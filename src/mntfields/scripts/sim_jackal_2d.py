#!/usr/bin/env python3
"""
sim_jackal_2d.py — headless 2D simulator of the Clearpath Jackal setup, for
debugging antfields_jackal_mapping.launch.py without the robot.

Simulates what the real robot provides over ROS, under a Clearpath-style
namespace:
  /<ns>/cmd_vel                      geometry_msgs/TwistStamped (subscribed)
  /<ns>/sensors/lidar2d_0/scan       sensor_msgs/LaserScan, 40 Hz
  /<ns>/tf        odom -> base_link  (ground-truth odometry, 50 Hz)
  /<ns>/tf_static base_link -> lidar2d_0_laser (rear mount, facing backwards)
  /<ns>/sim/ground_truth             geometry_msgs/PoseStamped (debugging)

World: square room with border walls and a box in the centre, or (--map) the
occupied cells of a saved occupancy grid, e.g. a SLAM map of the real maze, so
the real passages can be tested. Robot footprint collisions are blocked and counted.

Run it on an isolated ROS domain so it can never talk to a real robot:
    ROS_DOMAIN_ID=42 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST \\
        python3 scripts/sim_jackal_2d.py --ns sim
"""

import argparse
import math

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped, TwistStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from tf2_msgs.msg import TFMessage

# Jackal geometry (base_link at the chassis centre).
HALF_LEN = 0.255
HALF_WID = 0.215
# Lidar on the rear mount, facing backwards (matches the real robot's TF).
LIDAR_XYZ = (-0.12, 0.0, 0.231)
LIDAR_YAW = math.pi


def yaw_quat(yaw):
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


class World:
    """Axis-aligned room (walls on the border) plus axis-aligned box obstacles."""

    def __init__(self, room, boxes):
        self.room = room  # half-size of the square room (m)
        self.boxes = boxes  # list of (xmin, ymin, xmax, ymax)
        segs = []
        r = room
        segs += [((-r, -r), (r, -r)), ((r, -r), (r, r)), ((r, r), (-r, r)), ((-r, r), (-r, -r))]
        for x0, y0, x1, y1 in boxes:
            segs += [((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))]
        s = np.array(segs, dtype=np.float64)
        self.seg_a = s[:, 0, :]
        self.seg_d = s[:, 1, :] - s[:, 0, :]

    def blocked(self, pts):
        """True if any of the Nx2 points is outside the room or inside a box."""
        if np.any(np.abs(pts) >= self.room):
            return True
        for x0, y0, x1, y1 in self.boxes:
            if np.any((pts[:, 0] > x0) & (pts[:, 0] < x1) & (pts[:, 1] > y0) & (pts[:, 1] < y1)):
                return True
        return False

    def raycast(self, origin, angles, max_range):
        """Distance along each world-frame angle to the nearest segment."""
        d = np.stack([np.cos(angles), np.sin(angles)], axis=1)  # R x 2
        a = self.seg_a[None, :, :] - origin[None, None, :]       # 1 x S x 2
        e = self.seg_d[None, :, :]                                # 1 x S x 2
        dd = d[:, None, :]                                        # R x 1 x 2
        denom = dd[..., 0] * e[..., 1] - dd[..., 1] * e[..., 0]   # R x S
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (a[..., 0] * e[..., 1] - a[..., 1] * e[..., 0]) / denom
            u = (a[..., 0] * dd[..., 1] - a[..., 1] * dd[..., 0]) / denom
        hit = (np.abs(denom) > 1e-12) & (t > 0.0) & (u >= 0.0) & (u <= 1.0)
        t = np.where(hit, t, np.inf)
        r = t.min(axis=1)
        return np.where(np.isfinite(r), r, max_range + 1.0)


class GridWorld:
    """Walls = occupied cells of an occupancy grid (row = y), plus a border around it."""

    def __init__(self, data, res, ox, oy, occ_threshold=50):
        occ = np.asarray(data) > occ_threshold
        occ[0, :] = occ[-1, :] = occ[:, 0] = occ[:, -1] = True  # close the outside
        self.occ, self.res, self.ox, self.oy = occ, float(res), float(ox), float(oy)
        self.h, self.w = occ.shape

    def _cells(self, pts):
        c = np.floor((pts[:, 0] - self.ox) / self.res).astype(int)
        r = np.floor((pts[:, 1] - self.oy) / self.res).astype(int)
        return r, c

    def blocked(self, pts):
        r, c = self._cells(pts)
        out = (r < 0) | (r >= self.h) | (c < 0) | (c >= self.w)
        if out.any():
            return True
        return bool(self.occ[r, c].any())

    def raycast(self, origin, angles, max_range, step=0.02):
        steps = np.arange(step, max_range, step)
        xs = origin[0] + np.cos(angles)[:, None] * steps[None, :]
        ys = origin[1] + np.sin(angles)[:, None] * steps[None, :]
        c = np.floor((xs - self.ox) / self.res).astype(int)
        r = np.floor((ys - self.oy) / self.res).astype(int)
        out = (r < 0) | (r >= self.h) | (c < 0) | (c >= self.w)
        hit = out | self.occ[np.clip(r, 0, self.h - 1), np.clip(c, 0, self.w - 1)]
        first = np.where(hit.any(axis=1), hit.argmax(axis=1), -1)
        return np.where(first >= 0, steps[np.maximum(first, 0)], max_range + 1.0)


def footprint_points(x, y, yaw, step=0.02):
    xs = np.arange(-HALF_LEN, HALF_LEN + 1e-9, step)
    ys = np.arange(-HALF_WID, HALF_WID + 1e-9, step)
    local = np.concatenate([
        np.stack([xs, np.full_like(xs, HALF_WID)], 1),
        np.stack([xs, np.full_like(xs, -HALF_WID)], 1),
        np.stack([np.full_like(ys, HALF_LEN), ys], 1),
        np.stack([np.full_like(ys, -HALF_LEN), ys], 1),
    ])
    c, s = math.cos(yaw), math.sin(yaw)
    return np.stack([x + c * local[:, 0] - s * local[:, 1], y + s * local[:, 0] + c * local[:, 1]], 1)


class SimJackal(Node):
    def __init__(self, args):
        super().__init__("sim_jackal_2d")
        ns = "/" + args.ns.strip("/") if args.ns.strip("/") else ""
        if args.map:
            m = np.load(args.map)
            self.world = GridWorld(m["data"], m["res"], m["ox"], m["oy"])
        else:
            h = args.box / 2.0
            self.world = World(args.room / 2.0, [(-h, -h, h, h)] if args.box > 0 else [])
        self.x, self.y, self.yaw = args.x, args.y, args.yaw
        self.v_cmd = self.w_cmd = 0.0
        self.last_cmd_t = -1e9
        self.collisions = 0
        self.dist = 0.0
        self.noise = args.noise
        self.rng = np.random.default_rng(0)
        self.max_range = 30.0
        self.n_beams = 1081
        self.angle_min, self.angle_max = -2.356194496154785, 2.356194496154785

        tf_qos = QoSProfile(depth=100)
        static_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                                reliability=ReliabilityPolicy.RELIABLE)
        self.tf_pub = self.create_publisher(TFMessage, f"{ns}/tf", tf_qos)
        self.static_pub = self.create_publisher(TFMessage, f"{ns}/tf_static", static_qos)
        self.scan_pub = self.create_publisher(LaserScan, f"{ns}/sensors/lidar2d_0/scan",
                                              qos_profile_sensor_data)
        self.gt_pub = self.create_publisher(PoseStamped, f"{ns}/sim/ground_truth", 10)
        self.create_subscription(TwistStamped, f"{ns}/cmd_vel", self.cmd_cb, 10)

        self.publish_static_tf()
        self.dt = 0.02
        self.create_timer(self.dt, self.step)
        self.create_timer(1.0 / 40.0, self.publish_scan)
        self.create_timer(5.0, self.report)
        self.get_logger().info(
            f"[SIM] world {args.map or f'room {args.room:.1f} m, centre box {args.box:.1f} m'}, start "
            f"({self.x:.2f},{self.y:.2f},{math.degrees(self.yaw):.0f} deg), ns '{ns}'")

    def cmd_cb(self, msg):
        self.v_cmd = float(np.clip(msg.twist.linear.x, -1.0, 1.0))
        self.w_cmd = float(np.clip(msg.twist.angular.z, -2.0, 2.0))
        self.last_cmd_t = self.get_clock().now().nanoseconds * 1e-9

    def step(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        v, w = (self.v_cmd, self.w_cmd) if now - self.last_cmd_t < 0.5 else (0.0, 0.0)
        nyaw = self.yaw + w * self.dt
        nx = self.x + v * math.cos(self.yaw) * self.dt
        ny = self.y + v * math.sin(self.yaw) * self.dt
        if self.world.blocked(footprint_points(nx, ny, nyaw)):
            if v != 0.0 or w != 0.0:
                self.collisions += 1
                if self.collisions == 1 or self.collisions % 25 == 0:
                    self.get_logger().warn(
                        f"[SIM] COLLISION #{self.collisions} at ({self.x:.2f},{self.y:.2f}) "
                        f"yaw {math.degrees(self.yaw):.0f} deg cmd v={v:+.2f} w={w:+.2f}")
        else:
            self.dist += abs(v) * self.dt
            self.x, self.y, self.yaw = nx, ny, math.atan2(math.sin(nyaw), math.cos(nyaw))
        stamp = self.get_clock().now().to_msg()
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = "odom"
        t.child_frame_id = "base_link"
        t.transform.translation.x, t.transform.translation.y = self.x, self.y
        q = yaw_quat(self.yaw)
        t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = q
        self.tf_pub.publish(TFMessage(transforms=[t]))
        gt = PoseStamped()
        gt.header.stamp, gt.header.frame_id = stamp, "odom"
        gt.pose.position.x, gt.pose.position.y = self.x, self.y
        gt.pose.orientation.z, gt.pose.orientation.w = q[2], q[3]
        self.gt_pub.publish(gt)

    def publish_static_tf(self):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "base_link"
        t.child_frame_id = "lidar2d_0_laser"
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = LIDAR_XYZ
        q = yaw_quat(LIDAR_YAW)
        t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = q
        self.static_pub.publish(TFMessage(transforms=[t]))

    def publish_scan(self):
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        origin = np.array([self.x + c * LIDAR_XYZ[0] - s * LIDAR_XYZ[1],
                           self.y + s * LIDAR_XYZ[0] + c * LIDAR_XYZ[1]])
        beams = np.linspace(self.angle_min, self.angle_max, self.n_beams)
        ranges = self.world.raycast(origin, beams + self.yaw + LIDAR_YAW, self.max_range)
        if self.noise > 0:
            ranges = ranges + self.rng.normal(0.0, self.noise, ranges.shape)
        msg = LaserScan()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "lidar2d_0_laser"
        msg.angle_min, msg.angle_max = self.angle_min, self.angle_max
        msg.angle_increment = (self.angle_max - self.angle_min) / (self.n_beams - 1)
        msg.scan_time, msg.time_increment = 1.0 / 40.0, 0.0
        msg.range_min, msg.range_max = 0.02, self.max_range
        msg.ranges = ranges.astype(np.float32).tolist()
        self.scan_pub.publish(msg)

    def report(self):
        self.get_logger().info(
            f"[SIM] pose ({self.x:.2f},{self.y:.2f},{math.degrees(self.yaw):.0f} deg) "
            f"driven {self.dist:.1f} m, collisions {self.collisions}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ns", default="sim", help="robot namespace (topics /<ns>/...)")
    p.add_argument("--room", type=float, default=8.0, help="side length of the square room (m)")
    p.add_argument("--box", type=float, default=1.5, help="side length of the centre box (m), 0 = none")
    p.add_argument("--map", default="", help="npz occupancy grid (data, res, ox, oy) to use as the world")
    p.add_argument("--x", type=float, default=-2.5)
    p.add_argument("--y", type=float, default=-2.5)
    p.add_argument("--yaw", type=float, default=0.0, help="start heading (rad)")
    p.add_argument("--noise", type=float, default=0.01, help="range noise std dev (m)")
    args, ros_args = p.parse_known_args()
    rclpy.init(args=ros_args)
    node = SimJackal(args)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        print(
            f"[SIM] final pose ({node.x:.2f},{node.y:.2f}) driven {node.dist:.1f} m, "
            f"collisions {node.collisions}", flush=True)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
