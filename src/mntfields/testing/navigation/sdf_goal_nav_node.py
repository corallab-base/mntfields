#!/usr/bin/env python3

import math
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Point, PoseStamped, PoseWithCovarianceStamped, Twist
from nav_msgs.msg import Odometry, Path
from visualization_msgs.msg import Marker, MarkerArray

from mntfields.mntfields_ros.mapping.navigation import Navigator


class SDFGoalNavNode(Node):
    def __init__(self):
        super().__init__('sdf_goal_nav_node')

        self.declare_parameter('max_lin', 0.15)
        self.declare_parameter('max_ang', 0.8)
        self.declare_parameter('goal_reached_dist', 0.25)
        self.declare_parameter('global_path_spacing', 0.10)
        self.declare_parameter('sdf_valid_sec', 2.0)
        self.declare_parameter('sdf_min_hold_sec', 1.0)
        self.declare_parameter('sdf_replace_threshold', 0.20)
        self.declare_parameter('control_dt', 0.10)
        self.declare_parameter('stuck_window', 25)
        self.declare_parameter('stuck_radius', 0.08)
        self.declare_parameter('global_frame', 'odom')

        self.robot_pos: np.ndarray | None = None
        self.robot_yaw: float = 0.0
        self.goal_xy: np.ndarray | None = None
        self._amcl_received = False
        self.nav_frame = str(self.get_parameter('global_frame').value)

        self.global_path: np.ndarray | None = None
        self._latest_sdf_path: np.ndarray | None = None
        self._latest_sdf_path_time: float = 0.0
        self._accepted_sdf_path: np.ndarray | None = None
        self._accepted_sdf_path_time: float = 0.0
        self._active_path: np.ndarray | None = None
        self._active_mode: str = 'idle'

        self.dt = float(self.get_parameter('control_dt').value)
        self._position_history = deque(
            maxlen=max(5, int(self.get_parameter('stuck_window').value)))

        self.path_pub = self.create_publisher(Path, '/nav_path', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/sdf_nav_markers', 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.navigator = Navigator(self)

        self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self._amcl_pose_cb, 10)
        self.create_subscription(Odometry, '/odom', self._odom_cb, 10)
        self.create_subscription(PoseStamped, '/goal_pose', self._goal_pose_cb, 10)
        self.create_subscription(Path, '/sdf_local_path', self._sdf_path_cb, 10)

        self.create_timer(self.dt, self._control_loop)

        self.get_logger().info(
            '[SDF NAV] Ready. RViz goal -> straight global path -> SDF local replanning.')

    def _amcl_pose_cb(self, msg: PoseWithCovarianceStamped):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.robot_pos = np.array([p.x, p.y, p.z], dtype=np.float32)
        sin_y = 2.0 * (q.w * q.z + q.x * q.y)
        cos_y = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.robot_yaw = math.atan2(sin_y, cos_y)
        self._amcl_received = True

    def _odom_cb(self, msg: Odometry):
        if self._amcl_received:
            return
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.robot_pos = np.array([p.x, p.y, p.z], dtype=np.float32)
        sin_y = 2.0 * (q.w * q.z + q.x * q.y)
        cos_y = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.robot_yaw = math.atan2(sin_y, cos_y)

    def _goal_pose_cb(self, msg: PoseStamped):
        if self.robot_pos is None:
            self.get_logger().warn('[SDF NAV] No pose yet; ignoring goal.')
            return
        self.goal_xy = np.array([msg.pose.position.x, msg.pose.position.y], dtype=np.float32)
        self.global_path = self._build_straight_path(self.robot_pos[:2], self.goal_xy)
        self._latest_sdf_path = None
        self._accepted_sdf_path = None
        self._active_path = self.global_path
        self._active_mode = 'straight_global'
        self._position_history.clear()
        self.navigator.reset_path()
        self._publish_global_path()
        self._publish_markers()
        self.get_logger().info(
            f'[SDF NAV] New goal: ({self.goal_xy[0]:.2f}, {self.goal_xy[1]:.2f}) '
            f'global_pts={len(self.global_path)}'
        )

    def _sdf_path_cb(self, msg: Path):
        if len(msg.poses) < 2:
            return
        self._latest_sdf_path = np.array(
            [[p.pose.position.x, p.pose.position.y] for p in msg.poses],
            dtype=np.float32,
        )
        self._latest_sdf_path_time = self.get_clock().now().nanoseconds * 1e-9

    def _build_straight_path(self, start_xy: np.ndarray, goal_xy: np.ndarray) -> np.ndarray:
        spacing = max(0.02, float(self.get_parameter('global_path_spacing').value))
        delta = goal_xy - start_xy
        dist = float(np.linalg.norm(delta))
        if dist < 1e-6:
            return np.array([start_xy, goal_xy], dtype=np.float32)
        count = max(2, int(math.ceil(dist / spacing)) + 1)
        alpha = np.linspace(0.0, 1.0, count, dtype=np.float32)
        pts = start_xy[None, :] + alpha[:, None] * delta[None, :]
        return pts.astype(np.float32)

    def _compose_active_path(self, now_sec: float) -> tuple[np.ndarray | None, str]:
        if self.global_path is None:
            return None, 'idle'

        active = self.global_path
        mode = 'straight_global'
        valid_for = float(self.get_parameter('sdf_valid_sec').value)
        accepted = self._maybe_accept_sdf_path(now_sec, valid_for)
        if accepted is not None and len(accepted) >= 2:
            active = self._stitch_local_to_global(accepted, self.global_path)
            mode = 'sdf_local'
        return active, mode

    def _maybe_accept_sdf_path(self, now_sec: float, valid_for: float) -> np.ndarray | None:
        latest = self._latest_sdf_path
        if (latest is None
                or now_sec - self._latest_sdf_path_time >= valid_for
                or len(latest) < 2):
            self._accepted_sdf_path = None
            return None

        accepted = self._accepted_sdf_path
        if accepted is None or now_sec - self._accepted_sdf_path_time >= valid_for:
            self._accepted_sdf_path = latest.copy()
            self._accepted_sdf_path_time = now_sec
            return self._accepted_sdf_path

        hold_sec = float(self.get_parameter('sdf_min_hold_sec').value)
        if now_sec - self._accepted_sdf_path_time < hold_sec:
            return accepted

        replace_threshold = float(self.get_parameter('sdf_replace_threshold').value)
        if self._path_change_metric(accepted, latest) < replace_threshold:
            return accepted

        self._accepted_sdf_path = latest.copy()
        self._accepted_sdf_path_time = now_sec
        return self._accepted_sdf_path

    @staticmethod
    def _path_change_metric(a: np.ndarray, b: np.ndarray, samples: int = 16) -> float:
        ra = SDFGoalNavNode._resample_path_for_compare(a, samples)
        rb = SDFGoalNavNode._resample_path_for_compare(b, samples)
        if ra is None or rb is None:
            return float('inf')
        return float(np.mean(np.linalg.norm(ra - rb, axis=1)))

    @staticmethod
    def _resample_path_for_compare(path: np.ndarray, samples: int) -> np.ndarray | None:
        pts = np.asarray(path, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 2:
            return None
        if len(pts) == samples:
            return pts

        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        total = float(s[-1])
        if total < 1e-6:
            return np.repeat(pts[:1], samples, axis=0)

        targets = np.linspace(0.0, total, samples, dtype=np.float32)
        x = np.interp(targets, s, pts[:, 0])
        y = np.interp(targets, s, pts[:, 1])
        return np.stack([x, y], axis=1).astype(np.float32)

    def _stitch_local_to_global(self, local_path: np.ndarray, global_path: np.ndarray) -> np.ndarray:
        local = np.asarray(local_path, dtype=np.float32)
        global_pts = np.asarray(global_path, dtype=np.float32)
        if len(local) < 2 or len(global_pts) < 2:
            return global_pts

        d0 = np.linalg.norm(global_pts - local[-1], axis=1)
        join_idx = int(np.argmin(d0))
        suffix = global_pts[join_idx + 1:]

        stitched = [local]
        if len(suffix):
            if float(np.linalg.norm(local[-1] - suffix[0])) < 1e-3:
                suffix = suffix[1:]
            if len(suffix):
                stitched.append(suffix)
        return np.vstack(stitched).astype(np.float32)

    def _publish_global_path(self):
        if self.global_path is None:
            return
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.nav_frame
        for p in self.global_path:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = float(p[0])
            ps.pose.position.y = float(p[1])
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        self.path_pub.publish(msg)

    def _make_line_marker(self, marker_id: int, ns: str, pts: np.ndarray, rgb: tuple[float, float, float],
                          width: float, alpha: float) -> Marker:
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = self.nav_frame
        marker.ns = ns
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = width
        marker.color.r = rgb[0]
        marker.color.g = rgb[1]
        marker.color.b = rgb[2]
        marker.color.a = alpha
        marker.points = [Point(x=float(p[0]), y=float(p[1]), z=0.05) for p in pts]
        return marker

    def _make_text_marker(self, marker_id: int, text: str) -> Marker:
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = self.nav_frame
        marker.ns = 'status'
        marker.id = marker_id
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.pose.position.z = 0.5
        if self.robot_pos is not None:
            marker.pose.position.x = float(self.robot_pos[0])
            marker.pose.position.y = float(self.robot_pos[1])
        marker.scale.z = 0.18
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 0.95
        marker.text = text
        return marker

    def _publish_markers(self):
        arr = MarkerArray()

        for idx, ns in enumerate(['goal', 'global', 'local', 'active', 'status']):
            delete = Marker()
            delete.header.stamp = self.get_clock().now().to_msg()
            delete.header.frame_id = self.nav_frame
            delete.ns = ns
            delete.id = idx
            delete.action = Marker.DELETEALL
            arr.markers.append(delete)

        if self.goal_xy is not None:
            goal = Marker()
            goal.header.stamp = self.get_clock().now().to_msg()
            goal.header.frame_id = self.nav_frame
            goal.ns = 'goal'
            goal.id = 0
            goal.type = Marker.SPHERE
            goal.action = Marker.ADD
            goal.pose.position.x = float(self.goal_xy[0])
            goal.pose.position.y = float(self.goal_xy[1])
            goal.pose.position.z = 0.10
            goal.pose.orientation.w = 1.0
            goal.scale.x = 0.25
            goal.scale.y = 0.25
            goal.scale.z = 0.25
            goal.color.r = 1.0
            goal.color.g = 0.1
            goal.color.b = 0.1
            goal.color.a = 1.0
            arr.markers.append(goal)

        if self.global_path is not None and len(self.global_path) >= 2:
            arr.markers.append(
                self._make_line_marker(0, 'global', self.global_path, (1.0, 0.85, 0.10), 0.03, 0.65))

        if self._latest_sdf_path is not None and len(self._latest_sdf_path) >= 2:
            arr.markers.append(
                self._make_line_marker(0, 'local', self._latest_sdf_path, (0.10, 0.95, 1.0), 0.05, 0.65))

        if self._accepted_sdf_path is not None and len(self._accepted_sdf_path) >= 2:
            arr.markers.append(
                self._make_line_marker(1, 'local', self._accepted_sdf_path, (0.00, 0.65, 1.0), 0.08, 0.90))

        if self._active_path is not None and len(self._active_path) >= 2:
            arr.markers.append(
                self._make_line_marker(0, 'active', self._active_path, (0.20, 1.0, 0.20), 0.07, 0.95))

        arr.markers.append(self._make_text_marker(0, f'mode: {self._active_mode}'))
        self.marker_pub.publish(arr)

    def _is_stuck(self) -> bool:
        if len(self._position_history) < self._position_history.maxlen:
            return False
        pts = np.asarray(self._position_history, dtype=np.float32)
        disp = float(np.linalg.norm(pts[-1] - pts[0]))
        return disp < float(self.get_parameter('stuck_radius').value)

    def _control_loop(self):
        self._publish_global_path()

        if self.robot_pos is None:
            self._active_path = None
            self._active_mode = 'waiting_pose'
            self._publish_markers()
            self.navigator.stop_robot()
            return

        self._position_history.append(self.robot_pos[:2].copy())

        if self.goal_xy is None:
            self._active_path = None
            self._active_mode = 'idle'
            self._publish_markers()
            self.navigator.stop_robot()
            return

        robot_xy = self.robot_pos[:2]
        d_goal = float(np.linalg.norm(robot_xy - self.goal_xy))
        if d_goal < float(self.get_parameter('goal_reached_dist').value):
            self.get_logger().info(f'[SDF NAV] Goal reached: d={d_goal:.3f}m')
            self.navigator.stop_robot()
            self.navigator.reset_path()
            self.goal_xy = None
            self.global_path = None
            self._latest_sdf_path = None
            self._accepted_sdf_path = None
            self._active_path = None
            self._active_mode = 'goal_reached'
            self._publish_markers()
            return

        now_sec = self.get_clock().now().nanoseconds * 1e-9
        self._active_path, self._active_mode = self._compose_active_path(now_sec)
        self._publish_markers()

        if self._active_path is None:
            self.navigator.stop_robot()
            return

        if self._is_stuck():
            self.get_logger().warn(
                f'[SDF NAV] No progress detected while following {self._active_mode}.',
                throttle_duration_sec=2.0)

        self.navigator.drive_path(self._active_path)


def main(args=None):
    rclpy.init(args=args)
    node = SDFGoalNavNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.navigator.stop_robot()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
