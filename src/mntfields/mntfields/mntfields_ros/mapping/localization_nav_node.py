#!/usr/bin/env python3
"""
localization_nav_node.py

Localization + Navigation node for MNTFields.

- Subscribes to /map (static OccupancyGrid from map_server)
- Runs AMCL for localization (external, see instructions below)
- Accepts 2D Pose Estimate from RViz → seeds AMCL initial pose
- Accepts 2D Goal Pose from RViz → plans and drives using MNTFields regions
- Loads all saved region weights from disk
- Plans trajectory via region.generate_field_path()
- Drives via Navigator

Usage:
    ros2 run <your_pkg> localization_nav_node \
        --ros-args -p map_yaml:=/path/to/map.yaml \
                   -p model_dir:=/home/user/turtlebot3_ws/mntfields/model/mntfields_output \
                   -p max_lin:=0.15 \
                   -p max_ang:=0.8 \
                   -p goal_reached_dist:=0.25
"""

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
import tf2_ros
import numpy as np
import os
import glob
import json
import torch
import math

from nav_msgs.msg import OccupancyGrid, Path, Odometry
from geometry_msgs.msg import (
    Twist, PoseStamped, PoseWithCovarianceStamped, TransformStamped
)
from visualization_msgs.msg import Marker, MarkerArray
from scipy.spatial.transform import Rotation

from mntfields.mntfieldscore.core.region import Region
from mntfields.mntfieldsros.mapping.navigation import Navigator


# ── Dummy stubs required by Region.__init__ ──────────────────────────────────

class DummyRenderer:
    def render(self, *a, **kw):
        return None


class DummyModel:
    def __init__(self, model_dir: str):
        self.folder   = model_dir
        self.renderer = "cuda:0"   # must be a string, not an object
        self.h        = 0.0


# ─────────────────────────────────────────────────────────────────────────────

class LocalizationNavNode(Node):

    def __init__(self):
        super().__init__('localization_nav_node')

        # ── Parameters ───────────────────────────────────────────────
        self.declare_parameter('model_dir',
            '/home/user/turtlebot3_ws/mntfields/model/mntfields_output')
        self.declare_parameter('max_lin',  0.15)
        self.declare_parameter('max_ang',  0.8)
        self.declare_parameter('goal_reached_dist', 0.25)
        self.declare_parameter('step_size',  0.10)   # field path step (m)
        self.declare_parameter('max_steps',  400)    # field path max iters
        self.declare_parameter('use_sim_time', False)

        self.set_parameters([
            rclpy.parameter.Parameter(
                'use_sim_time',
                rclpy.Parameter.Type.BOOL,
                self.get_parameter('use_sim_time').value
            )
        ])

        self.model_dir = self.get_parameter('model_dir').value

        # ── State ─────────────────────────────────────────────────────
        self.robot_pos: np.ndarray | None = None   # [x, y, z]
        self.robot_yaw: float             = 0.0
        self.goal_xy:   np.ndarray | None = None
        self.path:      np.ndarray | None = None   # Nx2 active path
        self.path_gen_running             = False

        # Region management
        self.all_regions: dict[int, Region] = {}
        self.active_region: Region | None   = None

        # ── Load all saved regions ────────────────────────────────────
        self._load_all_regions()

        # ── TF ────────────────────────────────────────────────────────
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ── Navigator (reuses same interface as gridmap_node) ─────────
        # Navigator expects node to have: cmd_pub, robot_pos, robot_yaw,
        # and ROS parameters max_lin, max_ang, goal_reached_dist
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.navigator = Navigator(self)

        # ── Subscribers ───────────────────────────────────────────────
        # AMCL pose — best-estimate localization result
        self.create_subscription(
            PoseWithCovarianceStamped,
            '/amcl_pose',
            self._amcl_pose_cb,
            10
        )

        # Odometry fallback (used until first AMCL fix arrives)
        self.create_subscription(
            Odometry,
            '/odom',
            self._odom_cb,
            10
        )

        # 2D Pose Estimate from RViz → seeds AMCL initial pose
        # (you publish this in RViz with the "2D Pose Estimate" tool)
        # The AMCL node itself subscribes to /initialpose — we just
        # relay it so we can also log it here.
        self.create_subscription(
            PoseWithCovarianceStamped,
            '/initialpose',
            self._initial_pose_cb,
            10
        )

        # 2D Goal Pose from RViz
        self.create_subscription(
            PoseStamped,
            '/goal_pose',
            self._goal_pose_cb,
            10
        )

        # ── Publishers ────────────────────────────────────────────────
        self.path_pub   = self.create_publisher(Path,        '/nav_path',      10)
        self.marker_pub = self.create_publisher(MarkerArray, '/nav_markers',   10)

        # ── Update timer (10 Hz) ──────────────────────────────────────
        self.create_timer(0.10, self._update)

        self.get_logger().info(
            f'[LocNavNode] Started. Loaded {len(self.all_regions)} region(s). '
            f'Waiting for 2D Pose Estimate then 2D Goal Pose in RViz.'
        )

    # ── Region loading ────────────────────────────────────────────────────────

    def _load_all_regions(self):
        """
        Scan model_dir for region_N/ sub-folders and load each saved region bundle.
        A region folder is valid if it contains region_N_meta.json.
        """
        if not os.path.isdir(self.model_dir):
            self.get_logger().error(
                f'[LocNavNode] model_dir not found: {self.model_dir}'
            )
            return

        dummy_model = DummyModel(self.model_dir)
        pattern     = os.path.join(self.model_dir, 'region_*')

        for region_dir in sorted(glob.glob(pattern)):
            folder_name = os.path.basename(region_dir)
            try:
                ver = int(folder_name.split('_')[1])
            except (IndexError, ValueError):
                continue

            meta_path = os.path.join(region_dir, f'region_{ver}_meta.json')
            if not os.path.isfile(meta_path):
                self.get_logger().warn(
                    f'[LocNavNode] region {ver} metadata missing, skipping.'
                )
                continue

            try:
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                bound = torch.tensor(meta.get('bound'), dtype=torch.float32)

                region = Region(
                    model=dummy_model,
                    bound=bound,
                    version=ver,
                    init_network=True,
                    eval=False
                )
                region.saved = True
                region.load_from_disk()
                self.all_regions[ver] = region
                self.get_logger().info(
                    f'[LocNavNode] Loaded region {ver} '
                    f'bound=[{bound[0,0]:.2f},{bound[0,1]:.2f}]'
                    f'→[{bound[1,0]:.2f},{bound[1,1]:.2f}] '
                    f'epoch={region.epoch}'
                )
            except Exception as e:
                self.get_logger().error(
                    f'[LocNavNode] Failed loading region {ver}: {e}'
                )

        if not self.all_regions:
            self.get_logger().warn(
                '[LocNavNode] No regions loaded — navigation will not work. '
                'Make sure you ran exploration first and model_dir is correct.'
            )

    def _region_for_pos(self, xy: np.ndarray) -> 'Region | None':
        """Return the region whose bounding box contains xy."""
        for region in self.all_regions.values():
            b = region.bound.detach().cpu().numpy()
            if (b[0, 0] <= xy[0] <= b[1, 0] and
                    b[0, 1] <= xy[1] <= b[1, 1]):
                return region
        return None

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _amcl_pose_cb(self, msg: PoseWithCovarianceStamped):
        """Best pose estimate from AMCL — use this as primary position."""
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.robot_pos = np.array([p.x, p.y, p.z], dtype=np.float32)
        sin_y = 2.0 * (q.w * q.z + q.x * q.y)
        cos_y = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.robot_yaw = math.atan2(sin_y, cos_y)

    def _odom_cb(self, msg: Odometry):
        """Fallback odometry — only used if AMCL has not published yet."""
        if self.robot_pos is not None:
            return   # AMCL is already providing pose
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.robot_pos = np.array([p.x, p.y, p.z], dtype=np.float32)
        sin_y = 2.0 * (q.w * q.z + q.x * q.y)
        cos_y = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.robot_yaw = math.atan2(sin_y, cos_y)

    def _initial_pose_cb(self, msg: PoseWithCovarianceStamped):
        """
        RViz '2D Pose Estimate' click.
        AMCL subscribes to /initialpose directly — we just log it here
        so the user knows AMCL received the seed.
        """
        p = msg.pose.pose.position
        self.get_logger().info(
            f'[LocNavNode] Initial pose set: ({p.x:.2f}, {p.y:.2f}) — '
            f'AMCL should now localize.'
        )

    def _goal_pose_cb(self, msg: PoseStamped):
        """RViz '2D Goal Pose' click — triggers path planning + navigation."""
        gx = msg.pose.position.x
        gy = msg.pose.position.y
        self.goal_xy = np.array([gx, gy], dtype=np.float32)
        self.path    = None
        self.navigator.reset_path()
        self.path_gen_running = False

        self.get_logger().info(
            f'[LocNavNode] New goal: ({gx:.2f}, {gy:.2f})'
        )

        # Immediately try to find the region for the goal and start point
        if self.robot_pos is not None:
            self._trigger_path_gen()

    # ── Path generation ───────────────────────────────────────────────────────

    def _trigger_path_gen(self):
        """
        Find the best region that covers the robot (or goal) and generate
        a field path in a background thread, matching gridmap_node pattern.
        """
        if self.path_gen_running:
            return
        if self.robot_pos is None or self.goal_xy is None:
            return

        robot_xy = self.robot_pos[:2].copy()
        goal_xy  = self.goal_xy.copy()

        # Pick region: prefer one containing the robot, else closest to robot
        region = self._region_for_pos(robot_xy)
        if region is None:
            # Fall back: pick region whose center is closest to robot
            if not self.all_regions:
                self.get_logger().warn('[LocNavNode] No regions available for path gen.')
                return
            region = min(
                self.all_regions.values(),
                key=lambda r: np.linalg.norm(r.center[:2] - robot_xy)
            )
            self.get_logger().warn(
                f'[LocNavNode] Robot not inside any region bound, '
                f'using closest region {region.version}'
            )

        self.active_region = region
        self.path_gen_running = True
        step  = float(self.get_parameter('step_size').value)
        steps = int(self.get_parameter('max_steps').value)

        import threading

        def _worker():
            try:
                raw = region.generate_field_path(
                    robot_xy, goal_xy,
                    stepsize=step,
                    maxsteps=steps
                )
                if raw is not None and len(raw) > 1:
                    path = np.asarray(raw, dtype=np.float32)
                    path = self._decimate(path, spacing=0.20)
                    path = self._smooth(path, window=3)
                    self.path = np.asarray(path, dtype=np.float32)
                    self.navigator.reset_path()
                    self.get_logger().info(
                        f'[LocNavNode] Path generated: {len(self.path)} waypoints '
                        f'via region {region.version}'
                    )
                else:
                    self.get_logger().warn('[LocNavNode] generate_field_path returned empty.')
            except Exception as e:
                self.get_logger().error(f'[LocNavNode] Path gen error: {e}')
            finally:
                self.path_gen_running = False

        threading.Thread(target=_worker, daemon=True).start()

    # ── Path utilities (mirrors gridmap_node) ────────────────────────────────

    def _decimate(self, path_np: np.ndarray, spacing: float = 0.20) -> np.ndarray:
        pts     = path_np.astype(np.float32)
        new_pts = [pts[0]]
        for i in range(1, len(pts)):
            if np.linalg.norm(pts[i] - new_pts[-1]) >= spacing:
                new_pts.append(pts[i])
        if np.linalg.norm(new_pts[-1] - pts[-1]) > 0.05:
            new_pts.append(pts[-1])
        return np.array(new_pts, dtype=np.float32)

    def _smooth(self, path_np: np.ndarray, window: int = 5) -> np.ndarray:
        pts = path_np.astype(np.float32)
        if len(pts) < window:
            return pts
        sm = pts.copy()
        k  = window // 2
        for i in range(len(pts)):
            a = max(0, i - k)
            b = min(len(pts), i + k + 1)
            sm[i] = pts[a:b].mean(axis=0)
        sm[0]  = pts[0]
        sm[-1] = pts[-1]
        return sm

    # ── Path publishing ───────────────────────────────────────────────────────

    def _publish_path(self):
        if self.path is None:
            return
        msg = Path()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        for p in self.path:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x  = float(p[0])
            ps.pose.position.y  = float(p[1])
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        self.path_pub.publish(msg)

    def _publish_goal_marker(self):
        if self.goal_xy is None:
            return
        arr = MarkerArray()
        m = Marker()
        m.header.stamp    = self.get_clock().now().to_msg()
        m.header.frame_id = 'map'
        m.ns   = 'goal'
        m.id   = 0
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x  = float(self.goal_xy[0])
        m.pose.position.y  = float(self.goal_xy[1])
        m.pose.position.z  = 0.1
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 0.25
        m.color.r = 1.0; m.color.g = 0.2; m.color.b = 0.0; m.color.a = 1.0
        arr.markers.append(m)
        self.marker_pub.publish(arr)

    # ── Main update loop ──────────────────────────────────────────────────────

    def _update(self):
        self._publish_path()
        self._publish_goal_marker()

        if self.robot_pos is None:
            return

        robot_xy = self.robot_pos[:2]

        # No goal → stop
        if self.goal_xy is None:
            self.navigator.stop_robot()
            return

        # Path not ready yet → try to generate
        if self.path is None:
            if not self.path_gen_running:
                self._trigger_path_gen()
            self.navigator.stop_robot()
            return

        # Check goal reached
        dist = float(np.linalg.norm(robot_xy - self.goal_xy))
        goal_reached_dist = float(
            self.get_parameter('goal_reached_dist').value
        )
        if dist < goal_reached_dist:
            self.get_logger().info(
                f'[LocNavNode] Goal reached! dist={dist:.3f}m'
            )
            self.navigator.stop_robot()
            self.goal_xy = None
            self.path    = None
            return

        # Drive along path
        done = self.navigator.drive_path(self.path)
        if done:
            self.get_logger().info('[LocNavNode] Navigator reported path complete.')
            self.goal_xy = None
            self.path    = None


def main(args=None):
    rclpy.init(args=args)
    node = LocalizationNavNode()
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
