"""
antfields_gridmap_node.py

Single-region AntFields-style exploration node for a real TurtleBot.

This node reuses the existing Region neural field, Navigator controller,
Lidar2DAdapter scan conversion, and LidarGridmap SLAM-frontier extraction, but
does not create or switch between multiple regions and does not use room
segmentation / ownership logic.
"""

import math
import os
import threading
import time

import numpy as np
import rclpy
import rclpy.time
import tf2_ros
import torch
from scipy.ndimage import binary_dilation, distance_transform_edt, label
from skimage.graph import MCP_Geometric
from scipy.spatial.transform import Rotation
from geometry_msgs.msg import Twist, TwistStamped
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Path
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import LaserScan, PointCloud2, PointField
from visualization_msgs.msg import Marker

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg", force=True)

from mntfields.mntfields_core.core.region import Region
from mntfields.mntfields_core.mapping.lidar_gridmap import LidarGridmap
from mntfields.mntfields_ros.mapping.markers import MarkerPublisher
from mntfields.mntfields_ros.mapping.navigation import Navigator
from mntfields.mntfields_ros.sensing.lidar_adapter import Lidar2DAdapter
from mntfields.utils.mntfields_utils import straight_line_path


DEFAULT_MODEL_DIR = os.path.expanduser("~/mntfields_models/antfields")


class _StampedCmdPublisher:
    """Publishes Twist commands as TwistStamped (e.g. Clearpath Jazzy cmd_vel)."""

    def __init__(self, node, topic, frame_id):
        self._node = node
        self._pub = node.create_publisher(TwistStamped, topic, 10)
        self._frame_id = frame_id

    def publish(self, twist: Twist):
        msg = TwistStamped()
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.header.frame_id = self._frame_id
        msg.twist = twist
        self._pub.publish(msg)


class _SafetyFilteredPublisher:
    """Runs every velocity command through filter_fn before publishing it."""

    def __init__(self, inner, filter_fn):
        self._inner = inner
        self._filter_fn = filter_fn

    def publish(self, twist: Twist):
        self._inner.publish(self._filter_fn(twist))


class AntfieldsGridmapNode(Node):
    def __init__(self):
        super().__init__("antfields_gridmap")

        self.declare_parameter("model_dir", DEFAULT_MODEL_DIR)
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("map_topic", "/map")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("cmd_vel_stamped", False)
        self.declare_parameter("base_frame", "base_footprint")
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("lidar_max_range_m", 3.5)
        self.declare_parameter("wall_min_m", 0.05)
        self.declare_parameter("wall_max_m", 0.80)
        # Robot clearance for the field: walls are treated as inflated by this much, so
        # the speed band above (and how strongly walls repel paths) is unchanged.
        self.declare_parameter("wall_offset_m", 0.0)

        self.model_dir = os.path.expanduser(str(self.get_parameter("model_dir").value))
        os.makedirs(self.model_dir, exist_ok=True)
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.odom_frame = str(self.get_parameter("odom_frame").value)

        self.declare_parameter("train_timer_sec", 0.2)
        self.declare_parameter("train_samples", 12000)
        self.declare_parameter("train_epoch", 30)
        self.declare_parameter("region_max_raw_samples", 80000)
        self.declare_parameter("region_raw_memory_cell_size_m", 0.25)
        self.declare_parameter("enable_region_debug_vis", True)
        self.declare_parameter("goal_reached_dist", 0.30)
        # Path follower: distance to the end of the current path that counts as done.
        # <= 0 uses goal_reached_dist.
        self.declare_parameter("path_end_tolerance_m", -1.0)
        self.declare_parameter("max_lin", 0.12)
        self.declare_parameter("max_ang", 0.90)
        self.declare_parameter("control_timer_sec", 0.10)
        self.declare_parameter("lidar_height_m", 0.17)
        self.declare_parameter("vis_timer_sec", 3.0)
        self.declare_parameter("min_train_epochs_for_nav", 80)
        self.declare_parameter("follow_stuck_timeout_sec", 4.0)
        self.declare_parameter("follow_progress_eps", 0.10)
        self.declare_parameter("path_commit_max_m", 0.50)
        self.declare_parameter("path_truncate_trigger_m", 0.80)
        self.declare_parameter("frontier_max_goal_dist_m", 6.0)
        self.declare_parameter("goal_blacklist_radius_m", 0.50)
        # Abandoned (not reached) goals are only blacklisted for this long; they are
        # often reachable again once the field has trained or more map is known.
        self.declare_parameter("abandoned_goal_ttl_sec", 60.0)
        self.declare_parameter("train_min_translation_m", 0.05)
        self.declare_parameter("train_min_rotation_rad", 0.12)
        self.declare_parameter("train_force_period_sec", 0.75)
        self.declare_parameter("region_padding_m", 0.75)
        self.declare_parameter("region_min_size_m", 6.0)
        self.declare_parameter("fixed_region_size_m", 20.0)
        self.declare_parameter("expand_region_with_map", False)
        self.declare_parameter("collision_pad_cells", 2)
        # >= 0: SLAM collision = closer than this (Euclidean) to an occupied cell, via a
        # distance map; 0 = only occupied cells themselves (no inflation).
        # < 0: the square collision_pad_cells check (original behaviour).
        self.declare_parameter("collision_radius_m", -1.0)
        self.declare_parameter("path_collision_sample_step_m", 0.03)
        self.declare_parameter("unknown_is_collision", False)
        self.declare_parameter("trim_at_unknown", False)
        self.declare_parameter("enable_path_validation", False)
        self.declare_parameter("allow_straight_line_fallback", False)
        self.declare_parameter("field_vis_timer_sec", 1.0)
        self.declare_parameter("field_vis_radius_m", 6.0)
        self.declare_parameter("field_vis_resolution_m", 0.10)
        self.declare_parameter("field_vis_free_only", True)
        # Ignore SLAM collisions this close to the path start, so a robot that
        # ends up near a wall can still plan away (the scan safety stop covers it).
        self.declare_parameter("collision_check_skip_m", 0.0)
        # > 0: only validate this much of each path; only path_commit_max_m is driven
        # before replanning, and the rest is re-checked then. 0 validates the whole path.
        self.declare_parameter("validation_horizon_m", 0.0)
        # Reject paths whose first step points away from the goal. Off for maps with
        # obstacles: a valid path often has to start by going around one.
        self.declare_parameter("require_goalward_start", True)
        # A frontier goal can sit closer to a wall than collision_radius_m allows. If a
        # path only becomes unsafe within this distance of the goal, drive to its last
        # safe point instead of rejecting it, and count the goal as reached there.
        self.declare_parameter("goal_approach_m", 1.0)
        # > 0: route to far frontiers on the SLAM map (clearance-aware shortest path) and
        # let the field plan to a point this far along the route, one piece at a time.
        # The single field's rollout gets trapped in local minima over long routes.
        # 0 = plan straight to the frontier (original behaviour).
        self.declare_parameter("route_subgoal_dist_m", 0.0)
        # > 0: ignore frontier points that only border an unknown patch smaller than
        # this many cells (holes in mapped free space that no beam happened to hit).
        self.declare_parameter("min_frontier_unknown_cells", 0)
        # Live-scan forward safety stop (needed for lidars with a rear blind spot).
        self.declare_parameter("safety_stop_enabled", False)
        self.declare_parameter("robot_front_m", 0.30)
        self.declare_parameter("robot_rear_m", 0.30)
        # Drive rear-first along paths (for a lidar that faces the robot's rear).
        self.declare_parameter("drive_reverse", False)
        self.declare_parameter("robot_half_width_m", 0.25)
        self.declare_parameter("safety_stop_dist_m", 0.25)
        self.declare_parameter("safety_slow_dist_m", 0.60)
        self.declare_parameter("safety_scan_timeout_sec", 0.5)
        # > 0: before turning on the spot, need this much SLAM-map clearance (the radius
        # the robot's corners sweep); otherwise creep to where there is room first.
        self.declare_parameter("robot_turn_radius_m", 0.0)
        # One full turn in place before exploring, to fill a lidar blind spot.
        self.declare_parameter("initial_spin", False)
        self.declare_parameter("initial_spin_speed", 0.4)

        self.robot_pos = None
        self.robot_yaw = 0.0

        self.region = None
        self._region_store = {}

        self.gridmap = None
        self._slam_arr = None
        self._slam_cell_size = 0.05
        self._slam_origin = np.zeros(2, dtype=np.float32)
        self._slam_map_received = False

        self.latest_scan = None
        self._latest_world_pts = None
        self._latest_robot_pose_for_pts = None
        self._last_train_pose = None
        self._last_train_yaw = None
        self._last_train_wall_t = 0.0

        self._path = None
        self._current_goal = None
        self._active_path_goal = None
        self._visited_goals = []
        self._abandoned_goals = []  # [(xy, monotonic time abandoned)]
        self._goal_blacklist_radius = float(
            self.get_parameter("goal_blacklist_radius_m").value
        )
        self._path_total_dist = None
        self._midpoint_replanned = False
        self._best_goal_dist = float("inf")
        self._last_progress_time = 0.0
        self._last_path_progress_s = 0.0
        self._controller_path_done = False
        self._stuck_replan_count = 0
        self._pathgen_fail_count = 0
        self._exploration_tick_count = 0
        self.goal_xy = None
        self._frontier_pick_stats = {}

        self._train_running = False
        self._train_lock = threading.Lock()

        self.tf_buffer = tf2_ros.Buffer()
        # Receive TF on the listener's own internal node and thread. Lookups with a
        # timeout (and planning) block this node's executor; if TF were received on it
        # too, every lookup would starve and the buffer would fall further behind.
        # Passing node=None matters: with our node, spin_thread would spin *this* node
        # on a second executor. Process-wide remaps (/tf -> /<ns>/tf) still apply.
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, None, spin_thread=True)

        self.adapter = Lidar2DAdapter(
            max_range=float(self.get_parameter("lidar_max_range_m").value),
            lidar_height=float(self.get_parameter("lidar_height_m").value),
        )

        # Planning can take seconds. On the executor thread it would also stop scans and
        # velocity commands (the robot's twist_mux then halts it), so the executor only runs
        # the fast callbacks (scans, control, training trigger) and everything slow or
        # map-reading runs on one planning thread (see _planning_loop). The map callback
        # only hands the newest map over, so the map never changes mid-plan.
        # Guards the active path and the Navigator between control_tick and planning.
        self._nav_lock = threading.RLock()
        self._pending_map = None
        self._planning_stop = threading.Event()

        # Sensor-data (best-effort) QoS receives from both reliable and best-effort
        # lidar publishers; a reliable subscription silently gets nothing from the latter.
        self.create_subscription(
            LaserScan, str(self.get_parameter("scan_topic").value), self.lidar_cb,
            qos_profile_sensor_data,
        )

        map_qos = QoSProfile(
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
            depth=1,
        )
        self.create_subscription(
            OccupancyGrid, str(self.get_parameter("map_topic").value),
            lambda msg: setattr(self, "_pending_map", msg), map_qos,
        )

        self.nav_frame = "map"

        cmd_topic = str(self.get_parameter("cmd_vel_topic").value)
        if bool(self.get_parameter("cmd_vel_stamped").value):
            self.cmd_pub = _StampedCmdPublisher(self, cmd_topic, self.base_frame)
        else:
            self.cmd_pub = self.create_publisher(Twist, cmd_topic, 10)
        if (bool(self.get_parameter("safety_stop_enabled").value)
                or float(self.get_parameter("robot_turn_radius_m").value) > 0.0):
            self.cmd_pub = _SafetyFilteredPublisher(self.cmd_pub, self._filter_cmd)
        self._latest_scan_wall_t = 0.0
        self._safety_blocked = False
        self._spin_done = not bool(self.get_parameter("initial_spin").value)
        self._spin_accum = 0.0
        self._spin_last_yaw = None
        self.field_path_pub = self.create_publisher(Path, "/field_path", 10)
        self.field_path_pub_alt = self.create_publisher(
            Path, "/mntfields/local_model_path", 10
        )
        self.nav_path_pub = self.create_publisher(Path, "/navigator_path", 10)
        self.inflation_pub = self.create_publisher(OccupancyGrid, "/mntfields/inflation", map_qos)
        self.frontier_occ_pub = self.create_publisher(
            OccupancyGrid, "/mntfields/frontier", map_qos
        )
        self.frontier_marker_pub = self.create_publisher(Marker, "/mntfields/frontier_markers", 10)
        self.goal_marker_pub = self.create_publisher(Marker, "/mntfields/goal_marker", 10)
        self.travel_time_pub = self.create_publisher(PointCloud2, "/mntfields/travel_time", 1)
        self.speed_pub = self.create_publisher(PointCloud2, "/mntfields/speed", 1)

        self.drive_reverse = bool(self.get_parameter("drive_reverse").value)
        self.markers = MarkerPublisher(self)
        self.navigator = Navigator(self)

        train_dt = float(self.get_parameter("train_timer_sec").value)
        ctrl_dt = float(self.get_parameter("control_timer_sec").value)
        vis_dt = float(self.get_parameter("vis_timer_sec").value)
        self.create_timer(train_dt, self.train_tick)
        self.create_timer(ctrl_dt, self.control_tick)
        self._planning_thread = threading.Thread(
            target=self._planning_loop, args=(ctrl_dt, vis_dt), daemon=True
        )
        self._planning_thread.start()

        self.get_logger().info("[AntFieldsNode] initialized")

    def _ensure_model_training_state(self, region: Region) -> None:
        model = getattr(region, "model", None)
        if model is None:
            return
        if not hasattr(model, "prev_state_queue"):
            model.prev_state_queue = []
        if not hasattr(model, "prev_optimizer_queue"):
            model.prev_optimizer_queue = []
        if not hasattr(model, "prev_loss_queue"):
            model.prev_loss_queue = []

    def _lookup_robot_transform(self):
        candidates = [self.base_frame] + [
            f for f in ("base_footprint", "base_link") if f != self.base_frame
        ]
        for base_frame in candidates:
            try:
                return self.tf_buffer.lookup_transform(
                    "map",
                    base_frame,
                    rclpy.time.Time(),
                    timeout=Duration(seconds=0.05),
                )
            except Exception:
                pass
            # "Latest" through map is pinned to the newest map->odom stamp; if SLAM
            # stalls, odom->base at that time has already left the buffer. Combine the
            # latest map->odom with the latest odom->base instead.
            try:
                return self.tf_buffer.lookup_transform_full(
                    "map",
                    rclpy.time.Time(),
                    base_frame,
                    rclpy.time.Time(),
                    self.odom_frame,
                    timeout=Duration(seconds=0.05),
                )
            except Exception:
                continue
        return None

    def _warn_if_slam_stale(self):
        try:
            t = self.tf_buffer.lookup_transform("map", self.odom_frame, rclpy.time.Time())
        except Exception:
            return
        age = (self.get_clock().now() - rclpy.time.Time.from_msg(t.header.stamp)).nanoseconds * 1e-9
        if age > 3.0:
            self.get_logger().warn(
                f"[SLAM] map->{self.odom_frame} is {age:.1f}s old; slam_toolbox is not "
                "updating (map frozen, using last known map->odom)",
                throttle_duration_sec=5.0,
            )

    def _update_robot_pose_from_tf(self):
        t = self._lookup_robot_transform()
        if t is None:
            return
        p = t.transform.translation
        q = t.transform.rotation
        self.robot_pos = np.array([p.x, p.y, p.z], dtype=np.float32)
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.robot_yaw = float(np.arctan2(siny_cosp, cosy_cosp))

    def lidar_cb(self, msg: LaserScan):
        self.latest_scan = msg
        self._update_robot_pose_from_tf()
        if self.robot_pos is None:
            return

        pts_sensor = self.adapter.scan_to_xy(msg)
        if pts_sensor is None or len(pts_sensor) == 0:
            return

        world_pts = self._transform_scan_points_to_map(pts_sensor, msg)
        if world_pts is None:
            return

        # The field is 2D and Region queries its obstacle KD-tree at z=0; keeping the
        # lidar's TF height here would add it to every wall distance.
        world_pts[:, 2] = 0.0
        self._latest_world_pts = world_pts
        self._latest_scan_wall_t = time.monotonic()
        self._latest_robot_pose_for_pts = self.robot_pos.copy()

    def _transform_scan_points_to_map(self, pts_sensor: np.ndarray, msg: LaserScan):
        """Transform Nx3 scan-frame points into the map frame using TF."""
        source_frame = str(msg.header.frame_id).strip()
        if not source_frame:
            self.get_logger().warn("[LIDAR] scan header has empty frame_id; dropping scan")
            return None

        stamp = rclpy.time.Time.from_msg(msg.header.stamp)
        try:
            tf_msg = self.tf_buffer.lookup_transform(
                "map",
                source_frame,
                stamp,
                timeout=Duration(seconds=0.05),
            )
        except Exception:
            try:
                # Lidar pose in odom at the scan time, combined with the latest map->odom.
                tf_msg = self.tf_buffer.lookup_transform_full(
                    "map",
                    rclpy.time.Time(),
                    source_frame,
                    stamp,
                    self.odom_frame,
                    timeout=Duration(seconds=0.05),
                )
            except Exception as exc:
                self.get_logger().warn(
                    f"[LIDAR] map <- {source_frame} TF unavailable; dropping scan: {exc}"
                )
                return None

        t = tf_msg.transform.translation
        q = tf_msg.transform.rotation
        rot = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix().astype(np.float32)
        trans = np.array([t.x, t.y, t.z], dtype=np.float32)
        return (pts_sensor @ rot.T) + trans

    def _planning_loop(self, period, vis_dt):
        """Exploration, map ingestion and visualisation, off the executor thread."""
        field_dt = float(self.get_parameter("field_vis_timer_sec").value)
        last_vis = last_field = 0.0
        while not self._planning_stop.is_set() and rclpy.ok():
            t0 = time.monotonic()
            for name, fn in (("map", self._ingest_pending_map), ("exploration", self.exploration_tick)):
                try:
                    fn()
                except Exception as exc:
                    self.get_logger().error(f"[PLANNING] {name} failed: {exc}")
            if t0 - last_field >= field_dt:
                last_field = t0
                try:
                    self.field_vis_tick()
                except Exception as exc:
                    self.get_logger().warn(f"[PLANNING] field vis failed: {exc}")
            if t0 - last_vis >= vis_dt:
                last_vis = t0
                try:
                    self._publish_frontier_markers()
                    self.vis_tick()
                except Exception as exc:
                    self.get_logger().warn(f"[PLANNING] vis failed: {exc}")
            self._planning_stop.wait(max(0.0, period - (time.monotonic() - t0)))

    def _ingest_pending_map(self):
        msg, self._pending_map = self._pending_map, None
        if msg is not None:
            self.slam_map_cb(msg)

    def slam_map_cb(self, msg: OccupancyGrid):
        info = msg.info
        self._slam_cell_size = float(info.resolution)
        self._slam_origin = np.array(
            [info.origin.position.x, info.origin.position.y], dtype=np.float32
        )

        h = int(info.height)
        w = int(info.width)
        arr = np.array(msg.data, dtype=np.int8).reshape(h, w)
        # Distance (m) from every cell to the nearest occupied cell, for collision checks.
        self._slam_occ_dist = distance_transform_edt(arr <= 50) * self._slam_cell_size
        self._slam_arr = arr
        self._publish_inflation(msg)

        map_size = float(max(h, w) * self._slam_cell_size)
        center_xy = self._slam_origin + np.array(
            [w * self._slam_cell_size * 0.5, h * self._slam_cell_size * 0.5],
            dtype=np.float32,
        )

        if self.gridmap is None or self.gridmap.n_cells != int(max(h, w)):
            n_cells = int(max(h, w))
            trav = np.ones((n_cells, n_cells), dtype=np.uint8)
            self.gridmap = LidarGridmap(
                trav_map=trav,
                regions=self._region_store,
                n_cells=n_cells,
                near_dist=2.0,
                map_size=map_size,
                origin=center_xy,
            )
        else:
            self.gridmap.origin = center_xy.astype(np.float32)
            self.gridmap.map_size = map_size
            self.gridmap.cell_size = self.gridmap.map_size / self.gridmap.n_cells
            self.gridmap.regions = self._region_store

        self.gridmap.ingest_slam_map(msg)
        self._drop_hole_frontiers(arr)
        self._ensure_global_region()

        if not self._slam_map_received:
            self._slam_map_received = True
            self.get_logger().info(
                "[SLAM MAP] first map received "
                f"size={w}x{h} res={self._slam_cell_size:.3f}m "
                f"origin=({self._slam_origin[0]:.2f},{self._slam_origin[1]:.2f})"
            )

    def _target_region_bound(self):
        if self._slam_arr is None:
            return None

        h, w = self._slam_arr.shape
        pad = float(self.get_parameter("region_padding_m").value)
        min_size = float(self.get_parameter("region_min_size_m").value)
        fixed_size = float(self.get_parameter("fixed_region_size_m").value)

        xmin = float(self._slam_origin[0]) - pad
        ymin = float(self._slam_origin[1]) - pad
        xmax = float(self._slam_origin[0] + w * self._slam_cell_size) + pad
        ymax = float(self._slam_origin[1] + h * self._slam_cell_size) + pad

        cx = 0.5 * (xmin + xmax)
        cy = 0.5 * (ymin + ymax)
        sx = max(min_size, fixed_size, xmax - xmin)
        sy = max(min_size, fixed_size, ymax - ymin)

        return torch.tensor(
            [
                [cx - 0.5 * sx, cy - 0.5 * sy, -0.1],
                [cx + 0.5 * sx, cy + 0.5 * sy, 0.1],
            ],
            dtype=torch.float32,
        )

    def _ensure_global_region(self):
        bound = self._target_region_bound()
        if bound is None:
            return

        if self.region is None:
            self.region = Region(
                model_folder=self.model_dir,
                bound=bound,
                version=1,
                device="cuda:0",
                wall_min=float(self.get_parameter("wall_min_m").value),
                wall_max=float(self.get_parameter("wall_max_m").value),
                wall_offset=float(self.get_parameter("wall_offset_m").value),
            )
            self.region.max_raw_samples = int(
                self.get_parameter("region_max_raw_samples").value
            )
            self.region.raw_memory_cell_size_m = float(
                self.get_parameter("region_raw_memory_cell_size_m").value
            )
            self.region.enable_debug_vis = bool(
                self.get_parameter("enable_region_debug_vis").value
            )
            self._ensure_model_training_state(self.region)
            self._region_store[self.region.version] = self.region
            b = self.region.bound.detach().cpu().numpy()
            self.get_logger().info(
                "[REGION] created single global region "
                f"bound=[({b[0,0]:.2f},{b[0,1]:.2f}) -> ({b[1,0]:.2f},{b[1,1]:.2f})]"
            )
            return

        if not bool(self.get_parameter("expand_region_with_map").value):
            return

        cur = self.region.bound.detach().cpu().numpy()
        tgt = bound.detach().cpu().numpy()
        grow = np.array(
            [
                [min(cur[0, 0], tgt[0, 0]), min(cur[0, 1], tgt[0, 1]), -0.1],
                [max(cur[1, 0], tgt[1, 0]), max(cur[1, 1], tgt[1, 1]), 0.1],
            ],
            dtype=np.float32,
        )
        if np.max(np.abs(grow[:, :2] - cur[:, :2])) > 1e-3:
            self.region.set_bound(torch.from_numpy(grow))
            self.get_logger().info(
                "[REGION] expanded global region "
                f"to [({grow[0,0]:.2f},{grow[0,1]:.2f}) -> ({grow[1,0]:.2f},{grow[1,1]:.2f})]"
            )

    def _should_train_now(self):
        if self.robot_pos is None:
            return False
        if self._last_train_pose is None:
            return True

        trans_eps = float(self.get_parameter("train_min_translation_m").value)
        rot_eps = float(self.get_parameter("train_min_rotation_rad").value)
        force_period = float(self.get_parameter("train_force_period_sec").value)
        now_sec = float(self.get_clock().now().nanoseconds) * 1e-9

        dtrans = float(np.linalg.norm(self.robot_pos[:2] - self._last_train_pose[:2]))
        dyaw = abs(
            ((self.robot_yaw - self._last_train_yaw + math.pi) % (2.0 * math.pi))
            - math.pi
        )
        if dtrans >= trans_eps or dyaw >= rot_eps:
            return True
        return (now_sec - self._last_train_wall_t) >= force_period

    def train_tick(self):
        if self._train_running or self.region is None:
            return
        if self._latest_world_pts is None or self._latest_robot_pose_for_pts is None:
            return
        if getattr(self.region.model, "network", None) is None:
            return
        if len(self._latest_world_pts) < 50:
            return
        if not self._should_train_now():
            return

        world_pts = np.asarray(self._latest_world_pts, dtype=np.float32).copy()
        robot_pose = np.asarray(self._latest_robot_pose_for_pts, dtype=np.float32).copy()
        region = self.region
        self._ensure_model_training_state(region)
        self._train_running = True

        def _worker():
            try:
                epoch = int(self.get_parameter("train_epoch").value)
                n_samples = int(self.get_parameter("train_samples").value)
                region.last_robot_pose = np.array(
                    [robot_pose[0], robot_pose[1], 0.0], dtype=np.float32
                )

                result = region.get_data_from_lidar(
                    world_pts,
                    robot_pose,
                    device=region.device,
                    n_samples=n_samples,
                )
                if result is None:
                    return
                points, closest = result
                if points.shape[0] == 0:
                    return

                raw = torch.cat([points, closest], dim=1)
                if not torch.isfinite(raw).all():
                    return

                raw_cpu = region._filter_raw_rows_to_bound(raw.detach().cpu())
                if len(raw_cpu) == 0:
                    return

                with self._train_lock:
                    if region.all_raw_frame_data is None:
                        region.all_raw_frame_data = []
                    region.all_raw_frame_data.append(raw_cpu)
                    merged = torch.cat(region.all_raw_frame_data, dim=0)
                    merged = region._filter_raw_rows_to_bound(merged)
                    if len(merged) == 0:
                        return
                    max_raw = max(1, int(region.max_raw_samples))
                    if len(merged) > max_raw:
                        merged = region._spatially_cap_raw_history(merged, max_raw)
                    region.all_raw_frame_data = [merged]

                n14 = region.calculate_speed_and_normal(merged.detach().cpu())
                if n14 is None or len(n14) == 0:
                    return
                n14 = n14.to(region.device)

                region.model.Params["Training"]["Learning Rate"] = region._current_lr()
                self._ensure_model_training_state(region)

                loss = None
                completed = 0
                for _ in range(max(1, epoch)):
                    all_data = n14
                    if len(all_data) > 7000:
                        idx = torch.randperm(len(all_data), device=region.device)[:7000]
                        all_data = all_data[idx]
                    loss_new, _ = region.model.train_core(
                        epoch=1, frame_data=all_data, is_one_frame=True
                    )
                    completed += 1
                    if loss_new is not None:
                        loss = loss_new

                region._sync_model_refs()
                if loss is not None:
                    region.epoch += completed
                    self._last_train_pose = robot_pose.copy()
                    self._last_train_yaw = float(self.robot_yaw)
                    self._last_train_wall_t = (
                        float(self.get_clock().now().nanoseconds) * 1e-9
                    )
                    if region.epoch % 25 == 0:
                        loss_val = (
                            loss.item() if isinstance(loss, torch.Tensor) else float(loss)
                        )
                        self.get_logger().info(
                            f"[TRAIN] epoch={region.epoch} loss={loss_val:.3e} raw={len(region.all_raw_frame_data[0])}"
                        )
                if region.epoch > 0 and region.epoch % 50 == 0:
                    region.save_model()
            except Exception as exc:
                self.get_logger().error(f"[TRAIN] {exc}")
                import traceback
                traceback.print_exc()
            finally:
                self._train_running = False

        threading.Thread(target=_worker, daemon=True).start()

    def _filter_cmd(self, twist: Twist) -> Twist:
        twist = self._apply_turn_guard(twist)
        if bool(self.get_parameter("safety_stop_enabled").value):
            twist = self._apply_safety_stop(twist)
        return twist

    def _apply_turn_guard(self, twist: Twist) -> Twist:
        """Hold turns on the spot where the robot's corners would hit a mapped wall,
        creeping forward or back to more clearance first."""
        turn_r = float(self.get_parameter("robot_turn_radius_m").value)
        if turn_r <= 0.0 or abs(twist.linear.x) > 0.02 or abs(twist.angular.z) < 1e-3:
            return twist
        if self.robot_pos is None:
            return twist
        here = self._clearance_at(self.robot_pos[:2])
        if here is None or here >= turn_r:
            return twist
        heading = np.array([math.cos(self.robot_yaw), math.sin(self.robot_yaw)], dtype=np.float32)
        options = []
        for direction in (1.0, -1.0):
            c = self._clearance_at(self.robot_pos[:2] + direction * 0.10 * heading)
            if c is not None:
                options.append((c, direction))
        best = max(options) if options else None
        twist.angular.z = 0.0
        if best is not None and best[0] > here + 0.005:
            twist.linear.x = 0.05 * best[1]
            self.get_logger().info(
                f"[TURN] {here:.2f}m from a wall is too tight to turn (needs {turn_r:.2f}m); "
                f"creeping {'forward' if best[1] > 0 else 'back'} first",
                throttle_duration_sec=2.0,
            )
        else:
            self.get_logger().warn(
                f"[TURN] {here:.2f}m from a wall is too tight to turn and no room to creep; holding",
                throttle_duration_sec=2.0,
            )
        return twist

    def _apply_safety_stop(self, twist: Twist) -> Twist:
        """Limit linear speed using live scan points in a corridor ahead of the footprint
        in the direction of travel (forward or reverse).

        Rotation is left alone so the robot can still turn away and replan.
        """
        if twist.linear.x == 0.0:
            return twist
        direction = 1.0 if twist.linear.x > 0.0 else -1.0
        pts = self._latest_world_pts
        stale = (time.monotonic() - self._latest_scan_wall_t) > float(
            self.get_parameter("safety_scan_timeout_sec").value
        )
        if pts is None or self.robot_pos is None or stale:
            twist.linear.x = 0.0
            self.get_logger().warn(
                "[SAFETY] no fresh scan; blocking linear motion", throttle_duration_sec=2.0
            )
            return twist

        # Scan points (map frame) -> robot frame.
        c, s = math.cos(self.robot_yaw), math.sin(self.robot_yaw)
        d = pts[:, :2] - self.robot_pos[:2]
        x = direction * (c * d[:, 0] + s * d[:, 1])  # distance along the travel direction
        y = -s * d[:, 0] + c * d[:, 1]
        front = float(
            self.get_parameter("robot_front_m" if direction > 0 else "robot_rear_m").value
        )
        half_w = float(self.get_parameter("robot_half_width_m").value)
        in_corridor = (x > 0.0) & (np.abs(y) < half_w)
        if not in_corridor.any():
            self._safety_blocked = False
            return twist

        gap = float(np.min(x[in_corridor])) - front
        stop = float(self.get_parameter("safety_stop_dist_m").value)
        slow = max(stop + 1e-3, float(self.get_parameter("safety_slow_dist_m").value))
        if gap < stop:
            twist.linear.x = 0.0
            if not self._safety_blocked:
                self.get_logger().warn(
                    f"[SAFETY] obstacle {gap:.2f}m {'ahead' if direction > 0 else 'behind'}; "
                    "stopping linear motion"
                )
            self._safety_blocked = True
        else:
            twist.linear.x *= min(1.0, (gap - stop) / (slow - stop))
            self._safety_blocked = False
        return twist

    def _initial_spin_tick(self) -> bool:
        """Turn one full revolution in place. Returns True while still spinning."""
        if self._spin_done:
            return False
        if self.robot_pos is None:
            return True
        if self._spin_last_yaw is not None:
            dyaw = math.atan2(
                math.sin(self.robot_yaw - self._spin_last_yaw),
                math.cos(self.robot_yaw - self._spin_last_yaw),
            )
            self._spin_accum += abs(dyaw)
        else:
            self.get_logger().info("[SPIN] initial 360 deg spin to cover the lidar blind spot")
        self._spin_last_yaw = self.robot_yaw
        if self._spin_accum >= 2.0 * math.pi:
            self._spin_done = True
            self.navigator.stop_robot()
            self.get_logger().info("[SPIN] done")
            return False
        cmd = Twist()
        cmd.angular.z = float(self.get_parameter("initial_spin_speed").value)
        self.cmd_pub.publish(cmd)
        return True

    def control_tick(self):
        with self._nav_lock:
            if self._initial_spin_tick():
                return
            if self.robot_pos is None or self._path is None or self._current_goal is None:
                self.navigator.stop_robot()
                return
            try:
                if self.navigator.drive_path(self._path):
                    self._controller_path_done = True
            except Exception as exc:
                self.get_logger().warn(f"[CONTROL] {exc}")
                self.navigator.stop_robot()

    def _point_in_collision_slam(self, xy, pad_cells=None):
        if self._slam_arr is None:
            return False
        radius = float(self.get_parameter("collision_radius_m").value)
        if radius >= 0.0:
            return self._point_in_collision_radius(xy, radius)
        if pad_cells is None:
            pad_cells = int(self.get_parameter("collision_pad_cells").value)
        unknown_is_collision = bool(
            self.get_parameter("unknown_is_collision").value
        )
        c = int((float(xy[0]) - float(self._slam_origin[0])) / self._slam_cell_size)
        r = int((float(xy[1]) - float(self._slam_origin[1])) / self._slam_cell_size)
        h, w = self._slam_arr.shape
        for dr in range(-pad_cells, pad_cells + 1):
            for dc in range(-pad_cells, pad_cells + 1):
                rr, cc = r + dr, c + dc
                if not (0 <= rr < h and 0 <= cc < w):
                    if unknown_is_collision:
                        return True
                    continue
                val = int(self._slam_arr[rr, cc])
                if val > 50:
                    return True
                if unknown_is_collision and val < 0:
                    return True
        return False

    def _drop_hole_frontiers(self, arr):
        """Remove frontier points bordering only small enclosed unknown patches."""
        min_cells = int(self.get_parameter("min_frontier_unknown_cells").value)
        pts = getattr(self.gridmap, "slam_frontier_points", None)
        if min_cells <= 0 or pts is None or len(pts) == 0:
            return
        unknown = arr < 0
        labels, n = label(unknown, structure=np.ones((3, 3), bool))
        if n == 0:
            return
        sizes = np.bincount(labels.ravel())
        big = np.zeros_like(unknown)
        big_ids = np.flatnonzero(sizes >= min_cells)
        big_ids = big_ids[big_ids != 0]
        big[np.isin(labels, big_ids)] = True
        near_big = binary_dilation(big, structure=np.ones((3, 3), bool), iterations=2)
        res = float(self._slam_cell_size)
        h, w = arr.shape
        c = np.clip(((pts[:, 0] - self._slam_origin[0]) / res).astype(int), 0, w - 1)
        r = np.clip(((pts[:, 1] - self._slam_origin[1]) / res).astype(int), 0, h - 1)
        keep = near_big[r, c]
        if not keep.all():
            self.gridmap.slam_frontier_points = pts[keep].astype(np.float32, copy=False)
            self.get_logger().info(
                f"[FRONTIER] ignored {int((~keep).sum())} frontier points at small unknown holes "
                f"(< {min_cells} cells) inside the mapped area",
                throttle_duration_sec=30.0,
            )

    def _publish_inflation(self, map_msg: OccupancyGrid):
        """Inflation layer for RViz ("costmap" colour scheme): 100 = wall, 99 = closer than
        collision_radius_m (paths rejected), 50 = where the field is slowest
        (wall_offset_m + wall_min_m)."""
        if self.inflation_pub.get_subscription_count() == 0:
            return
        radius = float(self.get_parameter("collision_radius_m").value)
        if radius < 0.0:
            pad = int(self.get_parameter("collision_pad_cells").value)
            radius = pad * self._slam_cell_size
        clearance = float(self.get_parameter("wall_offset_m").value) + float(
            self.get_parameter("wall_min_m").value
        )
        dist = self._slam_occ_dist
        grid = np.zeros(dist.shape, dtype=np.int8)
        grid[dist < clearance] = 50
        grid[dist < radius] = 99
        grid[self._slam_arr > 50] = 100
        out = OccupancyGrid()
        out.header = map_msg.header
        out.info = map_msg.info
        out.data = grid.ravel().tolist()
        self.inflation_pub.publish(out)

    def _point_in_collision_radius(self, xy, radius):
        arr, dist = self._slam_arr, self._slam_occ_dist
        unknown_is_collision = bool(self.get_parameter("unknown_is_collision").value)
        c = int((float(xy[0]) - float(self._slam_origin[0])) / self._slam_cell_size)
        r = int((float(xy[1]) - float(self._slam_origin[1])) / self._slam_cell_size)
        h, w = arr.shape
        if not (0 <= r < h and 0 <= c < w):
            return unknown_is_collision
        if unknown_is_collision and arr[r, c] < 0:
            return True
        # Occupied cells have distance 0 and always collide, even with radius 0.
        return float(dist[r, c]) < max(radius, 0.5 * self._slam_cell_size)

    def _segment_in_collision_slam(self, a, b, pad_cells=None):
        if self._slam_arr is None:
            return False
        a = np.asarray(a[:2], dtype=np.float32)
        b = np.asarray(b[:2], dtype=np.float32)
        d = float(np.linalg.norm(b - a))
        if d < 1e-6:
            return self._point_in_collision_slam(a, pad_cells=pad_cells)

        sample_step = float(self.get_parameter("path_collision_sample_step_m").value)
        sample_step = max(0.5 * float(self._slam_cell_size), sample_step)
        n = max(2, int(np.ceil(d / sample_step)) + 1)
        ts = np.linspace(0.0, 1.0, n, dtype=np.float32)
        for t in ts:
            p = a + t * (b - a)
            if self._point_in_collision_slam(p, pad_cells=pad_cells):
                return True
        return False

    def _clearance_at(self, xy):
        """Distance (m) from xy to the nearest occupied SLAM cell, or None off-map."""
        dist = getattr(self, "_slam_occ_dist", None)
        if dist is None:
            return None
        c = int((float(xy[0]) - float(self._slam_origin[0])) / self._slam_cell_size)
        r = int((float(xy[1]) - float(self._slam_origin[1])) / self._slam_cell_size)
        if not (0 <= r < dist.shape[0] and 0 <= c < dist.shape[1]):
            return None
        return float(dist[r, c])

    def _path_is_safe_slam(self, pts, pad_cells=None):
        return self._path_collision_reason(pts, pad_cells=pad_cells)[0] is None

    def _path_collision_reason(self, pts, pad_cells=None):
        """(None, None) if the path is clear of SLAM obstacles, else (description of the
        hit, index of the first point that is unsafe or ends an unsafe segment)."""
        arr = np.asarray(pts, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 2 or len(arr) < 2:
            return "path has fewer than 2 points", 0
        skip_m = float(self.get_parameter("collision_check_skip_m").value)
        near_start = np.linalg.norm(arr - arr[0], axis=1) < skip_m
        along = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(arr, axis=0), axis=1))])
        start_clear = self._clearance_at(arr[0])
        for i in range(len(arr)):
            if near_start[i]:
                # Near the start, only moving closer to a wall than the robot already is
                # counts: the robot may leave a tight spot but not drive further into it.
                clear = self._clearance_at(arr[i])
                if (start_clear is not None and clear is not None
                        and self._point_in_collision_slam(arr[i], pad_cells=pad_cells)
                        and clear < start_clear - 0.02):
                    return (
                        f"point {i}/{len(arr)} at ({arr[i, 0]:.2f},{arr[i, 1]:.2f}), "
                        f"{along[i]:.2f}m along, moves closer to a wall ({clear:.2f}m) "
                        f"than the robot is ({start_clear:.2f}m)"
                    ), i
                continue
            if self._point_in_collision_slam(arr[i], pad_cells=pad_cells):
                return (
                    f"point {i}/{len(arr)} at ({arr[i, 0]:.2f},{arr[i, 1]:.2f}), "
                    f"{along[i]:.2f}m along, too close to an occupied cell"
                ), i
        for i in range(len(arr) - 1):
            if near_start[i] or near_start[i + 1]:
                continue
            if self._segment_in_collision_slam(arr[i], arr[i + 1], pad_cells=pad_cells):
                return (
                    f"segment {i}->{i + 1} ({along[i]:.2f}m along) passes too close to an occupied cell"
                ), i + 1
        return None, None

    def pick_reachable_frontier(self):
        if self.robot_pos is None or self.gridmap is None:
            self._frontier_pick_stats = {"source": "none", "raw": 0}
            return None

        if not self.gridmap.has_raw_slam_frontiers():
            self._frontier_pick_stats = {"source": "gridmap", "raw": 0}
            self.get_logger().info(
                "[NAV] no frontiers left in the map: exploration complete",
                throttle_duration_sec=15.0,
            )
            return None

        robot = self.robot_pos[:2]
        cand = np.asarray(self.gridmap.slam_frontier_points, dtype=np.float32)
        if cand.ndim == 1 and cand.size == 2:
            cand = cand.reshape(1, 2)
        if cand.size == 0:
            self._frontier_pick_stats = {"source": "gridmap", "raw": 0}
            return None

        stats = {"source": "gridmap", "raw": int(len(cand))}
        d = np.linalg.norm(cand - robot[None, :], axis=1)

        # Farther than goal_reached_dist, or a goal counts as reached the moment it is picked.
        min_standoff = max(
            0.15, float(self.get_parameter("goal_reached_dist").value) + 0.10
        )
        keep = d >= min_standoff
        cand = cand[keep]
        d = d[keep]
        stats["after_standoff"] = int(len(cand))
        if len(cand) == 0:
            self._frontier_pick_stats = stats
            return None

        ttl = float(self.get_parameter("abandoned_goal_ttl_sec").value)
        now = time.monotonic()
        self._abandoned_goals = [(g, t) for g, t in self._abandoned_goals if now - t < ttl]

        def not_blacklisted(cand):
            blocked = list(self._visited_goals) + [g for g, _ in self._abandoned_goals]
            if not blocked:
                return np.ones(len(cand), dtype=bool)
            bl = np.asarray(blocked, dtype=np.float32)
            min_vd = np.min(np.linalg.norm(cand[:, None, :] - bl[None, :, :], axis=2), axis=1)
            return min_vd >= self._goal_blacklist_radius

        keep = not_blacklisted(cand)
        if not keep.any() and self._abandoned_goals:
            # Every frontier is blacklisted: give abandoned goals another try rather than
            # stalling. Reached goals stay blacklisted, or the robot loops on them.
            self.get_logger().info(
                f"[NAV] all {len(cand)} frontier points blacklisted; retrying "
                f"{len(self._abandoned_goals)} abandoned goals"
            )
            self._abandoned_goals = []
            keep = not_blacklisted(cand)
        if not keep.any():
            self.get_logger().info(
                f"[NAV] all {len(cand)} remaining frontier points are next to goals already "
                "reached; nothing left to explore from here",
                throttle_duration_sec=10.0,
            )
        cand = cand[keep]
        d = d[keep]
        stats["after_blacklist"] = int(len(cand))
        if len(cand) and bool(self.get_parameter("enable_path_validation").value):
            # A frontier hugging a wall can never pass path validation; skip it up front
            # instead of failing on it repeatedly.
            keep = np.array([not self._point_in_collision_slam(p) for p in cand], dtype=bool)
            cand = cand[keep]
            d = d[keep]
            stats["after_clearance"] = int(len(cand))
        if len(cand) and float(self.get_parameter("route_subgoal_dist_m").value) > 0.0:
            # Frontiers the robot cannot drive to (e.g. space outside the maze seen through
            # gaps between wall panels) would be chased forever; keep only reachable ones.
            reach = self._reachable_from_robot()
            if reach is not None:
                keep = np.array([reach(p) for p in cand], dtype=bool)
                cand = cand[keep]
                d = d[keep]
                stats["after_reachable"] = int(len(cand))
        if len(cand) == 0:
            self._frontier_pick_stats = stats
            return None

        # Distance cap last, so blacklisted or wall-hugging frontiers nearby cannot hide
        # reachable ones farther away.
        max_goal_dist = float(self.get_parameter("frontier_max_goal_dist_m").value)
        keep = d <= max_goal_dist
        if not keep.any():
            # Nothing within the cap: head for the nearest frontiers instead of stopping.
            # Paths are driven in short committed pieces, so a far goal is still safe.
            keep = d <= float(d.min()) + 1.0
            self.get_logger().info(
                f"[NAV] no frontier within frontier_max_goal_dist_m={max_goal_dist:.1f}m; "
                f"using the nearest ones ({float(d.min()):.1f}m)",
                throttle_duration_sec=10.0,
            )
        cand = cand[keep]
        d = d[keep]
        stats["after_distcap"] = int(len(cand))
        self._frontier_pick_stats = stats


        try:
            from sklearn.cluster import DBSCAN

            best_rep = None
            best_score = -1.0
            labels = DBSCAN(eps=0.30, min_samples=1).fit(cand).labels_
            for label in np.unique(labels):
                members = cand[labels == label]
                centroid = members.mean(axis=0)
                rep = members[
                    np.argmin(np.linalg.norm(members - centroid[None, :], axis=1))
                ]
                d_rep = float(np.linalg.norm(rep - robot))
                score = (float(len(members)) ** 0.5) / math.log(d_rep + math.e)
                if score > best_score:
                    best_score = score
                    best_rep = rep
            if best_rep is not None:
                return best_rep.astype(np.float32)
        except Exception:
            pass

        return cand[np.argmin(d)].astype(np.float32)

    def _publish_goal_marker(self):
        self.markers.publish_goal_marker()

    def _publish_frontier_markers(self):
        if self.gridmap is None or self.gridmap.slam_raw_data is None:
            return

        now = self.get_clock().now().to_msg()
        raw = self.gridmap.slam_raw_data
        slam_h, slam_w = raw.shape
        occ = OccupancyGrid()
        occ.header.frame_id = "map"
        occ.header.stamp = now
        occ.info.map_load_time = now
        occ.info.resolution = float(self.gridmap.slam_raw_resolution)
        occ.info.width = int(slam_w)
        occ.info.height = int(slam_h)
        occ.info.origin.position.x = float(self.gridmap.slam_raw_origin[0])
        occ.info.origin.position.y = float(self.gridmap.slam_raw_origin[1])
        occ.info.origin.orientation.w = 1.0

        flat = np.full(slam_h * slam_w, -1, dtype=np.int8)
        if self.gridmap.has_raw_slam_frontiers() and len(self.gridmap.slam_frontier_points) > 0:
            pts = self.gridmap.slam_frontier_points
            res = self.gridmap.slam_raw_resolution
            ox, oy = self.gridmap.slam_raw_origin
            sc = ((pts[:, 0] - ox) / res).astype(np.int32)
            sr = ((pts[:, 1] - oy) / res).astype(np.int32)
            valid = (sc >= 0) & (sc < slam_w) & (sr >= 0) & (sr < slam_h)
            flat[sr[valid] * slam_w + sc[valid]] = 100
        occ.data = flat.tolist()
        self.frontier_occ_pub.publish(occ)

        self.markers.publish_frontier_markers()
        if self._current_goal is not None:
            self.markers.publish_goal_marker()

    def _truncate_path_by_arclen(self, pts, max_len_m):
        arr = np.asarray(pts, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 2 or len(arr) < 2:
            return arr
        if max_len_m <= 0.0:
            return arr[:2].copy()

        out = [arr[0].copy()]
        acc = 0.0
        for i in range(1, len(arr)):
            a = arr[i - 1]
            b = arr[i]
            seg = float(np.linalg.norm(b - a))
            if seg < 1e-6:
                continue
            if acc + seg <= max_len_m:
                out.append(b.copy())
                acc += seg
                continue
            remain = max_len_m - acc
            if remain > 1e-4:
                t = remain / seg
                out.append((a + t * (b - a)).astype(np.float32))
            break

        if len(out) < 2:
            out.append(arr[1].copy())
        return np.asarray(out, dtype=np.float32)

    def decimate_path(self, path_xy, spacing=0.4):
        pts = np.asarray(path_xy, dtype=np.float32)
        if pts.ndim == 1:
            pts = pts.reshape(1, 2)
        if len(pts) < 2:
            return pts
        new_pts = [pts[0]]
        for i in range(1, len(pts)):
            if np.linalg.norm(pts[i] - new_pts[-1]) >= spacing:
                new_pts.append(pts[i])
        if np.linalg.norm(new_pts[-1] - pts[-1]) > 0.05:
            new_pts.append(pts[-1])
        return np.asarray(new_pts, dtype=np.float32)

    def smooth_path(self, path_xy, window=5):
        if path_xy is None:
            return None
        pts = np.asarray(path_xy, dtype=np.float32)
        if pts.ndim == 1:
            pts = pts.reshape(1, 2)
        if len(pts) < window:
            return pts
        sm = pts.copy()
        k = window // 2
        for i in range(len(pts)):
            a = max(0, i - k)
            b = min(len(pts), i + k + 1)
            sm[i] = pts[a:b].mean(axis=0)
        sm[0] = pts[0]
        sm[-1] = pts[-1]
        return sm

    def _sanitize_path(self, robot_xy, goal_xy, pts, append_goal=True):
        arr = np.asarray(pts, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 2 or len(arr) < 2:
            return None

        d0 = float(np.linalg.norm(arr[0] - robot_xy))
        d1 = float(np.linalg.norm(arr[-1] - robot_xy))
        if d1 < d0:
            arr = arr[::-1].copy()

        if float(np.linalg.norm(arr[0] - robot_xy)) > 0.05:
            arr = np.vstack([robot_xy.astype(np.float32), arr])
        else:
            arr[0] = robot_xy.astype(np.float32)

        gap_to_goal = float(np.linalg.norm(arr[-1] - goal_xy))
        if not append_goal or gap_to_goal > 0.30:
            # A path that stops short of the goal (cut at unmapped space, or a field
            # rollout that never reached it) stays short: a straight leg to the goal
            # would cross whatever made it stop.
            pass
        elif gap_to_goal > 0.10:
            arr = np.vstack([arr, goal_xy.astype(np.float32)])
        else:
            arr[-1] = goal_xy.astype(np.float32)

        goal_dir = goal_xy.astype(np.float32) - robot_xy.astype(np.float32)
        goal_norm = float(np.linalg.norm(goal_dir))
        if goal_norm < 1e-6:
            return arr
        goal_dir /= goal_norm

        first_step = None
        for i in range(1, len(arr)):
            step = arr[i] - arr[i - 1]
            step_norm = float(np.linalg.norm(step))
            if step_norm > 0.05:
                first_step = step / step_norm
                break

        if first_step is None:
            return None
        if (bool(self.get_parameter("require_goalward_start").value)
                and float(np.dot(first_step, goal_dir)) < 0.0):
            return None
        return arr

    def _trim_at_unvisited(self, pts):
        arr = np.asarray(pts, dtype=np.float32)
        if self._slam_arr is None:
            return arr

        cell = self._slam_cell_size
        ox = float(self._slam_origin[0])
        oy = float(self._slam_origin[1])
        h, w = self._slam_arr.shape

        gx = ((arr[:, 0] - ox) / cell).astype(int)
        gy = ((arr[:, 1] - oy) / cell).astype(int)
        inb = (0 <= gx) & (gx < w) & (0 <= gy) & (gy < h)

        trim_at_unknown = bool(self.get_parameter("trim_at_unknown").value)
        traversable = np.ones(len(arr), dtype=bool)
        traversable[~inb] = False
        if trim_at_unknown:
            traversable[inb] = self._slam_arr[gy[inb], gx[inb]] == 0
        else:
            traversable[inb] = self._slam_arr[gy[inb], gx[inb]] <= 50

        blocked = np.where(~traversable)[0]
        if len(blocked) == 0 or blocked[0] <= 1:
            return arr
        trimmed = arr[: blocked[0]]
        return trimmed if len(trimmed) >= 2 else None

    def publish_field_path(self, path_xy, frame_id="map"):
        pts = np.asarray(path_xy, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 2:
            return
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        poses = []
        for p in pts:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = float(p[0])
            ps.pose.position.y = float(p[1])
            ps.pose.orientation.w = 1.0
            poses.append(ps)
        msg.poses = poses
        self.field_path_pub.publish(msg)
        self.field_path_pub_alt.publish(msg)

    def _publish_nav_path(self, path_xy, frame_id="map"):
        pts = np.asarray(path_xy, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 2:
            return
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        poses = []
        for p in pts:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = float(p[0])
            ps.pose.position.y = float(p[1])
            ps.pose.orientation.w = 1.0
            poses.append(ps)
        msg.poses = poses
        self.nav_path_pub.publish(msg)

    def _accept_raw_path(
        self,
        robot_xy,
        path_goal,
        raw,
        forward_only,
        label="field",
        frontier_goal=None,
    ):
        if raw is None or len(raw) <= 1:
            self.get_logger().info(f"[PATH] rejecting {label} path: field produced no path")
            return False
        enable_path_validation = bool(
            self.get_parameter("enable_path_validation").value
        )
        raw_arr = np.asarray(raw, dtype=np.float32)
        decim = self.decimate_path(raw_arr, spacing=0.10)
        smooth = self.smooth_path(decim, window=3)
        if smooth is None or len(smooth) < 2:
            self.get_logger().info(f"[PATH] rejecting {label} path: too short after smoothing")
            return False
        trimmed = False
        if enable_path_validation:
            before = len(smooth)
            smooth = self._trim_at_unvisited(smooth)
            if smooth is None or len(smooth) < 2:
                self.get_logger().info(
                    f"[PATH] rejecting {label} path: cut to <2 points at the first cell "
                    f"SLAM has not mapped as free (trim_at_unknown, {before} points before)"
                )
                return False
            trimmed = len(smooth) < before
        # A path cut at unmapped space must end at the cut: appending the goal would
        # add a straight leg through exactly the space the cut excluded.
        smooth = self._sanitize_path(robot_xy, path_goal, smooth, append_goal=not trimmed)
        if smooth is None or len(smooth) < 2:
            self.get_logger().info(f"[PATH] rejecting {label} path: dropped by sanitize")
            return False

        full_path = np.asarray(smooth, dtype=np.float32)
        check_path = full_path
        horizon = float(self.get_parameter("validation_horizon_m").value)
        if horizon > 0.0:
            check_path = self._truncate_path_by_arclen(full_path, horizon)
        reason, bad_i = (
            self._path_collision_reason(check_path) if enable_path_validation else (None, None)
        )
        goal_approach = False
        if reason is not None:
            approach_m = float(self.get_parameter("goal_approach_m").value)
            bad_xy = check_path[min(bad_i, len(check_path) - 1)]
            near_goal = float(np.linalg.norm(bad_xy - np.asarray(path_goal[:2], dtype=np.float32)))
            to_frontier = frontier_goal is None or float(np.linalg.norm(
                np.asarray(path_goal[:2], dtype=np.float32) - np.asarray(frontier_goal[:2], dtype=np.float32)
            )) < 1e-3
            if (label == "field" and to_frontier and approach_m > 0.0 and bad_i
                    and near_goal <= approach_m):
                # Only the approach to the goal is too close to a wall: stop short of it.
                safe = full_path[:bad_i]
                safe_len = (
                    float(np.sum(np.linalg.norm(np.diff(safe, axis=0), axis=1)))
                    if len(safe) >= 2 else 0.0
                )
                if safe_len < 0.15:
                    self.get_logger().info(
                        f"[NAV] frontier ({path_goal[0]:.2f},{path_goal[1]:.2f}) reached as close "
                        "as collision_radius_m allows"
                    )
                    self._stop_and_clear_goal(reached=True)
                    return True
                short_m = float(np.linalg.norm(safe[-1] - np.asarray(path_goal[:2], dtype=np.float32)))
                self.get_logger().info(
                    f"[PATH] {label} path ends too close to a wall ({reason}); "
                    f"stopping {short_m:.2f}m short of the goal at the last safe point"
                )
                full_path = safe
                goal_approach = True
            else:
                self.get_logger().info(
                    f"[PATH] rejecting {label} path after collision validation: {reason}"
                )
                return False
        self._publish_nav_path(full_path)
        full_len = float(np.sum(np.linalg.norm(np.diff(full_path, axis=0), axis=1)))
        commit_trigger = float(self.get_parameter("path_truncate_trigger_m").value)
        commit_max = float(self.get_parameter("path_commit_max_m").value)
        exec_path = full_path
        truncated = False
        if full_len > commit_trigger:
            exec_path = self._truncate_path_by_arclen(full_path, commit_max)
            truncated = len(exec_path) >= 2 and len(exec_path) < len(full_path)
        if enable_path_validation and not self._path_is_safe_slam(exec_path):
            self.get_logger().info(
                f"[PATH] rejecting {label} exec path after truncation collision validation"
            )
            return False
        exec_reach = float(np.linalg.norm(np.asarray(exec_path[-1]) - np.asarray(robot_xy[:2])))
        if exec_reach < 0.20 and float(np.linalg.norm(np.asarray(path_goal[:2]) - robot_xy[:2])) > 0.50:
            # The committed piece folds back to the robot (field rollout dithering in a
            # local minimum); following it would end at once and replan the same thing.
            self.get_logger().info(
                f"[PATH] rejecting {label} path: committed piece ends {exec_reach:.2f}m from the "
                "robot (no progress)"
            )
            return False

        # Reaching the end of a stopped-short path means the goal is as reached as it gets.
        self._path_is_goal_approach = goal_approach and not truncated
        with self._nav_lock:
            self._path = exec_path
            self._active_path_goal = np.asarray(path_goal[:2], dtype=np.float32).copy()
            self.navigator.reset_path()
            self._controller_path_done = False
        self._path_total_dist = float(
            np.sum(np.linalg.norm(np.diff(self._path, axis=0), axis=1))
        )
        self._midpoint_replanned = False
        self._best_goal_dist = float(np.linalg.norm(robot_xy - self._active_path_goal))
        self._last_progress_time = float(self.get_clock().now().nanoseconds) * 1e-9
        self._last_path_progress_s = 0.0
        self._pathgen_fail_count = 0
        self.region.latest_path = full_path.tolist()
        self.region.latest_goal = self._active_path_goal.copy()
        self.publish_field_path(self._path)
        goal_for_log = frontier_goal if frontier_goal is not None else path_goal
        self.get_logger().info(
            f"[PATH] {label} ({'fwd' if forward_only else 'bi'}) accepted "
            f"wps={len(self._path)} raw={len(raw_arr)} full_len={full_len:.2f}m "
            f"truncated={truncated} path_goal=({path_goal[0]:.2f},{path_goal[1]:.2f}) "
            f"frontier=({goal_for_log[0]:.2f},{goal_for_log[1]:.2f})"
        )
        return True

    def _clear_active_path(self):
        with self._nav_lock:
            self._path = None
            self._active_path_goal = None
            self.navigator.reset_path()
            self._controller_path_done = False

    def _stop_and_clear_goal(self, reached=True):
        if reached and self._current_goal is not None:
            self._visited_goals.append(self._current_goal[:2].copy())
        with self._nav_lock:
            self.navigator.stop_robot()
            self._clear_active_path()
            self._current_goal = None
        self.goal_xy = None
        self._path_total_dist = None
        self._midpoint_replanned = False
        self._best_goal_dist = float("inf")

    def exploration_tick(self):
        self._exploration_tick_count += 1
        log_now = self._exploration_tick_count % 10 == 0

        self._update_robot_pose_from_tf()
        if log_now:
            self._warn_if_slam_stale()
        if self.robot_pos is None:
            if log_now:
                self.get_logger().info("[NAV] waiting for robot pose")
            return
        if self.region is None:
            if log_now:
                self.get_logger().info("[NAV] waiting for first SLAM map")
            return
        if not self._spin_done:
            # control_tick owns the velocity output during the initial spin.
            return

        robot_xy = self.robot_pos[:2]
        if self._active_path_goal is not None:
            self.region.latest_goal = self._active_path_goal.copy()
        else:
            self.region.latest_goal = self._current_goal.copy() if self._current_goal is not None else None

        if self._path is not None and len(self._path) >= 2:
            self.publish_field_path(self._path)

            if self._current_goal is None:
                self._clear_active_path()
                return

            now_sec = float(self.get_clock().now().nanoseconds) * 1e-9
            nav_goal = (
                self._active_path_goal
                if self._active_path_goal is not None
                else self._current_goal
            )
            d_goal = float(np.linalg.norm(robot_xy - nav_goal))
            d_frontier = float(np.linalg.norm(robot_xy - self._current_goal))
            d_path_end = float(np.linalg.norm(robot_xy - self._path[-1]))

            progress_eps = float(self.get_parameter("follow_progress_eps").value)
            stuck_timeout = float(self.get_parameter("follow_stuck_timeout_sec").value)
            eff_eps = min(progress_eps, max(0.01, 0.05 * d_goal))
            path_progress_s = float(getattr(self.navigator, "last_proj_s", 0.0))
            path_advanced = path_progress_s > (self._last_path_progress_s + 0.08)

            if d_goal + eff_eps < self._best_goal_dist or d_goal + 0.01 < self._best_goal_dist:
                self._best_goal_dist = d_goal
                self._last_progress_time = now_sec
                self._stuck_replan_count = 0
            elif path_advanced:
                self._last_path_progress_s = path_progress_s
                self._last_progress_time = now_sec

            if log_now:
                self.get_logger().info(
                    f"[FOLLOWING] d_nav={d_goal:.2f}m d_frontier={d_frontier:.2f}m "
                    f"d_path_end={d_path_end:.2f}m path_wps={len(self._path)} epoch={self.region.epoch}"
                )

            goal_reached_dist = float(self.get_parameter("goal_reached_dist").value)

            if d_frontier < goal_reached_dist:
                self.get_logger().info(
                    f"[NAV] goal reached ({self._current_goal[0]:.2f},{self._current_goal[1]:.2f})"
                )
                self._stop_and_clear_goal(reached=True)
                return

            if d_path_end < 0.20 and d_frontier > 0.50:
                self.get_logger().info(
                    f"[PATH EXTEND] local target done, {d_frontier:.2f}m to frontier"
                )
                self._clear_active_path()
                self._stuck_replan_count = 0
                self._best_goal_dist = float("inf")
                return

            if self._controller_path_done:
                self._controller_path_done = False
                if d_frontier < 0.35 or getattr(self, "_path_is_goal_approach", False):
                    self._path_is_goal_approach = False
                    self._stop_and_clear_goal(reached=True)
                    return
                self.get_logger().warn(
                    f"[FOLLOWING] path endpoint reached but frontier still {d_frontier:.2f}m away; replanning"
                )
                self._clear_active_path()
                self._stuck_replan_count = 0
                self._best_goal_dist = float("inf")
                return

            adaptive_timeout = stuck_timeout * (2.0 if d_goal < 1.0 else 1.0)
            if now_sec - self._last_progress_time > adaptive_timeout:
                self._stuck_replan_count += 1
                self.get_logger().warn(
                    f"[FOLLOWING] no goal progress for {adaptive_timeout:.1f}s; forcing replan "
                    f"(attempt {self._stuck_replan_count})"
                )
                self._clear_active_path()
                self._last_progress_time = now_sec
                if self._stuck_replan_count >= 3 and self._current_goal is not None:
                    self.get_logger().warn(
                        f"[NAV] abandoning unreachable frontier ({self._current_goal[0]:.2f},{self._current_goal[1]:.2f})"
                    )
                    self._abandon_goal(self._current_goal)
                    self._current_goal = None
                    self.goal_xy = None
                    self._stuck_replan_count = 0
                return

            if (
                self._path_total_dist is not None
                and self._path_total_dist > 2.0
                and not self._midpoint_replanned
                and d_goal < self._path_total_dist * 0.5
            ):
                self._midpoint_replanned = True
                self.get_logger().info(
                    f"[REPLAN] halfway triggered at d_goal={d_goal:.2f}m"
                )
                self._clear_active_path()
                return

            return

        self._clear_active_path()

        goal = self._current_goal.copy() if self._current_goal is not None else self.pick_reachable_frontier()
        if goal is None:
            if log_now:
                self.get_logger().info(
                    f"[NAV] no reachable frontier stats={self._frontier_pick_stats}"
                )
            self.navigator.stop_robot()
            return

        min_train_epochs = int(self.get_parameter("min_train_epochs_for_nav").value)
        if self.region.epoch < min_train_epochs:
            if log_now:
                self.get_logger().info(
                    f"[NAV] warmup waiting epoch={self.region.epoch} < {min_train_epochs}"
                )
            self.navigator.stop_robot()
            self._current_goal = None
            self.goal_xy = None
            self._pathgen_fail_count = 0
            self._stuck_replan_count = 0
            return

        new_goal = self._current_goal is None
        if new_goal:
            self._stuck_replan_count = 0
            self._pathgen_fail_count = 0
        self._current_goal = goal
        self.goal_xy = goal.copy()
        self.region.latest_goal = goal.copy()
        self._publish_goal_marker()

        self.get_logger().info(
            f"[NAV] {'picked frontier' if new_goal else 'replanning to frontier'} "
            f"({goal[0]:.2f},{goal[1]:.2f}) epoch={self.region.epoch}"
        )

        use_field = (
            self.region.epoch >= min_train_epochs
            and getattr(self.region.model, "function", None) is not None
        )
        allow_straight_line_fallback = bool(
            self.get_parameter("allow_straight_line_fallback").value
        )

        if use_field:
            b = self.region.bound.detach().cpu().numpy()
            forward_only = not (
                b[0, 0] <= goal[0] <= b[1, 0] and b[0, 1] <= goal[1] <= b[1, 1]
            )
            plan_goal = self._route_subgoal(robot_xy, goal)
            raw = self.region.generate_field_path(
                robot_xy,
                plan_goal,
                step_size=0.10,
                max_steps=220,
                forward_only=forward_only,
            )
            if self._accept_raw_path(
                robot_xy, plan_goal, raw, forward_only, label="field", frontier_goal=goal
            ):
                return

            safe_wp = self.region.find_safe_waypoint(
                robot_xy, plan_goal, collision_fn=self._point_in_collision_slam
            )
            if safe_wp is not None:
                raw = self.region.generate_field_path(
                    robot_xy,
                    safe_wp,
                    step_size=0.10,
                    max_steps=220,
                    forward_only=False,
                )
                if self._accept_raw_path(
                    robot_xy,
                    safe_wp,
                    raw,
                    False,
                    label="safe-waypoint",
                    frontier_goal=goal,
                ):
                    return

            self.get_logger().info(
                "[PATH] no field-derived local path accepted; holding position for more training"
            )
            self.navigator.stop_robot()
            self._count_path_failure(goal)
            return

        if allow_straight_line_fallback:
            sl = straight_line_path(
                robot_xy,
                goal,
                step=0.10,
                collision_fn=self._point_in_collision_slam,
            )
            sl_len = (
                float(np.sum(np.linalg.norm(np.diff(sl, axis=0), axis=1)))
                if len(sl) > 1
                else 0.0
            )
            if len(sl) >= 3 and sl_len >= 0.15:
                self._path = sl
                self._active_path_goal = goal.copy()
                self.navigator.reset_path()
                self._controller_path_done = False
                self._path_total_dist = sl_len
                self._midpoint_replanned = False
                self._best_goal_dist = float(np.linalg.norm(robot_xy - self._active_path_goal))
                self._last_progress_time = float(self.get_clock().now().nanoseconds) * 1e-9
                self._last_path_progress_s = 0.0
                self.publish_field_path(self._path)
                self.get_logger().info(
                    f"[PATH] straight-line fallback wps={len(sl)} len={sl_len:.2f}m"
                )
                return

        self._count_path_failure(goal)

    def _reachable_from_robot(self, approach_m=0.6):
        """Function xy -> True if a clearance-respecting route over mapped-free cells
        gets within approach_m of xy (close enough to map it), or None without a map."""
        arr, dist = self._slam_arr, getattr(self, "_slam_occ_dist", None)
        if arr is None or dist is None or arr.shape != dist.shape or self.robot_pos is None:
            return None
        res = float(self._slam_cell_size)
        ox, oy = float(self._slam_origin[0]), float(self._slam_origin[1])
        h, w = arr.shape
        radius = float(self.get_parameter("collision_radius_m").value)
        if radius < 0.0:
            radius = int(self.get_parameter("collision_pad_cells").value) * res
        sr = int(np.clip((self.robot_pos[1] - oy) / res, 0, h - 1))
        sc = int(np.clip((self.robot_pos[0] - ox) / res, 0, w - 1))
        yy, xx = np.mgrid[0:h, 0:w]
        near_start = np.hypot(yy - sr, xx - sc) * res < max(
            0.35, float(self.get_parameter("collision_check_skip_m").value))
        ok = ((dist >= radius) | near_start) & (arr >= 0) & (arr <= 50)
        try:
            costs, _ = MCP_Geometric(np.where(ok, 1.0, np.inf)).find_costs([(sr, sc)])
        except Exception as exc:
            self.get_logger().warn(f"[ROUTE] reachability failed: {exc}", throttle_duration_sec=10.0)
            return None
        reach_dist = distance_transform_edt(~np.isfinite(costs)) * res

        def reachable(xy):
            r = int((xy[1] - oy) / res)
            c = int((xy[0] - ox) / res)
            return 0 <= r < h and 0 <= c < w and reach_dist[r, c] <= approach_m

        return reachable

    def _route_subgoal(self, robot_xy, goal):
        """Point route_subgoal_dist_m along the SLAM-map route to goal, or goal itself.

        The route is a minimum-cost path over cells at least collision_radius_m from
        walls (except right around the robot, which may start close to one), preferring
        the middle of corridors. Mapped-free cells first; if that fails, unknown cells
        too, but only near the goal (a frontier borders unknown space): elsewhere unknown
        cells are often the inside of obstacles whose outline has gaps.
        """
        hop = float(self.get_parameter("route_subgoal_dist_m").value)
        arr, dist = self._slam_arr, getattr(self, "_slam_occ_dist", None)
        if hop <= 0.0 or arr is None or dist is None or arr.shape != dist.shape:
            return goal
        res = float(self._slam_cell_size)
        ox, oy = float(self._slam_origin[0]), float(self._slam_origin[1])
        h, w = arr.shape

        def cell(xy):
            return (int(np.clip((xy[1] - oy) / res, 0, h - 1)),
                    int(np.clip((xy[0] - ox) / res, 0, w - 1)))

        radius = float(self.get_parameter("collision_radius_m").value)
        if radius < 0.0:
            radius = int(self.get_parameter("collision_pad_cells").value) * res
        start, end = cell(robot_xy), cell(goal)
        yy, xx = np.mgrid[0:h, 0:w]
        near_start = np.hypot(yy - start[0], xx - start[1]) * res < max(
            0.35, float(self.get_parameter("collision_check_skip_m").value))
        # Cheaper in the middle of corridors than along the clearance limit.
        base = 1.0 + 2.0 * np.clip((radius + 0.30 - dist) / 0.30, 0.0, 1.0)
        clear = (dist >= radius) | near_start
        near_goal = np.hypot(yy - end[0], xx - end[1]) * res < 1.0
        for allow_unknown in (False, True):
            ok = clear & (arr <= 50) & ((arr >= 0) | (allow_unknown & near_goal))
            ok[end] = True
            cost = np.where(ok, base * np.where(arr < 0, 3.0, 1.0), np.inf)
            try:
                mcp = MCP_Geometric(cost)
                costs, _ = mcp.find_costs([start], [end])
                if not np.isfinite(costs[end]):
                    continue
                route = np.array(mcp.traceback(end), dtype=np.float32)
            except Exception as exc:
                self.get_logger().warn(f"[ROUTE] failed: {exc}", throttle_duration_sec=10.0)
                return goal
            pts = np.stack([ox + (route[:, 1] + 0.5) * res, oy + (route[:, 0] + 0.5) * res], axis=1)
            along = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
            if along[-1] <= hop + 0.30:
                return goal
            sub = pts[int(np.searchsorted(along, hop))]
            self.get_logger().info(
                f"[ROUTE] frontier ({goal[0]:.2f},{goal[1]:.2f}) is {along[-1]:.1f}m away by map route"
                f"{' (via unknown cells near it)' if allow_unknown else ''}; field plans to "
                f"({sub[0]:.2f},{sub[1]:.2f}) first",
                throttle_duration_sec=2.0,
            )
            return sub.astype(np.float32)
        self.get_logger().info(
            f"[ROUTE] no map route to frontier ({goal[0]:.2f},{goal[1]:.2f}); planning straight to it",
            throttle_duration_sec=5.0,
        )
        return goal

    def _count_path_failure(self, goal):
        self._pathgen_fail_count += 1
        if self._pathgen_fail_count >= 5:
            self.get_logger().warn(
                f"[NAV] abandoning frontier ({goal[0]:.2f},{goal[1]:.2f}) after repeated path failures"
            )
            self._abandon_goal(goal)
            self._current_goal = None
            self.goal_xy = None
            self._pathgen_fail_count = 0

    def _abandon_goal(self, goal):
        self._abandoned_goals.append((np.asarray(goal[:2], dtype=np.float32).copy(), time.monotonic()))

    def vis_tick(self):
        if self.region is not None and self.region.enable_debug_vis:
            if self.region.last_robot_pose is not None and self.region.model.function is not None:
                try:
                    src = self.region.last_robot_pose
                    self.region.save_debug_plot(src, view_radius=3.5, robot_xy=src[:2])
                except Exception as exc:
                    self.get_logger().warn(f"[VIS] debug plot failed: {exc}")

    def _slam_free_mask(self, X_w, Y_w):
        """True where the SLAM map marks the cell as free (unknown/occupied -> False)."""
        if self._slam_arr is None:
            return np.zeros(X_w.shape, dtype=bool)
        h, w = self._slam_arr.shape
        ix = np.floor((X_w - self._slam_origin[0]) / self._slam_cell_size).astype(np.int64)
        iy = np.floor((Y_w - self._slam_origin[1]) / self._slam_cell_size).astype(np.int64)
        inside = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
        vals = np.full(X_w.shape, -1, dtype=np.int16)
        vals[inside] = self._slam_arr[iy[inside], ix[inside]]
        return (vals >= 0) & (vals < 50)

    def _make_field_cloud(self, xs, ys, values, stamp):
        """Flat PointCloud2 (x, y, z, intensity) in the map frame."""
        pts = np.zeros(len(xs), dtype=[
            ("x", np.float32), ("y", np.float32), ("z", np.float32), ("intensity", np.float32)
        ])
        pts["x"] = xs
        pts["y"] = ys
        pts["z"] = 0.01
        pts["intensity"] = values
        msg = PointCloud2()
        msg.header.frame_id = "map"
        msg.header.stamp = stamp
        msg.height = 1
        msg.width = len(pts)
        msg.fields = [
            PointField(name=name, offset=4 * i, datatype=PointField.FLOAT32, count=1)
            for i, name in enumerate(("x", "y", "z", "intensity"))
        ]
        msg.is_bigendian = False
        msg.point_step = 16
        msg.row_step = 16 * len(pts)
        msg.is_dense = True
        msg.data = pts.tobytes()
        return msg

    def field_vis_tick(self):
        """Publish the travel-time field from the robot (and the speed field) for RViz."""
        if self.region is None or self.robot_pos is None:
            return
        if self.travel_time_pub.get_subscription_count() == 0 and \
                self.speed_pub.get_subscription_count() == 0:
            return
        radius = float(self.get_parameter("field_vis_radius_m").value)
        res = float(self.get_parameter("field_vis_resolution_m").value)
        n = int(2.0 * radius / max(res, 0.02)) + 1
        try:
            grid = self.region.evaluate_field_grid(self.robot_pos, view_radius=radius, n=n)
        except Exception as exc:
            self.get_logger().warn(f"[FIELD VIS] inference failed: {exc}", throttle_duration_sec=10.0)
            return
        if grid is None:
            return
        X_w, Y_w, TT, V, inb = grid
        keep = inb
        if bool(self.get_parameter("field_vis_free_only").value):
            keep = keep & self._slam_free_mask(X_w, Y_w)
        stamp = self.get_clock().now().to_msg()
        # Speed is infinite at the source cell itself, so filter each field separately.
        for pub, values in ((self.travel_time_pub, TT), (self.speed_pub, V)):
            m = keep & np.isfinite(values)
            if m.any():
                pub.publish(self._make_field_cloud(X_w[m], Y_w[m], values[m], stamp))


def main(args=None):
    rclpy.init(args=args)
    node = AntfieldsGridmapNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.cmd_pub.publish(Twist())
        except Exception:
            pass
        node._planning_stop.set()
        node._planning_thread.join(timeout=5.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
