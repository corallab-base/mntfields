"""
mapping_node.py — Line-based room segmentation + per-room NTField training.

Mirrors gridmap_node.py's proven exploration/training/control loop, replacing
TopologyExtractor + Voronoi region spawning with Warberg et al. line-segment
room segmentation.

Lifecycle:
  1. Spawn bootstrap Region at robot pos → WARMUP (train, no movement)
  2. EXPLORE → frontier-based, train in background thread, collect frames
  3. Segmentation runs passively every 3s → rooms + doorways
  4. When robot crosses into a confirmed room with no Region → spawn Region,
     seed from current region, switch, warmup, continue
  5. Old region is saved + unloaded from GPU (one active region at a time)
  6. Re-entering a room with an existing Region → reload + short warmup
"""

import os
import math
import json
import time
import threading
import colorsys
from math import atan2, sqrt, pi, copysign

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
import rclpy.time
import rclpy.parameter

import numpy as np
import torch
import cv2
import tf2_ros
import networkx as nx
from scipy.spatial.transform import Rotation

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import Twist, PoseStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA, Header, String

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from mntfields.mntfields_ros.sensing.lidar_adapter import Lidar2DAdapter
from mntfields.mntfields_ros.mapping.navigation import Navigator, resample_polyline
from mntfields.mntfields_core.core.region import Region
from mntfields.mntfields_core.mapping.line_segments import (
    extract_segments, merge_segments, build_directed_graph, LineSegment,
)
from mntfields.mntfields_core.mapping.room_segmenter import (
    RoomSegment, Doorway,  # kept for room_graph.load() backward compat
)
from mntfields.mntfields_core.mapping.watershed_segmenter import (
    WatershedSegmenter, WatershedRoom, WatershedDoorway,
)
from mntfields.mntfields_core.mapping.room_graph import RoomGraph
from mntfields.utils.mntfields_utils import straight_line_path


# ── Constants ────────────────────────────────────────────────────────────────

MODEL_FOLDER = os.path.expanduser("~/turtlebot3_ws/mntfields_model")
REGION_RADIUS = 2.5
ROOM_BOUND_PAD = 0.3
MIN_SEGMENTS_FOR_ROOMS = 8
MIN_REGION_ROOM_AREA_M2 = 2.0   # watershed room must be ≥ this to get a dedicated region
WARMUP_EPOCHS = 100
REVISIT_WARMUP_EPOCHS = 20
ROOM_PERSIST_TICKS = 3
ROOM_FORGET_TICKS = 5
ROOM_SWITCH_STABILITY_TICKS = 2
ROOM_SWITCH_MIN_MOVE_M = 0.40
ROOM_ASSIGN_MIN_REGIONS = 3
ROOM_ASSIGN_OVERLAP_PRIMARY = 0.35
ROOM_ASSIGN_HOLD_OVERLAP = 0.20
ROOM_ASSIGN_REBIND_MARGIN = 0.25
REGION_SPAWN_MARGIN = 0.40
_REGION_SWITCH_MARGIN = 0.30


def _normalize_angle(a: float) -> float:
    while a > pi:
        a -= 2.0 * pi
    while a < -pi:
        a += 2.0 * pi
    return a


def _bbox_overlap_ratio_np(bb1: np.ndarray, bb2: np.ndarray) -> float:
    """Overlap of two bboxes [[xmin,ymin],[xmax,ymax]] relative to the smaller box."""
    ix_min = max(bb1[0, 0], bb2[0, 0])
    iy_min = max(bb1[0, 1], bb2[0, 1])
    ix_max = min(bb1[1, 0], bb2[1, 0])
    iy_max = min(bb1[1, 1], bb2[1, 1])
    iw = max(0.0, float(ix_max - ix_min))
    ih = max(0.0, float(iy_max - iy_min))
    inter = iw * ih
    if inter < 1e-9:
        return 0.0
    a1 = max(1e-9, float((bb1[1, 0] - bb1[0, 0]) * (bb1[1, 1] - bb1[0, 1])))
    a2 = max(1e-9, float((bb2[1, 0] - bb2[0, 0]) * (bb2[1, 1] - bb2[0, 1])))
    return inter / min(a1, a2)


# ── Node ─────────────────────────────────────────────────────────────────────

class MappingNode(Node): 
    """
    ROS node for MNTFields mapping: Watershed room segmentation + per-region NTField training
    Parameters:
    train_timer_sec: float, seconds between training ticks (default: 0.5)
    train_samples: int, max samples to use for training each region (default: 10000)
    train_epoch: int, number of training epochs to run per training tick (default: 40)
    control_timer_sec: float, seconds between control ticks (default: 0.10)
    vis_timer_sec: float, seconds between visualization updates (default: 3.0)
    segmentation_timer_sec: float, seconds between room segmentation runs (default: 3.0)
    max_lin: float, max linear velocity for navigation (default: 0.20 m/s)
    max_ang: float, max angular velocity for navigation (default: 0.80 rad/s)
    goal_reached_dist: float, distance threshold to consider goal reached (default: 0.30 m)
    lidar_height_m: float, height of the LiDAR sensor from the ground (default: 0.17 m)
    enable_region_debug_vis: bool, whether to enable debug visualization for regions (default: True)
    min_train_epochs_for_nav: int, minimum training epochs before enabling navigation (default: 120)
    follow_stuck_timeout_sec: float, seconds of no progress before considering the robot stuck and replanning (default: 2.0)
    follow_progress_eps: float, minimum distance progress to reset the stuck timer (default: 0.10 m)
    path_commit_max_m: float, maximum distance along a new path to commit to when replanning (default: 0.40 m)
    path_truncate_trigger_m: float, distance threshold to trigger path truncation when deviating from the original path (default: 0.40 m)
    segment_replan_dist_m: float, distance threshold to trigger replanning when deviating from the current path (default: 0.35 m)
    plot_world_span_m: float, size of the world to plot in visualization (default: 0.0, which means auto-fit)
    robot_type: str, "sim" or "real", applies different default parameters for real robot operation (default: "sim")
    lidar_max_range: float, maximum range of the LiDAR sensor in meters (default: 3.5 m)
    """

    def __init__(self):
        super().__init__("mntfields_mapping")

        os.makedirs(MODEL_FOLDER, exist_ok=True)

        # ── Parameters ───────────────────────────────────────────────────────
        self.declare_parameter("semantic_topic",
                               "/semantic_segmentation/objects")
        self.declare_parameter("train_timer_sec", 0.5)
        self.declare_parameter("train_samples", 10000)
        self.declare_parameter("train_epoch", 40)
        self.declare_parameter("control_timer_sec", 0.10)
        self.declare_parameter("vis_timer_sec", 6.0)
        self.declare_parameter("segmentation_timer_sec", 3.0)
        self.declare_parameter("max_lin", 0.20)
        self.declare_parameter("max_ang", 0.80)
        self.declare_parameter("goal_reached_dist", 0.30)
        self.declare_parameter("lidar_height_m", 0.17)
        self.declare_parameter("base_frame", "base_footprint")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("enable_region_debug_vis", True)
        self.declare_parameter("min_train_epochs_for_nav", 120)
        self.declare_parameter("follow_stuck_timeout_sec", 2.0)
        self.declare_parameter("follow_progress_eps", 0.10)
        self.declare_parameter("path_commit_max_m", 0.40)
        self.declare_parameter("path_truncate_trigger_m", 0.40)
        self.declare_parameter("segment_replan_dist_m", 0.35)
        self.declare_parameter("plot_world_span_m", 0.0)
        self.declare_parameter("robot_type", "sim")   # "sim" | "real"
        self.declare_parameter("lidar_max_range", 3.5)
        # Do not touch use_sim_time here. ROS 2 declares it automatically, and
        # overriding it inside the node breaks real-robot timers.

        # ── Robot-type profile ────────────────────────────────────────────────
        _robot_type = str(self.get_parameter("robot_type").value)
        _is_real = (_robot_type == "real")
        self.is_real_robot: bool       = _is_real
        self.lidar_max_range: float     = float(self.get_parameter("lidar_max_range").value)
        self.region_radius: float       = 4.5  if _is_real else REGION_RADIUS
        self.room_bound_pad: float      = 0.5  if _is_real else ROOM_BOUND_PAD
        self.warmup_epochs: int         = 150  if _is_real else WARMUP_EPOCHS
        self.path_max_steps: int        = 300  if _is_real else 220
        self.frontier_cap_factor: float = 1.33  if _is_real else 4.0
        # For real robot, apply higher defaults unless caller overrode via ROS param
        _min_train_param = int(self.get_parameter("min_train_epochs_for_nav").value)
        _train_samples_param = int(self.get_parameter("train_samples").value)
        _path_commit_max_param = float(self.get_parameter("path_commit_max_m").value)
        self._min_train_epochs: int = (200 if _is_real and _min_train_param == 120
                                       else _min_train_param)
        self._train_samples: int    = (15000 if _is_real and _train_samples_param == 10000
                                       else _train_samples_param)
        _train_epoch_param = int(self.get_parameter("train_epoch").value)
        self._train_epoch_per_tick: int = (10 if _is_real and _train_epoch_param == 40
                                           else _train_epoch_param)
        self._path_commit_max: float = (0.70 if _is_real and abs(_path_commit_max_param - 0.40) < 1e-6
                                        else _path_commit_max_param)
        _plot_span_param = float(self.get_parameter("plot_world_span_m").value)
        self._plot_world_span_m: float = (
            _plot_span_param if _plot_span_param > 0.0
            else (25.0 if _is_real else 18.0)
        )

        # ── Robot state ──────────────────────────────────────────────────────
        self.robot_pos: np.ndarray | None = None
        self.robot_yaw: float = 0.0
        self.robot_pos_odom: np.ndarray | None = None
        self.robot_yaw_odom: float = 0.0
        self._latest_world_pts: np.ndarray | None = None
        self._latest_robot_pose: np.ndarray | None = None

        # ── Active region (single active, like gridmap_node) ─────────────────
        self.region: Region | None = None
        self.all_regions: dict = {}  # version -> Region

        # ── Room→Region mapping ──────────────────────────────────────────────
        self.room_regions: dict[int, Region] = {}  # room_id -> Region
        self._spawned_room_ids: set[int] = set()   # rooms that already have a Region
        self.active_room_id: int | None = None
        self._prev_room_id: int | None = None      # detect room crossings
        self._active_room_candidate_id: int | None = None
        self._active_room_candidate_ticks: int = 0
        self._pending_switch_room_id: int | None = None
        self._pending_switch_reason: str | None = None
        self._pending_region: Region | None = None
        self._pending_warmup_epochs: int | None = None
        self._last_region_anchor_xy: np.ndarray | None = None

        # ── Line-segment accumulation ────────────────────────────────────────
        self._all_segments: list[LineSegment] = []

        # ── Room segmentation (watershed) ────────────────────────────────────
        self.segmenter = WatershedSegmenter(
            min_peak_clearance_m  = 0.35 if _is_real else 0.30,
            min_peak_separation_m = 1.5  if _is_real else 1.2,
            min_room_area_m2      = 0.8,
            door_min_m            = 0.55,
            door_max_m            = 3.50,
        )
        self.room_graph = RoomGraph()
        self._rooms: list = []
        self._visible_rooms: list = []
        self._accepted_rooms: dict[int, WatershedRoom] = {}
        self._doorways: list = []
        self._room_persist: dict[int, int] = {}
        self._room_miss: dict[int, int] = {}
        self._confirmed_room_ids: set[int] = set()
        self._seg_running: bool = False          # gate: one seg thread at a time
        self._seg_result: tuple | None = None    # (rooms, doorways) pending apply

        # ── Mode ─────────────────────────────────────────────────────────────
        self.mode: str = "EXPLORE"  # starts as EXPLORE, _switch_region→WARMUP
        self._warmup_start_epoch: int = 0
        self._warmup_epoch_target: int = self.warmup_epochs

        # ── Exploration state (ported from gridmap_node) ─────────────────────
        self._path: np.ndarray | None = None
        self._control_path: np.ndarray | None = None
        self._current_goal: np.ndarray | None = None
        self.goal_xy: np.ndarray | None = None
        self._visited_goals: list = []
        self._goal_blacklist_radius: float = 0.5
        self._pathgen_fail_count: int = 0
        self._nonprogress_pathgen_count: int = 0
        self._last_path_reject_reason: str | None = None
        self._exploration_tick_count: int = 0
        self._controller_path_done: bool = False
        self._train_running: bool = False
        self._train_lock = threading.Lock()
        self._path_total_dist = None
        self._midpoint_replanned = False
        self._best_goal_dist = float('inf')
        self._last_progress_time = 0.0
        self._last_path_progress_s = 0.0
        self._stuck_replan_count: int = 0
        self._segment_start_xy: np.ndarray | None = None
        self._pathgen_running: bool = False
        self._pathgen_result: tuple | None = None
        self._pathgen_lock = threading.Lock()
        self._robot_traj: list = []          # [(x,y), ...] robot positions for vis
        self._last_frontier_candidate_count: int = 0
        self._last_frontier_selection_method: str = "none"
        self._lidar_drop_count: int = 0
        self._lidar_drop_max_lag_s: float = 0.0
        self._frontier_pick_times_ms: list[float] = []
        self._path_gen_times_ms: list[float] = []
        self._goal_metrics: list[dict] = []
        self._current_goal_metric: dict | None = None
        self._run_start_wall_time: float = time.time()
        self._run_start_mono: float = time.monotonic()
        self._current_goal_kind: str | None = None
        self._inter_region_target_room_id: int | None = None
        self._inter_region_target_region_ver: int | None = None
        self._inter_region_route_room_ids: list[int] = []
        self._region_frontier_banks: dict[int, list[list[float]]] = {}
        self._mapping_complete: bool = False
        self._mapping_completion_saved: bool = False
        # Semantic detection diagnostics
        self._sem_msg_count: int = 0       # total messages received on topic
        self._sem_stored_total: int = 0    # total unique objects stored so far
        self._sem_dedup_count: int = 0     # messages where everything was deduplicated
        self._sem_no_xyz_count: int = 0    # objects missing usable map/camera coords
        self._sem_tf_fail_count: int = 0   # camera-frame detections without usable TF
        self._sem_no_region_count: int = 0 # objects with coords but no region to store in
        self._sem_last_log_time: float = 0.0
        self._lidar_drop_window_start: float = time.monotonic()
        self._vis_running: bool = False
        self._plots2_extent: tuple[float, float, float, float] | None = None
        self._vis_region_contour_cache: dict = {}
        self._vis_contour_lock = threading.Lock()

        # ── SLAM map ─────────────────────────────────────────────────────────
        self._slam_arr: np.ndarray | None = None
        self._slam_cell_size: float = 0.05
        self._slam_origin: np.ndarray = np.zeros(2, dtype=np.float32)
        self._slam_H: int = 0
        self._slam_W: int = 0
        self._slam_map_received: bool = False

        # ── TF ───────────────────────────────────────────────────────────────
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ── LiDAR adapter ────────────────────────────────────────────────────
        self.adapter = Lidar2DAdapter(
            max_range=self.lidar_max_range,
            lidar_height=float(self.get_parameter("lidar_height_m").value),
        )

        # ── Subscribers ──────────────────────────────────────────────────────
        self.create_subscription(LaserScan, "/scan", self._lidar_cb, 10)
        self.create_subscription(
            String,
            str(self.get_parameter("semantic_topic").value),
            self._semantic_tracked_cb, 10)
        map_qos = QoSProfile(
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
            depth=1,
        )
        self.create_subscription(OccupancyGrid, "/map", self._slam_map_cb, map_qos)

        # ── Publishers ───────────────────────────────────────────────────────
        self.base_frame = str(self.get_parameter("base_frame").value)
        cmd_vel_topic = str(self.get_parameter("cmd_vel_topic").value)
        self.cmd_pub = self.create_publisher(Twist, cmd_vel_topic, 10)
        self.field_path_pub = self.create_publisher(Path, "/field_path", 10)
        self.nav_path_pub = self.create_publisher(Path, "/navigator_path", 10)
        self.room_marker_pub = self.create_publisher(
            MarkerArray, "/mntfields/rooms", 10)
        self.seg_marker_pub = self.create_publisher(
            MarkerArray, "/mntfields/line_segments", 10)
        self.door_marker_pub = self.create_publisher(
            MarkerArray, "/mntfields/doorways", 10)
        self.goal_marker_pub = self.create_publisher(
            Marker, "/mntfields/goal_marker", 10)
        self.frontier_marker_pub = self.create_publisher(
            Marker, "/mntfields/frontier_markers", 10)
        self.frontier_occ_pub = self.create_publisher(
            OccupancyGrid, "/mntfields/frontier", map_qos)

        # ── Navigator ────────────────────────────────────────────────────────
        self.dt = float(self.get_parameter("control_timer_sec").value)
        self.nav_frame = "map"
        self.use_continuous_real_follower = True
        self.navigator = Navigator(self)

        # ── Timers ───────────────────────────────────────────────────────────
        train_dt = float(self.get_parameter("train_timer_sec").value)
        ctrl_dt = float(self.get_parameter("control_timer_sec").value)
        vis_dt = float(self.get_parameter("vis_timer_sec").value)
        seg_dt = float(self.get_parameter("segmentation_timer_sec").value)

        self.create_timer(train_dt, self._train_tick)
        self.create_timer(ctrl_dt, self._exploration_tick)
        self.create_timer(ctrl_dt, self._control_tick)
        self.create_timer(seg_dt, self._segmentation_tick)
        self.create_timer(vis_dt, self._vis_tick)

        self.get_logger().info("[MappingNode] initialized")

    # ══════════════════════════════════════════════════════════════════════════
    # Subscribers
    # ══════════════════════════════════════════════════════════════════════════

    def _lidar_cb(self, msg: LaserScan):
        """Process incoming LiDAR scan: transform to map frame, extract line segments (old)"""
        self._update_robot_pose_from_tf()
        if self.robot_pos is None:
            return

        pts_sensor = self.adapter.scan_to_xy(msg)
        if pts_sensor is None:
            return

        world_pts = self._transform_scan_points_to_map(pts_sensor, msg)
        if world_pts is None:
            return

        self._latest_world_pts = world_pts
        self._latest_robot_pose = self.robot_pos.copy()

        # Extract and merge line segments for room segmentation
        new_segs = extract_segments(world_pts[:, :2], self.robot_pos[:2])
        if new_segs:
            self._all_segments = merge_segments(self._all_segments, new_segs)

        # Auto-spawn first region
        if self.region is None and self._latest_world_pts is not None:
            self._spawn_first_region()

    def _semantic_tracked_cb(self, msg: String):
        """Store furniture detections into the owning region.

        Accepts the schema published by semantic_segmentation_rgbd_node.py.
        Preferred inputs are world_xyz_m/map_xy. If those are unavailable,
        camera_xyz_m is transformed into map here, so the camera pipeline can
        run on an isolated ROS domain and bridge only object JSON + /tf_static.
        No "status" filter — every detection with valid world coordinates is stored
        (deduplication is handled by region.add_semantic_object's dedup_radius).
        """
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warn("[SEM] received malformed JSON on semantic topic")
            return

        objects = data.get("objects", [])
        self._sem_msg_count += 1

        if not objects:
            # Throttled: warn every 30 messages if topic is alive but always empty
            if self._sem_msg_count % 30 == 1:
                self.get_logger().warn(
                    f"[SEM] topic alive but 'objects' list is always empty "
                    f"(msg #{self._sem_msg_count})")
            return

        stored_this_msg = 0
        no_xyz_this_msg = 0
        tf_fail_this_msg = 0
        no_region_this_msg = 0
        stamp = self._semantic_json_stamp(data)

        for obj in objects:
            class_name = obj.get("class_name", "")
            # Accept either world_xyz_m (3-element) or map_xy (2-element)
            world_xyz = obj.get("world_xyz_m") or obj.get("map_xy")

            if not class_name:
                continue

            if world_xyz is not None:
                xy = np.asarray(world_xyz[:2], dtype=np.float32)
            else:
                xy = self._semantic_camera_xyz_to_map_xy(obj, stamp)

            if xy is None:
                no_xyz_this_msg += 1
                self._sem_no_xyz_count += 1
                if obj.get("camera_xyz_m") is not None:
                    tf_fail_this_msg += 1
                    self._sem_tf_fail_count += 1
                continue

            # Assign to whichever region's bound contains this point;
            # fall back to the active region.
            region = None
            for reg in self.all_regions.values():
                if reg.in_bound(xy):
                    region = reg
                    break
            if region is None:
                region = self.region

            if region is None:
                no_region_this_msg += 1
                self._sem_no_region_count += 1
                continue

            created = region.add_semantic_object(
                class_name, xy,
                stamp=(float(stamp.nanoseconds) * 1e-9 if stamp is not None else None),
            )
            after = len(region.semantic_objects.get(class_name, []))
            if created:
                stored_this_msg += 1
                self._sem_stored_total += 1
                self.get_logger().info(
                    f"[SEM] NEW '{class_name}' at ({xy[0]:.2f},{xy[1]:.2f}) "
                    f"→ R{region.version} (region total={after}, "
                    f"all-time stored={self._sem_stored_total})")

        # ── Per-message summary when nothing was stored ──────────────────────
        if stored_this_msg == 0:
            self._sem_dedup_count += 1
            reasons = []
            if no_xyz_this_msg:
                reasons.append(f"{no_xyz_this_msg} missing coords")
            if tf_fail_this_msg:
                reasons.append(f"{tf_fail_this_msg} tf-failed")
            if no_region_this_msg:
                reasons.append(f"{no_region_this_msg} no region")
            dedup = len(objects) - no_xyz_this_msg - no_region_this_msg
            if dedup > 0:
                reasons.append(f"{dedup} deduplicated")
            has_issue = no_xyz_this_msg > 0 or no_region_this_msg > 0
            if has_issue or self._sem_dedup_count % 20 == 1:
                self.get_logger().info(
                    f"[SEM] msg #{self._sem_msg_count}: {len(objects)} object(s) — "
                    + (", ".join(reasons) if reasons else "all skipped"))

        # ── Periodic health summary (every 20 messages) ───────────────────────
        if self._sem_msg_count % 20 == 0:
            self.get_logger().info(
                f"[SEM SUMMARY] msgs={self._sem_msg_count} "
                f"stored={self._sem_stored_total} "
                f"all-dedup={self._sem_dedup_count} "
                f"no-xyz={self._sem_no_xyz_count} "
                f"tf-fail={self._sem_tf_fail_count} "
                f"no-region={self._sem_no_region_count}")

    @staticmethod
    def _semantic_json_stamp(data: dict) -> rclpy.time.Time | None:
        sec = data.get("stamp_sec")
        nsec = data.get("stamp_nanosec")
        if sec is None or nsec is None:
            return None
        try:
            return rclpy.time.Time(seconds=int(sec), nanoseconds=int(nsec))
        except Exception:
            return None

    def _semantic_camera_xyz_to_map_xy(
        self,
        obj: dict,
        stamp: rclpy.time.Time | None,
    ) -> np.ndarray | None:
        camera_xyz = obj.get("camera_xyz_m")
        frame_id = str(obj.get("frame_id", "")).strip()
        if camera_xyz is None or not frame_id:
            return None

        try:
            pt = np.asarray(camera_xyz[:3], dtype=np.float32)
            if pt.shape[0] < 3 or not np.all(np.isfinite(pt[:3])):
                return None
        except Exception:
            return None

        tf_msg = None
        if stamp is not None:
            try:
                tf_msg = self.tf_buffer.lookup_transform(
                    "map",
                    frame_id,
                    stamp,
                    timeout=Duration(seconds=0.03),
                )
            except Exception:
                tf_msg = None

        if tf_msg is None:
            try:
                tf_msg = self.tf_buffer.lookup_transform(
                    "map",
                    frame_id,
                    rclpy.time.Time(),
                    timeout=Duration(seconds=0.03),
                )
            except Exception as exc:
                if self._sem_tf_fail_count % 20 == 0:
                    self.get_logger().warn(
                        f"[SEM] cannot transform {frame_id}->map for camera_xyz_m: {exc}")
                return None

        t = tf_msg.transform.translation
        q = tf_msg.transform.rotation
        rot = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix().astype(np.float32)
        trans = np.array([t.x, t.y, t.z], dtype=np.float32)
        map_pt = rot @ pt[:3] + trans
        return map_pt[:2].astype(np.float32)

    def _slam_map_cb(self, msg: OccupancyGrid):
        """Process incoming SLAM map: store as numpy array, save to disk in map_server format for segmentation visualization."""
        info = msg.info
        self._slam_cell_size = float(info.resolution)
        self._slam_origin = np.array(
            [info.origin.position.x, info.origin.position.y], dtype=np.float32)
        H, W = info.height, info.width
        self._slam_H, self._slam_W = H, W
        self._slam_arr = np.array(msg.data, dtype=np.int8).reshape(H, W)

        if not self._slam_map_received:
            self._slam_map_received = True
            # self.get_logger().info(
            #     f"[SLAM MAP] first map size={W}x{H} res={self._slam_cell_size:.3f}m")
            self._save_slam_map()

    def _save_slam_map(self) -> None:
        """Save SLAM map to mntfields_model/map.pgm + map.yaml (map_server format)."""
        if self._slam_arr is None:
            return
        try:
            arr = self._slam_arr.copy()
            H, W = arr.shape
            img = np.full((H, W), 127, dtype=np.uint8)
            img[arr == 0] = 254       # free → white (occ_prob≈0.004, well below free_thresh=0.196)
            img[arr > 50] = 0         # occupied → black (occ_prob=1.0, well above occ_thresh=0.65)
            img = np.flipud(img)      # ROS origin is bottom-left

            pgm_path  = os.path.join(MODEL_FOLDER, "map.pgm")
            yaml_path = os.path.join(MODEL_FOLDER, "map.yaml")

            # PGM binary (P5)
            with open(pgm_path, "wb") as f:
                f.write(f"P5\n{W} {H}\n255\n".encode("ascii"))
                f.write(img.tobytes())

            # YAML (map_server format)
            ox = float(self._slam_origin[0])
            oy = float(self._slam_origin[1])
            with open(yaml_path, "w") as f:
                f.write(
                    f"image: map.pgm\n"
                    f"resolution: {self._slam_cell_size:.4f}\n"
                    f"origin: [{ox:.6f}, {oy:.6f}, 0.0]\n"
                    f"negate: 0\n"
                    f"occupied_thresh: 0.65\n"
                    f"free_thresh: 0.196\n"
                )
            # self.get_logger().info(f"[SLAM MAP SAVED] {W}x{H} → {pgm_path}")
        except Exception as e:
            self.get_logger().error(f"[SLAM MAP SAVE ERROR] {e}")

    # ══════════════════════════════════════════════════════════════════════════
    # TF
    # ══════════════════════════════════════════════════════════════════════════

    def _update_robot_pose_from_tf(self):
        try:
            t = self.tf_buffer.lookup_transform(
                "map", self.base_frame,
                rclpy.time.Time(), timeout=Duration(seconds=0.05))
            p = t.transform.translation
            q = t.transform.rotation
            self.robot_pos = np.array([p.x, p.y, p.z], dtype=np.float32)
            siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            self.robot_yaw = float(np.arctan2(siny_cosp, cosy_cosp))
        except Exception:
            pass
        try:
            t = self.tf_buffer.lookup_transform(
                "odom", self.base_frame,
                rclpy.time.Time(), timeout=Duration(seconds=0.05))
            p = t.transform.translation
            q = t.transform.rotation
            self.robot_pos_odom = np.array([p.x, p.y, p.z], dtype=np.float32)
            siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            self.robot_yaw_odom = float(np.arctan2(siny_cosp, cosy_cosp))
        except Exception:
            pass

    def _transform_path_map_to_odom(self, path_map: np.ndarray) -> np.ndarray | None:
        if path_map is None or len(path_map) == 0:
            return None
        try:
            tf_msg = self.tf_buffer.lookup_transform(
                "odom", "map", rclpy.time.Time(), timeout=Duration(seconds=0.05))
        except Exception:
            return None

        t = tf_msg.transform.translation
        q = tf_msg.transform.rotation
        rot = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix().astype(np.float32)
        trans = np.array([t.x, t.y, t.z], dtype=np.float32)
        pts3 = np.zeros((len(path_map), 3), dtype=np.float32)
        pts3[:, :2] = np.asarray(path_map, dtype=np.float32)
        pts_odom = (pts3 @ rot.T) + trans
        return pts_odom[:, :2]

    def _transform_scan_points_to_map(self, pts_sensor: np.ndarray, msg: LaserScan):
        """Transform Nx3 scan-frame points into the map frame using TF."""
        source_frame = str(msg.header.frame_id).strip()
        if not source_frame:
            self.get_logger().warn('[LIDAR] scan header has empty frame_id; dropping scan')
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
                tf_msg = self.tf_buffer.lookup_transform(
                    "map",
                    source_frame,
                    rclpy.time.Time(),
                    timeout=Duration(seconds=0.05),
                )
            except Exception as exc:
                now_msg = self.get_clock().now().to_msg()
                lag_s = ((now_msg.sec - msg.header.stamp.sec) +
                         (now_msg.nanosec - msg.header.stamp.nanosec) * 1e-9)
                self._lidar_drop_count += 1
                self._lidar_drop_max_lag_s = max(self._lidar_drop_max_lag_s, lag_s)
                now_mono = time.monotonic()
                if now_mono - self._lidar_drop_window_start >= 5.0:
                    self.get_logger().warn(
                        f"[LIDAR] dropped {self._lidar_drop_count} scans in last "
                        f"{now_mono - self._lidar_drop_window_start:.1f}s; "
                        f"max_scan_lag={self._lidar_drop_max_lag_s:.3f}s; "
                        f"latest_error={exc}")
                    self._lidar_drop_count = 0
                    self._lidar_drop_max_lag_s = 0.0
                    self._lidar_drop_window_start = now_mono
                return None

        t = tf_msg.transform.translation
        q = tf_msg.transform.rotation
        rot = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix().astype(np.float32)
        trans = np.array([t.x, t.y, t.z], dtype=np.float32)
        return (pts_sensor @ rot.T) + trans

    # ══════════════════════════════════════════════════════════════════════════
    # Region management
    # ══════════════════════════════════════════════════════════════════════════

    def _spawn_first_region(self):
        """Create the bootstrap region at the robot's spawn location."""
        reg = self._new_region(self.robot_pos[:2])
        self._switch_region(reg, self.warmup_epochs, "first region")

    def _new_region(self, center_xy, radius=None) -> Region:
        if radius is None:
            radius = self.region_radius
        cx, cy = float(center_xy[0]), float(center_xy[1])
        bound = torch.tensor([
            [cx - radius, cy - radius, -0.1],
            [cx + radius, cy + radius,  0.1],
        ], dtype=torch.float32)

        reg = Region(
            model_folder=MODEL_FOLDER,
            bound=bound,
            device="cuda:0",
        )
        reg.spawn_center = np.array([cx, cy], dtype=np.float32)
        reg.max_raw_samples = 50000
        reg.enable_debug_vis = bool(
            self.get_parameter("enable_region_debug_vis").value)

        self.all_regions[reg.version] = reg
        self.get_logger().info(
            f"[REGION {reg.version} NEW] center=({cx:.2f},{cy:.2f}) "
            f"bound=[({bound[0,0]:.2f},{bound[0,1]:.2f}) → "
            f"({bound[1,0]:.2f},{bound[1,1]:.2f})]")

        self._seed_region_from_neighbors(reg)
        return reg

    def _new_region_for_room(self, room: RoomSegment) -> Region:
        bb = room.bounding_box
        pad = self.room_bound_pad
        cx = float(0.5 * (bb[0, 0] + bb[1, 0]))
        cy = float(0.5 * (bb[0, 1] + bb[1, 1]))
        rx = float(bb[1, 0] - bb[0, 0]) * 0.5 + pad
        ry = float(bb[1, 1] - bb[0, 1]) * 0.5 + pad
        radius = max(rx, ry, self.region_radius * 0.5)

        bound = torch.tensor([
            [cx - rx, cy - ry, -0.1],
            [cx + rx, cy + ry,  0.1],
        ], dtype=torch.float32)

        reg = Region(
            model_folder=MODEL_FOLDER,
            bound=bound,
            device="cuda:0",
        )
        reg.spawn_center = room.centroid.copy()
        reg.max_raw_samples = 50000
        reg.enable_debug_vis = bool(
            self.get_parameter("enable_region_debug_vis").value)

        self.all_regions[reg.version] = reg
        self.room_regions[room.id] = reg
        self._spawned_room_ids.add(room.id)
        self.get_logger().info(
            f"[REGION {reg.version} NEW] for room {room.id} "
            f"bb=[{bb[0,0]:.1f},{bb[0,1]:.1f}]→[{bb[1,0]:.1f},{bb[1,1]:.1f}]")

        self._seed_region_from_neighbors(reg)
        return reg

    @staticmethod
    def _room_region_overlap(room: RoomSegment, region: Region) -> float:
        rb = np.asarray(room.bounding_box, dtype=np.float32)
        bb = region.bound.detach().cpu().numpy()
        ix0 = max(float(rb[0, 0]), float(bb[0, 0]))
        iy0 = max(float(rb[0, 1]), float(bb[0, 1]))
        ix1 = min(float(rb[1, 0]), float(bb[1, 0]))
        iy1 = min(float(rb[1, 1]), float(bb[1, 1]))
        iw = max(0.0, ix1 - ix0)
        ih = max(0.0, iy1 - iy0)
        inter = iw * ih
        room_area = max(1e-6, float(rb[1, 0] - rb[0, 0]) * float(rb[1, 1] - rb[0, 1]))
        return inter / room_area

    def _seed_region_from_neighbors(self, new_reg: Region):
        seeded_rows = []
        for donor in self.all_regions.values():
            if donor is new_reg:
                continue
            if donor.all_raw_frame_data is None or len(donor.all_raw_frame_data) == 0:
                continue
            donor_merged = torch.cat(donor.all_raw_frame_data, dim=0)
            filtered = new_reg._filter_raw_rows_to_bound(donor_merged)
            if len(filtered) > 0:
                seeded_rows.append(filtered)

        if seeded_rows:
            merged = torch.cat(seeded_rows, dim=0)
            max_raw = max(1, int(new_reg.max_raw_samples))
            if len(merged) > max_raw:
                idx = torch.randperm(len(merged))[:max_raw]
                merged = merged[idx]
            new_reg.all_raw_frame_data = [merged]
            # self.get_logger().info(
            #     f"[SEED] R{new_reg.version} seeded {len(merged)} rows")

        # Also seed lidar_obs point cloud
        nb = new_reg.bound.detach().cpu().numpy()
        for donor in self.all_regions.values():
            if donor is new_reg or donor.lidar_obs is None:
                continue
            donor_pts = np.asarray(donor.lidar_obs.points, dtype=np.float32)
            in_b = (
                (donor_pts[:, 0] >= nb[0, 0]) & (donor_pts[:, 0] <= nb[1, 0]) &
                (donor_pts[:, 1] >= nb[0, 1]) & (donor_pts[:, 1] <= nb[1, 1])
            )
            if in_b.sum() == 0:
                continue
            try:
                import open3d as o3d
                from mntfields.utils.mntfields_utils import combine_pcd
                subset = donor_pts[in_b]
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(subset.astype(np.float64))
                if new_reg.lidar_obs is None:
                    new_reg.lidar_obs = pcd
                else:
                    new_reg.lidar_obs = combine_pcd(pcd, new_reg.lidar_obs)
            except Exception:
                pass

        # if new_reg.lidar_obs is not None:
        #     self.get_logger().info(
        #         f"[SEED] R{new_reg.version} seeded "
        #         f"{len(np.asarray(new_reg.lidar_obs.points))} obs pts")

    # ── Region lifecycle ─────────────────────────────────────────────────────

    def _unload_current_region(self):
        """Save current region weights + data, free GPU memory."""
        if self.region is None:
            return
        try:
            if getattr(self.region.model, "network", None) is not None:
                self.region.save_model()
            # save_debug_plot removed: expensive GPU inference at unload time
            # try:
            #     rp = getattr(self.region, 'last_robot_pose', None)
            #     if rp is not None:
            #         self.region.save_debug_plot(rp, robot_xy=rp[:2])
            # except Exception:
            #     pass
            if getattr(self.region, "loaded", True):
                self.region.unload_to_disk()
            # self.get_logger().info(
            #     f"[REGION {self.region.version} UNLOADED] "
            #     f"epoch={self.region.epoch} saved + freed GPU")
        except Exception as e:
            self.get_logger().error(f"[UNLOAD ERROR] R{self.region.version}: {e}")

    def _switch_region(self, new_region: Region,
                       warmup_epochs: int = -1,
                       reason: str = "switch"):
        """Full lifecycle: unload old region, switch to new, enter warmup."""
        if warmup_epochs < 0:
            warmup_epochs = self.warmup_epochs
        old_ver = self.region.version if self.region is not None else "---"

        # 1. Save and unload old region
        self._unload_current_region()

        # 2. Reload if previously unloaded
        if getattr(new_region, 'loaded', True) is False:
            new_region.load_from_disk()
            # self.get_logger().info(
            #     f"[REGION {new_region.version}] reloaded from disk "
            #     f"(epoch={new_region.epoch})")

        # 3. Set active region
        self.region = new_region
        if self.robot_pos is not None:
            self._last_region_anchor_xy = self.robot_pos[:2].copy()
        self._pending_switch_room_id = None
        self._pending_switch_reason = None
        self._pending_region = None
        self._pending_warmup_epochs = None

        # 4. Enter warmup
        self._enter_warmup(
            new_region, warmup_epochs,
            f"[REGION] R{old_ver} → R{new_region.version}: {reason}")

    def _bind_room_to_region(self, room_id: int, region: Region, reason: str):
        prev = self.room_regions.get(room_id)
        if prev is region:
            return
        self.room_regions[room_id] = region
        self._spawned_room_ids.add(room_id)
        self.get_logger().info(
            f"[ROOM→REGION] room {room_id} -> R{region.version}: {reason}")

    def _maybe_bind_active_room_to_current_region(self):
        if self.region is None or self.active_room_id is None:
            return
        if self.active_room_id not in self._confirmed_room_ids:
            return
        if self.active_room_id in self.room_regions:
            return
        if self.robot_pos is None or not self.region.in_bound(self.robot_pos[:2]):
            return
        self._bind_room_to_region(
            self.active_room_id, self.region, "bootstrap active room")
        self._prev_room_id = self.active_room_id

    def _queue_room_switch(self, room_id: int, reason: str):
        self._pending_switch_room_id = room_id
        self._pending_switch_reason = reason
        self._pending_region = None
        self._pending_warmup_epochs = None
        self._path = None
        self._current_goal = None
        self.goal_xy = None
        self.navigator.reset_path()
        self._controller_path_done = False
        self._stop_robot()
        # self.get_logger().info(f"[ROOM SWITCH] queued room {room_id}: {reason}")

    def _queue_region_switch(self, region: Region, reason: str,
                             warmup_epochs: int | None = None):
        self._pending_region = region
        self._pending_switch_room_id = None
        self._pending_switch_reason = reason
        self._pending_warmup_epochs = warmup_epochs
        self._path = None
        self._current_goal = None
        self.goal_xy = None
        self.navigator.reset_path()
        self._controller_path_done = False
        self._stop_robot()

    def _process_pending_room_switch(self):
        if self._pending_region is not None:
            if self._train_running:
                return True
            reason = self._pending_switch_reason or "spawned exploration region"
            warmup_epochs = (self._pending_warmup_epochs
                             if self._pending_warmup_epochs is not None
                             else self.warmup_epochs)
            self._switch_region(self._pending_region, warmup_epochs, reason)
            return True

        room_id = self._pending_switch_room_id
        if room_id is None:
            return False
        if self._train_running:
            return True

        if room_id in self.room_regions:
            room_reg = self.room_regions[room_id]
            if room_reg is self.region:
                self._pending_switch_room_id = None
                self._pending_switch_reason = None
                return False
            reason = self._pending_switch_reason or f"re-entered room {room_id}"
            self._switch_region(room_reg, REVISIT_WARMUP_EPOCHS, reason)
            return True

        room = self.room_graph.rooms.get(room_id)
        if room is None:
            self._pending_switch_room_id = None
            self._pending_switch_reason = None
            return False

        new_reg = self._new_region_for_room(room)
        reason = self._pending_switch_reason or f"entered new room {room_id}"
        self._switch_region(new_reg, self.warmup_epochs, reason)
        return True

    def _maybe_spawn_region_from_exploration(self, robot_xy: np.ndarray):
        if self.region is None:
            return
        if self.mode == "WARMUP":
            return
        if self._pending_region is not None or self._pending_switch_room_id is not None:
            return
        if self.robot_pos is None:
            return
        if self.region.in_bound(robot_xy):
            return
        b = self.region.bound.detach().cpu().numpy()
        dx = max(b[0, 0] - robot_xy[0], 0.0, robot_xy[0] - b[1, 0])
        dy = max(b[0, 1] - robot_xy[1], 0.0, robot_xy[1] - b[1, 1])
        outside = max(dx, dy)
        if outside < REGION_SPAWN_MARGIN:
            return
        moved = (
            float(np.linalg.norm(robot_xy - self._last_region_anchor_xy))
            if self._last_region_anchor_xy is not None else float("inf")
        )
        if moved < ROOM_SWITCH_MIN_MOVE_M:
            return

        # Prefer room-shaped region if the robot is in a confirmed, large-enough room
        ws_room_id = self.segmenter.room_for_point(robot_xy)
        ws_room = self.segmenter.rooms.get(ws_room_id) if ws_room_id is not None else None

        if (ws_room is not None
                and ws_room.confirmed
                and ws_room.area_m2 >= MIN_REGION_ROOM_AREA_M2):
            new_reg = self._new_region_for_room(ws_room)
            reason = (f"room-driven spawn R{ws_room.id} "
                      f"area={ws_room.area_m2:.1f}m² clr={ws_room.peak_clearance_m:.2f}m")
        else:
            new_reg = self._new_region(robot_xy)
            reason = f"exploration-driven spawn near ({robot_xy[0]:.2f},{robot_xy[1]:.2f})"
        self._queue_region_switch(new_reg, reason)

    def _enter_warmup(self, region: Region, epochs_needed: int, reason: str):
        self.mode = "WARMUP"
        self._warmup_start_epoch = int(getattr(region, 'epoch', 0))
        self._warmup_epoch_target = max(1, int(epochs_needed))
        self._path = None
        self._current_goal = None
        self.goal_xy = None
        self.navigator.reset_path()
        self._controller_path_done = False
        self._stop_robot()
        self.get_logger().info(
            f"{reason} [warmup={self._warmup_epoch_target} eps, "
            f"start_ep={self._warmup_start_epoch}]")

    def _maybe_switch_for_room_crossing(self, robot_xy: np.ndarray):
        """On room crossing: log topology and spawn a new region if the room warrants one."""
        if self.active_room_id is None:
            return
        self._maybe_bind_active_room_to_current_region()
        if self.mode == "WARMUP":
            # Keep prev in sync so no stale crossing fires when warmup exits
            self._prev_room_id = self.active_room_id
            return

        # Detect crossing: room changed from last check
        if self._prev_room_id == self.active_room_id:
            return

        old_room = self._prev_room_id
        new_room = self.active_room_id

        # Only act on confirmed rooms
        if new_room not in self._confirmed_room_ids:
            return

        moved = (
            float(np.linalg.norm(robot_xy - self._last_region_anchor_xy))
            if self._last_region_anchor_xy is not None else float("inf")
        )
        if moved < ROOM_SWITCH_MIN_MOVE_M:
            self.get_logger().info(
                f"[CROSSING] suppressed room {old_room} → room {new_room}: "
                f"moved {moved:.2f}m since current region activation")
            self._prev_room_id = new_room
            return

        self._prev_room_id = new_room
        self.get_logger().info(f"[CROSSING] room {old_room} → room {new_room}")

        # Spawn a region for the new room if it's large enough and doesn't have one yet
        self._maybe_spawn_region_for_room(new_room)

    def _region_for_room(self, room_id: int) -> Region | None:
        """Get the region for a room, or None."""
        return self.room_regions.get(room_id)

    def _assign_confirmed_rooms_to_regions(self):
        if len(self.all_regions) < ROOM_ASSIGN_MIN_REGIONS:
            return
        for room in self._rooms:
            if room.id not in self._confirmed_room_ids:
                continue

            overlaps: dict[Region, float] = {}
            best_reg = None
            best_overlap = 0.0
            for reg in self.all_regions.values():
                overlap = self._room_region_overlap(room, reg)
                overlaps[reg] = overlap
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_reg = reg

            existing = self.room_regions.get(room.id)
            existing_overlap = overlaps.get(existing, 0.0) if existing is not None else 0.0

            # Shared boundary rooms are noisy: segmentation can make the same room
            # overlap two neighboring regions. Keep the current/assigned region
            # unless another region covers the room substantially better.
            if existing is not None and existing_overlap >= ROOM_ASSIGN_HOLD_OVERLAP:
                if (best_reg is None or best_reg is existing or
                        best_overlap < existing_overlap + ROOM_ASSIGN_REBIND_MARGIN):
                    continue

            # If the robot is currently inside this confirmed room, let the active
            # region hold the room when it has meaningful area coverage.
            if (room.id == self.active_room_id and self.region is not None):
                active_overlap = overlaps.get(self.region, 0.0)
                if active_overlap >= ROOM_ASSIGN_HOLD_OVERLAP:
                    if (existing is None or existing is self.region or
                            active_overlap >= existing_overlap):
                        self._bind_room_to_region(
                            room.id, self.region,
                            f"active hold overlap={active_overlap:.2f}")
                        continue

            if best_reg is not None and best_overlap >= ROOM_ASSIGN_OVERLAP_PRIMARY:
                self._bind_room_to_region(
                    room.id, best_reg, f"overlap={best_overlap:.2f}")

    def _maybe_spawn_region_for_room(self, room_id: int) -> None:
        """
        Spawn a room-shaped region when the robot enters a confirmed watershed room.
        This is the primary region-creation trigger — it fires on room crossings,
        independent of whether the robot is inside the current region bounds.
        """
        if self.mode == "WARMUP":
            return
        if self._pending_region is not None or self._pending_switch_room_id is not None:
            return
        if room_id in self._spawned_room_ids:
            # Room already has a region — switch back to it instead of spawning
            existing = self.room_regions.get(room_id)
            if existing is not None and existing is not self.region:
                self._queue_region_switch(
                    existing,
                    f"re-entered room {room_id} → R{existing.version}",
                    warmup_epochs=REVISIT_WARMUP_EPOCHS)
            return

        # Get watershed room properties
        ws_room = self.segmenter.rooms.get(room_id)
        if ws_room is None:
            return
        if ws_room.area_m2 < MIN_REGION_ROOM_AREA_M2:
            self.get_logger().info(
                f"[CROSSING] room {room_id} too small ({ws_room.area_m2:.1f}m² "
                f"< {MIN_REGION_ROOM_AREA_M2}m²) — no region spawn")
            return

        new_reg = self._new_region_for_room(ws_room)
        self.get_logger().info(
            f"[REGION NEW] room {room_id} → R{new_reg.version} "
            f"area={ws_room.area_m2:.1f}m² clr={ws_room.peak_clearance_m:.2f}m")
        self._queue_region_switch(
            new_reg, f"room-crossing spawn room {room_id} (area={ws_room.area_m2:.1f}m²)")

    # ══════════════════════════════════════════════════════════════════════════
    # Training tick (every 0.5s) — background thread (like gridmap_node)
    # ══════════════════════════════════════════════════════════════════════════

    def _train_tick(self):
        if self._train_running:
            return
        if self._pathgen_running:
            return
        if self.region is None:
            return
        if self._pending_switch_room_id is not None:
            return
        if self._latest_world_pts is None or self._latest_robot_pose is None:
            return
        if getattr(self.region.model, 'network', None) is None:
            return

        world_pts = np.asarray(self._latest_world_pts, dtype=np.float32).copy()
        robot_pose = np.asarray(self._latest_robot_pose, dtype=np.float32).copy()
        region = self.region

        if len(world_pts) < 50:
            return

        self._train_running = True

        def _worker():
            tick_t0 = time.perf_counter()
            try:
                # Bail if region was switched while queued
                if region is not self.region:
                    return

                epoch = int(self._train_epoch_per_tick)
                region.last_robot_pose = np.array(
                    [robot_pose[0], robot_pose[1], 0.0], dtype=np.float32)

                n_samples = self._train_samples

                # Use SLAM obstacle cloud for self-consistent training
                bound_np = region.bound.detach().cpu().numpy()
                slam_obs = self._slam_obstacle_cloud(bound_np)
                if slam_obs is not None and len(slam_obs) >= 20:
                    import open3d as o3d
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(
                        slam_obs.astype(np.float64))
                    pcd.estimate_normals(
                        search_param=o3d.geometry.KDTreeSearchParamHybrid(
                            radius=0.1, max_nn=30))
                    pcd.orient_normals_towards_camera_location(
                        camera_location=robot_pose)
                    region.lidar_obs = pcd
                    region.all_lidar = []

                result = region.get_data_from_lidar(
                    world_pts, robot_pose, device="cuda:0", n_samples=n_samples)
                if result is None:
                    return
                points, closest = result
                if points.shape[0] == 0:
                    return

                raw = torch.cat([points, closest], dim=1)
                if not torch.isfinite(raw).all():
                    return

                raw_cpu = raw.detach().cpu()
                raw_cpu = region._filter_raw_rows_to_bound(raw_cpu)
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

                with self._train_lock:
                    region.model.Params['Training']['Learning Rate'] = region._current_lr()

                    loss = None
                    completed = 0
                    for _ep in range(max(1, epoch)):
                        if self._pathgen_running:
                            break
                        all_data = n14
                        if len(all_data) > 7000:
                            idx = torch.randperm(len(all_data), device=region.device)[:7000]
                            all_data = all_data[idx]
                        loss_new, _ = region.model.train_core(
                            epoch=1, frame_data=all_data, is_one_frame=True)
                        completed += 1
                        if loss_new is not None:
                            loss = loss_new

                    region._sync_model_refs()
                    # Discard if region was switched mid-training
                    if region is not self.region:
                        return
                    if loss is not None:
                        region.epoch += completed
                        region.train_tick_ms.append(round((time.perf_counter() - tick_t0) * 1e3, 2))
                if region.epoch > 0 and region.epoch % 50 == 0:
                    region.save_model()
                    region.save_checkpoint(region.epoch)

                # save_training_vis disabled: expensive per-epoch disk I/O + GPU inference
                # if region.epoch > 0 and region.epoch % 40 == 0:
                #     try:
                #         n14_vis = n14[:, :6].detach()
                #         n14_spd = n14[:, 6:8].detach()
                #         region.save_training_vis(
                #             n14_vis, n14_spd, robot_xy=robot_pose[:2])
                #     except Exception:
                #         pass

            except Exception as e:
                self.get_logger().error(f"[TRAIN TICK] {e}")
                import traceback
                traceback.print_exc()
            finally:
                self._train_running = False

        threading.Thread(target=_worker, daemon=True).start()

    def _slam_obstacle_cloud(self, bound_np: np.ndarray) -> np.ndarray | None:
        if self._slam_arr is None:
            return None
        cell = float(self._slam_cell_size)
        ox, oy = float(self._slam_origin[0]), float(self._slam_origin[1])
        H, W = self._slam_arr.shape

        c0 = max(0, int((bound_np[0, 0] - ox) / cell))
        c1 = min(W, int((bound_np[1, 0] - ox) / cell) + 1)
        r0 = max(0, int((bound_np[0, 1] - oy) / cell))
        r1 = min(H, int((bound_np[1, 1] - oy) / cell) + 1)
        if c0 >= c1 or r0 >= r1:
            return None

        patch = self._slam_arr[r0:r1, c0:c1]
        occ_mask = patch > 50
        rows, cols = np.nonzero(occ_mask)
        if len(rows) == 0:
            return None

        xs = (cols + c0 + 0.5) * cell + ox
        ys = (rows + r0 + 0.5) * cell + oy
        zs = np.zeros_like(xs)
        return np.stack([xs, ys, zs], axis=1).astype(np.float32)

    @staticmethod
    def _clone_room(room) -> WatershedRoom:
        if isinstance(room, WatershedRoom):
            return WatershedRoom(
                id=room.id,
                centroid_xy=room.centroid_xy.copy(),
                peak_clearance_m=room.peak_clearance_m,
                bounding_box=room.bounding_box.copy(),
                area_m2=room.area_m2,
                confirmed=room.confirmed,
            )
        # Fallback for any legacy RoomSegment (e.g. from room_graph.load())
        return RoomSegment(
            id=room.id,
            line_segments=list(getattr(room, 'line_segments', [])),
            bounding_box=np.array(room.bounding_box, dtype=np.float32).copy(),
            centroid=np.array(room.centroid, dtype=np.float32).copy(),
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Room segmentation tick (every 3s, passive)
    # ══════════════════════════════════════════════════════════════════════════

    def _segmentation_tick(self):
        # ── Apply pending result from background thread (main-thread safe) ──
        if self._seg_result is not None:
            rooms, doorways = self._seg_result
            self._seg_result = None
            self._apply_segmentation_result(rooms, doorways)

        # ── Launch new computation if not already running ──
        if self._seg_running:
            return
        if self._slam_arr is None:
            return

        # Snapshot SLAM data for the background thread (avoid race on map updates)
        slam_snap  = self._slam_arr.copy()
        cell_m     = float(self._slam_cell_size)
        origin_xy  = self._slam_origin.copy()

        # Frontier mask: free cells adjacent to unknown cells
        arr16 = slam_snap.astype(np.int16)
        free  = (arr16 == 0)
        unk   = (arr16 < 0)
        H, W  = slam_snap.shape
        front = np.zeros((H, W), dtype=bool)
        front[:-1, :] |= free[:-1, :] & unk[1:,  :]
        front[1:,  :] |= free[1:,  :] & unk[:-1, :]
        front[:,  :-1] |= free[:, :-1] & unk[:,  1:]
        front[:,   1:] |= free[:,  1:] & unk[:,  :-1]

        self._seg_running = True

        def _worker():
            try:
                rooms, doorways = self.segmenter.update(
                    slam_snap, cell_m, origin_xy, frontier_mask=front)
                if rooms:
                    self._seg_result = (rooms, doorways)
            except Exception as e:
                self.get_logger().error(f"[SEG THREAD] {e}")
            finally:
                self._seg_running = False

        threading.Thread(target=_worker, daemon=True).start()

    def _apply_segmentation_result(self, rooms, doorways):
        """Apply segmentation output — called from main thread only."""
        self._visible_rooms = list(rooms)
        self._doorways = doorways

        # Persistence tracking
        current_ids = {r.id for r in rooms}
        new_persist: dict[int, int] = {}
        for rid in current_ids:
            new_persist[rid] = self._room_persist.get(rid, 0) + 1

        # Persistence inheritance: when rooms merge/split, new room IDs that cover
        # the same area as a vanished room inherit its persistence count so that
        # merge→re-split oscillations don't restart the confirmation clock.
        vanished_ids = set(self._room_persist.keys()) - current_ids
        appeared_ids = current_ids - set(self._room_persist.keys())
        if vanished_ids and appeared_ids:
            new_room_map = {r.id: r for r in rooms}
            for new_id in appeared_ids:
                nr = new_room_map.get(new_id)
                if nr is None:
                    continue
                best_inherit = 0
                for old_id in vanished_ids:
                    pr = self._accepted_rooms.get(old_id)
                    if pr is None:
                        continue
                    ov = _bbox_overlap_ratio_np(nr.bounding_box, pr.bounding_box)
                    if ov > 0.30:
                        best_inherit = max(best_inherit,
                                           self._room_persist.get(old_id, 0))
                if best_inherit > 0:
                    new_persist[new_id] = best_inherit  # carry over, don't reset

        self._room_persist = new_persist

        visible_room_map = {room.id: room for room in rooms}
        for rid, room in visible_room_map.items():
            self._accepted_rooms[rid] = self._clone_room(room)
            self._room_miss[rid] = 0

        for rid in list(self._accepted_rooms.keys()):
            if rid in current_ids:
                continue
            miss = self._room_miss.get(rid, 0) + 1
            self._room_miss[rid] = miss
            if miss >= ROOM_FORGET_TICKS:
                self._accepted_rooms.pop(rid, None)
                self._room_miss.pop(rid, None)
                self._room_persist.pop(rid, None)
                self._confirmed_room_ids.discard(rid)
                # self.get_logger().info(
                #     f"[ROOM FORGOTTEN] room {rid} (missing {miss} ticks)")

        graph_rooms = [self._accepted_rooms[rid]
                       for rid in sorted(self._accepted_rooms.keys())]
        if not graph_rooms:
            graph_rooms = list(rooms)
        self._rooms = graph_rooms
        self.room_graph.update(graph_rooms, doorways)

        # Promote rooms using watershed's own frontier-based confirmation.
        # Once confirmed, a room stays confirmed until it disappears (ROOM_FORGET_TICKS).
        # We don't remove on a single unconfirmed tick — frontier fluctuations cause false removals.
        for room in rooms:
            if room.confirmed and room.id not in self._confirmed_room_ids:
                self._confirmed_room_ids.add(room.id)
                self.get_logger().info(
                    f"[ROOM CONFIRMED] room {room.id} "
                    f"area={room.area_m2:.1f}m² clr={room.peak_clearance_m:.2f}m")

        self._assign_confirmed_rooms_to_regions()

        # Determine which room the robot is in (debounced)
        if self.robot_pos is not None:
            room_id = self.room_graph.room_for_point(self.robot_pos[:2])
            if room_id == self._active_room_candidate_id:
                self._active_room_candidate_ticks += 1
            else:
                self._active_room_candidate_id = room_id
                self._active_room_candidate_ticks = 1
            if (self._active_room_candidate_id is not None and
                    self._active_room_candidate_ticks >= ROOM_SWITCH_STABILITY_TICKS):
                self.active_room_id = self._active_room_candidate_id
            if self._prev_room_id is None:
                self._prev_room_id = self.active_room_id
            self._maybe_bind_active_room_to_current_region()

        # self.get_logger().info(
        #     f"[SEG] rooms={len(self._visible_rooms)} accepted={len(self._rooms)} "
        #     f"confirmed={len(self._confirmed_room_ids)} "
        #     f"doorways={len(doorways)} segs={len(self._all_segments)} "
        #     f"active_room={self.active_room_id}")

    # ══════════════════════════════════════════════════════════════════════════
    # Control tick — path following (ported from gridmap_node)
    # ══════════════════════════════════════════════════════════════════════════

    def _control_tick(self):
        self._update_robot_pose_from_tf()
        active_pos = self.robot_pos
        active_path = self._path
        if self._mapping_complete:
            self._stop_robot()
            return
        if active_pos is None or active_path is None or self._current_goal is None:
            self._stop_robot()
            return
        if self.mode == "WARMUP":
            self._stop_robot()
            return
        try:
            if self.navigator.drive_path(active_path):
                self._controller_path_done = True
            if (self._current_goal_metric is not None
                    and self._current_goal_metric.get("time_to_first_motion_s") is None
                    and abs(float(getattr(self.navigator, "last_cmd_v", 0.0))) > 0.02):
                self._current_goal_metric["time_to_first_motion_s"] = (
                    float(time.monotonic() - self._run_start_mono)
                    - float(self._current_goal_metric["start_monotonic_s"])
                )
        except Exception as e:
            self.get_logger().warn(f"[CONTROL] {e}")
            self._stop_robot()

    def _stop_robot(self):
        self.cmd_pub.publish(Twist())

    def _begin_goal_metric(self, goal_xy: np.ndarray, robot_xy: np.ndarray, selection_method: str) -> None:
        self._current_goal_metric = {
            "goal_xy": [float(goal_xy[0]), float(goal_xy[1])],
            "start_xy": [float(robot_xy[0]), float(robot_xy[1])],
            "start_monotonic_s": float(time.monotonic() - self._run_start_mono),
            "selection_method": str(selection_method),
            "frontier_candidates": int(self._last_frontier_candidate_count),
            "path_generation_attempts": 0,
            "replan_labels": [],
            "time_to_first_path_s": None,
            "time_to_first_motion_s": None,
            "time_to_goal_s": None,
            "time_in_rotate_only_s": 0.0,
            "distance_at_start_m": float(np.linalg.norm(goal_xy - robot_xy)),
            "distance_at_finish_m": None,
            "result": "active",
        }

    def _finalize_goal_metric(self, result: str, robot_xy: np.ndarray | None = None) -> None:
        gm = self._current_goal_metric
        if gm is None:
            return
        gm["result"] = str(result)
        finish_t = float(time.monotonic() - self._run_start_mono)
        start_t = float(gm.get("start_monotonic_s", finish_t))
        if gm.get("time_to_goal_s") is None:
            gm["time_to_goal_s"] = max(0.0, finish_t - start_t)
        if robot_xy is not None:
            goal_xy = np.asarray(gm["goal_xy"], dtype=np.float32)
            gm["distance_at_finish_m"] = float(np.linalg.norm(goal_xy - np.asarray(robot_xy, dtype=np.float32)))
        self._goal_metrics.append(gm)
        self._current_goal_metric = None

    def _mark_mapping_complete(self, reason: str) -> None:
        if self._mapping_complete:
            return
        self._mapping_complete = True
        self.mode = "COMPLETE"
        self._stop_robot()
        self._path = None
        self._control_path = None
        self._current_goal = None
        self._current_goal_kind = None
        self.goal_xy = None
        self.navigator.reset_path()
        self.get_logger().info(f"[FLOW] Mapping complete: {reason}")
        if not self._mapping_completion_saved:
            self._mapping_completion_saved = True
            try:
                self.save_all()
            except Exception as exc:
                self.get_logger().warn(f"[FLOW] mapping-complete save failed: {exc}")

    def _set_direct_path(
        self,
        robot_xy: np.ndarray,
        goal_xy: np.ndarray,
        path_xy: np.ndarray,
        *,
        goal_kind: str,
        label: str,
        target_room_id: int | None = None,
        target_region_ver: int | None = None,
    ) -> bool:
        path_arr = np.asarray(path_xy, dtype=np.float32)
        if path_arr.ndim != 2 or path_arr.shape[1] != 2 or len(path_arr) < 2:
            return False
        if float(np.linalg.norm(path_arr[0] - robot_xy)) > 0.03:
            path_arr = np.vstack([np.asarray(robot_xy, dtype=np.float32), path_arr])
        full_path = resample_polyline(path_arr, ds=0.05)
        if full_path is None or len(full_path) < 2:
            return False

        self._path = np.asarray(full_path, dtype=np.float32)
        self._control_path = self._path
        self._current_goal = np.asarray(goal_xy, dtype=np.float32).copy()
        self._current_goal_kind = str(goal_kind)
        self.goal_xy = self._current_goal.copy()
        self._inter_region_target_room_id = (int(target_room_id)
                                             if target_room_id is not None else None)
        self._inter_region_target_region_ver = (int(target_region_ver)
                                                if target_region_ver is not None else None)
        self.navigator.reset_path()
        self._controller_path_done = False
        self._path_total_dist = float(np.sum(np.linalg.norm(
            np.diff(self._path, axis=0), axis=1)))
        self._midpoint_replanned = False
        self._best_goal_dist = float(np.linalg.norm(robot_xy - self._current_goal))
        self._last_progress_time = float(self.get_clock().now().nanoseconds) * 1e-9
        self._last_path_progress_s = 0.0
        self._segment_start_xy = np.asarray(robot_xy, dtype=np.float32).copy()
        self._pathgen_fail_count = 0
        self._nonprogress_pathgen_count = 0
        self._publish_goal_marker(self._current_goal)
        self._publish_nav_path(self._path)
        self._publish_field_path(self._path)
        self.get_logger().info(
            f"[PATH] {label} direct accepted goal=({self._current_goal[0]:.2f},"
            f"{self._current_goal[1]:.2f}) wps={len(self._path)} "
            f"len={self._path_total_dist:.2f}m goal_kind={goal_kind}")
        return True

    def _region_center_xy(self, region: Region) -> np.ndarray:
        if getattr(region, "spawn_center", None) is not None:
            return np.asarray(region.spawn_center[:2], dtype=np.float32)
        b = region.bound.detach().cpu().numpy()
        return np.array([
            0.5 * (float(b[0, 0]) + float(b[1, 0])),
            0.5 * (float(b[0, 1]) + float(b[1, 1])),
        ], dtype=np.float32)

    def _region_for_frontier_point(self, point_xy: np.ndarray) -> Region | None:
        room_id = self.room_graph.room_for_point(np.asarray(point_xy, dtype=np.float32))
        if room_id is not None:
            reg = self.room_regions.get(int(room_id))
            if reg is not None:
                return reg

        contenders = []
        for reg in self.all_regions.values():
            try:
                if reg.in_bound(np.asarray(point_xy, dtype=np.float32)):
                    contenders.append(reg)
            except Exception:
                continue
        if not contenders:
            return None
        return min(
            contenders,
            key=lambda reg: float(np.linalg.norm(
                self._region_center_xy(reg) - np.asarray(point_xy, dtype=np.float32))))

    def _all_frontier_candidates(self) -> np.ndarray:
        if self._slam_arr is None or self.robot_pos is None:
            return np.zeros((0, 2), dtype=np.float32)

        arr = self._slam_arr
        cs = self._slam_cell_size
        ox, oy = float(self._slam_origin[0]), float(self._slam_origin[1])
        free = (arr == 0)
        unknown = (arr == -1)
        kernel = np.ones((3, 3), dtype=np.uint8)
        free_dilated = cv2.dilate(free.astype(np.uint8), kernel, iterations=1)
        frontier_mask = (free_dilated > 0) & unknown
        frontier_cells = np.argwhere(frontier_mask)
        if len(frontier_cells) == 0:
            return np.zeros((0, 2), dtype=np.float32)

        cand = np.zeros((len(frontier_cells), 2), dtype=np.float32)
        cand[:, 0] = ox + frontier_cells[:, 1] * cs + cs * 0.5
        cand[:, 1] = oy + frontier_cells[:, 0] * cs + cs * 0.5

        robot_xy = np.asarray(self.robot_pos[:2], dtype=np.float32)
        adjusted: list[np.ndarray] = []
        for pt in cand:
            adj = self._shift_frontier_goal_to_free_space(pt, robot_xy)
            if adj is None:
                continue
            adjusted.append(adj.astype(np.float32))
        if not adjusted:
            return np.zeros((0, 2), dtype=np.float32)
        return np.asarray(adjusted, dtype=np.float32)

    def _refresh_region_frontier_banks(self) -> None:
        banks: dict[int, list[list[float]]] = {
            int(ver): [] for ver in self.all_regions.keys()
        }
        if self.region is not None:
            banks.setdefault(int(self.region.version), [])

        cand = self._all_frontier_candidates()
        for pt in cand:
            reg = self._region_for_frontier_point(pt)
            if reg is None:
                continue
            banks.setdefault(int(reg.version), []).append(
                [float(pt[0]), float(pt[1])]
            )
        self._region_frontier_banks = banks

    def _filter_frontier_candidates(
        self,
        cand: np.ndarray,
        robot_xy: np.ndarray,
        *,
        apply_dist_cap: bool,
    ) -> tuple[np.ndarray, np.ndarray, str]:
        cand = np.asarray(cand, dtype=np.float32)
        if cand.ndim == 1 and cand.size == 2:
            cand = cand.reshape(1, 2)
        if cand.size == 0:
            return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32), "none:no_candidates"

        d = np.linalg.norm(cand - robot_xy[None, :], axis=1)
        min_standoff = max(0.15, float(self.get_parameter("goal_reached_dist").value) * 0.85)
        keep = d >= min_standoff
        cand = cand[keep]
        d = d[keep]
        if len(cand) == 0:
            return cand, d, "none:standoff_filter"

        if apply_dist_cap:
            keep = d < (self.region_radius * self.frontier_cap_factor)
            cand = cand[keep]
            d = d[keep]
            if len(cand) == 0:
                return cand, d, "none:distance_cap"

        if self._visited_goals:
            visited = np.asarray(self._visited_goals, dtype=np.float32)
            if visited.ndim == 2 and visited.shape[1] == 2:
                min_vd = np.min(np.linalg.norm(
                    cand[:, None, :] - visited[None, :, :], axis=2), axis=1)
                keep = min_vd >= self._goal_blacklist_radius
                cand = cand[keep]
                d = d[keep]
        if len(cand) == 0:
            return cand, d, "none:blacklist"
        return cand, d, ""

    def _choose_frontier_candidate(
        self,
        cand: np.ndarray,
        d: np.ndarray,
    ) -> tuple[np.ndarray | None, str]:
        if len(cand) == 0:
            return None, "none:no_candidates"

        best_cand = cand[np.argmin(d)]
        selection_method = "single_point"
        try:
            from sklearn.cluster import DBSCAN
            db = DBSCAN(eps=0.20, min_samples=6).fit(cand)
            labels = db.labels_
            cluster_reps = []
            for lbl in np.unique(labels):
                members = cand[labels == lbl]
                member_d = d[labels == lbl]
                if len(members) == 0:
                    continue
                centroid = members.mean(axis=0)
                rep = members[np.argmin(np.linalg.norm(
                    members - centroid[None, :], axis=1))]
                d_rep = float(np.min(member_d)) if len(member_d) > 0 else float("inf")
                cluster_reps.append((rep, d_rep))
            if cluster_reps:
                best_cand = min(cluster_reps, key=lambda item: item[1])[0]
                selection_method = f"clustered_dbscan(n={len(cluster_reps)})"
        except Exception:
            selection_method = "single_point:dbscan_failed"
        return np.asarray(best_cand, dtype=np.float32), selection_method

    def _nearest_region_route_with_frontiers(
        self,
        robot_xy: np.ndarray,
    ) -> tuple[list[dict] | None, int | None, int | None]:
        if self.region is None:
            return None, None, None
        current_room = self.active_room_id
        if current_room is None:
            current_room = self.room_graph.room_for_point(robot_xy)
        if current_room is None or current_room not in self.room_graph.rooms:
            return None, None, None

        region_to_rooms: dict[int, list[int]] = {}
        for room_id, reg in self.room_regions.items():
            region_to_rooms.setdefault(int(reg.version), []).append(int(room_id))

        best_route = None
        best_room = None
        best_region = None
        best_cost = float("inf")
        for region_ver, pts in self._region_frontier_banks.items():
            if int(region_ver) == int(self.region.version) or not pts:
                continue
            for room_id in region_to_rooms.get(int(region_ver), []):
                if room_id not in self.room_graph.rooms:
                    continue
                try:
                    route = self.room_graph.shortest_path(int(current_room), int(room_id))
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue
                cost = 0.0
                prev = np.asarray(robot_xy, dtype=np.float32)
                for step in route[1:]:
                    pos = np.asarray(step["position"][:2], dtype=np.float32)
                    cost += float(np.linalg.norm(pos - prev))
                    prev = pos
                if cost < best_cost:
                    best_cost = cost
                    best_route = route
                    best_room = int(room_id)
                    best_region = int(region_ver)
        return best_route, best_room, best_region

    def _route_positions_from_room_route(
        self,
        robot_xy: np.ndarray,
        room_route: list[dict],
    ) -> np.ndarray | None:
        if not room_route:
            return None
        points = [np.asarray(robot_xy, dtype=np.float32)]
        for step in room_route[1:]:
            pos = np.asarray(step["position"][:2], dtype=np.float32)
            if float(np.linalg.norm(pos - points[-1])) < 0.05:
                continue
            seg = straight_line_path(points[-1], pos, step=0.10, collision_fn=self._point_in_collision_slam)
            if seg is None or len(seg) < 2 or float(np.linalg.norm(seg[-1] - pos)) > 0.18:
                return None
            points.extend(np.asarray(seg[1:], dtype=np.float32))
        if len(points) < 2:
            return None
        return np.asarray(points, dtype=np.float32)

    def _start_inter_region_route(self, robot_xy: np.ndarray) -> bool:
        self._refresh_region_frontier_banks()
        room_route, target_room_id, target_region_ver = self._nearest_region_route_with_frontiers(robot_xy)
        if room_route is None or target_room_id is None or target_region_ver is None:
            return False

        path_xy = self._route_positions_from_room_route(robot_xy, room_route)
        if path_xy is None or len(path_xy) < 2:
            self.get_logger().warn(
                f"[NAV] no collision-free inter-region route to R{target_region_ver}")
            return False

        self._inter_region_route_room_ids = [
            int(step["id"]) for step in room_route if step.get("type") == "room"
        ]
        self._last_frontier_candidate_count = int(
            len(self._region_frontier_banks.get(int(target_region_ver), [])))
        self._last_frontier_selection_method = f"inter_region_route(R{target_region_ver})"
        route_str = " -> ".join(
            f"{step['type'][0].upper()}{step['id']}" for step in room_route
        )
        self.get_logger().info(
            f"[NAV] local frontiers exhausted in R{self.region.version}; "
            f"routing to nearest frontier region R{target_region_ver} via {route_str}")
        return self._set_direct_path(
            robot_xy,
            np.asarray(path_xy[-1], dtype=np.float32),
            path_xy,
            goal_kind="handoff",
            label="inter-region",
            target_room_id=target_room_id,
            target_region_ver=target_region_ver,
        )

    def _start_path_generation(self, robot_xy: np.ndarray, goal: np.ndarray,
                               modes, label: str) -> bool:
        if self.region is None or getattr(self.region.model, 'function', None) is None:
            return False
        with self._pathgen_lock:
            if self._pathgen_running:
                return False
            self._pathgen_running = True
            self._pathgen_result = None

        region = self.region
        robot_xy = np.asarray(robot_xy, dtype=np.float32).copy()
        goal = np.asarray(goal, dtype=np.float32).copy()
        max_steps = int(self.path_max_steps)
        if isinstance(modes, bool):
            modes = [modes]
        modes = list(modes)
        start_t0 = time.perf_counter()
        if self._current_goal_metric is not None:
            self._current_goal_metric["path_generation_attempts"] = (
                int(self._current_goal_metric.get("path_generation_attempts", 0)) + 1
            )
            self._current_goal_metric.setdefault("replan_labels", []).append(str(label))

        def _worker():
            attempts = []
            err = None
            try:
                with self._train_lock:
                    for forward_only in modes:
                        if region is not self.region:
                            break
                        raw = region.generate_field_path(
                            robot_xy, goal, step_size=0.10, max_steps=max_steps,
                            forward_only=forward_only)
                        attempts.append((raw, bool(forward_only)))
            except Exception as exc:
                err = exc
            finally:
                elapsed_ms = round((time.perf_counter() - start_t0) * 1e3, 2)
                with self._pathgen_lock:
                    self._pathgen_result = (
                        region, robot_xy, goal, attempts, label, err, elapsed_ms)
                    self._pathgen_running = False

        threading.Thread(target=_worker, daemon=True).start()
        return True

    def _consume_path_generation(self) -> bool:
        with self._pathgen_lock:
            result = self._pathgen_result
            self._pathgen_result = None
        if result is None:
            return False

        region, robot_xy, goal, attempts, label, err, elapsed_ms = result
        self._path_gen_times_ms.append(float(elapsed_ms))
        if err is not None or region is not self.region:
            if err is not None:
                self.get_logger().warn(f"[PATH] {label} generation error: {err}")
            return True
        if self._current_goal is None:
            return True
        if float(np.linalg.norm(goal - self._current_goal[:2])) > 0.25:
            return True

        reject_reasons = []
        for raw, forward_only in attempts:
            mode_label = "fwd" if forward_only else "bi"
            if self._accept_raw_path(robot_xy, goal, raw, forward_only, goal, label):
                if self._current_goal_metric is not None and self._current_goal_metric.get("time_to_first_path_s") is None:
                    self._current_goal_metric["time_to_first_path_s"] = float(time.monotonic() - self._run_start_mono) - float(self._current_goal_metric["start_monotonic_s"])
                if reject_reasons:
                    self.get_logger().info(
                        f"[PATH] {label} accepted after fallback; rejected="
                        f"{'; '.join(reject_reasons)}")
                return True
            reason = getattr(self, '_last_path_reject_reason', None) or 'unknown'
            reject_reasons.append(f"{mode_label}:{reason}")

        has_nonprogress = any("nonprogress_segment" in r for r in reject_reasons)
        if has_nonprogress:
            self._nonprogress_pathgen_count += 1
        else:
            self._nonprogress_pathgen_count = 0

        self.get_logger().info(
            f"[PATH] {label} rejected goal=({goal[0]:.2f},{goal[1]:.2f}) "
            f"attempts={'; '.join(reject_reasons) if reject_reasons else 'none'}")

        if self._nonprogress_pathgen_count >= 2 and self._current_goal is not None:
            fail_count = self._nonprogress_pathgen_count
            self.get_logger().warn(
                f"[NAV] abandoning frontier — path endpoint made no progress "
                f"{fail_count}x goal=({self._current_goal[0]:.2f},"
                f"{self._current_goal[1]:.2f}) reasons=[{'; '.join(reject_reasons)}]")
            self._visited_goals.append(self._current_goal[:2].copy())
            self._clear_nav_state(
                f"non-progress path generation {fail_count}x reasons=[{'; '.join(reject_reasons)}]")
            return True

        self._pathgen_fail_count += 1
        if self._pathgen_fail_count >= 3 and self._current_goal is not None:
            fail_count = self._pathgen_fail_count
            self.get_logger().warn(
                f"[NAV] abandoning frontier — path gen failed {fail_count}x "
                f"goal=({self._current_goal[0]:.2f},{self._current_goal[1]:.2f}) "
                f"reasons=[{'; '.join(reject_reasons)}]")
            self._visited_goals.append(self._current_goal[:2].copy())
            self._pathgen_fail_count = 0
            self._clear_nav_state(f"path generation failed {fail_count}x reasons=[{'; '.join(reject_reasons)}]")
        return True

    # ══════════════════════════════════════════════════════════════════════════
    # Exploration tick — ported from gridmap_node
    # ══════════════════════════════════════════════════════════════════════════

    def _exploration_tick(self):
        self._exploration_tick_count += 1
        log_now = (self._exploration_tick_count % 10 == 0)
        if self._mapping_complete:
            self._stop_robot()
            return

        self._update_robot_pose_from_tf()
        if self.robot_pos is None:
            # if log_now:
            #     self.get_logger().info("[NAV] waiting for robot pose (TF not ready)")
            return

        # Record trajectory (downsample: every 5th tick ≈ 0.5s)
        if self._exploration_tick_count % 5 == 0:
            self._robot_traj.append(self.robot_pos[:2].copy())

        if self._process_pending_room_switch():
            return

        robot_xy = self.robot_pos[:2]
        self._maybe_spawn_region_from_exploration(robot_xy)
        if self._process_pending_room_switch():
            return

        # Check for room boundary crossings → region switch
        self._maybe_switch_for_room_crossing(robot_xy)

        # Publish frontier markers every ~1s
        if self._exploration_tick_count % 10 == 0:
            self._publish_frontier_markers()

        # ── Warmup gate ──
        if self.mode == "WARMUP":
            self._stop_robot()
            if self.region is not None:
                cur_epochs = int(self.region.epoch) - self._warmup_start_epoch
                if cur_epochs >= self._warmup_epoch_target:
                    self.mode = "EXPLORE"
                    self.get_logger().info(
                        f"[FLOW] Warmup complete R{self.region.version} "
                        f"(trained {cur_epochs} epochs) → EXPLORE")
                elif log_now:
                    self.get_logger().info(
                        f"[WARMUP] R{self.region.version} "
                        f"epoch={self.region.epoch} "
                        f"({cur_epochs}/{self._warmup_epoch_target})")
            return

        if self._consume_path_generation():
            return

        # ── Following existing path ──
        if self._path is not None and len(self._path) >= 2:
            self._publish_field_path(self._path)
            if self._current_goal is None:
                self._path = None
                self._control_path = None
                self.navigator.reset_path()
                return
            if self._current_goal_kind == "frontier" and not self._goal_is_still_valid(self._current_goal):
                self.get_logger().info(
                    f"[NAV] invalidating obstructed goal "
                    f"({self._current_goal[0]:.2f},{self._current_goal[1]:.2f})")
                self._visited_goals.append(self._current_goal[:2].copy())
                self._clear_nav_state(
                    f"goal obstructed ({self._current_goal[0]:.2f},"
                    f"{self._current_goal[1]:.2f})")
                return

            now_sec = float(self.get_clock().now().nanoseconds) * 1e-9
            d_goal = float(np.linalg.norm(robot_xy - self._current_goal))
            d_path_end = float(np.linalg.norm(robot_xy - self._path[-1]))
            goal_reached_dist = float(self.get_parameter("goal_reached_dist").value)

            progress_eps = float(self.get_parameter("follow_progress_eps").value)
            stuck_timeout = float(self.get_parameter("follow_stuck_timeout_sec").value)
            eff_eps = min(progress_eps, max(0.01, 0.05 * d_goal))
            path_progress_s = float(getattr(self.navigator, 'last_proj_s', 0.0))
            path_advanced = path_progress_s > (self._last_path_progress_s + 0.08)

            segment_replan_dist = float(self.get_parameter("segment_replan_dist_m").value)
            if self._segment_start_xy is not None and segment_replan_dist > 0.0:
                segment_progress = float(np.linalg.norm(robot_xy - self._segment_start_xy))
                if (segment_progress >= segment_replan_dist and
                        d_goal > max(goal_reached_dist + 0.10, segment_replan_dist)):
                    self._start_path_generation(
                        robot_xy, self._current_goal, [False, True], "segment")

            if (d_goal + eff_eps < self._best_goal_dist) or (d_goal + 0.01 < self._best_goal_dist):
                self._best_goal_dist = d_goal
                self._last_progress_time = now_sec
                self._stuck_replan_count = 0
            elif path_advanced:
                self._last_path_progress_s = path_progress_s
                self._last_progress_time = now_sec
            elif getattr(self.navigator, '_rotating', False):
                # Rotation-in-place is legitimate progress — don't penalise heading alignment
                self._last_progress_time = now_sec
                if self._current_goal_metric is not None:
                    self._current_goal_metric["time_in_rotate_only_s"] = (
                        float(self._current_goal_metric.get("time_in_rotate_only_s", 0.0)) + self.dt
                    )

            # [FOLLOWING] debug removed — too verbose; re-enable by uncommenting if needed
            # if log_now:
            #     tgt = getattr(self.navigator, 'last_target', None)
            #     tgt_str = ("None" if tgt is None else
            #                f"({float(tgt[0]):.2f},{float(tgt[1]):.2f})")
            #     self.get_logger().info(
            #         f"[FOLLOWING] d_goal={d_goal:.2f}m d_path_end={d_path_end:.2f}m "
            #         f"path_wps={len(self._path)} "
            #         f"cmd=({float(self.navigator.last_cmd_v):.2f},"
            #         f"{float(self.navigator.last_cmd_w):.2f}) "
            #         f"target={tgt_str}")

            if d_goal < goal_reached_dist:
                if self._current_goal_kind == "frontier":
                    self._visited_goals.append(self._current_goal[:2].copy())
                    self._finalize_goal_metric("reached", robot_xy)
                self._clear_nav_state(
                    f"goal reached ({self._current_goal[0]:.2f},"
                    f"{self._current_goal[1]:.2f}) d={d_goal:.2f}m")
                return

            # Partial path exhausted but goal still far → extend
            if self._current_goal_kind != "handoff" and d_path_end < 0.20 and d_goal > 0.50:
                self._start_path_generation(
                    robot_xy, self._current_goal, [False, True], "extend")
                return

            if self._controller_path_done:
                self._controller_path_done = False
                if self._current_goal_kind == "handoff":
                    self._clear_nav_state(
                        f"inter-region handoff completed "
                        f"({self._current_goal[0]:.2f},{self._current_goal[1]:.2f})")
                    return
                if d_goal < 0.35:
                    self._visited_goals.append(self._current_goal[:2].copy())
                    self._clear_nav_state(
                        f"short segment completed near goal "
                        f"({self._current_goal[0]:.2f},{self._current_goal[1]:.2f}) "
                        f"d={d_goal:.2f}m")
                    return
                self._start_path_generation(
                    robot_xy, self._current_goal, [False, True], "endpoint")
                return

            # Stuck detection
            adaptive_timeout = stuck_timeout * (2.0 if d_goal < 1.0 else 1.0)
            if now_sec - self._last_progress_time > adaptive_timeout:
                self._stuck_replan_count += 1
                # self.get_logger().warn(
                #     f"[STUCK] no progress for {adaptive_timeout:.1f}s "
                #     f"(best={self._best_goal_dist:.2f}m now={d_goal:.2f}m) "
                #     f"replan #{self._stuck_replan_count}")
                if self._current_goal_kind == "handoff":
                    self._clear_nav_state(
                        f"inter-region handoff stuck "
                        f"({self._current_goal[0]:.2f},{self._current_goal[1]:.2f})")
                else:
                    self._start_path_generation(
                        robot_xy, self._current_goal, [False, True], "stuck")
                    self._path = None
                    self._control_path = None
                    self.navigator.reset_path()
                    self._last_progress_time = now_sec
                    if self._stuck_replan_count >= 2 and self._current_goal is not None:
                        self.get_logger().warn(
                            f"[NAV] clearing goal: stuck after {self._stuck_replan_count} "
                            f"replans ({self._current_goal[0]:.2f},"
                            f"{self._current_goal[1]:.2f})")
                        self._visited_goals.append(self._current_goal[:2].copy())
                        self._finalize_goal_metric("stuck_cleared", robot_xy)
                        self._current_goal = None
                        self.goal_xy = None
                        self._current_goal_kind = None
                        self._stuck_replan_count = 0
                return

            # Midpoint replan for long paths
            if (self._current_goal_kind != "handoff"
                    and self._path_total_dist is not None
                    and self._path_total_dist > 2.0
                    and not self._midpoint_replanned
                    and d_goal < self._path_total_dist * 0.5):
                self._midpoint_replanned = True
                self._start_path_generation(
                    robot_xy, self._current_goal, [False, True], "midpoint")
                self._path = None
                self._control_path = None
                self.navigator.reset_path()
                return

            return

        # ── No path — need to plan ──
        self._path = None
        self._control_path = None
        self.navigator.reset_path()

        if self.region is None:
            # if log_now:
            #     self.get_logger().info("[NAV] no region yet")
            return
        if getattr(self.region.model, 'function', None) is None:
            # if log_now:
            #     self.get_logger().info("[NAV] model not initialized")
            return
        if self._pathgen_running:
            self._stop_robot()
            return

        # Pick or keep goal
        frontier_t0 = time.perf_counter()
        goal = (self._current_goal.copy() if self._current_goal is not None
                else self._pick_frontier_goal())
        if self._current_goal is None:
            self._frontier_pick_times_ms.append(round((time.perf_counter() - frontier_t0) * 1e3, 2))
        if goal is None:
            if self._start_inter_region_route(robot_xy):
                return
            self._refresh_region_frontier_banks()
            total_frontiers = sum(len(v) for v in self._region_frontier_banks.values())
            if total_frontiers == 0:
                self._mark_mapping_complete("all frontier banks exhausted")
            self._stop_robot()
            return
        if not self._goal_is_still_valid(goal):
            self.get_logger().info(
                f"[NAV] clearing goal: newly selected goal invalid "
                f"({goal[0]:.2f},{goal[1]:.2f})")
            self._visited_goals.append(goal[:2].copy())
            self._current_goal = None
            self._current_goal_kind = None
            self.goal_xy = None
            self._stop_robot()
            return

        new_goal = self._current_goal is None
        if new_goal:
            self._stuck_replan_count = 0
            self._pathgen_fail_count = 0
            self._nonprogress_pathgen_count = 0
            self._current_goal_kind = "frontier"
            self._inter_region_target_room_id = None
            self._inter_region_target_region_ver = None
            self._inter_region_route_room_ids = []
            self._begin_goal_metric(goal, robot_xy, self._last_frontier_selection_method)
        self._current_goal = goal
        self.goal_xy = goal.copy()
        self.region.latest_goal = goal.copy()
        self._publish_goal_marker(goal)

        if new_goal or log_now:
            goal_dist = float(np.linalg.norm(goal - robot_xy))
            self.get_logger().info(
                f"[NAV] {'picked frontier' if new_goal else 'replanning'} "
                f"({goal[0]:.2f},{goal[1]:.2f}) from "
                f"({robot_xy[0]:.2f},{robot_xy[1]:.2f}) "
                f"dist={goal_dist:.2f}m candidates={self._last_frontier_candidate_count} "
                f"selection={self._last_frontier_selection_method} "
                f"epoch={self.region.epoch}")

        # Wait for enough training
        min_train = self._min_train_epochs
        if self.region.epoch < min_train:
            # Try bidirectional rollout with partial training (epoch >= 40)
            if self.region.epoch >= 40:
                self._start_path_generation(robot_xy, goal, True, "warmup-fwd")
                return
            if log_now:
                self.get_logger().info(
                    f"[NAV] warmup waiting epoch={self.region.epoch} < {min_train}")
            self._stop_robot()
            return

        # Full field path: try bidirectional first, then forward-only fallback.
        self._start_path_generation(robot_xy, goal, [False, True], "field")
        if self._path is None:
            self._stop_robot()

    def _clear_nav_state(self, reason: str | None = None):
        if reason:
            self.get_logger().info(f"[NAV] clearing goal: {reason}")
        if self._current_goal is not None and self._current_goal_kind == "frontier":
            self._finalize_goal_metric("cleared", self.robot_pos[:2] if self.robot_pos is not None else None)
        self._stop_robot()
        self._path = None
        self._control_path = None
        self._current_goal = None
        self._current_goal_kind = None
        self.goal_xy = None
        self.navigator.reset_path()
        self._controller_path_done = False
        self._path_total_dist = None
        self._midpoint_replanned = False
        self._best_goal_dist = float('inf')
        self._segment_start_xy = None
        self._pathgen_fail_count = 0
        self._nonprogress_pathgen_count = 0
        self._inter_region_target_room_id = None
        self._inter_region_target_region_ver = None
        self._inter_region_route_room_ids = []

    def _save_metrics_json(self) -> None:
        try:
            def _summary_stats(values):
                vals = [float(v) for v in values if v is not None]
                if not vals:
                    return {
                        "count": 0,
                        "mean": None,
                        "median": None,
                        "min": None,
                        "max": None,
                    }
                arr = np.asarray(vals, dtype=np.float64)
                return {
                    "count": int(arr.size),
                    "mean": float(arr.mean()),
                    "median": float(np.median(arr)),
                    "min": float(arr.min()),
                    "max": float(arr.max()),
                }

            reached_goals = [g for g in self._goal_metrics if g.get("result") == "reached"]
            motion_goals = [g for g in self._goal_metrics if g.get("time_to_first_motion_s") is not None]
            train_tick_all = []
            for reg in self.all_regions.values():
                train_tick_all.extend(list(getattr(reg, "train_tick_ms", [])))

            out = {
                "run_started_unix_s": float(self._run_start_wall_time),
                "run_duration_s": float(time.monotonic() - self._run_start_mono),
                "frontier_pick_ms": list(self._frontier_pick_times_ms),
                "path_gen_ms": list(self._path_gen_times_ms),
                "goal_metrics": list(self._goal_metrics),
                "lidar_drop_count": int(self._lidar_drop_count),
                "lidar_drop_max_lag_s": float(self._lidar_drop_max_lag_s),
                "regions": {
                    str(ver): {
                        "epoch": int(reg.epoch),
                        "train_tick_ms": list(reg.train_tick_ms),
                        "loss_history_len": int(len(getattr(reg, "loss_history", []))),
                    }
                    for ver, reg in self.all_regions.items()
                },
            }
            if self.region is not None:
                out["regions"].setdefault(
                    str(self.region.version),
                    {
                        "epoch": int(self.region.epoch),
                        "train_tick_ms": list(self.region.train_tick_ms),
                        "loss_history_len": int(len(getattr(self.region, "loss_history", []))),
                    },
                )
                train_tick_all.extend(list(getattr(self.region, "train_tick_ms", [])))
            out["summary"] = {
                "goals_total": int(len(self._goal_metrics)),
                "goals_reached": int(len(reached_goals)),
                "goal_success_rate": (
                    float(len(reached_goals) / len(self._goal_metrics))
                    if self._goal_metrics else None
                ),
                "motion_success_rate": (
                    float(len(motion_goals) / len(self._goal_metrics))
                    if self._goal_metrics else None
                ),
                "frontier_pick_ms": _summary_stats(self._frontier_pick_times_ms),
                "path_gen_ms": _summary_stats(self._path_gen_times_ms),
                "train_tick_ms": _summary_stats(train_tick_all),
                "time_to_first_path_s": _summary_stats(
                    [g.get("time_to_first_path_s") for g in self._goal_metrics]
                ),
                "time_to_first_motion_s": _summary_stats(
                    [g.get("time_to_first_motion_s") for g in self._goal_metrics]
                ),
                "time_to_goal_s": _summary_stats(
                    [g.get("time_to_goal_s") for g in reached_goals]
                ),
                "time_in_rotate_only_s": _summary_stats(
                    [g.get("time_in_rotate_only_s") for g in self._goal_metrics]
                ),
                "replans_per_goal": _summary_stats(
                    [len(g.get("replan_labels", [])) for g in self._goal_metrics]
                ),
            }
            path = os.path.join(MODEL_FOLDER, "metrics.json")
            with open(path, "w") as f:
                json.dump(out, f, indent=2)
            self.get_logger().info(
                f"[SAVE] metrics.json goals={len(self._goal_metrics)} "
                f"path_gen={len(self._path_gen_times_ms)} train_regions={len(out['regions'])}")
        except Exception as e:
            self.get_logger().error(f"[METRICS SAVE ERROR] {e}")

    # ── Path processing (ported from gridmap_node) ───────────────────────────

    def _accept_raw_path(self, robot_xy, goal, raw, forward_only, plan_goal,
                         label="field") -> bool:
        if raw is None or len(raw) <= 1:
            self._last_path_reject_reason = "raw_none_or_short"
            return False
        raw_arr = np.asarray(raw, dtype=np.float32)
        decim = self._decimate_path(raw_arr, spacing=0.10)
        smooth = self._smooth_path(decim, window=3)
        if smooth is None or len(smooth) < 2:
            self._last_path_reject_reason = "smooth_too_short"
            return False
        smooth = self._trim_at_unvisited(smooth)
        if smooth is None or len(smooth) < 2:
            self._last_path_reject_reason = f"trim_too_short(pts={0 if smooth is None else len(smooth)})"
            return False
        smooth = self._sanitize_path(robot_xy, goal, smooth)
        if smooth is None or len(smooth) < 2:
            self._last_path_reject_reason = "sanitize_rejected"
            return False
        self._last_path_reject_reason = None

        full_path = resample_polyline(np.asarray(smooth, dtype=np.float32), ds=0.05)
        if full_path is None or len(full_path) < 2:
            self._last_path_reject_reason = "resample_too_short"
            return False

        self._publish_nav_path(full_path)
        full_len = float(np.sum(np.linalg.norm(np.diff(full_path, axis=0), axis=1)))

        commit_trigger = float(self.get_parameter("path_truncate_trigger_m").value)
        commit_max = float(self._path_commit_max)
        exec_path = full_path
        truncated = False
        if full_len > min(commit_trigger, commit_max):
            exec_path = self._truncate_path_by_arclen(full_path, commit_max)
            if exec_path is not None and len(exec_path) >= 2:
                exec_len = float(np.sum(np.linalg.norm(
                    np.diff(np.asarray(exec_path, dtype=np.float32), axis=0), axis=1)))
                truncated = exec_len < full_len - 0.05

        if exec_path is None or len(exec_path) < 2:
            self._last_path_reject_reason = "exec_too_short"
            return False

        # Keep the executable segment anchored at the latest pose used for this
        # plan. Smoothing/decimation can otherwise make the controller chase a
        # stale first point and repeatedly complete tiny endpoint segments.
        exec_path = np.asarray(exec_path, dtype=np.float32)
        start_gap = float(np.linalg.norm(exec_path[0] - robot_xy))
        if start_gap > 0.03:
            exec_path = np.vstack([np.asarray(robot_xy, dtype=np.float32), exec_path])

        d_goal = float(np.linalg.norm(np.asarray(goal, dtype=np.float32) - robot_xy))
        exec_end_dist = float(np.linalg.norm(exec_path[-1] - robot_xy))
        goal_reached_dist = float(self.get_parameter("goal_reached_dist").value)
        min_segment_advance = max(goal_reached_dist + 0.08, 0.38)
        if d_goal > max(0.75, min_segment_advance + 0.20) and exec_end_dist < min_segment_advance:
            self._last_path_reject_reason = (
                f"nonprogress_segment(end={exec_end_dist:.2f},goal={d_goal:.2f},"
                f"robot=({robot_xy[0]:.2f},{robot_xy[1]:.2f}),"
                f"path_end=({exec_path[-1][0]:.2f},{exec_path[-1][1]:.2f}))")
            return False

        self._path = exec_path
        self._control_path = exec_path
        self._segment_start_xy = np.asarray(robot_xy, dtype=np.float32).copy()
        self.navigator.reset_path()
        self._controller_path_done = False
        self._path_total_dist = float(np.sum(np.linalg.norm(
            np.diff(self._path, axis=0), axis=1)))
        self._midpoint_replanned = False
        self._best_goal_dist = float(np.linalg.norm(robot_xy - self._current_goal))
        self._last_progress_time = float(self.get_clock().now().nanoseconds) * 1e-9
        self._last_path_progress_s = 0.0
        self._pathgen_fail_count = 0
        self._nonprogress_pathgen_count = 0
        self.region.latest_path = full_path.tolist()
        self._publish_field_path(self._path)
        fwd_label = 'fwd' if forward_only else 'bi'
        first_wp = self._path[1] if len(self._path) > 1 else self._path[0]
        self.get_logger().info(
            f"[PATH] {label} ({fwd_label}) accepted goal=({goal[0]:.2f},{goal[1]:.2f}) "
            f"wps={len(self._path)} full_len={full_len:.2f}m truncated={truncated} "
            f"first_wp=({first_wp[0]:.2f},{first_wp[1]:.2f})")
        return True

    def _decimate_path(self, path_xy, spacing=0.4):
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

    def _smooth_path(self, path_xy, window=5):
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

    def _sanitize_path(self, robot_xy, goal_xy, pts):
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

        if float(np.linalg.norm(arr[-1] - goal_xy)) > 0.10:
            arr = np.vstack([arr, goal_xy.astype(np.float32)])
        else:
            arr[-1] = goal_xy.astype(np.float32)

        return arr

    def _trim_at_unvisited(self, pts):
        try:
            arr = np.asarray(pts, dtype=np.float32)
            if self._slam_arr is None:
                return arr
            cell = self._slam_cell_size
            ox, oy = float(self._slam_origin[0]), float(self._slam_origin[1])
            H, W = self._slam_arr.shape

            gx = ((arr[:, 0] - ox) / cell).astype(int)
            gy = ((arr[:, 1] - oy) / cell).astype(int)
            inb = ((0 <= gx) & (gx < W) & (0 <= gy) & (gy < H))

            free = np.zeros(len(arr), dtype=bool)
            free[inb] = (self._slam_arr[gy[inb], gx[inb]] == 0)

            unvis = np.where(~free)[0]
            if len(unvis) == 0:
                return arr
            if unvis[0] <= 1:
                return arr
            trimmed = arr[:unvis[0]]
            return trimmed if len(trimmed) >= 2 else None
        except Exception:
            return pts

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
                p = a + t * (b - a)
                out.append(p.astype(np.float32))
            break

        if len(out) < 2:
            out.append(arr[1].copy())
        return np.asarray(out, dtype=np.float32)

    # ── Frontier selection ───────────────────────────────────────────────────

    def _pick_frontier_goal(self) -> np.ndarray | None:
        if self._slam_arr is None or self.robot_pos is None or self.region is None:
            self._last_frontier_candidate_count = 0
            self._last_frontier_selection_method = "none:no_slam_or_pose"
            return None

        self._refresh_region_frontier_banks()
        all_cand = self._all_frontier_candidates()
        robot_xy = np.asarray(self.robot_pos[:2], dtype=np.float32)
        cand, d, fail_reason = self._filter_frontier_candidates(
            all_cand, robot_xy, apply_dist_cap=True)
        if len(cand) == 0:
            self._last_frontier_candidate_count = 0
            self._last_frontier_selection_method = fail_reason
            return None

        self._last_frontier_candidate_count = int(len(cand))
        best_cand, selection_method = self._choose_frontier_candidate(cand, d)
        self._last_frontier_selection_method = selection_method
        return best_cand

    def _point_in_collision_slam(self, xy, pad_cells=1) -> bool:
        if self._slam_arr is None:
            return False
        c = int((float(xy[0]) - float(self._slam_origin[0])) / self._slam_cell_size)
        r = int((float(xy[1]) - float(self._slam_origin[1])) / self._slam_cell_size)
        H, W = self._slam_arr.shape
        for dr in range(-pad_cells, pad_cells + 1):
            for dc in range(-pad_cells, pad_cells + 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < H and 0 <= cc < W and int(self._slam_arr[rr, cc]) > 50:
                    return True
        return False

    def _point_has_slam_clearance(self, xy, clearance_m=0.22) -> bool:
        if self._slam_arr is None:
            return True
        pad_cells = max(1, int(math.ceil(clearance_m / max(self._slam_cell_size, 1e-6))))
        return not self._point_in_collision_slam(xy, pad_cells=pad_cells)

    def _goal_is_still_valid(self, goal_xy: np.ndarray | None) -> bool:
        if goal_xy is None:
            return False
        if self._point_in_collision_slam(goal_xy, pad_cells=1):
            return False
        if not self._point_has_slam_clearance(goal_xy, clearance_m=0.20):
            return False
        return True

    def _shift_frontier_goal_to_free_space(
        self,
        frontier_xy: np.ndarray,
        robot_xy: np.ndarray,
        *,
        max_backoff_m: float = 0.45,
        step_m: float = 0.05,
        clearance_m: float = 0.20,
    ) -> np.ndarray | None:
        """Move a frontier goal back toward the robot until it lies in clear free space."""
        if self._slam_arr is None:
            return np.asarray(frontier_xy, dtype=np.float32)

        frontier = np.asarray(frontier_xy, dtype=np.float32)
        robot = np.asarray(robot_xy, dtype=np.float32)
        delta = robot - frontier
        dist = float(np.linalg.norm(delta))
        if dist < 1e-6:
            cand = frontier
            return cand if self._goal_is_still_valid(cand) else None

        direction = delta / dist
        samples = max(1, int(math.ceil(max_backoff_m / max(step_m, 1e-6))))
        for i in range(samples + 1):
            backoff = min(max_backoff_m, i * step_m)
            cand = frontier + direction * backoff
            if self._goal_is_still_valid(cand) and np.linalg.norm(cand - robot) > 0.15:
                return cand.astype(np.float32)

        for i in range(samples + 1):
            backoff = min(max_backoff_m, i * step_m)
            cand = frontier + direction * backoff
            if not self._point_in_collision_slam(cand, pad_cells=1):
                return cand.astype(np.float32)
        return None

    # ══════════════════════════════════════════════════════════════════════════
    # Visualization
    # ══════════════════════════════════════════════════════════════════════════

    def _vis_tick(self):
        # ── PNG-based composite visualization (non-blocking daemon thread) ──
        if (self.region is not None
                and self.robot_pos is not None
                and self._slam_arr is not None):
            if self._vis_running:
                self._publish_frontier_markers()
                self._publish_room_markers()
                self._publish_segment_markers()
                self._publish_doorway_markers()
                return
            region = self.region
            robot_pos = self.robot_pos.copy()
            goal_xy = self.goal_xy.copy() if self.goal_xy is not None else None
            path_snap = (self._path.copy() if self._path is not None
                         and len(self._path) >= 2 else None)
            traj_snap = list(self._robot_traj)
            rooms_snap = list(self._rooms)
            doorways_snap = list(self._doorways)
            confirmed_snap = set(self._confirmed_room_ids)
            persist_snap = dict(self._room_persist)
            room_regions_snap = dict(self.room_regions)
            active_room_snap = self.active_room_id
            slam_arr = self._slam_arr.copy()
            slam_cs = self._slam_cell_size
            slam_ox, slam_oy = float(self._slam_origin[0]), float(self._slam_origin[1])
            slam_H, slam_W = self._slam_H, self._slam_W
            all_segs_snap = list(self._all_segments)
            step = self._exploration_tick_count
            mode = self.mode
            all_regions_snap = dict(self.all_regions)
            _rm = self.segmenter.get_room_map()
            room_map_snap = _rm.copy() if _rm is not None else None
            self._vis_running = True

            def _plot():
                try:
                    self._save_vis_plot(
                        region, robot_pos, goal_xy, path_snap, traj_snap,
                        rooms_snap, doorways_snap, confirmed_snap, persist_snap,
                        room_regions_snap, active_room_snap,
                        slam_arr, slam_cs, slam_ox, slam_oy, slam_H, slam_W,
                        all_segs_snap, step, mode, all_regions_snap,
                        room_map=room_map_snap)
                except Exception as exc:
                    import traceback
                    self.get_logger().error(f"[VIS] {exc}\n{traceback.format_exc()}")
                finally:
                    self._vis_running = False

            threading.Thread(target=_plot, daemon=True).start()

        # ── RViz markers (main-thread safe) ─────────────────────────────────
        self._publish_frontier_markers()
        self._publish_room_markers()
        self._publish_segment_markers()
        self._publish_doorway_markers()

    def _save_vis_plot(self, region, robot_pos, goal_xy, path, traj,
                       rooms, doorways, confirmed_ids, persist, room_regions,
                       active_room, slam_arr, slam_cs, slam_ox, slam_oy,
                       slam_H, slam_W, all_segs, step, mode, all_regions=None,
                       room_map=None):
        """Save a single PNG: occupancy map + watershed room overlay (no GPU inference)."""

        robot_xy = robot_pos[:2]

        # Stable canvas — initialized once, only expands
        x0, x1, y0, y1 = self._update_plots2_extent(
            robot_xy, goal_xy, path, traj, all_regions or {region.version: region},
            slam_arr, slam_cs, slam_ox, slam_oy, slam_H, slam_W,
        )

        fig, ax = plt.subplots(1, 1, figsize=(13, 13))
        canvas_blue = '#080a78'
        fig.patch.set_facecolor(canvas_blue)
        ax.set_facecolor(canvas_blue)

        # ── 1. Occupancy map ─────────────────────────────────────────────────
        if slam_H > 0 and slam_W > 0:
            bg = np.zeros((slam_H, slam_W, 3), dtype=np.uint8)
            bg[:, :] = [8, 10, 120]              # unknown/canvas: original-style blue
            known = (slam_arr == 0) | (slam_arr > 50)
            bg[known] = [255, 248, 0]            # known SLAM cells: bright yellow
            ax.imshow(
                bg,
                origin='lower',
                extent=[slam_ox, slam_ox + slam_W * slam_cs,
                        slam_oy, slam_oy + slam_H * slam_cs],
                interpolation='nearest',
                zorder=0,
            )

        # ── 2. Room tint/boundaries are intentionally disabled in this PNG.
        # They are still published to RViz, but the saved plot should resemble
        # original plots2: occupancy + learned field + contours.
        room_map_c = None

        # ── 3. Room boundary contours via cv2 ────────────────────────────────
        if room_map_c is not None:
            unique_lbls = np.unique(room_map_c)
            unique_lbls = unique_lbls[unique_lbls > 0]
            cmap20 = plt.get_cmap('tab20')
            for i, lbl in enumerate(unique_lbls):
                mask_u8 = (room_map_c == lbl).astype(np.uint8)
                cnts, _ = cv2.findContours(
                    mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                c = cmap20(i % 20)
                for cnt in cnts:
                    pts = cnt[:, 0, :]           # [N, 2]: col, row
                    wx = slam_ox + pts[:, 0] * slam_cs
                    wy = slam_oy + pts[:, 1] * slam_cs
                    # close the contour
                    ax.plot(np.append(wx, wx[0]), np.append(wy, wy[0]),
                            color=c, linewidth=1.2, alpha=0.70, zorder=2)

        # ── 4. NTField travel-time contours (skipped when model not ready) ─────
        try:
            fn = region.function
            if fn is not None:
                b = region.bound.detach().cpu().numpy()
                rx0 = max(x0, float(b[0, 0]))
                rx1 = min(x1, float(b[1, 0]))
                ry0 = max(y0, float(b[0, 1]))
                ry1 = min(y1, float(b[1, 1]))
                if rx1 <= rx0 or ry1 <= ry0:
                    raise RuntimeError("active region outside plot extent")

                # Sample the field on the active region, not the full canvas.
                # Otherwise a large SLAM canvas makes the contours too sparse.
                region_span = max(rx1 - rx0, ry1 - ry0)
                grid_n = int(np.clip(region_span / 0.045, 120, 220))
                xs_g = np.linspace(rx0, rx1, grid_n, dtype=np.float32)
                ys_g = np.linspace(ry0, ry1, grid_n, dtype=np.float32)
                X_w, Y_w = np.meshgrid(xs_g, ys_g)

                src_norm = region.normalize(
                    np.array([[robot_xy[0], robot_xy[1], 0.0]], dtype=np.float32))[0]
                query_world = np.stack(
                    [X_w.flatten(), Y_w.flatten(),
                     np.zeros(X_w.size, dtype=np.float32)], axis=1,
                )
                query_norm = region.normalize(query_world)

                dim = region.model.dim
                N = X_w.size
                XP = np.zeros((N, 2 * dim), dtype=np.float32)
                XP[:, :dim] = src_norm
                XP[:, dim:] = src_norm
                XP[:, dim:dim + min(dim, query_norm.shape[1])] = query_norm[:, :dim]

                XP_t = torch.tensor(XP, dtype=torch.float32, device=region.device)
                with torch.no_grad():
                    TT = fn.TravelTimes(XP_t).cpu().numpy().reshape(X_w.shape)
                    # Do not draw the speed heatmap in this compact plots2-style
                    # view. Keep only occupancy + travel-time contours.

                in_region = (
                    (X_w >= float(b[0, 0])) & (X_w <= float(b[1, 0])) &
                    (Y_w >= float(b[0, 1])) & (Y_w <= float(b[1, 1]))
                )
                known_field = in_region
                if slam_H > 0 and slam_W > 0:
                    cols = np.floor((X_w - slam_ox) / slam_cs).astype(np.int32)
                    rows = np.floor((Y_w - slam_oy) / slam_cs).astype(np.int32)
                    inside = (
                        (rows >= 0) & (rows < slam_H) &
                        (cols >= 0) & (cols < slam_W)
                    )
                    known_field = np.zeros_like(in_region, dtype=bool)
                    known_field[inside] = (
                        (slam_arr[rows[inside], cols[inside]] == 0) |
                        (slam_arr[rows[inside], cols[inside]] > 50)
                    )
                    known_field &= in_region

                TT = np.where(np.isfinite(TT) & known_field, TT, np.nan)

                with self._vis_contour_lock:
                    self._vis_region_contour_cache[int(region.version)] = (
                        X_w.copy(), Y_w.copy(), TT.copy())
        except Exception:
            pass

        # Draw all known region contours, including contours from regions that
        # are no longer active. This keeps each region visible as last seen.
        try:
            with self._vis_contour_lock:
                contour_items = list(self._vis_region_contour_cache.items())
            active_version = int(region.version)
            for version, (X_c, Y_c, TT_c) in contour_items:
                is_active = int(version) == active_version
                ax.contour(
                    X_c, Y_c, TT_c, levels=np.arange(0, 10, 0.05),
                    colors='#00cccc',
                    linewidths=0.85 if is_active else 0.65,
                    alpha=0.96 if is_active else 0.68,
                    zorder=3 if is_active else 2.9,
                )
                ax.contour(
                    X_c, Y_c, TT_c, levels=np.arange(0, 10, 0.25),
                    colors='#00b3b3',
                    linewidths=0.38 if is_active else 0.30,
                    alpha=0.55 if is_active else 0.38,
                    zorder=3.1 if is_active else 2.95,
                )
        except Exception:
            pass

        # ── 5. Doorways ───────────────────────────────────────────────────────
        if False:
            for door in doorways:
                dx, dy = float(door.position[0]), float(door.position[1])
                ax.scatter([dx], [dy], c='#00f5ff', s=28, marker='D',
                           zorder=4, alpha=0.95, linewidths=0)

        # ── 6. Room centroid labels ───────────────────────────────────────────
        if False:
            for room in rooms:
                if room.id not in confirmed_ids:
                    continue
                cx, cy = float(room.centroid_xy[0]), float(room.centroid_xy[1])
                has_reg = room.id in room_regions
                txt_col = 'white' if has_reg else '#ffdd55'
                prefix  = '◆' if has_reg else '○'
                ax.text(cx, cy, f'{prefix}R{room.id}',
                        color=txt_col, fontsize=7, ha='center', va='center',
                        fontweight='bold', zorder=5,
                        bbox=dict(boxstyle='round,pad=0.15',
                                  fc='#000000', alpha=0.25, ec='none'))

        # ── 7. Robot trajectory + planned path ───────────────────────────────
        if len(traj) > 1:
            t_arr = np.asarray(traj, dtype=np.float32)
            ax.plot(t_arr[:, 0], t_arr[:, 1],
                    color='#ff00ff', linewidth=1.5, alpha=0.80, zorder=6)
        if path is not None and len(path) >= 2:
            p_arr = np.asarray(path, dtype=np.float32)
            ax.plot(p_arr[:, 0], p_arr[:, 1],
                    color='#ff00ff', linewidth=2.4, alpha=0.98, zorder=7)

        # ── 8. Goal + robot marker ────────────────────────────────────────────
        if goal_xy is not None:
            ax.scatter([goal_xy[0]], [goal_xy[1]],
                       c='#40ff00', s=110, marker='*', zorder=8,
                       edgecolors='black', linewidths=0.5)
        ax.scatter([robot_xy[0]], [robot_xy[1]],
                   c='red', s=72, zorder=9, edgecolors='white', linewidths=0.6)

        # ── 8. Canvas framing ────────────────────────────────────────────────
        ax.set_aspect('equal')
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.axis('off')
        n_confirmed = sum(1 for r in rooms if r.id in confirmed_ids)
        ax.set_title(
            f'step {step}  mode={mode}  rooms={n_confirmed}  ep={region.epoch}',
            fontsize=9, color='white', pad=4)
        fig.tight_layout(pad=0.3)

        vis_dir = os.path.join(MODEL_FOLDER, 'vis')
        os.makedirs(vis_dir, exist_ok=True)
        fig.savefig(os.path.join(vis_dir, f'plot_{step:06d}.png'), dpi=95,
                    facecolor=fig.get_facecolor())
        fig.savefig(os.path.join(MODEL_FOLDER, 'plot.png'), dpi=95,
                    facecolor=fig.get_facecolor())
        plt.close(fig)

    def _update_plots2_extent(self, robot_xy, goal_xy, path, traj, all_regions,
                              slam_arr, slam_cs, slam_ox, slam_oy, slam_H, slam_W):
        """Canvas anchored to the SLAM map; only ever expands outward, never re-centers.

        The canvas is driven by the known SLAM map extent (all mapped cells), not
        the robot position, so the view stays fixed as the robot moves.  Robot
        position is only used as a fallback before the first SLAM frame arrives.
        """
        span = max(float(self._plot_world_span_m), 4.0)
        pad  = max(0.8, 0.04 * span)

        # ── Gather world bounds from SLAM map (primary source) ───────────────
        xs, ys = [], []
        if slam_arr is not None and slam_H > 0 and slam_W > 0:
            known = (slam_arr == 0) | (slam_arr > 50)
            if known.any():
                rows, cols = np.where(known)
                xs.extend([
                    float(slam_ox + cols.min() * slam_cs) - pad,
                    float(slam_ox + (cols.max() + 1) * slam_cs) + pad,
                ])
                ys.extend([
                    float(slam_oy + rows.min() * slam_cs) - pad,
                    float(slam_oy + (rows.max() + 1) * slam_cs) + pad,
                ])

        # ── Fallback: no SLAM yet — centre on robot with minimum span ────────
        if not xs:
            cx, cy = float(robot_xy[0]), float(robot_xy[1])
            half = 0.5 * span
            xs = [cx - half, cx + half]
            ys = [cy - half, cy + half]

        new_x0, new_x1 = min(xs), max(xs)
        new_y0, new_y1 = min(ys), max(ys)

        # Enforce minimum canvas size (span × span), centred on current data
        cx = 0.5 * (new_x0 + new_x1)
        cy = 0.5 * (new_y0 + new_y1)
        half_x = max(0.5 * (new_x1 - new_x0), 0.5 * span)
        half_y = max(0.5 * (new_y1 - new_y0), 0.5 * span)
        new_x0, new_x1 = cx - half_x, cx + half_x
        new_y0, new_y1 = cy - half_y, cy + half_y

        # ── Merge with existing extent: only ever expand, never shrink ───────
        if self._plots2_extent is None:
            self._plots2_extent = (new_x0, new_x1, new_y0, new_y1)
        else:
            cur_x0, cur_x1, cur_y0, cur_y1 = self._plots2_extent
            self._plots2_extent = (
                min(cur_x0, new_x0),
                max(cur_x1, new_x1),
                min(cur_y0, new_y0),
                max(cur_y1, new_y1),
            )

        return self._plots2_extent

    def _publish_frontier_markers(self):
        """Publish all raw frontier cells as a single SPHERE_LIST Marker + OccupancyGrid."""
        if self._slam_arr is None:
            return

        arr = self._slam_arr
        cs = self._slam_cell_size
        ox, oy = float(self._slam_origin[0]), float(self._slam_origin[1])
        H, W = self._slam_H, self._slam_W

        # Frontier: unknown cells adjacent to free cells
        free = (arr == 0)
        unknown = (arr == -1)
        kernel = np.ones((3, 3), dtype=np.uint8)
        free_dilated = cv2.dilate(free.astype(np.uint8), kernel, iterations=1)
        frontier_mask = (free_dilated > 0) & unknown

        now = self.get_clock().now().to_msg()
        frontier_cells = np.argwhere(frontier_mask)

        # 1. Publish as OccupancyGrid (for RViz Map display)
        flat = np.full(H * W, -1, dtype=np.int8)
        if len(frontier_cells) > 0:
            rows, cols = frontier_cells[:, 0], frontier_cells[:, 1]
            flat[rows * W + cols] = 100
        occ = OccupancyGrid()
        occ.header.frame_id = "map"
        occ.header.stamp = now
        occ.info.map_load_time = now
        occ.info.resolution = float(cs)
        occ.info.width = int(W)
        occ.info.height = int(H)
        occ.info.origin.position.x = float(ox)
        occ.info.origin.position.y = float(oy)
        occ.info.origin.orientation.w = 1.0
        occ.data = flat.tolist()
        self.frontier_occ_pub.publish(occ)

        # 2. Publish all raw frontier points as a single SPHERE_LIST Marker
        m = Marker()
        m.header = Header(stamp=now, frame_id="map")
        m.ns = "frontiers"
        m.id = 1
        m.type = Marker.SPHERE_LIST
        m.action = Marker.ADD
        m.scale.x = m.scale.y = m.scale.z = 0.07
        m.color = ColorRGBA(r=0.4, g=1.0, b=0.4, a=0.90)

        if len(frontier_cells) > 0:
            fxy_x = ox + frontier_cells[:, 1] * cs + cs * 0.5
            fxy_y = oy + frontier_cells[:, 0] * cs + cs * 0.5
            for x, y in zip(fxy_x, fxy_y):
                p = Point()
                p.x = float(x)
                p.y = float(y)
                p.z = 0.08
                m.points.append(p)

        self.frontier_marker_pub.publish(m)

    def _publish_room_markers(self):
        ma = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        active_room = self.active_room_id

        for i, room in enumerate(self._rooms):
            confirmed = room.id in self._confirmed_room_ids
            is_active = (room.id == active_room)
            has_region = room.id in self.room_regions

            # Fill: confirmed=solid colour, unconfirmed=faint grey
            mf = Marker()
            mf.header = Header(stamp=stamp, frame_id="map")
            mf.ns = "rooms"
            mf.id = room.id
            mf.type = Marker.CUBE
            mf.action = Marker.ADD
            bb = room.bounding_box
            mf.pose.position.x = float(0.5 * (bb[0, 0] + bb[1, 0]))
            mf.pose.position.y = float(0.5 * (bb[0, 1] + bb[1, 1]))
            mf.pose.position.z = 0.02
            mf.scale.x = float(bb[1, 0] - bb[0, 0])
            mf.scale.y = float(bb[1, 1] - bb[0, 1])
            mf.scale.z = 0.01
            hue = (room.id * 0.618) % 1.0  # stable hue tied to room ID
            sat = 0.7 if confirmed else 0.2
            val = 0.95 if confirmed else 0.5
            r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
            alpha = 0.55 if confirmed else 0.12
            mf.color = ColorRGBA(r=float(r), g=float(g), b=float(b), a=alpha)
            ma.markers.append(mf)

            # Outline cube for the robot's current room (bright white border)
            if is_active:
                mo = Marker()
                mo.header = mf.header
                mo.ns = "active_room_outline"
                mo.id = room.id
                mo.type = Marker.CUBE
                mo.action = Marker.ADD
                mo.pose = mf.pose
                mo.scale.x = mf.scale.x + 0.04
                mo.scale.y = mf.scale.y + 0.04
                mo.scale.z = 0.005
                mo.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=0.9)  # yellow outline
                ma.markers.append(mo)

            # Label
            mc = Marker()
            mc.header = mf.header
            mc.ns = "room_labels"
            mc.id = room.id
            mc.type = Marker.TEXT_VIEW_FACING
            mc.action = Marker.ADD
            mc.pose.position.x = float(room.centroid[0])
            mc.pose.position.y = float(room.centroid[1])
            mc.pose.position.z = 0.35
            mc.scale.z = 0.18
            mc.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            reg = self.room_regions.get(room.id)
            persist = self._room_persist.get(room.id, 0)
            status = "OK" if confirmed else f"{persist}/{ROOM_PERSIST_TICKS}"
            if reg is not None:
                mc.text = f"R{room.id} [{status}] reg={reg.version} ep={reg.epoch}"
            else:
                mc.text = f"R{room.id} [{status}]"
            ma.markers.append(mc)

        self.room_marker_pub.publish(ma)

    def _publish_segment_markers(self):
        ma = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        for seg in self._all_segments:
            m = Marker()
            m.header = Header(stamp=stamp, frame_id="map")
            m.ns = "line_segments"
            m.id = seg.id
            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD
            m.points = [
                Point(x=float(seg.p1[0]), y=float(seg.p1[1]), z=0.1),
                Point(x=float(seg.p2[0]), y=float(seg.p2[1]), z=0.1),
            ]
            m.scale.x = 0.03
            m.color = ColorRGBA(r=0.2, g=0.8, b=1.0, a=0.9)
            ma.markers.append(m)
        self.seg_marker_pub.publish(ma)

    def _publish_doorway_markers(self):
        ma = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        for door in self._doorways:
            m = Marker()
            m.header = Header(stamp=stamp, frame_id="map")
            m.ns = "doorways"
            m.id = door.id
            m.type = Marker.CYLINDER
            m.action = Marker.ADD
            m.pose.position.x = float(door.position[0])
            m.pose.position.y = float(door.position[1])
            m.pose.position.z = 0.15
            m.scale.x = m.scale.y = float(door.width)
            m.scale.z = 0.3
            m.color = ColorRGBA(r=1.0, g=0.3, b=0.3, a=0.5)
            ma.markers.append(m)
        self.door_marker_pub.publish(ma)

    def _publish_field_path(self, path_xy: np.ndarray):
        if path_xy is None:
            return
        pts = np.asarray(path_xy, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 2:
            return
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        for pt in pts:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = float(pt[0])
            ps.pose.position.y = float(pt[1])
            msg.poses.append(ps)
        self.field_path_pub.publish(msg)

    def _publish_nav_path(self, path_xy: np.ndarray):
        """Publish full (untruncated) path for visualization."""
        if path_xy is None:
            return
        pts = np.asarray(path_xy, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 2:
            return
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        for pt in pts:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = float(pt[0])
            ps.pose.position.y = float(pt[1])
            msg.poses.append(ps)
        self.nav_path_pub.publish(msg)

    def _publish_goal_marker(self, goal: np.ndarray):
        m = Marker()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = "map"
        m.ns = "goal"
        m.id = 0
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = float(goal[0])
        m.pose.position.y = float(goal[1])
        m.pose.position.z = 0.3
        m.scale.x = m.scale.y = m.scale.z = 0.2
        m.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)
        self.goal_marker_pub.publish(m)

    # ══════════════════════════════════════════════════════════════════════════
    # Save / shutdown
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _json_has_payload(path: str, keys: tuple[str, ...]) -> bool:
        try:
            if not os.path.exists(path):
                return False
            with open(path, "r") as f:
                raw = json.load(f)
            if not keys:
                return isinstance(raw, dict) and len(raw) > 0
            for key in keys:
                val = raw.get(key)
                if isinstance(val, dict) and len(val) > 0:
                    return True
                if isinstance(val, list) and len(val) > 0:
                    return True
            return False
        except Exception:
            return False

    def _save_topology_json(self) -> None:
        """Save topology.json for region_graph_inspector compatibility.

        Source of truth:
          - regions  → self.all_regions (Region bounds + version)
          - rooms    → accepted watershed rooms, falling back to latest visible rooms
          - doors    → latest watershed doorways
          - edges    → self.room_graph._graph (room connectivity)
        """
        try:
            # ── regions ──────────────────────────────────────────────────────
            if self.region is not None:
                self.all_regions.setdefault(self.region.version, self.region)
            regions_out = {}
            for ver, reg in self.all_regions.items():
                b = reg.bound.detach().cpu().numpy()
                center = [float(0.5 * (b[0, 0] + b[1, 0])),
                          float(0.5 * (b[0, 1] + b[1, 1]))]
                regions_out[str(ver)] = {
                    "version": int(ver),
                    "center": center,
                    "bound": b.tolist(),
                }

            # ── rooms from watershed segmenter ───────────────────────────────
            rooms_out = []
            ws_rooms = (self._accepted_rooms if self._accepted_rooms
                        else self.segmenter.rooms)
            ws_doors = ({d.id: d for d in self._doorways} if self._doorways
                        else self.segmenter.doorways)
            for room in ws_rooms.values():
                centroid = getattr(room, "centroid_xy", getattr(room, "centroid", None))
                area_m2 = getattr(room, "area_m2", 0.0)
                confirmed = getattr(room, "confirmed",
                                    room.id in self._confirmed_room_ids)
                door_ids = [int(d.id) for d in ws_doors.values()
                            if room.id in d.room_ids]
                rooms_out.append({
                    "id": int(room.id),
                    "centroid_xy": np.asarray(centroid, dtype=np.float32).tolist(),
                    "bounding_box": room.bounding_box.tolist(),
                    "confirmed": bool(confirmed),
                    "area_m2": float(area_m2),
                    "doors": door_ids,
                })

            # ── doorways from watershed segmenter ────────────────────────────
            doors_out = [
                {
                    "id": int(d.id),
                    "pos_xy": d.position_xy.tolist(),
                    "room_ids": [int(r) for r in d.room_ids],
                    "width": float(d.width_m),
                }
                for d in ws_doors.values()
            ]

            # ── room graph edges ─────────────────────────────────────────────
            edges_out = []
            for u, v, edata in self.room_graph._graph.edges(data=True):
                edges_out.append({
                    "rooms": [int(u), int(v)],
                    "doorway_id": int(edata.get("doorway_id", -1)),
                    "weight": float(edata.get("weight", 1.0)),
                    "position": edata.get("position", [0.0, 0.0]),
                })

            topo = {
                "regions": regions_out,
                "graph_edges": edges_out,
                "rooms": rooms_out,
                "doors": doors_out,
                "connection_points": [],
                "frontier_banks": dict(self._region_frontier_banks),
            }
            topo_path = os.path.join(MODEL_FOLDER, "topology.json")
            has_payload = bool(regions_out or edges_out or rooms_out or doors_out)
            if not has_payload:
                if self._json_has_payload(
                        topo_path, ("regions", "graph_edges", "rooms", "doors")):
                    self.get_logger().warn(
                        "[TOPOLOGY] skip empty save; keeping existing topology.json")
                else:
                    self.get_logger().warn(
                        "[TOPOLOGY] skip empty save; no regions/rooms available yet")
                return
            with open(topo_path, "w") as f:
                json.dump(topo, f, indent=2)
            self.get_logger().info(
                f"[TOPOLOGY] {len(regions_out)} regions, "
                f"{len(rooms_out)} rooms, {len(doors_out)} doors")
        except Exception as e:
            self.get_logger().error(f"[TOPOLOGY SAVE ERROR] {e}")

    def save_all(self):
        if self._current_goal_metric is not None:
            self._finalize_goal_metric(
                "shutdown_active_goal",
                self.robot_pos[:2] if self.robot_pos is not None else None,
            )
        try:
            self._refresh_region_frontier_banks()
        except Exception:
            pass
        # Save active region
        if self.region is not None:
            try:
                self.all_regions.setdefault(self.region.version, self.region)
                self.region.save_model()
            except Exception:
                pass

        # Save room graph
        graph_path = os.path.join(MODEL_FOLDER, "room_graph.json")
        graph_has_payload = bool(self.room_graph.rooms or self.room_graph.doorways or
                                 self.room_graph._graph.number_of_edges() > 0)
        if graph_has_payload:
            self.room_graph.save(graph_path)
            self.get_logger().info(
                f"[SAVE] room_graph.json rooms={len(self.room_graph.rooms)} "
                f"doors={len(self.room_graph.doorways)}")
        elif self._json_has_payload(graph_path, ("rooms", "doorways", "edges")):
            self.get_logger().warn(
                "[SAVE] skip empty room_graph.json; keeping existing file")
        else:
            self.get_logger().warn(
                "[SAVE] skip room_graph.json; no rooms available yet")

        # Save SLAM map (map_server format for region_graph_inspector)
        self._save_slam_map()

        # Save topology.json (for region_graph_inspector)
        self._save_topology_json()

        # Room→Region mapping
        mapping = {str(rid): reg.version
                   for rid, reg in self.room_regions.items()}
        map_path = os.path.join(MODEL_FOLDER, "room_region_map.json")
        if mapping:
            with open(map_path, "w") as f:
                json.dump(mapping, f, indent=2)
            self.get_logger().info(
                f"[SAVE] room_region_map.json entries={len(mapping)}")
        elif self._json_has_payload(map_path, tuple()):
            self.get_logger().warn(
                "[SAVE] skip empty room_region_map.json; keeping existing file")
        else:
            self.get_logger().warn(
                "[SAVE] skip room_region_map.json; no room-region bindings yet")

        self._save_metrics_json()

        # self.get_logger().info(
        #     f"[SAVE] {len(self.all_regions)} regions + room graph")

    def destroy_node(self):
        self.save_all()
        super().destroy_node()


# ── Entry point ──────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = MappingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save_all()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
