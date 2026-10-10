"""
nav_node.py — Room-graph navigation with per-room NTFields.

Loads saved regions (one per room) + room graph from mapping phase.
On receiving a 2D Goal Pose from RViz:
  1. Identify current room and goal room via room_graph.
  2. Compute shortest path through room graph (room → door → room → ...).
  3. For each room segment, generate field path via region.generate_field_path().
  4. Concatenate, smooth, and follow.

Also monitors semantic changes and retrains affected room fields.

State machine:  WAITING_FOR_POSE → IDLE → PLANNING → READY → DRIVING → IDLE
"""

import os
import json
import glob
import math
import copy
import shutil
import threading
import time as _time

import numpy as np
import torch
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
import rclpy.time
import rclpy.parameter

from geometry_msgs.msg import Twist, PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid, Path
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String, ColorRGBA, Header
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import Buffer, TransformListener
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

import torch

from mntfields.mntfields_core.core.region import Region, is_semantic_entry_confirmed
from mntfields.mntfields_core.mapping.room_graph import RoomGraph
from mntfields.mntfields_ros.mapping.navigation import (
    Navigator, resample_polyline,
)
from mntfields.mntfields_ros.sensing.lidar_adapter import Lidar2DAdapter
from mntfields.utils.mntfields_utils import (
    compute_ownership_map,
    compute_crossing_points,
    owner_region_for_xy,
)

os.environ.setdefault("MPLBACKEND", "Agg")

# ── Constants ────────────────────────────────────────────────────────────────

MODEL_FOLDER = os.path.expanduser("~/turtlebot3_ws/mntfields_model")
GOAL_REACHED_DIST = 0.25
SEMANTIC_CHECK_INTERVAL_SEC = 10.0
RETRAIN_EPOCHS = 150          # epochs to run on merged data after a change
PRUNE_RADIUS_M = 1.0          # spatial radius for stale-data pruning
LIDAR_ACCUM_EVERY_N = 5       # accumulate 1-in-N scans during navigation
OBSTACLE_PAINT_RADIUS_M = 0.175  # footprint radius painted onto map for moved object

# Minimum rows remaining after prune+synth to justify running a retrain.
# Below this the retrain is skipped (running on <500 noisy rows at LR=1e-4
# for 100 epochs can degrade a well-trained field).
RETRAIN_MIN_ROWS = 500

# Synthetic Nx12 rows generated from the painted map around each new obstacle.
# Teaches the NTField "wall at new_xy" directly, independent of whether the
# robot's LiDAR has swept the area after the obstacle moved.
SYNTH_ROWS_PER_EVENT = 300
SYNTH_DISK_RADIUS_M = 1.5

# Only these classes trigger retraining when they move or disappear.
# - other_furniture (lamps, curtains, mirrors, paintings) excluded — wall-mounted
#   or non-blocking, causing spurious retrains.
# - door excluded — doors are structural fixtures; apparent "moves" are usually
#   newly discovered doors elsewhere in the room, not a physical relocation of
#   the stored one, and the NTField already treats walls around them correctly.
RETRAIN_CLASSES = {"table", "chair", "bed", "sofa", "cabinet"}

# Camera FOV guard — only fire "removed" events when the stored object is
# actually within the camera's viewing frustum. Prevents spurious retrains
# when the robot drives past an object and it falls out of frame.
CAM_HFOV_DEG  = 69.0   # RealSense D435/D455 horizontal FOV
CAM_MAX_RANGE_M = 4.0  # beyond this, detection is unreliable — skip removal check

# Hold-off window: after the first change event queues, wait this long before
# flushing the retrain pass. Lets the camera sweep over additional moved
# objects (and the 360° LiDAR to accumulate new geometry) so one retrain
# covers everything instead of firing N sequential retrains.
RETRAIN_HOLDOFF_SEC = 7.0
RETRAIN_MIN_LOCAL_ROWS = 600
RETRAIN_LOCAL_RADIUS_M = 1.0

# Grid size for dedup: two events at old_xy within this distance are treated
# as the same object. Small enough to allow two chairs in the same region.
RETRAIN_DEDUP_GRID_M = 0.5

# Position-persistence buffer for change detection. Any incoming detection
# (regardless of class) bumps a cell at its grid-rounded xy. A cell is only
# acted on once it has accumulated STABLE_MIN_COUNT detections, and cells
# that go unseen for STABLE_TIMEOUT_S seconds are pruned. This decouples the
# retrain decision from class labels entirely — class is kept as metadata
# only (logging, painting, RViz) but is not used for matching.
POSITION_GRID_M = 0.3
STABLE_MIN_COUNT = 3
STABLE_TIMEOUT_S = 5.0
# A stable cell within REASSOCIATE_RADIUS of a stored landmark is the same
# physical object (class flips ignored). Between REASSOCIATE and MOVED_MAX
# it is treated as the same object having moved. Beyond MOVED_MAX it is
# considered unrelated and the stored landmark is a candidate for removal.
REASSOCIATE_RADIUS_M = 0.8
MOVED_MAX_RADIUS_M = 2.0
# Consecutive _check_semantic_changes cycles with no stable cell near a
# stored landmark (while the landmark is inside FOV) before a "removed"
# event fires. Prevents cold-start spam when the persistence buffer has
# not yet produced any stable cells, and tolerates brief occlusions.
MISS_THRESHOLD = 3


class NavNode(Node):

    def __init__(self):
        super().__init__("mntfields_nav")

        # ── Parameters ───────────────────────────────────────────────────────
        self.declare_parameter("model_dir", MODEL_FOLDER)
        self.declare_parameter("max_lin", 0.15)
        self.declare_parameter("max_ang", 0.80)
        self.declare_parameter("goal_reached_dist", GOAL_REACHED_DIST)
        self.declare_parameter("use_keyboard_confirm", True)
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("planner_mode", "region_graph")
        self.declare_parameter("field_step_size", 0.10)
        self.declare_parameter("field_max_steps", 220)
        self.declare_parameter("region_bound_margin_m", 0.05)
        self.declare_parameter("doorway_cushion_offsets_m", [0.0, 0.18, 0.30, 0.45, 0.60])
        self.declare_parameter("semantic_topic",
                               "/semantic_segmentation/objects")
        self.declare_parameter("metrics_path", "")

        self.model_dir = str(self.get_parameter("model_dir").value)
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.planner_mode = str(self.get_parameter("planner_mode").value)
        self.field_step_size = float(self.get_parameter("field_step_size").value)
        self.field_max_steps = int(self.get_parameter("field_max_steps").value)
        self.region_bound_margin_m = float(self.get_parameter("region_bound_margin_m").value)
        self.doorway_cushion_offsets_m = [
            float(v) for v in self.get_parameter("doorway_cushion_offsets_m").value
        ]
        _metrics_path = str(self.get_parameter("metrics_path").value).strip()
        self._metrics_path = (_metrics_path if _metrics_path
                              else os.path.join(self.model_dir, "line_navigation_metrics.json"))

        # ── State ────────────────────────────────────────────────────────────
        self.state = "WAITING_FOR_POSE"
        self.robot_pos: np.ndarray | None = None
        self.robot_yaw: float = 0.0

        self._goal_xy: np.ndarray | None = None
        self._nav_path: np.ndarray | None = None
        self._path_idx: int = 0
        self._planning_thread: threading.Thread | None = None
        self._metrics_run_start_wall_time = _time.time()
        self._metrics_run_start_mono = _time.monotonic()
        self._goal_request_seq = 0
        self._active_goal_request_id: int | None = None
        self._goal_request_metrics: list[dict] = []
        self._trajectory_plan_metrics: list[dict] = []

        # ── SDF local path (from sdf_local_replanner_node) ───────────────────
        self._sdf_path: np.ndarray | None = None
        self._sdf_path_time: float = 0.0
        _SDF_VALID_SEC = 2.0  # fall back to global path after this many seconds

        # ── Regions + room graph ─────────────────────────────────────────────
        self.all_regions: dict[int, Region] = {}      # version -> Region
        self.room_regions: dict[int, Region] = {}     # room_id -> Region
        self.room_graph = RoomGraph()
        self._topology_edges: list[dict] = []
        self._topology_doors: list[dict] = []
        self._strict_roomwise_topology: bool = False

        # ── Semantic change tracking ─────────────────────────────────────────
        self._rooms_needing_retrain: set[int] = set()
        self._retrain_lock = threading.Lock()
        # Each event: {region_ver, old_xy, new_xy, class_name}
        self._pending_retrain_events: list[dict] = []
        self._retraining: bool = False
        self._retrain_thread: threading.Thread | None = None
        # Saved state to resume after retrain completes. None when not retraining.
        self._state_before_retrain: str | None = None
        # Hold-off window: monotonic time before which pending events should
        # not be flushed. Set when the first event is queued into an empty
        # list; lets the camera/LiDAR sweep gather the rest of the scene.
        self._retrain_hold_until: float = 0.0

        # Position-persistence buffer keyed by (gx, gy) grid cell.
        # Value: {xy, count, last_seen, last_class}.
        self._position_buffer: dict[tuple, dict] = {}
        # Per-landmark miss counters keyed by (region_ver, gx, gy) of the
        # stored landmark's current xy. Incremented when no stable cell is
        # near the landmark while it is in FOV; reset when a cell is found.
        self._miss_counters: dict[tuple, int] = {}
        self._candidate_root = os.path.join(self.model_dir, "_candidates")
        self._candidate_backup_root = os.path.join(self.model_dir, "_backups")

        # ── LiDAR accumulation during navigation ─────────────────────────────
        self._lidar_adapter = Lidar2DAdapter(max_range=3.5, lidar_height=0.17)
        self._scan_count: int = 0

        # ── SLAM map (for collision checking) ────────────────────────────────
        self._map_occ: np.ndarray | None = None
        self._map_free: np.ndarray | None = None
        self._map_resolution: float | None = None
        self._map_origin: list[float] | None = None

        # ── TF ───────────────────────────────────────────────────────────────
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ── Load saved data ──────────────────────────────────────────────────
        self._load_room_graph()
        self._load_topology()
        self._load_all_regions()
        self._build_room_region_map()
        self._build_room_to_region_map()

        # ── Subscribers ──────────────────────────────────────────────────────
        self.create_subscription(
            PoseWithCovarianceStamped, "/amcl_pose", self._amcl_cb, 10)
        self.create_subscription(
            PoseStamped, "/goal_pose", self._goal_cb, 10)

        map_qos = QoSProfile(
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE, depth=1)
        self.create_subscription(OccupancyGrid, "/map", self._map_cb, map_qos)

        self.create_subscription(LaserScan, "/scan", self._scan_cb, 10)
        self.create_subscription(String, "/semantic/changes", self._semantic_cb, 10)
        self.create_subscription(
            String,
            str(self.get_parameter("semantic_topic").value),
            self._tracked_objects_cb, 10)
        self.create_subscription(Path, "/sdf_local_path", self._sdf_path_cb, 10)

        # ── Publishers ───────────────────────────────────────────────────────
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.nav_path_pub = self.create_publisher(Path, "/nav_path", 10)
        self.goal_marker_pub = self.create_publisher(Marker, "/mntfields/goal_marker", 10)
        self.room_marker_pub = self.create_publisher(
            MarkerArray, "/mntfields/rooms", 10)

        # ── Navigator (pure-pursuit) ─────────────────────────────────────────
        self.dt = 0.10
        self.nav_frame = "map"
        self.navigator = Navigator(self)

        # ── Timers ───────────────────────────────────────────────────────────
        self.create_timer(0.10, self._update_tick)
        self.create_timer(SEMANTIC_CHECK_INTERVAL_SEC, self._semantic_retrain_tick)

        # ── Keyboard confirm thread ──────────────────────────────────────────
        if bool(self.get_parameter("use_keyboard_confirm").value):
            self._keyboard_event = threading.Event()
            t = threading.Thread(target=self._keyboard_thread, daemon=True)
            t.start()
        else:
            self._keyboard_event = None

        self.get_logger().info(
            f"[NAV] Started. {len(self.all_regions)} region(s), "
            f"{self.room_graph.num_rooms} room(s), "
            f"{self.room_graph.num_doorways} doorway(s).")

    # ══════════════════════════════════════════════════════════════════════════
    # Loading
    # ══════════════════════════════════════════════════════════════════════════

    def _load_room_graph(self):
        path = os.path.join(self.model_dir, "room_graph.json")
        if os.path.isfile(path):
            self.room_graph.load(path)
            self.get_logger().info(
                f"[NAV] Loaded room graph: {self.room_graph.num_rooms} rooms, "
                f"{self.room_graph.num_doorways} doorways")
        else:
            self.get_logger().warn(f"[NAV] room_graph.json not found at {path}")

    def _load_topology(self):
        path = os.path.join(self.model_dir, "topology.json")
        if not os.path.isfile(path):
            self.get_logger().warn(f"[NAV] topology.json not found at {path}")
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                topo = json.load(f)
            topo_graph = topo.get("topology_graph") or {}
            self._topology_edges = topo.get("graph_edges", []) or topo_graph.get("edges", [])
            self._topology_doors = topo.get("doors", []) or ((topo.get("room_topology") or {}).get("doors")) or []
            self.get_logger().info(
                f"[NAV] Loaded topology: edges={len(self._topology_edges)} "
                f"doors={len(self._topology_doors)}")
        except Exception as exc:
            self.get_logger().warn(f"[NAV] Failed loading topology.json: {exc}")

    def _load_all_regions(self):
        """Discover and load all region_N/ folders."""
        if not os.path.isdir(self.model_dir):
            self.get_logger().error(f"[NAV] model_dir not found: {self.model_dir}")
            return

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.get_logger().info(
            f"[NAV] Loading regions from {self.model_dir} on {device}")

        pattern = os.path.join(self.model_dir, "region_*")
        for region_dir in sorted(glob.glob(pattern)):
            if not os.path.isdir(region_dir):
                continue

            folder_name = os.path.basename(region_dir)
            try:
                ver = int(folder_name.split("_")[1])
            except (IndexError, ValueError):
                continue

            meta_path = os.path.join(region_dir, f"region_{ver}_meta.json")
            if not os.path.isfile(meta_path):
                self.get_logger().warn(
                    f"[NAV] Skipping region {ver}: missing {meta_path}")
                continue

            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                bound = torch.tensor(meta["bound"], dtype=torch.float32)

                region = Region(
                    model_folder=self.model_dir,
                    bound=bound,
                    version=ver,
                    device=device,
                )
                if not region.load_model():
                    self.get_logger().warn(
                        f"[NAV] Skipping region {ver}: incomplete saved bundle")
                    continue
                self.all_regions[ver] = region
                self.get_logger().info(
                    f"[NAV] Loaded region v{ver} epoch={region.epoch}")
            except Exception as e:
                self.get_logger().error(
                    f"[NAV] Failed loading region {ver} from {region_dir}: {e}")

    def _build_room_region_map(self):
        """Map room IDs to regions using the saved room_region_map.json."""
        map_path = os.path.join(self.model_dir, "room_region_map.json")
        if not os.path.isfile(map_path):
            self.get_logger().warn("[NAV] room_region_map.json not found")
            return
        with open(map_path) as f:
            mapping = json.load(f)
        for room_id_str, region_ver in mapping.items():
            room_id = int(room_id_str)
            region_ver = int(region_ver)
            if region_ver in self.all_regions:
                self.room_regions[room_id] = self.all_regions[region_ver]
        self.get_logger().info(
            f"[NAV] Room-region map: {len(self.room_regions)} entries")

    def _build_room_to_region_map(self):
        """Build room_id → region_version lookup from room_region_map.json.

        Offline-refined: identity (room_id == region_version).
        Online: multiple rooms may share one region.
        Falls back to identity if file is missing.
        """
        self._room_to_region: dict[int, int] = {}
        map_path = os.path.join(self.model_dir, "room_region_map.json")
        if os.path.isfile(map_path):
            try:
                with open(map_path) as f:
                    raw = json.load(f)
                self._room_to_region = {int(k): int(v) for k, v in raw.items()}
            except Exception as exc:
                self.get_logger().warn(f"[NAV] room_region_map parse error: {exc}")
        if not self._room_to_region:
            # Fallback: identity mapping from loaded region versions
            for ver in self.all_regions:
                self._room_to_region[ver] = ver
        self._strict_roomwise_topology = (
            bool(self._room_to_region)
            and all(int(k) == int(v) for k, v in self._room_to_region.items())
        )
        self.get_logger().info(
            f"[NAV] room_to_region={self._room_to_region} "
            f"strict_roomwise={self._strict_roomwise_topology}"
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Subscribers
    # ══════════════════════════════════════════════════════════════════════════

    def _amcl_cb(self, msg: PoseWithCovarianceStamped):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.robot_pos = np.array([p.x, p.y, p.z], dtype=np.float32)
        r = Rotation.from_quat([q.x, q.y, q.z, q.w])
        self.robot_yaw = float(r.as_euler("xyz")[2])
        if self.state == "WAITING_FOR_POSE":
            self.state = "IDLE"
            self.get_logger().info("[NAV] Got pose → IDLE. Click 2D Goal Pose.")

    def _goal_cb(self, msg: PoseStamped):
        if self.robot_pos is None:
            self.get_logger().warn("[NAV] No robot pose yet.")
            return
        gx = msg.pose.position.x
        gy = msg.pose.position.y
        self._goal_request_seq += 1
        self._active_goal_request_id = self._goal_request_seq
        self._goal_request_metrics.append({
            "goal_request_id": int(self._active_goal_request_id),
            "goal_xy": [float(gx), float(gy)],
            "start_xy": [float(self.robot_pos[0]), float(self.robot_pos[1])],
            "start_monotonic_s": float(_time.monotonic() - self._metrics_run_start_mono),
        })
        self._goal_xy = np.array([gx, gy], dtype=np.float32)
        self._sdf_path = None
        self._sdf_path_time = 0.0
        # Don't clobber RETRAINING — the retrain worker's finally block will
        # pick up the new _goal_xy and transition to PLANNING once done.
        if self.state == "RETRAINING":
            self.get_logger().info(
                f"[NAV] Goal received during retrain: ({gx:.2f}, {gy:.2f}) — "
                "will plan once retrain completes.")
            return
        self.state = "PLANNING"
        self.get_logger().info(f"[NAV] Goal received: ({gx:.2f}, {gy:.2f})")

    def _sdf_path_cb(self, msg: Path):
        """Receive locally-optimized path from sdf_local_replanner_node."""
        if not msg.poses:
            return
        frame_id = str(msg.header.frame_id).strip()
        if frame_id and frame_id != "map":
            self.get_logger().warn(
                f"[NAV] Ignoring SDF path in frame '{frame_id}' (expected map).",
                throttle_duration_sec=2.0)
            return

        path = np.array(
            [[p.pose.position.x, p.pose.position.y] for p in msg.poses],
            dtype=np.float32)
        if len(path) < 2:
            return
        if self.robot_pos is not None:
            start_dist = float(np.linalg.norm(path[0, :2] - self.robot_pos[:2]))
            if start_dist > 0.50:
                self.get_logger().warn(
                    f"[NAV] Ignoring stale SDF path start_dist={start_dist:.2f}m.",
                    throttle_duration_sec=1.0)
                return
        self._sdf_path = path
        self._sdf_path_time = _time.monotonic()

    def _map_cb(self, msg: OccupancyGrid):
        info = msg.info
        self._map_resolution = float(info.resolution)
        self._map_origin = [info.origin.position.x, info.origin.position.y]
        H, W = info.height, info.width
        arr = np.array(msg.data, dtype=np.int8).reshape(H, W)
        self._map_occ = (arr > 50)  # occupied cells
        self._map_free = (arr == 0)

    def _scan_cb(self, msg: LaserScan):
        """Update robot pose from TF and accumulate LiDAR into the current region."""
        self._update_pose_from_tf()
        if self.robot_pos is None or self.state not in ("DRIVING", "IDLE"):
            return

        self._scan_count += 1
        if self._scan_count % LIDAR_ACCUM_EVERY_N != 0:
            return

        region = self._region_for_robot()
        if region is None:
            return

        pts_sensor = self._lidar_adapter.scan_to_xy(msg)
        if pts_sensor is None:
            return
        world_pts = self._transform_scan_to_map(pts_sensor, msg)
        if world_pts is None or len(world_pts) < 20:
            return

        try:
            result = region.get_data_from_lidar(
                world_pts, self.robot_pos.copy(), device="cuda:0", n_samples=2000)
            if result is None:
                return
            points, closest = result
            raw = torch.cat([points, closest], dim=1).detach().cpu()
            raw = self._filter_region_owned_rows(region, raw)
            if len(raw) > 0:
                self._accumulate_region_owned_raw(region, raw)
        except Exception as exc:
            self.get_logger().debug(f"[NAV] LiDAR accumulate error: {exc}")

    def _filter_region_owned_rows(self, region, raw_rows: torch.Tensor) -> torch.Tensor:
        """Keep rows owned by this region via x0 only.

        Navigation retraining needs richer room-local support than the default
        strict region filter, which drops any row whose x1 crosses a doorway or
        neighboring room boundary. For nav updates we keep rows whose source
        sample x0 is inside the region bound.
        """
        if raw_rows is None or raw_rows.ndim != 2 or raw_rows.shape[1] != 12:
            return torch.zeros((0, 12), dtype=torch.float32)
        rows = raw_rows.detach().cpu().float()
        b = region.bound.detach().cpu().numpy()
        x0 = rows[:, 0:2].numpy()
        owned = (
            (x0[:, 0] >= float(b[0, 0])) & (x0[:, 0] <= float(b[1, 0])) &
            (x0[:, 1] >= float(b[0, 1])) & (x0[:, 1] <= float(b[1, 1]))
        )
        if not np.any(owned):
            return torch.zeros((0, 12), dtype=torch.float32)
        return rows[owned]

    def _owner_region_for_xy(self, xy: np.ndarray) -> "Region | None":
        """Return canonical owner region for world XY using the occupancy ownership map."""
        if self._map_occ is None or self._map_free is None or self._map_resolution is None:
            for reg in self.all_regions.values():
                if reg.in_bound(xy):
                    return reg
            return None
        origin = np.asarray(self._map_origin, dtype=np.float32)
        free = np.asarray(self._map_free, dtype=bool)
        ownership = compute_ownership_map(
            free=free,
            regions=self.all_regions,
            origin=origin,
            cell_size=float(self._map_resolution),
        )
        return owner_region_for_xy(
            xy,
            ownership,
            self.all_regions,
            origin,
            float(self._map_resolution),
        )

    def _accumulate_region_owned_raw(self, region, raw_rows: torch.Tensor) -> bool:
        """Accumulate navigation raw rows using x0 ownership instead of strict x0/x1 bounds."""
        if raw_rows is None or raw_rows.shape[0] == 0 or raw_rows.shape[1] != 12:
            return False
        rows = self._filter_region_owned_rows(region, raw_rows)
        if len(rows) == 0:
            return False
        if region.all_raw_frame_data is None:
            region.all_raw_frame_data = []
        region.all_raw_frame_data.append(rows)
        merged = torch.cat(region.all_raw_frame_data, dim=0)
        merged = self._filter_region_owned_rows(region, merged)
        max_raw = max(1, int(getattr(region, "max_raw_samples", 10_000)))
        if len(merged) > max_raw:
            merged = region._spatially_cap_raw_history(merged, max_raw)
        region.all_raw_frame_data = [merged]
        return True

    def _deduplicate_raw_rows(self, raw_rows: torch.Tensor) -> torch.Tensor:
        """Remove exact duplicate Nx12 rows while preserving dtype/device semantics."""
        if raw_rows is None or raw_rows.ndim != 2 or raw_rows.shape[1] != 12 or len(raw_rows) == 0:
            return torch.zeros((0, 12), dtype=torch.float32)
        rows = raw_rows.detach().cpu().float().contiguous()
        uniq = torch.unique(rows, dim=0)
        return uniq

    def _transform_scan_to_map(
        self, pts_sensor: np.ndarray, msg: LaserScan
    ) -> "np.ndarray | None":
        """Transform Nx3 sensor-frame points to map frame using TF."""
        frame_id = msg.header.frame_id or "rplidar_link"
        try:
            stamp = rclpy.time.Time.from_msg(msg.header.stamp)
            tf = self.tf_buffer.lookup_transform(
                "map", frame_id, stamp, timeout=Duration(seconds=0.05))
        except Exception:
            try:
                tf = self.tf_buffer.lookup_transform(
                    "map", frame_id, rclpy.time.Time(),
                    timeout=Duration(seconds=0.05))
            except Exception:
                return None
        t = tf.transform.translation
        q = tf.transform.rotation
        rot = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        trans = np.array([t.x, t.y, t.z], dtype=np.float64)
        pts = pts_sensor.astype(np.float64)
        return (rot @ pts.T).T + trans

    def _region_for_robot(self) -> "Region | None":
        """Return the region whose bounding box contains the current robot position."""
        if self.robot_pos is None:
            return None
        xy = self.robot_pos[:2].astype(np.float32)
        best: Region | None = None
        best_area = float("inf")
        for reg in self.all_regions.values():
            if reg.in_bound(xy):
                b = reg.bound.detach().cpu().numpy()
                area = float((b[1, 0] - b[0, 0]) * (b[1, 1] - b[0, 1]))
                if area < best_area:
                    best_area = area
                    best = reg
        return best

    def _semantic_cb(self, msg: String):
        """Handle semantic change events from SemanticRegionStore (legacy channel)."""
        # This channel carries pre-parsed events with region_version already set.
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        for ev in data.get("events", []):
            ev_type = ev.get("type", "")
            if ev_type not in ("object_moved", "object_missing"):
                continue
            rv = ev.get("region_version")
            if rv is None:
                continue
            old_xy = ev.get("stored_xy")
            new_xy = ev.get("observed_xy")
            if old_xy is None:
                continue
            self._queue_retrain_event(
                region_ver=int(rv),
                old_xy=np.asarray(old_xy[:2], dtype=np.float32),
                new_xy=(np.asarray(new_xy[:2], dtype=np.float32)
                        if new_xy is not None else None),
                class_name=ev.get("class_name", "?"),
            )

    def _tracked_objects_cb(self, msg: String):
        """Detect furniture changes by comparing live detections to stored semantic objects.

        Accepts the schema from semantic_segmentation_rgbd_node.py:
          { "objects": [{ "class_name": ..., "world_xyz_m": [x,y,z] }] }
        Normalises world_xyz_m → map_xy so _check_semantic_changes has a uniform interface.
        No "status" filter — all objects with valid world coordinates are checked.
        """
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        objects = data.get("objects", [])
        # Normalise: ensure each object has a "map_xy" key (2D world coords)
        confirmed = []
        for obj in objects:
            world_xyz = obj.get("world_xyz_m") or obj.get("map_xy")
            if world_xyz is None:
                continue
            normalised = dict(obj)
            normalised["map_xy"] = world_xyz[:2]
            confirmed.append(normalised)
        if not confirmed:
            return
        self.get_logger().info(
            f"[SEM] received {len(confirmed)} object(s): "
            + ", ".join(f"{o['class_name']}@({o['map_xy'][0]:.2f},{o['map_xy'][1]:.2f})"
                        for o in confirmed),
            throttle_duration_sec=5.0,
        )
        self._check_semantic_changes(confirmed)

    def _object_in_camera_fov(self, s_xy: np.ndarray) -> bool:
        """Return True if the stored world XY is inside the camera FOV from the current robot pose.

        Uses robot yaw + horizontal FOV cone + max range. Returns True when
        robot pose is unknown (fail-open so we don't suppress all checks).
        """
        if self.robot_pos is None:
            return True
        robot_xy = self.robot_pos[:2]
        vec = s_xy - robot_xy
        dist = float(np.linalg.norm(vec))
        if dist > CAM_MAX_RANGE_M:
            return False
        angle_to_obj = math.atan2(float(vec[1]), float(vec[0]))
        angle_diff = angle_to_obj - self.robot_yaw
        angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi
        return abs(angle_diff) <= math.radians(CAM_HFOV_DEG / 2.0)

    def _ingest_position_buffer(self, confirmed: list[dict]):
        """Update the position-persistence buffer with the latest detections.

        Each detection bumps a grid-rounded cell regardless of class — the
        class is stored as metadata only. Cells older than STABLE_TIMEOUT_S
        are pruned so stale observations decay out of the buffer.
        """
        now = _time.monotonic()
        for obj in confirmed:
            o_xy = np.asarray(obj["map_xy"][:2], dtype=np.float32)
            gx = int(round(float(o_xy[0]) / POSITION_GRID_M))
            gy = int(round(float(o_xy[1]) / POSITION_GRID_M))
            key = (gx, gy)
            entry = self._position_buffer.get(key)
            if entry is None:
                self._position_buffer[key] = {
                    "xy": o_xy.copy(),
                    "count": 1,
                    "last_seen": now,
                    "first_seen": now,
                    "last_class": obj["class_name"],
                }
            else:
                entry["count"] += 1
                entry["last_seen"] = now
                entry["last_class"] = obj["class_name"]
                alpha = 1.0 / float(entry["count"])
                entry["xy"] = (1.0 - alpha) * entry["xy"] + alpha * o_xy
        stale = [k for k, v in self._position_buffer.items()
                 if (now - v["last_seen"]) > STABLE_TIMEOUT_S]
        for k in stale:
            del self._position_buffer[k]

    def _stable_cells(self) -> list[dict]:
        """Return position-buffer cells that have passed the persistence gate."""
        return [v for v in self._position_buffer.values()
                if v["count"] >= STABLE_MIN_COUNT]

    def _check_semantic_changes(self, confirmed: list[dict]):
        """Position-based change detection. Class is metadata, not a matcher.

        - Any incoming detection accumulates in a grid-cell buffer. A cell
          becomes 'stable' once it has STABLE_MIN_COUNT detections.
        - Each stored confirmed landmark is compared to stable cells by
          position only:
            * within REASSOCIATE_RADIUS → same physical object, class flips
              are ignored and no retrain fires.
            * within MOVED_MAX_RADIUS (but beyond REASSOCIATE) → moved;
              queue retrain.
            * no stable cell within MOVED_MAX_RADIUS and landmark is inside
              the camera FOV → removed; queue retrain.
          Anything else is deferred (robot likely hasn't looked yet).
        """
        self._ingest_position_buffer(confirmed)
        stable = self._stable_cells()

        for reg in self.all_regions.values():
            stored = getattr(reg, "semantic_objects", {})
            if isinstance(stored, list):
                # Defensive fallback for corrupted runtime state; preserve node
                # liveness instead of crashing the semantic loop.
                stored = {}
                reg.semantic_objects = stored
            elif not isinstance(stored, dict):
                stored = {}
                reg.semantic_objects = stored
            if not stored:
                continue
            # Stable cells owned by this region. Do not use raw in_bound()
            # membership here because refined room regions overlap near
            # doorways and one cell can otherwise trigger duplicate retrains
            # in multiple regions.
            region_stable = []
            for c in stable:
                owner = self._owner_region_for_xy(np.asarray(c["xy"][:2], dtype=np.float32))
                if owner is not None and int(owner.version) == int(reg.version):
                    region_stable.append(c)

            self.get_logger().info(
                f"[SEM] R{reg.version} stored: "
                + ", ".join(f"{k}×{len(v)}" for k, v in stored.items())
                + f"  | stable cells in region: {len(region_stable)}",
                throttle_duration_sec=10.0,
            )

            # Flatten this region's confirmed stored landmarks for mutual-NN.
            all_stored: list[tuple] = []
            all_stored_any: list[tuple] = []
            for class_name, positions in stored.items():
                if class_name not in RETRAIN_CLASSES:
                    continue
                for stored_pos in positions:
                    s_xy = self._semantic_entry_xy(stored_pos)
                    if s_xy is None:
                        continue
                    owner = self._owner_region_for_xy(s_xy)
                    if owner is not None and int(owner.version) != int(reg.version):
                        continue
                    all_stored_any.append((class_name, positions, stored_pos, s_xy))
                    if not is_semantic_entry_confirmed(stored_pos):
                        continue
                    all_stored.append((class_name, positions, stored_pos, s_xy))

            for class_name, positions, stored_pos, s_xy in all_stored:
                best, best_dist = None, float("inf")
                for cell in region_stable:
                    d = float(np.linalg.norm(cell["xy"] - s_xy))
                    if d < best_dist:
                        best_dist = d
                        best = cell

                # Mutual-NN guard: beyond the reassociate radius, only claim a
                # stable cell if no other stored landmark in this region is
                # even closer to it. Prevents an unrelated landmark from
                # "moving" to a cell that clearly belongs to another.
                if best is not None and best_dist > REASSOCIATE_RADIUS_M:
                    for o_cls, _, o_sp, o_xy in all_stored:
                        if o_sp is stored_pos:
                            continue
                        if float(np.linalg.norm(best["xy"] - o_xy)) < best_dist:
                            self.get_logger().debug(
                                f"[SEM] '{class_name}' @ "
                                f"({s_xy[0]:.2f},{s_xy[1]:.2f}) nearest cell "
                                f"is closer to '{o_cls}' — not claiming",
                                throttle_duration_sec=10.0,
                            )
                            best = None
                            best_dist = float("inf")
                            break

                key = (reg.version,
                       int(round(float(s_xy[0]) / POSITION_GRID_M)),
                       int(round(float(s_xy[1]) / POSITION_GRID_M)))

                if best is None or best_dist > MOVED_MAX_RADIUS_M:
                    # No stable cell near this landmark. Only count misses
                    # while the landmark is inside FOV — if the robot hasn't
                    # looked there, absence is not evidence of anything.
                    if not self._object_in_camera_fov(s_xy):
                        self._miss_counters.pop(key, None)
                        continue
                    cnt = self._miss_counters.get(key, 0) + 1
                    self._miss_counters[key] = cnt
                    if cnt < MISS_THRESHOLD:
                        self.get_logger().info(
                            f"[SEM] '{class_name}' at ({s_xy[0]:.2f},{s_xy[1]:.2f}) "
                            f"miss {cnt}/{MISS_THRESHOLD} (awaiting persistence)",
                            throttle_duration_sec=3.0,
                        )
                        continue
                    self._miss_counters.pop(key, None)
                    self.get_logger().info(
                        f"[SEM] '{class_name}' at ({s_xy[0]:.2f},{s_xy[1]:.2f}) "
                        f"absent for {MISS_THRESHOLD} cycles — queuing retrain (removed)")
                    self._queue_retrain_event(
                        region_ver=reg.version,
                        old_xy=s_xy, new_xy=None, class_name=class_name,
                    )
                    positions.remove(stored_pos)
                    continue

                # Got a cell associated with this landmark — reset miss count.
                self._miss_counters.pop(key, None)

                if best_dist <= REASSOCIATE_RADIUS_M:
                    if best["last_class"] != class_name:
                        self.get_logger().info(
                            f"[SEM] R{reg.version} cell @ "
                            f"({best['xy'][0]:.2f},{best['xy'][1]:.2f}) "
                            f"stored as '{class_name}', observed as "
                            f"'{best['last_class']}' — same cell, no retrain",
                            throttle_duration_sec=10.0,
                        )
                    continue

                # REASSOCIATE < best_dist ≤ MOVED_MAX → object moved.
                self.get_logger().info(
                    f"[SEM] '{class_name}' at ({s_xy[0]:.2f},{s_xy[1]:.2f}) "
                    f"→ stable cell ({best['xy'][0]:.2f},{best['xy'][1]:.2f}) "
                    f"Δ={best_dist:.2f}m (count={best['count']})")
                self._queue_retrain_event(
                    region_ver=reg.version,
                    old_xy=s_xy,
                    new_xy=best["xy"].astype(np.float32).copy(),
                    class_name=class_name,
                )
                if isinstance(stored_pos, dict):
                    stored_pos["xy"] = best["xy"].astype(np.float32).tolist()
                else:
                    positions[positions.index(stored_pos)] = \
                        best["xy"].astype(np.float32).tolist()

            # Brand-new stable obstacle: no stored landmark (confirmed or tentative)
            # is near this cell, so treat it as a new object appearance.
            for cell in region_stable:
                cell_xy = np.asarray(cell["xy"][:2], dtype=np.float32)
                cell_class = str(cell.get("last_class", "")).strip()
                if cell_class not in RETRAIN_CLASSES:
                    continue

                claimed = False
                for _, _, _, s_xy in all_stored_any:
                    if float(np.linalg.norm(cell_xy - s_xy)) <= REASSOCIATE_RADIUS_M:
                        claimed = True
                        break
                if claimed:
                    continue

                self.get_logger().info(
                    f"[SEM] new stable '{cell_class}' at "
                    f"({cell_xy[0]:.2f},{cell_xy[1]:.2f}) "
                    f"count={int(cell.get('count', 0))} — queuing retrain (appeared)"
                )
                self._queue_retrain_event(
                    region_ver=reg.version,
                    old_xy=cell_xy,
                    new_xy=cell_xy.copy(),
                    class_name=cell_class,
                )

                # Seed a confirmed landmark immediately so this stable cell does
                # not re-trigger every semantic check cycle before/after retrain.
                entry = {
                    "xy": cell_xy.astype(np.float32).tolist(),
                    "count": max(int(cell.get("count", 0)), STABLE_MIN_COUNT),
                    "last_seen": _time.time(),
                    "class_name": cell_class,
                    "class_votes": {
                        cell_class: max(int(cell.get("count", 0)), STABLE_MIN_COUNT)
                    },
                }
                stored.setdefault(cell_class, []).append(entry)
                all_stored_any.append((cell_class, stored[cell_class], entry, cell_xy))

    @staticmethod
    def _semantic_entry_xy(entry) -> "np.ndarray | None":
        try:
            if isinstance(entry, dict):
                raw = entry.get("xy", entry.get("world_xy", entry.get("pos")))
            else:
                raw = entry
            xy = np.asarray(raw[:2], dtype=np.float32)
            if xy.shape[0] < 2 or not np.all(np.isfinite(xy[:2])):
                return None
            return xy[:2]
        except Exception:
            return None

    def _queue_retrain_event(
        self,
        region_ver: int,
        old_xy: np.ndarray,
        new_xy: "np.ndarray | None",
        class_name: str,
    ):
        """Add a retrain event if this exact region+class+location isn't queued.

        Dedup key includes a grid-rounded old_xy so multiple moved instances
        of the same class in one region (e.g. two chairs) are not collapsed
        to a single event.
        """
        gx = round(float(old_xy[0]) / RETRAIN_DEDUP_GRID_M)
        gy = round(float(old_xy[1]) / RETRAIN_DEDUP_GRID_M)
        with self._retrain_lock:
            for ev in self._pending_retrain_events:
                if (ev["region_ver"] == region_ver
                        and ev["class_name"] == class_name
                        and ev.get("_key") == (gx, gy)):
                    return  # already queued
            was_empty = len(self._pending_retrain_events) == 0
            self._pending_retrain_events.append({
                "region_ver": region_ver,
                "old_xy": old_xy,
                "new_xy": new_xy,
                "class_name": class_name,
                "_key": (gx, gy),
                "queued_at": _time.monotonic(),
            })
            if was_empty:
                # First event of a new burst — start the hold-off window so
                # subsequent events from the same camera/LiDAR sweep get
                # bundled into a single retrain.
                self._retrain_hold_until = _time.monotonic() + RETRAIN_HOLDOFF_SEC
        self.get_logger().info(
            f"[SEM] Queued retrain: R{region_ver} "
            f"{class_name} moved from ({old_xy[0]:.2f},{old_xy[1]:.2f})"
            + (f" to ({new_xy[0]:.2f},{new_xy[1]:.2f})"
               if new_xy is not None else " (missing)")
            + f" (hold-off {RETRAIN_HOLDOFF_SEC:.0f}s)"
        )

    def _event_focus_points(self, evs: list[dict]) -> list[np.ndarray]:
        pts: list[np.ndarray] = []
        for ev in evs:
            for raw in (ev.get("new_xy"), ev.get("old_xy")):
                if raw is None:
                    continue
                xy = np.asarray(raw[:2], dtype=np.float32)
                if not np.all(np.isfinite(xy)):
                    continue
                pts.append(xy)
        return pts

    def _merged_region_raw_rows(self, region: "Region") -> torch.Tensor:
        frames = getattr(region, "all_raw_frame_data", None) or []
        if not frames:
            return torch.zeros((0, 12), dtype=torch.float32)
        normed: list[torch.Tensor] = []
        for frame in frames:
            if not isinstance(frame, torch.Tensor):
                continue
            if frame.ndim != 2 or frame.shape[1] != 12:
                continue
            normed.append(frame.detach().cpu().float())
        if not normed:
            return torch.zeros((0, 12), dtype=torch.float32)
        return normed[0] if len(normed) == 1 else torch.cat(normed, dim=0)

    def _candidate_local_row_count(
        self,
        region: "Region",
        evs: list[dict],
        radius_m: float = RETRAIN_LOCAL_RADIUS_M,
    ) -> int:
        merged = self._merged_region_raw_rows(region)
        merged = self._filter_region_owned_rows(region, merged)
        if len(merged) == 0:
            return 0
        x0 = merged[:, 0:2].detach().cpu().numpy()
        focus = self._event_focus_points(evs)
        if not focus:
            return 0
        keep = np.zeros(len(x0), dtype=bool)
        for p in focus:
            d = np.linalg.norm(x0 - p[None, :], axis=1)
            keep |= (d <= float(radius_m))
        return int(np.count_nonzero(keep))

    def _candidate_model_dir(self, region_ver: int) -> str:
        return os.path.join(self._candidate_root, f"region_{int(region_ver)}")

    def _candidate_backup_dir(self, region_ver: int) -> str:
        ts = int(_time.time())
        return os.path.join(self._candidate_backup_root, f"region_{int(region_ver)}_{ts}")

    def _prepare_candidate_region(self, region_ver: int) -> str | None:
        src_region_dir = os.path.join(self.model_dir, f"region_{int(region_ver)}")
        if not os.path.isdir(src_region_dir):
            self.get_logger().warn(f"[SEM] live region dir missing for R{region_ver}: {src_region_dir}")
            return None

        os.makedirs(self._candidate_root, exist_ok=True)
        candidate_root = self._candidate_model_dir(region_ver)
        try:
            if os.path.isdir(candidate_root):
                shutil.rmtree(candidate_root)
            os.makedirs(candidate_root, exist_ok=True)
            shutil.copytree(
                src_region_dir,
                os.path.join(candidate_root, f"region_{int(region_ver)}"),
            )
            return candidate_root
        except Exception as exc:
            self.get_logger().error(f"[SEM] failed preparing candidate R{region_ver}: {exc}")
            return None

    def _load_region_from_root(self, model_root: str, region_ver: int) -> "Region | None":
        meta_path = os.path.join(model_root, f"region_{region_ver}", f"region_{region_ver}_meta.json")
        if not os.path.isfile(meta_path):
            return None
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            bound = torch.tensor(meta["bound"], dtype=torch.float32)
            region = Region(
                model_folder=model_root,
                bound=bound,
                version=int(region_ver),
                device="cuda:0" if torch.cuda.is_available() else "cpu",
            )
            if not region.load_model():
                return None
            return region
        except Exception as exc:
            self.get_logger().warn(f"[SEM] failed loading candidate R{region_ver}: {exc}")
            return None

    def _validate_candidate_region(self, region_ver: int, candidate_root: str) -> bool:
        candidate = self._load_region_from_root(candidate_root, region_ver)
        if candidate is None:
            return False
        incumbent = self.all_regions.get(int(region_ver))
        if incumbent is None:
            return False

        touched_room_ids = [rid for rid, ver in self._room_to_region.items() if int(ver) == int(region_ver)]
        room_prev = {rid: self.room_regions.get(rid) for rid in touched_room_ids}

        self.all_regions[int(region_ver)] = candidate
        for rid in touched_room_ids:
            self.room_regions[rid] = candidate
        try:
            if self.robot_pos is None or self._goal_xy is None:
                return True
            if self._point_in_collision(self._goal_xy):
                return False
            path = self._plan_path()
            return path is not None and len(path) >= 2 and not self._path_hits_wall(path)
        except Exception as exc:
            self.get_logger().warn(f"[SEM] candidate validation failed for R{region_ver}: {exc}")
            return False
        finally:
            self.all_regions[int(region_ver)] = incumbent
            for rid, prev in room_prev.items():
                if prev is not None:
                    self.room_regions[rid] = prev

    def _candidate_training_rows(
        self,
        region: "Region",
        evs: list[dict],
        radius_m: float = RETRAIN_LOCAL_RADIUS_M,
    ) -> torch.Tensor:
        merged = self._merged_region_raw_rows(region)
        merged = self._filter_region_owned_rows(region, merged)
        if len(merged) == 0:
            return merged

        pruned = merged
        for ev in evs:
            old_xy = ev.get("old_xy")
            new_xy = ev.get("new_xy")
            if old_xy is None and new_xy is None:
                continue
            old_xy = np.asarray(old_xy, dtype=np.float32)[:2] if old_xy is not None else None
            new_xy = np.asarray(new_xy, dtype=np.float32)[:2] if new_xy is not None else None
            x0 = pruned[:, 0:2].detach().cpu().numpy()
            keep = np.ones(len(pruned), dtype=bool)
            if old_xy is not None:
                keep &= (np.linalg.norm(x0 - old_xy[None, :], axis=1) > float(radius_m))
            if new_xy is not None:
                keep &= (np.linalg.norm(x0 - new_xy[None, :], axis=1) > float(radius_m))
            pruned = pruned[torch.from_numpy(keep).to(pruned.device)]
            if len(pruned) == 0:
                break

        focus = self._event_focus_points(evs)
        if not focus:
            return pruned

        merged_np = merged[:, 0:2].detach().cpu().numpy()
        local_keep = np.zeros(len(merged_np), dtype=bool)
        for p in focus:
            d = np.linalg.norm(merged_np - p[None, :], axis=1)
            local_keep |= (d <= float(radius_m))
        local_rows = merged[torch.from_numpy(local_keep).to(merged.device)]
        if len(local_rows) == 0:
            return pruned

        combined = torch.cat([pruned, local_rows], dim=0)
        return self._deduplicate_raw_rows(combined)

    def _promote_candidate_region(self, region_ver: int, candidate_root: str, evs: list[dict]) -> bool:
        src_region_dir = os.path.join(candidate_root, f"region_{int(region_ver)}")
        live_region_dir = os.path.join(self.model_dir, f"region_{int(region_ver)}")
        if not os.path.isdir(src_region_dir):
            self.get_logger().warn(f"[SEM] candidate dir missing for R{region_ver}: {src_region_dir}")
            return False

        os.makedirs(self._candidate_backup_root, exist_ok=True)
        backup_dir = self._candidate_backup_dir(region_ver)
        try:
            if os.path.isdir(live_region_dir):
                shutil.copytree(live_region_dir, backup_dir)
                shutil.rmtree(live_region_dir)
            shutil.copytree(src_region_dir, live_region_dir)
            new_region = self._load_region_from_root(self.model_dir, region_ver)
            if new_region is None:
                raise RuntimeError("promoted region failed to reload")
            self.all_regions[int(region_ver)] = new_region
            for rid, ver in self._room_to_region.items():
                if int(ver) == int(region_ver):
                    self.room_regions[int(rid)] = new_region
            for ev in evs:
                if ev.get("new_xy") is not None:
                    self._paint_obstacle_on_map(ev["new_xy"])
            return True
        except Exception as exc:
            self.get_logger().error(f"[SEM] failed promoting candidate R{region_ver}: {exc}")
            return False

    # ══════════════════════════════════════════════════════════════════════════
    # TF
    # ══════════════════════════════════════════════════════════════════════════

    def _update_pose_from_tf(self):
        frames = [self.base_frame]
        if self.base_frame != "base_footprint":
            frames.append("base_footprint")
        if self.base_frame != "base_link":
            frames.append("base_link")
        try:
            t = None
            last_exc = None
            for frame in frames:
                try:
                    t = self.tf_buffer.lookup_transform(
                        "map", frame,
                        rclpy.time.Time(), timeout=Duration(seconds=0.05))
                    break
                except Exception as exc:
                    last_exc = exc
            if t is None:
                raise last_exc if last_exc is not None else RuntimeError("no base TF")
            p = t.transform.translation
            q = t.transform.rotation
            self.robot_pos = np.array([p.x, p.y, p.z], dtype=np.float32)
            r = Rotation.from_quat([q.x, q.y, q.z, q.w])
            self.robot_yaw = float(r.as_euler("xyz")[2])
            if self.state == "WAITING_FOR_POSE":
                self.state = "IDLE"
        except Exception:
            pass

    # ══════════════════════════════════════════════════════════════════════════
    # Path planning
    # ══════════════════════════════════════════════════════════════════════════

    def _plan_path(self):
        """Plan a multi-room path from robot to goal."""
        robot_xy = self.robot_pos[:2].copy()
        goal_xy = self._goal_xy.copy()
        metric = {
            "goal_request_id": int(self._active_goal_request_id) if self._active_goal_request_id is not None else None,
            "planner_mode": str(self.planner_mode),
            "start_xy": [float(robot_xy[0]), float(robot_xy[1])],
            "goal_xy": [float(goal_xy[0]), float(goal_xy[1])],
            "total_plan_ms": None,
            "path_points": 0,
            "path_length_m": None,
            "success": False,
            "failure_reason": None,
            "segments": [],
        }
        t0 = _time.monotonic()

        if self.planner_mode == "region_graph":
            path, reason = self._plan_region_graph_path(robot_xy, goal_xy, metric)
            if path is not None and len(path) >= 2:
                metric["success"] = True
                metric["path_points"] = int(len(path))
                metric["path_length_m"] = float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
                metric["total_plan_ms"] = round((_time.monotonic() - t0) * 1e3, 3)
                self._trajectory_plan_metrics.append(metric)
                return path
            self.get_logger().warn(
                f"[NAV] Region-graph planner failed ({reason}); "
                "falling back to room graph.")
            metric["failure_reason"] = f"region_graph_failed:{reason}"

        path = self._plan_room_graph_path(robot_xy, goal_xy, metric)
        metric["success"] = bool(path is not None and len(path) >= 2)
        if metric["success"]:
            metric["path_points"] = int(len(path))
            metric["path_length_m"] = float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
        elif metric["failure_reason"] is None:
            metric["failure_reason"] = "room_graph_failed"
        metric["total_plan_ms"] = round((_time.monotonic() - t0) * 1e3, 3)
        self._trajectory_plan_metrics.append(metric)
        return path

    def _plan_room_graph_path(self, robot_xy: np.ndarray, goal_xy: np.ndarray, metric: dict | None = None):
        """Legacy room-graph planner."""

        cur_room = self.room_graph.room_for_point(robot_xy)
        goal_room = self.room_graph.room_for_point(goal_xy)

        if cur_room is None or goal_room is None:
            self.get_logger().warn("[NAV] Robot or goal not in any room.")
            return None

        if cur_room == goal_room:
            # Same room — single field path
            region = self.room_regions.get(cur_room)
            if region is None:
                self.get_logger().warn(f"[NAV] No region for room {cur_room}")
                if metric is not None:
                    metric["failure_reason"] = f"no_region_for_room_{cur_room}"
                return None
            seg_t0 = _time.monotonic()
            path = region.generate_field_path(robot_xy, goal_xy,
                                              step_size=0.15, max_steps=150)
            seg_ms = (_time.monotonic() - seg_t0) * 1e3
            if metric is not None:
                metric["segments"].append({
                    "region_version": int(region.version),
                    "field_generate_ms": round(seg_ms, 3),
                    "fallback_used": False,
                    "segment_points": int(len(path)) if path is not None else 0,
                    "segment_length_m": (
                        float(np.sum(np.linalg.norm(np.diff(np.asarray(path, dtype=np.float32), axis=0), axis=1)))
                        if path is not None and len(path) >= 2 else None
                    ),
                    "success": bool(path is not None and len(path) >= 2),
                })
            if path is not None and len(path) >= 2:
                return np.array(path, dtype=np.float32)
            # fallback
            if metric is not None:
                metric["segments"][-1]["fallback_used"] = True
            return np.array([robot_xy, goal_xy], dtype=np.float32)

        # Different rooms — use room graph
        try:
            route = self.room_graph.shortest_path(cur_room, goal_room)
        except Exception as e:
            self.get_logger().warn(f"[NAV] No path: {e}")
            if metric is not None:
                metric["failure_reason"] = f"room_graph_no_path:{e}"
            return None

        # Build chain of (entry, exit, region) tuples
        all_waypoints: list[np.ndarray] = [robot_xy]

        for i, node in enumerate(route):
            if node["type"] == "doorway":
                all_waypoints.append(np.array(node["position"], dtype=np.float32))
            elif node["type"] == "room" and i == len(route) - 1:
                # goal room — add actual goal
                all_waypoints.append(goal_xy)

        # Generate field path for each segment
        full_path_parts: list[np.ndarray] = []
        for seg_i in range(len(all_waypoints) - 1):
            start = all_waypoints[seg_i]
            end = all_waypoints[seg_i + 1]

            # Determine which room this segment is in
            mid = 0.5 * (start + end)
            room_id = self.room_graph.room_for_point(mid)
            region = self.room_regions.get(room_id) if room_id else None

            if region is not None:
                seg_t0 = _time.monotonic()
                path = region.generate_field_path(start, end,
                                                  step_size=0.15, max_steps=120,
                                                  forward_only=True)
                seg_ms = (_time.monotonic() - seg_t0) * 1e3
                if path is not None and len(path) >= 2:
                    if metric is not None:
                        arr = np.array(path, dtype=np.float32)
                        metric["segments"].append({
                            "region_version": int(region.version),
                            "field_generate_ms": round(seg_ms, 3),
                            "fallback_used": False,
                            "segment_points": int(len(arr)),
                            "segment_length_m": float(np.sum(np.linalg.norm(np.diff(arr, axis=0), axis=1))),
                            "success": True,
                        })
                    full_path_parts.append(np.array(path, dtype=np.float32))
                    continue
                if metric is not None:
                    metric["segments"].append({
                        "region_version": int(region.version),
                        "field_generate_ms": round(seg_ms, 3),
                        "fallback_used": True,
                        "segment_points": 2,
                        "segment_length_m": float(np.linalg.norm(end - start)),
                        "success": True,
                    })

            # Fallback: straight line
            full_path_parts.append(np.array([start, end], dtype=np.float32))

        if not full_path_parts:
            if metric is not None and metric["failure_reason"] is None:
                metric["failure_reason"] = "no_full_path_parts"
            return None

        # Concatenate, removing duplicate junction points
        combined = [full_path_parts[0]]
        for part in full_path_parts[1:]:
            if len(combined[-1]) > 0 and len(part) > 0:
                if np.linalg.norm(combined[-1][-1] - part[0]) < 0.1:
                    part = part[1:]  # skip duplicate start
            if len(part) > 0:
                combined.append(part)

        full_path = np.concatenate(combined, axis=0)

        # Resample for smooth following
        full_path = resample_polyline(full_path, ds=0.05)
        return full_path

    def _plan_region_graph_path(self, robot_xy: np.ndarray, goal_xy: np.ndarray, metric: dict | None = None):
        """Plan using the same region-level idea used by /region_graph_path.

        This avoids assigning the clicked goal to a room first.  RegionGraphInspector
        routes between field regions, uses crossing/doorway transition points, then
        decimates and smooths the final path with occupancy validation.
        """
        if not self.all_regions:
            return None, "no_regions_loaded"
        if self._map_occ is None or self._map_free is None or self._map_resolution is None:
            return None, "map_not_ready"

        origin = np.asarray(self._map_origin, dtype=np.float32)
        occ = np.asarray(self._map_occ, dtype=bool)
        free = np.asarray(self._map_free, dtype=bool)
        ownership = compute_ownership_map(
            free=free,
            regions=self.all_regions,
            origin=origin,
            cell_size=float(self._map_resolution),
        )
        crossings = compute_crossing_points(
            ownership=ownership,
            origin=origin,
            cell_size=float(self._map_resolution),
            occ_grid=occ,
        )

        start_region = owner_region_for_xy(
            robot_xy, ownership, self.all_regions, origin, float(self._map_resolution))
        goal_region = owner_region_for_xy(
            goal_xy, ownership, self.all_regions, origin, float(self._map_resolution))
        if start_region is None or goal_region is None:
            return None, "start_or_goal_region_unassigned"

        s_ver = int(start_region.version)
        g_ver = int(goal_region.version)
        route = self._region_route(s_ver, g_ver, crossings)
        if not route:
            return None, (
                f"no_region_route R{s_ver}->R{g_ver} "
                f"crossings={len(crossings)} topo_edges={len(self._topology_edges)}"
            )

        waypoints = [robot_xy.astype(np.float32)]
        for u, v in zip(route[:-1], route[1:]):
            transition = self._transition_point(u, v, crossings, goal_xy)
            if transition is None:
                return None, f"no_transition_point R{u}->R{v} crossings={len(crossings)}"
            waypoints.append(transition.astype(np.float32))
        waypoints.append(goal_xy.astype(np.float32))

        segments: list[np.ndarray] = []
        used_regions: list[int] = []
        cur_xy = robot_xy.astype(np.float32)
        for i in range(len(route) - 1):
            reg_ver = int(route[i])
            next_reg_ver = int(route[i + 1])
            door_xy = waypoints[i + 1].astype(np.float32)
            source_attempts = self._doorway_anchors_in_region(reg_ver, door_xy, goal_xy)
            if not source_attempts:
                return None, (
                    f"no_source_doorway_anchor R{reg_ver}->R{next_reg_ver} "
                    f"door=({door_xy[0]:.2f},{door_xy[1]:.2f}) route={route}"
                )

            hop_ok = False
            last_reason = "unknown"
            for src_offset, src_anchor in source_attempts:
                seg_t0 = _time.monotonic()
                seg, reason = self._field_segment(
                    reg_ver,
                    cur_xy,
                    src_anchor,
                    max_steps=self.field_max_steps,
                )
                seg_ms = (_time.monotonic() - seg_t0) * 1e3
                if seg is None:
                    last_reason = reason
                    continue

                seg_use = np.asarray(seg, dtype=np.float32)
                seg_reason = reason
                bridge_used = False
                if src_offset > 1e-6:
                    bridge = straight_line_path(
                        src_anchor[:2], door_xy[:2], step=0.05,
                        collision_fn=lambda xy: self._point_in_collision(xy),
                    )
                    if len(bridge) < 2:
                        last_reason = f"source_bridge_too_short@{src_offset:.2f}"
                        continue
                    bridge_np = np.asarray(bridge, dtype=np.float32)
                    if self._path_hits_wall(bridge_np):
                        last_reason = f"source_bridge_collides@{src_offset:.2f}"
                        continue
                    seg_use = np.vstack([seg_use, bridge_np[1:]])
                    seg_reason = f"doorway_cushion_{src_offset:.2f}m+{reason}"
                    bridge_used = True

                segments.append(seg_use)
                used_regions.append(reg_ver)
                if metric is not None:
                    metric["segments"].append({
                        "region_version": int(reg_ver),
                        "field_generate_ms": round(seg_ms, 3),
                        "fallback_used": bool(bridge_used),
                        "segment_points": int(len(seg_use)),
                        "segment_length_m": float(np.sum(np.linalg.norm(np.diff(seg_use, axis=0), axis=1))),
                        "success": True,
                        "reason": str(seg_reason),
                    })
                cur_xy = door_xy.copy()
                hop_ok = True
                break

            if not hop_ok:
                return None, (
                    f"segment_failed R{reg_ver}->{next_reg_ver} reason={last_reason} "
                    f"start=({cur_xy[0]:.2f},{cur_xy[1]:.2f}) "
                    f"door=({door_xy[0]:.2f},{door_xy[1]:.2f}) route={route}"
                )

        final_reg_ver = int(route[-1])
        final_t0 = _time.monotonic()
        final_seg, final_reason = self._field_from_doorway(
            final_reg_ver, cur_xy, goal_xy.astype(np.float32), self.field_max_steps
        )
        final_ms = (_time.monotonic() - final_t0) * 1e3
        if final_seg is None:
            return None, (
                f"final_segment_failed R{final_reg_ver} reason={final_reason} "
                f"start=({cur_xy[0]:.2f},{cur_xy[1]:.2f}) "
                f"goal=({goal_xy[0]:.2f},{goal_xy[1]:.2f}) route={route}"
            )
        final_seg_np = np.asarray(final_seg, dtype=np.float32)
        segments.append(final_seg_np)
        used_regions.append(final_reg_ver)
        if metric is not None:
            metric["segments"].append({
                "region_version": int(final_reg_ver),
                "field_generate_ms": round(final_ms, 3),
                "fallback_used": False,
                "segment_points": int(len(final_seg_np)),
                "segment_length_m": float(np.sum(np.linalg.norm(np.diff(final_seg_np, axis=0), axis=1))),
                "success": True,
                "reason": str(final_reason),
            })

        full = self._concat_paths(segments)
        if full is None or len(full) < 2:
            return None, f"concat_failed route={route} n_segments={len(segments)}"

        decimated = self._decimate(full, spacing=0.10)
        smoothed = self._smooth(decimated, window=7)
        if self._path_hits_wall(smoothed):
            final_path = decimated
            if self._path_hits_wall(final_path):
                return None, (
                    f"path_collides_after_smooth_and_decimate route={route} "
                    f"raw_pts={len(full)} decimated_pts={len(decimated)}"
                )
        else:
            final_path = smoothed

        final_path = resample_polyline(final_path, ds=0.05)
        plen = self._path_length(final_path)
        route_str = " -> ".join(f"R{v}" for v in route)
        self.get_logger().info(
            f"[NAV] Region graph path ready: pts={len(final_path)} "
            f"len={plen:.2f}m route=[{route_str}]")
        return final_path, "ok"

    def _region_route(self, start_ver: int, goal_ver: int, crossings: dict) -> list[int] | None:
        if start_ver == goal_ver:
            return [start_ver]

        adj: dict[int, set[int]] = {int(v): set() for v in self.all_regions}
        if not self._strict_roomwise_topology:
            for u, v in crossings.keys():
                if u in adj and v in adj:
                    adj[int(u)].add(int(v))
                    adj[int(v)].add(int(u))

        # Topology edges — supports {"rooms":[a,b]} (current format) and
        # {"v1":a,"v2":b} (legacy). Room IDs translated to region versions.
        for edge in self._topology_edges:
            if not isinstance(edge, dict):
                continue
            rooms = edge.get("rooms")
            if rooms and len(rooms) >= 2:
                u = self._room_to_region.get(int(rooms[0]), int(rooms[0]))
                v = self._room_to_region.get(int(rooms[1]), int(rooms[1]))
            elif "v1" in edge and "v2" in edge:
                u, v = int(edge["v1"]), int(edge["v2"])
            else:
                continue
            if u in adj and v in adj and u != v:
                adj[u].add(v)
                adj[v].add(u)

        if goal_ver in adj.get(start_ver, set()):
            return [start_ver, goal_ver]

        queue = [start_ver]
        parent = {start_ver: None}
        while queue:
            cur = queue.pop(0)
            if cur == goal_ver:
                break
            for nb in sorted(adj.get(cur, set())):
                if nb not in parent:
                    parent[nb] = cur
                    queue.append(nb)
        if goal_ver not in parent:
            return None
        out = []
        cur = goal_ver
        while cur is not None:
            out.append(cur)
            cur = parent[cur]
        return list(reversed(out))

    def _transition_point(self, u: int, v: int, crossings: dict, goal_xy: np.ndarray):
        key = (min(int(u), int(v)), max(int(u), int(v)))
        candidates = []
        if (not self._strict_roomwise_topology) and key in crossings:
            candidates.append(np.asarray(crossings[key], dtype=np.float32))

        reg_u = self.all_regions.get(int(u))
        reg_v = self.all_regions.get(int(v))
        if reg_u is not None and reg_v is not None:
            for door in self._topology_doors:
                if not isinstance(door, dict):
                    continue
                pos = door.get("pos_xy") or door.get("position")
                if pos is None:
                    continue
                p = np.asarray(pos, dtype=np.float32).reshape(-1)[:2]
                if self._point_in_collision(p):
                    continue

                # Primary: match door room_ids → region versions
                room_ids = door.get("room_ids", [])
                region_ids = {self._room_to_region.get(int(r), int(r)) for r in room_ids}
                if region_ids == {int(u), int(v)}:
                    candidates.append(p)
                    continue

                if not self._strict_roomwise_topology:
                    # Fallback: spatial in_bound with small margin
                    MARGIN = 0.3
                    def _in_margin(reg, pt):
                        b = reg.bound.numpy() if hasattr(reg.bound, 'numpy') else np.array(reg.bound)
                        return (b[0,0]-MARGIN <= pt[0] <= b[1,0]+MARGIN and
                                b[0,1]-MARGIN <= pt[1] <= b[1,1]+MARGIN)
                    if _in_margin(reg_u, p) and _in_margin(reg_v, p):
                        candidates.append(p)

        if not candidates:
            return None
        goal_xy = np.asarray(goal_xy[:2], dtype=np.float32)
        candidates.sort(key=lambda p: float(np.linalg.norm(p - goal_xy)))
        return candidates[0]

    def _doorway_anchors_in_region(self, region_ver: int, door_xy: np.ndarray,
                                   toward_xy: np.ndarray):
        region = self.all_regions.get(int(region_ver))
        if region is None:
            return []

        door = np.asarray(door_xy[:2], dtype=np.float32)
        target = np.asarray(toward_xy[:2], dtype=np.float32)
        direction = target - door
        dist = float(np.linalg.norm(direction))
        if dist < 1e-6:
            direction = np.array([1.0, 0.0], dtype=np.float32)
        else:
            direction = (direction / dist).astype(np.float32)

        attempts = []
        seen = set()
        offsets = self.doorway_cushion_offsets_m or [0.0]
        for offset in offsets:
            cand = door if float(offset) <= 1e-6 else (door + direction * float(offset)).astype(np.float32)
            cand = self._clip_to_region_bounds(region_ver, cand)
            cand_free = self._nearest_free_xy(cand, region=region)
            if cand_free is None:
                continue
            cand_free = np.asarray(cand_free[:2], dtype=np.float32)
            if not region.in_bound(cand_free):
                continue
            key = (int(round(float(cand_free[0]) * 100.0)),
                   int(round(float(cand_free[1]) * 100.0)))
            if key in seen:
                continue
            seen.add(key)
            attempts.append((float(offset), cand_free))
        return attempts

    def _field_from_doorway(self, region_ver: int, door_xy: np.ndarray,
                            goal_xy: np.ndarray, max_steps: int):
        attempts = self._doorway_anchors_in_region(region_ver, door_xy, goal_xy)
        if not attempts:
            return None, "no_goal_region_doorway_anchor"

        last_reason = "unknown"
        for offset, anchor_xy in attempts:
            seg, reason = self._field_segment(region_ver, anchor_xy, goal_xy, max_steps)
            if seg is None:
                last_reason = reason
                continue
            if offset <= 1e-6:
                return np.asarray(seg, dtype=np.float32), reason

            bridge = straight_line_path(
                door_xy[:2], anchor_xy[:2], step=0.05,
                collision_fn=lambda xy: self._point_in_collision(xy),
            )
            if len(bridge) < 2:
                last_reason = f"goal_bridge_too_short@{offset:.2f}"
                continue
            bridge_np = np.asarray(bridge, dtype=np.float32)
            if self._path_hits_wall(bridge_np):
                last_reason = f"goal_bridge_collides@{offset:.2f}"
                continue
            combined = np.vstack([bridge_np, np.asarray(seg, dtype=np.float32)[1:]])
            return combined, f"doorway_cushion_{offset:.2f}m+{reason}"

        return None, f"all_goal_doorway_cushions_failed;{last_reason}"

    def _field_segment(self, region_ver: int, start_xy: np.ndarray,
                       goal_xy: np.ndarray, max_steps: int):
        region = self.all_regions.get(int(region_ver))
        if region is None:
            return None, "region_not_loaded"

        raw_np = None
        raw_reason = "ok"
        try:
            raw = region.generate_field_path(
                start_xy.astype(np.float32),
                goal_xy.astype(np.float32),
                step_size=self.field_step_size,
                max_steps=max_steps,
            )
        except Exception as exc:
            raw = None
            raw_reason = f"raw_exception:{type(exc).__name__}"

        if raw is not None:
            raw_np = np.asarray(raw, dtype=np.float32)
            if len(raw_np) < 2:
                raw_reason = "raw_too_few_points"
            elif self._path_hits_wall(raw_np):
                raw_reason = "raw_collides"
            else:
                return raw_np, "ok"
        elif raw_reason == "ok":
            raw_reason = "raw_returned_none"

        constrained, reason = self._constrained_field_rollout(
            region_ver, start_xy, goal_xy, max_steps)
        if constrained is not None and len(constrained) >= 2:
            return constrained, f"ok_constrained_after_{raw_reason}"

        if raw_np is not None and len(raw_np) >= 2 and not self._path_hits_wall(raw_np):
            return raw_np, f"ok_raw_after_{reason}"

        return None, f"{reason};{raw_reason}"

    def _constrained_field_rollout(self, region_ver: int, start_xy: np.ndarray,
                                   goal_xy: np.ndarray, max_steps: int):
        region = self.all_regions.get(int(region_ver))
        if region is None:
            return None, "constrained_region_not_loaded"

        start = self._nearest_free_xy(
            self._clip_to_region_bounds(region_ver, start_xy.astype(np.float32)),
            region=region,
        )
        goal = self._nearest_free_xy(
            self._clip_to_region_bounds(region_ver, goal_xy.astype(np.float32)),
            region=region,
        )
        if start is None:
            return None, "constrained_no_free_start"
        if goal is None:
            return None, "constrained_no_free_goal"

        cur = start.astype(np.float32)
        goal = goal.astype(np.float32)
        pts = [cur.copy()]
        best_goal_dist = float(np.linalg.norm(cur - goal))

        for k in range(max(10, int(max_steps))):
            goal_dist = float(np.linalg.norm(cur - goal))
            best_goal_dist = min(best_goal_dist, goal_dist)
            if goal_dist <= max(0.12, self.field_step_size):
                pts.append(goal.copy())
                out = np.asarray(pts, dtype=np.float32)
                if self._path_hits_wall(out):
                    return None, "constrained_final_collides"
                return out, "ok"

            try:
                nxt = region.get_next_waypoint(
                    cur.astype(np.float32),
                    goal.astype(np.float32),
                    step_size=self.field_step_size,
                    debug=False,
                )
                nxt = np.asarray(nxt, dtype=np.float32).reshape(-1)[:2]
            except Exception:
                return None, "constrained_next_waypoint_exception"

            if not np.isfinite(nxt).all():
                return None, "constrained_non_finite_step"

            nxt = self._clip_to_region_bounds(region_ver, nxt)
            if self._point_in_collision(nxt) or self._segment_hits_wall(cur, nxt):
                nxt = self._rescue_field_step(region_ver, cur, goal, self.field_step_size)
                if nxt is None:
                    return None, "constrained_step_blocked"

            if float(np.linalg.norm(nxt - cur)) < 1e-4:
                return None, (
                    f"constrained_stalled(step={k},dist={goal_dist:.2f},best={best_goal_dist:.2f})"
                )

            pts.append(nxt.copy())
            cur = nxt

        end_dist = float(np.linalg.norm(cur - goal))
        progress = max(0.0, float(np.linalg.norm(start - goal) - end_dist))
        return None, (
            f"constrained_max_steps(end_dist={end_dist:.2f},best={best_goal_dist:.2f},"
            f"progress={progress:.2f})"
        )

    def _rescue_field_step(self, region_ver: int, cur: np.ndarray,
                           goal: np.ndarray, step: float):
        d = goal - cur
        dn = float(np.linalg.norm(d))
        if dn > 1e-6:
            cand = self._clip_to_region_bounds(region_ver, cur + step * d / dn)
            if not self._point_in_collision(cand) and not self._segment_hits_wall(cur, cand):
                return cand.astype(np.float32)

        base = math.atan2(float(d[1]), float(d[0])) if dn > 1e-6 else 0.0
        for ang in np.linspace(-math.pi, math.pi, 24, endpoint=False):
            cand = np.array(
                [cur[0] + step * math.cos(base + ang),
                 cur[1] + step * math.sin(base + ang)],
                dtype=np.float32,
            )
            cand = self._clip_to_region_bounds(region_ver, cand)
            if float(np.linalg.norm(cand - cur)) < 1e-4:
                continue
            if self._point_in_collision(cand):
                continue
            if self._segment_hits_wall(cur, cand):
                continue
            return cand.astype(np.float32)
        return None

    def _clip_to_region_bounds(self, region_ver: int, xy: np.ndarray) -> np.ndarray:
        region = self.all_regions.get(int(region_ver))
        if region is None:
            return xy.copy()
        b = region.bound.detach().cpu().numpy()
        margin = max(0.0, float(self.region_bound_margin_m))
        xmin, ymin = float(b[0, 0] + margin), float(b[0, 1] + margin)
        xmax, ymax = float(b[1, 0] - margin), float(b[1, 1] - margin)
        if xmin > xmax:
            xmin, xmax = float(b[0, 0]), float(b[1, 0])
        if ymin > ymax:
            ymin, ymax = float(b[0, 1]), float(b[1, 1])
        return np.array([
            np.clip(float(xy[0]), xmin, xmax),
            np.clip(float(xy[1]), ymin, ymax),
        ], dtype=np.float32)

    def _nearest_free_xy(self, xy: np.ndarray, region: Region | None = None):
        if self._map_occ is None or self._map_free is None or self._map_resolution is None:
            return np.asarray(xy[:2], dtype=np.float32)
        rc = self._world_to_map(xy)
        if rc is None:
            return None
        r0, c0 = rc
        H, W = self._map_occ.shape
        if self._map_free[r0, c0] and (region is None or region.in_bound(xy)):
            return np.asarray(xy[:2], dtype=np.float32)
        for rad in range(1, 30):
            r_min, r_max = max(0, r0 - rad), min(H - 1, r0 + rad)
            c_min, c_max = max(0, c0 - rad), min(W - 1, c0 + rad)
            candidates = []
            for r in range(r_min, r_max + 1):
                for c in (c_min, c_max):
                    if self._map_free[r, c]:
                        candidates.append((r, c))
            for c in range(c_min + 1, c_max):
                for r in (r_min, r_max):
                    if self._map_free[r, c]:
                        candidates.append((r, c))
            best = None
            best_d = float("inf")
            for r, c in candidates:
                p = self._map_to_world(r, c)
                if region is not None and not region.in_bound(p):
                    continue
                d = float(np.linalg.norm(p - xy[:2]))
                if d < best_d:
                    best = p
                    best_d = d
            if best is not None:
                return best.astype(np.float32)
        return None

    def _world_to_map(self, xy: np.ndarray):
        if self._map_origin is None or self._map_resolution is None or self._map_occ is None:
            return None
        ox, oy = float(self._map_origin[0]), float(self._map_origin[1])
        c = int((float(xy[0]) - ox) / float(self._map_resolution))
        r = int((float(xy[1]) - oy) / float(self._map_resolution))
        H, W = self._map_occ.shape
        if 0 <= r < H and 0 <= c < W:
            return r, c
        return None

    def _map_to_world(self, r: int, c: int) -> np.ndarray:
        ox, oy = float(self._map_origin[0]), float(self._map_origin[1])
        res = float(self._map_resolution)
        return np.array([ox + (c + 0.5) * res, oy + (r + 0.5) * res], dtype=np.float32)

    def _paint_obstacle_on_map(
        self,
        world_xy: np.ndarray,
        radius_m: float = OBSTACLE_PAINT_RADIUS_M,
    ) -> bool:
        """Mark a circular footprint as occupied at world_xy.

        Updates both the in-memory collision grid (_map_occ / _map_free)
        and the saved map.pgm on disk so the new obstacle persists across
        sessions.  Returns True if the disk write succeeded.
        """
        if self._map_origin is None or self._map_resolution is None or self._map_occ is None:
            return False

        res = float(self._map_resolution)
        H, W = self._map_occ.shape
        ox, oy = float(self._map_origin[0]), float(self._map_origin[1])

        cx = float(world_xy[0])
        cy = float(world_xy[1])
        r_cells = max(1, int(math.ceil(radius_m / res)))
        grid_c = int((cx - ox) / res)
        grid_r = int((cy - oy) / res)

        # Paint a filled circle in the in-memory grids
        for dr in range(-r_cells, r_cells + 1):
            for dc in range(-r_cells, r_cells + 1):
                if dr * dr + dc * dc > r_cells * r_cells:
                    continue
                rr, cc = grid_r + dr, grid_c + dc
                if 0 <= rr < H and 0 <= cc < W:
                    self._map_occ[rr, cc] = True
                    if self._map_free is not None:
                        self._map_free[rr, cc] = False

        # Update map.pgm on disk
        pgm_path = os.path.join(self.model_dir, "map.pgm")
        if not os.path.isfile(pgm_path):
            return False
        try:
            with open(pgm_path, "rb") as f:
                raw = f.read()

            # Parse P5 header
            lines = []
            idx = 0
            while len(lines) < 3:
                end = raw.index(b"\n", idx)
                line = raw[idx:end].decode("ascii").strip()
                if not line.startswith("#"):
                    lines.append(line)
                idx = end + 1
            assert lines[0] == "P5", "Not a P5 PGM"
            pgm_W, pgm_H = map(int, lines[1].split())
            data_offset = idx

            img = np.frombuffer(raw[data_offset:], dtype=np.uint8).reshape(pgm_H, pgm_W).copy()

            # PGM is stored flipped (row 0 = top = high y).
            # pgm_row = (H - 1) - grid_row
            for dr in range(-r_cells, r_cells + 1):
                for dc in range(-r_cells, r_cells + 1):
                    if dr * dr + dc * dc > r_cells * r_cells:
                        continue
                    rr, cc = grid_r + dr, grid_c + dc
                    if 0 <= rr < pgm_H and 0 <= cc < pgm_W:
                        pgm_row = (pgm_H - 1) - rr
                        img[pgm_row, cc] = 0  # occupied → black

            with open(pgm_path, "wb") as f:
                header = f"P5\n{pgm_W} {pgm_H}\n255\n".encode("ascii")
                f.write(header)
                f.write(img.tobytes())

            self.get_logger().info(
                f"[MAP] Painted obstacle @ "
                f"({cx:.2f},{cy:.2f}) r={radius_m:.2f}m onto {pgm_path}")
            return True

        except Exception as exc:
            self.get_logger().error(f"[MAP] Failed to update map.pgm: {exc}")
            return False

    def _point_in_collision(self, xy: np.ndarray) -> bool:
        rc = self._world_to_map(xy)
        if rc is None or self._map_occ is None:
            return False
        r, c = rc
        return bool(self._map_occ[r, c])

    def _segment_hits_wall(self, a: np.ndarray, b: np.ndarray) -> bool:
        a = np.asarray(a[:2], dtype=np.float32)
        b = np.asarray(b[:2], dtype=np.float32)
        d = float(np.linalg.norm(b - a))
        if d < 1e-6:
            return self._point_in_collision(a)
        n = max(2, int(d / max(0.02, float(self._map_resolution or 0.05))))
        for t in np.linspace(0.0, 1.0, n):
            p = a + t * (b - a)
            if self._point_in_collision(p):
                return True
        return False

    def _path_hits_wall(self, path_xy: np.ndarray) -> bool:
        pts = np.asarray(path_xy, dtype=np.float32)
        if len(pts) < 2:
            return False
        for p in pts:
            if self._point_in_collision(p):
                return True
        for a, b in zip(pts[:-1], pts[1:]):
            if self._segment_hits_wall(a, b):
                return True
        return False

    @staticmethod
    def _concat_paths(paths: list[np.ndarray]) -> np.ndarray | None:
        if not paths:
            return None
        out = [paths[0]]
        for seg in paths[1:]:
            if len(seg) == 0:
                continue
            if np.linalg.norm(out[-1][-1] - seg[0]) < 1e-3:
                out.append(seg[1:])
            else:
                out.append(seg)
        valid = [s for s in out if len(s) > 0]
        if not valid:
            return None
        cat = np.concatenate(valid, axis=0)
        return cat if len(cat) >= 2 else None

    @staticmethod
    def _decimate(path_xy: np.ndarray, spacing: float = 0.10) -> np.ndarray:
        pts = path_xy.astype(np.float32)
        out = [pts[0]]
        for i in range(1, len(pts)):
            if np.linalg.norm(pts[i] - out[-1]) >= spacing:
                out.append(pts[i])
        if np.linalg.norm(out[-1] - pts[-1]) > 0.05:
            out.append(pts[-1])
        return np.asarray(out, dtype=np.float32)

    @staticmethod
    def _smooth(path_xy: np.ndarray, window: int = 7) -> np.ndarray:
        pts = path_xy.astype(np.float32)
        if len(pts) < window:
            return pts
        sm = pts.copy()
        half = window // 2
        for i in range(1, len(pts) - 1):
            lo = max(0, i - half)
            hi = min(len(pts), i + half + 1)
            sm[i] = pts[lo:hi].mean(axis=0)
        sm[0] = pts[0]
        sm[-1] = pts[-1]
        return sm.astype(np.float32)

    @staticmethod
    def _path_length(path_xy: np.ndarray) -> float:
        pts = np.asarray(path_xy, dtype=np.float32)
        if len(pts) < 2:
            return 0.0
        return float(np.linalg.norm(np.diff(pts[:, :2], axis=0), axis=1).sum())

    # ══════════════════════════════════════════════════════════════════════════
    # Main update loop
    # ══════════════════════════════════════════════════════════════════════════

    def _update_tick(self):
        if self.state == "RETRAINING":
            # Hard halt while the background retrain worker runs. Pose TF can
            # still update via _scan_cb's early return, but no planning, no
            # driving, no new goals advance until _do_retrain restores state.
            self.cmd_pub.publish(Twist())
            return

        if self.state == "WAITING_FOR_POSE":
            self._update_pose_from_tf()
            return

        if self.state == "IDLE":
            self.cmd_pub.publish(Twist())
            return

        if self.state == "PLANNING":
            if self._planning_thread is None or not self._planning_thread.is_alive():
                self._planning_thread = threading.Thread(
                    target=self._do_plan, daemon=True)
                self._planning_thread.start()
            return

        if self.state == "READY":
            # Show path, wait for keyboard confirm (or auto-proceed)
            if self._keyboard_event is not None:
                if self._keyboard_event.is_set():
                    self._keyboard_event.clear()
                    self.navigator.reset_path()
                    self.state = "DRIVING"
                    self.get_logger().info("[NAV] DRIVING")
            else:
                self.navigator.reset_path()
                self.state = "DRIVING"
            return

        if self.state == "DRIVING":
            self._drive_tick()
            return

    def _do_plan(self):
        """Run planning in a background thread."""
        path = self._plan_path()
        if path is not None and len(path) >= 2:
            self._nav_path = path
            self._publish_nav_path(path)
            self._publish_goal_marker(self._goal_xy)
            if self._keyboard_event is None:
                self.navigator.reset_path()
                self.state = "DRIVING"
                self.get_logger().info(
                    f"[NAV] Path ready ({len(path)} pts). Driving.")
            else:
                self.state = "READY"
                self.get_logger().info(
                    f"[NAV] Path ready ({len(path)} pts). Press ENTER to drive.")
        else:
            self.get_logger().warn("[NAV] Planning failed.")
            self.state = "IDLE"

    @staticmethod
    def _summary_stats(values):
        vals = [float(v) for v in values if v is not None]
        if not vals:
            return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
        vals = sorted(vals)
        n = len(vals)
        mid = n // 2
        median = vals[mid] if n % 2 == 1 else 0.5 * (vals[mid - 1] + vals[mid])
        return {
            "count": int(n),
            "mean": float(sum(vals) / n),
            "median": float(median),
            "min": float(vals[0]),
            "max": float(vals[-1]),
        }

    def _save_metrics_json(self):
        try:
            os.makedirs(os.path.dirname(self._metrics_path), exist_ok=True)
            per_region = {}
            for plan in self._trajectory_plan_metrics:
                for seg in plan.get("segments", []):
                    ver = seg.get("region_version")
                    if ver is None:
                        continue
                    per_region.setdefault(str(ver), []).append(seg.get("field_generate_ms"))
            payload = {
                "run_started_unix_s": float(self._metrics_run_start_wall_time),
                "run_duration_s": float(_time.monotonic() - self._metrics_run_start_mono),
                "goal_requests": list(self._goal_request_metrics),
                "trajectory_plan_metrics": list(self._trajectory_plan_metrics),
                "summary": {
                    "trajectory_generation_ms": self._summary_stats(
                        [m.get("total_plan_ms") for m in self._trajectory_plan_metrics]
                    ),
                    "trajectory_generation_ms_by_region": {
                        ver: self._summary_stats(vals) for ver, vals in per_region.items()
                    },
                },
            }
            with open(self._metrics_path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
            self.get_logger().info(f"[NAV] Saved line navigation metrics: {self._metrics_path}")
        except Exception as e:
            self.get_logger().warn(f"[NAV] Failed to save line navigation metrics: {e}")

    def destroy_node(self):
        self._save_metrics_json()
        super().destroy_node()

    def _drive_tick(self):
        if self.robot_pos is None or self._nav_path is None:
            self.state = "IDLE"
            return

        # Check goal reached
        dist = float(np.linalg.norm(
            self.robot_pos[:2] - self._goal_xy))
        if dist < float(self.get_parameter("goal_reached_dist").value):
            self.get_logger().info("[NAV] Goal reached!")
            self.cmd_pub.publish(Twist())
            self.state = "IDLE"
            self._nav_path = None
            self._sdf_path = None
            self._sdf_path_time = 0.0
            self.navigator.reset_path()
            return

        # Prefer the SDF-optimized local path when fresh; fall back to global.
        _SDF_VALID_SEC = 2.0
        active = (self._sdf_path
                  if (self._sdf_path is not None
                      and _time.monotonic() - self._sdf_path_time < _SDF_VALID_SEC)
                  else self._nav_path)

        if active is None:
            return

        done = self.navigator.drive_path(active)
        if done or self.navigator.path_complete:
            self.get_logger().info("[NAV] End of path.")
            self.state = "IDLE"
            self._sdf_path = None
            self._sdf_path_time = 0.0
            self.navigator.reset_path()

    # ══════════════════════════════════════════════════════════════════════════
    # Semantic retraining
    # ══════════════════════════════════════════════════════════════════════════

    def _semantic_retrain_tick(self):
        """Drain pending retrain events; run each in a background thread.

        Waits until the hold-off window (set when the first event was queued)
        has elapsed so the camera + 360° LiDAR have had time to sweep up any
        other moved objects — those extra events then get bundled into one
        retrain instead of triggering a second pass.
        """
        with self._retrain_lock:
            if not self._pending_retrain_events:
                return
            if self._retraining:
                return  # previous retrain still running
            if _time.monotonic() < self._retrain_hold_until:
                remaining = self._retrain_hold_until - _time.monotonic()
                self.get_logger().info(
                    f"[SEM] Holding {len(self._pending_retrain_events)} event(s) "
                    f"for {remaining:.1f}s to coalesce further changes.",
                    throttle_duration_sec=5.0,
                )
                return
            by_region: dict[int, list[dict]] = {}
            for ev in self._pending_retrain_events:
                by_region.setdefault(int(ev["region_ver"]), []).append(ev)

            ready: list[dict] = []
            waiting: list[dict] = []
            waiting_msgs: list[str] = []
            for rv, evs in by_region.items():
                region = self.all_regions.get(rv)
                if region is None:
                    waiting.extend(evs)
                    waiting_msgs.append(f"R{rv}: region not loaded")
                    continue
                local_rows = self._candidate_local_row_count(region, evs)
                if local_rows >= RETRAIN_MIN_LOCAL_ROWS:
                    ready.extend(evs)
                else:
                    waiting.extend(evs)
                    waiting_msgs.append(f"R{rv}: {local_rows}/{RETRAIN_MIN_LOCAL_ROWS} local rows")

            if not ready:
                if waiting:
                    self._pending_retrain_events = waiting
                    self._retrain_hold_until = _time.monotonic() + RETRAIN_HOLDOFF_SEC
                    self.get_logger().info(
                        "[SEM] Waiting for more local retrain data: " + ", ".join(waiting_msgs),
                        throttle_duration_sec=5.0,
                    )
                return

            self._pending_retrain_events = waiting
            if waiting:
                self._retrain_hold_until = _time.monotonic() + RETRAIN_HOLDOFF_SEC
            self._retraining = True
            # Stash the state we were in so _do_retrain can decide how to resume.
            # Transition to RETRAINING here (not inside the worker) so the main
            # update loop halts on the very next tick.
            self._state_before_retrain = self.state
            self.state = "RETRAINING"
            self.get_logger().info(
                f"[SEM] Entering RETRAINING (was {self._state_before_retrain}) "
                f"with {len(ready)} ready event(s).")

        self._retrain_thread = threading.Thread(
            target=self._do_retrain, args=(ready,), daemon=True)
        self._retrain_thread.start()

    def _synth_rows_from_painted_map(
        self,
        new_xy: np.ndarray,
        region: "Region",
        n_samples: int = SYNTH_ROWS_PER_EVENT,
        disk_radius: float = SYNTH_DISK_RADIUS_M,
    ) -> torch.Tensor:
        """Synthesise Nx12 training rows around a newly-painted obstacle.

        The painted map is authoritative for where obstacles are *now*. For
        each new obstacle centre we sample free-space x0 in a disk, find the
        nearest painted occupied cell (cp0), pick a local x1 displacement,
        and look up cp1 the same way. The resulting Nx12 rows teach the
        NTField "wall at new_xy" directly, without waiting for the robot's
        LiDAR to sweep the area.
        """
        empty = torch.zeros((0, 12), dtype=torch.float32)
        if (self._map_occ is None or self._map_free is None
                or self._map_origin is None or self._map_resolution is None):
            return empty

        res = float(self._map_resolution)
        ox, oy = float(self._map_origin[0]), float(self._map_origin[1])
        H, W = self._map_occ.shape

        # Window of painted occupied cells around new_xy, padded by the near-wall
        # band so cp-lookups can reach walls at the disk's edge.
        pad = disk_radius + float(region.maximum) + 2.0 * res
        pad_cells = int(math.ceil(pad / res))
        gc = int((float(new_xy[0]) - ox) / res)
        gr = int((float(new_xy[1]) - oy) / res)
        r_lo, r_hi = max(0, gr - pad_cells), min(H, gr + pad_cells + 1)
        c_lo, c_hi = max(0, gc - pad_cells), min(W, gc + pad_cells + 1)
        sub_occ = self._map_occ[r_lo:r_hi, c_lo:c_hi]
        if not sub_occ.any():
            return empty
        rs, cs = np.where(sub_occ)
        occ_world = np.stack([
            ox + (cs + c_lo + 0.5) * res,
            oy + (rs + r_lo + 0.5) * res,
        ], axis=-1).astype(np.float32)
        occ_tree = cKDTree(occ_world)

        rng = np.random.default_rng()
        b = region.bound.detach().cpu().numpy()
        mn, mx = float(region.minimum), float(region.maximum)

        # Sample x0 uniformly in a disk around new_xy (area-uniform via sqrt).
        n_try = n_samples * 4
        theta = rng.uniform(0.0, 2.0 * np.pi, n_try).astype(np.float32)
        r = (disk_radius * np.sqrt(rng.uniform(0.0, 1.0, n_try))).astype(np.float32)
        x0 = np.stack([
            float(new_xy[0]) + r * np.cos(theta),
            float(new_xy[1]) + r * np.sin(theta),
        ], axis=-1).astype(np.float32)

        def _keep_in_bound_and_free(pts: np.ndarray) -> np.ndarray:
            m = ((pts[:, 0] >= b[0, 0]) & (pts[:, 0] <= b[1, 0]) &
                 (pts[:, 1] >= b[0, 1]) & (pts[:, 1] <= b[1, 1]))
            if not m.any():
                return m
            ci = ((pts[:, 0] - ox) / res).astype(np.int32)
            ri = ((pts[:, 1] - oy) / res).astype(np.int32)
            g = (ri >= 0) & (ri < H) & (ci >= 0) & (ci < W)
            m &= g
            if not m.any():
                return m
            # Free-space mask lookup only for the cells we actually keep
            ri_k = np.clip(ri, 0, H - 1)
            ci_k = np.clip(ci, 0, W - 1)
            m &= self._map_free[ri_k, ci_k]
            return m

        keep0 = _keep_in_bound_and_free(x0)
        x0 = x0[keep0]
        if len(x0) == 0:
            return empty

        d0, i0 = occ_tree.query(x0, k=1)
        band0 = (d0 > mn) & (d0 < mx)
        x0 = x0[band0]
        cp0 = occ_world[i0[band0]]
        if len(x0) == 0:
            return empty

        # Sample x1 as a local displacement from x0.
        n = len(x0)
        t = rng.uniform(0.0, 2.0 * np.pi, n).astype(np.float32)
        L = rng.uniform(0.1, disk_radius, n).astype(np.float32)
        x1 = x0 + np.stack([np.cos(t), np.sin(t)], axis=-1) * L[:, None]

        keep1 = _keep_in_bound_and_free(x1)
        x0 = x0[keep1]; cp0 = cp0[keep1]; x1 = x1[keep1]
        if len(x0) == 0:
            return empty

        d1, i1 = occ_tree.query(x1, k=1)
        band1 = (d1 > mn) & (d1 < mx)
        x0 = x0[band1]; cp0 = cp0[band1]; x1 = x1[band1]
        cp1 = occ_world[i1[band1]]
        if len(x0) == 0:
            return empty

        if len(x0) > n_samples:
            idx = rng.choice(len(x0), n_samples, replace=False)
            x0 = x0[idx]; cp0 = cp0[idx]
            x1 = x1[idx]; cp1 = cp1[idx]

        z = np.zeros((len(x0), 1), dtype=np.float32)
        rows = np.hstack([x0, z, x1, z, cp0, z, cp1, z]).astype(np.float32)
        return torch.from_numpy(rows)

    def _do_retrain(self, events: list[dict]):
        """Background worker: train candidate regions, validate, promote, restore state."""
        try:
            promoted_any = False
            by_region: dict[int, list[dict]] = {}
            for ev in events:
                by_region.setdefault(ev["region_ver"], []).append(ev)

            for rv, evs in by_region.items():
                region = self.all_regions.get(rv)
                if region is None:
                    self.get_logger().warn(f"[SEM] R{rv} not loaded, skipping retrain.")
                    continue

                merged = self._candidate_training_rows(region, evs)
                if len(merged) == 0:
                    self.get_logger().warn(f"[SEM] R{rv}: no candidate rows after local overwrite, skipping.")
                    continue
                if len(merged) < RETRAIN_MIN_ROWS:
                    self.get_logger().warn(
                        f"[SEM] R{rv}: only {len(merged)} candidate rows "
                        f"(floor={RETRAIN_MIN_ROWS}) — skipping retrain to avoid "
                        "degrading the field on sparse data.")
                    continue

                candidate_root = self._prepare_candidate_region(rv)
                if candidate_root is None:
                    continue
                candidate = self._load_region_from_root(candidate_root, rv)
                if candidate is None:
                    continue
                candidate.all_raw_frame_data = [merged.detach().cpu()]
                candidate.semantic_objects = copy.deepcopy(getattr(region, "semantic_objects", {}))

                n14 = candidate.calculate_speed_and_normal(merged.detach().cpu())
                if n14 is None or len(n14) == 0:
                    continue
                n14 = n14.to(candidate.device)
                self.get_logger().info(
                    f"[SEM] R{rv}: candidate retraining on {len(merged)} rows "
                    f"for {RETRAIN_EPOCHS} epochs …")
                for ep in range(RETRAIN_EPOCHS):
                    batch = n14
                    if len(batch) > 7000:
                        idx = torch.randperm(len(batch), device=candidate.device)[:7000]
                        batch = batch[idx]
                    candidate.model.train_core(epoch=1, frame_data=batch,
                                               is_one_frame=True)
                candidate._sync_model_refs()
                candidate.save_model()

                if not self._validate_candidate_region(rv, candidate_root):
                    self.get_logger().warn(f"[SEM] R{rv}: candidate failed validation; keeping incumbent.")
                    continue
                if not self._promote_candidate_region(rv, candidate_root, evs):
                    self.get_logger().warn(f"[SEM] R{rv}: candidate promotion failed; keeping incumbent.")
                    continue

                promoted_any = True
                self.get_logger().info(f"[SEM] R{rv}: candidate promoted and saved.")

            # 7. Decide how to resume. If a goal exists, either cancel it (now
            #    in collision) or force a fresh plan from the current pose
            #    against the updated fields. Otherwise restore the prior state.
            prev = self._state_before_retrain or "IDLE"
            if self._goal_xy is not None and promoted_any:
                if self._point_in_collision(self._goal_xy):
                    self.get_logger().warn(
                        f"[SEM] Goal ({self._goal_xy[0]:.2f},{self._goal_xy[1]:.2f}) "
                        "is now in collision after furniture move — cancelling goal.")
                    self._goal_xy = None
                    self._nav_path = None
                    self.state = "IDLE"
                    self.cmd_pub.publish(Twist())
                else:
                    self.get_logger().info(
                        "[SEM] Replanning after retrain from current position.")
                    self.state = "PLANNING"
            else:
                self.state = prev

        except Exception as exc:
            self.get_logger().error(f"[SEM] Retrain error: {exc}")
            # Make sure we don't leave the node stuck in RETRAINING on error.
            if self.state == "RETRAINING":
                self.state = self._state_before_retrain or "IDLE"
        finally:
            self._state_before_retrain = None
            self._retraining = False

    # ══════════════════════════════════════════════════════════════════════════
    # Keyboard thread
    # ══════════════════════════════════════════════════════════════════════════

    def _keyboard_thread(self):
        while rclpy.ok():
            try:
                input()
                if self._keyboard_event is not None:
                    self._keyboard_event.set()
            except EOFError:
                break

    # ══════════════════════════════════════════════════════════════════════════
    # Visualization
    # ══════════════════════════════════════════════════════════════════════════

    def _publish_nav_path(self, path_xy: np.ndarray):
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        for pt in path_xy:
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
        m.scale.x = m.scale.y = m.scale.z = 0.25
        m.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)
        self.goal_marker_pub.publish(m)


# ── Entry point ──────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = NavNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
