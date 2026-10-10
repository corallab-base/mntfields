import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
import rclpy.time
import torch
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree
import signal
import threading
from typing import List, Optional

import numpy as np
import math
import cv2
import tf2_ros
from tf2_ros import TransformBroadcaster
import tf2_geometry_msgs

import sensor_msgs_py.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, LaserScan
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from geometry_msgs.msg import TransformStamped, Twist, Point, PointStamped, PoseStamped, PoseWithCovarianceStamped
from visualization_msgs.msg import Marker, MarkerArray

from mntfields.mntfields_ros.sensing.lidar_adapter import Lidar2DAdapter
from mntfields.mntfields_ros.mapping.map_publisher import MapPublisher 
from mntfields.mntfields_ros.mapping.markers import MarkerPublisher
from mntfields.mntfields_ros.mapping.navigation import Navigator
from mntfields.mntfields_ros.mapping.exploration_graph import ExplorationGraph
from mntfields.mntfields_ros.mapping.semantic_mapper_node import SemanticMapperNode
from mntfields.mntfields_ros.mapping.plotting import MappingPlotter
from mntfields.mntfields_core.mapping.lidar_gridmap import LidarGridmap
from mntfields.mntfields_core.core.region import Region
from mntfields.mntfields_core.core.model_base import Model

from collections import deque
from itertools import combinations
import threading

import os
import json
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg", force=True)
from matplotlib import cm, colors
import matplotlib.pyplot as plt

REGION_RADIUS = 2.5
REGION_EXIT_MARGIN = 0.4                     # kept for CP-based spawn check
MAX_RANGE = 4.0
WARMUP_EPOCHS = 100
VISITED_MARGIN  = 0.2    # m — pad visited-AABB on each side when growing bound
MAX_REGION_SIDE = 25.0   # m — hard cap on any single region side length
MIN_REGION_SIDE = 4.0    # 4.0 m — avoid tiny regions that never get useful frontiers
ROOM_BOUND_MARGIN = 0.5  # m — pad room AABB when creating/updating region bound

#print("NEW CODE")
class DummyRenderer:
    def __init__(self):
        pass

    def render(self, *args, **kwargs):
        return None

class DummyModel:
    def __init__(self):
        self.folder = "/home/mayank/turtlebot3_ws/mntfields_model"
        self.renderer = "cuda:0"   # IMPORTANT: must be string device
        self.h = 0.0

class FakePCD:
    def __init__(self, pts):
        self.points = pts
        self.normals = np.zeros_like(pts)

class GridmapNode(Node):
    def __init__(self):
        super().__init__('mntfields_gridmap')
        self.folder = os.path.join("/home/mayank/turtlebot3_ws/mntfields_model/mntfields_output")
        os.makedirs(self.folder, exist_ok=True)

        # Training frames are consumed FIFO in train_tick(); maxlen is set from
        # parameter below.
        self.train_queue = deque()
        self.train_lock = threading.Lock()
        self.train_running = False
        self._traj_history = []
        self._composite_step = 0
        self._composite_plot_running = False
        self._occupancy_snapshot_running = False
        self._ros_map_save_running = False

        # Tune these
        self.declare_parameter('train_timer_sec', 0.5)      # training loop cadence
        self.declare_parameter('train_samples', 6000)       # n_samples to get_data_from_lidar
        self.declare_parameter('train_epoch', 40)           # epoch passed into region.train_core
        self.declare_parameter('train_queue_maxlen', 64)   # buffered training frames
        self.declare_parameter('region_seed_frames', 5000)  # raw rows to inherit from previous region
        self.declare_parameter('region_max_raw_samples', 30000)  # hard cap per region
        self.declare_parameter('region_raw_memory_cell_size_m', 0.35)
        self.declare_parameter('strict_pair_validity', False)  # legacy sampling: endpoint-valid x1 only
        self.declare_parameter('train_after_region', True)  # toggle
        self.declare_parameter('train_min_robot_motion_m', 0.35)
        self.declare_parameter('enable_region_debug_vis', True)
        self.declare_parameter('enable_composite_plot', True)
        self.declare_parameter('enable_occupancy_snapshot', False)
        # Composite plot viewport controls:
        # keep active-region field centered around robot while always showing the
        # full active-region bound.
        self.declare_parameter('plot_follow_active_region', True)
        self.declare_parameter('plot_active_region_margin_m', 1.0)
        # Door-node matching tolerance:
        # - `door_match_radius_m` > 0 overrides directly (meters).
        # - otherwise auto = `door_match_radius_cells * cell_size`.
        self.declare_parameter('door_match_radius_m', -1.0)
        self.declare_parameter('door_match_radius_cells', 6.0)
        self.declare_parameter('follow_stuck_timeout_sec', 4.0)
        self.declare_parameter('follow_progress_eps', 0.10)
        self.radius = REGION_RADIUS
        train_q_max = int(self.get_parameter('train_queue_maxlen').value)
        self.train_queue = deque(maxlen=max(8, train_q_max))
        self._region_max_raw_samples = int(
            self.get_parameter('region_max_raw_samples').value)
        self._last_train_enqueue_pose = None
        self._last_train_enqueue_region = None

        # Start a separate timer JUST for training
        train_dt = float(self.get_parameter('train_timer_sec').value)
        self.create_timer(train_dt, self.train_tick)
        
        self.declare_parameter('n_cells', 400)
        self.declare_parameter('near_dist', 2.0)
        self.declare_parameter('map_size', 40.0)  # Add this

        n_cells = self.get_parameter('n_cells').value
        near_dist = self.get_parameter('near_dist').value
        map_size = self.get_parameter('map_size').value

        self.map_to_odom = None
        self.origin_initialized = False
        self._origin_set_count = 0  # track how many times odom_cb sets origin (should be exactly 1)
        self.set_parameters([rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)])
        # --- exploration / control params ---
        self.declare_parameter('goal_reached_dist', 0.20)
        self.declare_parameter('yaw_tol', 0.25)          # rad
        self.declare_parameter('kp_yaw', 0.8)
        self.declare_parameter('kp_lin', 0.25)
        self.declare_parameter('max_lin', 0.15)          # m/s
        self.declare_parameter('max_ang', 0.8)           # rad/s
        self.declare_parameter('frontier_min_cells', 2)
        self.declare_parameter('goal_cooldown_sec', 2.0)
        self.declare_parameter('region_exit_switch_margin', 0.35)
        self.declare_parameter('save_satellites_in_periodic_map', False)
        self.declare_parameter('simple_mode', False)
        self.declare_parameter('region_exit_respawn_margin', 0.30)
        self.declare_parameter('region_exit_respawn_hits', 1)
        self.declare_parameter('region_edge_spawn_enable', True)
        self.declare_parameter('region_edge_spawn_margin', 0.20)
        self.declare_parameter('region_edge_spawn_hits', 2)
        self.declare_parameter('region_spawn_cooldown_sec', 6.0)
        self.declare_parameter('allow_region_soft_expand', True)
        self.declare_parameter('soft_expand_max_side_m', 7.0)
        self.declare_parameter('soft_expand_step_m', 0.25)
        self.declare_parameter('backtrack_stack_max', 200)
        self.declare_parameter('backtrack_anchor_spacing_m', 0.8)
        self.declare_parameter('backtrack_revisit_radius_m', 0.6)

        self.path = None        #these 4 lines are for the path holding for stable paths - latest diff
        self.path_last_update = self.get_clock().now()
        self.path_update_period = 2.0   # seconds
        self.path_index = 0
        self.declare_parameter('control_timer_sec', 0.05)
        self.dt = float(self.get_parameter('control_timer_sec').value)
        self._controller_path_done = False

        self.goal_xy = None
        self.last_goal_time = self.get_clock().now()
        self.mode = "BOOTSTRAP"   # BOOTSTRAP -> EXPLORE
        #--diff
        self.declare_parameter('warmup_epochs', WARMUP_EPOCHS)
        self.warmup_epochs_needed = int(self.get_parameter('warmup_epochs').value)
        self.declare_parameter('revisit_warmup_epochs', 4)
        self.revisit_warmup_epochs = int(
            self.get_parameter('revisit_warmup_epochs').value)
        self._warmup_epoch_target = self.warmup_epochs_needed
        self.warmup_done = False
        self.warmup_start_epoch = 0
        self.nav_state = "IDLE"
        self.wait_start_time = None
        self.wait_duration = 3.0
        self.latest_candidate_path = None
        #--
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.semantic_mapper = SemanticMapperNode(self)
        
        self.navigator = Navigator(self)
        self.map_publisher = MapPublisher(self)
        self.markers = MarkerPublisher(self)

        self.robot_pos = None
        self.robot_yaw = 0.0
        self.robot_vx   = 0.0
        self.robot_vy   = 0.0
        self.robot_vyaw = 0.0
        self.latest_cloud = None

        self.all_regions: dict = {}
        self.region_initialized = False
        self.bootstrap_frames = 0
        self.bootstrap_threshold = 10 # frames before region starts
        
        self.adapter = Lidar2DAdapter()

        # Traversability map (1 = free, 0 = blocked)
        self.trav_map = np.ones((n_cells, n_cells), dtype=np.uint8)
        self.gridmap = LidarGridmap(
            trav_map=self.trav_map,
            regions=self.all_regions,#{}
            n_cells=n_cells,
            near_dist=2.0,
            map_size=map_size,           # Add this
            origin=(0.0, 0.0), #(origin_x, origin_y)  # Add this
        )
        self._last_door_match_radius_m = None
        self._apply_door_match_radius_config(force_log=True)
        self.origin_initialized = False
        self._origin_set_count = 0
        self.declare_parameter('occ_height', 0.3)
        self.occ_height = float(self.get_parameter('occ_height').value)
        self.occ_height_tol = 0.01

        self.create_subscription(LaserScan,'/scan',self.lidar_cb,10)
        #self.create_subscription(PoseWithCovarianceStamped,'/amcl_pose',self.pose_cb,5) - will be used for a built map
        self.create_subscription(Odometry, '/odom', self.odom_cb, 10)

        map_qos = QoSProfile(
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
            depth=1,
        )
        self.create_subscription(OccupancyGrid, '/map', self.slam_map_cb, map_qos)
        self._slam_map_received = False

        self.occ_pub = self.create_publisher(OccupancyGrid,'/mntfields/occ_map',5)
        self.frontier_pub = self.create_publisher(OccupancyGrid,'/mntfields/frontier',5)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.goal_marker_pub = self.create_publisher(Marker, '/mntfields/goal_marker', 10)
        self.frontier_marker_pub = self.create_publisher(Marker, '/mntfields/frontier_markers', 10)
        self.doorway_marker_pub = self.create_publisher(MarkerArray, '/mntfields/doorway_markers', 10)
        self.field_path_pub = self.create_publisher(Path, '/field_path', 10)

        self.create_timer(0.2, self.update)
        self.create_timer(self.dt, self.control_tick)

        self.model = DummyModel()
        self.region = None

        self.exploration_graph = ExplorationGraph()
        self._goal_blacklist_radius = 0.5 #was 0.3
        self._visited_goals: List[np.ndarray] = []
        self._goal_kind = "none"  # "frontier" | "backtrack" | "graph_route" | "none"
        self._frontier_anchor_stack: List[dict] = []
        self._recent_backtrack_anchors: deque = deque(maxlen=24)
        self._active_backtrack_anchor = None
        self._outside_bound_hits: int = 0
        self._edge_spawn_hits: int = 0
        self._last_region_spawn_time: float = -1e9
        self._resume_goal_after_warmup: Optional[np.ndarray] = None
        self._resume_goal_kind: str = "none"

        self.train_counter = 0
        self.train_every_n_frames = 1

        # Per-region grid tracking which map cells have been sampled with a
        # REAL LiDAR hit.  Keyed by region.version so it resets automatically
        # when the region changes.  Used to prevent synthetic free-ray
        # endpoints from overwriting real-hit training data.
        self._real_hit_grid_version: int = -1
        self._real_hit_grid: np.ndarray = np.zeros((1, 1), dtype=bool)

        self._map_update_counter = 0 #claude difff
        self._path_gen_running   = False
        self._path_refresh_ticks = 0
        self._path_total_dist    = None   # arc-length of path when first accepted in FOLLOWING
        self._midpoint_replanned = False  # fires once per goal
        self._best_goal_dist     = float("inf")
        self._last_progress_time = 0.0
        self._last_path_progress_s = 0.0
        self._stabilizing_retries: int = 0
        self._graph_route_chain: List[np.ndarray] = []
        self._graph_route_idx:   int = 0
        self._plot_world_center: Optional[np.ndarray] = None

        # ── Region switch hysteresis ──────────────────────────────────────
        self._last_switch_from_version: int = -1
        self._last_switch_time: float = 0.0
        self._switch_candidate_version: int = -1
        self._switch_candidate_hits: int = 0

        # ── Goal-dwell training burst ─────────────────────────────────────
        self.declare_parameter('dwell_frames_needed', 6)    # frames to collect at goal
        self.declare_parameter('dwell_epochs',       20)    # epochs to train before resuming
        self.dwell_active       = False
        self.dwell_frames       = []                        # accumulated during dwell
        self.dwell_frames_count = 0
        self.dwell_epoch_start  = 0
        self.dwell_phase        = "COLLECT"  # "COLLECT" | "TRAIN" | "DONE"
        self.simple_mode = bool(self.get_parameter('simple_mode').value)
        self.plotter = MappingPlotter(self)

    def _launch_background_task(self, flag_attr: str, fn):
        if getattr(self, flag_attr, False):
            return
        setattr(self, flag_attr, True)

        def _runner():
            try:
                fn()
            except Exception as e:
                self.get_logger().warn(str(e))
            finally:
                setattr(self, flag_attr, False)

        threading.Thread(target=_runner, daemon=True).start()

    def lidar_cb(self, msg):
        self.latest_cloud = msg
        if not hasattr(self, "_last_lidar_log"):
            self._last_lidar_log = self.get_clock().now()
            #self.get_logger().info("Received first LiDAR scan")
            return
        now = self.get_clock().now()
        if (now - self._last_lidar_log).nanoseconds > 2e9:  # 2 seconds
            #self.get_logger().info("Receiving LiDAR scans...")
            self._last_lidar_log = now
    
    def odom_cb(self, msg):
        # Velocities only — pose/yaw now come from TF (map → base_footprint).
        self.robot_vx   = float(msg.twist.twist.linear.x)
        self.robot_vy   = float(msg.twist.twist.linear.y)
        self.robot_vyaw = float(msg.twist.twist.angular.z)

    def slam_map_cb(self, msg):
        """Receive SLAM Toolbox /map, set grid origin once and ingest occupancy each tick."""
        if not self._slam_map_received:
            info = msg.info
            cx = info.origin.position.x + info.width  * info.resolution / 2.0
            cy = info.origin.position.y + info.height * info.resolution / 2.0
            self.gridmap.origin = np.array([cx, cy], dtype=np.float32)
            self.origin_initialized = True
            self._slam_map_received = True
            self.get_logger().info(
                f"[SLAM MAP] Grid origin set to ({cx:.2f},{cy:.2f}) from SLAM map")
        self.gridmap.ingest_slam_map(msg)

    def _update_robot_pose_from_tf(self):
        """Update robot_pos and robot_yaw from TF map → base_footprint."""
        try:
            t = self.tf_buffer.lookup_transform(
                'map', 'base_footprint', rclpy.time.Time(),
                timeout=Duration(seconds=0.05))
            p = t.transform.translation
            q = t.transform.rotation
            self.robot_pos = np.array([p.x, p.y, p.z], dtype=np.float32)
            siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            self.robot_yaw = float(np.arctan2(siny_cosp, cosy_cosp))
        except Exception:
            pass   # keep last known pose; SLAM TF not yet available

    def _apply_region_training_config(self, region: 'Region'):
        if region is None:
            return
        max_raw = int(self.get_parameter('region_max_raw_samples').value)
        region.max_raw_samples = max_raw
        region.raw_memory_cell_size_m = float(
            self.get_parameter('region_raw_memory_cell_size_m').value
        )

    def init_region(self, robot_pose):
        ''' Initializes a region around the robot and registers it. '''
        # Grab seed data from the old region BEFORE unloading it.
        # This seeds the new region with walls already mapped near the boundary
        # so the new field doesn't treat them as open space.
        seed_rows = None
        n_seed = int(self.get_parameter('region_seed_frames').value)
        if n_seed > 0 and self.region is not None:
            old_data = getattr(self.region, 'all_raw_frame_data', [])
            if old_data:
                try:
                    merged = torch.cat(old_data, dim=0)
                    if len(merged) > 0:
                        # Bias seed towards rows whose x0 falls INSIDE the new
                        # region's bound — these capture boundary walls the new
                        # field must know about.  Fill remainder randomly.
                        new_cx, new_cy = float(robot_pose[0]), float(robot_pose[1])
                        new_r = float(self.radius)
                        x0_world = merged[:, :2].cpu().numpy()
                        in_new = (
                            (x0_world[:, 0] >= new_cx - new_r) &
                            (x0_world[:, 0] <= new_cx + new_r) &
                            (x0_world[:, 1] >= new_cy - new_r) &
                            (x0_world[:, 1] <= new_cy + new_r)
                        )
                        in_idx  = np.where(in_new)[0]
                        out_idx = np.where(~in_new)[0]
                        n_in  = min(n_seed, len(in_idx))
                        n_out = min(max(0, n_seed - n_in), len(out_idx))
                        chosen = []
                        if n_in > 0:
                            perm = torch.randperm(len(in_idx))[:n_in]
                            chosen.append(merged[torch.from_numpy(in_idx[perm.numpy()])])
                        if n_out > 0:
                            perm2 = torch.randperm(len(out_idx))[:n_out]
                            chosen.append(merged[torch.from_numpy(out_idx[perm2.numpy()])])
                        seed_rows = torch.cat(chosen, dim=0).clone() if chosen else None
                except Exception:
                    seed_rows = None

        # Save old region weights before overwriting
        if self.region is not None:
            self._unload_current_region()
        with self.train_lock:
            self.train_queue.clear()
        self.train_running = False

        center_x = robot_pose[0]
        center_y = robot_pose[1]
        radius = self.radius

        # Fix world center for composite plot on first region init
        if self._plot_world_center is None:
            self._plot_world_center = np.array([center_x, center_y], dtype=np.float32)

        bound = torch.tensor([
            [center_x - radius, center_y - radius, -0.1],
            [center_x + radius, center_y + radius,  0.1],
        ], dtype=torch.float32)

        new_region = Region(
            model=self.model,
            bound=bound,
            init_network=True,
            eval=False,)
        new_region.gridmap = self.gridmap
        self._apply_region_training_config(new_region)
        new_region.enable_debug_vis = bool(
            self.get_parameter('enable_region_debug_vis').value)
        self.all_regions[new_region.version] = new_region

        new_region.all_lidar = []
        new_region.spawn_center = np.array([center_x, center_y], dtype=np.float32)

        # Seed new region with sampled rows from the old region so that walls
        # already mapped near the boundary are not treated as open space.
        # The new region's train_core will filter these to its own bound on the
        # first training step via _filter_raw_rows_to_bound.
        if seed_rows is not None and len(seed_rows) > 0:
            new_region.all_raw_frame_data = [seed_rows]
            self.get_logger().info(
                f"[REGION {new_region.version} SEED] inherited {len(seed_rows)} raw rows "
                f"from previous region")

        # Record crossing edge between old and new region in the exploration graph
        new_center_xy = np.array([center_x, center_y], dtype=np.float32)
        if self.region is not None:
            old_ver = self.region.version
            old_b = self.region.bound.detach().cpu().numpy()
            old_center_xy = np.array(
                [(old_b[0, 0] + old_b[1, 0]) / 2.0,
                 (old_b[0, 1] + old_b[1, 1]) / 2.0], dtype=np.float32)
            cp_xy = np.asarray(robot_pose[:2], dtype=np.float32)
            dist = float(np.linalg.norm(old_center_xy - new_center_xy))
            self.exploration_graph.add_cp_edge(old_ver, new_region.version, cp_xy, weight=dist)
        self.exploration_graph.ensure_region(new_region.version, new_center_xy)

        self.region = new_region
        self.region_initialized = True
        self._last_train_enqueue_pose = None
        self._last_train_enqueue_region = None
        self.region.last_robot_pose = np.array([center_x, center_y, 0.0], dtype=np.float32)
        self.get_logger().info(
            f"[REGION {new_region.version} INIT] "
            f"center=({center_x:.2f}, {center_y:.2f}) "
            f"bound=[({bound[0,0]:.2f},{bound[0,1]:.2f}) → "
            f"({bound[1,0]:.2f},{bound[1,1]:.2f})]")
        self._path_total_dist    = None
        self._midpoint_replanned = False
        # Reset real-hit grid for the new region
        n = self.gridmap.n_cells
        self._real_hit_grid         = np.zeros((n, n), dtype=bool)
        self._real_hit_grid_version = new_region.version
        self._enter_warmup(
            new_region,
            self.warmup_epochs_needed,
            f"[FLOW] Region {new_region.version} → entering WARMUP",
        )

    def _enter_warmup(self, region: 'Region', epochs_needed: int, reason: str):
        self.mode = "WARMUP"
        self.warmup_start_epoch = int(getattr(region, 'epoch', 0))
        self._warmup_epoch_target = max(1, int(epochs_needed))
        self.warmup_done = False
        self.goal_xy = None
        self.path = None
        self.nav_state = "IDLE"
        self.latest_candidate_path = None
        self._path_gen_running = False
        self.navigator.stop_robot()
        self.get_logger().info(
            f"{reason} [warmup_epochs={self._warmup_epoch_target}, "
            f"start_epoch={self.warmup_start_epoch}]"
        )

    def _apply_door_match_radius_config(self, force_log: bool = False):
        gm = self.gridmap
        radius_m = float(self.get_parameter('door_match_radius_m').value)
        radius_cells = float(self.get_parameter('door_match_radius_cells').value)
        gm.set_door_match_radius(radius_m=radius_m, radius_cells=radius_cells)

        current = float(gm.door_match_radius_m)
        if force_log or self._last_door_match_radius_m is None or abs(current - self._last_door_match_radius_m) > 1e-6:
            mode = "meters override" if radius_m > 0.0 else "cell-scaled auto"
            self.get_logger().info(
                f"[DOOR MATCH] radius={current:.3f} m ({mode}, cells={float(gm.door_match_radius_cells):.2f}, cell={gm.cell_size:.3f} m)"
            )
            self._last_door_match_radius_m = current

    def feed_region(self, world_pts, robot_pose, non_hit_dirs=None):
        """
        world_pts: real LiDAR hit points in world frame (Nx3), NO synthetic endpoints.
        non_hit_dirs: (K,2) float32 unit vectors of non-hitting ray directions (optional).
        get_data_from_lidar accumulates these into the global obstacle cloud and
        returns Nx12 (x0, x1, cp0, cp1) world-coord training pairs.
        train_core normalizes, accumulates history, and runs one backward pass.
        """
        if self.region is None or getattr(self.region, 'function', None) is None:
            return
        if len(world_pts) < 50:
            self.get_logger().warn(f"Too few points: {len(world_pts)}")
            return
        try:
            n_samples = int(self.get_parameter('train_samples').value)
            epoch = int(self.get_parameter('train_epoch').value)

            result = self.region.get_data_from_lidar(
                world_pts,
                robot_pose,
                device='cuda:0',
                n_samples=n_samples,
                non_hit_dirs=non_hit_dirs,
            )
            if result is None:
                self.get_logger().warn("get_data_from_lidar returned None")
                return
            points, closest = result
            if points.shape[0] == 0:
                self.get_logger().warn("get_data_from_lidar returned 0 points")
                return

            # Nx12: (x0, x1, cp0, cp1) all in world coords
            raw = torch.cat([points, closest], dim=1)

            if not torch.isfinite(raw).all():
                bad = (~torch.isfinite(raw)).sum().item()
                self.get_logger().error(f"[FEED] raw has non-finite values: {bad}")
                return

            self.region.last_robot_pose = np.array(
                [robot_pose[0], robot_pose[1], 0.0], dtype=np.float32)
            self.region.strict_pair_validity = bool(
                self.get_parameter('strict_pair_validity').value
            )
            self.get_logger().info(
                f"[FEED] calling train_core epoch={epoch} "
                f"region_epoch_before={int(getattr(self.region, 'epoch', -1))}"
            )
            self.region.train_core(epoch=epoch, raw_frame_data=raw)
            self.get_logger().info(
                f"[FEED] train_core returned epoch_after={int(getattr(self.region, 'epoch', -1))}"
            )

            # Cross-region data sharing: any neighbouring region whose bound
            # contains LiDAR wall points accumulates those points into its buffer
            # WITHOUT a backward pass.  This fixes the "gap at junction" problem
            # where boundary walls are only seen while the robot is in a different
            # region and therefore never appear in the neighbour's training data.
            world_pts_arr = world_pts if isinstance(world_pts, np.ndarray) else np.asarray(world_pts)
            for sat_ver, sat in list(self.all_regions.items()):
                if sat_ver == self.region.version:
                    continue
                if sat.bound is None:
                    continue
                try:
                    b = sat.bound.detach().cpu().numpy()
                    in_b = ((world_pts_arr[:, 0] >= b[0, 0]) & (world_pts_arr[:, 0] <= b[1, 0]) &
                            (world_pts_arr[:, 1] >= b[0, 1]) & (world_pts_arr[:, 1] <= b[1, 1]))
                    if in_b.sum() < 20:
                        continue
                    sat_pts_world = world_pts_arr[in_b]
                    sat_result = sat.get_data_from_lidar(
                        sat_pts_world, robot_pose, device='cuda:0',
                        n_samples=min(1000, n_samples // 4))
                    if sat_result is not None:
                        sat_p, sat_c = sat_result
                        if sat_p.shape[0] > 0:
                            sat_raw = torch.cat([sat_p, sat_c], dim=1)
                            sat.accumulate_raw_data(sat_raw)
                except Exception:
                    pass  # best-effort; never block current region training
        except Exception as e:
            self.get_logger().error(f"feed_region error: {e}")
            import traceback
            traceback.print_exc()

    def _current_region_frontier_count(self) -> int:
        if self.region is None:
            return 0
        rec = self.exploration_graph.records.get(self.region.version)
        if rec is None:
            return 0
        return int(len(rec.saved_frontiers))

    def _push_frontier_anchor(self, robot_xy: np.ndarray):
        """
        Save a backtracking anchor when the current region has live frontiers.
        Anchors are deduplicated by spacing and kept as a bounded LIFO stack.
        """
        if self.region is None or robot_xy is None:
            return
        if self._current_region_frontier_count() <= 0:
            return

        spacing = float(self.get_parameter('backtrack_anchor_spacing_m').value)
        max_sz = max(8, int(self.get_parameter('backtrack_stack_max').value))
        pos = np.asarray(robot_xy[:2], dtype=np.float32).copy()

        if self._frontier_anchor_stack:
            last = self._frontier_anchor_stack[-1]
            if int(last.get("region_version", -1)) == int(self.region.version):
                last_xy = np.asarray(last.get("xy", pos), dtype=np.float32)
                if float(np.linalg.norm(last_xy - pos)) < spacing:
                    return

        self._frontier_anchor_stack.append({
            "xy": pos,
            "region_version": int(self.region.version),
        })
        if len(self._frontier_anchor_stack) > max_sz:
            drop = len(self._frontier_anchor_stack) - max_sz
            del self._frontier_anchor_stack[:drop]

    def _is_recent_backtrack_anchor(self, xy: np.ndarray) -> bool:
        rad = float(self.get_parameter('backtrack_revisit_radius_m').value)
        if not self._recent_backtrack_anchors:
            return False
        pos = np.asarray(xy[:2], dtype=np.float32)
        for prev in self._recent_backtrack_anchors:
            if float(np.linalg.norm(pos - prev[:2])) <= rad:
                return True
        return False

    def _pick_backtrack_goal(self, robot_xy: np.ndarray):
        """
        Pop frontier anchors in LIFO order until a reachable backtrack target
        is found. This replaces global-remote-frontier selection.
        """
        while self._frontier_anchor_stack:
            anchor = self._frontier_anchor_stack.pop()
            anchor_xy_obj = anchor.get("xy", None)
            if anchor_xy_obj is None:
                continue
            anchor_xy = np.asarray(anchor_xy_obj, dtype=np.float32)
            if anchor_xy.shape != (2,):
                continue
            if float(np.linalg.norm(anchor_xy - robot_xy)) < 0.35:
                continue
            if self._is_recent_backtrack_anchor(anchor_xy):
                continue
            self._recent_backtrack_anchors.append(anchor_xy.copy())
            self._active_backtrack_anchor = anchor
            self.get_logger().info(
                f"[BACKTRACK] heading to anchor in R{anchor.get('region_version', -1)} "
                f"at ({anchor_xy[0]:.2f}, {anchor_xy[1]:.2f})"
            )
            return anchor_xy
        self._active_backtrack_anchor = None
        return None

    def pick_goal(self):
        # 1) Always prefer the nearest currently reachable live frontier.
        g = self.pick_reachable_frontier()
        if g is not None:
            self._goal_kind = "frontier"
            self._active_backtrack_anchor = None
            self._graph_route_chain = []
            return g

        robot_xy = self.robot_pos[:2]

        # 2) No local frontier: try graph route to a remote region's frontier.
        if self.region is not None:
            result = self.exploration_graph.best_remote_frontier(
                self.region.version, robot_xy)
            if result is not None:
                remote_ver, frontier_xy, waypoints = result
                self._graph_route_chain = list(waypoints) + [frontier_xy]
                self._graph_route_idx   = 0
                self._goal_kind         = "graph_route"
                self._active_backtrack_anchor = None
                next_wp = self._graph_route_chain[0]
                self.get_logger().info(
                    f"[GRAPH ROUTE] {len(self._graph_route_chain)}-step chain → R{remote_ver}, "
                    f"first wp=({next_wp[0]:.2f},{next_wp[1]:.2f})"
                )
                return np.asarray(next_wp, dtype=np.float32)

        # 3) Backtrack to previous frontier anchors until a place with live
        # frontiers is reached.
        bg = self._pick_backtrack_goal(robot_xy)
        if bg is not None:
            self._goal_kind = "backtrack"
            self._graph_route_chain = []
            return np.asarray(bg, dtype=np.float32)

        self._goal_kind = "none"
        self._graph_route_chain = []
        print("[NO REACHABLE FRONTIER OR BACKTRACK ANCHOR]")
        return None

    def publish_field_path(self, path_xy, frame_id='odom'):
        """
        path_xy: list[np.ndarray] or np.ndarray of shape (N,2)
        """
        if path_xy is None:
            return
        if isinstance(path_xy, list):
            pts = np.asarray(path_xy, dtype=np.float32)
        else:
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
            ps.pose.position.z = 0.0
            ps.pose.orientation.w = 1.0
            poses.append(ps)
        msg.poses = poses
        self.field_path_pub.publish(msg)

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
            a = max(0, i-k)
            b = min(len(pts), i+k+1)
            sm[i] = pts[a:b].mean(axis=0)
        sm[0] = pts[0]
        sm[-1] = pts[-1]
        return sm 

    def is_path_significantly_different(self, old_path, new_path, threshold=0.25):
        old_path = np.asarray(old_path)
        new_path = np.asarray(new_path)
        if len(old_path) < 2 or len(new_path) < 2:
            return True
        # compare up to min length
        n = min(len(old_path), len(new_path))
        # compare corresponding waypoints
        diffs = np.linalg.norm(old_path[:n] - new_path[:n], axis=1)
        max_diff = np.max(diffs)
        print("Max waypoint diff:", max_diff)
        return max_diff > threshold

    def train_tick(self):
        if self.region is None:
            return
        # STABILIZING is a brief critical window right after a new path is
        # computed — skip training here to avoid GPU contention while the
        # path-gen thread is also running.  FOLLOWING is long-duration and
        # training is beneficial; yielding the inference_lock between epochs
        # means path-gen can always interleave.
        if self.mode == "EXPLORE" and self.nav_state == "STABILIZING":
            return
        if self.train_running:
            _tnow = self.get_clock().now().nanoseconds * 1e-9
            if _tnow - getattr(self, '_train_stuck_log_t', 0.0) >= 5.0:
                self._train_stuck_log_t = _tnow
                self.get_logger().warn(
                    f"[TRAIN TICK] train_running=True — previous job still active "
                    f"queue={len(self.train_queue)} "
                    f"region_epoch={int(getattr(self.region, 'epoch', -1))}"
                )
            return
        with self.train_lock:
            if len(self.train_queue) == 0:
                return
            self.train_running = True
            _item = self.train_queue.popleft()
            world_pts, robot_pose = _item[0], _item[1]
            non_hit_dirs_q = _item[2] if len(_item) > 2 else None
            self.get_logger().info(
                f"[TRAIN TICK] dequeue queue_after={len(self.train_queue)} "
                f"pts={len(world_pts)} pose=({float(robot_pose[0]):.2f},{float(robot_pose[1]):.2f}) "
                f"non_hit_dirs={0 if non_hit_dirs_q is None else len(non_hit_dirs_q)}"
            )

        def _do_train():
            _epoch_before = int(getattr(self.region, 'epoch', -1)) if self.region else -1
            self.get_logger().info(f"[TRAIN START] epoch_before={_epoch_before}")
            try:
                self.feed_region(world_pts, robot_pose, non_hit_dirs=non_hit_dirs_q)
            except Exception as e:
                self.get_logger().error(f"[TRAIN] failed: {e}")
                import traceback
                traceback.print_exc()
            finally:
                _epoch_after = int(getattr(self.region, 'epoch', -1)) if self.region else -1
                self.get_logger().info(f"[TRAIN DONE] epoch_after={_epoch_after}")
                self.train_running = False

        t = threading.Thread(target=_do_train, daemon=True)
        t.start()

    # maybe_expand_region removed — region sizing handled by fit_region_to_lidar()

    def pick_reachable_frontier(self, max_candidates=300):
        frontier = self.gridmap.frontier
        iy, ix = np.where(frontier > 0)
        if self.gridmap.has_raw_slam_frontiers():
            cand = np.asarray(self.gridmap.slam_frontier_points, dtype=np.float32)
            ij = None
        else:
            if len(ix) == 0:
                return None
            ij = np.stack([iy, ix], axis=1)
            cell   = self.gridmap.cell_size
            origin = self.gridmap.origin
            half   = self.gridmap.map_size / 2.0
            fx = (origin[0] - half) + (ix + 0.5) * cell
            fy = (origin[1] - half) + (iy + 0.5) * cell
            cand = np.stack([fx, fy], axis=1)
        robot = self.robot_pos[:2]
        d = np.linalg.norm(cand - robot[None, :], axis=1)
        # Do not select micro-frontiers that are effectively under the robot.
        # With SLAM-map semantics, tiny free/unknown notches can appear around
        # the robot footprint and immediately satisfy the frontier test.
        MIN_FRONTIER_STANDOFF_M = max(0.45, float(self.get_parameter('goal_reached_dist').value) * 2.0)
        far_enough = d >= MIN_FRONTIER_STANDOFF_M
        if far_enough.sum() == 0:
            return None
        cand = cand[far_enough]
        d    = d[far_enough]
        if ij is not None:
            ij = ij[far_enough]
        # Cutoff covers full region diameter + margin (REGION_RADIUS * 4 = 10m).
        # The old 2.0m limit incorrectly discarded frontiers at the far side of
        # the current region (radius 2.5m, diameter 5m), causing "fully explored"
        # even when 3+ live frontiers existed in R(cur).
        valid = d < (REGION_RADIUS * 4)
        if valid.sum() == 0:
            return None
        cand = cand[valid]
        d    = d[valid]
        if ij is not None:
            ij = ij[valid]
        # Drop frontier candidates near already reached goals so the planner
        # does not oscillate on tiny remnants left after local frontier clearing.
        if self._visited_goals:
            visited_goals = np.asarray(self._visited_goals, dtype=np.float32)
            if visited_goals.ndim == 2 and visited_goals.shape[1] == 2:
                min_goal_dist = np.min(
                    np.linalg.norm(cand[:, None, :] - visited_goals[None, :, :], axis=2),
                    axis=1,
                )
                keep = min_goal_dist >= float(self._goal_blacklist_radius)
                if keep.sum() == 0:
                    return None
                cand = cand[keep]
                d    = d[keep]
                if ij is not None:
                    ij = ij[keep]
        # Keep frontier candidates connected through known free SLAM cells when
        # available. This makes frontier selection follow `/map` semantics
        # directly instead of the custom visited-boundary heuristic.
        try:
            _gm  = self.gridmap
            _ox  = _gm.origin[0] - _gm.map_size / 2.0
            _oy  = _gm.origin[1] - _gm.map_size / 2.0
            rix = int((robot[0] - _ox) / _gm.cell_size)
            riy = int((robot[1] - _oy) / _gm.cell_size)
            if _gm.has_slam_semantics() and 0 <= rix < _gm.n_cells and 0 <= riy < _gm.n_cells:
                free_known = (_gm.slam_known & _gm.slam_free & (_gm.trav_map == 1)).astype(np.uint8)
                if free_known[riy, rix]:
                    _, labels = cv2.connectedComponents(free_known, connectivity=4)
                    robot_label = int(labels[riy, rix])
                    if robot_label > 0:
                        _same_cc = np.zeros(len(cand), dtype=bool)
                        for _ki, _c in enumerate(cand):
                            _gx = int((_c[0] - _ox) / _gm.cell_size)
                            _gy = int((_c[1] - _oy) / _gm.cell_size)
                            if 0 <= _gx < _gm.n_cells and 0 <= _gy < _gm.n_cells:
                                _same_cc[_ki] = int(labels[_gy, _gx]) == robot_label
                        if _same_cc.any():
                            cand = cand[_same_cc]
                            d    = d[_same_cc]
                            if ij is not None:
                                ij = ij[_same_cc]
        except Exception:
            pass
        if self.region is not None:
            # Shrink the effective region bound inward by VISITED_MARGIN so that
            # selected frontiers are comfortably inside the trained field, not
            # right at the 2.5m non-hitting-ray boundary where training data
            # is sparse.  This guarantees a safe field-planned path to the goal.
            _b = self.region.bound.detach().cpu().numpy()
            _margin = VISITED_MARGIN  # 0.2 m inward
            in_bound = np.array([
                (c[0] >= _b[0, 0] + _margin) and (c[0] <= _b[1, 0] - _margin) and
                (c[1] >= _b[0, 1] + _margin) and (c[1] <= _b[1, 1] - _margin)
                for c in cand])
            if in_bound.sum() > 0:
                cand = cand[in_bound]
                d    = d[in_bound]
                if ij is not None:
                    ij = ij[in_bound]
            # else: keep full candidate set — CC + adjacency filters ensure reachability

        # Filter by visited-cell AABB: frontiers that are deep into LiDAR-mapped
        # but physically unvisited space produce random field gradients and cause
        # path gen to fail every time.  Allow PAD_M beyond the furthest visited
        # cell so genuine boundary frontiers are kept.
        _vis_cells = np.argwhere(self.gridmap.visited > 0)
        if _vis_cells.size > 0 and len(cand) > 0:
            _cell = self.gridmap.cell_size
            _ox   = self.gridmap.origin[0] - self.gridmap.map_size / 2.0
            _oy   = self.gridmap.origin[1] - self.gridmap.map_size / 2.0
            _PAD  = 1.0  # accept frontiers up to 1 m beyond the visited boundary
            _vx   = _vis_cells[:, 1].astype(float) * _cell + _ox
            _vy   = _vis_cells[:, 0].astype(float) * _cell + _oy
            _in_vis = np.array([
                (_vx.min() - _PAD <= c[0] <= _vx.max() + _PAD) and
                (_vy.min() - _PAD <= c[1] <= _vy.max() + _PAD)
                for c in cand])
            if _in_vis.any():
                cand = cand[_in_vis]
                d    = d[_in_vis]
                if ij is not None:
                    ij = ij[_in_vis]

        if len(cand) == 0:
            return None

        # ── Frontier clustering ────────────────────────────────────────────
        # Group nearby frontier cells using connected components on a dilated
        # frontier bitmap, then score each cluster by expected coverage gain
        # divided by travel cost.  The representative returned is always an
        # ACTUAL frontier cell (nearest frontier to the cluster centroid) so
        # it never lands in untrained / unknown space.
        #
        # Score = cluster_size^0.5 / (distance_to_rep + 1.0)
        # sqrt(size) prevents very large clusters from completely dominating.
        # ──────────────────────────────────────────────────────────────────
        # Smaller merge radius prevents over-merging in large rooms where
        # a single huge near-cluster would dominate over all distant clusters.
        CLUSTER_RADIUS_M = 0.45
        try:
            _gm  = self.gridmap
            _ox  = _gm.origin[0] - _gm.map_size / 2.0
            _oy  = _gm.origin[1] - _gm.map_size / 2.0
            # Build a bitmap of our candidate frontier cells only
            _fmap = np.zeros((_gm.n_cells, _gm.n_cells), dtype=np.uint8)
            _cix  = np.clip(((cand[:, 0] - _ox) / _gm.cell_size).astype(int),
                            0, _gm.n_cells - 1)
            _ciy  = np.clip(((cand[:, 1] - _oy) / _gm.cell_size).astype(int),
                            0, _gm.n_cells - 1)
            _fmap[_ciy, _cix] = 1
            # Dilate to merge frontiers within CLUSTER_RADIUS_M
            _r_cells = max(1, int(CLUSTER_RADIUS_M / _gm.cell_size))
            _kern = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * _r_cells + 1, 2 * _r_cells + 1))
            _dilated = cv2.dilate(_fmap, _kern)
            _n_cc, _cc_labels = cv2.connectedComponents(_dilated, connectivity=8)
            # Map each candidate to its cluster label
            _cand_labels = _cc_labels[_ciy, _cix]   # shape (N,)
            # Pick best cluster.
            # Score = sqrt(size) / log(dist + e)
            # Log distance penalty: distant clusters compete much better than with
            # linear denominator. At dist=5m the log denominator is only 2.0×
            # the dist=1m value (vs 3.3× with linear). This lets large far clusters
            # beat small near ones — ensuring the robot covers the full room rather
            # than looping around nearby frontiers indefinitely.
            best_cand_idx = None
            best_score    = -1.0
            for _lbl in np.unique(_cand_labels):
                if _lbl == 0:
                    continue
                _members_mask = _cand_labels == _lbl
                _members      = cand[_members_mask]
                _size         = float(_members_mask.sum())
                _centroid     = _members.mean(axis=0)
                # Representative = frontier member closest to centroid
                _dist_to_cen  = np.linalg.norm(_members - _centroid[None, :], axis=1)
                _rep_local    = int(np.argmin(_dist_to_cen))
                _rep          = _members[_rep_local]
                _d_rep        = float(np.linalg.norm(_rep - robot))
                _score        = (_size ** 0.5) / math.log(_d_rep + math.e)
                if _score > best_score:
                    best_score    = _score
                    best_cand_idx = np.where(_members_mask)[0][_rep_local]
            if best_cand_idx is not None:
                print(f"[FRONTIER CLUSTER] {_n_cc-1} clusters, "
                      f"best score={best_score:.2f}, "
                      f"goal={cand[best_cand_idx]}")
                return cand[best_cand_idx]
        except Exception:
            pass

        # Fallback: no field yet or clustering failed → nearest candidate
        if len(cand) > max_candidates:
            cand = cand[np.random.choice(len(cand), max_candidates, replace=False)]
            d    = np.linalg.norm(cand - robot[None, :], axis=1)
        if self.region is None or getattr(self.region, "function", None) is None:
            return cand[np.argmin(d)]
        x0  = np.tile(np.array([robot[0], robot[1], 0.0]), (len(cand), 1))
        x1  = np.concatenate([cand, np.zeros((len(cand), 1))], axis=1)
        xp  = np.concatenate([x0, x1], axis=1)
        xp_t = torch.tensor(xp, dtype=torch.float32)
        xp_n = self.region.normalize_tensor(xp_t).to(self.region.device)
        with torch.no_grad():
            tt = self.region.function.TravelTimes(xp_n).cpu().numpy().reshape(-1)
        ok = np.isfinite(tt)
        if ok.sum() == 0:
            return cand[np.argmin(d)]
        print("frontier from pick_reachable_frontier")
        return cand[ok][np.argmin(tt[ok])]

    def maybe_create_new_region(self):
        """
        Spawn the first region once minimal local coverage exists.
        """
        if self.region is not None:
            return
        if self.robot_pos is None:
            return
        # Spawn the first region once minimal visited coverage is present.
        robot_xy = self.robot_pos[:2]
        gm   = self.gridmap
        cell = gm.cell_size
        ox   = gm.origin[0] - gm.map_size / 2.0
        oy   = gm.origin[1] - gm.map_size / 2.0
        max_region_side = max(
            MIN_REGION_SIDE,
            float(self.get_parameter('soft_expand_max_side_m').value),
        )
        max_step_clip = max(
            0.05,
            float(self.get_parameter('soft_expand_step_m').value),
        )
        r = self.radius
        ix0 = max(0, int((robot_xy[0] - r - ox) / cell))
        ix1 = min(gm.n_cells - 1, int((robot_xy[0] + r - ox) / cell))
        iy0 = max(0, int((robot_xy[1] - r - oy) / cell))
        iy1 = min(gm.n_cells - 1, int((robot_xy[1] + r - oy) / cell))
        area = (ix1 - ix0) * (iy1 - iy0)
        if area <= 0:
            return
        coverage = float(gm.visited[iy0:iy1, ix0:ix1].sum()) / float(area)
        if coverage >= 0.05:
            self.init_region(self.robot_pos)

    def fit_region_to_lidar(self):
        """
        Fit the current region's bound to the discovered footprint.
        Primary: LiDAR footprint AABB (last 12 scans) + VISITED_MARGIN.
        Fallback: discovery_map AABB clipped to current bound + 0.8m pad.
        """
        if self.region is None:
            return
        if self.robot_pos is not None:
            # Do not keep stretching the active region once the robot is already
            # near an exit/switch boundary. Late bound growth re-normalizes the
            # whole field against a larger box and can make the last debug plots
            # before a region switch look corrupted.
            bcur = self.region.bound.detach().cpu().numpy()
            robot_xy = np.asarray(self.robot_pos[:2], dtype=np.float32)
            switch_margin = float(self.get_parameter('region_exit_switch_margin').value)
            edge_gap = min(
                float(robot_xy[0] - bcur[0, 0]),
                float(bcur[1, 0] - robot_xy[0]),
                float(robot_xy[1] - bcur[0, 1]),
                float(bcur[1, 1] - robot_xy[1]),
            )
            if edge_gap <= max(0.0, switch_margin):
                return

        gm   = self.gridmap
        cell = gm.cell_size
        ox   = gm.origin[0] - gm.map_size / 2.0
        oy   = gm.origin[1] - gm.map_size / 2.0
        max_region_side = max(
            MIN_REGION_SIDE,
            float(self.get_parameter('soft_expand_max_side_m').value),
        )
        max_step_clip = max(
            0.05,
            float(self.get_parameter('soft_expand_step_m').value),
        )

        if self.simple_mode:
            # Fit strictly to discovery_map footprint.
            ver = self.region.version
            bit = np.uint64(1) << np.uint64(ver % 64)
            own_mask = ((gm.discovery_map & bit) != 0) & (gm.visited > 0)
            ij = np.argwhere(own_mask)
            if len(ij) < 4:
                return
            iy_min, ix_min = ij[:, 0].min(), ij[:, 1].min()
            iy_max, ix_max = ij[:, 0].max(), ij[:, 1].max()
            wx_min = ox + ix_min * cell - VISITED_MARGIN
            wx_max = ox + (ix_max + 1) * cell + VISITED_MARGIN
            wy_min = oy + iy_min * cell - VISITED_MARGIN
            wy_max = oy + (iy_max + 1) * cell + VISITED_MARGIN
        else:
            # ── Primary: local LiDAR footprint ───────────────────────────────
            _pairs = getattr(self.region, 'all_lidar', [])
            pts2d = None
            if _pairs is not None and len(_pairs) > 0:
                chunks = []
                for scan, _ in _pairs[-12:]:
                    if scan is not None and len(scan) > 0:
                        chunks.append(np.asarray(scan)[:, :2])
                if chunks:
                    pts2d = np.vstack(chunks)

            if pts2d is not None and len(pts2d) >= 25:
                wx_min = float(np.min(pts2d[:, 0])) - VISITED_MARGIN
                wx_max = float(np.max(pts2d[:, 0])) + VISITED_MARGIN
                wy_min = float(np.min(pts2d[:, 1])) - VISITED_MARGIN
                wy_max = float(np.max(pts2d[:, 1])) + VISITED_MARGIN
            else:
                # ── Fallback: discovery AABB near current bound ───────────────
                ver = self.region.version
                bit = np.uint64(1) << np.uint64(ver % 64)
                own_mask = ((gm.discovery_map & bit) != 0) & (gm.visited > 0) & (gm.data == 0)
                bcur = self.region.bound.detach().cpu().numpy()
                pad = 0.8
                xmask = ((ox + (np.arange(gm.n_cells) + 0.5) * cell) >= bcur[0, 0] - pad) & \
                        ((ox + (np.arange(gm.n_cells) + 0.5) * cell) <= bcur[1, 0] + pad)
                ymask = ((oy + (np.arange(gm.n_cells) + 0.5) * cell) >= bcur[0, 1] - pad) & \
                        ((oy + (np.arange(gm.n_cells) + 0.5) * cell) <= bcur[1, 1] + pad)
                own_mask &= ymask[:, None] & xmask[None, :]
                ij = np.argwhere(own_mask)
                if len(ij) < 4:
                    return
                iy_min, ix_min = ij[:, 0].min(), ij[:, 1].min()
                iy_max, ix_max = ij[:, 0].max(), ij[:, 1].max()
                wx_min = ox + ix_min * cell - VISITED_MARGIN
                wx_max = ox + (ix_max + 1) * cell + VISITED_MARGIN
                wy_min = oy + iy_min * cell - VISITED_MARGIN
                wy_max = oy + (iy_max + 1) * cell + VISITED_MARGIN

        new_xmin = wx_min
        new_ymin = wy_min
        new_xmax = wx_max
        new_ymax = wy_max

        # Hard cap anchored to spawn_center so the region cannot walk across the map.
        oldb = self.region.bound.detach().cpu().numpy()
        sc = getattr(self.region, 'spawn_center', None)
        if sc is not None:
            cx, cy = float(sc[0]), float(sc[1])
        else:
            cx = float((oldb[0, 0] + oldb[1, 0]) * 0.5)
            cy = float((oldb[0, 1] + oldb[1, 1]) * 0.5)
        half = max_region_side / 2.0
        new_xmin = max(new_xmin, cx - half)
        new_xmax = min(new_xmax, cx + half)
        new_ymin = max(new_ymin, cy - half)
        new_ymax = min(new_ymax, cy + half)

        # Enforce a minimum footprint so regions are not too tiny to collect
        # stable frontiers/door structure.
        w = new_xmax - new_xmin
        h = new_ymax - new_ymin
        min_half = MIN_REGION_SIDE / 2.0
        if w < MIN_REGION_SIDE:
            new_xmin = cx - min_half
            new_xmax = cx + min_half
        if h < MIN_REGION_SIDE:
            new_ymin = cy - min_half
            new_ymax = cy + min_half

        map_xmin = gm.origin[0] - gm.map_size / 2.0
        map_xmax = gm.origin[0] + gm.map_size / 2.0
        map_ymin = gm.origin[1] - gm.map_size / 2.0
        map_ymax = gm.origin[1] + gm.map_size / 2.0
        new_xmin = max(new_xmin, map_xmin)
        new_xmax = min(new_xmax, map_xmax)
        new_ymin = max(new_ymin, map_ymin)
        new_ymax = min(new_ymax, map_ymax)

        # Bounds are monotonically expanding: they grow to cover newly explored space
        # but never shrink back (which would cause the region to "walk" and lose
        # coverage of previously trained areas). Expansion is rate-limited per tick.
        max_step = max_step_clip
        new_xmin = np.clip(min(new_xmin, oldb[0, 0]), oldb[0, 0] - max_step, oldb[0, 0])
        new_xmax = np.clip(max(new_xmax, oldb[1, 0]), oldb[1, 0], oldb[1, 0] + max_step)
        new_ymin = np.clip(min(new_ymin, oldb[0, 1]), oldb[0, 1] - max_step, oldb[0, 1])
        new_ymax = np.clip(max(new_ymax, oldb[1, 1]), oldb[1, 1], oldb[1, 1] + max_step)

        new_bound = torch.tensor([
            [new_xmin, new_ymin, -0.1],
            [new_xmax, new_ymax,  0.1],
        ], dtype=torch.float32)
        self.region.set_bound(new_bound)

    def _switch_region(self, new_region: 'Region'):
        old_ver = self.region.version
        self._apply_region_training_config(new_region)
        # Ensure incoming region has an active network before unloading current.
        # If it was previously unloaded its optimizer/network will be None.
        if getattr(new_region, 'optimizer', None) is None:
            if getattr(new_region, 'saved', False):
                new_region.load_from_disk()
            else:
                new_region.init_network()
        self._unload_current_region()
        # Flush stale training frames from the old region before switching
        with self.train_lock:
            self.train_queue.clear()
        self.train_running = False   # abort any in-flight train for old region
        self.region = new_region
        self.region.enable_debug_vis = bool(
            self.get_parameter('enable_region_debug_vis').value)
        self._last_switch_from_version = old_ver
        self._last_switch_time = self.get_clock().now().nanoseconds * 1e-9
        self._switch_candidate_version = -1
        self._switch_candidate_hits = 0
        self.get_logger().info(f"[REGION SWITCH] {old_ver} → {new_region.version}")
        self.goal_xy        = None
        self.last_goal_time = self.get_clock().now()
        # Do NOT wipe gridmap.frontier here — reconcile_all_banks evicts stale
        # points every tick against the live grid.  Wiping the grid here would
        # immediately empty every region bank on the next reconcile call, making
        # the robot think all regions are fully explored.
        # Snapshot any remaining live frontier cells into the OLD region's bank
        # before leaving, so the robot can return to finish it later.
        if old_ver in self.exploration_graph.records:
            old_rec = self.exploration_graph.records[old_ver]
            frontier_cells = np.argwhere(self.gridmap.frontier > 0)
            disc = self.gridmap.discovery_map
            for ci, cj in frontier_cells:
                cell_bits = disc[ci, cj]
                versions_here = self.gridmap.discover_regions(cell_bits)
                if old_ver in versions_here or len(versions_here) == 0:
                    wx, wy = self.gridmap._grid_to_world(int(ci), int(cj))
                    pt = np.array([wx, wy], dtype=np.float32)
                    old_region_obj = self.all_regions.get(old_ver)
                    if old_region_obj is not None and old_region_obj.in_bound(pt):
                        old_rec.add_frontier(pt, ExplorationGraph.BLACKLIST_RADIUS)
            print(f"[REGION SWITCH] Saved {len(old_rec.saved_frontiers)} frontiers for R{old_ver}")
        # Reset real-hit grid for the incoming region
        n = self.gridmap.n_cells
        self._real_hit_grid         = np.zeros((n, n), dtype=bool)
        self._real_hit_grid_version = new_region.version
        cur_epoch = int(getattr(new_region, 'epoch', 0))
        if cur_epoch < self.warmup_epochs_needed:
            warmup_epochs = self.warmup_epochs_needed - cur_epoch
            reason = (
                f"[FLOW] Region {new_region.version} is undertrained "
                f"(epoch={cur_epoch}) → entering WARMUP"
            )
        else:
            warmup_epochs = self.revisit_warmup_epochs
            reason = (
                f"[FLOW] Region {new_region.version} re-activated "
                f"(epoch={cur_epoch}) → refresh WARMUP"
            )
        self._enter_warmup(new_region, warmup_epochs, reason)

    def _unload_current_region(self):
        """Save current region weights to disk and free GPU memory."""
        if self.region is None:
            return
        try:
            self.region.save_model()          # saves checkpoint to region folder
            self.region.unload_to_disk()      # releases GPU tensors
            self.get_logger().info(
                f"[REGION {self.region.version} UNLOADED]"
            )
        except Exception as e:
            self.get_logger().error(f"[UNLOAD ERROR] {e}")

    def _trim_at_unvisited(self, pts: np.ndarray) -> 'np.ndarray | None':
        """
        Trim a path at the first waypoint that lands in an unvisited grid cell.

        Skips indices 0–1 (robot may straddle a cell boundary at its current
        position).  Returns the trimmed array, the original array if all
        waypoints are visited, or None if the trimmed result is shorter than
        2 points.
        """
        try:
            gm  = self.gridmap
            ox  = gm.origin[0] - gm.map_size / 2.0
            oy  = gm.origin[1] - gm.map_size / 2.0
            arr = np.asarray(pts, dtype=np.float32)
            gx  = ((arr[:, 0] - ox) / gm.cell_size).astype(int)
            gy  = ((arr[:, 1] - oy) / gm.cell_size).astype(int)
            inb = ((0 <= gx) & (gx < gm.n_cells) &
                   (0 <= gy) & (gy < gm.n_cells))
            vis = np.zeros(len(arr), dtype=bool)
            vis[inb] = (gm.visited[gy[inb], gx[inb]] == 1)
            unvis = np.where(~vis)[0]
            if len(unvis) == 0:
                return arr
            if unvis[0] <= 1:
                return arr   # too early to trim — keep path as-is
            trimmed = arr[:unvis[0]]
            return trimmed if len(trimmed) >= 2 else None
        except Exception:
            return pts

    def _launch_path_gen(self, robot_xy, goal_xy): #claude diff
        """Offload generate_field_path to a background thread so update() never blocks."""
        if robot_xy is None or goal_xy is None:          # ← add this guard
            return
        if self.region is None or getattr(self.region, "function", None) is None:
            return
        if getattr(self, '_path_gen_running', False):
            return  # already running, skip
        self._path_gen_running = True
        robot_xy = robot_xy.copy()
        goal_xy  = goal_xy.copy()

        def _worker():
            try:
                active_region = self.region
                if active_region is not None:
                    active_region.latest_goal = goal_xy.copy()
                # Use the bidirectional field rollout so the raw path comes
                # from the neural field itself rather than a one-way rollout
                # with synthetic straight-line completion.
                raw = active_region.generate_field_path(
                    robot_xy, goal_xy, step_size=0.10, max_steps=220)
                raw_path = None
                if raw is not None:
                    raw_arr = np.asarray(raw, dtype=np.float32)
                    if raw_arr.ndim == 2 and len(raw_arr) > 1:
                        raw_path = raw_arr
                        # Detect partial path: goal-side froze in untrained territory
                        # but start side made real progress toward the region boundary.
                        # Redirect goal_xy to the rollout endpoint so the robot dwells
                        # there, trains, and then replans toward the original frontier.
                        _ep = raw_arr[-1]
                        if float(np.linalg.norm(_ep - goal_xy)) > 0.40:
                            self.goal_xy = _ep.copy()
                            self.get_logger().info(
                                f"[PARTIAL PATH] goal redirected to trained boundary "
                                f"({_ep[0]:.2f},{_ep[1]:.2f})")

                if raw_path is not None and len(raw_path) > 1:
                    decimate = self.decimate_path(raw_path, spacing=0.10)
                    # Small window (3) avoids cutting wall corners while still
                    # smoothing out field-gradient jitter.  Window=7 was too
                    # aggressive and could pull waypoints into obstacle cells.
                    smooth   = self.smooth_path(decimate, window=3)

                    if len(smooth) < 2:
                        return

                    # Reject paths that pass through or lead into exact wall cells.
                    # If a wall hit is found: trim there and check if the trimmed
                    # endpoint is within 1 cell of the wall — if so, reject the
                    # entire path (the field is pointing INTO the wall and trimming
                    # would just leave the robot stranded at the wall face).
                    try:
                        _obs_raw = self.gridmap.data
                        _gm = self.gridmap
                        _ox = _gm.origin[0] - _gm.map_size / 2.0
                        _oy = _gm.origin[1] - _gm.map_size / 2.0
                        _pa = np.asarray(smooth, dtype=np.float32)
                        _gx = ((_pa[:, 0] - _ox) / _gm.cell_size).astype(int)
                        _gy = ((_pa[:, 1] - _oy) / _gm.cell_size).astype(int)
                        _inb = ((0 <= _gx) & (_gx < _gm.n_cells)
                                & (0 <= _gy) & (_gy < _gm.n_cells))
                        _in_wall = np.zeros(len(_pa), dtype=bool)
                        _in_wall[_inb] = (_obs_raw[_gy[_inb], _gx[_inb]] == 1)
                        _hits = np.where(_in_wall[2:])[0]
                        if _hits.size > 0:
                            _trim_at = int(_hits[0]) + 2
                            # Check: is the last kept point adjacent to the wall?
                            # If trim_at <= 3 (wall hit within first few steps), reject.
                            if _trim_at <= 2:
                                self.get_logger().debug(
                                    "[PATH REJECT] wall hit within 2 steps — discarding path")
                                return
                            smooth = smooth[:_trim_at]
                            self.get_logger().debug(
                                f"[PATH TRIM] Trimmed to {_trim_at} wp at wall cell")
                    except Exception:
                        pass

                    # Reject a 2-wp path whose endpoint is already within goal_reached_dist
                    # — it would immediately trigger PATH EXTEND without moving the robot.
                    if len(smooth) == 2:
                        ep_dist = float(np.linalg.norm(
                            np.asarray(smooth[-1]) - robot_xy))
                        if ep_dist < 0.22:
                            self.get_logger().debug(
                                "[PATH REJECT] 2-wp path endpoint already reached")
                            return  # leave latest_candidate_path None → _stabilizing_retries counts up

                    self.latest_candidate_path = np.asarray(smooth, dtype=np.float32)
                    if active_region is not None:
                        active_region.latest_path = self.latest_candidate_path
                    self.get_logger().info(
                        f"[PATH UPDATED] field path accepted, {len(smooth)} waypoints")
                else:
                    self.get_logger().debug("[PATH GEN] no field path this tick")
            except Exception as e:
                self.get_logger().warn(f"[PATH GEN] {e}")
            finally:
                self._path_gen_running = False
        threading.Thread(target=_worker, daemon=True).start() #end claude diff

    def _maybe_create_new_region_guarded(self):
        """
        Bound-exit respawn policy:
        when robot leaves the active region bound for several consecutive
        ticks, pause navigation, create a new region at robot pose, warm up,
        then resume the interrupted goal.
        """
        if self.region is None:
            self.maybe_create_new_region()
            return

        if self.robot_pos is None:
            return

        robot_xy = np.asarray(self.robot_pos[:2], dtype=np.float32)
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        margin = float(self.get_parameter('region_exit_respawn_margin').value)
        need_hits = max(1, int(self.get_parameter('region_exit_respawn_hits').value))
        b = self.region.bound.detach().cpu().numpy()
        in_bound = bool(
            (b[0, 0] - margin) <= robot_xy[0] <= (b[1, 0] + margin) and
            (b[0, 1] - margin) <= robot_xy[1] <= (b[1, 1] + margin)
        )
        if in_bound:
            self._outside_bound_hits = 0
            # If soft-expand reached its side cap, allow spawning a new region
            # when the robot keeps running near the current bound edge.
            edge_spawn_enable = bool(self.get_parameter('region_edge_spawn_enable').value)
            if edge_spawn_enable:
                sx = float(b[1, 0] - b[0, 0])
                sy = float(b[1, 1] - b[0, 1])
                max_side = max(sx, sy)
                side_cap = float(self.get_parameter('soft_expand_max_side_m').value)
                near_cap = max_side >= max(0.5, side_cap - 0.25)
                edge_margin = float(self.get_parameter('region_edge_spawn_margin').value)
                dist_to_edges = min(
                    float(abs(robot_xy[0] - b[0, 0])),
                    float(abs(b[1, 0] - robot_xy[0])),
                    float(abs(robot_xy[1] - b[0, 1])),
                    float(abs(b[1, 1] - robot_xy[1])),
                )
                near_edge = dist_to_edges <= edge_margin
                cooldown = float(self.get_parameter('region_spawn_cooldown_sec').value)
                if near_cap and near_edge:
                    self._edge_spawn_hits += 1
                else:
                    self._edge_spawn_hits = 0
                need_edge_hits = max(1, int(self.get_parameter('region_edge_spawn_hits').value))
                if (self._edge_spawn_hits >= need_edge_hits and
                        now_sec - self._last_region_spawn_time >= cooldown):
                    self._edge_spawn_hits = 0
                    self._last_region_spawn_time = now_sec
                    self._pause_for_region_respawn(
                        f"[REGION EDGE SPAWN] R{self.region.version} near soft cap "
                        f"(side={max_side:.2f}m), robot near edge ({dist_to_edges:.2f}m) -> new region"
                    )
                    self.init_region(self.robot_pos)
            return

        self._edge_spawn_hits = 0
        self._outside_bound_hits += 1
        if self._outside_bound_hits < need_hits:
            return
        self._outside_bound_hits = 0
        cooldown = float(self.get_parameter('region_spawn_cooldown_sec').value)
        if now_sec - self._last_region_spawn_time < cooldown:
            return
        self._last_region_spawn_time = now_sec

        self._pause_for_region_respawn(
            f"[REGION EXIT] Robot left R{self.region.version} bound at "
            f"({robot_xy[0]:.2f},{robot_xy[1]:.2f}); creating new region"
        )
        self.init_region(self.robot_pos)

    def _assign_frontiers_to_regions(self, current_version: int):
        """
        Assign live frontier cells to region banks based on which regions'
        current bounds contain each cell.

        Rules (in priority order):
          1. Current region's bound contains the frontier   → current region.
          2. No region's bound contains it (uncovered)      → current region.
          3. Exactly one non-current region covers it AND
             that region is not exhausted                   → that remote region.
          4. Multiple non-current regions cover it          → current region.
        """
        frontier_cells = np.argwhere(self.gridmap.frontier > 0)
        if len(frontier_cells) == 0:
            return
        if self.simple_mode:
            for ci, cj in frontier_cells:
                wx, wy = self.gridmap._grid_to_world(int(ci), int(cj))
                self.exploration_graph.records[current_version].add_frontier(
                    np.array([wx, wy], dtype=np.float32),
                    ExplorationGraph.BLACKLIST_RADIUS
                )
            return

        # Vectorised world-position of all frontier cells
        gm      = self.gridmap
        cell_sz = gm.cell_size
        ox      = gm.origin[0] - gm.map_size / 2.0
        oy      = gm.origin[1] - gm.map_size / 2.0
        pts = np.stack([
            ox + (frontier_cells[:, 1] + 0.5) * cell_sz,   # j → x
            oy + (frontier_cells[:, 0] + 0.5) * cell_sz,   # i → y
        ], axis=1).astype(np.float32)                        # (N, 2)

        N = len(pts)

        # Pre-extract bounds for all regions
        region_versions = list(self.all_regions.keys())
        bounds_list     = []
        for v in region_versions:
            b = self.all_regions[v].bound
            if hasattr(b, 'cpu'):
                b = b.cpu().numpy()
            bounds_list.append(b)            # (2, 3) numpy array

        R = len(region_versions)
        if R == 0:
            for pt in pts:
                self.exploration_graph.records[current_version].add_frontier(
                    pt, ExplorationGraph.BLACKLIST_RADIUS)
            return

        # coverage[ri, i] = True if region ri's bound contains frontier i
        coverage = np.zeros((R, N), dtype=bool)
        cur_ri   = -1
        for ri, (v, b) in enumerate(zip(region_versions, bounds_list)):
            coverage[ri] = (
                (pts[:, 0] >= b[0, 0]) & (pts[:, 0] <= b[1, 0]) &
                (pts[:, 1] >= b[0, 1]) & (pts[:, 1] <= b[1, 1])
            )
            if v == current_version:
                cur_ri = ri

        coverage_sum   = coverage.sum(axis=0)                           # (N,)
        current_covers = coverage[cur_ri] if cur_ri >= 0 else np.zeros(N, dtype=bool)

        exclusive_pts: dict = {}   # {version: list[np.ndarray]}
        current_pts: list   = []

        for i in range(N):
            pt = pts[i]
            if current_covers[i] or coverage_sum[i] == 0:
                # Rule 1 & 2: current covers it, or nobody covers it → current
                current_pts.append(pt)
            elif coverage_sum[i] == 1:
                # Rule 3: exactly one non-current region covers this frontier.
                # Only assign to that region if it has TRAINED near this cell
                # (its discovery_map bit is stamped here or adjacent).
                # Without this check, frontiers in the remote region's 0.2m
                # padding zone (outside its visited footprint) would be
                # assigned there, and its field would produce wall-crossing
                # paths to reach them.
                ri = int(np.argmax(coverage[:, i]))
                v  = region_versions[ri]
                rec = self.exploration_graph.records.get(v)
                _remote_bit = np.uint64(1) << np.uint64(v % 64)
                _ci = int(frontier_cells[i, 0])
                _cj = int(frontier_cells[i, 1])
                _remote_has_data = bool(
                    self.gridmap.discovery_map[_ci, _cj] & _remote_bit)
                if rec is not None and not rec.is_exhausted() and _remote_has_data:
                    exclusive_pts.setdefault(v, []).append(pt)
                else:
                    current_pts.append(pt)
            else:
                # Rule 4: multiple non-current regions → current
                current_pts.append(pt)

        # Assign exclusive frontiers to their remote regions
        for version, _pts in exclusive_pts.items():
            region_obj = self.all_regions.get(version)
            if region_obj is None:
                continue
            b = region_obj.bound
            if hasattr(b, 'cpu'):
                b = b.cpu().numpy()
            center = ((b[0, :2] + b[1, :2]) / 2.0).astype(np.float32)
            self.exploration_graph.ensure_region(version, center)
            for pt in _pts:
                self.exploration_graph.records[version].add_frontier(
                    pt, ExplorationGraph.BLACKLIST_RADIUS)

        # All current + overlap frontiers → current region
        for pt in current_pts:
            self.exploration_graph.records[current_version].add_frontier(
                pt, ExplorationGraph.BLACKLIST_RADIUS)

    def _build_is_frontier_fn(self):
        """Returns a closure over the current frontier grid for bank reconciliation."""
        frontier_cells = np.argwhere(self.gridmap.frontier > 0)
        if len(frontier_cells) == 0:
            def is_frontier_fn(xy, radius):
                return False
            return is_frontier_fn
        # World coords of all live frontier cells
        world_pts = np.array([
            self.gridmap._grid_to_world(int(ci), int(cj))
            for ci, cj in frontier_cells
        ])
        tree = cKDTree(world_pts)
        def is_frontier_fn(xy: np.ndarray, radius: float) -> bool:
            hits = tree.query_ball_point(xy[:2], r=radius)
            return len(hits) > 0
        return is_frontier_fn

    def _start_goal_dwell(self):
        dwell_frames_needed = int(self.get_parameter('dwell_frames_needed').value)
        if self.region is None or dwell_frames_needed <= 0:
            self.goal_xy   = None
            self.nav_state = "IDLE"
            self._goal_kind = "none"
            return

        self.dwell_active       = True
        self.dwell_frames       = []
        self.dwell_frames_count = 0
        self.dwell_phase        = "COLLECT"
        self.nav_state          = "DWELL"
        self.goal_xy            = None
        self._goal_kind         = "none"
        self._best_goal_dist    = float("inf")
        self.navigator.stop_robot()
        # Keep queued frames so accumulated map coverage is not dropped.
        # Dwell still uses an epoch baseline from "now".
        self.dwell_epoch_start = int(self.region.epoch)
        #print(f"[DWELL] Starting — collecting {dwell_frames_needed} frames "
        #      f"then training {int(self.get_parameter('dwell_epochs').value)} epochs "
        #      f"(baseline epoch={self.dwell_epoch_start})")

    def _tick_goal_dwell(self, world_points_hit: np.ndarray, robot_xy: np.ndarray):
        dwell_frames_needed = int(self.get_parameter('dwell_frames_needed').value)
        dwell_epochs        = int(self.get_parameter('dwell_epochs').value)
        # If region changed under us (new region created), abort dwell cleanly
        if self.region is not None and int(self.region.epoch) < self.dwell_epoch_start:
            print("[DWELL] Region switched during dwell — ending dwell, entering WARMUP")
            self._end_goal_dwell()
            # Trigger warmup for the new region
            self._enter_warmup(
                self.region,
                self.revisit_warmup_epochs,
                "[DWELL] Region switched during dwell → entering WARMUP"
            )
            return
        if self.dwell_phase == "COLLECT":
            if self.region is not None and len(world_points_hit) > 5:
                self.dwell_frames.append((
                    world_points_hit.copy(),
                    self.robot_pos.copy() if self.robot_pos is not None else robot_xy))
                self.dwell_frames_count += 1
                #print(f"[DWELL] Collected frame {self.dwell_frames_count}/{dwell_frames_needed}")

            if self.dwell_frames_count >= dwell_frames_needed:
                with self.train_lock:
                    for frame in self.dwell_frames:
                        self.train_queue.append(frame)
                self.dwell_epoch_start = int(self.region.epoch)
                self.dwell_phase       = "TRAIN"
                print(f"[DWELL] Collection done — training for {dwell_epochs} epochs "
                      f"from epoch {self.dwell_epoch_start}")

        elif self.dwell_phase == "TRAIN":
            if self.region is None:
                self._end_goal_dwell()
                return
            epochs_done = int(self.region.epoch) - self.dwell_epoch_start
            now_sec = self.get_clock().now().nanoseconds * 1e-9
            if now_sec - getattr(self, '_dwell_log_t', 0.0) >= 1.0:
                self._dwell_log_t = now_sec
                self.get_logger().info(
                    f"[DWELL] waiting for training: epochs={epochs_done}/{dwell_epochs} "
                    f"queue={len(self.train_queue)} running={self.train_running}"
                )
            #print(f"[DWELL] Training epoch {epochs_done}/{dwell_epochs}")
            if epochs_done >= dwell_epochs:
                print(f"[DWELL] Training complete — resuming navigation")
                self._end_goal_dwell()

    def _end_goal_dwell(self):
        """Clean up dwell state and return to normal navigation."""
        self.dwell_active  = False
        self.dwell_frames  = []
        self.dwell_phase   = "COLLECT"
        self.nav_state     = "IDLE"
        self.goal_xy       = None
        self._goal_kind    = "none"
        self._best_goal_dist = float("inf")

    def _pause_for_region_respawn(self, reason: str):
        """
        Stop navigation and preserve the current goal so it can be resumed
        after the new active region warms up.
        """
        if self.goal_xy is not None and self._goal_kind in ("frontier", "backtrack"):
            self._resume_goal_after_warmup = self.goal_xy.copy()
            self._resume_goal_kind = str(self._goal_kind)
        else:
            self._resume_goal_after_warmup = None
            self._resume_goal_kind = "none"
        self.get_logger().warn(reason)

        self.navigator.stop_robot()
        self.navigator.reset_path()
        self.path = None
        self.latest_candidate_path = None
        self._path_gen_running = False
        self._path_total_dist = None
        self._midpoint_replanned = False
        self._path_refresh_ticks = 0
        self._best_goal_dist = float("inf")
        self._last_path_progress_s = 0.0
        self.goal_xy = None
        self._goal_kind = "none"
        self._active_backtrack_anchor = None
        self.nav_state = "IDLE"

    def _resume_pending_goal_after_warmup(self, robot_xy: np.ndarray, now_sec: float) -> bool:
        goal = self._resume_goal_after_warmup
        if goal is None:
            return False
        goal = np.asarray(goal, dtype=np.float32)
        if not np.isfinite(goal).all():
            self._resume_goal_after_warmup = None
            self._resume_goal_kind = "none"
            return False

        self.goal_xy = goal
        self._goal_kind = (
            self._resume_goal_kind
            if self._resume_goal_kind in ("frontier", "backtrack")
            else "frontier"
        )
        self.wait_start_time = now_sec
        self._path_gen_running = False
        self.latest_candidate_path = None
        self._best_goal_dist = float(np.linalg.norm(robot_xy - self.goal_xy))
        self._last_progress_time = now_sec
        self._last_path_progress_s = 0.0
        self.nav_state = "STABILIZING"
        self._resume_goal_after_warmup = None
        self._resume_goal_kind = "none"
        self.get_logger().info(
            f"[FLOW] Warmup complete → resuming pending {self._goal_kind} goal "
            f"at ({self.goal_xy[0]:.2f},{self.goal_xy[1]:.2f})"
        )
        return True

    def update(self):
        try:
            self._update_impl()
        except Exception as e:
            import traceback
            self.get_logger().error(f"[UPDATE] Unhandled exception: {e}")
            traceback.print_exc()
            try:
                self.navigator.stop_robot()
            except Exception:
                pass

    def control_tick(self):
        if self.robot_pos is None:
            return
        if self.mode != "EXPLORE" or self.nav_state != "FOLLOWING" or self.path is None or self.goal_xy is None:
            self.navigator.stop_robot()
            return
        try:
            if self.navigator.drive_path(self.path):
                self._controller_path_done = True
        except Exception as e:
            self.get_logger().warn(f"[CONTROL] {e}")
            self.navigator.stop_robot()

    def _update_impl(self):
        now = self.get_clock().now()
        self._apply_door_match_radius_config()
        # ── Prerequisites ────────────────────────────────────────────────
        self._update_robot_pose_from_tf()
        if not self._slam_map_received:
            if not hasattr(self, '_diag_no_slam_logged'):
                self._diag_no_slam_logged = False
            if not self._diag_no_slam_logged:
                self.get_logger().warn("[DIAG] Waiting for SLAM Toolbox /map ...")
                self._diag_no_slam_logged = True
            return
        self._diag_no_slam_logged = False
        if self.robot_pos is None or self.latest_cloud is None:
            if not hasattr(self, '_diag_no_data_logged'):
                self._diag_no_data_logged = False
            if not self._diag_no_data_logged:
                self.get_logger().warn(
                    f"[DIAG] Waiting for data: robot_pos={'OK' if self.robot_pos is not None else 'MISSING'}, "
                    f"lidar={'OK' if self.latest_cloud is not None else 'MISSING'}")
                self._diag_no_data_logged = True
            return
        self._diag_no_data_logged = False
        if self.region is not None:
            self.region.last_robot_pose = np.array(
                [self.robot_pos[0], self.robot_pos[1], 0.0], dtype=np.float32
            )
        result = self.adapter.update(self.latest_cloud, self.robot_pos)
        points_lidar, speeds = result if (result is not None) else (None, None)
        if points_lidar is None or len(points_lidar) == 0:
            if not hasattr(self, '_diag_lidar_logged'):
                self._diag_lidar_logged = False
            if not self._diag_lidar_logged:
                self.get_logger().warn("[DIAG] adapter.update() returned no points — check LiDAR topic/z-filter")
                self._diag_lidar_logged = True
            return
        self._diag_lidar_logged = False
        # ── Scan-time TF: look up map←sensor directly at the scan's timestamp ──
        # A single map←source_frame lookup at the scan's own timestamp is the
        # correct, minimal way to transform LiDAR points to world frame.
        # Splitting into static (sensor→base_link) + dynamic (base_link→map)
        # introduces two separate lookups and manual yaw extraction, which was
        # the source of the "spinning map" bug: the yaw-only 2-D rotation matrix
        # missed pitch/roll corrections and was error-prone in the fallback path.
        source_frame = self.latest_cloud.header.frame_id
        scan_stamp   = self.latest_cloud.header.stamp
        scan_time    = rclpy.time.Time.from_msg(scan_stamp)

        # Try to look up map←source_frame at the exact scan timestamp.
        # Falls back to Time(0) (latest available) if the buffer doesn't yet
        # have a transform for that stamp — which can happen in the first few
        # ticks of a simulation before the TF history fills up.
        try:
            tf_map_sensor = self.tf_buffer.lookup_transform(
                'map', source_frame, scan_time,
                timeout=Duration(seconds=0.1))
        except Exception:
            try:
                # Second fallback: latest available transform (Time(0))
                tf_map_sensor = self.tf_buffer.lookup_transform(
                    'map', source_frame, rclpy.time.Time(),
                    timeout=Duration(seconds=0.05))
            except Exception as e:
                if not getattr(self, '_tf_map_sensor_warned', False):
                    self.get_logger().warn(f"[TF] map←{source_frame} unavailable: {e}")
                    self._tf_map_sensor_warned = True
                return

        self._tf_map_sensor_warned = False   # reset on success
        t_ms  = tf_map_sensor.transform.translation
        q_ms  = tf_map_sensor.transform.rotation
        R_ms  = Rotation.from_quat([q_ms.x, q_ms.y, q_ms.z, q_ms.w]).as_matrix().astype(np.float32)
        t_ms_arr = np.array([t_ms.x, t_ms.y, t_ms.z], dtype=np.float32)

        # Transform all sensor-frame points directly to map frame in one step.
        world_points = (R_ms @ points_lidar.T).T + t_ms_arr

        # ── Bound filter ──────────────────────────────────────────────────
        robot_xy = self.robot_pos[:2]
        half = self.gridmap.map_size / 2
        ox, oy = self.gridmap.origin
        in_bounds = ((world_points[:, 0] > ox - half) & (world_points[:, 0] < ox + half) &
                     (world_points[:, 1] > oy - half) & (world_points[:, 1] < oy + half))
        world_points_filtered = world_points[in_bounds]
        speeds_filtered = speeds[in_bounds]
        if len(world_points_filtered) < 5:
            return
        # Angular downsampling + free-ray synthesis
        # For each of 360 angular bins:
        #   - Real hit bin  → closest point from sensor; hit_dist = measured dist
        #   - Empty bin     → synthetic endpoint at MAX_RANGE; hit_dist = inf
        # The raycast loop later uses hit_dist to decide whether to mark a wall.
        robot = robot_xy
        vecs   = world_points_filtered[:, :2] - robot_xy[None, :]
        dists  = np.linalg.norm(vecs, axis=1) + 1e-9
        angles = np.arctan2(vecs[:, 1], vecs[:, 0])
        bins   = ((angles + np.pi) / (2 * np.pi) * 359).astype(np.int32)

        closest_idx  = {}
        closest_dist = np.full(360, np.inf, dtype=np.float32)
        for k, b in enumerate(bins):
            if dists[k] < closest_dist[b]:
                closest_dist[b] = dists[k]
                closest_idx[b]  = k

        # Real-hit cloud — identical to the original method; used for raycast + gridmap
        keep = np.array(list(closest_idx.values()), dtype=np.int32)
        world_points_hit = world_points_filtered[keep]
        speeds_hit       = speeds_filtered[keep]

        # Full 360-bin cloud — real hits + synthetic free-ray endpoints
        # Used ONLY for training data collection; never touches gridmap/raycast.
        #
        # Free-ray endpoints are only included if the map cell they land on has
        # NOT already been claimed by a real hit this region lifetime.  This
        # prevents the synthetic endpoints from overwriting real-hit training
        # data as the robot moves and new free directions emerge.

        # ── Ensure the real-hit grid matches the current region ───────────
        if (self.region is not None and
                self.region.version != self._real_hit_grid_version):
            n = self.gridmap.n_cells
            self._real_hit_grid = np.zeros((n, n), dtype=bool)
            self._real_hit_grid_version = self.region.version

        _cell   = self.gridmap.cell_size
        _origin = self.gridmap.origin
        _half   = self.gridmap.map_size / 2.0

        def _world_to_grid_clamp(x, y):
            ix = int((x - (_origin[0] - _half)) / _cell)
            iy = int((y - (_origin[1] - _half)) / _cell)
            n  = self.gridmap.n_cells
            return (max(0, min(n - 1, ix)),
                    max(0, min(n - 1, iy)))

        full_endpoints  = []
        full_speeds     = []
        non_hit_dirs    = []   # unit 2-D directions of non-hitting (free) rays
        free_z = float(self.robot_pos[2]) if len(self.robot_pos) > 2 else 0.0

        for b in range(360):
            if b in closest_idx:
                idx = closest_idx[b]
                pt  = world_points_filtered[idx]
                full_endpoints.append(pt)
                full_speeds.append(speeds_filtered[idx])
                # Mark this cell as claimed by a real hit
                gx, gy = _world_to_grid_clamp(pt[0], pt[1])
                if self._real_hit_grid.shape[0] > gy and self._real_hit_grid.shape[1] > gx:
                    self._real_hit_grid[gy, gx] = True
            else:
                theta  = (b / 359.0) * 2.0 * np.pi - np.pi
                dir_xy = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
                non_hit_dirs.append(dir_xy)   # collect for interior training
                free_xy = robot_xy + MAX_RANGE * dir_xy
                gx, gy  = _world_to_grid_clamp(free_xy[0], free_xy[1])
                if self._real_hit_grid.shape[0] > gy and self._real_hit_grid.shape[1] > gx:
                    if self._real_hit_grid[gy, gx]:
                        # A real hit already claimed this cell — skip the synthetic
                        # endpoint so we don't corrupt the training data there.
                        continue
                full_endpoints.append(
                    np.array([free_xy[0], free_xy[1], free_z], dtype=np.float32))
                full_speeds.append(0.0)

        world_points_360 = np.asarray(full_endpoints, dtype=np.float32)

        # ── Raycast: update gridmap.data + visited (real hits only) ──────
        cell   = self.gridmap.cell_size
        origin = self.gridmap.origin
        half_  = self.gridmap.map_size / 2

        def world_to_ij(x, y):
            return (int((x - origin[0] + half_) / cell),
                    int((y - origin[1] + half_) / cell))

        _newly_cleared_frontier_pts = []

        for pt3d in world_points_hit:
            pt = pt3d[:2]
            dx, dy = pt[0] - robot_xy[0], pt[1] - robot_xy[1]
            dist = float(np.hypot(dx, dy))
            if dist > MAX_RANGE:
                scale = MAX_RANGE / dist
                dx *= scale; dy *= scale; dist = MAX_RANGE
            steps = int(dist / cell)
            if steps < 1:
                ix, iy = world_to_ij(pt[0], pt[1])
                if 0 <= ix < self.gridmap.n_cells and 0 <= iy < self.gridmap.n_cells:
                    self.gridmap.data[iy, ix] = 1
                    self.gridmap.visited[iy, ix] = 1
                continue
            for s in range(steps - 1):
                x = robot_xy[0] + dx * s / steps
                y = robot_xy[1] + dy * s / steps
                ix, iy = world_to_ij(x, y)
                if not (0 <= ix < self.gridmap.n_cells and
                        0 <= iy < self.gridmap.n_cells):
                    break
                if self.gridmap.data[iy, ix] == 1:
                    break
                if self.gridmap.frontier[iy, ix]:
                    _newly_cleared_frontier_pts.append(
                        np.array([x, y], dtype=np.float32))
                    self.gridmap.frontier[iy, ix] = 0   # eagerly clear ghost frontier
                self.gridmap.visited[iy, ix] = 1
            ix_end, iy_end = world_to_ij(pt[0], pt[1])
            if (0 <= ix_end < self.gridmap.n_cells and
                    0 <= iy_end < self.gridmap.n_cells):
                self.gridmap.data[iy_end, ix_end] = 1
                self.gridmap.visited[iy_end, ix_end] = 1

        # ── Raycast non-hitting rays: mark free, no obstacle at end ──────────
        # Cells swept by a non-hitting (max-range) ray are confirmed free.
        # We mark them visited so x1 sampling can land there.
        # We do NOT mark the endpoint as an obstacle — that would be a fake hit.
        # Depth is capped at REGION_RADIUS (not MAX_RANGE): cells within the
        # current region get marked visited (enabling x1 sampling), while cells
        # beyond REGION_RADIUS stay unknown so frontiers form there naturally.
        NON_HIT_DEPTH = REGION_RADIUS
        for dir_xy in non_hit_dirs:
            dx = float(dir_xy[0]) * NON_HIT_DEPTH
            dy = float(dir_xy[1]) * NON_HIT_DEPTH
            steps = max(1, int(NON_HIT_DEPTH / cell))
            for s in range(steps):
                x = robot_xy[0] + dx * s / steps
                y = robot_xy[1] + dy * s / steps
                ix, iy = world_to_ij(x, y)
                if not (0 <= ix < self.gridmap.n_cells and
                        0 <= iy < self.gridmap.n_cells):
                    break
                if self.gridmap.data[iy, ix] == 1:
                    break   # stop at a confirmed real obstacle
                self.gridmap.visited[iy, ix] = 1
            # intentionally no endpoint write — non-hitting rays have no obstacle

        # ── Immediately evict newly-visited frontier cells from ALL region banks ──
        if _newly_cleared_frontier_pts:
            for _rec in self.exploration_graph.records.values():
                for pt in _newly_cleared_frontier_pts:
                    _rec.mark_visited(pt)

        # ── Gridmap update ────────────────────────────────────────────────
        self._map_update_counter += 1

        # LiDAR drives real-time occupancy; SLAM merges corrections every ~0.5s.
        self.gridmap.update(
            lidar_scan=world_points_hit[:, :2],
            points=world_points_hit[:, :2],
            speeds=speeds_hit,
            cur_pos=robot_xy,
            region=self.region,
            regions=self.all_regions,
            update_room=False,
            update_occupancy=True,
        )

        # ── Doorway detection (every 10 ticks ≈ 2s) ─────────────────────
        if self._map_update_counter % 10 == 0:
            self._detect_doorways()

        # ── Snapshot live frontiers into ExplorationGraph ────────────────
        if self.region is not None:
            ver = self.region.version
            center = np.array(self.region.center[:2], dtype=np.float32)
            self.exploration_graph.ensure_region(ver, center)
            self._assign_frontiers_to_regions(ver)

            # Rate-limited per-region frontier count log
            _now = self.get_clock().now().nanoseconds * 1e-9
            if _now - getattr(self, '_frontier_log_t', 0.0) >= 5.0:
                self._frontier_log_t = _now
                cur_ver = self.region.version if self.region else None
                parts = [
                    f"R{v}({'cur' if v == cur_ver else 'rem'}):{len(rec.saved_frontiers)}"
                    for v, rec in self.exploration_graph.records.items()
                ]
                self.get_logger().info(f"[FRONTIERS] {' | '.join(parts) if parts else 'none'}")

        # Reconcile ALL region banks to evict stale frontiers.
        # Remote frontiers that have been mapped (wall or free) must be removed,
        # otherwise the robot re-routes to them indefinitely.
        is_frontier_fn = self._build_is_frontier_fn()
        for ver, rec in self.exploration_graph.records.items():
            rec.saved_frontiers = [
                pt for pt in rec.saved_frontiers
                if is_frontier_fn(pt, 0.6)
            ]
        self._push_frontier_anchor(robot_xy)

        # ── Enqueue training frame (ALL modes) ───────────────────────────
        # CRITICAL: pass world_points_HIT (real obstacle hits only).
        # world_points_360 contains synthetic free-ray endpoints placed at
        # MAX_RANGE in empty angular bins.  These are not real obstacles.
        # Passing them to get_data_from_lidar corrupts the global KD-tree,
        # giving every sampled point a falsely small clearance value and
        # collapsing the speed gradient that the field needs to learn.
        if self.region is not None and bool(self.get_parameter('train_after_region').value):
            self.train_counter += 1
            if self.train_counter % self.train_every_n_frames == 0:
                enqueue = (self.mode == "WARMUP")
                cur_region_ver = int(self.region.version)
                cur_pose = np.asarray(self.robot_pos[:2], dtype=np.float32)
                if not enqueue:
                    min_motion = float(self.get_parameter('train_min_robot_motion_m').value)
                    last_pose = getattr(self, '_last_train_enqueue_pose', None)
                    last_region = getattr(self, '_last_train_enqueue_region', None)
                    if last_pose is not None and last_region == cur_region_ver:
                        moved = float(np.linalg.norm(cur_pose - last_pose))
                        enqueue = moved >= min_motion
                if enqueue:
                    _nh_arr = np.array(non_hit_dirs, dtype=np.float32) if non_hit_dirs else None
                    with self.train_lock:
                        self.train_queue.append((
                            world_points_hit.copy(),
                            self.robot_pos.copy(),
                            _nh_arr,
                        ))
                    self._last_train_enqueue_pose = cur_pose.copy()
                    self._last_train_enqueue_region = cur_region_ver
        if (self._map_update_counter % 3 == 0
                and self.mode != "BOOTSTRAP"
                and bool(self.get_parameter('allow_region_soft_expand').value)):
            self.fit_region_to_lidar()
        # Satellite creation disabled for bound-exit region policy.
        # Inter-region graph edge rebuilding disabled: exploration now relies
        # on local frontier selection + frontier-anchor backtracking.

        # ── Region boundary check ─────────────────────────────────────────
        _region_ver_before_guard = int(self.region.version) if self.region is not None else -1
        self._maybe_create_new_region_guarded()
        _region_ver_after_guard = int(self.region.version) if self.region is not None else -1
        if (_region_ver_after_guard != _region_ver_before_guard
                and self.region is not None
                and len(world_points_hit) >= 50):
            _nh_arr2 = np.array(non_hit_dirs, dtype=np.float32) if non_hit_dirs else None
            with self.train_lock:
                self.train_queue.clear()
                self.train_queue.append((
                    world_points_hit.copy(),
                    self.robot_pos.copy(),
                    _nh_arr2,
                ))
            self._last_train_enqueue_pose = np.asarray(self.robot_pos[:2], dtype=np.float32).copy()
            self._last_train_enqueue_region = _region_ver_after_guard
        # ── Trajectory ───────────────────────────────────────────────────
        if self.robot_pos is not None:
            self._traj_history.append(self.robot_pos[:2].copy())

        # ── Mode state machine (navigation only, map already updated above) ──
        if self.mode == "BOOTSTRAP":
            self.navigator.stop_robot()
            self.bootstrap_frames += 1
            if self.region is None and self.bootstrap_frames >= self.bootstrap_threshold:
                self.get_logger().info("Initializing first region")
                self.init_region(self.robot_pos)
                self.last_goal_time    = now

        elif self.mode == "WARMUP":
            self.navigator.stop_robot()
            if self.region is not None:
                cur_epochs = int(self.region.epoch) - int(self.warmup_start_epoch)
                #print(f"[WARMUP] epoch {cur_epochs}/{self.warmup_epochs_needed}")
                if cur_epochs >= self._warmup_epoch_target:
                    self.mode        = "EXPLORE"
                    self.warmup_done = True
                    self.last_goal_time = now
                    if self.region is not None:
                        self.region.warmup_complete_epoch = int(self.region.epoch)
                    now_sec = now.nanoseconds * 1e-9
                    robot_xy = np.asarray(self.robot_pos[:2], dtype=np.float32)
                    resumed = self._resume_pending_goal_after_warmup(robot_xy, now_sec)
                    if not resumed:
                        self.goal_xy = None
                        self._goal_kind = "none"
                        self.nav_state = "IDLE"
                        self.get_logger().info("[FLOW] Warmup complete → starting EXPLORE")

        elif self.mode == "EXPLORE":
            robot_xy = np.asarray(self.robot_pos[:2], dtype=np.float32)
            now_sec  = now.nanoseconds * 1e-9

            # ── IDLE: pick a new frontier goal ───────────────────────────
            if self.nav_state == "IDLE":
                print("calling pick_goal")
                g = self.pick_goal()

                if g is None:
                    self.navigator.stop_robot()
                    print("[NO FRONTIER]")
                else:
                    self.goal_xy           = np.asarray(g, dtype=np.float32)
                    self.wait_start_time   = now_sec
                    self._path_gen_running = False
                    self._best_goal_dist   = float(np.linalg.norm(robot_xy - self.goal_xy))
                    self._last_progress_time = now_sec
                    self._last_path_progress_s = 0.0
                    print("[GOAL SET]", self.goal_xy)
                    self.latest_candidate_path = None
                    self._stabilizing_retries = 0
                    self._replan_count = 0

                    self.nav_state = "STABILIZING"
            # ── STABILIZING: generate path in background, don't block ────
            elif self.nav_state == "STABILIZING":
                self.navigator.stop_robot()
                # Capture running state BEFORE launching so the timeout branch
                # can distinguish "thread still alive from last tick" (extend)
                # from "thread finished with no result this tick" (count failure).
                _gen_was_running = self._path_gen_running
                self._launch_path_gen(robot_xy, self.goal_xy)

                path_ready = self.latest_candidate_path is not None
                timed_out  = now_sec - self.wait_start_time >= self.wait_duration
                if path_ready or timed_out:
                    if self.latest_candidate_path is not None:
                        print("[PATH FROZEN]")
                        self.path = self.latest_candidate_path.copy()
                        self.latest_candidate_path = None  # consume — prevent stale re-use
                        self.navigator.reset_path()
                        self._path_refresh_ticks = 0
                        self._path_total_dist = float(np.sum(
                            np.linalg.norm(np.diff(self.path, axis=0), axis=1)))
                        self._midpoint_replanned = False
                        _d_goal_now = float(np.linalg.norm(robot_xy - self.goal_xy))
                        if _d_goal_now < self._best_goal_dist - 0.50:
                            self._replan_count = 0  # significant progress — reset
                        self._best_goal_dist = _d_goal_now
                        self._last_progress_time = now_sec
                        self._last_path_progress_s = 0.0
                        # Abandon frontier if too many replans without meaningful progress
                        self._replan_count = getattr(self, '_replan_count', 0) + 1
                        if self._replan_count > 10 and _d_goal_now > 0.20:
                            self.get_logger().warn(
                                f"[STUCK] Goal ({self.goal_xy[0]:.2f},{self.goal_xy[1]:.2f}) "
                                f"unreachable after {self._replan_count} replans — abandoning"
                            )
                            self._visited_goals.append(self.goal_xy.copy())
                            self.goal_xy = None
                            self._goal_kind = "none"
                            self._replan_count = 0
                            self._stabilizing_retries = 0
                            self.nav_state = "IDLE"
                            return
                        self.nav_state = "FOLLOWING"
                    elif _gen_was_running:
                        # Previous path-gen thread is still alive (long rollout / lock
                        # contention) — extend the window without counting a failure.
                        self.wait_start_time = now_sec
                    else:
                        # Path gen finished this tick but returned no valid path.
                        self._stabilizing_retries += 1
                        if self._stabilizing_retries >= 5:
                            self.get_logger().warn(
                                f"[NO PATH] Goal ({self.goal_xy[0]:.2f},{self.goal_xy[1]:.2f}) "
                                f"unreachable after {self._stabilizing_retries} attempts "
                                f"(kind={self._goal_kind}). Abandoning."
                            )
                            if self._goal_kind == "graph_route":
                                self._graph_route_chain = []
                                self._graph_route_idx = 0
                            self._visited_goals.append(self.goal_xy.copy())
                            self.goal_xy = None
                            self._goal_kind = "none"
                            self._stabilizing_retries = 0
                            self.nav_state = "IDLE"
                        else:
                            print(f"[NO PATH FOUND — retrying {self._stabilizing_retries}/5]")
                            self.wait_start_time = now_sec  # extend wait

            # ── FOLLOWING: drive, continuously refresh path in background ─
            elif self.nav_state == "FOLLOWING":
                if self.goal_xy is None:
                    self.nav_state = "IDLE"
                    self.path = None
                    return
                if self.goal_xy is not None:
                    d_goal = float(np.linalg.norm(robot_xy - self.goal_xy))
                else:
                    d_goal = None
                #print(f"Distance to frontier: {d_goal:.2f}")
                if d_goal is None:
                    self.nav_state = "IDLE"
                    self.goal_xy = None
                    self.path = None 

                if d_goal is not None and d_goal < 0.25:
                    print("[FRONTIER REACHED]")
                    self.navigator.stop_robot()
                    self.navigator.reset_path()

                    if self.goal_xy is not None and self._goal_kind == "frontier":
                        # ── Clear frontier cells around reached goal ──────────────
                        gix = int((self.goal_xy[0] - (self.gridmap.origin[0] - self.gridmap.map_size/2))
                                  / self.gridmap.cell_size)
                        giy = int((self.goal_xy[1] - (self.gridmap.origin[1] - self.gridmap.map_size/2))
                                  / self.gridmap.cell_size)
                        pad = 3
                        for di in range(-pad, pad+1):
                            for dj in range(-pad, pad+1):
                                ii, jj = giy+di, gix+dj
                                if 0 <= ii < self.gridmap.n_cells and 0 <= jj < self.gridmap.n_cells:
                                    self.gridmap.frontier[ii, jj] = 0
                                    self.gridmap.visited[ii, jj]  = 1

                        # ── Update visited blacklist ──────────────────────────────
                        self._visited_goals.append(self.goal_xy.copy())
                        if len(self._visited_goals) > 50:
                            self._visited_goals.pop(0)

                        # ── Notify ExplorationGraph ───────────────────────────────
                        if self.region is not None:
                            self.exploration_graph.mark_goal_reached(
                                self.region.version, self.goal_xy
                            )
                    elif self.goal_xy is not None and self._goal_kind == "backtrack":
                        self.get_logger().info(
                            f"[BACKTRACK] anchor reached at ({self.goal_xy[0]:.2f}, {self.goal_xy[1]:.2f})"
                        )
                    elif self.goal_xy is not None and self._goal_kind == "graph_route":
                        # Advance to next waypoint in chain
                        self._graph_route_idx += 1
                        if self._graph_route_idx < len(self._graph_route_chain):
                            next_wp = self._graph_route_chain[self._graph_route_idx]
                            self.goal_xy = np.asarray(next_wp, dtype=np.float32)
                            self._best_goal_dist = float(np.linalg.norm(robot_xy - self.goal_xy))
                            self._last_progress_time = now_sec
                            self._last_path_progress_s = 0.0
                            self.latest_candidate_path = None
                            self._path_gen_running = False
                            self.path = None
                            self.nav_state = "STABILIZING"
                            self.wait_start_time = now_sec
                            self._stabilizing_retries = 0
                            self.get_logger().info(
                                f"[GRAPH ROUTE] Waypoint {self._graph_route_idx}/"
                                f"{len(self._graph_route_chain)-1} reached, "
                                f"advancing to ({next_wp[0]:.2f},{next_wp[1]:.2f})"
                            )
                            return
                        else:
                            self.get_logger().info("[GRAPH ROUTE] Chain complete — frontier reached")
                            self._graph_route_chain = []
                            self._goal_kind = "frontier"  # treat as frontier for dwell/bookkeeping

                    self.path = None
                    if self._goal_kind == "backtrack":
                        self.goal_xy = None
                        self._goal_kind = "none"
                        self._active_backtrack_anchor = None
                        self.nav_state = "IDLE"
                    else:
                        self._start_goal_dwell()
                    if not self.dwell_active:
                        self.goal_xy = None
                        self.nav_state = "IDLE"

                else:
                    if self.goal_xy is None:
                        self.nav_state = "IDLE"
                        return
                    progress_eps = float(self.get_parameter('follow_progress_eps').value)
                    stuck_timeout = float(self.get_parameter('follow_stuck_timeout_sec').value)
                    # Near goal, progress naturally becomes small; do not require
                    # large eps-sized jumps to count as progress.
                    eff_eps = min(progress_eps, max(0.01, 0.05 * d_goal))
                    path_progress_s = float(getattr(self.navigator, 'last_proj_s', 0.0))
                    heading_err = abs(float(getattr(self.navigator, 'last_heading_err', 0.0)))
                    path_advanced = path_progress_s > (self._last_path_progress_s + 0.08)
                    if (d_goal + eff_eps < self._best_goal_dist) or (d_goal + 0.01 < self._best_goal_dist):
                        self._best_goal_dist = d_goal
                        self._last_progress_time = now_sec
                    elif path_advanced:
                        self._last_path_progress_s = path_progress_s
                        self._last_progress_time = now_sec
                    if self.path is not None:
                        d_path_end = float(np.linalg.norm(robot_xy - self.path[-1]))
                        if now_sec - getattr(self, '_follow_log_t', 0.0) >= 1.0:
                            self._follow_log_t = now_sec
                            print(f"[FOLLOWING] d_goal={d_goal:.2f}m d_path_end={d_path_end:.2f}m path_wps={len(self.path)}")
                        if d_path_end < 0.20 and d_goal > 0.50:
                            # Reached the end of this partial path but goal is still far.
                            # Clear current path so update() picks up the next path_gen
                            # result, which will be generated from the new robot position.
                            print(f"[PATH EXTEND] partial path done, {d_goal:.2f}m to goal")
                            self.path = None
                            self.navigator.reset_path()
                            self.latest_candidate_path = None
                            self._path_gen_running = False
                            self.wait_start_time = now_sec
                            self.nav_state = "STABILIZING"
                            if not self._path_gen_running:
                                self._launch_path_gen(robot_xy, self.goal_xy)
                            return
                        path_done = self._controller_path_done
                        if path_done:
                            self._controller_path_done = False
                        if path_done:
                            if d_goal < 0.35:
                                self.get_logger().info(
                                    f"[FOLLOWING] Navigator reported path complete near goal "
                                    f"(d_goal={d_goal:.2f}m); treating as reached"
                                )
                                self.navigator.stop_robot()
                                self.navigator.reset_path()
                                if self.goal_xy is not None and self._goal_kind == "frontier":
                                    self._visited_goals.append(self.goal_xy.copy())
                                    if len(self._visited_goals) > 50:
                                        self._visited_goals.pop(0)
                                    if self.region is not None:
                                        self.exploration_graph.mark_goal_reached(
                                            self.region.version, self.goal_xy
                                        )
                                self.path = None
                                if self._goal_kind == "backtrack":
                                    self.goal_xy = None
                                    self._goal_kind = "none"
                                    self._active_backtrack_anchor = None
                                    self.nav_state = "IDLE"
                                else:
                                    self._start_goal_dwell()
                                if not self.dwell_active:
                                    self.goal_xy = None
                                    self.nav_state = "IDLE"
                                return
                            self.get_logger().warn(
                                f"[FOLLOWING] path endpoint reached but goal still {d_goal:.2f}m away; replanning"
                            )
                            self.path = None
                            self.navigator.reset_path()
                            self.latest_candidate_path = None
                            self._path_gen_running = False
                            self.wait_start_time = now_sec
                            self.nav_state = "STABILIZING"
                            return
                    adaptive_timeout = stuck_timeout * (2.0 if d_goal < 1.0 else 1.0)
                    if getattr(self.navigator, '_rotating', False) or heading_err > 0.8:
                        adaptive_timeout *= 2.0
                    if now_sec - self._last_progress_time > adaptive_timeout:
                        if not self.region.in_bound(robot_xy):
                            # Robot is outside region bound — _maybe_create_new_region_guarded
                            # will handle this. Suppress faulty stuck trigger.
                            self._last_progress_time = now_sec
                        else:
                            self.get_logger().warn(
                                f"[FOLLOWING] no goal progress for {adaptive_timeout:.1f}s "
                                f"(best={self._best_goal_dist:.2f}m current={d_goal:.2f}m); forcing replan"
                            )
                            self.path = None
                            self.navigator.reset_path()
                            self.latest_candidate_path = None
                            self._path_gen_running = False
                            self.wait_start_time = now_sec
                            self.nav_state = "STABILIZING"
                            self._last_progress_time = now_sec
                            return
                    # ── Periodically re-launch path gen to keep path fresh ───
                    self._path_refresh_ticks += 1
                    if self._path_refresh_ticks % 10 == 0 and not self._path_gen_running:
                        self._launch_path_gen(robot_xy, self.goal_xy)
                    # ── Mid-path replan: once, when halfway through a long path ─
                    if (self._path_total_dist is not None
                            and self._path_total_dist > 2.0
                            and not self._midpoint_replanned
                            and d_goal < self._path_total_dist * 0.5
                            and not self._path_gen_running):
                        self._midpoint_replanned = True
                        self._launch_path_gen(robot_xy, self.goal_xy)
                        self.get_logger().info(
                            f"[REPLAN] Halfway triggered at d_goal={d_goal:.2f}m "
                            f"(path was {self._path_total_dist:.2f}m)"
                        )
                    # ── Consume updated path from background thread ──────────
                    # Hot-swapping the tracked path every refresh makes the robot
                    # hunt between similar field paths. Only accept a refreshed
                    # path once the current one is nearly exhausted.
                    if (self.latest_candidate_path is not None
                            and self.path is not None
                            and d_path_end < 0.60
                            and not np.array_equal(self.latest_candidate_path, self.path)):
                        self.path = self.latest_candidate_path.copy()
                        self.latest_candidate_path = None
                        self.navigator.reset_path()

            elif self.nav_state == "DWELL":
                self.navigator.stop_robot()
                self._tick_goal_dwell(world_points_hit, robot_xy)

        # ── Composite plot ────────────────────────────────────────────────
        if bool(self.get_parameter('enable_composite_plot').value):
            if self._composite_step % 20 == 0:
                self._launch_background_task(
                    '_composite_plot_running',
                    lambda: self.plotter.save_composite_plot()
                )
        if bool(self.get_parameter('enable_occupancy_snapshot').value):
            if self._composite_step % 20 == 0:
                self._launch_background_task(
                    '_occupancy_snapshot_running',
                    lambda: self.plotter.save_occupancy_snapshot()
                )
        if self._composite_step % 200 == 0:
            save_sat = bool(
                self.get_parameter('save_satellites_in_periodic_map').value
            )
            self._launch_background_task(
                '_ros_map_save_running',
                lambda: self.plotter.save_ros_map(save_satellites=save_sat)
            )
        self._composite_step += 1

        # ── Always publish (every tick, every mode) ───────────────────────
        self.map_publisher.publish_maps()
        self.markers.publish_goal_marker()
        self.markers.publish_frontier_markers()
        self.markers.publish_doorway_markers()
        self.publish_field_path(self.path)

    def save_composite_plot(self):
        """
        plot2-style composite:
          - world-scale background (unknown / visited / obstacles)
          - per-region travel-time contours that PERSIST via a numeric cache (no translucent overlays)
          - exploration graph (nodes/edges/doors)
          - ONLY the latest saved frontiers per region (not the entire bank)
        """
        from matplotlib.lines import Line2D
        os.makedirs(self.folder, exist_ok=True)
        out_dir = os.path.join(self.folder, "plots2")
        os.makedirs(out_dir, exist_ok=True)

        # World-map bounds (gridmap is world-sized)
        b_map = [
            [self.gridmap.origin[0] - self.gridmap.map_size / 2,
             self.gridmap.origin[1] - self.gridmap.map_size / 2],
            [self.gridmap.origin[0] + self.gridmap.map_size / 2,
             self.gridmap.origin[1] + self.gridmap.map_size / 2],
        ]

        fig, ax = plt.subplots(1, 1, figsize=(12, 12))

        # ── Background: visited=dark green, obstacle=black, unknown=dark blue ──
        data = self.gridmap.data
        vis  = self.gridmap.visited
        n    = self.gridmap.n_cells
        cell = self.gridmap.cell_size

        bg = np.zeros((n, n, 3), dtype=np.uint8)
        bg[:, :] = [10, 10, 80]        # unknown
        bg[vis == 1]  = [220, 220, 40]     # visited free
        bg[data == 1] = [0, 0, 0]      # obstacles

        extent = [b_map[0][0], b_map[1][0], b_map[0][1], b_map[1][1]]
        ax.imshow(bg, origin='lower', extent=extent, zorder=0, aspect='auto')
        # ── Persistent numeric cache for region fields ─────────────────────────
        # Cache: version -> dict(X_w, Y_w, TT_masked, extent, src_world, step, epoch)
        if not hasattr(self, "_region_field_cache"):
            self._region_field_cache = {}

        # Tuning
        GRID_N = 90                 # resolution of per-region TT grid
        CONTOUR_LEVELS = np.arange(0, 20, 0.05)        # number of contour levels
        FRONTIER_LAST_K = 30        # only show last K saved frontiers per region
        CACHE_UPDATE_EVERY = 1      # update active region every step (keep expanding field visible)

        def _build_region_mask(b, grid_n=GRID_N):
            """Mask out cells outside visited/obstacle footprint (vectorized)."""
            i_arr = np.arange(grid_n)
            j_arr = np.arange(grid_n)
            wx = b[0, 0] + (i_arr + 0.5) / grid_n * (b[1, 0] - b[0, 0])  # (grid_n,) x coords
            wy = b[0, 1] + (j_arr + 0.5) / grid_n * (b[1, 1] - b[0, 1])  # (grid_n,) y coords
            gx = ((wx - b_map[0][0]) / cell).astype(int)   # column indices in gridmap
            gy = ((wy - b_map[0][1]) / cell).astype(int)   # row indices in gridmap
            # mask[j, i]: rows=j (y axis), cols=i (x axis)
            gx_grid, gy_grid = np.meshgrid(gx, gy)         # both (grid_n, grid_n)
            inb = (0 <= gx_grid) & (gx_grid < n) & (0 <= gy_grid) & (gy_grid < n)
            gx_s = np.clip(gx_grid, 0, n - 1)
            gy_s = np.clip(gy_grid, 0, n - 1)
            mask = np.zeros((grid_n, grid_n), dtype=bool)
            mask[inb] = (vis[gy_s[inb], gx_s[inb]] == 1) | (data[gy_s[inb], gx_s[inb]] == 1)
            return mask

        def _compute_tt_grid(region, src_world, b, mask, grid_n=GRID_N):
            """Compute TT(x0=src, x1=grid) over region bound and apply mask => TT_masked."""
            xmin, ymin = float(b[0, 0]), float(b[0, 1])
            xmax, ymax = float(b[1, 0]), float(b[1, 1])
            xs = np.linspace(xmin, xmax, grid_n)
            ys = np.linspace(ymin, ymax, grid_n)
            X_w, Y_w = np.meshgrid(xs, ys)

            # world -> normalized using region bound
            sx = (xmax - xmin) + 1e-9
            sy = (ymax - ymin) + 1e-9
            X_n = (X_w - xmin) / sx - 0.5
            Y_n = (Y_w - ymin) / sy - 0.5

            src_norm = region.normalize(src_world.reshape(1, 3))[0]  # (3,)

            XP = np.zeros((X_w.size, 2 * region.dim), dtype=np.float32)
            XP[:, :region.dim] = src_norm
            XP[:, region.dim + 0] = X_n.reshape(-1)
            XP[:, region.dim + 1] = Y_n.reshape(-1)
            XP[:, region.dim + 2] = src_norm[2]

            XP_t = torch.tensor(XP, dtype=torch.float32, device=region.device)

            with torch.no_grad():
                TT_t = region.function.TravelTimes(XP_t)
                if TT_t.ndim == 2 and TT_t.shape[1] == 1:
                    TT_t = TT_t[:, 0]
                TT = TT_t.detach().cpu().numpy().reshape(X_w.shape)

            TT_masked = TT.copy()
            TT_masked[~mask] = np.nan

            return X_w, Y_w, TT_masked

        def _pick_src_world(region, b):
            """Stable source for plotting TT contours."""
            # Active region uses robot position if available
            if self.region is not None and region.version == self.region.version and self.robot_pos is not None:
                return np.array([self.robot_pos[0], self.robot_pos[1], 0.0], dtype=np.float32)

            # Inactive: prefer cached source if present, else region center
            cached = self._region_field_cache.get(region.version)
            if cached is not None and "src_world" in cached:
                return cached["src_world"]

            rc = (b[0] + b[1]) / 2.0
            return np.array([float(rc[0]), float(rc[1]), 0.0], dtype=np.float32)

        # ── Update cache for the active region (so its field "expands" live) ──
        if self.region is not None and (self._composite_step % CACHE_UPDATE_EVERY == 0):
            try:
                reg = self.region
                if reg.function is not None:
                    b = reg.bound.detach().cpu().numpy()
                    mask = _build_region_mask(b, grid_n=GRID_N)
                    src_world = _pick_src_world(reg, b)
                    X_w, Y_w, TT_masked = _compute_tt_grid(reg, src_world, b, mask, grid_n=GRID_N)
                    self._region_field_cache[reg.version] = dict(
                        X_w=X_w, Y_w=Y_w, TT=TT_masked,
                        extent=[b[0, 0], b[1, 0], b[0, 1], b[1, 1]],
                        mask=mask, src_world=src_world,
                        step=int(self._composite_step),
                        epoch=int(getattr(reg, "epoch", 0)),
                    )
            except Exception as e:
                self.get_logger().warn(f"[COMPOSITE] active region cache update failed: {e}")

        # ── Ensure each region has at least one cached snapshot (persist) ─────
        for region in self.all_regions.values():
            if region.version in self._region_field_cache:
                continue
            # Only cache if model exists right now; otherwise it will remain missing until loaded once
            if getattr(region, "function", None) is None:
                continue
            try:
                b = region.bound.detach().cpu().numpy()
                mask = _build_region_mask(b, grid_n=GRID_N)
                src_world = _pick_src_world(region, b)
                X_w, Y_w, TT_masked = _compute_tt_grid(region, src_world, b, mask, grid_n=GRID_N)
                self._region_field_cache[region.version] = dict(
                    X_w=X_w, Y_w=Y_w, TT=TT_masked,
                    extent=[b[0, 0], b[1, 0], b[0, 1], b[1, 1]],
                    mask=mask, src_world=src_world,
                    step=int(self._composite_step),
                    epoch=int(getattr(region, "epoch", 0)),
                )
            except Exception as e:
                self.get_logger().warn(f"[COMPOSITE] region {region.version} initial cache failed: {e}")

        # ── Draw persistent contours for all cached regions ───────────────────
        for region in self.all_regions.values():
            b = region.bound.detach().cpu().numpy()

            cached = self._region_field_cache.get(region.version, None)
            if cached is not None:
                X_w = cached["X_w"]
                Y_w = cached["Y_w"]
                TT  = cached["TT"]

                if np.isfinite(TT).any():
                    ax.contour(
                        X_w, Y_w, TT,
                        levels=CONTOUR_LEVELS,
                        cmap="gist_heat",
                        linewidths=0.65,
                        alpha=0.9,
                        zorder=2
                    )

            # ── Region bounding box (always) ─────────────────────────────────
            rect_w = float(b[1, 0] - b[0, 0])
            rect_h = float(b[1, 1] - b[0, 1])
            is_active = (self.region is not None and region.version == self.region.version)
            edge_color = 'yellow' if is_active else 'gray'
            lw = 2.0 if is_active else 0.8

            rect = plt.Rectangle(
                (float(b[0, 0]), float(b[0, 1])), rect_w, rect_h,
                linewidth=lw, edgecolor=edge_color, facecolor='none', zorder=4
            )
            ax.add_patch(rect)
            ax.text(
                float(b[0, 0]) + 0.02, float(b[1, 1]) - 0.15,
                f"R{region.version}{'*' if is_active else ''}",
                color=edge_color, fontsize=7, zorder=5
            )

        # ── ExplorationGraph overlay: edges with waypoints ───────────────────
        # Each edge now carries 'waypoints' (free+visited cells) from
        # add_free_cell_edge.  We draw: center1 → wp1 → wp2 → ... → center2
        for v1, v2, edata in self.exploration_graph.graph.edges(data=True):
            rec1 = self.exploration_graph.records.get(v1)
            rec2 = self.exploration_graph.records.get(v2)
            if rec1 is None or rec2 is None:
                continue

            c1 = rec1.center
            c2 = rec2.center
            waypoints = edata.get('waypoints', [])

            # Build the polyline: c1 → sorted waypoints → c2
            # Sort waypoints from c1-side to c2-side for a clean visual chain
            if waypoints:
                wps = np.array(waypoints)
                # Sort by projection along c1→c2 axis
                axis = c2 - c1
                axis_len = np.linalg.norm(axis)
                if axis_len > 1e-6:
                    t_vals = (wps - c1[None, :]) @ axis / (axis_len ** 2)
                    order = np.argsort(t_vals)
                    wps = wps[order]
                chain_x = [c1[0]] + list(wps[:, 0]) + [c2[0]]
                chain_y = [c1[1]] + list(wps[:, 1]) + [c2[1]]
                ax.plot(chain_x, chain_y, color='cyan', linewidth=1.4,
                        linestyle='--', alpha=0.8, zorder=6)
                # Mark the waypoints themselves
                ax.scatter(wps[:, 0], wps[:, 1],
                           c='deepskyblue', s=25, marker='s',
                           zorder=7, alpha=0.9)
            else:
                # Edge exists but no waypoints yet (proximity bridge pending)
                ax.plot([c1[0], c2[0]], [c1[1], c2[1]],
                        color='cyan', linewidth=0.8,
                        linestyle=':', alpha=0.5, zorder=6)

        # ── ExplorationGraph nodes (region centers) ──────────────────────────
        for ver, rec in self.exploration_graph.records.items():
            c = rec.center
            has_f = rec.has_frontiers()
            node_color = 'lime' if has_f else 'red'
            ax.scatter(c[0], c[1], c=node_color, s=120,
                       marker='o', zorder=8, edgecolors='white', linewidths=1.0)
            ax.text(c[0] + 0.05, c[1] + 0.05,
                    f"R{ver}\n({len(rec.saved_frontiers)}f)",
                    color='white', fontsize=6, zorder=9,
                    bbox=dict(boxstyle='round,pad=0.15', fc='black', alpha=0.5))

        # ── Latest saved frontier points per region (NOT all) ─────────────────
        for ver, rec in self.exploration_graph.records.items():
            if not rec.saved_frontiers:
                continue
            pts = rec.saved_frontiers[-FRONTIER_LAST_K:]  # latest K (append order)
            for fp in pts:
                ax.scatter(fp[0], fp[1], c='orange', s=14,
                           marker='.', zorder=7, alpha=0.85)

        # ── Robot, trajectory, goal ─────────────────────────────────────────
        if self.robot_pos is not None:
            ax.scatter(self.robot_pos[0], self.robot_pos[1],
                       c='red', s=100, zorder=11, marker='o', label='Robot')

        if len(self._traj_history) > 1:
            traj = np.array(self._traj_history)
            ax.plot(traj[:, 0], traj[:, 1], 'm-', linewidth=1.5, zorder=9)

        if self.goal_xy is not None:
            ax.scatter(self.goal_xy[0], self.goal_xy[1],
                       c='lime', s=180, zorder=11, marker='*', label='Goal')

        # ── Legend ──────────────────────────────────────────────────────────
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='lime',
                   markersize=8, label='Region (has frontiers)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                   markersize=8, label='Region (exhausted)'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='deepskyblue',
                   markersize=7, label='Graph waypoint (free cell)'),
            Line2D([0], [0], color='cyan', linestyle='--', label='Graph edge (connected)'),
            Line2D([0], [0], color='cyan', linestyle=':', label='Graph edge (no waypoints yet)'),
            Line2D([0], [0], marker='.', color='w', markerfacecolor='orange',
                   markersize=6, label=f'Latest {FRONTIER_LAST_K} frontiers'),
            Line2D([0], [0], marker='*', color='w', markerfacecolor='lime',
                   markersize=10, label='Current goal'),
            Line2D([0], [0], marker='D', color='w', markerfacecolor='magenta',
                   markersize=7, label='Connection point (unvisited)'),
            Line2D([0], [0], marker='D', color='w', markerfacecolor='gray',
                   markersize=7, label='Connection point (visited)'),
        ]
        ax.legend(handles=legend_elements, loc='upper right',
                  fontsize=7, framealpha=0.7)

        # Fixed 20×20 world extent centred on first region spawn (stable across steps)
        if self._plot_world_center is not None:
            cx, cy = float(self._plot_world_center[0]), float(self._plot_world_center[1])
        else:
            cx = float(self.gridmap.origin[0])
            cy = float(self.gridmap.origin[1])
        ax.set_xlim(cx - 10.0, cx + 10.0)
        ax.set_ylim(cy - 10.0, cy + 10.0)
        ax.set_aspect('equal')
        ax.set_title(
            f"Step {self._composite_step}  |  regions={len(self.all_regions)}"
            f"  mode={self.mode}/{self.nav_state}"
        )

        fig.savefig(os.path.join(out_dir, f"plot_step_{self._composite_step}.png"), dpi=120)
        fig.savefig(os.path.join(self.folder, "plot2.png"))
        plt.close(fig)

    def save_ros_map(self, filepath_base: str = None, save_satellites: bool = False):
        """Save occupancy grid as ROS2 map_server PGM + YAML for AMCL localization."""
        import yaml as _yaml
        if filepath_base is None:
            filepath_base = os.path.join(self.folder, "ros_map", "map")
        os.makedirs(os.path.dirname(filepath_base), exist_ok=True)

        gm   = self.gridmap
        data = gm.data.copy()      # 1=occupied, 0=free-or-unknown
        vis  = gm.visited.copy()   # 1=confirmed free

        n    = gm.n_cells
        # PGM encoding: 254=free, 0=occupied, 205=unknown
        pgm  = np.full((n, n), 205, dtype=np.uint8)
        pgm[(data == 0) & (vis == 1)] = 254   # confirmed free
        pgm[data == 1]                = 0     # occupied
        # map_server expects row 0 = bottom of world → flip vertically
        pgm = np.flipud(pgm)

        pgm_path  = filepath_base + ".pgm"
        yaml_path = filepath_base + ".yaml"

        with open(pgm_path, 'wb') as f:
            header = f"P5\n{n} {n}\n255\n"
            f.write(header.encode())
            f.write(pgm.tobytes())

        origin_x = float(gm.origin[0] - gm.map_size / 2.0)
        origin_y = float(gm.origin[1] - gm.map_size / 2.0)
        meta = {
            'image':           os.path.basename(pgm_path),
            'resolution':      float(gm.cell_size),
            'origin':          [origin_x, origin_y, 0.0],
            'negate':          0,
            'occupied_thresh': 0.65,
            'free_thresh':     0.196,
        }
        with open(yaml_path, 'w') as f:
            _yaml.dump(meta, f, default_flow_style=False)

        self.get_logger().info(f"[MAP SAVED] {pgm_path}")
        # Optional: save loaded satellite region weights.
        # Default False for periodic map dumps to avoid repeatedly saving
        # newly-created, undertrained regions.
        if save_satellites:
            for ver, reg in self.all_regions.items():
                if self.region is not None and ver == self.region.version:
                    continue
                try:
                    if getattr(reg, 'loaded', True):
                        reg.save_model()
                except Exception as e:
                    self.get_logger().warn(f"[MAP SAVED] Could not save satellite R{ver}: {e}")

    def _detect_doorways(self):
        """Scan the occupancy grid for doorway-like gaps; update self._connection_points.

        Four-filter approach:
          1. Gap width in [0.6m, 1.0m], bounded by wall cells on both ends.
          2. Wall continues check: the bounding wall cells have wall neighbours
             extending OUTWARD (away from the gap) ≥ 0.8m in the same scan direction.
             Rejects corridor openings where the "wall" is the outer room boundary
             with nothing beyond it.
          3. Free-on-both-sides check: at the gap centre, both sides in the
             normal direction must have ≥ 0.8 m of free space.
             Rejects wall indentations / shallow recesses (one side hits solid wall).
          4. Connected-component validation: blocking the gap must disconnect the
             two free-space regions on either side. Rejects round obstacles whose
             curved walls create concave gaps that pass filters 1–3 but are not
             true doorways (both sides remain connected via paths around the object).
        """
        gm      = self.gridmap
        if not gm.has_slam_semantics():
            self._connection_points = []
            return
        occupied = gm.slam_occupied.astype(np.uint8)
        free     = gm.slam_free.astype(np.uint8)
        known    = gm.slam_known.astype(np.uint8)
        n       = gm.n_cells
        cell    = gm.cell_size

        DOOR_MIN   = max(2, int(round(0.6 / cell)))
        DOOR_MAX   = max(DOOR_MIN, int(round(1.0 / cell)))
        EXT        = max(2, int(round(0.8 / cell)))   # min wall extension cells
        FREE_DEPTH = max(3, int(round(0.8 / cell)))   # min free cells on each side

        def _free_run(arr_free, arr_known, start, step):
            """Count consecutive known-free cells from start in the given direction."""
            k = start + step
            run = 0
            while 0 <= k < len(arr_free):
                if arr_known[k] != 1 or arr_free[k] != 1:
                    break
                run += 1
                k += step
            return run

        def _cc_validates_door(temp_free, gap_rows, gap_cols, side_a, side_b):
            """Return True if blocking the gap disconnects the two sides.

            temp_free : copy of slam_free (will be temporarily modified in-place)
            gap_rows, gap_cols : arrays of grid indices for the gap cells to block
            side_a, side_b : (row, col) tuples for sample cells on each side
            """
            # Block the gap
            orig = temp_free[gap_rows, gap_cols].copy()
            temp_free[gap_rows, gap_cols] = 0

            _, labels = cv2.connectedComponents(temp_free.astype(np.uint8), connectivity=4)

            ra, ca = side_a
            rb, cb = side_b
            valid_a = (0 <= ra < n and 0 <= ca < n and temp_free[ra, ca] == 1)
            valid_b = (0 <= rb < n and 0 <= cb < n and temp_free[rb, cb] == 1)
            if valid_a and valid_b:
                la = labels[ra, ca]
                lb = labels[rb, cb]
                result = (la != 0 and lb != 0 and la != lb)
            else:
                result = False

            # Restore
            temp_free[gap_rows, gap_cols] = orig
            return result

        # Working copy for CC validation (reused across candidates)
        free_work = free.copy()

        cps = []

        # ── Row scan: gap runs along columns (x), normal is ±y ──────────────────
        for i in range(1, n - 1):
            j = 1
            while j < n - 1:
                if free[i, j] != 1:
                    j += 1
                    continue
                j_start = j
                while j < n and free[i, j] == 1:
                    j += 1
                j_end = j - 1   # inclusive

                span = j_end - j_start + 1
                if span < DOOR_MIN or span > DOOR_MAX:
                    continue
                if j_start == 0 or j_end >= n - 1:
                    continue
                j_left, j_right = j_start - 1, j_end + 1
                if occupied[i, j_left] != 1 or occupied[i, j_right] != 1:
                    continue
                if not np.all(known[i, j_start:j_end + 1] == 1):
                    continue
                # Filter 2: wall continues outward in the same row
                left_ext  = np.any(occupied[i, max(0, j_left - EXT):j_left] == 1)
                right_ext = np.any(occupied[i, j_right + 1:min(n, j_right + 1 + EXT)] == 1)
                if not (left_ext and right_ext):
                    continue
                # Filter 3: known free space ≥ FREE_DEPTH on both sides in normal (y) direction
                j_mid   = (j_start + j_end) // 2
                f_above = _free_run(free[:, j_mid], known[:, j_mid], i, +1)
                f_below = _free_run(free[:, j_mid], known[:, j_mid], i, -1)
                if f_above < FREE_DEPTH or f_below < FREE_DEPTH:
                    continue

                # Filter 4: CC validation — blocking the gap must disconnect the two sides
                gap_rows = np.full(j_end - j_start + 1, i, dtype=int)
                gap_cols = np.arange(j_start, j_end + 1, dtype=int)
                side_a = (i + FREE_DEPTH, j_mid)
                side_b = (i - FREE_DEPTH, j_mid)
                if not _cc_validates_door(free_work, gap_rows, gap_cols, side_a, side_b):
                    continue

                wx, wy = gm._grid_to_world(i, j_mid)
                cps.append({
                    "pos":     [wx, wy],
                    "normal":  [0.0, 1.0],
                    "width":   span * cell,
                    "kind":    "door",
                    "visited": False,
                })

        # ── Column scan: gap runs along rows (y), normal is ±x ──────────────────
        for j in range(1, n - 1):
            i = 1
            while i < n - 1:
                if free[i, j] != 1:
                    i += 1
                    continue
                i_start = i
                while i < n and free[i, j] == 1:
                    i += 1
                i_end = i - 1

                span = i_end - i_start + 1
                if span < DOOR_MIN or span > DOOR_MAX:
                    continue
                if i_start == 0 or i_end >= n - 1:
                    continue
                i_bot, i_top = i_start - 1, i_end + 1
                if occupied[i_bot, j] != 1 or occupied[i_top, j] != 1:
                    continue
                if not np.all(known[i_start:i_end + 1, j] == 1):
                    continue
                # Filter 2: wall continues outward in the same column
                bot_ext = np.any(occupied[max(0, i_bot - EXT):i_bot, j] == 1)
                top_ext = np.any(occupied[i_top + 1:min(n, i_top + 1 + EXT), j] == 1)
                if not (bot_ext and top_ext):
                    continue
                # Filter 3: known free space ≥ FREE_DEPTH on both sides in normal (x) direction
                i_mid    = (i_start + i_end) // 2
                f_right  = _free_run(free[i_mid, :], known[i_mid, :], j, +1)
                f_left   = _free_run(free[i_mid, :], known[i_mid, :], j, -1)
                if f_right < FREE_DEPTH or f_left < FREE_DEPTH:
                    continue

                # Filter 4: CC validation — blocking the gap must disconnect the two sides
                gap_rows = np.arange(i_start, i_end + 1, dtype=int)
                gap_cols = np.full(i_end - i_start + 1, j, dtype=int)
                side_a = (i_mid, j + FREE_DEPTH)
                side_b = (i_mid, j - FREE_DEPTH)
                if not _cc_validates_door(free_work, gap_rows, gap_cols, side_a, side_b):
                    continue

                wx, wy = gm._grid_to_world(i_mid, j)
                cps.append({
                    "pos":     [wx, wy],
                    "normal":  [1.0, 0.0],
                    "width":   span * cell,
                    "kind":    "door",
                    "visited": False,
                })

        # ── NMS: within 1.5m keep the narrower gap; within 0.3m always deduplicate ─
        # Sort by width ascending so narrower gaps are preferred
        cps.sort(key=lambda c: c["width"])
        merged = []
        for cp in cps:
            pos = np.array(cp["pos"])
            dominated = False
            for m in merged:
                d = np.linalg.norm(pos - np.array(m["pos"]))
                if d < 0.3:
                    dominated = True   # same doorway, already kept narrower
                    break
                if d < 1.5 and cp["width"] >= m["width"]:
                    dominated = True   # wider duplicate nearby
                    break
            if not dominated:
                merged.append(cp)

        prev_count = len(getattr(self, '_connection_points', []))
        self._connection_points = merged
        if len(merged) != prev_count:
            self.get_logger().info(
                f"[DOORWAYS] detected {len(merged)} doorway(s): "
                + ", ".join(f"({c['pos'][0]:.1f},{c['pos'][1]:.1f}) w={c['width']:.2f}m"
                            for c in merged))

    def _save_topology(self):
        """Persist region bounds + graph connectivity for downstream navigation/debug."""
        topo = {
            "regions": {},
            "connection_points": [],
            "graph_edges": [],
        }

        # Region bounds/centers
        for ver, reg in self.all_regions.items():
            try:
                b = reg.bound.detach().cpu().numpy()
                topo["regions"][str(ver)] = {
                    "center": [float(reg.center[0]), float(reg.center[1])],
                    "bound": [[float(b[0, 0]), float(b[0, 1])],
                              [float(b[1, 0]), float(b[1, 1])]],
                }
            except Exception:
                continue

        # Explicit connection points (door/corridor candidates)
        cp_seen = []
        for cp in getattr(self, "_connection_points", []):
            try:
                pos = np.asarray(cp["pos"], dtype=np.float32)
            except Exception:
                continue
            if any(np.linalg.norm(pos - prev) < 0.2 for prev in cp_seen):
                continue
            topo["connection_points"].append({
                "pos": [float(pos[0]), float(pos[1])],
                "normal": [float(cp.get("normal", [0.0, 0.0])[0]),
                           float(cp.get("normal", [0.0, 0.0])[1])],
                "width": float(cp.get("width", 0.0)),
                "kind": str(cp.get("kind", "door")),
                "visited": bool(cp.get("visited", False)),
            })
            cp_seen.append(pos)

        # Graph edges + canonical waypoints
        g = getattr(self.exploration_graph, "graph", None)
        if g is not None:
            for v1, v2, edata in g.edges(data=True):
                cp = edata.get("cp")
                cp_json = None
                if cp is not None:
                    cp = np.asarray(cp, dtype=np.float32)
                    cp_json = [float(cp[0]), float(cp[1])]
                    if not any(np.linalg.norm(cp - prev) < 0.2 for prev in cp_seen):
                        topo["connection_points"].append({
                            "pos": cp_json,
                            "normal": [0.0, 0.0],
                            "width": 0.0,
                            "kind": str(edata.get("edge_type", "free_cell")),
                            "visited": True,
                        })
                        cp_seen.append(cp)

                waypoints = []
                for wp in edata.get("waypoints", []):
                    try:
                        w = np.asarray(wp, dtype=np.float32)
                        waypoints.append([float(w[0]), float(w[1])])
                    except Exception:
                        continue

                topo["graph_edges"].append({
                    "v1": int(v1),
                    "v2": int(v2),
                    "edge_type": str(edata.get("edge_type", "free_cell")),
                    "cp": cp_json,
                    "waypoints": waypoints,
                })

        path = os.path.join(self.model.folder, "topology.json")
        try:
            with open(path, "w") as f:
                json.dump(topo, f, indent=2)
            self.get_logger().info(f"[TOPOLOGY] Saved to {path}")
        except Exception as e:
            self.get_logger().warn(f"[TOPOLOGY] Save failed: {e}")

    def save_occupancy_snapshot(self, tag=None):
        """
        Saves occupancy grid + visited + frontier as image and numpy.
        """
        out_dir = os.path.join(self.folder, "occupancy_snapshots")
        os.makedirs(out_dir, exist_ok=True)

        if tag is None:
            tag = f"step_{self._composite_step:06d}"

        data = self.gridmap.data.copy()
        visited = self.gridmap.visited.copy()
        frontier = self.gridmap.frontier.copy()

        n = self.gridmap.n_cells

        # ---- Create visualization image ----
        img = np.zeros((n, n, 3), dtype=np.uint8)

        # unknown
        img[:, :] = [10, 10, 80]

        # visited free
        img[visited == 1] = [200, 200, 0]   # yellow (clearer than dark green)

        # occupied
        img[data == 1] = [0, 0, 0]

        # frontier overlay (optional highlight)
        img[frontier == 1] = [255, 140, 0]  # orange

        plt.figure(figsize=(8, 8))
        plt.imshow(img, origin="lower")
        plt.title(f"Occupancy Snapshot {tag}")
        plt.axis("off")
        plt.savefig(os.path.join(out_dir, f"{tag}.png"), dpi=150, bbox_inches="tight")
        plt.close()

        # ---- Save raw arrays ----
        np.savez_compressed(
            os.path.join(out_dir, f"{tag}.npz"),
            occupancy=data,
            visited=visited,
            frontier=frontier
        )

        #self.get_logger().info(f"[OCC SNAPSHOT SAVED] {tag}")

    # ============ DEAD CODE (never called) ============

    # Last reviewed as dead: 2026-03-17
    def start_region_warmup(self):
        if self.region is None:
            return
        self._enter_warmup(
            self.region,
            self.warmup_epochs_needed,
            f"[WARMUP] Starting region {self.region.version}"
        )

    # Last reviewed as dead: 2026-03-17
    def _has_clear_los(self, robot_xy: np.ndarray, target_xy: np.ndarray) -> bool:
        """Vectorised ray-cast: returns True if no obstacle lies on the straight
        line from robot_xy to target_xy in the occupancy grid."""
        gm = self.gridmap
        ox = gm.origin[0] - gm.map_size / 2.0
        oy = gm.origin[1] - gm.map_size / 2.0
        dist = float(np.linalg.norm(target_xy - robot_xy))
        if dist < 0.01:
            return True
        n_samples = max(5, int(dist / gm.cell_size) + 2)
        ts  = np.linspace(0.0, 1.0, n_samples)
        pts = robot_xy[None] + ts[:, None] * (target_xy - robot_xy)[None]
        xi  = ((pts[:, 0] - ox) / gm.cell_size).astype(int)
        yi  = ((pts[:, 1] - oy) / gm.cell_size).astype(int)
        inb = (0 <= xi) & (xi < gm.n_cells) & (0 <= yi) & (yi < gm.n_cells)
        if not inb.all():
            return False   # ray exits mapped area → treat as blocked
        return not gm.data[yi, xi].any()

    # Last reviewed as dead: 2026-03-17
    def _sample_occupied(self, world_xy: np.ndarray) -> bool:
        gm = self.gridmap
        ox = gm.origin[0] - gm.map_size / 2.0
        oy = gm.origin[1] - gm.map_size / 2.0
        gx = int((world_xy[0] - ox) / gm.cell_size)
        gy = int((world_xy[1] - oy) / gm.cell_size)
        if not (0 <= gx < gm.n_cells and 0 <= gy < gm.n_cells):
            return False
        return bool(gm.data[gy, gx])

    # Last reviewed as dead: 2026-03-17
    def _occ_astar(self, start_xy: np.ndarray, goal_xy: np.ndarray) -> 'np.ndarray | None':
        """
        A* path on the occupancy grid through confirmed free+visited cells.
        Guaranteed wall-safe — result does NOT need _trim_at_unvisited or wall-rejection.
        Returns Nx2 float32 world-coord array, or None if no path exists.
        """
        import skimage.graph as skg
        import cv2 as _cv2

        gm  = self.gridmap
        ox  = gm.origin[0] - gm.map_size / 2.0
        oy  = gm.origin[1] - gm.map_size / 2.0
        n   = gm.n_cells

        def w2g(xy):
            gx = int((xy[0] - ox) / gm.cell_size)
            gy = int((xy[1] - oy) / gm.cell_size)
            return (gy, gx)   # (row, col)

        s_rc = w2g(start_xy)
        g_rc = w2g(goal_xy)
        for rc in (s_rc, g_rc):
            if not (0 <= rc[0] < n and 0 <= rc[1] < n):
                return None

        # Inflate obstacles by ROBOT_CELLS for clearance
        ROBOT_CELLS = 3
        kernel = _cv2.getStructuringElement(
            _cv2.MORPH_ELLIPSE, (2 * ROBOT_CELLS + 1, 2 * ROBOT_CELLS + 1))
        inflated = _cv2.dilate(gm.data.astype(np.uint8), kernel, iterations=1)

        # Cost: 1.0 for free+visited, 1e9 for wall/unvisited/inflated
        cost = np.full((n, n), 1e9, dtype=np.float64)
        cost[(inflated == 0) & (gm.visited > 0)] = 1.0
        # Always allow start/goal regardless of visited status (robot IS there)
        cost[s_rc[0], s_rc[1]] = 1.0
        cost[g_rc[0], g_rc[1]] = 1.0

        try:
            indices, _ = skg.route_through_array(
                cost, s_rc, g_rc, fully_connected=True, geometric=True)
        except Exception:
            return None

        if len(indices) < 2:
            return None

        cells = np.array(indices, dtype=float)        # (N, 2): (row, col)
        wx = ox + (cells[:, 1] + 0.5) * gm.cell_size
        wy = oy + (cells[:, 0] + 0.5) * gm.cell_size
        return np.column_stack([wx, wy]).astype(np.float32)

    # Last reviewed as dead: 2026-03-17
    def _abandon_current_goal(self, now_sec: float, reason: str):
        if self.goal_xy is not None:
            if self._goal_kind == "frontier":
                self._visited_goals.append(self.goal_xy.copy())
                if len(self._visited_goals) > 50:
                    self._visited_goals.pop(0)
            msg = (f"{reason} at ({self.goal_xy[0]:.2f},{self.goal_xy[1]:.2f}) "
                   f"[mode={self.mode}, nav={self.nav_state}]")
            if self.mode == "EXPLORE" and self.nav_state == "FOLLOWING":
                self.get_logger().warn(f"[GOAL ABANDONED] {msg}")
            else:
                self.get_logger().info(f"[GOAL RESET] {msg}")
        self.navigator.stop_robot()
        self.navigator.reset_path()
        self.path = None
        self.latest_candidate_path = None
        self._path_gen_running = False
        self._path_total_dist = None
        self._midpoint_replanned = False
        self._path_refresh_ticks = 0
        self._best_goal_dist = float("inf")
        self._last_progress_time = now_sec
        self._last_path_progress_s = 0.0
        self.goal_xy = None
        self._goal_kind = "none"
        self._active_backtrack_anchor = None
        self.nav_state = "IDLE"


def main():
    rclpy.init()
    node = GridmapNode()
    try:
        rclpy.spin(node)
    finally:
        try:
            node.plotter.save_ros_map(save_satellites=True)
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
