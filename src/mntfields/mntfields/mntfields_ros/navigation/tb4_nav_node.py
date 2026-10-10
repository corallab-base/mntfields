#!/usr/bin/env python3
"""
tb4_nav_node.py

Real-TurtleBot4 navigation node — adapts NavigationNode for physical hardware.

Changes vs. navigation_node.py:
  - use_sim_time=False
  - BEST_EFFORT QoS for /odom
  - cbreak keyboard loop (Enter to drive, 'c' to stop)
  - global_frame defaults to 'odom'
  - conservative max_lin=0.12, max_ang=0.6

State machine:
  WAITING_FOR_POSE → IDLE ↔ PLANNING → READY → DRIVING → IDLE

Usage:
    ros2 run mntfields tb4_nav_node \
        --ros-args \
        -p model_dir:=/home/mayank/turtlebot3_ws/mntfields_model \
        -p map_yaml:=/home/mayank/turtlebot3_ws/mntfields_model/mntfields_output/ros_map/map.yaml \
        -p global_frame:=odom
"""

import math
import os
import glob
import json
import select
import sys
import termios
import threading
import time as _time
import tty
import yaml

import numpy as np
import rclpy
import rclpy.parameter
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
import torch
from PIL import Image

from geometry_msgs.msg import Twist, PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import LaserScan, PointCloud2, PointField
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from mntfields.mntfields_core.core.region import Region
from mntfields.mntfields_ros.mapping.navigation import Navigator


# ── Dummy stubs required by Region.__init__ ──────────────────────────────────

class DummyRenderer:
    def render(self, *a, **kw):
        return None


class DummyModel:
    def __init__(self, model_dir: str):
        self.folder   = model_dir
        self.renderer = "cuda:0"   # must be string, not object
        self.h        = 0.0


# ─────────────────────────────────────────────────────────────────────────────

class TB4NavNode(Node):

    # ── BEST_EFFORT QoS override for /odom ───────────────────────────────────

    def create_subscription(self, msg_type, topic, callback, qos_profile, *args, **kwargs):
        if topic == "/odom":
            qos_profile = QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=10,
                reliability=ReliabilityPolicy.BEST_EFFORT,
            )
        return super().create_subscription(
            msg_type, topic, callback, qos_profile, *args, **kwargs)

    def __init__(self):
        super().__init__('tb4_nav_node')

        # ── Force real-robot time ─────────────────────────────────────────
        self.set_parameters([
            rclpy.parameter.Parameter(
                'use_sim_time', rclpy.Parameter.Type.BOOL, False),
        ])

        # ── Parameters ───────────────────────────────────────────────────────
        self.declare_parameter('model_dir',          '/home/mayank/turtlebot3_ws/mntfields_model')
        self.declare_parameter('max_lin',            0.12)   # conservative for real hardware
        self.declare_parameter('max_ang',            0.6)
        self.declare_parameter('goal_reached_dist',  0.25)
        self.declare_parameter('step_size',          0.10)
        self.declare_parameter('max_steps',          400)
        self.declare_parameter('replan_at_halfway',  True)
        self.declare_parameter('map_yaml',           '')
        self.declare_parameter('sdf_repo_dir',          '/home/mayank/comp-sdf-dyn-nav')
        self.declare_parameter('lidar_scan_topic',      '/scan')
        self.declare_parameter('replan_interval_sec',   0.15)
        self.declare_parameter('max_traj_pts',          8)
        self.declare_parameter('max_ahead_dist',        1.75)
        self.declare_parameter('opt_iterations',        12)
        self.declare_parameter('z_value',               0.225)
        self.declare_parameter('smoothing',             0.25)
        self.declare_parameter('trunc_dist_sdf',        0.05)
        self.declare_parameter('max_result_age_sec',    0.75)
        self.declare_parameter('max_robot_motion_m',    0.20)
        self.declare_parameter('min_publish_delta_m',   0.08)
        self.declare_parameter('local_seed_valid_sec',  1.0)
        self.declare_parameter('status_log_sec',        2.0)
        self.declare_parameter('global_frame',          'odom')  # real robot uses odom

        self.model_dir = self.get_parameter('model_dir').value
        self.nav_frame = str(self.get_parameter('global_frame').value)
        self._lidar_scan_topic = str(self.get_parameter('lidar_scan_topic').value)

        # ── State ─────────────────────────────────────────────────────────
        self.robot_pos: np.ndarray | None = None   # [x, y, z]
        self.robot_yaw: float             = 0.0
        self._amcl_received               = False

        self.goal_xy:   np.ndarray | None = None
        self.path:      np.ndarray | None = None   # Nx2 active path
        self.state: str                   = 'WAITING_FOR_POSE'

        self._path_gen_running   = False
        self._motion_enabled     = False
        self._sdf_opt_running    = False

        # Mid-path replan state
        self._path_total_dist    = None
        self._midpoint_replanned = False

        # Actual trajectory history
        self._robot_path_pts: list = []

        # Region management
        self.all_regions: dict[int, Region] = {}
        self.active_region: 'Region | None' = None
        self._map_occ: np.ndarray | None = None
        self._map_resolution: float | None = None
        self._map_origin: tuple[float, float] | None = None
        self._scan_msg: LaserScan | None = None
        self._sdf_lock = threading.Lock()
        self._sdf_available = False
        self._traj_opt = None
        self._sdf_device = None
        self._last_sdf_replan_time = 0.0
        self._last_sdf_local_path: np.ndarray | None = None
        self._last_sdf_local_path_time = 0.0
        self._last_status_log = 0.0
        self._logged_pc_frame = False

        # ── Load all saved regions ─────────────────────────────────────────
        self._load_all_regions()
        self._load_static_map()
        self._load_topology()

        # ── Publishers ───────────────────────────────────────────────────
        self.cmd_pub        = self.create_publisher(Twist,       '/cmd_vel',       10)
        self.path_pub       = self.create_publisher(Path,        '/nav_path',      10)
        self.robot_path_pub = self.create_publisher(Path,        '/robot_path',    10)
        self.marker_pub     = self.create_publisher(MarkerArray, '/nav_markers',   10)
        self.sdf_path_pub   = self.create_publisher(Path,        '/sdf_local_path', 10)
        self.sdf_debug_pc_pub = self.create_publisher(PointCloud2, '/sdf_points_debug', 10)

        self._tf_buf = Buffer()
        self._tf_listener = TransformListener(self._tf_buf, self)

        # ── Navigator (reuses same interface as gridmap_node) ─────────────
        self.navigator = Navigator(self)

        # ── Subscriptions ────────────────────────────────────────────────
        self.create_subscription(
            PoseWithCovarianceStamped, '/amcl_pose', self._amcl_pose_cb, 10)
        self.create_subscription(
            Odometry, '/odom', self._odom_cb, 10)
        self.create_subscription(
            LaserScan, self._lidar_scan_topic, self._scan_cb, qos_profile_sensor_data)
        self.create_subscription(
            PoseWithCovarianceStamped, '/initialpose', self._initial_pose_cb, 10)
        self.create_subscription(
            PoseStamped, '/goal_pose', self._goal_pose_cb, 10)
        self.create_subscription(Path, '/sdf_local_path', self._sdf_path_cb, 10)
        self._sdf_path: np.ndarray | None = None
        self._sdf_path_time: float = 0.0

        # ── Update timer (10 Hz) ─────────────────────────────────────────
        self.dt = 0.10
        self.create_timer(self.dt, self._update)

        threading.Thread(
            target=self._init_sdf_optimizer,
            args=(str(self.get_parameter('sdf_repo_dir').value),),
            daemon=True,
        ).start()

        # ── Keyboard thread (cbreak: Enter to drive, 'c' to stop) ────────
        if sys.stdin.isatty():
            threading.Thread(target=self._keyboard_loop, daemon=True).start()
        else:
            self.get_logger().warn(
                '[TB4NAV] stdin not a TTY — keyboard control disabled.')

        self.get_logger().info(
            f'[TB4NAV] Started. Loaded {len(self.all_regions)} region(s). '
            f'Waiting for /amcl_pose or /odom then 2D Goal Pose in RViz. '
            f'Press ENTER to drive, "c" to stop.'
        )

    # ── BEST_EFFORT /odom + cbreak keyboard ──────────────────────────────────

    def _keyboard_loop(self):
        fd = sys.stdin.fileno()
        old_attrs = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while True:
                ready, _, _ = select.select([sys.stdin], [], [], 0.2)
                if not ready:
                    continue
                ch = sys.stdin.read(1)
                if ch in ("\n", "\r"):
                    if self.state == 'READY':
                        self._motion_enabled = True
                        self.state = 'DRIVING'
                        self.get_logger().info('[TB4NAV] Motion enabled — driving to goal.')
                    else:
                        self.get_logger().info(
                            f'[TB4NAV] ENTER pressed in state={self.state}')
                elif ch in ("c", "C"):
                    self.navigator.stop_robot()
                    self.navigator.reset_path()
                    self.goal_xy = None
                    self.path = None
                    self._motion_enabled = False
                    self.state = 'IDLE'
                    self.get_logger().info('[TB4NAV] Navigation stopped by keyboard.')
        except Exception as e:
            self.get_logger().warn(f'[TB4NAV] Keyboard loop stopped: {e}')
        finally:
            try:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)
            except Exception:
                pass

    # ── Region loading ────────────────────────────────────────────────────────

    def _load_all_regions(self):
        if not os.path.isdir(self.model_dir):
            self.get_logger().error(f'[TB4NAV] model_dir not found: {self.model_dir}')
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
                self.get_logger().warn(f'[TB4NAV] No region metadata for region {ver}, skipping.')
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
                    f'[TB4NAV] Loaded region {ver} '
                    f'bound=[{bound[0,0]:.2f},{bound[0,1]:.2f}]'
                    f'→[{bound[1,0]:.2f},{bound[1,1]:.2f}] '
                    f'epoch={region.epoch}'
                )
            except Exception as e:
                self.get_logger().error(f'[TB4NAV] Failed loading region {ver}: {e}')

        if not self.all_regions:
            self.get_logger().warn(
                '[TB4NAV] No regions loaded — run exploration first and check model_dir.'
            )

    def _region_for_pos(self, xy: np.ndarray) -> 'Region | None':
        for region in self.all_regions.values():
            b = region.bound.detach().cpu().numpy()
            if b[0, 0] <= xy[0] <= b[1, 0] and b[0, 1] <= xy[1] <= b[1, 1]:
                return region
        return None

    def _load_topology(self):
        import json
        import networkx as nx
        topo_path = os.path.join(self.model_dir, "topology.json")
        self._topo_graph = nx.Graph()
        if not os.path.isfile(topo_path):
            self.get_logger().warn(
                '[TB4NAV] No topology.json — cross-region routing disabled'
            )
            self._connection_points = []
            self._topo_edges = []
            return
        with open(topo_path) as f:
            topo = json.load(f)
        self._connection_points = topo.get("connection_points", [])
        self._topo_edges = topo.get("graph_edges", [])
        for v_str, rdata in topo.get("regions", {}).items():
            self._topo_graph.add_node(int(v_str),
                                      center=rdata['center'], bound=rdata['bound'])
        for e in self._topo_edges:
            cp = e.get('cp')
            self._topo_graph.add_edge(
                e['v1'], e['v2'],
                waypoints=e.get('waypoints', []),
                edge_type=e.get('edge_type', 'free_cell'),
                cp=(np.asarray(cp, dtype=np.float32) if cp is not None else None),
            )
        self.get_logger().info(
            f'[TB4NAV] Topology: {self._topo_graph.number_of_nodes()} regions, '
            f'{self._topo_graph.number_of_edges()} edges, '
            f'{len(self._connection_points)} CPs'
        )

    def _load_static_map(self):
        map_yaml = str(self.get_parameter('map_yaml').value or '').strip()
        if not map_yaml:
            return
        if not os.path.isfile(map_yaml):
            self.get_logger().warn(f'[TB4NAV] map_yaml not found: {map_yaml}')
            return
        try:
            with open(map_yaml, 'r', encoding='utf-8') as f:
                meta = yaml.safe_load(f)
            image_path = meta.get('image')
            if not os.path.isabs(image_path):
                image_path = os.path.join(os.path.dirname(map_yaml), image_path)
            img = Image.open(image_path).convert('L')
            arr = np.asarray(img, dtype=np.float32) / 255.0
            negate = int(meta.get('negate', 0))
            occ_prob = arr if negate else (1.0 - arr)
            occ_thresh = float(meta.get('occupied_thresh', 0.65))
            self._map_occ = occ_prob >= occ_thresh
            self._map_resolution = float(meta['resolution'])
            origin = meta.get('origin', [0.0, 0.0, 0.0])
            self._map_origin = (float(origin[0]), float(origin[1]))
            self.get_logger().info(
                f'[TB4NAV] Loaded static map {self._map_occ.shape[1]}x{self._map_occ.shape[0]} '
                f'@ {self._map_resolution:.3f}m'
            )
        except Exception as e:
            self.get_logger().warn(f'[TB4NAV] Failed to load static map: {e}')
            self._map_occ = None
            self._map_resolution = None
            self._map_origin = None

    def _map_indices(self, xy: np.ndarray) -> tuple[int, int] | None:
        if self._map_occ is None or self._map_resolution is None or self._map_origin is None:
            return None
        mx = int(math.floor((float(xy[0]) - self._map_origin[0]) / self._map_resolution))
        my = int(math.floor((float(xy[1]) - self._map_origin[1]) / self._map_resolution))
        h, w = self._map_occ.shape
        row = h - 1 - my
        col = mx
        if not (0 <= row < h and 0 <= col < w):
            return None
        return row, col

    def _point_in_collision(self, xy: np.ndarray, pad_cells: int = 1) -> bool:
        idx = self._map_indices(xy)
        if idx is None:
            return False
        row, col = idx
        h, w = self._map_occ.shape
        r0 = max(0, row - pad_cells)
        r1 = min(h - 1, row + pad_cells)
        c0 = max(0, col - pad_cells)
        c1 = min(w - 1, col + pad_cells)
        return bool(np.any(self._map_occ[r0:r1 + 1, c0:c1 + 1]))

    def _segment_hits_wall(self, p0: np.ndarray, p1: np.ndarray, step: float = 0.05) -> bool:
        if self._map_occ is None:
            return False
        dist = float(np.linalg.norm(p1 - p0))
        n = max(2, int(math.ceil(dist / step)))
        for alpha in np.linspace(0.0, 1.0, n):
            pt = p0 + alpha * (p1 - p0)
            if self._point_in_collision(pt):
                return True
        return False

    def _path_hits_wall(self, path_np: np.ndarray) -> bool:
        if self._map_occ is None or len(path_np) < 2:
            return False
        for i in range(len(path_np) - 1):
            if self._segment_hits_wall(path_np[i], path_np[i + 1]):
                return True
        return False

    def _ordered_edge_waypoints(self, v1: int, v2: int, edata: dict) -> list[np.ndarray]:
        r1 = self.all_regions.get(v1)
        r2 = self.all_regions.get(v2)
        if r1 is None or r2 is None:
            return []

        c1 = np.asarray(r1.center[:2], dtype=np.float32)
        c2 = np.asarray(r2.center[:2], dtype=np.float32)
        axis = c2 - c1
        axis_len = float(np.linalg.norm(axis))
        if axis_len < 1e-6:
            return []

        raw_points = []
        cp = edata.get('cp')
        if cp is not None:
            raw_points.append(np.asarray(cp, dtype=np.float32))
        for wp in edata.get('waypoints', []):
            raw_points.append(np.asarray(wp, dtype=np.float32))

        deduped = []
        for pt in raw_points:
            if any(np.linalg.norm(pt - prev) < 0.2 for prev in deduped):
                continue
            deduped.append(pt)

        perp_limit = max(0.75, 0.30 * axis_len)
        kept = []
        for pt in deduped:
            rel = pt - c1
            t = float(rel @ axis / (axis_len ** 2))
            proj = c1 + np.clip(t, 0.0, 1.0) * axis
            perp = float(np.linalg.norm(pt - proj))
            if t < -0.20 or t > 1.20:
                continue
            if perp > perp_limit:
                continue
            if self._point_in_collision(pt):
                continue
            kept.append((t, pt))

        kept.sort(key=lambda item: item[0])
        ordered = [pt for _, pt in kept]

        if edata.get('edge_type') == 'doorway' and cp is not None:
            cp_arr = np.asarray(cp, dtype=np.float32)
            if not self._point_in_collision(cp_arr):
                return [cp_arr]

        return ordered

    def _append_chain_target(self, chain, region, target_xy):
        tgt = np.asarray(target_xy[:2], dtype=np.float32)
        if chain and np.linalg.norm(chain[-1][1] - tgt) < 0.10:
            chain[-1] = (region, tgt)
            return
        chain.append((region, tgt))

    def _build_nav_waypoint_chain(self, robot_region, goal_xy):
        """
        Build a (Region, target_pos) chain using the loaded topology graph.
        Returns None on failure so callers do not synthesize wall-crossing bridges.
        """
        import networkx as nx
        goal_region = self._region_for_pos(goal_xy)
        if goal_region is None and self.all_regions:
            goal_region = min(
                self.all_regions.values(),
                key=lambda r: float(np.linalg.norm(
                    r.bound.detach().cpu().numpy().mean(axis=0)[:2] - goal_xy)))

        if goal_region is None or goal_region.version == robot_region.version:
            return [(robot_region, goal_xy)]

        if not hasattr(self, '_topo_graph') or self._topo_graph.number_of_edges() == 0:
            return None

        try:
            path_vers = nx.shortest_path(
                self._topo_graph, robot_region.version, goal_region.version)
        except (nx.NetworkXNoPath, nx.NodeNotFound, nx.NetworkXError):
            self.get_logger().warn(
                f'[TB4NAV] No graph path R{robot_region.version}→R{goal_region.version}, '
                f'cannot plan multi-region route')
            return None

        chain = []
        for i in range(len(path_vers) - 1):
            v1, v2 = path_vers[i], path_vers[i + 1]
            edata = self._topo_graph.edges.get((v1, v2), {})
            src = self.all_regions.get(v1)
            dst = self.all_regions.get(v2)
            if src is None or dst is None:
                return None

            ordered_wps = self._ordered_edge_waypoints(v1, v2, edata)
            if not ordered_wps:
                self.get_logger().warn(
                    f'[TB4NAV] Edge R{v1}↔R{v2} has no valid transition waypoints')
                return None

            c1 = np.asarray(src.center[:2], dtype=np.float32)
            c2 = np.asarray(dst.center[:2], dtype=np.float32)
            axis = c2 - c1
            axis_den = float(axis @ axis) + 1e-9
            for wp in ordered_wps:
                t = float((wp - c1) @ axis / axis_den)
                owner = src if t <= 0.5 else dst
                self._append_chain_target(chain, owner, wp)

        self._append_chain_target(chain, goal_region, np.array(goal_xy[:2], dtype=np.float32))
        return chain

    # ── Callbacks ─────────────────────────────────────────────────────────────

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
            return   # AMCL takes priority
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.robot_pos = np.array([p.x, p.y, p.z], dtype=np.float32)
        sin_y = 2.0 * (q.w * q.z + q.x * q.y)
        cos_y = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.robot_yaw = math.atan2(sin_y, cos_y)

    def _scan_cb(self, msg: LaserScan):
        self._scan_msg = msg
        if not self._logged_pc_frame:
            self.get_logger().info(
                f'[SDF] Using lidar scan topic {self._lidar_scan_topic} in frame={msg.header.frame_id}')
            self._logged_pc_frame = True

    def _initial_pose_cb(self, msg: PoseWithCovarianceStamped):
        p = msg.pose.pose.position
        self.get_logger().info(
            f'[TB4NAV] Initial pose seed: ({p.x:.2f}, {p.y:.2f}) — AMCL localizing.'
        )

    def _sdf_path_cb(self, msg: Path):
        if not msg.poses:
            return
        self._sdf_path = np.array(
            [[p.pose.position.x, p.pose.position.y] for p in msg.poses],
            dtype=np.float32)
        self._sdf_path_time = _time.monotonic()

    def _goal_pose_cb(self, msg: PoseStamped):
        gx = msg.pose.position.x
        gy = msg.pose.position.y
        self.goal_xy = np.array([gx, gy], dtype=np.float32)
        self.path    = None
        self._sdf_path = None
        self._sdf_path_time = 0.0
        with self._sdf_lock:
            self._last_sdf_local_path = None
            self._last_sdf_local_path_time = 0.0
        self.navigator.reset_path()
        self._path_gen_running   = False
        self._motion_enabled     = False
        self._path_total_dist    = None
        self._midpoint_replanned = False
        self.state = 'PLANNING'
        self.get_logger().info(f'[TB4NAV] New goal: ({gx:.2f}, {gy:.2f}) — planning...')
        self._trigger_path_gen()

    def _init_sdf_optimizer(self, sdf_repo_dir: str):
        try:
            import types

            if sdf_repo_dir not in sys.path:
                sys.path.insert(0, sdf_repo_dir)

            if 'infer_single_view' not in sys.modules:
                _m = types.ModuleType('infer_single_view')
                class _FakeISV:
                    pass
                _m.InferenceSingleViewAugSDF = _FakeISV
                sys.modules['infer_single_view'] = _m

            for _name in ['lib_obj_det',
                          'lib_obj_det.object_detection_utils',
                          'lib_obj_det.object_detector']:
                if _name not in sys.modules:
                    sys.modules[_name] = types.ModuleType(_name)
            _lod_utils = sys.modules['lib_obj_det.object_detection_utils']
            if not hasattr(_lod_utils, 'YoloFramePredictions'):
                class _FakeYolo:
                    pass
                _lod_utils.YoloFramePredictions = _FakeYolo

            from lib_nn.sdf import DeepSdfDecoderMap
            from planner import TrajectoryOptimizerBodySDF

            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            ckpt_dir = os.path.join(sdf_repo_dir, 'ckpts', 'sdf-object')
            sdf_ckpt = os.path.join(ckpt_dir, 'sdf-obj-trunc0.5-latest.pth')
            latent_fp = os.path.join(ckpt_dir, 'latent-code-turtlebot4-bbox-tight-trunc0.5.pth')

            sdf_obj = DeepSdfDecoderMap(
                latent_size=256,
                dims=[512] * 8,
                dropout=list(range(8)),
                dropout_prob=0.2,
                norm_layers=list(range(8)),
                latent_in=[4],
                weight_norm=True,
            )

            state = torch.load(sdf_ckpt, map_location=device)
            if isinstance(state, dict) and 'model_state_dict' in state:
                state = state['model_state_dict']
            if isinstance(state, dict):
                state = {(k[7:] if k.startswith('module.') else k): v for k, v in state.items()}
            sdf_obj.load_state_dict(state)
            sdf_obj.to(device).eval()

            class _SDFMapStub:
                def __init__(self, sdf_obj, device):
                    self.sdf_obj = sdf_obj
                    self.device = device

            self._traj_opt = TrajectoryOptimizerBodySDF(
                sdf_infer_map=_SDFMapStub(sdf_obj, device),
                body_sdf_latent_code_filepath=latent_fp,
                iterations_per_update=int(self.get_parameter('opt_iterations').value),
                lr_init=0.1,
                lr_decay=0.876,
                trunc_dist_sdf=float(self.get_parameter('trunc_dist_sdf').value),
                smoothing=float(self.get_parameter('smoothing').value),
                pc_memory_radius=1.5,
            )
            self._sdf_device = device
            self._sdf_available = True
            self.get_logger().info(f'[SDF] Optimizer initialized on {device}.')
        except Exception as e:
            self.get_logger().warn(f'[SDF] Optimizer init failed: {e}')

    def _log_waiting_status(self, missing: list[str]):
        now = _time.monotonic()
        period = float(self.get_parameter('status_log_sec').value)
        if now - self._last_status_log < period:
            return
        self._last_status_log = now
        self.get_logger().info(f'[SDF] Waiting for: {", ".join(missing)}')

    def _maybe_trigger_sdf_replan(self):
        if not self._sdf_available or self.path is None or self.robot_pos is None:
            return
        now = _time.monotonic()
        interval = float(self.get_parameter('replan_interval_sec').value)
        with self._sdf_lock:
            if self._sdf_opt_running or now - self._last_sdf_replan_time < interval:
                return
            scan_msg = self._scan_msg
            local_seed = None if self._last_sdf_local_path is None else self._last_sdf_local_path.copy()
            local_seed_time = self._last_sdf_local_path_time
            self._sdf_opt_running = True
            self._last_sdf_replan_time = now

        missing = []
        if scan_msg is None:
            missing.append('lidar_scan')
        if missing:
            with self._sdf_lock:
                self._sdf_opt_running = False
            self._log_waiting_status(missing)
            return

        robot_xy = self.robot_pos[:2].copy()
        path = self.path.copy()
        threading.Thread(
            target=self._run_sdf_optimization,
            args=(scan_msg, robot_xy, path, local_seed, local_seed_time),
            daemon=True,
        ).start()

    def _run_sdf_optimization(self, scan_msg, robot_xy, path, local_seed, local_seed_time):
        t0 = _time.monotonic()
        try:
            pc_sensor = self._scan_to_pointcloud(scan_msg)
            if pc_sensor is None or len(pc_sensor) == 0:
                return
            c2w = self._get_c2w(scan_msg.header.stamp, scan_msg.header.frame_id)
            if c2w is None:
                return
            seg_global = self._extract_ahead_segment(
                path=path,
                robot_xy=robot_xy,
                max_ahead=float(self.get_parameter('max_ahead_dist').value),
                max_pts=int(self.get_parameter('max_traj_pts').value),
            )
            if seg_global is None or len(seg_global) < 3:
                return
            seg_seed = self._select_seed_segment(robot_xy, seg_global, local_seed, local_seed_time)
            pc_world = self._transform_pc(pc_sensor, c2w)
            self._publish_debug_pointcloud(pc_world)
            pc_t = torch.from_numpy(pc_world).float().to(self._sdf_device)
            self._traj_opt.pc_memory.set_robot_state(robot_xy)
            traj_t = torch.from_numpy(seg_seed).float()
            refined = self._traj_opt.optimize_trajectory(
                pc=pc_t,
                obj_det_preds=None,
                traj_pt2ds=traj_t,
                z_value=float(self.get_parameter('z_value').value),
            )
            refined_np = refined.detach().cpu().numpy().astype(np.float32)
            refined_np = self._stabilize_refined_path(refined_np, robot_xy, seg_global)
            if not self._should_publish_result(
                    refined_np=refined_np,
                    robot_xy_start=robot_xy,
                    seg_end=seg_global[-1],
                    elapsed_sec=_time.monotonic() - t0):
                return
            self._publish_refined_path(refined_np)
        except Exception as e:
            self.get_logger().warn(f'[SDF] Optimization error: {e}')
        finally:
            with self._sdf_lock:
                self._sdf_opt_running = False

    def _get_c2w(self, stamp, source_frame):
        try:
            tf = self._tf_buf.lookup_transform(
                self.nav_frame, source_frame, stamp,
                timeout=Duration(seconds=0.05),
            )
        except Exception:
            try:
                tf = self._tf_buf.lookup_transform(
                    self.nav_frame, source_frame, Time(),
                    timeout=Duration(seconds=0.05),
                )
            except Exception:
                return None
        t = tf.transform.translation
        q = tf.transform.rotation
        tx, ty, tz = t.x, t.y, t.z
        qx, qy, qz, qw = q.x, q.y, q.z, q.w
        r00 = 1.0 - 2.0 * (qy * qy + qz * qz)
        r01 = 2.0 * (qx * qy - qz * qw)
        r02 = 2.0 * (qx * qz + qy * qw)
        r10 = 2.0 * (qx * qy + qz * qw)
        r11 = 1.0 - 2.0 * (qx * qx + qz * qz)
        r12 = 2.0 * (qy * qz - qx * qw)
        r20 = 2.0 * (qx * qz - qy * qw)
        r21 = 2.0 * (qy * qz + qx * qw)
        r22 = 1.0 - 2.0 * (qx * qx + qy * qy)
        return np.array([
            [r00, r01, r02, tx],
            [r10, r11, r12, ty],
            [r20, r21, r22, tz],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float32)

    @staticmethod
    def _scan_to_pointcloud(msg: LaserScan) -> 'np.ndarray | None':
        if msg.angle_increment == 0.0:
            return None
        rmin = max(0.05, float(msg.range_min))
        rmax = float(msg.range_max)
        angles = msg.angle_min + np.arange(len(msg.ranges), dtype=np.float32) * msg.angle_increment
        ranges = np.asarray(msg.ranges, dtype=np.float32)
        valid = np.isfinite(ranges) & (ranges >= rmin) & (ranges <= rmax)
        if not np.any(valid):
            return None
        ranges = ranges[valid]
        angles = angles[valid]
        xs = ranges * np.cos(angles)
        ys = ranges * np.sin(angles)
        zs = np.zeros_like(xs)
        return np.stack([xs, ys, zs], axis=1).astype(np.float32)

    @staticmethod
    def _transform_pc(pc_sensor: np.ndarray, c2w: np.ndarray) -> np.ndarray:
        if len(pc_sensor) == 0:
            return np.zeros((0, 3), dtype=np.float32)
        ones = np.ones((len(pc_sensor), 1), dtype=np.float32)
        pts_h = np.concatenate([pc_sensor, ones], axis=1)
        pts_world = (c2w @ pts_h.T).T[:, :3]
        return pts_world.astype(np.float32)

    @staticmethod
    def _extract_ahead_segment(path: np.ndarray, robot_xy: np.ndarray, max_ahead: float, max_pts: int):
        if len(path) < 2:
            return None
        dists = np.linalg.norm(path - robot_xy, axis=1)
        closest = int(np.argmin(dists))
        seg = path[closest:]
        if len(seg) < 2:
            return None
        deltas = np.linalg.norm(np.diff(seg, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(deltas)])
        end_idx = int(np.searchsorted(cum, max_ahead, side='right'))
        seg = seg[:end_idx + 1]
        if len(seg) < 2:
            return None
        if len(seg) > max_pts:
            idxs = np.linspace(0, len(seg) - 1, max_pts, dtype=int)
            seg = seg[idxs]
        return seg.astype(np.float32)

    def _select_seed_segment(self, robot_xy, seg_global, local_seed, local_seed_time):
        valid_sec = float(self.get_parameter('local_seed_valid_sec').value)
        if (local_seed is None
                or _time.monotonic() - local_seed_time > valid_sec
                or len(local_seed) < 3):
            return seg_global
        seg_local = self._extract_ahead_segment(
            path=local_seed,
            robot_xy=robot_xy,
            max_ahead=float(self.get_parameter('max_ahead_dist').value),
            max_pts=int(self.get_parameter('max_traj_pts').value),
        )
        if seg_local is None or len(seg_local) < 3:
            return seg_global
        if float(np.linalg.norm(seg_local[-1, :2] - seg_global[-1, :2])) > 0.50:
            return seg_global
        seg_local = seg_local.astype(np.float32, copy=True)
        seg_local[0, :2] = robot_xy[:2]
        seg_local[-1, :2] = seg_global[-1, :2]
        return seg_local

    def _stabilize_refined_path(self, refined_np, robot_xy, seg):
        if refined_np.ndim != 2 or len(refined_np) == 0:
            return refined_np
        out = np.asarray(refined_np, dtype=np.float32).copy()
        out[0, :2] = robot_xy[:2]
        if len(out) > 1:
            out[-1, :2] = seg[-1, :2]
        keep = [0]
        for i in range(1, len(out)):
            if np.linalg.norm(out[i, :2] - out[keep[-1], :2]) >= 0.03:
                keep.append(i)
        if keep[-1] != len(out) - 1:
            keep.append(len(out) - 1)
        return out[keep].astype(np.float32)

    def _should_publish_result(self, refined_np, robot_xy_start, seg_end, elapsed_sec):
        if len(refined_np) < 2:
            return False
        if elapsed_sec > float(self.get_parameter('max_result_age_sec').value):
            return False
        if self.robot_pos is not None:
            moved = float(np.linalg.norm(self.robot_pos[:2] - robot_xy_start))
            if moved > float(self.get_parameter('max_robot_motion_m').value):
                return False
        if float(np.linalg.norm(refined_np[-1, :2] - seg_end[:2])) > 0.25:
            return False
        with self._sdf_lock:
            last_local = None if self._last_sdf_local_path is None else self._last_sdf_local_path.copy()
        if last_local is not None and len(last_local) >= 2:
            delta = self._path_delta_m(last_local, refined_np)
            if delta < float(self.get_parameter('min_publish_delta_m').value):
                return False
        return True

    @staticmethod
    def _path_delta_m(path_a: np.ndarray, path_b: np.ndarray) -> float:
        if len(path_a) == 0 or len(path_b) == 0:
            return float('inf')
        n = min(len(path_a), len(path_b), 12)
        idx_a = np.linspace(0, len(path_a) - 1, n, dtype=int)
        idx_b = np.linspace(0, len(path_b) - 1, n, dtype=int)
        pts_a = path_a[idx_a, :2]
        pts_b = path_b[idx_b, :2]
        return float(np.mean(np.linalg.norm(pts_a - pts_b, axis=1)))

    def _publish_refined_path(self, pts: np.ndarray):
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.nav_frame
        for p in pts:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = float(p[0])
            ps.pose.position.y = float(p[1])
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        self.sdf_path_pub.publish(msg)
        self._sdf_path = np.asarray(pts, dtype=np.float32).copy()
        self._sdf_path_time = _time.monotonic()
        with self._sdf_lock:
            self._last_sdf_local_path = self._sdf_path.copy()
            self._last_sdf_local_path_time = self._sdf_path_time

    def _publish_debug_pointcloud(self, pts: np.ndarray):
        if pts is None or len(pts) == 0:
            return
        xyz = np.asarray(pts, dtype=np.float32)
        msg = PointCloud2()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.nav_frame
        msg.height = 1
        msg.width = int(len(xyz))
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = msg.point_step * msg.width
        msg.is_dense = True
        msg.data = np.ascontiguousarray(xyz).tobytes()
        self.sdf_debug_pc_pub.publish(msg)

    # ── Path generation ───────────────────────────────────────────────────────

    def _trigger_path_gen(self):
        if self._path_gen_running:
            return
        if self.robot_pos is None or self.goal_xy is None:
            return

        robot_xy = self.robot_pos[:2].copy()
        goal_xy  = self.goal_xy.copy()

        region = self._region_for_pos(robot_xy)
        if region is None:
            if not self.all_regions:
                self.get_logger().warn('[TB4NAV] No regions available for path gen.')
                return
            region = min(
                self.all_regions.values(),
                key=lambda r: float(np.linalg.norm(
                    r.bound.detach().cpu().numpy().mean(axis=0)[:2] - robot_xy))
            )
            self.get_logger().warn(
                f'[TB4NAV] Robot outside all regions, using closest: region {region.version}'
            )

        self.active_region = region
        self._path_gen_running = True
        step  = float(self.get_parameter('step_size').value)
        steps = int(self.get_parameter('max_steps').value)

        def _worker():
            try:
                robot_region = self.active_region
                chain = self._build_nav_waypoint_chain(robot_region, goal_xy)
                if chain is None:
                    self.get_logger().warn(
                        f'[TB4NAV] No valid transition chain R{robot_region.version}→goal')
                    return

                cur = robot_xy.copy()
                segments = []
                steps_per_seg = max(60, steps // len(chain))
                for region, tgt in chain:
                    seg = region.generate_field_path(
                        cur, tgt[:2], step_size=step, max_steps=steps_per_seg)
                    if seg is None or len(seg) < 2:
                        self.get_logger().warn(
                            f'[TB4NAV] Segment failed in R{region.version} toward '
                            f'({tgt[0]:.2f},{tgt[1]:.2f})')
                        return
                    seg_np = np.asarray(seg, dtype=np.float32)
                    if self._path_hits_wall(seg_np):
                        self.get_logger().warn(
                            f'[TB4NAV] Raw segment intersects wall in R{region.version}')
                        return
                    segments.append(seg_np)
                    cur = seg_np[-1]

                full_path = np.concatenate(segments, axis=0) if segments else None

                if full_path is not None and len(full_path) > 1:
                    decimated = self._decimate(full_path, spacing=0.10)
                    smoothed  = self._smooth(decimated, window=7)
                    if self._path_hits_wall(smoothed):
                        self.get_logger().warn('[TB4NAV] Smoothed multi-region path intersects wall')
                        return
                    self.path = np.asarray(smoothed, dtype=np.float32)
                    self.navigator.reset_path()
                    self._path_total_dist = float(np.sum(
                        np.linalg.norm(np.diff(self.path, axis=0), axis=1)))
                    self._midpoint_replanned = False
                    if self.state == 'PLANNING':
                        self.state = 'READY'
                        print(
                            f'\n[TB4NAV] Path ready ({len(self.path)} pts, '
                            f'{self._path_total_dist:.2f}m, '
                            f'{len(chain)}-region chain). Press ENTER to drive...'
                        )
                    self.get_logger().info(
                        f'[TB4NAV] Path: {len(chain)}-segment chain '
                        f'R{robot_region.version}→R{chain[-1][0].version}'
                    )
                else:
                    self.get_logger().warn('[TB4NAV] generate_field_path returned empty.')
            except Exception as e:
                self.get_logger().error(f'[TB4NAV] Path gen error: {e}')
            finally:
                self._path_gen_running = False

        threading.Thread(target=_worker, daemon=True).start()

    # ── Path utilities ────────────────────────────────────────────────────────

    def _decimate(self, path_np: np.ndarray, spacing: float = 0.10) -> np.ndarray:
        pts     = path_np.astype(np.float32)
        new_pts = [pts[0]]
        for i in range(1, len(pts)):
            if np.linalg.norm(pts[i] - new_pts[-1]) >= spacing:
                new_pts.append(pts[i])
        if np.linalg.norm(new_pts[-1] - pts[-1]) > 0.05:
            new_pts.append(pts[-1])
        return np.array(new_pts, dtype=np.float32)

    def _smooth(self, path_np: np.ndarray, window: int = 7) -> np.ndarray:
        pts = path_np.astype(np.float32)
        if len(pts) < window:
            return pts
        sm  = pts.copy()
        k   = window // 2
        for i in range(len(pts)):
            a = max(0, i - k)
            b = min(len(pts), i + k + 1)
            sm[i] = pts[a:b].mean(axis=0)
        sm[0]  = pts[0]
        sm[-1] = pts[-1]
        return sm

    # ── Publishers ────────────────────────────────────────────────────────────

    def _publish_nav_path(self):
        if self.path is None:
            return
        msg = Path()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = self.nav_frame
        for p in self.path:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x  = float(p[0])
            ps.pose.position.y  = float(p[1])
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        self.path_pub.publish(msg)

    def _publish_robot_path(self):
        if not self._robot_path_pts:
            return
        msg = Path()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = self.nav_frame
        for xy in self._robot_path_pts:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x  = float(xy[0])
            ps.pose.position.y  = float(xy[1])
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        self.robot_path_pub.publish(msg)

    def _publish_goal_marker(self):
        if self.goal_xy is None:
            return
        arr = MarkerArray()
        m = Marker()
        m.header.stamp    = self.get_clock().now().to_msg()
        m.header.frame_id = self.nav_frame
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

    # ── Main update loop (10 Hz) ──────────────────────────────────────────────

    def _update(self):
        # Always publish nav path and goal marker
        self._publish_nav_path()
        self._publish_goal_marker()

        # Track robot trajectory
        if self.robot_pos is not None:
            xy = self.robot_pos[:2].copy()
            if (not self._robot_path_pts or
                    np.linalg.norm(xy - self._robot_path_pts[-1]) > 0.05):
                self._robot_path_pts.append(xy)
                if len(self._robot_path_pts) > 5000:
                    self._robot_path_pts.pop(0)
            self._publish_robot_path()

        # ── WAITING_FOR_POSE ──────────────────────────────────────────────
        if self.state == 'WAITING_FOR_POSE':
            if self.robot_pos is not None:
                self.state = 'IDLE'
                self.get_logger().info(
                    '[TB4NAV] Pose received — waiting for 2D Goal Pose in RViz.'
                )
            else:
                self.get_logger().info(
                    '[TB4NAV] Waiting for pose (/amcl_pose or /odom)...', throttle_duration_sec=5.0
                )
            return

        if self.robot_pos is None:
            return

        robot_xy = self.robot_pos[:2]

        # ── IDLE ─────────────────────────────────────────────────────────
        if self.state == 'IDLE':
            self.navigator.stop_robot()
            return

        # ── PLANNING ─────────────────────────────────────────────────────
        if self.state == 'PLANNING':
            self.navigator.stop_robot()
            if not self._path_gen_running and self.path is None:
                self._trigger_path_gen()
            return

        # ── READY — path shown, waiting for ENTER ─────────────────────────
        if self.state == 'READY':
            self.navigator.stop_robot()
            return

        # ── DRIVING ──────────────────────────────────────────────────────
        if self.state == 'DRIVING':
            if self.goal_xy is None:
                self.state = 'IDLE'
                return

            d_goal = float(np.linalg.norm(robot_xy - self.goal_xy))
            goal_reached_dist = float(self.get_parameter('goal_reached_dist').value)

            if d_goal < goal_reached_dist:
                self.get_logger().info(f'[TB4NAV] Goal reached! d={d_goal:.3f}m')
                self.navigator.stop_robot()
                self.navigator.reset_path()
                self.goal_xy = None
                self.path    = None
                self._sdf_path = None
                self._sdf_path_time = 0.0
                self.state   = 'IDLE'
                return

            if self.path is not None:
                self._maybe_trigger_sdf_replan()

            _SDF_VALID = 2.0
            active = (self._sdf_path
                      if (self._sdf_path is not None
                          and _time.monotonic() - self._sdf_path_time < _SDF_VALID)
                      else self.path)
            if active is not None:
                self.navigator.drive_path(active)

            # Mid-path replan: once, when halfway through a long path (>2m)
            replan = self.get_parameter('replan_at_halfway').value
            if (replan
                    and self._path_total_dist is not None
                    and self._path_total_dist > 2.0
                    and not self._midpoint_replanned
                    and d_goal < self._path_total_dist * 0.5
                    and not self._path_gen_running):
                self._midpoint_replanned = True
                self.get_logger().info(
                    f'[TB4NAV] Halfway replan triggered at d_goal={d_goal:.2f}m '
                    f'(path was {self._path_total_dist:.2f}m)'
                )
                self._trigger_path_gen()


def main(args=None):
    rclpy.init(args=args)
    node = TB4NavNode()
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
