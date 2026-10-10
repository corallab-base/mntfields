"""
gridmap_node.py — Refactored GridmapNode (~500 lines).

No threading, no state machine, no ExplorationGraph, no LidarGridmap.
Uses TopologyExtractor + SLAM /map + simple proportional controller.
"""

import json
import os
import math
import threading
import time
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
from scipy.spatial.transform import Rotation

import sensor_msgs_py.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, LaserScan
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from geometry_msgs.msg import TransformStamped, Twist, Point, PointStamped, PoseStamped, PoseWithCovarianceStamped
from visualization_msgs.msg import Marker, MarkerArray
from builtin_interfaces.msg import Duration as DurationMsg

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg", force=True)

from mntfields.mntfields_ros.sensing.lidar_adapter import Lidar2DAdapter
from mntfields.mntfields_ros.mapping.topology_extractor import TopologyExtractor
from mntfields.mntfields_ros.mapping.markers import MarkerPublisher
from mntfields.mntfields_ros.mapping.navigation import Navigator
import open3d as o3d
from mntfields.mntfields_core.core.region import Region
from mntfields.mntfields_core.mapping.lidar_gridmap import LidarGridmap
from mntfields.utils.mntfields_utils import (
    compute_ownership_map, compute_crossing_points, owner_region_for_xy,
    compute_region_room_ownership, straight_line_path, combine_pcd)

# ── Constants ────────────────────────────────────────────────────────────────

REGION_RADIUS = 2.5
VISITED_MARGIN = 0.2
_REGION_SWITCH_MARGIN = 0.30   # metres inset from bound before switching regions
MODEL_FOLDER = "/home/mayank/turtlebot3_ws/mntfields_model"
WARMUP_EPOCHS = 100
REVISIT_WARMUP_EPOCHS = 4


def _normalize_angle(a: float) -> float:
    """Wrap angle to (-pi, pi]."""
    while a > pi:
        a -= 2.0 * pi
    while a < -pi:
        a += 2.0 * pi
    return a


# ── Node ─────────────────────────────────────────────────────────────────────

class GridmapNode(Node):

    def __init__(self):
        super().__init__('mntfields_gridmap')

        os.makedirs(MODEL_FOLDER, exist_ok=True)

        # ── Parameters ───────────────────────────────────────────────────────
        self.declare_parameter('train_timer_sec', 0.5)
        self.declare_parameter('train_samples', 10000)
        self.declare_parameter('train_epoch', 40)
        self.declare_parameter('region_max_raw_samples', 50000)
        self.declare_parameter('region_raw_memory_cell_size_m', 0.35)
        self.declare_parameter('enable_region_debug_vis', True)
        self.declare_parameter('goal_reached_dist', 0.30)
        self.declare_parameter('max_lin', 0.20)
        self.declare_parameter('max_ang', 0.80)
        self.declare_parameter('control_timer_sec', 0.10)
        self.declare_parameter('lidar_height_m', 0.17)  # TurtleBot3 Waffle Pi
        self.declare_parameter('base_frame', 'base_footprint')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('follower_heading_gain', 1.2)
        self.declare_parameter('follower_rotate_thresh', 1.5)
        self.declare_parameter('follower_rotate_exit', 0.15)
        self.declare_parameter('follower_near_goal_rotate_thresh', 0.55)
        self.declare_parameter('follower_forward_suppress_thresh', 1.0)
        self.declare_parameter('follower_w_rate', 3.0)
        self.declare_parameter('vis_timer_sec', 3.0)
        self.declare_parameter('min_train_epochs_for_nav', 120)
        self.declare_parameter('room_region_spawn_dedupe_dist_m', 0.9)
        self.declare_parameter('ownership_spawn_grace_dist_m', 0.75)
        self.declare_parameter('follow_stuck_timeout_sec', 4.0)
        self.declare_parameter('follow_progress_eps', 0.10)
        self.declare_parameter('path_commit_max_m', 0.40)
        self.declare_parameter('path_truncate_trigger_m', 0.60)
        self.set_parameters([rclpy.parameter.Parameter(
            'use_sim_time', rclpy.Parameter.Type.BOOL, True)])

        # ── State ─────────────────────────────────────────────────────────────
        self.robot_pos: np.ndarray = None   # [x, y, z]
        self.robot_yaw: float = 0.0
        self.robot_vx: float = 0.0
        self.robot_vy: float = 0.0

        self.region: Region = None
        self.all_regions: dict = {}         # reg.version -> Region
        self._spawned_room_ids: set = set() # room IDs that already have a Region

        self._path: np.ndarray = None       # Nx2
        self._path_idx: int = 0
        self._current_goal: np.ndarray = None
        self._visited_goals: list = []
        self._goal_blacklist_radius: float = 0.5
        self._pathgen_fail_count: int = 0
        self._exploration_tick_count: int = 0   # for throttled logging/publishing
        self._active_handoff_target: np.ndarray = None
        self._controller_path_done = False
        self._train_running = False
        self._train_lock = threading.Lock()
        self._frontier_pick_times_ms: list = []   # ms per pick_reachable_frontier call
        self._path_gen_times_ms: list = []        # ms per generate_field_path call
        self.mode: str = "EXPLORE"                   # "WARMUP" or "EXPLORE"
        self._warmup_start_epoch: int = 0
        self._warmup_epoch_target: int = WARMUP_EPOCHS
        self._path_total_dist = None
        self._midpoint_replanned = False
        self._best_goal_dist = float('inf')
        self._best_heading_err = float('inf')
        self._last_progress_time = 0.0
        self._last_path_progress_s = 0.0
        self._stuck_replan_count: int = 0       # consecutive stuck replans for same goal

        self.latest_cloud = None            # most recent LaserScan message
        self._latest_world_pts: np.ndarray = None   # transformed scan in world frame
        self._latest_robot_pose_for_pts: np.ndarray = None

        self._last_coverage_spawn_time: float = 0.0   # cooldown for coverage spawning

        # ── Voronoi ownership ─────────────────────────────────────────────────
        self._ownership_map: np.ndarray = None   # int32 [H,W], 0=unowned
        self._crossing_points: dict = {}         # (ver_a,ver_b) -> xy
        self._region_room_ownership: dict = {'by_region': {}, 'by_room': {}}

        # ── TopologyExtractor ─────────────────────────────────────────────────
        self.extractor = TopologyExtractor()
        self.gridmap = None
        self._slam_map_received = False
        self._slam_arr: np.ndarray = None
        self._slam_cell_size: float = 0.05
        self._slam_origin: np.ndarray = np.zeros(2, dtype=np.float32)

        # ── TF ────────────────────────────────────────────────────────────────
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ── LiDAR adapter ─────────────────────────────────────────────────────
        self.adapter = Lidar2DAdapter(
            max_range=3.5,
            lidar_height=float(self.get_parameter('lidar_height_m').value)
        )

        # ── Subscribers ───────────────────────────────────────────────────────
        self.create_subscription(LaserScan, '/scan', self.lidar_cb, 10)

        map_qos = QoSProfile(
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
            depth=1,
        )
        self.create_subscription(OccupancyGrid, '/map', self.slam_map_cb, map_qos)

        # ── Publishers ────────────────────────────────────────────────────────
        cmd_vel_topic = str(self.get_parameter('cmd_vel_topic').value)
        self.cmd_pub = self.create_publisher(Twist, cmd_vel_topic, 10)
        # Path display — publish on both legacy and current RViz topics
        self.field_path_pub = self.create_publisher(Path, '/field_path', 10)
        self.field_path_pub_alt = self.create_publisher(Path, '/mntfields/local_model_path', 10)
        # Frontier OccupancyGrid — rviz Map display expects nav_msgs/OccupancyGrid
        self.frontier_occ_pub = self.create_publisher(
            OccupancyGrid, '/mntfields/frontier', map_qos)
        # Frontier spheres — separate MarkerArray for 3D visualization
        self.frontier_marker_pub = self.create_publisher(
            Marker, '/mntfields/frontier_markers', 10)
        self.goal_marker_pub = self.create_publisher(
            Marker, '/mntfields/goal_marker', 10)
        # Full untruncated neural-field path (exec path stays on /field_path)
        self.nav_path_pub = self.create_publisher(Path, '/navigator_path', 10)
        # Topology markers: confirmed doors, door candidates, room centroids
        self.topology_marker_pub = self.create_publisher(
            MarkerArray, '/mntfields/topology', 10)
        self.markers = MarkerPublisher(self)

        # ── Timers ────────────────────────────────────────────────────────────
        train_dt = float(self.get_parameter('train_timer_sec').value)
        self.train_timer = self.create_timer(train_dt, self.train_tick)

        ctrl_dt = float(self.get_parameter('control_timer_sec').value)
        self.exploration_timer = self.create_timer(ctrl_dt, self.exploration_tick)
        self.control_timer = self.create_timer(ctrl_dt, self.control_tick)

        vis_dt = float(self.get_parameter('vis_timer_sec').value)
        self.vis_timer = self.create_timer(vis_dt, self.vis_tick)

        self.dt = ctrl_dt
        self.nav_frame = "map"
        self.base_frame = str(self.get_parameter('base_frame').value)
        self.navigator = Navigator(self)

        self.get_logger().info("[GridmapNode] initialized")

    # ── Subscribers ──────────────────────────────────────────────────────────

    def lidar_cb(self, msg: LaserScan):
        """Store latest scan; transform to world frame for training."""
        self.latest_cloud = msg
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
        self._latest_robot_pose_for_pts = self.robot_pos.copy()

    def slam_map_cb(self, msg: OccupancyGrid):
        """Receive SLAM /map, update topology + old LidarGridmap frontier state."""
        info = msg.info
        self._slam_cell_size = float(info.resolution)
        self._slam_origin = np.array(
            [info.origin.position.x, info.origin.position.y],
            dtype=np.float32)

        H, W = info.height, info.width
        arr = np.array(msg.data, dtype=np.int8).reshape(H, W)
        self._slam_arr = arr

        self.extractor.update(arr, self._slam_cell_size, self._slam_origin)

        map_size = float(max(H, W) * self._slam_cell_size)
        center_xy = self._slam_origin + np.array([W * self._slam_cell_size * 0.5, H * self._slam_cell_size * 0.5], dtype=np.float32)
        if self.gridmap is None or self.gridmap.n_cells != int(max(H, W)):
            n_cells = int(max(H, W))
            trav = np.ones((n_cells, n_cells), dtype=np.uint8)
            self.gridmap = LidarGridmap(
                trav_map=trav, regions=self.all_regions, n_cells=n_cells,
                near_dist=2.0, map_size=map_size, origin=center_xy)
        else:
            self.gridmap.origin = center_xy.astype(np.float32)
            self.gridmap.map_size = map_size
            self.gridmap.cell_size = self.gridmap.map_size / self.gridmap.n_cells
            self.gridmap.regions = self.all_regions

        self.gridmap.ingest_slam_map(msg)
        self._update_regions_from_rooms()
        self._rebuild_ownership()
        self._stamp_discovery_map()
        self._publish_topology_markers()

        # Throttled topology log every 20 map callbacks (~20s at 1Hz)
        self._topology_log_count = getattr(self, '_topology_log_count', 0) + 1
        if self._topology_log_count % 20 == 1:
            n_tracks = len(self.extractor._door_tracks)
            n_doors  = len(self.extractor.doors)
            n_rooms  = len(self.extractor.confirmed_rooms)
            n_fronts = len(self.extractor.frontier_centroids)
            self.get_logger().info(
                f"[TOPO] rooms={n_rooms} doors={n_doors} "
                f"door_tracks={n_tracks} frontiers={n_fronts}")
            # Log door candidate rejection reasons for diagnosis
            if self.extractor._door_debug:
                from collections import Counter
                reasons = Counter(d['reason'] for d in self.extractor._door_debug)
                self.get_logger().info(f"[DOOR DBG] candidates: {dict(reasons)}")
            for rd in self.extractor._door_debug_rooms:
                summary = " ".join(f"{k}={v}" for k, v in sorted(rd.items()))
                self.get_logger().info(f"[DOOR DBG] {summary}")

        if not self._slam_map_received:
            self._slam_map_received = True
            self.get_logger().info(
                f"[SLAM MAP] first map received "
                f"size={W}x{H} res={self._slam_cell_size:.3f}m "
                f"origin=({self._slam_origin[0]:.2f},{self._slam_origin[1]:.2f})")

    # ── TF ───────────────────────────────────────────────────────────────────

    def _update_robot_pose_from_tf(self):
        """Look up nav_frame -> base_frame and store robot_pos, robot_yaw."""
        try:
            t = self.tf_buffer.lookup_transform(
                self.nav_frame, self.base_frame,
                rclpy.time.Time(),          # latest available
                timeout=Duration(seconds=0.05))
            p = t.transform.translation
            q = t.transform.rotation
            self.robot_pos = np.array([p.x, p.y, p.z], dtype=np.float32)
            siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            self.robot_yaw = float(np.arctan2(siny_cosp, cosy_cosp))
        except Exception:
            pass

    def _transform_scan_points_to_map(self, pts_sensor: np.ndarray, msg: LaserScan):
        """Transform Nx3 scan-frame points into the map frame using TF."""
        source_frame = str(msg.header.frame_id).strip()
        if not source_frame:
            self.get_logger().warn('[LIDAR] scan header has empty frame_id; dropping scan')
            return None

        stamp = rclpy.time.Time.from_msg(msg.header.stamp)
        try:
            tf_msg = self.tf_buffer.lookup_transform(
                'map',
                source_frame,
                stamp,
                timeout=Duration(seconds=0.05),
            )
        except Exception:
            try:
                tf_msg = self.tf_buffer.lookup_transform(
                    'map',
                    source_frame,
                    rclpy.time.Time(),
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

    # ── Region management ────────────────────────────────────────────────────

    def _new_region(self, center_xy, room_id: int = -1,
                    radius: float = REGION_RADIUS) -> Region:
        """Create a Region with bound centered on center_xy."""
        cx, cy = float(center_xy[0]), float(center_xy[1])
        bound = torch.tensor([
            [cx - radius, cy - radius, -0.1],
            [cx + radius, cy + radius,  0.1],
        ], dtype=torch.float32)

        reg = Region(
            model_folder=MODEL_FOLDER,
            bound=bound,
            device='cuda:0',
        )
        reg.spawn_center = np.array([cx, cy], dtype=np.float32)
        reg.max_raw_samples = int(
            self.get_parameter('region_max_raw_samples').value)
        reg.raw_memory_cell_size_m = float(
            self.get_parameter('region_raw_memory_cell_size_m').value)
        reg.enable_debug_vis = bool(
            self.get_parameter('enable_region_debug_vis').value)

        self.all_regions[reg.version] = reg
        self.get_logger().info(
            f"[REGION {reg.version} NEW] center=({cx:.2f},{cy:.2f}) "
            f"bound=[({bound[0,0]:.2f},{bound[0,1]:.2f}) -> "
            f"({bound[1,0]:.2f},{bound[1,1]:.2f})]")

        # Seed new region with data from overlapping neighbors
        self._seed_region_from_neighbors(reg)

        self._rebuild_ownership()
        return reg

    def _seed_region_from_neighbors(self, new_reg: Region) -> None:
        """Copy training data + LiDAR obs from overlapping regions into new_reg."""
        nb = new_reg.bound.detach().cpu().numpy()

        # 1. Seed raw training data (Nx12 world-coord pairs)
        seeded_rows = []
        for donor in self.all_regions.values():
            if donor is new_reg:
                continue
            if donor.all_raw_frame_data is None or len(donor.all_raw_frame_data) == 0:
                continue
            donor_merged = torch.cat(donor.all_raw_frame_data, dim=0)
            # Filter to rows where both x0 and x1 fall within new region's bound
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
            self.get_logger().info(
                f"[SEED] R{new_reg.version} seeded with {len(merged)} "
                f"raw training rows from neighbors")

        # 2. Seed LiDAR obstacle point cloud
        for donor in self.all_regions.values():
            if donor is new_reg:
                continue
            if donor.lidar_obs is None:
                continue
            donor_pts = np.asarray(donor.lidar_obs.points, dtype=np.float32)
            in_bound = (
                (donor_pts[:, 0] >= nb[0, 0]) & (donor_pts[:, 0] <= nb[1, 0]) &
                (donor_pts[:, 1] >= nb[0, 1]) & (donor_pts[:, 1] <= nb[1, 1])
            )
            if in_bound.sum() == 0:
                continue
            subset = donor_pts[in_bound]
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(subset.astype(np.float64))
            if new_reg.lidar_obs is None:
                new_reg.lidar_obs = pcd
            else:
                new_reg.lidar_obs = combine_pcd(pcd, new_reg.lidar_obs)

        if new_reg.lidar_obs is not None:
            n_obs = len(np.asarray(new_reg.lidar_obs.points))
            self.get_logger().info(
                f"[SEED] R{new_reg.version} seeded with {n_obs} obstacle points")

    # ── Region lifecycle ──────────────────────────────────────────────────

    def _unload_current_region(self) -> None:
        """Save current region weights to disk and free GPU memory."""
        if self.region is None:
            return
        try:
            self.region.save_model()
            try:
                rp = getattr(self.region, 'last_robot_pose', None)
                if rp is not None:
                    self.region.save_debug_plot(rp, robot_xy=rp[:2])
            except Exception:
                pass
            self.region.unload_to_disk()
            self.get_logger().info(
                f"[REGION {self.region.version} UNLOADED] "
                f"epoch={self.region.epoch} saved + freed GPU")
        except Exception as e:
            self.get_logger().error(
                f"[UNLOAD ERROR] R{self.region.version}: {e}")

    def _enter_warmup(self, region: Region, epochs_needed: int,
                      reason: str) -> None:
        """Enter warmup mode: stop robot, train new region for N epochs."""
        self.mode = "WARMUP"
        self._warmup_start_epoch = int(getattr(region, 'epoch', 0))
        self._warmup_epoch_target = max(1, int(epochs_needed))
        # Clear navigation state
        self._path = None
        self._path_idx = 0
        self._current_goal = None
        self.goal_xy = None
        self.navigator.reset_path()
        self._controller_path_done = False
        self._stop_robot()
        self.get_logger().info(
            f"{reason} [warmup={self._warmup_epoch_target} eps, "
            f"start_ep={self._warmup_start_epoch}]")

    def _switch_region(self, new_region: Region,
                       warmup_epochs: int = WARMUP_EPOCHS,
                       reason: str = "switch") -> None:
        """Full lifecycle: unload old region, switch to new, enter warmup."""
        old_ver = self.region.version if self.region is not None else '---'

        # 1. Save and unload old region (stops it from being trained)
        self._unload_current_region()

        # 2. Reload if previously unloaded
        if getattr(new_region, 'loaded', True) is False:
            new_region.load_from_disk()
            self.get_logger().info(
                f"[REGION {new_region.version}] reloaded from disk "
                f"(epoch={new_region.epoch})")

        # 3. Set active region
        self.region = new_region

        # 4. Enter warmup
        self._enter_warmup(
            new_region, warmup_epochs,
            f"[REGION] R{old_ver} → R{new_region.version}: {reason}")

    def _find_existing_region_near_xy(self, xy: np.ndarray, max_dist: float) -> Region | None:
        if not self.all_regions:
            return None
        xy = np.asarray(xy[:2], dtype=np.float32)
        best = None
        best_d = float('inf')
        for reg in self.all_regions.values():
            center_xy = None
            if getattr(reg, 'spawn_center', None) is not None:
                center_xy = np.asarray(reg.spawn_center[:2], dtype=np.float32)
            else:
                b = reg.bound.detach().cpu().numpy()
                center_xy = np.array([(b[0, 0] + b[1, 0]) * 0.5, (b[0, 1] + b[1, 1]) * 0.5], dtype=np.float32)
            d = float(np.linalg.norm(center_xy - xy))
            if d < best_d:
                best_d = d
                best = reg
        if best is not None and best_d <= float(max_dist):
            return best
        return None

    def _regions_containing_xy(self, xy: np.ndarray, margin: float = 0.0) -> list[Region]:
        xy = np.asarray(xy[:2], dtype=np.float32)
        contained = []
        for reg in self.all_regions.values():
            b = reg.bound.detach().cpu().numpy()
            if (b[0, 0] - margin <= xy[0] <= b[1, 0] + margin and
                    b[0, 1] - margin <= xy[1] <= b[1, 1] + margin):
                contained.append(reg)
        return contained

    def _update_regions_from_rooms(self):
        """Use confirmed rooms as hints — spawn at room centroid only if no region covers it."""
        if self.robot_pos is None:
            return
        robot_xy = self.robot_pos[:2]
        dedupe_dist = float(self.get_parameter('room_region_spawn_dedupe_dist_m').value)

        room_source = self.extractor.confirmed_rooms
        for room_id, room in room_source.items():
            if room_id in self._spawned_room_ids:
                continue

            centroid = room.centroid_xy if hasattr(room, "centroid_xy") else np.array(
                self.gridmap._grid_to_world(
                    int(np.mean(np.where(room.mask)[0])),
                    int(np.mean(np.where(room.mask)[1]))
                ), dtype=np.float32)

            # If any region already covers this centroid, skip
            existing = self._find_existing_region_near_xy(centroid, dedupe_dist)
            if existing is not None:
                self._spawned_room_ids.add(room_id)
                continue

            # Also skip if centroid is inside any region's AABB (already covered)
            contained = self._regions_containing_xy(centroid, margin=0.0)
            if contained:
                self._spawned_room_ids.add(room_id)
                continue

            # Room centroid is far from all regions — spawn as a hint
            reg = self._new_region(centroid, room_id=room_id)
            self._spawned_room_ids.add(room_id)
            self.get_logger().info(
                f"[REGION] room hint spawn R{reg.version} for room {room_id} "
                f"at ({centroid[0]:.2f},{centroid[1]:.2f})")

        if self.region is None and self._latest_world_pts is not None:
            reg = self._new_region(robot_xy)
            self._switch_region(reg, WARMUP_EPOCHS, "first region (rooms)")

    def _rebuild_ownership(self):
        """Recompute Voronoi ownership, crossings, and region↔room support."""
        if self._slam_arr is None or not self.all_regions:
            return
        free = (self._slam_arr == 0)
        self._ownership_map = compute_ownership_map(
            free, self.all_regions, self._slam_origin, self._slam_cell_size)
        self._crossing_points = compute_crossing_points(
            self._ownership_map, self._slam_origin, self._slam_cell_size,
            occ_grid=(self._slam_arr != 0))
        room_map = getattr(self.extractor, '_confirmed_room_map', None)
        self._region_room_ownership = compute_region_room_ownership(
            self._ownership_map, room_map, self.all_regions)
        n_owned = int(np.count_nonzero(self._ownership_map))
        n_room_links = int(len(self._region_room_ownership.get('by_region', {})))
        self.get_logger().debug(
            f"[OWNERSHIP] owned_cells={n_owned} "
            f"crossings={len(self._crossing_points)} room_links={n_room_links}")

    # ── Discovery-map helpers ─────────────────────────────────────────────────

    def _stamp_discovery_map(self) -> None:
        """Stamp the current region's bit into visited+free cells within its bound."""
        gm = self.gridmap
        if gm is None or self.region is None:
            return
        ver = int(self.region.version)
        if ver < 0 or ver >= 64:
            return
        b = self.region.bound.detach().cpu().numpy()
        i0, j0 = gm._world_to_grid(float(b[0, 0]), float(b[0, 1]))
        i1, j1 = gm._world_to_grid(float(b[1, 0]), float(b[1, 1]))
        r0, r1 = max(0, min(i0, i1)), min(gm.n_cells, max(i0, i1) + 1)
        c0, c1 = max(0, min(j0, j1)), min(gm.n_cells, max(j0, j1) + 1)
        if r0 >= r1 or c0 >= c1:
            return
        patch_vis = gm.visited[r0:r1, c0:c1]
        patch_occ = gm.data[r0:r1, c0:c1]
        mask = (patch_vis > 0) & (patch_occ == 0)
        gm.discovery_map[r0:r1, c0:c1][mask] |= np.uint64(1 << ver)

    def _best_region_by_coverage(self, xy: np.ndarray, radius: float = 1.0):
        """Return the region with most observation coverage near *xy*."""
        gm = self.gridmap
        if gm is None or not self.all_regions:
            return None
        ci, cj = gm._world_to_grid(float(xy[0]), float(xy[1]))
        r_cells = max(1, int(radius / gm.cell_size))
        i0 = max(0, ci - r_cells)
        i1 = min(gm.n_cells, ci + r_cells + 1)
        j0 = max(0, cj - r_cells)
        j1 = min(gm.n_cells, cj + r_cells + 1)
        patch = gm.discovery_map[i0:i1, j0:j1]
        if patch.size == 0:
            return None
        best_ver, best_count = None, 0
        for ver in self.all_regions:
            v = int(ver)
            if v < 0 or v >= 64:
                continue
            count = int(np.count_nonzero(patch & np.uint64(1 << v)))
            if count > best_count:
                best_count = count
                best_ver = ver
        if best_ver is None:
            return None
        return self.all_regions.get(best_ver)

    def _maybe_switch_or_spawn_region(self, robot_xy: np.ndarray) -> None:
        """Switch to the owning region at robot_xy, spawning only as a last resort.
        All transitions go through _switch_region() for proper lifecycle."""
        if self.region is None:
            return

        # Check if robot is still inside current region's bound
        b = self.region.bound.detach().cpu().numpy()
        margin = _REGION_SWITCH_MARGIN
        in_bound = (
            b[0, 0] + margin <= robot_xy[0] <= b[1, 0] - margin and
            b[0, 1] + margin <= robot_xy[1] <= b[1, 1] - margin
        )
        if in_bound:
            return  # still within current region — no switch needed

        # Robot is outside current region's bound — find best candidate
        new_owner = owner_region_for_xy(
            robot_xy, self._ownership_map, self.all_regions,
            self._slam_origin, self._slam_cell_size)

        # Discovery-map override: prefer region with best local observation coverage
        coverage_best = self._best_region_by_coverage(robot_xy, radius=1.0)
        if coverage_best is not None and coverage_best is not new_owner:
            min_ep = int(self.get_parameter('min_train_epochs_for_nav').value)
            if coverage_best.epoch >= min_ep:
                new_owner = coverage_best

        if new_owner is not None and new_owner is self.region:
            return

        if new_owner is not None and new_owner is not self.region:
            # Switch to existing region (short warmup — already trained)
            self._switch_region(new_owner, REVISIT_WARMUP_EPOCHS, "ownership")
            return

        # No owner — check containment with grace margin
        grace = float(self.get_parameter('ownership_spawn_grace_dist_m').value)
        contained = self._regions_containing_xy(robot_xy, margin=grace)
        if contained:
            chosen = min(contained, key=lambda r: float(np.linalg.norm(
                np.asarray(r.center[:2], dtype=np.float32) - robot_xy[:2])))
            if chosen is not self.region:
                self._switch_region(chosen, REVISIT_WARMUP_EPOCHS, "containment")
            return

        nearby = self._find_existing_region_near_xy(robot_xy, max_dist=REGION_RADIUS * 0.8)
        if nearby is not None and nearby is not self.region:
            self._switch_region(nearby, REVISIT_WARMUP_EPOCHS, "nearby reuse")
            return

        # No existing region covers this position — spawn new
        new_reg = self._new_region(robot_xy)
        self._switch_region(new_reg, WARMUP_EPOCHS, "spawn at uncovered position")

    def _coverage_spawn_check(self, robot_xy: np.ndarray) -> None:
        """Spawn a new region when the robot is far from all existing region centers."""
        now_sec = float(self.get_clock().now().nanoseconds) * 1e-9
        if now_sec - self._last_coverage_spawn_time < 5.0:
            return  # cooldown

        if not self.all_regions:
            # No regions at all — spawn one
            reg = self._new_region(robot_xy)
            self._switch_region(reg, WARMUP_EPOCHS, "first region")
            self._last_coverage_spawn_time = now_sec
            return

        # Check distance to nearest existing region center
        dedupe_dist = float(self.get_parameter('room_region_spawn_dedupe_dist_m').value)
        spawn_threshold = REGION_RADIUS * 0.85
        nearest = self._find_existing_region_near_xy(robot_xy, max_dist=spawn_threshold)
        if nearest is not None:
            return  # still within coverage

        # Also check dedupe — don't double-spawn
        too_close = self._find_existing_region_near_xy(robot_xy, max_dist=dedupe_dist)
        if too_close is not None:
            return

        reg = self._new_region(robot_xy)
        self._switch_region(reg, WARMUP_EPOCHS, "coverage spawn")
        self._last_coverage_spawn_time = now_sec

    def _point_in_collision_slam(self, xy, pad_cells=1) -> bool:
        """Check if world point xy is in collision on the SLAM grid.
        Only occupied cells (val > 50) count as collision; unknown (-1) and
        free (0) are safe — matching antfields behaviour."""
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

    def _segment_in_collision_slam(self, a, b, pad_cells=1) -> bool:
        """Check if straight-line segment a→b hits a SLAM wall."""
        if self._slam_arr is None:
            return False
        a = np.asarray(a[:2], dtype=np.float32)
        b = np.asarray(b[:2], dtype=np.float32)
        d = float(np.linalg.norm(b - a))
        if d < 1e-6:
            return self._point_in_collision_slam(a, pad_cells=pad_cells)
        step = max(0.5 * float(self._slam_cell_size), 0.03)
        n = max(2, int(np.ceil(d / step)) + 1)
        for t in np.linspace(0.0, 1.0, n, dtype=np.float32):
            if self._point_in_collision_slam(a + t * (b - a), pad_cells=pad_cells):
                return True
        return False

    def _path_is_safe_slam(self, pts, pad_cells=1) -> bool:
        """Return True if every waypoint and segment of *pts* is collision-free."""
        arr = np.asarray(pts, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 2 or len(arr) < 2:
            return False
        for i in range(len(arr)):
            if self._point_in_collision_slam(arr[i], pad_cells=pad_cells):
                return False
        for i in range(len(arr) - 1):
            if self._segment_in_collision_slam(arr[i], arr[i + 1], pad_cells=pad_cells):
                return False
        return True

    def _slam_obstacle_cloud(self, bound_np: np.ndarray) -> np.ndarray | None:
        """Extract occupied-cell centres from the SLAM map within a bound.

        Returns Nx3 float32 array of (x, y, 0) world-frame positions,
        or None if the SLAM map isn't available yet.
        """
        if self._slam_arr is None:
            return None
        cell = float(self._slam_cell_size)
        ox, oy = float(self._slam_origin[0]), float(self._slam_origin[1])
        H, W = self._slam_arr.shape

        # Grid index range for the bound
        c0 = max(0, int((bound_np[0, 0] - ox) / cell))
        c1 = min(W, int((bound_np[1, 0] - ox) / cell) + 1)
        r0 = max(0, int((bound_np[0, 1] - oy) / cell))
        r1 = min(H, int((bound_np[1, 1] - oy) / cell) + 1)
        if c0 >= c1 or r0 >= r1:
            return None

        patch = self._slam_arr[r0:r1, c0:c1]
        occ_mask = patch > 50  # only occupied, not unknown
        rows, cols = np.nonzero(occ_mask)
        if len(rows) == 0:
            return None

        xs = (cols + c0 + 0.5) * cell + ox
        ys = (rows + r0 + 0.5) * cell + oy
        zs = np.zeros_like(xs)
        return np.stack([xs, ys, zs], axis=1).astype(np.float32)

    # ── Training ─────────────────────────────────────────────────────────────

    def train_tick(self):
        """Queue one background training job so control timers stay responsive."""
        if self._train_running:
            return
        if self.region is None:
            return
        if self._latest_world_pts is None:
            return
        if self._latest_robot_pose_for_pts is None:
            return
        if getattr(self.region.model, 'network', None) is None:
            return

        world_pts = np.asarray(self._latest_world_pts, dtype=np.float32).copy()
        robot_pose = np.asarray(self._latest_robot_pose_for_pts, dtype=np.float32).copy()
        region = self.region
        if len(world_pts) < 50:
            return

        self._train_running = True

        def _worker():
            try:
                # Abort if region was switched while we were queued
                if region is not self.region:
                    return

                epoch = int(self.get_parameter('train_epoch').value)
                region.last_robot_pose = np.array(
                    [robot_pose[0], robot_pose[1], 0.0], dtype=np.float32)

                n_samples = int(self.get_parameter('train_samples').value)

                # Replace accumulated LiDAR obs with SLAM-derived obstacle
                # cloud.  Accumulated raw LiDAR drifts when SLAM corrects its
                # pose — the SLAM map is always self-consistent.
                bound_np = region.bound.detach().cpu().numpy()
                slam_obs = self._slam_obstacle_cloud(bound_np)
                if slam_obs is not None and len(slam_obs) >= 20:
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(
                        slam_obs.astype(np.float64))
                    pcd.estimate_normals(
                        search_param=o3d.geometry.KDTreeSearchParamHybrid(
                            radius=0.1, max_nn=30))
                    pcd.orient_normals_towards_camera_location(
                        camera_location=robot_pose)
                    region.lidar_obs = pcd
                    region.all_lidar = []  # clear stale scan history

                result = region.get_data_from_lidar(
                    world_pts, robot_pose, device='cuda:0', n_samples=n_samples)
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

                # Cosine warm-restart LR
                region.model.Params['Training']['Learning Rate'] = region._current_lr()

                loss = None
                completed = 0
                _tick_t0 = time.perf_counter()
                for _ep in range(max(1, epoch)):
                    all_data = n14
                    if len(all_data) > 7000:
                        idx = torch.randperm(len(all_data), device=region.device)[:7000]
                        all_data = all_data[idx]
                    loss_new, _ = region.model.train_core(
                        epoch=1, frame_data=all_data, is_one_frame=True)
                    completed += 1
                    if loss_new is not None:
                        loss = loss_new
                _tick_ms = (time.perf_counter() - _tick_t0) * 1e3
                if loss is not None:
                    region.train_tick_ms.append(round(_tick_ms, 2))
                    region.loss_history.append(round(float(loss), 6))

                region._sync_model_refs()
                # Discard results if region was switched mid-training
                if region is not self.region:
                    return
                if loss is not None:
                    region.epoch += completed
                if region.epoch > 0 and region.epoch % 50 == 0:
                    region.save_model()
                    region.save_checkpoint(region.epoch)
                    self._save_slam_map()
                    self._save_perf_log()

                # Save training data distribution plot every 40 epochs
                if region.epoch > 0 and region.epoch % 40 == 0:
                    try:
                        n14_vis = n14[:, :6].detach()
                        n14_spd = n14[:, 6:8].detach()
                        region.save_training_vis(
                            n14_vis, n14_spd,
                            robot_xy=robot_pose[:2])
                    except Exception:
                        pass

            except Exception as e:
                self.get_logger().error(f"[TRAIN TICK] {e}")
                import traceback
                traceback.print_exc()
            finally:
                self._train_running = False

        threading.Thread(target=_worker, daemon=True).start()

    # ── Exploration ─────────────────────────────────────────────────────────-

    def control_tick(self):
        if self.robot_pos is None or self._path is None or self._current_goal is None:
            self.navigator.stop_robot()
            return
        try:
            if self.navigator.drive_path(self._path):
                self._controller_path_done = True
        except Exception as e:
            self.get_logger().warn(f"[CONTROL] {e}")
            self.navigator.stop_robot()

    def _accept_raw_path(self, robot_xy, goal, raw, forward_only, plan_goal, label="field") -> bool:
        """Process a raw waypoint list → decimate/smooth/trim/sanitize/commit.
        Returns True if path was accepted and navigation state updated."""
        if raw is None or len(raw) <= 1:
            return False
        raw_arr = np.asarray(raw, dtype=np.float32)
        decim = self.decimate_path(raw_arr, spacing=0.10)
        smooth = self.smooth_path(decim, window=3)
        if smooth is None or len(smooth) < 2:
            return False
        smooth = self._trim_at_unvisited(smooth)
        if smooth is not None and len(smooth) >= 2:
            smooth = self._sanitize_path(robot_xy, goal, smooth)
        if smooth is None or len(smooth) < 2:
            return False

        full_path = np.asarray(smooth, dtype=np.float32)

        self._publish_nav_path(full_path)
        full_len = float(np.sum(np.linalg.norm(np.diff(full_path, axis=0), axis=1)))
        commit_trigger = float(self.get_parameter('path_truncate_trigger_m').value)
        commit_max = float(self.get_parameter('path_commit_max_m').value)
        exec_path = full_path
        truncated = False
        if full_len > commit_trigger:
            exec_path = self._truncate_path_by_arclen(full_path, commit_max)
            truncated = len(exec_path) >= 2 and len(exec_path) < len(full_path)

        exec_path = self._ensure_forward_start(robot_xy, self.robot_yaw, exec_path)
        if exec_path is None or len(exec_path) < 2:
            return False

        self._path = exec_path
        self._path_idx = 0
        self.navigator.reset_path()
        self._controller_path_done = False
        self._path_total_dist = float(np.sum(np.linalg.norm(np.diff(self._path, axis=0), axis=1)))
        self._midpoint_replanned = False
        self._best_goal_dist = float(np.linalg.norm(robot_xy - self._current_goal))
        self._best_heading_err = float('inf')
        self._last_progress_time = float(self.get_clock().now().nanoseconds) * 1e-9
        self._last_path_progress_s = 0.0
        self._pathgen_fail_count = 0
        self.region.latest_path = full_path.tolist()
        self.publish_field_path(self._path)
        fwd_label = 'fwd' if forward_only else 'bi'
        self.get_logger().info(
            f"[PATH] {label} ({fwd_label}) accepted wps={len(self._path)} raw={len(raw_arr)} "
            f"decim={len(decim)} full_len={full_len:.2f}m truncated={truncated} plan_to "
            f"({plan_goal[0]:.2f},{plan_goal[1]:.2f}) frontier=({goal[0]:.2f},{goal[1]:.2f}) "
            f"first={tuple(np.round(self._path[min(1, len(self._path)-1)], 2))}")
        return True

    def exploration_tick(self):
        """Navigation tick — follow path or pick new frontier."""
        self._exploration_tick_count += 1
        log_now = (self._exploration_tick_count % 10 == 0)   

        self._update_robot_pose_from_tf()
        if self.robot_pos is None:
            if log_now:
                self.get_logger().info("[NAV] waiting for robot pose (TF not ready)")
            return

        robot_xy = self.robot_pos[:2]

        # Keep the active region aligned with the robot's current position
        self._maybe_switch_or_spawn_region(robot_xy)
        # Coverage-based spawning — spawn when robot is far from all region centers
        self._coverage_spawn_check(robot_xy)

        # ── Warmup gate: stop robot while new region trains ──
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
            return  # skip all navigation while warming up

        # Publish frontier markers every 10 ticks (~1s) so rviz stays fresh
        if self._exploration_tick_count % 10 == 0:
            self._publish_frontier_markers()

        # Follow existing path with original-style progress watchdogs.
        if self._path is not None and len(self._path) >= 2:
            self.publish_field_path(self._path)
            if self._current_goal is None:
                self._path = None
                self._path_idx = 0
                self.navigator.reset_path()
                return

            now_sec = float(self.get_clock().now().nanoseconds) * 1e-9
            d_goal = float(np.linalg.norm(robot_xy - self._current_goal))
            d_path_end = float(np.linalg.norm(robot_xy - self._path[-1]))

            progress_eps = float(self.get_parameter('follow_progress_eps').value)
            stuck_timeout = float(self.get_parameter('follow_stuck_timeout_sec').value)
            eff_eps = min(progress_eps, max(0.01, 0.05 * d_goal))
            path_progress_s = float(getattr(self.navigator, 'last_proj_s', 0.0))
            path_advanced = path_progress_s > (self._last_path_progress_s + 0.08)
            heading_err_abs = abs(float(getattr(self.navigator, 'last_heading_err', 0.0)))
            cmd_v_abs = abs(float(getattr(self.navigator, 'last_cmd_v', 0.0)))

            if (d_goal + eff_eps < self._best_goal_dist) or (d_goal + 0.01 < self._best_goal_dist):
                self._best_goal_dist = d_goal
                self._best_heading_err = min(self._best_heading_err, heading_err_abs)
                self._last_progress_time = now_sec
                self._stuck_replan_count = 0
            elif path_advanced:
                self._last_path_progress_s = path_progress_s
                self._best_heading_err = min(self._best_heading_err, heading_err_abs)
                self._last_progress_time = now_sec
            elif cmd_v_abs < 0.02 and heading_err_abs + 0.15 < self._best_heading_err:
                self._best_heading_err = heading_err_abs
                self._last_progress_time = now_sec

            if log_now:
                self.get_logger().info(
                    f"[FOLLOWING] d_goal={d_goal:.2f}m d_path_end={d_path_end:.2f}m "
                    f"path_wps={len(self._path)} cmd_v={getattr(self.navigator, 'last_cmd_v', 0.0):.2f} "
                    f"cmd_w={getattr(self.navigator, 'last_cmd_w', 0.0):.2f} "
                    f"head_err={getattr(self.navigator, 'last_heading_err', 0.0):.2f}")

            if d_goal < float(self.get_parameter('goal_reached_dist').value):
                self.get_logger().info(
                    f"[NAV] goal reached ({self._current_goal[0]:.2f},{self._current_goal[1]:.2f})")
                self._visited_goals.append(self._current_goal[:2].copy())
                if self.extractor is not None:
                    self.extractor.mark_reached(self._current_goal)
                self._stop_robot()
                self._path = None
                self._path_idx = 0
                self._current_goal = None
                self.goal_xy = None
                self.navigator.reset_path()
                self._controller_path_done = False
                self._path_total_dist = None
                self._midpoint_replanned = False
                self._best_goal_dist = float('inf')
                self._best_heading_err = float('inf')
                return

            if d_path_end < 0.20 and d_goal > 0.50:
                self.get_logger().info(f"[PATH EXTEND] partial path done, {d_goal:.2f}m to goal")
                self._path = None
                self._path_idx = 0
                self.navigator.reset_path()
                self._controller_path_done = False
                return

            if self._controller_path_done:
                self._controller_path_done = False
                if d_goal < 0.35:
                    self.get_logger().info(
                        f"[FOLLOWING] Navigator reported path complete near goal (d_goal={d_goal:.2f}m); treating as reached")
                    self._visited_goals.append(self._current_goal[:2].copy())
                    if self.extractor is not None:
                        self.extractor.mark_reached(self._current_goal)
                    self._stop_robot()
                    self._path = None
                    self._path_idx = 0
                    self._current_goal = None
                    self.goal_xy = None
                    self.navigator.reset_path()
                    self._path_total_dist = None
                    self._midpoint_replanned = False
                    self._best_goal_dist = float('inf')
                    self._best_heading_err = float('inf')
                    return
                self.get_logger().warn(
                    f"[FOLLOWING] path endpoint reached but goal still {d_goal:.2f}m away; replanning")
                self._path = None
                self._path_idx = 0
                self.navigator.reset_path()
                self._best_heading_err = float('inf')
                return

            turning_in_place = (cmd_v_abs < 0.02 and heading_err_abs > 0.8)
            adaptive_timeout = stuck_timeout * (2.0 if d_goal < 1.0 else 1.0) * (3.0 if turning_in_place else 1.0)
            if now_sec - self._last_progress_time > adaptive_timeout:
                self._stuck_replan_count += 1
                self.get_logger().warn(
                    f"[FOLLOWING] no goal progress for {adaptive_timeout:.1f}s "
                    f"(best={self._best_goal_dist:.2f}m current={d_goal:.2f}m); "
                    f"forcing replan (attempt {self._stuck_replan_count})")
                self._path = None
                self._path_idx = 0
                self.navigator.reset_path()
                self._best_heading_err = float('inf')
                self._last_progress_time = now_sec
                if self._stuck_replan_count >= 3 and self._current_goal is not None:
                    self.get_logger().warn(
                        f"[NAV] abandoning unreachable frontier "
                        f"({self._current_goal[0]:.2f},{self._current_goal[1]:.2f}) after "
                        f"{self._stuck_replan_count} stuck replans")
                    self._visited_goals.append(self._current_goal[:2].copy())
                    self._current_goal = None
                    self.goal_xy = None
                    self._stuck_replan_count = 0
                return

            if (self._path_total_dist is not None
                    and self._path_total_dist > 2.0
                    and not self._midpoint_replanned
                    and d_goal < self._path_total_dist * 0.5):
                self._midpoint_replanned = True
                self.get_logger().info(
                    f"[REPLAN] Halfway triggered at d_goal={d_goal:.2f}m (path was {self._path_total_dist:.2f}m)")
                self._path = None
                self._path_idx = 0
                self.navigator.reset_path()
                self._best_heading_err = float('inf')
                return

            return

        # No path — keep current goal if we are replanning, otherwise pick one.
        self._path = None
        self._path_idx = 0
        self.navigator.reset_path()

        if self.region is None:
            if log_now:
                self.get_logger().info(
                    "[NAV] no region yet — waiting for SLAM map + lidar")
            return
        if getattr(self.region.model, 'function', None) is None:
            if log_now:
                self.get_logger().info("[NAV] region exists but model not initialized")
            return

        if self._current_goal is not None:
            goal = self._current_goal.copy()
        else:
            _fp_t0 = time.perf_counter()
            goal = self.pick_reachable_frontier()
            self._frontier_pick_times_ms.append(
                round((time.perf_counter() - _fp_t0) * 1e3, 2))
        if goal is None:
            if log_now:
                stats = getattr(self, '_frontier_pick_stats', {}) or {}
                self.get_logger().info(
                    f"[NAV] no reachable frontier "
                    f"(source={stats.get('source','?')} raw={stats.get('raw',0)} "
                    f"standoff={stats.get('after_standoff',0)} distcap={stats.get('after_distcap',0)} "
                    f"blacklist={stats.get('after_blacklist',0)}) — waiting")
            self._stop_robot()
            return

        new_goal = self._current_goal is None
        if new_goal:
            self._stuck_replan_count = 0
            self._pathgen_fail_count = 0
        self._current_goal = goal
        self.goal_xy = goal.copy()
        self.region.latest_goal = goal.copy()
        self._publish_goal_marker()
        plan_goal = self._select_planning_target(goal)
        if new_goal or log_now:
            if self._active_handoff_target is not None:
                self.get_logger().info(
                    f"[NAV] {'picked frontier' if new_goal else 'replanning'} ({goal[0]:.2f},{goal[1]:.2f}), "
                    f"handoff via ({plan_goal[0]:.2f},{plan_goal[1]:.2f}) epoch={self.region.epoch}")
            else:
                self.get_logger().info(
                    f"[NAV] {'picked frontier' if new_goal else 'replanning'} ({goal[0]:.2f},{goal[1]:.2f}), "
                    f"from ({robot_xy[0]:.2f},{robot_xy[1]:.2f}) epoch={self.region.epoch}")

        # Wait for a meaningful warmup before attempting navigation.
        min_train_epochs = int(self.get_parameter('min_train_epochs_for_nav').value)
        if self.region.epoch < min_train_epochs:
            # Try safe-waypoint if field has enough signal (epoch >= 40)
            if self.region.epoch >= 40:
                safe_wp = self.region.find_safe_waypoint(
                    robot_xy, goal, collision_fn=self._point_in_collision_slam)
                if safe_wp is not None:
                    self.get_logger().info(
                        f"[NAV] warmup safe-waypoint ({safe_wp[0]:.2f},{safe_wp[1]:.2f}) "
                        f"epoch={self.region.epoch}")
                    _pg_t0 = time.perf_counter()
                    raw = self.region.generate_field_path(
                        robot_xy, safe_wp, step_size=0.10, max_steps=220,
                        forward_only=False)
                    self._path_gen_times_ms.append(
                        round((time.perf_counter() - _pg_t0) * 1e3, 2))
                    if self._accept_raw_path(robot_xy, goal, raw, False, safe_wp, "safe-warmup"):
                        return
            if log_now:
                self.get_logger().info(
                    f"[NAV] warmup waiting epoch={self.region.epoch} < {min_train_epochs}")
            self._stop_robot()
            return

        # Only use neural field once it has had enough training
        if self.region.epoch >= min_train_epochs:
            # Pick the region with best observation coverage near the plan goal
            path_region = self.region
            goal_coverage = self._best_region_by_coverage(plan_goal, radius=1.0)
            if (goal_coverage is not None
                    and goal_coverage is not self.region
                    and goal_coverage.epoch >= min_train_epochs):
                path_region = goal_coverage
                self.get_logger().info(
                    f"[PATH] using R{path_region.version} (discovery-best) "
                    f"instead of R{self.region.version} for plan_goal")

            # Check if plan_goal is inside the path region's bound
            _b = path_region.bound.detach().cpu().numpy()
            plan_goal_in_bound = (
                _b[0, 0] <= plan_goal[0] <= _b[1, 0] and
                _b[0, 1] <= plan_goal[1] <= _b[1, 1]
            )
            # If goal is outside bound, use forward-only rollout.
            # The robot will approach the boundary and _maybe_switch_or_spawn_region
            # will handle the lifecycle transition to a new region.
            forward_only = not plan_goal_in_bound

            _pg_t0 = time.perf_counter()
            raw = path_region.generate_field_path(
                robot_xy, plan_goal, step_size=0.10, max_steps=220,
                forward_only=forward_only)
            self._path_gen_times_ms.append(
                round((time.perf_counter() - _pg_t0) * 1e3, 2))

            if self._accept_raw_path(robot_xy, goal, raw, forward_only, plan_goal):
                return

            if log_now:
                self.get_logger().info(
                    f"[PATH] field path failed (epoch={path_region.epoch}) "
                    f"plan_to ({plan_goal[0]:.2f},{plan_goal[1]:.2f}) "
                    f"frontier=({goal[0]:.2f},{goal[1]:.2f})")

        self._pathgen_fail_count += 1
        if self._pathgen_fail_count >= 20:
            self.get_logger().warn(
                f"[NAV] abandoning frontier ({goal[0]:.2f},{goal[1]:.2f}) — "
                f"path generation failed {self._pathgen_fail_count} times")
            self._visited_goals.append(self._current_goal[:2].copy())
            self._current_goal = None
            self.goal_xy = None
            self._pathgen_fail_count = 0
        self._stop_robot()
        return

    def _follow_path_step(self, robot_xy: np.ndarray, robot_yaw: float):
        """Proportional controller following self._path."""
        if self._path is None or self._path_idx >= len(self._path):
            self._stop_robot()
            self._path = None
            return

        wp = self._path[self._path_idx]
        dx = float(wp[0] - robot_xy[0])
        dy = float(wp[1] - robot_xy[1])
        dist = sqrt(dx * dx + dy * dy)

        goal_reached_dist = float(
            self.get_parameter('goal_reached_dist').value)

        if dist < goal_reached_dist:
            self._path_idx += 1
            if self._path_idx >= len(self._path):
                # Reached end of path — blacklist goal
                if self._current_goal is not None:
                    self.get_logger().info(
                        f"[NAV] goal reached "
                        f"({self._current_goal[0]:.2f},{self._current_goal[1]:.2f})")
                    self._visited_goals.append(
                        self._current_goal[:2].copy())
                    if self.extractor is not None:
                        self.extractor.mark_reached(self._current_goal)
                self._stop_robot()
                self._path = None
                self._current_goal = None
                self.goal_xy = None
            return

        heading = atan2(dy, dx)
        err = _normalize_angle(heading - robot_yaw)

        twist = Twist()
        if abs(err) > 0.4:
            twist.linear.x = 0.0
            twist.angular.z = float(0.6 * copysign(1.0, err))
        else:
            twist.linear.x = float(min(0.25, 0.5 * dist))
            twist.angular.z = float(1.0 * err)

        self.cmd_pub.publish(twist)

    def _stop_robot(self):
        self.cmd_pub.publish(Twist())

    # ── Frontier selection ────────────────────────────────────────────────────

    def _select_planning_target(self, frontier_xy: np.ndarray) -> np.ndarray:
        """Use room/door topology to hand off toward frontiers outside current room support."""
        self._active_handoff_target = None
        try:
            if frontier_xy is None or self.robot_pos is None or self.extractor is None:
                return frontier_xy
            robot_xy = np.asarray(self.robot_pos[:2], dtype=np.float32)
            goal_xy = np.asarray(frontier_xy[:2], dtype=np.float32)
            cur_room = self.extractor.get_room_at_xy(robot_xy)
            goal_room = self.extractor.get_room_at_xy(goal_xy)
            if cur_room is None:
                return frontier_xy
            if goal_room is not None and goal_room.id == cur_room.id:
                return frontier_xy
            doors = getattr(cur_room, 'doors', []) or []
            if len(doors) == 0:
                return frontier_xy

            best_target = None
            best_score = float('inf')
            for door in doors:
                door_xy = np.asarray(door.pos_xy[:2], dtype=np.float32)
                dir_vec = goal_xy - door_xy
                dn = float(np.linalg.norm(dir_vec))
                if dn > 1e-6:
                    dir_unit = dir_vec / dn
                else:
                    dir_unit = np.zeros(2, dtype=np.float32)
                # Step slightly beyond the doorway toward the target room/frontier.
                cand = door_xy + 0.20 * dir_unit
                score = float(np.linalg.norm(cand - robot_xy) + 0.35 * np.linalg.norm(goal_xy - cand))
                if score < best_score:
                    best_score = score
                    best_target = cand
            if best_target is not None:
                self._active_handoff_target = best_target.astype(np.float32)
                return self._active_handoff_target
        except Exception:
            pass
        return frontier_xy

    def pick_reachable_frontier(self) -> np.ndarray:
        """Return one reachable frontier goal, preferring raw LidarGridmap SLAM frontier points."""
        if self.robot_pos is None:
            self._frontier_pick_stats = {'source': 'none', 'raw': 0, 'after_standoff': 0, 'after_distcap': 0, 'after_blacklist': 0}
            return None

        robot = self.robot_pos[:2]
        source = 'extractor'
        cand = None

        if self.gridmap is not None and self.gridmap.has_raw_slam_frontiers():
            pts = np.asarray(self.gridmap.slam_frontier_points, dtype=np.float32)
            if len(pts) > 0:
                cand = pts.copy()
                source = 'gridmap'

        if cand is None:
            cand = np.asarray(self.extractor.get_frontier_centroids(), dtype=np.float32)

        if cand.ndim == 1 and cand.size == 2:
            cand = cand.reshape(1, 2)
        if cand.size == 0:
            self._frontier_pick_stats = {'source': source, 'raw': 0, 'after_standoff': 0, 'after_distcap': 0, 'after_blacklist': 0}
            return None

        stats = {'source': source, 'raw': int(len(cand)), 'after_standoff': 0, 'after_distcap': 0, 'after_blacklist': 0}

        d = np.linalg.norm(cand - robot[None, :], axis=1)

        MIN_STANDOFF = max(0.15, float(self.get_parameter('goal_reached_dist').value) * 0.85)
        far_enough = d >= MIN_STANDOFF
        cand = cand[far_enough]
        d = d[far_enough]
        stats['after_standoff'] = int(len(cand))
        if len(cand) == 0:
            self._frontier_pick_stats = stats
            return None

        valid = d < (REGION_RADIUS * 4)
        cand = cand[valid]
        d = d[valid]
        stats['after_distcap'] = int(len(cand))
        if len(cand) == 0:
            self._frontier_pick_stats = stats
            return None

        if self._visited_goals:
            visited = np.asarray(self._visited_goals, dtype=np.float32)
            if visited.ndim == 2 and visited.shape[1] == 2:
                min_vd = np.min(np.linalg.norm(cand[:, None, :] - visited[None, :, :], axis=2), axis=1)
                keep = min_vd >= self._goal_blacklist_radius
                cand = cand[keep]
                d = d[keep]
        stats['after_blacklist'] = int(len(cand))
        self._frontier_pick_stats = stats

        # If gridmap source was used but all points got filtered, fall back to
        # extractor centroids (different safety-radius, less aggressive filtering)
        if len(cand) == 0 and source == 'gridmap':
            ext = np.asarray(self.extractor.get_frontier_centroids(), dtype=np.float32)
            if ext.size > 0:
                cand = ext.reshape(-1, 2)
                source = 'extractor_fallback'
                d = np.linalg.norm(cand - robot[None, :], axis=1)
                # Re-apply standoff only (blacklist already handled above)
                cand = cand[d >= MIN_STANDOFF]
                d = d[d >= MIN_STANDOFF]
                stats['after_blacklist'] = int(len(cand))
                self.get_logger().info(
                    f"[FRONTIER] gridmap filtered to 0, extractor fallback: {len(cand)} candidates")

        if len(cand) == 0:
            return None

        try:
            from sklearn.cluster import DBSCAN
            best_cand = None
            best_score = -1.0
            db = DBSCAN(eps=0.30, min_samples=1).fit(cand)
            labels = db.labels_
            for lbl in np.unique(labels):
                members = cand[labels == lbl]
                if len(members) == 0:
                    continue
                centroid = members.mean(axis=0)
                rep = members[np.argmin(np.linalg.norm(members - centroid[None, :], axis=1))]
                d_rep = float(np.linalg.norm(rep - robot))
                score = (float(len(members)) ** 0.5) / math.log(d_rep + math.e)
                if score > best_score:
                    best_score = score
                    best_cand = rep
            if best_cand is not None:
                return best_cand.astype(np.float32)
        except Exception:
            pass

        return cand[np.argmin(d)].astype(np.float32)

    # ── Visualization tick ───────────────────────────────────────────────────

    def vis_tick(self):
        """Periodic visualization — save debug plot and publish frontier markers."""
        if self.region is not None and self.region.enable_debug_vis:
            if (self.region.last_robot_pose is not None and
                    self.region.model.function is not None):
                try:
                    src = self.region.last_robot_pose
                    self.region.save_debug_plot(src, view_radius=3.5,
                                                 robot_xy=src[:2])
                except Exception as e:
                    self.get_logger().warn(f"[VIS] debug plot failed: {e}")

        self._publish_frontier_markers()
        self._publish_topology_markers()

    def _publish_goal_marker(self):
        self.markers.publish_goal_marker()

    def _publish_frontier_markers(self):
        now = self.get_clock().now().to_msg()
        n_frontier = 0
        if self.gridmap is not None and self.gridmap.slam_raw_data is not None:
            raw = self.gridmap.slam_raw_data
            slam_h, slam_w = raw.shape
            occ = OccupancyGrid()
            occ.header.frame_id = 'map'
            occ.header.stamp = now
            occ.info.map_load_time = now
            occ.info.resolution = float(self.gridmap.slam_raw_resolution)
            occ.info.width = int(slam_w)
            occ.info.height = int(slam_h)
            occ.info.origin.position.x = float(self.gridmap.slam_raw_origin[0])
            occ.info.origin.position.y = float(self.gridmap.slam_raw_origin[1])
            occ.info.origin.orientation.w = 1.0
            # Publish frontier-only occupancy so RViz's dedicated frontier Map
            # display shows the raw frontier cells directly instead of re-drawing
            # the whole SLAM map underneath them.
            flat = np.full(slam_h * slam_w, -1, dtype=np.int8)
            if self.gridmap.has_raw_slam_frontiers() and len(self.gridmap.slam_frontier_points) > 0:
                pts = self.gridmap.slam_frontier_points
                n_frontier = int(len(pts))
                res = self.gridmap.slam_raw_resolution
                ox, oy = self.gridmap.slam_raw_origin
                sc = ((pts[:, 0] - ox) / res).astype(np.int32)
                sr = ((pts[:, 1] - oy) / res).astype(np.int32)
                valid = (sc >= 0) & (sc < slam_w) & (sr >= 0) & (sr < slam_h)
                flat[sr[valid] * slam_w + sc[valid]] = 100
            occ.data = flat.tolist()
            self.frontier_occ_pub.publish(occ)

        if self._exploration_tick_count % 10 == 0:
            n_centroids = len(self.extractor.get_frontier_centroids()) if self.gridmap is None else 0
            self.get_logger().info(f"[FRONTIER PUB] cells={n_frontier} centroids={n_centroids}")

        self.markers.publish_frontier_markers()
        if self._current_goal is not None:
            self.markers.publish_goal_marker()

    def _truncate_path_by_arclen(self, pts: np.ndarray, max_len_m: float):
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

    def _sanitize_path(self, robot_xy: np.ndarray, goal_xy: np.ndarray, pts: np.ndarray):
        arr = np.asarray(pts, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 2 or len(arr) < 2:
            return None

        # If the path is reversed, flip it so it starts near the robot.
        d0 = float(np.linalg.norm(arr[0] - robot_xy))
        d1 = float(np.linalg.norm(arr[-1] - robot_xy))
        if d1 < d0:
            arr = arr[::-1].copy()

        # Ensure exact anchors at robot and goal for the controller.
        if float(np.linalg.norm(arr[0] - robot_xy)) > 0.05:
            arr = np.vstack([robot_xy.astype(np.float32), arr])
        else:
            arr[0] = robot_xy.astype(np.float32)

        if float(np.linalg.norm(arr[-1] - goal_xy)) > 0.10:
            arr = np.vstack([arr, goal_xy.astype(np.float32)])
        else:
            arr[-1] = goal_xy.astype(np.float32)

        # First meaningful step must generally point toward the goal.
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

        if float(np.dot(first_step, goal_dir)) < 0.0:
            return None

        return arr

    def _ensure_forward_start(self, robot_xy: np.ndarray, robot_yaw: float, pts: np.ndarray):
        """Drop leading points until the first meaningful step is in front of the robot."""
        arr = np.asarray(pts, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 2 or len(arr) < 2:
            return arr

        fwd = np.array([math.cos(robot_yaw), math.sin(robot_yaw)], dtype=np.float32)
        keep_from = 1
        for i in range(1, len(arr)):
            step = arr[i] - robot_xy.astype(np.float32)
            step_norm = float(np.linalg.norm(step))
            if step_norm <= 0.05:
                continue
            step_dir = step / step_norm
            if float(np.dot(step_dir, fwd)) > 0.0:
                keep_from = i
                break

        out = np.vstack([robot_xy.astype(np.float32), arr[keep_from:]])
        if len(out) < 2:
            return None
        return out

    def _trim_at_unvisited(self, pts: np.ndarray):
        """
        Trim path at first waypoint landing in an unvisited SLAM cell.
        Uses the SLAM occupancy map (free = 0, unknown = -1/255, occ = 100).
        Skips indices 0-1 (robot may straddle a cell boundary).
        """
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

            # free = 0 in SLAM map; unknown/occupied = not free
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

    def publish_field_path(self, path_xy, frame_id='map'):
        if path_xy is None:
            return
        pts = np.asarray(path_xy, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 2:
            self.get_logger().info('[PATH PUB] skipped invalid path payload')
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
        self.field_path_pub_alt.publish(msg)
        #self.get_logger().info(f'[PATH PUB] published poses={len(poses)} frame={frame_id}')

    def _publish_nav_path(self, path_xy, frame_id='map'):
        """Publish full untruncated neural-field path to /navigator_path."""
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
        self.nav_path_pub.publish(msg)

    def _publish_topology_markers(self):
        """Publish confirmed doors, door candidates, and room centroids as markers."""
        if self.extractor is None:
            return
        arr = MarkerArray()
        now = self.get_clock().now().to_msg()

        del_m = Marker()
        del_m.action = Marker.DELETEALL
        del_m.header.frame_id = 'map'
        del_m.header.stamp = now
        arr.markers.append(del_m)

        mid = 0

        # Confirmed doors — green cylinders
        for door in self.extractor.doors:
            m = Marker()
            m.header.frame_id = 'map'; m.header.stamp = now
            m.ns = 'doors'; m.id = mid; mid += 1
            m.type = Marker.CYLINDER; m.action = Marker.ADD
            m.pose.position.x = float(door.pos_xy[0])
            m.pose.position.y = float(door.pos_xy[1])
            m.pose.position.z = 0.05
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = 0.25; m.scale.z = 0.10
            m.color.r = 0.0; m.color.g = 1.0; m.color.b = 0.2; m.color.a = 1.0
            arr.markers.append(m)

        # Unconfirmed door tracks — yellow, semi-transparent
        for tid, track in self.extractor._door_tracks.items():
            m = Marker()
            m.header.frame_id = 'map'; m.header.stamp = now
            m.ns = 'door_tracks'; m.id = mid; mid += 1
            m.type = Marker.CYLINDER; m.action = Marker.ADD
            m.pose.position.x = float(track['pos_xy'][0])
            m.pose.position.y = float(track['pos_xy'][1])
            m.pose.position.z = 0.05
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = 0.15; m.scale.z = 0.10
            m.color.r = 1.0; m.color.g = 1.0; m.color.b = 0.0; m.color.a = 0.5
            arr.markers.append(m)

        # Confirmed room centroids — blue cubes + text labels
        for room in self.extractor.confirmed_rooms.values():
            m = Marker()
            m.header.frame_id = 'map'; m.header.stamp = now
            m.ns = 'rooms'; m.id = mid; mid += 1
            m.type = Marker.CUBE; m.action = Marker.ADD
            m.pose.position.x = float(room.centroid_xy[0])
            m.pose.position.y = float(room.centroid_xy[1])
            m.pose.position.z = 0.05
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.30
            m.color.r = 0.2; m.color.g = 0.4; m.color.b = 1.0; m.color.a = 0.8
            arr.markers.append(m)
            t = Marker()
            t.header.frame_id = 'map'; t.header.stamp = now
            t.ns = 'room_labels'; t.id = mid; mid += 1
            t.type = Marker.TEXT_VIEW_FACING; t.action = Marker.ADD
            t.pose.position.x = float(room.centroid_xy[0])
            t.pose.position.y = float(room.centroid_xy[1])
            t.pose.position.z = 0.25
            t.pose.orientation.w = 1.0
            t.scale.z = 0.18
            t.color.r = t.color.g = t.color.b = 1.0; t.color.a = 1.0
            t.text = f"R{room.id} d={len(room.doors)}"
            arr.markers.append(t)

        self.topology_marker_pub.publish(arr)

    def _save_topology(self) -> None:
        """Persist region bounds, crossings, rooms, doors, and room-region ownership."""
        topo = {
            'regions': {},
            'connection_points': [],
            'graph_edges': [],
            'rooms': [],
            'doors': [],
            'frontier_banks': {},
            'map_snapshot': {},
            'region_room_ownership': self._region_room_ownership or {'by_region': {}, 'by_room': {}},
        }

        for ver, reg in sorted(self.all_regions.items()):
            try:
                b = reg.bound.detach().cpu().numpy()
                room_meta = (self._region_room_ownership or {}).get('by_region', {}).get(int(ver), {})
                topo['regions'][str(ver)] = {
                    'version': int(ver),
                    'center': [float(reg.center[0]), float(reg.center[1])],
                    'bound': [[float(b[0, 0]), float(b[0, 1])], [float(b[1, 0]), float(b[1, 1])]],
                    'primary_room_id': room_meta.get('primary_room_id'),
                    'room_scores': room_meta.get('scores', []),
                }
            except Exception:
                continue

        for (a, b), crossing in sorted(self._crossing_points.items()):
            cp = np.asarray(crossing, dtype=np.float32)
            topo['connection_points'].append({
                'pos': [float(cp[0]), float(cp[1])],
                'kind': 'voronoi_crossing',
                'regions': [int(a), int(b)],
                'visited': True,
            })
            topo['graph_edges'].append({
                'v1': int(a),
                'v2': int(b),
                'edge_type': 'voronoi_crossing',
                'cp': [float(cp[0]), float(cp[1])],
                'waypoints': [[float(cp[0]), float(cp[1])]],
            })

        for room in sorted(self.extractor.confirmed_rooms.values(), key=lambda r: int(r.id)):
            mask = np.asarray(room.mask, dtype=bool)
            rows, cols = np.where(mask)
            bbox = None
            if len(rows) > 0:
                bbox = [int(cols.min()), int(rows.min()), int(cols.max()), int(rows.max())]
            topo['rooms'].append({
                'id': int(room.id),
                'centroid_xy': [float(room.centroid_xy[0]), float(room.centroid_xy[1])],
                'area_cells': int(mask.sum()),
                'bbox_rc': bbox,
                'door_ids': [int(d.id) for d in getattr(room, 'doors', [])],
            })

        for door in getattr(self.extractor, 'doors', []):
            topo['doors'].append({
                'id': int(door.id),
                'pos_xy': [float(door.pos_xy[0]), float(door.pos_xy[1])],
                'room_ids': sorted(int(v) for v in getattr(door, 'room_ids', set())),
            })

        frontier_pts = np.asarray(self.extractor.get_frontier_centroids(), dtype=np.float32)
        if frontier_pts.ndim == 2 and len(frontier_pts) > 0:
            topo['frontier_banks']['global'] = frontier_pts[:, :2].astype(float).tolist()

        topo['map_snapshot'] = {
            'map_yaml': os.path.join(MODEL_FOLDER, 'map.yaml'),
            'map_pgm': os.path.join(MODEL_FOLDER, 'map.pgm'),
            'resolution': float(self._slam_cell_size),
            'origin': [float(self._slam_origin[0]), float(self._slam_origin[1])],
        }

        room_map = getattr(self.extractor, '_confirmed_room_map', None)
        if room_map is not None:
            room_map_path = os.path.join(MODEL_FOLDER, 'room_map_snapshot.npz')
            np.savez_compressed(
                room_map_path,
                room_map=np.asarray(room_map, dtype=np.int32),
                origin=np.asarray(self._slam_origin, dtype=np.float32),
                resolution=np.asarray([self._slam_cell_size], dtype=np.float32),
            )
            topo['room_map'] = {
                'path': room_map_path,
                'resolution': float(self._slam_cell_size),
                'origin': [float(self._slam_origin[0]), float(self._slam_origin[1])],
                'shape': [int(room_map.shape[0]), int(room_map.shape[1])],
            }

        topo_path = os.path.join(MODEL_FOLDER, 'topology.json')
        with open(topo_path, 'w', encoding='utf-8') as f:
            import json
            json.dump(topo, f, indent=2)
        self.get_logger().info(f"[TOPOLOGY] saved {topo_path}")


    def _save_perf_log(self) -> None:
        """Write per-tick timing data for all loaded regions + node-level stats."""
        log = {}
        for ver, reg in self.all_regions.items():
            log[f"region_{ver}"] = {
                "train_tick_ms": list(reg.train_tick_ms),
                "loss_history": list(reg.loss_history),
                "epoch": int(reg.epoch),
            }
        log["frontier_pick_ms"] = list(self._frontier_pick_times_ms)
        log["path_gen_ms"] = list(self._path_gen_times_ms)
        path = os.path.join(MODEL_FOLDER, "perf_log.json")
        try:
            with open(path, "w") as f:
                json.dump(log, f, indent=2)
        except Exception as e:
            self.get_logger().warn(f"[PERF] could not write perf_log.json: {e}")

    def _save_slam_map(self) -> None:
        """Save current SLAM OccupancyGrid as map.pgm + map.yaml (ROS nav2 format)."""
        if self._slam_arr is None or self._slam_cell_size is None or self._slam_origin is None:
            return
        import yaml as _yaml
        arr = self._slam_arr          # int8 [H, W]: -1=unknown, 0=free, >=50=occupied
        H, W = arr.shape
        pgm = np.full((H, W), 205, dtype=np.uint8)   # 205 = unknown
        pgm[arr == 0]  = 254                          # free
        pgm[arr >= 50] = 0                            # occupied
        pgm = np.flipud(pgm)                          # PGM origin = bottom-left

        pgm_path  = os.path.join(MODEL_FOLDER, 'map.pgm')
        yaml_path = os.path.join(MODEL_FOLDER, 'map.yaml')
        with open(pgm_path, 'wb') as f:
            f.write(f'P5\n{W} {H}\n255\n'.encode())
            f.write(pgm.tobytes())
        meta = {
            'image':          'map.pgm',
            'resolution':     float(self._slam_cell_size),
            'origin':         [float(self._slam_origin[0]),
                               float(self._slam_origin[1]), 0.0],
            'negate':         0,
            'occupied_thresh': 0.65,
            'free_thresh':    0.196,
        }
        with open(yaml_path, 'w') as f:
            _yaml.dump(meta, f, default_flow_style=False)
        self.get_logger().info(
            f"[MAP] saved {pgm_path} ({W}×{H} px, res={self._slam_cell_size:.3f}m)")
        try:
            self._save_topology()
        except Exception as exc:
            self.get_logger().warn(f"[TOPOLOGY] save failed: {exc}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = GridmapNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Save all loaded regions before exit
        for reg in list(node.all_regions.values()):
            try:
                if (getattr(reg, 'loaded', True) and
                        getattr(reg.model, 'network', None) is not None):
                    reg.save_model()
                    reg.save_checkpoint(reg.epoch)
                    print(f"[SHUTDOWN] saved R{reg.version} epoch={reg.epoch}")
            except Exception as e:
                print(f"[SHUTDOWN] R{reg.version} save failed: {e}")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
