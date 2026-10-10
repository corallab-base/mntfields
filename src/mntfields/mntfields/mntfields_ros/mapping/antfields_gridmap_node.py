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

import numpy as np
import rclpy
import rclpy.time
import tf2_ros
import torch
from scipy.spatial.transform import Rotation
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Path
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import LaserScan
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


MODEL_FOLDER = "/home/mayank/turtlebot3_ws/antfields_model"


class AntfieldsGridmapNode(Node):
    def __init__(self):
        super().__init__("antfields_gridmap")

        os.makedirs(MODEL_FOLDER, exist_ok=True)

        self.declare_parameter("train_timer_sec", 0.2)
        self.declare_parameter("train_samples", 12000)
        self.declare_parameter("train_epoch", 30)
        self.declare_parameter("region_max_raw_samples", 80000)
        self.declare_parameter("region_raw_memory_cell_size_m", 0.25)
        self.declare_parameter("enable_region_debug_vis", True)
        self.declare_parameter("goal_reached_dist", 0.30)
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
        self.declare_parameter("train_min_translation_m", 0.05)
        self.declare_parameter("train_min_rotation_rad", 0.12)
        self.declare_parameter("train_force_period_sec", 0.75)
        self.declare_parameter("region_padding_m", 0.75)
        self.declare_parameter("region_min_size_m", 6.0)
        self.declare_parameter("fixed_region_size_m", 20.0)
        self.declare_parameter("expand_region_with_map", False)
        self.declare_parameter("collision_pad_cells", 2)
        self.declare_parameter("path_collision_sample_step_m", 0.03)
        self.declare_parameter("unknown_is_collision", False)
        self.declare_parameter("trim_at_unknown", False)
        self.declare_parameter("enable_path_validation", False)
        self.declare_parameter("allow_straight_line_fallback", False)

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
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.adapter = Lidar2DAdapter(
            max_range=3.5,
            lidar_height=float(self.get_parameter("lidar_height_m").value),
        )

        self.create_subscription(LaserScan, "/scan", self.lidar_cb, 10)

        map_qos = QoSProfile(
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
            depth=1,
        )
        self.create_subscription(OccupancyGrid, "/map", self.slam_map_cb, map_qos)

        self.nav_frame = "map"

        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.field_path_pub = self.create_publisher(Path, "/field_path", 10)
        self.field_path_pub_alt = self.create_publisher(
            Path, "/mntfields/local_model_path", 10
        )
        self.nav_path_pub = self.create_publisher(Path, "/navigator_path", 10)
        self.frontier_occ_pub = self.create_publisher(
            OccupancyGrid, "/mntfields/frontier", map_qos
        )
        self.frontier_marker_pub = self.create_publisher(Marker, "/mntfields/frontier_markers", 10)
        self.goal_marker_pub = self.create_publisher(Marker, "/mntfields/goal_marker", 10)

        self.markers = MarkerPublisher(self)
        self.navigator = Navigator(self)

        train_dt = float(self.get_parameter("train_timer_sec").value)
        ctrl_dt = float(self.get_parameter("control_timer_sec").value)
        vis_dt = float(self.get_parameter("vis_timer_sec").value)
        self.create_timer(train_dt, self.train_tick)
        self.create_timer(ctrl_dt, self.exploration_tick)
        self.create_timer(ctrl_dt, self.control_tick)
        self.create_timer(vis_dt, self.vis_tick)

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
        for base_frame in ("base_footprint", "base_link"):
            try:
                return self.tf_buffer.lookup_transform(
                    "map",
                    base_frame,
                    rclpy.time.Time(),
                    timeout=Duration(seconds=0.05),
                )
            except Exception:
                continue
        return None

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

        self._latest_world_pts = world_pts
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
                tf_msg = self.tf_buffer.lookup_transform(
                    "map",
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

    def slam_map_cb(self, msg: OccupancyGrid):
        info = msg.info
        self._slam_cell_size = float(info.resolution)
        self._slam_origin = np.array(
            [info.origin.position.x, info.origin.position.y], dtype=np.float32
        )

        h = int(info.height)
        w = int(info.width)
        arr = np.array(msg.data, dtype=np.int8).reshape(h, w)
        self._slam_arr = arr

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
                model_folder=MODEL_FOLDER,
                bound=bound,
                version=1,
                device="cuda:0",
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

    def control_tick(self):
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

    def _path_is_safe_slam(self, pts, pad_cells=None):
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

    def pick_reachable_frontier(self):
        if self.robot_pos is None or self.gridmap is None:
            self._frontier_pick_stats = {"source": "none", "raw": 0}
            return None

        if not self.gridmap.has_raw_slam_frontiers():
            self._frontier_pick_stats = {"source": "gridmap", "raw": 0}
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

        min_standoff = max(
            0.15, float(self.get_parameter("goal_reached_dist").value) * 0.85
        )
        keep = d >= min_standoff
        cand = cand[keep]
        d = d[keep]
        stats["after_standoff"] = int(len(cand))
        if len(cand) == 0:
            self._frontier_pick_stats = stats
            return None

        max_goal_dist = float(self.get_parameter("frontier_max_goal_dist_m").value)
        keep = d <= max_goal_dist
        cand = cand[keep]
        d = d[keep]
        stats["after_distcap"] = int(len(cand))
        if len(cand) == 0:
            self._frontier_pick_stats = stats
            return None

        if self._visited_goals:
            visited = np.asarray(self._visited_goals, dtype=np.float32)
            min_vd = np.min(
                np.linalg.norm(cand[:, None, :] - visited[None, :, :], axis=2), axis=1
            )
            keep = min_vd >= self._goal_blacklist_radius
            cand = cand[keep]
            d = d[keep]
        stats["after_blacklist"] = int(len(cand))
        self._frontier_pick_stats = stats
        if len(cand) == 0:
            return None

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
            return False
        enable_path_validation = bool(
            self.get_parameter("enable_path_validation").value
        )
        raw_arr = np.asarray(raw, dtype=np.float32)
        decim = self.decimate_path(raw_arr, spacing=0.10)
        smooth = self.smooth_path(decim, window=3)
        if smooth is None or len(smooth) < 2:
            return False
        if enable_path_validation:
            smooth = self._trim_at_unvisited(smooth)
        if smooth is not None and len(smooth) >= 2:
            smooth = self._sanitize_path(robot_xy, path_goal, smooth)
        if smooth is None or len(smooth) < 2:
            return False

        full_path = np.asarray(smooth, dtype=np.float32)
        if enable_path_validation and not self._path_is_safe_slam(full_path):
            self.get_logger().info(
                f"[PATH] rejecting {label} path after collision validation"
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
        self._path = None
        self._active_path_goal = None
        self.navigator.reset_path()
        self._controller_path_done = False

    def _stop_and_clear_goal(self, reached=True):
        if reached and self._current_goal is not None:
            self._visited_goals.append(self._current_goal[:2].copy())
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
        if self.robot_pos is None:
            if log_now:
                self.get_logger().info("[NAV] waiting for robot pose")
            return
        if self.region is None:
            if log_now:
                self.get_logger().info("[NAV] waiting for first SLAM map")
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
                if d_frontier < 0.35:
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
                    self._visited_goals.append(self._current_goal[:2].copy())
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
            raw = self.region.generate_field_path(
                robot_xy,
                goal,
                step_size=0.10,
                max_steps=220,
                forward_only=forward_only,
            )
            if self._accept_raw_path(
                robot_xy, goal, raw, forward_only, label="field", frontier_goal=goal
            ):
                return

            safe_wp = self.region.find_safe_waypoint(
                robot_xy, goal, collision_fn=self._point_in_collision_slam
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

        self._pathgen_fail_count += 1
        if self._pathgen_fail_count >= 5:
            self.get_logger().warn(
                f"[NAV] abandoning frontier ({goal[0]:.2f},{goal[1]:.2f}) after repeated path failures"
            )
            self._visited_goals.append(goal[:2].copy())
            self._current_goal = None
            self.goal_xy = None

    def vis_tick(self):
        if self.region is not None and self.region.enable_debug_vis:
            if self.region.last_robot_pose is not None and self.region.model.function is not None:
                try:
                    src = self.region.last_robot_pose
                    self.region.save_debug_plot(src, view_radius=3.5, robot_xy=src[:2])
                except Exception as exc:
                    self.get_logger().warn(f"[VIS] debug plot failed: {exc}")
        self._publish_frontier_markers()


def main(args=None):
    rclpy.init(args=args)
    node = AntfieldsGridmapNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
