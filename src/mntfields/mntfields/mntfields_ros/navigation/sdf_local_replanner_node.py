#!/usr/bin/env python3
"""
sdf_local_replanner_node.py

Subscribes to the mntfields global path (/nav_path) and a lidar scan
(/scan), then uses TrajectoryOptimizerBodySDF from comp-sdf-dyn-nav
to locally refine the trajectory around obstacles.

Publishes the refined path on /sdf_local_path at ~replan_interval_sec Hz.
The navigation_node falls back to the global path if no fresh SDF path
arrives within 2 s.

Architecture:
    /nav_path (global)  ──→  SDF Local Replanner
    /scan               ──→      (optimize every 0.5 s in background thread)
    TF: global_frame → lidar frame        │
                                 ↓
                        /sdf_local_path  →  navigation_node
"""

import sys
import threading
import time as _time
import json
import os
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time

from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Path as PathMsg
from sensor_msgs.msg import LaserScan, PointCloud2, PointField
from tf2_ros import Buffer, TransformListener


# ── Stub ──────────────────────────────────────────────────────────────────────

class _SDFMapStub:
    """Minimal stub satisfying TrajectoryOptimizerBodySDF's interface.

    TrajectoryOptimizerBodySDF only reads:
      - sdf_infer_map.device
      - sdf_infer_map.sdf_obj.decode_sdf(queries, latent_vector)
    No iGibson / InferenceSingleViewAugSDF needed.
    """

    def __init__(self, sdf_obj, device):
        self.sdf_obj = sdf_obj
        self.device  = device


# ── Node ──────────────────────────────────────────────────────────────────────

class SDFLocalReplannerNode(Node):

    def __init__(self):
        super().__init__('sdf_local_replanner')

        # ── Parameters ───────────────────────────────────────────────────────
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
        self.declare_parameter('global_frame',          'map')
        self.declare_parameter('base_frame',            'base_link')
        self.declare_parameter('metrics_path',          '')

        sdf_repo_dir = self.get_parameter('sdf_repo_dir').value
        self._global_frame = str(self.get_parameter('global_frame').value)
        self._base_frame = str(self.get_parameter('base_frame').value)
        self._lidar_scan_topic = str(self.get_parameter('lidar_scan_topic').value)
        _metrics_path = str(self.get_parameter('metrics_path').value).strip()
        model_dir = os.path.expanduser("~/turtlebot3_ws/mntfields_model")
        self._metrics_path = (_metrics_path if _metrics_path
                              else os.path.join(model_dir, 'sdf_replanner_metrics.json'))

        # ── Shared state (all accesses protected by _lock) ────────────────────
        self._lock       = threading.Lock()
        self._path_xy: np.ndarray | None = None    # Nx2 global path
        self._pc_xyz:  np.ndarray | None = None    # Mx3 points in lidar frame
        self._pc_stamp                   = None
        self._pc_frame: str              = ''
        self._robot_xy: np.ndarray | None = None   # [x, y] in map frame
        self._opt_running                = False

        # SDF optimizer state (set by _init_sdf_optimizer, read in _run_optimization)
        self._traj_opt      = None
        self._sdf_available = False
        self._device        = None

        self._amcl_received = False
        self._last_status_log = 0.0
        self._logged_pc_frame = False
        self._last_local_path: np.ndarray | None = None
        self._last_local_path_time = 0.0
        self._metrics_run_start_wall_time = _time.time()
        self._metrics_run_start_mono = _time.monotonic()
        self._sdf_opt_metrics: list[dict] = []

        # ── TF listener ──────────────────────────────────────────────────────
        self._tf_buf      = Buffer()
        self._tf_listener = TransformListener(self._tf_buf, self)

        # ── Publisher ────────────────────────────────────────────────────────
        self._path_pub = self.create_publisher(PathMsg, '/sdf_local_path', 10)
        self._debug_pc_pub = self.create_publisher(PointCloud2, '/sdf_points_debug', 10)

        # ── Subscriptions ────────────────────────────────────────────────────
        self.create_subscription(PathMsg,   '/nav_path',
                                 self._nav_path_cb, 10)
        self.create_subscription(
            LaserScan,
            self._lidar_scan_topic,
            self._scan_cb,
            qos_profile_sensor_data,
        )
        self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose',
                                 self._amcl_pose_cb, 10)
        # ── Replan timer ─────────────────────────────────────────────────────
        interval = float(self.get_parameter('replan_interval_sec').value)
        self.create_timer(interval, self._replan_tick)

        # ── Load SDF optimizer in background so node starts immediately ───────
        threading.Thread(
            target=self._init_sdf_optimizer,
            args=(sdf_repo_dir,),
            daemon=True,
        ).start()

        self.get_logger().info('[SDF] Node started. Loading SDF optimizer...')

    # ── SDF initialisation ────────────────────────────────────────────────────

    def _init_sdf_optimizer(self, sdf_repo_dir: str):
        try:
            import types
            import torch
            sys.path.insert(0, sdf_repo_dir)

            # planner.py imports InferenceSingleViewAugSDF and YoloFramePredictions at
            # module level — both trigger deep YOLO / iGibson chains that aren't
            # available here.  Install lightweight stubs into sys.modules BEFORE the
            # import so Python resolves them without running those chains.
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
            self._device       = device

            ckpt_dir  = Path(sdf_repo_dir) / 'ckpts' / 'sdf-object'
            sdf_ckpt  = ckpt_dir / 'sdf-obj-trunc0.5-latest.pth'
            latent_fp = ckpt_dir / 'latent-code-turtlebot4-bbox-tight-trunc0.5.pth'

            # Build DeepSDF decoder with config from inference_single_view.yaml
            sdf_obj = DeepSdfDecoderMap(
                latent_size=256,
                dims=[512] * 8,
                dropout=list(range(8)),
                dropout_prob=0.2,
                norm_layers=list(range(8)),
                latent_in=[4],
                weight_norm=True,
            )

            # Load weights (may be wrapped under 'model_state_dict')
            state = torch.load(str(sdf_ckpt), map_location=device)
            if isinstance(state, dict) and 'model_state_dict' in state:
                state = state['model_state_dict']
            if isinstance(state, dict):
                state = {
                    (k[7:] if k.startswith('module.') else k): v
                    for k, v in state.items()
                }
            sdf_obj.load_state_dict(state)
            sdf_obj.to(device).eval()

            stub = _SDFMapStub(sdf_obj, device)

            trunc  = float(self.get_parameter('trunc_dist_sdf').value)
            smooth = float(self.get_parameter('smoothing').value)
            opt_it = int(self.get_parameter('opt_iterations').value)

            self._traj_opt = TrajectoryOptimizerBodySDF(
                sdf_infer_map=stub,
                body_sdf_latent_code_filepath=latent_fp,
                iterations_per_update=opt_it,
                lr_init=0.1,
                lr_decay=0.876,
                trunc_dist_sdf=trunc,
                smoothing=smooth,
                pc_memory_radius=1.5,
            )

            self._sdf_available = True
            self.get_logger().info(f'[SDF] Optimizer initialized on {device}.')

        except Exception as e:
            self.get_logger().error(f'[SDF] Optimizer init failed: {e}')
            import traceback
            traceback.print_exc()

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _nav_path_cb(self, msg: PathMsg):
        if not msg.poses:
            return
        pts = np.array(
            [[p.pose.position.x, p.pose.position.y] for p in msg.poses],
            dtype=np.float32,
        )
        with self._lock:
            self._path_xy = pts

    def _scan_cb(self, msg: LaserScan):
        pts = self._scan_to_pointcloud(msg)
        if pts is None or len(pts) == 0:
            return
        effective_frame = msg.header.frame_id
        if not self._logged_pc_frame:
            self.get_logger().info(
                f'[SDF] Using lidar scan topic {self._lidar_scan_topic} in frame={effective_frame}')
            self._logged_pc_frame = True
        with self._lock:
            self._pc_xyz   = pts
            self._pc_stamp = msg.header.stamp
            self._pc_frame = effective_frame

    def _amcl_pose_cb(self, msg: PoseWithCovarianceStamped):
        p = msg.pose.pose.position
        with self._lock:
            self._robot_xy = np.array([p.x, p.y], dtype=np.float32)
        self._amcl_received = True

    def _update_robot_pose_from_map_tf(self) -> None:
        frames = [self._base_frame]
        if self._base_frame != 'base_footprint':
            frames.append('base_footprint')
        if self._base_frame != 'base_link':
            frames.append('base_link')

        for frame in frames:
            try:
                tf_msg = self._tf_buf.lookup_transform(
                    self._global_frame,
                    frame,
                    Time(),
                    timeout=Duration(seconds=0.02),
                )
            except Exception:
                continue
            p = tf_msg.transform.translation
            with self._lock:
                self._robot_xy = np.array([p.x, p.y], dtype=np.float32)
            return

    # ── Replan timer ──────────────────────────────────────────────────────────

    def _replan_tick(self):
        if not self._sdf_available:
            return
        self._update_robot_pose_from_map_tf()

        with self._lock:
            if self._opt_running:
                return
            missing = []
            if self._path_xy is None:
                missing.append('nav_path')
            if self._pc_xyz is None or self._pc_stamp is None:
                missing.append('lidar_scan')
            if self._robot_xy is None:
                missing.append('robot_pose')
            if missing:
                self._log_waiting_status(missing)
                return

            # Snapshot shared state so the background thread doesn't need the lock
            pc_sensor  = self._pc_xyz.copy()
            pc_stamp   = self._pc_stamp
            pc_frame   = self._pc_frame
            robot_xy   = self._robot_xy.copy()
            path       = self._path_xy.copy()
            local_seed = (None if self._last_local_path is None
                          else self._last_local_path.copy())
            local_seed_time = self._last_local_path_time
            self._opt_running = True

        threading.Thread(
            target=self._run_optimization,
            args=(pc_sensor, pc_stamp, pc_frame, robot_xy, path, local_seed, local_seed_time),
            daemon=True,
        ).start()

    def _run_optimization(self, pc_sensor, pc_stamp, pc_frame, robot_xy, path, local_seed, local_seed_time):
        import torch
        t0 = _time.monotonic()
        metric = {
            'start_xy': [float(robot_xy[0]), float(robot_xy[1])],
            'pc_frame': str(pc_frame),
            'input_path_points': int(len(path)) if path is not None else 0,
            'seed_path_points': 0,
            'ahead_segment_points': 0,
            'pointcloud_points': int(len(pc_sensor)) if pc_sensor is not None else 0,
            'prep_ms': None,
            'optimizer_inference_ms': None,
            'total_sdf_cycle_ms': None,
            'published': False,
            'failure_reason': None,
        }
        try:
            c2w = self._get_c2w(pc_stamp, pc_frame)
            if c2w is None:
                metric['failure_reason'] = 'missing_tf'
                self.get_logger().warn(
                    f'[SDF] No TF {self._global_frame} <- {pc_frame} for point cloud; skipping replan.',
                    throttle_duration_sec=2.0)
                return

            seg_global = self._extract_ahead_segment(
                path=path,
                robot_xy=robot_xy,
                max_ahead=float(self.get_parameter('max_ahead_dist').value),
                max_pts=int(self.get_parameter('max_traj_pts').value),
            )
            if seg_global is None or len(seg_global) < 3:
                metric['failure_reason'] = 'ahead_segment_too_short'
                self.get_logger().warn(
                    '[SDF] Ahead segment too short for optimization; keeping global path.',
                    throttle_duration_sec=2.0)
                return
            metric['ahead_segment_points'] = int(len(seg_global))

            seg_seed = self._select_seed_segment(
                robot_xy=robot_xy,
                seg_global=seg_global,
                local_seed=local_seed,
                local_seed_time=local_seed_time,
            )
            metric['seed_path_points'] = int(len(seg_seed)) if seg_seed is not None else 0

            pc_world = self._transform_pc(pc_sensor, c2w)
            self._publish_debug_pointcloud(pc_world)
            pc_t = torch.from_numpy(pc_world).float().to(self._device)

            # Lidar input does not come with pinhole intrinsics. For this path we
            # only need bounded obstacle memory around the robot, so update the
            # robot state directly and skip the camera-frustum pruning step.
            self._traj_opt.pc_memory.set_robot_state(robot_xy)

            traj_t = torch.from_numpy(seg_seed).float()
            z_val  = float(self.get_parameter('z_value').value)
            prep_done = _time.monotonic()

            refined = self._traj_opt.optimize_trajectory(
                pc=pc_t,
                obj_det_preds=None,
                traj_pt2ds=traj_t,
                z_value=z_val,
            )
            infer_done = _time.monotonic()
            metric['prep_ms'] = round((prep_done - t0) * 1e3, 3)
            metric['optimizer_inference_ms'] = round((infer_done - prep_done) * 1e3, 3)

            refined_np = refined.detach().cpu().numpy().astype(np.float32)
            refined_np = self._stabilize_refined_path(refined_np, robot_xy, seg_global)
            if not self._should_publish_result(
                    refined_np=refined_np,
                    robot_xy_start=robot_xy,
                    seg_end=seg_global[-1],
                    elapsed_sec=_time.monotonic() - t0):
                metric['failure_reason'] = 'result_rejected'
                return

            self._publish_refined_path(refined_np, _time.monotonic() - t0)
            metric['published'] = True

        except Exception as e:
            metric['failure_reason'] = f'exception:{e}'
            self.get_logger().warn(f'[SDF] Optimization error: {e}')
        finally:
            metric['total_sdf_cycle_ms'] = round((_time.monotonic() - t0) * 1e3, 3)
            self._sdf_opt_metrics.append(metric)
            with self._lock:
                self._opt_running = False

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _log_waiting_status(self, missing: list[str]):
        now = _time.monotonic()
        period = float(self.get_parameter('status_log_sec').value)
        if now - self._last_status_log < period:
            return
        self._last_status_log = now
        self.get_logger().info(f'[SDF] Waiting for: {", ".join(missing)}')

    def _stabilize_refined_path(
        self,
        refined_np: np.ndarray,
        robot_xy: np.ndarray,
        seg: np.ndarray,
    ) -> np.ndarray:
        if refined_np.ndim != 2 or len(refined_np) == 0:
            return refined_np

        out = np.asarray(refined_np, dtype=np.float32).copy()

        # Keep local-path handoff stable for the downstream tracker:
        # start exactly at the live robot pose and preserve the original
        # segment endpoint so each refresh targets the same horizon.
        out[0, :2] = robot_xy[:2]
        if len(out) > 1:
            out[-1, :2] = seg[-1, :2]

        # Remove near-duplicate consecutive points that can create heading jitter.
        keep = [0]
        for i in range(1, len(out)):
            if np.linalg.norm(out[i, :2] - out[keep[-1], :2]) >= 0.03:
                keep.append(i)
        if keep[-1] != len(out) - 1:
            keep.append(len(out) - 1)
        return out[keep].astype(np.float32)

    def _select_seed_segment(
        self,
        robot_xy: np.ndarray,
        seg_global: np.ndarray,
        local_seed: np.ndarray | None,
        local_seed_time: float,
    ) -> np.ndarray:
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

        # Keep the rolling horizon attached to the global path endpoint so the
        # local replanner refines a short window instead of drifting off-route.
        if float(np.linalg.norm(seg_local[-1, :2] - seg_global[-1, :2])) > 0.50:
            return seg_global

        seg_local = seg_local.astype(np.float32, copy=True)
        seg_local[0, :2] = robot_xy[:2]
        seg_local[-1, :2] = seg_global[-1, :2]
        return seg_local

    def _should_publish_result(
        self,
        refined_np: np.ndarray,
        robot_xy_start: np.ndarray,
        seg_end: np.ndarray,
        elapsed_sec: float,
    ) -> bool:
        max_age = float(self.get_parameter('max_result_age_sec').value)
        if elapsed_sec > max_age:
            self.get_logger().info(
                f'[SDF] Dropping stale local path ({elapsed_sec:.2f}s > {max_age:.2f}s).',
                throttle_duration_sec=1.0)
            return False

        with self._lock:
            robot_xy_now = None if self._robot_xy is None else self._robot_xy.copy()
            last_local = None if self._last_local_path is None else self._last_local_path.copy()

        if robot_xy_now is not None:
            moved = float(np.linalg.norm(robot_xy_now - robot_xy_start))
            max_motion = float(self.get_parameter('max_robot_motion_m').value)
            if moved > max_motion:
                self.get_logger().info(
                    f'[SDF] Dropping stale local path after robot moved {moved:.2f}m.',
                    throttle_duration_sec=1.0)
                return False

        if len(refined_np) < 2:
            return False

        # Keep the optimizer attached to the same local horizon.
        if float(np.linalg.norm(refined_np[-1, :2] - seg_end[:2])) > 0.25:
            self.get_logger().info(
                '[SDF] Dropping unstable local path with drifting endpoint.',
                throttle_duration_sec=1.0)
            return False

        if last_local is not None and len(last_local) >= 2:
            delta = self._path_delta_m(last_local, refined_np)
            min_delta = float(self.get_parameter('min_publish_delta_m').value)
            if delta < min_delta:
                return False

        with self._lock:
            self._last_local_path = refined_np.copy()
            self._last_local_path_time = _time.monotonic()
        return True

    @staticmethod
    def _path_delta_m(path_a: np.ndarray, path_b: np.ndarray) -> float:
        if len(path_a) == 0 or len(path_b) == 0:
            return float('inf')
        n = min(len(path_a), len(path_b), 12)
        if n <= 0:
            return float('inf')
        idx_a = np.linspace(0, len(path_a) - 1, n, dtype=int)
        idx_b = np.linspace(0, len(path_b) - 1, n, dtype=int)
        pts_a = path_a[idx_a, :2]
        pts_b = path_b[idx_b, :2]
        return float(np.mean(np.linalg.norm(pts_a - pts_b, axis=1)))

    def _get_c2w(self, stamp, source_frame) -> 'np.ndarray | None':
        """Lookup TF global_frame → source_frame, return 4×4 float32 homogeneous matrix."""
        try:
            tf = self._tf_buf.lookup_transform(
                self._global_frame, source_frame, stamp,
                timeout=Duration(seconds=0.05),
            )
        except Exception:
            try:
                tf = self._tf_buf.lookup_transform(
                    self._global_frame, source_frame, Time(),
                    timeout=Duration(seconds=0.05),
                )
            except Exception:
                return None

        t  = tf.transform.translation
        q  = tf.transform.rotation
        tx, ty, tz       = t.x, t.y, t.z
        qx, qy, qz, qw  = q.x, q.y, q.z, q.w

        # Quaternion → rotation matrix (passive, right-handed)
        r00 = 1.0 - 2.0*(qy*qy + qz*qz)
        r01 = 2.0*(qx*qy - qz*qw)
        r02 = 2.0*(qx*qz + qy*qw)
        r10 = 2.0*(qx*qy + qz*qw)
        r11 = 1.0 - 2.0*(qx*qx + qz*qz)
        r12 = 2.0*(qy*qz - qx*qw)
        r20 = 2.0*(qx*qz - qy*qw)
        r21 = 2.0*(qy*qz + qx*qw)
        r22 = 1.0 - 2.0*(qx*qx + qy*qy)

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
        """Transform Nx3 sensor-frame points → world frame (Nx3)."""
        if len(pc_sensor) == 0:
            return np.zeros((0, 3), dtype=np.float32)
        ones     = np.ones((len(pc_sensor), 1), dtype=np.float32)
        pts_h    = np.concatenate([pc_sensor, ones], axis=1)   # Nx4
        pts_world = (c2w @ pts_h.T).T[:, :3]                 # Nx3
        return pts_world.astype(np.float32)

    @staticmethod
    def _extract_ahead_segment(
        path: np.ndarray,
        robot_xy: np.ndarray,
        max_ahead: float,
        max_pts: int,
    ) -> 'np.ndarray | None':
        """Extract a map-frame horizon starting at the robot's path projection."""
        if len(path) < 2:
            return None

        pts = np.asarray(path[:, :2], dtype=np.float32)
        robot = np.asarray(robot_xy[:2], dtype=np.float32)
        a = pts[:-1]
        b = pts[1:]
        ab = b - a
        denom = np.einsum('ij,ij->i', ab, ab)
        denom = np.where(denom > 1e-9, denom, 1e-9)
        t = np.clip(np.einsum('ij,ij->i', robot[None, :] - a, ab) / denom, 0.0, 1.0)
        proj = a + t[:, None] * ab
        d2 = np.einsum('ij,ij->i', proj - robot[None, :], proj - robot[None, :])
        seg_i = int(np.argmin(d2))
        seg_t = float(t[seg_i])

        seg = [robot.copy()]
        if seg_t < 0.98:
            seg.append(pts[seg_i + 1].copy())
            tail_start = seg_i + 2
        else:
            tail_start = seg_i + 2
        for p in pts[tail_start:]:
            seg.append(p.copy())

        seg = np.asarray(seg, dtype=np.float32)
        if len(seg) < 2:
            return None

        # Cap by arc-length
        deltas  = np.linalg.norm(np.diff(seg, axis=0), axis=1)
        cum     = np.concatenate([[0.0], np.cumsum(deltas)])
        end_idx = int(np.searchsorted(cum, max_ahead, side='right'))
        seg     = seg[:end_idx + 1]

        if len(seg) < 2:
            return None

        # Subsample
        if len(seg) > max_pts:
            idxs = np.linspace(0, len(seg) - 1, max_pts, dtype=int)
            seg  = seg[idxs]

        return seg.astype(np.float32)

    def _publish_refined_path(self, pts: np.ndarray, elapsed_sec: float | None = None):
        msg = PathMsg()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = self._global_frame
        for p in pts:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x    = float(p[0])
            ps.pose.position.y    = float(p[1])
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        self._path_pub.publish(msg)
        suffix = ''
        if elapsed_sec is not None:
            suffix = f' in {elapsed_sec:.2f}s'
        self.get_logger().info(
            f'[SDF] Published local path with {len(msg.poses)} points{suffix}.',
            throttle_duration_sec=0.5)

    def _publish_debug_pointcloud(self, pts: np.ndarray):
        if pts is None or len(pts) == 0:
            return
        xyz = np.asarray(pts, dtype=np.float32)
        msg = PointCloud2()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._global_frame
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
        self._debug_pc_pub.publish(msg)

    @staticmethod
    def _summary_stats(values):
        vals = [float(v) for v in values if v is not None]
        if not vals:
            return {'count': 0, 'mean': None, 'median': None, 'min': None, 'max': None}
        vals = sorted(vals)
        n = len(vals)
        mid = n // 2
        median = vals[mid] if n % 2 == 1 else 0.5 * (vals[mid - 1] + vals[mid])
        return {
            'count': int(n),
            'mean': float(sum(vals) / n),
            'median': float(median),
            'min': float(vals[0]),
            'max': float(vals[-1]),
        }

    def _save_metrics_json(self):
        try:
            Path(self._metrics_path).parent.mkdir(parents=True, exist_ok=True)
            payload = {
                'run_started_unix_s': float(self._metrics_run_start_wall_time),
                'run_duration_s': float(_time.monotonic() - self._metrics_run_start_mono),
                'sdf_optimization_metrics': list(self._sdf_opt_metrics),
                'summary': {
                    'sdf_optimizer_inference_ms': self._summary_stats(
                        [m.get('optimizer_inference_ms') for m in self._sdf_opt_metrics]
                    ),
                    'sdf_total_cycle_ms': self._summary_stats(
                        [m.get('total_sdf_cycle_ms') for m in self._sdf_opt_metrics]
                    ),
                },
            }
            Path(self._metrics_path).write_text(json.dumps(payload, indent=2), encoding='utf-8')
            self.get_logger().info(f'[SDF] Saved replanner metrics: {self._metrics_path}')
        except Exception as e:
            self.get_logger().warn(f'[SDF] Failed to save replanner metrics: {e}')

    def destroy_node(self):
        self._save_metrics_json()
        super().destroy_node()


# ── Entry point ───────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = SDFLocalReplannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
