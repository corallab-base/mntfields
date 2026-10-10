#!/usr/bin/env python3
"""
astar_sdf_nav_node.py

Lightweight navigation stack for demos:
  - global planner: inflated occupancy-grid A*
  - local planner:  /sdf_local_path from sdf_local_replanner_node
  - controller:     Navigator path follower

Workflow:
  1. AMCL provides /amcl_pose on a saved 2D map.
  2. RViz 2D Goal Pose triggers A* on the occupancy map.
  3. The global path is published on /nav_path.
  4. sdf_local_replanner_node refines it using the robot-body SDF and /scan.
  5. This node follows the freshest local SDF path, or falls back to the
     global A* path when no recent refinement is available.
"""

import heapq
import math
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import Point, PoseStamped, PoseWithCovarianceStamped, Twist
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from visualization_msgs.msg import Marker, MarkerArray

from mntfields.mntfields_ros.mapping.navigation import Navigator, resample_polyline


class AstarSdfNavNode(Node):
    def __init__(self):
        super().__init__("astar_sdf_nav")

        self.declare_parameter("max_lin", 0.12)
        self.declare_parameter("max_ang", 0.6)
        self.declare_parameter("goal_reached_dist", 0.25)
        self.declare_parameter("global_frame", "map")
        self.declare_parameter("global_path_spacing", 0.08)
        self.declare_parameter("smoothing_window", 5)
        self.declare_parameter("inflation_cells", 2)
        self.declare_parameter("occ_threshold", 50)
        self.declare_parameter("unknown_is_blocked", True)
        self.declare_parameter("sdf_valid_sec", 1.0)
        self.declare_parameter("sdf_min_hold_sec", 0.75)
        self.declare_parameter("sdf_replace_threshold", 0.15)
        self.declare_parameter("control_dt", 0.10)
        self.declare_parameter("stuck_window", 25)
        self.declare_parameter("stuck_radius", 0.08)

        self.nav_frame = str(self.get_parameter("global_frame").value)

        self.robot_pos: np.ndarray | None = None
        self.robot_yaw: float = 0.0
        self._amcl_received = False
        self.goal_xy: np.ndarray | None = None

        self._map_occ: np.ndarray | None = None
        self._map_blocked: np.ndarray | None = None
        self._map_resolution: float | None = None
        self._map_origin: Tuple[float, float] | None = None

        self.global_path: np.ndarray | None = None
        self._latest_sdf_path: np.ndarray | None = None
        self._latest_sdf_path_time: float = 0.0
        self._accepted_sdf_path: np.ndarray | None = None
        self._accepted_sdf_path_time: float = 0.0
        self._active_path: np.ndarray | None = None
        self._active_mode: str = "idle"

        self.dt = float(self.get_parameter("control_dt").value)
        self._position_history = deque(
            maxlen=max(5, int(self.get_parameter("stuck_window").value))
        )

        self.path_pub = self.create_publisher(Path, "/nav_path", 10)
        self.marker_pub = self.create_publisher(MarkerArray, "/astar_sdf_nav_markers", 10)
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        self.navigator = Navigator(self)

        map_qos = QoSProfile(
            depth=1,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(OccupancyGrid, "/map", self._map_cb, map_qos)
        self.create_subscription(PoseWithCovarianceStamped, "/amcl_pose", self._amcl_pose_cb, 10)
        self.create_subscription(Odometry, "/odom", self._odom_cb, 10)
        self.create_subscription(PoseStamped, "/goal_pose", self._goal_pose_cb, 10)
        self.create_subscription(Path, "/sdf_local_path", self._sdf_path_cb, 10)

        self.create_timer(self.dt, self._control_loop)

        self.get_logger().info(
            "[A*+SDF NAV] Ready. Set initial pose with '2D Pose Estimate' and goal with "
            "'2D Goal Pose' in RViz."
        )

    # ── Pose callbacks ──────────────────────────────────────────────────────

    def _pose_to_xyyaw(self, pos, ori) -> tuple[np.ndarray, float]:
        sin_y = 2.0 * (ori.w * ori.z + ori.x * ori.y)
        cos_y = 1.0 - 2.0 * (ori.y * ori.y + ori.z * ori.z)
        yaw = math.atan2(sin_y, cos_y)
        return np.array([pos.x, pos.y, pos.z], dtype=np.float32), yaw

    def _amcl_pose_cb(self, msg: PoseWithCovarianceStamped):
        self.robot_pos, self.robot_yaw = self._pose_to_xyyaw(
            msg.pose.pose.position, msg.pose.pose.orientation
        )
        self._amcl_received = True

    def _odom_cb(self, msg: Odometry):
        if self._amcl_received:
            return
        self.robot_pos, self.robot_yaw = self._pose_to_xyyaw(
            msg.pose.pose.position, msg.pose.pose.orientation
        )

    # ── Map handling ────────────────────────────────────────────────────────

    def _map_cb(self, msg: OccupancyGrid):
        w = int(msg.info.width)
        h = int(msg.info.height)
        if w <= 0 or h <= 0:
            return
        data = np.asarray(msg.data, dtype=np.int16).reshape(h, w)
        occ_thresh = int(self.get_parameter("occ_threshold").value)
        unknown_blocked = bool(self.get_parameter("unknown_is_blocked").value)

        occ = data >= occ_thresh
        if unknown_blocked:
            occ |= data < 0

        self._map_occ = occ
        self._map_blocked = self._inflate_occ(occ, int(self.get_parameter("inflation_cells").value))
        self._map_resolution = float(msg.info.resolution)
        self._map_origin = (
            float(msg.info.origin.position.x),
            float(msg.info.origin.position.y),
        )

    @staticmethod
    def _inflate_occ(occ: np.ndarray, pad: int) -> np.ndarray:
        if pad <= 0:
            return occ.copy()
        h, w = occ.shape
        out = occ.copy()
        occ_idx = np.argwhere(occ)
        for r, c in occ_idx:
            r0 = max(0, r - pad)
            r1 = min(h, r + pad + 1)
            c0 = max(0, c - pad)
            c1 = min(w, c + pad + 1)
            out[r0:r1, c0:c1] = True
        return out

    def _map_indices(self, xy: np.ndarray) -> Optional[Tuple[int, int]]:
        if self._map_resolution is None or self._map_origin is None or self._map_occ is None:
            return None
        col = int(math.floor((float(xy[0]) - self._map_origin[0]) / self._map_resolution))
        row_from_bottom = int(math.floor((float(xy[1]) - self._map_origin[1]) / self._map_resolution))
        h, w = self._map_occ.shape
        row = h - 1 - row_from_bottom
        if row < 0 or row >= h or col < 0 or col >= w:
            return None
        return row, col

    def _cell_to_world(self, row: int, col: int) -> np.ndarray:
        assert self._map_resolution is not None and self._map_origin is not None and self._map_occ is not None
        h, _ = self._map_occ.shape
        row_from_bottom = h - 1 - row
        x = self._map_origin[0] + (col + 0.5) * self._map_resolution
        y = self._map_origin[1] + (row_from_bottom + 0.5) * self._map_resolution
        return np.array([x, y], dtype=np.float32)

    def _nearest_free_xy(self, xy: np.ndarray, max_rad: int = 25) -> Optional[np.ndarray]:
        idx = self._map_indices(xy)
        if idx is None or self._map_blocked is None:
            return None
        r0, c0 = idx
        h, w = self._map_blocked.shape
        if not self._map_blocked[r0, c0]:
            return self._cell_to_world(r0, c0)
        for rad in range(1, max_rad + 1):
            rr_min = max(0, r0 - rad)
            rr_max = min(h - 1, r0 + rad)
            cc_min = max(0, c0 - rad)
            cc_max = min(w - 1, c0 + rad)
            best = None
            best_d2 = 1e18
            for rr in range(rr_min, rr_max + 1):
                for cc in range(cc_min, cc_max + 1):
                    if abs(rr - r0) != rad and abs(cc - c0) != rad:
                        continue
                    if self._map_blocked[rr, cc]:
                        continue
                    d2 = float((rr - r0) ** 2 + (cc - c0) ** 2)
                    if d2 < best_d2:
                        best_d2 = d2
                        best = (rr, cc)
            if best is not None:
                return self._cell_to_world(best[0], best[1])
        return None

    # ── A* planning ─────────────────────────────────────────────────────────

    def _astar_cells(
        self, start_rc: Tuple[int, int], goal_rc: Tuple[int, int]
    ) -> Optional[List[Tuple[int, int]]]:
        if self._map_blocked is None:
            return None

        h, w = self._map_blocked.shape
        sr, sc = start_rc
        gr, gc = goal_rc
        if not (0 <= sr < h and 0 <= sc < w and 0 <= gr < h and 0 <= gc < w):
            return None
        if self._map_blocked[sr, sc] or self._map_blocked[gr, gc]:
            return None

        def hcost(r: int, c: int) -> float:
            return math.hypot(r - gr, c - gc)

        nbrs = [
            (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, math.sqrt(2.0)), (-1, 1, math.sqrt(2.0)),
            (1, -1, math.sqrt(2.0)), (1, 1, math.sqrt(2.0)),
        ]

        open_heap: List[Tuple[float, float, Tuple[int, int]]] = []
        heapq.heappush(open_heap, (hcost(sr, sc), 0.0, (sr, sc)))
        g_score: Dict[Tuple[int, int], float] = {(sr, sc): 0.0}
        parent: Dict[Tuple[int, int], Tuple[int, int]] = {}
        closed: Set[Tuple[int, int]] = set()

        while open_heap:
            _f, g_cur, cur = heapq.heappop(open_heap)
            if cur in closed:
                continue
            if cur == (gr, gc):
                path = [cur]
                while cur in parent:
                    cur = parent[cur]
                    path.append(cur)
                path.reverse()
                return path

            closed.add(cur)
            r, c = cur
            for dr, dc, step_cost in nbrs:
                nr = r + dr
                nc = c + dc
                if nr < 0 or nr >= h or nc < 0 or nc >= w:
                    continue
                if self._map_blocked[nr, nc]:
                    continue
                ng = g_cur + step_cost
                old = g_score.get((nr, nc))
                if old is not None and ng >= old:
                    continue
                g_score[(nr, nc)] = ng
                parent[(nr, nc)] = (r, c)
                heapq.heappush(open_heap, (ng + hcost(nr, nc), ng, (nr, nc)))

        return None

    def _astar_world(self, start_xy: np.ndarray, goal_xy: np.ndarray) -> Optional[np.ndarray]:
        s = self._nearest_free_xy(start_xy)
        g = self._nearest_free_xy(goal_xy)
        if s is None or g is None:
            return None
        src = self._map_indices(s)
        dst = self._map_indices(g)
        if src is None or dst is None:
            return None
        cells = self._astar_cells(src, dst)
        if not cells:
            return None
        pts = np.asarray([self._cell_to_world(r, c) for r, c in cells], dtype=np.float32)
        return self._postprocess_path(pts)

    def _postprocess_path(self, path_xy: np.ndarray) -> np.ndarray:
        path = np.asarray(path_xy, dtype=np.float32)
        if len(path) < 2:
            return path
        ds = max(0.04, float(self.get_parameter("global_path_spacing").value))
        path = resample_polyline(path, ds=ds)
        window = max(1, int(self.get_parameter("smoothing_window").value))
        if window > 1 and len(path) >= window:
            sm = path.copy()
            k = window // 2
            for i in range(len(path)):
                a = max(0, i - k)
                b = min(len(path), i + k + 1)
                sm[i] = path[a:b].mean(axis=0)
            sm[0] = path[0]
            sm[-1] = path[-1]
            path = sm
        return path.astype(np.float32)

    # ── Goal and local path handling ────────────────────────────────────────

    def _goal_pose_cb(self, msg: PoseStamped):
        if self.robot_pos is None:
            self.get_logger().warn("[A*+SDF NAV] No pose yet; ignoring goal.")
            return
        if self._map_blocked is None:
            self.get_logger().warn("[A*+SDF NAV] No map yet; ignoring goal.")
            return

        self.goal_xy = np.array([msg.pose.position.x, msg.pose.position.y], dtype=np.float32)
        self._latest_sdf_path = None
        self._accepted_sdf_path = None
        self._active_path = None
        self.navigator.reset_path()
        self._position_history.clear()

        path = self._astar_world(self.robot_pos[:2], self.goal_xy)
        if path is None or len(path) < 2:
            self.global_path = None
            self._active_mode = "astar_failed"
            self.get_logger().warn("[A*+SDF NAV] A* failed to find a path to goal.")
            self._publish_markers()
            return

        self.global_path = path
        self._active_path = self.global_path
        self._active_mode = "astar_global"
        self._publish_global_path()
        self._publish_markers()
        self.get_logger().info(
            f"[A*+SDF NAV] New goal: ({self.goal_xy[0]:.2f}, {self.goal_xy[1]:.2f}) "
            f"global_pts={len(self.global_path)}"
        )

    def _sdf_path_cb(self, msg: Path):
        if len(msg.poses) < 2:
            return
        self._latest_sdf_path = np.array(
            [[p.pose.position.x, p.pose.position.y] for p in msg.poses],
            dtype=np.float32,
        )
        self._latest_sdf_path_time = self.get_clock().now().nanoseconds * 1e-9

    def _maybe_accept_sdf_path(self, now_sec: float, valid_for: float) -> Optional[np.ndarray]:
        latest = self._latest_sdf_path
        if latest is None or now_sec - self._latest_sdf_path_time >= valid_for or len(latest) < 2:
            self._accepted_sdf_path = None
            return None

        accepted = self._accepted_sdf_path
        if accepted is None or now_sec - self._accepted_sdf_path_time >= valid_for:
            self._accepted_sdf_path = latest.copy()
            self._accepted_sdf_path_time = now_sec
            return self._accepted_sdf_path

        hold_sec = float(self.get_parameter("sdf_min_hold_sec").value)
        if now_sec - self._accepted_sdf_path_time < hold_sec:
            return accepted

        replace_threshold = float(self.get_parameter("sdf_replace_threshold").value)
        if self._path_change_metric(accepted, latest) < replace_threshold:
            return accepted

        self._accepted_sdf_path = latest.copy()
        self._accepted_sdf_path_time = now_sec
        return self._accepted_sdf_path

    @staticmethod
    def _resample_path_for_compare(path: np.ndarray, samples: int) -> Optional[np.ndarray]:
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

    @staticmethod
    def _path_change_metric(a: np.ndarray, b: np.ndarray, samples: int = 16) -> float:
        ra = AstarSdfNavNode._resample_path_for_compare(a, samples)
        rb = AstarSdfNavNode._resample_path_for_compare(b, samples)
        if ra is None or rb is None:
            return float("inf")
        return float(np.mean(np.linalg.norm(ra - rb, axis=1)))

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

    # ── Publishing helpers ───────────────────────────────────────────────────

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

    def _make_line_marker(
        self,
        marker_id: int,
        ns: str,
        pts: np.ndarray,
        rgb: tuple[float, float, float],
        width: float,
        alpha: float,
    ) -> Marker:
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
        marker.ns = "status"
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
        for idx, ns in enumerate(["goal", "global", "local", "active", "status"]):
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
            goal.ns = "goal"
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
                self._make_line_marker(0, "global", self.global_path, (1.0, 0.85, 0.10), 0.03, 0.65)
            )
        if self._latest_sdf_path is not None and len(self._latest_sdf_path) >= 2:
            arr.markers.append(
                self._make_line_marker(0, "local", self._latest_sdf_path, (0.10, 0.95, 1.0), 0.05, 0.65)
            )
        if self._accepted_sdf_path is not None and len(self._accepted_sdf_path) >= 2:
            arr.markers.append(
                self._make_line_marker(1, "local", self._accepted_sdf_path, (0.00, 0.65, 1.0), 0.08, 0.90)
            )
        if self._active_path is not None and len(self._active_path) >= 2:
            arr.markers.append(
                self._make_line_marker(0, "active", self._active_path, (0.20, 1.0, 0.20), 0.07, 0.95)
            )

        arr.markers.append(self._make_text_marker(0, f"mode: {self._active_mode}"))
        self.marker_pub.publish(arr)

    # ── Control loop ────────────────────────────────────────────────────────

    def _is_stuck(self) -> bool:
        if len(self._position_history) < self._position_history.maxlen:
            return False
        pts = np.asarray(self._position_history, dtype=np.float32)
        disp = float(np.linalg.norm(pts[-1] - pts[0]))
        return disp < float(self.get_parameter("stuck_radius").value)

    def _control_loop(self):
        self._publish_global_path()

        if self.robot_pos is None:
            self._active_path = None
            self._active_mode = "waiting_pose"
            self._publish_markers()
            self.navigator.stop_robot()
            return

        self._position_history.append(self.robot_pos[:2].copy())

        if self.goal_xy is None:
            self._active_path = None
            self._active_mode = "idle"
            self._publish_markers()
            self.navigator.stop_robot()
            return

        robot_xy = self.robot_pos[:2]
        d_goal = float(np.linalg.norm(robot_xy - self.goal_xy))
        if d_goal < float(self.get_parameter("goal_reached_dist").value):
            self.get_logger().info(f"[A*+SDF NAV] Goal reached: d={d_goal:.3f}m")
            self.navigator.stop_robot()
            self.navigator.reset_path()
            self.goal_xy = None
            self.global_path = None
            self._latest_sdf_path = None
            self._accepted_sdf_path = None
            self._active_path = None
            self._active_mode = "goal_reached"
            self._publish_markers()
            return

        now_sec = self.get_clock().now().nanoseconds * 1e-9
        active = self.global_path
        mode = "astar_global"
        accepted = self._maybe_accept_sdf_path(now_sec, float(self.get_parameter("sdf_valid_sec").value))
        if accepted is not None and self.global_path is not None and len(accepted) >= 2:
            active = self._stitch_local_to_global(accepted, self.global_path)
            mode = "sdf_local"

        self._active_path = active
        self._active_mode = mode
        self._publish_markers()

        if self._active_path is None:
            self.navigator.stop_robot()
            return

        if self._is_stuck():
            self.get_logger().warn(
                f"[A*+SDF NAV] No progress detected while following {self._active_mode}.",
                throttle_duration_sec=2.0,
            )

        self.navigator.drive_path(self._active_path)


def main(args=None):
    rclpy.init(args=args)
    node = AstarSdfNavNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.navigator.stop_robot()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
