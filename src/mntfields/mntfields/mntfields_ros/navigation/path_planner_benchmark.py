#!/usr/bin/env python3
"""
path_planner_benchmark.py

Extends RegionGraphInspector to benchmark four path planners side-by-side:
  1. NTFields    — region graph + neural metric field gradient descent
  2. A*          — grid-based A* on the inflated occupancy map
  3. RRT*        — asymptotically-optimal sampling-based planner
  4. RRT         — basic rapidly-exploring random tree (first feasible path)

Workflow (same as RegionGraphInspector):
  - Set start with '2D Pose Estimate' in RViz  → /initialpose
  - Set goal  with '2D Goal Pose'  in RViz      → /goal_pose
  - All planners fire immediately; timing + path length logged to console.

Topics published:
  /region_graph_path    — NTFields   (green, thick)
  /astar_path           — A*         (yellow)
  /rrtstar_path         — RRT*       (cyan)
  /rrt_path             — RRT        (orange)

Launch:
  ros2 launch mntfields path_planning.launch.py model_dir:=/path/to/mntfields_model
"""

import threading
import time
from typing import List, Optional, Tuple

import networkx as nx
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Path
from rclpy.node import Node

from mntfields.mntfields_ros.navigation.region_graph_inspector import RegionGraphInspector
from mntfields.utils.mntfields_utils import straight_line_path


class PathPlannerBenchmark(RegionGraphInspector):
    """
    Subclass of RegionGraphInspector that adds A* and RRT* planners for comparison.
    All three planners run on each new start+goal query and results are logged.
    """

    def __init__(self):
        super().__init__()

        self.astar_path_pub = self.create_publisher(Path, "/astar_path", 10)
        self.rrtstar_path_pub = self.create_publisher(Path, "/rrtstar_path", 10)
        self.rrt_path_pub = self.create_publisher(Path, "/rrt_path", 10)
        self._benchmark_running = False

        # RRT* params
        self.declare_parameter("rrtstar_max_iter", 5000)
        self.declare_parameter("rrtstar_step_size", 0.40)
        self.declare_parameter("rrtstar_neighborhood_radius", 0.50)
        self.declare_parameter("rrtstar_goal_sample_rate", 0.10)
        self.declare_parameter("rrtstar_goal_tolerance", 0.35)

        # RRT params
        self.declare_parameter("rrt_max_iter", 5000)
        self.declare_parameter("rrt_step_size", 0.40)
        self.declare_parameter("rrt_goal_sample_rate", 0.10)
        self.declare_parameter("rrt_goal_tolerance", 0.35)

        self.get_logger().info(
            "[BENCHMARK] Ready. Planners: NTFields + A* + RRT* + RRT. "
            "Set start with '2D Pose Estimate', goal with '2D Goal Pose'."
        )

    # ── Input callbacks — override parent to use benchmark flow ───────────────

    def _initialpose_cb(self, msg: PoseWithCovarianceStamped):
        p = msg.pose.pose.position
        self.start_xy = np.array([p.x, p.y], dtype=np.float32)
        self.get_logger().info(
            f"[BENCHMARK] Start: ({self.start_xy[0]:.3f}, {self.start_xy[1]:.3f})"
        )
        if self.goal_xy is not None and not self._benchmark_running:
            self._trigger_benchmark()

    def _goal_cb(self, msg: PoseStamped):
        p = msg.pose.position
        self.goal_xy = np.array([p.x, p.y], dtype=np.float32)
        self.get_logger().info(
            f"[BENCHMARK] Goal: ({self.goal_xy[0]:.3f}, {self.goal_xy[1]:.3f})"
        )
        if self.start_xy is None:
            self.get_logger().warn(
                "[BENCHMARK] Start not set. Use '2D Pose Estimate' first."
            )
            return
        if not self._benchmark_running:
            self._trigger_benchmark()

    # ── Main benchmark dispatch ────────────────────────────────────────────────

    def _trigger_benchmark(self):
        if self._benchmark_running:
            return
        if self.start_xy is None or self.goal_xy is None:
            return
        if self._map_blocked is None:
            self.get_logger().warn("[BENCHMARK] No map loaded — cannot plan.")
            return
        if self._route_graph.number_of_nodes() == 0:
            self.get_logger().warn("[BENCHMARK] Route graph empty — NTFields unavailable.")

        start = self.start_xy.copy()
        goal = self.goal_xy.copy()
        self._benchmark_running = True

        def _worker():
            try:
                results: dict = {}

                # 1. NTFields
                t0 = time.perf_counter()
                ntf_path = self._plan_ntfields(start, goal)
                results["ntfields"] = (time.perf_counter() - t0, ntf_path)

                # 2. A*
                t0 = time.perf_counter()
                astar_path = self._astar_world(start, goal)
                results["astar"] = (time.perf_counter() - t0, astar_path)

                # 3. RRT*
                t0 = time.perf_counter()
                rrtstar_path = self._plan_rrtstar(start, goal)
                results["rrtstar"] = (time.perf_counter() - t0, rrtstar_path)

                # 4. RRT
                t0 = time.perf_counter()
                rrt_path = self._plan_rrt(start, goal)
                results["rrt"] = (time.perf_counter() - t0, rrt_path)

                self._report_and_publish(results, start, goal)

            except Exception as exc:
                import traceback
                self.get_logger().error(
                    f"[BENCHMARK] Unhandled error: {exc}\n{traceback.format_exc()}"
                )
            finally:
                self._benchmark_running = False

        threading.Thread(target=_worker, daemon=True).start()

    # ── NTFields planner (reuses parent's internal methods) ───────────────────

    def _plan_ntfields(
        self, start: np.ndarray, goal: np.ndarray
    ) -> Optional[np.ndarray]:
        """Region-graph + neural field gradient descent path."""
        s_region = self._resolve_region_for_xy(start)
        g_region = self._resolve_region_for_xy(goal)
        if s_region is None or g_region is None:
            self.get_logger().warn(
                "[BENCHMARK] NTFields: could not resolve start/goal regions."
            )
            return None
        s_ver = s_region.version
        g_ver = g_region.version
        if s_ver not in self._route_graph or g_ver not in self._route_graph:
            self.get_logger().warn(
                f"[BENCHMARK] NTFields: R{s_ver} or R{g_ver} not in route graph."
            )
            return None

        # Route through doorway-connected edges only so every hop has a real
        # doorway — prevents the field from trying to navigate through walls
        # along long-distance edges that skip intermediate rooms.
        if s_ver == g_ver:
            region_route = [s_ver]
        else:
            doorway_graph = nx.Graph()
            for u, v, data in self._route_graph.edges(data=True):
                reg_u = self.regions.get(u)
                reg_v = self.regions.get(v)
                if reg_u is None or reg_v is None:
                    continue
                if self._all_doorways_for_regions(reg_u, reg_v, goal):
                    doorway_graph.add_edge(u, v, weight=data.get("weight", 1.0))
            try:
                region_route = nx.shortest_path(
                    doorway_graph, s_ver, g_ver, weight="weight"
                )
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                try:
                    region_route = nx.shortest_path(
                        self._route_graph, s_ver, g_ver, weight="weight"
                    )
                except nx.NetworkXNoPath:
                    self.get_logger().warn(
                        f"[BENCHMARK] NTFields: no graph path R{s_ver}→R{g_ver}."
                    )
                    return None

        self.get_logger().info(
            "[BENCHMARK] NTFields route: "
            + " → ".join(f"R{v}" for v in region_route)
        )

        seg_budget = max(80, self.max_steps // max(len(region_route), 1))
        segments: List[np.ndarray] = []
        cur = start.copy()

        # ── Single-region case ────────────────────────────────────────────────
        if s_ver == g_ver:
            direct, _, _ = self._field_segment(g_ver, cur, goal, self.max_steps)
            if direct is None:
                self.get_logger().warn("[BENCHMARK] NTFields: single-region field failed.")
                return None
            segments.append(direct)

        else:
            # ── Direct path if start already inside goal region ───────────────
            skip_final = False
            if g_region.in_bound(start[:2].astype(np.float32)):
                direct, _, _ = self._field_segment(g_ver, cur, goal, self.max_steps)
                if direct is not None:
                    segments.append(direct)
                    skip_final = True

            # ── Multi-hop via doorways ────────────────────────────────────────
            if not skip_final:
                hop_failed = False
                for hop_idx in range(len(region_route) - 1):
                    src_ver = region_route[hop_idx]
                    dst_ver = region_route[hop_idx + 1]
                    src_reg = self.regions.get(src_ver)
                    dst_reg = self.regions.get(dst_ver)
                    is_last = hop_idx == len(region_route) - 2

                    if src_reg is None or dst_reg is None:
                        self.get_logger().warn(
                            f"[BENCHMARK] NTFields: R{src_ver} or R{dst_ver} not loaded."
                        )
                        hop_failed = True
                        break

                    if is_last:
                        hop_target = goal
                    else:
                        nv = region_route[hop_idx + 2]
                        nr = self.regions.get(nv)
                        hop_target = nr.bound.mean(dim=0)[:2].numpy() if nr else goal

                    all_doors = self._all_doorways_for_regions(src_reg, dst_reg, hop_target, cur_xy=cur)
                    if not all_doors:
                        key = (min(src_ver, dst_ver), max(src_ver, dst_ver))
                        crossing = self._crossing_points.get(key)
                        if crossing is None and self._route_graph.has_edge(src_ver, dst_ver):
                            ep = self._edge_path_oriented(src_ver, dst_ver)
                            crossing = ep[len(ep) // 2].copy()
                        if crossing is None:
                            self.get_logger().warn(
                                f"[BENCHMARK] NTFields: no doorway/crossing "
                                f"R{src_ver}→R{dst_ver}."
                            )
                            hop_failed = True
                            break
                        all_doors = [np.asarray(crossing[:2], dtype=np.float32)]

                    # One doorway per hop: nearest to cur (all_doors already sorted)
                    door_xy = all_doors[0]
                    anchors = self._doorway_anchors_in_region(
                        src_ver, door_xy, cur  # push anchor back into src region
                    )
                    door_ok = False
                    if anchors:
                        for offset, anchor_xy in anchors:
                            fseg, _, _ = self._field_segment(
                                src_ver, cur, anchor_xy, seg_budget
                            )
                            if fseg is None:
                                continue
                            seg = np.asarray(fseg, dtype=np.float32)
                            if offset > 1e-6:
                                bridge_pts = straight_line_path(
                                    anchor_xy[:2], door_xy[:2], step=0.05,
                                    collision_fn=lambda xy: self._point_in_collision(xy),
                                )
                                if len(bridge_pts) < 2:
                                    continue
                                bridge = np.asarray(bridge_pts, dtype=np.float32)
                                if self._path_hits_wall(bridge):
                                    continue
                                seg = np.vstack([seg, bridge[1:]])
                            segments.append(seg)
                            cur = seg[-1][:2].astype(np.float32)
                            door_ok = True
                            break

                    if not door_ok:
                        self.get_logger().warn(
                            f"[BENCHMARK] NTFields: hop R{src_ver}→R{dst_ver} failed."
                        )
                        hop_failed = True
                        break

                if hop_failed:
                    return None

                final_field, reason = self._field_from_doorway(
                    g_ver, cur, goal, seg_budget
                )
                if final_field is None:
                    self.get_logger().warn(
                        f"[BENCHMARK] NTFields: final leg failed ({reason})."
                    )
                    return None
                segments.append(final_field)

        full = self._concat_paths(segments)
        if full is None:
            return None
        decimated = self._decimate(full, spacing=0.10)
        smoothed = self._smooth(decimated, window=7)
        return smoothed if not self._path_hits_wall(smoothed) else decimated

    # ── RRT* planner ──────────────────────────────────────────────────────────

    def _plan_rrtstar(
        self,
        start_xy: np.ndarray,
        goal_xy: np.ndarray,
        max_iter: Optional[int] = None,
        step_size: Optional[float] = None,
        neighborhood_radius: Optional[float] = None,
        goal_sample_rate: Optional[float] = None,
        goal_tolerance: Optional[float] = None,
    ) -> Optional[np.ndarray]:
        """RRT* on the 2D occupancy map. Returns smoothed path or None."""
        if self._map_blocked is None or self._map_origin is None or self._map_resolution is None:
            return None

        max_iter = max_iter or int(self.get_parameter("rrtstar_max_iter").value)
        step_size = step_size or float(self.get_parameter("rrtstar_step_size").value)
        neighborhood_radius = neighborhood_radius or float(
            self.get_parameter("rrtstar_neighborhood_radius").value
        )
        goal_sample_rate = goal_sample_rate or float(
            self.get_parameter("rrtstar_goal_sample_rate").value
        )
        goal_tolerance = goal_tolerance or float(
            self.get_parameter("rrtstar_goal_tolerance").value
        )

        start = self._nearest_free_xy(start_xy)
        goal = self._nearest_free_xy(goal_xy)
        if start is None or goal is None:
            self.get_logger().warn("[BENCHMARK] RRT*: start/goal in collision.")
            return None

        h, w = self._map_blocked.shape
        ox, oy = self._map_origin
        res = self._map_resolution
        x_min, x_max = float(ox), float(ox + w * res)
        y_min, y_max = float(oy), float(oy + h * res)

        # Tree storage
        nodes_list: List[np.ndarray] = [start.copy()]
        costs: List[float] = [0.0]
        parents: List[int] = [-1]

        best_path: Optional[List[np.ndarray]] = None
        best_cost = float("inf")

        rng = np.random.default_rng(seed=42)

        def _sample() -> np.ndarray:
            if rng.random() < goal_sample_rate:
                return goal.copy()
            for _ in range(100):
                x = rng.uniform(x_min, x_max)
                y = rng.uniform(y_min, y_max)
                xy = np.array([x, y], dtype=np.float32)
                if not self._point_in_collision(xy):
                    return xy
            return goal.copy()

        def _steer(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
            d = float(np.linalg.norm(dst - src))
            if d < step_size:
                return dst.copy()
            return (src + (dst - src) / d * step_size).astype(np.float32)

        def _extract_path(node_idx: int) -> List[np.ndarray]:
            path = [goal.copy(), nodes_list[node_idx].copy()]
            idx = parents[node_idx]
            while idx >= 0:
                path.append(nodes_list[idx].copy())
                idx = parents[idx]
            path.reverse()
            return path

        for _ in range(max_iter):
            rnd = _sample()

            # Nearest node
            tree_arr = np.array(nodes_list, dtype=np.float32)
            dists = np.linalg.norm(tree_arr - rnd, axis=1)
            nearest_idx = int(np.argmin(dists))
            nearest = nodes_list[nearest_idx]

            new_xy = _steer(nearest, rnd)

            if self._point_in_collision(new_xy):
                continue
            if self._segment_hits_wall(nearest, new_xy):
                continue

            # Neighbors within radius
            dists_to_new = np.linalg.norm(tree_arr - new_xy, axis=1)
            neighbor_idxs = [int(i) for i in np.where(dists_to_new <= neighborhood_radius)[0]]

            # Choose minimum-cost collision-free parent
            min_cost = costs[nearest_idx] + float(np.linalg.norm(new_xy - nearest))
            min_parent = nearest_idx
            for ni in neighbor_idxs:
                c = costs[ni] + float(dists_to_new[ni])
                if c < min_cost and not self._segment_hits_wall(nodes_list[ni], new_xy):
                    min_cost = c
                    min_parent = ni

            nodes_list.append(new_xy.copy())
            costs.append(min_cost)
            parents.append(min_parent)
            new_idx = len(nodes_list) - 1

            # Rewire
            for ni in neighbor_idxs:
                rewire_cost = min_cost + float(dists_to_new[ni])
                if rewire_cost < costs[ni] and not self._segment_hits_wall(new_xy, nodes_list[ni]):
                    parents[ni] = new_idx
                    costs[ni] = rewire_cost

            # Goal check
            dist_g = float(np.linalg.norm(new_xy - goal))
            if dist_g <= goal_tolerance:
                total = min_cost + dist_g
                if total < best_cost and not self._segment_hits_wall(new_xy, goal):
                    best_cost = total
                    best_path = _extract_path(new_idx)

        if best_path is None:
            self.get_logger().warn(
                f"[BENCHMARK] RRT*: no path found in {max_iter} iterations."
            )
            return None

        path_arr = np.array(best_path, dtype=np.float32)
        decimated = self._decimate(path_arr, spacing=0.10)
        smoothed = self._smooth(decimated, window=5)
        return smoothed if not self._path_hits_wall(smoothed) else decimated

    # ── RRT planner ───────────────────────────────────────────────────────────

    def _plan_rrt(
        self,
        start_xy: np.ndarray,
        goal_xy: np.ndarray,
        max_iter: Optional[int] = None,
        step_size: Optional[float] = None,
        goal_sample_rate: Optional[float] = None,
        goal_tolerance: Optional[float] = None,
    ) -> Optional[np.ndarray]:
        """Basic RRT — returns the first collision-free path found (not optimal)."""
        if self._map_blocked is None or self._map_origin is None or self._map_resolution is None:
            return None

        max_iter = max_iter or int(self.get_parameter("rrt_max_iter").value)
        step_size = step_size or float(self.get_parameter("rrt_step_size").value)
        goal_sample_rate = goal_sample_rate or float(
            self.get_parameter("rrt_goal_sample_rate").value
        )
        goal_tolerance = goal_tolerance or float(
            self.get_parameter("rrt_goal_tolerance").value
        )

        start = self._nearest_free_xy(start_xy)
        goal = self._nearest_free_xy(goal_xy)
        if start is None or goal is None:
            self.get_logger().warn("[BENCHMARK] RRT: start/goal in collision.")
            return None

        h, w = self._map_blocked.shape
        ox, oy = self._map_origin
        res = self._map_resolution
        x_min, x_max = float(ox), float(ox + w * res)
        y_min, y_max = float(oy), float(oy + h * res)

        nodes_list: List[np.ndarray] = [start.copy()]
        parents: List[int] = [-1]

        rng = np.random.default_rng(seed=42)

        def _sample() -> np.ndarray:
            if rng.random() < goal_sample_rate:
                return goal.copy()
            for _ in range(100):
                xy = np.array(
                    [rng.uniform(x_min, x_max), rng.uniform(y_min, y_max)],
                    dtype=np.float32,
                )
                if not self._point_in_collision(xy):
                    return xy
            return goal.copy()

        def _steer(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
            d = float(np.linalg.norm(dst - src))
            if d < step_size:
                return dst.copy()
            return (src + (dst - src) / d * step_size).astype(np.float32)

        for _ in range(max_iter):
            rnd = _sample()

            tree_arr = np.array(nodes_list, dtype=np.float32)
            nearest_idx = int(np.argmin(np.linalg.norm(tree_arr - rnd, axis=1)))
            nearest = nodes_list[nearest_idx]

            new_xy = _steer(nearest, rnd)

            if self._point_in_collision(new_xy):
                continue
            if self._segment_hits_wall(nearest, new_xy):
                continue

            nodes_list.append(new_xy.copy())
            parents.append(nearest_idx)
            new_idx = len(nodes_list) - 1

            if float(np.linalg.norm(new_xy - goal)) <= goal_tolerance:
                if not self._segment_hits_wall(new_xy, goal):
                    path = [goal.copy(), new_xy.copy()]
                    idx = parents[new_idx]
                    while idx >= 0:
                        path.append(nodes_list[idx].copy())
                        idx = parents[idx]
                    path.reverse()
                    path_arr = np.array(path, dtype=np.float32)
                    decimated = self._decimate(path_arr, spacing=0.10)
                    smoothed = self._smooth(decimated, window=5)
                    return smoothed if not self._path_hits_wall(smoothed) else decimated

        self.get_logger().warn(
            f"[BENCHMARK] RRT: no path found in {max_iter} iterations."
        )
        return None

    # ── Result reporting and publishing ───────────────────────────────────────

    def _report_and_publish(
        self, results: dict, start: np.ndarray, goal: np.ndarray
    ):
        ntf_elapsed,     ntf_path     = results.get("ntfields", (None, None))
        astar_elapsed,   astar_path   = results.get("astar",    (None, None))
        rrtstar_elapsed, rrtstar_path = results.get("rrtstar",  (None, None))
        rrt_elapsed,     rrt_path     = results.get("rrt",      (None, None))

        def _fmt(elapsed: Optional[float], path: Optional[np.ndarray]) -> str:
            if path is None or elapsed is None:
                return "FAILED"
            return f"time={elapsed:.3f}s  len={self._path_length(path):.2f}m  pts={len(path)}"

        self.get_logger().info(
            "\n[BENCHMARK] ══════════════════════════════════════════════════\n"
            f"  start=({start[0]:.3f}, {start[1]:.3f})  "
            f"goal=({goal[0]:.3f}, {goal[1]:.3f})\n"
            f"  NTFields : {_fmt(ntf_elapsed,   ntf_path)}\n"
            f"  A*       : {_fmt(astar_elapsed, astar_path)}\n"
            f"  RRT*     : {_fmt(rrtstar_elapsed, rrtstar_path)}\n"
            f"  RRT      : {_fmt(rrt_elapsed,   rrt_path)}\n"
            "[BENCHMARK] ══════════════════════════════════════════════════"
        )

        self.path_xy = ntf_path

        if ntf_path is not None:
            self._publish_path_to_pub(ntf_path,     self.path_pub,         z=0.22)
        if astar_path is not None:
            self._publish_path_to_pub(astar_path,   self.astar_path_pub,   z=0.18)
        if rrtstar_path is not None:
            self._publish_path_to_pub(rrtstar_path, self.rrtstar_path_pub, z=0.14)
        if rrt_path is not None:
            self._publish_path_to_pub(rrt_path,     self.rrt_path_pub,     z=0.10)


def main(args=None):
    rclpy.init(args=args)
    node = PathPlannerBenchmark()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
