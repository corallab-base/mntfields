# exploration_graph.py
import numpy as np
import networkx as nx
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class RegionFrontierRecord:
    version: int
    center: np.ndarray
    saved_frontiers: List[np.ndarray]
    visited_goals: List[np.ndarray]

    def has_frontiers(self) -> bool:
        return len(self.saved_frontiers) > 0

    def pop_best_frontier(self, robot_xy: np.ndarray) -> Optional[np.ndarray]:
        if not self.saved_frontiers:
            return None
        pts = np.array(self.saved_frontiers)
        dists = np.linalg.norm(pts - robot_xy[None, :], axis=1)
        idx = int(np.argmin(dists))
        return self.saved_frontiers.pop(idx)

    def add_frontier(self, pt: np.ndarray, blacklist_radius: float = 0.3):
        if len(self.saved_frontiers) > 0:
            pts = np.array(self.saved_frontiers)
            if np.linalg.norm(pts - pt[None, :], axis=1).min() < blacklist_radius:
                return
        self.saved_frontiers.append(pt.copy())

    def mark_visited(self, pt: np.ndarray):
        self.visited_goals.append(pt.copy())
        self.saved_frontiers = [
            f for f in self.saved_frontiers
            if np.linalg.norm(f - pt) > 0.5
        ]

    def is_exhausted(self) -> bool:
        """True when the region has been explored (no active frontiers remain)."""
        return not self.has_frontiers()


class ExplorationGraph:
    """
    Per-region frontier banks with graph connectivity via shared
    visited/free cells (never door midpoints or boundary midpoints).

    Frontier ownership rules
    ------------------------
    - All frontiers in an overlap zone belong to the CURRENT (active) region.
    - If a region's frontier bank is exhausted, it never receives overlap frontiers.
    - Frontiers cleared by LiDAR rays or robot visits are evicted immediately.
    """

    BLACKLIST_RADIUS = 0.5  # metres

    def __init__(self):
        self.records: Dict[int, RegionFrontierRecord] = {}
        # nodes = region versions
        # edge attrs: {'waypoints': List[np.ndarray], 'edge_type': str}
        self.graph = nx.Graph()

    # ------------------------------------------------------------------ #
    #  Region registration                                                 #
    # ------------------------------------------------------------------ #

    def ensure_region(self, version: int, center: np.ndarray):
        if version not in self.records:
            self.records[version] = RegionFrontierRecord(
                version=version,
                center=center.copy(),
                saved_frontiers=[],
                visited_goals=[],
            )
        else:
            self.records[version].center = center.copy()
        if version not in self.graph:
            self.graph.add_node(version)

    # ------------------------------------------------------------------ #
    #  Graph edges — only via shared visited/free cells                   #
    # ------------------------------------------------------------------ #

    def add_free_cell_edge(
        self,
        v1: int,
        v2: int,
        shared_waypoints: List[np.ndarray],
        weight: float = 1.0,
    ):
        """
        Connect two regions using world-xy points that both regions have
        visited or that are confirmed free/traversable on the gridmap.

        Never call this with boundary midpoints or door positions — only
        pass cells where self.gridmap.visited[i,j] > 0 AND the cell is
        within both regions' discovered areas.

        Store exactly one canonical traversable transition point per edge.
        Multiple waypoints per pair make the graph noisy and produce unstable
        cross-region routing.

        weight: Euclidean distance between region centers (or other travel cost).
        Used by best_remote_frontier for weighted-shortest-path routing.
        """
        if not shared_waypoints:
            return

        canonical = np.asarray(shared_waypoints[0], dtype=np.float32).copy()
        if self.graph.has_edge(v1, v2):
            edge = self.graph.edges[v1, v2]
            edge['cp'] = canonical
            edge['waypoints'] = [canonical]
            edge['edge_type'] = 'free_cell'
            edge['weight'] = weight
        else:
            self.graph.add_edge(
                v1, v2,
                cp=canonical,
                waypoints=[canonical],
                edge_type='free_cell',
                weight=weight,
            )

    # Keep add_door_edge for legacy callers but internally store as free_cell
    def add_door_edge(self, v1: int, v2: int, door_xy: np.ndarray, weight: float = 1.0):
        """Store door position as a waypoint only if the door cell is
        known-free (caller's responsibility to verify).
        weight: travel cost between the two regions (e.g. centre-to-centre distance)."""
        self.add_free_cell_edge(v1, v2, [door_xy], weight=weight)

    # add_spatial_edge is replaced entirely — boundary midpoints are never used
    # Kept as a no-op so old call sites don't crash during migration
    def add_spatial_edge(self, v1: int, v2: int, midpoint_xy: np.ndarray):
        """DEPRECATED. Boundary midpoints may be inside walls.
        Use add_free_cell_edge with verified free cells instead."""
        pass  # intentional no-op

    def add_cp_edge(self, v1: int, v2: int, cp_xy: np.ndarray, weight: float = 1.0):
        """
        Connect two regions via exactly one canonical connection point (a
        doorway or corridor crossing).  If an edge already exists, the CP
        position is updated only if it's a strictly new point.
        """
        if self.graph.has_edge(v1, v2):
            existing_cp = self.graph.edges[v1, v2].get('cp')
            if existing_cp is not None and np.linalg.norm(existing_cp - cp_xy) < 0.3:
                return   # same doorway, no update needed
        self.graph.add_edge(v1, v2, cp=cp_xy.copy(), edge_type='doorway',
                            waypoints=[cp_xy.copy()], weight=weight)

    # ------------------------------------------------------------------ #
    #  Frontier bank management                                            #
    # ------------------------------------------------------------------ #

    def snapshot_frontiers(
        self,
        current_version: int,
        frontier_pts: np.ndarray,
        all_versions_at_pts: Optional[Dict[int, List[np.ndarray]]] = None,
    ):
        """
        Assign frontier points to regions according to ownership rules:

        1. If a frontier point lies in an overlap zone (multiple regions
           cover it), it belongs exclusively to `current_version`.
        2. A region that is exhausted (no active frontiers anywhere in its
           own un-overlapped area) does NOT receive overlap-zone frontiers.
        3. Non-overlapping frontier points are assigned to `current_version`.

        Args:
            current_version: the region the robot is currently inside.
            frontier_pts: Nx2 world-xy array of live frontier cells.
            all_versions_at_pts: optional mapping {version: [pts exclusive
                to that region]}. When provided, exclusive points go to
                their owning region; overlap points always go to current.
        """
        if current_version not in self.records:
            return

        if all_versions_at_pts is None:
            # Fallback: all frontiers go to current region
            for pt in frontier_pts:
                self.records[current_version].add_frontier(
                    pt, self.BLACKLIST_RADIUS
                )
            return

        # Give exclusive frontiers to their respective (non-exhausted) regions
        for version, pts in all_versions_at_pts.items():
            if version == current_version:
                continue
            rec = self.records.get(version)
            if rec is None:
                continue
            # Only assign to this remote region if it is NOT exhausted
            if not rec.is_exhausted():
                for pt in pts:
                    rec.add_frontier(pt, self.BLACKLIST_RADIUS)

        # All overlap frontiers + current region's own frontiers → current region
        for pt in frontier_pts:
            self.records[current_version].add_frontier(
                pt, self.BLACKLIST_RADIUS
            )

    def reconcile_all_banks(
        self,
        is_frontier_fn,
        evict_radius: float = 0.6,
    ):
        """
        Evict saved frontier points that are no longer live.

        A point is removed if:
        - No live frontier cell exists within `evict_radius` of it, OR
        - It has been cleared by LiDAR rays (covered by is_frontier_fn).

        Called every update tick.
        """
        for rec in self.records.values():
            rec.saved_frontiers = [
                pt for pt in rec.saved_frontiers
                if is_frontier_fn(pt, evict_radius)
            ]

    def mark_goal_reached(self, version: int, goal_xy: np.ndarray):
        if version in self.records:
            self.records[version].mark_visited(goal_xy)

    # ------------------------------------------------------------------ #
    #  Querying                                                            #
    # ------------------------------------------------------------------ #

    def current_region_has_frontiers(self, version: int) -> bool:
        return version in self.records and self.records[version].has_frontiers()

    def best_remote_frontier(
        self,
        current_version: int,
        robot_xy: np.ndarray,
    ) -> Optional[Tuple[int, np.ndarray, List[np.ndarray]]]:

        candidates = [
            v for v, rec in self.records.items()
            if v != current_version and rec.has_frontiers()
        ]
        if not candidates:
            return None

        # Score candidates by (graph hops, euclidean distance).
        # Only graph-connected regions are eligible. Without a graph path the
        # caller would fall back to direct geometric travel across regions.
        best            = None
        best_path       = None
        best_score      = (float('inf'),)

        for v in candidates:
            try:
                path_versions = nx.shortest_path(self.graph, current_version, v, weight='weight')
                path_cost = float(nx.shortest_path_length(self.graph, current_version, v, weight='weight'))
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue

            score = (path_cost,)
            if score < best_score:
                best_score = score
                best       = v
                best_path  = path_versions

        if best is None:
            print("[EXPLORATION GRAPH] No graph-connected remote region has frontiers")
            return None

        # Build waypoint chain from the graph path edges.
        # Each edge contributes one canonical transition point.
        waypoints: List[np.ndarray] = []
        for i in range(len(best_path) - 1):
            u, v = best_path[i], best_path[i + 1]
            if not self.graph.has_edge(u, v):
                return None
            edge_data = self.graph.edges[u, v]
            cp = edge_data.get('cp')
            if cp is not None:
                waypoints.append(np.asarray(cp, dtype=np.float32))
                continue
            edge_wps = edge_data.get('waypoints', [])
            if not edge_wps:
                return None
            waypoints.append(np.asarray(edge_wps[0], dtype=np.float32))

        target_frontier = self.records[best].pop_best_frontier(robot_xy)
        if target_frontier is None:
            return None
        print(f"[EXPLORATION GRAPH] Route: {best_path}, waypoints={len(waypoints)}")
        return best, target_frontier, waypoints
