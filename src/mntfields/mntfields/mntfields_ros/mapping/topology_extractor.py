"""
topology_extractor.py — Frontiers, rooms, and doors from SLAM occupancy grid.

Pure Python/numpy — no ROS.  Called from slam_map_cb in GridmapNode.

SLAM grid conventions (ROS OccupancyGrid):
  0   = free
  100 = occupied   (any value > 50 treated as occupied)
  -1  = unknown     (stored as uint8 → 255 in raw data)

Room segmentation is performed online from the free-space topology:
  - clearance field from the distance transform of observed free space
  - skeleton graph of the free space
  - portal candidates as clearance bottlenecks along skeleton edges
  - room masks from connected components after cutting persistent bottlenecks
"""

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import skimage.morphology
import skimage.segmentation
from sklearn.cluster import DBSCAN


# ── data structures ─────────────────────────────────────────────────────────

@dataclass
class Door:
    id: int
    pos_xy: np.ndarray          # [x, y] world coords
    room_ids: set               # set of room.id values this door connects


@dataclass
class Room:
    id: int
    mask: np.ndarray            # bool [H, W] in grid frame
    centroid_xy: np.ndarray     # [x, y] world coords
    confirmed: bool = False
    doors: list = field(default_factory=list)


# ── main class ───────────────────────────────────────────────────────────────

class TopologyExtractor:
    """
    Maintains frontiers, confirmed rooms, and doors from a SLAM OccupancyGrid.

    Call update(slam_array, cell_size_m, origin_xy) after each /map message.
    """

    # Frontier cleanup / clustering
    _FRONTIER_OCC_CLEAR_DILATE = 2   # cells; keeps frontiers off mapped walls
    _DBSCAN_EPS                = 4   # cells
    _DBSCAN_MIN_SAMPLES        = 1

    # Reached-goal blacklist radius
    _BLACKLIST_DIST = 0.4   # metres

    # Topological room segmentation / confirmation knobs
    _ROOM_SEG_REF_CELL_M = 0.10
    _ROOM_SEG_KERNEL_REF = 7   # compatibility knob; no longer used directly
    _ROOM_SEG_ITER = 2         # compatibility knob; no longer used directly
    _ROOM_MIN_CC_CELLS = 10
    _ROOM_SMALL_AREA_REF_CELLS = 200
    _ROOM_SMALL_BOUNDARY_FRONTIERS = 15
    _ROOM_LARGE_BOUNDARY_FRONTIERS = 80
    _ROOM_LARGE_INSIDE_FRONTIERS = 50
    _ROOM_OUTER_OPENING_REF_CELLS = 20

    _PORTAL_MIN_CHAMBER_CLEARANCE_M = 0.45
    _PORTAL_MIN_PEAK_DIFF_M = 0.10
    _PORTAL_REL_CLEARANCE_RATIO = 0.82
    _PORTAL_MAX_LOW_SPAN_M = 1.2
    _PORTAL_MIN_LOW_SPAN_M = 0.05
    _PORTAL_CUT_CLEARANCE_PAD_M = 0.15
    _PORTAL_CUT_THICKNESS_M = 0.12
    _PORTAL_EDGE_MARGIN_M = 0.35
    _PORTAL_SMOOTH_WINDOW_M = 0.25
    _PORTAL_PROBE_RADIUS_M = 1.2
    _PORTAL_MIN_LOBE_AREA_M2 = 0.80
    _PORTAL_SPLIT_PROBE_RADIUS_M = 1.2
    _PORTAL_SPLIT_CENTER_BLOCK_M = 0.20
    _PORTAL_SPLIT_MAX_WIDTH_M = 1.05
    _PORTAL_SPLIT_DEDUP_RADIUS_M = 0.35
    _PORTAL_SPLIT_MIN_BALANCE = 0.75
    _ROOM_FRAGMENT_MAX_AREA_M2 = 0.75
    _ROOM_FRAGMENT_MERGE_RADIUS_M = 0.20

    # Door detection width limits (metres) — matching original lidar_gridmap
    _DOOR_WIDTH_MIN_M  = 0.60
    _DOOR_WIDTH_MAX_M  = 1.80
    _DOOR_FLANK_MIN_M  = 0.50   # compatibility knob; bottleneck support replaces flank checks
    _DOOR_NMS_RADIUS_M = 1.50

    # Door track persistence
    _DOOR_PERSIST_RADIUS_M  = 0.35
    _DOOR_PERSIST_HITS      = 2
    _DOOR_TRACK_MAX_MISSES  = 4

    def __init__(self):
        self.frontier_centroids: np.ndarray = np.zeros((0, 2), dtype=np.float32)
        self.confirmed_rooms: dict[int, Room] = {}
        self.unconfirmed_rooms: dict[int, Room] = {}
        self.doors: list[Door] = []

        self._confirmed_room_map: Optional[np.ndarray] = None   # int32 [H, W]
        self._room_map: Optional[np.ndarray] = None             # int32 all rooms
        self._outer_label: int = 0
        self._frontier_cells: Optional[np.ndarray] = None       # bool [H, W]

        self._n_rooms: int = 0
        self._n_doors: int = 0
        self._door_tracks: dict[int, dict] = {}

        self._cell_size: float = 0.05   # metres per grid cell
        self._origin: np.ndarray = np.zeros(2, dtype=np.float32)
        self._shape: tuple = (0, 0)

        self._blacklist: list[np.ndarray] = []   # list of Nx2 reached goal positions
        self._door_debug: list[dict] = []
        self._door_debug_rooms: list[dict] = []
        self._distance_field_m: Optional[np.ndarray] = None
        self._skeleton: Optional[np.ndarray] = None
        self._portal_candidates: list[dict] = []
        self._portal_cut_mask: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        slam_array: np.ndarray,
        cell_size_m: float,
        origin_xy: np.ndarray,
    ) -> None:
        """
        Update topology from the latest SLAM occupancy grid.

        slam_array : np.ndarray [H, W] int8 or uint8
            0=free, 100=occupied, -1 or 255=unknown
        cell_size_m : resolution in metres per cell
        origin_xy   : [x, y] world coords of grid cell (0, 0)
        """
        self._cell_size = float(cell_size_m)
        self._origin    = np.array(origin_xy[:2], dtype=np.float32)
        H, W            = slam_array.shape
        self._shape     = (H, W)

        # OccupancyGrid data is int8: 0=free, 100=occupied, -1=unknown
        arr = slam_array.astype(np.int16)

        free     = (arr == 0)
        occupied = (arr > 50)
        unknown  = (arr < 0)

        if self._confirmed_room_map is None or self._confirmed_room_map.shape != (H, W):
            self._confirmed_room_map = np.zeros((H, W), dtype=np.int32)
            self._room_map = np.zeros((H, W), dtype=np.int32)
            self.confirmed_rooms.clear()
            self.unconfirmed_rooms.clear()
            self.doors = []
            self._door_tracks.clear()
            self._door_debug = []
            self._door_debug_rooms = []
            self._n_rooms = 0

        self._door_debug = []
        self._door_debug_rooms = []
        self._compute_frontiers(free, unknown, occupied)
        self._compute_topology(free, occupied)
        self._compute_rooms(free, occupied)
        self._compute_doors(free, occupied)

    def get_frontier_centroids(self) -> np.ndarray:
        """Return Nx2 world coords of frontier clusters not in the blacklist."""
        if len(self.frontier_centroids) == 0:
            return self.frontier_centroids
        if not self._blacklist:
            return self.frontier_centroids
        mask = np.ones(len(self.frontier_centroids), dtype=bool)
        for bl in self._blacklist:
            dists = np.linalg.norm(self.frontier_centroids - bl, axis=1)
            mask &= (dists > self._BLACKLIST_DIST)
        return self.frontier_centroids[mask]

    def get_room_at_xy(self, xy: np.ndarray) -> Optional[Room]:
        """Return confirmed room containing world point xy, or None."""
        if self._confirmed_room_map is None:
            return None
        i, j = self.world_to_grid(float(xy[0]), float(xy[1]))
        H, W = self._shape
        if not (0 <= i < H and 0 <= j < W):
            return None
        room_id = int(self._confirmed_room_map[i, j])
        return self.confirmed_rooms.get(room_id)

    def mark_reached(self, xy: np.ndarray) -> None:
        """Blacklist a reached goal position so it is not re-selected."""
        self._blacklist.append(np.array(xy[:2], dtype=np.float32))

    def grid_to_world(self, i: int, j: int) -> np.ndarray:
        """Convert grid indices (row, col) → world [x, y]."""
        x = self._origin[0] + j * self._cell_size
        y = self._origin[1] + i * self._cell_size
        return np.array([x, y], dtype=np.float32)

    def world_to_grid(self, x: float, y: float) -> tuple[int, int]:
        """Convert world [x, y] → grid indices (row, col)."""
        j = int((x - self._origin[0]) / self._cell_size)
        i = int((y - self._origin[1]) / self._cell_size)
        return i, j

    # ------------------------------------------------------------------
    # Frontier extraction
    # ------------------------------------------------------------------

    def _compute_frontiers(
        self, free: np.ndarray, unknown: np.ndarray, occupied: Optional[np.ndarray] = None
    ) -> None:
        """
        Match the simpler raw-SLAM frontier logic from lidar_gridmap:
        strict 4-neighbor free->unknown frontier cells, then clear them away
        from occupied walls using a safety-radius dilation.
        """
        if occupied is None:
            occupied = np.zeros_like(free, dtype=bool)

        frontier = np.zeros_like(free, dtype=bool)
        frontier[:-1, :] |= free[:-1, :] & unknown[1:, :]
        frontier[1:, :] |= free[1:, :] & unknown[:-1, :]
        frontier[:, :-1] |= free[:, :-1] & unknown[:, 1:]
        frontier[:, 1:] |= free[:, 1:] & unknown[:, :-1]

        robot_safety_m = 0.20   # reduced from 0.40 so corridor-end frontiers survive
        slam_r = max(2, int(np.ceil(robot_safety_m / max(1e-6, self._cell_size))))
        occ_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * slam_r + 1, 2 * slam_r + 1)
        )
        occ_dilated = cv2.dilate(
            occupied.astype(np.uint8), occ_kernel, iterations=1
        ).astype(bool)
        frontier &= ~occ_dilated

        self._frontier_cells = frontier

        pts = np.stack(np.where(frontier), axis=1).astype(np.float32)
        if len(pts) == 0:
            self.frontier_centroids = np.zeros((0, 2), dtype=np.float32)
            return

        if len(pts) < self._DBSCAN_MIN_SAMPLES:
            self.frontier_centroids = np.asarray(
                [self.grid_to_world(int(i), int(j)) for i, j in pts],
                dtype=np.float32,
            )
            return

        db = DBSCAN(
            eps=self._DBSCAN_EPS,
            min_samples=self._DBSCAN_MIN_SAMPLES,
            n_jobs=1,
        ).fit(pts)
        labels = db.labels_

        centroids = []
        for lbl in np.unique(labels):
            if lbl < 0:
                continue
            cluster_pts = pts[labels == lbl]
            if len(cluster_pts) == 0:
                continue
            ij_mean = cluster_pts.mean(axis=0)
            d = np.linalg.norm(cluster_pts - ij_mean[None, :], axis=1)
            ci, cj = cluster_pts[int(np.argmin(d))].astype(np.int32)
            centroids.append(self.grid_to_world(int(ci), int(cj)))

        self.frontier_centroids = (
            np.array(centroids, dtype=np.float32)
            if centroids
            else np.zeros((0, 2), dtype=np.float32)
        )

    # ------------------------------------------------------------------
    # Topological preprocessing
    # ------------------------------------------------------------------

    def _compute_topology(self, free: np.ndarray, occupied: np.ndarray) -> None:
        free_u8 = free.astype(np.uint8)
        self._distance_field_m = (
            cv2.distanceTransform(free_u8, cv2.DIST_L2, 5).astype(np.float32) * self._cell_size
        )
        self._skeleton = skimage.morphology.skeletonize(free)
        skeleton_candidates = self._extract_portal_candidates(free, occupied)
        split_candidates = self._extract_split_space_candidates(free, occupied)
        self._portal_candidates = self._nms_portal_candidates(skeleton_candidates + split_candidates)

    # ------------------------------------------------------------------
    # Room segmentation from portal cuts
    # ------------------------------------------------------------------

    def _compute_rooms(self, free: np.ndarray, occupied: np.ndarray) -> None:
        """
        Segment observed free space into room hypotheses by cutting persistent
        topological bottlenecks and then taking free-space connected components.
        """
        cut_mask = self._build_portal_cut_mask(free)
        self._portal_cut_mask = cut_mask

        segmented_free = free & ~cut_mask
        _, labels = cv2.connectedComponents(segmented_free.astype(np.uint8), connectivity=8)
        labels = labels.astype(np.int32)
        labels = self._merge_small_components(labels)
        self._outer_label = 0
        self._room_map = np.zeros_like(labels, dtype=np.int32)
        self._confirmed_room_map = np.zeros_like(labels, dtype=np.int32)

        prev_confirmed = list(self.confirmed_rooms.values())
        remaining_prev = {int(room.id): room for room in prev_confirmed}
        new_confirmed: dict[int, Room] = {}
        self.unconfirmed_rooms.clear()

        for lbl in np.unique(labels):
            if lbl == 0:
                continue

            mask = (labels == lbl)
            mask = self._largest_cc(mask)
            if mask.sum() < int(self._ROOM_MIN_CC_CELLS):
                continue

            room_id = self._match_room_id(mask, remaining_prev)
            centroid_xy = self._mask_centroid(mask)
            room = Room(id=int(room_id), mask=mask, centroid_xy=centroid_xy)
            self._room_map[mask] = int(room_id)

            if self._room_is_confirmed(room, occupied):
                room.confirmed = True
                new_confirmed[int(room_id)] = room
                self._confirmed_room_map[mask] = int(room_id)
            else:
                self.unconfirmed_rooms[int(room_id)] = room

        self.confirmed_rooms = new_confirmed

    def _merge_small_components(self, labels: np.ndarray) -> np.ndarray:
        cell_area = max(self._cell_size * self._cell_size, 1e-6)
        max_area_cells = int(np.ceil(self._ROOM_FRAGMENT_MAX_AREA_M2 / cell_area))
        if max_area_cells <= 0:
            return labels
        dilate_r = max(1, int(np.ceil(self._ROOM_FRAGMENT_MERGE_RADIUS_M / max(self._cell_size, 1e-6))))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_r + 1, 2 * dilate_r + 1))

        labels = labels.copy()
        changed = True
        while changed:
            changed = False
            ids = [int(v) for v in np.unique(labels) if int(v) != 0]
            ids.sort(key=lambda rid: int((labels == rid).sum()))
            for rid in ids:
                mask = labels == rid
                area = int(mask.sum())
                if area == 0 or area > max_area_cells:
                    continue
                dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
                neighbors = [int(v) for v in np.unique(labels[dilated]) if int(v) not in (0, rid)]
                if not neighbors:
                    continue
                best_id = None
                best_score = -1
                for nid in neighbors:
                    contact = int(np.logical_and(dilated, labels == nid).sum())
                    n_area = int((labels == nid).sum())
                    score = 1000 * contact + n_area
                    if score > best_score:
                        best_score = score
                        best_id = nid
                if best_id is None:
                    continue
                labels[mask] = int(best_id)
                changed = True
                break
        return labels

    def _room_is_confirmed(self, room: Room, occupied: np.ndarray) -> bool:
        """
        A room is confirmed when:
          - Its boundary has few frontier cells.
          - Its interior has few frontier cells.
          - It does not have a large contiguous opening into unexplored space.
        """
        if self._frontier_cells is None:
            return False

        cell = max(1e-6, self._cell_size)
        ref_cell = float(self._ROOM_SEG_REF_CELL_M)
        scale_lin  = ref_cell / cell          # perimeter counts scale linearly
        scale_area = scale_lin ** 2           # area counts scale quadratically

        # Original size split: 200 cells at 0.10 m/cell = ~2 m²
        small_room_thresh = int(float(self._ROOM_SMALL_AREA_REF_CELLS) * scale_area)

        bounds = skimage.segmentation.find_boundaries(room.mask, mode='outer')
        bounds[occupied] = False

        n_bdry_frontiers   = int(self._frontier_cells[bounds].sum())
        n_inside_frontiers = int(self._frontier_cells[room.mask].sum())

        if room.mask.sum() < small_room_thresh:
            # Small room: only check boundary frontier count
            if n_bdry_frontiers > int(float(self._ROOM_SMALL_BOUNDARY_FRONTIERS) * scale_lin):
                return False
            return True

        # Large room: stricter checks
        if n_bdry_frontiers > int(float(self._ROOM_LARGE_BOUNDARY_FRONTIERS) * scale_lin):
            return False
        if n_inside_frontiers > int(float(self._ROOM_LARGE_INSIDE_FRONTIERS) * scale_area):
            return False

        open_boundary = (bounds & self._frontier_cells).astype(np.uint8)
        _, open_labels = cv2.connectedComponents(open_boundary)
        for ol in np.unique(open_labels):
            if ol == 0:
                continue
            if (open_labels == ol).sum() >= int(float(self._ROOM_OUTER_OPENING_REF_CELLS) * scale_lin):
                return False

        return True

    # ------------------------------------------------------------------
    # Door detection  (port of lidar_gridmap.add_doors)
    # ------------------------------------------------------------------

    def _compute_doors(self, free: np.ndarray, occupied: np.ndarray) -> None:
        """
        Confirm doorways from the current frame's portal candidates and the
        multi-frame persistence tracker.
        """
        for room in self.confirmed_rooms.values():
            room.doors = []
        self.doors = []

        candidates: list[tuple[np.ndarray, set[int]]] = []
        for cand in self._portal_candidates:
            if cand.get('status') != 'accepted':
                continue
            pos_xy = cand['pos_xy'].astype(np.float32)
            room_ids = self._room_ids_near_point(pos_xy)
            if len(room_ids) < 2:
                continue
            candidates.append((pos_xy, room_ids))

        self._update_door_tracks(candidates)
        self._realize_doors_from_tracks()

    # ------------------------------------------------------------------
    # Topological portal extraction
    # ------------------------------------------------------------------

    def _extract_portal_candidates(
        self,
        free: np.ndarray,
        occupied: np.ndarray,
    ) -> list[dict]:
        skeleton = self._skeleton
        clearance = self._distance_field_m
        if skeleton is None or clearance is None or not np.any(skeleton):
            return []

        raw_candidates: list[dict] = []
        skel_pts = np.argwhere(skeleton)
        edge_count = int(len(skel_pts))

        for ci, cj in skel_pts:
            cand = self._portal_candidate_at_point(int(ci), int(cj), skeleton, clearance, free, occupied)
            if cand is not None:
                raw_candidates.append(cand)

        self._door_debug_rooms.append({
            'skeleton_nodes': int(len(skel_pts)),
            'skeleton_edges': int(edge_count),
            'raw_portal_candidates': int(len(raw_candidates)),
        })
        return raw_candidates

    def _extract_split_space_candidates(
        self,
        free: np.ndarray,
        occupied: np.ndarray,
    ) -> list[dict]:
        clearance = self._distance_field_m
        if clearance is None:
            return []

        width_m = 2.0 * clearance
        local_max = clearance >= cv2.dilate(clearance, np.ones((3, 3), dtype=np.float32))
        candidate_mask = (
            free
            & local_max
            & (width_m >= float(self._DOOR_WIDTH_MIN_M))
            & (width_m <= float(min(self._DOOR_WIDTH_MAX_M, self._PORTAL_SPLIT_MAX_WIDTH_M)))
        )
        pts = np.argwhere(candidate_mask)
        if len(pts) == 0:
            return []

        dedup_r = max(2, int(np.ceil(self._PORTAL_SPLIT_DEDUP_RADIUS_M / max(self._cell_size, 1e-6))))
        order = np.argsort(width_m[pts[:, 0], pts[:, 1]])
        kept_pts: list[np.ndarray] = []
        out: list[dict] = []
        min_lobe_cells = int(np.ceil(self._PORTAL_MIN_LOBE_AREA_M2 / max(self._cell_size * self._cell_size, 1e-6)))

        for idx in order:
            ci, cj = (int(pts[idx, 0]), int(pts[idx, 1]))
            ij = np.array([ci, cj], dtype=np.float32)
            if kept_pts:
                d = np.linalg.norm(np.stack(kept_pts, axis=0) - ij[None, :], axis=1)
                if np.any(d < float(dedup_r)):
                    continue

            cand = self._split_space_candidate(ci, cj, free, occupied, min_lobe_cells)
            if cand is None:
                continue
            kept_pts.append(ij)
            out.append(cand)

        self._door_debug_rooms.append({
            'split_space_points': int(len(pts)),
            'split_space_candidates': int(len(out)),
        })
        return out

    def _split_space_candidate(
        self,
        ci: int,
        cj: int,
        free: np.ndarray,
        occupied: np.ndarray,
        min_lobe_cells: int,
    ) -> Optional[dict]:
        clearance = self._distance_field_m
        if clearance is None:
            return None
        cell = max(1e-6, self._cell_size)
        probe_r = max(6, int(np.ceil(self._PORTAL_SPLIT_PROBE_RADIUS_M / cell)))
        center_block_r = max(2, int(np.ceil(self._PORTAL_SPLIT_CENTER_BLOCK_M / cell)))

        r0 = max(0, ci - probe_r)
        r1 = min(self._shape[0], ci + probe_r + 1)
        c0 = max(0, cj - probe_r)
        c1 = min(self._shape[1], cj + probe_r + 1)
        patch = free[r0:r1, c0:c1].copy().astype(np.uint8)
        py = ci - r0
        px = cj - c0
        yy, xx = np.ogrid[:patch.shape[0], :patch.shape[1]]
        block = (yy - py) ** 2 + (xx - px) ** 2 <= center_block_r ** 2
        patch[block] = 0
        n_cc, cc = cv2.connectedComponents(patch, connectivity=8)

        border = np.zeros_like(patch, dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        border_areas = []
        for lbl in range(1, n_cc):
            comp = cc == lbl
            area = int(comp.sum())
            if area < min_lobe_cells:
                continue
            if not np.any(comp & border):
                continue
            border_areas.append(area)
        border_areas.sort(reverse=True)
        if len(border_areas) < 2:
            return None
        balance = float(border_areas[1]) / max(float(border_areas[0]), 1.0)
        if balance < float(self._PORTAL_SPLIT_MIN_BALANCE):
            return None

        tangent_rc = self._estimate_local_tangent(ci, cj, self._skeleton if self._skeleton is not None else free)
        if tangent_rc is None:
            axis = self._estimate_axis_from_free(ci, cj, free)
            tangent_rc = axis
        normal_rc = np.array([-tangent_rc[1], tangent_rc[0]], dtype=np.float32)

        return {
            'grid_center_rc': [int(ci), int(cj)],
            'pos_xy': self.grid_to_world(ci, cj).astype(np.float32),
            'clearance_m': float(clearance[ci, cj]),
            'width_m': float(2.0 * clearance[ci, cj]),
            'peak_diff_m': float(0.0),
            'ratio': float(1.0),
            'low_span_m': float(0.0),
            'score': float(0.4 + 0.1 * len(border_areas) + 0.001 * min(border_areas[1], 2000)),
            'tangent_rc': tangent_rc.astype(np.float32),
            'normal_rc': normal_rc.astype(np.float32),
            'lobe_areas_cells': [int(v) for v in border_areas[:4]],
            'lobe_balance': float(balance),
            'status': 'candidate',
            'reason': 'split_space_local_max',
            'bbox_rc': [int(r0), int(c0), int(r1 - 1), int(c1 - 1)],
        }

    def _estimate_axis_from_free(
        self,
        ci: int,
        cj: int,
        free: np.ndarray,
    ) -> np.ndarray:
        max_steps = max(4, int(np.ceil(self._DOOR_WIDTH_MAX_M / max(self._cell_size, 1e-6))))
        horiz = self._sample_axis_width(ci, cj, (0, 1), free, max_steps)
        vert = self._sample_axis_width(ci, cj, (1, 0), free, max_steps)
        if horiz <= vert:
            return np.array([0.0, 1.0], dtype=np.float32)
        return np.array([1.0, 0.0], dtype=np.float32)

    def _free_run_length(
        self,
        start_i: int,
        start_j: int,
        direction: tuple[int, int],
        free: np.ndarray,
        max_steps: int,
    ) -> int:
        steps = 0
        di, dj = direction
        for step in range(1, max_steps + 1):
            ci = start_i + di * step
            cj = start_j + dj * step
            if not (0 <= ci < self._shape[0] and 0 <= cj < self._shape[1]):
                break
            if not free[ci, cj]:
                break
            steps += 1
        return steps

    def _sample_axis_width(
        self,
        ci: int,
        cj: int,
        direction: tuple[int, int],
        free: np.ndarray,
        max_steps: int,
    ) -> int:
        return 1 + self._free_run_length(ci, cj, direction, free, max_steps) + self._free_run_length(ci, cj, (-direction[0], -direction[1]), free, max_steps)

    def _portal_candidate_at_point(
        self,
        ci: int,
        cj: int,
        skeleton: np.ndarray,
        clearance: np.ndarray,
        free: np.ndarray,
        occupied: np.ndarray,
    ) -> Optional[dict]:
        cell = max(1e-6, self._cell_size)
        width_min = float(self._DOOR_WIDTH_MIN_M)
        width_max = float(self._DOOR_WIDTH_MAX_M)
        c = float(clearance[ci, cj])
        width_m = 2.0 * c
        if width_m < width_min or width_m > width_max:
            return None
        tangent_rc = self._estimate_local_tangent(ci, cj, skeleton)
        if tangent_rc is None:
            return None
        lookahead_steps = max(3, int(np.ceil(1.0 / cell)))
        forward = self._sample_clearance_along_ray(ci, cj, tangent_rc, clearance, free, lookahead_steps)
        backward = self._sample_clearance_along_ray(ci, cj, -tangent_rc, clearance, free, lookahead_steps)
        if len(forward) < 2 or len(backward) < 2:
            return None

        left_peak = float(np.max(backward))
        right_peak = float(np.max(forward))
        chamber_peak = min(left_peak, right_peak)
        if chamber_peak < float(self._PORTAL_MIN_CHAMBER_CLEARANCE_M):
            return None
        peak_diff = chamber_peak - c
        if peak_diff < float(self._PORTAL_MIN_PEAK_DIFF_M):
            return None
        ratio = c / max(chamber_peak, 1e-6)
        if ratio > float(self._PORTAL_REL_CLEARANCE_RATIO):
            return None

        low_span_steps = self._low_clearance_span(ci, cj, tangent_rc, clearance, free, c + 0.08, lookahead_steps)
        low_span_m = float(low_span_steps * cell)
        if low_span_m < float(self._PORTAL_MIN_LOW_SPAN_M) or low_span_m > float(self._PORTAL_MAX_LOW_SPAN_M):
            return None

        support = self._portal_boundary_support(ci, cj, tangent_rc, occupied, c)
        if support < 2:
            return None
        normal_rc = np.array([-tangent_rc[1], tangent_rc[0]], dtype=np.float32)
        lobe_areas = self._portal_lobe_areas(ci, cj, normal_rc, c, free)
        min_lobe_area_cells = int(np.ceil(self._PORTAL_MIN_LOBE_AREA_M2 / max(cell * cell, 1e-6)))
        if len(lobe_areas) < 2 or lobe_areas[1] < min_lobe_area_cells:
            return None

        pos_xy = self.grid_to_world(ci, cj)
        return {
            'grid_center_rc': [int(ci), int(cj)],
            'pos_xy': pos_xy.astype(np.float32),
            'clearance_m': float(c),
            'width_m': float(width_m),
            'peak_diff_m': float(peak_diff),
            'ratio': float(ratio),
            'low_span_m': float(low_span_m),
            'score': float(peak_diff + 0.5 * (1.0 - ratio)),
            'tangent_rc': tangent_rc.astype(np.float32),
            'normal_rc': normal_rc.astype(np.float32),
            'lobe_areas_cells': [int(v) for v in lobe_areas[:4]],
            'status': 'candidate',
            'reason': 'clearance_bottleneck',
            'bbox_rc': [int(ci), int(cj), int(ci), int(cj)],
        }

    def _estimate_local_tangent(
        self,
        ci: int,
        cj: int,
        skeleton: np.ndarray,
    ) -> Optional[np.ndarray]:
        rad = max(2, int(np.ceil(self._PORTAL_EDGE_MARGIN_M / max(self._cell_size, 1e-6))))
        r0 = max(0, ci - rad)
        r1 = min(self._shape[0], ci + rad + 1)
        c0 = max(0, cj - rad)
        c1 = min(self._shape[1], cj + rad + 1)
        rr, cc = np.where(skeleton[r0:r1, c0:c1])
        if len(rr) < 3:
            return None
        pts = np.column_stack([rr.astype(np.float32) + r0, cc.astype(np.float32) + c0])
        pts -= pts.mean(axis=0, keepdims=True)
        cov = pts.T @ pts
        eigvals, eigvecs = np.linalg.eigh(cov)
        tangent = eigvecs[:, int(np.argmax(eigvals))]
        norm = float(np.hypot(tangent[0], tangent[1]))
        if norm < 1e-6:
            return None
        return (tangent / norm).astype(np.float32)

    def _sample_clearance_along_ray(
        self,
        ci: int,
        cj: int,
        direction_rc: np.ndarray,
        clearance: np.ndarray,
        free: np.ndarray,
        max_steps: int,
    ) -> np.ndarray:
        vals: list[float] = []
        seen: set[tuple[int, int]] = set()
        for step in range(1, max_steps + 1):
            ri = int(round(ci + float(direction_rc[0]) * step))
            rj = int(round(cj + float(direction_rc[1]) * step))
            if not (0 <= ri < self._shape[0] and 0 <= rj < self._shape[1]):
                break
            if (ri, rj) in seen:
                continue
            seen.add((ri, rj))
            if not free[ri, rj]:
                break
            vals.append(float(clearance[ri, rj]))
        return np.asarray(vals, dtype=np.float32)

    def _low_clearance_span(
        self,
        ci: int,
        cj: int,
        tangent_rc: np.ndarray,
        clearance: np.ndarray,
        free: np.ndarray,
        threshold: float,
        max_steps: int,
    ) -> int:
        span = 1
        for sign in (-1.0, 1.0):
            seen: set[tuple[int, int]] = set()
            for step in range(1, max_steps + 1):
                ri = int(round(ci + sign * float(tangent_rc[0]) * step))
                rj = int(round(cj + sign * float(tangent_rc[1]) * step))
                if not (0 <= ri < self._shape[0] and 0 <= rj < self._shape[1]):
                    break
                if (ri, rj) in seen:
                    continue
                seen.add((ri, rj))
                if not free[ri, rj] or float(clearance[ri, rj]) > threshold:
                    break
                span += 1
        return int(span)

    def _portal_lobe_areas(
        self,
        ci: int,
        cj: int,
        normal_rc: np.ndarray,
        clearance_m: float,
        free: np.ndarray,
    ) -> list[int]:
        cell = max(1e-6, self._cell_size)
        probe_r = max(4, int(np.ceil(self._PORTAL_PROBE_RADIUS_M / cell)))
        r0 = max(0, ci - probe_r)
        r1 = min(self._shape[0], ci + probe_r + 1)
        c0 = max(0, cj - probe_r)
        c1 = min(self._shape[1], cj + probe_r + 1)
        patch_free = free[r0:r1, c0:c1].copy().astype(np.uint8)

        yy, xx = np.ogrid[r0:r1, c0:c1]
        disk = ((yy - ci) ** 2 + (xx - cj) ** 2) <= probe_r ** 2
        patch_free &= disk.astype(np.uint8)

        local_ci = ci - r0
        local_cj = cj - c0
        half_len = max(
            2,
            int(
                np.ceil(
                    (float(clearance_m) + float(self._PORTAL_CUT_CLEARANCE_PAD_M))
                    / cell
                )
            ),
        )
        thickness = max(1, int(np.ceil(self._PORTAL_CUT_THICKNESS_M / cell)))
        p0 = (
            int(round(local_cj - float(normal_rc[1]) * half_len)),
            int(round(local_ci - float(normal_rc[0]) * half_len)),
        )
        p1 = (
            int(round(local_cj + float(normal_rc[1]) * half_len)),
            int(round(local_ci + float(normal_rc[0]) * half_len)),
        )
        cv2.line(patch_free, p0, p1, 0, thickness=thickness)
        n_cc, labels = cv2.connectedComponents(patch_free, connectivity=8)
        if n_cc <= 1:
            return []
        areas = sorted((int((labels == l).sum()) for l in range(1, n_cc)), reverse=True)
        return areas

    def _portal_boundary_support(
        self,
        ci: int,
        cj: int,
        tangent_rc: np.ndarray,
        occupied: np.ndarray,
        clearance_m: float,
    ) -> int:
        normal = np.array([-tangent_rc[1], tangent_rc[0]], dtype=np.float32)
        half_len = max(2, int(np.ceil((clearance_m + self._PORTAL_CUT_CLEARANCE_PAD_M) / max(self._cell_size, 1e-6))))
        hits = 0
        for sign in (-1.0, 1.0):
            found = False
            for step in range(1, half_len + 2):
                ri = int(round(ci + sign * normal[0] * step))
                rj = int(round(cj + sign * normal[1] * step))
                if not (0 <= ri < self._shape[0] and 0 <= rj < self._shape[1]):
                    break
                if occupied[ri, rj]:
                    found = True
                    break
            if found:
                hits += 1
        return hits

    def _nms_portal_candidates(self, candidates: list[dict]) -> list[dict]:
        candidates = sorted(candidates, key=lambda c: float(c['score']), reverse=True)
        kept: list[dict] = []
        for cand in candidates:
            pos_xy = cand['pos_xy']
            suppressed = False
            for kept_cand in kept:
                if float(np.linalg.norm(kept_cand['pos_xy'] - pos_xy)) < float(self._DOOR_NMS_RADIUS_M):
                    suppressed = True
                    break
            if suppressed:
                cand['status'] = 'rejected'
                cand['reason'] = 'suppressed_by_nms'
            else:
                cand['status'] = 'accepted'
                cand['reason'] = 'accepted'
                kept.append(cand)
            self._door_debug.append(self._debuggable_candidate(cand))
        return kept

    @staticmethod
    def _debuggable_candidate(cand: dict) -> dict:
        debug = {}
        for key, value in cand.items():
            if isinstance(value, np.ndarray):
                debug[key] = value.tolist()
            else:
                debug[key] = value
        return debug

    def _build_portal_cut_mask(self, free: np.ndarray) -> np.ndarray:
        cut_mask = np.zeros(self._shape, dtype=np.uint8)
        cell = max(1e-6, self._cell_size)
        thickness = max(1, int(np.ceil(self._PORTAL_CUT_THICKNESS_M / cell)))

        for cand in self._portal_candidates:
            if cand.get('status') != 'accepted':
                continue
            ci, cj = cand['grid_center_rc']
            normal = np.asarray(cand['normal_rc'], dtype=np.float32)
            half_len = max(
                2,
                int(
                    np.ceil(
                        (float(cand['clearance_m']) + float(self._PORTAL_CUT_CLEARANCE_PAD_M))
                        / cell
                    )
                ),
            )
            p0 = (
                int(round(cj - normal[1] * half_len)),
                int(round(ci - normal[0] * half_len)),
            )
            p1 = (
                int(round(cj + normal[1] * half_len)),
                int(round(ci + normal[0] * half_len)),
            )
            cv2.line(cut_mask, p0, p1, 1, thickness=thickness)

        cut_mask = (cut_mask > 0) & free
        return cut_mask

    def _match_room_id(
        self,
        mask: np.ndarray,
        previous_rooms: dict[int, Room],
    ) -> int:
        best_id = None
        best_score = 0.0
        for rid, prev_room in previous_rooms.items():
            inter = float(np.logical_and(mask, prev_room.mask).sum())
            if inter <= 0:
                continue
            union = float(np.logical_or(mask, prev_room.mask).sum())
            if union <= 0:
                continue
            score = inter / union
            if score > best_score:
                best_score = score
                best_id = int(rid)

        if best_id is not None and best_score >= 0.20:
            previous_rooms.pop(best_id, None)
            return int(best_id)

        self._n_rooms += 1
        return int(self._n_rooms)

    def _room_ids_near_point(self, pos_xy: np.ndarray) -> set[int]:
        room_ids: set[int] = set()
        if self._room_map is None:
            return room_ids
        ci, cj = self.world_to_grid(float(pos_xy[0]), float(pos_xy[1]))
        rad = max(2, int(np.ceil((self._DOOR_WIDTH_MAX_M / 2.0 + 0.15) / max(self._cell_size, 1e-6))))
        r0 = max(0, ci - rad)
        r1 = min(self._shape[0], ci + rad + 1)
        c0 = max(0, cj - rad)
        c1 = min(self._shape[1], cj + rad + 1)
        for rid in np.unique(self._room_map[r0:r1, c0:c1]):
            rid = int(rid)
            if rid > 0 and rid in self.confirmed_rooms:
                room_ids.add(rid)
        return room_ids

    # ------------------------------------------------------------------
    # Flanking-wall helpers  (port of lidar_gridmap)
    # ------------------------------------------------------------------

    @staticmethod
    def _wall_is_linear(
        wall_i: np.ndarray, wall_j: np.ndarray, max_deviation_cells: float = 1.5
    ) -> bool:
        """Return True if wall cells fit a straight line (PCA minor-axis residual)."""
        if len(wall_i) < 3:
            return True   # too few points to judge — accept
        pts = np.column_stack([wall_j.astype(float), wall_i.astype(float)])
        centroid = pts.mean(axis=0)
        centered = pts - centroid
        cov = centered.T @ centered
        _, eigvecs = np.linalg.eigh(cov)
        minor_axis = eigvecs[:, 0]   # eigenvector for smallest eigenvalue
        residuals = np.abs(centered @ minor_axis)
        return float(residuals.max()) <= max_deviation_cells

    def _collect_wall_strip(
        self,
        start_i: int,
        start_j: int,
        direction: tuple[int, int],
        length: int,
        occupied: np.ndarray,
        search_width: int = 2,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Walk from (start_i, start_j) in `direction` for up to `length` steps,
        collecting occupied cells within ±search_width perpendicular cells.
        Stops when no wall cell is found in a step.
        """
        H, W = occupied.shape
        di, dj = direction
        found_i, found_j = [], []
        for step in range(1, length + 1):
            ci = start_i + di * step
            cj = start_j + dj * step
            hit = False
            for w in range(-search_width, search_width + 1):
                wi = ci + (0 if di != 0 else w)
                wj = cj + (w if di != 0 else 0)
                if 0 <= wi < H and 0 <= wj < W and occupied[wi, wj]:
                    found_i.append(wi)
                    found_j.append(wj)
                    hit = True
            if not hit:
                break
        return np.array(found_i, dtype=int), np.array(found_j, dtype=int)

    def _gap_has_flanking_walls(
        self,
        rows: np.ndarray,
        cols: np.ndarray,
        occupied: np.ndarray,
        min_wall_len_cells: int,
    ) -> bool:
        """
        Check that a boundary gap has straight flanking walls on both sides.

        Mirrors lidar_gridmap._gap_has_flanking_walls.
        """
        if len(rows) == 0:
            return False

        i_span = int(rows.max() - rows.min())
        j_span = int(cols.max() - cols.min())
        search_len = min_wall_len_cells + 3

        if j_span >= i_span:
            # Horizontal gap — flanking walls extend in ±j direction
            i_c  = int(np.median(rows))
            j_lo = int(cols.min())
            j_hi = int(cols.max())

            left_i, left_j = self._collect_wall_strip(
                i_c, j_lo, (0, -1), search_len, occupied)
            if len(left_i) < min_wall_len_cells:
                return False
            if not self._wall_is_linear(left_i, left_j):
                return False

            right_i, right_j = self._collect_wall_strip(
                i_c, j_hi, (0, 1), search_len, occupied)
            if len(right_i) < min_wall_len_cells:
                return False
            if not self._wall_is_linear(right_i, right_j):
                return False
        else:
            # Vertical gap — flanking walls extend in ±i direction
            j_c  = int(np.median(cols))
            i_lo = int(rows.min())
            i_hi = int(rows.max())

            top_i, top_j = self._collect_wall_strip(
                i_lo, j_c, (-1, 0), search_len, occupied)
            if len(top_i) < min_wall_len_cells:
                return False
            if not self._wall_is_linear(top_i, top_j):
                return False

            bot_i, bot_j = self._collect_wall_strip(
                i_hi, j_c, (1, 0), search_len, occupied)
            if len(bot_i) < min_wall_len_cells:
                return False
            if not self._wall_is_linear(bot_i, bot_j):
                return False

        return True

    # ------------------------------------------------------------------
    # Door track persistence
    # ------------------------------------------------------------------

    def _update_door_tracks(self, candidates: list[tuple[np.ndarray, set[int]]]) -> None:
        for track in self._door_tracks.values():
            track['misses'] += 1

        used_tracks: set[int] = set()
        for pos_xy, room_ids in candidates:
            best_id = None
            best_d = float('inf')
            for tid, track in self._door_tracks.items():
                if tid in used_tracks:
                    continue
                d = float(np.linalg.norm(track['pos_xy'] - pos_xy))
                if d < self._DOOR_PERSIST_RADIUS_M and d < best_d:
                    best_d = d
                    best_id = tid

            if best_id is None:
                tid = self._n_doors
                self._n_doors += 1
                self._door_tracks[tid] = {
                    'pos_xy': pos_xy.astype(np.float32),
                    'room_ids': set(int(v) for v in room_ids),
                    'hits': 1,
                    'misses': 0,
                }
                used_tracks.add(tid)
            else:
                track = self._door_tracks[best_id]
                track['pos_xy'] = 0.5 * track['pos_xy'] + 0.5 * pos_xy.astype(np.float32)
                track['room_ids'] |= set(int(v) for v in room_ids)
                track['hits'] += 1
                track['misses'] = 0
                used_tracks.add(best_id)

        self._door_tracks = {
            tid: track
            for tid, track in self._door_tracks.items()
            if track['misses'] <= self._DOOR_TRACK_MAX_MISSES
        }

    def _realize_doors_from_tracks(self) -> None:
        for tid, track in self._door_tracks.items():
            if int(track['hits']) < self._DOOR_PERSIST_HITS:
                continue
            room_ids = set(int(v) for v in track['room_ids'] if int(v) in self.confirmed_rooms)
            if not room_ids:
                continue
            door = Door(id=int(tid), pos_xy=track['pos_xy'].astype(np.float32), room_ids=room_ids)
            self.doors.append(door)
            for rid in room_ids:
                room = self.confirmed_rooms.get(int(rid))
                if room is not None:
                    room.doors.append(door)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _largest_cc(mask: np.ndarray) -> np.ndarray:
        """Return mask with only the largest connected component."""
        lbl_img = mask.astype(np.uint8)
        n_cc, labels = cv2.connectedComponents(lbl_img)
        if n_cc <= 1:
            return mask
        areas = [(labels == l).sum() for l in range(1, n_cc)]
        best  = int(np.argmax(areas)) + 1
        return labels == best

    def _mask_centroid(self, mask: np.ndarray) -> np.ndarray:
        """World-coord centroid of a boolean grid mask."""
        rows, cols = np.where(mask)
        ci = float(rows.mean())
        cj = float(cols.mean())
        return self.grid_to_world(int(ci), int(cj))
