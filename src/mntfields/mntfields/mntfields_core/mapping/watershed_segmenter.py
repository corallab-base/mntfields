"""
watershed_segmenter.py — Room segmentation via watershed on the SLAM distance field.

Algorithm (Bormann et al. 2016 style):
  1. Distance transform of free-space binary image.
     Each free cell value = clearance from nearest obstacle/unknown cell (metres).
  2. Find local maxima with minimum inter-peak separation.
     Each maximum = one room seed (the point farthest from all walls).
  3. Watershed flood from seeds on the negated distance field.
     Each basin = one room.  Basin boundaries sit at saddle points of the
     distance field, which are geometrically the narrowest passages = doorways.
  4. Doorway detection: scan boundaries between adjacent rooms; where the
     distance field is in [door_min_m, door_max_m] → doorway.
  5. Room confirmation: a room is confirmed when frontier count on its
     boundary drops below threshold (room is fully explored).

Pure numpy/scipy/cv2 — no ROS dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class WatershedRoom:
    id: int
    centroid_xy: np.ndarray        # world [x, y] of the peak (farthest from walls)
    peak_clearance_m: float        # distance-to-wall at the peak
    bounding_box: np.ndarray       # [[xmin,ymin],[xmax,ymax]] world coords
    area_m2: float
    confirmed: bool = False

    # ------------------------------------------------------------------
    # RoomSegment-compatible interface (used by room_graph.py and mapping_node.py)
    # ------------------------------------------------------------------
    @property
    def centroid(self) -> np.ndarray:
        """Alias for centroid_xy — matches RoomSegment.centroid interface."""
        return self.centroid_xy

    @property
    def line_segments(self) -> list:
        """Stub — watershed rooms have no line segments."""
        return []

    def contains_point(self, xy: np.ndarray) -> bool:
        bb = self.bounding_box
        return bool(
            bb[0, 0] <= xy[0] <= bb[1, 0] and
            bb[0, 1] <= xy[1] <= bb[1, 1]
        )


@dataclass
class WatershedDoorway:
    id: int
    position_xy: np.ndarray        # world [x, y] midpoint of the doorway
    width_m: float                 # clearance at the doorway (metres)
    room_ids: tuple[int, int]      # the two rooms this connects

    # ------------------------------------------------------------------
    # Doorway-compatible interface (used by room_graph.py)
    # ------------------------------------------------------------------
    @property
    def position(self) -> np.ndarray:
        """Alias for position_xy — matches Doorway.position interface."""
        return self.position_xy

    @property
    def width(self) -> float:
        """Alias for width_m — matches Doorway.width interface."""
        return self.width_m


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

# Room seed detection
_MIN_PEAK_CLEARANCE_M   = 0.40   # robot must fit: ignore peaks below this clearance
_MIN_PEAK_SEPARATION_M  = 2.0    # minimum distance between room seeds
_MIN_ROOM_AREA_M2       = 0.8    # discard rooms smaller than this

# Doorway detection
_DOOR_MIN_M             = 0.55   # min doorway width (robot must fit)
_DOOR_MAX_M             = 3.50   # max doorway width (wide archways)

# Confirmation: frontier count thresholds on room boundary
_CONFIRM_BDRY_FRONTIER_CELLS = 12   # boundary frontier cells allowed
_CONFIRM_INNER_FRONTIER_FRAC = 0.04 # fraction of room interior allowed to be frontier

# Temporal stability: room must survive this many consecutive updates to confirm
_CONFIRM_MIN_TICKS = 2


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------

def _build_distance_field(free: np.ndarray, cell_size_m: float) -> np.ndarray:
    """Euclidean distance transform of free space (metres)."""
    dist = cv2.distanceTransform(
        free.astype(np.uint8), cv2.DIST_L2, 5
    ).astype(np.float32)
    return dist * cell_size_m


def _find_room_seeds(
    dist: np.ndarray,
    free: np.ndarray,
    cell_size_m: float,
    min_clearance_m: float = _MIN_PEAK_CLEARANCE_M,
    min_separation_m: float = _MIN_PEAK_SEPARATION_M,
) -> np.ndarray:
    """
    Return (N, 2) array of grid (row, col) peak positions.
    Each peak is a local maximum of the distance field with:
      - clearance >= min_clearance_m
      - separation from other peaks >= min_separation_m
    """
    min_sep_cells = max(2, int(round(min_separation_m / cell_size_m)))

    # threshold_abs must be in the same units as dist (metres)
    peaks = peak_local_max(
        dist,
        min_distance=min_sep_cells,
        threshold_abs=float(min_clearance_m),
        exclude_border=False,
        labels=free.astype(np.uint8),
    )
    return peaks   # shape (N, 2): each row is [row_i, col_j]


def _run_watershed(
    dist: np.ndarray,
    seeds_ij: np.ndarray,
    free: np.ndarray,
) -> np.ndarray:
    """
    Watershed on negated distance field seeded at peaks.
    Returns integer label map [H, W]: 0 = wall/unknown, 1..N = room IDs.
    """
    if len(seeds_ij) == 0:
        # No seeds: treat entire free space as one room
        out = np.zeros_like(dist, dtype=np.int32)
        out[free] = 1
        return out

    # Build marker image: each seed gets a unique label
    markers = np.zeros(dist.shape, dtype=np.int32)
    for idx, (r, c) in enumerate(seeds_ij, start=1):
        markers[r, c] = idx

    # Watershed floods uphill on dist (downhill on -dist) from each seed
    labels = watershed(-dist, markers, mask=free.astype(bool))
    return labels.astype(np.int32)


def _find_doorways(
    room_map: np.ndarray,
    dist: np.ndarray,
    cell_size_m: float,
    door_min_m: float = _DOOR_MIN_M,
    door_max_m: float = _DOOR_MAX_M,
) -> tuple[list[tuple[tuple[int, int], float, np.ndarray]], list[str]]:
    """
    Scan boundaries between adjacent rooms. Where the distance field is in
    [door_min_m, door_max_m], record a doorway.

    Returns (doorways, debug_lines) where:
      doorways  = list of (room_id_pair, width_m, boundary_cells_ij)
      debug_lines = human-readable debug strings (one per contact pair found)
    """
    H, W = room_map.shape
    room_ids = [int(r) for r in np.unique(room_map) if r > 0]

    # Dilate each room by 1 cell and find where it touches a different room
    doorways = []
    debug_lines: list[str] = []
    seen_pairs: set[tuple[int, int]] = set()

    kernel = np.ones((3, 3), np.uint8)

    for rid in room_ids:
        mask = (room_map == rid).astype(np.uint8)
        dilated = cv2.dilate(mask, kernel, iterations=1).astype(bool)

        # Cells in dilation that belong to a different room
        border_mask = dilated & (room_map > 0) & (room_map != rid)
        neighbor_ids = [int(v) for v in np.unique(room_map[border_mask]) if int(v) != rid]

        for nid in neighbor_ids:
            pair = (min(rid, nid), max(rid, nid))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)

            # Boundary cells = dilation of rid touching nid
            contact = border_mask & (room_map == nid)
            contact_ij = np.argwhere(contact)
            if len(contact_ij) == 0:
                debug_lines.append(
                    f"  pair(lbl {rid},{nid}): contact_ij empty → skip")
                continue

            # Measure clearance at the boundary.
            # The cell with MAX clearance is the center of the passage opening;
            # max_clr × 2 estimates the passage width at that point.
            # (min_clr would just be the outermost edge cell, often 1 cell = 0.05m)
            clearances = dist[contact_ij[:, 0], contact_ij[:, 1]]
            min_clr    = float(np.min(clearances))
            median_clr = float(np.median(clearances))
            max_clr    = float(np.max(clearances))
            width = max_clr * 2.0

            verdict = "ACCEPT" if door_min_m <= width <= door_max_m else f"REJECT(range [{door_min_m:.2f},{door_max_m:.2f}])"
            debug_lines.append(
                f"  pair(lbl {rid},{nid}): {len(contact_ij)} cells  "
                f"clr min={min_clr:.3f} med={median_clr:.3f} max={max_clr:.3f} → "
                f"width={width:.3f}m  {verdict}"
            )

            if door_min_m <= width <= door_max_m:
                # Doorway position = contact cell with maximum clearance (passage centre)
                best_idx = int(np.argmax(clearances))
                doorways.append((pair, width, contact_ij))

    debug_lines.insert(0,
        f"_find_doorways: {len(room_ids)} labels, {len(seen_pairs)} pairs → {len(doorways)} passed width filter")
    return doorways, debug_lines


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class WatershedSegmenter:
    """
    Online watershed-based room segmenter. Call update() each time the SLAM
    map changes. Rooms are stable — they can grow as exploration proceeds but
    do not split or merge unless the topology genuinely changes.

    Usage::

        seg = WatershedSegmenter()
        rooms, doorways = seg.update(slam_arr, cell_size_m, origin_xy,
                                     frontier_mask=frontier_mask)
    """

    def __init__(
        self,
        min_peak_clearance_m: float = _MIN_PEAK_CLEARANCE_M,
        min_peak_separation_m: float = _MIN_PEAK_SEPARATION_M,
        min_room_area_m2: float = _MIN_ROOM_AREA_M2,
        door_min_m: float = _DOOR_MIN_M,
        door_max_m: float = _DOOR_MAX_M,
    ):
        self.min_peak_clearance_m  = min_peak_clearance_m
        self.min_peak_separation_m = min_peak_separation_m
        self.min_room_area_m2      = min_room_area_m2
        self.door_min_m            = door_min_m
        self.door_max_m            = door_max_m

        # Persistent state
        self._room_map: Optional[np.ndarray] = None   # [H, W] int32
        self._dist:     Optional[np.ndarray] = None   # [H, W] float32 metres
        self._cell_size_m: float = 0.05
        self._origin_xy: np.ndarray = np.zeros(2)

        self.rooms:     dict[int, WatershedRoom]    = {}
        self.doorways:  dict[int, WatershedDoorway] = {}

        # Temporal confirmation tracking: room_id -> consecutive ticks seen
        self._room_ticks:      dict[int, int] = {}
        self._confirm_reasons: dict[int, str] = {}
        self._next_door_id: int = 0

        # Debug info populated each update (for external logging)
        self._door_debug_lines:       list[str] = []
        self._build_door_debug_lines: list[str] = []

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def update(
        self,
        slam_arr:      np.ndarray,
        cell_size_m:   float,
        origin_xy:     np.ndarray,
        frontier_mask: Optional[np.ndarray] = None,
    ) -> tuple[list[WatershedRoom], list[WatershedDoorway]]:
        """
        Run segmentation on the current SLAM occupancy grid.

        Parameters
        ----------
        slam_arr      : [H, W] int8/uint8 — 0=free, >50=occupied, <0/255=unknown
        cell_size_m   : metres per cell
        origin_xy     : [x, y] world coords of grid cell (0, 0)
        frontier_mask : optional [H, W] bool — True where frontier (free↔unknown)

        Returns
        -------
        rooms, doorways
        """
        self._cell_size_m = float(cell_size_m)
        self._origin_xy   = np.asarray(origin_xy[:2], dtype=np.float32)

        arr  = slam_arr.astype(np.int16)
        free = (arr == 0)

        # 1. Distance field
        dist = _build_distance_field(free, cell_size_m)
        self._dist = dist

        # 2. Room seeds (local maxima)
        seeds_ij = _find_room_seeds(
            dist, free, cell_size_m,
            min_clearance_m=self.min_peak_clearance_m,
            min_separation_m=self.min_peak_separation_m,
        )

        # 3. Watershed
        room_map = _run_watershed(dist, seeds_ij, free)

        # 4. Remove undersized basins
        min_cells = int(self.min_room_area_m2 / (cell_size_m ** 2))
        for rid in [int(r) for r in np.unique(room_map) if r > 0]:
            if int((room_map == rid).sum()) < min_cells:
                room_map[room_map == rid] = 0

        self._room_map = room_map

        # 5. Build room objects with stable IDs
        rooms = self._build_rooms(room_map, dist, seeds_ij, free,
                                  cell_size_m, frontier_mask)

        # 6. Extract doorways — pass new rooms directly so label→ID mapping is current
        raw_doors, self._door_debug_lines = _find_doorways(
            room_map, dist, cell_size_m, self.door_min_m, self.door_max_m)
        doorways, self._build_door_debug_lines = self._build_doorways(
            raw_doors, room_map, rooms)

        self.rooms    = {r.id: r for r in rooms}
        self.doorways = {d.id: d for d in doorways}

        return rooms, doorways

    # ------------------------------------------------------------------
    # Room building with stable IDs and confirmation
    # ------------------------------------------------------------------

    def _build_rooms(
        self,
        room_map:     np.ndarray,
        dist:         np.ndarray,
        seeds_ij:     np.ndarray,
        free:         np.ndarray,
        cell_size_m:  float,
        frontier_mask: Optional[np.ndarray],
    ) -> list[WatershedRoom]:
        """
        Build WatershedRoom objects.  IDs are stable: a new room that
        overlaps a previous room inherits its ID.
        """
        # Map seed index → room label
        seed_to_label: dict[int, int] = {}
        for idx, (r, c) in enumerate(seeds_ij, start=1):
            lbl = int(room_map[r, c])
            if lbl > 0:
                seed_to_label[idx] = lbl

        label_to_peak: dict[int, np.ndarray] = {}
        for idx, (r, c) in enumerate(seeds_ij, start=1):
            lbl = seed_to_label.get(idx)
            if lbl is not None:
                label_to_peak[lbl] = np.array([r, c], dtype=np.int32)

        rooms: list[WatershedRoom] = []
        current_labels = set(int(v) for v in np.unique(room_map) if v > 0)
        new_ticks: dict[int, int] = {}

        # Match current labels to previous rooms by centroid proximity
        prev_centroids = {rid: r.centroid_xy for rid, r in self.rooms.items()}

        label_to_id: dict[int, int] = {}
        used_prev_ids: set[int] = set()

        for lbl in current_labels:
            mask = (room_map == lbl)
            peak_ij = label_to_peak.get(lbl)
            if peak_ij is None:
                # fallback: use mask centroid
                ij = np.argwhere(mask)
                peak_ij = ij[int(np.argmax(dist[mask]))]

            peak_xy = self._grid_to_world(int(peak_ij[0]), int(peak_ij[1]))

            # Match to nearest previous room centroid within 3 m
            best_id = None
            best_dist = 3.0
            for prev_id, prev_c in prev_centroids.items():
                if prev_id in used_prev_ids:
                    continue
                d = float(np.linalg.norm(prev_c - peak_xy))
                if d < best_dist:
                    best_dist = d
                    best_id = prev_id

            if best_id is not None:
                room_id = best_id
                used_prev_ids.add(best_id)
            else:
                # New room — assign fresh ID = next integer not already used
                room_id = max((list(self.rooms.keys()) + list(label_to_id.values()) + [0])) + 1

            label_to_id[lbl] = room_id

        # Build room objects
        for lbl, room_id in label_to_id.items():
            mask = (room_map == lbl)
            peak_ij = label_to_peak.get(lbl)
            if peak_ij is None:
                ij = np.argwhere(mask)
                peak_ij = ij[int(np.argmax(dist[mask]))]

            peak_xy   = self._grid_to_world(int(peak_ij[0]), int(peak_ij[1]))
            peak_clr  = float(dist[int(peak_ij[0]), int(peak_ij[1])])
            area_m2   = float(mask.sum()) * (cell_size_m ** 2)
            bb        = self._mask_bbox_world(mask, cell_size_m)

            # Confirmation
            ticks = self._room_ticks.get(room_id, 0) + 1
            new_ticks[room_id] = ticks
            conf_ok, conf_reason = self._check_confirmed(mask, frontier_mask)
            confirmed = (ticks >= _CONFIRM_MIN_TICKS and conf_ok)
            # Store reason for external logging
            self._confirm_reasons[room_id] = conf_reason

            rooms.append(WatershedRoom(
                id=room_id,
                centroid_xy=peak_xy,
                peak_clearance_m=peak_clr,
                bounding_box=bb,
                area_m2=area_m2,
                confirmed=confirmed,
            ))

        self._room_ticks = new_ticks
        return rooms

    def _check_confirmed(
        self,
        room_mask: np.ndarray,
        frontier_mask: Optional[np.ndarray],
    ) -> tuple[bool, str]:
        """Return (confirmed, reason_string) for logging."""
        if frontier_mask is None:
            return True, "no frontier info"

        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(room_mask.astype(np.uint8), kernel).astype(bool)
        boundary = dilated & ~room_mask

        bdry_frontiers = int(frontier_mask[boundary].sum())
        if bdry_frontiers > _CONFIRM_BDRY_FRONTIER_CELLS:
            return False, f"bdry_frontiers={bdry_frontiers}>{_CONFIRM_BDRY_FRONTIER_CELLS}"

        inner_frontiers = int(frontier_mask[room_mask].sum())
        room_cells = max(1, int(room_mask.sum()))
        frac = inner_frontiers / room_cells
        if frac > _CONFIRM_INNER_FRONTIER_FRAC:
            return False, f"inner_frontier_frac={frac:.3f}>{_CONFIRM_INNER_FRONTIER_FRAC}"

        return True, "ok"

    # ------------------------------------------------------------------
    # Doorway building
    # ------------------------------------------------------------------

    def _build_doorways(
        self,
        raw_doors: list[tuple[tuple[int, int], float, np.ndarray]],
        room_map: np.ndarray,
        new_rooms: list[WatershedRoom],
    ) -> tuple[list[WatershedDoorway], list[str]]:
        """Convert raw door detections to WatershedDoorway objects with stable IDs."""
        debug: list[str] = []

        # Map current watershed labels to stable room IDs using the freshly built rooms
        # (self.rooms is still the previous frame here — must use new_rooms)
        label_to_id: dict[int, int] = {}
        for room in new_rooms:
            peak_ij_arr = self._world_to_grid(room.centroid_xy[0], room.centroid_xy[1])
            r_i = max(0, min(room_map.shape[0] - 1, peak_ij_arr[0]))
            c_i = max(0, min(room_map.shape[1] - 1, peak_ij_arr[1]))
            lbl = int(room_map[r_i, c_i])
            debug.append(f"  room_id={room.id} centroid={room.centroid_xy} → grid=({r_i},{c_i}) → lbl={lbl}")
            if lbl > 0:
                label_to_id[lbl] = room.id

        debug.insert(0, f"_build_doorways: {len(raw_doors)} raw doors, label_to_id={label_to_id}")

        doorways: list[WatershedDoorway] = []
        for (lbl1, lbl2), width_m, contact_ij in raw_doors:
            rid1 = label_to_id.get(lbl1)
            rid2 = label_to_id.get(lbl2)
            if rid1 is None or rid2 is None:
                debug.append(f"  raw door lbl({lbl1},{lbl2}) → rid1={rid1} rid2={rid2} SKIP (unmapped label)")
                continue

            # Doorway position = contact cell with highest clearance (passage centre)
            clearances_here = self._dist[contact_ij[:, 0], contact_ij[:, 1]]
            best_idx = int(np.argmax(clearances_here))
            pos_xy = self._grid_to_world(
                int(contact_ij[best_idx, 0]), int(contact_ij[best_idx, 1]))

            # Reuse existing doorway ID if same pair exists
            existing_id = None
            for did, d in self.doorways.items():
                if set(d.room_ids) == {rid1, rid2}:
                    existing_id = did
                    break

            if existing_id is None:
                self._next_door_id += 1
                existing_id = self._next_door_id

            debug.append(f"  raw door lbl({lbl1},{lbl2}) → room({rid1},{rid2}) w={width_m:.3f}m id={existing_id} ADDED")
            doorways.append(WatershedDoorway(
                id=existing_id,
                position_xy=pos_xy,
                width_m=width_m,
                room_ids=(rid1, rid2),
            ))

        return doorways, debug

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def room_for_point(self, xy: np.ndarray) -> Optional[int]:
        """
        Return the room ID at world position xy, using the room_map directly.
        Much more precise than bounding-box lookup.
        """
        if self._room_map is None:
            return None
        r, c = self._world_to_grid(float(xy[0]), float(xy[1]))
        H, W = self._room_map.shape
        if not (0 <= r < H and 0 <= c < W):
            return self._fallback_nearest_room(xy)

        # Direct lookup in watershed map
        lbl = int(self._room_map[r, c])
        if lbl == 0:
            # Edge case: on a boundary — find nearest room by expanding
            return self._fallback_nearest_room(xy)

        # Map label to stable ID via centroid matching
        for rid, room in self.rooms.items():
            peak_ij = self._world_to_grid(room.centroid_xy[0], room.centroid_xy[1])
            pr = max(0, min(self._room_map.shape[0]-1, peak_ij[0]))
            pc = max(0, min(self._room_map.shape[1]-1, peak_ij[1]))
            if int(self._room_map[pr, pc]) == lbl:
                return rid

        return self._fallback_nearest_room(xy)

    def _fallback_nearest_room(self, xy: np.ndarray) -> Optional[int]:
        if not self.rooms:
            return None
        xy = np.asarray(xy[:2])
        return min(self.rooms.keys(),
                   key=lambda rid: float(np.linalg.norm(
                       self.rooms[rid].centroid_xy - xy)))

    def get_room_map(self) -> Optional[np.ndarray]:
        """Return the raw [H, W] integer room label map."""
        return self._room_map

    def get_distance_field(self) -> Optional[np.ndarray]:
        """Return the [H, W] float32 clearance field in metres."""
        return self._dist

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def _grid_to_world(self, row: int, col: int) -> np.ndarray:
        x = self._origin_xy[0] + col * self._cell_size_m
        y = self._origin_xy[1] + row * self._cell_size_m
        return np.array([x, y], dtype=np.float32)

    def _world_to_grid(self, x: float, y: float) -> tuple[int, int]:
        col = int((x - self._origin_xy[0]) / self._cell_size_m)
        row = int((y - self._origin_xy[1]) / self._cell_size_m)
        return row, col

    def _mask_bbox_world(self, mask: np.ndarray, cell_size_m: float) -> np.ndarray:
        ij = np.argwhere(mask)
        if len(ij) == 0:
            return np.zeros((2, 2), dtype=np.float32)
        rows, cols = ij[:, 0], ij[:, 1]
        r_min, r_max = int(rows.min()), int(rows.max())
        c_min, c_max = int(cols.min()), int(cols.max())
        xy_min = self._grid_to_world(r_min, c_min)
        xy_max = self._grid_to_world(r_max, c_max)
        return np.array([xy_min, xy_max], dtype=np.float32)
