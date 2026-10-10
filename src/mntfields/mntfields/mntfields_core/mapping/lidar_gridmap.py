import numpy as np
import networkx as nx
import skimage
import skimage.morphology
import skimage.segmentation
import cv2
import torch
import scipy
import json
import os

from typing import Set, List, Dict, Union, Tuple
from itertools import combinations
# core classes
from mntfields.mntfields_core.mapping.rooms_doors import (
    Room,
    Door,
    ConfirmedRoom,
    UnconfirmedRoom,
)
from mntfields.mntfields_core.core.region import Region
from mntfields.mntfields_core.core.exploration_state import FrontierNode
# utils (your merged utils file)
from mntfields.utils.mntfields_utils import (
    get_neighbor_at_least,
    connected_components,
)

class LidarGridmap:
    def __init__(self, trav_map, regions, n_cells=100, occupancy_threshold=0.01, 
             near_dist=0.1, room_seg_config=(7, 2), map_size=20.0, origin=(0.0, 0.0)):
        """
        Initialize the gridmap.
        :param n_cells: Number of cells along each dimension of the gridmap.
        """
        self.n_cells = n_cells
        self.map_size = map_size  # meters
        self.origin = np.array(origin)  # map center
        self.cell_size = map_size / n_cells  # meters per cell

        self.data = np.zeros((n_cells, n_cells), dtype=np.uint8)
        self.visited = np.zeros_like(self.data)
        self.frontier = np.zeros_like(self.data)
        # SLAM semantics at the MNTFields grid resolution.
        # These preserve the free / occupied / unknown distinction that
        # `self.data` alone cannot represent.
        self.slam_known = np.zeros_like(self.data, dtype=bool)
        self.slam_free = np.zeros_like(self.data, dtype=bool)
        self.slam_occupied = np.zeros_like(self.data, dtype=bool)
        self.slam_raw_data = None
        self.slam_raw_resolution = None
        self.slam_raw_origin = None
        self.slam_raw_shape = None
        self.slam_frontier_points = np.zeros((0, 2), dtype=np.float32)

        self.confirmed_rooms: Dict[int, ConfirmedRoom] = {}
        self.confirmed_room_map = np.zeros_like(self.data)
        self.unconfirmed_rooms: Dict[int, UnconfirmedRoom] = {}
        self.room_seg = np.zeros_like(self.data)
        self.outer_label = 1
        self.new_confirmed_rooms: List[ConfirmedRoom] = []
        self.rooms = []

        self.occupancy_threshold = occupancy_threshold

        self.confirmed_label = 50
        self.unconfirmed_label = 100

        self.room_graph = nx.Graph()
        self.room_connectivity_graph = nx.Graph()
        self.n_doors = 0
        self.doors: Set[Door] = set()

        self.trav_map = trav_map
        self.near_dist = near_dist

        self.boundary_graph = nx.Graph()
        self.discovery_map = np.zeros_like(self.data, dtype=np.uint64)

        self.all_lidar = None
        self.kd_tree = None
        
        self.room_seg_config = room_seg_config
        self.regions: Dict[int, Region] = regions
        # Door nodes belonging to the same physical opening should be close.
        # Auto-scale by grid resolution so behavior stays stable across maps.
        self.door_match_radius_cells = 6.0
        self.door_match_radius_m = max(0.2, self.door_match_radius_cells * self.cell_size)

    def ingest_slam_map(self, slam_msg):
        """Ingest a SLAM Toolbox OccupancyGrid into self.data.
        Handles mismatched resolution and origin via vectorized coordinate mapping.
        Only updates occupancy (self.data); visited/frontier tracking is unchanged.
        """
        info = slam_msg.info
        slam_res = info.resolution
        slam_ox  = info.origin.position.x   # lower-left x in map frame
        slam_oy  = info.origin.position.y   # lower-left y in map frame
        slam_w   = info.width
        slam_h   = info.height
        if slam_w == 0 or slam_h == 0:
            return
        slam_data = np.array(slam_msg.data, dtype=np.int8).reshape(slam_h, slam_w)
        self.slam_raw_data = slam_data
        self.slam_raw_resolution = float(slam_res)
        self.slam_raw_origin = np.array([slam_ox, slam_oy], dtype=np.float32)
        self.slam_raw_shape = (int(slam_h), int(slam_w))

        n = self.n_cells
        ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')

        # Centre of each cell in our grid (map frame).
        # Convention throughout lidar_gridmap: data[i=row=y_based, j=col=x_based].
        # With indexing='ij': ii[a,b]=a (first axis = row = y), jj[a,b]=b (second axis = col = x).
        world_x = self.origin[0] - self.map_size / 2 + (jj + 0.5) * self.cell_size  # col → x
        world_y = self.origin[1] - self.map_size / 2 + (ii + 0.5) * self.cell_size  # row → y

        # Corresponding SLAM map cell indices (col = x-axis, row = y-axis)
        slam_col = ((world_x - slam_ox) / slam_res).astype(np.int32)
        slam_row = ((world_y - slam_oy) / slam_res).astype(np.int32)

        valid = ((slam_col >= 0) & (slam_col < slam_w) &
                 (slam_row >= 0) & (slam_row < slam_h))

        vals = slam_data[slam_row[valid], slam_col[valid]]
        valid_rows, valid_cols = np.where(valid)

        slam_known = np.zeros((n, n), dtype=bool)
        slam_free = np.zeros((n, n), dtype=bool)
        slam_occupied = np.zeros((n, n), dtype=bool)
        known_mask = (vals != -1)
        free_mask = (vals == 0)
        occ_mask = (vals > 50)

        slam_known[valid_rows[known_mask], valid_cols[known_mask]] = True
        slam_free[valid_rows[free_mask], valid_cols[free_mask]] = True
        slam_occupied[valid_rows[occ_mask], valid_cols[occ_mask]] = True
        self.slam_known = slam_known
        self.slam_free = slam_free
        self.slam_occupied = slam_occupied

        # Merge SLAM into existing data rather than replacing it wholesale.
        # SLAM Toolbox reports three states:
        #   100 = occupied  →  our data = 1  (authoritative: override LiDAR free)
        #     0 = free      →  our data = 0  (authoritative: clear stale LiDAR hits)
        #    -1 = unknown   →  keep existing data (LiDAR hits survive in unexplored zones)
        new_data = self.data.copy()
        new_data[valid_rows[known_mask], valid_cols[known_mask]] = (
            vals[known_mask] > 50).astype(np.uint8)
        self.data = new_data
        self._refresh_frontiers_from_raw_slam()
        # SLAM is authoritative for visited: every SLAM-free cell is reachable
        # free space, replacing the LiDAR-sweep sector with the full known map.
        self.visited = self.slam_free.astype(np.uint8)

    def set_door_match_radius(self, radius_m=None, radius_cells=None):
        """
        Configure max distance for matching door nodes across neighboring rooms.
        Priority:
          1) explicit radius_m (if > 0)
          2) radius_cells * cell_size
          3) existing defaults
        """
        if radius_m is not None and float(radius_m) > 0.0:
            self.door_match_radius_m = float(radius_m)
            return

        if radius_cells is not None and float(radius_cells) > 0.0:
            self.door_match_radius_cells = float(radius_cells)
            self.door_match_radius_m = max(0.2, self.door_match_radius_cells * self.cell_size)

    def has_slam_semantics(self):
        return bool(np.any(self.slam_known))

    def has_raw_slam_frontiers(self):
        return self.slam_frontier_points.shape[0] > 0

    def _refresh_frontiers_from_raw_slam(self):
        if self.slam_raw_data is None or self.slam_raw_resolution is None or self.slam_raw_origin is None:
            self.slam_frontier_points = np.zeros((0, 2), dtype=np.float32)
            return

        slam_data = self.slam_raw_data
        slam_free = slam_data == 0
        slam_unknown = slam_data == -1
        slam_occupied = slam_data > 50

        frontier = np.zeros_like(slam_free, dtype=bool)
        frontier[:-1, :] |= slam_free[:-1, :] & slam_unknown[1:, :]
        frontier[1:, :] |= slam_free[1:, :] & slam_unknown[:-1, :]
        frontier[:, :-1] |= slam_free[:, :-1] & slam_unknown[:, 1:]
        frontier[:, 1:] |= slam_free[:, 1:] & slam_unknown[:, :-1]

        # Keep frontier cells off mapped walls in the native SLAM raster before
        # projecting them to the coarser local grid.
        # Use robot safety radius (0.40 m) to compute erosion in SLAM-map pixels.
        _robot_safety_m = 0.40
        _slam_r = max(2, int(np.ceil(_robot_safety_m / self.slam_raw_resolution)))
        occ_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * _slam_r + 1, 2 * _slam_r + 1))
        occ_dilated = cv2.dilate(slam_occupied.astype(np.uint8), occ_kernel, iterations=1) > 0
        frontier &= ~occ_dilated

        fy, fx = np.where(frontier)
        if len(fx) == 0:
            self.slam_frontier_points = np.zeros((0, 2), dtype=np.float32)
            return

        world_x = self.slam_raw_origin[0] + (fx.astype(np.float32) + 0.5) * self.slam_raw_resolution
        world_y = self.slam_raw_origin[1] + (fy.astype(np.float32) + 0.5) * self.slam_raw_resolution
        pts = np.stack([world_x, world_y], axis=1)

        in_bounds = (
            (pts[:, 0] >= self.origin[0] - self.map_size / 2.0) &
            (pts[:, 0] <= self.origin[0] + self.map_size / 2.0) &
            (pts[:, 1] >= self.origin[1] - self.map_size / 2.0) &
            (pts[:, 1] <= self.origin[1] + self.map_size / 2.0)
        )
        pts = pts[in_bounds]
        self.slam_frontier_points = pts.astype(np.float32, copy=False)

        frontier_grid = np.zeros_like(self.frontier)
        if len(pts) == 0:
            self.frontier = frontier_grid
            return

        gx = ((pts[:, 0] - (self.origin[0] - self.map_size / 2.0)) / self.cell_size).astype(np.int32)
        gy = ((pts[:, 1] - (self.origin[1] - self.map_size / 2.0)) / self.cell_size).astype(np.int32)
        valid = (
            (gx >= 0) & (gx < self.n_cells) &
            (gy >= 0) & (gy < self.n_cells)
        )
        frontier_grid[gy[valid], gx[valid]] = 1
        frontier_grid[self.slam_occupied] = 0
        self.frontier = frontier_grid.astype(np.uint8)

    def _world_to_grid(self, x, y):
        """
        Convert world coordinates to grid coordinates.
        :param x: x-coordinate in world space (meters)
        :param y: y-coordinate in world space (meters)
        :return: (i, j) grid coordinates
        """
        # Map goes from [origin - map_size/2] to [origin + map_size/2]
        x_rel = x - (self.origin[0] - self.map_size / 2)
        y_rel = y - (self.origin[1] - self.map_size / 2)

        j = int(x_rel / self.cell_size)
        i = int(y_rel / self.cell_size)

        # Clamp to valid indices
        i = max(0, min(i, self.n_cells - 1))
        j = max(0, min(j, self.n_cells - 1))
        return i, j

    def _grid_to_world(self, i, j):
        x = (j + 0.5) * self.cell_size + (self.origin[0] - self.map_size / 2)
        y = (i + 0.5) * self.cell_size + (self.origin[1] - self.map_size / 2)
        return x, y

    def discover_regions(self, value):
        regs = []
        i = 0
        # Accept both numpy scalars and plain Python ints
        if hasattr(value, 'astype'):
            value = int(value.astype(np.int64))
        else:
            value = int(value)
        while value != 0:
            if value & 1 != 0:
                regs.append(i)
            value = value >> 1
            i += 1
        return regs


    def room_is_confirmed(self, room: Room):
        room_bounds = skimage.segmentation.find_boundaries(room.mask, mode='outer')
        room_bounds[self.data == 1] = 0
        room_bounds = room_bounds.astype(np.uint8)
        if room.mask.sum() < 200: #40: All these values are for real-scale environments
            if self.frontier[room_bounds == 1].sum() > 15: #5:
                return False
            return True
        if self.frontier[room_bounds == 1].sum() > 80: #15:
            return False
        if self.frontier[room.mask].sum() > 50: #10:
            return False
        search = np.zeros_like(room_bounds)
        search[(self.room_seg == self.outer_label) & (room_bounds == 1)] = 1
        threshold = cv2.threshold(search, 0.5, 1, cv2.THRESH_BINARY)[1]
        n_cc, door_labels = cv2.connectedComponents(threshold)
        for door_label in np.unique(door_labels):
            if door_label == 0:
                continue
            if (door_labels == door_label).sum() >= 20: #20:
                return False
        return True

    def add_confirmed_room(self, room: Room):
        m = np.unique(self.confirmed_room_map[room.mask])
        if len(m) != 1:
            return None   # overlaps existing confirmed room — skip silently
        if m[0] != 0:
            return None   # already confirmed
        # Size sanity check — reject tiny fragments
        if room.mask.sum() < 100:   # < 1 m² at 0.1m/cell
            return None
        confirmed_room = ConfirmedRoom(room.mask)
        self.room_connectivity_graph.add_node(confirmed_room.id)
        self.confirmed_rooms[confirmed_room.id] = confirmed_room
        self.confirmed_room_map[confirmed_room.mask] = confirmed_room.id
        return confirmed_room

    def get_closest_door(self, room: ConfirmedRoom, pos):
        min_dist = 999
        min_door = None
        for door in room.doors:
            dist = np.linalg.norm(door.pos - pos)
            if dist < min_dist:
                min_door = door
                min_dist = dist
        if min_dist < self.door_match_radius_m:
            return min_door
        return None

    def compute_doors_in_room(self, regions: Dict[int, Region]):
        edges = []
        for room in list(self.confirmed_rooms.values()) + list(self.unconfirmed_rooms.values()):
            for door_pair in combinations(room.doors, 2):
                weight = np.linalg.norm(door_pair[0].pos - door_pair[1].pos)
                
                region = room.region
                u = door_pair[0].id
                v = door_pair[1].id
                if not self.room_graph.has_edge(u, v):
                    edges.append((door_pair[0].id, door_pair[1].id, {'weight': weight, 'region': region.version}))
                else:
                    self.room_graph.edges[(u, v)]['region'] = region.version
        for edge in edges:
            self.room_graph.add_edge(edge[0], edge[1], **edge[2])

    def _wall_is_linear(self, wall_cells_i, wall_cells_j, max_deviation_cells=1.5):
        """Check if a set of wall cells form a roughly straight line.
        Round objects produce curved walls; real doorway flanks are straight.
        Returns True if the wall cells fit a line with residual <= max_deviation_cells."""
        if len(wall_cells_i) < 3:
            return True  # too few points to judge
        pts = np.column_stack([wall_cells_j.astype(float), wall_cells_i.astype(float)])
        # PCA: fit line via eigenvector of covariance
        centroid = pts.mean(axis=0)
        centered = pts - centroid
        cov = centered.T @ centered
        eigvals, eigvecs = np.linalg.eigh(cov)
        # Minor axis eigenvector = direction of max spread perpendicular to line
        minor_axis = eigvecs[:, 0]  # smallest eigenvalue
        residuals = np.abs(centered @ minor_axis)
        return residuals.max() <= max_deviation_cells

    def _collect_wall_strip(self, start_i, start_j, direction, length, search_width=2):
        """Collect wall cells extending from (start_i, start_j) in a given
        direction for up to `length` cells.  `direction` is (di, dj).
        Returns arrays of (wall_i, wall_j) found."""
        n = self.n_cells
        di, dj = direction
        found_i, found_j = [], []
        for step in range(1, length + 1):
            ci = start_i + di * step
            cj = start_j + dj * step
            # Search perpendicular to direction within search_width
            hit = False
            for w in range(-search_width, search_width + 1):
                # Perpendicular offset: if moving in i, offset in j; vice versa
                wi = ci + (0 if di != 0 else w)
                wj = cj + (w if di != 0 else 0)
                if 0 <= wi < n and 0 <= wj < n and self.data[wi, wj] == 1:
                    found_i.append(wi)
                    found_j.append(wj)
                    hit = True
            if not hit:
                break  # wall ended
        return np.array(found_i, dtype=int), np.array(found_j, dtype=int)

    def _gap_has_flanking_walls(self, door_idx_i, door_idx_j, min_wall_len_cells=5):
        """Check that BOTH sides of a gap have STRAIGHT wall cells extending at
        least min_wall_len_cells.  Rejects curved walls from round obstacles."""
        if len(door_idx_i) == 0:
            return False
        i_span = door_idx_i.max() - door_idx_i.min()
        j_span = door_idx_j.max() - door_idx_j.min()
        n = self.n_cells
        search_len = min_wall_len_cells + 3

        if j_span >= i_span:
            # Gap runs horizontally (span in j) — flanking walls extend in -j and +j
            i_c = int(np.median(door_idx_i))
            j_lo, j_hi = int(door_idx_j.min()), int(door_idx_j.max())

            # Left flank: wall extending in -j direction from gap left edge
            left_i, left_j = self._collect_wall_strip(i_c, j_lo, (0, -1), search_len)
            if len(left_i) < min_wall_len_cells:
                return False
            if not self._wall_is_linear(left_i, left_j):
                print(f"[DOOR REJECT] left flank is curved (round object)")
                return False

            # Right flank: wall extending in +j direction from gap right edge
            right_i, right_j = self._collect_wall_strip(i_c, j_hi, (0, 1), search_len)
            if len(right_i) < min_wall_len_cells:
                return False
            if not self._wall_is_linear(right_i, right_j):
                print(f"[DOOR REJECT] right flank is curved (round object)")
                return False

            return True
        else:
            # Gap runs vertically (span in i) — flanking walls extend in -i and +i
            j_c = int(np.median(door_idx_j))
            i_lo, i_hi = int(door_idx_i.min()), int(door_idx_i.max())

            # Top flank: wall extending in -i from gap top
            top_i, top_j = self._collect_wall_strip(i_lo, j_c, (-1, 0), search_len)
            if len(top_i) < min_wall_len_cells:
                return False
            if not self._wall_is_linear(top_i, top_j):
                print(f"[DOOR REJECT] top flank is curved (round object)")
                return False

            # Bottom flank: wall extending in +i from gap bottom
            bot_i, bot_j = self._collect_wall_strip(i_hi, j_c, (1, 0), search_len)
            if len(bot_i) < min_wall_len_cells:
                return False
            if not self._wall_is_linear(bot_i, bot_j):
                print(f"[DOOR REJECT] bottom flank is curved (round object)")
                return False

            return True

    def add_doors(self, room, region, regions):
        room_map = np.zeros_like(self.data, dtype=np.int32)
        for confirmed_room in self.confirmed_rooms.values():
            room_map[confirmed_room.mask] = confirmed_room.id
        for unconfirmed_room in self.unconfirmed_rooms.values():
            room_map[unconfirmed_room.mask] = unconfirmed_room.id

        room_bounds = skimage.segmentation.find_boundaries(room.mask, mode='outer')
        room_bounds[self.data == 1] = 0
        room_bounds = room_bounds.astype(np.uint8)

        # ── Only keep boundary pixels that touch outer_label (true gaps) ──
        # A real door gap is where room boundary meets unassigned/outer space
        # NOT where it touches another already-segmented room
        gap_mask = np.zeros_like(room_bounds, dtype=np.uint8)
        gap_mask[(room_bounds == 1) & (self.room_seg == self.outer_label)] = 1
        # Also include gaps touching visited-but-unsegmented cells
        gap_mask[(room_bounds == 1) & (self.visited == 1) & (self.confirmed_room_map == 0)] = 1

        threshold = cv2.threshold(gap_mask, 0.5, 1, cv2.THRESH_BINARY)[1]
        n_cc, door_labels = cv2.connectedComponents(threshold)

        # Minimum wall length on each side of gap (0.5m at current cell_size)
        min_wall_cells = max(3, int(0.5 / self.cell_size))
        # Door width limits in cells
        min_gap_cells = max(3, int(0.6 / self.cell_size))
        max_gap_cells = max(6, int(1.2 / self.cell_size))

        # First pass: collect candidate doors
        candidate_doors = []
        for door_label in np.unique(door_labels):
            if door_label == 0:
                continue
            door_idx_i, door_idx_j = np.where(door_labels == door_label)
            gap_cells = len(door_idx_i)

            # Width filter: 0.6m–1.2m (tightened from 3.0m to reject corridor openings)
            if gap_cells < min_gap_cells or gap_cells > max_gap_cells:
                continue

            # Flanking-wall check: both sides must have walls >= 0.5m
            if not self._gap_has_flanking_walls(door_idx_i, door_idx_j,
                                                 min_wall_len_cells=min_wall_cells):
                print(f"[DOOR REJECT] gap_cells={gap_cells} — insufficient flanking walls")
                continue

            i_c = int((door_idx_i.max() + door_idx_i.min()) // 2)
            j_c = int((door_idx_j.max() + door_idx_j.min()) // 2)
            door_center = np.array(self._grid_to_world(i_c, j_c))

            candidate_doors.append((door_label, door_idx_i, door_idx_j,
                                     gap_cells, door_center))

        # NMS: if two candidate doors are within 1.5m, keep only the one
        # with the most gap cells (more confident detection).
        nms_radius = 1.5
        keep = [True] * len(candidate_doors)
        for a in range(len(candidate_doors)):
            if not keep[a]:
                continue
            for b in range(a + 1, len(candidate_doors)):
                if not keep[b]:
                    continue
                dist = np.linalg.norm(candidate_doors[a][4] - candidate_doors[b][4])
                if dist < nms_radius:
                    # Suppress the smaller detection
                    if candidate_doors[a][3] >= candidate_doors[b][3]:
                        keep[b] = False
                    else:
                        keep[a] = False
                        break

        nodes = []
        edges = []
        for idx, (door_label, door_idx_i, door_idx_j, gap_cells, door_center) in enumerate(candidate_doors):
            if not keep[idx]:
                print(f"[DOOR NMS] suppressed door at ({door_center[0]:.2f},{door_center[1]:.2f})")
                continue

            print(f"[DOOR DEBUG] world=({door_center[0]:.2f},{door_center[1]:.2f})"
                  f" gap_cells={gap_cells}")

            door_id = self.n_doors
            self.n_doors += 1
            room_in_doors = np.unique(room_map[door_labels == door_label])
            room_in_doors = [r for r in room_in_doors if r not in [0, room.id]]

            nodes.append((door_id, {'room': room.id, 'pos': door_center}))
            door = Door(door_id, door_center, room)
            self.doors.add(door)
            room.doors.append(door)

            for r in room_in_doors:
                if r == 0:
                    continue
                if r in self.confirmed_rooms:
                    other_room = self.confirmed_rooms[r]
                elif r in self.unconfirmed_rooms:
                    other_room = self.unconfirmed_rooms[r]
                else:
                    continue
                closest_door = self.get_closest_door(other_room, door_center)
                if closest_door is None:
                    continue
                edges.append((door_id, closest_door.id, {
                    'weight': np.linalg.norm(door_center - closest_door.pos),
                    'region': -1
                }))

        new_node_ids = [x[0] for x in nodes]
        for door_pair in combinations(room.doors, 2):
            weight = np.linalg.norm(door_pair[0].pos - door_pair[1].pos)
            for reg in regions.values():
                if reg.in_bound(door_pair[0].pos) and reg.in_bound(door_pair[1].pos):
                    edges.append((door_pair[0].id, door_pair[1].id, {
                        'weight': weight, 'region': reg.version
                    }))
                    break

        self.room_graph.add_nodes_from(nodes)
        for edge in edges:
            self.room_graph.add_edge(edge[0], edge[1], **edge[2])
    

    def find_confirmed_room(self, room: Room):
        # returns the ConfirmedRoom object if room is a previously confirmed room
        for cr in self.confirmed_rooms.values():
            if (room.mask == cr.mask).all():
                return cr
        return None

    def update_rooms(self):
        self.rooms = []
        for label in np.unique(self.room_seg):
            if label == 0:
                continue
            if label == self.outer_label:
                continue
            room = Room(self.room_seg == label)
            confirmed_room = self.find_confirmed_room(room)
            self.rooms.append((room, confirmed_room))

    def update_unconfirmed_rooms(self):
        # coalesce unconfirmed rooms
        clusters: List[List[UnconfirmedRoom]] = []
        for room in self.unconfirmed_rooms.values():
            room_b_mask = self.get_boundary_mask(room)
            found_cluster = False
            for cluster in clusters:
                belongs_to_cluster = False
                for other_room in cluster:
                    other_room_b_mask = self.get_boundary_mask(other_room)
                    if (room_b_mask & other_room_b_mask).sum() > 0:
                        belongs_to_cluster = True
                        break
                if belongs_to_cluster:
                    found_cluster = True
                    cluster.append(room)
                    break
            if not found_cluster:
                clusters.append([room])
        for cluster in clusters:
            for room in cluster:
                del self.unconfirmed_rooms[room.id]
        for cluster in clusters:
            new_mask = cluster[0].mask
            for room in cluster:
                new_mask = new_mask | room.mask
            new_unconfirmed_room = UnconfirmedRoom(new_mask)
            self.unconfirmed_rooms[new_unconfirmed_room.id] = new_unconfirmed_room
            doors = [door for room in cluster for door in room.doors]
            for door in doors:
                door.room = new_unconfirmed_room
                if door.id in self.room_graph.nodes:
                    self.room_graph.nodes[door.id]['room'] = new_unconfirmed_room.id
            new_unconfirmed_room.doors = doors
        
        # split large unconfirmed rooms:
        for room in list(self.unconfirmed_rooms.values()):
            self.split_room(room)
    
    def split_from_point(self, room_mask: np.ndarray, point, split_axis, max_split_len_x=20, max_split_len_y=25, min_split_size=400): #10, 15, 400
        # point (i, j)
        split_mask = room_mask.copy().astype(np.uint8)
        split_line_mask = np.zeros_like(room_mask)
        n = split_mask.shape[0]
        split_len = 0
        if split_axis == 0:
            # split horizontally
            i = point[0]
            for j in range(max_split_len_x):
                j = point[1] - j
                if j < 0:
                    break
                if self.visited[i, j] != 1:
                    break
                if self.data[i, j] == 1:
                    break
                split_mask[i, j] = 0
                split_line_mask[i, j] = 1
                split_len += 1
            for j in range(max_split_len_x):
                j = point[1] + j
                if j >= n:
                    break
                if self.visited[i, j] != 1:
                    break
                if self.data[i, j] == 1:
                    break
                split_mask[i, j] = 0
                split_line_mask[i, j] = 1
                split_len += 1
        else:
            j = point[1]
            for i in range(max_split_len_y):
                i = point[0] - i
                if i < 0:
                    break
                if self.visited[i, j] != 1:
                    break
                if self.data[i, j] == 1:
                    break
                split_mask[i, j] = 0
                split_line_mask[i, j] = 1
                split_len += 1
            for i in range(max_split_len_y):
                i = point[0] + i
                if i >= n:
                    break
                if self.visited[i, j] != 1:
                    break
                if self.data[i, j] == 1:
                    break
                split_mask[i, j] = 0
                split_line_mask[i, j] = 1
                split_len += 1
        ccs = connected_components(split_mask)
        ccs[split_line_mask == 1] = 1
        labels = np.unique(ccs)
        if len(labels) < 3: # 0, 1, _
            return None, None
        
        for label in labels:
            if label == 0:
                continue
            if (ccs == label).sum() < min_split_size:
                return None, None
        
        return ccs, split_len
    
    def split_room(self, room: Room):
        room_mask = room.mask
        mask = room_mask.astype(np.uint8)
        skel = skimage.morphology.skeletonize(mask)
        
        n = skel.shape[0]
        skel_idxs = np.stack(np.where(skel), axis=-1)
        ccss = []
        for idx in skel_idxs:
            if idx[0] > 0 and idx[0] < n:
                if skel[idx[0] - 1, idx[1]] == 1 and skel[idx[0] + 1, idx[1]] == 1:
                    ccs, split_len = self.split_from_point(room_mask, idx, 0)
                    if ccs is not None:
                        ccss.append((split_len, ccs))
            if idx[1] > 0 and idx[1] < n:
                if skel[idx[0], idx[1] - 1] == 1 and skel[idx[0], idx[1] + 1] == 1:
                    ccs, split_len = self.split_from_point(room_mask, idx, 1)
                    if ccs is not None:
                        ccss.append((split_len, ccs))
        ccs = None
        if len(ccss) > 0:
            ccs = min(ccss, key=lambda x: x[0])[1]
        if ccs is not None:
            self.remove_room(room)
            labels = np.unique(ccs)
            for label in labels:
                if label == 0:
                    continue
                room = Room(ccs == label)
                if self.room_is_confirmed(room):
                    new_room = self.add_confirmed_room(room)
                    if new_room is not None:
                        print('found newly confirmed room')
                        self.new_confirmed_rooms.append(new_room)

                        if self.regions:  # self.regions is passed in from gridmap_node - new diff for room region attachment
                            ri = int(np.mean(np.where(new_room.mask)[0]))
                            rj = int(np.mean(np.where(new_room.mask)[1]))
                            rx, ry = self._grid_to_world(ri, rj)
                            for region in self.regions.values():
                                b = region.bound.numpy()
                                if (b[0,0] <= rx <= b[1,0] and b[0,1] <= ry <= b[1,1]):
                                    region.add_room(new_room)  # sets new_room.region = region
                                    break

                        self.add_doors(new_room, None, self.regions)
                else:
                    unconfirmed_room = UnconfirmedRoom(ccs == label)
                    self.unconfirmed_rooms[unconfirmed_room.id] = unconfirmed_room
                    #self.add_doors(unconfirmed_room, None, self.regions)
        
    def remove_room(self, room: Union[ConfirmedRoom, UnconfirmedRoom]):
        if room.id in self.confirmed_rooms:
            return
        elif room.id in self.unconfirmed_rooms:
            del self.unconfirmed_rooms[room.id]
        for door in room.doors:
            self.doors.remove(door)
            if door.id in self.room_graph.nodes:
                self.room_graph.remove_node(door.id)
          
    def clear_unconfirmed_rooms(self):
        for r in self.unconfirmed_rooms.values():
            for door in r.doors:
                self.doors.discard(door)   # discard won't raise if missing
                if door.id in self.room_graph.nodes:
                    self.room_graph.remove_node(door.id)
            if r.id in self.room_connectivity_graph.nodes:
                self.room_connectivity_graph.remove_node(r.id)
        self.unconfirmed_rooms = {}

    def update(self, lidar_scan, points, speeds, cur_pos, region: Region, regions: Dict[int, Region], update_room=True, update_occupancy=True):
        """
        Update the gridmap with a lidar scan.
        :param lidar_scan: A numpy array of shape (n, 3), where each row represents (x, y, intensity).
        :param update_occupancy: If False, skip LiDAR-based occupancy accumulation (used when
            occupancy is driven externally, e.g. from SLAM Toolbox via ingest_slam_map()).
        """
        def bresenham_line(x0, y0, x1, y1):
            """Generate grid cells along the line from (x0, y0) to (x1, y1) using Bresenham's algorithm."""
            cells = []
            dx = abs(x1 - x0)
            dy = abs(y1 - y0)
            sx = 1 if x0 < x1 else -1
            sy = 1 if y0 < y1 else -1
            err = dx - dy

            while True:
                cells.append((x0, y0))
                if x0 == x1 and y0 == y1:
                    break
                e2 = 2 * err
                if e2 > -dy:
                    err -= dy
                    x0 += sx
                if e2 < dx:
                    err += dx
                    y0 += sy
            return cells

        def is_blocked(start, end):
            """Check if there is an occupied grid between start and end points in the occupancy map."""
            for x, y in bresenham_line(start[0], start[1], end[0], end[1]):
                if self.data[x, y] == 1:  # Assuming 1 represents occupied space
                    return True
            return False

        n_angles = 90
        angle_bins = np.ones(n_angles) * 999
        angle_to_bin = lambda a: int((a + np.pi) / (2 * np.pi) * (n_angles - 1))
        # Always populate angle_bins from the current LiDAR scan — the frontier
        # expansion loop uses this to know which angular directions are blocked.
        # Only accumulate occupancy hits when update_occupancy=True.
        if update_occupancy:
            occ = np.zeros(self.data.shape, dtype=np.int64)
        robot_gi, robot_gj = self._world_to_grid(cur_pos[0], cur_pos[1])
        for point in lidar_scan:
            x, y = point
            angle = np.arctan2(y - cur_pos[1], x - cur_pos[0])
            dist = np.sqrt((x - cur_pos[0]) ** 2 + (y - cur_pos[1]) ** 2)
            bin = angle_to_bin(angle)
            angle_bins[bin] = min(angle_bins[bin], dist)

            if update_occupancy:
                hit_i, hit_j = self._world_to_grid(x, y)
                if 0 <= hit_i < self.n_cells and 0 <= hit_j < self.n_cells:
                    occ[hit_i, hit_j] += 1
                # Ray-cast: clear stale occupied cells along ray BEFORE hit
                ray_cells = bresenham_line(robot_gi, robot_gj, hit_i, hit_j)
                # All cells except the last (the hit endpoint) are free
                for ri, rj in ray_cells[:-1]:
                    if 0 <= ri < self.n_cells and 0 <= rj < self.n_cells:
                        if self.data[ri, rj] == 1:
                            self.data[ri, rj] = 0
        if update_occupancy:
            self.data[occ > 5] = 1
        for wall_idx in np.stack(np.where(self.data == 1), -1):
            x, y = self._grid_to_world(wall_idx[0], wall_idx[1])
            angle = np.arctan2(y - cur_pos[1], x - cur_pos[0])
            dist = np.sqrt((x - cur_pos[0]) ** 2 + (y - cur_pos[1]) ** 2)
            bin = angle_to_bin(angle)
            angle_bins[bin] = min(angle_bins[bin], dist)

        cur_pos_idx = self._world_to_grid(cur_pos[0], cur_pos[1])
        min_speed = 0.06
        cur_visited = np.zeros_like(self.visited)
        prev_frontier = self.frontier.copy()
        near_idx = np.linalg.norm(points - cur_pos, axis=1) <= self.near_dist
        near_points = points[near_idx]
        near_speeds = speeds[near_idx]
        visited_count = 0
        out_of_bounds = 0
        not_traversable = 0

        for point, speed in zip(near_points, near_speeds):
            x, y = point
            i, j = self._world_to_grid(x, y)
            if not (0 <= i < self.n_cells and 0 <= j < self.n_cells):
                out_of_bounds += 1
                continue

            if self.trav_map[i, j] != 1:
                not_traversable += 1
                continue

            self.visited[i, j] = 1
            cur_visited[i, j] = 1
            self.frontier[i, j] = 0
            visited_count += 1

        near_mask = np.linalg.norm(points - cur_pos, axis=1) > self.near_dist
        far_mask = np.linalg.norm(points - cur_pos, axis=1) <= self.near_dist + 0.05
        far_idx = near_mask & far_mask
        far_points = points[far_idx]
        far_speeds = speeds[far_idx]
        for point, speed in zip(far_points, far_speeds):
            x, y = point
            angle = np.arctan2(y - cur_pos[1], x - cur_pos[0])
            dist = np.sqrt((x - cur_pos[0]) ** 2 + (y - cur_pos[1]) ** 2)
            bin = angle_to_bin(angle)
            if angle_bins[bin] <= dist:
                continue
            if speed <= min_speed:
                continue
            i, j = self._world_to_grid(x, y)

            if 0 <= i < self.n_cells and 0 <= j < self.n_cells:
                if self.trav_map[i, j] == 1 and self.visited[i, j] == 0: #and not is_blocked(cur_pos_idx, (i, j)):
                    self.frontier[i, j] = 1
        
        old_dm = self.discovery_map.copy()
        self.visited[self.trav_map == 0] = 0

        if self.has_slam_semantics():
            self._refresh_frontiers_from_raw_slam()
            # Override LiDAR-sector visited with SLAM free space
            self.visited = self.slam_free.astype(np.uint8)
        else:
            unexplored_map = self.trav_map.copy()
            unexplored_map[self.visited == 1] = 0
            unexplored_map[self.data == 1] = 0

            # Only re-add boundary frontiers for cells NOT visited this tick
            # (avoids re-creating frontiers that raycasting just cleared)
            boundary_mask = skimage.segmentation.find_boundaries(self.visited, mode='inner') == 1
            boundary_mask &= (cur_visited == 0)
            self.frontier[boundary_mask] = 1
            self.frontier[self.data == 1] = 0
            erosion_pixels = 5  # 5 cells = 0.5m — robot radius (0.22m) + rotation safety margin
            erosion_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (2 * erosion_pixels + 1, 2 * erosion_pixels + 1)
            )
            occupied_dilated = cv2.dilate(
                self.data.astype(np.uint8),
                erosion_kernel,
                iterations=1
            )
            self.frontier[occupied_dilated == 1] = 0
            self.frontier[cv2.erode(1 - self.data, np.ones((5, 5)), iterations=1) == 0] = 0
            self.frontier = get_neighbor_at_least(unexplored_map, self.frontier, 1)
        if not update_room:
            return

        self._run_room_segmentation(cur_pos, region, regions)

    def _run_room_segmentation(self, cur_pos, region, regions):
        """Run room segmentation only — safe to call from a background thread.
        Reads self.data / self.visited / self.confirmed_room_map (written by the
        fast path of update()); writes self.room_seg / self.rooms /
        self.new_confirmed_rooms / self.unconfirmed_rooms.
        Slight staleness of the occupancy arrays is acceptable at 1 Hz cadence.
        """
        cur_pos_idx = self._world_to_grid(cur_pos[0], cur_pos[1])

        kernel_size = int(self.room_seg_config[0])   # = 7
        iterations  = int(self.room_seg_config[1])   # = 2      # = 7
        eroded = cv2.erode((1 - self.data).astype(np.uint8),np.ones((kernel_size, kernel_size), dtype=np.uint8),iterations=self.room_seg_config[1])
        #eroded = cv2.erode(1 - self.data, np.ones(self.room_seg_config[0]), iterations=self.room_seg_config[1])
        eroded[self.confirmed_room_map != 0] = 0
        
        threshold = cv2.threshold(eroded, 0.5, 1, cv2.THRESH_BINARY)[1]
        n_cc, labels = cv2.connectedComponents(threshold)
        for _ in range(100):
            labels = skimage.segmentation.expand_labels(labels, distance=1)
            labels[self.data == 1] = 0
            labels[self.confirmed_room_map != 0] = 0
        self.outer_label = labels[0, 0]
        labels[self.visited == 0] = 0
        unconfirmed_space = np.zeros(labels.shape, dtype=np.uint8)
        unconfirmed_space[labels == self.outer_label] = 1
        unconfirmed_space[(labels == 0) & (self.visited == 1) & (self.confirmed_room_map == 0) & (self.data == 0)] = 1
        # breakpoint()
        threshold2 = cv2.threshold(unconfirmed_space, 0.5, 1, cv2.THRESH_BINARY)[1]
        _, labels2 = cv2.connectedComponents(threshold2)
        for label2 in np.unique(labels2):
            if (labels2 == label2).sum() < 10:
                labels2[labels2 == label2] = -1
        # breakpoint()
        add_mask = (unconfirmed_space != 0) & (labels2 != -1)
        labels[add_mask] = labels2[add_mask] + self.unconfirmed_label

        self.room_seg = labels
        unique_labels = np.unique(labels)

        self.new_confirmed_rooms = []
        self.clear_unconfirmed_rooms()
        all_unconfirmed_rooms = {}
        unconfirmed_room_connectivity = nx.Graph()
        for label in unique_labels:
            if label == self.outer_label:
                continue
            if label == 0:
                continue
            # clean up room
            ccs = connected_components(labels == label)
            cc_masks = [ccs == cc for cc in np.unique(ccs) if np.unique(labels[ccs == cc])[0] == label]
            cc = max(cc_masks, key=lambda m: m.sum())
            room = Room(cc)
            
            if self.room_is_confirmed(room):
                new_room = self.add_confirmed_room(room)
                if new_room is not None:
                    print('found newly confirmed room')
                    # ✅ DEBUG: check the robot's grid cell + confirmed room map value
                    i, j = cur_pos_idx  # already computed earlier in update()
                    print("Robot grid:", i, j)
                    print("Room map value:", self.confirmed_room_map[i, j])
                    # ✅ DEBUG: sanity check that the mask is stamped with this room id
                    vals = np.unique(self.confirmed_room_map[new_room.mask])
                    print("Unique confirmed_room_map values inside new_room.mask:", vals)
                    # ── Assign room to owning region ──────────────────────────────
                    mask_ij = np.stack(np.where(new_room.mask), axis=1)  # (N, 2)
                    centroid_i = int(mask_ij[:, 0].mean())
                    centroid_j = int(mask_ij[:, 1].mean())
                    cx, cy = self._grid_to_world(centroid_i, centroid_j)
                    for region in self.regions.values():
                        b = region.bound.numpy()
                        if b[0,0] <= cx <= b[1,0] and b[0,1] <= cy <= b[1,1]:
                            region.add_room(new_room)   # sets new_room.region = region
                            break
                    # ─────────────────────────────────────────────────────────────
                    self.new_confirmed_rooms.append(new_room)
                    self.add_doors(new_room, region, regions)
            else:
                # all_unconfirmed_rooms[label] = cc
                unconfirmed_room = UnconfirmedRoom(cc) #parent_id = None was added.
                self.unconfirmed_rooms[unconfirmed_room.id] = unconfirmed_room
                #self.add_doors(unconfirmed_room, region, regions)
        
        self.update_unconfirmed_rooms()
        self.update_rooms()
    
    def get_boundary_mask(self, room: Room, mode='thick'):
        mask = room.mask.copy()
        mask = skimage.segmentation.find_boundaries(mask.astype(np.uint8), mode=mode)
        mask[self.data == 1] = 0
        return mask
    
    def get_map_img(self, plot=True):
        im = np.ones((*self.data.shape, 3), dtype=np.uint8) * 255
        im[self.data == 1] = 0
        if plot:
            plt.imshow(im, origin='lower')
        return im

## Added here for the room segmentation + graph
    def get_rooms(self):
        """
        Returns dictionary of all rooms (confirmed + unconfirmed).
        """
        rooms = {}

        # Confirmed rooms
        for room_id, room in self.confirmed_rooms.items():
            mask = room.mask
            cells = np.stack(np.where(mask), axis=-1)

            boundary = skimage.segmentation.find_boundaries(mask, mode='outer')
            boundary[self.data == 1] = 0
            boundary_cells = np.stack(np.where(boundary), axis=-1)

            rooms[room_id] = {
                "cells": cells,
                "boundary": boundary_cells,
                "confirmed": True
            }

        # Unconfirmed rooms
        for room_id, room in self.unconfirmed_rooms.items():
            mask = room.mask
            cells = np.stack(np.where(mask), axis=-1)

            boundary = skimage.segmentation.find_boundaries(mask, mode='outer')
            boundary[self.data == 1] = 0
            boundary_cells = np.stack(np.where(boundary), axis=-1)

            rooms[room_id] = {
                "cells": cells,
                "boundary": boundary_cells,
                "confirmed": False
            }

        return rooms

    def get_room_graph(self):
        graph_dict = {room_id: [] for room_id in self.confirmed_rooms.keys()}
        for u, v, edge_data in self.room_graph.edges(data=True):
            pos_u = self.room_graph.nodes[u].get('pos')
            pos_v = self.room_graph.nodes[v].get('pos')
            room_u = self.room_graph.nodes[u].get('room', -1)
            room_v = self.room_graph.nodes[v].get('room', -1)
            if pos_u is None or pos_v is None:
                continue
            door_xy = np.array([(pos_u[0]+pos_v[0])/2,
                                 (pos_u[1]+pos_v[1])/2], dtype=np.float32)
            if room_u in graph_dict:
                graph_dict[room_u].append({'to': room_v, 'door_xy': door_xy, 'weight': 1.0})
            if room_v in graph_dict:
                graph_dict[room_v].append({'to': room_u, 'door_xy': door_xy, 'weight': 1.0})
        return graph_dict


    def get_room_at_position(self, pos):
        """
        Returns room_id at given world (x,y), or None.
        """
        i, j = self._world_to_grid(pos[0], pos[1])
        room_id = self.confirmed_room_map[i, j]
        #print("Robot grid:", i, j)
        #print("Room map value:", self.confirmed_room_map[i, j])
        if room_id != 0:
            return room_id
        return None
## Above is the new diff for room seg + graph building
    
    def save(self, path):
        save_dir = os.path.join(path, 'lidar_gridmap')
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        np.save(os.path.join(save_dir, 'occ_map.npy'), self.data)
        np.save(os.path.join(save_dir, 'visited.npy'), self.visited)
        np.save(os.path.join(save_dir, 'frontier.npy'), self.frontier)
        with open(os.path.join(save_dir, 'confirmed_rooms.json'), 'w') as f:
            json.dump({rid: room.serialize() for rid, room in self.confirmed_rooms.items()}, f)
        with open(os.path.join(save_dir, 'unconfirmed_rooms.json'), 'w') as f:
            json.dump({int(rid): room.serialize() for rid, room in self.unconfirmed_rooms.items()}, f)
        np.save(os.path.join(save_dir, 'confirmed_room_map.npy'), self.confirmed_room_map)
        nx.write_adjlist(self.room_graph, os.path.join(save_dir, 'room_graph.adjlist'))
        edge_info = {str(edge): self.room_graph.edges[edge] for edge in self.room_graph.edges}
        with open(os.path.join(save_dir, 'room_graph_edges.json'), 'w') as f:
            json.dump(edge_info, f)
        nx.write_adjlist(self.room_connectivity_graph, os.path.join(save_dir, 'room_connectivity_graph.adjlist'))
        with open(os.path.join(save_dir, 'doors.json'), 'w') as f:
            json.dump([door.serialize() for door in self.doors], f)
        np.save(os.path.join(save_dir, 'discovery_map.npy'), self.discovery_map)

    # ============ DEAD CODE (never called) ============

    # Last reviewed as dead: 2026-03-17
    def can_access(self, region_version, pos):
        i, j = self._world_to_grid(pos[0], pos[1])
        return self.discovery_map[i, j].astype(int) & (1 << region_version) != 0

    # Last reviewed as dead: 2026-03-17
    def set_access(self, region_version, mask):
        self.discovery_map[mask] |= 1 << region_version

    # Last reviewed as dead: 2026-03-17
    def get_target_version(self, pos):
        i, j = self._world_to_grid(pos[0], pos[1])
        found = False
        r = 0
        while not found:
            mi = max(i - r, 0)
            mj = max(j - r, 0)
            Mi = min(i + r + 1, self.n_cells)
            Mj = min(j + r + 1, self.n_cells)
            values, counts = np.unique(self.discovery_map[mi:Mi, mj:Mj], return_counts=True)
            versions = []
            for value in values:
                versions += self.discover_regions(value)
            if len(versions) > 0:
                return max(versions)
            r += 1
            if r >= 10:
                breakpoint()
                return None

    # Last reviewed as dead: 2026-03-17
    def get_discovery_boundary(self, pos):
        i, j = self._world_to_grid(pos[0], pos[1])
        if self.discovery_map[i, j] != 0:
            return self.discovery_map[i, j]
        r = 10
        mi = max(i - r, 0)
        mj = max(j - r, 0)
        Mi = min(i + r + 1, self.n_cells)
        Mj = min(j + r + 1, self.n_cells)
        versions, counts = np.unique(self.discovery_map[mi:Mi, mj:Mj], return_counts=True)
        sorted_idx = np.argsort(counts)
        if versions[sorted_idx[-1]] == 0:
            if len(versions) == 1:
                breakpoint()
            return versions[sorted_idx[-2]]
        return versions[sorted_idx[-1]]

    # Last reviewed as dead: 2026-03-17
    def get_room_from_mask(self, mask):
        labels, counts = np.unique(self.confirmed_room_map[mask], return_counts=True)
        if len(labels) == 1 and labels[0] == 0:
            return -1
        sorted_idx = np.argsort(counts)
        labels = labels[sorted_idx]
        if labels[-1] == 0:
            return labels[-2]
        return labels[-1]

    # Last reviewed as dead: 2026-03-17
    def in_confirmed_room(self, pos):
        i, j = self._world_to_grid(pos[0], pos[1])
        return self.confirmed_room_map[i, j] > 0

    # Last reviewed as dead: 2026-03-17
    def get_room_unexplored(self, room: Room):
        room_bounds = skimage.segmentation.find_boundaries(room.mask, mode='outer')
        room_bounds[self.data == 1] = 0
        bound_masked = self.room_seg.copy()
        bound_masked[room_bounds == 0] = 0
        if (bound_masked == self.outer_label).sum() == 0:
            return None
        return np.stack(np.where(bound_masked == self.outer_label), -1)

    # Last reviewed as dead: 2026-03-17
    def get_door_idx(self, room: Room):
        room_bounds = skimage.segmentation.find_boundaries(room.mask, mode='outer')
        room_bounds[self.data == 1] = 0
        bound_masked = self.room_seg.copy()
        bound_masked[room_bounds == 0] = 0
        if (bound_masked != 0).sum() == 0:
            return None
        return np.stack(np.where(bound_masked != 0), -1)

    # Last reviewed as dead: 2026-03-17
    def get_confirmed_room_from_id(self, id):
        for room in self.confirmed_rooms.values():
            if room.id == id:
                return room
        return None

    # Last reviewed as dead: 2026-03-17
    def is_in_room(self, point, threshold=15):
        def raycast(point, direction, max_dist=50):
            n = self.data.shape[0]
            x = np.array(point)
            dx = np.array(direction)
            for i in range(max_dist):
                if x[0] < 0 or x[0] >= n or x[1] < 0 or x[1] >= n:
                    return max_dist
                if self.data[x[0], x[1]] > 0:
                    break
                x += dx
            return np.linalg.norm(x - point)

        horizontal = [raycast(point, np.array([0, -1])), raycast(point, np.array([0, 1]))]
        vertical = [raycast(point, np.array([-1, 0])), raycast(point, np.array([1, 0]))]
        for d1 in horizontal:
            for d2 in vertical:
                if d1 < threshold and d2 < threshold * 2:
                    return True
        return False
