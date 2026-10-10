"""
Line segment extraction from 2D LiDAR scans and directed graph construction.

Implements the first two stages of Warberg et al. (2024):
  1. Extract line segments from sequential LiDAR points (cluster → split-and-merge → refit)
  2. Build directed graph G^D: connect corners, split at corners, split at doorways

Pure algorithm — no ROS dependencies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

_next_segment_id = 0


def _new_id() -> int:
    global _next_segment_id
    _next_segment_id += 1
    return _next_segment_id


@dataclass
class LineSegment:
    """A 2D line segment extracted from LiDAR."""
    p1: np.ndarray           # [x, y] endpoint 1
    p2: np.ndarray           # [x, y] endpoint 2
    normal: np.ndarray       # [nx, ny] outward-pointing unit normal
    robot_pos: np.ndarray    # [x, y] robot position when the segment was observed
    id: int = field(default_factory=_new_id)

    # --- derived helpers ---
    @property
    def midpoint(self) -> np.ndarray:
        return 0.5 * (self.p1 + self.p2)

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.p2 - self.p1))

    @property
    def direction(self) -> np.ndarray:
        d = self.p2 - self.p1
        n = np.linalg.norm(d)
        return d / n if n > 1e-9 else np.array([1.0, 0.0])

    def point_on_positive_side(self, pt: np.ndarray) -> bool:
        """True if *pt* is on the outward (normal) side of the segment."""
        return float(np.dot(pt - self.midpoint, self.normal)) > 0.0

    def distance_to_point(self, pt: np.ndarray) -> float:
        """Shortest distance from *pt* to the finite segment."""
        v = self.p2 - self.p1
        u = pt - self.p1
        t = np.clip(np.dot(u, v) / (np.dot(v, v) + 1e-12), 0.0, 1.0)
        proj = self.p1 + t * v
        return float(np.linalg.norm(pt - proj))

    def closest_endpoint_dist(self, other: "LineSegment") -> float:
        """Min distance between any pair of endpoints of self and other."""
        dists = [
            np.linalg.norm(self.p1 - other.p1),
            np.linalg.norm(self.p1 - other.p2),
            np.linalg.norm(self.p2 - other.p1),
            np.linalg.norm(self.p2 - other.p2),
        ]
        return float(min(dists))

    def angle_with(self, other: "LineSegment") -> float:
        """Acute angle (radians) between direction vectors."""
        d1 = self.direction
        d2 = other.direction
        cos_a = np.clip(abs(np.dot(d1, d2)), 0.0, 1.0)
        return float(math.acos(cos_a))


# ---------------------------------------------------------------------------
# 1.  Extract line segments from a single 2-D LiDAR scan
# ---------------------------------------------------------------------------

# Tunables
_CLUSTER_DIST_THRESH = 0.15       # m – max gap between consecutive scan points in a cluster
_MIN_CLUSTER_SIZE = 4             # minimum points to attempt line fitting
_RDP_EPSILON = 0.05               # Ramer-Douglas-Peucker tolerance (m)
_MIN_SEGMENT_LENGTH = 0.20        # discard segments shorter than this (m)
_TLS_RESIDUAL_THRESH = 0.06       # total-least-squares max residual to accept fit (m)


def _cluster_scan_points(points: np.ndarray) -> list[np.ndarray]:
    """Split sequential scan points into clusters at distance gaps."""
    if len(points) < 2:
        return [points] if len(points) else []
    diffs = np.linalg.norm(np.diff(points, axis=0), axis=1)
    splits = np.where(diffs > _CLUSTER_DIST_THRESH)[0] + 1
    return np.split(points, splits)


def _rdp(points: np.ndarray, eps: float) -> np.ndarray:
    """Ramer-Douglas-Peucker simplification.  Returns indices."""
    if len(points) <= 2:
        return np.arange(len(points))
    # distance of each point from the line (first → last)
    line_vec = points[-1] - points[0]
    line_len = np.linalg.norm(line_vec)
    if line_len < 1e-9:
        return np.array([0, len(points) - 1])
    line_dir = line_vec / line_len
    perp = np.array([-line_dir[1], line_dir[0]])
    dists = np.abs((points - points[0]) @ perp)
    idx_max = int(np.argmax(dists))
    if dists[idx_max] > eps:
        left = _rdp(points[: idx_max + 1], eps)
        right = _rdp(points[idx_max:], eps) + idx_max
        return np.unique(np.concatenate([left, right]))
    return np.array([0, len(points) - 1])


def _fit_line_tls(points: np.ndarray):
    """Total-least-squares line fit. Returns (direction, centroid, max_residual)."""
    centroid = points.mean(axis=0)
    centered = points - centroid
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    direction = vt[0]
    residuals = np.abs(centered @ vt[1]) if len(vt) > 1 else np.zeros(len(points))
    return direction, centroid, float(residuals.max())


def _compute_normal(direction: np.ndarray, midpoint: np.ndarray,
                    robot_pos: np.ndarray) -> np.ndarray:
    """Inward normal: perpendicular to direction, pointing toward robot (into the room)."""
    n = np.array([-direction[1], direction[0]])
    # Normal should point toward the robot (into the room)
    if np.dot(n, robot_pos - midpoint) < 0:
        n = -n
    return n


def _segments_from_cluster(cluster: np.ndarray,
                           robot_pos: np.ndarray) -> list[LineSegment]:
    """Extract line segments from one contiguous cluster via split-and-merge."""
    if len(cluster) < _MIN_CLUSTER_SIZE:
        return []
    key_idx = _rdp(cluster, _RDP_EPSILON)
    if len(key_idx) < 2:
        return []
    segments: list[LineSegment] = []
    for i in range(len(key_idx) - 1):
        sub = cluster[key_idx[i]: key_idx[i + 1] + 1]
        if len(sub) < 2:
            continue
        direction, centroid, max_res = _fit_line_tls(sub)
        if max_res > _TLS_RESIDUAL_THRESH:
            # poor fit — skip (could recurse, but RDP already split)
            continue
        # project endpoints onto fitted line
        projections = (sub - centroid) @ direction
        t_min, t_max = projections.min(), projections.max()
        p1 = centroid + t_min * direction
        p2 = centroid + t_max * direction
        if np.linalg.norm(p2 - p1) < _MIN_SEGMENT_LENGTH:
            continue
        normal = _compute_normal(direction, 0.5 * (p1 + p2), robot_pos)
        segments.append(LineSegment(p1=p1, p2=p2, normal=normal,
                                    robot_pos=robot_pos.copy()))
    return segments


def extract_segments(points_xy: np.ndarray,
                     robot_pos: np.ndarray) -> list[LineSegment]:
    """
    Extract line segments from a single 2-D LiDAR frame.

    Parameters
    ----------
    points_xy : (N, 2) array – LiDAR hit points in world frame.
    robot_pos : (2,) array – robot [x, y] in world frame.

    Returns
    -------
    List of LineSegment (may be empty if scan is too sparse).
    """
    points_xy = np.asarray(points_xy, dtype=np.float64)
    robot_pos = np.asarray(robot_pos, dtype=np.float64).ravel()[:2]
    if len(points_xy) < _MIN_CLUSTER_SIZE:
        return []
    clusters = _cluster_scan_points(points_xy)
    all_segs: list[LineSegment] = []
    for cl in clusters:
        all_segs.extend(_segments_from_cluster(cl, robot_pos))
    return all_segs


# ---------------------------------------------------------------------------
# 2.  Merge segments across scans  (global accumulator)
# ---------------------------------------------------------------------------

_MERGE_ANGLE_THRESH = math.radians(8.0)   # max angle difference for collinear
_MERGE_GAP_THRESH = 0.25                   # max gap between segment endpoints (m)
_MERGE_LATERAL_THRESH = 0.10               # max perpendicular offset (m)


def _segments_are_collinear(a: LineSegment, b: LineSegment) -> bool:
    """Check if two segments are approximately collinear."""
    if a.angle_with(b) > _MERGE_ANGLE_THRESH:
        return False
    # check perpendicular distance of b's midpoint from a's line
    d = a.direction
    perp = np.array([-d[1], d[0]])
    lateral = abs(float(np.dot(b.midpoint - a.midpoint, perp)))
    return lateral < _MERGE_LATERAL_THRESH


def _gap_between(a: LineSegment, b: LineSegment) -> float:
    """
    Gap between two roughly-collinear segments.
    Projects both onto a's direction and returns the gap.
    """
    d = a.direction
    projs_a = [np.dot(a.p1 - a.p1, d), np.dot(a.p2 - a.p1, d)]
    projs_b = [np.dot(b.p1 - a.p1, d), np.dot(b.p2 - a.p1, d)]
    a_min, a_max = min(projs_a), max(projs_a)
    b_min, b_max = min(projs_b), max(projs_b)
    gap = max(b_min - a_max, a_min - b_max)
    return max(gap, 0.0)


def _merge_two(a: LineSegment, b: LineSegment) -> LineSegment:
    """Merge two collinear segments into one."""
    d = a.direction
    all_pts = np.array([a.p1, a.p2, b.p1, b.p2])
    projs = (all_pts - a.p1) @ d
    i_min, i_max = int(np.argmin(projs)), int(np.argmax(projs))
    p1, p2 = all_pts[i_min], all_pts[i_max]
    mid = 0.5 * (p1 + p2)
    # keep a's robot_pos (arbitrary choice — could average)
    normal = _compute_normal(d, mid, a.robot_pos)
    return LineSegment(p1=p1, p2=p2, normal=normal, robot_pos=a.robot_pos.copy())


def merge_segments(existing: list[LineSegment],
                   new_segs: list[LineSegment]) -> list[LineSegment]:
    """
    Merge *new_segs* into the global *existing* list.

    Collinear segments within *_MERGE_GAP_THRESH* are merged into one longer
    segment.  Non-collinear segments are appended.

    Returns the updated segment list (may be shorter than existing + new).
    """
    if not new_segs:
        return list(existing)
    merged = list(existing)
    for ns in new_segs:
        was_merged = False
        for i, es in enumerate(merged):
            if _segments_are_collinear(es, ns) and _gap_between(es, ns) < _MERGE_GAP_THRESH:
                merged[i] = _merge_two(es, ns)
                was_merged = True
                break
        if not was_merged:
            merged.append(ns)
    return merged


# ---------------------------------------------------------------------------
# 3.  Directed graph G^D — corners and doorway splits
# ---------------------------------------------------------------------------

# Paper parameters (Section IV)
_D_C = 0.40          # max endpoint distance to form a corner (m)
_D_MIN = 0.80        # min doorway gap (m)
_D_MAX = 3.00        # max doorway gap (m)
_PARALLEL_THRESH = math.radians(12.0)   # segments more parallel than this can't form corners


def _connect_corners(segments: list[LineSegment],
                     D_c: float) -> list[tuple[int, int, np.ndarray]]:
    """
    Find corner connections between non-parallel segment endpoints.

    Returns list of (seg_idx_a, seg_idx_b, corner_point).
    """
    corners: list[tuple[int, int, np.ndarray]] = []
    n = len(segments)
    for i in range(n):
        for j in range(i + 1, n):
            si, sj = segments[i], segments[j]
            # must be non-parallel
            if si.angle_with(sj) < _PARALLEL_THRESH:
                continue
            # check all four endpoint pairs
            pairs = [
                (si.p1, sj.p1), (si.p1, sj.p2),
                (si.p2, sj.p1), (si.p2, sj.p2),
            ]
            best_dist = float("inf")
            best_pair: Optional[tuple[np.ndarray, np.ndarray]] = None
            for pa, pb in pairs:
                d = float(np.linalg.norm(pa - pb))
                if d < best_dist:
                    best_dist = d
                    best_pair = (pa, pb)
            if best_dist < D_c and best_pair is not None:
                corner_pt = 0.5 * (best_pair[0] + best_pair[1])
                corners.append((i, j, corner_pt))
    return corners


def _split_segment_at(seg: LineSegment, split_pt: np.ndarray,
                      min_length: float = _MIN_SEGMENT_LENGTH
                      ) -> Optional[tuple[LineSegment, LineSegment]]:
    """
    Split a segment at *split_pt* (projected onto the segment line).
    Returns two new segments or None if either child is too short.
    """
    d = seg.direction
    t = float(np.dot(split_pt - seg.p1, d))
    seg_len = seg.length
    if t < min_length or (seg_len - t) < min_length:
        return None
    mid_pt = seg.p1 + t * d
    s1 = LineSegment(p1=seg.p1.copy(), p2=mid_pt.copy(),
                     normal=seg.normal.copy(), robot_pos=seg.robot_pos.copy())
    s2 = LineSegment(p1=mid_pt.copy(), p2=seg.p2.copy(),
                     normal=seg.normal.copy(), robot_pos=seg.robot_pos.copy())
    return (s1, s2)


def _find_doorway_splits(segments: list[LineSegment],
                         d_min: float, d_max: float
                         ) -> list[tuple[int, np.ndarray]]:
    """
    Find positions where a segment should be split because a non-parallel
    segment's closest endpoint is at doorway-width distance along it.

    Returns list of (segment_index, split_point).
    """
    splits: list[tuple[int, np.ndarray]] = []
    n = len(segments)
    for i in range(n):
        si = segments[i]
        for j in range(n):
            if i == j:
                continue
            sj = segments[j]
            if si.angle_with(sj) < _PARALLEL_THRESH:
                continue
            # check closest endpoint of sj to segment si
            for ep in [sj.p1, sj.p2]:
                # project ep onto si's line
                d = si.direction
                t = float(np.dot(ep - si.p1, d))
                if t < _MIN_SEGMENT_LENGTH or t > si.length - _MIN_SEGMENT_LENGTH:
                    continue
                # perpendicular distance from ep to si
                proj = si.p1 + t * d
                perp_dist = float(np.linalg.norm(ep - proj))
                if d_min <= perp_dist <= d_max:
                    splits.append((i, proj))
    return splits


def build_directed_graph(
    segments: list[LineSegment],
    D_c: float = _D_C,
    d_min: float = _D_MIN,
    d_max: float = _D_MAX,
) -> tuple[list[LineSegment], list[tuple[int, int]]]:
    """
    Process segments into directed graph G^D per Warberg et al. Section II-A.

    1. Connect corners (snap non-parallel endpoints within D_c).
    2. Split segments at corner intersection points.
    3. Split segments at doorway gaps [d_min, d_max].

    Returns
    -------
    processed_segments : list[LineSegment]
        Updated segment list (may be longer due to splits).
    edges : list[(seg_id_a, seg_id_b)]
        Directed edges representing corner connections.
    """
    # --- Step 1: connect corners ---
    corners = _connect_corners(segments, D_c)

    # Snap endpoints to corner points
    for i, j, corner_pt in corners:
        si, sj = segments[i], segments[j]
        # snap the closest endpoint of each segment to the corner
        for seg in [si, sj]:
            d1 = float(np.linalg.norm(seg.p1 - corner_pt))
            d2 = float(np.linalg.norm(seg.p2 - corner_pt))
            if d1 < d2:
                seg.p1 = corner_pt.copy()
            else:
                seg.p2 = corner_pt.copy()

    # --- Step 2: split segments at corners (interior intersections) ---
    split_queue: list[tuple[int, np.ndarray]] = []
    for i, j, corner_pt in corners:
        si, sj = segments[i], segments[j]
        # check if corner_pt is interior to either segment
        for idx, seg in [(i, si), (j, sj)]:
            d = seg.direction
            t = float(np.dot(corner_pt - seg.p1, d))
            if _MIN_SEGMENT_LENGTH < t < seg.length - _MIN_SEGMENT_LENGTH:
                split_queue.append((idx, corner_pt))

    # --- Step 3: find doorway splits ---
    door_splits = _find_doorway_splits(segments, d_min, d_max)
    split_queue.extend(door_splits)

    # --- Apply splits (process highest-index first to keep indices stable) ---
    # Group by segment index, sort split points along segment
    from collections import defaultdict
    splits_by_seg: dict[int, list[np.ndarray]] = defaultdict(list)
    for seg_idx, pt in split_queue:
        splits_by_seg[seg_idx].append(pt)

    new_segments: list[LineSegment] = []
    split_indices = set(splits_by_seg.keys())

    for idx, seg in enumerate(segments):
        if idx not in split_indices:
            new_segments.append(seg)
            continue
        # sort split points along segment direction
        pts = splits_by_seg[idx]
        d = seg.direction
        ts = [float(np.dot(pt - seg.p1, d)) for pt in pts]
        order = sorted(range(len(ts)), key=lambda k: ts[k])

        # build chain: p1 → split1 → split2 → ... → p2
        prev = seg.p1.copy()
        for k in order:
            split_pt = seg.p1 + ts[order[k]] * d  # use projected point
            if np.linalg.norm(split_pt - prev) >= _MIN_SEGMENT_LENGTH:
                child = LineSegment(p1=prev.copy(), p2=split_pt.copy(),
                                    normal=seg.normal.copy(),
                                    robot_pos=seg.robot_pos.copy())
                new_segments.append(child)
                prev = split_pt.copy()
        # last child
        if np.linalg.norm(seg.p2 - prev) >= _MIN_SEGMENT_LENGTH:
            child = LineSegment(p1=prev.copy(), p2=seg.p2.copy(),
                                normal=seg.normal.copy(),
                                robot_pos=seg.robot_pos.copy())
            new_segments.append(child)

    # --- Build edge list (corner connections by segment ID) ---
    # Re-find corners on the processed segments (IDs changed)
    edges: list[tuple[int, int]] = []
    id_map = {s.id: s for s in new_segments}
    # For each original corner, find the child segments whose endpoint matches
    for _, _, corner_pt in corners:
        touching = []
        for s in new_segments:
            if (np.linalg.norm(s.p1 - corner_pt) < D_c or
                    np.linalg.norm(s.p2 - corner_pt) < D_c):
                touching.append(s.id)
        for a in range(len(touching)):
            for b in range(a + 1, len(touching)):
                edges.append((touching[a], touching[b]))

    return new_segments, edges
