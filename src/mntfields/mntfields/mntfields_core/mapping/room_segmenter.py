"""
Visibility-graph construction and spectral clustering for room segmentation.

Implements Sections II-B, II-C, II-D of Warberg et al. (2024):
  - Build undirected weighted visibility graph G^V from processed line segments.
  - Spectral clustering on G^V to assign each segment to a room.
  - Split / merge checks (Fiedler value, cross-room ratio).
  - Doorway extraction from inter-room gaps.

Pure algorithm — no ROS dependencies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import linalg as la

from .line_segments import LineSegment


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

_next_room_id = 0


def _new_room_id() -> int:
    global _next_room_id
    _next_room_id += 1
    return _next_room_id


@dataclass
class RoomSegment:
    """A detected room: a cluster of line segments."""
    id: int = field(default_factory=_new_room_id)
    line_segments: list[LineSegment] = field(default_factory=list)
    bounding_box: np.ndarray = field(default_factory=lambda: np.zeros((2, 2)))
    centroid: np.ndarray = field(default_factory=lambda: np.zeros(2))

    def recompute_bounds(self, padding: float = 0.3) -> None:
        if not self.line_segments:
            return
        pts = np.array([s.p1 for s in self.line_segments]
                       + [s.p2 for s in self.line_segments])
        self.bounding_box = np.array([pts.min(axis=0) - padding,
                                      pts.max(axis=0) + padding])
        self.centroid = pts.mean(axis=0)

    def contains_point(self, xy: np.ndarray) -> bool:
        bb = self.bounding_box
        return bool(bb[0, 0] <= xy[0] <= bb[1, 0] and
                     bb[0, 1] <= xy[1] <= bb[1, 1])


@dataclass
class Doorway:
    """A detected doorway connecting two rooms."""
    id: int
    position: np.ndarray       # [x, y] midpoint
    width: float               # gap width in metres
    room_ids: tuple[int, int]  # the two rooms this door connects
    segment_pair: tuple[int, int] = (-1, -1)  # flanking segment IDs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _segment_segment_intersects(a_p1, a_p2, b_p1, b_p2) -> bool:
    """Check if two finite line segments intersect (2D cross-product method)."""
    def cross2(u, v):
        return u[0] * v[1] - u[1] * v[0]

    d = a_p2 - a_p1
    e = b_p2 - b_p1
    dxe = cross2(d, e)
    if abs(dxe) < 1e-12:
        return False  # parallel
    t = cross2(b_p1 - a_p1, e) / dxe
    u = cross2(b_p1 - a_p1, d) / dxe
    return 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0


def _mutual_visibility(si: LineSegment, sj: LineSegment,
                       all_segments: list[LineSegment]) -> bool:
    """
    True if si and sj are mutually visible:
      - No other segment occludes the line between their midpoints.
      - Each midpoint lies in the other's positive half-plane.
    """
    mi, mj = si.midpoint, sj.midpoint

    # positive half-plane check
    if not si.point_on_positive_side(mj):
        return False
    if not sj.point_on_positive_side(mi):
        return False

    # occlusion check: ray from mi to mj must not cross any other segment
    for sk in all_segments:
        if sk.id == si.id or sk.id == sj.id:
            continue
        if _segment_segment_intersects(mi, mj, sk.p1, sk.p2):
            return False
    return True


# ---------------------------------------------------------------------------
# Visibility graph construction  (Section II-B)
# ---------------------------------------------------------------------------

# Paper default parameters (Section IV)
_D_V = 8.0           # max distance for visibility edges (m)
_GAMMA_D = 0.02      # distance weight decay
_GAMMA_R = 0.005     # robot-position weight decay
_D_MIN = 0.80        # doorway min gap (m) for collinear connections
_D_MAX = 3.00        # doorway max gap (m)
_COLLINEAR_ANGLE = math.radians(8.0)


def _build_visibility_graph(
    segments: list[LineSegment],
    D_v: float = _D_V,
    gamma_d: float = _GAMMA_D,
    gamma_r: float = _GAMMA_R,
    d_min: float = _D_MIN,
    d_max: float = _D_MAX,
) -> np.ndarray:
    """
    Build the undirected weighted visibility graph G^V.

    Returns an (N, N) affinity matrix A where A[i,j] = W_ij > 0 for
    connected pairs and 0 otherwise.
    """
    n = len(segments)
    A = np.zeros((n, n), dtype=np.float64)
    max_len = max((s.length for s in segments), default=1.0)

    # Build index map: segment.id -> list index
    id_to_idx = {s.id: i for i, s in enumerate(segments)}

    for i in range(n):
        si = segments[i]
        for j in range(i + 1, n):
            sj = segments[j]

            # --- 1. Neighbouring / corner-connected edges (weight 1) ---
            # These come from the directed graph — endpoints that meet
            ep_dist = si.closest_endpoint_dist(sj)
            if ep_dist < 0.05:
                A[i, j] = A[j, i] = 1.0
                continue

            # --- 2. Collinear passage edges (weight 1) ---
            angle = si.angle_with(sj)
            if angle < _COLLINEAR_ANGLE:
                # check if gap is door-width
                gap = _collinear_gap(si, sj)
                if d_min <= gap <= d_max:
                    A[i, j] = A[j, i] = 1.0
                    continue

            # --- 3. Visibility edges (weighted) ---
            d_ij = float(np.linalg.norm(si.midpoint - sj.midpoint))
            if d_ij > D_v:
                continue
            if not _mutual_visibility(si, sj, segments):
                continue

            # Eq (2): distance weight
            W_d = math.exp(-gamma_d * d_ij * d_ij)
            # Eq (3): robot position weight
            W_r = math.exp(-gamma_r * float(np.linalg.norm(si.robot_pos - sj.robot_pos)))
            # Eq (4): length weight
            W_l = (si.length + sj.length) / (2.0 * max_len) if max_len > 0 else 1.0

            W_ij = W_d * W_r * W_l
            if W_ij > 1e-6:
                A[i, j] = A[j, i] = W_ij

    return A


def _collinear_gap(a: LineSegment, b: LineSegment) -> float:
    """Gap between two roughly-collinear segments projected onto a's direction."""
    d = a.direction
    projs_a = [float(np.dot(a.p1, d)), float(np.dot(a.p2, d))]
    projs_b = [float(np.dot(b.p1, d)), float(np.dot(b.p2, d))]
    a_min, a_max = min(projs_a), max(projs_a)
    b_min, b_max = min(projs_b), max(projs_b)
    gap = max(b_min - a_max, a_min - b_max)
    return max(gap, 0.0)


# ---------------------------------------------------------------------------
# Spectral clustering  (Section II-C)
# ---------------------------------------------------------------------------

_MIN_SEGMENTS_FOR_CLUSTERING = 6
_FIEDLER_THRESHOLD = 0.18     # T_lambda — split if below
_MERGE_RATIO_THRESHOLD = 0.5   # T_e — merge if above, matching the paper


def _spectral_cluster(A: np.ndarray, k_override: Optional[int] = None) -> np.ndarray:
    """
    Spectral clustering on affinity matrix A.

    Returns integer labels of length N (one per row of A).
    """
    n = A.shape[0]
    if n <= 1:
        return np.zeros(n, dtype=int)

    # Degree matrix
    D_diag = A.sum(axis=1)
    # Avoid division by zero for isolated nodes
    D_inv_sqrt = np.zeros(n)
    mask = D_diag > 1e-12
    D_inv_sqrt[mask] = 1.0 / np.sqrt(D_diag[mask])
    D_inv_sqrt_mat = np.diag(D_inv_sqrt)

    # Normalized symmetric Laplacian: L_sym = I - D^{-1/2} A D^{-1/2}
    L_sym = np.eye(n) - D_inv_sqrt_mat @ A @ D_inv_sqrt_mat

    # Eigendecomposition (L_sym is symmetric)
    eigenvalues, eigenvectors = la.eigh(L_sym)

    # Eigengap heuristic for k
    if k_override is not None:
        k = k_override
    else:
        k = _eigengap_k(eigenvalues)

    k = max(1, min(k, n))

    if k == 1:
        return np.zeros(n, dtype=int)

    # First k eigenvectors
    V = eigenvectors[:, :k].copy()

    # Row-normalise
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    V = V / norms

    # Column-pivoted QR for cluster assignment (Damle et al. 2019)
    labels = _qr_cluster(V, k)
    return labels


def _eigengap_k(eigenvalues: np.ndarray, max_k: int = 15,
                eig_ceiling: float = 1.0) -> int:
    """Pick k from the largest eigengap among the small eigenvalues.

    Standard spectral-clustering eigengap heuristic:
      k = argmax_{i in 0..limit} (λ_{i+1} - λ_i) + 1

    We search ALL gaps including gap[0] (λ_1 − λ_0): for a single
    well-connected room, gap[0] is the largest gap and signals k=1.
    We only search eigenvalues below *eig_ceiling* (default 1.0) to avoid
    spurious large gaps in the upper part of the spectrum (the normalised
    Laplacian has eigenvalues in [0, 2]).
    """
    n = len(eigenvalues)
    if n <= 2:
        return 1
    # Only consider eigenvalues below the ceiling
    valid = np.where(eigenvalues < eig_ceiling)[0]
    if len(valid) < 2:
        return 1
    limit = min(int(valid[-1]) + 2, n, max_k + 1)  # +1 to include one past ceiling
    eigs = eigenvalues[:limit]
    gaps = np.diff(eigs)
    if len(gaps) < 1:
        return 1
    # Search all gaps — include gaps[0] to allow k=1 (single well-connected room)
    search = gaps
    best_idx = int(np.argmax(search))
    k = best_idx + 1   # 1-based k
    return max(1, min(k, n))


def _qr_cluster(V: np.ndarray, k: int) -> np.ndarray:
    """
    Cluster rows of V into k groups using column-pivoted QR.
    Simple approach: find k pivot rows, assign each row to nearest pivot.
    """
    n = V.shape[0]
    # Column-pivoted QR on V^T to find representative rows
    _, _, piv = la.qr(V.T, pivoting=True)
    pivots = piv[:k]
    centres = V[pivots]

    # Assign each row to nearest centre
    labels = np.zeros(n, dtype=int)
    for i in range(n):
        dists = np.linalg.norm(centres - V[i], axis=1)
        labels[i] = int(np.argmin(dists))
    return labels


# ---------------------------------------------------------------------------
# Split / merge checks
# ---------------------------------------------------------------------------

def _fiedler_value(A_sub: np.ndarray) -> float:
    """Second smallest eigenvalue of the normalised Laplacian of a sub-graph."""
    n = A_sub.shape[0]
    if n <= 1:
        return float("inf")
    D_diag = A_sub.sum(axis=1)
    D_inv_sqrt = np.zeros(n)
    mask = D_diag > 1e-12
    D_inv_sqrt[mask] = 1.0 / np.sqrt(D_diag[mask])
    L = np.eye(n) - np.diag(D_inv_sqrt) @ A_sub @ np.diag(D_inv_sqrt)
    eigs = la.eigvalsh(L)
    return float(eigs[1]) if len(eigs) > 1 else float("inf")


def _split_edge_ratio(A_sub: np.ndarray, sub_labels: np.ndarray) -> float:
    """Paper-style split acceptance ratio: cross edges / smaller cluster size."""
    idx0 = np.where(sub_labels == 0)[0]
    idx1 = np.where(sub_labels == 1)[0]
    if len(idx0) == 0 or len(idx1) == 0:
        return float("inf")
    cross = A_sub[np.ix_(idx0, idx1)] > 1e-6
    n_cross = float(np.count_nonzero(cross))
    return n_cross / float(min(len(idx0), len(idx1)))


def _check_splits(rooms: list[RoomSegment],
                  seg_labels: np.ndarray,
                  A: np.ndarray,
                  segments: list[LineSegment],
                  T_lambda: float = _FIEDLER_THRESHOLD,
                  T_c: float = _MERGE_RATIO_THRESHOLD,
                  target_room_id: Optional[int] = None,
                  ) -> tuple[list[RoomSegment], np.ndarray]:
    """Split only the target room when Fiedler and edge-ratio checks pass."""
    new_rooms: list[RoomSegment] = []
    new_labels = seg_labels.copy()

    for room in rooms:
        if target_room_id is not None and room.id != target_room_id:
            new_rooms.append(room)
            continue
        idxs = np.where(seg_labels == room.id)[0]
        if len(idxs) < 4:
            new_rooms.append(room)
            continue
        A_sub = A[np.ix_(idxs, idxs)]
        fv = _fiedler_value(A_sub)
        if fv < T_lambda and len(idxs) >= 6:
            sub_labels = _spectral_cluster(A_sub, k_override=2)
            ratio = _split_edge_ratio(A_sub, sub_labels)
            if ratio >= T_c:
                new_rooms.append(room)
                continue
            segs_0 = [segments[idxs[j]] for j in range(len(idxs)) if sub_labels[j] == 0]
            segs_1 = [segments[idxs[j]] for j in range(len(idxs)) if sub_labels[j] == 1]
            if len(segs_0) >= len(segs_1):
                room_a = RoomSegment(id=room.id, line_segments=segs_0)
                room_b = RoomSegment(line_segments=segs_1)
            else:
                room_a = RoomSegment(line_segments=segs_0)
                room_b = RoomSegment(id=room.id, line_segments=segs_1)
            room_a.recompute_bounds()
            room_b.recompute_bounds()
            new_rooms.extend([room_a, room_b])
            # update labels
            for j, gi in enumerate(idxs):
                new_labels[gi] = room_a.id if sub_labels[j] == 0 else room_b.id
        else:
            new_rooms.append(room)

    return new_rooms, new_labels


# ---------------------------------------------------------------------------
# Doorway extraction
# ---------------------------------------------------------------------------

_next_door_id = 0


def _extract_doorways(rooms: list[RoomSegment],
                      seg_labels: np.ndarray,
                      segments: list[LineSegment],
                      d_min: float = _D_MIN,
                      d_max: float = _D_MAX) -> list[Doorway]:
    """
    Identify doorways: gaps between collinear segments assigned to different rooms.
    """
    global _next_door_id
    doorways: list[Doorway] = []
    seg_room = {}  # segment.id -> room.id
    for i, seg in enumerate(segments):
        if i < len(seg_labels):
            seg_room[seg.id] = int(seg_labels[i])

    n = len(segments)
    seen: set[tuple[int, int]] = set()
    for i in range(n):
        si = segments[i]
        ri = seg_room.get(si.id, -1)
        for j in range(i + 1, n):
            sj = segments[j]
            rj = seg_room.get(sj.id, -1)
            if ri == rj or ri < 0 or rj < 0:
                continue
            # check collinear gap
            angle = si.angle_with(sj)
            if angle > _COLLINEAR_ANGLE:
                continue
            gap = _collinear_gap(si, sj)
            if d_min <= gap <= d_max:
                key = (min(ri, rj), max(ri, rj))
                if key in seen:
                    continue
                seen.add(key)
                # doorway midpoint: average of the two closest endpoints
                dists = [
                    (np.linalg.norm(si.p1 - sj.p1), si.p1, sj.p1),
                    (np.linalg.norm(si.p1 - sj.p2), si.p1, sj.p2),
                    (np.linalg.norm(si.p2 - sj.p1), si.p2, sj.p1),
                    (np.linalg.norm(si.p2 - sj.p2), si.p2, sj.p2),
                ]
                dists.sort(key=lambda x: x[0])
                pos = 0.5 * (dists[0][1] + dists[0][2])
                _next_door_id += 1
                doorways.append(Doorway(
                    id=_next_door_id,
                    position=pos,
                    width=gap,
                    room_ids=(ri, rj),
                    segment_pair=(si.id, sj.id),
                ))

    # Also detect doorways from non-collinear segments with perpendicular gaps
    for i in range(n):
        si = segments[i]
        ri = seg_room.get(si.id, -1)
        for j in range(i + 1, n):
            sj = segments[j]
            rj = seg_room.get(sj.id, -1)
            if ri == rj or ri < 0 or rj < 0:
                continue
            key = (min(ri, rj), max(ri, rj))
            if key in seen:
                continue
            # non-collinear: check if closest endpoints form a door-width gap
            ep_dist = si.closest_endpoint_dist(sj)
            if d_min <= ep_dist <= d_max:
                # find the closest endpoint pair
                pairs = [(si.p1, sj.p1), (si.p1, sj.p2),
                         (si.p2, sj.p1), (si.p2, sj.p2)]
                best = min(pairs, key=lambda p: float(np.linalg.norm(p[0] - p[1])))
                pos = 0.5 * (best[0] + best[1])
                _next_door_id += 1
                seen.add(key)
                doorways.append(Doorway(
                    id=_next_door_id,
                    position=pos,
                    width=ep_dist,
                    room_ids=(ri, rj),
                    segment_pair=(si.id, sj.id),
                ))

    return doorways


# ---------------------------------------------------------------------------
# Bbox helpers (used by _match_or_create_room for merge detection)
# ---------------------------------------------------------------------------

def _segs_bbox(segs: list[LineSegment]) -> np.ndarray:
    """Return [[xmin, ymin], [xmax, ymax]] bounding box for a list of segments."""
    pts = np.array([s.p1 for s in segs] + [s.p2 for s in segs])
    return np.array([pts.min(axis=0), pts.max(axis=0)])


def _bbox_overlap_ratio(bb1: np.ndarray, bb2: np.ndarray) -> float:
    """IoU-style overlap ratio between two bboxes [[xmin,ymin],[xmax,ymax]]."""
    ix_min = max(bb1[0, 0], bb2[0, 0])
    iy_min = max(bb1[0, 1], bb2[0, 1])
    ix_max = min(bb1[1, 0], bb2[1, 0])
    iy_max = min(bb1[1, 1], bb2[1, 1])
    iw = max(0.0, ix_max - ix_min)
    ih = max(0.0, iy_max - iy_min)
    inter = iw * ih
    if inter < 1e-9:
        return 0.0
    a1 = max(1e-9, (bb1[1, 0] - bb1[0, 0]) * (bb1[1, 1] - bb1[0, 1]))
    a2 = max(1e-9, (bb2[1, 0] - bb2[0, 0]) * (bb2[1, 1] - bb2[0, 1]))
    return float(inter / min(a1, a2))  # overlap relative to smaller box


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

class RoomSegmenter:
    """
    Incremental room segmentation via visibility-graph spectral clustering.

    Usage::

        seg = RoomSegmenter()
        rooms, doorways = seg.update(processed_segments, edges)
    """

    def __init__(self, *,
                 D_v: float = _D_V,
                 gamma_d: float = _GAMMA_D,
                 gamma_r: float = _GAMMA_R,
                 d_min: float = _D_MIN,
                 d_max: float = _D_MAX,
                 T_lambda: float = _FIEDLER_THRESHOLD,
                 T_c: float = _MERGE_RATIO_THRESHOLD):
        self.D_v = D_v
        self.gamma_d = gamma_d
        self.gamma_r = gamma_r
        self.d_min = d_min
        self.d_max = d_max
        self.T_lambda = T_lambda
        self.T_c = T_c

        self._prev_rooms: list[RoomSegment] = []
        self._prev_k: int = 0

    def update(self, segments: list[LineSegment],
               edges: list[tuple[int, int]],
               current_room_id: Optional[int] = None,
               ) -> tuple[list[RoomSegment], list[Doorway]]:
        """
        Run room segmentation on the current set of processed segments.

        Parameters
        ----------
        segments : processed line segments from ``build_directed_graph``.
        edges : directed edges (corner connections) — currently used for
                context only; the visibility graph captures all connectivity.

        Returns
        -------
        rooms : list of RoomSegment with bounding boxes computed.
        doorways : list of Doorway between rooms.
        """
        n = len(segments)
        if n < _MIN_SEGMENTS_FOR_CLUSTERING:
            # Too few segments — return single room containing everything
            room = RoomSegment(line_segments=list(segments))
            room.recompute_bounds()
            return [room], []

        # 1. Build visibility graph
        A = _build_visibility_graph(
            segments,
            D_v=self.D_v,
            gamma_d=self.gamma_d,
            gamma_r=self.gamma_r,
            d_min=self.d_min,
            d_max=self.d_max,
        )

        # 2. Spectral clustering
        labels = _spectral_cluster(A)
        unique_labels = np.unique(labels)

        # 3. Build RoomSegment objects
        rooms: list[RoomSegment] = []
        # Map cluster label → room
        # Re-use previous room IDs where possible for temporal continuity.
        # Pass a shared used_ids set so no two new clusters steal the same previous ID.
        used_ids: set[int] = set()
        for cl in unique_labels:
            idxs = np.where(labels == cl)[0]
            segs = [segments[i] for i in idxs]
            room = self._match_or_create_room(segs, used_ids=used_ids)
            rooms.append(room)

        # Re-label by room ID
        seg_labels = np.zeros(n, dtype=int)
        for room in rooms:
            seg_ids = {s.id for s in room.line_segments}
            for i, s in enumerate(segments):
                if s.id in seg_ids:
                    seg_labels[i] = room.id

        # 4. Split refinement only when room count increases, and only for the
        # current room. This is closer to the paper's online update rule than a
        # global split/merge pass on every frame.
        if self._prev_k > 0 and len(rooms) > self._prev_k and current_room_id is not None:
            rooms, seg_labels = _check_splits(
                rooms, seg_labels, A, segments,
                T_lambda=self.T_lambda,
                T_c=self.T_c,
                target_room_id=current_room_id,
            )

        # Recompute bounds
        for r in rooms:
            r.recompute_bounds()

        # 5. Extract doorways
        doorways = _extract_doorways(rooms, seg_labels, segments,
                                     self.d_min, self.d_max)

        self._prev_rooms = rooms
        self._prev_k = len(rooms)
        return rooms, doorways

    # --- temporal continuity helpers ---

    def _match_or_create_room(self, segs: list[LineSegment],
                              used_ids: set[int] | None = None) -> RoomSegment:
        """
        Try to match a cluster to a previous room (centroid proximity, then bbox
        overlap fallback for merges).  ``used_ids`` prevents two new clusters from
        stealing the same previous room ID.
        """
        pts = np.array([s.midpoint for s in segs])
        centroid = pts.mean(axis=0)

        # --- pass 1: centroid proximity (widened to 3.5 m to catch merges) ---
        best_room: Optional[RoomSegment] = None
        best_dist = 3.5

        for prev in self._prev_rooms:
            if used_ids is not None and prev.id in used_ids:
                continue
            d = float(np.linalg.norm(prev.centroid - centroid))
            if d < best_dist:
                best_dist = d
                best_room = prev

        # --- pass 2: bbox overlap fallback (for large merges, centroid drifts far) ---
        if best_room is None:
            new_bb = _segs_bbox(segs)
            best_overlap = 0.0
            for prev in self._prev_rooms:
                if used_ids is not None and prev.id in used_ids:
                    continue
                ov = _bbox_overlap_ratio(new_bb, prev.bounding_box)
                if ov > 0.30 and ov > best_overlap:
                    best_overlap = ov
                    best_room = prev

        if best_room is not None:
            if used_ids is not None:
                used_ids.add(best_room.id)
            best_room.line_segments = list(segs)
            best_room.recompute_bounds()
            return best_room

        room = RoomSegment(line_segments=list(segs))
        room.recompute_bounds()
        return room
