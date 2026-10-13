"""
scan_safety.py — predictive footprint safety on live lidar points.

Obstacles are the last few seconds of lidar points (map frame, 2 cm voxels) plus
the SLAM map's occupied cells, so walls in a rear-facing lidar's blind wedge stay
known. Every velocity command is checked by predicting the robot's footprint along
its arc. Unlike the SLAM-grid guards this works at centimetre resolution, so the
robot can use the clearances a human operator would: where the normal margin
blocks a move, it is still allowed with a minimal margin at creeping speed.
"""

import math

import numpy as np


class ScanSafety:
    def __init__(self, half_len_front, half_len_rear, half_width, margin=0.03,
                 min_margin=0.01, creep_speed=0.05, near_speed=0.10, horizon=1.0, steps=10,
                 memory_sec=5.0, voxel=0.02, radius=2.0):
        self.base_front, self.base_rear, self.base_w = half_len_front, half_len_rear, half_width
        self.margin, self.min_margin, self.creep_speed = margin, min_margin, creep_speed
        self.near_speed = near_speed
        self.horizon = horizon
        self.steps = steps
        self.memory_sec = memory_sec
        self.voxel = voxel
        self.radius = radius
        self._mem = np.zeros((0, 3), dtype=np.float32)  # x, y, time
        self._static = np.zeros((0, 2), dtype=np.float32)

    # ------------------------------------------------------------------ memory
    def update(self, pts_xy, now):
        """Add one scan's points (Nx2, map frame) observed at time now (seconds)."""
        if pts_xy is None or len(pts_xy) == 0:
            return
        new = np.concatenate(
            [np.asarray(pts_xy, dtype=np.float32)[:, :2],
             np.full((len(pts_xy), 1), now, dtype=np.float32)], axis=1)
        mem = np.concatenate([self._mem[self._mem[:, 2] > now - self.memory_sec], new])
        # One point per voxel, keeping the newest observation.
        keys = np.floor(mem[:, :2] / self.voxel).astype(np.int64)
        flat = keys[:, 0] * 1_000_003 + keys[:, 1]
        _, idx = np.unique(flat[::-1], return_index=True)
        self._mem = mem[::-1][idx]

    def set_static(self, pts_xy):
        """Occupied cells of the SLAM map (Nx2), for walls out of the lidar's view."""
        self._static = np.asarray(pts_xy, dtype=np.float32).reshape(-1, 2)

    def points_near(self, xy):
        xy = np.asarray(xy[:2], dtype=np.float32)
        out = []
        for arr in (self._mem[:, :2], self._static):
            if len(arr):
                out.append(arr[np.sum((arr - xy) ** 2, axis=1) < self.radius ** 2])
        return np.concatenate(out) if out else np.zeros((0, 2), dtype=np.float32)

    # --------------------------------------------------------------- geometry
    def _depth(self, pts, pose, margin):
        """How far each point is inside the footprint (+margin) at pose (<= 0: outside)."""
        x, y, th = pose
        ct, st = math.cos(th), math.sin(th)
        dx, dy = pts[:, 0] - x, pts[:, 1] - y
        lx = ct * dx + st * dy
        ly = -st * dx + ct * dy
        return np.minimum(np.minimum(self.base_front + margin - lx, lx + self.base_rear + margin),
                          self.base_w + margin - np.abs(ly))

    def _hits(self, pts, pose0, pose, margin):
        """A point enters the footprint, or one already inside at pose0 goes deeper."""
        base = np.maximum(self._depth(pts, pose0, margin), 0.0)
        d = self._depth(pts, pose, margin)
        return bool(np.any((d > 0.0) & (d > base + 1e-4)))

    def arc(self, pose, v, w, horizon=None, steps=None):
        horizon = self.horizon if horizon is None else horizon
        steps = self.steps if steps is None else steps
        x, y, th = pose
        out, dt = [], horizon / steps
        for _ in range(steps):
            x += v * math.cos(th) * dt
            y += v * math.sin(th) * dt
            th += w * dt
            out.append((x, y, th))
        return out

    def _first_hit(self, pts, pose, v, w, margin):
        for k, p in enumerate(self.arc(pose, v, w), start=1):
            if self._hits(pts, pose, p, margin):
                return k
        return None

    def _turn_hits(self, pts, pose, sign, margin):
        return any(self._hits(pts, pose, (pose[0], pose[1], pose[2] + sign * a), margin)
                   for a in np.linspace(0.0, math.radians(20.0), 5)[1:])

    # ------------------------------------------------------------------ checks
    def check(self, pose, v, w):
        """(v, w, reason) after slowing/braking for predicted contact.

        Hard limit: the footprint + min_margin along the arc over `horizon`. Contact
        predicted at step k scales the command by (k - 1) / steps, so the robot brakes
        before it. Soft limit: the footprint + margin; entering it caps the speed at
        near_speed (how a tight passage is threaded by hand). Points already
        inside a footprint only count if they would go deeper. Turns on the spot are
        checked over a 20 degree sweep.
        """
        pts = self.points_near(pose)
        if len(pts) == 0 or (abs(v) < 1e-3 and abs(w) < 1e-3):
            return v, w, ""
        if abs(v) < 1e-3:
            sign = 1.0 if w > 0 else -1.0
            if self._turn_hits(pts, pose, sign, self.min_margin):
                return 0.0, 0.0, "turning on the spot would hit an obstacle"
            if self._turn_hits(pts, pose, sign, self.margin):
                return v, math.copysign(min(abs(w), 0.3), w), "tight turn: slowed"
            return v, w, ""
        k = self._first_hit(pts, pose, v, w, self.min_margin)
        if k is not None:
            s = (k - 1) / self.steps
            return v * s, w * s, (f"obstacle {k * self.horizon / self.steps:.1f}s ahead along the "
                                  f"{'forward' if v > 0 else 'reverse'} arc")
        if self._first_hit(pts, pose, v, w, self.margin) is not None:
            # Inside the soft margin only: cap the speed at near_speed. The hard limit
            # above still brakes before any real contact.
            s = min(1.0, self.near_speed / max(abs(v), 1e-6))
            if s < 1.0:
                return v * s, w * s, f"near obstacle: speed capped at {self.near_speed:.2f} m/s"
        return v, w, ""

    def side_clearance(self, pose):
        """Smallest gap (m) between the true footprint's sides and points alongside it."""
        pts = self.points_near(pose)
        if len(pts) == 0:
            return float("inf")
        x, y, th = pose
        ct, st = math.cos(th), math.sin(th)
        dx, dy = pts[:, 0] - x, pts[:, 1] - y
        lx = ct * dx + st * dy
        ly = -st * dx + ct * dy
        along = (lx < self.base_front) & (lx > -self.base_rear)
        if not along.any():
            return float("inf")
        return float(np.min(np.abs(ly[along]) - self.base_w))

    def find_creep(self, pose, w_sign, steps=(0.05, 0.10, 0.20, 0.30)):
        """Signed straight distance after which turning on the spot is clear, or None."""
        pts = self.points_near(pose)
        if len(pts) == 0:
            return None
        ct, st = math.cos(pose[2]), math.sin(pose[2])
        for s in steps:
            for direction in (1.0, -1.0):
                d = direction * s
                probe = (pose[0] + d * ct, pose[1] + d * st, pose[2])
                mid = (pose[0] + 0.5 * d * ct, pose[1] + 0.5 * d * st, pose[2])
                m = self.min_margin
                if self._hits(pts, pose, mid, m) or self._hits(pts, pose, probe, m):
                    continue
                if not self._turn_hits(pts, probe, w_sign, m):
                    return d
        return None
