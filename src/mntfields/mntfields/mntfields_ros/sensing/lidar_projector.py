"""
lidar_projector.py — Project 2D LaserScan points into a camera image plane.

Pure-math utility with no ROS dependencies.  Given camera intrinsics and
a rigid transform from the laser frame to the camera optical frame, converts
(range, angle) polar scans into pixel coordinates and retains the
sensor-frame 3D points for downstream map-frame transformation.
"""

import math
import numpy as np


class ProjectedPoints:
    """Result of a LiDAR-to-image projection."""

    __slots__ = ("pixels_uv", "sensor_xyz", "camera_xyz", "valid_mask")

    def __init__(
        self,
        pixels_uv: np.ndarray,
        sensor_xyz: np.ndarray,
        camera_xyz: np.ndarray,
        valid_mask: np.ndarray,
    ):
        self.pixels_uv = pixels_uv      # (N, 2) float32 — u, v in image
        self.sensor_xyz = sensor_xyz     # (N, 3) float32 — xyz in laser frame
        self.camera_xyz = camera_xyz     # (N, 3) float32 — xyz in camera frame
        self.valid_mask = valid_mask     # (N,)   bool    — True if in front of camera & in image


class LidarProjector:
    """Project 2D LaserScan points into the camera image plane."""

    def __init__(self):
        self._fx: float = 0.0
        self._fy: float = 0.0
        self._cx: float = 0.0
        self._cy: float = 0.0
        self._img_w: int = 0
        self._img_h: int = 0
        self._T: np.ndarray | None = None   # 4x4 camera_from_laser
        self._R: np.ndarray | None = None   # 3x3 rotation cache
        self._t: np.ndarray | None = None   # 3   translation cache
        self._ready = False

    # ------------------------------------------------------------------
    def set_intrinsics(self, fx: float, fy: float, cx: float, cy: float,
                       img_w: int, img_h: int) -> None:
        self._fx, self._fy = fx, fy
        self._cx, self._cy = cx, cy
        self._img_w, self._img_h = img_w, img_h
        self._check_ready()

    def set_transform(self, T_camera_from_laser: np.ndarray) -> None:
        """Set the 4x4 homogeneous transform from laser frame to camera
        optical frame.  Typically a static TF looked up once."""
        T = np.asarray(T_camera_from_laser, dtype=np.float64)
        assert T.shape == (4, 4), f"Expected 4x4, got {T.shape}"
        self._T = T
        self._R = T[:3, :3]
        self._t = T[:3, 3]
        self._check_ready()

    @property
    def ready(self) -> bool:
        return self._ready

    # ------------------------------------------------------------------
    def project(
        self,
        ranges: np.ndarray,
        angle_min: float,
        angle_increment: float,
        range_min: float = 0.05,
        range_max: float = 12.0,
    ) -> ProjectedPoints:
        """Project a LaserScan into the camera image.

        Args:
            ranges:           1-D array of range values from LaserScan.ranges
            angle_min:        start angle (rad)
            angle_increment:  angular step (rad)
            range_min/max:    valid range window (m)

        Returns:
            ProjectedPoints with N entries (one per range reading).
            Use ``valid_mask`` to select points that are in front of the
            camera and inside the image bounds.
        """
        assert self._ready, "Intrinsics or transform not set"

        n = len(ranges)
        ranges = np.asarray(ranges, dtype=np.float32)
        angles = angle_min + np.arange(n, dtype=np.float32) * angle_increment

        # --- Sensor-frame 3D (x, y, 0) -----------------------------------
        valid_range = np.isfinite(ranges) & (ranges >= range_min) & (ranges <= range_max)
        sensor_xyz = np.zeros((n, 3), dtype=np.float32)
        sensor_xyz[:, 0] = ranges * np.cos(angles)
        sensor_xyz[:, 1] = ranges * np.sin(angles)
        # z stays 0 for 2D LiDAR

        # --- Transform to camera optical frame ----------------------------
        # p_cam = R @ p_laser + t   (vectorised)
        cam = (self._R @ sensor_xyz.T).T + self._t   # (N, 3) float64
        cam = cam.astype(np.float32)

        # --- Pinhole projection -------------------------------------------
        Z = cam[:, 2]
        in_front = Z > 0.01  # must be in front of camera

        u = np.full(n, -1.0, dtype=np.float32)
        v = np.full(n, -1.0, dtype=np.float32)
        ok = in_front & valid_range
        if ok.any():
            inv_z = 1.0 / Z[ok]
            u[ok] = self._fx * cam[ok, 0] * inv_z + self._cx
            v[ok] = self._fy * cam[ok, 1] * inv_z + self._cy

        in_bounds = (
            ok
            & (u >= 0) & (u < self._img_w)
            & (v >= 0) & (v < self._img_h)
        )

        pixels = np.stack([u, v], axis=1)  # (N, 2)
        return ProjectedPoints(
            pixels_uv=pixels,
            sensor_xyz=sensor_xyz,
            camera_xyz=cam,
            valid_mask=in_bounds,
        )

    # ------------------------------------------------------------------
    def _check_ready(self):
        self._ready = (self._fx > 0 and self._T is not None)
