"""
region.py — Standalone neural metric field region.

Region holds a Model internally (no inheritance).
No threading, no inference_lock, no randomize_data in training path.
"""

import os
import json
import copy
import math
import gc
import warnings

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg", force=True)
from matplotlib import cm, colors
import matplotlib.pyplot as plt

import torch
import numpy as np
from scipy.spatial import cKDTree
import torch_kdtree
import open3d as o3d

from mntfields.mntfields_core.ntrl import model_network_metric as model_network
from mntfields.mntfields_core.ntrl import model_function_metric as model_function
from mntfields.mntfields_core.core.model_base import Model
from mntfields.utils.mntfields_utils import (
    np_to_pcd,
    combine_pcd,
)

# Fixed contour levels for TT field plots.
TT_CONTOUR_LEVELS = np.linspace(0, 200, 100)

warnings.filterwarnings(
    'ignore',
    message=r"input's size at dim=0 does not match num_features.*",
    category=UserWarning,
)


class Region:
    """Standalone neural metric field for a spatial region."""

    n_regions = 0

    def __init__(
        self,
        model_folder: str,
        bound: torch.Tensor,
        version: int = -1,
        device: str = "cuda:0",
        wall_min: float = 0.05,
        wall_max: float = 0.80,
        wall_offset: float = 0.0,
    ):
        """
        Args:
            model_folder: root output directory (region subfolder created inside)
            bound:        torch.Tensor [2, 3] – [[xmin,ymin,-0.1],[xmax,ymax,0.1]]
            version:      int region ID; auto-increments when -1
            device:       torch device string
            wall_min:     near-wall band lower edge [m]; speed is lowest (wall_min/wall_max) here
            wall_max:     near-wall band upper edge [m]; speed reaches 1 here
            wall_offset:  robot clearance [m]: distances are measured from walls inflated
                          by this much (configuration space), so the speed profile and
                          the wall contrast stay the same for a bigger robot
        """
        # Version
        if version == -1:
            Region.n_regions += 1
            self.version = Region.n_regions
        else:
            self.version = version

        self.device = device

        # Per-region folder
        self.folder = os.path.join(model_folder, f"region_{self.version}")
        os.makedirs(self.folder, exist_ok=True)

        # Bound
        bound = bound.clone().float()
        bound[0, 2] = -0.1
        bound[1, 2] = 0.1
        self.bound = bound

        # Near-wall band parameters
        self.minimum = float(wall_min)
        self.maximum = float(wall_max)
        self.wall_offset = float(wall_offset)

        # Internal Model instance
        self.model = Model(
            folder=self.folder,
            dim=3,
            B_scale=10,
            device=device,
            dim_cells=128,
            init_network=True,
            eval=False,
            lr=1e-4,
        )
        # Expose network and function at region level for callers
        self._sync_model_refs()

        # Training state
        self.epoch = 0
        self.alpha = 1
        self.train_tick_ms: list = []   # wall-clock time per training tick (ms)
        self.loss_history: list = []    # final loss value per training tick
        self.all_raw_frame_data = []
        self.max_raw_samples = 25_000
        self.raw_memory_cell_size_m = 0.35

        # Cosine warm-restart LR schedule
        self.lr_max = 5e-4
        self.lr_min = 5e-5
        self.lr_cycle_epochs = 200

        # LiDAR accumulation
        self.lidar_obs = None
        self.all_lidar = []
        self.kd_tree = None
        self.gpu_kd_tree = None
        self.max_lidar_scans = 40  # sliding window ~20s at 2Hz

        # Misc state
        self.saved = False
        self.loaded = True
        self.last_robot_pose = None
        self.latest_goal = None
        self.latest_path = None
        self.spawn_center = None
        self.enable_debug_vis = False
        self._last_vis_wall_t = 0.0

        # Semantic object storage: class_name → list of fused landmarks.
        # New entries use dicts: {"xy": [x, y], "count": n, "last_seen": t}.
        # Older saved maps may contain plain [x, y] entries; add_semantic_object
        # handles both formats.
        self.semantic_objects: dict = {}

        # Persistence paths (set by _region_paths)
        self.network_save_path = None
        self.B_save_path = None
        self.data_save_path = None
        self.meta_save_path = None

    # ------------------------------------------------------------------
    # Semantic object storage
    # ------------------------------------------------------------------

    def add_semantic_object(self, class_name: str, world_xy, dedup_radius: float = 0.8,
                            stamp: float | None = None) -> bool:
        """Fuse a furniture detection into a class-aware semantic landmark.

        Returns True when a new landmark is created, False when the detection is
        associated with and fused into an existing landmark. Plain legacy [x, y]
        entries are upgraded in place.
        """
        xy = np.asarray(world_xy[:2], dtype=np.float32)
        storage_class = class_name
        existing = self.semantic_objects.get(storage_class, [])
        now = float(stamp) if stamp is not None else 0.0

        best_idx = None
        best_class = storage_class
        best_dist = float("inf")
        candidate_classes = [storage_class]
        if class_name in SEMANTIC_FURNITURE_CLASSES:
            candidate_classes = [
                cls for cls in SEMANTIC_FURNITURE_CLASSES
                if cls in self.semantic_objects
            ] or [storage_class]

        for cls in candidate_classes:
            for idx, prev in enumerate(self.semantic_objects.get(cls, [])):
                prev_xy = self._semantic_entry_xy(prev)
                if prev_xy is None:
                    continue
                dist = float(np.linalg.norm(xy - prev_xy))
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx
                    best_class = cls

        if best_idx is not None and best_dist < dedup_radius:
            existing = self.semantic_objects.get(best_class, [])
            prev = existing[best_idx]
            if isinstance(prev, dict):
                count = max(1, int(prev.get("count", 1)))
                prev_xy = self._semantic_entry_xy(prev)
            else:
                count = 1
                prev_xy = self._semantic_entry_xy(prev)
                prev = {"xy": prev_xy.tolist(), "count": 1, "last_seen": 0.0}

            alpha = 1.0 / float(count + 1)
            fused_xy = (1.0 - alpha) * prev_xy + alpha * xy
            prev["xy"] = fused_xy.astype(np.float32).tolist()
            prev["count"] = count + 1
            prev["last_seen"] = now
            votes = prev.get("class_votes", {})
            votes[class_name] = int(votes.get(class_name, 0)) + 1
            prev["class_votes"] = votes
            prev["class_name"] = max(votes.items(), key=lambda item: item[1])[0]
            existing[best_idx] = prev
            self.semantic_objects[best_class] = existing
            return False

        existing = self.semantic_objects.get(storage_class, [])
        existing.append({
            "xy": xy.tolist(),
            "count": 1,
            "last_seen": now,
            "class_name": class_name,
            "class_votes": {class_name: 1},
        })
        self.semantic_objects[storage_class] = existing
        return True

    @staticmethod
    def _semantic_entry_xy(entry) -> np.ndarray | None:
        try:
            if isinstance(entry, dict):
                raw = entry.get("xy", entry.get("world_xy", entry.get("pos")))
            else:
                raw = entry
            xy = np.asarray(raw[:2], dtype=np.float32)
            if xy.shape[0] < 2 or not np.all(np.isfinite(xy[:2])):
                return None
            return xy[:2]
        except Exception:
            return None

    def prune_raw_data_near_object(
        self,
        old_xy: np.ndarray,
        new_xy: "np.ndarray | None",
        radius_m: float = 1.0,
    ) -> int:
        """Remove Nx12 training rows contaminated by a moved furniture object.

        Drops rows where:
        - cp0 or cp1 (closest obstacle contact, cols 6-8 / 9-11) is within
          radius_m of old_xy  — these recorded the old obstacle surface.
        - x0 or x1 (free-space sample, cols 0-2 / 3-5) is within
          radius_m*0.6 of new_xy — these are now inside the new obstacle
          footprint. Only applied when old→new displacement > radius_m*0.5
          (avoids over-pruning for tiny jitter moves).

        Returns the number of rows removed.
        """
        if not self.all_raw_frame_data:
            return 0

        data = (self.all_raw_frame_data[0]
                if len(self.all_raw_frame_data) == 1
                else torch.cat(self.all_raw_frame_data, dim=0))

        old = torch.tensor(old_xy[:2], dtype=torch.float32)

        cp0_xy = data[:, 6:8]
        cp1_xy = data[:, 9:11]
        stale = (
            (torch.norm(cp0_xy - old, dim=1) < radius_m) |
            (torch.norm(cp1_xy - old, dim=1) < radius_m)
        )

        if new_xy is not None:
            new = torch.tensor(new_xy[:2], dtype=torch.float32)
            if float(torch.norm(old - new)) > radius_m * 0.5:
                r_new = radius_m * 0.6
                x0_xy = data[:, 0:2]
                x1_xy = data[:, 3:5]
                blocked = (
                    (torch.norm(x0_xy - new, dim=1) < r_new) |
                    (torch.norm(x1_xy - new, dim=1) < r_new)
                )
                stale = stale | blocked

        n_removed = int(stale.sum())
        if n_removed > 0:
            self.all_raw_frame_data = [data[~stale]]
        return n_removed

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sync_model_refs(self):
        """Expose model internals at region level (convenience for callers)."""
        self.network = self.model.network
        self.function = self.model.function
        self.optimizer = self.model.optimizer
        self.B = self.model.B
        self.Params = self.model.Params
        self.prev_state_queue = self.model.prev_state_queue
        self.prev_optimizer_queue = self.model.prev_optimizer_queue

    def _region_paths(self):
        self.network_save_path = os.path.join(self.folder, f"region_{self.version}_network.pt")
        self.B_save_path = os.path.join(self.folder, f"region_{self.version}_B.pt")
        self.data_save_path = os.path.join(self.folder, f"region_{self.version}_data.pth")
        self.meta_save_path = os.path.join(self.folder, f"region_{self.version}_meta.json")
        return (self.network_save_path, self.B_save_path,
                self.data_save_path, self.meta_save_path)

    # ------------------------------------------------------------------
    # Bound management
    # ------------------------------------------------------------------

    def set_bound(self, new_bound: torch.Tensor):
        new_bound = new_bound.clone().float()
        new_bound[0, 2] = -0.1
        new_bound[1, 2] = 0.1
        self.bound = new_bound

    def in_bound(self, pos):
        """pos: (2,) numpy or list."""
        b = self.bound.detach().cpu().numpy()
        pos = np.asarray(pos, dtype=np.float32)
        return (pos >= b[0, :2]).all() and (pos <= b[1, :2]).all()

    def _xy_in_current_bound(self, pts_xy, margin=0.0):
        pts_xy = np.asarray(pts_xy, dtype=np.float32)
        if pts_xy.size == 0:
            return np.zeros((0,), dtype=bool)
        b = self.bound.detach().cpu().numpy()
        return (
            (pts_xy[:, 0] >= float(b[0, 0]) - margin) &
            (pts_xy[:, 0] <= float(b[1, 0]) + margin) &
            (pts_xy[:, 1] >= float(b[0, 1]) - margin) &
            (pts_xy[:, 1] <= float(b[1, 1]) + margin)
        )

    @property
    def center(self):
        b = self.bound.detach().cpu().numpy()
        return np.array([
            float((b[0, 0] + b[1, 0]) * 0.5),
            float((b[0, 1] + b[1, 1]) * 0.5),
            float((b[0, 2] + b[1, 2]) * 0.5),
        ], dtype=np.float32)

    # ------------------------------------------------------------------
    # Learning-rate schedule
    # ------------------------------------------------------------------

    def _current_lr(self) -> float:
        """Cosine annealing with warm restarts."""
        cycle_pos = (self.epoch % self.lr_cycle_epochs) / self.lr_cycle_epochs
        return self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (
            1.0 + math.cos(math.pi * cycle_pos))

    def is_field_unstable(self, hot_window=40) -> bool:
        """True during early phase of a cosine LR warm-restart cycle."""
        cycle_pos = self.epoch % self.lr_cycle_epochs
        return cycle_pos < hot_window

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def normalize_tensor(self, data):
        n_repeat = data.shape[-1] // 3
        bound_len = (self.bound[1] - self.bound[0]).repeat(n_repeat).to(data)
        bound_start = self.bound[0].repeat(n_repeat).to(data)
        return (data - bound_start) / bound_len - 0.5

    def normalize(self, data):
        if data.shape[-1] == 2:
            data = np.stack([data[..., 0], data[..., 1],
                             np.ones_like(data[..., 1])], axis=-1)
        bound_len = (self.bound[1] - self.bound[0]).detach().cpu().numpy()
        bound_start = self.bound[0].detach().cpu().numpy()
        return (data - bound_start) / bound_len - 0.5

    # ------------------------------------------------------------------
    # Raw history helpers
    # ------------------------------------------------------------------

    def _filter_raw_rows_to_bound(self, rows: torch.Tensor) -> torch.Tensor:
        if rows is None or len(rows) == 0:
            return rows
        arr = rows.detach().cpu().numpy()
        keep = (
            self._xy_in_current_bound(arr[:, 0:2]) &
            self._xy_in_current_bound(arr[:, 3:5])
        )
        if keep.all():
            return rows
        keep_idx = torch.as_tensor(np.where(keep)[0], dtype=torch.long)
        return rows[keep_idx]

    def _spatially_cap_raw_history(self, merged: torch.Tensor, max_raw: int) -> torch.Tensor:
        if merged is None or len(merged) <= max_raw:
            return merged

        arr = merged.detach().cpu().numpy()
        x0 = arr[:, :2]
        cell = max(0.05, float(getattr(self, "raw_memory_cell_size_m", 0.35)))
        bins = np.floor(x0 / cell).astype(np.int32)
        keys = (bins[:, 0].astype(np.int64) * np.int64(1_000_003)
                + bins[:, 1].astype(np.int64))
        uniq, inv = np.unique(keys, return_inverse=True)
        n_bins = int(len(uniq))

        if n_bins <= 0:
            idx = np.random.choice(len(arr), max_raw, replace=False)
            return merged[idx]

        base_quota = max(1, max_raw // n_bins)
        chosen = []
        chosen_mask = np.zeros(len(arr), dtype=bool)

        # Recency weights: higher index = newer = higher weight
        row_ages = np.arange(len(arr), dtype=np.float64) + 1.0

        for bi in range(n_bins):
            inds = np.where(inv == bi)[0]
            if len(inds) <= base_quota:
                pick = inds
            else:
                w = row_ages[inds]
                w = w / w.sum()
                pick = np.random.choice(inds, base_quota, replace=False, p=w)
            chosen.extend(pick.tolist())
            chosen_mask[pick] = True

        if len(chosen) < max_raw:
            remaining = np.where(~chosen_mask)[0]
            fill = min(max_raw - len(chosen), len(remaining))
            if fill > 0:
                w = row_ages[remaining]
                w = w / w.sum()
                extra = np.random.choice(remaining, fill, replace=False, p=w)
                chosen.extend(extra.tolist())

        if len(chosen) > max_raw:
            chosen = np.random.choice(
                np.asarray(chosen, dtype=np.int64), max_raw, replace=False
            ).tolist()

        idx = torch.as_tensor(np.asarray(chosen, dtype=np.int64), dtype=torch.long)
        return merged[idx]

    def accumulate_raw_data(self, raw_frame_data: torch.Tensor) -> bool:
        """Accumulate raw Nx12 world-coord pairs WITHOUT running a backward pass."""
        if raw_frame_data is None or raw_frame_data.shape[0] == 0:
            return False
        if raw_frame_data.shape[1] != 12:
            return False
        if self.all_raw_frame_data is None:
            self.all_raw_frame_data = []
        rows = self._filter_raw_rows_to_bound(raw_frame_data.detach().cpu())
        if len(rows) == 0:
            return False
        self.all_raw_frame_data.append(rows)
        merged = torch.cat(self.all_raw_frame_data, dim=0)
        merged = self._filter_raw_rows_to_bound(merged)
        max_raw = max(1, int(getattr(self, "max_raw_samples", 10_000)))
        if len(merged) > max_raw:
            merged = self._spatially_cap_raw_history(merged, max_raw)
        self.all_raw_frame_data = [merged]
        return True

    # ------------------------------------------------------------------
    # LiDAR data ingestion
    # ------------------------------------------------------------------

    def get_data_from_lidar(self, lidar_scan, cur_pos, device='cuda:0',
                            n_samples=4000, non_hit_dirs=None,
                            ownership_check_fn=None):
        """
        Sample training pairs using GPU KD-tree near-wall sampling.
        Returns (points, closest_points) both Nx3 tensors, or None.
        """
        obs_pts_2d = lidar_scan[:, :2]
        if len(obs_pts_2d) < 20:
            print("[LIDAR KD] too few obstacle points in current scan")
            return None

        # 1. Accumulate scan as oriented point cloud
        cur_lidar_pcd = np_to_pcd(lidar_scan, cur_pos, downsample=True)
        if self.lidar_obs is not None:
            self.lidar_obs = combine_pcd(cur_lidar_pcd, self.lidar_obs)
        else:
            self.lidar_obs = cur_lidar_pcd
        self.all_lidar.append((lidar_scan.copy(), cur_pos.copy()))

        # Sliding window: expire stale LiDAR scans
        if len(self.all_lidar) > self.max_lidar_scans:
            self.all_lidar = self.all_lidar[-self.max_lidar_scans:]
            # Rebuild lidar_obs from retained scans only
            self.lidar_obs = None
            for scan, pos in self.all_lidar:
                pcd = np_to_pcd(scan, pos, downsample=True)
                if self.lidar_obs is None:
                    self.lidar_obs = pcd
                else:
                    self.lidar_obs = combine_pcd(pcd, self.lidar_obs)

        all_obs = np.asarray(self.lidar_obs.points)
        if len(all_obs) < 20:
            print("[LIDAR KD] too few accumulated obstacle points")
            return None

        self.kd_tree = cKDTree(all_obs)
        self.gpu_kd_tree = torch_kdtree.build_kd_tree(
            torch.from_numpy(all_obs).float(),
            device=torch.device(device),
            squared_distances=False,
        )

        # 2. Local AABB clipped to region bound
        bb_min = all_obs.min(axis=0).astype(np.float32)
        bb_max = all_obs.max(axis=0).astype(np.float32)
        b = self.bound.detach().cpu().numpy().astype(np.float32)
        bb_min[:2] = np.maximum(bb_min[:2], b[0, :2])
        bb_max[:2] = np.minimum(bb_max[:2], b[1, :2])
        bb_min[2] = -0.05
        bb_max[2] = 0.05
        if np.any(bb_max <= bb_min):
            print("[LIDAR KD] invalid sampling box")
            return None

        # 3. Sample x0 uniformly in bounding box, keep near-wall band
        n_try = n_samples * 6
        x0_2d_cands = np.random.uniform(
            bb_min[:2], bb_max[:2], (n_try, 2)
        ).astype(np.float32)
        z0 = np.zeros((n_try, 1), dtype=np.float32)
        x0_3d_t = torch.from_numpy(np.hstack([x0_2d_cands, z0])).float().to(device)
        dist0_t, idx0_t = self.gpu_kd_tree.query(x0_3d_t, nr_nns_searches=1)
        dist0 = dist0_t[:, 0].cpu().numpy()
        idx0 = idx0_t[:, 0].cpu().numpy()

        dist0_eff = dist0 - float(self.wall_offset)
        near_wall_mask = (dist0_eff > float(self.minimum)) & (dist0_eff < float(self.maximum))
        if near_wall_mask.sum() == 0:
            print("[LIDAR KD] no x0 in near-wall band after uniform sampling")
            return None

        x0_2d = x0_2d_cands[near_wall_mask].astype(np.float32)
        cp0_arr = all_obs[idx0[near_wall_mask]].astype(np.float32)

        # Ownership filter: only train on cells this region owns
        if ownership_check_fn is not None:
            owned = ownership_check_fn(x0_2d)
            if owned.sum() == 0:
                print("[LIDAR KD] no x0 passes ownership filter")
                return None
            x0_2d = x0_2d[owned]
            cp0_arr = cp0_arr[owned]

        if len(x0_2d) > n_samples:
            sel2 = np.random.choice(len(x0_2d), n_samples, replace=False)
            x0_2d = x0_2d[sel2]
            cp0_arr = cp0_arr[sel2]

        n = len(x0_2d)
        x0_np = x0_2d
        z_n = np.zeros((n, 1), dtype=np.float32)

        x0 = torch.from_numpy(np.hstack([x0_np, z_n])).float().to(device)
        closest_obs0 = torch.from_numpy(cp0_arr).float().to(device)

        # 4. Sample x1 by local random displacement
        half_extent = max(
            float(bb_max[0] - bb_min[0]),
            float(bb_max[1] - bb_min[1])
        ) / 2.0
        half_extent = max(half_extent, 0.5)
        all_angles = np.random.uniform(0.0, 2.0 * np.pi, n)
        all_lengths = (np.random.uniform(0.0, 1.0, n) + 0.03) * half_extent
        x1_2d = (x0_np +
                 np.stack([np.cos(all_angles), np.sin(all_angles)], axis=-1)
                 * all_lengths[:, None]).astype(np.float32)

        keep = (
            (x1_2d[:, 0] >= bb_min[0]) & (x1_2d[:, 0] <= bb_max[0]) &
            (x1_2d[:, 1] >= bb_min[1]) & (x1_2d[:, 1] <= bb_max[1])
        )
        if keep.sum() < 10:
            print("[LIDAR KD] insufficient x1 candidates after displacement sampling")
            return None

        keep_idx = np.where(keep)[0]
        x0_np = x0_np[keep_idx]
        x1_2d = x1_2d[keep_idx]
        closest_obs0 = closest_obs0[keep_idx]

        # 5. Look up cp1 from global KD-tree
        x1_3d_q = np.hstack([x1_2d.astype(np.float32),
                              np.zeros((len(x1_2d), 1), dtype=np.float32)])
        x1_3d_t = torch.from_numpy(x1_3d_q).float().to(device)
        _, idx1_t = self.gpu_kd_tree.query(x1_3d_t, nr_nns_searches=1)
        idx1 = idx1_t[:, 0].cpu().numpy()
        cp1_3d = all_obs[idx1].astype(np.float32)

        # 6. Assemble Nx12 world-coord tensor
        z = np.zeros((len(x0_np), 1), dtype=np.float32)
        x0_3d = np.hstack([x0_np.astype(np.float32), z])
        x1_3d = np.hstack([x1_2d.astype(np.float32), z])
        cp0_3d = closest_obs0.detach().cpu().numpy().astype(np.float32)

        points = torch.from_numpy(np.hstack([x0_3d, x1_3d])).float().to(device)
        closest_points = torch.from_numpy(np.hstack([cp0_3d, cp1_3d])).float().to(device)
        return points, closest_points

    # ------------------------------------------------------------------
    # Speed and normal computation  (Nx12 → Nx14)
    # ------------------------------------------------------------------

    def calculate_speed_and_normal(self, data):
        """
        Convert Nx12 world-coord pairs (x0, x1, cp0, cp1) into Nx14 training rows.

        Output columns: [x0_n(3), x1_n(3), y0(1), y1(1), n0(3), n1(3)]
        """
        x0 = data[:, :3]
        x1 = data[:, 3:6]
        p0 = data[:, 6:9]
        p1 = data[:, 9:12]

        n0 = x0 - p0
        n1 = x1 - p1

        d0 = torch.norm(n0, dim=1, keepdim=True)
        d1 = torch.norm(n1, dim=1, keepdim=True)
        # Distances to walls inflated by the robot clearance.
        offset = float(getattr(self, "wall_offset", 0.0))
        y0 = d0 - offset
        y1 = d1 - offset

        maximum = float(self.maximum)
        minimum = float(self.minimum)

        valid = (y0[:, 0] > minimum) & (y0[:, 0] <= maximum)
        x0 = x0[valid]; x1 = x1[valid]
        y0 = y0[valid]; y1 = y1[valid]
        n0 = n0[valid]; n1 = n1[valid]
        d0 = d0[valid]; d1 = d1[valid]

        if len(x0) == 0:
            return torch.zeros((0, 14), dtype=torch.float32, device=data.device)

        n0 = n0 / (d0 + 1e-8)
        n1 = n1 / (d1 + 1e-8)

        x0_n = self.normalize_tensor(x0)
        x1_n = self.normalize_tensor(x1)

        y0 = torch.clamp(y0, min=minimum, max=maximum) / maximum
        y1 = torch.clamp(y1, min=minimum, max=maximum) / maximum

        return torch.cat([x0_n, x1_n, y0, y1, n0, n1], dim=1)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, epoch: int, world_pts, robot_pose, non_hit_dirs=None):
        """
        Accumulate Nx12 raw data, filter, convert to Nx14, train model.

        No threading, no inference_lock, no randomize_data.
        """
        if self.model.network is None or self.model.optimizer is None:
            return False

        result = self.get_data_from_lidar(
            world_pts, robot_pose, device=self.device, n_samples=4000,
            non_hit_dirs=non_hit_dirs,
        )
        if result is None:
            return False

        points, closest = result
        if points.shape[0] == 0:
            return False

        raw = torch.cat([points, closest], dim=1)  # Nx12
        if not torch.isfinite(raw).all():
            print("[TRAIN] non-finite values in raw data, skipping")
            return False

        raw_cpu = raw.detach().cpu()
        raw_cpu = self._filter_raw_rows_to_bound(raw_cpu)
        if len(raw_cpu) == 0:
            return False

        if self.all_raw_frame_data is None:
            self.all_raw_frame_data = []
        self.all_raw_frame_data.append(raw_cpu)

        merged = torch.cat(self.all_raw_frame_data, dim=0)
        merged = self._filter_raw_rows_to_bound(merged)
        if len(merged) == 0:
            return False

        max_raw = max(1, int(self.max_raw_samples))
        if len(merged) > max_raw:
            merged = self._spatially_cap_raw_history(merged, max_raw)
        self.all_raw_frame_data = [merged]

        # Nx14
        n14 = self.calculate_speed_and_normal(merged.detach().cpu())
        if n14 is None or len(n14) == 0:
            return False
        n14 = n14.to(self.device)

        # Cosine warm-restart LR
        self.model.Params['Training']['Learning Rate'] = self._current_lr()

        loss = None
        completed_epochs = 0

        for _ep in range(max(1, int(epoch))):
            all_data = n14
            if len(all_data) > 7000:
                idx = torch.randperm(len(all_data), device=self.device)[:7000]
                all_data = all_data[idx]
            loss_new, _ = self.model.train_core(
                epoch=1, frame_data=all_data, is_one_frame=True)
            completed_epochs += 1
            if loss_new is not None:
                loss = loss_new
                loss_val = (loss_new.item()
                            if isinstance(loss_new, torch.Tensor)
                            else float(loss_new))
                print(f"[TRAIN] epoch {self.epoch + completed_epochs:03d} "
                      f"loss={loss_val:.3e}")

        # Retry on catastrophic None
        if loss is None:
            for _retry in range(3):
                if self.model.optimizer is None or self.model.B is None:
                    break
                self.model.delete_network()
                self.model.init_network()
                self._sync_model_refs()
                all_data = n14
                if len(all_data) > 7000:
                    idx = torch.randperm(len(all_data), device=self.device)[:7000]
                    all_data = all_data[idx]
                loss, _ = self.model.train_core(
                    epoch=1, frame_data=all_data, is_one_frame=True)
                completed_epochs += 1
                if loss is not None:
                    break

        self._sync_model_refs()

        if loss is None:
            print("[TRAIN] failed after retries")
            return False

        self.epoch += completed_epochs
        loss_val = loss.item() if isinstance(loss, torch.Tensor) else float(loss)

        # Periodic save
        if self.epoch % 50 == 0:
            self.save_model()

        # Optional visualization
        if self.enable_debug_vis:
            import time as _time
            _now = _time.monotonic()
            _vis_interval = 20.0
            if (_now - self._last_vis_wall_t) >= _vis_interval:
                self._last_vis_wall_t = _now
                view_src = (self.last_robot_pose.copy()
                            if self.last_robot_pose is not None
                            else self.center)
                try:
                    self.save_debug_plot(view_src, view_radius=3.5,
                                         robot_xy=view_src[:2])
                    n14_vis = n14[:, :6]
                    n14_spd = n14[:, 6:8]
                    self.save_training_vis(n14_vis, n14_spd, robot_xy=view_src[:2])
                except Exception as e:
                    print(f"[VIS] skipped: {e}")

        return loss_val < 0.005

    # ------------------------------------------------------------------
    # Inference: waypoint and path generation
    # ------------------------------------------------------------------

    def get_next_waypoint(self, robot_xy, goal_xy, step_size=0.35, alpha=0.8,
                          debug=True):
        """
        Return a waypoint one step ahead by following -∇TT.
        Falls back to geometric steering if the field is unavailable.
        """
        dx = float(goal_xy[0] - robot_xy[0])
        dy = float(goal_xy[1] - robot_xy[1])
        dist = float(np.hypot(dx, dy))

        if dist < 1e-6:
            return np.array(goal_xy, dtype=np.float32)

        geom_dir = np.array([dx, dy], dtype=np.float32) / dist

        if self.model.function is None or self.model.network is None:
            step = min(step_size, dist)
            return (np.array(robot_xy, dtype=np.float32)
                    + geom_dir * step).astype(np.float32)

        try:
            device = self.device
            x0 = torch.tensor([[robot_xy[0], robot_xy[1], 0.0]],
                               dtype=torch.float32, device=device)
            x1 = torch.tensor([[goal_xy[0], goal_xy[1], 0.0]],
                               dtype=torch.float32, device=device)
            xp = torch.cat([x0, x1], dim=1)  # (1, 6)

            xp_n = self.normalize_tensor(xp.clone())
            xp_n.requires_grad_(True)

            tau, w, coords = self.model.network.out(xp_n)
            tt = tau[:, 0]

            grad = torch.autograd.grad(
                outputs=tt.sum(),
                inputs=coords,
                retain_graph=False,
                create_graph=False,
            )[0]

            if grad is None:
                direction = geom_dir
            else:
                grad_xy_norm = grad[0, :2].detach().cpu().numpy()
                b_np = self.bound.detach().cpu().numpy()
                bound_len = (b_np[1, :2] - b_np[0, :2]).astype(np.float32)
                grad_xy = grad_xy_norm / (bound_len + 1e-8)
                norm = float(np.linalg.norm(grad_xy))

                if norm < 1e-5:
                    direction = geom_dir
                else:
                    field_dir = (-grad_xy / norm).astype(np.float32)
                    direction = alpha * field_dir + (1.0 - alpha) * geom_dir
                    direction = direction / (np.linalg.norm(direction) + 1e-8)

            step = min(step_size, dist)
            waypoint = (np.array(robot_xy, dtype=np.float32)
                        + direction.astype(np.float32) * step)
            return waypoint.astype(np.float32)

        except Exception as e:
            print("FIELD WAYPOINT ERROR:", e)
            step = min(step_size, dist)
            return (np.array(robot_xy, dtype=np.float32)
                    + geom_dir * step).astype(np.float32)

    def generate_field_path(self, start_xy, goal_xy, step_size=0.15,
                            max_steps=120, gridmap=None, forward_only=False):
        """
        mNTFields rollout from start_xy to goal_xy.

        forward_only=False (default): bidirectional — two simultaneous rollouts
            meet in the middle.  Works well when both endpoints are inside the
            trained region.
        forward_only=True: single forward rollout start→goal, clipped at the
            region boundary.  A straight-line segment is appended from the
            rollout endpoint to goal_xy so the returned path always reaches the
            goal.  Use this when goal_xy is outside the trained region; avoids
            the random-gradient attractor problem of the goal-side rollout.
        """
        cur_s = np.array(start_xy, dtype=np.float32)
        cur_g = np.array(goal_xy, dtype=np.float32)
        pts_start = []
        pts_goal = []
        field_steps = 0

        b = self.bound.detach().cpu().numpy()

        def _in_wall(xy):
            if gridmap is None:
                return False
            cell = gridmap.cell_size
            origin = gridmap.origin
            half = gridmap.map_size / 2.0
            ix = int((xy[0] - (origin[0] - half)) / cell)
            iy = int((xy[1] - (origin[1] - half)) / cell)
            if not (0 <= ix < gridmap.n_cells and 0 <= iy < gridmap.n_cells):
                return True
            return gridmap.data[iy, ix] == 1

        def _clip_to_bound(xy):
            return np.array([
                np.clip(float(xy[0]), float(b[0, 0]), float(b[1, 0])),
                np.clip(float(xy[1]), float(b[0, 1]), float(b[1, 1])),
            ], dtype=np.float32)

        def _segment_hits_wall(a, c):
            if gridmap is None:
                return False
            d = float(np.linalg.norm(c - a))
            if d < 1e-6:
                return False
            n = max(2, int(np.ceil(
                d / max(1e-4, 0.5 * gridmap.cell_size))))
            ts = np.linspace(0.0, 1.0, n, dtype=np.float32)
            for t in ts:
                p = a + t * (c - a)
                if _in_wall(p):
                    return True
            return False

        def _advance(cur, target, clip_bound=True):
            """One step of get_next_waypoint, optionally clipped and wall-checked."""
            nxt = self.get_next_waypoint(cur, target, step_size=step_size)
            if nxt is None:
                return cur.copy(), False
            nxt = np.asarray(nxt, dtype=np.float32)
            if clip_bound:
                nxt = _clip_to_bound(nxt)
            if _in_wall(nxt) or _segment_hits_wall(cur, nxt):
                return cur.copy(), False
            return nxt, True

        if forward_only:
            # ── Forward-only rollout (start → goal, no bound clipping) ────────
            # The field guides the robot through the trained region; outside it
            # get_next_waypoint falls back to geometric steering.  Robot replans
            # incrementally so the path improves as it moves closer.
            for _ in range(max_steps):
                pts_start.append(cur_s.copy())
                if np.linalg.norm(cur_s - cur_g) <= max(step_size, 0.03):
                    cur_s = cur_g.copy()
                    break
                nxt_s, moved = _advance(cur_s, cur_g, clip_bound=False)
                if moved:
                    field_steps += 1
                else:
                    break
                cur_s = nxt_s

            pts_start.append(cur_s.copy())

            total_iters = len(pts_start) - 1
            if total_iters > 0:
                pct = int(100 * field_steps // max(1, total_iters))
                print(f"[PATH fwd] field={field_steps}/{total_iters} ({pct}%)")

            if not pts_start:
                return None

            path_arr = [np.asarray(p, dtype=np.float32) for p in pts_start]

        else:
            # ── Bidirectional rollout ─────────────────────────────────────────
            for _ in range(max_steps):
                pts_start.append(cur_s.copy())
                pts_goal.append(cur_g.copy())

                if np.linalg.norm(cur_s - cur_g) <= max(step_size, 0.03):
                    break

                nxt_s, ok_s = _advance(cur_s, cur_g)
                nxt_g, ok_g = _advance(cur_g, cur_s)

                if ok_s:
                    field_steps += 1
                if ok_g:
                    field_steps += 1

                if (np.linalg.norm(nxt_s - cur_s) < 1e-5 and
                        np.linalg.norm(nxt_g - cur_g) < 1e-5):
                    break

                cur_s = nxt_s
                cur_g = nxt_g

            pts_start.append(cur_s.copy())
            pts_goal.append(cur_g.copy())

            total_iters = len(pts_start) + len(pts_goal) - 2
            if total_iters > 0:
                pct = int(100 * field_steps // max(1, total_iters))
                print(f"[PATH bi] field={field_steps}/{total_iters} ({pct}%)")

            right = list(reversed(pts_goal))
            if not pts_start:
                path = right
            elif not right:
                path = pts_start
            else:
                gap = float(np.linalg.norm(pts_start[-1] - right[0]))
                if gap <= max(step_size, 0.03):
                    path = pts_start + right
                else:
                    start_travel = (
                        float(np.linalg.norm(pts_start[-1] - pts_start[0]))
                        if len(pts_start) > 1 else 0.0
                    )
                    if start_travel >= step_size * 3:
                        return [np.asarray(p, dtype=np.float32)
                                for p in pts_start]
                    return None

            path_arr = [np.asarray(p, dtype=np.float32) for p in path]

        # ── Common validation ─────────────────────────────────────────────────
        goal_dist = float(np.linalg.norm(
            np.array(start_xy, dtype=np.float32) -
            np.array(goal_xy, dtype=np.float32)))
        if len(path_arr) >= 2:
            total_len = sum(
                float(np.linalg.norm(path_arr[i + 1] - path_arr[i]))
                for i in range(len(path_arr) - 1)
            )
        else:
            total_len = 0.0

        if total_len < max(step_size * 2, 0.3) and goal_dist > 0.5:
            return None

        return path_arr

    # ------------------------------------------------------------------
    # Visualization helpers
    # ------------------------------------------------------------------

    def _view_grid_world(self, robot_xy, view_radius=2.0, n=80):
        b = self.bound.detach().cpu().numpy()
        rx, ry = float(robot_xy[0]), float(robot_xy[1])
        xmin_b, ymin_b = float(b[0, 0]), float(b[0, 1])
        xmax_b, ymax_b = float(b[1, 0]), float(b[1, 1])
        xmin_v, xmax_v = rx - view_radius, rx + view_radius
        ymin_v, ymax_v = ry - view_radius, ry + view_radius
        xmin = max(xmin_b, xmin_v)
        xmax = min(xmax_b, xmax_v)
        ymin = max(ymin_b, ymin_v)
        ymax = min(ymax_b, ymax_v)

        if not (xmin < xmax and ymin < ymax):
            xs = np.linspace(xmin_v, xmax_v, n)
            ys = np.linspace(ymin_v, ymax_v, n)
            X_w, Y_w = np.meshgrid(xs, ys)
            sx = (xmax_b - xmin_b) + 1e-9
            sy = (ymax_b - ymin_b) + 1e-9
            X_n = (X_w - xmin_b) / sx - 0.5
            Y_n = (Y_w - ymin_b) / sy - 0.5
            inb = np.zeros_like(X_w, dtype=bool)
            return X_w, Y_w, X_n, Y_n, inb

        xs = np.linspace(xmin, xmax, n)
        ys = np.linspace(ymin, ymax, n)
        X_w, Y_w = np.meshgrid(xs, ys)
        sx = (xmax_b - xmin_b) + 1e-9
        sy = (ymax_b - ymin_b) + 1e-9
        X_n = (X_w - xmin_b) / sx - 0.5
        Y_n = (Y_w - ymin_b) / sy - 0.5
        inb = np.ones_like(X_w, dtype=bool)
        return X_w, Y_w, X_n, Y_n, inb

    def evaluate_field_grid(self, src_world, view_radius=6.0, n=121):
        """Travel time from src_world and speed on an n x n grid around it.

        Returns (X_w, Y_w, TT, V, inb) as n x n arrays, or None when the
        model is not ready or the view does not overlap the region bound.
        """
        if self.model.function is None:
            return None
        src_world = np.asarray(src_world, dtype=np.float32)
        src_z = float(src_world[2]) if len(src_world) > 2 else 0.0
        src_norm = self.normalize(
            np.array([[src_world[0], src_world[1], src_z]], dtype=np.float32))[0]

        X_w, Y_w, X_n, Y_n, inb = self._view_grid_world(
            src_world[:2], view_radius=view_radius, n=n)
        if not inb.any():
            return None

        XP = np.zeros((X_w.size, 2 * self.model.dim), dtype=np.float32)
        XP[:, :self.model.dim] = src_norm
        XP[:, self.model.dim:] = src_norm
        XP[:, 0] = X_n.reshape(-1)
        XP[:, 1] = Y_n.reshape(-1)

        XP_t = torch.tensor(XP, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            TT = (self.model.function.TravelTimes(XP_t)
                  .detach().cpu().numpy().reshape(X_w.shape))
        XP_t2 = XP_t.clone().requires_grad_(True)
        with torch.enable_grad():
            V = (self.model.function.Speed(XP_t2)
                 .detach().cpu().numpy().reshape(X_w.shape))
        return X_w, Y_w, TT, V, inb

    def _denorm_points(self, pts_norm):
        b = self.bound.detach().cpu().numpy()
        scale = (b[1] - b[0])
        return (pts_norm + 0.5) * scale + b[0]

    def find_safe_waypoint(self, robot_xy, frontier_xy,
                           collision_fn=None, n=40, bias=0.5):
        """Find the highest-speed point in the field, biased toward frontier_xy.

        Returns np.ndarray([x, y]) or None.
        """
        if self.model.function is None:
            return None
        robot_xy = np.asarray(robot_xy[:2], dtype=np.float32)
        frontier_xy = np.asarray(frontier_xy[:2], dtype=np.float32)

        # Cover full region bound
        b = self.bound.detach().cpu().numpy()
        cx = float((b[0, 0] + b[1, 0]) / 2)
        cy = float((b[0, 1] + b[1, 1]) / 2)
        vr = max(float(b[1, 0] - b[0, 0]), float(b[1, 1] - b[0, 1])) / 2 + 0.1

        X_w, Y_w, X_n, Y_n, inb = self._view_grid_world(
            [cx, cy], view_radius=vr, n=n)
        if inb is None or not inb.any():
            return None

        # Build speed query tensor (same pattern as save_debug_plot)
        src_norm = self.normalize(
            np.array([[robot_xy[0], robot_xy[1], 0.0]], dtype=np.float32))[0]
        XP = np.zeros((X_w.size, 2 * self.model.dim), dtype=np.float32)
        XP[:, :self.model.dim] = src_norm
        XP[:, self.model.dim:] = src_norm
        XP[:, 0] = X_n.reshape(-1)
        XP[:, 1] = Y_n.reshape(-1)
        XP[:, 2] = src_norm[2]

        try:
            XP_t = torch.tensor(
                XP, dtype=torch.float32, device=self.device
            ).requires_grad_(True)
            with torch.enable_grad():
                V_t = self.model.function.Speed(XP_t)
            V = V_t.detach().cpu().numpy().reshape(X_w.shape)
        except Exception:
            return None

        # Flatten for scoring
        xf = X_w.reshape(-1)
        yf = Y_w.reshape(-1)
        vf = V.reshape(-1)
        inb_f = inb.reshape(-1)

        # Filter: in-bound, finite speed, min distance from robot
        valid = inb_f & np.isfinite(vf)
        dist_to_robot = np.sqrt((xf - robot_xy[0])**2 + (yf - robot_xy[1])**2)
        valid &= dist_to_robot > 0.3

        # Filter: collision check
        if collision_fn is not None:
            for i in np.where(valid)[0]:
                if collision_fn(np.array([xf[i], yf[i]])):
                    valid[i] = False

        if not valid.any():
            return None

        # Directional bias toward frontier
        fdir = frontier_xy - robot_xy
        fdir_norm = float(np.linalg.norm(fdir))
        if fdir_norm > 1e-6:
            fdir = fdir / fdir_norm
        else:
            fdir = np.zeros(2, dtype=np.float32)

        dx = xf[valid] - robot_xy[0]
        dy = yf[valid] - robot_xy[1]
        dot = dx * fdir[0] + dy * fdir[1]
        dot_max = max(float(np.abs(dot).max()), 1e-6)
        direction_score = dot / dot_max  # [-1, 1]

        score = vf[valid] * (1.0 + bias * np.clip(direction_score, 0, 1))
        best_idx = np.argmax(score)
        valid_indices = np.where(valid)[0]
        gi = valid_indices[best_idx]
        return np.array([xf[gi], yf[gi]], dtype=np.float32)

    def _goal_slice_gradient_overlay(self, goal_xy, sample_xy, stride=6):
        if self.model.function is None:
            return None
        sample_xy = np.asarray(sample_xy, dtype=np.float32)
        if sample_xy.ndim != 3 or sample_xy.shape[-1] != 2:
            return None

        xs = sample_xy[::stride, ::stride, 0]
        ys = sample_xy[::stride, ::stride, 1]
        if xs.size == 0:
            return None

        goal_xy = np.asarray(goal_xy[:2], dtype=np.float32)
        z = np.zeros((xs.size, 1), dtype=np.float32)
        x0 = np.stack([xs.reshape(-1), ys.reshape(-1)], axis=1).astype(np.float32)
        x1 = np.repeat(goal_xy.reshape(1, 2), len(x0), axis=0).astype(np.float32)
        xp = np.hstack([x0, z, x1, z])

        xp_t = torch.tensor(xp, dtype=torch.float32, device=self.device)
        xp_n = self.normalize_tensor(xp_t).requires_grad_(True)
        try:
            tau, _, coords = self.model.network.out(xp_n)
            grad = torch.autograd.grad(
                outputs=tau.sum(),
                inputs=coords,
                retain_graph=False,
                create_graph=False,
            )[0]
        except Exception:
            return None

        grad_xy_norm = grad[:, :2].detach().cpu().numpy()
        b = self.bound.detach().cpu().numpy()
        bound_len = (b[1, :2] - b[0, :2]).astype(np.float32)
        grad_xy_world = grad_xy_norm / (bound_len[None, :] + 1e-8)
        flow = -grad_xy_world
        mag = np.linalg.norm(flow, axis=1, keepdims=True)
        good = mag[:, 0] > 1e-6
        flow[good] /= mag[good]
        flow[~good] = 0.0
        return xs, ys, flow[:, 0].reshape(xs.shape), flow[:, 1].reshape(xs.shape)

    def save_debug_plot(self, src_world, view_radius=2.0, n=80, robot_xy=None):
        """Save speed-field diagnostic. Runs in calling thread — no daemon, no lock."""
        if self.model.function is None:
            return

        robot_x = float(src_world[0])
        robot_y = float(src_world[1])
        robot_z = float(src_world[2]) if len(src_world) > 2 else 0.0
        robot_x_dot = float(robot_xy[0]) if robot_xy is not None else robot_x
        robot_y_dot = float(robot_xy[1]) if robot_xy is not None else robot_y

        src_norm = self.normalize(
            np.array([[robot_x, robot_y, robot_z]], dtype=np.float32))[0]

        goal_xy = getattr(self, 'latest_goal', None)
        if goal_xy is not None:
            goal_xy = np.asarray(goal_xy[:2], dtype=np.float32)
        path = getattr(self, 'latest_path', None)
        if goal_xy is None and path is not None and len(path) > 0:
            goal_xy = np.asarray(path[-1][:2], dtype=np.float32)
        if goal_xy is None:
            goal_xy = np.array([robot_x_dot, robot_y_dot], dtype=np.float32)
        goal_norm = self.normalize(
            np.array([[goal_xy[0], goal_xy[1], robot_z]], dtype=np.float32))[0]

        X_w, Y_w, X_n, Y_n, inb = self._view_grid_world(
            [robot_x, robot_y], view_radius=view_radius, n=n)
        if inb is None or (hasattr(inb, 'any') and not inb.any()):
            return

        XP = np.zeros((X_w.size, 2 * self.model.dim), dtype=np.float32)
        XP[:, :self.model.dim] = goal_norm
        XP[:, self.model.dim:] = goal_norm
        XP[:, 0] = X_n.reshape(-1)
        XP[:, 1] = Y_n.reshape(-1)
        XP[:, 2] = src_norm[2]

        try:
            XP_t = torch.tensor(XP, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                TT = (self.model.function.TravelTimes(XP_t)
                      .detach().cpu().numpy().reshape(X_w.shape))
            XP_t2 = torch.tensor(
                XP, dtype=torch.float32, device=self.device).requires_grad_(True)
            with torch.enable_grad():
                V_t = self.model.function.Speed(XP_t2)
            V = V_t.detach().cpu().numpy().reshape(X_w.shape)
        except Exception as e:
            print(f"[DEBUG PLOT] inference failed: {e}")
            return

        cmap_speed = plt.get_cmap('viridis').copy()
        cmap_speed.set_bad(alpha=0.0)

        fig, ax = plt.subplots(figsize=(9, 9))
        m0 = ax.pcolormesh(X_w, Y_w, V, vmin=0.0, vmax=1.0,
                           cmap=cmap_speed, shading='auto')
        ax.contour(X_w, Y_w, TT, np.arange(0, 10, 0.03),
                   cmap='gist_heat', linewidths=0.5, alpha=0.8)
        fig.colorbar(m0, ax=ax, pad=0.02)
        ax.set_title(f"Speed field  epoch={self.epoch}")
        ax.scatter(robot_x_dot, robot_y_dot, c='red', s=120, zorder=6,
                   marker='o', label='Robot')

        if path is not None and len(path) > 1:
            path_arr = np.array(path)
            ax.plot(path_arr[:, 0], path_arr[:, 1],
                    color='cyan', linewidth=1.5, zorder=7,
                    linestyle='--', label='Path')
            ax.scatter(goal_xy[0], goal_xy[1], c='lime', s=110,
                       zorder=6, marker='*', label='Goal')

        ax.set_xlim(float(np.min(X_w)), float(np.max(X_w)))
        ax.set_ylim(float(np.min(Y_w)), float(np.max(Y_w)))
        ax.set_aspect('equal')
        ax.legend(fontsize=7)
        fig.suptitle(
            f"Region {self.version} | Epoch {self.epoch} | "
            f"Robot ({robot_x:.2f}, {robot_y:.2f})",
            fontsize=12,
        )

        out_dir = os.path.join(self.folder, "debug")
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(os.path.join(out_dir, f"field_{self.epoch:05d}.png"),
                    dpi=72, bbox_inches='tight')
        fig.savefig(os.path.join(self.folder, "field.png"),
                    dpi=72, bbox_inches='tight')
        plt.close(fig)

    def save_training_vis(self, points, speeds, robot_xy=None, view_radius=2.0):
        """Save training data visualization. Runs in calling thread — no lock."""
        if len(points) == 0:
            return

        pts = points.detach().cpu().numpy()
        spd = speeds.detach().cpu().numpy()

        x0_world = self._denorm_points(pts[:, :3])
        x1_world = self._denorm_points(pts[:, 3:6])
        dists_world = np.linalg.norm(x0_world[:, :2] - x1_world[:, :2], axis=1)
        far_enough = dists_world >= 0.05

        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        cmap = cm.get_cmap('viridis')
        vmin, vmax = float(spd.min()), float(spd.max())

        ax[0].scatter(x0_world[:, 0], x0_world[:, 1], c=spd[:, 0],
                      cmap=cmap, vmin=vmin, vmax=vmax, s=6, alpha=0.7)
        ax[0].set_title(f"Start x0+y0 (N={len(pts)})")
        ax[0].set_aspect('equal')

        x1_vis = x1_world[far_enough]
        spd1_vis = spd[far_enough, 1]
        sort_idx = np.argsort(spd1_vis)
        ax[1].scatter(x1_vis[sort_idx, 0], x1_vis[sort_idx, 1],
                      c=spd1_vis[sort_idx], cmap=cmap, vmin=vmin, vmax=vmax,
                      s=6, alpha=0.7)
        ax[1].set_title(f"Goal x1+y1 (N={far_enough.sum()})")
        ax[1].set_aspect('equal')

        ax[2].hist(dists_world, bins=40, alpha=0.8, color='orange', density=True)
        ax[2].axvline(dists_world.mean(), color='red', lw=2,
                      label=f"mu={dists_world.mean():.2f}m")
        ax[2].set_xlabel("World Distance (m)")
        ax[2].set_title("x0->x1 Dist")
        ax[2].legend()

        fig.colorbar(
            cm.ScalarMappable(
                cmap=cmap, norm=colors.Normalize(vmin=vmin, vmax=vmax)),
            ax=ax, label="Norm Speed", shrink=0.8,
        )

        b = self.bound.detach().cpu().numpy()
        xmin, ymin = float(b[0, 0]), float(b[0, 1])
        xmax, ymax = float(b[1, 0]), float(b[1, 1])
        for axi in ax[:2]:
            rect = plt.Rectangle(
                (b[0, 0], b[0, 1]), b[1, 0] - b[0, 0], b[1, 1] - b[0, 1],
                linewidth=2, edgecolor='white', facecolor='none', zorder=10)
            axi.add_patch(rect)
            axi.set_xlim(xmin, xmax)
            axi.set_ylim(ymin, ymax)

        out = os.path.join(self.folder, f"train_data_{self.epoch:03d}.png")
        fig.savefig(out, dpi=72, bbox_inches='tight')
        plt.close(fig)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _write_region_bundle(self):
        self._region_paths()
        torch.save(self.model.network.state_dict(), self.network_save_path)
        torch.save(self.model.B, self.B_save_path)
        torch.save({
            'frame_data': self.all_raw_frame_data,
            'all_lidar': self.all_lidar,
            'semantic_objects': getattr(self, 'semantic_objects', {}),
            'epoch': self.epoch,
            'train_tick_ms': list(self.train_tick_ms),
            'loss_history': list(self.loss_history),
        }, self.data_save_path)
        meta = {
            'version': self.version,
            'bound': [[float(x) for x in row]
                      for row in self.bound.detach().cpu().tolist()],
            'epoch': int(self.epoch),
            'doors': list(getattr(self, 'doors_id', [])),
            'rooms': [room.id for room in getattr(self, 'rooms', [])],
        }
        with open(self.meta_save_path, 'w') as f:
            json.dump(meta, f)

    def save_model(self):
        if self.model.network is None:
            return
        self._write_region_bundle()
        print(f"[Region {self.version}] saved to {self.folder}")

    def save_checkpoint(self, epoch: int) -> None:
        """Save a numbered snapshot of network weights for later comparison."""
        if self.model.network is None:
            return
        ckpt_dir = os.path.join(self.folder, 'checkpoints')
        os.makedirs(ckpt_dir, exist_ok=True)
        path = os.path.join(ckpt_dir, f'ep{epoch:05d}_network.pt')
        torch.save(self.model.network.state_dict(), path)
        print(f"[Region {self.version}] checkpoint saved: ep{epoch:05d}")

    def unload_to_disk(self):
        self._write_region_bundle()
        self.model.delete_network()
        self._sync_model_refs()
        self.saved = True
        self.loaded = False
        self.all_raw_frame_data = []
        self.all_lidar = []

    def load_from_disk(self):
        assert self.saved
        self._region_paths()
        device = self.device

        self.model.B = torch.load(
            self.B_save_path, map_location=device, weights_only=False)
        self.model.network = model_network.NN(device, self.model.dim, self.model.B).to(device)
        self.model.network.load_state_dict(
            torch.load(self.network_save_path, map_location=device, weights_only=False))

        try:
            loaded = torch.load(
                self.data_save_path,
                map_location=device,
                weights_only=False,
            )
            if isinstance(loaded, dict):
                self.all_raw_frame_data = loaded.get('frame_data', [])
                self.all_lidar = loaded.get('all_lidar', [])
                self.semantic_objects = loaded.get('semantic_objects', {})
                self.epoch = int(loaded.get('epoch', self.epoch))
                self.train_tick_ms = list(loaded.get('train_tick_ms', []))
                self.loss_history = list(loaded.get('loss_history', []))
            else:
                self.all_raw_frame_data = loaded if isinstance(loaded, list) else []
                self.all_lidar = []
        except Exception as e:
            print(f"[load_from_disk] could not load data: {e}")
            self.all_raw_frame_data = []
            self.all_lidar = []

        if os.path.exists(self.meta_save_path):
            try:
                with open(self.meta_save_path, 'r') as f:
                    meta = json.load(f)
                self.bound = torch.tensor(
                    meta.get('bound', self.bound.tolist()),
                    dtype=torch.float32)
                self.epoch = int(meta.get('epoch', self.epoch))
            except Exception as e:
                print(f"[load_from_disk] could not load meta: {e}")

        self.model.function = model_function.Function(
            self.folder, device, self.model.network, self.model.dim)
        lr = self.model.Params['Training'].get('Learning Rate', 1e-3)
        self.model.optimizer = torch.optim.AdamW(
            self.model.network.parameters(), lr=lr, weight_decay=0.1)
        self.model.scheduler = None
        self.saved = False
        self.loaded = True
        self._sync_model_refs()

    def load_model(self):
        self._region_paths()
        if not (self.network_save_path and
                os.path.exists(self.network_save_path) and
                os.path.exists(self.B_save_path) and
                os.path.exists(self.data_save_path)):
            print(f"[Region {self.version}] incomplete bundle in {self.folder}")
            return False
        self.saved = True
        self.load_from_disk()
        print(f"[Region {self.version}] loaded from epoch {self.epoch}")
        return True

    def destroy(self):
        if self.loaded:
            self.model.delete_network()
            self._sync_model_refs()
SEMANTIC_FURNITURE_CLASSES = {
    "table", "chair", "bed", "sofa", "cabinet", "other_furniture"
}

# Confirmation gate for semantic landmarks. A landmark is only treated as
# real by navigation / retraining / miss-counting once it has been detected
# at least this many times AND its plurality class has at least this fraction
# of the total votes. Keeps one-frame misclassifications from moving the map.
CONFIRM_MIN_COUNT = 3
CONFIRM_MIN_PLURALITY = 0.5


def is_semantic_entry_confirmed(entry) -> bool:
    """Return True if a stored semantic landmark has passed the confirmation gate.

    Legacy plain [x, y] entries have no vote history and are treated as
    unconfirmed so they do not drive retraining until re-observed.
    """
    if not isinstance(entry, dict):
        return False
    count = int(entry.get("count", 0))
    if count < CONFIRM_MIN_COUNT:
        return False
    votes = entry.get("class_votes") or {}
    if not votes:
        return False
    total = sum(int(v) for v in votes.values())
    if total <= 0:
        return False
    winner = max(votes.values())
    return int(winner) >= CONFIRM_MIN_PLURALITY * total
