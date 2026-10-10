#!/usr/bin/env python3
"""
field_training_diagnostics.py

Offline diagnostics for trained region fields.

Evaluates each region on random in-region start/goal pairs and reports:
  - returned_none / too_short
  - out_of_bounds
  - collides_with_occupied_cells
  - reached_goal
  - valid_success
"""

import argparse
import glob
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
import yaml

from mntfields.mntfields_core.core.region import Region


class DummyModel:
    def __init__(self, model_dir: str):
        self.folder = model_dir
        self.renderer = "cuda:0"
        self.h = 0.0


@dataclass
class MapData:
    occ: np.ndarray
    blocked: np.ndarray
    resolution: float
    origin_xy: Tuple[float, float]


class FieldDiagnostics:
    def __init__(
        self,
        model_dir: str,
        map_yaml: str,
        step_size: float,
        max_steps: int,
        samples: int,
        boundary_samples: int,
        goal_tol: float,
        inflation_cells: int,
        boundary_band_m: float,
        seed: int,
        rollout_mode: str,
    ):
        self.model_dir = model_dir
        self.map_yaml = map_yaml
        self.step_size = step_size
        self.max_steps = max_steps
        self.samples = samples
        self.boundary_samples = boundary_samples
        self.goal_tol = goal_tol
        self.inflation_cells = inflation_cells
        self.boundary_band_m = boundary_band_m
        self.rng = np.random.default_rng(seed)
        self.rollout_mode = rollout_mode

        self.regions: Dict[int, Region] = {}
        self.map_data: Optional[MapData] = None

    def run(self) -> dict:
        self._load_regions()
        self._load_map()
        if not self.regions:
            raise RuntimeError("No regions loaded.")
        if self.map_data is None:
            raise RuntimeError("Map failed to load.")

        per_region = {}
        for ver in sorted(self.regions.keys()):
            per_region[str(ver)] = self._evaluate_region(ver)

        global_summary = self._aggregate_global(per_region)
        return {
            "model_dir": self.model_dir,
            "map_yaml": self.map_yaml,
            "config": {
                "step_size": self.step_size,
                "max_steps": self.max_steps,
                "samples": self.samples,
                "boundary_samples": self.boundary_samples,
                "goal_tol": self.goal_tol,
                "inflation_cells": self.inflation_cells,
                "boundary_band_m": self.boundary_band_m,
                "rollout_mode": self.rollout_mode,
            },
            "global": global_summary,
            "regions": per_region,
        }

    def _suite_modes(self) -> List[str]:
        if self.rollout_mode == "both":
            return ["bidirectional", "oneway"]
        return [self.rollout_mode]

    def _load_regions(self):
        dummy = DummyModel(self.model_dir)
        for region_dir in sorted(glob.glob(os.path.join(self.model_dir, "region_*"))):
            name = os.path.basename(region_dir)
            try:
                ver = int(name.split("_")[1])
            except Exception:
                continue
            ckpt = os.path.join(region_dir, f"region_{ver}_checkpoint.pt")
            if not os.path.isfile(ckpt):
                continue
            data = torch.load(ckpt, map_location="cuda:0")
            bound = data.get("bound")
            if bound is None:
                continue
            if isinstance(bound, torch.Tensor):
                bound = bound.cpu()
            reg = Region(
                model=dummy,
                bound=bound,
                version=ver,
                init_network=True,
                eval=False,
            )
            reg.load_model()
            self.regions[ver] = reg

    @staticmethod
    def _inflate_occ(occ: np.ndarray, pad: int) -> np.ndarray:
        if pad <= 0:
            return occ.copy()
        h, w = occ.shape
        out = np.zeros_like(occ, dtype=bool)
        idx = np.argwhere(occ)
        for r, c in idx:
            r0 = max(0, r - pad)
            r1 = min(h, r + pad + 1)
            c0 = max(0, c - pad)
            c1 = min(w, c + pad + 1)
            out[r0:r1, c0:c1] = True
        return out

    def _load_map(self):
        with open(self.map_yaml, "r", encoding="utf-8") as fh:
            meta = yaml.safe_load(fh)
        image_path = meta["image"]
        if not os.path.isabs(image_path):
            image_path = os.path.join(os.path.dirname(self.map_yaml), image_path)
        arr = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32) / 255.0
        negate = int(meta.get("negate", 0))
        occ_prob = arr if negate else (1.0 - arr)
        occ_thresh = float(meta.get("occupied_thresh", 0.65))
        occ = occ_prob >= occ_thresh
        blocked = self._inflate_occ(occ, self.inflation_cells)
        self.map_data = MapData(
            occ=occ,
            blocked=blocked,
            resolution=float(meta["resolution"]),
            origin_xy=(float(meta["origin"][0]), float(meta["origin"][1])),
        )

    def _map_indices(self, xy: np.ndarray) -> Optional[Tuple[int, int]]:
        md = self.map_data
        assert md is not None
        mx = int(math.floor((float(xy[0]) - md.origin_xy[0]) / md.resolution))
        my = int(math.floor((float(xy[1]) - md.origin_xy[1]) / md.resolution))
        h, w = md.blocked.shape
        row = h - 1 - my
        col = mx
        if not (0 <= row < h and 0 <= col < w):
            return None
        return row, col

    def _cell_to_world(self, row: int, col: int) -> np.ndarray:
        md = self.map_data
        assert md is not None
        h, _ = md.blocked.shape
        mx = float(col) + 0.5
        my = float(h - 1 - row) + 0.5
        x = md.origin_xy[0] + mx * md.resolution
        y = md.origin_xy[1] + my * md.resolution
        return np.array([x, y], dtype=np.float32)

    def _point_in_collision(self, xy: np.ndarray) -> bool:
        md = self.map_data
        assert md is not None
        rc = self._map_indices(xy)
        if rc is None:
            return False
        return bool(md.blocked[rc[0], rc[1]])

    def _segment_hits_wall(self, p0: np.ndarray, p1: np.ndarray, step: float = 0.05) -> bool:
        dist = float(np.linalg.norm(p1 - p0))
        n = max(2, int(math.ceil(dist / step)))
        for a in np.linspace(0.0, 1.0, n):
            pt = p0 + a * (p1 - p0)
            if self._point_in_collision(pt):
                return True
        return False

    def _path_hits_wall(self, path_xy: np.ndarray) -> bool:
        if len(path_xy) < 2:
            return False
        for i in range(len(path_xy) - 1):
            if self._segment_hits_wall(path_xy[i], path_xy[i + 1]):
                return True
        return False

    def _point_in_region(self, ver: int, xy: np.ndarray) -> bool:
        b = self.regions[ver].bound.detach().cpu().numpy()
        return bool(b[0, 0] <= xy[0] <= b[1, 0] and b[0, 1] <= xy[1] <= b[1, 1])

    def _sample_free_points(self, ver: int) -> Tuple[np.ndarray, np.ndarray]:
        md = self.map_data
        assert md is not None
        b = self.regions[ver].bound.detach().cpu().numpy()
        h, w = md.blocked.shape
        pts = []
        boundary_pts = []
        for r in range(h):
            for c in range(w):
                if md.blocked[r, c]:
                    continue
                xy = self._cell_to_world(r, c)
                if not self._point_in_region(ver, xy):
                    continue
                pts.append(xy)
                d = min(
                    float(abs(xy[0] - b[0, 0])),
                    float(abs(xy[0] - b[1, 0])),
                    float(abs(xy[1] - b[0, 1])),
                    float(abs(xy[1] - b[1, 1])),
                )
                if d <= self.boundary_band_m:
                    boundary_pts.append(xy)
        pts_np = np.asarray(pts, dtype=np.float32) if pts else np.zeros((0, 2), dtype=np.float32)
        bnd_np = (
            np.asarray(boundary_pts, dtype=np.float32)
            if boundary_pts
            else np.zeros((0, 2), dtype=np.float32)
        )
        return pts_np, bnd_np

    def _rollout(
        self, reg: Region, s: np.ndarray, g: np.ndarray, mode: str
    ) -> Optional[np.ndarray]:
        if mode == "oneway":
            return reg.generate_field_path_oneway(
                s.astype(np.float32),
                g.astype(np.float32),
                step_size=self.step_size,
                max_steps=self.max_steps,
            )
        return reg.generate_field_path(
            s.astype(np.float32),
            g.astype(np.float32),
            step_size=self.step_size,
            max_steps=self.max_steps,
        )

    def _eval_suite(self, ver: int, starts: np.ndarray, goals: np.ndarray, name: str, mode: str) -> dict:
        reg = self.regions[ver]
        stats = {
            "suite": name,
            "rollout_mode": mode,
            "n": 0,
            "returned_none": 0,
            "too_short": 0,
            "out_of_bounds": 0,
            "collision": 0,
            "reached_goal": 0,
            "valid_success": 0,
        }
        if len(starts) == 0 or len(goals) == 0:
            return stats

        n = self.samples if name == "random_pairs" else self.boundary_samples
        for _ in range(n):
            s = starts[self.rng.integers(0, len(starts))]
            g = goals[self.rng.integers(0, len(goals))]
            if float(np.linalg.norm(s - g)) < 0.25:
                continue
            stats["n"] += 1
            try:
                seg = self._rollout(reg, s, g, mode)
            except Exception:
                seg = None
            if seg is None:
                stats["returned_none"] += 1
                continue
            seg_np = np.asarray(seg, dtype=np.float32)
            if len(seg_np) < 2:
                stats["too_short"] += 1
                continue
            if not np.all([self._point_in_region(ver, p) for p in seg_np]):
                stats["out_of_bounds"] += 1
            if self._path_hits_wall(seg_np):
                stats["collision"] += 1
            if float(np.linalg.norm(seg_np[-1] - g)) <= self.goal_tol:
                stats["reached_goal"] += 1
            if (
                float(np.linalg.norm(seg_np[-1] - g)) <= self.goal_tol
                and self._path_hits_wall(seg_np) is False
                and np.all([self._point_in_region(ver, p) for p in seg_np])
            ):
                stats["valid_success"] += 1
        return stats

    def _evaluate_region(self, ver: int) -> dict:
        pts, bnd = self._sample_free_points(ver)
        random_suite = {}
        boundary_suite = {}
        for mode in self._suite_modes():
            random_suite[mode] = self._eval_suite(ver, pts, pts, "random_pairs", mode)
            boundary_suite[mode] = self._eval_suite(
                ver, pts, (bnd if len(bnd) > 0 else pts), "boundary_targets", mode
            )
        return {
            "free_points": int(len(pts)),
            "boundary_points": int(len(bnd)),
            "random_pairs": random_suite,
            "boundary_targets": boundary_suite,
        }

    def _aggregate_global(self, per_region: dict) -> dict:
        def fold(key: str, mode: str) -> dict:
            out = {
                "n": 0,
                "returned_none": 0,
                "too_short": 0,
                "out_of_bounds": 0,
                "collision": 0,
                "reached_goal": 0,
                "valid_success": 0,
            }
            for _, r in per_region.items():
                s = r[key][mode]
                for k in out:
                    out[k] += int(s.get(k, 0))
            return out

        random_pairs = {}
        boundary_targets = {}
        for mode in self._suite_modes():
            random_pairs[mode] = fold("random_pairs", mode)
            boundary_targets[mode] = fold("boundary_targets", mode)

        return {
            "random_pairs": random_pairs,
            "boundary_targets": boundary_targets,
        }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True)
    p.add_argument("--map_yaml", required=True)
    p.add_argument("--samples", type=int, default=60)
    p.add_argument("--boundary_samples", type=int, default=40)
    p.add_argument("--step_size", type=float, default=0.10)
    p.add_argument("--max_steps", type=int, default=300)
    p.add_argument("--goal_tol", type=float, default=0.30)
    p.add_argument("--inflation_cells", type=int, default=1)
    p.add_argument("--boundary_band_m", type=float, default=0.30)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument(
        "--rollout_mode",
        type=str,
        default="bidirectional",
        choices=["bidirectional", "oneway", "both"],
    )
    p.add_argument("--out_json", type=str, default="")
    args = p.parse_args()

    diag = FieldDiagnostics(
        model_dir=args.model_dir,
        map_yaml=args.map_yaml,
        step_size=args.step_size,
        max_steps=args.max_steps,
        samples=args.samples,
        boundary_samples=args.boundary_samples,
        goal_tol=args.goal_tol,
        inflation_cells=args.inflation_cells,
        boundary_band_m=args.boundary_band_m,
        seed=args.seed,
        rollout_mode=args.rollout_mode,
    )
    report = diag.run()
    print(json.dumps(report, indent=2))
    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)


if __name__ == "__main__":
    main()
