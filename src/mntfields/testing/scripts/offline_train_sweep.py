#!/usr/bin/env python3
import argparse
import csv
import json
import os
import time
from dataclasses import dataclass
from itertools import product
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from mntfields.mntfields_core.core.region import Region
from mntfields.mntfields_core.mapping.lidar_gridmap import LidarGridmap


def _parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _parse_bool_list(s: str) -> List[bool]:
    vals = []
    for x in s.split(","):
        v = x.strip().lower()
        if not v:
            continue
        vals.append(v in ("1", "true", "yes", "y", "t"))
    return vals


@dataclass
class ReplaySample:
    pose: np.ndarray          # (3,)
    hit_points: np.ndarray    # (N,3)


class DummyModel:
    def __init__(self, folder: str, device: str):
        self.folder = folder
        self.renderer = device
        self.h = 0.0


def load_samples(dataset_dir: str) -> Tuple[dict, List[ReplaySample]]:
    meta_path = os.path.join(dataset_dir, "meta.json")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)

    chunks = sorted(
        p for p in os.listdir(dataset_dir)
        if p.startswith("chunk_") and p.endswith(".npz")
    )
    samples: List[ReplaySample] = []
    for c in chunks:
        arr = np.load(os.path.join(dataset_dir, c), allow_pickle=False)
        poses = arr["robot_pose"]
        hit = arr["world_points_hit"]
        hit_counts = arr["world_points_hit_counts"]
        for i in range(len(poses)):
            n = int(hit_counts[i])
            if n <= 0:
                continue
            pts = hit[i, :n, :].astype(np.float32)
            samples.append(ReplaySample(pose=poses[i].astype(np.float32), hit_points=pts))
    return meta, samples


def _segment_hits_wall(gridmap: LidarGridmap, p0: np.ndarray, p1: np.ndarray, step: float = 0.05) -> bool:
    d = float(np.linalg.norm(p1 - p0))
    n = max(2, int(np.ceil(d / step)))
    ts = np.linspace(0.0, 1.0, n, dtype=np.float32)
    pts = p0[None, :] + (p1 - p0)[None, :] * ts[:, None]
    for xy in pts:
        i, j = gridmap._world_to_grid(float(xy[0]), float(xy[1]))
        if gridmap.data[i, j] == 1:
            return True
    return False


def _path_hits_wall(gridmap: LidarGridmap, path_xy: np.ndarray) -> bool:
    if len(path_xy) < 2:
        return True
    for i in range(len(path_xy) - 1):
        if _segment_hits_wall(gridmap, path_xy[i], path_xy[i + 1]):
            return True
    return False


def _save_segmentation_png(gridmap: LidarGridmap, out_path: str):
    occ = gridmap.data
    vis = gridmap.visited
    img = np.full_like(occ, 205, dtype=np.uint8)  # unknown gray
    img[(occ == 0) & (vis == 1)] = 254            # free white
    img[occ == 1] = 0                             # occupied black
    plt.figure(figsize=(8, 8))
    plt.imshow(np.flipud(img), cmap="gray", vmin=0, vmax=255)
    plt.title("Offline Replay Occupancy")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def run_config(
    cfg: dict,
    samples: List[ReplaySample],
    meta: dict,
    run_dir: str,
    device: str,
    eval_pairs: int,
    eval_goal_tol: float,
    max_samples: int,
) -> dict:
    t0 = time.time()
    n_cells = int(meta.get("n_cells", 400))
    near_dist = float(meta.get("near_dist", 2.0))
    map_size = float(meta.get("map_size", 40.0))

    gridmap = LidarGridmap(
        trav_map=np.ones((n_cells, n_cells), dtype=np.uint8),
        regions={},
        n_cells=n_cells,
        near_dist=near_dist,
        room_seg_config=(cfg["room_seg_kernel"], cfg["room_seg_iters"]),
        map_size=map_size,
        origin=(0.0, 0.0),
    )
    gridmap.set_door_match_radius(radius_m=None, radius_cells=cfg["door_match_radius_cells"])
    if samples:
        gridmap.origin = samples[0].pose[:2].copy()

    model = DummyModel(folder=run_dir, device=device)
    bound = torch.tensor([
        [gridmap.origin[0] - map_size / 2.0, gridmap.origin[1] - map_size / 2.0, -0.1],
        [gridmap.origin[0] + map_size / 2.0, gridmap.origin[1] + map_size / 2.0,  0.1],
    ], dtype=torch.float32)
    region = Region(model=model, bound=bound, version=1, init_network=True, eval=False)
    region.gridmap = gridmap
    region.strict_pair_validity = bool(cfg["strict_pair_validity"])
    region.enable_debug_vis = False

    last_pts = None
    last_pose = None
    candidate_train_frames = 0
    train_frames = 0
    replay_samples = samples if max_samples <= 0 else samples[:max_samples]
    for idx, s in enumerate(replay_samples):
        pts = s.hit_points
        if len(pts) < 5:
            continue
        speeds = np.ones(len(pts), dtype=np.float32)
        do_room = ((idx % int(cfg["update_room_every_n"])) == 0)
        try:
            gridmap.update(
                lidar_scan=pts[:, :2],
                points=pts[:, :2],
                speeds=speeds,
                cur_pos=s.pose[:2],
                region=None,
                regions={},
                update_room=do_room,
            )
        except Exception:
            # Keep replay robust; region segmentation can fail on sparse frames.
            gridmap.update(
                lidar_scan=pts[:, :2],
                points=pts[:, :2],
                speeds=speeds,
                cur_pos=s.pose[:2],
                region=None,
                regions={},
                update_room=False,
            )

        if idx % int(cfg["train_every_n_frames"]) != 0:
            continue
        data = region.get_data_from_lidar(
            lidar_scan=pts,
            cur_pos=s.pose,
            device=device,
            n_samples=int(cfg["train_samples"]),
        )
        if data is None:
            continue
        points, closest = data
        if points is None or closest is None or len(points) == 0:
            continue
        candidate_train_frames += 1
        raw = torch.cat([points, closest], dim=1)
        prev_epoch = int(region.epoch)
        region.train_core(epoch=int(cfg["train_epoch"]), raw_frame_data=raw)
        if int(region.epoch) > prev_epoch:
            train_frames += 1
            last_pts = pts
            last_pose = s.pose

    # Eval from visited free cells.
    free = np.argwhere((gridmap.visited == 1) & (gridmap.data == 0))
    pair_trials = min(int(eval_pairs), max(0, len(free) // 2))
    success = 0
    non_wall_success = 0
    end_errors = []
    trained = (int(region.epoch) > 0 and train_frames > 0)
    if trained and pair_trials > 0:
        rng = np.random.default_rng(0)
        for _ in range(pair_trials):
            i0 = int(rng.integers(0, len(free)))
            i1 = int(rng.integers(0, len(free)))
            if i0 == i1:
                continue
            s_ij = free[i0]
            g_ij = free[i1]
            sx, sy = gridmap._grid_to_world(int(s_ij[0]), int(s_ij[1]))
            gx, gy = gridmap._grid_to_world(int(g_ij[0]), int(g_ij[1]))
            sxy = np.array([sx, sy], dtype=np.float32)
            gxy = np.array([gx, gy], dtype=np.float32)
            path = region.generate_field_path(sxy, gxy, step_size=0.15, max_steps=100)
            if path is None:
                continue
            p = np.asarray(path, dtype=np.float32)
            if len(p) < 2:
                continue
            err = float(np.linalg.norm(p[-1] - gxy))
            end_errors.append(err)
            if err <= float(eval_goal_tol):
                success += 1
            if err <= float(eval_goal_tol) and (not _path_hits_wall(gridmap, p)):
                non_wall_success += 1

    # Artifacts
    os.makedirs(run_dir, exist_ok=True)
    _save_segmentation_png(gridmap, os.path.join(run_dir, "occupancy.png"))
    if last_pts is not None:
        try:
            region.save_debug_plot(last_pts, view_radius=3.5, robot_xy=last_pose[:2] if last_pose is not None else None)
        except Exception:
            pass
    region.save_model()

    explored_ratio = float(gridmap.visited.sum()) / float(gridmap.visited.size + 1e-9)
    occupied_ratio = float((gridmap.data == 1).sum()) / float(gridmap.data.size + 1e-9)
    wall_safe_rate = float(non_wall_success) / float(max(1, pair_trials)) if trained else 0.0
    goal_rate = float(success) / float(max(1, pair_trials)) if trained else 0.0
    mean_end_error = (float(np.mean(end_errors)) if (trained and end_errors) else 1e9)
    elapsed = float(time.time() - t0)

    return {
        **cfg,
        "samples_total": len(replay_samples),
        "candidate_train_frames": candidate_train_frames,
        "train_frames_used": train_frames,
        "region_epoch": int(region.epoch),
        "explored_ratio": explored_ratio,
        "occupied_ratio": occupied_ratio,
        "goal_reach_rate": goal_rate,
        "wall_safe_success_rate": wall_safe_rate,
        "mean_goal_error_m": mean_end_error,
        "elapsed_sec": elapsed,
        "score": (2.0 * wall_safe_rate + 1.0 * goal_rate) - 0.05 * min(mean_end_error, 10.0),
        "run_dir": run_dir,
    }


def main():
    ap = argparse.ArgumentParser(description="Offline mntfields replay + training sweep.")
    ap.add_argument("--dataset_dir", required=True, help="Path to one session_* directory.")
    ap.add_argument("--out_dir", default="", help="Output dir. Default: <dataset_dir>/train_sweep_results")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--room_seg_kernel_values", default="7")
    ap.add_argument("--room_seg_iter_values", default="2")
    ap.add_argument("--door_match_radius_cells_values", default="6")
    ap.add_argument("--strict_pair_validity_values", default="true,false")
    ap.add_argument("--train_samples_values", default="2500")
    ap.add_argument("--train_epoch_values", default="20")
    ap.add_argument("--train_every_n_frames_values", default="2")
    ap.add_argument("--update_room_every_n", type=int, default=10)
    ap.add_argument("--max_samples", type=int, default=0, help="0 = use all samples")
    ap.add_argument("--eval_pairs", type=int, default=30)
    ap.add_argument("--eval_goal_tol", type=float, default=0.25)
    args = ap.parse_args()

    dataset_dir = os.path.abspath(args.dataset_dir)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.join(dataset_dir, "train_sweep_results")
    os.makedirs(out_dir, exist_ok=True)

    meta, samples = load_samples(dataset_dir)
    if not samples:
        raise SystemExit(f"No chunk samples found in: {dataset_dir}")

    kernels = _parse_int_list(args.room_seg_kernel_values)
    iters = _parse_int_list(args.room_seg_iter_values)
    doors = _parse_float_list(args.door_match_radius_cells_values)
    stricts = _parse_bool_list(args.strict_pair_validity_values)
    train_samples_list = _parse_int_list(args.train_samples_values)
    train_epoch_list = _parse_int_list(args.train_epoch_values)
    train_every_list = _parse_int_list(args.train_every_n_frames_values)

    configs = []
    for k, itv, d, s, ns, ep, te in product(
        kernels, iters, doors, stricts, train_samples_list, train_epoch_list, train_every_list
    ):
        configs.append({
            "room_seg_kernel": int(k),
            "room_seg_iters": int(itv),
            "door_match_radius_cells": float(d),
            "strict_pair_validity": bool(s),
            "train_samples": int(ns),
            "train_epoch": int(ep),
            "train_every_n_frames": int(te),
            "update_room_every_n": int(args.update_room_every_n),
        })

    results = []
    for i, cfg in enumerate(configs, start=1):
        run_name = (
            f"cfg_{i:03d}"
            f"_k{cfg['room_seg_kernel']}"
            f"_it{cfg['room_seg_iters']}"
            f"_d{int(cfg['door_match_radius_cells'])}"
            f"_sp{int(cfg['strict_pair_validity'])}"
            f"_ns{cfg['train_samples']}"
            f"_ep{cfg['train_epoch']}"
            f"_te{cfg['train_every_n_frames']}"
        )
        run_dir = os.path.join(out_dir, run_name)
        print(f"[SWEEP] {i}/{len(configs)} -> {run_name}")
        try:
            r = run_config(
                cfg=cfg,
                samples=samples,
                meta=meta,
                run_dir=run_dir,
                device=args.device,
                eval_pairs=int(args.eval_pairs),
                eval_goal_tol=float(args.eval_goal_tol),
                max_samples=int(args.max_samples),
            )
            results.append(r)
            print(
                f"  score={r['score']:.3f} wall_safe={r['wall_safe_success_rate']:.3f} "
                f"goal={r['goal_reach_rate']:.3f} err={r['mean_goal_error_m']:.3f} "
                f"epoch={r['region_epoch']} t={r['elapsed_sec']:.1f}s"
            )
        except Exception as e:
            print(f"  FAILED: {e}")

    if not results:
        raise SystemExit("All configs failed.")

    results = sorted(results, key=lambda x: x["score"], reverse=True)
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(out_dir, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        for r in results:
            w.writerow(r)
    with open(os.path.join(out_dir, "best_config.json"), "w") as f:
        json.dump(results[0], f, indent=2)

    print("\nBest config:")
    print(json.dumps(results[0], indent=2))
    print(f"\nSaved: {out_dir}")


if __name__ == "__main__":
    main()
