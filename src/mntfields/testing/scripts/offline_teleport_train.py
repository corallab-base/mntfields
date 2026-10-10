#!/usr/bin/env python3
import argparse
import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from mntfields.mntfields_core.core.region import Region
from mntfields.mntfields_core.mapping.lidar_gridmap import LidarGridmap


class DummyModel:
    def __init__(self, folder: str, device: str):
        self.folder = folder
        self.renderer = device
        self.h = 0.0


def load_dataset_samples(dataset_dir: str):
    meta_path = os.path.join(dataset_dir, "meta.json")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)

    chunks = sorted(
        p for p in os.listdir(dataset_dir)
        if p.startswith("chunk_") and p.endswith(".npz")
    )
    samples = []
    for c in chunks:
        arr = np.load(os.path.join(dataset_dir, c), allow_pickle=False)
        poses = arr["robot_pose"]
        hit = arr["world_points_hit"]
        hit_counts = arr["world_points_hit_counts"]
        for i in range(len(poses)):
            n = int(hit_counts[i])
            if n <= 0:
                continue
            samples.append((poses[i].astype(np.float32), hit[i, :n, :].astype(np.float32)))
    return meta, samples


def build_gridmap(samples, n_cells: int, near_dist: float, map_size: float, room_seg_kernel: int, room_seg_iters: int):
    gm = LidarGridmap(
        trav_map=np.ones((n_cells, n_cells), dtype=np.uint8),
        regions={},
        n_cells=n_cells,
        near_dist=near_dist,
        room_seg_config=(room_seg_kernel, room_seg_iters),
        map_size=map_size,
        origin=(0.0, 0.0),
    )
    if samples:
        gm.origin = samples[0][0][:2].copy()
    for idx, (pose, pts) in enumerate(samples):
        if len(pts) < 5:
            continue
        speeds = np.ones(len(pts), dtype=np.float32)
        try:
            gm.update(
                lidar_scan=pts[:, :2],
                points=pts[:, :2],
                speeds=speeds,
                cur_pos=pose[:2],
                region=None,
                regions={},
                update_room=(idx % 10 == 0),
            )
        except Exception:
            gm.update(
                lidar_scan=pts[:, :2],
                points=pts[:, :2],
                speeds=speeds,
                cur_pos=pose[:2],
                region=None,
                regions={},
                update_room=False,
            )
    return gm


def raycast_scan_hits(gm: LidarGridmap, robot_xy: np.ndarray, n_beams: int, max_range: float, step_m: float):
    hits = []
    if n_beams < 8:
        n_beams = 8
    for k in range(n_beams):
        ang = -np.pi + (2.0 * np.pi * k / n_beams)
        c = np.cos(ang)
        s = np.sin(ang)
        t = step_m
        last_free = None
        while t <= max_range + 1e-6:
            x = float(robot_xy[0] + c * t)
            y = float(robot_xy[1] + s * t)
            i, j = gm._world_to_grid(x, y)
            if gm.data[i, j] == 1:
                hx, hy = gm._grid_to_world(i, j)
                hits.append([hx, hy, 0.0])
                break
            last_free = (x, y)
            t += step_m
        if t > max_range and last_free is not None:
            # No hit in range -> endpoint at max range in free space.
            hits.append([last_free[0], last_free[1], 0.0])
    if not hits:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray(hits, dtype=np.float32)


def path_hits_wall(gm: LidarGridmap, path_xy: np.ndarray, step: float = 0.05) -> bool:
    if len(path_xy) < 2:
        return True
    for p0, p1 in zip(path_xy[:-1], path_xy[1:]):
        d = float(np.linalg.norm(p1 - p0))
        n = max(2, int(np.ceil(d / step)))
        ts = np.linspace(0.0, 1.0, n, dtype=np.float32)
        pts = p0[None, :] + (p1 - p0)[None, :] * ts[:, None]
        for xy in pts:
            i, j = gm._world_to_grid(float(xy[0]), float(xy[1]))
            if gm.data[i, j] == 1:
                return True
    return False


def save_occ_png(gm: LidarGridmap, out_path: str):
    occ = gm.data
    vis = gm.visited
    img = np.full_like(occ, 205, dtype=np.uint8)
    img[(occ == 0) & (vis == 1)] = 254
    img[occ == 1] = 0
    plt.figure(figsize=(8, 8))
    plt.imshow(np.flipud(img), cmap="gray", vmin=0, vmax=255)
    plt.title("Teleport Training Base Map")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Offline teleport training from full recorded map.")
    ap.add_argument("--dataset_dir", required=True, help="session_* folder")
    ap.add_argument("--out_dir", default="", help="default: <dataset_dir>/teleport_train")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--room_seg_kernel", type=int, default=7)
    ap.add_argument("--room_seg_iters", type=int, default=2)
    ap.add_argument("--train_steps", type=int, default=300)
    ap.add_argument("--train_epoch", type=int, default=20)
    ap.add_argument("--train_every_n_steps", type=int, default=1)
    ap.add_argument("--train_samples", type=int, default=2500)
    ap.add_argument("--strict_pair_validity", type=str, default="false")
    ap.add_argument("--n_beams", type=int, default=360)
    ap.add_argument("--max_range", type=float, default=3.2)
    ap.add_argument("--ray_step_m", type=float, default=0.05)
    ap.add_argument("--eval_pairs", type=int, default=50)
    ap.add_argument("--save_field_plots", type=str, default="true")
    ap.add_argument("--plot_every", type=int, default=25)
    ap.add_argument("--plot_view_radius", type=float, default=3.5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    strict = str(args.strict_pair_validity).lower() in ("1", "true", "yes", "y", "t")
    save_field_plots = str(args.save_field_plots).lower() in ("1", "true", "yes", "y", "t")
    dataset_dir = os.path.abspath(args.dataset_dir)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.join(dataset_dir, "teleport_train")
    os.makedirs(out_dir, exist_ok=True)
    run_t0 = time.time()

    meta, samples = load_dataset_samples(dataset_dir)
    if not samples:
        raise SystemExit(f"No chunk samples found in {dataset_dir}")

    n_cells = int(meta.get("n_cells", 400))
    near_dist = float(meta.get("near_dist", 2.0))
    map_size = float(meta.get("map_size", 40.0))

    print(f"[TELEPORT] Loaded {len(samples)} real samples, building base map...")
    gm = build_gridmap(
        samples=samples,
        n_cells=n_cells,
        near_dist=near_dist,
        map_size=map_size,
        room_seg_kernel=int(args.room_seg_kernel),
        room_seg_iters=int(args.room_seg_iters),
    )
    save_occ_png(gm, os.path.join(out_dir, "base_occupancy.png"))

    free_cells = np.argwhere((gm.visited == 1) & (gm.data == 0))
    if len(free_cells) < 50:
        raise SystemExit("Too few free cells after base-map build; record a longer dataset.")

    model = DummyModel(folder=out_dir, device=args.device)
    bound = torch.tensor([
        [gm.origin[0] - map_size / 2.0, gm.origin[1] - map_size / 2.0, -0.1],
        [gm.origin[0] + map_size / 2.0, gm.origin[1] + map_size / 2.0,  0.1],
    ], dtype=torch.float32)
    region = Region(model=model, bound=bound, version=1, init_network=True, eval=False)
    region.gridmap = gm
    region.strict_pair_validity = strict
    region.enable_debug_vis = False

    rng = np.random.default_rng(int(args.seed))
    train_used = 0
    candidate = 0
    last_scan = None
    last_pose = None
    plots_saved = 0

    for step in range(int(args.train_steps)):
        if step % max(1, int(args.train_every_n_steps)) != 0:
            continue
        ij = free_cells[int(rng.integers(0, len(free_cells)))]
        rx, ry = gm._grid_to_world(int(ij[0]), int(ij[1]))
        pose = np.array([rx, ry, 0.0], dtype=np.float32)
        scan = raycast_scan_hits(
            gm=gm,
            robot_xy=pose[:2],
            n_beams=int(args.n_beams),
            max_range=float(args.max_range),
            step_m=float(args.ray_step_m),
        )
        if len(scan) < 30:
            continue
        data = region.get_data_from_lidar(
            lidar_scan=scan,
            cur_pos=pose,
            device=args.device,
            n_samples=int(args.train_samples),
        )
        if data is None:
            continue
        points, closest = data
        if points is None or closest is None or len(points) == 0:
            continue
        candidate += 1
        prev_epoch = int(region.epoch)
        raw = torch.cat([points, closest], dim=1)
        region.train_core(epoch=int(args.train_epoch), raw_frame_data=raw)
        if int(region.epoch) > prev_epoch:
            train_used += 1
            last_scan = scan
            last_pose = pose
            if save_field_plots and (train_used % max(1, int(args.plot_every)) == 0):
                try:
                    region.save_debug_plot(
                        pose,
                        view_radius=float(args.plot_view_radius),
                        robot_xy=pose[:2],
                    )
                    plots_saved += 1
                except Exception as e:
                    print(f"[TELEPORT][PLOT] periodic save failed: {e}")

        if (step + 1) % 25 == 0:
            print(
                f"[TELEPORT] step={step+1}/{args.train_steps} "
                f"epoch={region.epoch} candidate={candidate} used={train_used}"
            )

    # Evaluate field success on random start-goal in free space
    eval_pairs = int(args.eval_pairs)
    success = 0
    wall_safe_success = 0
    end_errors = []
    if train_used > 0 and region.epoch > 0:
        for _ in range(eval_pairs):
            a = free_cells[int(rng.integers(0, len(free_cells)))]
            b = free_cells[int(rng.integers(0, len(free_cells)))]
            if np.all(a == b):
                continue
            sx, sy = gm._grid_to_world(int(a[0]), int(a[1]))
            gx, gy = gm._grid_to_world(int(b[0]), int(b[1]))
            sxy = np.array([sx, sy], dtype=np.float32)
            gxy = np.array([gx, gy], dtype=np.float32)
            path = region.generate_field_path(sxy, gxy, step_size=0.15, max_steps=120)
            if path is None:
                continue
            p = np.asarray(path, dtype=np.float32)
            if len(p) < 2:
                continue
            err = float(np.linalg.norm(p[-1] - gxy))
            end_errors.append(err)
            if err <= 0.25:
                success += 1
            if err <= 0.25 and (not path_hits_wall(gm, p)):
                wall_safe_success += 1

    if save_field_plots and last_scan is not None and last_pose is not None:
        try:
            region.save_debug_plot(
                last_pose,
                view_radius=float(args.plot_view_radius),
                robot_xy=last_pose[:2],
            )
            plots_saved += 1
        except Exception as e:
            print(f"[TELEPORT][PLOT] final save failed: {e}")

    region.save_model()
    summary = {
        "dataset_dir": dataset_dir,
        "device": args.device,
        "train_steps": int(args.train_steps),
        "train_epoch": int(args.train_epoch),
        "train_samples": int(args.train_samples),
        "strict_pair_validity": bool(strict),
        "n_beams": int(args.n_beams),
        "max_range": float(args.max_range),
        "ray_step_m": float(args.ray_step_m),
        "base_samples": len(samples),
        "free_cells": int(len(free_cells)),
        "candidate_train_frames": int(candidate),
        "train_frames_used": int(train_used),
        "region_epoch": int(region.epoch),
        "goal_reach_rate": float(success) / float(max(1, eval_pairs)),
        "wall_safe_success_rate": float(wall_safe_success) / float(max(1, eval_pairs)),
        "mean_goal_error_m": float(np.mean(end_errors)) if end_errors else 1e9,
        "field_plots_saved": int(plots_saved),
        "elapsed_sec": float(time.time() - run_t0),
        "output_dir": out_dir,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\n[TELEPORT] Summary:")
    print(json.dumps(summary, indent=2))
    print(f"[TELEPORT] Saved: {out_dir}")


if __name__ == "__main__":
    main()
