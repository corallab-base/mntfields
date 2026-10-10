#!/usr/bin/env python3
"""
offline_retrain_rooms.py — fine-tune selected room-regions from a merged raw pool.

This script:
  1. clones a saved model directory,
  2. pools raw region rows from one or more model directories,
  3. assigns pooled rows to rooms (default: x0 ownership),
  4. fine-tunes the corresponding room-region networks,
  5. writes Region.save_debug_plot() and Region.save_training_vis() outputs.

Usage:
    python3 src/mntfields/scripts/offline_retrain_rooms.py
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch


_WS = Path(__file__).resolve().parents[3]
_INSTALL_LIB = _WS / "install" / "mntfields" / "lib" / "python3.10" / "site-packages"
if _INSTALL_LIB.exists():
    sys.path.insert(0, str(_INSTALL_LIB))

_DEFAULT_MODEL_DIR = _WS / "mntfields_model"
_DEFAULT_OUTPUT_DIR = _WS / "mntfields_model_output"
_DEFAULT_RUNS_ROOT = _WS / "mntfield_runs"
_DEFAULT_MERGE_DIRS = [
    _DEFAULT_RUNS_ROOT / "mntfields_model_mapping",
    _DEFAULT_RUNS_ROOT / "mntfields_model_mapping_refined",
    _DEFAULT_RUNS_ROOT / "mntfields_model_navigation",
]

from mntfields.mntfields_core.core.region import Region


def _load_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _load_room_to_region(model_dir: Path) -> dict[int, int]:
    path = model_dir / "room_region_map.json"
    if not path.exists():
        return {}
    raw = _load_json(path)
    return {int(k): int(v) for k, v in raw.items()}


def _load_bound_from_meta(model_dir: Path, version: int) -> torch.Tensor:
    meta_path = model_dir / f"region_{version}" / f"region_{version}_meta.json"
    meta = _load_json(meta_path)
    return torch.tensor(meta["bound"], dtype=torch.float32)


def _load_region(model_dir: Path, version: int, device: str) -> Region:
    bound = _load_bound_from_meta(model_dir, version)
    region = Region(
        model_folder=str(model_dir),
        bound=bound,
        version=version,
        device=device,
    )
    if not region.load_model():
        raise RuntimeError(f"could not load region {version} from {model_dir}")
    return region


def _load_pooled_rows_from_model_dirs(model_dirs: list[Path]) -> torch.Tensor | None:
    parts: list[torch.Tensor] = []
    seen_dirs: set[Path] = set()
    for model_dir in model_dirs:
        model_dir = model_dir.resolve()
        if model_dir in seen_dirs:
            continue
        seen_dirs.add(model_dir)
        for region_dir in sorted(model_dir.glob("region_*")):
            if not region_dir.is_dir():
                continue
            try:
                ver = int(region_dir.name.split("_")[1])
            except Exception:
                continue
            data_path = region_dir / f"region_{ver}_data.pth"
            if not data_path.exists():
                continue
            try:
                loaded = torch.load(str(data_path), map_location="cpu", weights_only=False)
            except Exception as exc:
                print(f"[POOL] skip {data_path}: {exc}")
                continue
            frame_parts = []
            if isinstance(loaded, dict):
                frame_parts = loaded.get("frame_data", []) or []
            elif isinstance(loaded, list):
                frame_parts = loaded
            for tensor in frame_parts:
                if isinstance(tensor, torch.Tensor) and tensor.ndim == 2 and tensor.shape[1] >= 6:
                    parts.append(tensor.detach().cpu().float())
    if not parts:
        return None
    pooled = torch.cat(parts, dim=0)
    try:
        pooled = torch.unique(pooled, dim=0)
    except Exception:
        pass
    return pooled


def _room_rows(room: dict, pooled: torch.Tensor, row_mode: str) -> torch.Tensor | None:
    if pooled is None or len(pooled) == 0:
        return None
    arr = pooled.numpy()
    x0 = arr[:, 0:2]
    x1 = arr[:, 3:5]
    bb = room["bounding_box"]
    x0_in = (
        (x0[:, 0] >= float(bb[0][0])) & (x0[:, 0] <= float(bb[1][0])) &
        (x0[:, 1] >= float(bb[0][1])) & (x0[:, 1] <= float(bb[1][1]))
    )
    if row_mode == "x0":
        keep = x0_in
    else:
        x1_in = (
            (x1[:, 0] >= float(bb[0][0])) & (x1[:, 0] <= float(bb[1][0])) &
            (x1[:, 1] >= float(bb[0][1])) & (x1[:, 1] <= float(bb[1][1]))
        )
        keep = x0_in & x1_in
    if not keep.any():
        return None
    return pooled[keep]


def _cosine_lr(epoch: int, lr_min: float, lr_max: float, cycle: int) -> float:
    t = epoch % cycle
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * t / cycle))


def _pair_distance_stats(rows: torch.Tensor) -> dict[str, float] | None:
    if rows is None or len(rows) == 0:
        return None
    arr = rows.detach().cpu().numpy()
    x0 = arr[:, 0:2].astype("float32")
    x1 = arr[:, 3:5].astype("float32")
    d = ((x0 - x1) ** 2).sum(axis=1) ** 0.5
    if len(d) == 0:
        return None
    return {
        "min": float(d.min()),
        "p25": float(torch.tensor(d).quantile(0.25).item()),
        "p50": float(torch.tensor(d).quantile(0.50).item()),
        "p75": float(torch.tensor(d).quantile(0.75).item()),
        "p95": float(torch.tensor(d).quantile(0.95).item()),
        "max": float(d.max()),
        "std": float(d.std()),
    }


def _format_pair_stats(stats: dict[str, float] | None) -> str:
    if stats is None:
        return "none"
    return (
        f"min={stats['min']:.4f} p25={stats['p25']:.4f} p50={stats['p50']:.4f} "
        f"p75={stats['p75']:.4f} p95={stats['p95']:.4f} max={stats['max']:.4f} "
        f"std={stats['std']:.4f}"
    )


def _rebalance_pair_distance_rows(
    rows: torch.Tensor,
    n_bins: int,
    target_rows: int = 0,
) -> torch.Tensor:
    if rows is None or len(rows) == 0:
        return rows
    rows = rows.detach().cpu().float()
    if len(rows) <= 1:
        return rows

    arr = rows.numpy()
    x0 = arr[:, 0:2].astype("float32")
    x1 = arr[:, 3:5].astype("float32")
    d = ((x0 - x1) ** 2).sum(axis=1) ** 0.5
    d_min = float(d.min())
    d_max = float(d.max())
    if not np.isfinite(d_min) or not np.isfinite(d_max) or d_max <= d_min:
        return rows

    bins = np.linspace(d_min, d_max + 1e-6, max(2, int(n_bins)) + 1)
    bin_ids = np.clip(np.digitize(d, bins[1:-1], right=False), 0, len(bins) - 2)
    groups = [np.where(bin_ids == i)[0] for i in range(len(bins) - 1)]
    nonempty = [g for g in groups if len(g) > 0]
    if not nonempty:
        return rows

    if target_rows <= 0:
        counts = np.array([len(g) for g in nonempty], dtype=np.int32)
        target_rows = int(np.median(counts) * len(nonempty))
        target_rows = max(len(nonempty), min(int(len(rows)), target_rows))
    else:
        target_rows = max(len(nonempty), min(int(target_rows), int(len(rows))))

    target_per_bin = max(1, target_rows // len(nonempty))
    selected: list[np.ndarray] = []
    leftover: list[np.ndarray] = []
    for g in nonempty:
        if len(g) <= target_per_bin:
            selected.append(g)
            continue
        idx = np.linspace(0, len(g) - 1, target_per_bin).astype(np.int32)
        take = g[idx]
        selected.append(take)
        mask = np.ones(len(g), dtype=bool)
        mask[idx] = False
        leftover.append(g[mask])

    total = sum(len(g) for g in selected)
    remaining = max(0, target_rows - total)
    if remaining > 0 and leftover:
        pool = np.concatenate(leftover, axis=0)
        if len(pool) > remaining:
            idx = np.linspace(0, len(pool) - 1, remaining).astype(np.int32)
            pool = pool[idx]
        selected.append(pool)

    keep_idx = np.concatenate(selected, axis=0)
    keep_idx.sort()
    return rows[torch.from_numpy(keep_idx).long()]


def _finetune_region(
    region: Region,
    raw_rows: torch.Tensor,
    epochs: int,
    lr: float,
    checkpoint_every: int,
    batch_size: int,
    max_batches: int,
    debug_anchor: list[float],
) -> None:
    raw_rows = raw_rows.detach().cpu().float()
    region.all_raw_frame_data = [raw_rows]
    n14 = region.calculate_speed_and_normal(raw_rows.to(region.device))
    if n14 is None or len(n14) == 0:
        raise RuntimeError(f"region {region.version}: no valid Nx14 rows from pooled raw data")

    lr_min = lr * 0.1
    lr_max = lr
    lr_cycle = 200
    best_loss = float("inf")
    best_state = copy.deepcopy(region.model.network.state_dict())
    start_epoch = int(region.epoch)
    t0 = time.time()

    for ep in range(1, epochs + 1):
        cur_lr = _cosine_lr(ep, lr_min, lr_max, lr_cycle)
        region.model.optimizer.param_groups[0]["lr"] = cur_lr

        perm = torch.randperm(len(n14), device=region.device)
        shuffled = n14[perm]
        n = len(shuffled)
        n_batches = min(max_batches, math.ceil(n / batch_size))

        ep_loss = 0.0
        for i in range(n_batches):
            batch = shuffled[i * batch_size : (i + 1) * batch_size]
            points = batch[:, :6].float()
            speed = batch[:, 6:8].float()
            normal = batch[:, 8:14].float()
            speed = speed * speed * (2 - speed) * (2 - speed)

            loss_out = region.model.function.Loss(
                points, speed, normal, beta=1.0, gamma=0.001, epoch=start_epoch + ep
            )
            loss_value = loss_out[0] if isinstance(loss_out, tuple) else loss_out

            region.model.optimizer.zero_grad()
            loss_value.backward()
            torch.nn.utils.clip_grad_norm_(region.model.network.parameters(), 1.0)
            region.model.optimizer.step()
            ep_loss += float(loss_value.detach())

        ep_loss /= max(n_batches, 1)
        if ep_loss < best_loss:
            best_loss = ep_loss
            best_state = copy.deepcopy(region.model.network.state_dict())

        if ep % 50 == 0 or ep == epochs:
            elapsed = time.time() - t0
            print(
                f"  [Region {region.version}] ep {ep:4d}/{epochs}  loss={ep_loss:.6f}  "
                f"best={best_loss:.6f}  lr={cur_lr:.2e}  t={elapsed:.0f}s"
            )
        if checkpoint_every > 0 and ep % checkpoint_every == 0:
            region.save_checkpoint(start_epoch + ep)

    region.model.network.load_state_dict(best_state)
    region._sync_model_refs()
    region.epoch = start_epoch + epochs
    region.save_model()

    n14_vis = n14.detach().cpu()
    region.save_debug_plot(debug_anchor, view_radius=3.5, robot_xy=debug_anchor[:2])
    region.save_training_vis(n14_vis[:, :6], n14_vis[:, 6:8], robot_xy=debug_anchor[:2])


def main():
    parser = argparse.ArgumentParser(description="Offline retrain selected room-regions from a merged raw pool.")
    parser.add_argument(
        "--model-dir",
        default=str(_DEFAULT_MODEL_DIR),
        help="Current model directory whose room regions should be fine-tuned",
    )
    parser.add_argument(
        "--output-dir",
        default=str(_DEFAULT_OUTPUT_DIR),
        help="Output directory for the retrained model clone",
    )
    parser.add_argument(
        "--merge-model-dirs",
        nargs="*",
        default=[str(p) for p in _DEFAULT_MERGE_DIRS],
        help="Additional model directories to pool raw rows from",
    )
    parser.add_argument(
        "--rooms",
        nargs="*",
        type=int,
        default=[],
        help="Specific room ids to retrain (default: all rooms in topology.json)",
    )
    parser.add_argument(
        "--row-mode",
        choices=["x0", "both"],
        default="x0",
        help="Assign pooled rows to rooms by x0 ownership or strict x0/x1 inclusion",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8000)
    parser.add_argument("--max-batches", type=int, default=8)
    parser.add_argument(
        "--cap-room-rows",
        type=int,
        default=0,
        help="Spatially cap each room's raw rows before retraining (0 disables)",
    )
    parser.add_argument(
        "--rebalance-pair-dist",
        action="store_true",
        help="Downsample overrepresented x0->x1 distance bands before retraining",
    )
    parser.add_argument(
        "--rebalance-bins",
        type=int,
        default=8,
        help="Number of x0->x1 distance bins used for pair-distance rebalancing",
    )
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    model_dir = Path(os.path.expanduser(args.model_dir)).resolve()
    output_dir = (
        Path(os.path.expanduser(args.output_dir)).resolve()
        if args.output_dir
        else _DEFAULT_OUTPUT_DIR.resolve()
    )

    if output_dir.exists():
        raise RuntimeError(f"output dir already exists: {output_dir}")
    shutil.copytree(model_dir, output_dir)
    print(f"[COPY] cloned {model_dir} -> {output_dir}")

    topo = _load_json(output_dir / "topology.json")
    rooms = topo.get("rooms", [])
    if not rooms:
        raise RuntimeError(f"no rooms found in {output_dir / 'topology.json'}")
    requested_rooms = set(int(r) for r in args.rooms) if args.rooms else {int(r["id"]) for r in rooms}
    rooms = [r for r in rooms if int(r["id"]) in requested_rooms]
    if not rooms:
        raise RuntimeError("no matching rooms selected")

    room_to_region = _load_room_to_region(output_dir)
    merge_dirs = [model_dir] + [Path(os.path.expanduser(p)).resolve() for p in args.merge_model_dirs]
    pooled = _load_pooled_rows_from_model_dirs(merge_dirs)
    if pooled is None or len(pooled) == 0:
        raise RuntimeError("no pooled raw rows found across requested model dirs")
    print(f"[POOL] {len(pooled)} unique raw rows across {len(merge_dirs)} model dir(s)")

    for room in rooms:
        room_id = int(room["id"])
        region_ver = int(room_to_region.get(room_id, room_id))
        room_rows = _room_rows(room, pooled, args.row_mode)
        if room_rows is None or len(room_rows) == 0:
            print(f"[SKIP] room {room_id} region {region_ver}: no pooled rows")
            continue
        region = _load_region(output_dir, region_ver, args.device)
        before_n = len(room_rows)
        before_stats = _pair_distance_stats(room_rows)

        if int(args.cap_room_rows) > 0 and not args.rebalance_pair_dist:
            room_rows = region._spatially_cap_raw_history(room_rows, int(args.cap_room_rows))
        if args.rebalance_pair_dist:
            target_rows = int(args.cap_room_rows) if int(args.cap_room_rows) > 0 else 0
            room_rows = _rebalance_pair_distance_rows(
                room_rows,
                n_bins=int(args.rebalance_bins),
                target_rows=target_rows,
            )
        after_n = len(room_rows)
        after_stats = _pair_distance_stats(room_rows)

        print(f"\n[RETRAIN] room {room_id} -> region {region_ver} rows={before_n} -> {after_n}")
        print(f"  [PAIR BEFORE] {_format_pair_stats(before_stats)}")
        print(f"  [PAIR AFTER ] {_format_pair_stats(after_stats)}")
        centroid = [float(room["centroid_xy"][0]), float(room["centroid_xy"][1]), 0.0]
        _finetune_region(
            region=region,
            raw_rows=room_rows,
            epochs=int(args.epochs),
            lr=float(args.lr),
            checkpoint_every=int(args.checkpoint_every),
            batch_size=int(args.batch_size),
            max_batches=int(args.max_batches),
            debug_anchor=centroid,
        )
        print(f"[DONE] room {room_id} region {region_ver} -> {region.folder}")

    print(f"\n[DONE] offline retrained model written to {output_dir}")


if __name__ == "__main__":
    main()
