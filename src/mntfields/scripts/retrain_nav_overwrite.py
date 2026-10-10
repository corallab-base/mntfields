#!/usr/bin/env python3
"""
retrain_nav_overwrite.py — fine-tune with navigation-priority data pooling.

Problem: simple union of mapping + navigation data creates conflicting signals
in the obstacle area (mapping says "free space", navigation says "wall").
This causes the speed field to go blurry during fine-tuning.

Solution: for every spatial grid cell that has navigation x0 observations,
remove mapping rows whose x0 falls in that cell ("navigation overwrites mapping").
Then pool the remaining mapping rows + all navigation rows and fine-tune.

This preserves the sharp field structure from the base model (epoch=400)
while correctly incorporating the obstacle from the navigation data.

Usage:
    python3 src/mntfields/scripts/retrain_nav_overwrite.py \\
        --base-dir ~/turtlebot3_ws/mntfield_runs/refined_corrected \\
        --nav-dir  ~/turtlebot3_ws/mntfields_model \\
        --output-dir ~/turtlebot3_ws/mntfield_runs/rc_nav_overwrite_200ep
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


def _load_all_raw_rows(model_dir: Path) -> torch.Tensor | None:
    parts: list[torch.Tensor] = []
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
            print(f"  [SKIP] {data_path}: {exc}")
            continue
        frames = loaded.get("frame_data", []) if isinstance(loaded, dict) else loaded
        for t in frames:
            if isinstance(t, torch.Tensor) and t.ndim == 2 and t.shape[1] >= 6:
                parts.append(t.detach().cpu().float())
    if not parts:
        return None
    return torch.cat(parts, dim=0)


def _filter_by_room(rows: torch.Tensor, room: dict, row_mode: str) -> torch.Tensor | None:
    if rows is None or len(rows) == 0:
        return None
    arr = rows.numpy()
    x0 = arr[:, 0:2]
    bb = room["bounding_box"]
    x0_in = (
        (x0[:, 0] >= float(bb[0][0])) & (x0[:, 0] <= float(bb[1][0])) &
        (x0[:, 1] >= float(bb[0][1])) & (x0[:, 1] <= float(bb[1][1]))
    )
    if row_mode == "both":
        x1 = arr[:, 3:5]
        x1_in = (
            (x1[:, 0] >= float(bb[0][0])) & (x1[:, 0] <= float(bb[1][0])) &
            (x1[:, 1] >= float(bb[0][1])) & (x1[:, 1] <= float(bb[1][1]))
        )
        keep = x0_in & x1_in
    else:
        keep = x0_in
    if not keep.any():
        return None
    return rows[keep]


def _nav_overwrite_pool(
    map_rows: torch.Tensor,
    nav_rows: torch.Tensor,
    cell_size: float,
) -> torch.Tensor:
    """
    Remove mapping rows whose x0 falls in grid cells covered by navigation x0,
    then concatenate remaining mapping rows with all navigation rows.
    """
    nav_x0 = nav_rows[:, 0:2].numpy()
    nav_cells = set(
        tuple(c) for c in np.floor(nav_x0 / cell_size).astype(int).tolist()
    )

    map_x0 = map_rows[:, 0:2].numpy()
    map_cell_arr = np.floor(map_x0 / cell_size).astype(int)
    not_covered = np.array(
        [tuple(c) not in nav_cells for c in map_cell_arr],
        dtype=bool,
    )

    kept_map = map_rows[not_covered]
    print(
        f"    overwrite: nav_cells={len(nav_cells)}  "
        f"map_total={len(map_rows)}  map_kept={int(not_covered.sum())}  "
        f"nav_rows={len(nav_rows)}"
    )

    pooled = torch.cat([nav_rows, kept_map], dim=0)
    try:
        pooled = torch.unique(pooled, dim=0)
    except Exception:
        pass
    return pooled


def _cosine_lr(epoch: int, lr_min: float, lr_max: float, cycle: int) -> float:
    t = epoch % cycle
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * t / cycle))


def _finetune_region(
    region: Region,
    raw_rows: torch.Tensor,
    epochs: int,
    lr: float,
    batch_size: int,
    max_batches: int,
    debug_anchor: list[float],
) -> None:
    raw_rows = raw_rows.detach().cpu().float()
    region.all_raw_frame_data = [raw_rows]
    n14 = region.calculate_speed_and_normal(raw_rows.to(region.device))
    if n14 is None or len(n14) == 0:
        raise RuntimeError(f"region {region.version}: no valid Nx14 rows from pooled data")

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
        n_batches = min(max_batches, math.ceil(len(shuffled) / batch_size))

        ep_loss = 0.0
        for i in range(n_batches):
            batch = shuffled[i * batch_size:(i + 1) * batch_size]
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
                f"  [Region {region.version}] ep {ep:4d}/{epochs}  "
                f"loss={ep_loss:.6f}  best={best_loss:.6f}  "
                f"lr={cur_lr:.2e}  t={elapsed:.0f}s"
            )

    region.model.network.load_state_dict(best_state)
    region._sync_model_refs()
    region.epoch = start_epoch + epochs
    region.save_model()

    n14_vis = n14.detach().cpu()
    region.save_debug_plot(debug_anchor, view_radius=3.5, robot_xy=debug_anchor[:2])
    region.save_training_vis(n14_vis[:, :6], n14_vis[:, 6:8], robot_xy=debug_anchor[:2])


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune from base model with navigation-priority overwrite pooling."
    )
    parser.add_argument(
        "--base-dir",
        default=str(_WS / "mntfield_runs" / "refined_corrected"),
        help="Base model directory (good weights, mapping data)",
    )
    parser.add_argument(
        "--nav-dir",
        default=str(_WS / "mntfields_model"),
        help="Navigation model directory (obstacle data)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(_WS / "mntfield_runs" / "rc_nav_overwrite_200ep"),
        help="Output directory (must not exist)",
    )
    parser.add_argument(
        "--cell-size",
        type=float,
        default=0.15,
        help="Grid cell size (m) for navigation-coverage detection",
    )
    parser.add_argument(
        "--rooms",
        nargs="*",
        type=int,
        default=[],
        help="Specific room ids to retrain (default: all)",
    )
    parser.add_argument(
        "--row-mode",
        choices=["x0", "both"],
        default="x0",
        help="Assign rows to rooms by x0 only or strict x0+x1 inclusion",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help="Peak LR (lower than default 2e-4 to preserve sharp base weights)",
    )
    parser.add_argument("--batch-size", type=int, default=8000)
    parser.add_argument("--max-batches", type=int, default=8)
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    base_dir = Path(os.path.expanduser(args.base_dir)).resolve()
    nav_dir = Path(os.path.expanduser(args.nav_dir)).resolve()
    output_dir = Path(os.path.expanduser(args.output_dir)).resolve()

    if output_dir.exists():
        raise RuntimeError(f"output dir already exists: {output_dir}")
    shutil.copytree(base_dir, output_dir)
    print(f"[COPY] {base_dir} -> {output_dir}")

    topo = _load_json(output_dir / "topology.json")
    rooms = topo.get("rooms", [])
    if not rooms:
        raise RuntimeError(f"no rooms in {output_dir / 'topology.json'}")

    requested = set(int(r) for r in args.rooms) if args.rooms else {int(r["id"]) for r in rooms}
    rooms = [r for r in rooms if int(r["id"]) in requested]
    if not rooms:
        raise RuntimeError("no matching rooms selected")

    room_to_region = _load_room_to_region(output_dir)

    print(f"\n[LOAD] reading raw rows from base ({base_dir.name}) ...")
    map_all = _load_all_raw_rows(base_dir)
    print(f"[LOAD] reading raw rows from nav ({nav_dir.name}) ...")
    nav_all = _load_all_raw_rows(nav_dir)
    print(
        f"[LOAD] base={len(map_all) if map_all is not None else 0} rows  "
        f"nav={len(nav_all) if nav_all is not None else 0} rows\n"
    )

    for room in rooms:
        room_id = int(room["id"])
        region_ver = int(room_to_region.get(room_id, room_id))

        map_rows = _filter_by_room(map_all, room, args.row_mode)
        nav_rows = _filter_by_room(nav_all, room, args.row_mode)

        n_map = len(map_rows) if map_rows is not None else 0
        n_nav = len(nav_rows) if nav_rows is not None else 0
        print(f"[ROOM {room_id}] region={region_ver}  map={n_map}  nav={n_nav}")

        if nav_rows is None or n_nav == 0:
            if map_rows is None or n_map == 0:
                print(f"  [SKIP] no rows from either source\n")
                continue
            pooled = map_rows
            print(f"  [POOL] no nav rows — using map only ({n_map})\n")
        elif map_rows is None or n_map == 0:
            pooled = nav_rows
            print(f"  [POOL] no map rows — using nav only ({n_nav})\n")
        else:
            pooled = _nav_overwrite_pool(map_rows, nav_rows, args.cell_size)
            print(f"  [POOL] total after overwrite+dedup: {len(pooled)}\n")

        region = _load_region(output_dir, region_ver, args.device)
        centroid = [float(room["centroid_xy"][0]), float(room["centroid_xy"][1]), 0.0]
        _finetune_region(
            region=region,
            raw_rows=pooled,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            max_batches=args.max_batches,
            debug_anchor=centroid,
        )
        print(f"[DONE] room {room_id} region {region_ver}\n")

    print(f"[DONE] model written to {output_dir}")
    print(
        f"\nNext step — generate field plots:\n"
        f"  python3 src/mntfields/scripts/plot_room_contours.py \\\n"
        f"    --model-dir {output_dir} \\\n"
        f"    --debug-anchor-room-ids {' '.join(str(r['id']) for r in rooms)}\n"
    )


if __name__ == "__main__":
    main()
