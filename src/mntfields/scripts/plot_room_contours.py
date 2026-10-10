#!/usr/bin/env python3
"""
plot_room_contours.py — render offline diagnostics per saved room/region.

Supports two diagnostics:
  - contour plots using room centroid as the source
  - training-vis plots similar to Region.save_training_vis()

It writes:
  - one PNG per room
  - optionally one combined full-map PNG with all room contours overlaid

The output style matches the mapping vis plots:
  - dark blue unknown canvas
  - bright yellow known SLAM cells
  - cyan travel-time contours

Usage:
    python3 src/mntfields/scripts/plot_room_contours.py
    python3 src/mntfields/scripts/plot_room_contours.py \
        --model-dir ~/turtlebot3_ws/mntfields_model \
        --out-dir ~/turtlebot3_ws/mntfields_model/room_contours
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import numpy as np
from PIL import Image
import torch
import yaml

_WS = Path(__file__).resolve().parents[3]
_INSTALL_LIB = _WS / "install" / "mntfields" / "lib" / "python3.10" / "site-packages"
if _INSTALL_LIB.exists():
    sys.path.insert(0, str(_INSTALL_LIB))

from mntfields.mntfields_core.core.region import Region


def _load_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _load_room_to_region(model_dir: Path) -> dict[int, int]:
    path = model_dir / "room_region_map.json"
    if not path.exists():
        return {}
    raw = _load_json(path)
    return {int(k): int(v) for k, v in raw.items()}


def _load_regions(model_dir: Path, device: str) -> dict[int, Region]:
    out: dict[int, Region] = {}
    for region_dir in sorted(model_dir.glob("region_*")):
        if not region_dir.is_dir():
            continue
        try:
            ver = int(region_dir.name.split("_")[1])
        except Exception:
            continue
        meta_path = region_dir / f"region_{ver}_meta.json"
        if not meta_path.exists():
            print(f"[SKIP] region {ver}: missing meta file {meta_path}")
            continue
        try:
            meta = _load_json(meta_path)
            bound = torch.tensor(meta["bound"], dtype=torch.float32)
        except Exception as exc:
            print(f"[SKIP] region {ver}: could not load bound from meta ({exc})")
            continue

        region = Region(
            model_folder=str(model_dir),
            bound=bound,
            version=ver,
            device=device,
        )
        if region.load_model():
            out[ver] = region
    return out


def _load_pooled_rows_from_model_dirs(model_dirs: Iterable[Path]) -> torch.Tensor | None:
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


def _merged_raw_rows(region: Region) -> torch.Tensor | None:
    frames = getattr(region, "all_raw_frame_data", None) or []
    if not frames:
        return None
    try:
        merged = frames[0] if len(frames) == 1 else torch.cat(frames, dim=0)
        if not isinstance(merged, torch.Tensor) or merged.ndim != 2 or merged.shape[1] < 6:
            return None
        return merged.detach().cpu().float()
    except Exception:
        return None


def _room_sample_points(
    room: dict,
    sample_limit: int,
    rows: torch.Tensor | None = None,
    region: Region | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if rows is None and region is not None:
        rows = _merged_raw_rows(region)
    if rows is None or len(rows) == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)

    arr = rows.numpy()
    x0 = arr[:, 0:2].astype(np.float32)
    x1 = arr[:, 3:5].astype(np.float32)
    bb = room["bounding_box"]
    in_room_0 = (
        (x0[:, 0] >= float(bb[0][0])) & (x0[:, 0] <= float(bb[1][0])) &
        (x0[:, 1] >= float(bb[0][1])) & (x0[:, 1] <= float(bb[1][1]))
    )
    in_room_1 = (
        (x1[:, 0] >= float(bb[0][0])) & (x1[:, 0] <= float(bb[1][0])) &
        (x1[:, 1] >= float(bb[0][1])) & (x1[:, 1] <= float(bb[1][1]))
    )
    x0 = x0[in_room_0]
    x1 = x1[in_room_1]

    if sample_limit > 0 and len(x0) > sample_limit:
        idx = np.linspace(0, len(x0) - 1, sample_limit).astype(np.int32)
        x0 = x0[idx]
    if sample_limit > 0 and len(x1) > sample_limit:
        idx = np.linspace(0, len(x1) - 1, sample_limit).astype(np.int32)
        x1 = x1[idx]
    return x0, x1


def _room_raw_rows(room: dict, rows: torch.Tensor | None = None, region: Region | None = None) -> torch.Tensor | None:
    if rows is None and region is not None:
        rows = _merged_raw_rows(region)
    if rows is None or len(rows) == 0:
        return None
    arr = rows.numpy()
    x0 = arr[:, 0:2].astype(np.float32)
    x1 = arr[:, 3:5].astype(np.float32)
    bb = room["bounding_box"]
    in_room = (
        (x0[:, 0] >= float(bb[0][0])) & (x0[:, 0] <= float(bb[1][0])) &
        (x0[:, 1] >= float(bb[0][1])) & (x0[:, 1] <= float(bb[1][1])) &
        (x1[:, 0] >= float(bb[0][0])) & (x1[:, 0] <= float(bb[1][0])) &
        (x1[:, 1] >= float(bb[0][1])) & (x1[:, 1] <= float(bb[1][1]))
    )
    if not np.any(in_room):
        return None
    return rows[in_room]


def _room_owned_rows(room: dict, rows: torch.Tensor | None = None, region: Region | None = None) -> torch.Tensor | None:
    if rows is None and region is not None:
        rows = _merged_raw_rows(region)
    if rows is None or len(rows) == 0:
        return None
    arr = rows.numpy()
    x0 = arr[:, 0:2].astype(np.float32)
    bb = room["bounding_box"]
    owned = (
        (x0[:, 0] >= float(bb[0][0])) & (x0[:, 0] <= float(bb[1][0])) &
        (x0[:, 1] >= float(bb[0][1])) & (x0[:, 1] <= float(bb[1][1]))
    )
    if not np.any(owned):
        return None
    return rows[owned]


def _resolve_map_yaml(model_dir: Path) -> Path:
    map_yaml = model_dir / "map.yaml"
    if not map_yaml.exists():
        raise FileNotFoundError(f"map.yaml not found in {model_dir}")
    return map_yaml


def _load_map(model_dir: Path):
    map_yaml = _resolve_map_yaml(model_dir)
    meta = _load_yaml(map_yaml)
    image_name = str(meta["image"])
    image_path = Path(image_name)
    if not image_path.is_absolute():
        image_path = map_yaml.parent / image_name
    img = np.array(Image.open(image_path))
    if img.ndim == 3:
        img = img[..., 0]
    occ = np.flipud(img)
    resolution = float(meta["resolution"])
    origin = np.asarray(meta["origin"][:2], dtype=np.float32)
    return occ, resolution, origin, meta


def _known_mask_from_occ(occ: np.ndarray) -> np.ndarray:
    # Occupied and free are "known"; unknown is 205 in standard ROS maps.
    return occ != 205


def _build_background_rgb(occ: np.ndarray) -> np.ndarray:
    bg = np.zeros((occ.shape[0], occ.shape[1], 3), dtype=np.uint8)
    bg[:, :] = [8, 10, 120]                # canvas / unknown
    bg[occ != 205] = [255, 248, 0]         # known cells
    return bg


def _room_plot_extent(room: dict, pad_m: float):
    bb = room["bounding_box"]
    x0 = float(bb[0][0]) - pad_m
    y0 = float(bb[0][1]) - pad_m
    x1 = float(bb[1][0]) + pad_m
    y1 = float(bb[1][1]) + pad_m
    return x0, x1, y0, y1


def _pseudo_room_from_region(region: Region) -> dict:
    b = region.bound.detach().cpu().numpy()
    xmin, ymin = float(b[0, 0]), float(b[0, 1])
    xmax, ymax = float(b[1, 0]), float(b[1, 1])
    return {
        "id": int(region.version),
        "centroid_xy": [0.5 * (xmin + xmax), 0.5 * (ymin + ymax)],
        "bounding_box": [[xmin, ymin], [xmax, ymax]],
    }


def _sample_travel_times(
    region: Region,
    anchor_xy: np.ndarray,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    grid_n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.linspace(x0, x1, grid_n, dtype=np.float32)
    ys = np.linspace(y0, y1, grid_n, dtype=np.float32)
    X_w, Y_w = np.meshgrid(xs, ys)

    src_norm = region.normalize(
        np.array([[anchor_xy[0], anchor_xy[1], 0.0]], dtype=np.float32)
    )[0]
    query_world = np.stack(
        [X_w.reshape(-1), Y_w.reshape(-1), np.zeros(X_w.size, dtype=np.float32)],
        axis=1,
    )
    query_norm = region.normalize(query_world)

    dim = region.model.dim
    xp = np.zeros((X_w.size, 2 * dim), dtype=np.float32)
    xp[:, :dim] = src_norm
    xp[:, dim:] = src_norm
    xp[:, dim:dim + min(dim, query_norm.shape[1])] = query_norm[:, :dim]

    xp_t = torch.tensor(xp, dtype=torch.float32, device=region.device)
    with torch.no_grad():
        tt = region.function.TravelTimes(xp_t).detach().cpu().numpy().reshape(X_w.shape)
    return X_w, Y_w, tt


def _sample_debug_field(
    region: Region,
    anchor_xy: np.ndarray,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    grid_n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xs = np.linspace(x0, x1, grid_n, dtype=np.float32)
    ys = np.linspace(y0, y1, grid_n, dtype=np.float32)
    X_w, Y_w = np.meshgrid(xs, ys)

    anchor_norm = region.normalize(
        np.array([[anchor_xy[0], anchor_xy[1], 0.0]], dtype=np.float32)
    )[0]
    query_world = np.stack(
        [X_w.reshape(-1), Y_w.reshape(-1), np.zeros(X_w.size, dtype=np.float32)],
        axis=1,
    )
    query_norm = region.normalize(query_world)

    dim = region.model.dim
    xp = np.zeros((X_w.size, 2 * dim), dtype=np.float32)
    xp[:, :dim] = anchor_norm
    xp[:, dim:] = anchor_norm
    xp[:, : min(dim, query_norm.shape[1])] = query_norm[:, :dim]
    xp[:, 2] = anchor_norm[2]

    xp_t = torch.tensor(xp, dtype=torch.float32, device=region.device)
    with torch.no_grad():
        tt = region.function.TravelTimes(xp_t).detach().cpu().numpy().reshape(X_w.shape)
    xp_t_speed = torch.tensor(xp, dtype=torch.float32, device=region.device).requires_grad_(True)
    with torch.enable_grad():
        speed = region.function.Speed(xp_t_speed).detach().cpu().numpy().reshape(X_w.shape)
    return X_w, Y_w, tt, speed


def _finite_stats(arr: np.ndarray) -> dict[str, float] | None:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None
    return {
        "min": float(np.min(finite)),
        "p05": float(np.percentile(finite, 5)),
        "p25": float(np.percentile(finite, 25)),
        "p50": float(np.percentile(finite, 50)),
        "p75": float(np.percentile(finite, 75)),
        "p95": float(np.percentile(finite, 95)),
        "max": float(np.max(finite)),
        "std": float(np.std(finite)),
    }


def _format_stats(prefix: str, stats: dict[str, float] | None) -> str:
    if stats is None:
        return f"{prefix}: no finite samples"
    return (
        f"{prefix}: "
        f"min={stats['min']:.4f} p05={stats['p05']:.4f} p25={stats['p25']:.4f} "
        f"p50={stats['p50']:.4f} p75={stats['p75']:.4f} p95={stats['p95']:.4f} "
        f"max={stats['max']:.4f} std={stats['std']:.4f}"
    )


def _pair_distance_stats(rows: torch.Tensor | None) -> tuple[int, dict[str, float] | None]:
    if rows is None or len(rows) == 0:
        return 0, None
    arr = rows.detach().cpu().numpy()
    x0 = arr[:, 0:2].astype(np.float32)
    x1 = arr[:, 3:5].astype(np.float32)
    d = np.linalg.norm(x0 - x1, axis=1)
    return int(len(d)), _finite_stats(d)


def _mask_tt_to_room_and_map(
    region: Region,
    room: dict,
    X_w: np.ndarray,
    Y_w: np.ndarray,
    tt: np.ndarray,
    occ: np.ndarray,
    resolution: float,
    origin: np.ndarray,
) -> np.ndarray:
    b = region.bound.detach().cpu().numpy()
    bb = room["bounding_box"]

    in_region = (
        (X_w >= float(b[0, 0])) & (X_w <= float(b[1, 0])) &
        (Y_w >= float(b[0, 1])) & (Y_w <= float(b[1, 1]))
    )
    in_room = (
        (X_w >= float(bb[0][0])) & (X_w <= float(bb[1][0])) &
        (Y_w >= float(bb[0][1])) & (Y_w <= float(bb[1][1]))
    )

    H, W = occ.shape
    cols = np.floor((X_w - float(origin[0])) / resolution).astype(np.int32)
    rows = np.floor((Y_w - float(origin[1])) / resolution).astype(np.int32)
    inside = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)
    known = np.zeros_like(inside, dtype=bool)
    known[inside] = _known_mask_from_occ(occ)[rows[inside], cols[inside]]

    valid = np.isfinite(tt) & in_region & in_room & known
    return np.where(valid, tt, np.nan)


def _mask_field_to_room_and_map(
    region: Region,
    room: dict,
    X_w: np.ndarray,
    Y_w: np.ndarray,
    field: np.ndarray,
    occ: np.ndarray,
    resolution: float,
    origin: np.ndarray,
) -> np.ndarray:
    b = region.bound.detach().cpu().numpy()
    bb = room["bounding_box"]

    in_region = (
        (X_w >= float(b[0, 0])) & (X_w <= float(b[1, 0])) &
        (Y_w >= float(b[0, 1])) & (Y_w <= float(b[1, 1]))
    )
    in_room = (
        (X_w >= float(bb[0][0])) & (X_w <= float(bb[1][0])) &
        (Y_w >= float(bb[0][1])) & (Y_w <= float(bb[1][1]))
    )

    H, W = occ.shape
    cols = np.floor((X_w - float(origin[0])) / resolution).astype(np.int32)
    rows = np.floor((Y_w - float(origin[1])) / resolution).astype(np.int32)
    inside = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)
    known = np.zeros_like(inside, dtype=bool)
    known[inside] = _known_mask_from_occ(occ)[rows[inside], cols[inside]]

    valid = np.isfinite(field) & in_region & in_room & known
    return np.where(valid, field, np.nan)


def _plot_single_room(
    out_path: Path,
    room: dict,
    region: Region,
    occ: np.ndarray,
    resolution: float,
    origin: np.ndarray,
    pad_m: float,
    grid_n: int,
    contour_levels_major: np.ndarray,
    contour_levels_minor: np.ndarray,
    show_samples: bool,
    sample_limit: int,
    sample_rows: torch.Tensor | None = None,
    anchor_xy: np.ndarray | None = None,
    title_suffix: str = "",
) -> None:
    room_id = int(room["id"])
    centroid = np.asarray(room["centroid_xy"][:2], dtype=np.float32)
    anchor = centroid if anchor_xy is None else np.asarray(anchor_xy[:2], dtype=np.float32)
    x0, x1, y0, y1 = _room_plot_extent(room, pad_m)

    X_w, Y_w, tt = _sample_travel_times(region, anchor, x0, x1, y0, y1, grid_n)
    tt = _mask_tt_to_room_and_map(region, room, X_w, Y_w, tt, occ, resolution, origin)

    bg = _build_background_rgb(occ)
    H, W = occ.shape
    slam_x0 = float(origin[0])
    slam_y0 = float(origin[1])
    slam_x1 = slam_x0 + W * resolution
    slam_y1 = slam_y0 + H * resolution

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    canvas_blue = "#080a78"
    fig.patch.set_facecolor(canvas_blue)
    ax.set_facecolor(canvas_blue)

    ax.imshow(
        bg,
        origin="lower",
        extent=[slam_x0, slam_x1, slam_y0, slam_y1],
        interpolation="nearest",
        zorder=0,
    )

    finite = tt[np.isfinite(tt)]
    if finite.size > 0:
        t_min = float(np.min(finite))
        t_max = float(np.max(finite))
        major = contour_levels_major[
            (contour_levels_major >= t_min) & (contour_levels_major <= t_max)
        ]
        minor = contour_levels_minor[
            (contour_levels_minor >= t_min) & (contour_levels_minor <= t_max)
        ]
        if len(minor) >= 2:
            ax.contour(
                X_w, Y_w, tt,
                levels=minor,
                colors="#00b3b3",
                linewidths=0.35,
                alpha=0.55,
                zorder=2.9,
            )
        if len(major) >= 2:
            ax.contour(
                X_w, Y_w, tt,
                levels=major,
                colors="#00cccc",
                linewidths=0.85,
                alpha=0.96,
                zorder=3.0,
            )

    if show_samples:
        x0_pts, x1_pts = _room_sample_points(
            room,
            sample_limit=sample_limit,
            rows=sample_rows,
            region=region if sample_rows is None else None,
        )
        if len(x0_pts) > 0:
            ax.scatter(
                x0_pts[:, 0], x0_pts[:, 1],
                c="#ffffff", s=3, alpha=0.22, linewidths=0, zorder=4,
            )
        if len(x1_pts) > 0:
            ax.scatter(
                x1_pts[:, 0], x1_pts[:, 1],
                c="#ff4fd8", s=3, alpha=0.18, linewidths=0, zorder=4,
            )

    ax.scatter([float(anchor[0])], [float(anchor[1])],
               c="red", s=72, zorder=5, edgecolors="white", linewidths=0.6)

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        f"room {room_id}  region={region.version}  epoch={region.epoch}{title_suffix}",
        fontsize=10,
        color="white",
        pad=4,
    )
    fig.tight_layout(pad=0.25)
    fig.savefig(out_path, dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)


def _plot_single_room_debug_field(
    out_path: Path,
    room: dict,
    region: Region,
    occ: np.ndarray,
    resolution: float,
    origin: np.ndarray,
    pad_m: float,
    grid_n: int,
    contour_levels: np.ndarray,
    anchor_xy: np.ndarray | None = None,
    title_suffix: str = "",
) -> None:
    room_id = int(room["id"])
    centroid = np.asarray(room["centroid_xy"][:2], dtype=np.float32)
    anchor = centroid if anchor_xy is None else np.asarray(anchor_xy[:2], dtype=np.float32)
    x0, x1, y0, y1 = _room_plot_extent(room, pad_m)

    X_w, Y_w, tt, speed = _sample_debug_field(region, anchor, x0, x1, y0, y1, grid_n)
    tt = _mask_tt_to_room_and_map(region, room, X_w, Y_w, tt, occ, resolution, origin)
    speed = _mask_field_to_room_and_map(region, room, X_w, Y_w, speed, occ, resolution, origin)

    bg = _build_background_rgb(occ)
    H, W = occ.shape
    slam_x0 = float(origin[0])
    slam_y0 = float(origin[1])
    slam_x1 = slam_x0 + W * resolution
    slam_y1 = slam_y0 + H * resolution

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    canvas_blue = "#080a78"
    fig.patch.set_facecolor(canvas_blue)
    ax.set_facecolor(canvas_blue)

    ax.imshow(
        bg,
        origin="lower",
        extent=[slam_x0, slam_x1, slam_y0, slam_y1],
        interpolation="nearest",
        zorder=0,
    )

    speed_cmap = plt.get_cmap("viridis").copy()
    speed_cmap.set_bad(alpha=0.0)
    speed_plot = None
    speed_finite = speed[np.isfinite(speed)]
    if speed_finite.size > 0:
        speed_plot = ax.pcolormesh(
            X_w,
            Y_w,
            speed,
            vmin=0.0,
            vmax=1.0,
            cmap=speed_cmap,
            shading="auto",
            alpha=0.72,
            zorder=1.8,
        )

    tt_finite = tt[np.isfinite(tt)]
    if tt_finite.size > 0:
        levels = contour_levels[
            (contour_levels >= float(np.min(tt_finite))) &
            (contour_levels <= float(np.max(tt_finite)))
        ]
        if len(levels) >= 2:
            ax.contour(
                X_w,
                Y_w,
                tt,
                levels=levels,
                cmap="gist_heat",
                linewidths=0.5,
                alpha=0.78,
                zorder=2.8,
            )

    if speed_plot is not None:
        fig.colorbar(speed_plot, ax=ax, pad=0.02, shrink=0.8, label="Speed")

    ax.scatter([float(anchor[0])], [float(anchor[1])],
               c="red", s=72, zorder=5, edgecolors="white", linewidths=0.6)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        f"room {room_id}  region={region.version}  epoch={region.epoch}{title_suffix}",
        fontsize=10,
        color="white",
        pad=4,
    )
    fig.tight_layout(pad=0.25)
    fig.savefig(out_path, dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)


def _latest_backup_dir(model_dir: Path, region_ver: int) -> Path | None:
    backup_root = model_dir / "_backups"
    if not backup_root.exists():
        return None
    matches = sorted(
        backup_root.glob(f"region_{int(region_ver)}_*"),
        key=lambda p: p.stat().st_mtime,
    )
    return matches[-1] if matches else None


def _load_backup_region(backup_dir: Path, region_ver: int, device: str) -> Region | None:
    tmp_root = Path(tempfile.mkdtemp(prefix=f"mntfields_backup_r{int(region_ver)}_"))
    try:
        staged = tmp_root / f"region_{int(region_ver)}"
        shutil.copytree(backup_dir, staged)
        regions = _load_regions(tmp_root, device)
        region = regions.get(int(region_ver))
        if region is None:
            shutil.rmtree(tmp_root, ignore_errors=True)
            return None
        region._tmp_model_root = tmp_root
        return region
    except Exception:
        shutil.rmtree(tmp_root, ignore_errors=True)
        return None


def _cleanup_temp_region(region: Region | None) -> None:
    tmp_root = getattr(region, "_tmp_model_root", None)
    if tmp_root is not None:
        shutil.rmtree(tmp_root, ignore_errors=True)


def _plot_backup_compare_debug_field(
    out_path: Path,
    room: dict,
    current_region: Region,
    backup_region: Region,
    occ: np.ndarray,
    resolution: float,
    origin: np.ndarray,
    pad_m: float,
    grid_n: int,
    contour_levels: np.ndarray,
) -> None:
    room_id = int(room["id"])
    anchor = np.asarray(room["centroid_xy"][:2], dtype=np.float32)
    x0, x1, y0, y1 = _room_plot_extent(room, pad_m)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    canvas_blue = "#080a78"
    fig.patch.set_facecolor(canvas_blue)
    bg = _build_background_rgb(occ)
    H, W = occ.shape
    slam_x0 = float(origin[0])
    slam_y0 = float(origin[1])
    slam_x1 = slam_x0 + W * resolution
    slam_y1 = slam_y0 + H * resolution

    for ax, region, label in (
        (axes[0], current_region, "current"),
        (axes[1], backup_region, "latest backup"),
    ):
        ax.set_facecolor(canvas_blue)
        ax.imshow(
            bg,
            origin="lower",
            extent=[slam_x0, slam_x1, slam_y0, slam_y1],
            interpolation="nearest",
            zorder=0,
        )
        X_w, Y_w, tt, speed = _sample_debug_field(region, anchor, x0, x1, y0, y1, grid_n)
        tt = _mask_tt_to_room_and_map(region, room, X_w, Y_w, tt, occ, resolution, origin)
        speed = _mask_field_to_room_and_map(region, room, X_w, Y_w, speed, occ, resolution, origin)

        speed_cmap = plt.get_cmap("viridis").copy()
        speed_cmap.set_bad(alpha=0.0)
        speed_plot = None
        if np.isfinite(speed).any():
            speed_plot = ax.pcolormesh(
                X_w, Y_w, speed,
                vmin=0.0, vmax=1.0,
                cmap=speed_cmap,
                shading="auto",
                alpha=0.72,
                zorder=1.8,
            )
        tt_finite = tt[np.isfinite(tt)]
        if tt_finite.size > 0:
            levels = contour_levels[
                (contour_levels >= float(np.min(tt_finite))) &
                (contour_levels <= float(np.max(tt_finite)))
            ]
            if len(levels) >= 2:
                ax.contour(
                    X_w, Y_w, tt,
                    levels=levels,
                    cmap="gist_heat",
                    linewidths=0.5,
                    alpha=0.78,
                    zorder=2.8,
                )
        ax.scatter([float(anchor[0])], [float(anchor[1])],
                   c="red", s=72, zorder=5, edgecolors="white", linewidths=0.6)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"{label}  epoch={region.epoch}", fontsize=10, color="white", pad=4)
        if speed_plot is not None:
            fig.colorbar(speed_plot, ax=ax, pad=0.02, shrink=0.8, label="Speed")

    fig.suptitle(f"room {room_id}  region={current_region.version}  centroid field vs backup", fontsize=11, color="white")
    fig.tight_layout(pad=0.5)
    fig.savefig(out_path, dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)

    current_X, current_Y, current_tt, current_speed = _sample_debug_field(
        current_region, anchor, x0, x1, y0, y1, grid_n
    )
    current_tt = _mask_tt_to_room_and_map(current_region, room, current_X, current_Y, current_tt, occ, resolution, origin)
    current_speed = _mask_field_to_room_and_map(current_region, room, current_X, current_Y, current_speed, occ, resolution, origin)
    backup_X, backup_Y, backup_tt, backup_speed = _sample_debug_field(
        backup_region, anchor, x0, x1, y0, y1, grid_n
    )
    backup_tt = _mask_tt_to_room_and_map(backup_region, room, backup_X, backup_Y, backup_tt, occ, resolution, origin)
    backup_speed = _mask_field_to_room_and_map(backup_region, room, backup_X, backup_Y, backup_speed, occ, resolution, origin)

    current_rows_owned = _room_owned_rows(room, region=current_region)
    backup_rows_owned = _room_owned_rows(room, region=backup_region)
    current_pair_n, current_pair_stats = _pair_distance_stats(current_rows_owned)
    backup_pair_n, backup_pair_stats = _pair_distance_stats(backup_rows_owned)

    print(f"[STATS] room {room_id} region={current_region.version} centroid current speed   :: {_format_stats('speed', _finite_stats(current_speed))}")
    print(f"[STATS] room {room_id} region={current_region.version} centroid current tt      :: {_format_stats('tt', _finite_stats(current_tt))}")
    print(f"[STATS] room {room_id} region={current_region.version} current rows owned      :: count={current_pair_n} {_format_stats('x0->x1', current_pair_stats)}")
    print(f"[STATS] room {room_id} region={backup_region.version} centroid backup  speed   :: {_format_stats('speed', _finite_stats(backup_speed))}")
    print(f"[STATS] room {room_id} region={backup_region.version} centroid backup  tt      :: {_format_stats('tt', _finite_stats(backup_tt))}")
    print(f"[STATS] room {room_id} region={backup_region.version} backup  rows owned      :: count={backup_pair_n} {_format_stats('x0->x1', backup_pair_stats)}")


def _plot_training_vis_room(
    out_path: Path,
    room: dict,
    region: Region,
    rows: torch.Tensor | None = None,
    row_mode: str = "both",
) -> bool:
    if row_mode == "x0":
        rows = _room_owned_rows(room, rows=rows, region=region if rows is None else None)
    else:
        rows = _room_raw_rows(room, rows=rows, region=region if rows is None else None)
    if rows is None or len(rows) == 0:
        return False

    n14 = region.calculate_speed_and_normal(rows)
    if n14 is None or len(n14) == 0:
        return False

    pts = n14[:, :6].detach().cpu().numpy()
    spd = n14[:, 6:8].detach().cpu().numpy()

    x0_world = region._denorm_points(pts[:, :3])
    x1_world = region._denorm_points(pts[:, 3:6])
    dists_world = np.linalg.norm(x0_world[:, :2] - x1_world[:, :2], axis=1)
    far_enough = dists_world >= 0.05

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    cmap = cm.get_cmap("viridis")
    vmin, vmax = float(spd.min()), float(spd.max())

    ax[0].scatter(
        x0_world[:, 0], x0_world[:, 1],
        c=spd[:, 0], cmap=cmap, vmin=vmin, vmax=vmax, s=6, alpha=0.7,
    )
    ax[0].set_title(f"Start x0+y0 (N={len(pts)})")
    ax[0].set_aspect("equal")

    x1_vis = x1_world[far_enough]
    spd1_vis = spd[far_enough, 1]
    sort_idx = np.argsort(spd1_vis)
    ax[1].scatter(
        x1_vis[sort_idx, 0], x1_vis[sort_idx, 1],
        c=spd1_vis[sort_idx], cmap=cmap, vmin=vmin, vmax=vmax, s=6, alpha=0.7,
    )
    ax[1].set_title(f"Goal x1+y1 (N={far_enough.sum()})")
    ax[1].set_aspect("equal")

    ax[2].hist(dists_world, bins=40, alpha=0.8, color="orange", density=True)
    ax[2].axvline(
        dists_world.mean(), color="red", lw=2,
        label=f"mu={dists_world.mean():.2f}m",
    )
    ax[2].set_xlabel("World Distance (m)")
    ax[2].set_title("x0->x1 Dist")
    ax[2].legend()

    fig.colorbar(
        cm.ScalarMappable(cmap=cmap, norm=colors.Normalize(vmin=vmin, vmax=vmax)),
        ax=ax, label="Norm Speed", shrink=0.8,
    )

    b = region.bound.detach().cpu().numpy()
    xmin, ymin = float(b[0, 0]), float(b[0, 1])
    xmax, ymax = float(b[1, 0]), float(b[1, 1])
    for axi in ax[:2]:
        rect = plt.Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin,
            linewidth=2, edgecolor="white", facecolor="none", zorder=10,
        )
        axi.add_patch(rect)
        axi.set_xlim(xmin, xmax)
        axi.set_ylim(ymin, ymax)

    room_id = int(room["id"])
    fig.suptitle(
        f"room {room_id}  region={region.version}  epoch={region.epoch}",
        fontsize=11,
    )
    fig.savefig(out_path, dpi=90, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_combined_rooms(
    out_path: Path,
    rooms: list[dict],
    room_to_region: dict[int, int],
    regions: dict[int, Region],
    occ: np.ndarray,
    resolution: float,
    origin: np.ndarray,
    grid_n: int,
    contour_levels_major: np.ndarray,
    contour_levels_minor: np.ndarray,
    show_samples: bool,
    sample_limit: int,
    sample_rows_by_room: dict[int, torch.Tensor] | None = None,
) -> None:
    bg = _build_background_rgb(occ)
    H, W = occ.shape
    slam_x0 = float(origin[0])
    slam_y0 = float(origin[1])
    slam_x1 = slam_x0 + W * resolution
    slam_y1 = slam_y0 + H * resolution

    fig, ax = plt.subplots(1, 1, figsize=(13, 13))
    canvas_blue = "#080a78"
    fig.patch.set_facecolor(canvas_blue)
    ax.set_facecolor(canvas_blue)

    ax.imshow(
        bg,
        origin="lower",
        extent=[slam_x0, slam_x1, slam_y0, slam_y1],
        interpolation="nearest",
        zorder=0,
    )

    for room in rooms:
        room_id = int(room["id"])
        region_ver = int(room_to_region.get(room_id, room_id))
        region = regions.get(region_ver)
        if region is None:
            continue
        centroid = np.asarray(room["centroid_xy"][:2], dtype=np.float32)
        bb = room["bounding_box"]
        x0 = float(bb[0][0])
        y0 = float(bb[0][1])
        x1 = float(bb[1][0])
        y1 = float(bb[1][1])
        if x1 <= x0 or y1 <= y0:
            continue

        X_w, Y_w, tt = _sample_travel_times(region, centroid, x0, x1, y0, y1, grid_n)
        tt = _mask_tt_to_room_and_map(region, room, X_w, Y_w, tt, occ, resolution, origin)
        finite = tt[np.isfinite(tt)]
        if finite.size == 0:
            continue

        t_min = float(np.min(finite))
        t_max = float(np.max(finite))
        major = contour_levels_major[
            (contour_levels_major >= t_min) & (contour_levels_major <= t_max)
        ]
        minor = contour_levels_minor[
            (contour_levels_minor >= t_min) & (contour_levels_minor <= t_max)
        ]

        if len(minor) >= 2:
            ax.contour(
                X_w, Y_w, tt,
                levels=minor,
                colors="#00b3b3",
                linewidths=0.28,
                alpha=0.40,
                zorder=2.8,
            )
        if len(major) >= 2:
            ax.contour(
                X_w, Y_w, tt,
                levels=major,
                colors="#00cccc",
                linewidths=0.65,
                alpha=0.82,
                zorder=3.0,
            )

        if show_samples:
            room_rows = None if sample_rows_by_room is None else sample_rows_by_room.get(room_id)
            x0_pts, x1_pts = _room_sample_points(
                room,
                sample_limit=sample_limit,
                rows=room_rows,
                region=region if room_rows is None else None,
            )
            if len(x0_pts) > 0:
                ax.scatter(
                    x0_pts[:, 0], x0_pts[:, 1],
                    c="#ffffff", s=2, alpha=0.10, linewidths=0, zorder=4,
                )
            if len(x1_pts) > 0:
                ax.scatter(
                    x1_pts[:, 0], x1_pts[:, 1],
                    c="#ff4fd8", s=2, alpha=0.08, linewidths=0, zorder=4,
                )

        ax.scatter([float(centroid[0])], [float(centroid[1])],
                   c="#ff8844", s=32, zorder=5, edgecolors="white", linewidths=0.4)
        ax.text(
            float(centroid[0]), float(centroid[1]),
            f"R{region_ver}/room {room_id}",
            color="white",
            fontsize=8,
            ha="center",
            va="bottom",
            zorder=6,
        )

    ax.set_xlim(slam_x0, slam_x1)
    ax.set_ylim(slam_y0, slam_y1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("combined room contours", fontsize=11, color="white", pad=4)
    fig.tight_layout(pad=0.25)
    fig.savefig(out_path, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)


def _room_anchor_specs(room: dict, topology: dict) -> list[tuple[str, np.ndarray, str]]:
    specs: list[tuple[str, np.ndarray, str]] = []
    centroid = np.asarray(room["centroid_xy"][:2], dtype=np.float32)
    specs.append(("centroid", centroid, " [anchor=centroid]"))

    doors_by_id = {int(d["id"]): d for d in topology.get("doors", [])}
    for door_id in room.get("doors", []) or []:
        door = doors_by_id.get(int(door_id))
        if not door:
            continue
        pos = np.asarray(door["pos_xy"][:2], dtype=np.float32)
        specs.append((f"door_{int(door_id):03d}", pos, f" [anchor=door {int(door_id)}]"))
    return specs


def _plot_combined_debug_field(
    out_path: Path,
    rooms: list[dict],
    room_to_region: dict[int, int],
    regions: dict[int, Region],
    occ: np.ndarray,
    resolution: float,
    origin: np.ndarray,
    grid_n: int,
    contour_levels: np.ndarray,
) -> None:
    bg = _build_background_rgb(occ)
    H, W = occ.shape
    slam_x0 = float(origin[0])
    slam_y0 = float(origin[1])
    slam_x1 = slam_x0 + W * resolution
    slam_y1 = slam_y0 + H * resolution

    fig, ax = plt.subplots(1, 1, figsize=(13, 13))
    canvas_blue = "#080a78"
    fig.patch.set_facecolor(canvas_blue)
    ax.set_facecolor(canvas_blue)

    ax.imshow(
        bg,
        origin="lower",
        extent=[slam_x0, slam_x1, slam_y0, slam_y1],
        interpolation="nearest",
        zorder=0,
    )

    speed_mappable = None
    speed_cmap = plt.get_cmap("viridis").copy()
    speed_cmap.set_bad(alpha=0.0)

    for room in rooms:
        room_id = int(room["id"])
        region_ver = int(room_to_region.get(room_id, room_id))
        region = regions.get(region_ver)
        if region is None:
            continue
        centroid = np.asarray(room["centroid_xy"][:2], dtype=np.float32)
        bb = room["bounding_box"]
        x0 = float(bb[0][0])
        y0 = float(bb[0][1])
        x1 = float(bb[1][0])
        y1 = float(bb[1][1])
        if x1 <= x0 or y1 <= y0:
            continue

        X_w, Y_w, tt, speed = _sample_debug_field(region, centroid, x0, x1, y0, y1, grid_n)
        tt = _mask_tt_to_room_and_map(region, room, X_w, Y_w, tt, occ, resolution, origin)
        speed = _mask_field_to_room_and_map(region, room, X_w, Y_w, speed, occ, resolution, origin)

        speed_finite = speed[np.isfinite(speed)]
        if speed_finite.size > 0:
            speed_mappable = ax.pcolormesh(
                X_w,
                Y_w,
                speed,
                vmin=0.0,
                vmax=1.0,
                cmap=speed_cmap,
                shading="auto",
                alpha=0.70,
                zorder=1.8,
            )

        tt_finite = tt[np.isfinite(tt)]
        if tt_finite.size > 0:
            levels = contour_levels[(contour_levels >= float(np.min(tt_finite))) & (contour_levels <= float(np.max(tt_finite)))]
            if len(levels) >= 2:
                ax.contour(
                    X_w,
                    Y_w,
                    tt,
                    levels=levels,
                    cmap="gist_heat",
                    linewidths=0.45,
                    alpha=0.75,
                    zorder=2.8,
                )

        ax.scatter([float(centroid[0])], [float(centroid[1])],
                   c="red", s=24, zorder=4, edgecolors="white", linewidths=0.35)
        ax.text(
            float(centroid[0]), float(centroid[1]),
            f"R{region_ver}",
            color="white",
            fontsize=8,
            ha="center",
            va="bottom",
            zorder=5,
        )

    if speed_mappable is not None:
        fig.colorbar(speed_mappable, ax=ax, pad=0.02, shrink=0.8, label="Speed")

    ax.set_xlim(slam_x0, slam_x1)
    ax.set_ylim(slam_y0, slam_y1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("combined room debug field", fontsize=11, color="white", pad=4)
    fig.tight_layout(pad=0.25)
    fig.savefig(out_path, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Render one NTField contour plot per room.")
    parser.add_argument(
        "--model-dir",
        default=str(Path.home() / "turtlebot3_ws" / "mntfields_model"),
        help="Saved model directory containing region_*/ and topology.json",
    )
    parser.add_argument(
        "--out-dir",
        default="",
        help="Output directory for PNGs (default: <model-dir>/room_contours)",
    )
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Torch device for evaluating TravelTimes",
    )
    parser.add_argument(
        "--pad-m",
        type=float,
        default=0.6,
        help="Extra world-space padding around each room bounding box",
    )
    parser.add_argument(
        "--grid-n",
        type=int,
        default=160,
        help="Grid resolution per room plot",
    )
    parser.add_argument(
        "--skip-combined",
        action="store_true",
        help="Only write per-room PNGs, skip the combined full-map figure",
    )
    parser.add_argument(
        "--combined-field",
        action="store_true",
        help="Also write a stitched debug-style speed+contour plot for all rooms",
    )
    parser.add_argument(
        "--style",
        choices=["contours", "training", "both"],
        default="contours",
        help="Per-room output style",
    )
    parser.add_argument(
        "--show-samples",
        action="store_true",
        help="Overlay saved raw training sample endpoints (x0 in white, x1 in magenta)",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=2500,
        help="Maximum x0/x1 points to draw per room overlay",
    )
    parser.add_argument(
        "--merge-model-dirs",
        nargs="*",
        default=[],
        help=(
            "Additional model directories whose raw region data should be pooled, "
            "deduplicated, and repartitioned by the target room boxes for sample/training plots"
        ),
    )
    parser.add_argument(
        "--row-mode",
        choices=["both", "x0"],
        default="x0",
        help=(
            "How to assign raw rows to a room for training-style plots: "
            "'both' requires x0 and x1 inside the room box; "
            "'x0' keeps rows owned by the room via x0 only"
        ),
    )
    parser.add_argument(
        "--anchor-room-ids",
        nargs="*",
        type=int,
        default=[],
        help="Write extra per-anchor contour plots (centroid + each doorway) for these room ids",
    )
    parser.add_argument(
        "--debug-anchor-room-ids",
        nargs="*",
        type=int,
        default=[],
        help="Write extra per-anchor debug-field plots (field.png style) for these room ids",
    )
    parser.add_argument(
        "--compare-backup-room-ids",
        nargs="*",
        type=int,
        default=[],
        help="Write side-by-side current vs latest-backup centroid field plots for these room ids",
    )
    parser.add_argument(
        "--region-ids",
        nargs="*",
        type=int,
        default=[],
        help="Write direct centroid contour plots for these region ids using the region bounds",
    )
    parser.add_argument(
        "--debug-region-ids",
        nargs="*",
        type=int,
        default=[],
        help="Write direct centroid debug-field plots for these region ids using the region bounds",
    )
    args = parser.parse_args()

    model_dir = Path(os.path.expanduser(args.model_dir)).resolve()
    out_dir = Path(os.path.expanduser(args.out_dir)).resolve() if args.out_dir else (model_dir / "room_contours")
    out_dir.mkdir(parents=True, exist_ok=True)

    regions = _load_regions(model_dir, args.device)
    if not regions:
        raise RuntimeError(f"No region bundles could be loaded from {model_dir}")

    topo = _load_json(model_dir / "topology.json")
    rooms = topo.get("rooms", [])
    room_to_region = _load_room_to_region(model_dir)
    if not rooms and not args.region_ids and not args.debug_region_ids:
        raise RuntimeError(
            f"No rooms found in {model_dir / 'topology.json'} and no --region-ids/--debug-region-ids were provided"
        )

    merged_rows = None
    merged_room_rows: dict[int, torch.Tensor] = {}
    merge_dirs = [Path(os.path.expanduser(p)).resolve() for p in args.merge_model_dirs]
    if merge_dirs:
        pooled = _load_pooled_rows_from_model_dirs([model_dir, *merge_dirs])
        if pooled is not None and len(pooled) > 0:
            merged_rows = pooled
            for room in rooms:
                room_id = int(room["id"])
                if args.row_mode == "x0":
                    room_rows = _room_owned_rows(room, rows=merged_rows)
                else:
                    room_rows = _room_raw_rows(room, rows=merged_rows)
                if room_rows is not None and len(room_rows) > 0:
                    merged_room_rows[room_id] = room_rows
            print(
                f"[POOL] merged raw rows: {len(merged_rows)} unique rows across "
                f"{1 + len(merge_dirs)} model dir(s); usable rooms={sorted(merged_room_rows)}"
            )
        else:
            print("[POOL] merged raw rows: no usable rows found in requested model dirs")

    occ, resolution, origin, _meta = _load_map(model_dir)
    contour_levels_major = np.arange(0.0, 10.0, 0.05, dtype=np.float32)
    contour_levels_minor = np.arange(0.0, 10.0, 0.25, dtype=np.float32)
    debug_field_levels = np.arange(0.0, 10.0, 0.03, dtype=np.float32)

    written = 0
    anchor_room_ids = set(int(r) for r in args.anchor_room_ids)
    debug_anchor_room_ids = set(int(r) for r in args.debug_anchor_room_ids)
    compare_backup_room_ids = set(int(r) for r in args.compare_backup_room_ids)
    region_ids = set(int(r) for r in args.region_ids)
    debug_region_ids = set(int(r) for r in args.debug_region_ids)
    for room in rooms:
        room_id = int(room["id"])
        region_ver = int(room_to_region.get(room_id, room_id))
        region = regions.get(region_ver)
        if region is None:
            print(f"[SKIP] room {room_id}: region {region_ver} not loaded")
            continue
        if args.style in ("contours", "both"):
            out_path = out_dir / f"room_{room_id:03d}_region_{region_ver:03d}.png"
            _plot_single_room(
                out_path=out_path,
                room=room,
                region=region,
                occ=occ,
                resolution=resolution,
                origin=origin,
                pad_m=float(args.pad_m),
                grid_n=int(args.grid_n),
                contour_levels_major=contour_levels_major,
                contour_levels_minor=contour_levels_minor,
                show_samples=bool(args.show_samples),
                sample_limit=int(args.sample_limit),
                sample_rows=merged_room_rows.get(room_id),
            )
            print(f"[WRITE] room {room_id} contours -> {out_path}")
            written += 1
            if room_id in anchor_room_ids:
                for anchor_name, anchor_xy, title_suffix in _room_anchor_specs(room, topo):
                    anchor_out = out_dir / f"room_{room_id:03d}_region_{region_ver:03d}_{anchor_name}.png"
                    _plot_single_room(
                        out_path=anchor_out,
                        room=room,
                        region=region,
                        occ=occ,
                        resolution=resolution,
                        origin=origin,
                        pad_m=float(args.pad_m),
                        grid_n=int(args.grid_n),
                        contour_levels_major=contour_levels_major,
                        contour_levels_minor=contour_levels_minor,
                        show_samples=bool(args.show_samples),
                        sample_limit=int(args.sample_limit),
                        sample_rows=merged_room_rows.get(room_id),
                        anchor_xy=anchor_xy,
                        title_suffix=title_suffix,
                    )
                    print(f"[WRITE] room {room_id} anchor plot -> {anchor_out}")
                    written += 1
            if room_id in debug_anchor_room_ids:
                for anchor_name, anchor_xy, title_suffix in _room_anchor_specs(room, topo):
                    debug_anchor_out = out_dir / f"room_{room_id:03d}_region_{region_ver:03d}_{anchor_name}_field.png"
                    _plot_single_room_debug_field(
                        out_path=debug_anchor_out,
                        room=room,
                        region=region,
                        occ=occ,
                        resolution=resolution,
                        origin=origin,
                        pad_m=float(args.pad_m),
                        grid_n=int(args.grid_n),
                        contour_levels=debug_field_levels,
                        anchor_xy=anchor_xy,
                        title_suffix=title_suffix,
                    )
                    print(f"[WRITE] room {room_id} debug anchor plot -> {debug_anchor_out}")
                    written += 1
            if room_id in compare_backup_room_ids:
                backup_dir = _latest_backup_dir(model_dir, region_ver)
                backup_region = _load_backup_region(backup_dir, region_ver, args.device) if backup_dir else None
                if backup_region is None:
                    print(f"[SKIP] room {room_id}: no latest backup available for region {region_ver}")
                else:
                    compare_out = out_dir / f"room_{room_id:03d}_region_{region_ver:03d}_compare_backup_field.png"
                    try:
                        _plot_backup_compare_debug_field(
                            out_path=compare_out,
                            room=room,
                            current_region=region,
                            backup_region=backup_region,
                            occ=occ,
                            resolution=resolution,
                            origin=origin,
                            pad_m=float(args.pad_m),
                            grid_n=int(args.grid_n),
                            contour_levels=debug_field_levels,
                        )
                        print(f"[WRITE] room {room_id} backup compare -> {compare_out}")
                        written += 1
                    finally:
                        _cleanup_temp_region(backup_region)
        if args.style in ("training", "both"):
            train_out = out_dir / f"room_{room_id:03d}_region_{region_ver:03d}_training.png"
            if _plot_training_vis_room(
                train_out,
                room,
                region,
                rows=merged_room_rows.get(room_id),
                row_mode=str(args.row_mode),
            ):
                print(f"[WRITE] room {room_id} training -> {train_out}")
                written += 1
            else:
                print(f"[SKIP] room {room_id}: no usable training-vis rows")

    for region_ver in sorted(region_ids | debug_region_ids):
        region = regions.get(region_ver)
        if region is None:
            print(f"[SKIP] region {region_ver}: not loaded")
            continue
        pseudo_room = _pseudo_room_from_region(region)
        if region_ver in region_ids:
            out_path = out_dir / f"region_{region_ver:03d}_centroid.png"
            _plot_single_room(
                out_path=out_path,
                room=pseudo_room,
                region=region,
                occ=occ,
                resolution=resolution,
                origin=origin,
                pad_m=float(args.pad_m),
                grid_n=int(args.grid_n),
                contour_levels_major=contour_levels_major,
                contour_levels_minor=contour_levels_minor,
                show_samples=bool(args.show_samples),
                sample_limit=int(args.sample_limit),
                sample_rows=None,
                anchor_xy=np.asarray(pseudo_room["centroid_xy"][:2], dtype=np.float32),
                title_suffix=" [anchor=centroid]",
            )
            print(f"[WRITE] region {region_ver} centroid contour -> {out_path}")
            written += 1
        if region_ver in debug_region_ids:
            out_path = out_dir / f"region_{region_ver:03d}_centroid_field.png"
            _plot_single_room_debug_field(
                out_path=out_path,
                room=pseudo_room,
                region=region,
                occ=occ,
                resolution=resolution,
                origin=origin,
                pad_m=float(args.pad_m),
                grid_n=int(args.grid_n),
                contour_levels=debug_field_levels,
                anchor_xy=np.asarray(pseudo_room["centroid_xy"][:2], dtype=np.float32),
                title_suffix=" [anchor=centroid]",
            )
            print(f"[WRITE] region {region_ver} centroid debug field -> {out_path}")
            written += 1

    if rooms and (args.style in ("contours", "both")) and not args.skip_combined:
        combined_path = out_dir / "all_rooms_combined.png"
        _plot_combined_rooms(
            out_path=combined_path,
            rooms=rooms,
            room_to_region=room_to_region,
            regions=regions,
            occ=occ,
            resolution=resolution,
            origin=origin,
            grid_n=max(120, int(args.grid_n)),
            contour_levels_major=contour_levels_major,
            contour_levels_minor=contour_levels_minor,
            show_samples=bool(args.show_samples),
            sample_limit=int(args.sample_limit),
            sample_rows_by_room=merged_room_rows if merged_room_rows else None,
        )
        print(f"[WRITE] combined -> {combined_path}")
        if args.combined_field:
            combined_field_path = out_dir / "all_rooms_field_combined.png"
            _plot_combined_debug_field(
                out_path=combined_field_path,
                rooms=rooms,
                room_to_region=room_to_region,
                regions=regions,
                occ=occ,
                resolution=resolution,
                origin=origin,
                grid_n=max(120, int(args.grid_n)),
                contour_levels=debug_field_levels,
            )
            print(f"[WRITE] combined field -> {combined_field_path}")

    print(f"[DONE] wrote {written} plot(s) to {out_dir}")


if __name__ == "__main__":
    main()
