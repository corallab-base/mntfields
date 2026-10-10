#!/usr/bin/env python3
"""
Create a corrected refined bundle where rooms/regions 1 and 5 are merged.

Default:
  input  = ~/turtlebot3_ws/mntfield_runs/mntfields_model_mapping_refined
  output = ~/turtlebot3_ws/mntfield_runs/refined_corrected

Behavior:
  - copies the refined bundle to the output directory
  - merges room 5 into room 1
  - removes false doorway 14 (between 1 and 5)
  - remaps doorway 16 to connect 1 and 6
  - retrains region_1 from the combined raw rows of region_1 and region_5
  - removes region_5 from the output bundle
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

import torch

_WS = Path(__file__).resolve().parents[3]
_INSTALL_LIB = _WS / "install" / "mntfields" / "lib" / "python3.10" / "site-packages"
if _INSTALL_LIB.exists():
    sys.path.insert(0, str(_INSTALL_LIB))

from mntfields.mntfields_core.core.region import Region

DEFAULT_IN = _WS / "mntfield_runs" / "mntfields_model_mapping_refined"
DEFAULT_OUT = _WS / "mntfield_runs" / "refined_corrected"

def _load_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _write_json(path: Path, obj: dict) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def _bbox_union(bb_a, bb_b):
    return [
        [min(float(bb_a[0][0]), float(bb_b[0][0])), min(float(bb_a[0][1]), float(bb_b[0][1]))],
        [max(float(bb_a[1][0]), float(bb_b[1][0])), max(float(bb_a[1][1]), float(bb_b[1][1]))],
    ]


def _bound_union(bound_a, bound_b):
    return [
        [
            min(float(bound_a[0][0]), float(bound_b[0][0])),
            min(float(bound_a[0][1]), float(bound_b[0][1])),
            min(float(bound_a[0][2]), float(bound_b[0][2])),
        ],
        [
            max(float(bound_a[1][0]), float(bound_b[1][0])),
            max(float(bound_a[1][1]), float(bound_b[1][1])),
            max(float(bound_a[1][2]), float(bound_b[1][2])),
        ],
    ]


def _bbox_area(bb) -> float:
    return max(0.0, float(bb[1][0]) - float(bb[0][0])) * max(0.0, float(bb[1][1]) - float(bb[0][1]))


def _weighted_centroid(room_a: dict, room_b: dict):
    a1 = _bbox_area(room_a["bounding_box"])
    a2 = _bbox_area(room_b["bounding_box"])
    c1 = room_a.get("centroid_xy", room_a.get("centroid"))
    c2 = room_b.get("centroid_xy", room_b.get("centroid"))
    if a1 <= 0 and a2 <= 0:
        bb = _bbox_union(room_a["bounding_box"], room_b["bounding_box"])
        return [(bb[0][0] + bb[1][0]) * 0.5, (bb[0][1] + bb[1][1]) * 0.5]
    w1 = a1 / max(1e-6, a1 + a2)
    w2 = 1.0 - w1
    return [
        float(c1[0]) * w1 + float(c2[0]) * w2,
        float(c1[1]) * w1 + float(c2[1]) * w2,
    ]


def _load_region(out_dir: Path, version: int, device: str) -> Region:
    meta = _load_json(out_dir / f"region_{version}" / f"region_{version}_meta.json")
    region = Region(
        model_folder=str(out_dir),
        bound=torch.tensor(meta["bound"], dtype=torch.float32),
        version=version,
        device=device,
    )
    if not region.load_model():
        raise RuntimeError(f"could not load region {version} from {out_dir}")
    return region


def _concat_frame_rows(data_dict: dict) -> torch.Tensor:
    frames = data_dict.get("frame_data", []) or []
    parts = [p.detach().cpu().float() for p in frames if isinstance(p, torch.Tensor) and p.ndim == 2 and p.shape[1] == 12]
    if not parts:
        return torch.zeros((0, 12), dtype=torch.float32)
    merged = torch.cat(parts, dim=0)
    try:
        merged = torch.unique(merged, dim=0)
    except Exception:
        pass
    return merged


def _merge_semantics(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a or {})
    for cls, pts in (b or {}).items():
        out.setdefault(cls, []).extend(copy.deepcopy(pts))
    return out


def _cosine_lr(epoch: int, lr_min: float, lr_max: float, cycle: int) -> float:
    t = epoch % cycle
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * t / cycle))


def _retrain_merged_region(out_dir: Path, epochs: int, lr: float, device: str) -> None:
    data1 = torch.load(out_dir / "region_1" / "region_1_data.pth", map_location="cpu", weights_only=False)
    data5 = torch.load(out_dir / "region_5" / "region_5_data.pth", map_location="cpu", weights_only=False)
    rows = torch.cat([_concat_frame_rows(data1), _concat_frame_rows(data5)], dim=0)
    if len(rows) == 0:
        raise RuntimeError("no raw rows to retrain merged region")
    try:
        rows = torch.unique(rows, dim=0)
    except Exception:
        pass

    meta1 = _load_json(out_dir / "region_1" / "region_1_meta.json")
    meta5 = _load_json(out_dir / "region_5" / "region_5_meta.json")
    merged_bound = torch.tensor(_bound_union(meta1["bound"], meta5["bound"]), dtype=torch.float32)

    reg = _load_region(out_dir, 1, device)
    reg.bound = merged_bound.clone()
    reg._sync_model_refs()
    reg.all_raw_frame_data = [rows]
    reg.semantic_objects = _merge_semantics(
        data1.get("semantic_objects", {}) if isinstance(data1, dict) else {},
        data5.get("semantic_objects", {}) if isinstance(data5, dict) else {},
    )

    n14 = reg.calculate_speed_and_normal(rows.to(reg.device))
    if n14 is None or len(n14) == 0:
        raise RuntimeError("merged region has no valid Nx14 rows")

    lr_min = lr * 0.1
    lr_max = lr
    lr_cycle = 200
    best_loss = float("inf")
    best_state = copy.deepcopy(reg.model.network.state_dict())
    start_epoch = int(max(meta1.get("epoch", 0), meta5.get("epoch", 0)))
    t0 = time.time()

    for ep in range(1, epochs + 1):
        cur_lr = _cosine_lr(ep, lr_min, lr_max, lr_cycle)
        reg.model.optimizer.param_groups[0]["lr"] = cur_lr

        perm = torch.randperm(len(n14), device=reg.device)
        batch = n14[perm]
        if len(batch) > 8000:
            batch = batch[:8000]

        loss_out = reg.model.function.Loss(
            batch[:, :6].float(),
            (batch[:, 6:8].float() * batch[:, 6:8].float() * (2 - batch[:, 6:8].float()) * (2 - batch[:, 6:8].float())),
            batch[:, 8:14].float(),
            beta=1.0,
            gamma=0.001,
            epoch=start_epoch + ep,
        )
        loss = loss_out[0] if isinstance(loss_out, tuple) else loss_out
        reg.model.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(reg.model.network.parameters(), 1.0)
        reg.model.optimizer.step()

        loss_val = float(loss.detach())
        if loss_val < best_loss:
            best_loss = loss_val
            best_state = copy.deepcopy(reg.model.network.state_dict())
        if ep % 50 == 0 or ep == epochs:
            print(f"[MERGE] region_1 ep {ep:4d}/{epochs} loss={loss_val:.6f} best={best_loss:.6f} t={time.time()-t0:.0f}s")

    reg.model.network.load_state_dict(best_state)
    reg._sync_model_refs()
    reg.epoch = start_epoch + epochs
    reg.save_model()

    # Remove obsolete region_5 bundle after successful save of merged region_1.
    shutil.rmtree(out_dir / "region_5", ignore_errors=True)


def _rewrite_topology(out_dir: Path) -> None:
    topo = _load_json(out_dir / "topology.json")
    room_graph = _load_json(out_dir / "room_graph.json")
    room_region_map = _load_json(out_dir / "room_region_map.json")

    rooms_topo = {int(r["id"]): r for r in topo["rooms"]}
    room1 = copy.deepcopy(rooms_topo[1])
    room5 = copy.deepcopy(rooms_topo[5])
    merged_bbox = _bbox_union(room1["bounding_box"], room5["bounding_box"])
    merged_centroid = _weighted_centroid(room1, room5)
    merged_doors = sorted({int(d) for d in (room1.get("doors", []) + room5.get("doors", [])) if int(d) != 14})
    room1["bounding_box"] = merged_bbox
    room1["centroid_xy"] = merged_centroid
    room1["doors"] = merged_doors
    topo["rooms"] = [room1] + [r for r in topo["rooms"] if int(r["id"]) not in (1, 5)]
    topo["rooms"].sort(key=lambda r: int(r["id"]))

    topo["doors"] = [d for d in topo["doors"] if int(d["id"]) != 14]
    for d in topo["doors"]:
        if int(d["id"]) == 16:
            d["room_ids"] = [1, 6]

    new_edges = []
    seen = set()
    for e in topo["graph_edges"]:
        if int(e["doorway_id"]) == 14:
            continue
        rooms = [1 if int(r) == 5 else int(r) for r in e["rooms"]]
        if len(set(rooms)) < 2:
            continue
        rooms = sorted(rooms)
        key = tuple(rooms)
        if key in seen:
            continue
        seen.add(key)
        e = copy.deepcopy(e)
        e["rooms"] = rooms
        new_edges.append(e)
    topo["graph_edges"] = new_edges
    topo["regions"].pop("5", None)
    topo["regions"]["1"]["center"] = merged_centroid
    topo["regions"]["1"]["bound"] = [
        [merged_bbox[0][0], merged_bbox[0][1], -0.1],
        [merged_bbox[1][0], merged_bbox[1][1], 0.1],
    ]
    _write_json(out_dir / "topology.json", topo)

    rooms_rg = {int(k): v for k, v in room_graph["rooms"].items()}
    rg1 = copy.deepcopy(rooms_rg[1])
    rg5 = copy.deepcopy(rooms_rg[5])
    rg1["bounding_box"] = merged_bbox
    rg1["centroid"] = merged_centroid
    rg1["segment_count"] = int(rg1.get("segment_count", 0)) + int(rg5.get("segment_count", 0))
    new_rooms_rg = {"1": rg1}
    for k, v in room_graph["rooms"].items():
        if int(k) not in (1, 5):
            new_rooms_rg[str(k)] = v
    room_graph["rooms"] = new_rooms_rg

    room_graph["doorways"].pop("14", None)
    if "16" in room_graph["doorways"]:
        room_graph["doorways"]["16"]["room_ids"] = [1, 6]

    new_rg_edges = []
    seen = set()
    for e in room_graph["edges"]:
        if int(e["doorway_id"]) == 14:
            continue
        rooms = [1 if int(r) == 5 else int(r) for r in e["rooms"]]
        if len(set(rooms)) < 2:
            continue
        rooms = sorted(rooms)
        key = tuple(rooms)
        if key in seen:
            continue
        seen.add(key)
        e = copy.deepcopy(e)
        e["rooms"] = rooms
        new_rg_edges.append(e)
    room_graph["edges"] = new_rg_edges
    _write_json(out_dir / "room_graph.json", room_graph)

    room_region_map.pop("5", None)
    room_region_map["1"] = 1
    _write_json(out_dir / "room_region_map.json", room_region_map)


def main():
    parser = argparse.ArgumentParser(description="Create corrected refined bundle with rooms 1 and 5 merged.")
    parser.add_argument("--input-dir", default=str(DEFAULT_IN))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    in_dir = Path(args.input_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    if out_dir.exists():
        raise RuntimeError(f"output dir already exists: {out_dir}")

    shutil.copytree(in_dir, out_dir)
    print(f"[COPY] {in_dir} -> {out_dir}")
    _retrain_merged_region(out_dir, args.epochs, args.lr, args.device)
    _rewrite_topology(out_dir)
    print(f"[DONE] corrected bundle written to {out_dir}")


if __name__ == "__main__":
    main()
