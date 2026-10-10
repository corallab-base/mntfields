#!/usr/bin/env python3
"""
offline_refine.py — Post-mapping NTField refinement: one field per room.

Loads the saved mapping output (mntfields_model/), pools all accumulated
LiDAR data, partitions it per-room using each room's bounding box, trains
a fresh NTField for every room, and writes a new folder that is compatible
with region_graph_inspector and the navigator.

Room → Region mapping after refinement: each room gets its own Region with
version == room.id, so room_region_map becomes {room_id: room_id}.

Usage:
    python3 offline_refine.py \\
        --model-dir  ~/turtlebot3_ws/mntfields_model \\
        --output-dir ~/turtlebot3_ws/mntfields_model_refined \\
        [--epochs 3000] \\
        [--lr 2e-4] \\
        [--device cuda:0] \\
        [--checkpoint-every 500] \\
        [--bbox-margin 0.3]

After it finishes, rename the output dir to mntfields_model (or point the
inspector / nav launch at it):

    mv ~/turtlebot3_ws/mntfields_model ~/turtlebot3_ws/mntfields_model_online
    mv ~/turtlebot3_ws/mntfields_model_refined ~/turtlebot3_ws/mntfields_model
    ros2 launch mntfields mntfields_region_graph_inspector.launch.py
"""

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

# ── Make sure the mntfields package is importable ─────────────────────────
_WS = Path(__file__).resolve().parents[3]  # turtlebot3_ws/
_INSTALL_LIB = _WS / "install" / "mntfields" / "lib" / "python3.10" / "site-packages"
if _INSTALL_LIB.exists():
    sys.path.insert(0, str(_INSTALL_LIB))

from mntfields.mntfields_core.core.region import Region


# ═══════════════════════════════════════════════════════════════════════════
# Data loading helpers
# ═══════════════════════════════════════════════════════════════════════════

def load_topology(model_dir: str) -> dict:
    p = os.path.join(model_dir, "topology.json")
    if not os.path.exists(p):
        raise FileNotFoundError(f"topology.json not found in {model_dir}")
    with open(p) as f:
        return json.load(f)


def pool_all_raw_data(model_dir: str) -> torch.Tensor | None:
    """Concatenate all_raw_frame_data from every region into one Nx12 tensor."""
    all_parts = []
    region_dirs = sorted(
        d for d in Path(model_dir).iterdir()
        if d.is_dir() and d.name.startswith("region_")
    )
    if not region_dirs:
        return None

    for rdir in region_dirs:
        ver = rdir.name.split("_")[1]
        data_path = rdir / f"region_{ver}_data.pth"
        if not data_path.exists():
            continue
        try:
            loaded = torch.load(str(data_path), map_location="cpu", weights_only=False)
            parts = loaded.get("frame_data", []) if isinstance(loaded, dict) else (
                loaded if isinstance(loaded, list) else []
            )
            for p in parts:
                if isinstance(p, torch.Tensor) and p.ndim == 2 and p.shape[1] == 12:
                    all_parts.append(p.float())
            print(f"  [POOL] region_{ver}: loaded {sum(len(p) for p in parts if isinstance(p, torch.Tensor))} rows")
        except Exception as e:
            print(f"  [POOL] region_{ver}: could not load — {e}")

    if not all_parts:
        return None

    pooled = torch.cat(all_parts, dim=0)
    print(f"  [POOL] total pooled: {len(pooled)} rows from {len(region_dirs)} region(s)")
    return pooled


def pool_all_semantic_objects(model_dir: str) -> dict[str, list]:
    """Merge semantic_objects dicts from every region into one flat dict.

    Returns: {class_name: [[x, y], [x, y], ...]} with all observations pooled.
    """
    merged: dict[str, list] = {}
    region_dirs = sorted(
        d for d in Path(model_dir).iterdir()
        if d.is_dir() and d.name.startswith("region_")
    )
    for rdir in region_dirs:
        ver = rdir.name.split("_")[1]
        data_path = rdir / f"region_{ver}_data.pth"
        if not data_path.exists():
            continue
        try:
            loaded = torch.load(str(data_path), map_location="cpu", weights_only=False)
            sem = loaded.get("semantic_objects", {}) if isinstance(loaded, dict) else {}
            for cls, pts in sem.items():
                merged.setdefault(cls, []).extend(pts)
        except Exception as e:
            print(f"  [SEM POOL] region_{ver}: could not load — {e}")

    total = sum(len(v) for v in merged.values())
    print(f"  [SEM POOL] pooled {total} observation(s) across {len(merged)} class(es)")
    return merged


def _semantic_xy(entry):
    """Return np-compatible [x, y] from either legacy list or fused dict."""
    try:
        if isinstance(entry, dict):
            raw = entry.get("xy", entry.get("world_xy", entry.get("pos")))
        else:
            raw = entry
        if raw is None or len(raw) < 2:
            return None
        return [float(raw[0]), float(raw[1])]
    except Exception:
        return None


def partition_semantics_by_rooms(
    all_semantics: dict[str, list],
    rooms: list[dict],
    margin: float = 0.3,
) -> dict[int, dict[str, list]]:
    """Assign each semantic observation to its containing room (by XY bbox).

    Returns: {room_id: {class_name: [[x,y], ...]}}
    """
    result: dict[int, dict[str, list]] = {int(r["id"]): {} for r in rooms}

    for cls, pts in all_semantics.items():
        for pt in pts:
            xy = _semantic_xy(pt)
            if xy is None:
                continue
            best_room = None
            best_dist = float("inf")
            for room in rooms:
                rid = int(room["id"])
                bb = room["bounding_box"]
                xmin, ymin = bb[0][0] - margin, bb[0][1] - margin
                xmax, ymax = bb[1][0] + margin, bb[1][1] + margin
                if xmin <= xy[0] <= xmax and ymin <= xy[1] <= ymax:
                    cx = 0.5 * (bb[0][0] + bb[1][0])
                    cy = 0.5 * (bb[0][1] + bb[1][1])
                    dist = (xy[0] - cx) ** 2 + (xy[1] - cy) ** 2
                    if dist < best_dist:
                        best_dist = dist
                        best_room = rid
            if best_room is not None:
                result[best_room].setdefault(cls, []).append(pt)

    for room in rooms:
        rid = int(room["id"])
        n = sum(len(v) for v in result[rid].values())
        if n:
            print(f"  [SEM PARTITION] room {rid}: {n} observation(s) → "
                  f"{list(result[rid].keys())}")
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Room bounding-box partitioning
# ═══════════════════════════════════════════════════════════════════════════

def _room_bound_tensor(room: dict, margin: float) -> torch.Tensor:
    """Build a [2,3] bound tensor from a topology room entry."""
    bbox = room["bounding_box"]   # [[xmin,ymin],[xmax,ymax]]
    xmin, ymin = bbox[0][0] - margin, bbox[0][1] - margin
    xmax, ymax = bbox[1][0] + margin, bbox[1][1] + margin
    return torch.tensor([[xmin, ymin, -0.1], [xmax, ymax, 0.1]], dtype=torch.float32)


def partition_data_by_rooms(
    pooled: torch.Tensor,
    rooms: list[dict],
    margin: float = 0.3,
) -> dict[int, torch.Tensor]:
    """Assign each Nx12 row to the room whose (margined) bounding box contains x0.

    If a row falls in multiple rooms (overlap at doorways), it is assigned to
    the room whose centroid is closest to x0.  Rows that fall in no room are
    dropped.

    Returns:
        {room_id: Nx12 tensor}
    """
    x0_xy = pooled[:, :2]  # (N, 2) — x-y of the free-space sample point

    room_ids  = [int(r["id"]) for r in rooms]
    centroids = torch.tensor(
        [r["centroid_xy"] for r in rooms], dtype=torch.float32)  # (R, 2)
    bounds    = [_room_bound_tensor(r, margin) for r in rooms]

    # For each row, build a boolean mask per room (inside margined bbox)
    N = len(pooled)
    R = len(rooms)
    inside = torch.zeros(N, R, dtype=torch.bool)
    for ri, b in enumerate(bounds):
        inside[:, ri] = (
            (x0_xy[:, 0] >= b[0, 0]) & (x0_xy[:, 0] <= b[1, 0]) &
            (x0_xy[:, 1] >= b[0, 1]) & (x0_xy[:, 1] <= b[1, 1])
        )

    # Rows inside exactly one room → direct assignment
    # Rows inside multiple rooms → closest centroid wins
    # Rows in no room → dropped
    n_rooms_per_row = inside.sum(dim=1)  # (N,)

    assignment = torch.full((N,), -1, dtype=torch.long)

    # Single-room rows
    single_mask = n_rooms_per_row == 1
    if single_mask.any():
        assignment[single_mask] = inside[single_mask].long().argmax(dim=1)

    # Multi-room rows (doorway overlap) — assign to closest centroid
    multi_mask = n_rooms_per_row > 1
    if multi_mask.any():
        x0_multi = x0_xy[multi_mask]  # (M, 2)
        dists = torch.cdist(x0_multi.unsqueeze(0), centroids.unsqueeze(0))[0]  # (M, R)
        # Only consider rooms this row is inside
        inside_multi = inside[multi_mask].float()  # (M, R)
        dists_masked = dists + (1 - inside_multi) * 1e9
        assignment[multi_mask] = dists_masked.argmin(dim=1)

    result = {}
    for ri, room_id in enumerate(room_ids):
        mask = assignment == ri
        if mask.any():
            result[room_id] = pooled[mask]

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Per-room training
# ═══════════════════════════════════════════════════════════════════════════

def _cosine_lr(epoch: int, lr_min: float, lr_max: float, cycle: int) -> float:
    t = epoch % cycle
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * t / cycle))


def train_room_region(
    room_id: int,
    frame_data: torch.Tensor,
    bound: torch.Tensor,
    output_dir: str,
    epochs: int,
    lr: float,
    device: str,
    checkpoint_every: int,
    batch_size: int = 8000,
    max_batches: int = 8,
) -> Region:
    """Train a fresh NTField for one room.  Region version == room_id."""
    print(f"\n{'='*64}")
    print(f"  Room {room_id} → Region {room_id}  |  rows={len(frame_data)}  |  epochs={epochs}")
    print(f"  bound: [{bound[0,:2].tolist()}] → [{bound[1,:2].tolist()}]")
    print(f"{'='*64}")

    reg = Region(
        model_folder=output_dir,
        bound=bound.clone(),
        version=room_id,
        device=device,
    )

    raw_gpu = frame_data.to(device)
    n14 = reg.calculate_speed_and_normal(raw_gpu)

    if n14 is None or len(n14) == 0:
        print(f"  [Room {room_id}] WARNING: no valid Nx14 rows — saving untrained region")
        reg.all_raw_frame_data = [frame_data.cpu()]
        reg.save_model()
        return reg

    print(f"  [Room {room_id}] {len(n14)} valid Nx14 rows (from {len(frame_data)} raw)")
    reg.all_raw_frame_data = [frame_data.cpu()]

    lr_min   = lr * 0.1
    lr_max   = lr
    lr_cycle = 200

    best_loss  = float("inf")
    best_state = None
    t0 = time.time()

    for ep in range(1, epochs + 1):
        cur_lr = _cosine_lr(ep, lr_min, lr_max, lr_cycle)
        reg.model.optimizer.param_groups[0]["lr"] = cur_lr

        perm = torch.randperm(len(n14), device=device)
        shuffled = n14[perm]
        n = len(shuffled)
        n_batches = min(max_batches, math.ceil(n / batch_size))

        ep_loss = 0.0
        for i in range(n_batches):
            batch  = shuffled[i * batch_size : (i + 1) * batch_size]
            points = batch[:, :6].float()
            speed  = batch[:, 6:8].float()
            normal = batch[:, 8:14].float()

            speed = speed * speed * (2 - speed) * (2 - speed)

            loss_out = reg.model.function.Loss(
                points, speed, normal, beta=1.0, gamma=0.001, epoch=ep)
            loss_value = loss_out[0] if isinstance(loss_out, tuple) else loss_out

            reg.model.optimizer.zero_grad()
            loss_value.backward()
            torch.nn.utils.clip_grad_norm_(reg.model.network.parameters(), 1.0)
            reg.model.optimizer.step()
            ep_loss += float(loss_value.detach())

        ep_loss /= max(n_batches, 1)

        if ep_loss < best_loss:
            best_loss  = ep_loss
            best_state = copy.deepcopy(reg.model.network.state_dict())

        if ep % 100 == 0 or ep == epochs:
            elapsed = time.time() - t0
            print(f"  [Room {room_id}] ep {ep:4d}/{epochs}  loss={ep_loss:.6f}  "
                  f"best={best_loss:.6f}  lr={cur_lr:.2e}  t={elapsed:.0f}s")

        if checkpoint_every > 0 and ep % checkpoint_every == 0:
            reg.save_checkpoint(ep)

    if best_state is not None:
        reg.model.network.load_state_dict(best_state)
        reg._sync_model_refs()

    reg.epoch = epochs
    reg.save_model()
    print(f"  [Room {room_id}] done. best_loss={best_loss:.6f} → {reg.folder}")
    return reg


# ═══════════════════════════════════════════════════════════════════════════
# Topology rewrite
# ═══════════════════════════════════════════════════════════════════════════

def write_refined_topology(
    topology: dict,
    trained_room_ids: list[int],
    output_dir: str,
) -> None:
    """Write topology.json and room_region_map.json with one-to-one room→region mapping."""

    # New room_region_map: room_id → room_id (each room is its own region)
    room_region_map = {str(rid): rid for rid in trained_room_ids}

    # Rebuild topology regions block from room bounding boxes
    new_regions = {}
    for room in topology.get("rooms", []):
        rid = int(room["id"])
        if rid not in trained_room_ids:
            continue
        bbox = room["bounding_box"]
        center = [
            0.5 * (bbox[0][0] + bbox[1][0]),
            0.5 * (bbox[0][1] + bbox[1][1]),
        ]
        new_regions[str(rid)] = {
            "version": rid,
            "center":  center,
            "bound": [
                [bbox[0][0], bbox[0][1], -0.1],
                [bbox[1][0], bbox[1][1],  0.1],
            ],
        }

    new_topology = dict(topology)
    new_topology["regions"] = new_regions
    # graph_edges, rooms, doors, connection_points stay unchanged

    with open(os.path.join(output_dir, "topology.json"), "w") as f:
        json.dump(new_topology, f, indent=2)

    with open(os.path.join(output_dir, "room_region_map.json"), "w") as f:
        json.dump(room_region_map, f, indent=2)

    print(f"\n[GRAPH] topology.json  — {len(new_regions)} region(s) (one per room)")
    print(f"[GRAPH] room_region_map.json — {room_region_map}")
    print(f"        rooms={len(topology.get('rooms',[]))}  "
          f"doors={len(topology.get('doors',[]))}  "
          f"edges={len(topology.get('graph_edges',[]))}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Offline per-room NTField refinement")
    parser.add_argument(
        "--model-dir",
        default=os.path.expanduser("~/turtlebot3_ws/mntfields_model"),
        help="Existing mntfields_model directory")
    parser.add_argument(
        "--output-dir",
        default=os.path.expanduser("~/turtlebot3_ws/mntfields_model_refined"),
        help="Output directory for refined model")
    parser.add_argument("--epochs",           type=int,   default=3000)
    parser.add_argument("--lr",               type=float, default=2e-4)
    parser.add_argument("--device",           default="cuda:0")
    parser.add_argument("--checkpoint-every", type=int,   default=500)
    parser.add_argument("--batch-size",       type=int,   default=8000)
    parser.add_argument("--max-batches",      type=int,   default=8)
    parser.add_argument(
        "--bbox-margin", type=float, default=0.3,
        help="Metres to expand each room bbox when filtering training data (default 0.3)")
    args = parser.parse_args()

    model_dir  = os.path.expanduser(args.model_dir)
    output_dir = os.path.expanduser(args.output_dir)

    if not os.path.isdir(model_dir):
        print(f"[ERROR] model-dir not found: {model_dir}")
        sys.exit(1)

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print(f"[WARN] CUDA not available, falling back to CPU")
        device = "cpu"
    print(f"[DEVICE] {device}\n")

    os.makedirs(output_dir, exist_ok=True)

    # Copy static map files
    for fname in ("map.pgm", "map.yaml", "room_graph.json"):
        src = os.path.join(model_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(output_dir, fname))
            print(f"[COPY] {fname}")

    # Load topology
    topology = load_topology(model_dir)
    rooms = topology.get("rooms", [])
    if not rooms:
        print("[ERROR] No rooms found in topology.json")
        sys.exit(1)
    print(f"\n[ROOMS] {len(rooms)} room(s): {[r['id'] for r in rooms]}")

    # Pool all raw data across regions
    print("\n[POOL] Aggregating raw data from all regions...")
    pooled = pool_all_raw_data(model_dir)
    if pooled is None or len(pooled) == 0:
        print("[ERROR] No raw frame data found in any region")
        sys.exit(1)

    # Pool semantic objects from all regions
    print("\n[SEM POOL] Aggregating semantic objects from all regions...")
    all_semantics = pool_all_semantic_objects(model_dir)
    sem_by_room = partition_semantics_by_rooms(all_semantics, rooms, margin=args.bbox_margin)

    # Partition LiDAR data by room bounding box
    print(f"\n[PARTITION] Splitting {len(pooled)} rows by room bbox "
          f"(margin={args.bbox_margin}m)...")
    room_data = partition_data_by_rooms(pooled, rooms, margin=args.bbox_margin)
    del pooled  # free memory

    for room in rooms:
        rid = int(room["id"])
        n = len(room_data.get(rid, []))
        print(f"  room {rid}: {n} rows")
        if n == 0:
            print(f"           ↑ WARNING: no data — robot may not have entered this room")

    # Reset Region counter — we're assigning version=room_id explicitly
    Region.n_regions = max(int(r["id"]) for r in rooms)

    trained_ids = []
    skipped_ids = []

    for room in sorted(rooms, key=lambda r: int(r["id"])):
        rid = int(room["id"])
        frame_data = room_data.get(rid)

        if frame_data is None or len(frame_data) == 0:
            print(f"\n[SKIP] Room {rid}: no training data")
            skipped_ids.append(rid)
            continue

        # Match the margin used during partitioning so saved rows (whose x0 can
        # sit up to bbox_margin outside the tight bbox) remain inside the region
        # bound. Nav's _filter_region_owned_rows uses this bound directly, so a
        # tight=0 bound silently drops the border-band rows on first accumulate.
        bound = _room_bound_tensor(room, margin=args.bbox_margin)

        reg = train_room_region(
            room_id=rid,
            frame_data=frame_data,
            bound=bound,
            output_dir=output_dir,
            epochs=args.epochs,
            lr=args.lr,
            device=device,
            checkpoint_every=args.checkpoint_every,
            batch_size=args.batch_size,
            max_batches=args.max_batches,
        )

        # Attach semantic objects that belong to this room and re-save
        room_sem = sem_by_room.get(rid, {})
        if room_sem:
            reg.semantic_objects = room_sem
            reg.save_model()
            n_sem = sum(len(v) for v in room_sem.values())
            print(f"  [SEM] room {rid}: attached {n_sem} semantic observation(s) → saved")

        trained_ids.append(rid)

    # Write updated graph files
    write_refined_topology(topology, trained_ids, output_dir)

    # Summary
    print(f"\n{'='*64}")
    print(f"  Offline refinement complete — ONE FIELD PER ROOM")
    print(f"  Trained : rooms {trained_ids}")
    print(f"  Skipped : rooms {skipped_ids}")
    print(f"  Output  : {output_dir}")
    print(f"\n  To use:")
    print(f"    mv {model_dir} {model_dir}_online")
    print(f"    mv {output_dir} {model_dir}")
    print(f"    ros2 launch mntfields mntfields_region_graph_inspector.launch.py")
    print(f"{'='*64}\n")


if __name__ == "__main__":
    main()
