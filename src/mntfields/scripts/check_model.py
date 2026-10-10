#!/usr/bin/env python3
"""
check_model.py — inspect what is stored in mntfields_model/ after mapping or refinement.

Shows:
  • Raw mapping regions  (region_N/) — LiDAR frame count, semantic objects
  • Topology rooms       (topology.json) — room bounding boxes, centroids
  • Room-graph           (room_graph.json) — rooms and doorways
  • SLAM map             (map.pgm / map.yaml) — resolution and size

Usage:
    python3 src/mntfields/scripts/check_model.py
    python3 src/mntfields/scripts/check_model.py --model_dir ~/turtlebot3_ws/mntfields_model
"""

import argparse
import json
import os
import sys
from pathlib import Path

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# Mirror of region.is_semantic_entry_confirmed — inlined so this script
# stays runnable without importing the ROS package. Keep thresholds in sync.
_CONFIRM_MIN_COUNT = 3
_CONFIRM_MIN_PLURALITY = 0.5


def _is_confirmed(entry) -> bool:
    if not isinstance(entry, dict):
        return False
    count = int(entry.get("count", 0))
    if count < _CONFIRM_MIN_COUNT:
        return False
    votes = entry.get("class_votes") or {}
    total = sum(int(v) for v in votes.values())
    if total <= 0:
        return False
    return int(max(votes.values())) >= _CONFIRM_MIN_PLURALITY * total


# ── helpers ──────────────────────────────────────────────────────────────────

def _sep(title=""):
    w = 60
    if title:
        pad = (w - len(title) - 2) // 2
        print("─" * pad + f" {title} " + "─" * (w - pad - len(title) - 2))
    else:
        print("─" * w)


def _check_raw_regions(model_dir: Path):
    _sep("RAW MAPPING REGIONS  (region_N/)")
    dirs = sorted(
        d for d in model_dir.iterdir()
        if d.is_dir() and d.name.startswith("region_")
    )
    if not dirs:
        print("  (none found)")
        return

    for rdir in dirs:
        ver = rdir.name.split("_")[1]
        data_path = rdir / f"region_{ver}_data.pth"
        net_path   = rdir / f"region_{ver}_network.pt"

        has_net  = net_path.exists()
        has_data = data_path.exists()

        sem = {}
        n_frames = 0
        n_lidar  = 0
        epoch    = "?"
        bound    = None

        if has_data and HAS_TORCH:
            try:
                d = torch.load(str(data_path), map_location="cpu", weights_only=False)
                if isinstance(d, dict):
                    frames = d.get("frame_data", [])
                    n_frames = len(frames) if isinstance(frames, list) else 0
                    # frame_data entries may be tensors (Nx12) — count total points
                    n_lidar = sum(
                        f.shape[0] if hasattr(f, "shape") else len(f)
                        for f in (frames if isinstance(frames, list) else [])
                    )
                    sem = d.get("semantic_objects", {})
                    epoch = d.get("epoch", "?")
            except Exception as e:
                print(f"  region_{ver}: could not load data — {e}")

        meta_path = rdir / f"region_{ver}_meta.json"
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                bound = meta.get("bound")
            except Exception:
                pass

        trained_str = "trained" if has_net else "NOT trained"
        print(f"\n  region_{ver}  [{trained_str}]  epoch={epoch}")
        if bound:
            b = bound
            print(f"    bound: ({b[0][0]:.2f},{b[0][1]:.2f}) → ({b[1][0]:.2f},{b[1][1]:.2f})")
        print(f"    LiDAR frames={n_frames}  raw_points={n_lidar}")

        if sem:
            total = sum(len(v) for v in sem.values())
            total_conf = sum(1 for pts in sem.values() for p in pts if _is_confirmed(p))
            print(f"    semantic_objects ({total} total, {total_conf} confirmed):")
            for cls, pts in sem.items():
                n_conf = sum(1 for p in pts if _is_confirmed(p))
                descs = [d for d in (_semantic_desc(p) for p in pts) if d is not None]
                coords = "  ".join(descs) if descs else "(no valid xy)"
                print(f"      {cls} ×{len(pts)} ({n_conf} conf): {coords}")
        else:
            print("    semantic_objects: (none)")


def _check_topology(model_dir: Path):
    _sep("TOPOLOGY  (topology.json)")
    p = model_dir / "topology.json"
    if not p.exists():
        print("  topology.json NOT FOUND")
        return

    with open(p) as f:
        topo = json.load(f)

    rooms = topo.get("rooms", [])
    doors = topo.get("doors", [])
    regions = topo.get("regions", {})
    edges = topo.get("graph_edges", [])

    print(f"  regions: {len(regions)}   rooms: {len(rooms)}   "
          f"doors: {len(doors)}   edges: {len(edges)}")

    if rooms:
        print("\n  Rooms:")
        for room in rooms:
            rid = room.get("id", "?")
            c = room.get("centroid_xy", [None, None])
            bb = room.get("bounding_box", [[None, None], [None, None]])
            door_ids = room.get("doors", [])
            print(f"    room {rid}: centroid=({c[0]:.2f},{c[1]:.2f})  "
                  f"bbox=({bb[0][0]:.2f},{bb[0][1]:.2f})→({bb[1][0]:.2f},{bb[1][1]:.2f})  "
                  f"doors={door_ids}")

    if doors:
        print("\n  Doorways:")
        for door in doors:
            did  = door.get("id", "?")
            pos  = door.get("pos_xy", [None, None])
            rids = door.get("room_ids", [])
            w    = door.get("width", "?")
            print(f"    door {did}: pos=({pos[0]:.2f},{pos[1]:.2f})  "
                  f"rooms={rids}  width={w:.2f}m")


def _check_room_graph(model_dir: Path):
    _sep("ROOM GRAPH  (room_graph.json)")
    p = model_dir / "room_graph.json"
    if not p.exists():
        print("  room_graph.json NOT FOUND")
        return

    with open(p) as f:
        rg = json.load(f)

    rooms_raw = rg.get("rooms", {})
    doors_raw = rg.get("doorways", {})
    edges = rg.get("edges", [])

    # rooms/doorways may be a dict (keyed by str id) or a list
    rooms = list(rooms_raw.values()) if isinstance(rooms_raw, dict) else rooms_raw
    doors = list(doors_raw.values()) if isinstance(doors_raw, dict) else doors_raw

    print(f"  rooms={len(rooms)}  doorways={len(doors)}  edges={len(edges)}")
    for room in rooms:
        rid = room.get("id", "?")
        c = room.get("centroid", room.get("centroid_xy", [None, None]))
        print(f"    room {rid}: centroid=({c[0]:.2f},{c[1]:.2f})")
    for door in doors:
        did = door.get("id", "?")
        pos = door.get("position", door.get("pos_xy", [None, None]))
        rids = door.get("room_ids", [])
        print(f"    doorway {did}: pos=({pos[0]:.2f},{pos[1]:.2f})  rooms={rids}")


def _check_slam_map(model_dir: Path):
    _sep("SLAM MAP  (map.pgm / map.yaml)")
    pgm  = model_dir / "map.pgm"
    yaml = model_dir / "map.yaml"

    if yaml.exists():
        with open(yaml) as f:
            print("  map.yaml:")
            for line in f:
                print("    " + line.rstrip())
    else:
        print("  map.yaml NOT FOUND")

    if pgm.exists():
        size = pgm.stat().st_size
        print(f"  map.pgm: exists ({size} bytes)")
    else:
        print("  map.pgm NOT FOUND")


def _check_refined_regions(model_dir: Path):
    """Check room-level regions written by offline_refine.py (stored at top level as region_N/)."""
    _sep("REFINED ROOM-REGIONS  (after offline_refine.py)")
    dirs = sorted(
        d for d in model_dir.iterdir()
        if d.is_dir() and d.name.startswith("region_")
    )
    if not dirs:
        print("  (no region_N/ folders — run offline_refine.py first)")
        return

    # Cross-reference with topology to show room assignment
    topo_rooms: dict[int, dict] = {}
    topo_path = model_dir / "topology.json"
    if topo_path.exists():
        try:
            with open(topo_path) as f:
                topo = json.load(f)
            for room in topo.get("rooms", []):
                topo_rooms[int(room["id"])] = room
        except Exception:
            pass

    print(f"\n  {'Region':<10} {'Room ID':<10} {'Trained':<10} "
          f"{'Epoch':<8} {'Semantic objects'}")
    print("  " + "-" * 70)

    for rdir in dirs:
        ver = rdir.name.split("_")[1]
        try:
            ver_int = int(ver)
        except ValueError:
            ver_int = -1

        net_path  = rdir / f"region_{ver}_network.pt"
        data_path = rdir / f"region_{ver}_data.pth"

        trained = net_path.exists()
        sem = {}
        epoch = "?"

        if data_path.exists() and HAS_TORCH:
            try:
                d = torch.load(str(data_path), map_location="cpu", weights_only=False)
                if isinstance(d, dict):
                    sem = d.get("semantic_objects", {})
                    epoch = d.get("epoch", "?")
            except Exception:
                pass

        room_info = topo_rooms.get(ver_int, {})
        c = room_info.get("centroid_xy")
        room_str = f"{ver_int}" if c is None else f"{ver_int} @ ({c[0]:.1f},{c[1]:.1f})"

        sem_str = ", ".join(f"{k}×{len(v)}" for k, v in sem.items()) if sem else "(none)"
        trained_str = "yes" if trained else "NO"
        print(f"  region_{ver:<6} {room_str:<18} {trained_str:<10} {str(epoch):<8} {sem_str}")

        if sem:
            for cls, pts in sem.items():
                for pt in pts:
                    xy = _semantic_xy(pt)
                    if xy is None:
                        continue
                    extra = ""
                    if isinstance(pt, dict):
                        extra = f" count={int(pt.get('count', 1))}"
                    print(f"    └─ {cls}: map_xy=({xy[0]:.3f}, {xy[1]:.3f}){extra}")


# ── map visualisation ────────────────────────────────────────────────────────

def _load_pgm(pgm_path: Path):
    """Read a binary PGM (P5) file, return (H,W) uint8 array."""
    with open(pgm_path, "rb") as f:
        # Skip comment lines
        line = f.readline()
        assert line.strip() == b"P5", f"Expected P5 PGM, got {line}"
        while True:
            line = f.readline()
            if not line.startswith(b"#"):
                break
        W, H = map(int, line.split())
        maxval = int(f.readline())
        data = f.read()
    import numpy as np
    arr = np.frombuffer(data, dtype=np.uint8).reshape(H, W)
    return arr


def _load_map_yaml(yaml_path: Path) -> dict:
    result = {}
    with open(yaml_path) as f:
        for line in f:
            line = line.strip()
            if ":" not in line:
                continue
            k, _, v = line.partition(":")
            result[k.strip()] = v.strip()
    res = float(result.get("resolution", 0.05))
    origin_str = result.get("origin", "[0,0,0]").strip("[]")
    ox, oy = [float(x) for x in origin_str.split(",")[:2]]
    return {"resolution": res, "origin_x": ox, "origin_y": oy}


def _world_to_px(xy, origin_x, origin_y, resolution, H):
    """Convert world (x,y) metres to image pixel (col, row)."""
    import numpy as np
    xy = np.asarray(xy, dtype=float)
    col = (xy[0] - origin_x) / resolution
    row = H - (xy[1] - origin_y) / resolution
    return col, row


# Distinct colours per class (matplotlib named colours)
_CLASS_COLOURS = [
    "red", "lime", "cyan", "magenta", "yellow", "orange",
    "deepskyblue", "hotpink", "chartreuse", "gold", "violet", "tomato",
]


def _semantic_xy(entry):
    """Return (x, y) from either legacy [x, y] or fused landmark dict."""
    try:
        if isinstance(entry, dict):
            raw = entry.get("xy", entry.get("world_xy", entry.get("pos")))
        else:
            raw = entry
        if raw is None or len(raw) < 2:
            return None
        return float(raw[0]), float(raw[1])
    except Exception:
        return None


def _semantic_desc(entry):
    xy = _semantic_xy(entry)
    if xy is None:
        return None
    if isinstance(entry, dict):
        count = int(entry.get("count", 1))
        votes = entry.get("class_votes", {})
        suffix = f" n={count}" if count > 1 else ""
        if votes:
            suffix += " votes=" + ",".join(
                f"{k}:{v}" for k, v in sorted(votes.items()))
        return f"({xy[0]:.2f},{xy[1]:.2f}){suffix}"
    return f"({xy[0]:.2f},{xy[1]:.2f})"


def _plot_map(model_dir: Path, save_path: Path | None = None):
    if not HAS_MPL:
        print("  matplotlib not available — skipping plot")
        return

    import numpy as np

    pgm_path  = model_dir / "map.pgm"
    yaml_path = model_dir / "map.yaml"

    if not pgm_path.exists() or not yaml_path.exists():
        print("  map.pgm / map.yaml missing — cannot plot")
        return

    arr  = _load_pgm(pgm_path)
    meta = _load_map_yaml(yaml_path)
    res  = meta["resolution"]
    ox   = meta["origin_x"]
    oy   = meta["origin_y"]
    H, W = arr.shape

    # Collect all semantic objects from every region. Each entry is a
    # (xy, confirmed) tuple so the plot can colour tentative landmarks
    # differently from confirmed ones.
    all_objects: dict[str, list] = {}   # class_name → [(xy, confirmed), ...]
    if HAS_TORCH:
        for rdir in sorted(model_dir.glob("region_*")):
            ver = rdir.name.split("_")[1]
            data_path = rdir / f"region_{ver}_data.pth"
            if not data_path.exists():
                continue
            try:
                d = torch.load(str(data_path), map_location="cpu", weights_only=False)
                sem = d.get("semantic_objects", {}) if isinstance(d, dict) else {}
                for cls, pts in sem.items():
                    for p in pts:
                        xy = _semantic_xy(p)
                        if xy is None:
                            continue
                        all_objects.setdefault(cls, []).append(
                            (xy, _is_confirmed(p)))
            except Exception:
                pass

    # Collect room bounding boxes from topology
    rooms = []
    topo_path = model_dir / "topology.json"
    if topo_path.exists():
        try:
            with open(topo_path) as f:
                topo = json.load(f)
            rooms = topo.get("rooms", [])
        except Exception:
            pass

    # Collect doorways
    doorways = []
    rg_path = model_dir / "room_graph.json"
    if rg_path.exists():
        try:
            with open(rg_path) as f:
                rg = json.load(f)
            doors_raw = rg.get("doorways", {})
            doorways = list(doors_raw.values()) if isinstance(doors_raw, dict) else doors_raw
        except Exception:
            pass

    fig, ax = plt.subplots(figsize=(10, 10))

    # PGM row 0 = top of map (saved with flipud in _save_slam_map).
    # origin="upper" maps row 0 → max y, so no additional flip needed.
    ax.imshow(arr, cmap="gray", origin="upper",
              extent=[ox, ox + W * res, oy, oy + H * res])

    # Room bounding boxes
    room_colours = ["#4488ff", "#ff8844", "#44cc44", "#cc44cc", "#cccc44"]
    for i, room in enumerate(rooms):
        rid  = room.get("id", i)
        bb   = room.get("bounding_box", room.get("bound"))
        if bb is None:
            continue
        xmin, ymin = bb[0][0], bb[0][1]
        xmax, ymax = bb[1][0], bb[1][1]
        c = room_colours[i % len(room_colours)]
        rect = mpatches.Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin,
            linewidth=1.5, edgecolor=c, facecolor="none", linestyle="--",
            label=f"room {rid}",
        )
        ax.add_patch(rect)
        cx = room.get("centroid_xy", [(xmin + xmax) / 2, (ymin + ymax) / 2])
        ax.text(cx[0], cx[1], f"room {rid}", color=c, fontsize=9,
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.6))

    # Doorways
    for door in doorways:
        pos = door.get("position", door.get("pos_xy"))
        if pos is None:
            continue
        ax.plot(pos[0], pos[1], marker="D", color="white", markersize=8,
                markeredgecolor="black", markeredgewidth=1.0, zorder=5)
        ax.text(pos[0] + 0.05, pos[1] + 0.05,
                f"door {door.get('id', '?')}", color="white", fontsize=7,
                bbox=dict(boxstyle="round,pad=0.1", fc="black", alpha=0.5))

    # Semantic objects — confirmed are solid, tentative are hollow/translucent.
    class_list = sorted(all_objects.keys())
    legend_handles = []
    for ci, cls in enumerate(class_list):
        colour = _CLASS_COLOURS[ci % len(_CLASS_COLOURS)]
        pts = all_objects[cls]
        confirmed = [xy for xy, ok in pts if ok]
        tentative = [xy for xy, ok in pts if not ok]

        if confirmed:
            cx = [p[0] for p in confirmed]
            cy = [p[1] for p in confirmed]
            ax.scatter(cx, cy, s=120, color=colour, zorder=6,
                       edgecolors="black", linewidths=0.8)
        if tentative:
            tx = [p[0] for p in tentative]
            ty = [p[1] for p in tentative]
            ax.scatter(tx, ty, s=90, facecolors="none", edgecolors=colour,
                       linewidths=1.5, linestyle=":", alpha=0.8, zorder=6)

        for xy, ok in pts:
            label = cls if ok else f"{cls}?"
            ax.annotate(label, (xy[0], xy[1]), textcoords="offset points",
                        xytext=(6, 6), fontsize=8, color=colour,
                        bbox=dict(boxstyle="round,pad=0.2", fc="black",
                                  alpha=0.55 if ok else 0.3))

        legend_handles.append(
            mpatches.Patch(color=colour,
                           label=f"{cls} ×{len(confirmed)} (+{len(tentative)} tent.)"))

    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper right", fontsize=8,
                  framealpha=0.8)

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("mntfields model — semantic objects on SLAM map")
    ax.set_aspect("equal")
    plt.tight_layout()

    if save_path:
        plt.savefig(str(save_path), dpi=150)
        print(f"  Plot saved → {save_path}")
    else:
        plt.show()


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Inspect mntfields_model/ contents")
    parser.add_argument(
        "--model_dir",
        default=os.path.expanduser("~/turtlebot3_ws/mntfields_model"),
        help="Path to mntfields_model directory",
    )
    parser.add_argument(
        "--raw", action="store_true",
        help="Show raw mapping regions only (skip refined/topology sections)",
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Show map visualisation with semantic objects",
    )
    parser.add_argument(
        "--save_plot",
        default=None,
        help="Save plot to this path instead of showing interactively (e.g. /tmp/map.png)",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir).expanduser().resolve()
    if not model_dir.exists():
        print(f"ERROR: model directory not found: {model_dir}")
        sys.exit(1)

    print(f"\nmntfields_model: {model_dir}")
    print(f"torch available: {HAS_TORCH}\n")

    if args.raw:
        _check_raw_regions(model_dir)
    else:
        _check_raw_regions(model_dir)
        print()
        _check_topology(model_dir)
        print()
        _check_room_graph(model_dir)
        print()
        _check_slam_map(model_dir)

    _sep()
    print()

    if args.plot or args.save_plot:
        save = Path(args.save_plot) if args.save_plot else None
        _plot_map(model_dir, save_path=save)


if __name__ == "__main__":
    main()
