#!/usr/bin/env python3
"""
region_graph_inspector.py

Standalone visualization/planning tool for inspecting region connectivity.

What it does:
  - Loads saved region bundles from model_dir (region_*/region_*_{meta,network,B,data}.*)
  - Loads static map from map_yaml and publishes /map
  - Builds a sparse, grid-validated region graph
  - Publishes RViz markers for region bounds + sparse graph + selected route
  - Uses /initialpose as start, /goal_pose as goal
  - Plans and publishes trajectory on /region_graph_path
  - Optionally auto-launches RViz with a dedicated config
"""

import glob
import heapq
import json
import math
import os
import subprocess
import threading
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
from PIL import Image
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
import torch
import yaml
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Point, PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import MapMetaData, OccupancyGrid, Path
from visualization_msgs.msg import Marker, MarkerArray

from mntfields.mntfields_core.core.region import Region
from mntfields.mntfields_core.mapping.watershed_segmenter import WatershedSegmenter
from mntfields.utils.mntfields_utils import (
    compute_ownership_map, compute_crossing_points, owner_region_for_xy,
    room_id_for_xy, select_best_region_for_room, compute_region_room_ownership,
    straight_line_path)


# Per-region colour palette (distinct hues, good contrast on dark bg).
_REGION_COLOURS = [
    (0.2, 0.9, 0.4),   # green
    (0.3, 0.7, 1.0),   # sky blue
    (1.0, 0.6, 0.1),   # orange
    (0.9, 0.3, 0.9),   # magenta
    (1.0, 1.0, 0.2),   # yellow
    (0.1, 1.0, 0.9),   # cyan
    (1.0, 0.3, 0.3),   # red
    (0.6, 0.4, 1.0),   # purple
]

# Z-height layering so paths don't visually overlap.
_Z_FINAL      = 0.22
_Z_BIDIR      = 0.18
_Z_ONEWAY     = 0.14
_Z_GOAL2START = 0.10
_Z_DEBUG_SEG  = 0.26
_Z_SUMMARY    = 0.50


def _region_colour(version: int) -> tuple:
    return _REGION_COLOURS[version % len(_REGION_COLOURS)]


def _short_reason(reason: str, max_len: int = 60) -> str:
    """Truncate long diagnostic reason strings for RViz readability."""
    if len(reason) <= max_len:
        return reason
    return reason[:max_len - 3] + "..."


class RegionGraphInspector(Node):
    def __init__(self):
        super().__init__("region_graph_inspector")

        self.declare_parameter("model_dir", "/home/mayank/turtlebot3_ws/mntfields_model")
        self.declare_parameter(
            "map_yaml",
            "/home/mayank/turtlebot3_ws/mntfields_model/map.yaml",
        )
        self.declare_parameter("step_size", 0.10)
        self.declare_parameter("max_steps", 400)
        self.declare_parameter("auto_launch_rviz", True)
        self.declare_parameter("rviz_config", "")
        self.declare_parameter("adjacency_gap_m", 0.80)
        self.declare_parameter("max_neighbors_per_region", 3)
        self.declare_parameter("inflation_cells", 1)
        self.declare_parameter("region_bound_margin_m", 0.06)
        self.declare_parameter("doorway_cushion_offsets_m", [0.0, 0.18, 0.30, 0.45, 0.60])
        self.declare_parameter("compare_rollouts", True)
        self.declare_parameter("save_path_overlays", True)
        self.declare_parameter("path_overlay_grid_n", 120)
        self.declare_parameter("path_overlay_dir", "")
        self.declare_parameter("merge_small_rooms_area_cells", 2500)
        self.declare_parameter("merge_small_rooms_min_contact", 8)

        self.model_dir = str(self.get_parameter("model_dir").value)
        self.map_yaml = str(self.get_parameter("map_yaml").value)
        self.step_size = float(self.get_parameter("step_size").value)
        self.max_steps = int(self.get_parameter("max_steps").value)
        self.adjacency_gap_m = float(self.get_parameter("adjacency_gap_m").value)
        self.max_neighbors_per_region = int(
            self.get_parameter("max_neighbors_per_region").value
        )
        self.inflation_cells = int(self.get_parameter("inflation_cells").value)
        self.region_bound_margin_m = float(
            self.get_parameter("region_bound_margin_m").value
        )
        self.doorway_cushion_offsets_m = [
            float(v) for v in self.get_parameter("doorway_cushion_offsets_m").value
        ]
        self.compare_rollouts = bool(self.get_parameter("compare_rollouts").value)
        self.save_path_overlays = bool(self.get_parameter("save_path_overlays").value)
        self.path_overlay_grid_n = int(self.get_parameter("path_overlay_grid_n").value)
        self.path_overlay_dir = str(self.get_parameter("path_overlay_dir").value).strip()
        self.merge_small_rooms_area_cells = int(self.get_parameter("merge_small_rooms_area_cells").value)
        self.merge_small_rooms_min_contact = int(self.get_parameter("merge_small_rooms_min_contact").value)

        self.regions: Dict[int, Region] = {}
        self._bundle_region_meta: Dict[int, dict] = {}

        # Optional raw topology from disk (used as candidate edges only).
        self._raw_topo_edges: List[dict] = []
        self._connection_points: List[dict] = []
        self._saved_rooms: List[dict] = []
        self._saved_doors: List[dict] = []
        self._saved_frontier_banks: Dict[int, List[np.ndarray]] = {}
        self._saved_map_snapshot: Dict[str, Any] = {}

        # Sparse planning graph.
        self._route_graph = nx.Graph()
        self._region_anchor: Dict[int, np.ndarray] = {}
        self._active_region_route: List[int] = []
        self._debug_segments: List[dict] = []

        self.start_xy: Optional[np.ndarray] = None
        self.goal_xy: Optional[np.ndarray] = None
        self.path_xy: Optional[np.ndarray] = None
        self.path_raw_xy: Optional[np.ndarray] = None
        self.path_bidir_xy: Optional[np.ndarray] = None
        self.path_oneway_xy: Optional[np.ndarray] = None
        self.path_goal_to_start_xy: Optional[np.ndarray] = None
        self.path_speed_xy: Optional[np.ndarray] = None
        self._planning_running = False
        self._overlay_seq = 0

        self._map_occ: Optional[np.ndarray] = None
        self._map_blocked: Optional[np.ndarray] = None
        self._map_slam_arr: Optional[np.ndarray] = None
        self._map_resolution: Optional[float] = None
        self._map_origin: Optional[Tuple[float, float]] = None
        self._map_msg: Optional[OccupancyGrid] = None
        self._ownership_map: Optional[np.ndarray] = None  # int32 [H,W], 0=unowned
        self._crossing_points: dict = {}                  # (ver_a,ver_b) -> xy
        self._room_to_region: Dict[int, int] = {}         # room_id -> region_version
        self._strict_roomwise_topology: bool = False
        self._region_room_ownership: Dict[str, Any] = {'by_region': {}, 'by_room': {}}
        self._room_map: Optional[np.ndarray] = None
        self._room_map_origin: Optional[Tuple[float, float]] = None
        self._room_map_resolution: Optional[float] = None

        map_qos = QoSProfile(
            depth=1,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.map_pub = self.create_publisher(OccupancyGrid, "/map", map_qos)
        self.path_pub = self.create_publisher(Path, "/region_graph_path", 10)
        self.path_raw_pub = self.create_publisher(Path, "/region_graph_path_raw", 10)
        self.path_bidir_pub = self.create_publisher(Path, "/region_graph_path_bidirectional", 10)
        self.path_oneway_pub = self.create_publisher(Path, "/region_graph_path_oneway", 10)
        self.path_goal_to_start_pub = self.create_publisher(
            Path, "/region_graph_path_goal_to_start", 10
        )
        self.marker_pub = self.create_publisher(MarkerArray, "/region_graph_markers", 10)
        self.marker_debug_pub = self.create_publisher(
            MarkerArray, "/region_graph_markers_debug", 10
        )

        self.create_subscription(
            PoseWithCovarianceStamped, "/initialpose", self._initialpose_cb, 10
        )
        self.create_subscription(PoseStamped, "/goal_pose", self._goal_cb, 10)

        self._load_bundle_topology()
        self._load_all_regions()
        self._load_map()
        self._run_watershed()
        self._rebuild_ownership()
        self._build_sparse_route_graph()

        self._rviz_proc: Optional[subprocess.Popen] = None
        if bool(self.get_parameter("auto_launch_rviz").value):
            self._launch_rviz()

        self.create_timer(0.2, self._publish_once_active)
        self.create_timer(1.0, self._publish_graph_markers)

        self.get_logger().info(
            f"[INSPECTOR] Ready. regions={len(self.regions)}, "
            f"sparse_graph_edges={self._route_graph.number_of_edges()}. "
            "Set start with '2D Pose Estimate' and goal with '2D Goal Pose' in RViz."
        )

    def destroy_node(self):
        try:
            if self._rviz_proc is not None and self._rviz_proc.poll() is None:
                self._rviz_proc.terminate()
        except Exception:
            pass
        super().destroy_node()

    @staticmethod
    def _coerce_xy(val: Any) -> Optional[np.ndarray]:
        if val is None:
            return None
        arr = np.asarray(val, dtype=np.float32).reshape(-1)
        if arr.size < 2:
            return None
        return arr[:2].astype(np.float32)

    def _coerce_bound_tensor(self, bound_like: Any) -> Optional[torch.Tensor]:
        if bound_like is None:
            return None
        arr = np.asarray(bound_like, dtype=np.float32)
        if arr.shape == (2, 2):
            arr3 = np.zeros((2, 3), dtype=np.float32)
            arr3[:, :2] = arr
            arr3[0, 2] = -0.1
            arr3[1, 2] = 0.1
            arr = arr3
        if arr.shape != (2, 3):
            return None
        return torch.tensor(arr, dtype=torch.float32)

    def _region_bundle_complete(self, version: int) -> bool:
        region_dir = os.path.join(self.model_dir, f"region_{version}")
        req = [
            os.path.join(region_dir, f"region_{version}_network.pt"),
            os.path.join(region_dir, f"region_{version}_B.pt"),
            os.path.join(region_dir, f"region_{version}_data.pth"),
        ]
        return all(os.path.isfile(p) for p in req)

    def _discover_region_versions(self) -> List[int]:
        versions: Set[int] = set(self._bundle_region_meta.keys())
        for region_dir in glob.glob(os.path.join(self.model_dir, "region_*")):
            try:
                versions.add(int(os.path.basename(region_dir).split("_")[1]))
            except (IndexError, ValueError):
                continue
        return sorted(versions)

    def _load_all_regions(self):
        if not os.path.isdir(self.model_dir):
            self.get_logger().error(f"[INSPECTOR] model_dir not found: {self.model_dir}")
            return

        loaded = 0
        for version in self._discover_region_versions():
            meta = self._bundle_region_meta.get(version, {})
            bound = self._coerce_bound_tensor(meta.get("bound"))
            if bound is None:
                meta_path = os.path.join(
                    self.model_dir, f"region_{version}", f"region_{version}_meta.json"
                )
                if os.path.isfile(meta_path):
                    try:
                        with open(meta_path, "r", encoding="utf-8") as fh:
                            bound = self._coerce_bound_tensor(json.load(fh).get("bound"))
                    except Exception:
                        bound = None
            if bound is None:
                self.get_logger().warn(
                    f"[INSPECTOR] Skipping R{version}: no saved bound in topology or region meta."
                )
                continue
            if not self._region_bundle_complete(version):
                self.get_logger().warn(
                    f"[INSPECTOR] Skipping R{version}: incomplete region bundle."
                )
                continue

            try:
                region = Region(
                    model_folder=self.model_dir,
                    bound=bound,
                    version=version,
                    device="cpu",
                )
                if not region.load_model():
                    self.get_logger().warn(f"[INSPECTOR] R{version} failed to load bundle.")
                    continue

                topo_bound = self._coerce_bound_tensor(meta.get("bound"))
                if topo_bound is not None:
                    region.set_bound(topo_bound)

                self.regions[version] = region
                loaded += 1
                b = region.bound.detach().cpu().numpy()
                self.get_logger().info(
                    f"[INSPECTOR] Loaded region {version} "
                    f"[{b[0,0]:.2f},{b[0,1]:.2f}] -> [{b[1,0]:.2f},{b[1,1]:.2f}]"
                )
            except Exception as exc:
                self.get_logger().error(
                    f"[INSPECTOR] Failed loading region {version}: {exc}"
                )

        if loaded == 0:
            self.get_logger().warn("[INSPECTOR] No region bundles loaded.")

    def _parse_saved_rooms(self, topo: dict) -> List[dict]:
        rooms_raw = (
            topo.get("rooms")
            or topo.get("confirmed_rooms")
            or ((topo.get("room_topology") or {}).get("rooms"))
            or []
        )
        rooms: List[dict] = []
        if isinstance(rooms_raw, dict):
            items = rooms_raw.items()
        else:
            items = enumerate(rooms_raw)
        for key, raw in items:
            if not isinstance(raw, dict):
                continue
            room_id = raw.get("id", key)
            centroid = self._coerce_xy(raw.get("centroid_xy") or raw.get("centroid"))
            if centroid is None:
                continue
            doors = raw.get("doors", [])
            parsed_doors: List[int] = []
            for d in doors:
                if isinstance(d, dict):
                    if "id" in d:
                        parsed_doors.append(int(d["id"]))
                else:
                    try:
                        parsed_doors.append(int(d))
                    except Exception:
                        pass
            rooms.append(
                {
                    "id": int(room_id),
                    "centroid_xy": centroid,
                    "doors": parsed_doors,
                }
            )
        return rooms

    def _parse_saved_doors(self, topo: dict) -> List[dict]:
        doors_raw = topo.get("doors") or ((topo.get("room_topology") or {}).get("doors")) or []
        doors: List[dict] = []
        for raw in doors_raw:
            if not isinstance(raw, dict):
                continue
            pos = self._coerce_xy(raw.get("pos_xy") or raw.get("pos") or raw.get("xy"))
            if pos is None:
                continue
            room_ids = []
            for v in raw.get("room_ids", []):
                try:
                    room_ids.append(int(v))
                except Exception:
                    pass
            doors.append(
                {
                    "id": int(raw.get("id", len(doors))),
                    "pos_xy": pos,
                    "room_ids": room_ids,
                }
            )
        return doors

    def _parse_saved_frontiers(self, topo: dict) -> Dict[int, List[np.ndarray]]:
        raw = topo.get("frontier_banks") or topo.get("saved_frontiers") or {}
        out: Dict[int, List[np.ndarray]] = {}
        if isinstance(raw, list):
            for idx, entry in enumerate(raw):
                if not isinstance(entry, dict):
                    continue
                version = entry.get("version", idx)
                pts = entry.get("saved_frontiers") or entry.get("frontiers") or []
                parsed = [xy for xy in (self._coerce_xy(p) for p in pts) if xy is not None]
                out[int(version)] = parsed
            return out
        if isinstance(raw, dict):
            for version, pts in raw.items():
                parsed = [xy for xy in (self._coerce_xy(p) for p in (pts or [])) if xy is not None]
                try:
                    out[int(version)] = parsed
                except Exception:
                    continue
        return out

    def _load_bundle_topology(self):
        topo_path = os.path.join(self.model_dir, "topology.json")
        if not os.path.isfile(topo_path):
            self.get_logger().warn(f"[INSPECTOR] topology.json not found at {topo_path}")
            self._bundle_region_meta = {}
            self._raw_topo_edges = []
            self._connection_points = []
            self._saved_rooms = []
            self._saved_doors = []
            self._saved_frontier_banks = {}
            return

        with open(topo_path, "r", encoding="utf-8") as fh:
            topo = json.load(fh)

        regions = topo.get("regions", {})
        self._bundle_region_meta = {}
        if isinstance(regions, dict):
            for key, val in regions.items():
                try:
                    self._bundle_region_meta[int(key)] = val if isinstance(val, dict) else {}
                except Exception:
                    continue
        elif isinstance(regions, list):
            for item in regions:
                if not isinstance(item, dict):
                    continue
                try:
                    self._bundle_region_meta[int(item["version"])] = item
                except Exception:
                    continue

        topo_graph = topo.get("topology_graph") or {}
        self._raw_topo_edges = topo.get("graph_edges", []) or topo_graph.get("edges", [])
        self._connection_points = topo.get("connection_points", []) or topo_graph.get("connection_points", [])
        self._saved_rooms = self._parse_saved_rooms(topo)
        self._saved_doors = self._parse_saved_doors(topo)
        self._saved_frontier_banks = self._parse_saved_frontiers(topo)
        self._saved_map_snapshot = (
            topo.get("map_snapshot")
            or topo.get("occupancy_snapshot")
            or topo.get("occupancy_metadata")
            or {}
        )
        self._region_room_ownership = topo.get('region_room_ownership') or {'by_region': {}, 'by_room': {}}
        self._room_map = None
        self._room_map_origin = None
        self._room_map_resolution = None
        room_map_meta = topo.get('room_map') or {}
        room_map_path = str(room_map_meta.get('path') or '').strip()
        if room_map_path and not os.path.isabs(room_map_path):
            room_map_path = os.path.join(self.model_dir, room_map_path)
        if room_map_path and os.path.isfile(room_map_path):
            try:
                snap = np.load(room_map_path)
                room_map = np.asarray(snap['room_map'], dtype=np.int32)
                origin = np.asarray(snap.get('origin', room_map_meta.get('origin', [0.0, 0.0])), dtype=np.float32).reshape(-1)
                resolution = float(np.asarray(snap.get('resolution', [room_map_meta.get('resolution', self._map_resolution or 0.05)]), dtype=np.float32).reshape(-1)[0])
                self._room_map = room_map
                self._room_map_origin = (float(origin[0]), float(origin[1]))
                self._room_map_resolution = resolution
            except Exception as exc:
                self.get_logger().warn(f"[INSPECTOR] Failed loading room map snapshot: {exc}")
        # Load room_region_map.json — maps room_id → region_version.
        # In offline-refined models: room_id == region_version (identity mapping).
        # In online models: multiple rooms may map to one region.
        rrm_path = os.path.join(self.model_dir, "room_region_map.json")
        self._room_to_region = {}
        if os.path.isfile(rrm_path):
            try:
                with open(rrm_path) as f:
                    raw = json.load(f)
                self._room_to_region = {int(k): int(v) for k, v in raw.items()}
            except Exception as exc:
                self.get_logger().warn(f"[INSPECTOR] Failed loading room_region_map.json: {exc}")
        # If no file (old models), fall back to identity: assume room_id == region_version
        if not self._room_to_region:
            for rid in [r.get("id") for r in self._saved_rooms if r.get("id") is not None]:
                self._room_to_region[int(rid)] = int(rid)
        self._strict_roomwise_topology = (
            bool(self._room_to_region)
            and all(int(k) == int(v) for k, v in self._room_to_region.items())
        )

        self.get_logger().info(
            f"[INSPECTOR] Loaded bundle topology: "
            f"regions={len(self._bundle_region_meta)} "
            f"edges={len(self._raw_topo_edges)} "
            f"rooms={len(self._saved_rooms)} "
            f"doors={len(self._saved_doors)} "
            f"room_to_region={self._room_to_region} "
            f"strict_roomwise={self._strict_roomwise_topology}"
        )

    def _inflate_occ(self, occ: np.ndarray, pad: int) -> np.ndarray:
        if pad <= 0:
            return occ.copy()
        h, w = occ.shape
        out = np.zeros_like(occ, dtype=bool)
        occ_idx = np.argwhere(occ)
        for r, c in occ_idx:
            r0 = max(0, r - pad)
            r1 = min(h, r + pad + 1)
            c0 = max(0, c - pad)
            c1 = min(w, c + pad + 1)
            out[r0:r1, c0:c1] = True
        return out

    def _room_map_xy(self, row: int, col: int) -> np.ndarray:
        origin = self._room_map_origin or self._map_origin or (0.0, 0.0)
        res = self._room_map_resolution or self._map_resolution or 0.05
        return np.array([
            float(origin[0]) + float(col) * float(res),
            float(origin[1]) + float(row) * float(res),
        ], dtype=np.float32)

    def _saved_rooms_from_room_map(self) -> List[dict]:
        if self._room_map is None:
            return self._saved_rooms
        out: List[dict] = []
        for rid in sorted(int(v) for v in np.unique(self._room_map) if int(v) > 0):
            mask = (self._room_map == rid)
            rows, cols = np.where(mask)
            if len(rows) == 0:
                continue
            ci = int(np.round(rows.mean()))
            cj = int(np.round(cols.mean()))
            centroid = self._room_map_xy(ci, cj)
            out.append({
                'id': int(rid),
                'centroid_xy': [float(centroid[0]), float(centroid[1])],
                'area_cells': int(mask.sum()),
                'bbox_rc': [int(cols.min()), int(rows.min()), int(cols.max()), int(rows.max())],
            })
        return out

    def _postprocess_room_map(self) -> None:
        if self._room_map is None:
            return
        merge_small_area_cells = int(self.merge_small_rooms_area_cells)
        min_contact_cells = int(self.merge_small_rooms_min_contact)
        if merge_small_area_cells <= 0:
            self._saved_rooms = self._saved_rooms_from_room_map()
            return

        label_map = np.asarray(self._room_map, dtype=np.int32).copy()
        merged_pairs = []
        changed = True
        while changed:
            changed = False
            room_ids = sorted((int(v) for v in np.unique(label_map) if int(v) > 0), key=lambda rid: int(np.count_nonzero(label_map == rid)))
            for rid in room_ids:
                mask = (label_map == rid)
                area = int(np.count_nonzero(mask))
                if area == 0 or area > merge_small_area_cells:
                    continue
                neighbors = {}
                up = label_map[:-1, :][mask[1:, :]]
                down = label_map[1:, :][mask[:-1, :]]
                left = label_map[:, :-1][mask[:, 1:]]
                right = label_map[:, 1:][mask[:, :-1]]
                for arr in (up, down, left, right):
                    for lbl in arr:
                        lbl = int(lbl)
                        if lbl <= 0 or lbl == rid:
                            continue
                        neighbors[lbl] = neighbors.get(lbl, 0) + 1
                if not neighbors:
                    continue
                target_id, contact = max(neighbors.items(), key=lambda kv: (kv[1], int(np.count_nonzero(label_map == kv[0]))))
                if int(contact) < min_contact_cells:
                    continue
                label_map[mask] = int(target_id)
                merged_pairs.append({'source': int(rid), 'target': int(target_id), 'area_cells': int(area), 'contact_cells': int(contact)})
                changed = True
                break

        self._room_map = label_map
        self._saved_rooms = self._saved_rooms_from_room_map()
        if merged_pairs:
            self.get_logger().info(
                f"[INSPECTOR] Room postprocess merged {len(merged_pairs)} small rooms "
                f"(<= {merge_small_area_cells} cells, contact>={min_contact_cells})")

    def _load_map(self):
        map_yaml = self.map_yaml
        if not map_yaml or not os.path.isfile(map_yaml):
            inferred = os.path.join(self.model_dir, "map.yaml")
            if os.path.isfile(inferred):
                map_yaml = inferred
            else:
                saved_ref = self._saved_map_snapshot.get("map_yaml")
                if isinstance(saved_ref, str) and saved_ref:
                    candidate = saved_ref if os.path.isabs(saved_ref) else os.path.join(self.model_dir, saved_ref)
                    if os.path.isfile(candidate):
                        map_yaml = candidate
        if not map_yaml or not os.path.isfile(map_yaml):
            self.get_logger().warn(f"[INSPECTOR] map_yaml not found: {self.map_yaml}")
            return

        self.map_yaml = map_yaml

        try:
            with open(self.map_yaml, "r", encoding="utf-8") as fh:
                meta = yaml.safe_load(fh)

            image_path = meta["image"]
            if not os.path.isabs(image_path):
                image_path = os.path.join(os.path.dirname(self.map_yaml), image_path)

            img = Image.open(image_path).convert("L")
            arr = np.asarray(img, dtype=np.float32) / 255.0

            negate = int(meta.get("negate", 0))
            occ_prob = arr if negate else (1.0 - arr)
            occ_thresh = float(meta.get("occupied_thresh", 0.65))
            free_thresh = float(meta.get("free_thresh", 0.196))

            self._map_occ = occ_prob >= occ_thresh
            self._map_blocked = self._inflate_occ(self._map_occ, self.inflation_cells)
            self._map_resolution = float(meta["resolution"])
            origin = meta.get("origin", [0.0, 0.0, 0.0])
            self._map_origin = (float(origin[0]), float(origin[1]))

            occ_grid = np.full(arr.shape, -1, dtype=np.int8)
            occ_grid[occ_prob >= occ_thresh] = 100
            occ_grid[occ_prob <= free_thresh] = 0
            occ_grid = np.flipud(occ_grid)
            # Store post-flip array: row 0 = y=origin_y (bottom of world).
            # This matches the ROS OccupancyGrid convention and what the watershed
            # segmenter expects (same as mapping_node's self._slam_arr).
            self._map_slam_arr = occ_grid.copy()

            h, w = occ_grid.shape
            msg = OccupancyGrid()
            msg.header.frame_id = "map"
            msg.info = MapMetaData()
            msg.info.map_load_time = self.get_clock().now().to_msg()
            msg.info.resolution = self._map_resolution
            msg.info.width = w
            msg.info.height = h
            msg.info.origin.position.x = self._map_origin[0]
            msg.info.origin.position.y = self._map_origin[1]
            msg.info.origin.orientation.w = 1.0
            msg.data = occ_grid.reshape(-1).tolist()
            self._map_msg = msg

            self.get_logger().info(
                f"[INSPECTOR] Map loaded: {w}x{h} @ {self._map_resolution:.3f}m"
            )
        except Exception as exc:
            self.get_logger().error(f"[INSPECTOR] Failed loading map: {exc}")

    def _run_watershed(self) -> None:
        """Run watershed segmentation on the loaded SLAM map to populate rooms and doorways.

        If topology.json already has rooms/doors (from a mapping session), those take
        precedence. Otherwise fall back to computing fresh from the map.
        """
        if self._map_slam_arr is None or self._map_resolution is None:
            return
        # If topology already provided rooms and doors, use them.
        if self._saved_rooms and self._saved_doors:
            self.get_logger().info(
                f"[INSPECTOR] Using {len(self._saved_rooms)} rooms and "
                f"{len(self._saved_doors)} doors from topology.json"
            )
            return
        try:
            seg = WatershedSegmenter(
                min_peak_clearance_m=0.30,
                min_peak_separation_m=1.2,
                min_room_area_m2=0.8,
                door_min_m=0.55,
                door_max_m=3.50,
            )
            origin = self._map_origin or (0.0, 0.0)
            rooms, doorways = seg.update(
                slam_arr=self._map_slam_arr,
                cell_size_m=self._map_resolution,
                origin_xy=np.array(origin, dtype=np.float32),
                frontier_mask=None,
            )
            # Convert to the dict format _saved_rooms / _saved_doors expect
            self._saved_rooms = [
                {
                    "id": int(r.id),
                    "centroid_xy": np.asarray(r.centroid_xy, dtype=np.float32),
                    "bounding_box": np.asarray(r.bounding_box, dtype=np.float32),
                    "area_m2": float(r.area_m2),
                    "confirmed": True,
                    "doors": [int(d.id) for d in doorways if r.id in d.room_ids],
                }
                for r in rooms
            ]
            self._saved_doors = [
                {
                    "id": int(d.id),
                    "pos_xy": np.asarray(d.position_xy, dtype=np.float32),
                    "room_ids": list(d.room_ids),
                    "width": float(d.width_m),
                }
                for d in doorways
            ]
            self.get_logger().info(
                f"[INSPECTOR] Watershed computed {len(self._saved_rooms)} rooms, "
                f"{len(self._saved_doors)} doorways from map"
            )
        except Exception as exc:
            self.get_logger().error(f"[INSPECTOR] Watershed failed: {exc}")

    def _launch_rviz(self):
        rviz_config = str(self.get_parameter("rviz_config").value).strip()
        if not rviz_config:
            share = get_package_share_directory("mntfields")
            rviz_config = os.path.join(share, "rviz", "region_graph_inspector.rviz")
        if not os.path.isfile(rviz_config):
            self.get_logger().warn(f"[INSPECTOR] RViz config not found: {rviz_config}")
            return

        try:
            self._rviz_proc = subprocess.Popen(["rviz2", "-d", rviz_config])
            self.get_logger().info(f"[INSPECTOR] Launched RViz: {rviz_config}")
        except Exception as exc:
            self.get_logger().warn(f"[INSPECTOR] Could not launch RViz: {exc}")

    def _publish_once_active(self):
        if self._map_msg is not None:
            self._map_msg.header.stamp = self.get_clock().now().to_msg()
            self.map_pub.publish(self._map_msg)
        self._publish_path()
        self._publish_raw_path()
        self._publish_rollout_compare_paths()
        self._publish_start_goal_markers()

    def _publish_graph_markers(self):
        arr = MarkerArray()
        dbg = MarkerArray()
        now = self.get_clock().now().to_msg()
        marker_id = 0

        for version, region in sorted(self.regions.items()):
            b = region.bound.detach().cpu().numpy()
            xmin, ymin = float(b[0, 0]), float(b[0, 1])
            xmax, ymax = float(b[1, 0]), float(b[1, 1])
            rc = _region_colour(int(version))

            line = Marker()
            line.header.frame_id = "map"
            line.header.stamp = now
            line.ns = "regions"
            line.id = marker_id
            marker_id += 1
            line.type = Marker.LINE_STRIP
            line.action = Marker.ADD
            line.scale.x = 0.05
            line.color.r = float(rc[0])
            line.color.g = float(rc[1])
            line.color.b = float(rc[2])
            line.color.a = 0.9
            line.lifetime = Duration(seconds=2.0).to_msg()

            for x, y in (
                (xmin, ymin),
                (xmax, ymin),
                (xmax, ymax),
                (xmin, ymax),
                (xmin, ymin),
            ):
                p = Point()
                p.x = x
                p.y = y
                p.z = 0.05
                line.points.append(p)
            arr.markers.append(line)

            txt = Marker()
            txt.header.frame_id = "map"
            txt.header.stamp = now
            txt.ns = "region_ids"
            txt.id = marker_id
            marker_id += 1
            txt.type = Marker.TEXT_VIEW_FACING
            txt.action = Marker.ADD
            txt.pose.position.x = float((xmin + xmax) * 0.5)
            txt.pose.position.y = float((ymin + ymax) * 0.5)
            txt.pose.position.z = 0.25
            txt.scale.z = 0.35
            txt.color.r = 1.0
            txt.color.g = 1.0
            txt.color.b = 1.0
            txt.color.a = 1.0
            txt.text = f"R{version}"
            txt.lifetime = Duration(seconds=2.0).to_msg()
            dbg.markers.append(txt)

        for cp in self._connection_points:
            pos = self._coerce_xy(cp.get("pos") if isinstance(cp, dict) else None)
            if pos is None:
                continue
            m = Marker()
            m.header.frame_id = "map"
            m.header.stamp = now
            m.ns = "connection_points"
            m.id = marker_id
            marker_id += 1
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(pos[0])
            m.pose.position.y = float(pos[1])
            m.pose.position.z = 0.12
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.14
            m.color.r = 0.2
            m.color.g = 0.8
            m.color.b = 1.0
            m.color.a = 0.95
            m.lifetime = Duration(seconds=2.0).to_msg()
            dbg.markers.append(m)

        for door in self._saved_doors:
            pos = self._coerce_xy(door.get("pos_xy"))
            if pos is None:
                continue
            m = Marker()
            m.header.frame_id = "map"
            m.header.stamp = now
            m.ns = "saved_doors"
            m.id = marker_id
            marker_id += 1
            m.type = Marker.CYLINDER
            m.action = Marker.ADD
            m.pose.position.x = float(pos[0])
            m.pose.position.y = float(pos[1])
            m.pose.position.z = 0.10
            m.pose.orientation.w = 1.0
            m.scale.x = 0.20
            m.scale.y = 0.20
            m.scale.z = 0.04
            m.color.r = 0.1
            m.color.g = 0.95
            m.color.b = 0.25
            m.color.a = 0.95
            m.lifetime = Duration(seconds=2.0).to_msg()
            dbg.markers.append(m)

        for room in self._saved_rooms:
            centroid = self._coerce_xy(room.get("centroid_xy"))
            if centroid is None:
                continue
            m = Marker()
            m.header.frame_id = "map"
            m.header.stamp = now
            m.ns = "saved_rooms"
            m.id = marker_id
            marker_id += 1
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(centroid[0])
            m.pose.position.y = float(centroid[1])
            m.pose.position.z = 0.08
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.16
            m.color.r = 1.0
            m.color.g = 0.45
            m.color.b = 0.2
            m.color.a = 0.90
            m.lifetime = Duration(seconds=2.0).to_msg()
            dbg.markers.append(m)

            t = Marker()
            t.header.frame_id = "map"
            t.header.stamp = now
            t.ns = "saved_room_ids"
            t.id = marker_id
            marker_id += 1
            t.type = Marker.TEXT_VIEW_FACING
            t.action = Marker.ADD
            t.pose.position.x = float(centroid[0])
            t.pose.position.y = float(centroid[1])
            t.pose.position.z = 0.26
            t.scale.z = 0.20
            t.color.r = 1.0
            t.color.g = 1.0
            t.color.b = 1.0
            t.color.a = 0.95
            t.text = f"room {int(room.get('id', -1))}"
            t.lifetime = Duration(seconds=2.0).to_msg()
            dbg.markers.append(t)

        for version, pts in sorted(self._saved_frontier_banks.items()):
            for pt in pts[-40:]:
                m = Marker()
                m.header.frame_id = "map"
                m.header.stamp = now
                m.ns = "saved_frontiers"
                m.id = marker_id
                marker_id += 1
                m.type = Marker.SPHERE
                m.action = Marker.ADD
                m.pose.position.x = float(pt[0])
                m.pose.position.y = float(pt[1])
                m.pose.position.z = 0.06
                m.pose.orientation.w = 1.0
                m.scale.x = m.scale.y = m.scale.z = 0.09
                m.color.r = 1.0
                m.color.g = 0.55
                m.color.b = 0.0
                m.color.a = 0.85
                m.lifetime = Duration(seconds=2.0).to_msg()
                dbg.markers.append(m)

        # Sparse route graph edges.
        for u, v, edata in self._route_graph.edges(data=True):
            edge = Marker()
            edge.header.frame_id = "map"
            edge.header.stamp = now
            edge.ns = "sparse_graph"
            edge.id = marker_id
            marker_id += 1
            edge.type = Marker.LINE_STRIP
            edge.action = Marker.ADD
            edge.scale.x = 0.04
            edge.color.r = 1.0
            edge.color.g = 0.85
            edge.color.b = 0.1
            edge.color.a = 0.95
            edge.lifetime = Duration(seconds=2.0).to_msg()
            for xy in edata["path"]:
                p = Point()
                p.x = float(xy[0])
                p.y = float(xy[1])
                p.z = 0.10
                edge.points.append(p)
            dbg.markers.append(edge)

        # Selected region route highlighted.
        if len(self._active_region_route) >= 2:
            for i in range(len(self._active_region_route) - 1):
                u = self._active_region_route[i]
                v = self._active_region_route[i + 1]
                if not self._route_graph.has_edge(u, v):
                    continue
                route = self._edge_path_oriented(u, v)
                edge = Marker()
                edge.header.frame_id = "map"
                edge.header.stamp = now
                edge.ns = "selected_route"
                edge.id = marker_id
                marker_id += 1
                edge.type = Marker.LINE_STRIP
                edge.action = Marker.ADD
                edge.scale.x = 0.09
                edge.color.r = 0.2
                edge.color.g = 1.0
                edge.color.b = 0.2
                edge.color.a = 0.98
                edge.lifetime = Duration(seconds=2.0).to_msg()
                for xy in route:
                    p = Point()
                    p.x = float(xy[0])
                    p.y = float(xy[1])
                    p.z = 0.14
                    edge.points.append(p)
                dbg.markers.append(edge)

        # Per-segment debug overlays — colour-coded by region, staggered Z.
        for seg_idx, seg in enumerate(self._debug_segments):
            pts = seg.get("points")
            if pts is None or len(pts) < 2:
                continue
            kind = str(seg.get("kind", "unknown"))
            reason = str(seg.get("reason", ""))
            label = str(seg.get("label", ""))

            # Extract region version from label "R4" etc.
            reg_ver = -1
            if label.startswith("R"):
                try:
                    reg_ver = int(label[1:])
                except ValueError:
                    pass
            rc = _region_colour(reg_ver) if reg_ver >= 0 else (1.0, 1.0, 1.0)

            line = Marker()
            line.header.frame_id = "map"
            line.header.stamp = now
            line.ns = "segment_debug"
            line.id = marker_id
            marker_id += 1
            line.type = Marker.LINE_STRIP
            line.action = Marker.ADD
            line.lifetime = Duration(seconds=2.0).to_msg()

            # Colour: region hue modulated by segment kind.
            # field_ok = full region colour, failures = desaturated/dimmed.
            if kind == "field_ok":
                line.scale.x = 0.07
                line.color.r = rc[0]; line.color.g = rc[1]; line.color.b = rc[2]
                line.color.a = 0.95
            elif kind == "field_failed_raw":
                line.scale.x = 0.04
                line.color.r = 1.0; line.color.g = 0.2; line.color.b = 0.2
                line.color.a = 0.7
            elif "straight_line" in kind:
                line.scale.x = 0.05
                line.color.r = rc[0] * 0.6; line.color.g = rc[1] * 0.6; line.color.b = rc[2] * 0.6
                line.color.a = 0.85
            elif "edge_path" in kind:
                line.scale.x = 0.05
                line.color.r = 1.0; line.color.g = 0.2; line.color.b = 1.0
                line.color.a = 0.85
            elif kind == "connector":
                line.scale.x = 0.04
                line.color.r = 0.2; line.color.g = 0.6; line.color.b = 1.0
                line.color.a = 0.85
            else:
                line.scale.x = 0.04
                line.color.r = 0.7; line.color.g = 0.7; line.color.b = 0.7
                line.color.a = 0.7

            seg_z = _Z_DEBUG_SEG + seg_idx * 0.005  # slight stagger per segment
            for xy in pts:
                p = Point()
                p.x = float(xy[0])
                p.y = float(xy[1])
                p.z = seg_z
                line.points.append(p)
            dbg.markers.append(line)

            # Badge at segment start: concise "R4 field" or "R4 A*"
            badge_text = label
            if "field_ok" in kind:
                badge_text += " field"
            elif "straight_line" in kind:
                badge_text += " straight"
            elif "edge_path" in kind:
                badge_text += " edge"
            elif "astar" in kind:
                badge_text += " A*"
            else:
                badge_text += f" {kind[:12]}"

            start_pt = pts[0]
            badge = Marker()
            badge.header.frame_id = "map"
            badge.header.stamp = now
            badge.ns = "segment_badges"
            badge.id = marker_id
            marker_id += 1
            badge.type = Marker.TEXT_VIEW_FACING
            badge.action = Marker.ADD
            badge.pose.position.x = float(start_pt[0])
            badge.pose.position.y = float(start_pt[1])
            badge.pose.position.z = seg_z + 0.12
            badge.scale.z = 0.14
            badge.color.r = rc[0]; badge.color.g = rc[1]; badge.color.b = rc[2]
            badge.color.a = 1.0
            badge.text = badge_text
            badge.lifetime = Duration(seconds=2.0).to_msg()
            dbg.markers.append(badge)

            # Reason label at midpoint (only if there's a failure reason).
            if reason:
                mid = pts[len(pts) // 2]
                txt = Marker()
                txt.header.frame_id = "map"
                txt.header.stamp = now
                txt.ns = "segment_reasons"
                txt.id = marker_id
                marker_id += 1
                txt.type = Marker.TEXT_VIEW_FACING
                txt.action = Marker.ADD
                txt.pose.position.x = float(mid[0])
                txt.pose.position.y = float(mid[1])
                txt.pose.position.z = seg_z + 0.20
                txt.scale.z = 0.11
                txt.color.r = 1.0; txt.color.g = 0.85; txt.color.b = 0.85
                txt.color.a = 0.9
                txt.text = _short_reason(reason)
                txt.lifetime = Duration(seconds=2.0).to_msg()
                dbg.markers.append(txt)

        # Highlight start and goal regions with translucent filled rectangles.
        if self.start_xy is not None or self.goal_xy is not None:
            for tag, xy, rgba in [
                ("start_region", self.start_xy, (0.1, 0.9, 0.2, 0.12)),
                ("goal_region", self.goal_xy, (1.0, 0.2, 0.1, 0.12)),
            ]:
                if xy is None:
                    continue
                reg = self._region_for_xy(xy)
                if reg is None:
                    continue
                b = reg.bound.detach().cpu().numpy()
                m = Marker()
                m.header.frame_id = "map"
                m.header.stamp = now
                m.ns = tag
                m.id = marker_id
                marker_id += 1
                m.type = Marker.CUBE
                m.action = Marker.ADD
                cx = 0.5 * (b[0, 0] + b[1, 0])
                cy = 0.5 * (b[0, 1] + b[1, 1])
                m.pose.position.x = float(cx)
                m.pose.position.y = float(cy)
                m.pose.position.z = 0.01
                m.pose.orientation.w = 1.0
                m.scale.x = float(b[1, 0] - b[0, 0])
                m.scale.y = float(b[1, 1] - b[0, 1])
                m.scale.z = 0.005
                m.color.r, m.color.g, m.color.b, m.color.a = rgba
                m.lifetime = Duration(seconds=2.0).to_msg()
                arr.markers.append(m)

        # Planning summary text block (top-left area of map).
        if self._debug_segments and self.start_xy is not None and self.goal_xy is not None:
            lines = []
            route_str = " -> ".join(f"R{v}" for v in self._active_region_route)
            lines.append(f"Route: {route_str}")
            for seg in self._debug_segments:
                lbl = seg.get("label", "?")
                kind = seg.get("kind", "?")
                reason = seg.get("reason", "")
                n_pts = len(seg.get("points", []))
                seg_len = self._path_length(np.asarray(seg.get("points", []), dtype=np.float32))
                entry = f"  {lbl}: {kind} ({n_pts}pts, {seg_len:.2f}m)"
                if reason:
                    entry += f" [{_short_reason(reason, 40)}]"
                lines.append(entry)
            summary_text = "\n".join(lines)

            sm = Marker()
            sm.header.frame_id = "map"
            sm.header.stamp = now
            sm.ns = "planning_summary"
            sm.id = marker_id
            marker_id += 1
            sm.type = Marker.TEXT_VIEW_FACING
            sm.action = Marker.ADD
            # Place near the start point, elevated
            sm.pose.position.x = float(self.start_xy[0]) - 0.3
            sm.pose.position.y = float(self.start_xy[1]) + 0.5
            sm.pose.position.z = _Z_SUMMARY
            sm.scale.z = 0.12
            sm.color.r = 1.0; sm.color.g = 1.0; sm.color.b = 1.0; sm.color.a = 0.95
            sm.text = summary_text
            sm.lifetime = Duration(seconds=2.0).to_msg()
            dbg.markers.append(sm)

        self.marker_pub.publish(arr)
        self.marker_debug_pub.publish(dbg)

    def _publish_start_goal_markers(self):
        arr = MarkerArray()
        now = self.get_clock().now().to_msg()

        if self.start_xy is not None:
            m = Marker()
            m.header.frame_id = "map"
            m.header.stamp = now
            m.ns = "start_goal"
            m.id = 0
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(self.start_xy[0])
            m.pose.position.y = float(self.start_xy[1])
            m.pose.position.z = 0.12
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.20
            m.color.r = 0.1
            m.color.g = 0.9
            m.color.b = 0.2
            m.color.a = 1.0
            m.lifetime = Duration(seconds=1.0).to_msg()
            arr.markers.append(m)

        if self.goal_xy is not None:
            m = Marker()
            m.header.frame_id = "map"
            m.header.stamp = now
            m.ns = "start_goal"
            m.id = 1
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(self.goal_xy[0])
            m.pose.position.y = float(self.goal_xy[1])
            m.pose.position.z = 0.12
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.20
            m.color.r = 1.0
            m.color.g = 0.2
            m.color.b = 0.1
            m.color.a = 1.0
            m.lifetime = Duration(seconds=1.0).to_msg()
            arr.markers.append(m)

        if arr.markers:
            self.marker_pub.publish(arr)

    def _publish_path(self):
        if self.path_xy is None or len(self.path_xy) < 2:
            return
        self._publish_path_to_pub(self.path_xy, self.path_pub, z=_Z_FINAL)

    def _publish_raw_path(self):
        if self.path_raw_xy is None or len(self.path_raw_xy) < 2:
            return
        self._publish_path_to_pub(self.path_raw_xy, self.path_raw_pub, z=_Z_FINAL + 0.02)

    def _publish_rollout_compare_paths(self):
        if self.path_bidir_xy is not None and len(self.path_bidir_xy) >= 2:
            self._publish_path_to_pub(self.path_bidir_xy, self.path_bidir_pub, z=_Z_BIDIR)
        if self.path_oneway_xy is not None and len(self.path_oneway_xy) >= 2:
            self._publish_path_to_pub(self.path_oneway_xy, self.path_oneway_pub, z=_Z_ONEWAY)
        if self.path_goal_to_start_xy is not None and len(self.path_goal_to_start_xy) >= 2:
            self._publish_path_to_pub(self.path_goal_to_start_xy, self.path_goal_to_start_pub, z=_Z_GOAL2START)

    def _publish_path_to_pub(self, path_xy: np.ndarray, pub, z: float = 0.05):
        msg = Path()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        for xy in path_xy:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = float(xy[0])
            ps.pose.position.y = float(xy[1])
            ps.pose.position.z = z
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        pub.publish(msg)

    def _initialpose_cb(self, msg: PoseWithCovarianceStamped):
        p = msg.pose.pose.position
        self.start_xy = np.array([p.x, p.y], dtype=np.float32)
        self.get_logger().info(
            f"[INSPECTOR] Start set: ({self.start_xy[0]:.2f}, {self.start_xy[1]:.2f})"
        )
        if self.goal_xy is not None and not self._planning_running:
            self._trigger_plan()

    def _goal_cb(self, msg: PoseStamped):
        p = msg.pose.position
        self.goal_xy = np.array([p.x, p.y], dtype=np.float32)
        self.get_logger().info(
            f"[INSPECTOR] Goal set: ({self.goal_xy[0]:.2f}, {self.goal_xy[1]:.2f})"
        )
        if self.start_xy is None:
            self.get_logger().warn(
                "[INSPECTOR] Set /initialpose first. Goal kept and planning will run after start is set."
            )
            return
        if not self._planning_running:
            self._trigger_plan()

    def _rebuild_ownership(self):
        """Recompute Voronoi ownership map, crossings, and merged room support."""
        if self._map_occ is None or self._map_resolution is None or self._map_origin is None:
            return
        if not self.regions:
            return
        free_flipped = np.flipud(~self._map_occ)
        origin_arr = np.array(self._map_origin, dtype=np.float32)
        self._ownership_map = compute_ownership_map(
            free_flipped, self.regions, origin_arr, self._map_resolution)
        self._crossing_points = compute_crossing_points(
            self._ownership_map, origin_arr, self._map_resolution,
            occ_grid=np.flipud(self._map_occ))
        if self._room_map is not None:
            self._postprocess_room_map()
            room_origin = np.array(self._room_map_origin or self._map_origin, dtype=np.float32)
            room_res = float(self._room_map_resolution or self._map_resolution)
            if self._room_map.shape == self._ownership_map.shape and np.allclose(room_origin, origin_arr) and abs(room_res - float(self._map_resolution)) < 1e-6:
                self._region_room_ownership = compute_region_room_ownership(
                    self._ownership_map, self._room_map, self.regions)
            else:
                self.get_logger().warn(
                    '[INSPECTOR] Room map snapshot shape/origin does not match ownership map; room-first selection may fall back.')
        self.get_logger().info(
            f"[INSPECTOR] Ownership map: owned_cells={int(np.count_nonzero(self._ownership_map))} "
            f"crossings={len(self._crossing_points)} room_links={len((self._region_room_ownership or {}).get('by_region', {}))}")

    def _region_for_xy(self, xy: np.ndarray) -> Optional[Region]:
        return owner_region_for_xy(
            xy, self._ownership_map, self.regions,
            np.array(self._map_origin or [0.0, 0.0], dtype=np.float32),
            self._map_resolution or 0.05)

    def _room_id_for_xy(self, xy: np.ndarray) -> Optional[int]:
        if self._room_map is None or self._room_map_origin is None or self._room_map_resolution is None:
            return None
        return room_id_for_xy(
            xy, self._room_map,
            np.array(self._room_map_origin, dtype=np.float32),
            float(self._room_map_resolution),
        )

    def _resolve_region_for_xy(self, xy: np.ndarray) -> Optional[Region]:
        room_id = self._room_id_for_xy(xy)
        if room_id is not None:
            reg = select_best_region_for_room(
                room_id, self._region_room_ownership, self.regions, goal_xy=xy)
            if reg is not None and self._is_field_viable(reg):
                return reg
        # Ownership-based (Voronoi)
        primary = self._region_for_xy(xy)
        if primary is not None and self._is_field_viable(primary):
            return primary
        # Primary region is non-viable — find nearest viable region whose AABB contains xy
        xy_f = np.asarray(xy[:2], dtype=np.float32)
        viable_containing = []
        for reg in self.regions.values():
            b = reg.bound.detach().cpu().numpy()
            if b[0, 0] <= xy_f[0] <= b[1, 0] and b[0, 1] <= xy_f[1] <= b[1, 1]:
                if self._is_field_viable(reg):
                    d = float(np.linalg.norm(
                        np.array([0.5*(b[0,0]+b[1,0]), 0.5*(b[0,1]+b[1,1])]) - xy_f))
                    viable_containing.append((d, reg))
        if viable_containing:
            viable_containing.sort(key=lambda t: t[0])
            self.get_logger().info(
                f"[INSPECTOR] Goal region R{primary.version if primary else '?'} non-viable, "
                f"using R{viable_containing[0][1].version} instead")
            return viable_containing[0][1]
        # No viable containing region — try nearest viable overall
        best = self._closest_viable_region(xy)
        if best is not None:
            self.get_logger().info(
                f"[INSPECTOR] Goal region R{primary.version if primary else '?'} non-viable, "
                f"using nearest viable R{best.version}")
            return best
        # Last resort: return whatever we have
        return primary or self._closest_region(xy)

    def _closest_region(self, xy: np.ndarray) -> Optional[Region]:
        if not self.regions:
            return None
        return min(
            self.regions.values(),
            key=lambda r: float(
                np.linalg.norm(r.bound.detach().cpu().numpy().mean(axis=0)[:2] - xy)
            ),
        )

    def _closest_viable_region(self, xy: np.ndarray) -> Optional[Region]:
        if not self.regions:
            return None
        xy_f = np.asarray(xy[:2], dtype=np.float32)
        viable = [(float(np.linalg.norm(
            r.bound.detach().cpu().numpy().mean(axis=0)[:2] - xy_f)), r)
            for r in self.regions.values() if self._is_field_viable(r)]
        if not viable:
            return None
        viable.sort(key=lambda t: t[0])
        return viable[0][1]

    def _map_indices(self, xy: np.ndarray) -> Optional[Tuple[int, int]]:
        if self._map_occ is None or self._map_resolution is None or self._map_origin is None:
            return None
        mx = int(math.floor((float(xy[0]) - self._map_origin[0]) / self._map_resolution))
        my = int(math.floor((float(xy[1]) - self._map_origin[1]) / self._map_resolution))
        h, w = self._map_occ.shape
        row = h - 1 - my
        col = mx
        if not (0 <= row < h and 0 <= col < w):
            return None
        return row, col

    def _cell_to_world(self, row: int, col: int) -> np.ndarray:
        assert self._map_resolution is not None and self._map_origin is not None and self._map_occ is not None
        h, _w = self._map_occ.shape
        mx = float(col) + 0.5
        my = float(h - 1 - row) + 0.5
        x = self._map_origin[0] + mx * self._map_resolution
        y = self._map_origin[1] + my * self._map_resolution
        return np.array([x, y], dtype=np.float32)

    def _point_in_collision(self, xy: np.ndarray) -> bool:
        idx = self._map_indices(xy)
        if idx is None or self._map_blocked is None:
            return False
        return bool(self._map_blocked[idx[0], idx[1]])

    def _segment_hits_wall(self, p0: np.ndarray, p1: np.ndarray, step: float = 0.05) -> bool:
        if self._map_blocked is None:
            return False
        dist = float(np.linalg.norm(p1 - p0))
        n = max(2, int(math.ceil(dist / step)))
        for alpha in np.linspace(0.0, 1.0, n):
            p = p0 + alpha * (p1 - p0)
            if self._point_in_collision(p):
                return True
        return False

    def _path_hits_wall(self, path_xy: np.ndarray) -> bool:
        if self._map_blocked is None or len(path_xy) < 2:
            return False
        for i in range(len(path_xy) - 1):
            if self._segment_hits_wall(path_xy[i], path_xy[i + 1]):
                return True
        return False

    def _nearest_free_xy(self, xy: np.ndarray, region: Optional[Region] = None) -> Optional[np.ndarray]:
        if self._map_blocked is None:
            return xy.copy()

        rc = self._map_indices(xy)
        if rc is None:
            return None

        h, w = self._map_blocked.shape
        r0, c0 = rc

        b = None
        if region is not None:
            b = region.bound.detach().cpu().numpy()

        def _inside_region(world_xy: np.ndarray) -> bool:
            if b is None:
                return True
            return bool(b[0, 0] <= world_xy[0] <= b[1, 0] and b[0, 1] <= world_xy[1] <= b[1, 1])

        if not self._map_blocked[r0, c0]:
            wxy = self._cell_to_world(r0, c0)
            if _inside_region(wxy):
                return wxy

        max_rad = int(max(8, round(2.5 / max(self._map_resolution or 0.1, 1e-3))))
        best = None
        best_d2 = 1e18

        for rad in range(1, max_rad + 1):
            rr_min = max(0, r0 - rad)
            rr_max = min(h - 1, r0 + rad)
            cc_min = max(0, c0 - rad)
            cc_max = min(w - 1, c0 + rad)

            for rr in range(rr_min, rr_max + 1):
                for cc in range(cc_min, cc_max + 1):
                    if abs(rr - r0) != rad and abs(cc - c0) != rad:
                        continue
                    if self._map_blocked[rr, cc]:
                        continue
                    wxy = self._cell_to_world(rr, cc)
                    if not _inside_region(wxy):
                        continue
                    d2 = float((rr - r0) ** 2 + (cc - c0) ** 2)
                    if d2 < best_d2:
                        best_d2 = d2
                        best = wxy
            if best is not None:
                return best

        # Fallback: nearest free globally near query.
        for rad in range(max_rad + 1, max_rad + 25):
            rr_min = max(0, r0 - rad)
            rr_max = min(h - 1, r0 + rad)
            cc_min = max(0, c0 - rad)
            cc_max = min(w - 1, c0 + rad)
            for rr in range(rr_min, rr_max + 1):
                for cc in range(cc_min, cc_max + 1):
                    if self._map_blocked[rr, cc]:
                        continue
                    return self._cell_to_world(rr, cc)
        return None

    @staticmethod
    def _rect_gap(b1: np.ndarray, b2: np.ndarray) -> float:
        dx = max(0.0, float(max(b1[0, 0] - b2[1, 0], b2[0, 0] - b1[1, 0])))
        dy = max(0.0, float(max(b1[0, 1] - b2[1, 1], b2[0, 1] - b1[1, 1])))
        return math.hypot(dx, dy)

    def _astar_cells(self, start_rc: Tuple[int, int], goal_rc: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        if self._map_blocked is None:
            return None

        h, w = self._map_blocked.shape
        sr, sc = start_rc
        gr, gc = goal_rc
        if not (0 <= sr < h and 0 <= sc < w and 0 <= gr < h and 0 <= gc < w):
            return None
        if self._map_blocked[sr, sc] or self._map_blocked[gr, gc]:
            return None

        def hcost(r: int, c: int) -> float:
            return math.hypot(r - gr, c - gc)

        nbrs = [
            (-1, 0, 1.0),
            (1, 0, 1.0),
            (0, -1, 1.0),
            (0, 1, 1.0),
            (-1, -1, math.sqrt(2.0)),
            (-1, 1, math.sqrt(2.0)),
            (1, -1, math.sqrt(2.0)),
            (1, 1, math.sqrt(2.0)),
        ]

        open_heap: List[Tuple[float, float, Tuple[int, int]]] = []
        heapq.heappush(open_heap, (hcost(sr, sc), 0.0, (sr, sc)))
        g_score: Dict[Tuple[int, int], float] = {(sr, sc): 0.0}
        parent: Dict[Tuple[int, int], Tuple[int, int]] = {}
        closed: Set[Tuple[int, int]] = set()

        expansions = 0
        max_expansions = 200000

        while open_heap:
            _f, g_cur, cur = heapq.heappop(open_heap)
            if cur in closed:
                continue
            if cur == (gr, gc):
                path = [cur]
                while cur in parent:
                    cur = parent[cur]
                    path.append(cur)
                path.reverse()
                return path

            closed.add(cur)
            expansions += 1
            if expansions > max_expansions:
                return None

            r, c = cur
            for dr, dc, step_cost in nbrs:
                nr = r + dr
                nc = c + dc
                if nr < 0 or nr >= h or nc < 0 or nc >= w:
                    continue
                if self._map_blocked[nr, nc]:
                    continue
                ng = g_cur + step_cost
                old = g_score.get((nr, nc))
                if old is not None and ng >= old:
                    continue
                g_score[(nr, nc)] = ng
                parent[(nr, nc)] = (r, c)
                heapq.heappush(open_heap, (ng + hcost(nr, nc), ng, (nr, nc)))

        return None

    def _astar_world(self, start_xy: np.ndarray, goal_xy: np.ndarray) -> Optional[np.ndarray]:
        s = self._nearest_free_xy(start_xy)
        g = self._nearest_free_xy(goal_xy)
        if s is None or g is None:
            return None
        src = self._map_indices(s)
        dst = self._map_indices(g)
        if src is None or dst is None:
            return None
        cells = self._astar_cells(src, dst)
        if not cells:
            return None
        return np.asarray([self._cell_to_world(r, c) for r, c in cells], dtype=np.float32)

    @staticmethod
    def _path_length(path_xy: np.ndarray) -> float:
        if path_xy is None or len(path_xy) < 2:
            return 0.0
        return float(np.sum(np.linalg.norm(np.diff(path_xy, axis=0), axis=1)))

    @staticmethod
    def _safe_pct(num: float, den: float) -> float:
        if den <= 1e-6:
            return 0.0
        return 100.0 * num / den

    def _is_field_viable(self, region: Region) -> bool:
        """Quick sanity check: does the field produce finite TT at the region center?"""
        if region is None:
            return False
        fn = getattr(region, 'function', None)
        if fn is None:
            return False
        b = region.bound.detach().cpu().numpy()
        cx = 0.5 * (b[0, 0] + b[1, 0])
        cy = 0.5 * (b[0, 1] + b[1, 1])
        dim = int(getattr(region, 'dim', getattr(region.model, 'dim', 3)))
        # Evaluate TT(center, center+small_offset) — if finite, field is alive
        src_n = region.normalize(np.array([[cx, cy, 0.0]], dtype=np.float32))[0]
        offsets = np.array([[0.1, 0.0], [-0.1, 0.0], [0.0, 0.1], [0.0, -0.1]], dtype=np.float32)
        XP = np.zeros((len(offsets), 2 * dim), dtype=np.float32)
        for i, off in enumerate(offsets):
            tgt = region.normalize(np.array([[cx + off[0], cy + off[1], 0.0]], dtype=np.float32))[0]
            XP[i, :dim] = src_n
            XP[i, dim:] = tgt
        XP_t = torch.tensor(XP, dtype=torch.float32, device=region.device)
        lock = getattr(region, 'inference_lock', None)
        if lock is not None and not lock.acquire(timeout=0.2):
            return False
        try:
            with torch.no_grad():
                tt = fn.TravelTimes(XP_t).detach().cpu().numpy()
            return bool(np.isfinite(tt).any())
        except Exception:
            return False
        finally:
            if lock is not None:
                lock.release()

    def _build_sparse_route_graph(self):
        self._route_graph.clear()
        self._region_anchor.clear()

        if not self.regions:
            return
        if self._map_blocked is None:
            self.get_logger().warn("[INSPECTOR] Map unavailable. Sparse graph cannot be grid-validated.")
            return

        # Add nodes + anchors.
        for ver, region in self.regions.items():
            center = np.asarray(region.center[:2], dtype=np.float32)
            anchor = self._nearest_free_xy(center, region=region)
            if anchor is None:
                anchor = self._nearest_free_xy(center)
            if anchor is None:
                self.get_logger().warn(f"[INSPECTOR] Region R{ver} has no reachable free anchor; skipping.")
                continue
            self._region_anchor[ver] = anchor
            self._route_graph.add_node(ver, center=center, anchor=anchor)

        nodes = sorted(self._route_graph.nodes())
        if len(nodes) < 2:
            return

        # Build candidate pairs from Voronoi crossing points (primary) and
        # topology hints (secondary).  AABB-gap heuristic is removed — crossing
        # points are topologically grounded and don't suffer from the
        # overlapping-AABB false-positive problem.
        candidate_pairs: Set[Tuple[int, int]] = set()
        kept = 0

        if not self._strict_roomwise_topology:
            for (u, v) in self._crossing_points:
                if u in self._route_graph and v in self._route_graph:
                    candidate_pairs.add((u, v))

        # Supplement with topology edges — added directly without A* validation
        # because topology edges already encode confirmed room connectivity.
        # Supports both {"v1":..,"v2":..} and {"rooms":[a,b],"position":[x,y]} formats.
        # Room IDs are translated to region versions via _room_to_region so this
        # works for both online models (many rooms → few regions) and offline-refined
        # models (room_id == region_version, identity mapping).
        topo_pairs: Set[Tuple[int, int]] = set()
        for e in self._raw_topo_edges:
            rooms = e.get("rooms")
            if rooms and len(rooms) >= 2:
                u = self._room_to_region.get(int(rooms[0]), int(rooms[0]))
                v = self._room_to_region.get(int(rooms[1]), int(rooms[1]))
            else:
                u = int(e.get("v1", -1))
                v = int(e.get("v2", -1))
            if u not in self._route_graph or v not in self._route_graph or u == v:
                continue
            key = (min(u, v), max(u, v))
            topo_pairs.add(key)
            candidate_pairs.add(key)
            # Register doorway position as crossing point
            pos = e.get("position")
            if pos and len(pos) >= 2:
                if key not in self._crossing_points:
                    self._crossing_points[key] = np.asarray(pos[:2], dtype=np.float32)
            # Add edge directly — topology connectivity is ground truth
            w = float(e.get("weight", np.linalg.norm(
                self._region_anchor.get(u, np.zeros(2)) -
                self._region_anchor.get(v, np.zeros(2)))))
            crossing_xy = self._crossing_points.get(key)
            if not self._route_graph.has_edge(u, v):
                self._route_graph.add_edge(
                    u, v, weight=max(w, 0.1),
                    path=[], endpoints=(u, v), crossing=crossing_xy)
                kept += 1

        # Check field viability for each region (cached for graph building).
        viable = {}
        for ver in self._route_graph.nodes():
            reg = self.regions.get(ver)
            v = self._is_field_viable(reg) if reg is not None else False
            viable[ver] = v
            if not v:
                self.get_logger().warn(
                    f"[INSPECTOR] R{ver} field non-viable (epoch={getattr(reg, 'epoch', '?')})")

        # Validate each candidate edge with A* between region anchors.
        # Penalize edges through non-viable regions so routing prefers trained fields.
        NON_VIABLE_PENALTY = 50.0  # metres added to weight
        kept = 0
        for u, v in sorted(candidate_pairs):
            a_u = self._region_anchor.get(u)
            a_v = self._region_anchor.get(v)
            if a_u is None or a_v is None:
                continue
            edge_path = self._astar_world(a_u, a_v)
            if edge_path is None or len(edge_path) < 2:
                continue
            plen = self._path_length(edge_path)
            d = float(np.linalg.norm(a_u - a_v))
            stretch = plen / max(0.05, d)
            if stretch > 3.5:
                continue
            crossing_xy = self._crossing_points.get((u, v))
            # Penalize if either endpoint region has a non-viable field
            w = plen
            if not viable.get(u, False):
                w += NON_VIABLE_PENALTY
            if not viable.get(v, False):
                w += NON_VIABLE_PENALTY
            self._route_graph.add_edge(
                u,
                v,
                weight=w,
                path=edge_path,
                endpoints=(u, v),
                crossing=crossing_xy,
            )
            kept += 1

        self.get_logger().info(
            f"[INSPECTOR] Sparse graph built: nodes={self._route_graph.number_of_nodes()}, "
            f"candidate_pairs={len(candidate_pairs)}, kept_edges={kept}"
        )

    def _all_doorways_for_regions(
        self, src_region: "Region", dst_region: "Region", goal_xy: np.ndarray,
        cur_xy: Optional[np.ndarray] = None,
    ) -> List[np.ndarray]:
        """Return doorways connecting src_region and dst_region.

        Sorted by distance to cur_xy (robot's current position) so the field
        always targets the nearest reachable doorway first — the closest doorway
        is most likely to be within the source region's well-trained coverage.
        Falls back to sorting by goal_xy when cur_xy is not provided.

        Primary method: match door.room_ids against region versions (exact, reliable).
        Fallback: spatial in_bound() with a small margin — covers cases where region
        version != room id (e.g. online-trained models).
        """
        if not self._saved_doors:
            return []

        s_ver = src_region.version
        g_ver = dst_region.version

        # ── Primary: topology room_ids match ────────────────────────────────
        candidates = [
            door for door in self._saved_doors
            if set(self._room_to_region.get(int(r), int(r))
                   for r in door.get('room_ids', [])) == {s_ver, g_ver}
        ]

        # ── Fallback: spatial in_bound with 0.3 m margin ────────────────────
        if not candidates and not self._strict_roomwise_topology:
            MARGIN = 0.3
            def _in_bound_margin(region, pos_xy):
                p = np.asarray(pos_xy[:2], dtype=np.float32)
                b = region.bound.numpy() if hasattr(region.bound, 'numpy') else np.array(region.bound)
                return (b[0, 0] - MARGIN <= p[0] <= b[1, 0] + MARGIN and
                        b[0, 1] - MARGIN <= p[1] <= b[1, 1] + MARGIN)
            candidates = [
                door for door in self._saved_doors
                if _in_bound_margin(src_region, door['pos_xy'])
                and _in_bound_margin(dst_region, door['pos_xy'])
            ]

        if not candidates:
            return []

        # Sort by distance from the robot's current position so the nearest
        # doorway (most likely within the source field's trained coverage) is
        # tried first.  Fall back to distance-to-goal if cur_xy not given.
        ref = np.asarray(cur_xy[:2], dtype=np.float32) if cur_xy is not None \
            else np.asarray(goal_xy[:2], dtype=np.float32)
        candidates.sort(key=lambda d: float(np.linalg.norm(
            np.asarray(d['pos_xy'][:2], dtype=np.float32) - ref)))
        self.get_logger().info(
            f"[INSPECTOR] Doorway candidates R{s_ver}→R{g_ver} "
            f"(sorted by {'cur' if cur_xy is not None else 'goal'}): "
            + ", ".join(
                f"id={d['id']} @ ({float(d['pos_xy'][0]):.2f},{float(d['pos_xy'][1]):.2f})"
                for d in candidates
            )
        )
        return [np.asarray(d['pos_xy'][:2], dtype=np.float32) for d in candidates]

    def _field_from_doorway(
        self, g_ver: int, door_xy: np.ndarray, goal: np.ndarray, seg_budget: int
    ) -> Tuple[Optional[np.ndarray], str]:
        """Try goal-region field from doorway; nudge toward goal if exact position fails.

        The doorway sits at the region boundary where training data may be sparse.
        Nudging the start a short distance into the goal region's well-covered interior
        often recovers a valid gradient. A collision-free bridge connects door → nudged start.

        Returns (path_Nx2, reason_str). path is None if all attempts fail.
        """
        attempts = self._doorway_anchors_in_region(g_ver, door_xy, goal)
        if not attempts:
            return None, "no_goal_region_doorway_anchor"

        last_reason = "unknown"
        for offset, anchor_xy in attempts:
            final_field, reason, _ = self._field_segment(g_ver, anchor_xy, goal, seg_budget)
            if final_field is None:
                last_reason = reason
                continue
            if offset <= 1e-6:
                return final_field, reason

            bridge = straight_line_path(
                door_xy[:2], anchor_xy[:2], step=0.05,
                collision_fn=lambda xy: self._point_in_collision(xy),
            )
            if len(bridge) < 2:
                last_reason = f"goal_bridge_too_short@{offset:.2f}"
                continue
            bridge_np = np.asarray(bridge, dtype=np.float32)
            if self._path_hits_wall(bridge_np):
                last_reason = f"goal_bridge_collides@{offset:.2f}"
                continue

            combined = np.vstack([bridge_np, np.asarray(final_field, dtype=np.float32)[1:]])
            self.get_logger().info(
                f"[INSPECTOR] R{g_ver} field succeeded after inward doorway cushion {offset:.2f}m."
            )
            return combined, f"doorway_cushion_{offset:.2f}m+{reason}"

        return None, f"all_goal_doorway_cushions_failed;{last_reason}"

    def _doorway_anchors_in_region(
        self, region_ver: int, door_xy: np.ndarray, toward_xy: np.ndarray
    ) -> List[Tuple[float, np.ndarray]]:
        """Return collision-free anchor candidates from a doorway into a region.

        The exact doorway center often lies on the room boundary. For refined
        room-wise fields, that makes the boundary act like a hard wall. This
        returns progressively inward anchor points toward ``toward_xy`` while
        keeping them inside the region and on free cells.
        """
        region = self.regions.get(region_ver)
        if region is None:
            return []

        door = np.asarray(door_xy[:2], dtype=np.float32)
        target = np.asarray(toward_xy[:2], dtype=np.float32)
        direction = target - door
        dist = float(np.linalg.norm(direction))
        if dist < 1e-6:
            direction = np.array([1.0, 0.0], dtype=np.float32)
        else:
            direction = (direction / dist).astype(np.float32)

        attempts: List[Tuple[float, np.ndarray]] = []
        seen: Set[Tuple[int, int]] = set()
        offsets = self.doorway_cushion_offsets_m or [0.0]

        for offset in offsets:
            cand = door if offset <= 1e-6 else (door + direction * float(offset)).astype(np.float32)
            cand = self._clip_to_region_bounds(region_ver, cand)
            cand_free = self._nearest_free_xy(cand, region=region)
            if cand_free is None:
                continue
            cand_free = np.asarray(cand_free[:2], dtype=np.float32)
            if not self._point_in_region(region_ver, cand_free):
                continue
            key = (int(round(float(cand_free[0]) * 100.0)), int(round(float(cand_free[1]) * 100.0)))
            if key in seen:
                continue
            seen.add(key)
            attempts.append((float(offset), cand_free))
        return attempts

    def _edge_path_oriented(self, u: int, v: int) -> np.ndarray:
        edata = self._route_graph.edges[u, v]
        p = np.asarray(edata["path"], dtype=np.float32)
        e0, e1 = edata.get("endpoints", (u, v))
        if e0 == u and e1 == v:
            return p
        return p[::-1].copy()

    def _point_in_region(self, region_ver: int, xy: np.ndarray) -> bool:
        region = self.regions.get(region_ver)
        if region is None:
            return False
        b = region.bound.detach().cpu().numpy()
        return bool(b[0, 0] <= xy[0] <= b[1, 0] and b[0, 1] <= xy[1] <= b[1, 1])

    def _extract_transition_on_edge(
        self, edge_path: np.ndarray, u: int, v: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract a stable inter-region connector from the validated edge path.

        This keeps the previous A*-derived connector behavior: use the first
        path sample that lies in region ``u`` as the exit, the last path sample
        that lies in region ``v`` as the entry, and the subpath between them as
        the connector. If region membership cannot be resolved cleanly, fall
        back to the full validated edge path.
        """
        path = np.asarray(edge_path, dtype=np.float32)
        if path.ndim != 2 or len(path) < 2:
            raise ValueError('edge_path must be Nx2 with at least 2 points')

        reg_u = self.regions.get(int(u))
        reg_v = self.regions.get(int(v))
        if reg_u is None or reg_v is None:
            return path[0].copy(), path[-1].copy(), path.copy()

        idx_u = []
        idx_v = []
        for i, pt in enumerate(path):
            if reg_u.in_bound(pt):
                idx_u.append(i)
            if reg_v.in_bound(pt):
                idx_v.append(i)

        if idx_u and idx_v:
            exit_idx = max(idx_u)
            entry_idx = min(idx_v)
            if entry_idx < exit_idx:
                exit_idx = min(idx_u)
                entry_idx = max(idx_v)
            if entry_idx < exit_idx:
                exit_idx = 0
                entry_idx = len(path) - 1
        else:
            exit_idx = 0
            entry_idx = len(path) - 1

        connector = path[exit_idx:entry_idx + 1].copy()
        if len(connector) < 2:
            connector = path.copy()
            exit_idx = 0
            entry_idx = len(path) - 1

        exit_xy = connector[0].copy()
        entry_xy = connector[-1].copy()
        return exit_xy, entry_xy, connector

    def _field_segment(
        self, region_ver: int, start_xy: np.ndarray, goal_xy: np.ndarray, max_steps: int
    ) -> Tuple[Optional[np.ndarray], str, Optional[np.ndarray]]:
        region = self.regions.get(region_ver)
        if region is None:
            return None, "region_not_loaded", None

        # Keep the raw field rollout for debugging/comparison.
        raw_seg_np: Optional[np.ndarray] = None
        raw_reason = "ok"
        try:
            raw_seg = region.generate_field_path(
                start_xy.astype(np.float32),
                goal_xy.astype(np.float32),
                step_size=self.step_size,
                max_steps=max_steps,
            )
        except Exception as exc:
            raw_reason = f"raw_exception:{type(exc).__name__}"
            raw_seg = None
        if raw_seg is None:
            if raw_reason == "ok":
                raw_reason = "raw_returned_none"
        else:
            raw_seg_np = np.asarray(raw_seg, dtype=np.float32)
            if len(raw_seg_np) < 2:
                raw_reason = "raw_too_few_points"
            elif not np.all([self._point_in_region(region_ver, p) for p in raw_seg_np]):
                raw_reason = "raw_out_of_region_bounds"
            elif self._path_hits_wall(raw_seg_np):
                raw_reason = "raw_collides_with_occupied_cells"

        constrained, c_reason = self._constrained_field_rollout(
            region_ver=region_ver,
            start_xy=start_xy,
            goal_xy=goal_xy,
            max_steps=max_steps,
        )
        if constrained is not None and len(constrained) >= 2:
            if raw_reason == "ok":
                return constrained, "ok", raw_seg_np
            return constrained, f"ok_constrained_after_{raw_reason}", raw_seg_np

        # If constrained rollout fails but raw field output is valid, keep the raw field.
        if raw_seg_np is not None and raw_reason == "ok":
            return raw_seg_np, f"ok_raw_after_{c_reason}", raw_seg_np

        return None, f"{c_reason};{raw_reason}", raw_seg_np

    def _clip_to_region_bounds(self, region_ver: int, xy: np.ndarray) -> np.ndarray:
        region = self.regions.get(region_ver)
        if region is None:
            return xy.copy()
        b = region.bound.detach().cpu().numpy()
        margin = max(0.0, self.region_bound_margin_m)
        xmin = float(b[0, 0] + margin)
        ymin = float(b[0, 1] + margin)
        xmax = float(b[1, 0] - margin)
        ymax = float(b[1, 1] - margin)
        if xmin > xmax:
            xmin, xmax = float(b[0, 0]), float(b[1, 0])
        if ymin > ymax:
            ymin, ymax = float(b[0, 1]), float(b[1, 1])
        return np.array(
            [np.clip(float(xy[0]), xmin, xmax), np.clip(float(xy[1]), ymin, ymax)],
            dtype=np.float32,
        )

    def _rescue_field_step(
        self, region_ver: int, cur: np.ndarray, goal: np.ndarray, step: float
    ) -> Optional[np.ndarray]:
        # Try direct-to-goal first.
        d = goal - cur
        dn = float(np.linalg.norm(d))
        if dn > 1e-6:
            cand = cur + (step * d / dn)
            cand = self._clip_to_region_bounds(region_ver, cand)
            if (not self._point_in_collision(cand)) and (not self._segment_hits_wall(cur, cand)):
                return cand

        # Then angular sweep around current heading.
        base = math.atan2(float(d[1]), float(d[0])) if dn > 1e-6 else 0.0
        for ang in np.linspace(-math.pi, math.pi, 24, endpoint=False):
            cand = np.array(
                [cur[0] + step * math.cos(base + ang), cur[1] + step * math.sin(base + ang)],
                dtype=np.float32,
            )
            cand = self._clip_to_region_bounds(region_ver, cand)
            if np.linalg.norm(cand - cur) < 1e-3:
                continue
            if self._point_in_collision(cand):
                continue
            if self._segment_hits_wall(cur, cand):
                continue
            return cand
        return None

    def _constrained_field_rollout(
        self, region_ver: int, start_xy: np.ndarray, goal_xy: np.ndarray, max_steps: int
    ) -> Tuple[Optional[np.ndarray], str]:
        region = self.regions.get(region_ver)
        if region is None:
            return None, "constrained_region_not_loaded"

        start = self._clip_to_region_bounds(region_ver, start_xy.astype(np.float32))
        goal = self._clip_to_region_bounds(region_ver, goal_xy.astype(np.float32))
        start_free = self._nearest_free_xy(start, region=region)
        goal_free = self._nearest_free_xy(goal, region=region)
        if start_free is None:
            return None, "constrained_no_free_start"
        if goal_free is None:
            return None, "constrained_no_free_goal"
        cur = start_free.astype(np.float32)
        goal = goal_free.astype(np.float32)
        pts = [cur.copy()]
        best_goal_dist = float(np.linalg.norm(cur - goal))
        prev_goal_dist = best_goal_dist

        for k in range(max(10, max_steps)):
            goal_dist = float(np.linalg.norm(cur - goal))
            best_goal_dist = min(best_goal_dist, goal_dist)
            if goal_dist <= max(0.12, self.step_size):
                pts.append(goal.copy())
                out = np.asarray(pts, dtype=np.float32)
                if self._path_hits_wall(out):
                    return None, "constrained_final_collides"
                return out, "ok"

            try:
                nxt = region.get_next_waypoint(
                    cur.astype(np.float32),
                    goal.astype(np.float32),
                    step_size=self.step_size,
                    debug=False,
                )
                nxt = np.asarray(nxt, dtype=np.float32)[:2]
            except Exception:
                return None, "constrained_next_waypoint_exception"

            if not np.isfinite(nxt).all():
                return None, "constrained_non_finite_step"

            nxt = self._clip_to_region_bounds(region_ver, nxt)
            if self._point_in_collision(nxt) or self._segment_hits_wall(cur, nxt):
                rescued = self._rescue_field_step(region_ver, cur, goal, self.step_size)
                if rescued is None:
                    return None, "constrained_step_blocked"
                nxt = rescued

            if np.linalg.norm(nxt - cur) < 1e-4:
                return None, (
                    f"constrained_stalled(step={k},dist={goal_dist:.2f},best={best_goal_dist:.2f})"
                )

            pts.append(nxt.copy())
            cur = nxt
            prev_goal_dist = goal_dist

        end_dist = float(np.linalg.norm(cur - goal))
        progress = max(0.0, float(np.linalg.norm(start_free - goal) - end_dist))
        return None, (
            "constrained_max_steps"
            f"(end_dist={end_dist:.2f},best={best_goal_dist:.2f},progress={progress:.2f})"
        )

    @staticmethod
    def _concat_paths(paths: List[np.ndarray]) -> Optional[np.ndarray]:
        if not paths:
            return None
        out = [paths[0]]
        for seg in paths[1:]:
            if len(seg) == 0:
                continue
            if np.linalg.norm(out[-1][-1] - seg[0]) < 1e-3:
                out.append(seg[1:])
            else:
                out.append(seg)
        cat = np.concatenate([s for s in out if len(s) > 0], axis=0)
        if len(cat) < 2:
            return None
        return cat

    @staticmethod
    def _decimate(path_xy: np.ndarray, spacing: float = 0.10) -> np.ndarray:
        pts = path_xy.astype(np.float32)
        out = [pts[0]]
        for i in range(1, len(pts)):
            if np.linalg.norm(pts[i] - out[-1]) >= spacing:
                out.append(pts[i])
        if np.linalg.norm(out[-1] - pts[-1]) > 0.05:
            out.append(pts[-1])
        return np.asarray(out, dtype=np.float32)

    @staticmethod
    def _smooth(path_xy: np.ndarray, window: int = 7) -> np.ndarray:
        pts = path_xy.astype(np.float32)
        if len(pts) < window:
            return pts
        sm = pts.copy()
        k = window // 2
        for i in range(len(pts)):
            a = max(0, i - k)
            b = min(len(pts), i + k + 1)
            sm[i] = pts[a:b].mean(axis=0)
        sm[0] = pts[0]
        sm[-1] = pts[-1]
        return sm

    def _trigger_plan(self):
        if self._planning_running:
            return
        if self.start_xy is None or self.goal_xy is None:
            return
        if self._map_blocked is None:
            self.get_logger().warn("[INSPECTOR] No map loaded. Cannot plan.")
            return
        if self._route_graph.number_of_nodes() == 0:
            self.get_logger().warn("[INSPECTOR] Sparse graph is empty. Cannot plan.")
            return

        start = self.start_xy.copy()
        goal = self.goal_xy.copy()

        s_region = self._resolve_region_for_xy(start)
        g_region = self._resolve_region_for_xy(goal)
        if s_region is None or g_region is None:
            self.get_logger().warn("[INSPECTOR] Could not resolve start/goal regions.")
            return
        if s_region.version not in self._route_graph or g_region.version not in self._route_graph:
            self.get_logger().warn("[INSPECTOR] Start/goal region has no sparse-graph node.")
            return

        self._planning_running = True

        def _worker():
            try:
                s_ver = s_region.version
                g_ver = g_region.version
                self._active_region_route = []
                self.path_raw_xy = None
                self.path_bidir_xy = None
                self.path_oneway_xy = None
                self.path_goal_to_start_xy = None
                self.path_speed_xy = None
                self._debug_segments = []

                # ── Find shortest region sequence via doorway-connected graph ───
                # Build a subgraph that only contains edges where a physical
                # doorway exists between the two regions.  The full _route_graph
                # can have long-distance edges (e.g. R1→R4 directly) that skip
                # intermediate rooms — routing on those causes the field to try a
                # straight shot through walls.  Every edge in doorway_graph is
                # guaranteed to have a real doorway, so every hop is field-navigable.
                if s_ver == g_ver:
                    region_route = [s_ver]
                else:
                    doorway_graph = nx.Graph()
                    for u, v, data in self._route_graph.edges(data=True):
                        reg_u = self.regions.get(u)
                        reg_v = self.regions.get(v)
                        if reg_u is None or reg_v is None:
                            continue
                        # Use a dummy goal direction for sorting; we only care
                        # whether ANY doorway exists, not which one is best yet.
                        if self._all_doorways_for_regions(reg_u, reg_v, goal):
                            doorway_graph.add_edge(
                                u, v, weight=data.get("weight", 1.0)
                            )
                    try:
                        region_route = nx.shortest_path(
                            doorway_graph, s_ver, g_ver, weight="weight"
                        )
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        # No doorway-connected path — fall back to raw route graph
                        self.get_logger().warn(
                            f"[INSPECTOR] No doorway-connected path R{s_ver}→R{g_ver}; "
                            f"falling back to raw route graph."
                        )
                        try:
                            region_route = nx.shortest_path(
                                self._route_graph, s_ver, g_ver, weight="weight"
                            )
                        except nx.NetworkXNoPath:
                            self.get_logger().warn(
                                f"[INSPECTOR] No graph path from R{s_ver} to R{g_ver}."
                            )
                            return
                self._active_region_route = list(region_route)
                self.get_logger().info(
                    f"[INSPECTOR] Region route: "
                    + " → ".join(f"R{v}" for v in region_route)
                )

                segments: List[np.ndarray] = []
                used_parts: List[Tuple[str, np.ndarray]] = []
                cur = start.copy()
                skip_final_leg = False

                seg_budget = max(80, self.max_steps // max(len(region_route), 1))

                if s_ver == g_ver:
                    # ── Same region: direct field path to goal ──────────────────
                    seg_budget = self.max_steps

                else:
                    # ── Multi-hop: execute each consecutive region pair ─────────
                    # For each hop (Ri → Rj) find the doorway between them, plan
                    # a field path from cur → doorway in Ri, then enter Rj.
                    # The final hop lands on the goal.

                    # If start is already inside the goal region, try direct path
                    if g_region.in_bound(start[:2].astype(np.float32)):
                        direct, direct_reason, _ = self._field_segment(
                            g_ver, cur, goal, self.max_steps
                        )
                        if direct is not None:
                            self.get_logger().info(
                                f"[INSPECTOR] Start is inside R{g_ver}; "
                                f"using direct g_ver field (no doorway needed)."
                            )
                            self._debug_segments.append(
                                {"kind": "field_ok", "reason": direct_reason,
                                 "label": f"R{g_ver}(direct)", "points": direct}
                            )
                            used_parts.append(("field_ok", direct))
                            segments.append(direct)
                            skip_final_leg = True

                    if not skip_final_leg:
                        # Walk each hop: (R0→R1), (R1→R2), ..., (R_{n-1}→R_n)
                        # The intermediate waypoint for hop i is the doorway between
                        # region_route[i] and region_route[i+1].
                        # The last hop goes all the way to the goal.
                        hop_failed = False
                        for hop_idx in range(len(region_route) - 1):
                            hop_src_ver = region_route[hop_idx]
                            hop_dst_ver = region_route[hop_idx + 1]
                            hop_src_region = self.regions.get(hop_src_ver)
                            hop_dst_region = self.regions.get(hop_dst_ver)
                            is_last_hop = (hop_idx == len(region_route) - 2)

                            if hop_src_region is None or hop_dst_region is None:
                                self.get_logger().warn(
                                    f"[INSPECTOR] Region R{hop_src_ver} or R{hop_dst_ver} "
                                    f"not loaded; aborting."
                                )
                                hop_failed = True
                                break

                            # Intermediate target: doorway for this hop, or goal on last hop
                            if is_last_hop:
                                hop_target = goal
                            else:
                                # Use centroid of next region as the waypoint target when
                                # searching doorways (pulls the field toward the next room)
                                next_ver = region_route[hop_idx + 2]
                                next_region = self.regions.get(next_ver)
                                if next_region is not None:
                                    hop_target = next_region.bound.mean(dim=0)[:2].numpy()
                                else:
                                    hop_target = goal

                            all_doors = self._all_doorways_for_regions(
                                hop_src_region, hop_dst_region, hop_target,
                                cur_xy=cur,
                            )
                            if not all_doors:
                                self.get_logger().warn(
                                    f"[INSPECTOR] No doorway between R{hop_src_ver} and "
                                    f"R{hop_dst_ver}. Trying Voronoi crossing."
                                )
                                crossing_key = (min(hop_src_ver, hop_dst_ver),
                                                max(hop_src_ver, hop_dst_ver))
                                crossing = self._crossing_points.get(crossing_key)
                                if crossing is None and self._route_graph.has_edge(
                                        hop_src_ver, hop_dst_ver):
                                    edge_path = self._edge_path_oriented(hop_src_ver, hop_dst_ver)
                                    crossing = edge_path[len(edge_path) // 2].copy()
                                if crossing is None:
                                    self.get_logger().warn(
                                        f"[INSPECTOR] No crossing for "
                                        f"R{hop_src_ver}→R{hop_dst_ver}; aborting."
                                    )
                                    hop_failed = True
                                    break
                                all_doors = [np.asarray(crossing[:2], dtype=np.float32)]

                            # One doorway per hop: always use the nearest doorway
                            # to cur (already first in all_doors after cur_xy sort).
                            # No fallback to other doorways — strictly 1 per region.
                            door_xy = all_doors[0]
                            self.get_logger().info(
                                f"[INSPECTOR] Hop {hop_idx+1}/{len(region_route)-1}: "
                                f"R{hop_src_ver}→R{hop_dst_ver} via doorway "
                                f"({door_xy[0]:.2f},{door_xy[1]:.2f})"
                            )

                            # Anchor direction: from doorway back toward cur (into the
                            # source region), not toward the far goal which would push
                            # the anchor out of the src region entirely.
                            source_anchor_attempts = self._doorway_anchors_in_region(
                                hop_src_ver, door_xy, cur
                            )
                            door_succeeded = False
                            if not source_anchor_attempts:
                                self.get_logger().warn(
                                    f"[INSPECTOR] No inward doorway anchor found in R{hop_src_ver}."
                                )
                            else:
                                last_reason_src = "unknown"
                                for src_offset, src_anchor_xy in source_anchor_attempts:
                                    field_src, reason_src, raw_src = self._field_segment(
                                        hop_src_ver, cur, src_anchor_xy, seg_budget
                                    )
                                    if field_src is None:
                                        last_reason_src = reason_src
                                        if raw_src is not None and len(raw_src) >= 2:
                                            self._debug_segments.append(
                                                {"kind": "field_failed_raw",
                                                 "reason": reason_src,
                                                 "label": f"R{hop_src_ver}",
                                                 "points": raw_src}
                                            )
                                        continue

                                    segment_to_use = np.asarray(field_src, dtype=np.float32)
                                    seg_reason = reason_src
                                    if src_offset > 1e-6:
                                        bridge = straight_line_path(
                                            src_anchor_xy[:2], door_xy[:2], step=0.05,
                                            collision_fn=lambda xy: self._point_in_collision(xy),
                                        )
                                        if len(bridge) < 2:
                                            last_reason_src = f"source_bridge_too_short@{src_offset:.2f}"
                                            continue
                                        bridge_np = np.asarray(bridge, dtype=np.float32)
                                        if self._path_hits_wall(bridge_np):
                                            last_reason_src = f"source_bridge_collides@{src_offset:.2f}"
                                            continue
                                        segment_to_use = np.vstack([segment_to_use, bridge_np[1:]])
                                        seg_reason = f"doorway_cushion_{src_offset:.2f}m+{reason_src}"

                                    self._debug_segments.append(
                                        {"kind": "field_ok", "reason": seg_reason,
                                         "label": f"R{hop_src_ver}", "points": segment_to_use}
                                    )
                                    used_parts.append(("field_ok", segment_to_use))
                                    segments.append(segment_to_use)
                                    cur = np.asarray(segment_to_use[-1][:2], dtype=np.float32)
                                    door_succeeded = True
                                    break

                                if not door_succeeded:
                                    self.get_logger().warn(
                                        f"[INSPECTOR] R{hop_src_ver} field to doorway "
                                        f"({door_xy[0]:.2f},{door_xy[1]:.2f}) failed "
                                        f"({last_reason_src})."
                                    )

                            if not door_succeeded:
                                self.get_logger().warn(
                                    f"[INSPECTOR] Hop R{hop_src_ver}→R{hop_dst_ver} failed; aborting."
                                )
                                hop_failed = True
                                break

                        if hop_failed:
                            return

                        # Final leg: from last doorway → goal using goal region's field
                        final_field, reason_g = self._field_from_doorway(
                            g_ver, cur, goal, seg_budget
                        )
                        if final_field is None:
                            self.get_logger().warn(
                                f"[INSPECTOR] Final leg R{g_ver} field failed "
                                f"({reason_g}); no valid path."
                            )
                            return
                        self._debug_segments.append(
                            {"kind": "field_ok", "reason": reason_g,
                             "label": f"R{g_ver}(final)", "points": final_field}
                        )
                        used_parts.append(("field_ok", final_field))
                        segments.append(final_field)
                        skip_final_leg = True

                if not skip_final_leg:
                    # ── Final leg: goal region field  →  goal ───────────────────
                    final_field, reason_g, raw_final = self._field_segment(
                        g_ver, cur, goal, seg_budget
                    )
                    if final_field is None:
                        if raw_final is not None and len(raw_final) >= 2:
                            self._debug_segments.append(
                                {"kind": "field_failed_raw", "reason": reason_g,
                                 "label": f"R{g_ver}", "points": raw_final}
                            )
                        self.get_logger().warn(
                            f"[INSPECTOR] Final field segment failed in R{g_ver} "
                            f"(reason={reason_g}); no valid path."
                        )
                        return
                    self._debug_segments.append(
                        {"kind": "field_ok", "reason": reason_g, "label": f"R{g_ver}",
                         "points": final_field}
                    )
                    used_parts.append(("field_ok", final_field))
                    segments.append(final_field)

                full = self._concat_paths(segments)
                if full is None:
                    self.get_logger().warn("[INSPECTOR] Empty path.")
                    return
                self.path_raw_xy = full.copy()

                decimated = self._decimate(full, spacing=0.10)
                smoothed = self._smooth(decimated, window=7)
                if self._path_hits_wall(smoothed):
                    self.get_logger().warn(
                        "[INSPECTOR] Smoothed path intersects occupied cells. "
                        "Using raw decimated path instead."
                    )
                    final_path = decimated
                    if self._path_hits_wall(decimated):
                        self.get_logger().warn(
                            "[INSPECTOR] Decimated path ALSO intersects occupied cells — "
                            "likely a gap between segments (teleportation through wall)."
                        )
                else:
                    final_path = smoothed
                self.path_xy = final_path
                self._compute_rollout_compare_paths(
                    start=start,
                    goal=goal,
                    s_ver=s_ver,
                    g_ver=g_ver,
                    region_route=region_route,
                    seg_budget=seg_budget,
                )
                self._save_path_overlay_figure(start, goal, region_route)
                total = self._path_length(final_path)
                field_len = float(
                    sum(
                        self._path_length(seg)
                        for kind, seg in used_parts
                        if kind == "field_ok"
                    )
                )
                fallback_len = float(
                    sum(
                        self._path_length(seg)
                        for kind, seg in used_parts
                        if kind == "straight_line_fallback"
                    )
                )
                n_field = sum(1 for kind, _ in used_parts if kind == "field_ok")
                n_fallback = sum(1 for kind, _ in used_parts if kind == "straight_line_fallback")
                route_str = " -> ".join(f"R{v}" for v in region_route)
                self.get_logger().info(
                    f"[INSPECTOR] Path ready: pts={len(final_path)}, len={total:.2f}m, "
                    f"route=[{route_str}], hops={max(0, len(region_route)-1)}"
                )
                self.get_logger().info(
                    "[INSPECTOR] Segment breakdown:"
                )
                for kind, seg in used_parts:
                    self.get_logger().info(
                        f"  {kind}: {len(seg)}pts, {self._path_length(seg):.2f}m"
                    )
                self.get_logger().info(
                    f"[INSPECTOR] Totals: "
                    f"field={field_len:.2f}m ({self._safe_pct(field_len, field_len + fallback_len):.1f}%), "
                    f"fallback={fallback_len:.2f}m ({self._safe_pct(fallback_len, field_len + fallback_len):.1f}%), "
                    f"segments(field={n_field}, fallback={n_fallback})"
                )
            except Exception as exc:
                self.get_logger().error(f"[INSPECTOR] Planning error: {exc}")
            finally:
                self._planning_running = False

        threading.Thread(target=_worker, daemon=True).start()

    def _compare_in_region_rollout(
        self,
        region_ver: int,
        start_xy: np.ndarray,
        goal_xy: np.ndarray,
        max_steps: int,
        mode: str,
    ) -> Optional[np.ndarray]:
        region = self.regions.get(region_ver)
        if region is None:
            return None
        if mode == "speed":
            return self._speed_rollout_in_region(
                region_ver=region_ver,
                start_xy=start_xy,
                goal_xy=goal_xy,
                max_steps=max_steps,
            )
        if mode == "bidirectional":
            seg = region.generate_field_path(
                start_xy.astype(np.float32),
                goal_xy.astype(np.float32),
                step_size=self.step_size,
                max_steps=max_steps,
            )
        elif mode == "goal_to_start":
            seg = region.generate_field_path_oneway(
                goal_xy.astype(np.float32),
                start_xy.astype(np.float32),
                step_size=self.step_size,
                max_steps=max_steps,
            )
        else:
            seg = region.generate_field_path_oneway(
                start_xy.astype(np.float32),
                goal_xy.astype(np.float32),
                step_size=self.step_size,
                max_steps=max_steps,
            )
        if seg is None:
            return None
        seg_np = np.asarray(seg, dtype=np.float32)
        if len(seg_np) < 2:
            return None
        return seg_np

    def _speed_rollout_in_region(
        self, region_ver: int, start_xy: np.ndarray, goal_xy: np.ndarray, max_steps: int
    ) -> Optional[np.ndarray]:
        region = self.regions.get(region_ver)
        if region is None:
            return None
        cur = self._clip_to_region_bounds(region_ver, start_xy.astype(np.float32))
        goal = self._clip_to_region_bounds(region_ver, goal_xy.astype(np.float32))
        cur_free = self._nearest_free_xy(cur, region=region)
        goal_free = self._nearest_free_xy(goal, region=region)
        if cur_free is None or goal_free is None:
            return None
        cur = cur_free.astype(np.float32)
        goal = goal_free.astype(np.float32)
        pts = [cur.copy()]
        step = float(max(0.03, self.step_size))
        for _ in range(max(10, max_steps)):
            d = float(np.linalg.norm(goal - cur))
            if d <= max(0.12, step):
                pts.append(goal.copy())
                break

            base = math.atan2(float(goal[1] - cur[1]), float(goal[0] - cur[0]))
            candidate_angles = [0.0] + list(np.linspace(-math.pi, math.pi, 24, endpoint=False))
            cands = []
            for da in candidate_angles:
                cand = np.array(
                    [cur[0] + step * math.cos(base + da), cur[1] + step * math.sin(base + da)],
                    dtype=np.float32,
                )
                cand = self._clip_to_region_bounds(region_ver, cand)
                if np.linalg.norm(cand - cur) < 1e-4:
                    continue
                if self._point_in_collision(cand):
                    continue
                if self._segment_hits_wall(cur, cand):
                    continue
                cands.append(cand)
            if not cands:
                break
            cands_np = np.asarray(cands, dtype=np.float32)
            z0 = np.zeros((len(cands_np), 1), dtype=np.float32)
            z1 = np.zeros((len(cands_np), 1), dtype=np.float32)
            goal_rep = np.repeat(goal.reshape(1, 2), len(cands_np), axis=0).astype(np.float32)
            xp = np.concatenate([cands_np, z0, goal_rep, z1], axis=1)
            xp_t = torch.tensor(xp, dtype=torch.float32, device=region.device)
            xp_n = region.normalize_tensor(xp_t.clone())
            lock = getattr(region, "inference_lock", None)
            if lock is not None and not lock.acquire(timeout=0.15):
                break
            try:
                with torch.no_grad():
                    sp = region.function.Speed(xp_n).detach().cpu().numpy()
            except Exception:
                sp = None
            finally:
                if lock is not None:
                    lock.release()
            if sp is None:
                break
            if sp.ndim > 1:
                sp = sp[:, 0]
            sp = sp.reshape(-1).astype(np.float32)
            goal_d = np.linalg.norm(goal_rep - cands_np, axis=1).astype(np.float32)
            score = sp - 0.05 * goal_d
            best = int(np.argmax(score))
            nxt = cands_np[best]
            if np.linalg.norm(nxt - cur) < 1e-4:
                break
            pts.append(nxt.copy())
            cur = nxt
        if len(pts) < 2:
            return None
        out = np.asarray(pts, dtype=np.float32)
        if self._path_hits_wall(out):
            return None
        return out

    def _build_compare_path_along_route(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        region_route: List[int],
        mode: str,
        seg_budget: int,
    ) -> Optional[np.ndarray]:
        if len(region_route) == 0:
            return None
        if mode == "goal_to_start":
            cur = goal.copy()
            route = list(reversed(region_route))
            end_target = start
            inreg_mode = "oneway"
        else:
            cur = start.copy()
            route = list(region_route)
            end_target = goal
            inreg_mode = mode

        segments: List[np.ndarray] = []
        for i in range(len(route) - 1):
            u = route[i]
            v = route[i + 1]
            if not self._route_graph.has_edge(u, v):
                return None
            edge_uv = self._edge_path_oriented(u, v)
            exit_u, entry_v, connector_uv = self._extract_transition_on_edge(edge_uv, u, v)
            try:
                field_u = self._compare_in_region_rollout(
                    region_ver=u,
                    start_xy=cur,
                    goal_xy=exit_u,
                    max_steps=seg_budget,
                    mode=inreg_mode,
                )
            except Exception:
                field_u = None
            if field_u is None:
                field_u = self._astar_world(cur, exit_u)
                if field_u is None or len(field_u) < 2:
                    return None
                field_u = np.asarray(field_u, dtype=np.float32)
            segments.append(field_u)
            segments.append(np.asarray(connector_uv, dtype=np.float32))
            cur = entry_v.copy()

        last_region = route[-1]
        try:
            final_field = self._compare_in_region_rollout(
                region_ver=last_region,
                start_xy=cur,
                goal_xy=end_target,
                max_steps=seg_budget,
                mode=inreg_mode,
            )
        except Exception:
            final_field = None
        if final_field is None:
            final_field = self._astar_world(cur, end_target)
            if final_field is None or len(final_field) < 2:
                return None
            final_field = np.asarray(final_field, dtype=np.float32)
        segments.append(final_field)
        return self._concat_paths(segments)

    def _compute_rollout_compare_paths(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        s_ver: int,
        g_ver: int,
        region_route: List[int],
        seg_budget: int,
    ) -> None:
        if not self.compare_rollouts:
            return
        if not region_route:
            return

        _MODE_DESC = {
            "bidirectional": "generate_field_path(start,goal) — rollout from both ends, meet in middle",
            "oneway":        "generate_field_path_oneway(start,goal) — gradient descent start->goal only",
            "goal_to_start": "generate_field_path_oneway(goal,start) — gradient descent goal->start, reversed",
            "speed":         "speed-guided greedy rollout — pick highest-speed collision-free step",
        }

        route_str = " -> ".join(f"R{v}" for v in region_route)
        self.get_logger().info(
            f"[INSPECTOR] Comparing rollout methods along route [{route_str}]:"
        )

        for mode in ("bidirectional", "oneway", "goal_to_start", "speed"):
            try:
                seg = self._build_compare_path_along_route(
                    start=start,
                    goal=goal,
                    region_route=region_route,
                    mode=mode,
                    seg_budget=seg_budget,
                )
            except Exception as exc:
                self.get_logger().warn(
                    f"[INSPECTOR]   {mode}: FAILED ({type(exc).__name__})"
                )
                seg = None
            if seg is None:
                self.get_logger().info(f"[INSPECTOR]   {mode}: no path")
                continue
            seg_np = np.asarray(seg, dtype=np.float32)
            if len(seg_np) < 2:
                self.get_logger().info(f"[INSPECTOR]   {mode}: too few points")
                continue
            if mode == "oneway":
                self.path_oneway_xy = seg_np
            elif mode == "goal_to_start":
                self.path_goal_to_start_xy = seg_np
            elif mode == "speed":
                self.path_speed_xy = seg_np
            else:
                self.path_bidir_xy = seg_np
            collision = self._path_hits_wall(seg_np)
            self.get_logger().info(
                f"[INSPECTOR]   {mode}: {len(seg_np)}pts, {self._path_length(seg_np):.2f}m"
                f"{', COLLIDES' if collision else ''}"
            )
            self.get_logger().info(
                f"[INSPECTOR]     ({_MODE_DESC.get(mode, mode)})"
            )

    def _path_runs_in_region(self, path_xy: Optional[np.ndarray], region_ver: int) -> List[np.ndarray]:
        if path_xy is None or len(path_xy) < 2:
            return []
        inside = np.array([self._point_in_region(region_ver, p) for p in path_xy], dtype=bool)
        runs: List[np.ndarray] = []
        i = 0
        while i < len(path_xy):
            while i < len(path_xy) and not inside[i]:
                i += 1
            if i >= len(path_xy):
                break
            j = i
            while j < len(path_xy) and inside[j]:
                j += 1
            seg = path_xy[i:j]
            if len(seg) >= 2:
                runs.append(seg)
            i = j
        return runs

    def _contour_grid_for_region(
        self, region: Region, anchor_xy: np.ndarray, n: int
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        if getattr(region, "function", None) is None:
            return (
                np.zeros((0, 0), dtype=np.float32),
                np.zeros((0, 0), dtype=np.float32),
                None,
                None,
            )
        b = region.bound.detach().cpu().numpy()
        xmin, ymin = float(b[0, 0]), float(b[0, 1])
        xmax, ymax = float(b[1, 0]), float(b[1, 1])
        xs = np.linspace(xmin, xmax, n, dtype=np.float32)
        ys = np.linspace(ymin, ymax, n, dtype=np.float32)
        X_w, Y_w = np.meshgrid(xs, ys)

        sxy = np.array([[float(anchor_xy[0]), float(anchor_xy[1]), 0.0]], dtype=np.float32)
        src_n = region.normalize(sxy)[0]
        sx = (xmax - xmin) + 1e-9
        sy = (ymax - ymin) + 1e-9
        X_n = (X_w - xmin) / sx - 0.5
        Y_n = (Y_w - ymin) / sy - 0.5

        dim = int(getattr(region, "dim", getattr(region.model, "dim", 3)))
        XP = np.zeros((X_w.size, 2 * dim), dtype=np.float32)
        # Match region.save_debug_plot convention: x0 fixed (anchor), x1 sweeps grid.
        XP[:, :dim] = src_n
        XP[:, dim:] = src_n
        XP[:, dim + 0] = X_n.reshape(-1)
        XP[:, dim + 1] = Y_n.reshape(-1)
        if dim > 2:
            XP[:, dim + 2] = src_n[2]
        lock = getattr(region, "inference_lock", None)
        if lock is not None and not lock.acquire(timeout=0.3):
            return X_w, Y_w, None, None
        try:
            # TravelTimes is a plain forward pass — no_grad is fine.
            XP_tt = torch.tensor(XP, dtype=torch.float32, device=region.device)
            with torch.no_grad():
                TT = region.function.TravelTimes(XP_tt).detach().cpu().numpy().reshape(X_w.shape)
            # Speed = 1/||∇τ|| needs autograd, so requires_grad on input,
            # and must NOT be inside torch.no_grad().
            XP_v = torch.tensor(XP, dtype=torch.float32, device=region.device,
                                requires_grad=True)
            V = region.function.Speed(XP_v).detach().cpu().numpy().reshape(X_w.shape)
        except Exception as exc:
            import traceback
            self.get_logger().warn(
                f"[INSPECTOR] _contour_grid exception: {exc}\n{traceback.format_exc()}")
            TT = None
            V = None
        finally:
            if lock is not None:
                lock.release()
        return X_w, Y_w, TT, V

    def _gradient_grid_for_region(
        self, region: Region, goal_xy: np.ndarray, n: int, stride: int
    ):
        if region is None or getattr(region, "function", None) is None:
            return None
        if not hasattr(region, "_goal_slice_gradient_overlay"):
            return None
        b = region.bound.detach().cpu().numpy()
        xs = np.linspace(float(b[0, 0]), float(b[1, 0]), n, dtype=np.float32)
        ys = np.linspace(float(b[0, 1]), float(b[1, 1]), n, dtype=np.float32)
        X_w, Y_w = np.meshgrid(xs, ys)
        sample_xy = np.stack([X_w, Y_w], axis=-1)
        return region._goal_slice_gradient_overlay(goal_xy, sample_xy, stride=stride)

    def _save_path_overlay_figure(
        self, start: np.ndarray, goal: np.ndarray, region_route: List[int]
    ) -> None:
        if not self.save_path_overlays:
            return
        if not region_route:
            return
        out_dir = self.path_overlay_dir if self.path_overlay_dir else os.path.join(
            self.model_dir, "inspector_overlays"
        )
        os.makedirs(out_dir, exist_ok=True)

        paths = [
            ("Final", self.path_xy, "#6CFF5A", "-", 2.0),
            ("Bidirectional", self.path_bidir_xy, "#00DCFF", "--", 1.8),
            ("OneWay", self.path_oneway_xy, "#FF46AA", "--", 1.8),
            ("GoalToStart", self.path_goal_to_start_xy, "#FFFFFF", "-.", 1.8),
            ("SpeedGuided", self.path_speed_xy, "#FF8A00", ":", 2.0),
        ]

        nreg = len(region_route)
        ncols = min(3, nreg)
        nrows = int(math.ceil(float(nreg) / float(ncols)))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 5.5 * nrows))
        axes_arr = np.atleast_1d(axes).reshape(-1)

        for idx, reg_ver in enumerate(region_route):
            ax = axes_arr[idx]
            region = self.regions.get(reg_ver)
            if region is None:
                ax.set_title(f"R{reg_ver} (missing)")
                ax.axis("off")
                continue
            b = region.bound.detach().cpu().numpy()
            anchor = None
            for _lbl, _path, _c, _ls, _lw in paths:
                runs_anchor = self._path_runs_in_region(_path, reg_ver)
                if runs_anchor:
                    anchor = runs_anchor[0][0]
                    break
            if anchor is None:
                if self._point_in_region(reg_ver, start):
                    anchor = start
                elif self._point_in_region(reg_ver, goal):
                    anchor = goal
                else:
                    anchor = np.array(
                        [0.5 * (b[0, 0] + b[1, 0]), 0.5 * (b[0, 1] + b[1, 1])],
                        dtype=np.float32,
                    )
            X_w, Y_w, TT, V = self._contour_grid_for_region(
                region, anchor_xy=np.asarray(anchor, dtype=np.float32), n=max(50, self.path_overlay_grid_n)
            )
            if V is not None and np.isfinite(V).any():
                ax.pcolormesh(
                    X_w,
                    Y_w,
                    V,
                    vmin=0.0,
                    vmax=1.0,
                    cmap="viridis",
                    shading="auto",
                    alpha=0.45,
                )
            if TT is not None and np.isfinite(TT).any():
                finite = TT[np.isfinite(TT)]
                tt_min = float(np.min(finite))
                tt_max = float(np.max(finite))
                fixed_levels = np.arange(0.0, 3.0, 0.025, dtype=np.float32)
                if tt_max <= tt_min + 1e-6:
                    levels = np.array([tt_min], dtype=np.float32)
                elif fixed_levels[-1] < tt_min or fixed_levels[0] > tt_max:
                    # Fixed [0,3) levels can miss entirely for some anchor-conditioned slices.
                    levels = np.linspace(tt_min, tt_max, 28, dtype=np.float32)
                else:
                    levels = fixed_levels[(fixed_levels >= tt_min) & (fixed_levels <= tt_max)]
                    if len(levels) < 4:
                        levels = np.linspace(tt_min, tt_max, 28, dtype=np.float32)
                if len(levels) >= 2:
                    ax.contour(X_w, Y_w, TT, levels=levels, cmap="gist_heat", linewidths=0.6, alpha=0.85)
                self.get_logger().info(
                    f"[INSPECTOR] Overlay TT range R{reg_ver}: min={tt_min:.3f}, max={tt_max:.3f}, "
                    f"levels={len(levels)}"
                )
            else:
                self.get_logger().warn(
                    f"[INSPECTOR] Overlay TT invalid for R{reg_ver} "
                    f"(TT={'None' if TT is None else f'shape={TT.shape},nan={np.isnan(TT).sum()},inf={np.isinf(TT).sum()}'}, "
                    f"function={'None' if getattr(region, 'function', None) is None else 'exists'}, "
                    f"device={getattr(region, 'device', '?')})")

            grad_overlay = self._gradient_grid_for_region(
                region,
                goal_xy=np.asarray(goal, dtype=np.float32),
                n=max(40, self.path_overlay_grid_n),
                stride=max(4, self.path_overlay_grid_n // 18),
            )
            if grad_overlay is not None:
                qx, qy, qu, qv = grad_overlay
                ax.quiver(
                    qx, qy, qu, qv,
                    color="white",
                    alpha=0.45,
                    scale=22,
                    width=0.003,
                    headwidth=3.0,
                    headlength=4.0,
                    headaxislength=3.6,
                    zorder=4,
                )

            for label, path, color, style, lw in paths:
                runs = self._path_runs_in_region(path, reg_ver)
                for ridx, run in enumerate(runs):
                    ax.plot(
                        run[:, 0],
                        run[:, 1],
                        color=color,
                        linestyle=style,
                        linewidth=lw,
                        alpha=0.95,
                        label=(label if ridx == 0 else None),
                    )

            ax.scatter(float(start[0]), float(start[1]), c="#44FF44", s=36, marker="o", zorder=8, label="Start")
            ax.scatter(float(goal[0]), float(goal[1]), c="#FF3A2E", s=48, marker="*", zorder=8, label="Goal")
            ax.set_xlim(float(b[0, 0]), float(b[1, 0]))
            ax.set_ylim(float(b[0, 1]), float(b[1, 1]))
            ax.set_aspect("equal")
            # Build subtitle with segment method used in this region.
            seg_info = ""
            for seg in self._debug_segments:
                if seg.get("label", "") == f"R{reg_ver}":
                    seg_kind = seg.get("kind", "?")
                    seg_reason = seg.get("reason", "")
                    n_pts = len(seg.get("points", []))
                    seg_info = f" | {seg_kind} ({n_pts}pts)"
                    if seg_reason:
                        seg_info += f"\n{_short_reason(seg_reason, 50)}"
                    break
            role = ""
            if self._point_in_region(reg_ver, start):
                role += " [START]"
            if self._point_in_region(reg_ver, goal):
                role += " [GOAL]"
            ax.set_title(
                f"R{reg_ver}{role}{seg_info}",
                fontsize=9,
            )
            ax.grid(alpha=0.2, linewidth=0.4)

        for j in range(nreg, len(axes_arr)):
            axes_arr[j].axis("off")

        handles, labels = axes_arr[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper right", fontsize=8)
        fig.suptitle(
            f"Inspector Path Overlay | start=({start[0]:.2f},{start[1]:.2f}) "
            f"goal=({goal[0]:.2f},{goal[1]:.2f})",
            fontsize=11,
        )
        fig.tight_layout(rect=[0.0, 0.0, 0.96, 0.95])
        self._overlay_seq += 1
        stamp = self.get_clock().now().nanoseconds
        out_file = os.path.join(out_dir, f"path_overlay_{self._overlay_seq:04d}_{stamp}.png")
        fig.savefig(out_file, dpi=120, bbox_inches="tight")
        plt.close(fig)
        self.get_logger().info(f"[INSPECTOR] Saved path overlay image: {out_file}")


def main(args=None):
    rclpy.init(args=args)
    node = RegionGraphInspector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
