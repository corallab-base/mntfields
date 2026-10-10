"""
semantic_region_store.py — Region-level semantic object storage + change monitor.

Helper module that runs on the GridmapNode executor (same pattern as the
old SemanticMapperNode).  Subscribes to ``/semantic/tracked_objects`` published
by the standalone SemanticDetectorNode, assigns confirmed objects to regions,
and monitors for semantic changes during revisits.
"""

import json
import time

import numpy as np

import rclpy
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

from mntfields.utils.mntfields_utils import owner_region_for_xy
from mntfields.mntfields_core.core.region import is_semantic_entry_confirmed


class SemanticRegionStore:
    """Assigns tracked semantic objects to regions and detects layout changes.

    Instantiated inside TB4GridmapNode.__init__ (or GridmapNode) — not a
    standalone ROS node.  All subscriptions / timers are created on the
    host node's executor.
    """

    def __init__(
        self,
        gridmap_node,
        observation_radius_m: float = 3.5,
        miss_passes_to_stale: int = 5,
        moved_threshold_m: float = 0.8,
        change_check_interval_s: float = 10.0,
    ):
        self._node = gridmap_node
        self._obs_radius = observation_radius_m
        self._miss_threshold = miss_passes_to_stale
        self._moved_thr = moved_threshold_m

        # Latest tracked objects from the detector node
        self._latest_tracked: list[dict] = []

        # Per-object miss counters: (region_version, class_name, xy_key) → int
        self._miss_counts: dict[tuple, int] = {}

        # ── Subscriptions (on host node) ─────────────────────────────────
        gridmap_node.create_subscription(
            String, "/semantic/tracked_objects",
            self._tracked_cb, 10)

        # ── Publishers ───────────────────────────────────────────────────
        self._change_pub = gridmap_node.create_publisher(
            String, "/semantic/changes", 10)
        self._marker_pub = gridmap_node.create_publisher(
            MarkerArray, "/semantic/region_objects", 10)

        # ── Change-check timer ───────────────────────────────────────────
        gridmap_node.create_timer(
            change_check_interval_s, self._change_check_tick)

        gridmap_node.get_logger().info(
            "[SEMANTIC STORE] Initialised — listening on /semantic/tracked_objects")

    # ==================================================================
    # Tracked-objects callback
    # ==================================================================

    def _tracked_cb(self, msg: String):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        objects = data.get("objects", [])
        self._latest_tracked = objects

        for obj in objects:
            if obj.get("status") != "confirmed":
                continue
            self._assign_to_region(
                obj["class_name"],
                np.array(obj["map_xy"], dtype=np.float32),
            )

    # ------------------------------------------------------------------
    def _assign_to_region(self, class_name: str, map_xy: np.ndarray):
        """Find the owning region for *map_xy* and record the object."""
        node = self._node
        region = None

        if (hasattr(node, "_ownership_map")
                and node._ownership_map is not None
                and hasattr(node, "all_regions")):
            region = owner_region_for_xy(
                map_xy,
                node._ownership_map,
                node.all_regions,
                node._slam_origin,
                node._slam_cell_size,
            )

        if region is None and hasattr(node, "region"):
            region = node.region  # fallback: active region

        if region is not None:
            region.add_semantic_object(class_name, map_xy.tolist())

    # ==================================================================
    # Change-detection tick
    # ==================================================================

    def _change_check_tick(self):
        node = self._node
        if node.robot_pos is None:
            return
        if not hasattr(node, "all_regions"):
            return

        robot_xy = node.robot_pos[:2]
        events: list[dict] = []

        # Collect stored objects near the robot
        for rv, reg in node.all_regions.items():
            stored = getattr(reg, "semantic_objects", {})
            if not stored:
                continue

            for cls, positions in stored.items():
                for pos in positions:
                    pos_arr = self._semantic_entry_xy(pos)
                    if pos_arr is None:
                        continue
                    if np.linalg.norm(pos_arr - robot_xy) > self._obs_radius:
                        continue
                    # Confirmation gate: tentative landmarks don't drive
                    # miss-counting or "moved" events — they may be a
                    # one-frame misclassification that will disappear on
                    # its own once better observations accumulate.
                    if not is_semantic_entry_confirmed(pos):
                        continue

                    # Look for a matching tracked object
                    matched = self._find_matching_tracked(cls, pos_arr)

                    key = (rv, cls, tuple(np.round(pos_arr, 2)))

                    if matched is not None:
                        # Check if moved significantly
                        dist = np.linalg.norm(
                            np.array(matched["map_xy"]) - pos_arr)
                        if dist > self._moved_thr:
                            events.append({
                                "type": "object_moved",
                                "class_name": cls,
                                "stored_xy": pos_arr.tolist(),
                                "observed_xy": matched["map_xy"],
                                "distance_m": round(float(dist), 3),
                                "region_version": rv,
                            })
                        self._miss_counts.pop(key, None)
                    else:
                        cnt = self._miss_counts.get(key, 0) + 1
                        self._miss_counts[key] = cnt
                        if cnt >= self._miss_threshold:
                            events.append({
                                "type": "object_missing",
                                "class_name": cls,
                                "stored_xy": pos_arr.tolist(),
                                "miss_count": cnt,
                                "region_version": rv,
                            })

        # Check for new objects not in any region's store
        for obj in self._latest_tracked:
            if obj.get("status") != "confirmed":
                continue
            obj_xy = np.array(obj["map_xy"], dtype=np.float32)
            if np.linalg.norm(obj_xy - robot_xy) > self._obs_radius:
                continue
            if not self._is_stored_anywhere(
                    obj["class_name"], obj_xy, node.all_regions):
                events.append({
                    "type": "new_object",
                    "class_name": obj["class_name"],
                    "observed_xy": obj["map_xy"],
                    "track_id": obj.get("track_id"),
                })

        if events:
            self._publish_changes(events)

    # ------------------------------------------------------------------
    def _find_matching_tracked(
        self, class_name: str, stored_xy: np.ndarray
    ) -> dict | None:
        """Find a confirmed tracked object matching class + position."""
        for obj in self._latest_tracked:
            if obj.get("status") != "confirmed":
                continue
            if obj["class_name"] != class_name:
                continue
            d = np.linalg.norm(
                np.array(obj["map_xy"], dtype=np.float32) - stored_xy)
            if d < self._moved_thr:
                return obj
        return None

    def _is_stored_anywhere(
        self, class_name: str, xy: np.ndarray, all_regions: dict
    ) -> bool:
        """Check if an object of *class_name* near *xy* is in any region."""
        for reg in all_regions.values():
            stored = getattr(reg, "semantic_objects", {})
            for pos in stored.get(class_name, []):
                pos_arr = self._semantic_entry_xy(pos)
                if pos_arr is not None and np.linalg.norm(xy - pos_arr) < 0.6:
                    return True
        return False

    @staticmethod
    def _semantic_entry_xy(entry) -> np.ndarray | None:
        try:
            if isinstance(entry, dict):
                raw = entry.get("xy", entry.get("world_xy", entry.get("pos")))
            else:
                raw = entry
            xy = np.asarray(raw[:2], dtype=np.float32)
            if xy.shape[0] < 2 or not np.all(np.isfinite(xy[:2])):
                return None
            return xy[:2]
        except Exception:
            return None

    # ------------------------------------------------------------------
    def _publish_changes(self, events: list[dict]):
        now = self._node.get_clock().now().to_msg()
        msg = String()
        msg.data = json.dumps({
            "stamp_sec": now.sec,
            "stamp_nanosec": now.nanosec,
            "events": events,
        })
        self._change_pub.publish(msg)

        for ev in events:
            self._node.get_logger().info(
                f"[SEMANTIC CHANGE] {ev['type']}: {ev['class_name']} "
                f"region={ev.get('region_version', '?')}")
