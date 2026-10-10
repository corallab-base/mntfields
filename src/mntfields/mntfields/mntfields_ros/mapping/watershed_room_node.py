"""
watershed_room_node.py — Standalone watershed room segmentation test node.

Subscribes to /map (SLAM OccupancyGrid), runs watershed segmentation, and
publishes rich visualizations so you can verify room detection in RViz.

Designed to run with ONLY:
  - SLAM Toolbox (online_sync)
  - Keyboard teleop
  - This node

No mapping_node, nav_node, or Gazebo required beyond what SLAM Toolbox needs.

Published topics:
  /watershed/room_grid       nav_msgs/OccupancyGrid   — coloured room label map
  /watershed/distance_field  nav_msgs/OccupancyGrid   — clearance field (0–100 scaled)
  /watershed/rooms           visualization_msgs/MarkerArray — room outlines + labels
  /watershed/doorways        visualization_msgs/MarkerArray — doorway spheres
  /watershed/peaks           visualization_msgs/MarkerArray — room seed points

Logged to console:
  [WATERSHED] N rooms (K confirmed), M doorways
  per-room: id, area m², clearance m, confirmed
"""

import os
import time

import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy

from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA, Header
from geometry_msgs.msg import Point

os.environ.setdefault("MPLBACKEND", "Agg")

from mntfields.mntfields_core.mapping.watershed_segmenter import (
    WatershedSegmenter, WatershedRoom, WatershedDoorway,
)


# Distinct colours for up to 12 rooms (RGB 0–1)
_ROOM_COLOURS = [
    (0.27, 0.51, 0.71),  # blue
    (0.80, 0.36, 0.36),  # red
    (0.40, 0.76, 0.40),  # green
    (0.90, 0.65, 0.20),  # orange
    (0.60, 0.40, 0.80),  # purple
    (0.30, 0.75, 0.75),  # cyan
    (0.85, 0.52, 0.73),  # pink
    (0.55, 0.45, 0.35),  # brown
    (0.70, 0.87, 0.35),  # lime
    (0.35, 0.70, 0.90),  # sky
    (0.95, 0.85, 0.30),  # yellow
    (0.60, 0.60, 0.60),  # grey
]


def _room_colour(room_id: int) -> tuple[float, float, float]:
    return _ROOM_COLOURS[(room_id - 1) % len(_ROOM_COLOURS)]


class WatershedRoomNode(Node):

    def __init__(self):
        super().__init__("watershed_room_node")

        # ── Parameters ───────────────────────────────────────────────────────
        self.declare_parameter("min_peak_clearance_m",  0.40)
        self.declare_parameter("min_peak_separation_m", 2.0)
        self.declare_parameter("min_room_area_m2",      0.8)
        self.declare_parameter("door_min_m",            0.55)
        self.declare_parameter("door_max_m",            2.20)
        self.declare_parameter("update_interval_sec",   1.0)   # throttle heavy computation

        # ── Segmenter ────────────────────────────────────────────────────────
        self.segmenter = WatershedSegmenter(
            min_peak_clearance_m  = float(self.get_parameter("min_peak_clearance_m").value),
            min_peak_separation_m = float(self.get_parameter("min_peak_separation_m").value),
            min_room_area_m2      = float(self.get_parameter("min_room_area_m2").value),
            door_min_m            = float(self.get_parameter("door_min_m").value),
            door_max_m            = float(self.get_parameter("door_max_m").value),
        )

        self._last_update_t: float = 0.0
        self._update_interval: float = float(
            self.get_parameter("update_interval_sec").value)

        # Cached latest map message
        self._latest_map_msg = None

        # ── QoS for map subscription (SLAM Toolbox publishes TRANSIENT_LOCAL) ─
        map_qos = QoSProfile(
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
            depth=1,
        )

        # ── Subscribers ──────────────────────────────────────────────────────
        self.create_subscription(OccupancyGrid, "/map", self._map_cb, map_qos)

        # ── Publishers ───────────────────────────────────────────────────────
        self._room_grid_pub  = self.create_publisher(
            OccupancyGrid, "/watershed/room_grid",      map_qos)
        self._dist_grid_pub  = self.create_publisher(
            OccupancyGrid, "/watershed/distance_field", map_qos)
        self._room_marker_pub = self.create_publisher(
            MarkerArray, "/watershed/rooms",    10)
        self._door_marker_pub = self.create_publisher(
            MarkerArray, "/watershed/doorways", 10)
        self._peak_marker_pub = self.create_publisher(
            MarkerArray, "/watershed/peaks",    10)

        # ── Timer ─────────────────────────────────────────────────────────────
        self.create_timer(self._update_interval, self._update_tick)

        self.get_logger().info(
            "[WatershedRoomNode] ready — waiting for /map")

    # ──────────────────────────────────────────────────────────────────────────

    def _map_cb(self, msg: OccupancyGrid):
        self._latest_map_msg = msg

    def _update_tick(self):
        if self._latest_map_msg is None:
            return

        msg = self._latest_map_msg
        t0 = time.monotonic()

        # ── Decode OccupancyGrid ──────────────────────────────────────────────
        H = msg.info.height
        W = msg.info.width
        cell_m = float(msg.info.resolution)
        origin_xy = np.array([
            msg.info.origin.position.x,
            msg.info.origin.position.y,
        ], dtype=np.float32)

        slam_arr = np.array(msg.data, dtype=np.int8).reshape(H, W)

        # ── Frontier mask (free cells adjacent to unknown) ────────────────────
        arr16  = slam_arr.astype(np.int16)
        free   = (arr16 == 0)
        unk    = (arr16 < 0)
        front  = np.zeros((H, W), dtype=bool)
        front[:-1, :] |= free[:-1, :] & unk[1:,  :]
        front[1:,  :] |= free[1:,  :] & unk[:-1, :]
        front[:,  :-1] |= free[:, :-1] & unk[:,  1:]
        front[:,   1:] |= free[:,  1:] & unk[:,  :-1]

        # ── Run segmentation ──────────────────────────────────────────────────
        try:
            rooms, doorways = self.segmenter.update(
                slam_arr, cell_m, origin_xy, frontier_mask=front)
        except Exception as e:
            self.get_logger().error(f"[WATERSHED] Segmentation error: {e}")
            return

        elapsed_ms = (time.monotonic() - t0) * 1000.0

        # ── Log summary ───────────────────────────────────────────────────────
        n_conf = sum(1 for r in rooms if r.confirmed)
        self.get_logger().info(
            f"[WATERSHED] {len(rooms)} rooms ({n_conf} confirmed), "
            f"{len(doorways)} doorways  [{elapsed_ms:.0f} ms]"
        )
        for r in sorted(rooms, key=lambda x: x.id):
            status = "CONFIRMED" if r.confirmed else "unconfirmed"
            reason = self.segmenter._confirm_reasons.get(r.id, "")
            reason_str = f"  [{reason}]" if reason and not r.confirmed else ""
            self.get_logger().info(
                f"  room {r.id:2d}: {r.area_m2:.1f} m²  "
                f"clr={r.peak_clearance_m:.2f} m  "
                f"center=({r.centroid_xy[0]:.2f},{r.centroid_xy[1]:.2f})  "
                f"{status}{reason_str}"
            )

        # Door debug — log every update so we can see exactly what's happening
        for line in self.segmenter._door_debug_lines:
            self.get_logger().info(f"[DOOR_DBG] {line}")
        for line in self.segmenter._build_door_debug_lines:
            self.get_logger().info(f"[DOOR_DBG] {line}")

        # ── Publish visualizations ────────────────────────────────────────────
        stamp  = self.get_clock().now().to_msg()
        frame  = msg.header.frame_id

        self._publish_room_grid(rooms, slam_arr, H, W, cell_m, origin_xy,
                                msg, stamp, frame)
        self._publish_dist_grid(slam_arr, H, W, cell_m, origin_xy,
                                msg, stamp, frame)
        self._publish_room_markers(rooms, stamp, frame)
        self._publish_door_markers(doorways, stamp, frame)
        self._publish_peak_markers(rooms, stamp, frame)

    # ── Publish helpers ────────────────────────────────────────────────────────

    def _publish_room_grid(self, rooms, slam_arr, H, W, cell_m, origin_xy,
                           orig_msg, stamp, frame):
        """
        Publish a coloured OccupancyGrid where each room has a unique value.
        0 = wall/unknown, 1..N = room IDs scaled to 0–100.
        (RViz's colour map will differentiate them.)
        """
        grid = np.zeros((H, W), dtype=np.int8)

        room_map = self.segmenter.get_room_map()
        if room_map is None:
            return

        # Assign each room a value spaced across 1–99
        room_ids = sorted(self.segmenter.rooms.keys())
        n = max(len(room_ids), 1)
        for i, rid in enumerate(room_ids):
            val = int(1 + (i * 98) // n)
            # Find labels in room_map corresponding to this room
            room = self.segmenter.rooms[rid]
            peak_ij = self.segmenter._world_to_grid(
                room.centroid_xy[0], room.centroid_xy[1])
            pr = max(0, min(H - 1, peak_ij[0]))
            pc = max(0, min(W - 1, peak_ij[1]))
            lbl = int(room_map[pr, pc])
            if lbl > 0:
                grid[room_map == lbl] = val

        msg = OccupancyGrid()
        msg.header.stamp    = stamp
        msg.header.frame_id = frame
        msg.info            = orig_msg.info
        msg.data            = grid.flatten().tolist()
        self._room_grid_pub.publish(msg)

    def _publish_dist_grid(self, slam_arr, H, W, cell_m, origin_xy,
                           orig_msg, stamp, frame):
        """Publish the distance field scaled to 0–100."""
        dist = self.segmenter.get_distance_field()
        if dist is None:
            return
        max_d = float(dist.max()) if dist.max() > 0 else 1.0
        scaled = np.clip((dist / max_d) * 100.0, 0, 100).astype(np.int8)

        msg = OccupancyGrid()
        msg.header.stamp    = stamp
        msg.header.frame_id = frame
        msg.info            = orig_msg.info
        msg.data            = scaled.flatten().tolist()
        self._dist_grid_pub.publish(msg)

    def _publish_room_markers(self, rooms: list[WatershedRoom], stamp, frame):
        arr = MarkerArray()

        # Delete stale markers
        del_m = Marker()
        del_m.action = Marker.DELETEALL
        del_m.header.frame_id = frame
        arr.markers.append(del_m)

        for room in rooms:
            r, g, b = _room_colour(room.id)
            alpha   = 1.0 if room.confirmed else 0.4

            # Bounding box outline (LINE_STRIP)
            bb = room.bounding_box
            corners = [
                (bb[0, 0], bb[0, 1], 0.0),
                (bb[1, 0], bb[0, 1], 0.0),
                (bb[1, 0], bb[1, 1], 0.0),
                (bb[0, 0], bb[1, 1], 0.0),
                (bb[0, 0], bb[0, 1], 0.0),
            ]
            box_m = Marker()
            box_m.header.stamp    = stamp
            box_m.header.frame_id = frame
            box_m.ns   = "room_boxes"
            box_m.id   = room.id
            box_m.type = Marker.LINE_STRIP
            box_m.action = Marker.ADD
            box_m.scale.x = 0.05
            box_m.color = ColorRGBA(r=r, g=g, b=b, a=alpha)
            for x, y, z in corners:
                box_m.points.append(Point(x=float(x), y=float(y), z=float(z)))
            arr.markers.append(box_m)

            # Label text
            txt_m = Marker()
            txt_m.header.stamp    = stamp
            txt_m.header.frame_id = frame
            txt_m.ns   = "room_labels"
            txt_m.id   = room.id
            txt_m.type = Marker.TEXT_VIEW_FACING
            txt_m.action = Marker.ADD
            txt_m.pose.position.x = float(room.centroid_xy[0])
            txt_m.pose.position.y = float(room.centroid_xy[1])
            txt_m.pose.position.z = 0.5
            txt_m.pose.orientation.w = 1.0
            txt_m.scale.z = 0.25
            txt_m.color   = ColorRGBA(r=r, g=g, b=b, a=1.0)
            status = "✓" if room.confirmed else "?"
            txt_m.text = f"R{room.id}{status}\n{room.area_m2:.1f}m²"
            arr.markers.append(txt_m)

        self._room_marker_pub.publish(arr)

    def _publish_door_markers(self, doorways: list[WatershedDoorway], stamp, frame):
        arr = MarkerArray()
        del_m = Marker()
        del_m.action = Marker.DELETEALL
        del_m.header.frame_id = frame
        arr.markers.append(del_m)

        for door in doorways:
            m = Marker()
            m.header.stamp    = stamp
            m.header.frame_id = frame
            m.ns   = "doorways"
            m.id   = door.id
            m.type = Marker.CYLINDER
            m.action = Marker.ADD
            m.pose.position.x = float(door.position_xy[0])
            m.pose.position.y = float(door.position_xy[1])
            m.pose.position.z = 0.15
            m.pose.orientation.w = 1.0
            m.scale.x = float(door.width_m)
            m.scale.y = float(door.width_m)
            m.scale.z = 0.30
            m.color = ColorRGBA(r=1.0, g=0.85, b=0.0, a=0.85)
            arr.markers.append(m)

            # Width label
            lbl = Marker()
            lbl.header.stamp    = stamp
            lbl.header.frame_id = frame
            lbl.ns   = "door_labels"
            lbl.id   = door.id
            lbl.type = Marker.TEXT_VIEW_FACING
            lbl.action = Marker.ADD
            lbl.pose.position.x = float(door.position_xy[0])
            lbl.pose.position.y = float(door.position_xy[1])
            lbl.pose.position.z = 0.5
            lbl.pose.orientation.w = 1.0
            lbl.scale.z = 0.18
            lbl.color   = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            lbl.text    = f"D{door.id}\n{door.width_m:.2f}m\nR{door.room_ids[0]}↔R{door.room_ids[1]}"
            arr.markers.append(lbl)

        self._door_marker_pub.publish(arr)

    def _publish_peak_markers(self, rooms: list[WatershedRoom], stamp, frame):
        arr = MarkerArray()
        del_m = Marker()
        del_m.action = Marker.DELETEALL
        del_m.header.frame_id = frame
        arr.markers.append(del_m)

        for room in rooms:
            r, g, b = _room_colour(room.id)
            m = Marker()
            m.header.stamp    = stamp
            m.header.frame_id = frame
            m.ns   = "peaks"
            m.id   = room.id
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(room.centroid_xy[0])
            m.pose.position.y = float(room.centroid_xy[1])
            m.pose.position.z = 0.05
            m.pose.orientation.w = 1.0
            sz = float(room.peak_clearance_m) * 0.4
            m.scale.x = m.scale.y = m.scale.z = max(0.1, sz)
            m.color = ColorRGBA(r=r, g=g, b=b, a=0.9)
            arr.markers.append(m)

        self._peak_marker_pub.publish(arr)


# ── Entry point ───────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = WatershedRoomNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
