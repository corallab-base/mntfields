from nav_msgs.msg import OccupancyGrid
import numpy as np


class MapPublisher:
    def __init__(self, node):
        self.node = node

    def publish_maps(self):
        now = self.node.get_clock().now()
        if not hasattr(self, '_last_publish_log'):
            self._last_publish_log = now

        self.node.occ_pub.publish(self.make_grid_msg(self.node.gridmap.data))
        self.node.frontier_pub.publish(self.make_frontier_msg())

    # ── helpers ──────────────────────────────────────────────────────────────

    def _slam_header(self, msg, gm):
        """Fill msg header/info using the raw SLAM map's coordinate frame."""
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.info.resolution = float(gm.slam_raw_resolution)
        slam_h, slam_w = gm.slam_raw_shape
        msg.info.width = int(slam_w)
        msg.info.height = int(slam_h)
        msg.info.origin.position.x = float(gm.slam_raw_origin[0])
        msg.info.origin.position.y = float(gm.slam_raw_origin[1])
        msg.info.origin.orientation.w = 1.0
        return slam_h, slam_w

    def _local_header(self, msg, gm):
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.info.resolution = float(gm.cell_size)
        msg.info.width = int(gm.n_cells)
        msg.info.height = int(gm.n_cells)
        msg.info.origin.position.x = float(gm.origin[0] - gm.map_size / 2)
        msg.info.origin.position.y = float(gm.origin[1] - gm.map_size / 2)
        msg.info.origin.orientation.w = 1.0

    # ── occ_map ───────────────────────────────────────────────────────────────

    def make_grid_msg(self, grid):
        """Publish occupancy in SLAM frame (free/occupied/unknown from /map)."""
        msg = OccupancyGrid()
        gm = self.node.gridmap

        if gm.has_slam_semantics() and gm.slam_raw_data is not None:
            slam_h, slam_w = self._slam_header(msg, gm)
            # Re-encode raw SLAM data: -1→-1, 0→0, >50→100, else→-1
            raw = gm.slam_raw_data  # (slam_h, slam_w) int8
            flat = np.where(raw == 0, np.int8(0),
                   np.where(raw > 50, np.int8(100), np.int8(-1))).flatten()
        else:
            # Before SLAM arrives: LiDAR-based fallback in local frame
            self._local_header(msg, gm)
            flat = np.full(gm.n_cells * gm.n_cells, -1, dtype=np.int8)
            occ = grid.flatten() > 0
            visited = gm.visited.flatten() > 0
            flat[visited & ~occ] = 0
            flat[occ] = 100

        msg.data = flat.tolist()
        return msg

    # ── frontier map ──────────────────────────────────────────────────────────

    def make_frontier_msg(self):
        """Publish frontiers in SLAM frame so they overlay /map with no local-grid border."""
        msg = OccupancyGrid()
        gm = self.node.gridmap

        if gm.has_slam_semantics() and gm.slam_raw_data is not None:
            slam_h, slam_w = self._slam_header(msg, gm)
            # Start with the SLAM occupancy values so unknown/free/occupied
            # match the /map display — no extra grey borders.
            raw = gm.slam_raw_data
            flat = np.where(raw == 0, np.int8(0),
                   np.where(raw > 50, np.int8(100), np.int8(-1))).flatten()

            # Stamp frontier cells (SLAM-resolution) as occupied so they
            # stand out dark against the white free background.
            if gm.has_raw_slam_frontiers() and len(gm.slam_frontier_points) > 0:
                pts = gm.slam_frontier_points          # Nx2 world coords
                res = gm.slam_raw_resolution
                ox, oy = gm.slam_raw_origin
                sc = ((pts[:, 0] - ox) / res).astype(np.int32)
                sr = ((pts[:, 1] - oy) / res).astype(np.int32)
                valid = (sc >= 0) & (sc < slam_w) & (sr >= 0) & (sr < slam_h)
                flat[sr[valid] * slam_w + sc[valid]] = 100
        else:
            # Fallback: local frame, only frontier cells marked
            self._local_header(msg, gm)
            flat = np.full(gm.n_cells * gm.n_cells, -1, dtype=np.int8)
            flat[gm.frontier.flatten() > 0] = 100

        msg.data = flat.tolist()
        return msg
