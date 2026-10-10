import json
import os
from datetime import datetime

import numpy as np
import yaml
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy
from rclpy.qos import qos_profile_sensor_data
from scipy.spatial.transform import Rotation
import tf2_ros

from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2, LaserScan

from mntfields.mntfields_ros.sensing.lidar_adapter import Lidar2DAdapter
from mntfields.mntfields_core.mapping.lidar_gridmap import LidarGridmap


class OfflineMapRecorderNode(Node):
    def __init__(self):
        super().__init__('offline_map_recorder')

        self.declare_parameter('output_root', '/home/mayank/turtlebot3_ws/mntfields_model/offline_datasets')
        self.declare_parameter('session_name', '')
        self.declare_parameter('map_size', 40.0)
        self.declare_parameter('n_cells', 400)
        self.declare_parameter('near_dist', 2.0)
        self.declare_parameter('max_range', 3.2)
        self.declare_parameter('save_every_n_ticks', 1)
        self.declare_parameter('flush_every', 200)
        self.declare_parameter('update_dt_sec', 0.2)

        root = str(self.get_parameter('output_root').value)
        session_name = str(self.get_parameter('session_name').value).strip()
        if not session_name:
            session_name = datetime.now().strftime('session_%Y%m%d_%H%M%S')
        self.out_dir = os.path.join(root, session_name)
        os.makedirs(self.out_dir, exist_ok=True)

        self.max_range = float(self.get_parameter('max_range').value)
        self.save_every_n_ticks = max(1, int(self.get_parameter('save_every_n_ticks').value))
        self.flush_every = max(20, int(self.get_parameter('flush_every').value))

        n_cells = int(self.get_parameter('n_cells').value)
        near_dist = float(self.get_parameter('near_dist').value)
        map_size = float(self.get_parameter('map_size').value)

        self.gridmap = LidarGridmap(
            # Match mapping node behavior: start traversability as unknown-free.
            # LidarGridmap.update() only marks visited/frontier where trav_map == 1.
            trav_map=np.ones((n_cells, n_cells), dtype=np.uint8),
            regions={},
            n_cells=n_cells,
            near_dist=near_dist,
            map_size=map_size,
            origin=(0.0, 0.0),
        )
        self.origin_initialized = False

        self.adapter = Lidar2DAdapter()
        self.robot_pos = None
        self.latest_scan = None
        self.latest_cloud = None

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.buffer = []
        self.chunk_idx = 0
        self.tick = 0

        self.create_subscription(LaserScan, '/scan', self.scan_cb, qos_profile_sensor_data)
        self.create_subscription(PointCloud2, '/cloud', self.cloud_cb, qos_profile_sensor_data)
        self.create_subscription(Odometry, '/odom', self.odom_cb, qos_profile_sensor_data)
        map_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.map_pub = self.create_publisher(OccupancyGrid, '/offline_map', map_qos)
        dt = float(self.get_parameter('update_dt_sec').value)
        self.create_timer(dt, self.update)

        meta = {
            'format_version': 1,
            'map_size': map_size,
            'n_cells': n_cells,
            'near_dist': near_dist,
            'max_range': self.max_range,
            'save_every_n_ticks': self.save_every_n_ticks,
            'flush_every': self.flush_every,
        }
        with open(os.path.join(self.out_dir, 'meta.json'), 'w') as f:
            json.dump(meta, f, indent=2)
        self.get_logger().info(f'[RECORDER] writing dataset to {self.out_dir}')

    def scan_cb(self, msg: LaserScan):
        self.latest_scan = msg

    def cloud_cb(self, msg: PointCloud2):
        self.latest_cloud = msg

    def odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        self.robot_pos = np.array([p.x, p.y, p.z], dtype=np.float32)
        if not self.origin_initialized:
            self.gridmap.origin = self.robot_pos[:2].copy()
            self.origin_initialized = True

    def _world_to_ij(self, x: float, y: float):
        c = self.gridmap.cell_size
        o = self.gridmap.origin
        h = self.gridmap.map_size / 2.0
        return (
            int((x - o[0] + h) / c),
            int((y - o[1] + h) / c),
        )

    def _flush_chunk(self):
        if not self.buffer:
            return

        n = len(self.buffer)
        steps = np.array([it['step'] for it in self.buffer], dtype=np.int64)
        stamp_ns = np.array([it['stamp_ns'] for it in self.buffer], dtype=np.int64)
        pose = np.stack([it['pose'] for it in self.buffer], axis=0).astype(np.float32)

        all_counts = np.array([len(it['all']) for it in self.buffer], dtype=np.int32)
        hit_counts = np.array([len(it['hit']) for it in self.buffer], dtype=np.int32)
        max_all = int(all_counts.max()) if n > 0 else 0
        max_hit = int(hit_counts.max()) if n > 0 else 0

        all_pts = np.full((n, max_all, 3), np.nan, dtype=np.float32)
        hit_pts = np.full((n, max_hit, 3), np.nan, dtype=np.float32)
        for i, it in enumerate(self.buffer):
            na = all_counts[i]
            nh = hit_counts[i]
            if na > 0:
                all_pts[i, :na] = it['all']
            if nh > 0:
                hit_pts[i, :nh] = it['hit']

        np.savez_compressed(
            os.path.join(self.out_dir, f'chunk_{self.chunk_idx:05d}.npz'),
            steps=steps,
            stamp_ns=stamp_ns,
            robot_pose=pose,
            world_points_all=all_pts,
            world_points_all_counts=all_counts,
            world_points_hit=hit_pts,
            world_points_hit_counts=hit_counts,
        )

        np.savez_compressed(
            os.path.join(self.out_dir, f'map_{self.chunk_idx:05d}.npz'),
            occupancy=self.gridmap.data.copy(),
            visited=self.gridmap.visited.copy(),
            frontier=self.gridmap.frontier.copy(),
            origin=np.asarray(self.gridmap.origin, dtype=np.float32),
            cell_size=np.float32(self.gridmap.cell_size),
            map_size=np.float32(self.gridmap.map_size),
        )

        self.get_logger().info(
            f'[RECORDER] flushed chunk {self.chunk_idx:05d} ({n} frames)'
        )
        self.chunk_idx += 1
        self.buffer = []

    @staticmethod
    def _bresenham(i0: int, j0: int, i1: int, j1: int):
        """Yield grid cells along line from (i0,j0) to (i1,j1), inclusive."""
        di = abs(i1 - i0)
        dj = abs(j1 - j0)
        si = 1 if i0 < i1 else -1
        sj = 1 if j0 < j1 else -1
        err = di - dj
        i, j = i0, j0
        while True:
            yield i, j
            if i == i1 and j == j1:
                break
            e2 = 2 * err
            if e2 > -dj:
                err -= dj
                i += si
            if e2 < di:
                err += di
                j += sj

    def _integrate_occupancy_from_hits(self, robot_xy: np.ndarray, hit_points_xy: np.ndarray):
        """Classic occupancy update: free along ray, occupied at endpoint."""
        ri, rj = self.gridmap._world_to_grid(float(robot_xy[0]), float(robot_xy[1]))
        n = self.gridmap.n_cells
        if not (0 <= ri < n and 0 <= rj < n):
            return
        self.gridmap.visited[ri, rj] = 1
        self.gridmap.data[ri, rj] = 0

        for x, y in hit_points_xy:
            ei, ej = self.gridmap._world_to_grid(float(x), float(y))
            cells = list(self._bresenham(ri, rj, ei, ej))
            if len(cells) <= 1:
                if 0 <= ei < n and 0 <= ej < n:
                    self.gridmap.data[ei, ej] = 1
                    self.gridmap.visited[ei, ej] = 1
                continue

            # Mark intermediate cells free/observed.
            for ci, cj in cells[:-1]:
                if 0 <= ci < n and 0 <= cj < n:
                    self.gridmap.visited[ci, cj] = 1
                    self.gridmap.data[ci, cj] = 0
            # Mark endpoint occupied (wall hit).
            li, lj = cells[-1]
            if 0 <= li < n and 0 <= lj < n:
                self.gridmap.visited[li, lj] = 1
                self.gridmap.data[li, lj] = 1

    def _publish_map(self):
        gm = self.gridmap
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'

        msg.info.resolution = float(gm.cell_size)
        msg.info.width = int(gm.n_cells)
        msg.info.height = int(gm.n_cells)
        msg.info.origin.position.x = float(gm.origin[0] - gm.map_size / 2.0)
        msg.info.origin.position.y = float(gm.origin[1] - gm.map_size / 2.0)
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0

        data = gm.data
        visited = gm.visited
        occ = np.full_like(data, -1, dtype=np.int16)
        occ[(data == 0) & (visited == 1)] = 0
        occ[data == 1] = 100
        # ROS OccupancyGrid expects row-major from (x=0,y=0) at map origin.
        # Gridmap storage already uses i->y (bottom to top), j->x, so no flip here.
        msg.data = occ.reshape(-1).astype(np.int8).tolist()
        self.map_pub.publish(msg)

    def _save_ros_map(self, filepath_base: str = None):
        if filepath_base is None:
            filepath_base = os.path.join(self.out_dir, 'ros_map', 'map')
        os.makedirs(os.path.dirname(filepath_base), exist_ok=True)

        gm = self.gridmap
        data = gm.data.copy()
        vis = gm.visited.copy()
        n_cells = gm.n_cells

        pgm = np.full((n_cells, n_cells), 205, dtype=np.uint8)
        pgm[(data == 0) & (vis == 1)] = 254
        pgm[data == 1] = 0
        pgm = np.flipud(pgm)

        pgm_path = filepath_base + '.pgm'
        yaml_path = filepath_base + '.yaml'
        with open(pgm_path, 'wb') as f:
            header = f'P5\n{n_cells} {n_cells}\n255\n'
            f.write(header.encode())
            f.write(pgm.tobytes())

        origin_x = float(gm.origin[0] - gm.map_size / 2.0)
        origin_y = float(gm.origin[1] - gm.map_size / 2.0)
        meta = {
            'image': os.path.basename(pgm_path),
            'resolution': float(gm.cell_size),
            'origin': [origin_x, origin_y, 0.0],
            'negate': 0,
            'occupied_thresh': 0.65,
            'free_thresh': 0.196,
        }
        with open(yaml_path, 'w') as f:
            yaml.safe_dump(meta, f, default_flow_style=False)

        self.get_logger().info(f'[RECORDER] saved map: {pgm_path}')

    def update(self):
        if self.robot_pos is None:
            return

        # Prefer /scan (same as mapping flow), fall back to /cloud if present.
        lidar_msg = self.latest_scan if self.latest_scan is not None else self.latest_cloud
        if lidar_msg is None:
            if self.origin_initialized:
                self._publish_map()
            return

        result = self.adapter.update(lidar_msg, self.robot_pos)
        if result is None:
            if self.origin_initialized:
                self._publish_map()
            return
        points_lidar, speeds = result
        if len(points_lidar) == 0:
            if self.origin_initialized:
                self._publish_map()
            return

        try:
            source_frame = lidar_msg.header.frame_id
            stamp = lidar_msg.header.stamp
            if not self.tf_buffer.can_transform('odom', source_frame, stamp, timeout=Duration(seconds=0.1)):
                # Fallback to latest available transform if exact-time lookup misses.
                tf = self.tf_buffer.lookup_transform('odom', source_frame, Time())
            else:
                tf = self.tf_buffer.lookup_transform('odom', source_frame, stamp)
        except Exception:
            if self.origin_initialized:
                self._publish_map()
            return

        t = tf.transform.translation
        q = tf.transform.rotation
        rot = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        world_points = (rot @ points_lidar.T).T + np.array([t.x, t.y, t.z], dtype=np.float32)

        # Keep points in map bounds only.
        half = self.gridmap.map_size / 2.0
        ox, oy = self.gridmap.origin
        inb = (
            (world_points[:, 0] > ox - half) & (world_points[:, 0] < ox + half) &
            (world_points[:, 1] > oy - half) & (world_points[:, 1] < oy + half)
        )
        world_points = world_points[inb]
        speeds = speeds[inb]
        if len(world_points) < 5:
            if self.origin_initialized:
                self._publish_map()
            return

        # Same angular nearest-hit reduction used by mapping path.
        robot_xy = self.robot_pos[:2]
        vecs = world_points[:, :2] - robot_xy[None, :]
        dists = np.linalg.norm(vecs, axis=1) + 1e-9
        angles = np.arctan2(vecs[:, 1], vecs[:, 0])
        bins = ((angles + np.pi) / (2 * np.pi) * 359).astype(np.int32)

        closest_idx = {}
        closest_dist = np.full(360, np.inf, dtype=np.float32)
        for k, b in enumerate(bins):
            if dists[k] < closest_dist[b]:
                closest_dist[b] = dists[k]
                closest_idx[b] = k

        keep = np.array(list(closest_idx.values()), dtype=np.int32)
        world_points_hit = world_points[keep]
        speeds_hit = speeds[keep]

        # Build standard occupancy semantics for RViz/map export:
        # free space along beams and occupied at hit endpoints.
        self._integrate_occupancy_from_hits(robot_xy, world_points_hit[:, :2])

        # Gridmap update.
        self.gridmap.update(
            lidar_scan=world_points_hit[:, :2],
            points=world_points_hit[:, :2],
            speeds=speeds_hit,
            cur_pos=robot_xy,
            region=None,
            regions={},
            update_room=False,
        )
        self._publish_map()

        self.tick += 1
        if (self.tick % self.save_every_n_ticks) == 0:
            self.buffer.append({
                'step': int(self.tick),
                'stamp_ns': int(self.get_clock().now().nanoseconds),
                'pose': self.robot_pos.copy().astype(np.float32),
                'all': world_points.astype(np.float32),
                'hit': world_points_hit.astype(np.float32),
            })
            if len(self.buffer) >= self.flush_every:
                self._flush_chunk()

    def destroy_node(self):
        self._flush_chunk()
        self._save_ros_map()
        return super().destroy_node()


def main():
    rclpy.init()
    node = OfflineMapRecorderNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
