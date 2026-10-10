import csv
import json
import os
from itertools import product

import numpy as np
import rclpy
from rclpy.node import Node

from mntfields.mntfields_core.mapping.lidar_gridmap import LidarGridmap


def _parse_int_list(s: str):
    return [int(x.strip()) for x in s.split(',') if x.strip()]


def _parse_float_list(s: str):
    return [float(x.strip()) for x in s.split(',') if x.strip()]


class OfflineSweepNode(Node):
    def __init__(self):
        super().__init__('offline_sweep_node')

        self.declare_parameter('dataset_dir', '/home/mayank/turtlebot3_ws/mntfields_model/offline_datasets')
        self.declare_parameter('room_seg_kernel_values', '5,7,9')
        self.declare_parameter('room_seg_iter_values', '1,2,3')
        self.declare_parameter('door_match_radius_cells_values', '4,6,8')
        self.declare_parameter('update_room_every_n', 10)

        self.dataset_dir = str(self.get_parameter('dataset_dir').value)
        self.room_seg_kernel_values = _parse_int_list(str(self.get_parameter('room_seg_kernel_values').value))
        self.room_seg_iter_values = _parse_int_list(str(self.get_parameter('room_seg_iter_values').value))
        self.door_match_radius_cells_values = _parse_float_list(
            str(self.get_parameter('door_match_radius_cells_values').value)
        )
        self.update_room_every_n = max(1, int(self.get_parameter('update_room_every_n').value))

        if not os.path.isdir(self.dataset_dir):
            raise RuntimeError(f'dataset_dir not found: {self.dataset_dir}')

        self.meta = self._load_meta()
        self.samples = self._load_samples()

        if len(self.samples) == 0:
            raise RuntimeError('No samples found in dataset chunks')

        self.results_dir = os.path.join(self.dataset_dir, 'sweep_results')
        os.makedirs(self.results_dir, exist_ok=True)

        self.get_logger().info(
            f"[SWEEP] loaded {len(self.samples)} samples from {self.dataset_dir}"
        )
        self.run_sweep()
        self.get_logger().info('[SWEEP] done')

    def _load_meta(self):
        meta_path = os.path.join(self.dataset_dir, 'meta.json')
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                return json.load(f)
        return {
            'map_size': 40.0,
            'n_cells': 400,
            'near_dist': 2.0,
        }

    def _load_samples(self):
        chunks = sorted([
            p for p in os.listdir(self.dataset_dir)
            if p.startswith('chunk_') and p.endswith('.npz')
        ])
        samples = []
        for c in chunks:
            arr = np.load(os.path.join(self.dataset_dir, c), allow_pickle=False)
            poses = arr['robot_pose']
            hit = arr['world_points_hit']
            hit_counts = arr['world_points_hit_counts']
            for i in range(len(poses)):
                n = int(hit_counts[i])
                pts = hit[i, :n, :]
                samples.append((poses[i], pts))
        return samples

    def _run_config(self, k: int, iters: int, door_cells: float):
        n_cells = int(self.meta.get('n_cells', 400))
        near_dist = float(self.meta.get('near_dist', 2.0))
        map_size = float(self.meta.get('map_size', 40.0))

        gridmap = LidarGridmap(
            # Match online mapping behavior; update() marks visited/frontier
            # only on traversable cells.
            trav_map=np.ones((n_cells, n_cells), dtype=np.uint8),
            regions={},
            n_cells=n_cells,
            near_dist=near_dist,
            room_seg_config=(k, iters),
            map_size=map_size,
            origin=(0.0, 0.0),
        )
        gridmap.set_door_match_radius(radius_m=None, radius_cells=door_cells)

        if len(self.samples) > 0:
            gridmap.origin = np.asarray(self.samples[0][0][:2], dtype=np.float32)

        for idx, (pose, pts) in enumerate(self.samples):
            if pts is None or len(pts) == 0:
                continue
            speeds = np.ones(len(pts), dtype=np.float32)
            do_room = ((idx % self.update_room_every_n) == 0)
            gridmap.update(
                lidar_scan=pts[:, :2],
                points=pts[:, :2],
                speeds=speeds,
                cur_pos=np.asarray(pose[:2], dtype=np.float32),
                region=None,
                regions={},
                update_room=do_room,
            )

        occ = gridmap.data
        vis = gridmap.visited
        frontier = gridmap.frontier

        explored_ratio = float(vis.sum()) / float(vis.size + 1e-9)
        occupied_ratio = float((occ == 1).sum()) / float(occ.size + 1e-9)
        frontier_count = int((frontier > 0).sum())
        doors = int(len(gridmap.doors))
        confirmed_rooms = int(len(gridmap.confirmed_rooms))
        room_graph_edges = int(gridmap.room_graph.number_of_edges()) if hasattr(gridmap, 'room_graph') else 0

        score = explored_ratio - 0.0005 * frontier_count + 0.02 * confirmed_rooms + 0.005 * doors

        return {
            'room_seg_kernel': k,
            'room_seg_iters': iters,
            'door_match_radius_cells': door_cells,
            'explored_ratio': explored_ratio,
            'occupied_ratio': occupied_ratio,
            'frontier_count': frontier_count,
            'doors': doors,
            'confirmed_rooms': confirmed_rooms,
            'room_graph_edges': room_graph_edges,
            'score': score,
        }

    def run_sweep(self):
        configs = list(product(
            self.room_seg_kernel_values,
            self.room_seg_iter_values,
            self.door_match_radius_cells_values,
        ))

        results = []
        for i, (k, iters, dcell) in enumerate(configs, start=1):
            self.get_logger().info(
                f"[SWEEP] {i}/{len(configs)} k={k} iters={iters} door_cells={dcell}"
            )
            try:
                results.append(self._run_config(k, iters, dcell))
            except Exception as e:
                self.get_logger().warn(f"[SWEEP] config failed: k={k}, iters={iters}, d={dcell} | {e}")

        if not results:
            raise RuntimeError('All sweep configs failed')

        results = sorted(results, key=lambda r: r['score'], reverse=True)

        out_json = os.path.join(self.results_dir, 'results.json')
        with open(out_json, 'w') as f:
            json.dump(results, f, indent=2)

        out_csv = os.path.join(self.results_dir, 'results.csv')
        with open(out_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            for r in results:
                writer.writerow(r)

        best = results[0]
        with open(os.path.join(self.results_dir, 'best_config.json'), 'w') as f:
            json.dump(best, f, indent=2)

        self.get_logger().info(
            '[SWEEP] best config: '
            f"k={best['room_seg_kernel']} iters={best['room_seg_iters']} "
            f"door_cells={best['door_match_radius_cells']} score={best['score']:.4f}"
        )


def main():
    rclpy.init()
    node = None
    try:
        node = OfflineSweepNode()
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
