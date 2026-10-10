#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

from mntfields.mntfields_ros.mapping.offline_topology_from_map import build_arg_parser, run_once

DEFAULT_SWEEP = [
    {
        'name': 'rooms_k5_i1_min100',
        'room_seg_kernel_ref': 5,
        'room_seg_iter': 1,
        'room_min_cc_cells': 100,
    },
    {
        'name': 'rooms_k7_i1_min100',
        'room_seg_kernel_ref': 7,
        'room_seg_iter': 1,
        'room_min_cc_cells': 100,
    },
    {
        'name': 'rooms_k9_i1_min100',
        'room_seg_kernel_ref': 9,
        'room_seg_iter': 1,
        'room_min_cc_cells': 100,
        'room_large_boundary_frontiers': 120,
        'room_large_inside_frontiers': 80,
    },
    {
        'name': 'rooms_k9_i1_min250',
        'room_seg_kernel_ref': 9,
        'room_seg_iter': 1,
        'room_min_cc_cells': 250,
        'room_large_boundary_frontiers': 120,
        'room_large_inside_frontiers': 80,
    },
    {
        'name': 'rooms_k9_i1_min500',
        'room_seg_kernel_ref': 9,
        'room_seg_iter': 1,
        'room_min_cc_cells': 500,
        'room_large_boundary_frontiers': 120,
        'room_large_inside_frontiers': 80,
    },
    {
        'name': 'rooms_k11_i1_min250',
        'room_seg_kernel_ref': 11,
        'room_seg_iter': 1,
        'room_min_cc_cells': 250,
        'room_large_boundary_frontiers': 120,
        'room_large_inside_frontiers': 80,
    },
    {
        'name': 'rooms_k11_i1_min500',
        'room_seg_kernel_ref': 11,
        'room_seg_iter': 1,
        'room_min_cc_cells': 500,
        'room_large_boundary_frontiers': 120,
        'room_large_inside_frontiers': 80,
    },
    {
        'name': 'rooms_k13_i1_min500',
        'room_seg_kernel_ref': 13,
        'room_seg_iter': 1,
        'room_min_cc_cells': 500,
        'room_large_boundary_frontiers': 120,
        'room_large_inside_frontiers': 80,
    },
]



def sanitize(name: str) -> str:
    return ''.join(ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in name)


def main():
    ap = argparse.ArgumentParser(description='Run a batch of offline topology configurations and store each result in one workspace folder.')
    ap.add_argument('--map-yaml', default='/home/mayank/turtlebot3_ws/ros_map/map.yaml')
    ap.add_argument('--out-root', default='/home/mayank/turtlebot3_ws/offline_testing/topology_sweeps')
    args = ap.parse_args()

    root = Path(args.out_root)
    root.mkdir(parents=True, exist_ok=True)

    base_parser = build_arg_parser()
    defaults = vars(base_parser.parse_args([]))
    manifest = []

    for idx, cfg in enumerate(DEFAULT_SWEEP, start=1):
        run_name = f'{idx:02d}_{sanitize(cfg["name"])}'
        run_dir = root / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        params = defaults.copy()
        params.update(cfg)
        params['map_yaml'] = args.map_yaml
        params['out_dir'] = str(run_dir)
        ns = SimpleNamespace(**params)

        print(f'\n=== {run_name} ===')
        run_once(ns)

        summary_path = run_dir / 'summary.json'
        if summary_path.exists():
            with open(summary_path, 'r', encoding='utf-8') as fh:
                summary = json.load(fh)
            manifest.append({
                'name': run_name,
                'dir': str(run_dir),
                'counts': summary.get('counts', {}),
                'tuning': summary.get('tuning', {}),
                'overlay_png': summary.get('overlay_png'),
            })

    manifest_path = root / 'manifest.json'
    with open(manifest_path, 'w', encoding='utf-8') as fh:
        json.dump(manifest, fh, indent=2)

    print(f'\nmanifest={manifest_path}')


if __name__ == '__main__':
    main()
