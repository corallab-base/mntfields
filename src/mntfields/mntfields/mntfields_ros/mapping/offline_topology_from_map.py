#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage
from sklearn.cluster import DBSCAN
import yaml

from mntfields.mntfields_ros.mapping.topology_extractor import TopologyExtractor


ROOM_PALETTE = [
    (60, 180, 255),
    (255, 180, 60),
    (120, 220, 120),
    (220, 120, 220),
    (255, 110, 110),
    (110, 220, 220),
    (180, 140, 255),
    (255, 215, 120),
]


def load_ros_map(map_yaml_path: str):
    map_yaml_path = os.path.abspath(map_yaml_path)
    with open(map_yaml_path, 'r', encoding='utf-8') as fh:
        meta = yaml.safe_load(fh)

    image_path = meta['image']
    if not os.path.isabs(image_path):
        image_path = os.path.join(os.path.dirname(map_yaml_path), image_path)

    img = Image.open(image_path).convert('L')
    arr = np.asarray(img, dtype=np.float32) / 255.0

    negate = int(meta.get('negate', 0))
    occ_prob = arr if negate else (1.0 - arr)
    occ_thresh = float(meta.get('occupied_thresh', 0.65))
    free_thresh = float(meta.get('free_thresh', 0.196))

    slam = np.full(arr.shape, -1, dtype=np.int16)
    slam[occ_prob >= occ_thresh] = 100
    slam[occ_prob <= free_thresh] = 0

    resolution = float(meta['resolution'])
    origin = np.asarray(meta.get('origin', [0.0, 0.0, 0.0])[:2], dtype=np.float32)
    return slam, resolution, origin, image_path


def apply_tuning(args):
    if args.room_seg_ref_cell_m is not None:
        TopologyExtractor._ROOM_SEG_REF_CELL_M = float(args.room_seg_ref_cell_m)
    if args.room_seg_kernel_ref is not None:
        TopologyExtractor._ROOM_SEG_KERNEL_REF = int(args.room_seg_kernel_ref)
    if args.room_seg_iter is not None:
        TopologyExtractor._ROOM_SEG_ITER = int(args.room_seg_iter)
    if args.room_min_cc_cells is not None:
        TopologyExtractor._ROOM_MIN_CC_CELLS = int(args.room_min_cc_cells)
    if args.room_small_area_ref_cells is not None:
        TopologyExtractor._ROOM_SMALL_AREA_REF_CELLS = int(args.room_small_area_ref_cells)
    if args.room_small_boundary_frontiers is not None:
        TopologyExtractor._ROOM_SMALL_BOUNDARY_FRONTIERS = int(args.room_small_boundary_frontiers)
    if args.room_large_boundary_frontiers is not None:
        TopologyExtractor._ROOM_LARGE_BOUNDARY_FRONTIERS = int(args.room_large_boundary_frontiers)
    if args.room_large_inside_frontiers is not None:
        TopologyExtractor._ROOM_LARGE_INSIDE_FRONTIERS = int(args.room_large_inside_frontiers)
    if args.room_outer_opening_ref_cells is not None:
        TopologyExtractor._ROOM_OUTER_OPENING_REF_CELLS = int(args.room_outer_opening_ref_cells)
    if args.door_width_min_m is not None:
        TopologyExtractor._DOOR_WIDTH_MIN_M = float(args.door_width_min_m)
    if args.door_width_max_m is not None:
        TopologyExtractor._DOOR_WIDTH_MAX_M = float(args.door_width_max_m)
    if args.door_flank_min_m is not None:
        TopologyExtractor._DOOR_FLANK_MIN_M = float(args.door_flank_min_m)
    if args.door_nms_radius_m is not None:
        TopologyExtractor._DOOR_NMS_RADIUS_M = float(args.door_nms_radius_m)
    if args.frontier_occ_clear_dilate is not None:
        TopologyExtractor._FRONTIER_OCC_CLEAR_DILATE = int(args.frontier_occ_clear_dilate)
    if args.dbscan_eps is not None:
        TopologyExtractor._DBSCAN_EPS = int(args.dbscan_eps)
    if args.dbscan_min_samples is not None:
        TopologyExtractor._DBSCAN_MIN_SAMPLES = int(args.dbscan_min_samples)


def _blend_mask(img: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float) -> None:
    if not np.any(mask):
        return
    color_arr = np.asarray(color, dtype=np.float32)
    base = img[mask].astype(np.float32)
    img[mask] = np.clip((1.0 - alpha) * base + alpha * color_arr, 0, 255).astype(np.uint8)


def _room_color(idx: int) -> tuple[int, int, int]:
    return ROOM_PALETTE[idx % len(ROOM_PALETTE)]


def _draw_label(img: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int], scale: float = 0.5) -> None:
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def _base_map_rgb(slam: np.ndarray) -> np.ndarray:
    free = (slam == 0)
    occ = (slam > 50)
    unknown = (slam < 0)
    img = np.zeros((slam.shape[0], slam.shape[1], 3), dtype=np.uint8)
    img[unknown] = [185, 185, 185]
    img[free] = [250, 250, 250]
    img[occ] = [20, 20, 20]
    return img


def draw_overlay(slam: np.ndarray, extractor: TopologyExtractor, out_png: str):
    img = _base_map_rgb(slam)

    for idx, room in enumerate(sorted(extractor.confirmed_rooms.values(), key=lambda r: int(r.id))):
        color = _room_color(idx)
        _blend_mask(img, room.mask, color, 0.35)
        mask_u8 = room.mask.astype(np.uint8)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        outline = tuple(max(0, int(c * 0.45)) for c in color)
        cv2.drawContours(img, contours, -1, outline, 2)
        rows, cols = np.where(room.mask)
        if len(rows) > 0:
            x0, y0 = int(cols.min()), int(rows.min())
            x1, y1 = int(cols.max()), int(rows.max())
            cv2.rectangle(img, (x0, y0), (x1, y1), outline, 2)
            ci, cj = extractor.world_to_grid(float(room.centroid_xy[0]), float(room.centroid_xy[1]))
            cv2.circle(img, (cj, ci), 5, outline, -1)
            _draw_label(img, f'R{room.id}', cj + 8, ci - 8, outline)

    for room in sorted(extractor.unconfirmed_rooms.values(), key=lambda r: int(r.id)):
        color = (255, 170, 60)
        _blend_mask(img, room.mask, color, 0.35)
        contours, _ = cv2.findContours(room.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, (180, 90, 0), 2)
        ci, cj = extractor.world_to_grid(float(room.centroid_xy[0]), float(room.centroid_xy[1]))
        cv2.circle(img, (cj, ci), 5, (180, 90, 0), -1)
        _draw_label(img, f'U{room.id}', cj + 8, ci - 8, (180, 90, 0))

    if extractor._frontier_cells is not None:
        img[extractor._frontier_cells] = [0, 200, 255]

    for dbg in getattr(extractor, '_door_debug', []):
        r0, c0, r1, c1 = dbg.get('bbox_rc', [0, 0, 0, 0])
        ci, cj = dbg.get('grid_center_rc', [0, 0])
        status = dbg.get('status', 'rejected')
        reason = str(dbg.get('reason', ''))
        if status == 'accepted':
            color = (220, 30, 30)
        elif reason == 'too_narrow':
            color = (255, 120, 0)
        elif reason == 'too_wide':
            color = (160, 0, 255)
        elif reason == 'flanking_walls_failed':
            color = (255, 0, 180)
        elif reason == 'suppressed_by_nms':
            color = (120, 120, 255)
        else:
            color = (150, 150, 150)
        cv2.rectangle(img, (int(c0), int(r0)), (int(c1), int(r1)), color, 2)
        cv2.circle(img, (int(cj), int(ci)), 4, color, -1)
        short_reason = {
            'accepted': 'OK',
            'too_narrow': 'N',
            'too_wide': 'W',
            'flanking_walls_failed': 'F',
            'suppressed_by_nms': 'X',
        }.get(reason, '?')
        _draw_label(img, short_reason, int(cj) + 6, int(ci) + 14, color)

    for door in extractor.doors:
        di, dj = extractor.world_to_grid(float(door.pos_xy[0]), float(door.pos_xy[1]))
        cv2.circle(img, (dj, di), 6, (220, 30, 30), -1)
        _draw_label(img, f'D{door.id}', dj + 8, di + 14, (180, 0, 0))

    cv2.imwrite(out_png, img)


def _crop_bounds(ci: int, cj: int, radius: int, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = shape
    r0 = max(0, ci - radius)
    r1 = min(h, ci + radius + 1)
    c0 = max(0, cj - radius)
    c1 = min(w, cj + radius + 1)
    return r0, r1, c0, c1


def _wall_is_linear(wall_i: np.ndarray, wall_j: np.ndarray, max_deviation_cells: float = 1.5) -> bool:
    if len(wall_i) < 3:
        return True
    pts = np.column_stack([wall_j.astype(float), wall_i.astype(float)])
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    cov = centered.T @ centered
    _, eigvecs = np.linalg.eigh(cov)
    minor_axis = eigvecs[:, 0]
    residuals = np.abs(centered @ minor_axis)
    return float(residuals.max()) <= max_deviation_cells


def _collect_wall_strip(start_i: int, start_j: int, direction: tuple[int, int], length: int, occupied: np.ndarray, search_width: int = 2) -> tuple[np.ndarray, np.ndarray]:
    h, w = occupied.shape
    di, dj = direction
    found_i, found_j = [], []
    for step in range(1, length + 1):
        ci = start_i + di * step
        cj = start_j + dj * step
        hit = False
        for offs in range(-search_width, search_width + 1):
            wi = ci + (0 if di != 0 else offs)
            wj = cj + (offs if di != 0 else 0)
            if 0 <= wi < h and 0 <= wj < w and occupied[wi, wj]:
                found_i.append(wi)
                found_j.append(wj)
                hit = True
        if not hit:
            break
    return np.asarray(found_i, dtype=int), np.asarray(found_j, dtype=int)


def _free_run_length(start_i: int, start_j: int, direction: tuple[int, int], free: np.ndarray, max_steps: int) -> int:
    h, w = free.shape
    di, dj = direction
    steps = 0
    for step in range(1, max_steps + 1):
        ci = start_i + di * step
        cj = start_j + dj * step
        if not (0 <= ci < h and 0 <= cj < w):
            break
        if not free[ci, cj]:
            break
        steps += 1
    return steps


def _sample_axis_width(ci: int, cj: int, direction: tuple[int, int], free: np.ndarray, max_steps: int) -> int:
    return 1 + _free_run_length(ci, cj, direction, free, max_steps) + _free_run_length(ci, cj, (-direction[0], -direction[1]), free, max_steps)


def _estimate_door_axis(ci: int, cj: int, free: np.ndarray, resolution: float, max_width_m: float) -> dict:
    max_steps = max(4, int(np.ceil(max_width_m / max(resolution, 1e-6))))
    horizontal_width_cells = _sample_axis_width(ci, cj, (0, 1), free, max_steps)
    vertical_width_cells = _sample_axis_width(ci, cj, (1, 0), free, max_steps)
    if horizontal_width_cells <= vertical_width_cells:
        return {
            'door_axis': 'horizontal',
            'width_cells': int(horizontal_width_cells),
            'normal_dirs': ((-1, 0), (1, 0)),
        }
    return {
        'door_axis': 'vertical',
        'width_cells': int(vertical_width_cells),
        'normal_dirs': ((0, -1), (0, 1)),
    }


def _find_shoulder_wall(start_i: int, start_j: int, normal_dir: tuple[int, int], occupied: np.ndarray, max_search_steps: int, lateral_search: int = 2) -> tuple[int, int] | None:
    h, w = occupied.shape
    ni, nj = normal_dir
    for step in range(1, max_search_steps + 1):
        ci = start_i + ni * step
        cj = start_j + nj * step
        for offs in range(-lateral_search, lateral_search + 1):
            wi = ci + (offs if nj != 0 else 0)
            wj = cj + (offs if ni != 0 else 0)
            if 0 <= wi < h and 0 <= wj < w and occupied[wi, wj]:
                return int(wi), int(wj)
    return None


def _built_map_wall_support(ci: int, cj: int, free: np.ndarray, occupied: np.ndarray, resolution: float, flank_min_m: float = 0.5, free_probe_m: float = 1.5, width_max_m: float = 2.0) -> dict:
    min_wall_cells = max(3, int(np.ceil(flank_min_m / max(resolution, 1e-6))))
    search_len = min_wall_cells + 3
    free_probe_cells = max(6, int(np.ceil(free_probe_m / max(resolution, 1e-6))))

    axis = _estimate_door_axis(ci, cj, free, resolution, width_max_m)
    normal_a, normal_b = axis['normal_dirs']
    shoulder_search_steps = max(2, int(np.ceil(0.5 * axis['width_cells'])) + 2)
    shoulder_a = _find_shoulder_wall(ci, cj, normal_a, occupied, shoulder_search_steps)
    shoulder_b = _find_shoulder_wall(ci, cj, normal_b, occupied, shoulder_search_steps)

    if axis['door_axis'] == 'horizontal':
        flank_dir_neg = (0, -1)
        flank_dir_pos = (0, 1)
        passage_cells = _free_run_length(ci, cj, (-1, 0), free, free_probe_cells) + _free_run_length(ci, cj, (1, 0), free, free_probe_cells)
        orientation = 'horizontal_gap'
    else:
        flank_dir_neg = (-1, 0)
        flank_dir_pos = (1, 0)
        passage_cells = _free_run_length(ci, cj, (0, -1), free, free_probe_cells) + _free_run_length(ci, cj, (0, 1), free, free_probe_cells)
        orientation = 'vertical_gap'

    if shoulder_a is not None:
        side_a_neg_i, side_a_neg_j = _collect_wall_strip(shoulder_a[0], shoulder_a[1], flank_dir_neg, search_len, occupied)
        side_a_pos_i, side_a_pos_j = _collect_wall_strip(shoulder_a[0], shoulder_a[1], flank_dir_pos, search_len, occupied)
        side_a_len = max(len(side_a_neg_i), len(side_a_pos_i))
        side_a_linear = _wall_is_linear(side_a_neg_i, side_a_neg_j) or _wall_is_linear(side_a_pos_i, side_a_pos_j)
    else:
        side_a_len = 0
        side_a_linear = False

    if shoulder_b is not None:
        side_b_neg_i, side_b_neg_j = _collect_wall_strip(shoulder_b[0], shoulder_b[1], flank_dir_neg, search_len, occupied)
        side_b_pos_i, side_b_pos_j = _collect_wall_strip(shoulder_b[0], shoulder_b[1], flank_dir_pos, search_len, occupied)
        side_b_len = max(len(side_b_neg_i), len(side_b_pos_i))
        side_b_linear = _wall_is_linear(side_b_neg_i, side_b_neg_j) or _wall_is_linear(side_b_pos_i, side_b_pos_j)
    else:
        side_b_len = 0
        side_b_linear = False

    wall_len = min(side_a_len, side_b_len)
    wall_linear = bool(side_a_linear and side_b_linear)
    orientation_score = (wall_len if wall_linear else 0) + 0.35 * passage_cells

    return {
        'orientation': orientation,
        'door_axis': axis['door_axis'],
        'estimated_width_cells': int(axis['width_cells']),
        'shoulder_a_rc': list(shoulder_a) if shoulder_a is not None else None,
        'shoulder_b_rc': list(shoulder_b) if shoulder_b is not None else None,
        'side_a_wall_len_cells': int(side_a_len),
        'side_b_wall_len_cells': int(side_b_len),
        'side_a_linear': bool(side_a_linear),
        'side_b_linear': bool(side_b_linear),
        'wall_support_cells': int(wall_len),
        'wall_support_m': float(wall_len * resolution),
        'wall_linear': bool(wall_linear),
        'passage_cells': int(passage_cells),
        'passage_m': float(passage_cells * resolution),
        'orientation_score': float(orientation_score),
    }


def _mask_centroid_world(extractor: TopologyExtractor, mask: np.ndarray) -> np.ndarray:
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return np.zeros((2,), dtype=np.float32)
    ci = int(np.round(rows.mean()))
    cj = int(np.round(cols.mean()))
    return extractor.grid_to_world(ci, cj)


def postprocess_small_rooms(extractor: TopologyExtractor, merge_small_area_cells: int = 0, min_contact_cells: int = 8) -> dict | None:
    if merge_small_area_cells <= 0 or not extractor.confirmed_rooms:
        return None

    rooms = {int(rid): room for rid, room in extractor.confirmed_rooms.items()}
    label_map = np.zeros_like(next(iter(rooms.values())).mask, dtype=np.int32)
    for rid, room in rooms.items():
        label_map[room.mask] = int(rid)

    merged_pairs = []
    changed = True
    while changed:
        changed = False
        room_ids = sorted(rooms.keys(), key=lambda rid: int(rooms[rid].mask.sum()))
        for rid in room_ids:
            if rid not in rooms:
                continue
            room = rooms[rid]
            area = int(room.mask.sum())
            if area > int(merge_small_area_cells):
                continue

            mask = room.mask
            neighbors = {}
            for shifted in (label_map[1:, :], label_map[:-1, :], label_map[:, 1:], label_map[:, :-1]):
                pass
            # 4-neighbor contact counts with adjacent room labels
            up_mask = mask[1:, :]
            up_lbls = label_map[:-1, :][up_mask]
            down_mask = mask[:-1, :]
            down_lbls = label_map[1:, :][down_mask]
            left_mask = mask[:, 1:]
            left_lbls = label_map[:, :-1][left_mask]
            right_mask = mask[:, :-1]
            right_lbls = label_map[:, 1:][right_mask]
            for arr in (up_lbls, down_lbls, left_lbls, right_lbls):
                for lbl in arr:
                    lbl = int(lbl)
                    if lbl == 0 or lbl == rid or lbl not in rooms:
                        continue
                    neighbors[lbl] = neighbors.get(lbl, 0) + 1
            if not neighbors:
                continue
            target_id, contact = max(neighbors.items(), key=lambda kv: (kv[1], int(rooms[kv[0]].mask.sum())))
            if int(contact) < int(min_contact_cells):
                continue

            target = rooms[target_id]
            target.mask = np.logical_or(target.mask, room.mask)
            target.centroid_xy = _mask_centroid_world(extractor, target.mask)
            if hasattr(target, 'doors') and hasattr(room, 'doors'):
                target.doors = list(getattr(target, 'doors', [])) + list(getattr(room, 'doors', []))
            del rooms[rid]
            label_map[mask] = int(target_id)
            merged_pairs.append({'source': int(rid), 'target': int(target_id), 'area_cells': int(area), 'contact_cells': int(contact)})
            changed = True
            break

    extractor.confirmed_rooms = {int(rid): room for rid, room in rooms.items()}
    return {
        'merge_small_area_cells': int(merge_small_area_cells),
        'min_contact_cells': int(min_contact_cells),
        'merged_pairs': merged_pairs,
        'merged_count': int(len(merged_pairs)),
        'remaining_confirmed_rooms': int(len(extractor.confirmed_rooms)),
    }


def analyze_built_map_necks(
    slam: np.ndarray,
    resolution: float,
    out_dir: str,
    width_min_m: float = 0.3,
    width_max_m: float = 2.0,
    probe_radius_m: float = 1.2,
    center_block_m: float = 0.20,
    min_lobe_area_m2: float = 0.35,
    dedup_radius_m: float = 0.35,
    cluster_eps_cells: int = 12,
    top_k: int = 12,
    flank_min_m: float = 0.5,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = str(out_dir / 'built_map_necks.png')
    out_json = str(out_dir / 'built_map_necks.json')
    crops_dir = out_dir / 'built_map_neck_crops'
    crops_dir.mkdir(parents=True, exist_ok=True)

    free = (slam == 0)
    occupied = (slam > 50)
    free_u8 = free.astype(np.uint8)
    dt = cv2.distanceTransform(free_u8, cv2.DIST_L2, 5).astype(np.float32) * float(resolution)
    width_m = 2.0 * dt

    local_max = dt >= ndimage.maximum_filter(dt, size=3, mode='nearest')
    candidate_mask = free & local_max & (width_m >= float(width_min_m)) & (width_m <= float(width_max_m))
    pts = np.stack(np.where(candidate_mask), axis=1) if np.any(candidate_mask) else np.zeros((0, 2), dtype=np.int32)

    probe_r = max(6, int(np.ceil(probe_radius_m / max(float(resolution), 1e-6))))
    center_block_r = max(2, int(np.ceil(center_block_m / max(float(resolution), 1e-6))))
    min_lobe_cells = max(8, int(np.ceil(min_lobe_area_m2 / max(float(resolution), 1e-6) ** 2)))
    dedup_r = max(3, int(np.ceil(dedup_radius_m / max(float(resolution), 1e-6))))
    order = np.argsort(width_m[pts[:, 0], pts[:, 1]]) if len(pts) > 0 else np.zeros((0,), dtype=np.int32)

    candidates = []
    deduped_pts = []
    for idx in order:
        ci, cj = pts[idx]
        ij = np.array([ci, cj], dtype=np.float32)
        if deduped_pts:
            d = np.linalg.norm(np.stack(deduped_pts, axis=0) - ij[None, :], axis=1)
            if np.any(d < dedup_r):
                continue

        r0, r1, c0, c1 = _crop_bounds(int(ci), int(cj), probe_r, free.shape)
        patch = free[r0:r1, c0:c1].copy().astype(np.uint8)
        py = int(ci) - r0
        px = int(cj) - c0
        yy, xx = np.ogrid[:patch.shape[0], :patch.shape[1]]
        block = (yy - py) ** 2 + (xx - px) ** 2 <= center_block_r ** 2
        patch[block] = 0
        n_cc, cc = cv2.connectedComponents(patch)

        border = np.zeros_like(patch, dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        border_components = 0
        border_areas = []
        for lbl in range(1, n_cc):
            comp = (cc == lbl)
            area = int(comp.sum())
            if area < min_lobe_cells:
                continue
            if not np.any(comp & border):
                continue
            border_components += 1
            border_areas.append(area)

        support = _built_map_wall_support(
            int(ci),
            int(cj),
            free,
            occupied,
            float(resolution),
            flank_min_m=flank_min_m,
            width_max_m=width_max_m,
        )
        width_here = float(width_m[int(ci), int(cj)])
        target_width = 0.9
        width_score = max(0.0, 1.0 - abs(width_here - target_width) / max(width_max_m - width_min_m, 1e-6))
        doorway_score = float(
            1.2 * support['wall_support_m']
            + 0.4 * support['passage_m']
            + 0.6 * width_score
            + 0.5 * border_components
        )
        entry = {
            'grid_rc': [int(ci), int(cj)],
            'width_m': width_here,
            'status': 'accepted' if border_components >= 2 else 'rejected',
            'reason': 'splits_space' if border_components >= 2 else 'no_two_lobes',
            'border_components': int(border_components),
            'border_areas': border_areas,
            'probe_radius_m': float(probe_radius_m),
            'center_block_m': float(center_block_m),
            'flank_min_m': float(flank_min_m),
            'score': doorway_score,
            'support': support,
        }
        candidates.append(entry)
        deduped_pts.append(ij)

    clusters = []
    if candidates:
        pts2 = np.asarray([c['grid_rc'] for c in candidates], dtype=np.float32)
        labels = DBSCAN(eps=max(1, int(cluster_eps_cells)), min_samples=1, n_jobs=1).fit_predict(pts2)
        for lbl in sorted(np.unique(labels)):
            member_indices = [int(i) for i in range(len(candidates)) if int(labels[i]) == int(lbl)]
            members = [candidates[i] for i in member_indices]
            rows = np.asarray([m['grid_rc'][0] for m in members], dtype=np.int32)
            cols = np.asarray([m['grid_rc'][1] for m in members], dtype=np.int32)
            widths = np.asarray([m['width_m'] for m in members], dtype=np.float32)
            border_counts = np.asarray([m['border_components'] for m in members], dtype=np.int32)
            support_scores = np.asarray([m['support']['orientation_score'] for m in members], dtype=np.float32)
            wall_supports = np.asarray([m['support']['wall_support_m'] for m in members], dtype=np.float32)
            best = max(members, key=lambda m: (m['score'], m['support']['wall_support_m'], -abs(m['width_m'] - 0.9)))
            accepted_count = int(sum(1 for m in members if m['status'] == 'accepted'))
            cluster_score = float(
                best['score']
                + 0.25 * len(members)
                + 0.35 * float(wall_supports.max())
                + 0.1 * float(support_scores.mean())
                + 0.2 * accepted_count
            )
            clusters.append({
                'cluster_id': 0,
                'count': int(len(members)),
                'accepted_count': accepted_count,
                'best_status': best['status'],
                'best_reason': best['reason'],
                'best_width_m': float(best['width_m']),
                'width_min_m': float(widths.min()),
                'width_mean_m': float(widths.mean()),
                'width_max_m': float(widths.max()),
                'border_components_max': int(border_counts.max()),
                'wall_support_max_m': float(wall_supports.max()),
                'orientation': best['support']['orientation'],
                'score': cluster_score,
                'center_rc': [int(np.round(rows.mean())), int(np.round(cols.mean()))],
                'bbox_rc': [int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())],
                'member_indices': member_indices,
                'best_candidate': best,
            })

    clusters.sort(key=lambda c: (-c['score'], c['best_width_m']))
    for idx, cluster in enumerate(clusters, start=1):
        cluster['cluster_id'] = idx

    img = _base_map_rgb(slam)
    for cluster in clusters[:top_k]:
        r0, c0, r1, c1 = cluster['bbox_rc']
        ci, cj = cluster['center_rc']
        color = (30, 30, 230) if cluster['wall_support_max_m'] >= flank_min_m else (220, 0, 220)
        pad = 4
        cv2.rectangle(img, (max(0, c0 - pad), max(0, r0 - pad)), (min(img.shape[1] - 1, c1 + pad), min(img.shape[0] - 1, r1 + pad)), color, 2)
        cv2.circle(img, (int(cj), int(ci)), 5, color, -1)
        label = f"C{cluster['cluster_id']} {cluster['best_width_m']:.2f}m {cluster['orientation'][0].upper()}"
        _draw_label(img, label, int(cj) + 6, int(ci) - 6, color, scale=0.45)

    for entry in candidates:
        ci, cj = entry['grid_rc']
        support_ok = entry['support']['wall_support_m'] >= flank_min_m
        color = (30, 30, 230) if support_ok else (160, 80, 220)
        radius = 4 if support_ok else 2
        cv2.circle(img, (int(cj), int(ci)), radius, color, -1)

    for cluster in clusters[:top_k]:
        ci, cj = cluster['center_rc']
        crop_r = max(probe_r, 28)
        r0, r1, c0, c1 = _crop_bounds(int(ci), int(cj), crop_r, slam.shape)
        crop = _base_map_rgb(slam)[r0:r1, c0:c1].copy()
        crop_center = (int(cj - c0), int(ci - r0))
        cv2.circle(crop, crop_center, max(2, center_block_r), (220, 0, 220), 2)
        for member_idx in cluster['member_indices']:
            mci, mcj = candidates[member_idx]['grid_rc']
            local = (int(mcj - c0), int(mci - r0))
            cv2.circle(crop, local, 2, (60, 180, 255), -1)
        crop_path = crops_dir / f"cluster_{cluster['cluster_id']:02d}.png"
        cv2.imwrite(str(crop_path), crop)
        cluster['crop_png'] = str(crop_path)

    cv2.imwrite(out_png, img)

    analysis = {
        'width_min_m': float(width_min_m),
        'width_max_m': float(width_max_m),
        'probe_radius_m': float(probe_radius_m),
        'center_block_m': float(center_block_m),
        'min_lobe_area_m2': float(min_lobe_area_m2),
        'dedup_radius_m': float(dedup_radius_m),
        'cluster_eps_cells': int(cluster_eps_cells),
        'flank_min_m': float(flank_min_m),
        'candidate_count': len(candidates),
        'accepted_count': int(sum(1 for c in candidates if c['status'] == 'accepted')),
        'cluster_count': len(clusters),
        'top_clusters': clusters[:top_k],
        'candidates': candidates,
        'overlay_png': out_png,
        'crops_dir': str(crops_dir),
    }
    with open(out_json, 'w', encoding='utf-8') as fh:
        json.dump(analysis, fh, indent=2)
    return analysis


def build_summary(extractor: TopologyExtractor, map_yaml: str, image_path: str, out_png: str, built_map_necks: dict | None = None, room_postprocess: dict | None = None):
    return {
        'map_yaml': os.path.abspath(map_yaml),
        'map_image': os.path.abspath(image_path),
        'overlay_png': os.path.abspath(out_png),
        'counts': {
            'confirmed_rooms': len(extractor.confirmed_rooms),
            'unconfirmed_rooms': len(extractor.unconfirmed_rooms),
            'doors': len(extractor.doors),
            'frontier_centroids': int(len(extractor.frontier_centroids)),
            'door_tracks': int(len(extractor._door_tracks)),
        },
        'tuning': {
            'room_seg_ref_cell_m': TopologyExtractor._ROOM_SEG_REF_CELL_M,
            'room_seg_kernel_ref': TopologyExtractor._ROOM_SEG_KERNEL_REF,
            'room_seg_iter': TopologyExtractor._ROOM_SEG_ITER,
            'room_min_cc_cells': TopologyExtractor._ROOM_MIN_CC_CELLS,
            'room_small_area_ref_cells': TopologyExtractor._ROOM_SMALL_AREA_REF_CELLS,
            'room_small_boundary_frontiers': TopologyExtractor._ROOM_SMALL_BOUNDARY_FRONTIERS,
            'room_large_boundary_frontiers': TopologyExtractor._ROOM_LARGE_BOUNDARY_FRONTIERS,
            'room_large_inside_frontiers': TopologyExtractor._ROOM_LARGE_INSIDE_FRONTIERS,
            'room_outer_opening_ref_cells': TopologyExtractor._ROOM_OUTER_OPENING_REF_CELLS,
            'door_width_min_m': TopologyExtractor._DOOR_WIDTH_MIN_M,
            'door_width_max_m': TopologyExtractor._DOOR_WIDTH_MAX_M,
            'door_flank_min_m': TopologyExtractor._DOOR_FLANK_MIN_M,
            'door_nms_radius_m': TopologyExtractor._DOOR_NMS_RADIUS_M,
            'frontier_occ_clear_dilate': TopologyExtractor._FRONTIER_OCC_CLEAR_DILATE,
            'dbscan_eps': TopologyExtractor._DBSCAN_EPS,
            'dbscan_min_samples': TopologyExtractor._DBSCAN_MIN_SAMPLES,
        },
        'rooms': [
            {
                'id': int(room.id),
                'centroid_xy': [float(room.centroid_xy[0]), float(room.centroid_xy[1])],
                'area_cells': int(room.mask.sum()),
                'doors': [int(door.id) for door in room.doors],
                'confirmed': True,
            }
            for room in extractor.confirmed_rooms.values()
        ],
        'unconfirmed_rooms': [
            {
                'id': int(room.id),
                'centroid_xy': [float(room.centroid_xy[0]), float(room.centroid_xy[1])],
                'area_cells': int(room.mask.sum()),
                'confirmed': False,
            }
            for room in extractor.unconfirmed_rooms.values()
        ],
        'door_debug_rooms': getattr(extractor, '_door_debug_rooms', []),
        'door_debug': getattr(extractor, '_door_debug', []),
        'built_map_necks': built_map_necks,
        'room_postprocess': room_postprocess,
        'doors': [
            {
                'id': int(door.id),
                'pos_xy': [float(door.pos_xy[0]), float(door.pos_xy[1])],
                'room_ids': sorted(int(v) for v in door.room_ids),
            }
            for door in extractor.doors
        ],
    }


def run_once(args):
    apply_tuning(args)
    slam, resolution, origin, image_path = load_ros_map(args.map_yaml)
    extractor = TopologyExtractor()
    extractor.update(slam, resolution, origin)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    overlay_png = str(out_dir / 'topology_overlay.png')
    summary_json = str(out_dir / 'summary.json')

    room_postprocess = postprocess_small_rooms(
        extractor,
        merge_small_area_cells=int(getattr(args, 'merge_small_rooms_area_cells', 0) or 0),
        min_contact_cells=int(getattr(args, 'merge_small_rooms_min_contact', 8) or 8),
    )
    draw_overlay(slam, extractor, overlay_png)
    built_map_necks = None
    if getattr(args, 'built_map_neck_debug', False):
        built_map_necks = analyze_built_map_necks(
            slam,
            resolution,
            str(out_dir),
            width_min_m=float(getattr(args, 'neck_width_min_m', 0.3)),
            width_max_m=float(getattr(args, 'neck_width_max_m', 2.0)),
            probe_radius_m=float(getattr(args, 'neck_probe_radius_m', 1.2)),
            center_block_m=float(getattr(args, 'neck_center_block_m', 0.20)),
            min_lobe_area_m2=float(getattr(args, 'neck_min_lobe_area_m2', 0.35)),
            dedup_radius_m=float(getattr(args, 'neck_dedup_radius_m', 0.35)),
            cluster_eps_cells=int(getattr(args, 'neck_cluster_eps_cells', 12)),
            top_k=int(getattr(args, 'neck_top_k', 12)),
            flank_min_m=float(getattr(args, 'neck_flank_min_m', 0.5)),
        )
    summary = build_summary(extractor, args.map_yaml, image_path, overlay_png, built_map_necks=built_map_necks, room_postprocess=room_postprocess)
    with open(summary_json, 'w', encoding='utf-8') as fh:
        json.dump(summary, fh, indent=2)

    print(json.dumps(summary['counts'], indent=2))
    print(f'overlay={overlay_png}')
    print(f'summary={summary_json}')
    if built_map_necks is not None:
        print(f"built_map_necks={built_map_necks['overlay_png']}")
        print(f"built_map_neck_clusters={built_map_necks['cluster_count']}")


def build_arg_parser():
    ap = argparse.ArgumentParser(description='Run TopologyExtractor offline on a ROS map.yaml/map.pgm pair.')
    ap.add_argument('--map-yaml', default='/home/mayank/turtlebot3_ws/mntfields_model/map.yaml')
    ap.add_argument('--out-dir', default='/tmp/topology_map_test')
    ap.add_argument('--room-seg-ref-cell-m', type=float)
    ap.add_argument('--room-seg-kernel-ref', type=int)
    ap.add_argument('--room-seg-iter', type=int)
    ap.add_argument('--room-min-cc-cells', type=int)
    ap.add_argument('--room-small-area-ref-cells', type=int)
    ap.add_argument('--room-small-boundary-frontiers', type=int)
    ap.add_argument('--room-large-boundary-frontiers', type=int)
    ap.add_argument('--room-large-inside-frontiers', type=int)
    ap.add_argument('--room-outer-opening-ref-cells', type=int)
    ap.add_argument('--door-width-min-m', type=float)
    ap.add_argument('--door-width-max-m', type=float)
    ap.add_argument('--door-flank-min-m', type=float)
    ap.add_argument('--door-nms-radius-m', type=float)
    ap.add_argument('--frontier-occ-clear-dilate', type=int)
    ap.add_argument('--dbscan-eps', type=int)
    ap.add_argument('--dbscan-min-samples', type=int)
    ap.add_argument('--merge-small-rooms-area-cells', type=int, default=0)
    ap.add_argument('--merge-small-rooms-min-contact', type=int, default=8)
    ap.add_argument('--built-map-neck-debug', action='store_true')
    ap.add_argument('--neck-width-min-m', type=float, default=0.3)
    ap.add_argument('--neck-width-max-m', type=float, default=2.0)
    ap.add_argument('--neck-probe-radius-m', type=float, default=1.2)
    ap.add_argument('--neck-center-block-m', type=float, default=0.20)
    ap.add_argument('--neck-min-lobe-area-m2', type=float, default=0.35)
    ap.add_argument('--neck-dedup-radius-m', type=float, default=0.35)
    ap.add_argument('--neck-cluster-eps-cells', type=int, default=12)
    ap.add_argument('--neck-top-k', type=int, default=12)
    ap.add_argument('--neck-flank-min-m', type=float, default=0.5)
    return ap


def main():
    ap = build_arg_parser()
    args = ap.parse_args()
    run_once(args)


if __name__ == '__main__':
    main()
