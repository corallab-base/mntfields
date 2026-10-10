#!/usr/bin/env python3
"""
segformer_realtime.py — Standalone RealSense + SegFormer real-time demo.

Launches an Intel RealSense D435, runs SegFormer-B0 ADE20K for semantic
segmentation, and displays an annotated
overlay with door/furniture detections.

No ROS dependency — can be tested directly with the camera plugged in.

Usage:
    python3 segformer_realtime.py [--width 640] [--height 480] [--fps 30]
                                   [--seg-rate 5] [--depth-max 4.0]
                                   [--no-depth] [--save-dir ./captures]

Keys:
    q / ESC — quit
    s       — save current frame + segmentation + depth to --save-dir
    d       — toggle depth overlay
    p       — pause / unpause segmentation

Active model: nvidia/segformer-b0-finetuned-ade-512-512
Previous model: shi-labs/oneformer_ade20k_swin_tiny
"""

import argparse
import os
import sys
import time
from collections import defaultdict

import cv2
import numpy as np
import torch

# ---------------------------------------------------------------------------
# ADE20K class mapping → useful indoor classes
# ---------------------------------------------------------------------------

# ADE20K 150 classes → reduced set for indoor room/furniture understanding.
# Class indices are ADE20K/SceneParse150 labels used by SegFormer/OneFormer.

REDUCED_LABELS = {
    0: "background",
#    1: "floor",
#    2: "wall",
    3: "door",
#    4: "window",
    5: "table",
    6: "chair", #was commented
    7: "bed",
    8: "sofa",
    9: "cabinet",
#    10: "appliance",
    11: "other_furniture",
}

# ADE20K class index → reduced class index
# fmt: off
_ADE20K_TO_REDUCED = np.zeros(150, dtype=np.uint8)  # 150 classes (0-149)

#_ADE20K_TO_REDUCED[[3, 28]]                                    = 1   # floor, rug
#_ADE20K_TO_REDUCED[[0, 42]]                                    = 2   # wall, column/pillar
_ADE20K_TO_REDUCED[[14, 58]]                                   = 3   # door, screen_door
#_ADE20K_TO_REDUCED[[8]]                                        = 4   # windowpane
_ADE20K_TO_REDUCED[[15, 33, 45, 77, 64, 70, 73]]              = 5   # table, desk, counter, bar, coffee_table, countertop, kitchen_island
_ADE20K_TO_REDUCED[[19, 30, 75, 31, 69, 110]]                 = 6   #COMMENT chair, armchair, swivel_chair, seat, bench, stool
_ADE20K_TO_REDUCED[[7]]                                        = 7   # bed
_ADE20K_TO_REDUCED[[23, 97]]                                   = 8   # sofa, ottoman/pouf
_ADE20K_TO_REDUCED[[10, 35, 44, 62, 24, 55, 99]]              = 9   # cabinet, wardrobe, chest_of_drawers, bookcase, shelf, display_case, buffet/sideboard
#_ADE20K_TO_REDUCED[[50, 71, 118, 47, 65, 37, 107, 129, 124]]  = 10  # refrigerator, stove, oven, sink, toilet, bathtub, washer, dishwasher, microwave
_ADE20K_TO_REDUCED[[36, 89, 18, 27, 22, 39, 41, 17, 85, 63]]  = 11  # lamp, tv, curtain, mirror, painting, cushion, box, plant, chandelier, blind
# fmt: on

# Colours for overlay (BGR)
COLOURS = np.array([
    [0,   0,   0],     # 0  background (transparent)
    [0,   0,   0],      #[180, 180, 140],   # 1  floor
    [0,   0,   0],      #[100, 100, 100],   # 2  wall
    [0,   0,   255],   # 3  door          ← RED (high visibility)
    [0,   0,   0],      #[255, 200, 0],     # 4  window
    [0,   180, 0],     # 5  table
    [0,   220, 220],#[0,   220, 220],   # 6  chair # make it 0, 0, 0
    [180, 0,   180],   # 7  bed
    [220, 140, 0],     # 8  sofa
    [50,  130, 220],   # 9  cabinet
    [0,   0,   0],      #[200, 200, 0],     # 10 appliance
    [140, 100, 180],   # 11 other_furniture
], dtype=np.uint8)

# Classes to draw bounding boxes + labels for
_HIGHLIGHT_CLASSES = {3, 5, 6, 7, 8, 9, 11} #{3, 4, 5, 6, 7, 8, 9, 10, 11}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
# ----------SegFormer--------------------------------------------------------
#def load_model(device: torch.device):
#    """Load SegFormer-B0 ADE20K from HuggingFace.
#
#    To swap models, change only model_name if the replacement uses the same
#    ADE20K 150-class label space. If the label space changes, also update
#    _ADE20K_TO_REDUCED above.
#    """
#    from transformers import (
#        SegformerImageProcessor,
#        SegformerForSemanticSegmentation,
#    )
#
#    model_name = os.environ.get(
#        "MNTFIELDS_SEMSEG_MODEL",
#        "nvidia/segformer-b0-finetuned-ade-512-512",
#    )
#    print(f"[MODEL] Loading {model_name} ...")
#    processor = SegformerImageProcessor.from_pretrained(model_name)
#    model = SegformerForSemanticSegmentation.from_pretrained(model_name)
#    model = model.to(device).eval()
#
#    # Warm up
#    dummy = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
#    inputs = processor(images=dummy, return_tensors="pt")
#    inputs = {k: v.to(device) for k, v in inputs.items()}
#    with torch.no_grad():
#        model(**inputs)
#    print(f"[MODEL] Loaded on {device}, warm-up done.")
#    return processor, model
#
#
#def segment_frame(rgb: np.ndarray, processor, model, device: torch.device,
#                  target_size: tuple = None):
#    """Run SegFormer semantic segmentation.
#
#    Args:
#        rgb: (H, W, 3) uint8 RGB image.
#        target_size: (H, W) for output resolution. Defaults to input size.
#
#    Returns:
#        (H, W) uint8 array of reduced class labels (0-11).
#    """
#    if target_size is None:
#        target_size = (rgb.shape[0], rgb.shape[1])
#
#    inputs = processor(images=rgb, return_tensors="pt")
#    inputs = {k: v.to(device) for k, v in inputs.items()}
#
#    with torch.no_grad():
#        outputs = model(**inputs)
#
#    # Post-process → per-pixel ADE20K class IDs at target resolution
#    seg_list = processor.post_process_semantic_segmentation(
#        outputs, target_sizes=[target_size])
#    ade_labels = seg_list[0].cpu().numpy().astype(np.int16)
#
#    # Map 150 → 12 reduced classes
#    ade_labels = np.clip(ade_labels, 0, len(_ADE20K_TO_REDUCED) - 1)
#    return _ADE20K_TO_REDUCED[ade_labels]
#-----------------------------------------------------------------------------

# ---------OneFormer----------------------------------------------------------
# Previous OneFormer implementation kept for rollback/reference.
# To restore it, replace load_model() and segment_frame() above with this pair.
def load_model(device: torch.device):
    """Load OneFormer Swin-Tiny ADE20K from HuggingFace."""
    from transformers import (
        OneFormerProcessor,
        OneFormerForUniversalSegmentation,
    )
    model_name = os.environ.get(
        "MNTFIELDS_SEMSEG_MODEL",
        "shi-labs/oneformer_ade20k_swin_tiny",
    )
    print(f"[MODEL] Loading {model_name} ...")
    processor = OneFormerProcessor.from_pretrained(model_name)
    model = OneFormerForUniversalSegmentation.from_pretrained(model_name)
    model = model.to(device).eval()
    dummy = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    inputs = processor(images=dummy, task_inputs=["semantic"], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        model(**inputs)
    print(f"[MODEL] Loaded on {device}, warm-up done.")
    return processor, model

def segment_frame(rgb: np.ndarray, processor, model, device: torch.device,
                  target_size: tuple = None, conf_thr: float = 0.0):
    """Run OneFormer semantic segmentation.

    When ``conf_thr > 0``, pixels whose top-class probability is below the
    threshold are reassigned to class 0 (background). This suppresses
    low-confidence label flicker at class boundaries and ambiguous surfaces
    without changing the label space. ``post_process_semantic_segmentation``
    only returns hard labels, so the per-pixel probability is reconstructed
    from the query outputs inline.
    """
    if target_size is None:
        target_size = (rgb.shape[0], rgb.shape[1])
    inputs = processor(images=rgb, task_inputs=["semantic"], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    seg_list = processor.post_process_semantic_segmentation(
        outputs, target_sizes=[target_size])
    ade_labels = seg_list[0].cpu().numpy().astype(np.int16)
    ade_labels = np.clip(ade_labels, 0, len(_ADE20K_TO_REDUCED) - 1)

    if conf_thr > 0.0:
        with torch.no_grad():
            # OneFormer combined seg =
            #   softmax(class_queries_logits)[..., :-1] · sigmoid(masks_queries_logits)
            masks = outputs.masks_queries_logits.sigmoid()                # (B, Q, Hm, Wm)
            classes = outputs.class_queries_logits.softmax(-1)[..., :-1]  # (B, Q, C)
            seg_probs = torch.einsum("bqc,bqhw->bchw", classes, masks)    # (B, C, Hm, Wm)
            top_probs = seg_probs.max(dim=1).values                       # (B, Hm, Wm)
            top_probs = torch.nn.functional.interpolate(
                top_probs.unsqueeze(1), size=target_size,
                mode="bilinear", align_corners=False,
            ).squeeze(1)
            conf = top_probs[0].cpu().numpy()
        ade_labels[conf < conf_thr] = 0

    return _ADE20K_TO_REDUCED[ade_labels]
#--------------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def overlay_segmentation(bgr: np.ndarray, seg: np.ndarray, alpha: float = 0.45):
    """Blend segmentation colour overlay onto BGR image."""
    colour_map = COLOURS[seg]
    mask = seg > 0
    blended = bgr.copy()
    if not np.any(mask):
        return blended

    # Avoid boolean-indexed addWeighted slices; some OpenCV builds return None
    # for that input shape. Blend the full frame, then copy masked pixels back.
    overlay = cv2.addWeighted(bgr, 1.0 - alpha, colour_map, alpha, 0)
    blended[mask] = overlay[mask]
    return blended


def draw_class_boxes(overlay: np.ndarray, seg: np.ndarray):
    """Draw bounding boxes and labels for highlighted classes."""
    for cls_id in _HIGHLIGHT_CLASSES:
        mask = (seg == cls_id).astype(np.uint8)
        if mask.sum() < 200:
            continue

        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8)
        for i in range(1, n_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 500:
                continue
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]

            colour = COLOURS[cls_id].tolist()
            cv2.rectangle(overlay, (x, y), (x + w, y + h), colour, 2)

            label = REDUCED_LABELS[cls_id]
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(overlay, (x, y - th - 6), (x + tw + 4, y), colour, -1)
            cv2.putText(overlay, label, (x + 2, y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return overlay


def draw_legend(overlay: np.ndarray, seg: np.ndarray):
    """Draw class percentage legend in top-left corner."""
    h, w = seg.shape
    total = h * w
    y0 = 20
    for cls_id in range(1, 12):
        count = int((seg == cls_id).sum())
        if count == 0:
            continue
        pct = 100.0 * count / total
        label = f"{REDUCED_LABELS[cls_id]}: {pct:.1f}%"
        colour = COLOURS[cls_id].tolist()
        cv2.putText(overlay, label, (10, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1)
        y0 += 18
    return overlay


def depth_colourmap(depth_frame: np.ndarray, max_m: float = 4.0):
    """Convert uint16 depth (mm) to a colour visualization."""
    depth_m = depth_frame.astype(np.float32) / 1000.0
    norm = (np.clip(depth_m, 0, max_m) / max_m * 255).astype(np.uint8)
    return cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="RealSense + OneFormer real-time semantic segmentation")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seg-rate", type=int, default=5,
                        help="Target segmentation Hz (default 5)")
    parser.add_argument("--depth-max", type=float, default=4.0)
    parser.add_argument("--no-depth", action="store_true")
    parser.add_argument("--save-dir", type=str, default="./captures")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    # ── Device ─────────────────────────────────────────────────────────
    if args.cpu or not torch.cuda.is_available():
        device = torch.device("cpu")
        print("[DEVICE] CPU")
    else:
        device = torch.device("cuda:0")
        print(f"[DEVICE] CUDA — {torch.cuda.get_device_name(0)}")

    # ── Load model ─────────────────────────────────────────────────────
    processor, model = load_model(device)

    # ── RealSense pipeline ─────────────────────────────────────────────
    import pyrealsense2 as rs

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, args.width, args.height,
                         rs.format.bgr8, args.fps)
    if not args.no_depth:
        config.enable_stream(rs.stream.depth, args.width, args.height,
                             rs.format.z16, args.fps)

    print(f"[CAMERA] Starting D435 at {args.width}x{args.height} @ {args.fps}fps ...")
    try:
        profile = pipeline.start(config)
    except RuntimeError as e:
        print(f"[ERROR] Could not start RealSense: {e}")
        print("  Make sure the camera is connected and not in use.")
        sys.exit(1)

    align = rs.align(rs.stream.color) if not args.no_depth else None

    intrinsics = profile.get_stream(
        rs.stream.color).as_video_stream_profile().get_intrinsics()
    print(f"[CAMERA] Intrinsics: fx={intrinsics.fx:.1f} fy={intrinsics.fy:.1f} "
          f"cx={intrinsics.ppx:.1f} cy={intrinsics.ppy:.1f}")

    # ── Main loop ──────────────────────────────────────────────────────
    seg_interval = max(1, args.fps // args.seg_rate)
    frame_count = 0
    seg_result = None
    paused = False
    show_depth = False
    fps_t0 = time.time()
    fps_count = 0
    display_fps = 0.0
    seg_ms = 0.0

    print(f"[RUNNING] Segmentation every {seg_interval} frames (~{args.seg_rate} Hz)")
    print("  Keys: q=quit  s=save  d=depth  p=pause")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            if align is not None:
                frames = align.process(frames)

            colour_frame = frames.get_color_frame()
            if not colour_frame:
                continue

            bgr = np.asanyarray(colour_frame.get_data())
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            depth_image = None
            if not args.no_depth:
                df = frames.get_depth_frame()
                if df:
                    depth_image = np.asanyarray(df.get_data())

            # ── Segmentation ──────────────────────────────────────────
            frame_count += 1
            if not paused and frame_count % seg_interval == 0:
                t0 = time.time()
                seg_result = segment_frame(rgb, processor, model, device)
                seg_ms = (time.time() - t0) * 1000.0

            # ── Overlay ───────────────────────────────────────────────
            if seg_result is not None:
                overlay = overlay_segmentation(bgr, seg_result)
                overlay = draw_class_boxes(overlay, seg_result)
                overlay = draw_legend(overlay, seg_result)
            else:
                overlay = bgr.copy()

            # FPS
            fps_count += 1
            elapsed = time.time() - fps_t0
            if elapsed > 1.0:
                display_fps = fps_count / elapsed
                fps_count = 0
                fps_t0 = time.time()

            info = f"FPS: {display_fps:.0f} | Seg: {seg_ms:.0f}ms (~{1000/max(seg_ms,1):.0f} Hz)"
            if paused:
                info += " | PAUSED"
            cv2.putText(overlay, info, (10, overlay.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow("OneFormer Real-Time", overlay)

            if show_depth and depth_image is not None:
                cv2.imshow("Depth", depth_colourmap(depth_image, args.depth_max))

            # ── Keys ──────────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('s'):
                os.makedirs(args.save_dir, exist_ok=True)
                ts = int(time.time() * 1000)
                cv2.imwrite(os.path.join(args.save_dir, f"rgb_{ts}.png"), bgr)
                cv2.imwrite(os.path.join(args.save_dir, f"seg_{ts}.png"), overlay)
                if seg_result is not None:
                    np.save(os.path.join(args.save_dir, f"labels_{ts}.npy"),
                            seg_result)
                if depth_image is not None:
                    np.save(os.path.join(args.save_dir, f"depth_{ts}.npy"),
                            depth_image)
                print(f"[SAVED] {args.save_dir}/*_{ts}.*")
            elif key == ord('d'):
                show_depth = not show_depth
                if not show_depth:
                    try:
                        cv2.destroyWindow("Depth")
                    except cv2.error:
                        pass
            elif key == ord('p'):
                paused = not paused
                print(f"[SEG] {'paused' if paused else 'resumed'}")

    except KeyboardInterrupt:
        print("\n[STOPPED]")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
