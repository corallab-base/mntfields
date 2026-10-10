#!/usr/bin/env python3
"""
ROS2 RGB-D semantic segmentation node for a RealSense-class camera.

Subscribes to color + aligned depth + CameraInfo, runs the lightweight
semantic segmentation model from segformer_realtime.py, and publishes:

  /semantic_segmentation/overlay   sensor_msgs/Image  (bgr8)
  /semantic_segmentation/labels    sensor_msgs/Image  (mono8)
  /semantic_segmentation/objects   std_msgs/String    (JSON summary)

The JSON output contains connected components for semantic classes with
camera-frame and world-frame centroids recovered from aligned depth + TF.
"""

import json
import os
import threading
import time

import cv2
import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
import tf2_ros
import torch
from visualization_msgs.msg import Marker, MarkerArray

from mntfields.mntfields_ros.sensing.segformer_realtime import (
    COLOURS,
    REDUCED_LABELS,
    draw_class_boxes,
    draw_legend,
    load_model,
    overlay_segmentation,
    segment_frame,
)

_IMAGE_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=ReliabilityPolicy.BEST_EFFORT,
)

class SemanticSegmentationRGBDNode(Node):
    def __init__(self) -> None:
        super().__init__("semantic_segmentation_rgbd")

        self.declare_parameter("rgb_topic", "/camera/camera/color/image_raw")
        self.declare_parameter(
            "depth_topic", "/camera/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter(
            "camera_info_topic",
            "/camera/camera/aligned_depth_to_color/camera_info",
        )
        self.declare_parameter("global_frame", "map")
        self.declare_parameter("tick_hz", 2.0)
        self.declare_parameter("max_sync_slop_sec", 0.20)
        self.declare_parameter("sync_queue_size", 2)
        self.declare_parameter("max_frame_age_sec", 1.0)
        self.declare_parameter("allow_rgb_only", False)
        self.declare_parameter("min_depth_m", 0.20)
        self.declare_parameter("max_depth_m", 6.0)
        self.declare_parameter("min_component_area_px", 600)
        # Per-pixel confidence gate applied inside segment_frame. Pixels
        # whose top-class probability is below this threshold are reassigned
        # to background so low-confidence boundary flicker does not survive
        # as real detections. 0.0 disables the gate (legacy behaviour).
        self.declare_parameter("conf_thr", 0.5)
        self.declare_parameter("summary_stride_px", 4)
        self.declare_parameter("enable_world_coordinates", True)
        self.declare_parameter("tf_timeout_sec", 0.30)
        self.declare_parameter("show_window", False)
        self.declare_parameter("window_name", "Semantic Segmentation RGB-D")
        self.declare_parameter("use_cpu", False)
        self.declare_parameter("marker_topic", "/semantic_segmentation/markers")
        self.declare_parameter("publish_markers", True)
        self.declare_parameter(
            "snapshot_dir",
            os.path.expanduser("~/turtlebot3_ws/mntfields_model/semantic"),
        )
        self.declare_parameter("snapshot_interval_sec", 0.0)
        self.declare_parameter("health_log_sec", 5.0)

        self._rgb_topic = str(self.get_parameter("rgb_topic").value)
        self._depth_topic = str(self.get_parameter("depth_topic").value)
        self._camera_info_topic = str(
            self.get_parameter("camera_info_topic").value)
        self._global_frame = str(self.get_parameter("global_frame").value)
        self._tick_hz = float(self.get_parameter("tick_hz").value)
        self._max_sync_slop_sec = float(
            self.get_parameter("max_sync_slop_sec").value)
        self._sync_queue_size = max(1, int(
            self.get_parameter("sync_queue_size").value))
        self._max_frame_age_sec = float(
            self.get_parameter("max_frame_age_sec").value)
        self._allow_rgb_only = bool(self.get_parameter("allow_rgb_only").value)
        self._min_depth_m = float(self.get_parameter("min_depth_m").value)
        self._max_depth_m = float(self.get_parameter("max_depth_m").value)
        self._min_component_area_px = int(
            self.get_parameter("min_component_area_px").value)
        self._conf_thr = float(self.get_parameter("conf_thr").value)
        self._summary_stride_px = max(
            1, int(self.get_parameter("summary_stride_px").value))
        self._enable_world_coordinates = bool(
            self.get_parameter("enable_world_coordinates").value)
        self._tf_timeout_sec = float(self.get_parameter("tf_timeout_sec").value)
        self._show_window = bool(self.get_parameter("show_window").value)
        self._window_name = str(self.get_parameter("window_name").value)
        self._marker_topic = str(self.get_parameter("marker_topic").value)
        self._publish_markers_enabled = bool(
            self.get_parameter("publish_markers").value)
        self._snapshot_dir = str(self.get_parameter("snapshot_dir").value)
        self._snapshot_interval_sec = float(
            self.get_parameter("snapshot_interval_sec").value)
        self._health_log_sec = float(self.get_parameter("health_log_sec").value)
        use_cpu = bool(self.get_parameter("use_cpu").value)

        if self._show_window and "DISPLAY" not in os.environ:
            self.get_logger().warn(
                "[SEMSEG] DISPLAY is not set; disabling preview window")
            self._show_window = False

        self._device = self._select_device(use_cpu)
        self._processor, self._model = load_model(self._device)

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        self._rgb_msg: Image | None = None
        self._depth_msg: Image | None = None
        self._raw_rgb_msg: Image | None = None
        self._raw_depth_msg: Image | None = None
        self._intrinsics: tuple[float, float, float, float] | None = None
        self._inference_running = False
        self._latest_preview_bgr: np.ndarray | None = None
        self._latest_rgb_bgr: np.ndarray | None = None
        self._latest_segmentation: np.ndarray | None = None
        self._latest_infer_ms: float = 0.0
        self._last_snapshot_time: float = 0.0
        self._snapshot_count: int = 0
        self._last_processed_pair_key: tuple[int, int, int, int] | None = None
        self._last_processed_rgb_key: tuple[int, int] | None = None
        self._stale_drop_count: int = 0
        self._rgbd_count: int = 0
        self._raw_rgb_count: int = 0
        self._raw_depth_count: int = 0
        self._latest_pair_fallback_count: int = 0
        self._rgb_only_count: int = 0
        self._caminfo_count: int = 0
        self._infer_count: int = 0
        self._objects_pub_count: int = 0
        self._last_health_log_time: float = 0.0
        self._last_tf_matrix: np.ndarray | None = None
        self._last_tf_stamp_sec: float | None = None
        self._last_tf_source: str | None = None
        self._last_tf_target: str | None = None
        self._latest_tf_source_frame: str | None = None

        # Create snapshot directory immediately at startup
        try:
            os.makedirs(self._snapshot_dir, exist_ok=True)
        except Exception as exc:
            self.get_logger().warn(f"[SEMSEG] could not create snapshot dir: {exc}")

        self._raw_rgb_sub = self.create_subscription(
            Image, self._rgb_topic, self._raw_rgb_cb, _IMAGE_QOS)
        self._raw_depth_sub = self.create_subscription(
            Image, self._depth_topic, self._raw_depth_cb, _IMAGE_QOS)
        self.create_subscription(
            CameraInfo, self._camera_info_topic, self._caminfo_cb, 1)

        self._overlay_pub = self.create_publisher(
            Image, "/semantic_segmentation/overlay", 1)
        self._label_pub = self.create_publisher(
            Image, "/semantic_segmentation/labels", 1)
        self._objects_pub = self.create_publisher(
            String, "/semantic_segmentation/objects", 10)
        self._marker_pub = self.create_publisher(
            MarkerArray, self._marker_topic, 10)

        self.create_timer(1.0 / max(self._tick_hz, 1e-3), self._tick)
        if self._enable_world_coordinates:
            self.create_timer(0.25, self._refresh_tf_cache)
        if self._show_window:
            self._latest_preview_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(
                self._latest_preview_bgr,
                "Waiting for semantic segmentation frames...",
                (20, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
            self.create_timer(1.0 / 15.0, self._update_window)

        self.get_logger().info(
            f"[SEMSEG] ready device={self._device} "
            f"rgb={self._rgb_topic} depth={self._depth_topic} "
            f"info={self._camera_info_topic} global_frame={self._global_frame} "
            f"world_coords={self._enable_world_coordinates} "
            f"tf_timeout={self._tf_timeout_sec:.2f}s "
            f"show_window={self._show_window} "
            f"snapshot_dir={self._snapshot_dir} "
            f"snapshot_interval={self._snapshot_interval_sec}s "
            f"max_frame_age={self._max_frame_age_sec}s "
            f"sync_queue={self._sync_queue_size} "
            f"allow_rgb_only={self._allow_rgb_only} "
            f"health_log={self._health_log_sec}s"
        )

    def _select_device(self, use_cpu: bool) -> torch.device:
        if use_cpu or not torch.cuda.is_available():
            self.get_logger().info("[SEMSEG] using CPU")
            return torch.device("cpu")
        self.get_logger().info(
            f"[SEMSEG] using CUDA: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda:0")

    def _raw_rgb_cb(self, msg: Image) -> None:
        self._raw_rgb_count += 1
        self._raw_rgb_msg = msg
        rgb = self._decode_rgb(msg)
        if rgb is not None:
            self._latest_rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def _raw_depth_cb(self, msg: Image) -> None:
        self._raw_depth_count += 1
        self._raw_depth_msg = msg
        if msg.header.frame_id:
            self._latest_tf_source_frame = str(msg.header.frame_id)

    def _caminfo_cb(self, msg: CameraInfo) -> None:
        self._caminfo_count += 1
        fx = float(msg.k[0])
        fy = float(msg.k[4])
        cx = float(msg.k[2])
        cy = float(msg.k[5])
        if fx > 0.0 and fy > 0.0:
            self._intrinsics = (fx, fy, cx, cy)

    def _refresh_tf_cache(self) -> None:
        if not self._enable_world_coordinates:
            return
        source = self._latest_tf_source_frame
        if not source:
            return
        try:
            tf_msg = self._tf_buffer.lookup_transform(
                self._global_frame,
                source,
                rclpy.time.Time(),
                timeout=Duration(seconds=max(0.01, min(self._tf_timeout_sec, 0.10))),
            )
        except Exception:
            return
        self._last_tf_matrix = self._transform_to_matrix(tf_msg.transform)
        self._last_tf_stamp_sec = self._stamp_sec_from_ros_time(tf_msg.header.stamp)
        self._last_tf_source = source
        self._last_tf_target = self._global_frame

    def _tick(self) -> None:
        self._maybe_log_health()
        if self._inference_running:
            return

        depth_msg = None
        fallback_pair = self._latest_raw_pair()
        if fallback_pair is not None and self._intrinsics is not None:
            rgb_msg, depth_msg = fallback_pair
            pair_key = self._pair_key(rgb_msg, depth_msg)
            if pair_key == self._last_processed_pair_key:
                return
            self._rgb_msg, self._depth_msg = fallback_pair
            self._rgbd_count += 1
            self._latest_pair_fallback_count += 1
            if self._frame_is_stale(rgb_msg, depth_msg):
                self._rgb_msg = None
                self._depth_msg = None
                return
            self._last_processed_pair_key = pair_key
        elif self._allow_rgb_only and self._raw_rgb_msg is not None:
            rgb_msg = self._raw_rgb_msg
            rgb_key = self._rgb_key(rgb_msg)
            if rgb_key == self._last_processed_rgb_key:
                return
            if self._rgb_frame_is_stale(rgb_msg):
                self._raw_rgb_msg = None
                return
            self._last_processed_rgb_key = rgb_key
            self._rgb_only_count += 1
        else:
            return

        self._inference_running = True
        threading.Thread(
            target=self._infer_once,
            args=(rgb_msg, depth_msg),
            daemon=True,
        ).start()

    def _infer_once(self, rgb_msg: Image, depth_msg: Image | None) -> None:
        try:
            if depth_msg is not None and self._frame_is_stale(rgb_msg, depth_msg):
                return
            if depth_msg is None and self._rgb_frame_is_stale(rgb_msg):
                return
            rgb = self._decode_rgb(rgb_msg)
            depth_m = self._decode_depth(depth_msg) if depth_msg is not None else None
            if rgb is None:
                return

            t0 = time.time()
            seg = segment_frame(
                rgb,
                self._processor,
                self._model,
                self._device,
                target_size=rgb.shape[:2],
                conf_thr=self._conf_thr,
            )
            infer_ms = (time.time() - t0) * 1000.0
            self._latest_segmentation = seg.copy()
            self._latest_infer_ms = infer_ms

            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            overlay = self._make_overlay(bgr)
            self._maybe_save_snapshot(overlay)

            objects = (
                self._summarize_components(seg, depth_m, rgb_msg, depth_msg)
                if depth_m is not None and depth_msg is not None and self._intrinsics is not None
                else []
            )
            self._infer_count += 1
            self._publish_overlay(overlay, rgb_msg)
            self._publish_labels(seg, rgb_msg)
            self._publish_objects(objects, rgb_msg)
            self._publish_markers(objects, rgb_msg)
        except Exception as exc:
            self.get_logger().error(f"[SEMSEG] inference failed: {exc}")
        finally:
            self._inference_running = False

    def _frame_is_stale(self, rgb_msg: Image, depth_msg: Image) -> bool:
        if self._max_frame_age_sec <= 0.0:
            return False
        now_sec = float(self.get_clock().now().nanoseconds) * 1e-9
        rgb_age = now_sec - (
            float(rgb_msg.header.stamp.sec) + float(rgb_msg.header.stamp.nanosec) * 1e-9)
        depth_age = now_sec - (
            float(depth_msg.header.stamp.sec) + float(depth_msg.header.stamp.nanosec) * 1e-9)
        age = max(rgb_age, depth_age)
        if age <= self._max_frame_age_sec:
            return False
        self._stale_drop_count += 1
        self.get_logger().warn(
            f"[SEMSEG] dropping stale RGB-D frame age={age:.2f}s "
            f"(limit={self._max_frame_age_sec:.2f}s, drops={self._stale_drop_count})",
            throttle_duration_sec=5.0,
        )
        return True

    def _rgb_frame_is_stale(self, rgb_msg: Image) -> bool:
        if self._max_frame_age_sec <= 0.0:
            return False
        now_sec = float(self.get_clock().now().nanoseconds) * 1e-9
        age = now_sec - self._stamp_sec(rgb_msg)
        if age <= self._max_frame_age_sec:
            return False
        self._stale_drop_count += 1
        self.get_logger().warn(
            f"[SEMSEG] dropping stale RGB frame age={age:.2f}s "
            f"(limit={self._max_frame_age_sec:.2f}s, drops={self._stale_drop_count})",
            throttle_duration_sec=5.0,
        )
        return True

    def _maybe_log_health(self) -> None:
        if self._health_log_sec <= 0.0:
            return
        now = time.time()
        if now - self._last_health_log_time < self._health_log_sec:
            return
        self._last_health_log_time = now

        missing = []
        has_pair = (
            (self._rgb_msg is not None and self._depth_msg is not None) or
            self._latest_raw_pair() is not None
        )
        has_rgb_only = self._allow_rgb_only and self._raw_rgb_msg is not None
        if not has_pair and not has_rgb_only:
            missing.append("synced_rgbd")
        if self._raw_rgb_msg is None:
            missing.append("raw_rgb")
        if self._raw_depth_msg is None:
            missing.append("raw_depth")
        if self._intrinsics is None:
            missing.append("camera_info")
        if self._inference_running:
            missing.append("inference_busy")

        rgb_age = self._msg_age_sec(self._raw_rgb_msg)
        depth_age = self._msg_age_sec(self._raw_depth_msg)
        if np.isfinite(rgb_age) and rgb_age > 3.0:
            missing.append("fresh_rgb")
        if np.isfinite(depth_age) and depth_age > 3.0 and not self._allow_rgb_only:
            missing.append("fresh_depth")
        elif np.isfinite(depth_age) and depth_age > 3.0 and self._raw_depth_msg is not None:
            missing.append("fresh_depth")

        status = "ok" if not missing else "waiting_for=" + ",".join(missing)
        self.get_logger().info(
            f"[SEMSEG HEALTH] {status} rgbd={self._rgbd_count} "
            f"raw_rgb={self._raw_rgb_count} raw_depth={self._raw_depth_count} "
            f"fallback={self._latest_pair_fallback_count} "
            f"rgb_only={self._rgb_only_count} "
            f"rgb_age={rgb_age:.1f}s depth_age={depth_age:.1f}s "
            f"caminfo={self._caminfo_count} infer={self._infer_count} "
            f"objects_msgs={self._objects_pub_count} stale_drops={self._stale_drop_count}")

    def _latest_raw_pair(self) -> tuple[Image, Image] | None:
        if self._raw_rgb_msg is None or self._raw_depth_msg is None:
            return None
        rgb_t = self._stamp_sec(self._raw_rgb_msg)
        depth_t = self._stamp_sec(self._raw_depth_msg)
        max_dt = max(self._max_sync_slop_sec, 0.25)
        if abs(rgb_t - depth_t) > max_dt:
            return None
        return self._raw_rgb_msg, self._raw_depth_msg

    @staticmethod
    def _stamp_sec(msg: Image) -> float:
        return float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9

    @staticmethod
    def _pair_key(rgb_msg: Image, depth_msg: Image) -> tuple[int, int, int, int]:
        return (
            int(rgb_msg.header.stamp.sec),
            int(rgb_msg.header.stamp.nanosec),
            int(depth_msg.header.stamp.sec),
            int(depth_msg.header.stamp.nanosec),
        )

    @staticmethod
    def _rgb_key(rgb_msg: Image) -> tuple[int, int]:
        return (
            int(rgb_msg.header.stamp.sec),
            int(rgb_msg.header.stamp.nanosec),
        )

    def _msg_age_sec(self, msg: Image | None) -> float:
        if msg is None:
            return float("nan")
        now_sec = float(self.get_clock().now().nanoseconds) * 1e-9
        return max(0.0, now_sec - self._stamp_sec(msg))

    def _maybe_save_snapshot(self, overlay: np.ndarray) -> None:
        if self._snapshot_interval_sec <= 0.0:
            return
        now = time.time()
        if now - self._last_snapshot_time < self._snapshot_interval_sec:
            return
        try:
            from datetime import datetime
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._snapshot_count += 1
            fname = os.path.join(
                self._snapshot_dir,
                f"semantic_{ts}_{self._snapshot_count:04d}.jpg",
            )
            cv2.imwrite(fname, overlay)
            self._last_snapshot_time = now
        except Exception as exc:
            self.get_logger().warn(f"[SEMSEG] snapshot save failed: {exc}")

    def _summarize_components(
        self,
        seg: np.ndarray,
        depth_m: np.ndarray,
        rgb_msg: Image,
        depth_msg: Image,
    ) -> list[dict]:
        fx, fy, cx, cy = self._intrinsics
        transform = None
        tf_debug = {"mode": "disabled", "stamp_sec": None}
        if self._enable_world_coordinates:
            transform, tf_debug = self._lookup_transform_matrix(
                self._global_frame, depth_msg.header.frame_id,
                depth_msg.header.stamp)

        rgb_depth_dt_ms = 1000.0 * abs(
            self._stamp_sec(rgb_msg) - self._stamp_sec(depth_msg))

        results: list[dict] = []
        for class_id in range(1, len(REDUCED_LABELS)):
            mask = (seg == class_id).astype(np.uint8)
            if int(mask.sum()) < self._min_component_area_px:
                continue

            n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                mask, connectivity=8)
            for idx in range(1, n_labels):
                area = int(stats[idx, cv2.CC_STAT_AREA])
                if area < self._min_component_area_px:
                    continue

                ys_all, xs_all = np.where(labels == idx)
                if xs_all.size == 0:
                    continue

                depths_all = depth_m[ys_all, xs_all]
                valid_all = np.isfinite(depths_all)
                valid_all &= depths_all >= self._min_depth_m
                valid_all &= depths_all <= self._max_depth_m
                if not np.any(valid_all):
                    continue

                depth_p10, depth_p50, depth_p90 = np.percentile(
                    depths_all[valid_all], [10, 50, 90])
                depth_spread = float(depth_p90 - depth_p10)

                xs = xs_all[::self._summary_stride_px]
                ys = ys_all[::self._summary_stride_px]
                if xs.size == 0:
                    continue

                depths = depth_m[ys, xs]
                valid = np.isfinite(depths)
                valid &= depths >= self._min_depth_m
                valid &= depths <= self._max_depth_m
                if not np.any(valid):
                    continue

                xs = xs[valid]
                ys = ys[valid]
                depths = depths[valid]

                z = float(np.median(depths))
                u = float(np.median(xs))
                v = float(np.median(ys))
                x = (u - cx) * z / fx
                y = (v - cy) * z / fy
                camera_xyz = np.array([x, y, z], dtype=np.float64)

                world_xyz = None
                if transform is not None:
                    world_xyz = (
                        transform[:3, :3] @ camera_xyz + transform[:3, 3]
                    ).tolist()

                x0 = int(stats[idx, cv2.CC_STAT_LEFT])
                y0 = int(stats[idx, cv2.CC_STAT_TOP])
                w = int(stats[idx, cv2.CC_STAT_WIDTH])
                h = int(stats[idx, cv2.CC_STAT_HEIGHT])

                obj_out = {
                    "class_id": class_id,
                    "class_name": REDUCED_LABELS[class_id],
                    "area_px": area,
                    "bbox_xyxy": [x0, y0, x0 + w, y0 + h],
                    "camera_xyz_m": [round(val, 4) for val in camera_xyz.tolist()],
                    "world_xyz_m": (
                        [round(float(val), 4) for val in world_xyz]
                        if world_xyz is not None else None
                    ),
                    "frame_id": (
                        self._global_frame if world_xyz is not None
                        else depth_msg.header.frame_id
                    ),
                    "debug": {
                        "rgb_depth_dt_ms": round(float(rgb_depth_dt_ms), 3),
                        "mask_px": int(area),
                        "depth_p10_m": round(float(depth_p10), 4),
                        "depth_p50_m": round(float(depth_p50), 4),
                        "depth_p90_m": round(float(depth_p90), 4),
                        "depth_spread_m": round(depth_spread, 4),
                        "tf_mode": tf_debug.get("mode"),
                        "tf_stamp_sec": (
                            round(float(tf_debug["stamp_sec"]), 6)
                            if tf_debug.get("stamp_sec") is not None else None
                        ),
                    },
                }
                results.append(obj_out)

                self.get_logger().info(
                    "[SEMSEG DEBUG] "
                    f"{REDUCED_LABELS[class_id]} area={area}px "
                    f"rgb-depth={rgb_depth_dt_ms:.1f}ms "
                    f"depth[p10,p50,p90]=({depth_p10:.2f},{depth_p50:.2f},{depth_p90:.2f})m "
                    f"spread={depth_spread:.2f}m "
                    f"camera_xyz=({camera_xyz[0]:.2f},{camera_xyz[1]:.2f},{camera_xyz[2]:.2f}) "
                    f"world_xyz="
                    + (
                        f"({world_xyz[0]:.2f},{world_xyz[1]:.2f},{world_xyz[2]:.2f}) "
                        if world_xyz is not None else "None "
                    )
                    + f"tf={tf_debug.get('mode')}@{tf_debug.get('stamp_sec')}"
                )
        return results

    def _lookup_transform_matrix(self, target: str, source: str, stamp):
        timeout = Duration(seconds=max(0.01, self._tf_timeout_sec))
        try:
            tf_msg = self._tf_buffer.lookup_transform(
                target,
                source,
                rclpy.time.Time(seconds=stamp.sec, nanoseconds=stamp.nanosec),
                timeout=timeout,
            )
            matrix = self._transform_to_matrix(tf_msg.transform)
            stamp_sec = self._stamp_sec_from_ros_time(tf_msg.header.stamp)
            self._last_tf_matrix = matrix
            self._last_tf_stamp_sec = stamp_sec
            self._last_tf_source = source
            self._last_tf_target = target
            return matrix, {
                "mode": "stamped",
                "stamp_sec": stamp_sec,
            }
        except Exception as exc:
            self.get_logger().warn(
                f"[SEMSEG] stamped TF {source}->{target} unavailable: {exc}; "
                f"falling back to latest transform",
                throttle_duration_sec=5.0,
            )
        try:
            tf_msg = self._tf_buffer.lookup_transform(
                target,
                source,
                rclpy.time.Time(),
                timeout=timeout,
            )
            matrix = self._transform_to_matrix(tf_msg.transform)
            stamp_sec = self._stamp_sec_from_ros_time(tf_msg.header.stamp)
            self._last_tf_matrix = matrix
            self._last_tf_stamp_sec = stamp_sec
            self._last_tf_source = source
            self._last_tf_target = target
            return matrix, {
                "mode": "latest",
                "stamp_sec": stamp_sec,
            }
        except Exception as exc:
            self.get_logger().warn(
                f"[SEMSEG] TF {source}->{target} unavailable: {exc}",
                throttle_duration_sec=5.0,
            )
            if (
                self._last_tf_matrix is not None
                and self._last_tf_source == source
                and self._last_tf_target == target
            ):
                return self._last_tf_matrix.copy(), {
                    "mode": "cached",
                    "stamp_sec": self._last_tf_stamp_sec,
                }
            return None, {"mode": "unavailable", "stamp_sec": None}

    @staticmethod
    def _transform_to_matrix(transform) -> np.ndarray:
        tr = transform.translation
        q = transform.rotation
        x, y, z, w = q.x, q.y, q.z, q.w
        rot = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ], dtype=np.float64)
        out = np.eye(4, dtype=np.float64)
        out[:3, :3] = rot
        out[:3, 3] = [tr.x, tr.y, tr.z]
        return out

    @staticmethod
    def _stamp_sec_from_ros_time(stamp) -> float:
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    def _publish_overlay(self, image_bgr: np.ndarray, src_msg: Image) -> None:
        msg = Image()
        msg.header = src_msg.header
        msg.height, msg.width = image_bgr.shape[:2]
        msg.encoding = "bgr8"
        msg.step = msg.width * 3
        msg.data = image_bgr.tobytes()
        self._overlay_pub.publish(msg)

    def _publish_labels(self, labels: np.ndarray, src_msg: Image) -> None:
        msg = Image()
        msg.header = src_msg.header
        msg.height, msg.width = labels.shape
        msg.encoding = "mono8"
        msg.step = msg.width
        msg.data = labels.astype(np.uint8).tobytes()
        self._label_pub.publish(msg)

    def _publish_objects(self, objects: list[dict], src_msg: Image) -> None:
        msg = String()
        msg.data = json.dumps({
            "stamp_sec": src_msg.header.stamp.sec,
            "stamp_nanosec": src_msg.header.stamp.nanosec,
            "global_frame": (
                self._global_frame if self._enable_world_coordinates
                else src_msg.header.frame_id
            ),
            "world_coordinates_enabled": self._enable_world_coordinates,
            "objects": objects,
        })
        self._objects_pub.publish(msg)
        self._objects_pub_count += 1
        if objects or self._objects_pub_count == 1 or self._objects_pub_count % 10 == 0:
            self.get_logger().info(
                f"[SEMSEG OBJECTS] published msg #{self._objects_pub_count} "
                f"objects={len(objects)} frame={src_msg.header.frame_id} "
                f"world_coords={self._enable_world_coordinates}")

    def _publish_markers(self, objects: list[dict], src_msg: Image) -> None:
        if not self._publish_markers_enabled:
            return

        marker_frame = (
            self._global_frame if self._enable_world_coordinates
            else src_msg.header.frame_id
        )
        now = self.get_clock().now().to_msg()
        markers = MarkerArray()

        delete_all = Marker()
        delete_all.header.frame_id = marker_frame
        delete_all.header.stamp = now
        delete_all.action = Marker.DELETEALL
        markers.markers.append(delete_all)

        marker_id = 0
        for obj in objects:
            coords = obj.get("world_xyz_m")
            if coords is None:
                coords = obj.get("camera_xyz_m")
            if coords is None or len(coords) != 3:
                continue

            colour = self._marker_colour(obj.get("class_id", 0))

            sphere = Marker()
            sphere.header.frame_id = marker_frame
            sphere.header.stamp = now
            sphere.ns = "semantic_segmentation"
            sphere.id = marker_id
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose.position.x = float(coords[0])
            sphere.pose.position.y = float(coords[1])
            sphere.pose.position.z = float(coords[2])
            sphere.pose.orientation.w = 1.0
            sphere.scale.x = 0.18
            sphere.scale.y = 0.18
            sphere.scale.z = 0.18
            sphere.color.r = colour[2] / 255.0
            sphere.color.g = colour[1] / 255.0
            sphere.color.b = colour[0] / 255.0
            sphere.color.a = 0.85
            sphere.lifetime = Duration(seconds=1.0).to_msg()
            markers.markers.append(sphere)
            marker_id += 1

            label = Marker()
            label.header = sphere.header
            label.ns = "semantic_segmentation_labels"
            label.id = marker_id
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position.x = float(coords[0])
            label.pose.position.y = float(coords[1])
            label.pose.position.z = float(coords[2]) + 0.18
            label.pose.orientation.w = 1.0
            label.scale.z = 0.12
            label.color.r = 1.0
            label.color.g = 1.0
            label.color.b = 1.0
            label.color.a = 0.95
            label.text = (
                f"{obj.get('class_name', 'unknown')} "
                f"({coords[0]:.2f}, {coords[1]:.2f}, {coords[2]:.2f})"
            )
            label.lifetime = Duration(seconds=1.0).to_msg()
            markers.markers.append(label)
            marker_id += 1

        self._marker_pub.publish(markers)

    @staticmethod
    def _marker_colour(class_id: int) -> tuple[int, int, int]:
        if 0 <= int(class_id) < len(COLOURS):
            return tuple(int(v) for v in COLOURS[int(class_id)])
        return (255, 255, 255)

    def _update_window(self) -> None:
        if not self._show_window:
            return
        frame = self._latest_preview_bgr
        if self._latest_rgb_bgr is not None:
            frame = self._make_overlay(self._latest_rgb_bgr)
        if frame is None:
            return
        self._latest_preview_bgr = frame
        cv2.imshow(self._window_name, frame)
        cv2.waitKey(1)

    def _make_overlay(self, bgr: np.ndarray) -> np.ndarray:
        frame = bgr.copy()
        if self._latest_segmentation is not None:
            if self._latest_segmentation.shape[:2] == frame.shape[:2]:
                frame = overlay_segmentation(frame, self._latest_segmentation)
                frame = draw_class_boxes(frame, self._latest_segmentation)
                frame = draw_legend(frame, self._latest_segmentation)
        cv2.putText(
            frame,
            f"Seg {self._latest_infer_ms:.0f} ms",
            (10, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
        )
        return frame

    def _decode_rgb(self, msg: Image) -> np.ndarray | None:
        h = int(msg.height)
        w = int(msg.width)
        if msg.encoding not in ("rgb8", "bgr8"):
            self.get_logger().warn(
                f"[SEMSEG] unsupported RGB encoding: {msg.encoding}",
                throttle_duration_sec=5.0,
            )
            return None
        row_stride = int(msg.step)
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, row_stride)
        arr = arr[:, :w * 3].reshape(h, w, 3)
        if msg.encoding == "rgb8":
            return arr.copy()
        return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)

    def _decode_depth(self, msg: Image) -> np.ndarray | None:
        h = int(msg.height)
        w = int(msg.width)
        if msg.encoding == "16UC1":
            row_stride = int(msg.step) // np.dtype(np.uint16).itemsize
            arr = np.frombuffer(msg.data, dtype=np.uint16).reshape(h, row_stride)
            arr = arr[:, :w]
            return arr.astype(np.float32) * 0.001
        if msg.encoding == "32FC1":
            row_stride = int(msg.step) // np.dtype(np.float32).itemsize
            arr = np.frombuffer(msg.data, dtype=np.float32).reshape(h, row_stride)
            return arr[:, :w].copy()
        self.get_logger().warn(
            f"[SEMSEG] unsupported depth encoding: {msg.encoding}",
            throttle_duration_sec=5.0,
        )
        return None

def main(args=None) -> None:
    rclpy.init(args=args)
    node = SemanticSegmentationRGBDNode()
    try:
        rclpy.spin(node)
    finally:
        if getattr(node, "_show_window", False):
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
