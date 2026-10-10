#!/usr/bin/env python3
"""
SemanticMapperNode — helper module (NOT a standalone ROS node).

Piggybacks on the GridmapNode executor.  Each inference tick:
  1. Runs LRASPP segmentation on the latest RGB frame (in a daemon thread).
  2. Back-projects detections to world coordinates via depth + TF.
  3. Calls region.add_semantic_object() on the current active region.

Subscriptions are created on the host gridmap_node so they share its executor.

Topics (all configurable via __init__ kwargs):
  rgb_topic         /camera/color/image_raw
  depth_topic       /camera/aligned_depth_to_color/image_raw
  camera_info_topic /camera/aligned_depth_to_color/camera_info

Camera TF:
  If publish_static_camera_tf=True (default) a static transform
  base_link → camera_color_optical_frame is broadcast once at startup.
"""

import math
import os
import threading
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped, TransformStamped
from rclpy.duration import Duration
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from visualization_msgs.msg import Marker, MarkerArray
import tf2_ros


FURNITURE_CLASSES = {
    9:  "chair",
    11: "diningtable",
    18: "sofa",
    20: "tvmonitor",
}

_IMAGE_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)


class SemanticMapperNode:
    """
    Helper module that runs inside the GridmapNode executor.

    Args:
        gridmap_node:             host GridmapNode (provides executor, logger, tf_buffer, .region)
        rate_hz:                  inference tick frequency (default 1 Hz)
        rgb_topic:                RGB image topic
        depth_topic:              aligned depth topic
        camera_info_topic:        camera intrinsics topic
        global_frame:             world frame for 3-D poses (default "odom")
        camera_frame:             camera optical frame (default "camera_color_optical_frame")
        base_frame:               robot base frame (default "base_link")
        min_pixels:               minimum segmented pixels to accept a detection (default 300)
        min_confidence:           minimum mean class probability (default 0.65)
        min_depth_m:              minimum valid depth in metres (default 0.15)
        max_depth_m:              maximum valid depth in metres (default 8.0)
        tf_timeout_sec:           TF lookup timeout (default 0.15)
        publish_static_camera_tf: broadcast a static base_link→camera TF on startup (default True)
        camera_offset_xyz:        (x, y, z) offset of camera from base_link in metres
        snapshot_dir:             where to save debug PNG snapshots (default mntfields_model/semantic_snapshots)
        snapshot_every_n_ticks:   save a snapshot every N inference ticks (0 = disabled, default 60)
    """

    def __init__(
        self,
        gridmap_node,
        rate_hz: float = 1.0,
        rgb_topic: str = "/camera/color/image_raw",
        depth_topic: str = "/camera/aligned_depth_to_color/image_raw",
        camera_info_topic: str = "/camera/aligned_depth_to_color/camera_info",
        global_frame: str = "odom",
        camera_frame: str = "camera_color_optical_frame",
        base_frame: str = "base_link",
        min_pixels: int = 300,
        min_confidence: float = 0.65,
        min_depth_m: float = 0.15,
        max_depth_m: float = 8.0,
        tf_timeout_sec: float = 0.15,
        publish_static_camera_tf: bool = True,
        camera_offset_xyz: Tuple[float, float, float] = (-0.05, 0.0, 0.20),
        snapshot_dir: Optional[str] = None,
        snapshot_every_n_ticks: int = 60,
    ):
        self.node = gridmap_node
        self.global_frame = global_frame
        self.camera_frame = camera_frame
        self.base_frame = base_frame
        self.min_pixels = min_pixels
        self.min_confidence = min_confidence
        self.min_depth_m = min_depth_m
        self.max_depth_m = max_depth_m
        self.tf_timeout = Duration(seconds=tf_timeout_sec)
        self.camera_offset_xyz = camera_offset_xyz

        self.snapshot_every_n_ticks = snapshot_every_n_ticks
        if snapshot_dir is None:
            snapshot_dir = os.path.join(
                getattr(gridmap_node, "model_dir",
                        "/home/mayank/turtlebot3_ws/mntfields_model"),
                "semantic_snapshots",
            )
        self.snapshot_dir = snapshot_dir
        os.makedirs(self.snapshot_dir, exist_ok=True)

        self.bridge = CvBridge()

        # TF buffer — re-use the host node's buffer when available
        if hasattr(gridmap_node, "tf_buffer") and gridmap_node.tf_buffer is not None:
            self._tf_buffer = gridmap_node.tf_buffer
            self._own_tf_listener = False
        else:
            self._tf_buffer = tf2_ros.Buffer()
            self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, gridmap_node)
            self._own_tf_listener = True

        # Camera intrinsics
        self.fx: Optional[float] = None
        self.fy: Optional[float] = None
        self.cx: Optional[float] = None
        self.cy: Optional[float] = None

        # Latest raw messages (decoded only in the inference thread)
        self.rgb_msg: Optional[Image] = None
        self.depth_msg: Optional[Image] = None

        # Threading
        self._inference_running = False
        self._tick_count = 0

        # Debug publishers on the host node
        self._marker_pub = gridmap_node.create_publisher(
            MarkerArray, "/semantic_mapper/markers", 10
        )
        self._debug_img_pub = gridmap_node.create_publisher(
            Image, "/semantic_mapper/debug_image", 10
        )

        # Static TF broadcaster
        if publish_static_camera_tf:
            self._static_tf_broadcaster = tf2_ros.StaticTransformBroadcaster(gridmap_node)
            self._publish_camera_static_tf()

        # Subscriptions on the host node executor
        gridmap_node.create_subscription(Image, rgb_topic,         self._rgb_cb,      _IMAGE_QOS)
        gridmap_node.create_subscription(Image, depth_topic,       self._depth_cb,    _IMAGE_QOS)
        gridmap_node.create_subscription(CameraInfo, camera_info_topic, self._caminfo_cb, 1)

        # Model — LRASPP MobileNetV3-Large (much faster than DeepLabV3)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        from torchvision import transforms
        from torchvision.models.segmentation import (
            lraspp_mobilenet_v3_large,
            LRASPP_MobileNet_V3_Large_Weights,
        )
        weights = LRASPP_MobileNet_V3_Large_Weights.DEFAULT
        self.model = lraspp_mobilenet_v3_large(weights=weights).to(self.device).eval()
        self.transform = transforms.ToTensor()

        # Inference tick timer on host executor
        gridmap_node.create_timer(1.0 / max(rate_hz, 1e-3), self._tick)

        gridmap_node.get_logger().info(
            f"[SEMANTIC] Ready — device={self.device} rate={rate_hz:.1f} Hz "
            f"frame={global_frame} camera_frame={camera_frame} "
            f"min_confidence={min_confidence}"
        )

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _rgb_cb(self, msg: Image) -> None:
        self.rgb_msg = msg
        # Passthrough to debug topic at camera framerate (no decode here)
        if self._debug_img_pub.get_subscription_count() > 0:
            self._debug_img_pub.publish(msg)

    def _depth_cb(self, msg: Image) -> None:
        self.depth_msg = msg

    def _caminfo_cb(self, msg: CameraInfo) -> None:
        self.fx = float(msg.k[0])
        self.fy = float(msg.k[4])
        self.cx = float(msg.k[2])
        self.cy = float(msg.k[5])

    # ── Inference tick ────────────────────────────────────────────────────────

    def _tick(self) -> None:
        self._tick_count += 1
        if self._inference_running:
            return  # previous inference still running

        if not self._intrinsics_ready():
            if self._tick_count % 20 == 0:
                self.node.get_logger().info(
                    f"[SEMANTIC] waiting — rgb={self.rgb_msg is not None} "
                    f"depth={self.depth_msg is not None} K={self._intrinsics_ready()}"
                )
            return

        if self.rgb_msg is None or self.depth_msg is None:
            return

        # Decode once per tick (not in every callback)
        try:
            rgb_bgr        = self.bridge.imgmsg_to_cv2(self.rgb_msg, desired_encoding="bgr8")
            depth          = self.bridge.imgmsg_to_cv2(self.depth_msg)
        except Exception as e:
            self.node.get_logger().warn(f"[SEMANTIC] decode error: {e}", throttle_duration_sec=5.0)
            return

        depth_msg_snap = self.depth_msg
        rgb_msg_snap   = self.rgb_msg
        tick_n         = self._tick_count

        self._inference_running = True
        threading.Thread(
            target=self._infer_and_store,
            args=(rgb_bgr, depth, depth_msg_snap, rgb_msg_snap, tick_n),
            daemon=True,
        ).start()

    def _infer_and_store(
        self,
        rgb_bgr: np.ndarray,
        depth: np.ndarray,
        depth_msg_snap: Image,
        rgb_msg_snap: Image,
        tick_n: int,
    ) -> None:
        try:
            img_rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
            tensor = self.transform(img_rgb).unsqueeze(0).to(self.device)
            with torch.no_grad():
                out   = self.model(tensor)["out"][0]
                probs = torch.softmax(out, dim=0)
                mask  = out.argmax(0).cpu().numpy().astype(np.int32)

            detections = self._extract_detections(mask, probs)
            debug_img  = rgb_bgr.copy()
            markers    = MarkerArray()
            # Clear old markers
            del_m = Marker()
            del_m.header.frame_id = self.global_frame
            del_m.header.stamp    = self.node.get_clock().now().to_msg()
            del_m.action          = Marker.DELETEALL
            markers.markers.append(del_m)

            if not detections:
                cv2.putText(debug_img, "No furniture detected",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                self._publish_debug(debug_img, markers, rgb_msg_snap)
                return

            for det_idx, det in enumerate(detections):
                depth_m = self._depth_at_bbox(det["bbox"], depth, depth_msg_snap)
                x1, y1, x2, y2 = det["bbox"]
                label = f"{det['class']} {det['confidence']:.2f}"

                cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                if depth_m is None:
                    cv2.putText(debug_img, f"{label} depth=invalid",
                                (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 0, 255), 2)
                    continue

                point_cam = self._backproject(det["center"], depth_m, depth_msg_snap)
                point_world = self._transform_to_world(
                    point_cam, depth_msg_snap.header.frame_id, depth_msg_snap.header.stamp
                )

                if point_world is None:
                    cv2.putText(debug_img, f"{label} tf=missing",
                                (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 165, 255), 2)
                    continue

                wx = float(point_world.point.x)
                wy = float(point_world.point.y)

                # Store in the current active region
                region = getattr(self.node, "region", None)
                if region is not None:
                    region.add_semantic_object(det["class"], [wx, wy])
                    self.node.get_logger().info(
                        f"[SEMANTIC] {det['class']} @ ({wx:.2f},{wy:.2f}) "
                        f"depth={depth_m:.2f}m → region {region.version}",
                        throttle_duration_sec=2.0,
                    )

                cv2.putText(
                    debug_img,
                    f"{label} ({wx:.1f},{wy:.1f})",
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2,
                )
                markers.markers.append(self._make_marker(det_idx, det["class"], point_world))

            self._publish_debug(debug_img, markers, rgb_msg_snap)

            if self.snapshot_every_n_ticks > 0 and tick_n % self.snapshot_every_n_ticks == 0:
                self._save_snapshot(debug_img, tick_n)

        except Exception as e:
            self.node.get_logger().warn(f"[SEMANTIC] inference error: {e}")
        finally:
            self._inference_running = False

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _intrinsics_ready(self) -> bool:
        return (
            self.fx is not None and self.fy is not None
            and self.cx is not None and self.cy is not None
            and self.fx > 1e-6 and self.fy > 1e-6
        )

    def _extract_detections(self, mask: np.ndarray, probs: torch.Tensor) -> list:
        detections = []
        for class_id, class_name in FURNITURE_CLASSES.items():
            ys, xs = np.where(mask == class_id)
            if xs.size < self.min_pixels:
                continue
            x1, x2 = int(xs.min()), int(xs.max())
            y1, y2 = int(ys.min()), int(ys.max())
            confidence = float(probs[class_id, ys, xs].mean().item())
            if confidence < self.min_confidence:
                continue
            detections.append({
                "class":      class_name,
                "bbox":       (x1, y1, x2, y2),
                "center":     ((x1 + x2) // 2, (y1 + y2) // 2),
                "confidence": confidence,
            })
        return detections

    def _depth_at_bbox(
        self,
        bbox: Tuple[int, int, int, int],
        depth: np.ndarray,
        depth_msg_snap: Image,
    ) -> Optional[float]:
        x1, y1, x2, y2 = bbox
        crop = depth[y1:y2 + 1, x1:x2 + 1].astype(np.float32)
        if crop.size == 0:
            return None
        if depth_msg_snap.encoding == "16UC1":
            crop *= 0.001
        valid = np.isfinite(crop) & (crop > self.min_depth_m) & (crop < self.max_depth_m)
        vals = crop[valid]
        return float(np.median(vals)) if vals.size > 0 else None

    def _backproject(
        self, pixel: Tuple[int, int], depth_m: float, depth_msg_snap: Image
    ) -> PointStamped:
        u, v = pixel
        x = (float(u) - float(self.cx)) * depth_m / float(self.fx)
        y = (float(v) - float(self.cy)) * depth_m / float(self.fy)
        pt = PointStamped()
        pt.header = depth_msg_snap.header
        pt.point.x = x
        pt.point.y = y
        pt.point.z = depth_m
        return pt

    def _transform_to_world(
        self, point_cam: PointStamped, source_frame: str, stamp
    ) -> Optional[PointStamped]:
        try:
            tf = self._tf_buffer.lookup_transform(
                self.global_frame, source_frame, stamp, timeout=self.tf_timeout
            )
        except Exception:
            try:
                tf = self._tf_buffer.lookup_transform(
                    self.global_frame, source_frame, Time()
                )
            except Exception as e:
                self.node.get_logger().warn(
                    f"[SEMANTIC] TF {self.global_frame}←{source_frame} failed: {e}",
                    throttle_duration_sec=2.0,
                )
                return None
        try:
            return tf2_ros.TransformRegistration().get(type(point_cam))(point_cam, tf)
        except Exception:
            import tf2_geometry_msgs  # noqa: F401
            return tf2_geometry_msgs.do_transform_point(point_cam, tf)

    def _make_marker(self, marker_id: int, label: str, point: PointStamped) -> Marker:
        m = Marker()
        m.header      = point.header
        m.ns          = "semantic_mapper"
        m.id          = marker_id
        m.type        = Marker.SPHERE
        m.action      = Marker.ADD
        m.pose.position   = point.point
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 0.20
        m.color.a = 0.9
        m.color.r = 0.1
        m.color.g = 0.9
        m.color.b = 0.2
        m.text = label
        return m

    def _publish_debug(
        self, debug_img: np.ndarray, markers: MarkerArray, rgb_msg_snap: Optional[Image]
    ) -> None:
        self._marker_pub.publish(markers)
        if self._debug_img_pub.get_subscription_count() > 0:
            msg = self.bridge.cv2_to_imgmsg(debug_img, encoding="bgr8")
            if rgb_msg_snap is not None:
                msg.header = rgb_msg_snap.header
            self._debug_img_pub.publish(msg)

    def _save_snapshot(self, debug_img: np.ndarray, tick_n: int) -> None:
        path = os.path.join(self.snapshot_dir, f"semantic_{tick_n:06d}.png")
        ok = cv2.imwrite(path, debug_img)
        self.node.get_logger().info(f"[SEMANTIC SNAPSHOT] write={ok} → {path}")

    def _publish_camera_static_tf(self) -> None:
        tf_msg = TransformStamped()
        tf_msg.header.stamp    = self.node.get_clock().now().to_msg()
        tf_msg.header.frame_id = self.base_frame
        tf_msg.child_frame_id  = self.camera_frame
        ox, oy, oz = self.camera_offset_xyz
        tf_msg.transform.translation.x = ox
        tf_msg.transform.translation.y = oy
        tf_msg.transform.translation.z = oz
        # base_link: x forward, y left, z up
        # optical frame: x right, y down, z forward  →  RPY = (-π/2, 0, -π/2)
        qx, qy, qz, qw = self._quaternion_from_euler(-math.pi / 2.0, 0.0, -math.pi / 2.0)
        tf_msg.transform.rotation.x = qx
        tf_msg.transform.rotation.y = qy
        tf_msg.transform.rotation.z = qz
        tf_msg.transform.rotation.w = qw
        self._static_tf_broadcaster.sendTransform(tf_msg)
        self.node.get_logger().info(
            f"[SEMANTIC] static TF {self.base_frame}→{self.camera_frame} "
            f"xyz=({ox:.2f},{oy:.2f},{oz:.2f})"
        )

    @staticmethod
    def _quaternion_from_euler(roll: float, pitch: float, yaw: float):
        cr, sr = math.cos(roll * 0.5),  math.sin(roll * 0.5)
        cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
        cy, sy = math.cos(yaw * 0.5),   math.sin(yaw * 0.5)
        return (
            sr * cp * cy - cr * sp * sy,   # qx
            cr * sp * cy + sr * cp * sy,   # qy
            cr * cp * sy - sr * sp * cy,   # qz
            cr * cp * cy + sr * sp * sy,   # qw
        )
