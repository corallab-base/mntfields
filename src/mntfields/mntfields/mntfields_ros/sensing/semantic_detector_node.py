"""
semantic_detector_node.py — Standalone ROS2 node for furniture detection.

Subscribes to compressed RGB + LaserScan + CameraInfo + TF.
Runs RetinaNet object detection, projects LiDAR into the image,
fuses 2D boxes with LiDAR points for map-frame anchoring,
and maintains a temporal object tracker.

Publishes:
    /semantic/tracked_objects  (String, JSON) — confirmed + candidate tracks
    /semantic/detections       (String, JSON) — per-frame raw detections
    /semantic/debug_image      (Image)        — annotated image
    /semantic/markers          (MarkerArray)  — RViz object markers
"""

import json
import math
import threading
import time

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
from torchvision.models.detection import (
    retinanet_resnet50_fpn,
    RetinaNet_ResNet50_FPN_Weights,
)

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.duration import Duration

from sensor_msgs.msg import CompressedImage, LaserScan, CameraInfo, Image
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

import tf2_ros

from mntfields.mntfields_ros.sensing.lidar_projector import LidarProjector
from mntfields.mntfields_ros.sensing.object_tracker import ObjectTracker


# COCO class IDs for target furniture
_TARGET_CLASSES = {
    62: "chair",
    63: "couch",
    67: "dining_table",
    72: "tv",
}

# Colours for debug drawing (BGR)
_CLASS_COLOURS = {
    "chair":        (0, 200, 0),
    "couch":        (200, 100, 0),
    "dining_table": (0, 100, 200),
    "tv":           (200, 0, 200),
}


class SemanticDetectorNode(Node):
    """Laptop-side semantic detection: RetinaNet + LiDAR fusion + tracking."""

    def __init__(self):
        super().__init__("semantic_detector")

        # ── Parameters ───────────────────────────────────────────────────
        self.declare_parameter("detection_rate_hz", 2.0)
        self.declare_parameter("score_threshold", 0.45)
        self.declare_parameter("min_lidar_pts_in_box", 2)
        self.declare_parameter("confirm_threshold", 5)
        self.declare_parameter("lost_threshold", 30)
        self.declare_parameter("association_radius_m", 0.6)
        self.declare_parameter("camera_image_topic",
                               "/camera/color/image_raw/compressed")
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("camera_info_topic",
                               "/camera/color/camera_info")
        self.declare_parameter("laser_frame", "base_scan")
        self.declare_parameter("camera_frame", "camera_color_optical_frame")
        self.declare_parameter("map_frame", "map")

        rate_hz = self.get_parameter("detection_rate_hz").value
        self._score_thr = self.get_parameter("score_threshold").value
        self._min_lidar = int(self.get_parameter("min_lidar_pts_in_box").value)
        cam_topic = self.get_parameter("camera_image_topic").value
        scan_topic = self.get_parameter("scan_topic").value
        caminfo_topic = self.get_parameter("camera_info_topic").value
        self._laser_frame = self.get_parameter("laser_frame").value
        self._camera_frame = self.get_parameter("camera_frame").value
        self._map_frame = self.get_parameter("map_frame").value

        # ── TF ───────────────────────────────────────────────────────────
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ── Model ────────────────────────────────────────────────────────
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self.get_logger().info(
            f"[SEMANTIC] Loading RetinaNet on {self._device} …")
        self._model = retinanet_resnet50_fpn(
            weights=RetinaNet_ResNet50_FPN_Weights.COCO_V1)
        self._model.to(self._device)
        self._model.eval()
        self.get_logger().info("[SEMANTIC] RetinaNet ready.")

        # ── Sub-components ───────────────────────────────────────────────
        self._projector = LidarProjector()
        self._tracker = ObjectTracker(
            confirm_threshold=int(
                self.get_parameter("confirm_threshold").value),
            lost_threshold=int(self.get_parameter("lost_threshold").value),
            association_radius_m=float(
                self.get_parameter("association_radius_m").value),
        )

        # ── Subscribers ──────────────────────────────────────────────────
        best_effort = QoSProfile(
            history=HistoryPolicy.KEEP_LAST, depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT)

        self.create_subscription(
            CompressedImage, cam_topic, self._image_cb, best_effort)
        self.create_subscription(
            LaserScan, scan_topic, self._scan_cb, best_effort)
        self.create_subscription(
            CameraInfo, caminfo_topic, self._caminfo_cb, 1)

        # ── Publishers ───────────────────────────────────────────────────
        self._det_pub = self.create_publisher(
            String, "/semantic/detections", 10)
        self._track_pub = self.create_publisher(
            String, "/semantic/tracked_objects", 10)
        self._debug_pub = self.create_publisher(
            Image, "/semantic/debug_image", 1)
        self._marker_pub = self.create_publisher(
            MarkerArray, "/semantic/markers", 10)

        # ── Timer ────────────────────────────────────────────────────────
        self.create_timer(1.0 / rate_hz, self._tick)

        # ── State ────────────────────────────────────────────────────────
        self._latest_img_msg: CompressedImage | None = None
        self._latest_scan_msg: LaserScan | None = None
        self._intrinsics: tuple | None = None       # (fx, fy, cx, cy, w, h)
        self._T_cam_laser: np.ndarray | None = None  # cached static TF
        self._inference_running = False

        self.get_logger().info(
            f"[SEMANTIC] Node ready — rate={rate_hz}Hz, "
            f"score_thr={self._score_thr}, cam={cam_topic}")

    # ==================================================================
    # Callbacks
    # ==================================================================

    def _image_cb(self, msg: CompressedImage):
        self._latest_img_msg = msg

    def _scan_cb(self, msg: LaserScan):
        self._latest_scan_msg = msg

    def _caminfo_cb(self, msg: CameraInfo):
        if self._intrinsics is not None:
            return
        fx = float(msg.k[0])
        fy = float(msg.k[4])
        cx = float(msg.k[2])
        cy = float(msg.k[5])
        w = int(msg.width)
        h = int(msg.height)
        if fx > 0 and fy > 0:
            self._intrinsics = (fx, fy, cx, cy, w, h)
            self._projector.set_intrinsics(fx, fy, cx, cy, w, h)
            self.get_logger().info(
                f"[SEMANTIC] Intrinsics: fx={fx:.1f} fy={fy:.1f} "
                f"cx={cx:.1f} cy={cy:.1f} {w}x{h}")

    # ==================================================================
    # Timer tick
    # ==================================================================

    def _tick(self):
        if self._inference_running:
            return
        if self._latest_img_msg is None or self._latest_scan_msg is None:
            return
        if self._intrinsics is None:
            return

        # Cache static TF laser → camera (one-time)
        if self._T_cam_laser is None:
            self._T_cam_laser = self._lookup_static_tf(
                self._camera_frame, self._laser_frame)
            if self._T_cam_laser is None:
                return
            self._projector.set_transform(self._T_cam_laser)

        img_msg = self._latest_img_msg
        scan_msg = self._latest_scan_msg
        self._inference_running = True
        threading.Thread(
            target=self._detect_and_fuse,
            args=(img_msg, scan_msg),
            daemon=True,
        ).start()

    # ==================================================================
    # Detection + fusion (runs in daemon thread)
    # ==================================================================

    def _detect_and_fuse(self, img_msg: CompressedImage, scan_msg: LaserScan):
        try:
            stamp = img_msg.header.stamp
            ts = float(stamp.sec) + float(stamp.nanosec) * 1e-9

            # 1. Decode compressed image
            buf = np.frombuffer(img_msg.data, dtype=np.uint8)
            img_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img_bgr is None:
                return

            # 2. RetinaNet → 2D detections
            dets_2d = self._run_retinanet(img_bgr)

            # 3. Project LiDAR into image
            projected = self._projector.project(
                ranges=np.array(scan_msg.ranges, dtype=np.float32),
                angle_min=scan_msg.angle_min,
                angle_increment=scan_msg.angle_increment,
                range_min=max(0.05, scan_msg.range_min),
                range_max=min(12.0, scan_msg.range_max),
            )

            # 4. Fuse boxes with LiDAR → map-frame anchors
            dets_3d = self._fuse_with_lidar(dets_2d, projected, stamp)

            # 5. Tracker update
            tracked = self._tracker.update(dets_3d, ts)

            # 6. Publish
            self._publish_detections(dets_3d, stamp)
            self._publish_tracked(tracked, stamp)
            self._publish_debug_image(img_bgr, dets_2d, projected, tracked)
            self._publish_markers(tracked, stamp)

        except Exception as e:
            self.get_logger().error(f"[SEMANTIC] detect_and_fuse: {e}")
        finally:
            self._inference_running = False

    # ------------------------------------------------------------------
    # RetinaNet inference
    # ------------------------------------------------------------------

    def _run_retinanet(self, img_bgr: np.ndarray) -> list[dict]:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        tensor = F.to_tensor(img_rgb).unsqueeze(0).to(self._device)
        with torch.no_grad():
            preds = self._model(tensor)[0]

        results = []
        for box, label, score in zip(
            preds["boxes"], preds["labels"], preds["scores"]
        ):
            lid = int(label)
            sc = float(score)
            if lid in _TARGET_CLASSES and sc >= self._score_thr:
                results.append({
                    "class_id": lid,
                    "class_name": _TARGET_CLASSES[lid],
                    "confidence": sc,
                    "bbox_xyxy": box.cpu().numpy().astype(int).tolist(),
                })
        return results

    # ------------------------------------------------------------------
    # Box ↔ LiDAR fusion
    # ------------------------------------------------------------------

    def _fuse_with_lidar(
        self,
        dets_2d: list[dict],
        projected,
        stamp,
    ) -> list[dict]:
        """Associate 2D boxes with projected LiDAR points.
        Returns detections augmented with map-frame anchor positions."""
        if not dets_2d:
            return []

        # Transform LiDAR sensor-frame points → map frame
        T_map_laser = self._lookup_tf(self._map_frame, self._laser_frame, stamp)
        if T_map_laser is None:
            return []

        R = T_map_laser[:3, :3]
        t = T_map_laser[:3, 3]
        map_pts = (R @ projected.sensor_xyz.T).T + t  # (N, 3)

        u = projected.pixels_uv[:, 0]
        v = projected.pixels_uv[:, 1]
        valid = projected.valid_mask

        results = []
        for det in dets_2d:
            x1, y1, x2, y2 = det["bbox_xyxy"]
            mask = (
                valid
                & (u >= x1) & (u <= x2)
                & (v >= y1) & (v <= y2)
            )
            n_pts = int(mask.sum())

            if n_pts < self._min_lidar:
                continue  # insufficient LiDAR support — skip

            anchor_xy = np.median(map_pts[mask, :2], axis=0)
            anchor_range = float(
                np.median(np.linalg.norm(
                    projected.sensor_xyz[mask, :2], axis=1)))

            det3d = dict(det)
            det3d["anchor_map_xy"] = anchor_xy.tolist()
            det3d["anchor_range_m"] = round(anchor_range, 3)
            det3d["n_lidar_pts_in_box"] = n_pts
            # Fields needed by tracker
            det3d["map_xy"] = anchor_xy.tolist()
            results.append(det3d)

        return results

    # ==================================================================
    # TF helpers
    # ==================================================================

    def _lookup_static_tf(self, target: str, source: str) -> np.ndarray | None:
        """Lookup a (presumably static) 4x4 transform. Returns None if not yet available."""
        try:
            t = self.tf_buffer.lookup_transform(
                target, source,
                rclpy.time.Time(),
                timeout=Duration(seconds=1.0))
            return self._stamped_to_mat(t)
        except Exception as e:
            self.get_logger().warn(
                f"[SEMANTIC] TF {source}→{target} not available: {e}")
            return None

    def _lookup_tf(self, target: str, source: str, stamp) -> np.ndarray | None:
        """Lookup a 4x4 transform at the given stamp, falling back to latest."""
        try:
            t = self.tf_buffer.lookup_transform(
                target, source,
                rclpy.time.Time(
                    seconds=stamp.sec, nanoseconds=stamp.nanosec),
                timeout=Duration(seconds=0.05))
            return self._stamped_to_mat(t)
        except Exception:
            pass
        # Fallback: latest
        try:
            t = self.tf_buffer.lookup_transform(
                target, source,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.05))
            return self._stamped_to_mat(t)
        except Exception:
            return None

    @staticmethod
    def _stamped_to_mat(t) -> np.ndarray:
        """Convert a TransformStamped to a 4x4 numpy matrix."""
        tr = t.transform.translation
        q = t.transform.rotation
        # Quaternion to rotation matrix
        x, y, z, w = q.x, q.y, q.z, q.w
        R = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
            [2*(x*y + z*w),     1 - 2*(x*x + z*z),  2*(y*z - x*w)],
            [2*(x*z - y*w),     2*(y*z + x*w),       1 - 2*(x*x + y*y)],
        ], dtype=np.float64)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = [tr.x, tr.y, tr.z]
        return T

    # ==================================================================
    # Publishers
    # ==================================================================

    def _publish_detections(self, dets: list[dict], stamp):
        msg = String()
        msg.data = json.dumps({
            "stamp_sec": stamp.sec,
            "stamp_nanosec": stamp.nanosec,
            "detections": dets,
        })
        self._det_pub.publish(msg)

    def _publish_tracked(self, tracked, stamp):
        msg = String()
        msg.data = json.dumps({
            "stamp_sec": stamp.sec,
            "stamp_nanosec": stamp.nanosec,
            "objects": [t.to_dict() for t in tracked],
        })
        self._track_pub.publish(msg)

    # ------------------------------------------------------------------
    # Debug image
    # ------------------------------------------------------------------

    def _publish_debug_image(self, img_bgr, dets_2d, projected, tracked):
        if self._debug_pub.get_subscription_count() == 0:
            return

        vis = img_bgr.copy()

        # Draw projected LiDAR points (small green dots)
        valid = projected.valid_mask
        for i in np.where(valid)[0]:
            u = int(projected.pixels_uv[i, 0])
            v = int(projected.pixels_uv[i, 1])
            cv2.circle(vis, (u, v), 2, (0, 255, 0), -1)

        # Draw 2D detection boxes
        for det in dets_2d:
            cls = det["class_name"]
            x1, y1, x2, y2 = det["bbox_xyxy"]
            colour = _CLASS_COLOURS.get(cls, (255, 255, 255))
            cv2.rectangle(vis, (x1, y1), (x2, y2), colour, 2)
            label = f"{cls} {det['confidence']:.2f}"
            cv2.putText(vis, label, (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)

        # Draw tracked object IDs
        for trk in tracked:
            if trk.status == "confirmed":
                label = f"#{trk.track_id} {trk.class_name}"
                # Project track world pos back to image (approximate)
                cv2.putText(vis, label, (10, 20 + 18 * trk.track_id),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # Publish as ROS Image
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.height, msg.width = vis.shape[:2]
        msg.encoding = "bgr8"
        msg.step = vis.shape[1] * 3
        msg.data = vis.tobytes()
        self._debug_pub.publish(msg)

    # ------------------------------------------------------------------
    # RViz markers
    # ------------------------------------------------------------------

    def _publish_markers(self, tracked, stamp):
        if self._marker_pub.get_subscription_count() == 0:
            return

        ma = MarkerArray()
        now = self.get_clock().now().to_msg()

        for trk in tracked:
            # Sphere
            m = Marker()
            m.header.frame_id = self._map_frame
            m.header.stamp = now
            m.ns = "semantic_objects"
            m.id = trk.track_id * 2
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(trk.map_xy[0])
            m.pose.position.y = float(trk.map_xy[1])
            m.pose.position.z = 0.3
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.25
            if trk.status == "confirmed":
                m.color.r, m.color.g, m.color.b, m.color.a = 0.0, 0.9, 0.0, 0.8
            else:
                m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 1.0, 0.0, 0.5
            m.lifetime.sec = 2
            ma.markers.append(m)

            # Label
            lbl = Marker()
            lbl.header = m.header
            lbl.ns = "semantic_labels"
            lbl.id = trk.track_id * 2 + 1
            lbl.type = Marker.TEXT_VIEW_FACING
            lbl.action = Marker.ADD
            lbl.pose.position.x = float(trk.map_xy[0])
            lbl.pose.position.y = float(trk.map_xy[1])
            lbl.pose.position.z = 0.55
            lbl.pose.orientation.w = 1.0
            lbl.scale.z = 0.15
            lbl.color.r = lbl.color.g = lbl.color.b = 1.0
            lbl.color.a = 0.9
            lbl.text = f"{trk.class_name} #{trk.track_id}"
            lbl.lifetime.sec = 2
            ma.markers.append(lbl)

        self._marker_pub.publish(ma)


# ======================================================================
def main(args=None):
    rclpy.init(args=args)
    node = SemanticDetectorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
