#!/usr/bin/env python3
import math
import threading
from typing import List, Optional, Tuple

import cv2
import numpy as np
import rclpy
from rclpy.time import Time
import torch
import tf2_ros
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped, TransformStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CameraInfo, Image
from visualization_msgs.msg import Marker, MarkerArray


FURNITURE_CLASSES = {
    9: "chair",
    11: "diningtable",
    18: "sofa",
    20: "tvmonitor",
}

_IMAGE_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)


class SemanticMapperTestNode(Node):
    def __init__(self) -> None:
        super().__init__("semantic_mapper_test")

        self.declare_parameter("rgb_topic", "/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/aligned_depth_to_color/camera_info")
        self.declare_parameter("global_frame", "odom")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("camera_frame", "camera_color_optical_frame")
        self.declare_parameter("tick_hz", 1.0)
        self.declare_parameter("min_pixels", 300)
        self.declare_parameter("min_depth_m", 0.15)
        self.declare_parameter("max_depth_m", 8.0)
        self.declare_parameter("tf_timeout_sec", 0.15)
        self.declare_parameter("publish_static_camera_tf", True)
        self.declare_parameter("camera_offset_x_m", -0.05)
        self.declare_parameter("camera_offset_y_m", 0.0)
        self.declare_parameter("camera_offset_z_m", 0.20)
        self.declare_parameter("debug_image_topic", "/semantic_mapper/debug_image")
        self.declare_parameter("marker_topic", "/semantic_mapper/debug_markers")
        self.declare_parameter("min_confidence", 0.65)

        self.rgb_topic = str(self.get_parameter("rgb_topic").value)
        self.depth_topic = str(self.get_parameter("depth_topic").value)
        self.camera_info_topic = str(self.get_parameter("camera_info_topic").value)
        self.global_frame = str(self.get_parameter("global_frame").value)
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.camera_frame = str(self.get_parameter("camera_frame").value)
        self.tick_hz = float(self.get_parameter("tick_hz").value)
        self.min_pixels = int(self.get_parameter("min_pixels").value)
        self.min_depth_m = float(self.get_parameter("min_depth_m").value)
        self.max_depth_m = float(self.get_parameter("max_depth_m").value)
        self.publish_static_camera_tf = bool(
            self.get_parameter("publish_static_camera_tf").value
        )
        self.camera_offset_x_m = float(self.get_parameter("camera_offset_x_m").value)
        self.camera_offset_y_m = float(self.get_parameter("camera_offset_y_m").value)
        self.camera_offset_z_m = float(self.get_parameter("camera_offset_z_m").value)
        self.tf_timeout = Duration(
            seconds=float(self.get_parameter("tf_timeout_sec").value)
        )
        self.min_confidence = float(self.get_parameter("min_confidence").value)

        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.static_tf_broadcaster = tf2_ros.StaticTransformBroadcaster(self)

        self.rgb_msg: Optional[Image] = None
        self.depth_msg: Optional[Image] = None
        self.fx: Optional[float] = None
        self.fy: Optional[float] = None
        self.cx: Optional[float] = None
        self.cy: Optional[float] = None
        self._inference_running: bool = False

        self.create_subscription(Image, self.rgb_topic,   self.rgb_cb,   _IMAGE_QOS)
        self.create_subscription(Image, self.depth_topic, self.depth_cb, _IMAGE_QOS)
        self.create_subscription(CameraInfo, self.camera_info_topic, self.caminfo_cb, 1)

        self.debug_image_pub = self.create_publisher(
            Image, str(self.get_parameter("debug_image_topic").value), 10
        )
        self.marker_pub = self.create_publisher(
            MarkerArray, str(self.get_parameter("marker_topic").value), 10
        )
        self.point_pub = self.create_publisher(PointStamped, "/semantic_mapper/debug_point", 10)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        from torchvision import transforms
        from torchvision.models.segmentation import (
            LRASPP_MobileNet_V3_Large_Weights,
            lraspp_mobilenet_v3_large,
        )

        weights = LRASPP_MobileNet_V3_Large_Weights.DEFAULT
        self.model = lraspp_mobilenet_v3_large(weights=weights).to(self.device).eval()
        self.transform = transforms.ToTensor()

        self.timer = self.create_timer(1.0 / max(self.tick_hz, 1e-3), self.tick)
        self._tick_count = 0

        if self.publish_static_camera_tf:
            self.publish_camera_static_tf()

        self.get_logger().info(
            "[SEMANTIC TEST] ready "
            f"rgb={self.rgb_topic} depth={self.depth_topic} info={self.camera_info_topic} "
            f"frame={self.global_frame} camera_frame={self.camera_frame} "
            f"device={self.device} min_confidence={self.min_confidence}"
        )

    def rgb_cb(self, msg: Image) -> None:
        self.rgb_msg = msg          # store raw msg only
        # Passthrough: keep debug topic live at camera framerate (no decode needed)
        if self.debug_image_pub.get_subscription_count() > 0:
            self.debug_image_pub.publish(msg)

    def depth_cb(self, msg: Image) -> None:
        self.depth_msg = msg        # store raw msg only

    def caminfo_cb(self, msg: CameraInfo) -> None:
        self.fx = float(msg.k[0])
        self.fy = float(msg.k[4])
        self.cx = float(msg.k[2])
        self.cy = float(msg.k[5])

    def intrinsics_ready(self) -> bool:
        return (
            self.fx is not None
            and self.fy is not None
            and self.cx is not None
            and self.cy is not None
            and self.fx > 1e-6
            and self.fy > 1e-6
        )

    def tick(self) -> None:
        self._tick_count += 1
        if self._inference_running:
            return  # previous inference still running, skip

        if self.rgb_msg is None or self.depth_msg is None or not self.intrinsics_ready():
            if self._tick_count % 20 == 0:
                self.get_logger().info(
                    f"[SEMANTIC TEST] waiting rgb={self.rgb_msg is not None} "
                    f"depth={self.depth_msg is not None} K={self.intrinsics_ready()}"
                )
            return

        # Decode once per inference tick (not in every callback)
        rgb_bgr       = self.bridge.imgmsg_to_cv2(self.rgb_msg, desired_encoding="bgr8")
        depth         = self.bridge.imgmsg_to_cv2(self.depth_msg)
        depth_msg_snap = self.depth_msg
        rgb_msg_snap   = self.rgb_msg

        self._inference_running = True
        threading.Thread(
            target=self._infer_and_publish,
            args=(rgb_bgr, depth, depth_msg_snap, rgb_msg_snap),
            daemon=True,
        ).start()

    def _infer_and_publish(
        self,
        rgb_bgr: np.ndarray,
        depth: np.ndarray,
        depth_msg_snap: Image,
        rgb_msg_snap: Image,
    ) -> None:
        try:
            img_rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
            tensor = self.transform(img_rgb).unsqueeze(0).to(self.device)
            with torch.no_grad():
                out = self.model(tensor)["out"][0]
                probs = torch.softmax(out, dim=0)
                mask = out.argmax(0).cpu().numpy().astype(np.int32)

            detections = self.extract_detections(mask, probs)
            debug = rgb_bgr.copy()
            markers = MarkerArray()
            delete_marker = Marker()
            delete_marker.header.frame_id = self.global_frame
            delete_marker.header.stamp = self.get_clock().now().to_msg()
            delete_marker.action = Marker.DELETEALL
            markers.markers.append(delete_marker)

            if not detections:
                cv2.putText(
                    debug,
                    "No tracked furniture class found",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )
                self.publish_debug(debug, markers, rgb_msg_snap)
                return

            for idx, det in enumerate(detections):
                depth_m = self.depth_at_bbox(det["bbox"], depth, depth_msg_snap)
                label = f"{det['class']} conf={det['confidence']:.2f}"
                color = (0, 255, 0)
                x1, y1, x2, y2 = det["bbox"]
                cv2.rectangle(debug, (x1, y1), (x2, y2), color, 2)

                if depth_m is None:
                    cv2.putText(
                        debug,
                        f"{label} depth=invalid",
                        (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 0, 255),
                        2,
                    )
                    continue

                point_cam = self.backproject_pixel(det["center"], depth_m, depth_msg_snap)
                point_global = self.transform_point(
                    point_cam, depth_msg_snap.header.frame_id, depth_msg_snap.header.stamp
                )

                if point_global is None:
                    cv2.putText(
                        debug,
                        f"{label} tf=missing",
                        (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 165, 255),
                        2,
                    )
                    continue

                cv2.putText(
                    debug,
                    (
                        f"{label} z={depth_m:.2f}m "
                        f"map=({point_global.point.x:.2f},{point_global.point.y:.2f})"
                    ),
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    2,
                )
                markers.markers.append(self.make_marker(idx, det["class"], point_global))
                self.point_pub.publish(point_global)

            self.publish_debug(debug, markers, rgb_msg_snap)
        finally:
            self._inference_running = False

    def extract_detections(self, mask: np.ndarray, probs: torch.Tensor) -> List[dict]:
        detections: List[dict] = []
        for class_id, class_name in FURNITURE_CLASSES.items():
            ys, xs = np.where(mask == class_id)
            if xs.size < self.min_pixels:
                continue

            x1 = int(xs.min())
            x2 = int(xs.max())
            y1 = int(ys.min())
            y2 = int(ys.max())
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            confidence = float(probs[class_id, ys, xs].mean().item())
            if confidence < self.min_confidence:
                continue
            detections.append(
                {
                    "class": class_name,
                    "bbox": (x1, y1, x2, y2),
                    "center": center,
                    "confidence": confidence,
                }
            )
        return detections

    def depth_at_bbox(
        self,
        bbox: Tuple[int, int, int, int],
        depth: np.ndarray,
        depth_msg_snap: Image,
    ) -> Optional[float]:
        x1, y1, x2, y2 = bbox
        crop = depth[y1:y2 + 1, x1:x2 + 1]
        if crop.size == 0:
            return None

        crop = crop.astype(np.float32)
        if depth_msg_snap.encoding == "16UC1":
            crop *= 0.001

        valid = np.isfinite(crop)
        valid &= crop > self.min_depth_m
        valid &= crop < self.max_depth_m
        valid_values = crop[valid]
        if valid_values.size == 0:
            return None
        return float(np.median(valid_values))

    def backproject_pixel(
        self, pixel: Tuple[int, int], depth_m: float, depth_msg_snap: Image
    ) -> PointStamped:
        u, v = pixel
        x = (float(u) - float(self.cx)) * depth_m / float(self.fx)
        y = (float(v) - float(self.cy)) * depth_m / float(self.fy)

        msg = PointStamped()
        msg.header = depth_msg_snap.header
        msg.point.x = x
        msg.point.y = y
        msg.point.z = depth_m
        return msg

    def publish_camera_static_tf(self) -> None:
        tf_msg = TransformStamped()
        tf_msg.header.stamp = self.get_clock().now().to_msg()
        tf_msg.header.frame_id = self.base_frame
        tf_msg.child_frame_id = self.camera_frame
        tf_msg.transform.translation.x = self.camera_offset_x_m
        tf_msg.transform.translation.y = self.camera_offset_y_m
        tf_msg.transform.translation.z = self.camera_offset_z_m

        # Base frame: x forward, y left, z up
        # Optical frame: x right, y down, z forward
        qx, qy, qz, qw = self.quaternion_from_euler(-math.pi / 2.0, 0.0, -math.pi / 2.0)
        tf_msg.transform.rotation.x = qx
        tf_msg.transform.rotation.y = qy
        tf_msg.transform.rotation.z = qz
        tf_msg.transform.rotation.w = qw

        self.static_tf_broadcaster.sendTransform(tf_msg)
        self.get_logger().info(
            "[SEMANTIC TEST] published static TF "
            f"{self.base_frame} -> {self.camera_frame} "
            f"xyz=({self.camera_offset_x_m:.2f}, {self.camera_offset_y_m:.2f}, "
            f"{self.camera_offset_z_m:.2f})"
        )

    @staticmethod
    def quaternion_from_euler(roll: float, pitch: float, yaw: float) -> Tuple[float, float, float, float]:
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)

        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        return qx, qy, qz, qw

    def transform_point(
        self, point_cam: PointStamped, source_frame: str, stamp
    ) -> Optional[PointStamped]:
        try:
            tf = self.tf_buffer.lookup_transform(self.global_frame, source_frame, stamp)
        except Exception as stamped_exc:
            try:
                # Fall back to the latest available transform when exact image
                # timestamp lookup misses due to clock skew or TF latency.
                tf = self.tf_buffer.lookup_transform(
                    self.global_frame, source_frame, Time()
                )
            except Exception as latest_exc:
                self.get_logger().warn(
                    "[SEMANTIC TEST] TF lookup failed "
                    f"{self.global_frame} <- {source_frame}; "
                    f"stamped={stamped_exc}; latest={latest_exc}",
                    throttle_duration_sec=2.0,
                )
                return None

        try:
            return tf2_ros.TransformRegistration().get(type(point_cam))(point_cam, tf)
        except Exception:
            import tf2_geometry_msgs  # noqa: F401

            return tf2_geometry_msgs.do_transform_point(point_cam, tf)

    def make_marker(self, marker_id: int, label: str, point: PointStamped) -> Marker:
        marker = Marker()
        marker.header = point.header
        marker.ns = "semantic_mapper_test"
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position = point.point
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.20
        marker.scale.y = 0.20
        marker.scale.z = 0.20
        marker.color.a = 0.9
        marker.color.r = 0.1
        marker.color.g = 0.9
        marker.color.b = 0.2
        marker.text = label
        return marker

    def publish_debug(
        self, debug_img: np.ndarray, markers: MarkerArray, rgb_msg_snap: Optional[Image]
    ) -> None:
        self.marker_pub.publish(markers)
        if self.debug_image_pub.get_subscription_count() > 0:
            debug_msg = self.bridge.cv2_to_imgmsg(debug_img, encoding="bgr8")
            if rgb_msg_snap is not None:
                debug_msg.header = rgb_msg_snap.header
            self.debug_image_pub.publish(debug_msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SemanticMapperTestNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
