#!/usr/bin/env python3
"""Relay semantic object JSON between two ROS_DOMAIN_ID values.

This intentionally bridges only /semantic_segmentation/objects so RealSense
RGB/depth/image topics stay isolated from the robot mapping/navigation domain.
It can also bridge /tf_static so the robot domain knows the camera optical
frames without importing the high-bandwidth camera streams.
"""

from __future__ import annotations

import argparse
import sys
import time

import rclpy
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage


_TF_STATIC_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)


class _SemanticObjectsBridge:
    def __init__(
        self,
        *,
        camera_domain: int,
        robot_domain: int,
        input_topic: str,
        output_topic: str,
        bridge_tf_static: bool,
        tf_static_topic: str,
        log_every: int,
    ) -> None:
        self.camera_context = Context()
        self.robot_context = Context()
        rclpy.init(args=[], context=self.camera_context, domain_id=camera_domain)
        rclpy.init(args=[], context=self.robot_context, domain_id=robot_domain)

        self.input_node = Node(
            "semantic_objects_bridge_in",
            context=self.camera_context,
        )
        self.output_node = Node(
            "semantic_objects_bridge_out",
            context=self.robot_context,
        )
        self.input_executor = SingleThreadedExecutor(context=self.camera_context)
        self.output_executor = SingleThreadedExecutor(context=self.robot_context)
        self.input_executor.add_node(self.input_node)
        self.output_executor.add_node(self.output_node)

        self.publisher = self.output_node.create_publisher(String, output_topic, 10)
        self.tf_static_publisher = None
        self.count = 0
        self.tf_static_count = 0
        self.log_every = max(1, int(log_every))
        self.input_node.create_subscription(
            String,
            input_topic,
            self._objects_cb,
            10,
        )
        if bridge_tf_static:
            self.tf_static_publisher = self.output_node.create_publisher(
                TFMessage,
                tf_static_topic,
                _TF_STATIC_QOS,
            )
            self.input_node.create_subscription(
                TFMessage,
                tf_static_topic,
                self._tf_static_cb,
                _TF_STATIC_QOS,
            )

        self.input_node.get_logger().info(
            f"[SEM BRIDGE] camera_domain={camera_domain} topic={input_topic}")
        self.output_node.get_logger().info(
            f"[SEM BRIDGE] robot_domain={robot_domain} topic={output_topic}")
        if bridge_tf_static:
            self.output_node.get_logger().info(
                f"[SEM BRIDGE] bridging {tf_static_topic} camera_domain={camera_domain} "
                f"-> robot_domain={robot_domain}")

    def _objects_cb(self, msg: String) -> None:
        out = String()
        out.data = msg.data
        self.publisher.publish(out)
        self.count += 1
        if self.count == 1 or self.count % self.log_every == 0:
            self.output_node.get_logger().info(
                f"[SEM BRIDGE] relayed {self.count} semantic object messages")

    def _tf_static_cb(self, msg: TFMessage) -> None:
        if self.tf_static_publisher is None:
            return
        self.tf_static_publisher.publish(msg)
        self.tf_static_count += 1
        if self.tf_static_count == 1 or self.tf_static_count % self.log_every == 0:
            frames = ", ".join(t.child_frame_id for t in msg.transforms[:5])
            suffix = "" if len(msg.transforms) <= 5 else ", ..."
            self.output_node.get_logger().info(
                f"[SEM BRIDGE] relayed /tf_static msg #{self.tf_static_count} "
                f"transforms={len(msg.transforms)} [{frames}{suffix}]")

    def spin(self) -> None:
        try:
            while rclpy.ok(context=self.camera_context) and rclpy.ok(context=self.robot_context):
                self.input_executor.spin_once(timeout_sec=0.05)
                # Give the output context a chance to service graph events.
                self.output_executor.spin_once(timeout_sec=0.0)
        finally:
            self.input_executor.remove_node(self.input_node)
            self.output_executor.remove_node(self.output_node)
            self.input_node.destroy_node()
            self.output_node.destroy_node()
            self.camera_context.shutdown()
            self.robot_context.shutdown()


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bridge semantic object JSON from camera ROS domain to robot ROS domain.")
    parser.add_argument("--camera-domain", type=int, default=20)
    parser.add_argument("--robot-domain", type=int, default=0)
    parser.add_argument(
        "--input-topic",
        default="/semantic_segmentation/objects",
        help="Topic to subscribe to in the camera domain.")
    parser.add_argument(
        "--output-topic",
        default="/semantic_segmentation/objects",
        help="Topic to publish to in the robot domain.")
    parser.add_argument(
        "--tf-static-topic",
        default="/tf_static",
        help="Static TF topic to bridge from camera domain to robot domain.")
    parser.add_argument(
        "--no-tf-static",
        action="store_true",
        help="Do not bridge /tf_static.")
    parser.add_argument("--log-every", type=int, default=20)
    return parser.parse_args(argv)


def main(args=None) -> None:
    ns = _parse_args(sys.argv[1:] if args is None else list(args))
    bridge = _SemanticObjectsBridge(
        camera_domain=ns.camera_domain,
        robot_domain=ns.robot_domain,
        input_topic=ns.input_topic,
        output_topic=ns.output_topic,
        bridge_tf_static=not ns.no_tf_static,
        tf_static_topic=ns.tf_static_topic,
        log_every=ns.log_every,
    )
    bridge.spin()


if __name__ == "__main__":
    main()
