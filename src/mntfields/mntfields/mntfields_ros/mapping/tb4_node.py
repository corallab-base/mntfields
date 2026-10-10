"""
tb4_node.py — TurtleBot4 real-robot wrapper around GridmapNode.

Thin subclass that adds:
  • BEST_EFFORT QoS override for /scan (real TB4 requirement)
  • Keyboard control: Enter → log "mapping running", c → no-op (no nav to stop)
  • /mntfields/nav_enable topic (Bool) for remote control
  • SemanticMapperNode integration (optional, logs if unavailable)

Everything else (LiDAR processing, region management, training) is handled
by the base GridmapNode.
"""

import select
import sys
import termios
import threading
import tty

import rclpy
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool

from mntfields.mntfields_ros.mapping.gridmap_node import GridmapNode, REGION_RADIUS


class TB4GridmapNode(GridmapNode):
    """GridmapNode configured for the TurtleBot4 real robot."""

    # ── QoS override: use BEST_EFFORT for /scan on the real TB4 ─────────────
    def create_subscription(self, msg_type, topic, callback, qos_profile, *args, **kwargs):
        if topic in ("/scan", "/odom"):
            qos_profile = QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=10,
                reliability=ReliabilityPolicy.BEST_EFFORT,
            )
        return super().create_subscription(
            msg_type, topic, callback, qos_profile, *args, **kwargs
        )

    def __init__(self):
        super().__init__()

        # ── Override sim-time to False for real robot ────────────────────────
        self.set_parameters([
            rclpy.parameter.Parameter(
                "use_sim_time", rclpy.Parameter.Type.BOOL, False
            ),
            rclpy.parameter.Parameter(
                "min_train_epochs_for_nav", rclpy.Parameter.Type.INTEGER, 150
            ),
        ])

        # ── Keyboard control (daemon thread — user input only, not training) ─
        self._kbd_stop = threading.Event()
        if sys.stdin.isatty():
            self._kbd_thread = threading.Thread(
                target=self._keyboard_loop, daemon=True
            )
            self._kbd_thread.start()
        else:
            self._kbd_thread = None
            self.get_logger().warn(
                "[TB4] stdin is not a TTY — keyboard control disabled."
            )

        # ── /mntfields/nav_enable topic ──────────────────────────────────────
        self.create_subscription(
            Bool, "/mntfields/nav_enable", self._nav_enable_cb, 10
        )

        # ── Semantic region store (replaces old SemanticMapperNode) ────────
        try:
            from mntfields.mntfields_ros.mapping.semantic_region_store import (
                SemanticRegionStore,
            )
            self.semantic_store = SemanticRegionStore(gridmap_node=self)
        except Exception as e:
            self.semantic_store = None
            self.get_logger().warn(
                f"[TB4] Semantic region store init failed: {e}")

        self.get_logger().info(
            "[TB4] Ready — press Enter in terminal to confirm mapping is running, "
            "or publish True to /mntfields/nav_enable."
        )

    # ------------------------------------------------------------------
    # Nav enable/disable (no state machine — just logging for now)
    # ------------------------------------------------------------------

    def _nav_enable_cb(self, msg: Bool):
        if bool(msg.data):
            self.get_logger().info("[TB4] Mapping/navigation enabled via topic.")
        else:
            self.get_logger().info("[TB4] Mapping paused via topic.")

    # ------------------------------------------------------------------
    # Keyboard loop (daemon thread — safe: only calls logger, no training)
    # ------------------------------------------------------------------

    def _keyboard_loop(self):
        fd = sys.stdin.fileno()
        old_attrs = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while not self._kbd_stop.is_set():
                ready, _, _ = select.select([sys.stdin], [], [], 0.2)
                if not ready:
                    continue
                ch = sys.stdin.read(1)
                if ch in ("\n", "\r"):
                    self.get_logger().info("[TB4] Mapping running.")
                elif ch in ("c", "C"):
                    self.get_logger().info("[TB4] 'c' pressed — no navigation to stop.")
        except Exception as e:
            self.get_logger().warn(f"[TB4] Keyboard loop stopped: {e}")
        finally:
            try:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)
            except Exception:
                pass


def main(args=None):
    rclpy.init(args=args)
    node = TB4GridmapNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
