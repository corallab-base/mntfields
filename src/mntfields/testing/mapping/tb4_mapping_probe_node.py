from geometry_msgs.msg import Twist

import rclpy

from mntfields.mntfields_ros.mapping.tb4_node import TB4GridmapNode


class TB4MappingProbeNode(TB4GridmapNode):
    def _keyboard_loop(self):
        self._kbd_stop.wait()

    def __init__(self):
        super().__init__()
        self.set_parameters([
            rclpy.parameter.Parameter(
                "train_after_region",
                rclpy.Parameter.Type.BOOL,
                False,
            ),
        ])
        self.enable_navigation()
        with self.train_lock:
            self.train_queue.clear()

        self.declare_parameter("probe_rotation_enable", True)
        self.declare_parameter("probe_rotation_speed", 0.35)
        self.declare_parameter("probe_rotation_duration_sec", 1.2)
        self.declare_parameter("probe_forward_speed", 0.05)
        self.declare_parameter("probe_forward_duration_sec", 1.0)
        self.declare_parameter("probe_pause_duration_sec", 3.0)
        self.declare_parameter("probe_cmd_timer_sec", 0.05)

        self._probe_phase = "pause"
        self._probe_phase_started = self.get_clock().now().nanoseconds * 1e-9
        probe_dt = float(self.get_parameter("probe_cmd_timer_sec").value)
        self.create_timer(probe_dt, self._probe_motion_tick)
        self.get_logger().info(
            "[TB4 PROBE] Wrapper mode active. Same mapper as tb4_node, "
            "training disabled, scripted motion enabled."
        )

    def control_tick(self):
        return

    def pick_goal(self, *args, **kwargs):
        return None

    def _probe_motion_tick(self):
        if not bool(self.get_parameter("probe_rotation_enable").value):
            self.navigator.stop_robot()
            return
        if self.region is None or self.mode == "BOOTSTRAP":
            self.navigator.stop_robot()
            return

        now_sec = self.get_clock().now().nanoseconds * 1e-9
        elapsed = now_sec - self._probe_phase_started
        rot_speed = float(self.get_parameter("probe_rotation_speed").value)
        rot_dur = float(self.get_parameter("probe_rotation_duration_sec").value)
        fwd_speed = float(self.get_parameter("probe_forward_speed").value)
        fwd_dur = float(self.get_parameter("probe_forward_duration_sec").value)
        pause_dur = float(self.get_parameter("probe_pause_duration_sec").value)

        twist = Twist()
        if self._probe_phase == "rotate":
            if elapsed >= rot_dur:
                self._probe_phase = "forward"
                self._probe_phase_started = now_sec
            else:
                twist.angular.z = rot_speed
        elif self._probe_phase == "forward":
            if elapsed >= fwd_dur:
                self._probe_phase = "pause"
                self._probe_phase_started = now_sec
            else:
                twist.linear.x = fwd_speed
        else:
            if elapsed >= pause_dur:
                self._probe_phase = "rotate"
                self._probe_phase_started = now_sec
                twist.angular.z = rot_speed
        self.cmd_pub.publish(twist)

    def update(self):
        super().update()
        if self.mode != "BOOTSTRAP":
            self.map_publisher.publish_maps()
            self.markers.publish_goal_marker()
            self.markers.publish_frontier_markers()
            self.publish_field_path(self.path)

    def destroy_node(self):
        try:
            self.navigator.stop_robot()
        except Exception:
            pass
        return super().destroy_node()


def main():
    rclpy.init()
    node = TB4MappingProbeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.navigator.stop_robot()
            node.plotter.save_ros_map(save_satellites=True)
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
