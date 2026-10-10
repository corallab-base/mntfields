import numpy as np
import rclpy
from geometry_msgs.msg import Twist

from mntfields.mntfields_ros.mapping.tb4_node import TB4GridmapNode, MAX_RANGE


class TB4SkewTestNode(TB4GridmapNode):
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
        with self.train_lock:
            self.train_queue.clear()

        self.declare_parameter("skew_forward_distance_m", 1.0)
        self.declare_parameter("skew_forward_speed_mps", 0.08)
        self.declare_parameter("skew_turn_angle_deg", 180.0)
        self.declare_parameter("skew_turn_speed_rps", 0.25)
        self.declare_parameter("skew_motion_timer_sec", 0.05)

        self._motion_state = "WAIT_REGION"
        self._motion_start_xy = None
        self._motion_start_yaw = None
        self._test_done = False

        dt = float(self.get_parameter("skew_motion_timer_sec").value)
        self.create_timer(dt, self._motion_tick)
        self.get_logger().info(
            "[TB4 SKEW TEST] Mapping-only skew test active. "
            "Sequence: forward 1m, slow turn, drive back."
        )

    @staticmethod
    def _angle_diff(a, b):
        return float(np.arctan2(np.sin(a - b), np.cos(a - b)))

    def control_tick(self):
        return

    def pick_goal(self, *args, **kwargs):
        return None

    def _apply_mapping_only_update(self):
        if self.latest_cloud is None or self.robot_pos is None:
            return
        if self.latest_cloud.__class__.__name__ != "LaserScan":
            return

        deskewed = self._deskew_scan_world_points(self.latest_cloud)
        if deskewed is None:
            return

        world_points, speeds, scan_robot_pose, scan_robot_yaw = deskewed
        if self.region is not None:
            self.region.last_robot_pose = np.array(
                [scan_robot_pose[0], scan_robot_pose[1], 0.0], dtype=np.float32
            )

        half = self.gridmap.map_size / 2
        ox, oy = self.gridmap.origin
        in_bounds = (
            (world_points[:, 0] > ox - half)
            & (world_points[:, 0] < ox + half)
            & (world_points[:, 1] > oy - half)
            & (world_points[:, 1] < oy + half)
        )
        world_points_filtered = world_points[in_bounds]
        speeds_filtered = speeds[in_bounds]
        if len(world_points_filtered) < 5:
            return

        robot_xy = scan_robot_pose[:2]
        vecs = world_points_filtered[:, :2] - robot_xy[None, :]
        dists = np.linalg.norm(vecs, axis=1) + 1e-9
        angles = np.arctan2(vecs[:, 1], vecs[:, 0])
        bins = ((angles + np.pi) / (2 * np.pi) * 359).astype(np.int32)

        closest_idx = {}
        closest_dist = np.full(360, np.inf, dtype=np.float32)
        for k, b in enumerate(bins):
            if dists[k] < closest_dist[b]:
                closest_dist[b] = dists[k]
                closest_idx[b] = k

        keep = np.array(list(closest_idx.values()), dtype=np.int32)
        world_points_hit = world_points_filtered[keep]
        speeds_hit = speeds_filtered[keep]

        cell = self.gridmap.cell_size
        origin = self.gridmap.origin
        half_ = self.gridmap.map_size / 2

        def world_to_ij(x, y):
            return (
                int((x - origin[0] + half_) / cell),
                int((y - origin[1] + half_) / cell),
            )

        for pt3d in world_points_hit:
            pt = pt3d[:2]
            dx, dy = pt[0] - robot_xy[0], pt[1] - robot_xy[1]
            dist = float(np.hypot(dx, dy))
            if dist > MAX_RANGE:
                scale = MAX_RANGE / dist
                dx *= scale
                dy *= scale
                dist = MAX_RANGE
            steps = int(dist / cell)
            if steps < 1:
                ix, iy = world_to_ij(pt[0], pt[1])
                if 0 <= ix < self.gridmap.n_cells and 0 <= iy < self.gridmap.n_cells:
                    self.gridmap.data[iy, ix] = 1
                    self.gridmap.visited[iy, ix] = 1
                continue
            for s_idx in range(steps - 1):
                x = robot_xy[0] + dx * s_idx / steps
                y = robot_xy[1] + dy * s_idx / steps
                ix, iy = world_to_ij(x, y)
                if not (0 <= ix < self.gridmap.n_cells and 0 <= iy < self.gridmap.n_cells):
                    break
                if self.gridmap.data[iy, ix] == 1:
                    break
                self.gridmap.visited[iy, ix] = 1
            ix_end, iy_end = world_to_ij(pt[0], pt[1])
            if 0 <= ix_end < self.gridmap.n_cells and 0 <= iy_end < self.gridmap.n_cells:
                self.gridmap.data[iy_end, ix_end] = 1
                self.gridmap.visited[iy_end, ix_end] = 1

        self.gridmap.update(
            lidar_scan=world_points_hit[:, :2],
            points=world_points_hit[:, :2],
            speeds=speeds_hit,
            cur_pos=robot_xy,
            region=self.region,
            regions=self.all_regions,
            update_room=False,
        )
        self.robot_yaw = scan_robot_yaw
        self.map_publisher.publish_maps()
        self.markers.publish_goal_marker()
        self.markers.publish_frontier_markers()
        self.publish_field_path(None)

    def update(self):
        try:
            if self.region is None:
                if self.robot_pos is not None and self.latest_cloud is not None:
                    self.bootstrap_frames += 1
                    if self.bootstrap_frames >= self.bootstrap_threshold:
                        self.get_logger().info("[TB4 SKEW TEST] Initializing first region")
                        self.init_region(self.robot_pos)
            self._apply_mapping_only_update()
        except Exception as e:
            import traceback
            self.get_logger().error(f"[TB4 SKEW TEST] Unhandled exception: {e}")
            traceback.print_exc()
            try:
                self.navigator.stop_robot()
            except Exception:
                pass

    def _motion_tick(self):
        if self.region is None or self.robot_pos is None:
            return
        if self._test_done:
            self.navigator.stop_robot()
            return

        fwd_dist = float(self.get_parameter("skew_forward_distance_m").value)
        fwd_speed = float(self.get_parameter("skew_forward_speed_mps").value)
        turn_angle = np.deg2rad(float(self.get_parameter("skew_turn_angle_deg").value))
        turn_speed = float(self.get_parameter("skew_turn_speed_rps").value)

        twist = Twist()
        robot_xy = np.asarray(self.robot_pos[:2], dtype=np.float32)
        yaw = float(self.robot_yaw)

        if self._motion_state == "WAIT_REGION":
            self._motion_state = "FORWARD"
            self._motion_start_xy = robot_xy.copy()
        elif self._motion_state == "FORWARD":
            traveled = float(np.linalg.norm(robot_xy - self._motion_start_xy))
            if traveled >= fwd_dist:
                self._motion_state = "TURN"
                self._motion_start_yaw = yaw
            else:
                twist.linear.x = fwd_speed
        elif self._motion_state == "TURN":
            turned = abs(self._angle_diff(yaw, self._motion_start_yaw))
            if turned >= turn_angle:
                self._motion_state = "BACK"
                self._motion_start_xy = robot_xy.copy()
            else:
                twist.angular.z = turn_speed
        elif self._motion_state == "BACK":
            traveled = float(np.linalg.norm(robot_xy - self._motion_start_xy))
            if traveled >= fwd_dist:
                self._motion_state = "DONE"
                self._test_done = True
            else:
                twist.linear.x = fwd_speed

        self.cmd_pub.publish(twist)

    def destroy_node(self):
        try:
            self.navigator.stop_robot()
        except Exception:
            pass
        return super().destroy_node()


def main():
    rclpy.init()
    node = TB4SkewTestNode()
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
