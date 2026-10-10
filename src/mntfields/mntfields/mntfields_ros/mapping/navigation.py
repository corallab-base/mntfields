import math
import numpy as np
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Path


def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def resample_polyline(path_xy: np.ndarray, ds: float = 0.05) -> np.ndarray:
    """
    Densify a polyline to roughly fixed spacing ds (meters).
    This does NOT "smooth corners" much; it mainly prevents sparse-point jumpiness.
    """
    path_xy = np.asarray(path_xy, dtype=np.float32)
    if len(path_xy) < 2:
        return path_xy

    out = [path_xy[0]]
    acc = 0.0
    cur = path_xy[0].astype(np.float32)

    for i in range(1, len(path_xy)):
        nxt = path_xy[i].astype(np.float32)
        seg = float(np.linalg.norm(nxt - cur))
        if seg < 1e-6:
            continue

        direction = (nxt - cur) / seg
        dist_left = seg
        while dist_left >= (ds - acc):
            step = (ds - acc)
            cur = cur + direction * step
            out.append(cur.copy())
            dist_left -= step
            acc = 0.0
        acc += dist_left
        cur = nxt

    if float(np.linalg.norm(out[-1] - path_xy[-1])) > 1e-3:
        out.append(path_xy[-1])

    return np.asarray(out, dtype=np.float32)


def cumulative_s(path_xy: np.ndarray) -> np.ndarray:
    """Cumulative arc length along polyline points."""
    if len(path_xy) == 0:
        return np.zeros((0,), dtype=np.float32)
    s = np.zeros((len(path_xy),), dtype=np.float32)
    for i in range(1, len(path_xy)):
        s[i] = s[i - 1] + float(np.linalg.norm(path_xy[i] - path_xy[i - 1]))
    return s


def closest_segment_projection(path_xy: np.ndarray, robot_xy: np.ndarray, hint_i: int, window: int = 80):
    """
    Find closest point on a segment near hint_i.
    Returns:
      seg_i: index i such that segment is (i -> i+1)
      proj: projected point on that segment
      t: interpolation factor in [0,1]
      dist2: squared distance robot->proj
      sign: cross-track sign (+ left of segment direction, - right)
    """
    n = len(path_xy)
    if n < 2:
        return 0, path_xy[0].copy(), 0.0, float("inf"), 0.0

    a = max(0, min(int(hint_i), n - 2))
    b = min(n - 1, a + int(window))
    best = None

    rx, ry = float(robot_xy[0]), float(robot_xy[1])

    for i in range(a, b):
        p0 = path_xy[i]
        p1 = path_xy[i + 1]
        vx = float(p1[0] - p0[0])
        vy = float(p1[1] - p0[1])
        denom = vx * vx + vy * vy
        if denom < 1e-9:
            continue

        wx = rx - float(p0[0])
        wy = ry - float(p0[1])
        t = (wx * vx + wy * vy) / denom
        t = max(0.0, min(1.0, t))
        proj = np.array([float(p0[0]) + t * vx, float(p0[1]) + t * vy], dtype=np.float32)

        dx = rx - float(proj[0])
        dy = ry - float(proj[1])
        d2 = dx * dx + dy * dy

        cross = vx * (ry - float(p0[1])) - vy * (rx - float(p0[0]))
        sign = 1.0 if cross > 0.0 else -1.0

        if best is None or d2 < best[3]:
            best = (i, proj, t, d2, sign)

    if best is None:
        return a, path_xy[a].copy(), 0.0, float("inf"), 0.0

    return best


def arc_lookahead_point(path_xy: np.ndarray, s: np.ndarray, seg_i: int, seg_t: float, L: float):
    """
    Starting from a projection that lies on segment seg_i at interpolation seg_t,
    walk forward along arc length by L and return that point and the index anchor.
    """
    n = len(path_xy)
    if n < 2:
        return path_xy[-1].copy(), n - 1

    s0 = float(s[seg_i] + seg_t * (s[seg_i + 1] - s[seg_i]))
    target = s0 + float(L)

    k = int(np.searchsorted(s, target))
    if k <= 0:
        return path_xy[0].copy(), 0
    if k >= n:
        return path_xy[-1].copy(), n - 1

    sA, sB = float(s[k - 1]), float(s[k])
    frac = 0.0 if (sB - sA) < 1e-9 else (target - sA) / (sB - sA)
    p = path_xy[k - 1] + frac * (path_xy[k] - path_xy[k - 1])
    return p.astype(np.float32), k


class Navigator:
    def __init__(self, node):
        self.node = node
        # /navigator_path is now published from gridmap_node (full untruncated field path)
        # self.path_pub = node.create_publisher(Path, "/navigator_path", 10)
        self.frame_id = getattr(node, 'nav_frame', 'odom')
        #self.is_real = bool(getattr(node, 'is_real_robot', False))
        self.is_real = bool(False)
        self.use_simple_real = self.is_real and not bool(
            getattr(node, 'use_continuous_real_follower', False))

        self.path_msg = Path()
        self.path_msg.header.frame_id = self.frame_id

        def _param(name: str, default: float) -> float:
            try:
                return float(node.get_parameter(name).value)
            except Exception:
                return default

        self.k_heading = 0.45 if self.is_real else _param('follower_heading_gain', 1.2)

        base_v_max = float(node.get_parameter('max_lin').value)
        base_w_max = float(node.get_parameter('max_ang').value)
        # Previous real-robot hard caps: v_max=0.08m/s, w_max=0.22 rad/s (overly conservative)
        self.v_max = min(base_v_max, 0.12) if self.is_real else base_v_max
        self.w_max = min(base_w_max, 0.40) if self.is_real else min(base_w_max, 1.2)
        self.v_min = 0.02 if self.is_real else 0.04

        self.rotate_thresh = 1.1 if self.is_real else _param('follower_rotate_thresh', 1.5)
        self.rotate_exit = 0.35 if self.is_real else _param('follower_rotate_exit', 0.15)
        self.near_goal_rotate_thresh = 0.95 if self.is_real else _param('follower_near_goal_rotate_thresh', 0.55)
        self.forward_suppress_thresh = 0.8 if self.is_real else _param('follower_forward_suppress_thresh', 1.0)
        self.WP_RADIUS = 0.12 if self.is_real else 0.20
        self._rotating = False
        self.rotate_exit_stable_ticks = 3 if self.is_real else 1
        self._rotate_exit_count = 0

        # Previous real-robot angular acceleration: 0.20 rad/s² (took >1s to reach w_max)
        self.w_rate = 0.40 if self.is_real else _param('follower_w_rate', 3.0)
        self._w_prev = 0.0

        self.path_complete = False

        self._raw_path = None
        self._path_xy = None
        self._path_s = None
        self._wp_idx = 0
        self._latched_target = None
        self.last_proj_s = 0.0
        self.last_heading_err = 0.0
        self.last_cross_track = 0.0
        self.last_cmd_v = 0.0
        self.last_cmd_w = 0.0
        self.last_target = None
        self.last_desired_yaw = 0.0
        self.last_robot_yaw = 0.0

    def stop_robot(self):
        self.node.cmd_pub.publish(Twist())
        self.last_cmd_v = 0.0
        self.last_cmd_w = 0.0

    def reset_path(self):
        self.path_complete = False
        self._wp_idx = 0
        self._raw_path = None
        self._rotating = False
        self._w_prev = 0.0
        self._path_xy = None
        self._path_s = None
        self.last_proj_s = 0.0
        self.last_heading_err = 0.0
        self.last_cross_track = 0.0
        self.last_cmd_v = 0.0
        self.last_cmd_w = 0.0
        self.last_target = None
        self.last_desired_yaw = 0.0
        self.last_robot_yaw = 0.0
        self._rotate_exit_count = 0
        self._latched_target = None

    def publish_path(self, path_np):
        pass  # moved to gridmap_node._publish_nav_path (full untruncated field path)

    def _rate_limit(self, w_des: float, dt: float) -> float:
        if dt <= 0.0:
            return w_des
        dw = float(np.clip(w_des - self._w_prev, -self.w_rate * dt, self.w_rate * dt))
        w = self._w_prev + dw
        self._w_prev = w
        return w

    def drive_path(self, path):
        self.path_complete = False

        if self.node.robot_pos is None:
            return False
        if path is None:
            self.stop_robot()
            return False

        raw = np.asarray(path, dtype=np.float32)
        if raw.ndim != 2 or raw.shape[1] != 2 or len(raw) < 2:
            self.stop_robot()
            return False

        if self.use_simple_real:
            return self._drive_path_simple(raw)

        if (self._raw_path is None
                or self._raw_path.shape != raw.shape
                or not np.array_equal(self._raw_path, raw)):
            self._raw_path = raw.copy()
            self._path_xy = resample_polyline(raw, ds=0.05)
            self._path_s = cumulative_s(self._path_xy)
            self._wp_idx = 0
            self._w_prev = 0.0
            # Do NOT reset _rotating here.  Resetting it while the robot is
            # mid-turn causes heading_err to drop slightly below rotate_thresh
            # on the next tick, which drops back to normal-driving with
            # _w_prev=0 — a sudden angular-velocity step that appears as a
            # jerk.  Let the heading-error checks below control rotation state.

        path_xy = self._path_xy
        if self.is_real and self.frame_id == "odom":
            robot_src = getattr(self.node, 'robot_pos_odom', None)
            yaw_src = getattr(self.node, 'robot_yaw_odom', 0.0)
        else:
            robot_src = self.node.robot_pos
            yaw_src = self.node.robot_yaw

        robot_xy = np.asarray(robot_src[:2], dtype=np.float32)
        yaw = float(yaw_src)

        goal_reached_dist = float(self.node.get_parameter('goal_reached_dist').value)
        dist_to_final = float(np.linalg.norm(path_xy[-1] - robot_xy))
        if dist_to_final < goal_reached_dist:
            self.stop_robot()
            self.path_complete = True
            return True

        self.publish_path(path_xy)

        seg_i, proj, seg_t, dist2, sign = closest_segment_projection(
            path_xy, robot_xy, self._wp_idx, window=100)
        self._wp_idx = seg_i
        path_s = self._path_s
        proj_s = float(path_s[seg_i] + seg_t * (path_s[seg_i + 1] - path_s[seg_i]))
        self.last_proj_s = max(self.last_proj_s, proj_s)
        self.last_cross_track = float(math.sqrt(max(0.0, dist2))) * float(sign)
        near_goal = dist_to_final < 0.45
        if near_goal:
            L = float(np.clip(dist_to_final * 0.6, 0.06, 0.20))
            target = path_xy[-1].copy()
        else:
            L = float(np.clip(dist_to_final * 0.3, 0.20, 0.35))
            target, _ = arc_lookahead_point(path_xy, path_s, seg_i, seg_t, L)
        self.last_target = target.copy()

        dx = float(target[0] - robot_xy[0])
        dy = float(target[1] - robot_xy[1])
        desired_yaw = math.atan2(dy, dx)
        heading_err = wrap_pi(desired_yaw - yaw)
        self.last_heading_err = float(heading_err)
        self.last_desired_yaw = float(desired_yaw)
        self.last_robot_yaw = float(yaw)

        if not self._rotating and abs(heading_err) > self.rotate_thresh:
            self._rotating = True
            self._w_prev = 0.0
            self._rotate_exit_count = 0

        if self._rotating:
            if abs(heading_err) < self.rotate_exit:
                self._rotate_exit_count += 1
                if self._rotate_exit_count >= self.rotate_exit_stable_ticks:
                    self._rotating = False
                    self._w_prev = 0.0
                    self._rotate_exit_count = 0
            else:
                self._rotate_exit_count = 0
                w = math.copysign(
                    float(np.clip(self.k_heading * abs(heading_err), 0.15, self.w_max)),
                    heading_err)
                cmd = Twist()
                cmd.angular.z = w
                self.node.cmd_pub.publish(cmd)
                self.last_cmd_v = 0.0
                self.last_cmd_w = w
                return False

        if near_goal and abs(heading_err) > self.near_goal_rotate_thresh:
            w = math.copysign(
                float(np.clip(self.k_heading * abs(heading_err), 0.12, self.w_max)),
                heading_err)
            cmd = Twist()
            cmd.angular.z = w
            self.node.cmd_pub.publish(cmd)
            self.last_cmd_v = 0.0
            self.last_cmd_w = w
            return False

        v = self.v_max * max(0.10, math.cos(heading_err) ** 2)
        if dist_to_final < 0.5:
            v *= dist_to_final / 0.5
        if near_goal:
            v = min(v, 0.12)
            v = float(np.clip(v, 0.0, self.v_max))
        else:
            v = float(np.clip(v, self.v_min, self.v_max))

        if abs(heading_err) > self.forward_suppress_thresh:
            v = 0.0

        w_des = float(np.clip(self.k_heading * heading_err, -self.w_max, self.w_max))
        dt = float(getattr(self.node, 'dt', 0.05))
        w = self._rate_limit(w_des, dt)

        cmd = Twist()
        cmd.linear.x = v
        cmd.angular.z = w
        self.node.cmd_pub.publish(cmd)
        self.last_cmd_v = v
        self.last_cmd_w = w
        return False

    def _drive_path_simple(self, raw: np.ndarray):
        if (self._raw_path is None
                or self._raw_path.shape != raw.shape
                or not np.array_equal(self._raw_path, raw)):
            self._raw_path = raw.copy()
            self._path_xy = resample_polyline(raw, ds=0.08)
            self._wp_idx = 1 if len(self._path_xy) > 1 else 0
            self._latched_target = None
            self._rotating = False
            self._rotate_exit_count = 0
            self._w_prev = 0.0

        path_xy = self._path_xy
        robot_src = getattr(self.node, 'robot_pos_odom', None)
        yaw_src = getattr(self.node, 'robot_yaw_odom', 0.0)
        if robot_src is None:
            self.stop_robot()
            return False
        robot_xy = np.asarray(robot_src[:2], dtype=np.float32)
        yaw = float(yaw_src)
        self.last_robot_yaw = yaw

        goal_reached_dist = float(self.node.get_parameter('goal_reached_dist').value)
        dist_to_final = float(np.linalg.norm(path_xy[-1] - robot_xy))
        if dist_to_final < goal_reached_dist:
            self.stop_robot()
            self.path_complete = True
            return True

        if self._latched_target is None:
            self._latched_target = path_xy[min(self._wp_idx, len(path_xy) - 1)].copy()

        # Progress-aware waypoint advancement for real hardware. If the robot has
        # already moved closer to a later path sample than the current latched one,
        # stop trying to return to the stale waypoint.
        start_i = max(0, min(self._wp_idx - 1, len(path_xy) - 1))
        rem = path_xy[start_i:]
        if len(rem) > 0:
            d_rem = np.linalg.norm(rem - robot_xy[None, :], axis=1)
            closest_rel = int(np.argmin(d_rem))
            closest_idx = start_i + closest_rel
            closest_dist = float(d_rem[closest_rel])
            if closest_idx > self._wp_idx and closest_dist < 0.22:
                self._wp_idx = closest_idx
                self._latched_target = path_xy[min(self._wp_idx, len(path_xy) - 1)].copy()
                self._rotating = False
                self._rotate_exit_count = 0
                self._w_prev = 0.0

        while self._wp_idx < len(path_xy) - 1:
            wp_dist = float(np.linalg.norm(self._latched_target - robot_xy))
            if wp_dist <= self.WP_RADIUS:
                self._wp_idx += 1
                self._latched_target = path_xy[min(self._wp_idx, len(path_xy) - 1)].copy()
                self._rotating = False
                self._rotate_exit_count = 0
                self._w_prev = 0.0
            else:
                break

        target = self._latched_target.copy()
        self.last_target = target.copy()
        self.last_proj_s = float(self._wp_idx)
        self.last_cross_track = 0.0

        dx = float(target[0] - robot_xy[0])
        dy = float(target[1] - robot_xy[1])
        target_dist = math.hypot(dx, dy)
        desired_yaw = math.atan2(dy, dx)
        heading_err = wrap_pi(desired_yaw - yaw)
        self.last_desired_yaw = desired_yaw
        self.last_heading_err = heading_err

        rotate_only_thresh = 0.35
        rotate_exit = 0.12
        if self._rotating:
            if abs(heading_err) < rotate_exit:
                self._rotate_exit_count += 1
                if self._rotate_exit_count >= self.rotate_exit_stable_ticks:
                    self._rotating = False
                    self._rotate_exit_count = 0
                    self._w_prev = 0.0
            else:
                self._rotate_exit_count = 0
        else:
            if abs(heading_err) > rotate_only_thresh:
                self._rotating = True
                self._rotate_exit_count = 0
                self._w_prev = 0.0

        if self._rotating:
            w_des = float(np.clip(0.55 * heading_err, -self.w_max, self.w_max))
            w = self._rate_limit(w_des, float(getattr(self.node, 'dt', 0.05)))
            cmd = Twist()
            cmd.angular.z = w
            self.node.cmd_pub.publish(cmd)
            self.last_cmd_v = 0.0
            self.last_cmd_w = w
            return False

        v = float(np.clip(0.35 * target_dist, self.v_min, self.v_max))
        if abs(heading_err) > 0.20:
            v *= 0.20
        if dist_to_final < 0.5:
            v = min(v, 0.05)

        w_des = float(np.clip(0.45 * heading_err, -self.w_max, self.w_max))
        w = self._rate_limit(w_des, float(getattr(self.node, 'dt', 0.05)))
        cmd = Twist()
        cmd.linear.x = v
        cmd.angular.z = w
        self.node.cmd_pub.publish(cmd)
        self.last_cmd_v = v
        self.last_cmd_w = w
        return False
