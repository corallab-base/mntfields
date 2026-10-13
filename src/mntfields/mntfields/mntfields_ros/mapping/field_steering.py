"""
field_steering.py — drive by following the travel-time field's gradient directly.

Instead of planning a path piece, validating it and following it (and stopping to
replan), every control tick rolls the field out a few short steps from the robot
toward the current target (a map-route subgoal or the frontier) and steers toward
that lookahead point: pure pursuit on the field's own streamline.
"""

import math

import numpy as np


def _wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


class FieldSteering:
    def __init__(self, v_max, w_max, reverse=False, lookahead_steps=4, step=0.10, alpha=0.8,
                 rotate_thresh=1.2, k_w=1.8, accel=0.5, ang_accel=2.5, v_min=0.04,
                 max_field_deviation=math.radians(60.0)):
        self.v_max, self.w_max = v_max, w_max
        self.reverse = reverse
        self.lookahead_steps, self.step, self.alpha = lookahead_steps, step, alpha
        self.rotate_thresh, self.k_w = rotate_thresh, k_w
        self.accel, self.ang_accel, self.v_min = accel, ang_accel, v_min
        self.max_field_deviation = max_field_deviation
        self.route_overrides = 0
        self._v = 0.0
        self._w = 0.0
        self.last_rollout = None

    def reset(self):
        self._v = 0.0
        self._w = 0.0

    def rollout(self, region, xy, target):
        pts = [np.asarray(xy[:2], dtype=np.float32)]
        cur = pts[0]
        for _ in range(self.lookahead_steps):
            if float(np.linalg.norm(target - cur)) <= self.step:
                pts.append(np.asarray(target, dtype=np.float32))
                break
            cur = region.get_next_waypoint(cur, target, step_size=self.step, alpha=self.alpha)
            pts.append(np.asarray(cur, dtype=np.float32))
        self.last_rollout = np.array(pts)
        return self.last_rollout

    def compute(self, region, xy, yaw, target, dist_to_goal, dt):
        """(v, w) toward target along the field; dist_to_goal slows down at the very end."""
        target = np.asarray(target[:2], dtype=np.float32)
        pts = self.rollout(region, xy, target)
        look = pts[-1]
        xy = np.asarray(xy[:2], dtype=np.float32)
        d = look - xy
        if float(np.linalg.norm(d)) < 1e-3:
            return self._smooth(0.0, 0.0, dt)
        # The target is a nearby point on the map route. Where the field disagrees with
        # it strongly (e.g. it has not learned a narrow passage), follow the route.
        to_target = target - xy
        if float(np.linalg.norm(to_target)) > 1e-3:
            cosang = float(np.dot(d, to_target) / (np.linalg.norm(d) * np.linalg.norm(to_target)))
            if cosang < math.cos(self.max_field_deviation):
                d = to_target
                self.route_overrides += 1
        heading = yaw + (math.pi if self.reverse else 0.0)
        err = _wrap(math.atan2(float(d[1]), float(d[0])) - heading)
        if abs(err) > self.rotate_thresh:
            v = 0.0
            w = math.copysign(min(self.w_max, max(0.3, self.k_w * abs(err))), err)
        else:
            v = self.v_max * max(0.0, math.cos(err)) ** 2
            if dist_to_goal < 0.5:
                v *= max(0.2, dist_to_goal / 0.5)
            v = max(v, self.v_min)
            w = float(np.clip(self.k_w * err, -self.w_max, self.w_max))
        if self.reverse:
            v = -v
        return self._smooth(v, w, dt)

    def _smooth(self, v, w, dt):
        dv = float(np.clip(v - self._v, -self.accel * dt, self.accel * dt))
        dw = float(np.clip(w - self._w, -self.ang_accel * dt, self.ang_accel * dt))
        # Reversing direction of travel: come to a stop first.
        if self._v * v < 0.0:
            dv = float(np.clip(-self._v, -self.accel * dt, self.accel * dt))
        self._v += dv
        self._w += dw
        return self._v, self._w
