import numpy as np
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from builtin_interfaces.msg import Duration as DurationMsg


class MarkerPublisher:
    def __init__(self, node):
        self.node = node

    def _goal_xy(self):
        goal = getattr(self.node, '_current_goal', None)
        if goal is None:
            goal = getattr(self.node, 'goal_xy', None)
        return goal

    def _frontier_world_points(self):
        gridmap = getattr(self.node, 'gridmap', None)
        raw_pts = getattr(gridmap, 'slam_frontier_points', None) if gridmap is not None else None
        if raw_pts is not None and len(raw_pts) > 0:
            return np.asarray(raw_pts, dtype=float)

        extractor = getattr(self.node, 'extractor', None)
        if extractor is not None and getattr(extractor, '_frontier_cells', None) is not None:
            rows, cols = np.where(extractor._frontier_cells)
            if len(rows) > 0:
                pts = [extractor.grid_to_world(int(r), int(c)) for r, c in zip(rows, cols)]
                return np.asarray(pts, dtype=float)

        if gridmap is not None and getattr(gridmap, 'frontier', None) is not None:
            frontier = gridmap.frontier
            iy, ix = np.where(frontier > 0)
            if len(ix) == 0:
                return np.zeros((0, 2), dtype=float)
            cell = gridmap.cell_size
            origin = gridmap.origin
            half = gridmap.map_size / 2.0
            px = (origin[0] - half) + (ix + 0.5) * cell
            py = (origin[1] - half) + (iy + 0.5) * cell
            return np.stack([px, py], axis=1)

        return np.zeros((0, 2), dtype=float)

    def _frontier_centroids(self):
        extractor = getattr(self.node, 'extractor', None)
        if extractor is not None:
            cands = extractor.get_frontier_centroids()
            if cands is not None and len(cands) > 0:
                return np.asarray(cands, dtype=float)
        return np.zeros((0, 2), dtype=float)

    def publish_goal_marker(self):
        goal_xy = self._goal_xy()
        if goal_xy is None:
            return

        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.node.get_clock().now().to_msg()
        marker.ns = 'goal'
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = float(goal_xy[0])
        marker.pose.position.y = float(goal_xy[1])
        marker.pose.position.z = 0.15
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.25
        marker.scale.y = 0.25
        marker.scale.z = 0.25
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.2
        marker.color.a = 1.0
        self.node.goal_marker_pub.publish(marker)

    def publish_frontiers(self):
        self.publish_frontier_markers()

    def publish_frontier_markers(self):
        pts = self._frontier_world_points()
        stamp = self.node.get_clock().now().to_msg()

        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = stamp
        marker.ns = 'frontiers'
        marker.id = 1
        marker.type = Marker.SPHERE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.07
        marker.scale.y = 0.07
        marker.scale.z = 0.07
        marker.color.r = 0.4
        marker.color.g = 1.0
        marker.color.b = 0.4
        marker.color.a = 0.90
        dur = DurationMsg()
        dur.sec = 12
        marker.lifetime = dur

        for x, y in pts:
            p = Point()
            p.x = float(x)
            p.y = float(y)
            p.z = 0.08
            marker.points.append(p)

        self.node.frontier_marker_pub.publish(marker)

    def publish_doorway_markers(self):
        cps = getattr(self.node, '_connection_points', [])

        array = MarkerArray()
        stamp = self.node.get_clock().now().to_msg()

        del_marker = Marker()
        del_marker.header.frame_id = 'map'
        del_marker.header.stamp = stamp
        del_marker.ns = 'doorways'
        del_marker.id = 0
        del_marker.action = Marker.DELETEALL
        array.markers.append(del_marker)

        seen = []
        for idx, cp in enumerate(cps):
            try:
                pos = cp['pos']
            except (KeyError, TypeError):
                continue
            pos_arr = np.array(pos, dtype=float)
            if any(np.linalg.norm(pos_arr - s) < 0.2 for s in seen):
                continue
            seen.append(pos_arr)

            visited = bool(cp.get('visited', False))
            width = float(cp.get('width', 0.5))
            kind = str(cp.get('kind', 'door'))

            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp = stamp
            m.ns = 'doorways'
            m.id = idx * 2
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(pos[0])
            m.pose.position.y = float(pos[1])
            m.pose.position.z = 0.5
            m.pose.orientation.w = 1.0
            sphere_r = max(0.15, width * 0.3)
            m.scale.x = sphere_r
            m.scale.y = sphere_r
            m.scale.z = sphere_r
            m.color.a = 0.9
            if visited:
                m.color.r = 0.0; m.color.g = 0.9; m.color.b = 0.2
            else:
                m.color.r = 1.0; m.color.g = 0.5; m.color.b = 0.0
            m.lifetime.sec = 2
            array.markers.append(m)

            t = Marker()
            t.header.frame_id = 'map'
            t.header.stamp = stamp
            t.ns = 'doorways'
            t.id = idx * 2 + 1
            t.type = Marker.TEXT_VIEW_FACING
            t.action = Marker.ADD
            t.pose.position.x = float(pos[0])
            t.pose.position.y = float(pos[1])
            t.pose.position.z = 0.9
            t.pose.orientation.w = 1.0
            t.scale.z = 0.18
            t.color.r = 1.0; t.color.g = 1.0; t.color.b = 1.0; t.color.a = 1.0
            t.text = f"{kind[0].upper()}{idx} {width:.2f}m"
            t.lifetime.sec = 2
            array.markers.append(t)

        self.node.doorway_marker_pub.publish(array)
