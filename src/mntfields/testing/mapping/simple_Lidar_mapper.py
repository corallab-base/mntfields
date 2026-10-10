#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import math

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid, Odometry
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped

class SimpleLidarMapper(Node):
    def __init__(self):
        super().__init__('simple_lidar_mapper')
        
        # Parameters
        self.declare_parameter('map_size', 20.0)
        self.declare_parameter('resolution', 0.05)
        
        map_size = self.get_parameter('map_size').value
        self.resolution = self.get_parameter('resolution').value
        
        # Calculate grid dimensions
        self.width = int(map_size / self.resolution)
        self.height = int(map_size / self.resolution)
        
        # Initialize map (all unknown = -1)
        self.map_data = np.full(self.width * self.height, -1, dtype=np.int8)
        
        # Robot state
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_theta = 0.0
        
        # TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # Subscribers
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        
        # Publisher
        self.map_pub = self.create_publisher(OccupancyGrid, '/map', 10)
        
        # Timer to publish map
        self.create_timer(0.1, self.publish_map)
        
        self.get_logger().info('Simple LiDAR Mapper started')
    
    def odom_callback(self, msg):
        """Update robot position from odometry"""
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        
        # Extract yaw from quaternion
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.robot_theta = math.atan2(siny_cosp, cosy_cosp)
    
    def scan_callback(self, msg):
        """Process laser scan and update map"""
        angle = msg.angle_min
        
        for i, distance in enumerate(msg.ranges):
            # Skip invalid readings
            if math.isnan(distance) or math.isinf(distance):
                angle += msg.angle_increment
                continue
                
            if distance < msg.range_min or distance > msg.range_max:
                angle += msg.angle_increment
                continue
            
            # Calculate obstacle position in world frame
            obstacle_x = self.robot_x + distance * math.cos(angle + self.robot_theta)
            obstacle_y = self.robot_y + distance * math.sin(angle + self.robot_theta)
            
            # Convert to grid coordinates (center of map is origin)
            grid_x = int((obstacle_x + self.width * self.resolution / 2) / self.resolution)
            grid_y = int((obstacle_y + self.height * self.resolution / 2) / self.resolution)
            
            # Check bounds
            if 0 <= grid_x < self.width and 0 <= grid_y < self.height:
                index = grid_y * self.width + grid_x
                self.map_data[index] = 100
            
            angle += msg.angle_increment
    
    def publish_map(self):
        """Publish TF and occupancy grid"""
        now = self.get_clock().now().to_msg()
        
        # Publish map->odom transform FIRST
        t = TransformStamped()
        t.header.stamp = now
        t.header.frame_id = 'map'
        t.child_frame_id = 'odom'
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0
        self.tf_broadcaster.sendTransform(t)
        
        # Then publish map with same timestamp
        msg = OccupancyGrid()
        msg.header.stamp = now
        msg.header.frame_id = 'map'
        
        msg.info.resolution = self.resolution
        msg.info.width = self.width
        msg.info.height = self.height
        
        # Origin at center of map
        msg.info.origin.position.x = -self.width * self.resolution / 2
        msg.info.origin.position.y = -self.height * self.resolution / 2
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0
        
        msg.data = self.map_data.tolist()
        self.map_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = SimpleLidarMapper()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
    
import numpy as np
import networkx as nx
import skimage
import skimage.morphology
import skimage.segmentation
import cv2
import torch
import scipy
import json
import os

from typing import Set, List, Dict, Union, Tuple
from itertools import combinations
# core classes
from mntfields.mntfields_core.mapping.rooms_doors import (
    Room,
    Door,
    ConfirmedRoom,
    UnconfirmedRoom,
)
from mntfields.mntfields_core.core.region import Region
from mntfields.mntfields_core.core.exploration_state import FrontierNode
# utils (your merged utils file)
from mntfields.utils.mntfields_utils import (
    get_neighbor_at_least,
    connected_components,
)

class LidarGridmap:
    def __init__(self, trav_map, regions, n_cells=100, occupancy_threshold=0.01, 
             near_dist=0.1, room_seg_config=(7, 2), map_size=20.0, origin=(0.0, 0.0)):
        """
        Initialize the gridmap.
        :param n_cells: Number of cells along each dimension of the gridmap.
        """
        self.n_cells = n_cells
        self.map_size = map_size  # meters
        self.origin = np.array(origin)  # map center
        self.cell_size = map_size / n_cells  # meters per cell

        self.data = np.zeros((n_cells, n_cells), dtype=np.uint8)
        self.visited = np.zeros_like(self.data)
        self.frontier = np.zeros_like(self.data)

        self.confirmed_rooms: Dict[int, ConfirmedRoom] = {}
        self.confirmed_room_map = np.zeros_like(self.data)
        self.unconfirmed_rooms: Dict[int, UnconfirmedRoom] = {}
        self.room_seg = np.zeros_like(self.data)
        self.outer_label = 1
        self.new_confirmed_rooms: List[ConfirmedRoom] = []
        self.rooms = []

        self.occupancy_threshold = occupancy_threshold

        self.confirmed_label = 50
        self.unconfirmed_label = 100

        self.room_graph = nx.Graph()
        self.room_connectivity_graph = nx.Graph()
        self.n_doors = 0
        self.doors: Set[Door] = set()

        self.trav_map = trav_map
        self.near_dist = near_dist

        self.boundary_graph = nx.Graph()
        self.discovery_map = np.zeros_like(self.data, dtype=np.uint64)

        self.all_lidar = None
        self.kd_tree = None
        
        self.room_seg_config = room_seg_config
        self.regions: Dict[int, Region] = regions

    def _world_to_grid(self, x, y):
        """
        Convert world coordinates to grid coordinates.
        :param x: x-coordinate in world space (meters)
        :param y: y-coordinate in world space (meters)
        :return: (i, j) grid coordinates
        """
        # Map goes from [origin - map_size/2] to [origin + map_size/2]
        x_rel = x - (self.origin[0] - self.map_size / 2)
        y_rel = y - (self.origin[1] - self.map_size / 2)

        j = int(x_rel / self.cell_size)
        i = int(y_rel / self.cell_size)

        # Clamp to valid indices
        i = max(0, min(i, self.n_cells - 1))
        j = max(0, min(j, self.n_cells - 1))
        return i, j

    def _grid_to_world(self, i, j):
        x = (j + 0.5) * self.cell_size + (self.origin[0] - self.map_size / 2)
        y = (i + 0.5) * self.cell_size + (self.origin[1] - self.map_size / 2)
        return x, y

    def discover_regions(self, value):
        regs = []
        i = 0
        value = value.astype(int)
        while value != 0:
            if value & 1 != 0:
                regs.append(i)
            value = value >> 1
            i += 1
        return regs
    
    def can_access(self, region_version, pos):
        i, j = self._world_to_grid(pos[0], pos[1])
        return self.discovery_map[i, j].astype(int) & (1 << region_version) != 0
    
    def set_access(self, region_version, mask):
        self.discovery_map[mask] |= 1 << region_version
    
    def get_target_version(self, pos):
        i, j = self._world_to_grid(pos[0], pos[1])
        found = False
        r = 0
        while not found:
            mi = max(i - r, 0)
            mj = max(j - r, 0)
            Mi = min(i + r + 1, self.n_cells)
            Mj = min(j + r + 1, self.n_cells)
            values, counts = np.unique(self.discovery_map[mi:Mi, mj:Mj], return_counts=True)
            versions = []
            for value in values:
                versions += self.discover_regions(value)
            if len(versions) > 0:
                return max(versions)
            r += 1
            if r >= 10:
                breakpoint()
                return None
            
    def get_discovery_boundary(self, pos):
        i, j = self._world_to_grid(pos[0], pos[1])
        if self.discovery_map[i, j] != 0:
            return self.discovery_map[i, j]
        r = 10
        mi = max(i - r, 0)
        mj = max(j - r, 0)
        Mi = min(i + r + 1, self.n_cells)
        Mj = min(j + r + 1, self.n_cells)
        versions, counts = np.unique(self.discovery_map[mi:Mi, mj:Mj], return_counts=True)
        sorted_idx = np.argsort(counts)
        if versions[sorted_idx[-1]] == 0:
            if len(versions) == 1:
                breakpoint()
            return versions[sorted_idx[-2]]
        return versions[sorted_idx[-1]]

    def get_room_from_mask(self, mask):
        labels, counts = np.unique(self.confirmed_room_map[mask], return_counts=True)
        if len(labels) == 1 and labels[0] == 0:
            return -1
        sorted_idx = np.argsort(counts)
        labels = labels[sorted_idx]
        if labels[-1] == 0:
            return labels[-2]
        return labels[-1]

    def room_is_confirmed(self, room: Room):
        room_bounds = skimage.segmentation.find_boundaries(room.mask, mode='outer')
        room_bounds[self.data == 1] = 0
        room_bounds = room_bounds.astype(np.uint8)
        if room.mask.sum() < 200: #40: All these values are for real-scale environments
            if self.frontier[room_bounds == 1].sum() > 15: #5:
                return False
            return True
        if self.frontier[room_bounds == 1].sum() > 80: #15:
            return False
        if self.frontier[room.mask].sum() > 50: #10:
            return False
        search = np.zeros_like(room_bounds)
        search[(self.room_seg == self.outer_label) & (room_bounds == 1)] = 1
        threshold = cv2.threshold(search, 0.5, 1, cv2.THRESH_BINARY)[1]
        n_cc, door_labels = cv2.connectedComponents(threshold)
        for door_label in np.unique(door_labels):
            if door_label == 0:
                continue
            if (door_labels == door_label).sum() >= 20: #20:
                return False
        return True

    def in_confirmed_room(self, pos):
        i, j = self._world_to_grid(pos[0], pos[1])
        return self.confirmed_room_map[i, j] > 0

    def add_confirmed_room(self, room: Room):
        m = np.unique(self.confirmed_room_map[room.mask])
        if len(m) != 1:
            return None   # overlaps existing confirmed room — skip silently
        if m[0] != 0:
            return None   # already confirmed
        # Size sanity check — reject tiny fragments
        if room.mask.sum() < 100:   # < 1 m² at 0.1m/cell
            return None
        confirmed_room = ConfirmedRoom(room.mask)
        self.room_connectivity_graph.add_node(confirmed_room.id)
        self.confirmed_rooms[confirmed_room.id] = confirmed_room
        self.confirmed_room_map[confirmed_room.mask] = confirmed_room.id
        return confirmed_room

    def get_room_unexplored(self, room: Room):
        room_bounds = skimage.segmentation.find_boundaries(room.mask, mode='outer')
        room_bounds[self.data == 1] = 0
        bound_masked = self.room_seg.copy()
        bound_masked[room_bounds == 0] = 0
        if (bound_masked == self.outer_label).sum() == 0:
            return None
        return np.stack(np.where(bound_masked == self.outer_label), -1)
    
    def get_door_idx(self, room: Room):
        room_bounds = skimage.segmentation.find_boundaries(room.mask, mode='outer')
        room_bounds[self.data == 1] = 0
        bound_masked = self.room_seg.copy()
        bound_masked[room_bounds == 0] = 0
        if (bound_masked != 0).sum() == 0:
            return None
        return np.stack(np.where(bound_masked != 0), -1)

    def get_confirmed_room_from_id(self, id):
        for room in self.confirmed_rooms.values():
            if room.id == id:
                return room
        return None

    def get_closest_door(self, room: ConfirmedRoom, pos):
        min_dist = 999
        min_door = None
        for door in room.doors:
            dist = np.linalg.norm(door.pos - pos)
            if dist < min_dist:
                min_door = door
                min_dist = dist
        if min_dist < 3.0:
            return min_door
        return None

    def compute_doors_in_room(self, regions: Dict[int, Region]):
        edges = []
        for room in list(self.confirmed_rooms.values()) + list(self.unconfirmed_rooms.values()):
            for door_pair in combinations(room.doors, 2):
                weight = np.linalg.norm(door_pair[0].pos - door_pair[1].pos)
                
                region = room.region
                u = door_pair[0].id
                v = door_pair[1].id
                if not self.room_graph.has_edge(u, v):
                    edges.append((door_pair[0].id, door_pair[1].id, {'weight': weight, 'region': region.version}))
                else:
                    self.room_graph.edges[(u, v)]['region'] = region.version
        for edge in edges:
            self.room_graph.add_edge(edge[0], edge[1], **edge[2])

    def add_doors(self, room, region, regions):
        room_map = np.zeros_like(self.data, dtype=np.int32)
        for confirmed_room in self.confirmed_rooms.values():
            room_map[confirmed_room.mask] = confirmed_room.id
        for unconfirmed_room in self.unconfirmed_rooms.values():
            room_map[unconfirmed_room.mask] = unconfirmed_room.id
    
        room_bounds = skimage.segmentation.find_boundaries(room.mask, mode='outer')
        room_bounds[self.data == 1] = 0
        room_bounds = room_bounds.astype(np.uint8)
    
        # ── Only keep boundary pixels that touch outer_label (true gaps) ──
        # A real door gap is where room boundary meets unassigned/outer space
        # NOT where it touches another already-segmented room
        gap_mask = np.zeros_like(room_bounds, dtype=np.uint8)
        gap_mask[(room_bounds == 1) & (self.room_seg == self.outer_label)] = 1
        # Also include gaps touching visited-but-unsegmented cells
        gap_mask[(room_bounds == 1) & (self.visited == 1) & (self.confirmed_room_map == 0)] = 1
    
        threshold = cv2.threshold(gap_mask, 0.5, 1, cv2.THRESH_BINARY)[1]
        n_cc, door_labels = cv2.connectedComponents(threshold)
    
        nodes = []
        edges = []
        for door_label in np.unique(door_labels):
            if door_label == 0:
                continue
            door_idx_i, door_idx_j = np.where(door_labels == door_label)
            gap_cells = len(door_idx_i)
    
            # Real doorways: 0.6m–3.0m wide (6–30 cells at 0.1m/cell)
            if gap_cells < 6 or gap_cells > 30:
                continue
    
            i_c = int((door_idx_i.max() + door_idx_i.min()) // 2)
            j_c = int((door_idx_j.max() + door_idx_j.min()) // 2)
            door_center = np.array(self._grid_to_world(i_c, j_c))
    
            print(f"[DOOR DEBUG] grid=({i_c},{j_c}) → world=({door_center[0]:.2f},{door_center[1]:.2f})"
                  f" gap_cells={gap_cells}")
    
            door_id = self.n_doors
            self.n_doors += 1
            room_in_doors = np.unique(room_map[door_labels == door_label])
            room_in_doors = [r for r in room_in_doors if r not in [0, room.id]]
    
            nodes.append((door_id, {'room': room.id, 'pos': door_center}))
            door = Door(door_id, door_center, room)
            self.doors.add(door)
            room.doors.append(door)
    
            for r in room_in_doors:
                if r == 0:
                    continue
                if r in self.confirmed_rooms:
                    other_room = self.confirmed_rooms[r]
                elif r in self.unconfirmed_rooms:
                    other_room = self.unconfirmed_rooms[r]
                else:
                    continue
                closest_door = self.get_closest_door(other_room, door_center)
                if closest_door is None:
                    continue
                edges.append((door_id, closest_door.id, {
                    'weight': np.linalg.norm(door_center - closest_door.pos),
                    'region': -1
                }))
    
        new_node_ids = [x[0] for x in nodes]
        for door_pair in combinations(room.doors, 2):
            weight = np.linalg.norm(door_pair[0].pos - door_pair[1].pos)
            for reg in regions.values():
                if reg.in_bound(door_pair[0].pos) and reg.in_bound(door_pair[1].pos):
                    edges.append((door_pair[0].id, door_pair[1].id, {
                        'weight': weight, 'region': reg.version
                    }))
                    break
    
        self.room_graph.add_nodes_from(nodes)
        for edge in edges:
            self.room_graph.add_edge(edge[0], edge[1], **edge[2])

    def find_confirmed_room(self, room: Room):
        # returns the ConfirmedRoom object if room is a previously confirmed room
        for cr in self.confirmed_rooms.values():
            if (room.mask == cr.mask).all():
                return cr
        return None

    def update_rooms(self):
        self.rooms = []
        for label in np.unique(self.room_seg):
            if label == 0:
                continue
            if label == self.outer_label:
                continue
            room = Room(self.room_seg == label)
            confirmed_room = self.find_confirmed_room(room)
            self.rooms.append((room, confirmed_room))

    def update_unconfirmed_rooms(self):
        # coalesce unconfirmed rooms
        clusters: List[List[UnconfirmedRoom]] = []
        for room in self.unconfirmed_rooms.values():
            room_b_mask = self.get_boundary_mask(room)
            found_cluster = False
            for cluster in clusters:
                belongs_to_cluster = False
                for other_room in cluster:
                    other_room_b_mask = self.get_boundary_mask(other_room)
                    if (room_b_mask & other_room_b_mask).sum() > 0:
                        belongs_to_cluster = True
                        break
                if belongs_to_cluster:
                    found_cluster = True
                    cluster.append(room)
                    break
            if not found_cluster:
                clusters.append([room])
        for cluster in clusters:
            for room in cluster:
                del self.unconfirmed_rooms[room.id]
        for cluster in clusters:
            new_mask = cluster[0].mask
            for room in cluster:
                new_mask = new_mask | room.mask
            new_unconfirmed_room = UnconfirmedRoom(new_mask)
            self.unconfirmed_rooms[new_unconfirmed_room.id] = new_unconfirmed_room
            doors = [door for room in cluster for door in room.doors]
            for door in doors:
                door.room = new_unconfirmed_room
                if door.id in self.room_graph.nodes:
                    self.room_graph.nodes[door.id]['room'] = new_unconfirmed_room.id
            new_unconfirmed_room.doors = doors
        
        # split large unconfirmed rooms:
        for room in list(self.unconfirmed_rooms.values()):
            self.split_room(room)
    
    def split_from_point(self, room_mask: np.ndarray, point, split_axis, max_split_len_x=20, max_split_len_y=25, min_split_size=400): #10, 15, 400
        # point (i, j)
        split_mask = room_mask.copy().astype(np.uint8)
        split_line_mask = np.zeros_like(room_mask)
        n = split_mask.shape[0]
        split_len = 0
        if split_axis == 0:
            # split horizontally
            i = point[0]
            for j in range(max_split_len_x):
                j = point[1] - j
                if j < 0:
                    break
                if self.visited[i, j] != 1:
                    break
                if self.data[i, j] == 1:
                    break
                split_mask[i, j] = 0
                split_line_mask[i, j] = 1
                split_len += 1
            for j in range(max_split_len_x):
                j = point[1] + j
                if j >= n:
                    break
                if self.visited[i, j] != 1:
                    break
                if self.data[i, j] == 1:
                    break
                split_mask[i, j] = 0
                split_line_mask[i, j] = 1
                split_len += 1
        else:
            j = point[1]
            for i in range(max_split_len_y):
                i = point[0] - i
                if i < 0:
                    break
                if self.visited[i, j] != 1:
                    break
                if self.data[i, j] == 1:
                    break
                split_mask[i, j] = 0
                split_line_mask[i, j] = 1
                split_len += 1
            for i in range(max_split_len_y):
                i = point[0] + i
                if i >= n:
                    break
                if self.visited[i, j] != 1:
                    break
                if self.data[i, j] == 1:
                    break
                split_mask[i, j] = 0
                split_line_mask[i, j] = 1
                split_len += 1
        ccs = connected_components(split_mask)
        ccs[split_line_mask == 1] = 1
        labels = np.unique(ccs)
        if len(labels) < 3: # 0, 1, _
            return None, None
        
        for label in labels:
            if label == 0:
                continue
            if (ccs == label).sum() < min_split_size:
                return None, None
        
        return ccs, split_len
    
    def split_room(self, room: Room):
        room_mask = room.mask
        mask = room_mask.astype(np.uint8)
        skel = skimage.morphology.skeletonize(mask)
        
        n = skel.shape[0]
        skel_idxs = np.stack(np.where(skel), axis=-1)
        ccss = []
        for idx in skel_idxs:
            if idx[0] > 0 and idx[0] < n:
                if skel[idx[0] - 1, idx[1]] == 1 and skel[idx[0] + 1, idx[1]] == 1:
                    ccs, split_len = self.split_from_point(room_mask, idx, 0)
                    if ccs is not None:
                        ccss.append((split_len, ccs))
            if idx[1] > 0 and idx[1] < n:
                if skel[idx[0], idx[1] - 1] == 1 and skel[idx[0], idx[1] + 1] == 1:
                    ccs, split_len = self.split_from_point(room_mask, idx, 1)
                    if ccs is not None:
                        ccss.append((split_len, ccs))
        ccs = None
        if len(ccss) > 0:
            ccs = min(ccss, key=lambda x: x[0])[1]
        if ccs is not None:
            self.remove_room(room)
            labels = np.unique(ccs)
            for label in labels:
                if label == 0:
                    continue
                room = Room(ccs == label)
                if self.room_is_confirmed(room):
                    new_room = self.add_confirmed_room(room)
                    if new_room is not None:
                        print('found newly confirmed room')
                        self.new_confirmed_rooms.append(new_room)

                        if self.regions:  # self.regions is passed in from gridmap_node - new diff for room region attachment
                            ri = int(np.mean(np.where(new_room.mask)[0]))
                            rj = int(np.mean(np.where(new_room.mask)[1]))
                            rx, ry = self._grid_to_world(ri, rj)
                            for region in self.regions.values():
                                b = region.bound.numpy()
                                if (b[0,0] <= rx <= b[1,0] and b[0,1] <= ry <= b[1,1]):
                                    region.add_room(new_room)  # sets new_room.region = region
                                    break

                        self.add_doors(new_room, None, self.regions)
                else:
                    unconfirmed_room = UnconfirmedRoom(ccs == label)
                    self.unconfirmed_rooms[unconfirmed_room.id] = unconfirmed_room
                    #self.add_doors(unconfirmed_room, None, self.regions)
        
    def remove_room(self, room: Union[ConfirmedRoom, UnconfirmedRoom]):
        if room.id in self.confirmed_rooms:
            return
        elif room.id in self.unconfirmed_rooms:
            del self.unconfirmed_rooms[room.id]
        for door in room.doors:
            self.doors.remove(door)
            if door.id in self.room_graph.nodes:
                self.room_graph.remove_node(door.id)
          
    def clear_unconfirmed_rooms(self):
        for r in self.unconfirmed_rooms.values():
            for door in r.doors:
                self.doors.discard(door)   # discard won't raise if missing
                if door.id in self.room_graph.nodes:
                    self.room_graph.remove_node(door.id)
            if r.id in self.room_connectivity_graph.nodes:
                self.room_connectivity_graph.remove_node(r.id)
        self.unconfirmed_rooms = {}

    def is_in_room(self, point, threshold=15):
        def raycast(point, direction, max_dist=50):
            n = self.data.shape[0]
            x = np.array(point)
            dx = np.array(direction)
            for i in range(max_dist):
                if x[0] < 0 or x[0] >= n or x[1] < 0 or x[1] >= n:
                    return max_dist
                if self.data[x[0], x[1]] > 0:  
                    break
                x += dx
            return np.linalg.norm(x - point)
        
        horizontal = [raycast(point, np.array([0, -1])), raycast(point, np.array([0, 1]))]
        vertical = [raycast(point, np.array([-1, 0])), raycast(point, np.array([1, 0]))]
        for d1 in horizontal:
            for d2 in vertical:
                if d1 < threshold and d2 <threshold * 2:
                    return True
        return False

    def update(self, lidar_scan, points, speeds, cur_pos, region: Region, regions: Dict[int, Region], update_room=True):
        """
        Update the gridmap with a lidar scan.
        :param lidar_scan: A numpy array of shape (n, 3), where each row represents (x, y, intensity).
        """
        def bresenham_line(x0, y0, x1, y1):
            """Generate grid cells along the line from (x0, y0) to (x1, y1) using Bresenham's algorithm."""
            cells = []
            dx = abs(x1 - x0)
            dy = abs(y1 - y0)
            sx = 1 if x0 < x1 else -1
            sy = 1 if y0 < y1 else -1
            err = dx - dy

            while True:
                cells.append((x0, y0))
                if x0 == x1 and y0 == y1:
                    break
                e2 = 2 * err
                if e2 > -dy:
                    err -= dy
                    x0 += sx
                if e2 < dx:
                    err += dx
                    y0 += sy
            return cells

        def is_blocked(start, end):
            """Check if there is an occupied grid between start and end points in the occupancy map."""
            for x, y in bresenham_line(start[0], start[1], end[0], end[1]):
                if self.data[x, y] == 1:  # Assuming 1 represents occupied space
                    return True
            return False
        
        n_angles = 90
        angle_bins = np.ones(n_angles) * 999
        angle_to_bin = lambda a: int((a + np.pi) / (2 * np.pi) * (n_angles - 1))
        occ = np.zeros(self.data.shape, dtype=np.int64)
        for point in lidar_scan:
            x, y = point
            angle = np.arctan2(y - cur_pos[1], x - cur_pos[0])
            dist = np.sqrt((x - cur_pos[0]) ** 2 + (y - cur_pos[1]) ** 2)
            bin = angle_to_bin(angle)
            angle_bins[bin] = min(angle_bins[bin], dist)

            i, j = self._world_to_grid(x, y)
            if 0 <= i < self.n_cells and 0 <= j < self.n_cells:
                occ[i, j] += 1
        self.data[occ > 5] = 1
        for wall_idx in np.stack(np.where(self.data == 1), -1):
            x, y = self._grid_to_world(wall_idx[0], wall_idx[1])
            angle = np.arctan2(y - cur_pos[1], x - cur_pos[0])
            dist = np.sqrt((x - cur_pos[0]) ** 2 + (y - cur_pos[1]) ** 2)
            bin = angle_to_bin(angle)
            angle_bins[bin] = min(angle_bins[bin], dist)

        cur_pos_idx = self._world_to_grid(cur_pos[0], cur_pos[1])
        min_speed = 0.06
        cur_visited = np.zeros_like(self.visited)
        prev_frontier = self.frontier.copy()
        near_idx = np.linalg.norm(points - cur_pos, axis=1) <= self.near_dist
        near_points = points[near_idx]
        near_speeds = speeds[near_idx]
        visited_count = 0
        out_of_bounds = 0
        not_traversable = 0

        for point, speed in zip(near_points, near_speeds):
            x, y = point
            i, j = self._world_to_grid(x, y)
            if not (0 <= i < self.n_cells and 0 <= j < self.n_cells):
                out_of_bounds += 1
                continue

            if self.trav_map[i, j] != 1:
                not_traversable += 1
                continue

            self.visited[i, j] = 1
            cur_visited[i, j] = 1
            self.frontier[i, j] = 0
            visited_count += 1

        near_mask = np.linalg.norm(points - cur_pos, axis=1) > self.near_dist
        far_mask = np.linalg.norm(points - cur_pos, axis=1) <= self.near_dist + 0.05
        far_idx = near_mask & far_mask
        far_points = points[far_idx]
        far_speeds = speeds[far_idx]
        for point, speed in zip(far_points, far_speeds):
            x, y = point
            angle = np.arctan2(y - cur_pos[1], x - cur_pos[0])
            dist = np.sqrt((x - cur_pos[0]) ** 2 + (y - cur_pos[1]) ** 2)
            bin = angle_to_bin(angle)
            if angle_bins[bin] <= dist:
                continue
            if speed <= min_speed:
                continue
            i, j = self._world_to_grid(x, y)

            if 0 <= i < self.n_cells and 0 <= j < self.n_cells:
                if self.trav_map[i, j] == 1 and self.visited[i, j] == 0: #and not is_blocked(cur_pos_idx, (i, j)):
                    self.frontier[i, j] = 1
        
        old_dm = self.discovery_map.copy()
        self.visited[self.trav_map == 0] = 0

        unexplored_map = self.trav_map.copy()
        unexplored_map[self.visited == 1] = 0
        unexplored_map[self.data == 1] = 0
        
        self.frontier[skimage.segmentation.find_boundaries(self.visited, mode='inner') == 1] = 1
        self.frontier[self.data == 1] = 0
        ## DIFF From ORIGINAL
        #self.frontier[cv2.erode(1 - self.data, np.ones((3, 3)), iterations=1) == 0] = 0 
        # clean up frontier
        # ✅ STEP 1: Erode obstacles to create safety buffer
        erosion_pixels = 3  # 3 cells = 0.3m safety margin
        erosion_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (2 * erosion_pixels + 1, 2 * erosion_pixels + 1)
        )
        # Dilate occupied cells (pushes frontiers away from walls)
        occupied_dilated = cv2.dilate(
            self.data.astype(np.uint8), 
            erosion_kernel, 
            iterations=1
        )
        # Remove frontiers too close to obstacles 
        self.frontier[occupied_dilated == 1] = 0
        # ✅ STEP 2: Additional cleanup (your existing code)
        self.frontier[cv2.erode(1 - self.data, np.ones((3, 3)), iterations=1) == 0] = 0
        self.frontier = get_neighbor_at_least(unexplored_map, self.frontier, 1)
        if not update_room:
            return

        kernel_size = int(self.room_seg_config[0])   # = 7
        iterations  = int(self.room_seg_config[1])   # = 2      # = 7
        eroded = cv2.erode((1 - self.data).astype(np.uint8),np.ones((kernel_size, kernel_size), dtype=np.uint8),iterations=self.room_seg_config[1])
        #eroded = cv2.erode(1 - self.data, np.ones(self.room_seg_config[0]), iterations=self.room_seg_config[1])
        eroded[self.confirmed_room_map != 0] = 0
        
        threshold = cv2.threshold(eroded, 0.5, 1, cv2.THRESH_BINARY)[1]
        n_cc, labels = cv2.connectedComponents(threshold)
        for _ in range(100):
            labels = skimage.segmentation.expand_labels(labels, distance=1)
            labels[self.data == 1] = 0
            labels[self.confirmed_room_map != 0] = 0
        self.outer_label = labels[0, 0]
        labels[self.visited == 0] = 0
        unconfirmed_space = np.zeros(labels.shape, dtype=np.uint8)
        unconfirmed_space[labels == self.outer_label] = 1
        unconfirmed_space[(labels == 0) & (self.visited == 1) & (self.confirmed_room_map == 0) & (self.data == 0)] = 1
        # breakpoint()
        threshold2 = cv2.threshold(unconfirmed_space, 0.5, 1, cv2.THRESH_BINARY)[1]
        _, labels2 = cv2.connectedComponents(threshold2)
        for label2 in np.unique(labels2):
            if (labels2 == label2).sum() < 10:
                labels2[labels2 == label2] = -1
        # breakpoint()
        add_mask = (unconfirmed_space != 0) & (labels2 != -1)
        labels[add_mask] = labels2[add_mask] + self.unconfirmed_label

        self.room_seg = labels
        unique_labels = np.unique(labels)

        self.new_confirmed_rooms = []
        self.clear_unconfirmed_rooms()
        all_unconfirmed_rooms = {}
        unconfirmed_room_connectivity = nx.Graph()
        for label in unique_labels:
            if label == self.outer_label:
                continue
            if label == 0:
                continue
            # clean up room
            ccs = connected_components(labels == label)
            cc_masks = [ccs == cc for cc in np.unique(ccs) if np.unique(labels[ccs == cc])[0] == label]
            cc = max(cc_masks, key=lambda m: m.sum())
            room = Room(cc)
            
            if self.room_is_confirmed(room):
                new_room = self.add_confirmed_room(room)
                if new_room is not None:
                    print('found newly confirmed room')
                    # ✅ DEBUG: check the robot's grid cell + confirmed room map value
                    i, j = cur_pos_idx  # already computed earlier in update()
                    print("Robot grid:", i, j)
                    print("Room map value:", self.confirmed_room_map[i, j])
                    # ✅ DEBUG: sanity check that the mask is stamped with this room id
                    vals = np.unique(self.confirmed_room_map[new_room.mask])
                    print("Unique confirmed_room_map values inside new_room.mask:", vals)
                    # ── Assign room to owning region ──────────────────────────────
                    mask_ij = np.stack(np.where(new_room.mask), axis=1)  # (N, 2)
                    centroid_i = int(mask_ij[:, 0].mean())
                    centroid_j = int(mask_ij[:, 1].mean())
                    cx, cy = self._grid_to_world(centroid_i, centroid_j)
                    for region in self.regions.values():
                        b = region.bound.numpy()
                        if b[0,0] <= cx <= b[1,0] and b[0,1] <= cy <= b[1,1]:
                            region.add_room(new_room)   # sets new_room.region = region
                            break
                    # ─────────────────────────────────────────────────────────────
                    self.new_confirmed_rooms.append(new_room)
                    self.add_doors(new_room, region, regions)
            else:
                # all_unconfirmed_rooms[label] = cc
                unconfirmed_room = UnconfirmedRoom(cc) #parent_id = None was added.
                self.unconfirmed_rooms[unconfirmed_room.id] = unconfirmed_room
                #self.add_doors(unconfirmed_room, region, regions)
        
        self.update_unconfirmed_rooms()
        self.update_rooms()
    
    def get_boundary_mask(self, room: Room, mode='thick'):
        mask = room.mask.copy()
        mask = skimage.segmentation.find_boundaries(mask.astype(np.uint8), mode=mode)
        mask[self.data == 1] = 0
        return mask
    
    def get_map_img(self, plot=True):
        im = np.ones((*self.data.shape, 3), dtype=np.uint8) * 255
        im[self.data == 1] = 0
        if plot:
            plt.imshow(im, origin='lower')
        return im

## Added here for the room segmentation + graph
    def get_rooms(self):
        """
        Returns dictionary of all rooms (confirmed + unconfirmed).
        """
        rooms = {}

        # Confirmed rooms
        for room_id, room in self.confirmed_rooms.items():
            mask = room.mask
            cells = np.stack(np.where(mask), axis=-1)

            boundary = skimage.segmentation.find_boundaries(mask, mode='outer')
            boundary[self.data == 1] = 0
            boundary_cells = np.stack(np.where(boundary), axis=-1)

            rooms[room_id] = {
                "cells": cells,
                "boundary": boundary_cells,
                "confirmed": True
            }

        # Unconfirmed rooms
        for room_id, room in self.unconfirmed_rooms.items():
            mask = room.mask
            cells = np.stack(np.where(mask), axis=-1)

            boundary = skimage.segmentation.find_boundaries(mask, mode='outer')
            boundary[self.data == 1] = 0
            boundary_cells = np.stack(np.where(boundary), axis=-1)

            rooms[room_id] = {
                "cells": cells,
                "boundary": boundary_cells,
                "confirmed": False
            }

        return rooms

    def get_room_graph(self):
        graph_dict = {room_id: [] for room_id in self.confirmed_rooms.keys()}
        for u, v, edge_data in self.room_graph.edges(data=True):
            pos_u = self.room_graph.nodes[u].get('pos')
            pos_v = self.room_graph.nodes[v].get('pos')
            room_u = self.room_graph.nodes[u].get('room', -1)
            room_v = self.room_graph.nodes[v].get('room', -1)
            if pos_u is None or pos_v is None:
                continue
            door_xy = np.array([(pos_u[0]+pos_v[0])/2,
                                 (pos_u[1]+pos_v[1])/2], dtype=np.float32)
            if room_u in graph_dict:
                graph_dict[room_u].append({'to': room_v, 'door_xy': door_xy, 'weight': 1.0})
            if room_v in graph_dict:
                graph_dict[room_v].append({'to': room_u, 'door_xy': door_xy, 'weight': 1.0})
        return graph_dict


    def get_room_at_position(self, pos):
        """
        Returns room_id at given world (x,y), or None.
        """
        i, j = self._world_to_grid(pos[0], pos[1])
        room_id = self.confirmed_room_map[i, j]
        #print("Robot grid:", i, j)
        #print("Room map value:", self.confirmed_room_map[i, j])
        if room_id != 0:
            return room_id
        return None
## Above is the new diff for room seg + graph building
    
    def save(self, path):
        save_dir = os.path.join(path, 'lidar_gridmap')
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        np.save(os.path.join(save_dir, 'occ_map.npy'), self.data)
        np.save(os.path.join(save_dir, 'visited.npy'), self.visited)
        np.save(os.path.join(save_dir, 'frontier.npy'), self.frontier)
        with open(os.path.join(save_dir, 'confirmed_rooms.json'), 'w') as f:
            json.dump({rid: room.serialize() for rid, room in self.confirmed_rooms.items()}, f)
        with open(os.path.join(save_dir, 'unconfirmed_rooms.json'), 'w') as f:
            json.dump({int(rid): room.serialize() for rid, room in self.unconfirmed_rooms.items()}, f)
        np.save(os.path.join(save_dir, 'confirmed_room_map.npy'), self.confirmed_room_map)
        nx.write_adjlist(self.room_graph, os.path.join(save_dir, 'room_graph.adjlist'))
        edge_info = {str(edge): self.room_graph.edges[edge] for edge in self.room_graph.edges}
        with open(os.path.join(save_dir, 'room_graph_edges.json'), 'w') as f:
            json.dump(edge_info, f)
        nx.write_adjlist(self.room_connectivity_graph, os.path.join(save_dir, 'room_connectivity_graph.adjlist'))
        with open(os.path.join(save_dir, 'doors.json'), 'w') as f:
            json.dump([door.serialize() for door in self.doors], f)
        np.save(os.path.join(save_dir, 'discovery_map.npy'), self.discovery_map)


Latest update
    def update(self):
        now = self.get_clock().now()

        # ── Prerequisites ────────────────────────────────────────────────
        if self.robot_pos is None or self.latest_cloud is None:
            return

        result = self.adapter.update(self.latest_cloud, self.robot_pos)
        if result is None:
            return
        points_lidar, speeds = result
        if len(points_lidar) == 0:
            return

        # ── TF transform ─────────────────────────────────────────────────
        try:
            source_frame = self.latest_cloud.header.frame_id
            stamp        = self.latest_cloud.header.stamp
            if self.tf_buffer.can_transform('odom', source_frame, stamp,
                                            timeout=Duration(seconds=0.1)):
                transform = self.tf_buffer.lookup_transform('odom', source_frame, stamp)
            else:
                return
            t = transform.transform.translation
            q = transform.transform.rotation
            R = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
            world_points = (R @ points_lidar.T).T + np.array([t.x, t.y, t.z])
        except Exception as e:
            self.get_logger().warn(f"TF lookup failed: {e}")
            return

        # ── Bound filter ──────────────────────────────────────────────────
        robot_xy = self.robot_pos[:2]
        half     = self.gridmap.map_size / 2
        ox, oy   = self.gridmap.origin
        in_bounds = (
            (world_points[:, 0] >= ox - half) & (world_points[:, 0] <= ox + half) &
            (world_points[:, 1] >= oy - half) & (world_points[:, 1] <= oy + half)
        )
        world_points_filtered = world_points[in_bounds]
        speeds_filtered       = speeds[in_bounds]

        if len(world_points_filtered) < 5:
            return

        # ── Angular downsampling ──────────────────────────────────────────
        vecs   = world_points_filtered[:, :2] - robot_xy[None, :]
        dists  = np.linalg.norm(vecs, axis=1) + 1e-9
        angles = np.arctan2(vecs[:, 1], vecs[:, 0])
        bins   = ((angles + np.pi) / (2 * np.pi) * 359).astype(np.int32)
        closest_idx = {}
        for k, b in enumerate(bins):
            if (b not in closest_idx) or (dists[k] < dists[closest_idx[b]]):
                closest_idx[b] = k
        keep              = np.array(list(closest_idx.values()), dtype=np.int32)
        world_points_hit  = world_points_filtered[keep]
        speeds_hit        = speeds_filtered[keep]

        # ── Raycast: update gridmap.data + visited ────────────────────────
        cell   = self.gridmap.cell_size
        origin = self.gridmap.origin
        half   = self.gridmap.map_size / 2

        def world_to_ij(x, y):
            return (int((x - (origin[0] - half)) / cell),
                    int((y - (origin[1] - half)) / cell))

        for pt3d in world_points_hit:
            pt   = pt3d[:2]
            dx   = pt[0] - robot_xy[0]
            dy   = pt[1] - robot_xy[1]
            dist = float(np.hypot(dx, dy))
            if dist > MAX_RANGE:
                scale = MAX_RANGE / dist
                dx *= scale; dy *= scale; dist = MAX_RANGE
            steps = int(dist / cell)
            if steps <= 1:
                ix, iy = world_to_ij(pt[0], pt[1])
                if 0 <= ix < self.gridmap.n_cells and 0 <= iy < self.gridmap.n_cells:
                    self.gridmap.data[iy, ix]    = 1
                    self.gridmap.visited[iy, ix] = 1
                continue
            for s in range(steps - 1):
                x  = robot_xy[0] + dx * (s / steps)
                y  = robot_xy[1] + dy * (s / steps)
                ix, iy = world_to_ij(x, y)
                if not (0 <= ix < self.gridmap.n_cells and 0 <= iy < self.gridmap.n_cells):
                    break
                if self.gridmap.data[iy, ix] == 1:
                    break
                self.gridmap.visited[iy, ix] = 1
            ix_end, iy_end = world_to_ij(pt[0], pt[1])
            if 0 <= ix_end < self.gridmap.n_cells and 0 <= iy_end < self.gridmap.n_cells:
                self.gridmap.data[iy_end, ix_end]    = 1
                self.gridmap.visited[iy_end, ix_end] = 1

        # ── Gridmap update (rate-limit room segmentation to ~1 Hz) ───────
        self._map_update_counter += 1
        do_room_update = (self._map_update_counter % 5 == 0)  # every 5 × 0.2s = 1 Hz
        self.gridmap.update(
            lidar_scan=world_points_hit[:, :2],
            points=world_points_hit[:, :2],
            speeds=speeds_hit,
            cur_pos=robot_xy,
            region=self.region,
            regions=self.all_regions,
            update_room=do_room_update
        )

        # ── Snapshot live frontiers into ExplorationGraph ────────────────
        if self.region is not None:
            ver = self.region.version
            center = np.array(self.region.center[:2], dtype=np.float32)
            self.exploration_graph.ensure_region(ver, center)

            # Collect live frontier points for this region
            iy, ix = np.where(self.gridmap.frontier > 0)
            if len(ix) > 0:
                cell   = self.gridmap.cell_size
                origin = self.gridmap.origin
                half   = self.gridmap.map_size / 2.0
                fx = (origin[0] - half) + (ix + 0.5) * cell
                fy = (origin[1] - half) + (iy + 0.5) * cell
                live_frontiers = np.stack([fx, fy], axis=1)
                # Downsample to avoid flooding the bank
                if len(live_frontiers) > 50:
                    live_frontiers = live_frontiers[
                        np.random.choice(len(live_frontiers), 50, replace=False)
                    ]
                self.exploration_graph.snapshot_frontiers(ver, live_frontiers)

            # ── Evict stale frontier points from ALL region banks ─────────
            # Build a KD-tree of live frontier cells so the eviction check
            # is O(log N) per saved point rather than O(N²).
            if len(ix) > 0:
                cell_   = self.gridmap.cell_size
                origin_ = self.gridmap.origin
                half_   = self.gridmap.map_size / 2.0
                all_fx  = (origin_[0] - half_) + (ix + 0.5) * cell_
                all_fy  = (origin_[1] - half_) + (iy + 0.5) * cell_
                live_pts = np.stack([all_fx, all_fy], axis=1)
                import scipy.spatial
                _kd = scipy.spatial.cKDTree(live_pts)
                def _is_still_frontier(pt, radius):
                    return len(_kd.query_ball_point(pt, radius)) > 0
            else:
                # No live frontier cells at all — evict everything
                def _is_still_frontier(pt, radius):
                    return False

            self.exploration_graph.reconcile_all_banks(
                is_frontier_fn=_is_still_frontier,
                evict_radius=0.6,
            )

            # Register door edges from gridmap into ExplorationGraph
        #    for door in self.gridmap.doors:
        #        # door.room is the room id; map room → region via confirmedroommap
        #        door_xy = np.array(door.pos[:2], dtype=np.float32)
        #        # find which region versions are near this door
        #        for v1, rec1 in self.exploration_graph.records.items():
        #            for v2, rec2 in self.exploration_graph.records.items():
        #                if v1 >= v2:
        #                    continue
        #                reg1 = self.all_regions.get(v1)
        #                reg2 = self.all_regions.get(v2)
        #                if reg1 is None or reg2 is None:
        #                    continue
        #                b1 = reg1.bound.numpy()
        #                b2 = reg2.bound.numpy()
        #                in1 = (b1[0,0] <= door_xy[0] <= b1[1,0] and
        #                       b1[0,1] <= door_xy[1] <= b1[1,1])
        #                in2 = (b2[0,0] <= door_xy[0] <= b2[1,0] and
        #                       b2[0,1] <= door_xy[1] <= b2[1,1])
        #                if in1 and in2:
        #                    self.exploration_graph.add_door_edge(v1, v2, door_xy)
        if do_room_update:
            for u, v, edge_data in self.gridmap.room_graph.edges(data=True):
                room_u = self.gridmap.room_graph.nodes[u].get('room', -1)
                room_v = self.gridmap.room_graph.nodes[v].get('room', -1)

                # Only register edges between two confirmed rooms
                if room_u not in self.gridmap.confirmed_rooms:
                    continue
                if room_v not in self.gridmap.confirmed_rooms:
                    continue

                pos_u = self.gridmap.room_graph.nodes[u].get('pos')
                pos_v = self.gridmap.room_graph.nodes[v].get('pos')
                if pos_u is None or pos_v is None:
                    continue

                door_xy = np.array([(pos_u[0]+pos_v[0])/2,
                                     (pos_u[1]+pos_v[1])/2], dtype=np.float32)

                reg_u = getattr(self.gridmap.confirmed_rooms[room_u], 'region', None)
                reg_v = getattr(self.gridmap.confirmed_rooms[room_v], 'region', None)
                if reg_u is None or reg_v is None:
                    continue
                if reg_u.version == reg_v.version:
                    continue   # same region, spatial edge already covers this

                self.exploration_graph.add_door_edge(reg_u.version, reg_v.version, door_xy)

            # ── Spatial adjacency edges (intra-room, no door needed) ─────
            for v1, reg1 in self.all_regions.items():
                for v2, reg2 in self.all_regions.items():
                    if v1 >= v2:
                        continue
                    if self.exploration_graph.graph.has_edge(v1, v2):
                        continue  # already connected (door or spatial)

                    b1 = reg1.bound.numpy()
                    b2 = reg2.bound.numpy()

                    # Gap between the two bounding boxes in x and y
                    x_gap = max(0.0, max(b1[0,0], b2[0,0]) - min(b1[1,0], b2[1,0]))
                    y_gap = max(0.0, max(b1[0,1], b2[0,1]) - min(b1[1,1], b2[1,1]))

                    if x_gap < 1.0 and y_gap < 1.0:
                        # Use find_traversable_crossing to get a wall-free midpoint.
                        # This replaces the raw geometric midpoint that ignores walls.
                        crossing = self.find_traversable_crossing(
                            b1, b2, self.robot_pos[:2]
                        )
                        if crossing is None:
                            # Overlap exists geometrically but no free crossing
                            # found yet — skip until more of the area is visited.
                            continue
                        midpoint = crossing
                        self.exploration_graph.add_spatial_edge(v1, v2, midpoint)

        # ── Enqueue training frame (ALL modes) ───────────────────────────
        if self.region is not None and bool(self.get_parameter('train_after_region').value):
            self.train_counter += 1
            if self.train_counter % self.train_every_n_frames == 0:
                with self.train_lock:
                    self.train_queue.append((
                        world_points_hit.copy(),
                        self.robot_pos.copy()
                    ))

        # ── Process newly confirmed rooms (OG-style: absorb into region) ───
        if do_room_update and self.gridmap.new_confirmed_rooms:
            self.process_newly_confirmed_rooms(
                list(self.gridmap.new_confirmed_rooms)
            )

        # ── Region boundary check ─────────────────────────────────────────
        self.maybe_create_new_region()
        # ── Trajectory ───────────────────────────────────────────────────
        if self.robot_pos is not None:
            self._traj_history.append(self.robot_pos[:2].copy())

        # ── Mode state machine (navigation only, map already updated above) ──

        if self.mode == "BOOTSTRAP":
            self.navigator.stop_robot()
            self.bootstrap_frames += 1
            if self.bootstrap_frames >= self.bootstrap_threshold:
                if self.all_regions:
                    # process_newly_confirmed_rooms already created a region
                    # before bootstrap finished — adopt it instead of spawning
                    if self.region is None:
                        self.region = next(iter(self.all_regions.values()))
                        self.get_logger().info(
                            f"[FLOW] BOOTSTRAP: adopting existing region "
                            f"{self.region.version} created by room processing"
                        )
                    self.mode               = "WARMUP"
                    self.warmup_start_epoch = int(self.region.epoch)
                    self.goal_xy            = None
                    self.last_goal_time     = now
                    self.get_logger().info("[FLOW] Entering WARMUP phase")
                elif self.region is None:
                    self.get_logger().info("Initializing first region")
                    self.init_region(self.robot_pos)
                    self.mode               = "WARMUP"
                    self.warmup_start_epoch = int(self.region.epoch)
                    self.goal_xy            = None
                    self.last_goal_time     = now
                    self.get_logger().info("[FLOW] Entering WARMUP phase")

        elif self.mode == "WARMUP":
            # Stop robot, check epochs — but do NOT return; fall through to publish
            self.navigator.stop_robot()
            if self.region is not None:
                cur_epochs = int(self.region.epoch) - int(self.warmup_start_epoch)
                print(f"[WARMUP] epoch {cur_epochs}/{self.warmup_epochs_needed}")
                if cur_epochs >= self.warmup_epochs_needed:
                    self.mode        = "EXPLORE"
                    self.warmup_done = True
                    self.goal_xy     = None
                    self.last_goal_time = now
                    self.get_logger().info("[FLOW] Warmup complete → starting EXPLORE")

        elif self.mode == "EXPLORE":
            robot_xy = np.asarray(self.robot_pos[:2], dtype=np.float32)
            now_sec  = now.nanoseconds * 1e-9

            # ── IDLE: pick a new frontier goal ───────────────────────────
            if self.nav_state == "IDLE":
                current_room = self.gridmap.get_room_at_position(robot_xy)
                rooms        = self.gridmap.get_rooms()
                room_graph   = self.gridmap.get_room_graph()
                g = self.pick_room_aware_goal(current_room, rooms, room_graph)

                if g is None:
                    self.navigator.stop_robot()
                    print("[NO FRONTIER]")
                else:
                    self.goal_xy           = np.asarray(g, dtype=np.float32)
                    self.wait_start_time   = now_sec
                    self._path_gen_running = False
                    print("[GOAL SET]", self.goal_xy)

                    # ── If ExplorationGraph gave us door waypoints, build a
                    #    field-stitched cross-region path instead of a raw
                    #    waypoint chain (which ignores walls in overlap zones) ─
                    if self._remote_door_waypoints:
                        print(f"[EXPLORATION GRAPH] Building cross-region path "
                              f"via {len(self._remote_door_waypoints)} crossing(s) "
                              f"→ region {self._remote_target_version}")
                        stitched = self.build_cross_region_path(
                            robot_xy=robot_xy,
                            goal_xy=self.goal_xy,
                            door_waypoints=self._remote_door_waypoints,
                            target_version=self._remote_target_version,
                        )
                        if stitched is not None and len(stitched) >= 2:
                            dec = self.decimate_path(stitched, spacing=0.3)
                            smo = self.smooth_path(dec, window=5)
                            self.latest_candidate_path = np.asarray(smo, dtype=np.float32)
                            print(f"[EXPLORATION GRAPH] Stitched path: {len(stitched)} pts "
                                  f"→ {len(self.latest_candidate_path)} after decimate/smooth")
                        else:
                            # Stitching failed entirely — fall back to raw chain
                            self.get_logger().warn(
                                "[CROSS-REGION PATH] Stitching returned None, "
                                "falling back to raw waypoint chain"
                            )
                            full_wp = np.array(
                                self._remote_door_waypoints + [self.goal_xy],
                                dtype=np.float32
                            )
                            self.latest_candidate_path = full_wp
                        self._remote_door_waypoints = []   # consumed
                    else:
                        self.latest_candidate_path = None  # normal: path gen will fill this

                    self.nav_state = "STABILIZING"
            # ── STABILIZING: wait for field to be ready, then generate path ─
            elif self.nav_state == "STABILIZING":
                self.navigator.stop_robot()

                # Only attempt path generation once the field has trained
                # enough to be trustworthy.  Querying an untrained field
                # produces paths that cut through walls (Cause 1).
                field_ready = (
                    self.region is not None and
                    self.region.function is not None and
                    int(self.region.epoch) >= self.warmup_epochs_needed and
                    getattr(self.region, 'last_loss', 1.0) < 0.05
                )
                if field_ready:
                    self._launch_path_gen(robot_xy, self.goal_xy)

                if now_sec - self.wait_start_time >= self.wait_duration:
                    if self.latest_candidate_path is not None:
                        print("[PATH FROZEN]")
                        self.path = self.latest_candidate_path.copy()
                        self.navigator.reset_path()
                        self.nav_state = "FOLLOWING"
                    elif not field_ready:
                        # Field still training — extend wait silently
                        self.wait_start_time = now_sec
                    else:
                        print("[NO PATH FOUND — retrying]")
                        self.wait_start_time = now_sec  # extend wait

            # ── FOLLOWING: drive, continuously refresh path in background ─
            elif self.nav_state == "FOLLOWING":
                if self.goal_xy is None:
                    self.nav_state = "IDLE"
                    self.path = None
                    return
                if self.goal_xy is not None:
                    d_goal = float(np.linalg.norm(robot_xy - self.goal_xy))
                else:
                    d_goal = None
                #print(f"Distance to frontier: {d_goal:.2f}")
                if d_goal is None:
                    self.nav_state = "IDLE"
                    self.goal_xy = None
                    self.path = None 

                if d_goal is not None and d_goal < 0.25:
                    print("[FRONTIER REACHED]")
                    self.navigator.stop_robot()
                    self.navigator.reset_path()

                    if self.goal_xy is not None:
                        # ── Clear frontier cells around reached goal ──────────────
                        gix = int((self.goal_xy[0] - (self.gridmap.origin[0] - self.gridmap.map_size/2))
                                  / self.gridmap.cell_size)
                        giy = int((self.goal_xy[1] - (self.gridmap.origin[1] - self.gridmap.map_size/2))
                                  / self.gridmap.cell_size)
                        pad = 3
                        for di in range(-pad, pad+1):
                            for dj in range(-pad, pad+1):
                                ii, jj = giy+di, gix+dj
                                if 0 <= ii < self.gridmap.n_cells and 0 <= jj < self.gridmap.n_cells:
                                    self.gridmap.frontier[ii, jj] = 0
                                    self.gridmap.visited[ii, jj]  = 1

                        # ── Update visited blacklist ──────────────────────────────
                        self._visited_goals.append(self.goal_xy.copy())
                        if len(self._visited_goals) > 50:
                            self._visited_goals.pop(0)

                        # ── Notify ExplorationGraph ───────────────────────────────
                        if self.region is not None:
                            self.exploration_graph.mark_goal_reached(
                                self.region.version, self.goal_xy
                            )

                    self.goal_xy   = None
                    self.path      = None
                    self._remote_door_waypoints = []       # clear any leftover door chain
                    self._remote_target_version = None
                    self.nav_state = "IDLE"

                else:
                    if self.goal_xy is None:
                        self.nav_state = "IDLE"
                        return
                    # Refresh path only when not training (avoid corrupt field
                    # queries) and only every few ticks to avoid thrashing.
                    if not self.train_running:
                        self._launch_path_gen(robot_xy, self.goal_xy)
                    if self.latest_candidate_path is not None:
                        new_path = self.latest_candidate_path
                        self.latest_candidate_path = None
                        # Only replace if the new path is meaningfully different
                        # from the current one — prevents jitter from constant
                        # resets caused by small path variations (Cause 5).
                        if self.path is None or self.is_path_significantly_different(
                                self.path, new_path, threshold=0.35):
                            self.path = new_path
                            # Don't reset_path() — find where we currently are on
                            # the new path so navigation continues forward.
                            self.navigator.reset_path()
                    if self.path is not None:
                        self.navigator.drive_path(self.path)

        # ── Composite plot ────────────────────────────────────────────────
        if self._composite_step % 20 == 0:
            try:
                self.save_composite_plot()
                self.save_occupancy_snapshot()
            except Exception as e:
                self.get_logger().warn(f"[COMPOSITE PLOT] {e}")
        self._composite_step += 1

        # ── Always publish (every tick, every mode) ───────────────────────
        self.map_publisher.publish_maps()
        self.markers.publish_goal_marker()
        self.markers.publish_frontier_markers()
        self.publish_field_path(self.path)


import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
import rclpy.time
import torch
from scipy.spatial.transform import Rotation
import signal
import threading
from typing import Dict, List, Optional, Tuple

import numpy as np
import math
import tf2_ros
from tf2_ros import TransformBroadcaster
import tf2_geometry_msgs

import sensor_msgs_py.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from geometry_msgs.msg import TransformStamped, Twist, Point, PointStamped, PoseStamped, PoseWithCovarianceStamped
from visualization_msgs.msg import Marker

from mntfields.mntfields_ros.sensing.lidar_adapter import Lidar2DAdapter
from mntfields.mntfields_ros.mapping.map_publisher import MapPublisher 
from mntfields.mntfields_ros.mapping.markers import MarkerPublisher
from mntfields.mntfields_ros.mapping.navigation import Navigator
from mntfields.mntfields_ros.mapping.exploration_graph import ExplorationGraph

from mntfields.mntfields_core.mapping.lidar_gridmap import LidarGridmap
from mntfields.mntfields_core.core.region import Region
from mntfields.mntfields_core.core.model_base import Model

from collections import deque
import threading

import os
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg", force=True)
from matplotlib import cm, colors
import matplotlib.pyplot as plt


REGION_RADIUS = 2.5
REGION_EXIT_MARGIN = 0.4 #was 0.4
REGION_MIN_OVERLAP = 0.4 #was 0.4
MAX_RANGE = 3.5
WARMUP_MIN_EPOCHS = 5       # already exists as self.warmup_epochs_needed
REGION_WARMUP_EPOCHS = 8    # epochs needed before a new region is trusted

# Room→region absorption (mirrors run_mapping_og.py, scaled to real metres)
ROOM_FIT_MARGIN      = 0.15   # padding added around room bbox when fitting to region (m)
ROOM_OVERLAP_THRESH  = 0.6    # min fraction of room bbox that must overlap region bbox
MAX_ROOMS_PER_REGION = 8      # cap to stop one region swallowing the whole map
MAX_REGION_AREA_MULT = 2.5    # refuse absorption if merged area > N × base-area

#print("NEW CODE")
class DummyRenderer:
    def __init__(self):
        pass

    def render(self, *args, **kwargs):
        return None

class DummyModel:
    def __init__(self):
        self.folder = "/home/mayank/turtlebot3_ws/mntfields_model"
        self.renderer = "cuda:0"   # IMPORTANT: must be string device
        self.h = 0.0

class FakePCD:
    def __init__(self, pts):
        self.points = pts
        self.normals = np.zeros_like(pts)

class GridmapNode(Node):
    def __init__(self):
        super().__init__('mntfields_gridmap')
        self.folder = os.path.join("/home/mayank/turtlebot3_ws/mntfields_model/mntfields_output")
        os.makedirs(self.folder, exist_ok=True)

        #self.train_queue = deque(maxlen=3)   # keep last few frames only
        #self.train_lock = threading.Lock()
        #self.train_running = False
        #self._path_gen_running   = False
        #self.latest_candidate_path = None
        #self.wait_start_time = None
        #self.wait_duration = 3.0
        # Start a separate timer JUST for training
        #train_dt = float(self.get_parameter('train_timer_sec').value)
        #self.create_timer(train_dt, self.train_tick)

        self.declare_parameter('region_min_epochs', 10)  # train at least 20 times before 
        self._traj_history = []
        self._composite_step = 0

        self._frame_buffer: List[tuple] = []        # (world_pts, robot_pos) collected while moving
        self._last_collect_pos = np.zeros(2, dtype=np.float32)
        self._path_recompute_counter = 0
        self._current_path_computed = False         # True once path is computed for current goal
        self.warmup_epochs_per_goal  = 3            # epochs to train after each goal reached
        self.warmup_epochs_new_region = 8           # epochs for brand-new region
        # Tune these
        self.declare_parameter('train_timer_sec', 0.5)      # training loop cadence
        self.declare_parameter('train_samples', 10000)       # n_samples to get_data_from_lidar
        self.declare_parameter('train_epoch', 4)            # epoch passed into region.train_core
        self.declare_parameter('train_after_region', True)  # toggle
        self.radius = REGION_RADIUS

        self.declare_parameter('n_cells', 400)
        self.declare_parameter('near_dist', 2.0)
        self.declare_parameter('map_size', 40.0)  # Add this

        n_cells = self.get_parameter('n_cells').value
        near_dist = self.get_parameter('near_dist').value
        map_size = self.get_parameter('map_size').value

        self.map_to_odom = None
        self.origin_initialized = False
        self.set_parameters([rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)])
        # --- exploration / control params ---
        self.declare_parameter('goal_reached_dist', 0.20)
        self.declare_parameter('yaw_tol', 0.25)          # rad
        self.declare_parameter('kp_yaw', 0.8)
        self.declare_parameter('kp_lin', 0.25)
        self.declare_parameter('max_lin', 0.15)          # m/s
        self.declare_parameter('max_ang', 0.8)           # rad/s
        self.declare_parameter('frontier_min_cells', 2)
        self.declare_parameter('goal_cooldown_sec', 2.0)

        self.path = None        #these 4 lines are for the path holding for stable paths - latest diff
        self.path_last_update = self.get_clock().now()
        self.path_update_period = 2.0   # seconds
        self.path_index = 0

        self.goal_xy = None
        self.last_goal_time = self.get_clock().now()
        self.mode = "BOOTSTRAP"   # BOOTSTRAP -> EXPLORE
        #--diff
        self.warmup_frames_needed = 10
        self.warmup_epochs_needed = 12
        self.warmup_done = False
        self.warmup_start_epoch = 0
        self.nav_state = "IDLE"

        #--
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        self.navigator = Navigator(self)
        self.map_publisher = MapPublisher(self)
        self.markers = MarkerPublisher(self)

        self.robot_pos = None
        self.robot_yaw = 0.0
        self.latest_cloud = None

        self.all_regions: dict = {}
        self.region_initialized = False
        self.bootstrap_frames = 0
        self.bootstrap_threshold = 10 # frames before region starts
        
        self.adapter = Lidar2DAdapter()

        # Traversability map (1 = free, 0 = blocked)
        self.trav_map = np.ones((n_cells, n_cells), dtype=np.uint8)
        self.gridmap = LidarGridmap(
            trav_map=self.trav_map,
            regions=self.all_regions,#{}
            n_cells=n_cells,
            near_dist=2.0,
            map_size=map_size,           # Add this
            origin=(0.0, 0.0), #(origin_x, origin_y)  # Add this
            room_seg_config=(7,2),
        )
        self.origin_initialized = False
        self.declare_parameter('occ_height', 0.3)
        self.occ_height = float(self.get_parameter('occ_height').value)
        self.occ_height_tol = 0.01

        self.create_subscription(PointCloud2,'/lidar3d_points',self.lidar_cb,10)
        #self.create_subscription(PoseWithCovarianceStamped,'/amcl_pose',self.pose_cb,5) - will be used for a built map
        self.create_subscription(Odometry, '/odom', self.odom_cb, 10)

        self.occ_pub = self.create_publisher(OccupancyGrid,'/mntfields/occ_map',5)
        self.frontier_pub = self.create_publisher(OccupancyGrid,'/mntfields/frontier',5)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.goal_marker_pub = self.create_publisher(Marker, '/mntfields/goal_marker', 10)
        self.frontier_marker_pub = self.create_publisher(Marker, '/mntfields/frontier_markers', 10)
        self.field_path_pub = self.create_publisher(Path, '/field_path', 10)

        self.create_timer(0.2, self.update)

        self.model = DummyModel()
        self.region = None

        self.exploration_graph = ExplorationGraph() #graph diff
        self._remote_door_waypoints: List[np.ndarray] = []  # door waypoints for inter-region travel
        self._remote_target_version: Optional[int] = None
        self._goal_blacklist_radius = 0.5 #was 0.3
        self._visited_goals: List[np.ndarray] = []

        self.train_counter = 0
        self.train_every_n_frames = 1

        self._map_update_counter = 0 #claude difff

    def lidar_cb(self, msg):
        self.latest_cloud = msg
        if not hasattr(self, "_last_lidar_log"):
            self._last_lidar_log = self.get_clock().now()
            #self.get_logger().info("Received first LiDAR scan")
            return
        now = self.get_clock().now()
        if (now - self._last_lidar_log).nanoseconds > 2e9:  # 2 seconds
            #self.get_logger().info("Receiving LiDAR scans...")
            self._last_lidar_log = now
    
    def odom_cb(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation

        self.robot_pos = np.array([p.x, p.y, p.z], dtype=np.float32)
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.robot_yaw = np.arctan2(siny_cosp, cosy_cosp)

        # Lock map once, using the *initial* odom pose
        if not self.origin_initialized:
            self.gridmap.origin = np.array([p.x, p.y], dtype=np.float32)
            self.origin_initialized = True

    def _train_synchronous(self, n_epochs: int):
        """
        Blocking synchronous train on all buffered frames.
        Robot must be stopped before calling this.
        """
        if self.region is None:
            return
        if not self._frame_buffer:
            return

        print(f"[TRAIN] synchronous {n_epochs} epochs on "
              f"{len(self._frame_buffer)} buffered frames")

        for world_pts, robot_pose in self._frame_buffer:
            if len(world_pts) < 50:
                continue
            try:
                n_samples = int(self.get_parameter('train_samples').value)
                data = self.region.get_data_from_lidar(
                    world_pts, robot_pose,
                    device='cuda:0', n_samples=n_samples
                )
                if data is None:
                    continue
                points, closest = data
                if points.shape[0] == 0:
                    continue

                x0, x1   = points[:, :3], points[:, 3:6]
                cp0, cp1  = closest[:, :3], closest[:, 3:6]
                raw = torch.cat([x0, x1, cp0, cp1], dim=1)

                if not torch.isfinite(raw).all():
                    continue

                frame_data = self.region.calculate_speed_and_normal(raw)
                if frame_data.shape[0] == 0:
                    continue

                self.region.last_robot_pose = np.array(
                    [robot_pose[0], robot_pose[1], 0.0], dtype=np.float32
                )
                self.region.train_core(epoch=n_epochs, frame_data=frame_data)
                print(f"[TRAIN] done, epoch={self.region.epoch}")

            except Exception as e:
                self.get_logger().error(f"[TRAIN SYNC] {e}")
                import traceback; traceback.print_exc()

        self._frame_buffer.clear()
        print(f"[TRAIN] buffer cleared, region epoch={self.region.epoch}")

    def _collect_frame_if_moved(self, world_pts: np.ndarray, robot_pos: np.ndarray,
                                  min_dist: float = 0.15):
        """
        Append a frame to buffer only if robot moved min_dist since last collection.
        """
        robot_xy = robot_pos[:2]
        if np.linalg.norm(robot_xy - self._last_collect_pos) >= min_dist:
            self._frame_buffer.append((world_pts.copy(), robot_pos.copy()))
            self._last_collect_pos = robot_xy.copy()


    def init_region(self, robot_pose):
        ''' Initializes a region around the robot and registers it. '''

        center_x = robot_pose[0]
        center_y = robot_pose[1]
        radius = self.radius

        bound = torch.tensor([
            [center_x - radius, center_y - radius, -0.1],
            [center_x + radius, center_y + radius,  0.1],
        ], dtype=torch.float32)

        new_region = Region(
            model=self.model,
            bound=bound,
            init_network=True,
            eval=False,)
        new_region.gridmap = self.gridmap
        self.all_regions[new_region.version] = new_region

        if self.region is not None and self.region.all_lidar is not None:
            last = self.region.all_lidar[-5:]
            new_region.all_lidar = last.copy()
            new_region.lidar_obs = None
            new_region.kd_tree = None
        else:
            new_region.all_lidar = []
        self.region = new_region
        self.region_initialized = True
        self.get_logger().info(
            f"[REGION {new_region.version} INIT] "
            f"center=({center_x:.2f}, {center_y:.2f}) "
            f"bound=[({bound[0,0]:.2f},{bound[0,1]:.2f}) → "
            f"({bound[1,0]:.2f},{bound[1,1]:.2f})]")
        # ── Re-enter WARMUP for every new region ─────────────────────────
        self.mode               = "WARMUP"
        self.warmup_start_epoch = int(self.region.epoch)
        self.goal_xy            = None
        self.path               = None
        self.nav_state          = "IDLE"
        self.latest_candidate_path = None
        self._path_gen_running  = False
        self.get_logger().info(f"[FLOW] Region {new_region.version} → entering WARMUP")

    def start_region_warmup(self):
        self.mode = "WARMUP"
        self.region_warmup_start_epoch = self.region.epoch if self.region else 0
        self.navigator.stop_robot()
        self.get_logger().info(f"[WARMUP] Starting region {self.region.version}")

    def feed_region(self, world_pts, robot_pose, region=None):
        """Train a specific region. If region is None, falls back to self.region
        (legacy behaviour) but the preferred call is to pass the region
        explicitly so the training thread is not affected by later switches."""
        if region is None:
            region = self.region
        if region is None:
            return
        # Guard: if the region was unloaded between enqueue and execution,
        # skip silently — the next tick will re-enqueue with a loaded region.
        if not getattr(region, 'loaded', True):
            self.get_logger().warn(
                f"[FEED] Region {region.version} was unloaded before training — skipping"
            )
            return
        if len(world_pts) < 50:
            self.get_logger().warn(f"Too few points: {len(world_pts)}")
            return
        try:
            pts = torch.tensor(world_pts, dtype=torch.float32)
            pos = torch.tensor([robot_pose[0], robot_pose[1], 0.0], dtype=torch.float32)
            ##print("WORLD PTS MIN/MAX:", world_pts.min(), world_pts.max())
            n_samples = int(self.get_parameter('train_samples').value)
            epoch = int(self.get_parameter('train_epoch').value)
            data = region.get_data_from_lidar(
                pts.cpu().numpy(),
                pos.cpu().numpy(),
                device='cuda:0',
                n_samples=n_samples
            )
            if data is None:
                self.get_logger().warn("get_data_from_lidar returned None")
                return
            points, closest = data
            if points.shape[0] == 0:
                self.get_logger().warn("get_data_from_lidar returned 0 points")
                return

            if data is None:
                self.get_logger().warn("get_data_from_lidar returned None")
                return
            x0 = points[:, :3]
            x1 = points[:, 3:6]
            cp0 = closest[:, :3]
            cp1 = closest[:, 3:6]

            raw = torch.cat([x0, x1, cp0, cp1], dim=1)   # Nx12
            #print("RAW SHAPE:", raw.shape)

            if raw.ndim != 2 or raw.shape[1] != 12:
                self.get_logger().error(f"[DBG] raw has wrong shape: {tuple(raw.shape)} (expected Nx12)")
                return
            if not torch.isfinite(raw).all():
                bad = (~torch.isfinite(raw)).sum().item()
                self.get_logger().error(f"[DBG] raw has non-finite values: count={bad}")
                return
            with torch.no_grad():
                mn = raw.min(dim=0).values.cpu().numpy()
                mx = raw.max(dim=0).values.cpu().numpy()
            frame_data = region.calculate_speed_and_normal(raw)
            #frame_data = raw

            if frame_data.ndim != 2:
                self.get_logger().error(f"[DBG] frame_data not 2D: {tuple(frame_data.shape)}")
                return
            if frame_data.shape[1] < 12:
                self.get_logger().error(f"[DBG] frame_data has too few cols: {frame_data.shape[1]} (expected >=12)")
                return
            if not torch.isfinite(frame_data).all():
                bad = (~torch.isfinite(frame_data)).sum().item()
                self.get_logger().error(f"[DBG] frame_data non-finite: count={bad}")
                return
            with torch.no_grad():
                fd_min = frame_data.min().item()
                fd_max = frame_data.max().item()

            y0_col = frame_data[:, 6]
            y1_col = frame_data[:, 7]

            if frame_data.shape[0] == 0:
                self.get_logger().warn("No valid frame data after speed calculation")
                return
            region.last_robot_pose = np.array([robot_pose[0], robot_pose[1], 0.0], dtype=np.float32)

            if region.function is not None:
                try:
                    k = min(64, frame_data.shape[0])
                    xp_n = frame_data[:k, :6].clone().to(region.device)
                    with torch.no_grad():
                        tt = region.function.TravelTimes(xp_n).detach().cpu().numpy().reshape(-1)
                    with torch.enable_grad():
                        sp_t = region.function.Speed(xp_n)  # tensor
                    sp = sp_t.detach().cpu().numpy().reshape(-1)

                except Exception as e:
                    self.get_logger().error(f"[DBG MODEL] failed: {e}")
            #print(f"[FEED] frame_data shape={frame_data.shape}, " #UNCOMMENT FOR DEBUG ON FRAME DATA - CORRECT FOR NOW
            #      f"y0 range=[{frame_data[:,6].min():.4f}, {frame_data[:,6].max():.4f}]")
            #print(f"[FEED] calling train_core with {frame_data.shape[0]} rows")      
            success = region.train_core(epoch=epoch, frame_data=frame_data)
            loss_val = getattr(self.region, 'last_loss', None)
            try:
                test = frame_data[:128, :6].clone().to(region.device)
                #test.requires_grad_(True)
                test_n = region.normalize_tensor(test)
                with torch.enable_grad():
                    tau, w, coords_in = region.function.network.out(test_n)
                    tt = tau[:, 0]
                    sp = region.function.Speed(test_n)

                g = torch.autograd.grad(tt.sum(), coords_in, retain_graph=True)[0]
                g0 = g[:, :3]  # gradient w.r.t x0
                g0_norm = torch.sqrt((g0 * g0).sum(dim=1) + 1e-12)
            except Exception as e:
                print("MODEL DEBUG FAILED:", e)

        except Exception as e:
            self.get_logger().error(f"feed_region error: {e}")
            import traceback
            traceback.print_exc()

    def pick_frontier_goal(self):
        frontier = self.gridmap.frontier
        iy, ix = np.where(frontier > 0)
        min_frontier_cells = int(self.get_parameter('frontier_min_cells').value)

        if len(ix) < min_frontier_cells:
            return None
        # ✅ DIRECTLY call pick_reachable_frontier (it handles everything)
        goal = self.pick_reachable_frontier()
        if goal is not None:
            robot = self.robot_pos[:2]
            dist = np.linalg.norm(goal - robot)
        return goal

    def pick_reachable_frontier(self, max_candidates=300, current_region_only=True):
        """
        Return the best live frontier point.

        When current_region_only=True (the default used in pick_room_aware_goal),
        only frontier cells that lie inside the current region's bounding box are
        considered.  This prevents a single stale cell outside the current region
        from blocking the ExplorationGraph from routing the robot to a new region.
        """
        frontier = self.gridmap.frontier
        iy, ix = np.where(frontier > 0)
        if len(ix) == 0:
            return None

        cell   = self.gridmap.cell_size
        origin = self.gridmap.origin
        half   = self.gridmap.map_size / 2.0

        fx = (origin[0] - half) + (ix + 0.5) * cell
        fy = (origin[1] - half) + (iy + 0.5) * cell
        cand = np.stack([fx, fy], axis=1)

        robot = self.robot_pos[:2]
        d = np.linalg.norm(cand - robot[None, :], axis=1)

        # distance gating
        valid = (d > 0.2) & (d < MAX_RANGE)

        # Restrict to current region's bounding box so we don't pick up
        # frontier cells in already-switched-away-from regions.
        if current_region_only and self.region is not None:
            b = self.region.bound.numpy()
            inside_region = (
                (cand[:, 0] >= b[0, 0]) & (cand[:, 0] <= b[1, 0]) &
                (cand[:, 1] >= b[0, 1]) & (cand[:, 1] <= b[1, 1])
            )
            valid = valid & inside_region

        if valid.sum() == 0:
            return None

        cand = cand[valid]
        d    = d[valid]

        # no field yet
        if self.region is None or self.region.function is None:
            return cand[np.argmin(d)]

        # field scoring
        x0 = np.tile(np.array([robot[0], robot[1], 0.0]), (len(cand), 1))
        x1 = np.concatenate([cand, np.zeros((len(cand), 1))], axis=1)
        xp = np.concatenate([x0, x1], axis=1)

        xp_t = torch.tensor(xp, dtype=torch.float32)
        xp_n = self.region.normalize_tensor(xp_t).to(self.region.device)

        with torch.no_grad():
            tt = self.region.function.TravelTimes(xp_n).cpu().numpy().reshape(-1)

        ok = np.isfinite(tt)
        if ok.sum() == 0:
            return cand[np.argmin(d)]

        return cand[ok][np.argmin(tt[ok])]

    def pick_room_aware_goal(self, current_room, rooms, room_graph):
        robot_xy = self.robot_pos[:2]

        # 1. Try live local frontier (with blacklist)
        local = self._frontier_in_room(current_room, rooms)
        if local is not None:
            if (len(self._visited_goals) == 0 or
                    np.linalg.norm(np.array(self._visited_goals) - local[None,:], axis=1).min()
                    > self._goal_blacklist_radius):
                return local

        # 2. Try live frontier restricted to current region
        live = self.pick_reachable_frontier(current_region_only=True)
        if live is not None:
            return live

        # 3. Current region has no more live frontiers → check if any frontier
        #    is within direct driving range before invoking the full cross-region
        #    routing machinery.  A frontier just across a region boundary should
        #    be driven to directly, not via a stitched multi-segment path.
        any_live = self.pick_reachable_frontier(current_region_only=False)
        if any_live is not None:
            d_direct = float(np.linalg.norm(any_live - robot_xy))
            if d_direct < MAX_RANGE:
                # Close enough to drive straight — return without setting door waypoints
                # so STABILIZING uses the normal single-region path generator.
                print(f"[SHORTCUT] Frontier {d_direct:.2f}m away, skipping cross-region routing")
                return any_live

        # 4. Frontier requires cross-region routing via ExplorationGraph.
        if self.region is not None:
            ver = self.region.version
            result = self.exploration_graph.best_remote_frontier(ver, robot_xy)
            if result is not None:
                target_ver, target_frontier, door_waypoints = result
                print(f"[EXPLORATION GRAPH] Routing to region {target_ver} "
                      f"via {len(door_waypoints)} door(s), "
                      f"frontier={target_frontier}")
                self._remote_door_waypoints = door_waypoints
                self._remote_target_version = target_ver
                return target_frontier

        print("[EXPLORATION GRAPH] Fully explored — no frontiers anywhere")
        return None

    def _frontier_in_room(self, room_id, rooms):
        if room_id is None:
            return None
        if room_id not in rooms:
            return None
        room_cells = rooms[room_id]["cells"]
        frontier_pts = []
        for (i, j) in room_cells:
            if self.gridmap.frontier[i, j] == 1:
                x, y = self.gridmap._grid_to_world(i, j)
                frontier_pts.append([x, y])
        if len(frontier_pts) == 0:
            return None
        frontier_pts = np.array(frontier_pts)
        # choose closest to robot
        robot = self.robot_pos[:2]
        dists = np.linalg.norm(frontier_pts - robot[None, :], axis=1)
        idx = np.argmin(dists)
        return frontier_pts[idx] #END DIFF

    def publish_field_path(self, path_xy, frame_id='odom'):
        """
        path_xy: list[np.ndarray] or np.ndarray of shape (N,2)
        """
        if path_xy is None:
            return
        if isinstance(path_xy, list):
            pts = np.asarray(path_xy, dtype=np.float32)
        else:
            pts = np.asarray(path_xy, dtype=np.float32)

        if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 2:
            return

        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id

        poses = []
        for p in pts:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = float(p[0])
            ps.pose.position.y = float(p[1])
            ps.pose.position.z = 0.0
            ps.pose.orientation.w = 1.0
            poses.append(ps)

        msg.poses = poses
        self.field_path_pub.publish(msg)

    def decimate_path(self, path_xy, spacing=0.4):
        pts = np.asarray(path_xy, dtype=np.float32)

        if pts.ndim == 1:
            pts = pts.reshape(1, 2)

        if len(pts) < 2:
            return pts

        new_pts = [pts[0]]

        for i in range(1, len(pts)):
            if np.linalg.norm(pts[i] - new_pts[-1]) >= spacing:
                new_pts.append(pts[i])

        if np.linalg.norm(new_pts[-1] - pts[-1]) > 0.05:
            new_pts.append(pts[-1])

        return np.asarray(new_pts, dtype=np.float32)

    def smooth_path(self, path_xy, window=5):
        if path_xy is None:
            return None

        pts = np.asarray(path_xy, dtype=np.float32)

        if pts.ndim == 1:
            pts = pts.reshape(1, 2)

        if len(pts) < window:
            return pts

        sm = pts.copy()
        k = window // 2

        for i in range(len(pts)):
            a = max(0, i-k)
            b = min(len(pts), i+k+1)
            sm[i] = pts[a:b].mean(axis=0)

        sm[0] = pts[0]
        sm[-1] = pts[-1]
        return sm 

    def pick_lookahead(self, path, robot_xy, L=0.45):
        pts = np.asarray(path)
        r = np.asarray(robot_xy)

        d = np.linalg.norm(pts - r[None,:], axis=1)
        i0 = np.argmin(d)

        acc = 0.0
        i = i0
        while i < len(pts)-1 and acc < L:
            acc += np.linalg.norm(pts[i+1]-pts[i])
            i += 1

        return pts[i]
       #end diff

    def is_path_significantly_different(self, old_path, new_path, threshold=0.25):

        old_path = np.asarray(old_path)
        new_path = np.asarray(new_path)

        if len(old_path) < 2 or len(new_path) < 2:
            return True

        # compare up to min length
        n = min(len(old_path), len(new_path))

        # compare corresponding waypoints
        diffs = np.linalg.norm(old_path[:n] - new_path[:n], axis=1)

        max_diff = np.max(diffs)

        print("Max waypoint diff:", max_diff)

        return max_diff > threshold

    def debug_plot(self, field_waypoint=None):
        data = self.gridmap.data
        visited = self.gridmap.visited

        im = np.ones((*data.shape, 3), dtype=np.uint8) * 255
        im[data > 0.9] = np.array([0, 0, 0])      # walls
        im[visited == 1] = np.array([0, 255, 0])  # visited

        fig, ax = plt.subplots(1,1,figsize=(6,6))
        ax.imshow(im, origin='lower')

        # robot
        if self.robot_pos is not None:
            i,j = self.world_to_grid(self.robot_pos[0], self.robot_pos[1])
            ax.scatter(j,i,c='magenta',s=60,label='robot')

        # goal
        if self.goal_xy is not None:
            i,j = self.world_to_grid(self.goal_xy[0], self.goal_xy[1])
            ax.scatter(j,i,c='red',s=60,label='goal')

        # field waypoint
        if field_waypoint is not None:
            i,j = self.world_to_grid(field_waypoint[0], field_waypoint[1])
            ax.scatter(j,i,c='blue',s=60,label='field wp')
        ax.legend()
        plt.pause(0.01)

    def train_tick(self):
        if self.region is None:
            print(f"[TRAIN_TICK] region=False, running={self.train_running}, "
                  f"queue_len={len(self.train_queue)}, epoch=-1")
            return

        print(f"[TRAIN_TICK] region=True, running={self.train_running}, "
              f"queue_len={len(self.train_queue)}, epoch={self.region.epoch}")

        if self.train_running:
            return  # previous train still going

        with self.train_lock:
            if len(self.train_queue) == 0:
                return
            self.train_running = True     
            world_pts, robot_pose = self.train_queue.pop()
            self.train_queue.clear()

        # ✅ Set BEFORE spawning thread — eliminates race condition
       
        # Capture the region object NOW, before the thread starts.
        # This ensures the training thread always trains the region that
        # was active when the frame was enqueued, even if self.region
        # switches to a different region before the thread executes.
        captured_region = self.region

        def _do_train():
            try:
                ver = captured_region.version if captured_region else '?'
                print(f"[TRAIN THREAD] starting feed_region, region={ver}")
                self.feed_region(world_pts, robot_pose, region=captured_region)
                print(f"[TRAIN THREAD] done, region={ver}")
            except Exception as e:
                self.get_logger().error(f"[TRAIN] failed: {e}")
                import traceback
                traceback.print_exc()
            finally:
                self.train_running = False

        import threading
        t = threading.Thread(target=_do_train, daemon=True)
        t.start()

    def maybe_expand_region(self):
        if self.region is None:
            return

        robot = self.robot_pos[:2]
        bound = self.region.bound.numpy()

        margin = 0.3

        near_edge = (
            abs(robot[0] - bound[0,0]) < margin or
            abs(robot[0] - bound[1,0]) < margin or
            abs(robot[1] - bound[0,1]) < margin or
            abs(robot[1] - bound[1,1]) < margin
        )

        if near_edge:
            new_bound = torch.tensor([
                [bound[0,0]-0.3, bound[0,1]-0.3, -0.1],
                [bound[1,0]+0.3, bound[1,1]+0.3,  0.1],
            ], dtype=torch.float32)

            self.region.set_bound(new_bound)
            self.get_logger().info("[REGION EXPAND]")

    def _get_frontier_world_pts(self, max_pts=200):
        """Return frontier cell centres in world coords as (N,2) array."""
        iy, ix = np.where(self.gridmap.frontier > 0)
        if len(ix) == 0:
            return None
        cell   = self.gridmap.cell_size
        origin = self.gridmap.origin
        half   = self.gridmap.map_size / 2.0
        fx = (origin[0] - half) + (ix + 0.5) * cell
        fy = (origin[1] - half) + (iy + 0.5) * cell
        pts = np.stack([fx, fy], axis=1)
        if len(pts) > max_pts:
            pts = pts[np.random.choice(len(pts), max_pts, replace=False)]
        return pts

    def maybe_create_new_region(self):
        """
        Called every update() tick.
        If the robot has moved outside the current region's bound, either:
          (a) switch to an existing region that covers the robot, OR
          (b) spawn a new region centred on the robot.
        Old region is unloaded to disk to free GPU memory.
        """
        if self.region is None or self.robot_pos is None:
            return

        robot = self.robot_pos[:2]
        b = self.region.bound.numpy()

        # Is robot still comfortably inside current bound?
        inside = (
            robot[0] > b[0, 0] + REGION_EXIT_MARGIN and
            robot[0] < b[1, 0] - REGION_EXIT_MARGIN and
            robot[1] > b[0, 1] + REGION_EXIT_MARGIN and
            robot[1] < b[1, 1] - REGION_EXIT_MARGIN
        )
        if inside:
            return  # nothing to do

        # --- Robot has left (or is about to leave) the current region ---

        # (a) Check if any existing region already covers the robot.
        #     Require the robot to be at least REGION_EXIT_MARGIN inside the
        #     *candidate* region before switching — this prevents ping-pong
        #     when the robot sits exactly on the shared boundary.
        SWITCH_MARGIN = REGION_EXIT_MARGIN  # must be this far inside new region
        for ver, reg in self.all_regions.items():
            if ver == self.region.version:
                continue
            b2 = reg.bound.numpy()
            deep_inside = (
                robot[0] > b2[0, 0] + SWITCH_MARGIN and
                robot[0] < b2[1, 0] - SWITCH_MARGIN and
                robot[1] > b2[0, 1] + SWITCH_MARGIN and
                robot[1] < b2[1, 1] - SWITCH_MARGIN
            )
            if deep_inside:
                if not reg.loaded:
                    reg.load_from_disk()
                self._switch_region(reg)
                return

        # (b) No existing region covers the robot → create a new one
        #     But only if the area has enough visited free space to train on
        visited = self.gridmap.visited
        cell = self.gridmap.cell_size
        origin = self.gridmap.origin
        half = self.gridmap.map_size / 2.0

        # Count visited cells in the proposed new bound
        def count_visited_in_bound(cx, cy, r):
            ix0 = max(0, int((cx - r - (origin[0] - half)) / cell))
            ix1 = min(self.gridmap.n_cells - 1, int((cx + r - (origin[0] - half)) / cell))
            iy0 = max(0, int((cy - r - (origin[1] - half)) / cell))
            iy1 = min(self.gridmap.n_cells - 1, int((cy + r - (origin[1] - half)) / cell))
            region_area = (ix1 - ix0) * (iy1 - iy0)
            if region_area <= 0:
                return 0.0
            visited_count = visited[iy0:iy1, ix0:ix1].sum()
            return float(visited_count) / float(region_area)

        coverage = count_visited_in_bound(robot[0], robot[1], REGION_RADIUS)
        if coverage < REGION_MIN_OVERLAP:
            # Not enough data yet — keep extending current region temporarily
            # by softly expanding its bound rather than spawning premature region
            self._soft_extend_current_region(robot)
            return

        # Enough data: spawn the new region
        self.get_logger().info(
            f"[NEW REGION] Robot at ({robot[0]:.2f},{robot[1]:.2f}) "
            f"left region {self.region.version}. "
            f"Visited coverage={coverage:.2f}. Spawning new region."
        )
        self._unload_current_region()
        self.init_region(self.robot_pos)

    # ------------------------------------------------------------------ #
    #  OG-style room → region absorption                                  #
    # ------------------------------------------------------------------ #

    def _room_rect_world(self, room) -> Tuple[float, float, float, float]:
        """Return (xmin, ymin, xmax, ymax) of room in world coords."""
        xmin, ymin = self.gridmap._grid_to_world(room.imin, room.jmin)
        xmax, ymax = self.gridmap._grid_to_world(room.imax, room.jmax)
        return xmin, ymin, xmax, ymax

    @staticmethod
    def _percent_overlap(r1, r2) -> float:
        """
        Fraction of r1's area that is covered by r2.
        r1, r2 = (xmin, ymin, xmax, ymax) in world coords.
        Returns value in [0, 1].
        """
        ix0 = max(r1[0], r2[0]); iy0 = max(r1[1], r2[1])
        ix1 = min(r1[2], r2[2]); iy1 = min(r1[3], r2[3])
        if ix1 <= ix0 or iy1 <= iy0:
            return 0.0
        inter = (ix1 - ix0) * (iy1 - iy0)
        area1 = max((r1[2] - r1[0]) * (r1[3] - r1[1]), 1e-9)
        return float(inter / area1)

    def find_region_for_room(self, room):
        """
        Mirrors run_mapping_og.find_region_for_room().

        For each confirmed room that has just been detected, search for an
        existing region whose bounding box already covers (or substantially
        overlaps) the room.  If found, expand that region's bound to tightly
        wrap the room and return (region, new_bound).  If no region fits,
        return (None, None) so the caller can spawn a new one.

        Scale note: the OG used a normalised [-0.5,0.5] world with
        fit_margin=0.03.  Our world is real-scale (metres), so we use
        ROOM_FIT_MARGIN = 0.15 m.
        """
        base_area = (2 * REGION_RADIUS) ** 2
        m = ROOM_FIT_MARGIN
        xmin, ymin, xmax, ymax = self._room_rect_world(room)
        room_rect = (xmin, ymin, xmax, ymax)

        candidates = []
        for region in self.all_regions.values():
            b = region.bound.numpy()
            reg_rect = (b[0,0], b[0,1], b[1,0], b[1,1])
            overlap  = self._percent_overlap(room_rect, reg_rect)

            if overlap == 1.0:
                # Room fully inside region — just ensure margin padding
                nx0 = min(xmin - m, b[0,0])
                ny0 = min(ymin - m, b[0,1])
                nx1 = max(xmax + m, b[1,0])
                ny1 = max(ymax + m, b[1,1])
                new_area = (nx1 - nx0) * (ny1 - ny0)
                if new_area / base_area <= MAX_REGION_AREA_MULT:
                    new_bound = torch.tensor([
                        [nx0, ny0, -0.1],
                        [nx1, ny1,  0.1],
                    ], dtype=torch.float32)
                    return region, new_bound
                # Too large — try next candidate

            candidates.append((region, overlap))

        # Sort by descending overlap, try to expand
        candidates.sort(key=lambda x: x[1], reverse=True)
        for region, overlap in candidates:
            if overlap < ROOM_OVERLAP_THRESH:
                break  # sorted, so no point continuing
            if len(region.rooms) >= MAX_ROOMS_PER_REGION:
                continue

            b = region.bound.numpy()
            nx0 = min(xmin - m, b[0,0])
            ny0 = min(ymin - m, b[0,1])
            nx1 = max(xmax + m, b[1,0])
            ny1 = max(ymax + m, b[1,1])
            new_area = (nx1 - nx0) * (ny1 - ny0)
            if new_area / base_area > MAX_REGION_AREA_MULT:
                continue

            new_bound = torch.tensor([
                [nx0, ny0, -0.1],
                [nx1, ny1,  0.1],
            ], dtype=torch.float32)
            return region, new_bound

        return None, None

    def process_newly_confirmed_rooms(self, rooms: list):
        """
        Mirrors run_mapping_og.process_newly_confirmed_rooms().

        For each newly confirmed room:
          1. Try to find an existing region that can absorb it
             (find_region_for_room) and verify door connectivity
             (region.is_connected).
          2. If found  → expand region bound + assign room → region.
          3. If not    → spawn a new tight region around the room.

        After all rooms are processed, rebuild door edges and register
        the regions in ExplorationGraph.
        """
        if not rooms:
            return

        self.get_logger().info(
            f"[ROOM PROCESS] {len(rooms)} newly confirmed room(s)"
        )

        regions_changed: set = set()
        unassigned = list(rooms)

        for i in range(len(unassigned) - 1, -1, -1):
            room = unassigned[i]
            region, new_bound = self.find_region_for_room(room)

            if region is None:
                continue   # will be handled in the spawn-new step

            # Door-connectivity check (mirrors OG's is_connected guard)
            connected = region.is_connected(
                room,
                self.gridmap.room_graph,
                list(self.gridmap.confirmed_rooms.values()),
                list(self.gridmap.unconfirmed_rooms.values()),
            )
            if not connected:
                self.get_logger().info(
                    f"[ROOM PROCESS] Room {room.id} not door-connected "
                    f"to region {region.version} — will spawn new region"
                )
                continue

            self.get_logger().info(
                f"[ROOM PROCESS] Absorbing room {room.id} "
                f"into region {region.version}"
            )
            region.set_bound(new_bound)
            region.add_room(room, self.gridmap.room_connectivity_graph)
            regions_changed.add(region)
            unassigned.pop(i)

            # Update ExplorationGraph record for this region
            center = np.array(region.center[:2], dtype=np.float32)
            self.exploration_graph.ensure_region(region.version, center)

        # ── Spawn new region(s) for any rooms that couldn't be absorbed ──
        if unassigned:
            self.get_logger().info(
                f"[ROOM PROCESS] Spawning new region for "
                f"{len(unassigned)} unassigned room(s)"
            )
            # Compute tight bound around all unassigned rooms
            m = ROOM_FIT_MARGIN
            all_x0 = [self._room_rect_world(r)[0] for r in unassigned]
            all_y0 = [self._room_rect_world(r)[1] for r in unassigned]
            all_x1 = [self._room_rect_world(r)[2] for r in unassigned]
            all_y1 = [self._room_rect_world(r)[3] for r in unassigned]
            # Guarantee at least REGION_RADIUS on each side so there is
            # enough space to train the neural field
            cx = (min(all_x0) + max(all_x1)) / 2
            cy = (min(all_y0) + max(all_y1)) / 2
            rx = max((max(all_x1) - min(all_x0)) / 2 + m, REGION_RADIUS)
            ry = max((max(all_y1) - min(all_y0)) / 2 + m, REGION_RADIUS)
            new_bound = torch.tensor([
                [cx - rx, cy - ry, -0.1],
                [cx + rx, cy + ry,  0.1],
            ], dtype=torch.float32)

            new_region = Region(
                model=self.model, bound=new_bound,
                init_network=True, eval=False,
            )
            new_region.gridmap = self.gridmap
            new_region.all_lidar = []
            self.all_regions[new_region.version] = new_region

            for room in unassigned:
                new_region.add_room(room, self.gridmap.room_connectivity_graph)

            regions_changed.add(new_region)
            center = np.array(new_region.center[:2], dtype=np.float32)
            self.exploration_graph.ensure_region(new_region.version, center)
            self.get_logger().info(
                f"[ROOM PROCESS] New region {new_region.version} created "
                f"bound=[({new_bound[0,0]:.1f},{new_bound[0,1]:.1f}) → "
                f"({new_bound[1,0]:.1f},{new_bound[1,1]:.1f})]"
            )

            # If no active region yet, adopt this one so training frames
            # are attributed correctly and _unload_current_region can't
            # accidentally unload a region the training thread is still using.
            if self.region is None:
                self.region = new_region
                self.get_logger().info(
                    f"[ROOM PROCESS] Adopted region {new_region.version} "
                    f"as active region"
                )

            # Enter WARMUP for the new region only if we're not already in one
            if self.mode not in ("BOOTSTRAP", "WARMUP"):
                self.mode               = "WARMUP"
                self.warmup_start_epoch = int(new_region.epoch)
                self.goal_xy            = None
                self.nav_state          = "IDLE"
                self.latest_candidate_path = None
                self._path_gen_running  = False
                self.get_logger().info(
                    f"[ROOM PROCESS] Entering WARMUP for region {new_region.version}"
                )

        # ── Rebuild intra-room door edges for all changed regions ──────────
        self.gridmap.compute_doors_in_room(self.all_regions)

        # ── Register door-level edges in ExplorationGraph ─────────────────
        # (same logic as the update() tick block but targeted at changed regions)
        for u, v, edge_data in self.gridmap.room_graph.edges(data=True):
            room_u = self.gridmap.room_graph.nodes[u].get('room', -1)
            room_v = self.gridmap.room_graph.nodes[v].get('room', -1)
            if room_u not in self.gridmap.confirmed_rooms:
                continue
            if room_v not in self.gridmap.confirmed_rooms:
                continue
            pos_u = self.gridmap.room_graph.nodes[u].get('pos')
            pos_v = self.gridmap.room_graph.nodes[v].get('pos')
            if pos_u is None or pos_v is None:
                continue
            door_xy = np.array(
                [(pos_u[0]+pos_v[0])/2, (pos_u[1]+pos_v[1])/2],
                dtype=np.float32,
            )
            reg_u = getattr(self.gridmap.confirmed_rooms[room_u], 'region', None)
            reg_v = getattr(self.gridmap.confirmed_rooms[room_v], 'region', None)
            if reg_u is None or reg_v is None:
                continue
            if reg_u.version == reg_v.version:
                continue
            self.exploration_graph.add_door_edge(reg_u.version, reg_v.version, door_xy)


    def _switch_region(self, new_region: 'Region'):
        old_ver = self.region.version

        # keep last scans
        if self.region is not None and self.region.all_lidar is not None:
            new_region.all_lidar = self.region.all_lidar[-5:].copy()
            new_region.lidar_obs = None
            new_region.kd_tree = None

        self._unload_current_region()
        self.region = new_region
        self.get_logger().info(
            f"[REGION SWITCH] {old_ver} → {new_region.version}"
        )
        self.goal_xy = None
        self.last_goal_time = self.get_clock().now()
        self.gridmap.frontier[:] = 0

    def _unload_current_region(self):
        """Save current region weights to disk and free GPU memory."""
        if self.region is None:
            return
        try:
            self.region.save_model()          # saves checkpoint to region folder
            self.region.unload_to_disk()      # releases GPU tensors
            self.get_logger().info(
                f"[REGION {self.region.version} UNLOADED]"
            )
        except Exception as e:
            self.get_logger().error(f"[UNLOAD ERROR] {e}")

    def _soft_extend_current_region(self, robot):
        """
        Softly expand region a bit so we don't immediately spawn a new one,
        BUT stop expanding once region gets too large.
        """

        b = self.region.bound.numpy().copy()
        pad = 0.3

        # --- proposed new bound ---
        new_min_x = min(b[0, 0], robot[0] - REGION_RADIUS - pad)
        new_min_y = min(b[0, 1], robot[1] - REGION_RADIUS - pad)
        new_max_x = max(b[1, 0], robot[0] + REGION_RADIUS + pad)
        new_max_y = max(b[1, 1], robot[1] + REGION_RADIUS + pad)

        # --- current size ---
        cur_w = b[1, 0] - b[0, 0]
        cur_h = b[1, 1] - b[0, 1]

        # --- new size if applied ---
        new_w = new_max_x - new_min_x
        new_h = new_max_y - new_min_y

        # 🧠 IMPORTANT: cap region size
        MAX_SIZE = REGION_RADIUS * 2.0   # ≈ 2.5m if radius=1.0

        if new_w > MAX_SIZE or new_h > MAX_SIZE:
            # Stop expanding → allow new region to spawn
            self.get_logger().info(
                f"[SOFT EXTEND BLOCKED] region already large "
                f"(w={cur_w:.2f}, h={cur_h:.2f})"
            )
            return

        # --- apply extension ---
        b[0, 0] = new_min_x
        b[0, 1] = new_min_y
        b[1, 0] = new_max_x
        b[1, 1] = new_max_y

        new_bound = torch.tensor(b, dtype=torch.float32)
        self.region.set_bound(new_bound)

        self.get_logger().info(
            f"[SOFT EXTEND] new bound: "
            f"({b[0,0]:.2f},{b[0,1]:.2f}) → ({b[1,0]:.2f},{b[1,1]:.2f})"
        )

    def find_traversable_crossing(
        self,
        reg1_bound: np.ndarray,
        reg2_bound: np.ndarray,
        robot_xy: np.ndarray,
        n_candidates: int = 60,
    ) -> Optional[np.ndarray]:
        """
        Find the best traversable point inside the overlap zone between two
        region bounding boxes.

        Strategy:
          1. Compute the axis-aligned overlap rectangle.
          2. Sample a grid of candidate points inside it.
          3. Keep only points that are (a) visited/free in the gridmap and
             (b) not occupied.
          4. Return the candidate closest to robot_xy (biased toward the
             overlap centre so we don't skirt along the edge).

        Returns None if no traversable crossing is found.
        """
        # ── Overlap rectangle ─────────────────────────────────────────────
        ox0 = max(reg1_bound[0, 0], reg2_bound[0, 0])
        oy0 = max(reg1_bound[0, 1], reg2_bound[0, 1])
        ox1 = min(reg1_bound[1, 0], reg2_bound[1, 0])
        oy1 = min(reg1_bound[1, 1], reg2_bound[1, 1])

        if ox1 <= ox0 or oy1 <= oy0:
            return None  # no geometric overlap at all

        # ── Sample grid inside overlap ────────────────────────────────────
        side = int(np.ceil(np.sqrt(n_candidates)))
        xs = np.linspace(ox0, ox1, side)
        ys = np.linspace(oy0, oy1, side)
        XX, YY = np.meshgrid(xs, ys)
        candidates = np.stack([XX.ravel(), YY.ravel()], axis=1)  # (N, 2)

        # ── Filter: free and visited ──────────────────────────────────────
        cell   = self.gridmap.cell_size
        origin = self.gridmap.origin
        half   = self.gridmap.map_size / 2.0

        free = []
        for pt in candidates:
            ix = int((pt[0] - (origin[0] - half)) / cell)
            iy = int((pt[1] - (origin[1] - half)) / cell)
            if not (0 <= ix < self.gridmap.n_cells and
                    0 <= iy < self.gridmap.n_cells):
                continue
            if self.gridmap.data[iy, ix] == 1:
                continue   # occupied / wall
            if self.gridmap.visited[iy, ix] != 1:
                continue   # not yet seen — could hide a wall
            free.append(pt)

        if not free:
            return None

        free = np.array(free)

        # ── Pick candidate closest to robot (to minimise detour) ─────────
        dists = np.linalg.norm(free - robot_xy[None, :], axis=1)
        return free[np.argmin(dists)]

    def build_cross_region_path(
        self,
        robot_xy: np.ndarray,
        goal_xy: np.ndarray,
        door_waypoints: list,
        target_version: int,
    ) -> Optional[np.ndarray]:
        """
        Build a full field-stitched path from robot_xy → goal_xy that crosses
        one or more region boundaries.

        For each hop in door_waypoints the method tries to:
          (a) Replace the raw midpoint waypoint with a validated traversable
              crossing point (found by scanning visited/free cells in the
              overlap zone).
          (b) Generate a field-guided segment from the previous position to
              the crossing using the current region's field.

        The final segment (crossing → goal) uses the target region's field if
        that region is loaded; otherwise falls back to straight-line.

        Returns an (N, 2) float32 path, or None on failure.
        """
        if self.region is None:
            return None

        full_path: list = []
        cur_pos    = robot_xy.copy()
        cur_region = self.region

        # Collect the region objects along the route (current → ... → target)
        # door_waypoints has already been computed by ExplorationGraph and gives
        # one waypoint per region boundary crossing.
        for wp_idx, raw_wp in enumerate(door_waypoints):
            # Determine which two regions this crossing is between.
            # We find the *next* region along the chain by checking which
            # registered region (not cur_region) is closest to the raw waypoint.
            next_region = None
            min_d = float('inf')
            for ver, reg in self.all_regions.items():
                if ver == cur_region.version:
                    continue
                b = reg.bound.detach().cpu().numpy()
                center = np.array([(b[0, 0] + b[1, 0]) / 2,
                                   (b[0, 1] + b[1, 1]) / 2], dtype=np.float32)
                d = float(np.linalg.norm(raw_wp - center))
                if d < min_d:
                    min_d = d
                    next_region = reg

            if next_region is None:
                # Can't identify next region — append raw wp and continue
                full_path.append(raw_wp.copy())
                cur_pos = raw_wp.copy()
                continue

            # Try to find a real traversable crossing in the overlap zone
            b1 = cur_region.bound.detach().cpu().numpy()
            b2 = next_region.bound.detach().cpu().numpy()
            crossing = self.find_traversable_crossing(b1, b2, cur_pos)

            if crossing is None:
                # Fall back: use the raw midpoint from the exploration graph
                self.get_logger().warn(
                    f"[CROSS-REGION PATH] No traversable crossing found between "
                    f"R{cur_region.version}↔R{next_region.version}, using raw midpoint"
                )
                crossing = raw_wp.copy()

            # Generate field-guided segment: cur_pos → crossing
            seg = cur_region.generate_field_path_to_crossing(
                cur_pos, crossing, step_size=0.12, max_steps=100
            )
            if seg and len(seg) > 0:
                full_path.extend(seg)
            else:
                full_path.append(crossing.copy())

            cur_pos    = crossing.copy()
            cur_region = next_region

        # ── Final segment: last crossing → goal (use target region's field) ──
        target_reg = self.all_regions.get(target_version)
        if target_reg is not None and target_reg.function is not None and target_reg.loaded:
            final_seg = target_reg.generate_field_path(
                cur_pos, goal_xy, step_size=0.12, max_steps=100
            )
        else:
            # Target region not loaded — straight-line fallback
            n_steps = max(3, int(np.linalg.norm(goal_xy - cur_pos) / 0.25))
            final_seg = [
                cur_pos + t * (goal_xy - cur_pos)
                for t in np.linspace(0, 1, n_steps)
            ]

        if final_seg and len(final_seg) > 0:
            full_path.extend(final_seg)
        else:
            full_path.append(goal_xy.copy())

        if not full_path:
            return None

        pts = np.array(full_path, dtype=np.float32)
        # Deduplicate very close consecutive points
        if len(pts) > 1:
            keep = [0]
            for i in range(1, len(pts)):
                if np.linalg.norm(pts[i] - pts[keep[-1]]) > 0.05:
                    keep.append(i)
            pts = pts[keep]

        return pts

    def _compute_path_sync(self, robot_xy: np.ndarray,
                            goal_xy: np.ndarray) -> Optional[np.ndarray]:
        """
        Synchronously compute a path from robot_xy to goal_xy.
        Tries room-graph trajectory first, falls back to single-region field.
        Returns decimated+smoothed path or None.
        """
        if robot_xy is None or goal_xy is None:
            return None

        try:
            path = self.compute_trajectory_from_graph(robot_xy, goal_xy)

            if path is None or len(path) < 2:
                if self.region is not None and self.region.function is not None:
                    path = self._query_field_segment(
                        self.region, robot_xy, goal_xy,
                        step_size=0.20, max_steps=40
                    )
                    if path:
                        path = np.array(path, dtype=np.float32)

            if path is None or len(path) < 2:
                return None

            dec = self.decimate_path(path, spacing=0.3)
            smo = self.smooth_path(dec, window=5)
            return np.asarray(smo, dtype=np.float32)

        except Exception as e:
            self.get_logger().warn(f"[PATH SYNC] {e}")
            return None
    
    def update_region_doors(self):
        """
        Assign confirmed doors to regions by spatial containment.
        Two regions that share a door get a door edge in the exploration graph
        using the real door world position as the crossing waypoint.
        Called at 1 Hz (every do_room_update tick).
        """
        if not self.gridmap.doors:
            return
    
        # Clear and recompute door assignments for all regions
        for region in self.all_regions.values():
            if not hasattr(region, 'door_ids'):
                region.door_ids = set()
            region.door_ids.clear()
    
        # Assign each door to every region whose bounding box contains it
        for door in self.gridmap.doors:
            door_xy = np.array(door.pos[:2], dtype=np.float32)
            for region in self.all_regions.values():
                if region.in_bound(door_xy):
                    region.door_ids.add(door.id)
    
        # Build exploration graph edges from shared doors
        versions = list(self.all_regions.keys())
        for i, v1 in enumerate(versions):
            for v2 in versions[i+1:]:
                reg1 = self.all_regions[v1]
                reg2 = self.all_regions[v2]
                shared = getattr(reg1, 'door_ids', set()) & \
                         getattr(reg2, 'door_ids', set())
                if not shared:
                    continue
                # Use the door closest to the boundary midpoint as waypoint
                b1 = reg1.bound.numpy()
                b2 = reg2.bound.numpy()
                mid = np.array([
                    (max(b1[0,0], b2[0,0]) + min(b1[1,0], b2[1,0])) / 2.0,
                    (max(b1[0,1], b2[0,1]) + min(b1[1,1], b2[1,1])) / 2.0
                ], dtype=np.float32)
                best_door = min(
                    (d for d in self.gridmap.doors if d.id in shared),
                    key=lambda d: np.linalg.norm(
                        np.array(d.pos[:2], dtype=np.float32) - mid
                    )
                )
                door_xy = np.array(best_door.pos[:2], dtype=np.float32)
                self.exploration_graph.add_door_edge(v1, v2, door_xy)
    

    def _launch_path_gen(self, robot_xy, goal_xy):
        """
        Offload generate_field_path to a background thread.

        Guards:
          - Skip if training is actively running (prevents concurrent
            forward+backward through the same network weights).
          - Capture the region object at launch time so a region switch
            mid-flight doesn't corrupt the result.
          - After generating, re-anchor the path start to the robot's
            *current* position (not the stale captured position) so the
            first waypoint is never behind the robot.
        """
        if robot_xy is None or goal_xy is None:
            return
        if self.region is None or self.region.function is None:
            return
        if getattr(self, '_path_gen_running', False):
            return
        # Don't query the field while a backward pass is in progress —
        # PyTorch is not thread-safe for simultaneous forward + backward.
        if self.train_running:
            return
        self._path_gen_running = True
        robot_xy       = robot_xy.copy()
        goal_xy        = goal_xy.copy()
        captured_region = self.region   # snapshot — survives region switches

        def _worker():
            try:
                raw_path = captured_region.generate_field_path(
                    robot_xy, goal_xy, step_size=0.12, max_steps=100
                )
                if raw_path is None:
                    # generate_field_path returns None when too many geometric
                    # fallbacks occurred — field not ready, skip this tick
                    return
                raw_path = np.asarray(raw_path, dtype=np.float32)
                if len(raw_path) < 2:
                    return

                # Re-anchor to current robot position so the path never
                # starts behind the robot (Cause 4).
                cur_robot = self.robot_pos[:2] if self.robot_pos is not None else robot_xy
                dists = np.linalg.norm(raw_path - cur_robot[None, :], axis=1)
                anchor = int(np.argmin(dists))
                raw_path = raw_path[anchor:]   # trim everything before closest point
                if len(raw_path) < 2:
                    raw_path = np.vstack([cur_robot, raw_path])

                decimate = self.decimate_path(raw_path, spacing=0.3)
                smooth   = self.smooth_path(decimate, window=5)
                self.latest_candidate_path = np.asarray(smooth, dtype=np.float32)
            except Exception as e:
                self.get_logger().warn(f"[PATH GEN] {e}")
            finally:
                self._path_gen_running = False
        threading.Thread(target=_worker, daemon=True).start()

    def update(self):
        now = self.get_clock().now()

        if self.robot_pos is None or self.latest_cloud is None:
            return

        # ── LiDAR processing ─────────────────────────────────────────────
        result = self.adapter.update(self.latest_cloud, self.robot_pos)
        if result is None:
            return
        points_lidar, speeds = result
        if len(points_lidar) == 0:
            return

        try:
            source_frame = self.latest_cloud.header.frame_id
            stamp = self.latest_cloud.header.stamp
            if self.tf_buffer.can_transform('odom', source_frame, stamp,
                                             timeout=Duration(seconds=0.1)):
                transform = self.tf_buffer.lookup_transform('odom', source_frame, stamp)
            else:
                return
            t = transform.transform.translation
            q = transform.transform.rotation
            R = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
            world_points = (R @ points_lidar.T).T + np.array([t.x, t.y, t.z])
        except Exception as e:
            self.get_logger().warn(f"TF lookup failed: {e}")
            return

        robot_xy = self.robot_pos[:2]
        half = self.gridmap.map_size / 2
        ox, oy = self.gridmap.origin
        in_bounds = ((world_points[:, 0] > ox - half) & (world_points[:, 0] < ox + half) &
                     (world_points[:, 1] > oy - half) & (world_points[:, 1] < oy + half))
        world_points_filtered = world_points[in_bounds]
        speeds_filtered = speeds[in_bounds]
        if len(world_points_filtered) < 5:
            return

        # Angular downsampling
        vecs  = world_points_filtered[:, :2] - robot_xy[None, :]
        dists = np.linalg.norm(vecs, axis=1) + 1e-9
        angles = np.arctan2(vecs[:, 1], vecs[:, 0])
        bins  = ((angles + np.pi) / (2 * np.pi) * 359).astype(np.int32)
        closest_idx = {}
        for k, b in enumerate(bins):
            if b not in closest_idx or dists[k] < dists[closest_idx[b]]:
                closest_idx[b] = k
        keep = np.array(list(closest_idx.values()), dtype=np.int32)
        world_points_hit = world_points_filtered[keep]
        speeds_hit = speeds_filtered[keep]

        # Raycast
        cell = self.gridmap.cell_size
        origin = self.gridmap.origin
        half_  = self.gridmap.map_size / 2

        def world_to_ij(x, y):
            return (int((x - origin[0] + half_) / cell),
                    int((y - origin[1] + half_) / cell))

        for pt3d in world_points_hit:
            pt = pt3d[:2]
            dx, dy = pt[0] - robot_xy[0], pt[1] - robot_xy[1]
            dist = float(np.hypot(dx, dy))
            if dist > MAX_RANGE:
                scale = MAX_RANGE / dist
                dx *= scale; dy *= scale; dist = MAX_RANGE
            steps = int(dist / cell)
            if steps < 1:
                ix, iy = world_to_ij(pt[0], pt[1])
                if 0 <= ix < self.gridmap.n_cells and 0 <= iy < self.gridmap.n_cells:
                    self.gridmap.data[iy, ix] = 1
                    self.gridmap.visited[iy, ix] = 1
                continue
            for s in range(steps - 1):
                x = robot_xy[0] + dx * s / steps
                y = robot_xy[1] + dy * s / steps
                ix, iy = world_to_ij(x, y)
                if not (0 <= ix < self.gridmap.n_cells and
                        0 <= iy < self.gridmap.n_cells):
                    break
                if self.gridmap.data[iy, ix] == 1:
                    break
                self.gridmap.visited[iy, ix] = 1
            ix_end, iy_end = world_to_ij(pt[0], pt[1])
            if (0 <= ix_end < self.gridmap.n_cells and
                    0 <= iy_end < self.gridmap.n_cells):
                self.gridmap.data[iy_end, ix_end] = 1
                self.gridmap.visited[iy_end, ix_end] = 1

        # ── Gridmap update (1 Hz room segmentation) ──────────────────────
        self._map_update_counter += 1
        do_room_update = (self._map_update_counter % 5 == 0)
        self.gridmap.update(
            lidar_scan=world_points_hit[:, :2],
            points=world_points_hit[:, :2],
            speeds=speeds_hit,
            cur_pos=robot_xy,
            region=self.region,
            regions=self.all_regions,
            update_room=do_room_update
        )

        # ── Room-driven region spawning ───────────────────────────────────
        if do_room_update and self.gridmap.new_confirmed_rooms:
            self.process_newly_confirmed_rooms(list(self.gridmap.new_confirmed_rooms))

        # ── Frontier snapshot ─────────────────────────────────────────────
        if self.region is not None:
            ver    = self.region.version
            center = np.array(self.region.center[:2], dtype=np.float32)
            self.exploration_graph.ensure_region(ver, center)

            iy_f, ix_f = np.where(self.gridmap.frontier > 0)
            if len(ix_f) > 0:
                cell_  = self.gridmap.cell_size
                orig   = self.gridmap.origin
                half_m = self.gridmap.map_size / 2.0
                fx = (orig[0] - half_m) + (ix_f + 0.5) * cell_
                fy = (orig[1] - half_m) + (iy_f + 0.5) * cell_
                live_frontiers = np.stack([fx, fy], axis=1)
                dists_f = np.linalg.norm(live_frontiers - robot_xy[None, :], axis=1)
                live_frontiers = live_frontiers[
                    (dists_f >= 0.8) & (dists_f <= MAX_RANGE)
                ]
                if len(live_frontiers) > 50:
                    live_frontiers = live_frontiers[
                        np.random.choice(len(live_frontiers), 50, replace=False)
                    ]
                if len(live_frontiers) > 0:
                    self.exploration_graph.snapshot_frontiers(ver, live_frontiers)

        # ── Region door graph (1 Hz) ──────────────────────────────────────
        if do_room_update:
            self.update_region_doors()

        # ── Merge check (0.5 Hz) ─────────────────────────────────────────
        if self._map_update_counter % 10 == 0:
            self.maybe_merge_regions()

        # ── Trajectory history ────────────────────────────────────────────
        if self.robot_pos is not None:
            self._traj_history.append(self.robot_pos[:2].copy())

        # ══════════════════════════════════════════════════════════════════
        # MODE STATE MACHINE
        # ══════════════════════════════════════════════════════════════════

        if self.mode == "BOOTSTRAP":
            self.navigator.stop_robot()
            self.bootstrap_frames += 1
            if self.bootstrap_frames >= self.bootstrap_threshold:
                if self.all_regions:
                    self.region = next(iter(self.all_regions.values()))
                    self.get_logger().info(
                        f"[FLOW] BOOTSTRAP adopting existing region {self.region.version}"
                    )
                else:
                    self.init_region(self.robot_pos)
                self.mode               = "WARMUP"
                self.warmup_start_epoch = int(self.region.epoch)
                self.goal_xy            = None

        elif self.mode == "WARMUP":
            # Collect frames while stationary/roaming, train synchronously
            self._collect_frame_if_moved(world_points_hit[:, :2], self.robot_pos)
            self.navigator.stop_robot()

            cur_epochs = int(self.region.epoch) - int(self.warmup_start_epoch)
            print(f"[WARMUP] epoch {cur_epochs}/{self.warmup_epochs_new_region}")

            if len(self._frame_buffer) >= 3:
                self._train_synchronous(n_epochs=2)     # train 2 epochs per 3 frames

            if cur_epochs >= self.warmup_epochs_new_region:
                self._frame_buffer.clear()
                self.mode      = "EXPLORE"
                self.warmup_done = True
                self.goal_xy   = None
                self.nav_state = "IDLE"
                self.get_logger().info("[FLOW] Warmup complete → EXPLORE")

        elif self.mode == "EXPLORE":
            robot_xy = np.asarray(self.robot_pos[:2], dtype=np.float32)

            # ── IDLE: pick goal, train on buffered frames, compute path ──
            if self.nav_state == "IDLE":
                # 1. Train on any buffered frames from previous goal
                if self._frame_buffer:
                    self.navigator.stop_robot()
                    self._train_synchronous(n_epochs=self.warmup_epochs_per_goal)

                # 2. Pick next frontier goal
                current_room = self.gridmap.get_room_at_position(robot_xy)
                rooms        = self.gridmap.get_rooms()
                room_graph   = self.gridmap.get_room_graph()
                g = self.pick_room_aware_goal(current_room, rooms, room_graph)

                if g is None:
                    self.navigator.stop_robot()
                    print("[NO FRONTIER]")
                else:
                    self.goal_xy = np.asarray(g, dtype=np.float32)
                    print(f"[GOAL SET] {self.goal_xy}")

                    # 3. Compute path synchronously now (field just trained)
                    self.path = self._compute_path_sync(robot_xy, self.goal_xy)
                    self._current_path_computed = self.path is not None
                    self._path_recompute_counter = 0

                    if self.path is not None:
                        self.navigator.reset_path()
                        self.nav_state = "FOLLOWING"
                        print(f"[PATH SET] {len(self.path)} waypoints")
                    else:
                        print("[NO PATH] goal unreachable, staying IDLE")
                        self.goal_xy = None

                    # Load door waypoints if cross-region
                    if self._remote_door_waypoints and self.goal_xy is not None:
                        door_path = self._compute_path_sync(
                            robot_xy, self.goal_xy
                        )
                        if door_path is not None and len(door_path) > 1:
                            self.path = door_path
                            self.navigator.reset_path()
                        self._remote_door_waypoints = []

            # ── FOLLOWING: drive + collect frames + periodic path refresh ─
            elif self.nav_state == "FOLLOWING":
                if self.goal_xy is None:
                    self.nav_state = "IDLE"
                    self.path = None
                    return

                d_goal = float(np.linalg.norm(robot_xy - self.goal_xy))
                print(f"[FOLLOWING] dist={d_goal:.2f}m  "
                      f"frames={len(self._frame_buffer)}")

                # ── Goal reached ─────────────────────────────────────────
                if d_goal < 0.25:
                    print("[FRONTIER REACHED]")
                    self.navigator.stop_robot()
                    self.navigator.reset_path()

                    # Clear frontier cells around goal
                    gix = int((self.goal_xy[0] - (self.gridmap.origin[0]
                                - self.gridmap.map_size / 2))
                               / self.gridmap.cell_size)
                    giy = int((self.goal_xy[1] - (self.gridmap.origin[1]
                                - self.gridmap.map_size / 2))
                               / self.gridmap.cell_size)
                    pad = 3
                    for di in range(-pad, pad + 1):
                        for dj in range(-pad, pad + 1):
                            ii, jj = giy + di, gix + dj
                            if (0 <= ii < self.gridmap.n_cells and
                                    0 <= jj < self.gridmap.n_cells):
                                self.gridmap.frontier[ii, jj] = 0
                                self.gridmap.visited[ii, jj]  = 1

                    self._visited_goals.append(self.goal_xy.copy())
                    if len(self._visited_goals) > 50:
                        self._visited_goals.pop(0)

                    if self.region is not None:
                        self.exploration_graph.mark_goal_reached(
                            self.region.version, self.goal_xy
                        )

                    # Collect one final frame at goal position
                    self._collect_frame_if_moved(
                        world_points_hit[:, :2], self.robot_pos, min_dist=0.0
                    )

                    self.goal_xy   = None
                    self.path      = None
                    self._remote_door_waypoints = []
                    self._remote_target_version = None
                    self.nav_state = "IDLE"
                    # Training + next goal happens on next IDLE tick

                else:
                    # ── Collect position-gated frame ─────────────────────
                    self._collect_frame_if_moved(
                        world_points_hit[:, :2], self.robot_pos
                    )

                    # ── Periodic path recompute (every 5 ticks = 1 Hz) ───
                    self._path_recompute_counter += 1
                    if self._path_recompute_counter >= 5:
                        self._path_recompute_counter = 0
                        new_path = self._compute_path_sync(robot_xy, self.goal_xy)
                        if new_path is not None and len(new_path) > 1:
                            self.path = new_path
                            self.navigator.reset_path()

                    # ── Drive ─────────────────────────────────────────────
                    if self.path is not None:
                        self.navigator.drive_path(self.path)
                    else:
                        # Path lost — try once more then give up
                        self.path = self._compute_path_sync(robot_xy, self.goal_xy)
                        if self.path is not None:
                            self.navigator.reset_path()
                            self.navigator.drive_path(self.path)
                        else:
                            print("[PATH LOST] returning to IDLE")
                            self.navigator.stop_robot()
                            self.goal_xy   = None
                            self.nav_state = "IDLE"

        # ── Composite plot (every 20 ticks) ──────────────────────────────
        if self._composite_step % 20 == 0:
            try:
                self.save_composite_plot()
            except Exception as e:
                self.get_logger().warn(f"[COMPOSITE PLOT] {e}")
        self._composite_step += 1

        # ── Publish ───────────────────────────────────────────────────────
        self.map_publisher.publish_maps()
        self.markers.publish_goal_marker()
        self.markers.publish_frontier_markers()


    def save_composite_plot(self):
        """
        plot2-style composite:
          - world-scale background (unknown / visited / obstacles)
          - per-region travel-time contours that PERSIST via a numeric cache (no translucent overlays)
          - exploration graph (nodes/edges/doors)
          - ONLY the latest saved frontiers per region (not the entire bank)
        """
        from matplotlib.lines import Line2D
        os.makedirs(self.folder, exist_ok=True)
        out_dir = os.path.join(self.folder, "plots2")
        os.makedirs(out_dir, exist_ok=True)

        # World-map bounds (gridmap is world-sized)
        b_map = [
            [self.gridmap.origin[0] - self.gridmap.map_size / 2,
             self.gridmap.origin[1] - self.gridmap.map_size / 2],
            [self.gridmap.origin[0] + self.gridmap.map_size / 2,
             self.gridmap.origin[1] + self.gridmap.map_size / 2],
        ]

        fig, ax = plt.subplots(1, 1, figsize=(14, 14))

        # ── Background: visited=dark green, obstacle=black, unknown=dark blue ──
        data = self.gridmap.data
        vis  = self.gridmap.visited
        n    = self.gridmap.n_cells
        cell = self.gridmap.cell_size

        bg = np.zeros((n, n, 3), dtype=np.uint8)
        bg[:, :] = [10, 10, 80]        # unknown
        bg[vis == 1]  = [220, 220, 40]     # visited free
        bg[data == 1] = [0, 0, 0]      # obstacles

        extent = [b_map[0][0], b_map[1][0], b_map[0][1], b_map[1][1]]
        ax.imshow(bg, origin='lower', extent=extent, zorder=0, aspect='auto')

        # ── Persistent numeric cache for region fields ─────────────────────────
        # Cache: version -> dict(X_w, Y_w, TT_masked, extent, src_world, step, epoch)
        if not hasattr(self, "_region_field_cache"):
            self._region_field_cache = {}

        # Tuning
        GRID_N = 90                 # resolution of per-region TT grid
        CONTOUR_LEVELS = 40         # number of contour levels
        FRONTIER_LAST_K = 30        # only show last K saved frontiers per region
        CACHE_UPDATE_EVERY = 1      # update active region every step (keep expanding field visible)

        def _build_region_mask(b, grid_n=GRID_N):
            """Mask out cells outside visited/obstacle footprint (same logic as your original)."""
            mask = np.zeros((grid_n, grid_n), dtype=bool)
            for i in range(grid_n):
                for j in range(grid_n):
                    wx = b[0, 0] + (i + 0.5) / grid_n * (b[1, 0] - b[0, 0])
                    wy = b[0, 1] + (j + 0.5) / grid_n * (b[1, 1] - b[0, 1])
                    gx = int((wx - b_map[0][0]) / cell)
                    gy = int((wy - b_map[0][1]) / cell)
                    if 0 <= gx < n and 0 <= gy < n:
                        mask[j, i] = (vis[gy, gx] == 1 or data[gy, gx] == 1)
            return mask

        def _compute_tt_grid(region, src_world, b, mask, grid_n=GRID_N):
            """Compute TT(x0=src, x1=grid) over region bound and apply mask => TT_masked."""
            xmin, ymin = float(b[0, 0]), float(b[0, 1])
            xmax, ymax = float(b[1, 0]), float(b[1, 1])

            xs = np.linspace(xmin, xmax, grid_n)
            ys = np.linspace(ymin, ymax, grid_n)
            X_w, Y_w = np.meshgrid(xs, ys)

            # world -> normalized using region bound
            sx = (xmax - xmin) + 1e-9
            sy = (ymax - ymin) + 1e-9
            X_n = (X_w - xmin) / sx - 0.5
            Y_n = (Y_w - ymin) / sy - 0.5

            src_norm = region.normalize(src_world.reshape(1, 3))[0]  # (3,)

            XP = np.zeros((X_w.size, 2 * region.dim), dtype=np.float32)
            XP[:, :region.dim] = src_norm
            XP[:, region.dim + 0] = X_n.reshape(-1)
            XP[:, region.dim + 1] = Y_n.reshape(-1)
            XP[:, region.dim + 2] = src_norm[2]

            XP_t = torch.tensor(XP, dtype=torch.float32, device=region.device)

            with torch.no_grad():
                TT_t = region.function.TravelTimes(XP_t)
                if TT_t.ndim == 2 and TT_t.shape[1] == 1:
                    TT_t = TT_t[:, 0]
                TT = TT_t.detach().cpu().numpy().reshape(X_w.shape)

            TT_masked = TT.copy()
            TT_masked[~mask] = np.nan

            return X_w, Y_w, TT_masked

        def _pick_src_world(region, b):
            """Stable source for plotting TT contours."""
            # Active region uses robot position if available
            if self.region is not None and region.version == self.region.version and self.robot_pos is not None:
                return np.array([self.robot_pos[0], self.robot_pos[1], 0.0], dtype=np.float32)

            # Inactive: prefer cached source if present, else region center
            cached = self._region_field_cache.get(region.version)
            if cached is not None and "src_world" in cached:
                return cached["src_world"]

            rc = (b[0] + b[1]) / 2.0
            return np.array([float(rc[0]), float(rc[1]), 0.0], dtype=np.float32)

        # ── Update cache for the active region (so its field "expands" live) ──
        if self.region is not None and (self._composite_step % CACHE_UPDATE_EVERY == 0):
            try:
                reg = self.region
                if reg.function is not None:
                    b = reg.bound.detach().cpu().numpy()
                    mask = _build_region_mask(b, grid_n=GRID_N)
                    src_world = _pick_src_world(reg, b)
                    X_w, Y_w, TT_masked = _compute_tt_grid(reg, src_world, b, mask, grid_n=GRID_N)
                    self._region_field_cache[reg.version] = dict(
                        X_w=X_w, Y_w=Y_w, TT=TT_masked,
                        extent=[b[0, 0], b[1, 0], b[0, 1], b[1, 1]],
                        mask=mask, src_world=src_world,
                        step=int(self._composite_step),
                        epoch=int(getattr(reg, "epoch", 0)),
                    )
            except Exception as e:
                self.get_logger().warn(f"[COMPOSITE] active region cache update failed: {e}")

        # ── Ensure each region has at least one cached snapshot (persist) ─────
        for region in self.all_regions.values():
            if region.version in self._region_field_cache:
                continue
            # Only cache if model exists right now; otherwise it will remain missing until loaded once
            if region.function is None:
                continue
            try:
                b = region.bound.detach().cpu().numpy()
                mask = _build_region_mask(b, grid_n=GRID_N)
                src_world = _pick_src_world(region, b)
                X_w, Y_w, TT_masked = _compute_tt_grid(region, src_world, b, mask, grid_n=GRID_N)
                self._region_field_cache[region.version] = dict(
                    X_w=X_w, Y_w=Y_w, TT=TT_masked,
                    extent=[b[0, 0], b[1, 0], b[0, 1], b[1, 1]],
                    mask=mask, src_world=src_world,
                    step=int(self._composite_step),
                    epoch=int(getattr(region, "epoch", 0)),
                )
            except Exception as e:
                self.get_logger().warn(f"[COMPOSITE] region {region.version} initial cache failed: {e}")

        # ── Draw persistent contours for all cached regions ───────────────────
        for region in self.all_regions.values():
            b = region.bound.detach().cpu().numpy()

            cached = self._region_field_cache.get(region.version, None)
            if cached is not None:
                X_w = cached["X_w"]
                Y_w = cached["Y_w"]
                TT  = cached["TT"]

                if np.isfinite(TT).any():
                    ax.contour(
                        X_w, Y_w, TT,
                        levels=CONTOUR_LEVELS,
                        cmap="gist_heat",
                        linewidths=0.65,
                        alpha=0.9,
                        zorder=2
                    )

            # ── Region bounding box (always) ─────────────────────────────────
            rect_w = float(b[1, 0] - b[0, 0])
            rect_h = float(b[1, 1] - b[0, 1])
            is_active = (self.region is not None and region.version == self.region.version)
            edge_color = 'yellow' if is_active else 'gray'
            lw = 2.0 if is_active else 0.8

            rect = plt.Rectangle(
                (float(b[0, 0]), float(b[0, 1])), rect_w, rect_h,
                linewidth=lw, edgecolor=edge_color, facecolor='none', zorder=4
            )
            ax.add_patch(rect)
            ax.text(
                float(b[0, 0]) + 0.02, float(b[1, 1]) - 0.15,
                f"R{region.version}{'*' if is_active else ''}",
                color=edge_color, fontsize=7, zorder=5
            )

        # ── ExplorationGraph overlay: edges + door waypoints ─────────────────
        for v1, v2, edata in self.exploration_graph.graph.edges(data=True):
            rec1 = self.exploration_graph.records.get(v1)
            rec2 = self.exploration_graph.records.get(v2)
            if rec1 is None or rec2 is None:
                continue

            c1 = rec1.center
            c2 = rec2.center

            door_xys = edata.get('door_xys', [edata.get('door_xy')])
            door_xys = [d for d in door_xys if d is not None]

            if door_xys:
                for door_xy in door_xys:
                    xs = [c1[0], door_xy[0], c2[0]]
                    ys = [c1[1], door_xy[1], c2[1]]
                    ax.plot(xs, ys, color='cyan', linewidth=1.2,
                            linestyle='--', alpha=0.7, zorder=6)
                    ax.scatter(door_xy[0], door_xy[1],
                               c='cyan', s=40, marker='D',
                               zorder=7, alpha=0.9)
            else:
                ax.plot([c1[0], c2[0]], [c1[1], c2[1]],
                        color='cyan', linewidth=1.0,
                        linestyle='--', alpha=0.6, zorder=6)

        # ── ExplorationGraph nodes (region centers) ──────────────────────────
        for ver, rec in self.exploration_graph.records.items():
            c = rec.center
            has_f = rec.has_frontiers()
            node_color = 'lime' if has_f else 'red'
            ax.scatter(c[0], c[1], c=node_color, s=120,
                       marker='o', zorder=8, edgecolors='white', linewidths=1.0)
            ax.text(c[0] + 0.05, c[1] + 0.05,
                    f"R{ver}\n({len(rec.saved_frontiers)}f)",
                    color='white', fontsize=6, zorder=9,
                    bbox=dict(boxstyle='round,pad=0.15', fc='black', alpha=0.5))

        # ── Latest saved frontier points per region (NOT all) ─────────────────
        for ver, rec in self.exploration_graph.records.items():
            if not rec.saved_frontiers:
                continue
            pts = rec.saved_frontiers[-FRONTIER_LAST_K:]  # latest K (append order)
            for fp in pts:
                ax.scatter(fp[0], fp[1], c='orange', s=14,
                           marker='.', zorder=7, alpha=0.85)

        # ── Robot, trajectory, goal ─────────────────────────────────────────
        if self.robot_pos is not None:
            ax.scatter(self.robot_pos[0], self.robot_pos[1],
                       c='red', s=100, zorder=11, marker='o', label='Robot')

        if len(self._traj_history) > 1:
            traj = np.array(self._traj_history)
            ax.plot(traj[:, 0], traj[:, 1], 'm-', linewidth=1.5, zorder=9)

        if self.goal_xy is not None:
            ax.scatter(self.goal_xy[0], self.goal_xy[1],
                       c='lime', s=180, zorder=11, marker='*', label='Goal')

        # ── Active inter-region door route ───────────────────────────────────
        if getattr(self, "_remote_door_waypoints", None):
            if len(self._remote_door_waypoints) > 0:
                wps = np.array(self._remote_door_waypoints)
                ax.plot(wps[:, 0], wps[:, 1], color='magenta',
                        linewidth=2.0, linestyle='-', zorder=10, label='Door route')
                ax.scatter(wps[:, 0], wps[:, 1], c='magenta',
                           s=80, marker='D', zorder=11)

        # ── Legend ──────────────────────────────────────────────────────────
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='lime',
                   markersize=8, label='Region (has frontiers)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                   markersize=8, label='Region (exhausted)'),
            Line2D([0], [0], marker='D', color='w', markerfacecolor='cyan',
                   markersize=7, label='Door waypoint'),
            Line2D([0], [0], color='cyan', linestyle='--', label='Graph edge'),
            Line2D([0], [0], marker='.', color='w', markerfacecolor='orange',
                   markersize=6, label=f'Latest {FRONTIER_LAST_K} frontiers'),
            Line2D([0], [0], marker='*', color='w', markerfacecolor='lime',
                   markersize=10, label='Current goal'),
            Line2D([0], [0], color='magenta', label='Active door route'),
        ]
        ax.legend(handles=legend_elements, loc='upper right',
                  fontsize=7, framealpha=0.7)

        ax.set_xlim(b_map[0][0], b_map[1][0])
        ax.set_ylim(b_map[0][1], b_map[1][1])
        ax.set_aspect('equal')
        ax.set_title(
            f"Step {self._composite_step}  |  regions={len(self.all_regions)}"
            f"  doors={len(self.gridmap.doors)}"
            f"  mode={self.mode}/{self.nav_state}"
        )

        fig.savefig(os.path.join(out_dir, f"plot_step_{self._composite_step}.png"),
                    dpi=120, bbox_inches='tight')
        fig.savefig(os.path.join(self.folder, "plot2.png"), bbox_inches='tight')
        plt.close(fig)

    def save_occupancy_snapshot(self, tag=None):
        """
        Saves occupancy grid + visited + frontier as image and numpy.
        """
        out_dir = os.path.join(self.folder, "occupancy_snapshots")
        os.makedirs(out_dir, exist_ok=True)

        if tag is None:
            tag = f"step_{self._composite_step:06d}"

        data = self.gridmap.data.copy()
        visited = self.gridmap.visited.copy()
        frontier = self.gridmap.frontier.copy()

        n = self.gridmap.n_cells

        # ---- Create visualization image ----
        img = np.zeros((n, n, 3), dtype=np.uint8)

        # unknown
        img[:, :] = [10, 10, 80]

        # visited free
        img[visited == 1] = [200, 200, 0]   # yellow (clearer than dark green)

        # occupied
        img[data == 1] = [0, 0, 0]

        # frontier overlay (optional highlight)
        img[frontier == 1] = [255, 140, 0]  # orange

        plt.figure(figsize=(8, 8))
        plt.imshow(img, origin="lower")
        plt.title(f"Occupancy Snapshot {tag}")
        plt.axis("off")
        plt.savefig(os.path.join(out_dir, f"{tag}.png"), dpi=150, bbox_inches="tight")
        plt.close()

        # ---- Save raw arrays ----
        np.savez_compressed(
            os.path.join(out_dir, f"{tag}.npz"),
            occupancy=data,
            visited=visited,
            frontier=frontier
        )

        self.get_logger().info(f"[OCC SNAPSHOT SAVED] {tag}")

def main():
    rclpy.init()
    node = GridmapNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()


import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
import rclpy.time
import torch
from scipy.spatial.transform import Rotation
import signal
import threading
from typing import List

import numpy as np
import math
import tf2_ros
from tf2_ros import TransformBroadcaster
import tf2_geometry_msgs

import sensor_msgs_py.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from geometry_msgs.msg import TransformStamped, Twist, Point, PointStamped, PoseStamped, PoseWithCovarianceStamped
from visualization_msgs.msg import Marker

from mntfields.mntfields_ros.sensing.lidar_adapter import Lidar2DAdapter
from mntfields.mntfields_ros.mapping.map_publisher import MapPublisher 
from mntfields.mntfields_ros.mapping.markers import MarkerPublisher
from mntfields.mntfields_ros.mapping.navigation import Navigator
from mntfields.mntfields_ros.mapping.exploration_graph import ExplorationGraph

from mntfields.mntfields_core.mapping.lidar_gridmap import LidarGridmap
from mntfields.mntfields_core.core.region import Region
from mntfields.mntfields_core.core.model_base import Model

from collections import deque
import threading

import os
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg", force=True)
from matplotlib import cm, colors
import matplotlib.pyplot as plt


REGION_RADIUS = 3.5
REGION_EXIT_MARGIN = 0.4 #was 0.4
REGION_MIN_OVERLAP = 0.4 #was 0.4
MAX_RANGE = 3.2
WARMUP_EPOCHS = 6
#EXPLORATION_RANGE = 2.5   # used for raycasting / visited marking

#print("NEW CODE")
class DummyRenderer:
    def __init__(self):
        pass

    def render(self, *args, **kwargs):
        return None

class DummyModel:
    def __init__(self):
        self.folder = "/home/mayank/turtlebot3_ws/mntfields_model"
        self.renderer = "cuda:0"   # IMPORTANT: must be string device
        self.h = 0.0

class FakePCD:
    def __init__(self, pts):
        self.points = pts
        self.normals = np.zeros_like(pts)

class GridmapNode(Node):
    def __init__(self):
        super().__init__('mntfields_gridmap')
        self.folder = os.path.join("/home/mayank/turtlebot3_ws/mntfields_model/mntfields_output")
        os.makedirs(self.folder, exist_ok=True)

        self.train_queue = deque(maxlen=4)   # keep last few frames only
        self.train_lock = threading.Lock()
        self.train_running = False
        self.declare_parameter('region_min_epochs', 10)  # train at least 20 times before 
        self._traj_history = []
        self._composite_step = 0

        # Tune these
        self.declare_parameter('train_timer_sec', 0.5)      # training loop cadence
        self.declare_parameter('train_samples', 5000)       # n_samples to get_data_from_lidar
        self.declare_parameter('train_epoch', 4)            # epoch passed into region.train_core
        self.declare_parameter('train_after_region', True)  # toggle
        self.radius = REGION_RADIUS

        # Start a separate timer JUST for training
        train_dt = float(self.get_parameter('train_timer_sec').value)
        self.create_timer(train_dt, self.train_tick)
        
        self.declare_parameter('n_cells', 400)
        self.declare_parameter('near_dist', 2.0)
        self.declare_parameter('map_size', 40.0)  # Add this

        n_cells = self.get_parameter('n_cells').value
        near_dist = self.get_parameter('near_dist').value
        map_size = self.get_parameter('map_size').value

        self.map_to_odom = None
        self.origin_initialized = False
        self.set_parameters([rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)])
        # --- exploration / control params ---
        self.declare_parameter('goal_reached_dist', 0.20)
        self.declare_parameter('yaw_tol', 0.25)          # rad
        self.declare_parameter('kp_yaw', 0.8)
        self.declare_parameter('kp_lin', 0.25)
        self.declare_parameter('max_lin', 0.15)          # m/s
        self.declare_parameter('max_ang', 0.8)           # rad/s
        self.declare_parameter('frontier_min_cells', 2)
        self.declare_parameter('goal_cooldown_sec', 2.0)

        self.path = None        #these 4 lines are for the path holding for stable paths - latest diff
        self.path_last_update = self.get_clock().now()
        self.path_update_period = 2.0   # seconds
        self.path_index = 0

        self.goal_xy = None
        self.last_goal_time = self.get_clock().now()
        self.mode = "BOOTSTRAP"   # BOOTSTRAP -> EXPLORE
        #--diff
        self.warmup_frames_needed = 5
        self.warmup_epochs_needed = WARMUP_EPOCHS
        self.warmup_done = False
        self.warmup_start_epoch = 0
        self.nav_state = "IDLE"
        self.wait_start_time = None
        self.wait_duration = 3.0
        self.latest_candidate_path = None
        #--
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        self.navigator = Navigator(self)
        self.map_publisher = MapPublisher(self)
        self.markers = MarkerPublisher(self)

        self.robot_pos = None
        self.robot_yaw = 0.0
        self.latest_cloud = None

        self.all_regions: dict = {}
        self.region_initialized = False
        self.bootstrap_frames = 0
        self.bootstrap_threshold = 10 # frames before region starts
        
        self.adapter = Lidar2DAdapter()

        # Traversability map (1 = free, 0 = blocked)
        self.trav_map = np.ones((n_cells, n_cells), dtype=np.uint8)
        self.gridmap = LidarGridmap(
            trav_map=self.trav_map,
            regions=self.all_regions,#{}
            n_cells=n_cells,
            near_dist=2.0,
            map_size=map_size,           # Add this
            origin=(0.0, 0.0), #(origin_x, origin_y)  # Add this
            room_seg_config=(7,2),
        )
        self.origin_initialized = False
        self.declare_parameter('occ_height', 0.3)
        self.occ_height = float(self.get_parameter('occ_height').value)
        self.occ_height_tol = 0.01

        self.create_subscription(PointCloud2,'/lidar3d_points',self.lidar_cb,10)
        #self.create_subscription(PoseWithCovarianceStamped,'/amcl_pose',self.pose_cb,5) - will be used for a built map
        self.create_subscription(Odometry, '/odom', self.odom_cb, 10)

        self.occ_pub = self.create_publisher(OccupancyGrid,'/mntfields/occ_map',5)
        self.frontier_pub = self.create_publisher(OccupancyGrid,'/mntfields/frontier',5)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.goal_marker_pub = self.create_publisher(Marker, '/mntfields/goal_marker', 10)
        self.frontier_marker_pub = self.create_publisher(Marker, '/mntfields/frontier_markers', 10)
        self.field_path_pub = self.create_publisher(Path, '/field_path', 10)

        self.create_timer(0.2, self.update)

        self.model = DummyModel()
        self.region = None

        self.exploration_graph = ExplorationGraph() #graph diff
        self._remote_door_waypoints: List[np.ndarray] = []  # door waypoints for inter-region travel
        self._remote_target_version: Optional[int] = None
        self._goal_blacklist_radius = 0.5 #was 0.3
        self._visited_goals: List[np.ndarray] = []

        self.train_counter = 0
        self.train_every_n_frames = 1

        self._map_update_counter = 0 #claude difff
        self._path_gen_running   = False

        # ── Goal-dwell training burst ─────────────────────────────────────
        self.declare_parameter('dwell_frames_needed', 6)    # frames to collect at goal
        self.declare_parameter('dwell_epochs',        3)    # epochs to train before resuming
        self.dwell_active       = False
        self.dwell_frames       = []                        # accumulated during dwell
        self.dwell_frames_count = 0
        self.dwell_epoch_start  = 0
        self.dwell_phase        = "COLLECT"  # "COLLECT" | "TRAIN" | "DONE"


    def lidar_cb(self, msg):
        self.latest_cloud = msg
        if not hasattr(self, "_last_lidar_log"):
            self._last_lidar_log = self.get_clock().now()
            #self.get_logger().info("Received first LiDAR scan")
            return
        now = self.get_clock().now()
        if (now - self._last_lidar_log).nanoseconds > 2e9:  # 2 seconds
            #self.get_logger().info("Receiving LiDAR scans...")
            self._last_lidar_log = now
    
    def odom_cb(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation

        self.robot_pos = np.array([p.x, p.y, p.z], dtype=np.float32)
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.robot_yaw = np.arctan2(siny_cosp, cosy_cosp)

        # Lock map once, using the *initial* odom pose
        if not self.origin_initialized:
            self.gridmap.origin = np.array([p.x, p.y], dtype=np.float32)
            self.origin_initialized = True

    def init_region(self, robot_pose):
        ''' Initializes a region around the robot and registers it. '''

        center_x = robot_pose[0]
        center_y = robot_pose[1]
        radius = self.radius

        bound = torch.tensor([
            [center_x - radius, center_y - radius, -0.1],
            [center_x + radius, center_y + radius,  0.1],
        ], dtype=torch.float32)

        new_region = Region(
            model=self.model,
            bound=bound,
            init_network=True,
            eval=False,)
        new_region.gridmap = self.gridmap
        self.all_regions[new_region.version] = new_region

        if self.region is not None and self.region.all_lidar is not None:
            last = self.region.all_lidar[-5:]
            new_region.all_lidar = last.copy()
            new_region.lidar_obs = None
            new_region.kd_tree = None
        else:
            new_region.all_lidar = []
        self.region = new_region
        self.region_initialized = True
        self.get_logger().info(
            f"[REGION {new_region.version} INIT] "
            f"center=({center_x:.2f}, {center_y:.2f}) "
            f"bound=[({bound[0,0]:.2f},{bound[0,1]:.2f}) → "
            f"({bound[1,0]:.2f},{bound[1,1]:.2f})]")
        # ── Re-enter WARMUP for every new region ─────────────────────────
        self.mode               = "WARMUP"
        self.warmup_start_epoch = int(self.region.epoch)
        self.goal_xy            = None
        self.path               = None
        self.nav_state          = "IDLE"
        self.latest_candidate_path = None
        self._path_gen_running  = False
        self.get_logger().info(f"[FLOW] Region {new_region.version} → entering WARMUP")

    def start_region_warmup(self):
        self.mode = "WARMUP"
        self.region_warmup_start_epoch = self.region.epoch if self.region else 0
        self.navigator.stop_robot()
        self.get_logger().info(f"[WARMUP] Starting region {self.region.version}")

    def feed_region(self, world_pts, robot_pose):
        if len(world_pts) < 50:
            self.get_logger().warn(f"Too few points: {len(world_pts)}")
            return
        try:
            pts = torch.tensor(world_pts, dtype=torch.float32)
            pos = torch.tensor([robot_pose[0], robot_pose[1], 0.0], dtype=torch.float32)
            ##print("WORLD PTS MIN/MAX:", world_pts.min(), world_pts.max())
            n_samples = int(self.get_parameter('train_samples').value)
            epoch = int(self.get_parameter('train_epoch').value)
            data = self.region.get_data_from_lidar(
                pts.cpu().numpy(),
                pos.cpu().numpy(),
                device='cuda:0',
                n_samples=n_samples
            )
            if data is None:
                self.get_logger().warn("get_data_from_lidar returned None")
                return
            points, closest = data
            if points.shape[0] == 0:
                self.get_logger().warn("get_data_from_lidar returned 0 points")
                return

            if data is None:
                self.get_logger().warn("get_data_from_lidar returned None")
                return
            x0 = points[:, :3]
            x1 = points[:, 3:6]
            cp0 = closest[:, :3]
            cp1 = closest[:, 3:6]

            raw = torch.cat([x0, x1, cp0, cp1], dim=1)   # Nx12
            #print("RAW SHAPE:", raw.shape)

            if raw.ndim != 2 or raw.shape[1] != 12:
                self.get_logger().error(f"[DBG] raw has wrong shape: {tuple(raw.shape)} (expected Nx12)")
                return
            if not torch.isfinite(raw).all():
                bad = (~torch.isfinite(raw)).sum().item()
                self.get_logger().error(f"[DBG] raw has non-finite values: count={bad}")
                return
            with torch.no_grad():
                mn = raw.min(dim=0).values.cpu().numpy()
                mx = raw.max(dim=0).values.cpu().numpy()
            frame_data = self.region.calculate_speed_and_normal(raw)
            #frame_data = raw

            if frame_data.ndim != 2:
                self.get_logger().error(f"[DBG] frame_data not 2D: {tuple(frame_data.shape)}")
                return
            if frame_data.shape[1] < 12:
                self.get_logger().error(f"[DBG] frame_data has too few cols: {frame_data.shape[1]} (expected >=12)")
                return
            if not torch.isfinite(frame_data).all():
                bad = (~torch.isfinite(frame_data)).sum().item()
                self.get_logger().error(f"[DBG] frame_data non-finite: count={bad}")
                return
            with torch.no_grad():
                fd_min = frame_data.min().item()
                fd_max = frame_data.max().item()

            y0_col = frame_data[:, 6]
            y1_col = frame_data[:, 7]

            if frame_data.shape[0] == 0:
                self.get_logger().warn("No valid frame data after speed calculation")
                return
            self.region.last_robot_pose = np.array([robot_pose[0], robot_pose[1], 0.0], dtype=np.float32)

            if self.region.function is not None:
                try:
                    k = min(64, frame_data.shape[0])
                    xp_n = frame_data[:k, :6].clone().to(self.region.device)
                    with torch.no_grad():
                        tt = self.region.function.TravelTimes(xp_n).detach().cpu().numpy().reshape(-1)
                    with torch.enable_grad():
                        sp_t = self.region.function.Speed(xp_n)  # tensor
                    sp = sp_t.detach().cpu().numpy().reshape(-1)

                except Exception as e:
                    self.get_logger().error(f"[DBG MODEL] failed: {e}")
            #print(f"[FEED] frame_data shape={frame_data.shape}, " #UNCOMMENT FOR DEBUG ON FRAME DATA - CORRECT FOR NOW
            #      f"y0 range=[{frame_data[:,6].min():.4f}, {frame_data[:,6].max():.4f}]")
            #print(f"[FEED] calling train_core with {frame_data.shape[0]} rows")      
            success = self.region.train_core(epoch=epoch, frame_data=frame_data)
            loss_val = getattr(self.region, 'last_loss', None)
            try:
                test = frame_data[:128, :6].clone().to(self.region.device)
                #test.requires_grad_(True)
                test_n = self.region.normalize_tensor(test)
                with torch.enable_grad():
                    tau, w, coords_in = self.region.function.network.out(test_n)
                    tt = tau[:, 0]
                    sp = self.region.function.Speed(test_n)

                g = torch.autograd.grad(tt.sum(), coords_in, retain_graph=True)[0]
                g0 = g[:, :3]  # gradient w.r.t x0
                g0_norm = torch.sqrt((g0 * g0).sum(dim=1) + 1e-12)
            except Exception as e:
                print("MODEL DEBUG FAILED:", e)

        except Exception as e:
            self.get_logger().error(f"feed_region error: {e}")
            import traceback
            traceback.print_exc()

    def pick_frontier_goal(self):
        frontier = self.gridmap.frontier
        iy, ix = np.where(frontier > 0)
        min_frontier_cells = int(self.get_parameter('frontier_min_cells').value)

        if len(ix) < min_frontier_cells:
            return None
        # ✅ DIRECTLY call pick_reachable_frontier (it handles everything)
        goal = self.pick_reachable_frontier()
        if goal is not None:
            robot = self.robot_pos[:2]
            dist = np.linalg.norm(goal - robot)
        return goal

    def pick_room_aware_goal(self, current_room, rooms, room_graph):
        robot_xy = self.robot_pos[:2]

        # 1. Try live local frontier (with blacklist)
        local = self._frontier_in_room(current_room, rooms)
        if local is not None:
            if (len(self._visited_goals) == 0 or
                    np.linalg.norm(np.array(self._visited_goals) - local[None,:], axis=1).min()
                    > self._goal_blacklist_radius):
                return local

        # 2. Try global live frontier
        live = self.pick_reachable_frontier()
        if live is not None:
            return live

        # 3. Local region has no live frontiers → check ExplorationGraph bank
        if self.region is not None:
            ver = self.region.version
            result = self.exploration_graph.best_remote_frontier(ver, robot_xy)
            if result is not None:
                target_ver, target_frontier, door_waypoints = result
                print(f"[EXPLORATION GRAPH] Routing to region {target_ver} "
                      f"via {len(door_waypoints)} door(s), "
                      f"frontier={target_frontier}")
                self._remote_door_waypoints = door_waypoints
                self._remote_target_version = target_ver
                return target_frontier

        print("[EXPLORATION GRAPH] Fully explored — no frontiers anywhere")
        return None

    def _frontier_in_room(self, room_id, rooms):
        if room_id is None:
            return None
        if room_id not in rooms:
            return None
        room_cells = rooms[room_id]["cells"]
        frontier_pts = []
        for (i, j) in room_cells:
            if self.gridmap.frontier[i, j] == 1:
                x, y = self.gridmap._grid_to_world(i, j)
                frontier_pts.append([x, y])
        if len(frontier_pts) == 0:
            return None
        frontier_pts = np.array(frontier_pts)
        # choose closest to robot
        robot = self.robot_pos[:2]
        dists = np.linalg.norm(frontier_pts - robot[None, :], axis=1)
        idx = np.argmin(dists)
        return frontier_pts[idx] #END DIFF

    def publish_field_path(self, path_xy, frame_id='odom'):
        """
        path_xy: list[np.ndarray] or np.ndarray of shape (N,2)
        """
        if path_xy is None:
            return
        if isinstance(path_xy, list):
            pts = np.asarray(path_xy, dtype=np.float32)
        else:
            pts = np.asarray(path_xy, dtype=np.float32)

        if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 2:
            return

        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id

        poses = []
        for p in pts:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = float(p[0])
            ps.pose.position.y = float(p[1])
            ps.pose.position.z = 0.0
            ps.pose.orientation.w = 1.0
            poses.append(ps)

        msg.poses = poses
        self.field_path_pub.publish(msg)

    def decimate_path(self, path_xy, spacing=0.4):
        pts = np.asarray(path_xy, dtype=np.float32)

        if pts.ndim == 1:
            pts = pts.reshape(1, 2)

        if len(pts) < 2:
            return pts

        new_pts = [pts[0]]

        for i in range(1, len(pts)):
            if np.linalg.norm(pts[i] - new_pts[-1]) >= spacing:
                new_pts.append(pts[i])

        if np.linalg.norm(new_pts[-1] - pts[-1]) > 0.05:
            new_pts.append(pts[-1])

        return np.asarray(new_pts, dtype=np.float32)

    def smooth_path(self, path_xy, window=5):
        if path_xy is None:
            return None

        pts = np.asarray(path_xy, dtype=np.float32)

        if pts.ndim == 1:
            pts = pts.reshape(1, 2)

        if len(pts) < window:
            return pts

        sm = pts.copy()
        k = window // 2

        for i in range(len(pts)):
            a = max(0, i-k)
            b = min(len(pts), i+k+1)
            sm[i] = pts[a:b].mean(axis=0)

        sm[0] = pts[0]
        sm[-1] = pts[-1]
        return sm 

    def pick_lookahead(self, path, robot_xy, L=0.45):
        pts = np.asarray(path)
        r = np.asarray(robot_xy)

        d = np.linalg.norm(pts - r[None,:], axis=1)
        i0 = np.argmin(d)

        acc = 0.0
        i = i0
        while i < len(pts)-1 and acc < L:
            acc += np.linalg.norm(pts[i+1]-pts[i])
            i += 1

        return pts[i]
       #end diff

    def is_path_significantly_different(self, old_path, new_path, threshold=0.25):

        old_path = np.asarray(old_path)
        new_path = np.asarray(new_path)

        if len(old_path) < 2 or len(new_path) < 2:
            return True

        # compare up to min length
        n = min(len(old_path), len(new_path))

        # compare corresponding waypoints
        diffs = np.linalg.norm(old_path[:n] - new_path[:n], axis=1)

        max_diff = np.max(diffs)

        print("Max waypoint diff:", max_diff)

        return max_diff > threshold

    def debug_plot(self, field_waypoint=None):
        data = self.gridmap.data
        visited = self.gridmap.visited

        im = np.ones((*data.shape, 3), dtype=np.uint8) * 255
        im[data > 0.9] = np.array([0, 0, 0])      # walls
        im[visited == 1] = np.array([0, 255, 0])  # visited

        fig, ax = plt.subplots(1,1,figsize=(6,6))
        ax.imshow(im, origin='lower')

        # robot
        if self.robot_pos is not None:
            i,j = self.world_to_grid(self.robot_pos[0], self.robot_pos[1])
            ax.scatter(j,i,c='magenta',s=60,label='robot')

        # goal
        if self.goal_xy is not None:
            i,j = self.world_to_grid(self.goal_xy[0], self.goal_xy[1])
            ax.scatter(j,i,c='red',s=60,label='goal')

        # field waypoint
        if field_waypoint is not None:
            i,j = self.world_to_grid(field_waypoint[0], field_waypoint[1])
            ax.scatter(j,i,c='blue',s=60,label='field wp')
        ax.legend()
        plt.pause(0.01)

    def train_tick(self):
        if self.region is None:
            print(f"[TRAIN_TICK] region=False, running={self.train_running}, "
                  f"queue_len={len(self.train_queue)}, epoch=-1")
            return

        print(f"[TRAIN_TICK] region=True, running={self.train_running}, "
              f"queue_len={len(self.train_queue)}, epoch={self.region.epoch}")

        if self.train_running:
            return  # previous train still going

        with self.train_lock:
            if len(self.train_queue) == 0:
                return
            self.train_running = True     
            world_pts, robot_pose = self.train_queue.pop()
            self.train_queue.clear()

        # ✅ Set BEFORE spawning thread — eliminates race condition
       
        def _do_train():
            try:
                print(f"[TRAIN THREAD] starting feed_region, epoch={self.region.epoch}")
                self.feed_region(world_pts, robot_pose)
                print(f"[TRAIN THREAD] done, epoch={self.region.epoch}")
            except Exception as e:
                self.get_logger().error(f"[TRAIN] failed: {e}")
                import traceback
                traceback.print_exc()
            finally:
                self.train_running = False

        import threading
        t = threading.Thread(target=_do_train, daemon=True)
        t.start()

    def maybe_expand_region(self):
        if self.region is None:
            return

        robot = self.robot_pos[:2]
        bound = self.region.bound.numpy()

        margin = 0.3

        near_edge = (
            abs(robot[0] - bound[0,0]) < margin or
            abs(robot[0] - bound[1,0]) < margin or
            abs(robot[1] - bound[0,1]) < margin or
            abs(robot[1] - bound[1,1]) < margin
        )

        if near_edge:
            new_bound = torch.tensor([
                [bound[0,0]-0.3, bound[0,1]-0.3, -0.1],
                [bound[1,0]+0.3, bound[1,1]+0.3,  0.1],
            ], dtype=torch.float32)

            self.region.set_bound(new_bound)
            self.get_logger().info("[REGION EXPAND]")

    def _get_frontier_world_pts(self, max_pts=200):
        """Return frontier cell centres in world coords as (N,2) array."""
        iy, ix = np.where(self.gridmap.frontier > 0)
        if len(ix) == 0:
            return None
        cell   = self.gridmap.cell_size
        origin = self.gridmap.origin
        half   = self.gridmap.map_size / 2.0
        fx = (origin[0] - half) + (ix + 0.5) * cell
        fy = (origin[1] - half) + (iy + 0.5) * cell
        pts = np.stack([fx, fy], axis=1)
        if len(pts) > max_pts:
            pts = pts[np.random.choice(len(pts), max_pts, replace=False)]
        return pts

    def pick_reachable_frontier(self, max_candidates=300):
        frontier = self.gridmap.frontier
        iy, ix = np.where(frontier > 0)
        if len(ix) == 0:
            return None

        cell   = self.gridmap.cell_size
        origin = self.gridmap.origin
        half   = self.gridmap.map_size / 2.0
        fx = (origin[0] - half) + (ix + 0.5) * cell
        fy = (origin[1] - half) + (iy + 0.5) * cell
        cand = np.stack([fx, fy], axis=1)

        robot = self.robot_pos[:2]
        d = np.linalg.norm(cand - robot[None, :], axis=1)
        valid = d < 2.0
        if valid.sum() == 0:
            return None
        cand = cand[valid]
        d    = d[valid]

        # ✅ FIX 1: Only pick frontiers INSIDE the current region bound
        # The field has no knowledge of anything outside the bound
        if self.region is not None:
            in_bound = np.array([self.region.in_bound(c) for c in cand])
            if in_bound.sum() > 0:
                cand = cand[in_bound]
                d    = d[in_bound]
            # If ALL frontiers are outside the bound, fall back to closest-in-bound
            # rather than sending the robot off a cliff
            else:
                return cand[np.argmin(d)]   # geometric fallback

        if len(cand) > max_candidates:
            cand = cand[np.random.choice(len(cand), max_candidates, replace=False)]

        # No field yet → closest
        if self.region is None or self.region.function is None:
            return cand[np.argmin(d)]

        # Field query (all candidates guaranteed inside bound now)
        x0  = np.tile(np.array([robot[0], robot[1], 0.0]), (len(cand), 1))
        x1  = np.concatenate([cand, np.zeros((len(cand), 1))], axis=1)
        xp  = np.concatenate([x0, x1], axis=1)
        xp_t = torch.tensor(xp, dtype=torch.float32)
        xp_n = self.region.normalize_tensor(xp_t).to(self.region.device)

        with torch.no_grad():
            tt = self.region.function.TravelTimes(xp_n).cpu().numpy().reshape(-1)

        ok = np.isfinite(tt)
        if ok.sum() == 0:
            return cand[np.argmin(d)]

        return cand[ok][np.argmin(tt[ok])]

    def maybe_create_new_region(self):
        """
        Called every update() tick.
        If the robot has moved outside the current region's bound, either:
          (a) switch to an existing region that covers the robot, OR
          (b) spawn a new region centred on the robot.
        Old region is unloaded to disk to free GPU memory.
        """
        if self.region is None or self.robot_pos is None:
            return

        robot = self.robot_pos[:2]
        b = self.region.bound.numpy()

        # Is robot still comfortably inside current bound?
        inside = (
            robot[0] > b[0, 0] + REGION_EXIT_MARGIN and
            robot[0] < b[1, 0] - REGION_EXIT_MARGIN and
            robot[1] > b[0, 1] + REGION_EXIT_MARGIN and
            robot[1] < b[1, 1] - REGION_EXIT_MARGIN
        )
        if inside:
            return  # nothing to do

        # --- Robot has left (or is about to leave) the current region ---

        # (a) Check if any existing region already covers the robot
        for ver, reg in self.all_regions.items():
            if ver == self.region.version:
                continue
            if reg.in_bound(robot):
                if reg.loaded:
                    self._switch_region(reg)
                    return
                else:
                    reg.load_from_disk()
                    self._switch_region(reg)
                    return

        # (b) No existing region covers the robot → create a new one
        #     But only if the area has enough visited free space to train on
        visited = self.gridmap.visited
        cell = self.gridmap.cell_size
        origin = self.gridmap.origin
        half = self.gridmap.map_size / 2.0

        # Count visited cells in the proposed new bound
        def count_visited_in_bound(cx, cy, r):
            ix0 = max(0, int((cx - r - (origin[0] - half)) / cell))
            ix1 = min(self.gridmap.n_cells - 1, int((cx + r - (origin[0] - half)) / cell))
            iy0 = max(0, int((cy - r - (origin[1] - half)) / cell))
            iy1 = min(self.gridmap.n_cells - 1, int((cy + r - (origin[1] - half)) / cell))
            region_area = (ix1 - ix0) * (iy1 - iy0)
            if region_area <= 0:
                return 0.0
            visited_count = visited[iy0:iy1, ix0:ix1].sum()
            return float(visited_count) / float(region_area)

        coverage = count_visited_in_bound(robot[0], robot[1], REGION_RADIUS)
        if coverage < REGION_MIN_OVERLAP:
            # Not enough data yet — keep extending current region temporarily
            # by softly expanding its bound rather than spawning premature region
            self._soft_extend_current_region(robot)
            return

        # Enough data: spawn the new region
        self.get_logger().info(
            f"[NEW REGION] Robot at ({robot[0]:.2f},{robot[1]:.2f}) "
            f"left region {self.region.version}. "
            f"Visited coverage={coverage:.2f}. Spawning new region."
        )
        self._unload_current_region()
        self.init_region(self.robot_pos)

    def maybe_merge_regions(self):
        """
        For each confirmed room that spans multiple regions, merge those
        regions into one with a combined bounding box and retrain.
        """
        # Build room → [region_versions] mapping
        room_to_regions: Dict[int, List[int]] = {}
        for ver, region in self.all_regions.items():
            for room in region.rooms:
                if room.id not in room_to_regions:
                    room_to_regions[room.id] = []
                room_to_regions[room.id].append(ver)

        for room_id, versions in room_to_regions.items():
            if len(versions) <= 1:
                continue  # room already fits in one region

            # Only merge if all regions are confirmed (trained enough)
            regions_to_merge = [self.all_regions[v] for v in versions
                                if v in self.all_regions]
            if any(r.epoch < self.warmup_epochs_needed for r in regions_to_merge):
                continue  # wait until all are trained

            self.get_logger().info(
                f"[MERGE] Room {room_id} spans regions {versions} → merging"
            )
            self._merge_regions(regions_to_merge)


    def _merge_regions(self, regions: List):
        """
        Merge a list of regions into one. The merged region gets:
        - Combined bounding box
        - All rooms from all merged regions
        - All training data concatenated
        - Retraining triggered immediately
        """
        # ── Compute combined bound ────────────────────────────────────
        bounds = [r.bound.numpy() for r in regions]
        new_min = np.array([min(b[0, i] for b in bounds) for i in range(3)])
        new_max = np.array([max(b[1, i] for b in bounds) for i in range(3)])
        new_bound = torch.tensor(
            np.stack([new_min, new_max]), dtype=torch.float32
        )

        # ── Check merged region isn't too large ──────────────────────
        w = new_max[0] - new_min[0]
        h = new_max[1] - new_min[1]
        MAX_MERGE_SIZE = REGION_RADIUS * 2 * 2   # 2× normal region size
        if w > MAX_MERGE_SIZE or h > MAX_MERGE_SIZE:
            self.get_logger().warn(
                f"[MERGE SKIPPED] merged bound too large ({w:.1f}×{h:.1f}m)"
            )
            return

        # ── Create merged region ──────────────────────────────────────
        merged = Region(model=self.model, bound=new_bound,
                        init_network=True, eval=False)
        merged.gridmap = self.gridmap

        # Transfer all rooms
        all_rooms = [room for r in regions for room in r.rooms]
        for room in all_rooms:
            merged.add_room(room)   # also sets room.region = merged

        # Concatenate all_lidar from constituent regions
        lidar_chunks = [r.all_lidar for r in regions
                        if r.all_lidar is not None and len(r.all_lidar) > 0]
        if lidar_chunks:
            merged.all_lidar = np.concatenate(lidar_chunks, axis=0)

        # ── Remove old regions from all_regions and ExplorationGraph ─
        old_versions = [r.version for r in regions]
        for ver in old_versions:
            if ver in self.all_regions:
                try:
                    self.all_regions[ver].unload_to_disk()
                except Exception:
                    pass
                del self.all_regions[ver]
            if ver in self.exploration_graph.records:
                del self.exploration_graph.records[ver]
            if ver in self.exploration_graph.graph:
                self.exploration_graph.graph.remove_node(ver)

        # ── Register merged region ────────────────────────────────────
        self.all_regions[merged.version] = merged
        center = np.array(merged.center[:2], dtype=np.float32)
        self.exploration_graph.ensure_region(merged.version, center)

        # Rebuild spatial edges for merged region
        for v2, reg2 in self.all_regions.items():
            if v2 == merged.version:
                continue
            b1 = new_bound.numpy()
            b2 = reg2.bound.numpy()
            x_gap = max(0.0, max(b1[0,0], b2[0,0]) - min(b1[1,0], b2[1,0]))
            y_gap = max(0.0, max(b1[0,1], b2[0,1]) - min(b1[1,1], b2[1,1]))
            if x_gap < 1.0 and y_gap < 1.0:
                mx = (max(b1[0,0], b2[0,0]) + min(b1[1,0], b2[1,0])) / 2.0
                my = (max(b1[0,1], b2[0,1]) + min(b1[1,1], b2[1,1])) / 2.0
                self.exploration_graph.add_spatial_edge(
                    merged.version, v2, np.array([mx, my], dtype=np.float32)
                )

        # ── If robot was in one of the merged regions, switch to merged ─
        if self.region is not None and self.region.version in old_versions:
            self.region = merged
            self.get_logger().info(
                f"[MERGE] Active region was merged → now region {merged.version}"
            )

        # ── Trigger retraining on merged region ──────────────────────
        self.mode               = "WARMUP"
        self.warmup_start_epoch = int(merged.epoch)
        self.goal_xy            = None
        self.nav_state          = "IDLE"
        self.get_logger().info(
            f"[MERGE] Region {merged.version} created "
            f"(merged {old_versions}) → WARMUP"
        )


    #def _switch_region(self, new_region: 'Region'):
    #    old_ver = self.region.version
#
    #    # keep last scans
    #    if self.region is not None and self.region.all_lidar is not None:
    #        new_region.all_lidar = self.region.all_lidar[-5:].copy()
    #        new_region.lidar_obs = None
    #        new_region.kd_tree = None
#
    #    self._unload_current_region()
    #    self.region = new_region
    #    self.get_logger().info(
    #        f"[REGION SWITCH] {old_ver} → {new_region.version}"
    #    )
    #    self.goal_xy = None
    #    self.last_goal_time = self.get_clock().now()
    #    self.gridmap.frontier[:] = 0
    def _switch_region(self, new_region: 'Region'):
        old_ver = self.region.version

        if self.region is not None and self.region.all_lidar is not None:
            new_region.all_lidar = self.region.all_lidar[-5:].copy()
            new_region.lidar_obs = None
            new_region.kd_tree   = None

        self._unload_current_region()

        # Flush stale training frames from the old region before switching
        with self.train_lock:
            self.train_queue.clear()
        self.train_running = False   # abort any in-flight train for old region

        self.region = new_region
        self.get_logger().info(f"[REGION SWITCH] {old_ver} → {new_region.version}")
        self.goal_xy        = None
        self.last_goal_time = self.get_clock().now()
        self.gridmap.frontier[:] = 0


    def _unload_current_region(self):
        """Save current region weights to disk and free GPU memory."""
        if self.region is None:
            return
        try:
            self.region.save_model()          # saves checkpoint to region folder
            self.region.unload_to_disk()      # releases GPU tensors
            self.get_logger().info(
                f"[REGION {self.region.version} UNLOADED]"
            )
        except Exception as e:
            self.get_logger().error(f"[UNLOAD ERROR] {e}")

    def _soft_extend_current_region(self, robot):
        """
        Softly expand region a bit so we don't immediately spawn a new one,
        BUT stop expanding once region gets too large.
        """

        b = self.region.bound.numpy().copy()
        pad = 0.3

        # --- proposed new bound ---
        new_min_x = min(b[0, 0], robot[0] - REGION_RADIUS - pad)
        new_min_y = min(b[0, 1], robot[1] - REGION_RADIUS - pad)
        new_max_x = max(b[1, 0], robot[0] + REGION_RADIUS + pad)
        new_max_y = max(b[1, 1], robot[1] + REGION_RADIUS + pad)

        # --- current size ---
        cur_w = b[1, 0] - b[0, 0]
        cur_h = b[1, 1] - b[0, 1]

        # --- new size if applied ---
        new_w = new_max_x - new_min_x
        new_h = new_max_y - new_min_y

        # 🧠 IMPORTANT: cap region size
        MAX_SIZE = REGION_RADIUS * 2.0   # ≈ 2.5m if radius=1.0

        if new_w > MAX_SIZE or new_h > MAX_SIZE:
            # Stop expanding → allow new region to spawn
            self.get_logger().info(
                f"[SOFT EXTEND BLOCKED] region already large "
                f"(w={cur_w:.2f}, h={cur_h:.2f})"
            )
            return

        # --- apply extension ---
        b[0, 0] = new_min_x
        b[0, 1] = new_min_y
        b[1, 0] = new_max_x
        b[1, 1] = new_max_y

        new_bound = torch.tensor(b, dtype=torch.float32)
        self.region.set_bound(new_bound)

        self.get_logger().info(
            f"[SOFT EXTEND] new bound: "
            f"({b[0,0]:.2f},{b[0,1]:.2f}) → ({b[1,0]:.2f},{b[1,1]:.2f})"
        )

    def _launch_path_gen(self, robot_xy, goal_xy): #claude diff
        """Offload generate_field_path to a background thread so update() never blocks."""
        if robot_xy is None or goal_xy is None:          # ← add this guard
            return
        if self.region is None or self.region.function is None:
            return
        if getattr(self, '_path_gen_running', False):
            return  # already running, skip
        self._path_gen_running = True
        robot_xy = robot_xy.copy()
        goal_xy  = goal_xy.copy()

        def _worker():
            try:
                raw_path = self.region.generate_field_path(
                    robot_xy, goal_xy, step_size=0.10, max_steps=200
                )
                if raw_path is not None:
                    raw_path = np.asarray(raw_path, dtype=np.float32)
                    if len(raw_path) > 1:
                        decimate = self.decimate_path(raw_path, spacing=0.2)
                        smooth   = self.smooth_path(decimate, window=3)
                        self.latest_candidate_path = np.asarray(smooth, dtype=np.float32)
            except Exception as e:
                self.get_logger().warn(f"[PATH GEN] {e}")
            finally:
                self._path_gen_running = False
        threading.Thread(target=_worker, daemon=True).start() #end claude diff

    def _maybe_create_new_region_guarded(self):
        """
        Only allow region creation when robot is NOT actively navigating.
        """
        if self.region is None:
            self.maybe_create_new_region()
            return

        if self.mode != "EXPLORE":
            self.maybe_create_new_region()
            return

        if self.goal_xy is None:
            self.maybe_create_new_region()
            return

        # Actively navigating → block
        if self.nav_state in ("STABILIZING", "FOLLOWING"):
            robot_xy = self.robot_pos[:2]
            d = np.linalg.norm(robot_xy - self.goal_xy)
            if d < 0.20:
                self.maybe_create_new_region()
            else:
                return  # still en-route, block creation
        else:
            # IDLE or DWELL → both are safe, robot is stationary
            self.maybe_create_new_region()

    def _update_region_graph_via_free_cells(self):
        """
        Build/update ExplorationGraph edges between regions using cells that:
          - are free/traversable (gridmap.data == 0, not occupied)
          - have been visited by the robot (gridmap.visited > 0)
          - fall within the discovered area of BOTH regions simultaneously
            (discovery_map has bits set for both region versions)

        This replaces boundary-midpoint and door-based spatial edges entirely.
        """
        versions = list(self.all_regions.keys())
        if len(versions) < 2:
            return

        # Build a mask of cells that are free AND visited
        free_visited = (self.gridmap.data == 0) & (self.gridmap.visited > 0)
        if not free_visited.any():
            return

        cell_indices = np.argwhere(free_visited)  # Nx2 array of (i, j)
        disc = self.gridmap.discovery_map  # uint64 bitmask per cell

        for v1, v2 in combinations(versions, 2):
            bit1 = np.uint64(1) << np.uint64(v1)
            bit2 = np.uint64(1) << np.uint64(v2)

            # Cells visited by BOTH regions
            shared_mask = free_visited & \
                          ((disc & bit1) != 0) & \
                          ((disc & bit2) != 0)

            shared_cells = np.argwhere(shared_mask)
            if len(shared_cells) < 3:   # need at least a few shared cells
                continue

            # Convert a representative subset to world coords as waypoints
            # Use up to 5 spread-out points via simple stride sampling
            stride = max(1, len(shared_cells) // 5)
            sampled = shared_cells[::stride][:5]
            waypoints = [
                np.array(self.gridmap._grid_to_world(int(ci), int(cj)))
                for ci, cj in sampled
            ]

            self.exploration_graph.add_free_cell_edge(v1, v2, waypoints)

    def _assign_frontiers_to_regions(self, current_version: int):
        """
        Per-cell frontier ownership:
        - Cell discovered only by current region → current region.
        - Cell discovered only by one remote (non-exhausted) region → that region.
        - Cell in overlap with current region → current region (always).
        - Cell in overlap not including current, or remote is exhausted → current region.
        """
        frontier_cells = np.argwhere(self.gridmap.frontier > 0)
        if len(frontier_cells) == 0:
            return

        disc = self.gridmap.discovery_map

        exclusive: dict = {}   # {version: [np.ndarray, ...]}
        current_pts = []

        for ci, cj in frontier_cells:
            cell_bits = disc[ci, cj]
            versions_here = self.gridmap.discover_regions(cell_bits)

            wx, wy = self.gridmap._grid_to_world(int(ci), int(cj))
            pt = np.array([wx, wy], dtype=np.float32)

            if current_version in versions_here or len(versions_here) == 0:
                # Belongs to current (own cell, overlap with current, or undiscovered)
                current_pts.append(pt)
            elif len(versions_here) == 1:
                v = versions_here[0]
                rec = self.exploration_graph.records.get(v)
                if rec is not None and not rec.is_exhausted():
                    exclusive.setdefault(v, []).append(pt)
                else:
                    current_pts.append(pt)   # exhausted remote → current takes it
            else:
                # Multi-region overlap not including current → current takes it
                current_pts.append(pt)

        # Assign exclusive frontiers to non-exhausted remote regions
        for version, pts in exclusive.items():
            region_obj = self.all_regions.get(version)
            if region_obj is None:
                continue
            # Safe center extraction whether bound is a tensor or ndarray
            b = region_obj.bound
            if hasattr(b, 'cpu'):
                b = b.cpu().numpy()
            center = b[:2].mean(axis=0)[:2].astype(np.float32)
            self.exploration_graph.ensure_region(version, center)
            for pt in pts:
                self.exploration_graph.records[version].add_frontier(
                    pt, ExplorationGraph.BLACKLIST_RADIUS
                )

        # All current-region and overlap frontiers → current region
        for pt in current_pts:
            self.exploration_graph.records[current_version].add_frontier(
                pt, ExplorationGraph.BLACKLIST_RADIUS
            )

    def _build_is_frontier_fn(self):
        """Returns a closure over the current frontier grid for bank reconciliation."""
        frontier_cells = np.argwhere(self.gridmap.frontier > 0)
        if len(frontier_cells) == 0:
            def is_frontier_fn(xy, radius):
                return False
            return is_frontier_fn

        from scipy.spatial import cKDTree
        # World coords of all live frontier cells
        world_pts = np.array([
            self.gridmap._grid_to_world(int(ci), int(cj))
            for ci, cj in frontier_cells
        ])
        tree = cKDTree(world_pts)

        def is_frontier_fn(xy: np.ndarray, radius: float) -> bool:
            hits = tree.query_ball_point(xy[:2], r=radius)
            return len(hits) > 0
        return is_frontier_fn

    def _start_goal_dwell(self):
        dwell_frames_needed = int(self.get_parameter('dwell_frames_needed').value)
        if self.region is None or dwell_frames_needed <= 0:
            self.goal_xy   = None
            self.nav_state = "IDLE"
            return

        self.dwell_active       = True
        self.dwell_frames       = []
        self.dwell_frames_count = 0
        self.dwell_phase        = "COLLECT"
        self.nav_state          = "DWELL"
        self.goal_xy            = None
        self.navigator.stop_robot()

        # Flush any in-flight frames from navigation so we get a clean epoch baseline
        with self.train_lock:
            self.train_queue.clear()
        # Epoch baseline set AFTER flush so we don't count pre-dwell epochs
        self.dwell_epoch_start = int(self.region.epoch)

        print(f"[DWELL] Starting — collecting {dwell_frames_needed} frames "
              f"then training {int(self.get_parameter('dwell_epochs').value)} epochs "
              f"(baseline epoch={self.dwell_epoch_start})")

    def _tick_goal_dwell(self, world_points_360: np.ndarray, robot_xy: np.ndarray):
        dwell_frames_needed = int(self.get_parameter('dwell_frames_needed').value)
        dwell_epochs        = int(self.get_parameter('dwell_epochs').value)

        # If region changed under us (new region created), abort dwell cleanly
        if self.region is not None and int(self.region.epoch) < self.dwell_epoch_start:
            print("[DWELL] Region switched during dwell — ending dwell, entering WARMUP")
            self._end_goal_dwell()
            # Trigger warmup for the new region
            self.mode               = "WARMUP"
            self.warmup_start_epoch = int(self.region.epoch)
            self.warmup_done        = False
            return

        if self.dwell_phase == "COLLECT":
            if self.region is not None and len(world_points_360) > 5:
                self.dwell_frames.append((
                    world_points_360.copy(),
                    self.robot_pos.copy() if self.robot_pos is not None else robot_xy
                ))
                self.dwell_frames_count += 1
                print(f"[DWELL] Collected frame {self.dwell_frames_count}/{dwell_frames_needed}")

            if self.dwell_frames_count >= dwell_frames_needed:
                with self.train_lock:
                    for frame in self.dwell_frames:
                        self.train_queue.append(frame)
                self.dwell_epoch_start = int(self.region.epoch)
                self.dwell_phase       = "TRAIN"
                print(f"[DWELL] Collection done — training for {dwell_epochs} epochs "
                      f"from epoch {self.dwell_epoch_start}")

        elif self.dwell_phase == "TRAIN":
            if self.region is None:
                self._end_goal_dwell()
                return
            epochs_done = int(self.region.epoch) - self.dwell_epoch_start
            print(f"[DWELL] Training epoch {epochs_done}/{dwell_epochs}")
            if epochs_done >= dwell_epochs:
                print(f"[DWELL] Training complete — resuming navigation")
                self._end_goal_dwell()


    def _end_goal_dwell(self):
        """Clean up dwell state and return to normal navigation."""
        self.dwell_active  = False
        self.dwell_frames  = []
        self.dwell_phase   = "COLLECT"
        self.nav_state     = "IDLE"
        self.goal_xy       = None

    def update(self):
        now = self.get_clock().now()

        # ── Prerequisites ────────────────────────────────────────────────
        if self.robot_pos is None or self.latest_cloud is None:
            return

        result = self.adapter.update(self.latest_cloud, self.robot_pos)
        if result is None:
            return
        points_lidar, speeds = result
        if len(points_lidar) == 0:
            return

        # ── TF transform ─────────────────────────────────────────────────
        try:
            source_frame = self.latest_cloud.header.frame_id
            stamp        = self.latest_cloud.header.stamp
            if self.tf_buffer.can_transform('odom', source_frame, stamp,
                                            timeout=Duration(seconds=0.1)):
                transform = self.tf_buffer.lookup_transform('odom', source_frame, stamp)
            else:
                return
            t = transform.transform.translation
            q = transform.transform.rotation
            R = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
            world_points = (R @ points_lidar.T).T + np.array([t.x, t.y, t.z])
        except Exception as e:
            self.get_logger().warn(f"TF lookup failed: {e}")
            return

        # ── Bound filter ──────────────────────────────────────────────────
        robot_xy = self.robot_pos[:2]
        half = self.gridmap.map_size / 2
        ox, oy = self.gridmap.origin
        in_bounds = ((world_points[:, 0] > ox - half) & (world_points[:, 0] < ox + half) &
                     (world_points[:, 1] > oy - half) & (world_points[:, 1] < oy + half))
        world_points_filtered = world_points[in_bounds]
        speeds_filtered = speeds[in_bounds]
        if len(world_points_filtered) < 5:
            return

        # Angular downsampling + free-ray synthesis
        # For each of 360 angular bins:
        #   - Real hit bin  → closest point from sensor; hit_dist = measured dist
        #   - Empty bin     → synthetic endpoint at MAX_RANGE; hit_dist = inf
        # The raycast loop later uses hit_dist to decide whether to mark a wall.
        robot = robot_xy
        vecs   = world_points_filtered[:, :2] - robot_xy[None, :]
        dists  = np.linalg.norm(vecs, axis=1) + 1e-9
        angles = np.arctan2(vecs[:, 1], vecs[:, 0])
        bins   = ((angles + np.pi) / (2 * np.pi) * 359).astype(np.int32)

        closest_idx  = {}
        closest_dist = np.full(360, np.inf, dtype=np.float32)
        for k, b in enumerate(bins):
            if dists[k] < closest_dist[b]:
                closest_dist[b] = dists[k]
                closest_idx[b]  = k

        # Real-hit cloud — identical to the original method; used for raycast + gridmap
        keep = np.array(list(closest_idx.values()), dtype=np.int32)
        world_points_hit = world_points_filtered[keep]
        speeds_hit       = speeds_filtered[keep]

        # Full 360-bin cloud — real hits + synthetic free-ray endpoints
        # Used ONLY for training data collection; never touches gridmap/raycast.
        full_endpoints  = []
        full_speeds     = []
        for b in range(360):
            if b in closest_idx:
                idx = closest_idx[b]
                full_endpoints.append(world_points_filtered[idx])
                full_speeds.append(speeds_filtered[idx])
            else:
                theta  = (b / 359.0) * 2.0 * np.pi - np.pi
                dir_xy = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
                free_xy = robot_xy + MAX_RANGE * dir_xy
                free_z  = float(self.robot_pos[2]) if len(self.robot_pos) > 2 else 0.0
                full_endpoints.append(np.array([free_xy[0], free_xy[1], free_z], dtype=np.float32))
                full_speeds.append(0.0)
        world_points_360 = np.asarray(full_endpoints, dtype=np.float32)

        # ── Raycast: update gridmap.data + visited (real hits only) ──────
        cell   = self.gridmap.cell_size
        origin = self.gridmap.origin
        half_  = self.gridmap.map_size / 2

        def world_to_ij(x, y):
            return (int((x - origin[0] + half_) / cell),
                    int((y - origin[1] + half_) / cell))

        for pt3d in world_points_hit:
            pt = pt3d[:2]
            dx, dy = pt[0] - robot_xy[0], pt[1] - robot_xy[1]
            dist = float(np.hypot(dx, dy))
            if dist > MAX_RANGE:
                scale = MAX_RANGE / dist
                dx *= scale; dy *= scale; dist = MAX_RANGE
            steps = int(dist / cell)
            if steps < 1:
                ix, iy = world_to_ij(pt[0], pt[1])
                if 0 <= ix < self.gridmap.n_cells and 0 <= iy < self.gridmap.n_cells:
                    self.gridmap.data[iy, ix] = 1
                    self.gridmap.visited[iy, ix] = 1
                continue
            for s in range(steps - 1):
                x = robot_xy[0] + dx * s / steps
                y = robot_xy[1] + dy * s / steps
                ix, iy = world_to_ij(x, y)
                if not (0 <= ix < self.gridmap.n_cells and
                        0 <= iy < self.gridmap.n_cells):
                    break
                if self.gridmap.data[iy, ix] == 1:
                    break
                self.gridmap.visited[iy, ix] = 1
            ix_end, iy_end = world_to_ij(pt[0], pt[1])
            if (0 <= ix_end < self.gridmap.n_cells and
                    0 <= iy_end < self.gridmap.n_cells):
                self.gridmap.data[iy_end, ix_end] = 1
                self.gridmap.visited[iy_end, ix_end] = 1

        # ── Gridmap update (1 Hz room segmentation) ──────────────────────
        self._map_update_counter += 1
        do_room_update = (self._map_update_counter % 5 == 0)
        self.gridmap.update(
            lidar_scan=world_points_hit[:, :2],
            points=world_points_hit[:, :2],
            speeds=speeds_hit,
            cur_pos=robot_xy,
            region=self.region,
            regions=self.all_regions,
            update_room=do_room_update
        )

        # ── Snapshot live frontiers into ExplorationGraph ────────────────
        if self.region is not None:
            ver = self.region.version
            center = np.array(self.region.center[:2], dtype=np.float32)
            self.exploration_graph.ensure_region(ver, center)
            self._assign_frontiers_to_regions(ver)
            # Collect live frontier points for this region
            #iy, ix = np.where(self.gridmap.frontier > 0)
            #if len(ix) > 0:
            #    cell   = self.gridmap.cell_size
            #    origin = self.gridmap.origin
            #    half   = self.gridmap.map_size / 2.0
            #    fx = (origin[0] - half) + (ix + 0.5) * cell
            #    fy = (origin[1] - half) + (iy + 0.5) * cell
            #    live_frontiers = np.stack([fx, fy], axis=1)
            #    # Downsample to avoid flooding the bank
            #    if len(live_frontiers) > 50:
            #        live_frontiers = live_frontiers[
            #            np.random.choice(len(live_frontiers), 50, replace=False)
            #        ]
            #    self.exploration_graph.snapshot_frontiers(ver, live_frontiers)
        is_frontier_fn = self._build_is_frontier_fn()
        self.exploration_graph.reconcile_all_banks(is_frontier_fn, evict_radius=0.6)

        if do_room_update:
            for u, v, edge_data in self.gridmap.room_graph.edges(data=True):
                room_u = self.gridmap.room_graph.nodes[u].get('room', -1)
                room_v = self.gridmap.room_graph.nodes[v].get('room', -1)

                # Only register edges between two confirmed rooms
                if room_u not in self.gridmap.confirmed_rooms:
                    continue
                if room_v not in self.gridmap.confirmed_rooms:
                    continue

                pos_u = self.gridmap.room_graph.nodes[u].get('pos')
                pos_v = self.gridmap.room_graph.nodes[v].get('pos')
                if pos_u is None or pos_v is None:
                    continue

                door_xy = np.array([(pos_u[0]+pos_v[0])/2,
                                     (pos_u[1]+pos_v[1])/2], dtype=np.float32)

                reg_u = getattr(self.gridmap.confirmed_rooms[room_u], 'region', None)
                reg_v = getattr(self.gridmap.confirmed_rooms[room_v], 'region', None)
                if reg_u is None or reg_v is None:
                    continue
                if reg_u.version == reg_v.version:
                    continue   # same region, spatial edge already covers this

                self.exploration_graph.add_door_edge(reg_u.version, reg_v.version, door_xy)
                self._update_region_graph_via_free_cells()
#            # ── Spatial adjacency edges (intra-room, no door needed) ─────
#            for v1, reg1 in self.all_regions.items():
#                for v2, reg2 in self.all_regions.items():
#                    if v1 >= v2:
#                        continue
#                    #if self.exploration_graph.graph.has_edge(v1, v2):
#                    #    continue  # door edge already exists, skip
#
#                    b1 = reg1.bound.numpy()
#                    b2 = reg2.bound.numpy()
#
#                    # Gap between the two bounding boxes in x and y
#                    x_gap = max(0.0, max(b1[0,0], b2[0,0]) - min(b1[1,0], b2[1,0]))
#                    y_gap = max(0.0, max(b1[0,1], b2[0,1]) - min(b1[1,1], b2[1,1]))
#
#                    if x_gap < 1.0 and y_gap < 1.0:
#                        # Midpoint of the shared/nearest boundary zone
#                        mx = (max(b1[0,0], b2[0,0]) + min(b1[1,0], b2[1,0])) / 2.0
#                        my = (max(b1[0,1], b2[0,1]) + min(b1[1,1], b2[1,1])) / 2.0
#                        midpoint = np.array([mx, my], dtype=np.float32)
#                        #self.exploration_graph.add_spatial_edge(v1, v2, midpoint)
                #self._update_region_graph_via_free_cells()

        # ── Enqueue training frame (ALL modes) ───────────────────────────
        if self.region is not None and bool(self.get_parameter('train_after_region').value):
            self.train_counter += 1
            if self.train_counter % self.train_every_n_frames == 0:
                with self.train_lock:
                    self.train_queue.append((
                        world_points_360.copy(),
                        self.robot_pos.copy()
                    ))

        # ── Region boundary check ─────────────────────────────────────────
        self._maybe_create_new_region_guarded()
        if self._map_update_counter % 10 == 0:
            self.maybe_merge_regions()
        # ── Trajectory ───────────────────────────────────────────────────
        if self.robot_pos is not None:
            self._traj_history.append(self.robot_pos[:2].copy())

        # ── Mode state machine (navigation only, map already updated above) ──

        if self.mode == "BOOTSTRAP":
            self.navigator.stop_robot()
            self.bootstrap_frames += 1
            if self.region is None and self.bootstrap_frames >= self.bootstrap_threshold:
                self.get_logger().info("Initializing first region")
                self.init_region(self.robot_pos)
                self.mode              = "WARMUP"
                self.warmup_start_epoch = int(self.region.epoch)
                self.goal_xy           = None
                self.last_goal_time    = now
                self.get_logger().info("[FLOW] Entering WARMUP phase")

        elif self.mode == "WARMUP":
            # Stop robot, check epochs — but do NOT return; fall through to publish
            self.navigator.stop_robot()
            if self.region is not None:
                cur_epochs = int(self.region.epoch) - int(self.warmup_start_epoch)
                print(f"[WARMUP] epoch {cur_epochs}/{self.warmup_epochs_needed}")
                if cur_epochs >= self.warmup_epochs_needed:
                    self.mode        = "EXPLORE"
                    self.warmup_done = True
                    self.goal_xy     = None
                    self.last_goal_time = now
                    self.get_logger().info("[FLOW] Warmup complete → starting EXPLORE")

        elif self.mode == "EXPLORE":
            robot_xy = np.asarray(self.robot_pos[:2], dtype=np.float32)
            now_sec  = now.nanoseconds * 1e-9

            # ── IDLE: pick a new frontier goal ───────────────────────────
            if self.nav_state == "IDLE":
                current_room = self.gridmap.get_room_at_position(robot_xy)
                rooms        = self.gridmap.get_rooms()
                room_graph   = self.gridmap.get_room_graph()
                g = self.pick_room_aware_goal(current_room, rooms, room_graph)

                if g is None:
                    self.navigator.stop_robot()
                    print("[NO FRONTIER]")
                else:
                    self.goal_xy           = np.asarray(g, dtype=np.float32)
                    self.wait_start_time   = now_sec
                    self._path_gen_running = False
                    print("[GOAL SET]", self.goal_xy)

                    # ── If ExplorationGraph gave us door waypoints, pre-load them ─
                    if self._remote_door_waypoints:
                        print(f"[EXPLORATION GRAPH] Using {len(self._remote_door_waypoints)}"
                              f" door waypoints → region {self._remote_target_version}")
                        # Build composite path: door chain + final frontier goal
                        full_wp = np.array(
                            self._remote_door_waypoints + [self.goal_xy],
                            dtype=np.float32
                        )
                        self.latest_candidate_path = full_wp
                        self._remote_door_waypoints = []   # consumed
                    else:
                        self.latest_candidate_path = None  # normal: path gen will fill this

                    self.nav_state = "STABILIZING"
            # ── STABILIZING: generate path in background, don't block ────
            elif self.nav_state == "STABILIZING":
                self.navigator.stop_robot()
                # kick off path gen if not already running
                self._launch_path_gen(robot_xy, self.goal_xy)

                if now_sec - self.wait_start_time >= self.wait_duration:
                    if self.latest_candidate_path is not None:
                        print("[PATH FROZEN]")
                        self.path = self.latest_candidate_path.copy()
                        self.navigator.reset_path()
                        self.nav_state = "FOLLOWING"
                    else:
                        print("[NO PATH FOUND — retrying]")
                        self.wait_start_time = now_sec  # extend wait

            # ── FOLLOWING: drive, continuously refresh path in background ─
            elif self.nav_state == "FOLLOWING":
                if self.goal_xy is None:
                    self.nav_state = "IDLE"
                    self.path = None
                    return
                if self.goal_xy is not None:
                    d_goal = float(np.linalg.norm(robot_xy - self.goal_xy))
                else:
                    d_goal = None
                #print(f"Distance to frontier: {d_goal:.2f}")
                if d_goal is None:
                    self.nav_state = "IDLE"
                    self.goal_xy = None
                    self.path = None 

                if d_goal is not None and d_goal < 0.25:
                    print("[FRONTIER REACHED]")
                    self.navigator.stop_robot()
                    self.navigator.reset_path()

                    if self.goal_xy is not None:
                        # ── Clear frontier cells around reached goal ──────────────
                        gix = int((self.goal_xy[0] - (self.gridmap.origin[0] - self.gridmap.map_size/2))
                                  / self.gridmap.cell_size)
                        giy = int((self.goal_xy[1] - (self.gridmap.origin[1] - self.gridmap.map_size/2))
                                  / self.gridmap.cell_size)
                        pad = 3
                        for di in range(-pad, pad+1):
                            for dj in range(-pad, pad+1):
                                ii, jj = giy+di, gix+dj
                                if 0 <= ii < self.gridmap.n_cells and 0 <= jj < self.gridmap.n_cells:
                                    self.gridmap.frontier[ii, jj] = 0
                                    self.gridmap.visited[ii, jj]  = 1

                        # ── Update visited blacklist ──────────────────────────────
                        self._visited_goals.append(self.goal_xy.copy())
                        if len(self._visited_goals) > 50:
                            self._visited_goals.pop(0)

                        # ── Notify ExplorationGraph ───────────────────────────────
                        if self.region is not None:
                            self.exploration_graph.mark_goal_reached(
                                self.region.version, self.goal_xy
                            )

                    #self.goal_xy   = None
                    self.path      = None
                    self._remote_door_waypoints = []       # clear any leftover door chain
                    self._remote_target_version = None
                    self._start_goal_dwell()
                    if not self.dwell_active:
                        self.goal_xy = None
                        self.nav_state = "IDLE"

                else:
                    if self.goal_xy is None:       # ← double-check before path gen
                        self.nav_state = "IDLE"
                        return
                    # Continuously refresh the path as the network improves
                    self._launch_path_gen(robot_xy, self.goal_xy)
                    if self.latest_candidate_path is not None:
                        self.path                  = self.latest_candidate_path
                        self.latest_candidate_path = None
                        self.navigator.reset_path()
                    if self.path is not None:
                        self.navigator.drive_path(self.path)
            
            elif self.nav_state == "DWELL":
                self.navigator.stop_robot()
                self._tick_goal_dwell(world_points_360, robot_xy)

        # ── Composite plot ────────────────────────────────────────────────
        if self._composite_step % 20 == 0:
            try:
                self.save_composite_plot()
                self.save_occupancy_snapshot()
            except Exception as e:
                self.get_logger().warn(f"[COMPOSITE PLOT] {e}")
        self._composite_step += 1

        # ── Always publish (every tick, every mode) ───────────────────────
        self.map_publisher.publish_maps()
        self.markers.publish_goal_marker()
        self.markers.publish_frontier_markers()
        self.publish_field_path(self.path)

    def save_composite_plot(self):
        """
        plot2-style composite:
          - world-scale background (unknown / visited / obstacles)
          - per-region travel-time contours that PERSIST via a numeric cache (no translucent overlays)
          - exploration graph (nodes/edges/doors)
          - ONLY the latest saved frontiers per region (not the entire bank)
        """
        from matplotlib.lines import Line2D
        os.makedirs(self.folder, exist_ok=True)
        out_dir = os.path.join(self.folder, "plots2")
        os.makedirs(out_dir, exist_ok=True)

        # World-map bounds (gridmap is world-sized)
        b_map = [
            [self.gridmap.origin[0] - self.gridmap.map_size / 2,
             self.gridmap.origin[1] - self.gridmap.map_size / 2],
            [self.gridmap.origin[0] + self.gridmap.map_size / 2,
             self.gridmap.origin[1] + self.gridmap.map_size / 2],
        ]

        fig, ax = plt.subplots(1, 1, figsize=(14, 14))

        # ── Background: visited=dark green, obstacle=black, unknown=dark blue ──
        data = self.gridmap.data
        vis  = self.gridmap.visited
        n    = self.gridmap.n_cells
        cell = self.gridmap.cell_size

        bg = np.zeros((n, n, 3), dtype=np.uint8)
        bg[:, :] = [10, 10, 80]        # unknown
        bg[vis == 1]  = [220, 220, 40]     # visited free
        bg[data == 1] = [0, 0, 0]      # obstacles

        extent = [b_map[0][0], b_map[1][0], b_map[0][1], b_map[1][1]]
        ax.imshow(bg, origin='lower', extent=extent, zorder=0, aspect='auto')

        # ── Persistent numeric cache for region fields ─────────────────────────
        # Cache: version -> dict(X_w, Y_w, TT_masked, extent, src_world, step, epoch)
        if not hasattr(self, "_region_field_cache"):
            self._region_field_cache = {}

        # Tuning
        GRID_N = 90                 # resolution of per-region TT grid
        CONTOUR_LEVELS = 40         # number of contour levels
        FRONTIER_LAST_K = 30        # only show last K saved frontiers per region
        CACHE_UPDATE_EVERY = 1      # update active region every step (keep expanding field visible)

        def _build_region_mask(b, grid_n=GRID_N):
            """Mask out cells outside visited/obstacle footprint (same logic as your original)."""
            mask = np.zeros((grid_n, grid_n), dtype=bool)
            for i in range(grid_n):
                for j in range(grid_n):
                    wx = b[0, 0] + (i + 0.5) / grid_n * (b[1, 0] - b[0, 0])
                    wy = b[0, 1] + (j + 0.5) / grid_n * (b[1, 1] - b[0, 1])
                    gx = int((wx - b_map[0][0]) / cell)
                    gy = int((wy - b_map[0][1]) / cell)
                    if 0 <= gx < n and 0 <= gy < n:
                        mask[j, i] = (vis[gy, gx] == 1 or data[gy, gx] == 1)
            return mask

        def _compute_tt_grid(region, src_world, b, mask, grid_n=GRID_N):
            """Compute TT(x0=src, x1=grid) over region bound and apply mask => TT_masked."""
            xmin, ymin = float(b[0, 0]), float(b[0, 1])
            xmax, ymax = float(b[1, 0]), float(b[1, 1])

            xs = np.linspace(xmin, xmax, grid_n)
            ys = np.linspace(ymin, ymax, grid_n)
            X_w, Y_w = np.meshgrid(xs, ys)

            # world -> normalized using region bound
            sx = (xmax - xmin) + 1e-9
            sy = (ymax - ymin) + 1e-9
            X_n = (X_w - xmin) / sx - 0.5
            Y_n = (Y_w - ymin) / sy - 0.5

            src_norm = region.normalize(src_world.reshape(1, 3))[0]  # (3,)

            XP = np.zeros((X_w.size, 2 * region.dim), dtype=np.float32)
            XP[:, :region.dim] = src_norm
            XP[:, region.dim + 0] = X_n.reshape(-1)
            XP[:, region.dim + 1] = Y_n.reshape(-1)
            XP[:, region.dim + 2] = src_norm[2]

            XP_t = torch.tensor(XP, dtype=torch.float32, device=region.device)

            with torch.no_grad():
                TT_t = region.function.TravelTimes(XP_t)
                if TT_t.ndim == 2 and TT_t.shape[1] == 1:
                    TT_t = TT_t[:, 0]
                TT = TT_t.detach().cpu().numpy().reshape(X_w.shape)

            TT_masked = TT.copy()
            TT_masked[~mask] = np.nan

            return X_w, Y_w, TT_masked

        def _pick_src_world(region, b):
            """Stable source for plotting TT contours."""
            # Active region uses robot position if available
            if self.region is not None and region.version == self.region.version and self.robot_pos is not None:
                return np.array([self.robot_pos[0], self.robot_pos[1], 0.0], dtype=np.float32)

            # Inactive: prefer cached source if present, else region center
            cached = self._region_field_cache.get(region.version)
            if cached is not None and "src_world" in cached:
                return cached["src_world"]

            rc = (b[0] + b[1]) / 2.0
            return np.array([float(rc[0]), float(rc[1]), 0.0], dtype=np.float32)

        # ── Update cache for the active region (so its field "expands" live) ──
        if self.region is not None and (self._composite_step % CACHE_UPDATE_EVERY == 0):
            try:
                reg = self.region
                if reg.function is not None:
                    b = reg.bound.detach().cpu().numpy()
                    mask = _build_region_mask(b, grid_n=GRID_N)
                    src_world = _pick_src_world(reg, b)
                    X_w, Y_w, TT_masked = _compute_tt_grid(reg, src_world, b, mask, grid_n=GRID_N)
                    self._region_field_cache[reg.version] = dict(
                        X_w=X_w, Y_w=Y_w, TT=TT_masked,
                        extent=[b[0, 0], b[1, 0], b[0, 1], b[1, 1]],
                        mask=mask, src_world=src_world,
                        step=int(self._composite_step),
                        epoch=int(getattr(reg, "epoch", 0)),
                    )
            except Exception as e:
                self.get_logger().warn(f"[COMPOSITE] active region cache update failed: {e}")

        # ── Ensure each region has at least one cached snapshot (persist) ─────
        for region in self.all_regions.values():
            if region.version in self._region_field_cache:
                continue
            # Only cache if model exists right now; otherwise it will remain missing until loaded once
            if region.function is None:
                continue
            try:
                b = region.bound.detach().cpu().numpy()
                mask = _build_region_mask(b, grid_n=GRID_N)
                src_world = _pick_src_world(region, b)
                X_w, Y_w, TT_masked = _compute_tt_grid(region, src_world, b, mask, grid_n=GRID_N)
                self._region_field_cache[region.version] = dict(
                    X_w=X_w, Y_w=Y_w, TT=TT_masked,
                    extent=[b[0, 0], b[1, 0], b[0, 1], b[1, 1]],
                    mask=mask, src_world=src_world,
                    step=int(self._composite_step),
                    epoch=int(getattr(region, "epoch", 0)),
                )
            except Exception as e:
                self.get_logger().warn(f"[COMPOSITE] region {region.version} initial cache failed: {e}")

        # ── Draw persistent contours for all cached regions ───────────────────
        for region in self.all_regions.values():
            b = region.bound.detach().cpu().numpy()

            cached = self._region_field_cache.get(region.version, None)
            if cached is not None:
                X_w = cached["X_w"]
                Y_w = cached["Y_w"]
                TT  = cached["TT"]

                if np.isfinite(TT).any():
                    ax.contour(
                        X_w, Y_w, TT,
                        levels=CONTOUR_LEVELS,
                        cmap="gist_heat",
                        linewidths=0.65,
                        alpha=0.9,
                        zorder=2
                    )

            # ── Region bounding box (always) ─────────────────────────────────
            rect_w = float(b[1, 0] - b[0, 0])
            rect_h = float(b[1, 1] - b[0, 1])
            is_active = (self.region is not None and region.version == self.region.version)
            edge_color = 'yellow' if is_active else 'gray'
            lw = 2.0 if is_active else 0.8

            rect = plt.Rectangle(
                (float(b[0, 0]), float(b[0, 1])), rect_w, rect_h,
                linewidth=lw, edgecolor=edge_color, facecolor='none', zorder=4
            )
            ax.add_patch(rect)
            ax.text(
                float(b[0, 0]) + 0.02, float(b[1, 1]) - 0.15,
                f"R{region.version}{'*' if is_active else ''}",
                color=edge_color, fontsize=7, zorder=5
            )

        # ── ExplorationGraph overlay: edges + door waypoints ─────────────────
        for v1, v2, edata in self.exploration_graph.graph.edges(data=True):
            rec1 = self.exploration_graph.records.get(v1)
            rec2 = self.exploration_graph.records.get(v2)
            if rec1 is None or rec2 is None:
                continue

            c1 = rec1.center
            c2 = rec2.center

            door_xys = edata.get('door_xys', [edata.get('door_xy')])
            door_xys = [d for d in door_xys if d is not None]

            if door_xys:
                for door_xy in door_xys:
                    xs = [c1[0], door_xy[0], c2[0]]
                    ys = [c1[1], door_xy[1], c2[1]]
                    ax.plot(xs, ys, color='cyan', linewidth=1.2,
                            linestyle='--', alpha=0.7, zorder=6)
                    ax.scatter(door_xy[0], door_xy[1],
                               c='cyan', s=40, marker='D',
                               zorder=7, alpha=0.9)
            else:
                ax.plot([c1[0], c2[0]], [c1[1], c2[1]],
                        color='cyan', linewidth=1.0,
                        linestyle='--', alpha=0.6, zorder=6)

        # ── ExplorationGraph nodes (region centers) ──────────────────────────
        for ver, rec in self.exploration_graph.records.items():
            c = rec.center
            has_f = rec.has_frontiers()
            node_color = 'lime' if has_f else 'red'
            ax.scatter(c[0], c[1], c=node_color, s=120,
                       marker='o', zorder=8, edgecolors='white', linewidths=1.0)
            ax.text(c[0] + 0.05, c[1] + 0.05,
                    f"R{ver}\n({len(rec.saved_frontiers)}f)",
                    color='white', fontsize=6, zorder=9,
                    bbox=dict(boxstyle='round,pad=0.15', fc='black', alpha=0.5))

        # ── Latest saved frontier points per region (NOT all) ─────────────────
        for ver, rec in self.exploration_graph.records.items():
            if not rec.saved_frontiers:
                continue
            pts = rec.saved_frontiers[-FRONTIER_LAST_K:]  # latest K (append order)
            for fp in pts:
                ax.scatter(fp[0], fp[1], c='orange', s=14,
                           marker='.', zorder=7, alpha=0.85)

        # ── Robot, trajectory, goal ─────────────────────────────────────────
        if self.robot_pos is not None:
            ax.scatter(self.robot_pos[0], self.robot_pos[1],
                       c='red', s=100, zorder=11, marker='o', label='Robot')

        if len(self._traj_history) > 1:
            traj = np.array(self._traj_history)
            ax.plot(traj[:, 0], traj[:, 1], 'm-', linewidth=1.5, zorder=9)

        if self.goal_xy is not None:
            ax.scatter(self.goal_xy[0], self.goal_xy[1],
                       c='lime', s=180, zorder=11, marker='*', label='Goal')

        # ── Active inter-region door route ───────────────────────────────────
        if getattr(self, "_remote_door_waypoints", None):
            if len(self._remote_door_waypoints) > 0:
                wps = np.array(self._remote_door_waypoints)
                ax.plot(wps[:, 0], wps[:, 1], color='magenta',
                        linewidth=2.0, linestyle='-', zorder=10, label='Door route')
                ax.scatter(wps[:, 0], wps[:, 1], c='magenta',
                           s=80, marker='D', zorder=11)

        # ── Legend ──────────────────────────────────────────────────────────
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='lime',
                   markersize=8, label='Region (has frontiers)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                   markersize=8, label='Region (exhausted)'),
            Line2D([0], [0], marker='D', color='w', markerfacecolor='cyan',
                   markersize=7, label='Door waypoint'),
            Line2D([0], [0], color='cyan', linestyle='--', label='Graph edge'),
            Line2D([0], [0], marker='.', color='w', markerfacecolor='orange',
                   markersize=6, label=f'Latest {FRONTIER_LAST_K} frontiers'),
            Line2D([0], [0], marker='*', color='w', markerfacecolor='lime',
                   markersize=10, label='Current goal'),
            Line2D([0], [0], color='magenta', label='Active door route'),
        ]
        ax.legend(handles=legend_elements, loc='upper right',
                  fontsize=7, framealpha=0.7)

        ax.set_xlim(b_map[0][0], b_map[1][0])
        ax.set_ylim(b_map[0][1], b_map[1][1])
        ax.set_aspect('equal')
        ax.set_title(
            f"Step {self._composite_step}  |  regions={len(self.all_regions)}"
            f"  doors={len(self.gridmap.doors)}"
            f"  mode={self.mode}/{self.nav_state}"
        )

        fig.savefig(os.path.join(out_dir, f"plot_step_{self._composite_step}.png"),
                    dpi=120, bbox_inches='tight')
        fig.savefig(os.path.join(self.folder, "plot2.png"), bbox_inches='tight')
        plt.close(fig)

    def save_occupancy_snapshot(self, tag=None):
        """
        Saves occupancy grid + visited + frontier as image and numpy.
        """
        out_dir = os.path.join(self.folder, "occupancy_snapshots")
        os.makedirs(out_dir, exist_ok=True)

        if tag is None:
            tag = f"step_{self._composite_step:06d}"

        data = self.gridmap.data.copy()
        visited = self.gridmap.visited.copy()
        frontier = self.gridmap.frontier.copy()

        n = self.gridmap.n_cells

        # ---- Create visualization image ----
        img = np.zeros((n, n, 3), dtype=np.uint8)

        # unknown
        img[:, :] = [10, 10, 80]

        # visited free
        img[visited == 1] = [200, 200, 0]   # yellow (clearer than dark green)

        # occupied
        img[data == 1] = [0, 0, 0]

        # frontier overlay (optional highlight)
        img[frontier == 1] = [255, 140, 0]  # orange

        plt.figure(figsize=(8, 8))
        plt.imshow(img, origin="lower")
        plt.title(f"Occupancy Snapshot {tag}")
        plt.axis("off")
        plt.savefig(os.path.join(out_dir, f"{tag}.png"), dpi=150, bbox_inches="tight")
        plt.close()

        # ---- Save raw arrays ----
        np.savez_compressed(
            os.path.join(out_dir, f"{tag}.npz"),
            occupancy=data,
            visited=visited,
            frontier=frontier
        )

        self.get_logger().info(f"[OCC SNAPSHOT SAVED] {tag}")

def main():
    rclpy.init()
    node = GridmapNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
import rclpy.time
import torch
from scipy.spatial.transform import Rotation
import signal
import threading
from typing import List

import numpy as np
import math
import tf2_ros
from tf2_ros import TransformBroadcaster
import tf2_geometry_msgs

import sensor_msgs_py.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from geometry_msgs.msg import TransformStamped, Twist, Point, PointStamped, PoseStamped, PoseWithCovarianceStamped
from visualization_msgs.msg import Marker

from mntfields.mntfields_ros.sensing.lidar_adapter import Lidar2DAdapter
from mntfields.mntfields_ros.mapping.map_publisher import MapPublisher 
from mntfields.mntfields_ros.mapping.markers import MarkerPublisher
from mntfields.mntfields_ros.mapping.navigation import Navigator
from mntfields.mntfields_ros.mapping.exploration_graph import ExplorationGraph

from mntfields.mntfields_core.mapping.lidar_gridmap import LidarGridmap
from mntfields.mntfields_core.core.region import Region
from mntfields.mntfields_core.core.model_base import Model

from collections import deque
import threading

import os
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg", force=True)
from matplotlib import cm, colors
import matplotlib.pyplot as plt


REGION_RADIUS = 3.5
REGION_EXIT_MARGIN = 0.4 #was 0.4
REGION_MIN_OVERLAP = 0.4 #was 0.4
MAX_RANGE = 3.2
WARMUP_EPOCHS = 6
#EXPLORATION_RANGE = 2.5   # used for raycasting / visited marking

#print("NEW CODE")
class DummyRenderer:
    def __init__(self):
        pass

    def render(self, *args, **kwargs):
        return None

class DummyModel:
    def __init__(self):
        self.folder = "/home/mayank/turtlebot3_ws/mntfields_model"
        self.renderer = "cuda:0"   # IMPORTANT: must be string device
        self.h = 0.0

class FakePCD:
    def __init__(self, pts):
        self.points = pts
        self.normals = np.zeros_like(pts)

class GridmapNode(Node):
    def __init__(self):
        super().__init__('mntfields_gridmap')
        self.folder = os.path.join("/home/mayank/turtlebot3_ws/mntfields_model/mntfields_output")
        os.makedirs(self.folder, exist_ok=True)

        self.train_queue = deque(maxlen=4)   # keep last few frames only
        self.train_lock = threading.Lock()
        self.train_running = False
        self.declare_parameter('region_min_epochs', 10)  # train at least 20 times before 
        self._traj_history = []
        self._composite_step = 0

        # Tune these
        self.declare_parameter('train_timer_sec', 0.5)      # training loop cadence
        self.declare_parameter('train_samples', 5000)       # n_samples to get_data_from_lidar
        self.declare_parameter('train_epoch', 4)            # epoch passed into region.train_core
        self.declare_parameter('train_after_region', True)  # toggle
        self.radius = REGION_RADIUS

        # Start a separate timer JUST for training
        train_dt = float(self.get_parameter('train_timer_sec').value)
        self.create_timer(train_dt, self.train_tick)
        
        self.declare_parameter('n_cells', 400)
        self.declare_parameter('near_dist', 2.0)
        self.declare_parameter('map_size', 40.0)  # Add this

        n_cells = self.get_parameter('n_cells').value
        near_dist = self.get_parameter('near_dist').value
        map_size = self.get_parameter('map_size').value

        self.map_to_odom = None
        self.origin_initialized = False
        self.set_parameters([rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)])
        # --- exploration / control params ---
        self.declare_parameter('goal_reached_dist', 0.20)
        self.declare_parameter('yaw_tol', 0.25)          # rad
        self.declare_parameter('kp_yaw', 0.8)
        self.declare_parameter('kp_lin', 0.25)
        self.declare_parameter('max_lin', 0.15)          # m/s
        self.declare_parameter('max_ang', 0.8)           # rad/s
        self.declare_parameter('frontier_min_cells', 2)
        self.declare_parameter('goal_cooldown_sec', 2.0)

        self.path = None        #these 4 lines are for the path holding for stable paths - latest diff
        self.path_last_update = self.get_clock().now()
        self.path_update_period = 2.0   # seconds
        self.path_index = 0

        self.goal_xy = None
        self.last_goal_time = self.get_clock().now()
        self.mode = "BOOTSTRAP"   # BOOTSTRAP -> EXPLORE
        #--diff
        self.warmup_frames_needed = 5
        self.warmup_epochs_needed = WARMUP_EPOCHS
        self.warmup_done = False
        self.warmup_start_epoch = 0
        self.nav_state = "IDLE"
        self.wait_start_time = None
        self.wait_duration = 3.0
        self.latest_candidate_path = None
        #--
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        self.navigator = Navigator(self)
        self.map_publisher = MapPublisher(self)
        self.markers = MarkerPublisher(self)

        self.robot_pos = None
        self.robot_yaw = 0.0
        self.latest_cloud = None

        self.all_regions: dict = {}
        self.region_initialized = False
        self.bootstrap_frames = 0
        self.bootstrap_threshold = 10 # frames before region starts
        
        self.adapter = Lidar2DAdapter()

        # Traversability map (1 = free, 0 = blocked)
        self.trav_map = np.ones((n_cells, n_cells), dtype=np.uint8)
        self.gridmap = LidarGridmap(
            trav_map=self.trav_map,
            regions=self.all_regions,#{}
            n_cells=n_cells,
            near_dist=2.0,
            map_size=map_size,           # Add this
            origin=(0.0, 0.0), #(origin_x, origin_y)  # Add this
            room_seg_config=(7,2),
        )
        self.origin_initialized = False
        self.declare_parameter('occ_height', 0.3)
        self.occ_height = float(self.get_parameter('occ_height').value)
        self.occ_height_tol = 0.01

        self.create_subscription(PointCloud2,'/lidar3d_points',self.lidar_cb,10)
        #self.create_subscription(PoseWithCovarianceStamped,'/amcl_pose',self.pose_cb,5) - will be used for a built map
        self.create_subscription(Odometry, '/odom', self.odom_cb, 10)

        self.occ_pub = self.create_publisher(OccupancyGrid,'/mntfields/occ_map',5)
        self.frontier_pub = self.create_publisher(OccupancyGrid,'/mntfields/frontier',5)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.goal_marker_pub = self.create_publisher(Marker, '/mntfields/goal_marker', 10)
        self.frontier_marker_pub = self.create_publisher(Marker, '/mntfields/frontier_markers', 10)
        self.field_path_pub = self.create_publisher(Path, '/field_path', 10)

        self.create_timer(0.2, self.update)

        self.model = DummyModel()
        self.region = None

        self.exploration_graph = ExplorationGraph() #graph diff
        self._remote_door_waypoints: List[np.ndarray] = []  # door waypoints for inter-region travel
        self._remote_target_version: Optional[int] = None
        self._goal_blacklist_radius = 0.5 #was 0.3
        self._visited_goals: List[np.ndarray] = []

        self.train_counter = 0
        self.train_every_n_frames = 1

        self._map_update_counter = 0 #claude difff
        self._path_gen_running   = False

        # ── Goal-dwell training burst ─────────────────────────────────────
        self.declare_parameter('dwell_frames_needed', 6)    # frames to collect at goal
        self.declare_parameter('dwell_epochs',        3)    # epochs to train before resuming
        self.dwell_active       = False
        self.dwell_frames       = []                        # accumulated during dwell
        self.dwell_frames_count = 0
        self.dwell_epoch_start  = 0
        self.dwell_phase        = "COLLECT"  # "COLLECT" | "TRAIN" | "DONE"


    def lidar_cb(self, msg):
        self.latest_cloud = msg
        if not hasattr(self, "_last_lidar_log"):
            self._last_lidar_log = self.get_clock().now()
            #self.get_logger().info("Received first LiDAR scan")
            return
        now = self.get_clock().now()
        if (now - self._last_lidar_log).nanoseconds > 2e9:  # 2 seconds
            #self.get_logger().info("Receiving LiDAR scans...")
            self._last_lidar_log = now
    
    def odom_cb(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation

        self.robot_pos = np.array([p.x, p.y, p.z], dtype=np.float32)
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.robot_yaw = np.arctan2(siny_cosp, cosy_cosp)

        # Lock map once, using the *initial* odom pose
        if not self.origin_initialized:
            self.gridmap.origin = np.array([p.x, p.y], dtype=np.float32)
            self.origin_initialized = True

    def init_region(self, robot_pose):
        ''' Initializes a region around the robot and registers it. '''

        center_x = robot_pose[0]
        center_y = robot_pose[1]
        radius = self.radius

        bound = torch.tensor([
            [center_x - radius, center_y - radius, -0.1],
            [center_x + radius, center_y + radius,  0.1],
        ], dtype=torch.float32)

        new_region = Region(
            model=self.model,
            bound=bound,
            init_network=True,
            eval=False,)
        new_region.gridmap = self.gridmap
        self.all_regions[new_region.version] = new_region

        if self.region is not None and self.region.all_lidar is not None:
            last = self.region.all_lidar[-5:]
            new_region.all_lidar = last.copy()
            new_region.lidar_obs = None
            new_region.kd_tree = None
        else:
            new_region.all_lidar = []
        self.region = new_region
        self.region_initialized = True
        self.get_logger().info(
            f"[REGION {new_region.version} INIT] "
            f"center=({center_x:.2f}, {center_y:.2f}) "
            f"bound=[({bound[0,0]:.2f},{bound[0,1]:.2f}) → "
            f"({bound[1,0]:.2f},{bound[1,1]:.2f})]")
        # ── Re-enter WARMUP for every new region ─────────────────────────
        self.mode               = "WARMUP"
        self.warmup_start_epoch = int(self.region.epoch)
        self.goal_xy            = None
        self.path               = None
        self.nav_state          = "IDLE"
        self.latest_candidate_path = None
        self._path_gen_running  = False
        self.get_logger().info(f"[FLOW] Region {new_region.version} → entering WARMUP")

    def start_region_warmup(self):
        self.mode = "WARMUP"
        self.region_warmup_start_epoch = self.region.epoch if self.region else 0
        self.navigator.stop_robot()
        self.get_logger().info(f"[WARMUP] Starting region {self.region.version}")

    def feed_region(self, world_pts, robot_pose):
        if len(world_pts) < 50:
            self.get_logger().warn(f"Too few points: {len(world_pts)}")
            return
        try:
            pts = torch.tensor(world_pts, dtype=torch.float32)
            pos = torch.tensor([robot_pose[0], robot_pose[1], 0.0], dtype=torch.float32)
            ##print("WORLD PTS MIN/MAX:", world_pts.min(), world_pts.max())
            n_samples = int(self.get_parameter('train_samples').value)
            epoch = int(self.get_parameter('train_epoch').value)
            data = self.region.get_data_from_lidar(
                pts.cpu().numpy(),
                pos.cpu().numpy(),
                device='cuda:0',
                n_samples=n_samples
            )
            if data is None:
                self.get_logger().warn("get_data_from_lidar returned None")
                return
            points, closest = data
            if points.shape[0] == 0:
                self.get_logger().warn("get_data_from_lidar returned 0 points")
                return

            if data is None:
                self.get_logger().warn("get_data_from_lidar returned None")
                return
            x0 = points[:, :3]
            x1 = points[:, 3:6]
            cp0 = closest[:, :3]
            cp1 = closest[:, 3:6]

            raw = torch.cat([x0, x1, cp0, cp1], dim=1)   # Nx12
            #print("RAW SHAPE:", raw.shape)

            if raw.ndim != 2 or raw.shape[1] != 12:
                self.get_logger().error(f"[DBG] raw has wrong shape: {tuple(raw.shape)} (expected Nx12)")
                return
            if not torch.isfinite(raw).all():
                bad = (~torch.isfinite(raw)).sum().item()
                self.get_logger().error(f"[DBG] raw has non-finite values: count={bad}")
                return
            with torch.no_grad():
                mn = raw.min(dim=0).values.cpu().numpy()
                mx = raw.max(dim=0).values.cpu().numpy()
            frame_data = self.region.calculate_speed_and_normal(raw)
            #frame_data = raw

            if frame_data.ndim != 2:
                self.get_logger().error(f"[DBG] frame_data not 2D: {tuple(frame_data.shape)}")
                return
            if frame_data.shape[1] < 12:
                self.get_logger().error(f"[DBG] frame_data has too few cols: {frame_data.shape[1]} (expected >=12)")
                return
            if not torch.isfinite(frame_data).all():
                bad = (~torch.isfinite(frame_data)).sum().item()
                self.get_logger().error(f"[DBG] frame_data non-finite: count={bad}")
                return
            with torch.no_grad():
                fd_min = frame_data.min().item()
                fd_max = frame_data.max().item()

            y0_col = frame_data[:, 6]
            y1_col = frame_data[:, 7]

            if frame_data.shape[0] == 0:
                self.get_logger().warn("No valid frame data after speed calculation")
                return
            self.region.last_robot_pose = np.array([robot_pose[0], robot_pose[1], 0.0], dtype=np.float32)

            if self.region.function is not None:
                try:
                    k = min(64, frame_data.shape[0])
                    xp_n = frame_data[:k, :6].clone().to(self.region.device)
                    with torch.no_grad():
                        tt = self.region.function.TravelTimes(xp_n).detach().cpu().numpy().reshape(-1)
                    with torch.enable_grad():
                        sp_t = self.region.function.Speed(xp_n)  # tensor
                    sp = sp_t.detach().cpu().numpy().reshape(-1)

                except Exception as e:
                    self.get_logger().error(f"[DBG MODEL] failed: {e}")
            #print(f"[FEED] frame_data shape={frame_data.shape}, " #UNCOMMENT FOR DEBUG ON FRAME DATA - CORRECT FOR NOW
            #      f"y0 range=[{frame_data[:,6].min():.4f}, {frame_data[:,6].max():.4f}]")
            #print(f"[FEED] calling train_core with {frame_data.shape[0]} rows")      
            success = self.region.train_core(epoch=epoch, frame_data=frame_data)
            loss_val = getattr(self.region, 'last_loss', None)
            try:
                test = frame_data[:128, :6].clone().to(self.region.device)
                #test.requires_grad_(True)
                test_n = self.region.normalize_tensor(test)
                with torch.enable_grad():
                    tau, w, coords_in = self.region.function.network.out(test_n)
                    tt = tau[:, 0]
                    sp = self.region.function.Speed(test_n)

                g = torch.autograd.grad(tt.sum(), coords_in, retain_graph=True)[0]
                g0 = g[:, :3]  # gradient w.r.t x0
                g0_norm = torch.sqrt((g0 * g0).sum(dim=1) + 1e-12)
            except Exception as e:
                print("MODEL DEBUG FAILED:", e)

        except Exception as e:
            self.get_logger().error(f"feed_region error: {e}")
            import traceback
            traceback.print_exc()

    def pick_frontier_goal(self):
        frontier = self.gridmap.frontier
        iy, ix = np.where(frontier > 0)
        min_frontier_cells = int(self.get_parameter('frontier_min_cells').value)

        if len(ix) < min_frontier_cells:
            return None
        # ✅ DIRECTLY call pick_reachable_frontier (it handles everything)
        goal = self.pick_reachable_frontier()
        if goal is not None:
            robot = self.robot_pos[:2]
            dist = np.linalg.norm(goal - robot)
        return goal

    def pick_room_aware_goal(self, current_room, rooms, room_graph):
        robot_xy = self.robot_pos[:2]

        # 1. Try live local frontier (with blacklist)
        local = self._frontier_in_room(current_room, rooms)
        if local is not None:
            if (len(self._visited_goals) == 0 or
                    np.linalg.norm(np.array(self._visited_goals) - local[None,:], axis=1).min()
                    > self._goal_blacklist_radius):
                return local

        # 2. Try global live frontier
        live = self.pick_reachable_frontier()
        if live is not None:
            return live

        # 3. Local region has no live frontiers → check ExplorationGraph bank
        if self.region is not None:
            ver = self.region.version
            result = self.exploration_graph.best_remote_frontier(ver, robot_xy)
            if result is not None:
                target_ver, target_frontier, door_waypoints = result
                print(f"[EXPLORATION GRAPH] Routing to region {target_ver} "
                      f"via {len(door_waypoints)} door(s), "
                      f"frontier={target_frontier}")
                self._remote_door_waypoints = door_waypoints
                self._remote_target_version = target_ver
                return target_frontier

        print("[EXPLORATION GRAPH] Fully explored — no frontiers anywhere")
        return None

    def _frontier_in_room(self, room_id, rooms):
        if room_id is None:
            return None
        if room_id not in rooms:
            return None
        room_cells = rooms[room_id]["cells"]
        frontier_pts = []
        for (i, j) in room_cells:
            if self.gridmap.frontier[i, j] == 1:
                x, y = self.gridmap._grid_to_world(i, j)
                frontier_pts.append([x, y])
        if len(frontier_pts) == 0:
            return None
        frontier_pts = np.array(frontier_pts)
        # choose closest to robot
        robot = self.robot_pos[:2]
        dists = np.linalg.norm(frontier_pts - robot[None, :], axis=1)
        idx = np.argmin(dists)
        return frontier_pts[idx] #END DIFF

    def publish_field_path(self, path_xy, frame_id='odom'):
        """
        path_xy: list[np.ndarray] or np.ndarray of shape (N,2)
        """
        if path_xy is None:
            return
        if isinstance(path_xy, list):
            pts = np.asarray(path_xy, dtype=np.float32)
        else:
            pts = np.asarray(path_xy, dtype=np.float32)

        if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 2:
            return

        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id

        poses = []
        for p in pts:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = float(p[0])
            ps.pose.position.y = float(p[1])
            ps.pose.position.z = 0.0
            ps.pose.orientation.w = 1.0
            poses.append(ps)

        msg.poses = poses
        self.field_path_pub.publish(msg)

    def decimate_path(self, path_xy, spacing=0.4):
        pts = np.asarray(path_xy, dtype=np.float32)

        if pts.ndim == 1:
            pts = pts.reshape(1, 2)

        if len(pts) < 2:
            return pts

        new_pts = [pts[0]]

        for i in range(1, len(pts)):
            if np.linalg.norm(pts[i] - new_pts[-1]) >= spacing:
                new_pts.append(pts[i])

        if np.linalg.norm(new_pts[-1] - pts[-1]) > 0.05:
            new_pts.append(pts[-1])

        return np.asarray(new_pts, dtype=np.float32)

    def smooth_path(self, path_xy, window=5):
        if path_xy is None:
            return None

        pts = np.asarray(path_xy, dtype=np.float32)

        if pts.ndim == 1:
            pts = pts.reshape(1, 2)

        if len(pts) < window:
            return pts

        sm = pts.copy()
        k = window // 2

        for i in range(len(pts)):
            a = max(0, i-k)
            b = min(len(pts), i+k+1)
            sm[i] = pts[a:b].mean(axis=0)

        sm[0] = pts[0]
        sm[-1] = pts[-1]
        return sm 

    def pick_lookahead(self, path, robot_xy, L=0.45):
        pts = np.asarray(path)
        r = np.asarray(robot_xy)

        d = np.linalg.norm(pts - r[None,:], axis=1)
        i0 = np.argmin(d)

        acc = 0.0
        i = i0
        while i < len(pts)-1 and acc < L:
            acc += np.linalg.norm(pts[i+1]-pts[i])
            i += 1

        return pts[i]
       #end diff

    def is_path_significantly_different(self, old_path, new_path, threshold=0.25):

        old_path = np.asarray(old_path)
        new_path = np.asarray(new_path)

        if len(old_path) < 2 or len(new_path) < 2:
            return True

        # compare up to min length
        n = min(len(old_path), len(new_path))

        # compare corresponding waypoints
        diffs = np.linalg.norm(old_path[:n] - new_path[:n], axis=1)

        max_diff = np.max(diffs)

        print("Max waypoint diff:", max_diff)

        return max_diff > threshold

    def debug_plot(self, field_waypoint=None):
        data = self.gridmap.data
        visited = self.gridmap.visited

        im = np.ones((*data.shape, 3), dtype=np.uint8) * 255
        im[data > 0.9] = np.array([0, 0, 0])      # walls
        im[visited == 1] = np.array([0, 255, 0])  # visited

        fig, ax = plt.subplots(1,1,figsize=(6,6))
        ax.imshow(im, origin='lower')

        # robot
        if self.robot_pos is not None:
            i,j = self.world_to_grid(self.robot_pos[0], self.robot_pos[1])
            ax.scatter(j,i,c='magenta',s=60,label='robot')

        # goal
        if self.goal_xy is not None:
            i,j = self.world_to_grid(self.goal_xy[0], self.goal_xy[1])
            ax.scatter(j,i,c='red',s=60,label='goal')

        # field waypoint
        if field_waypoint is not None:
            i,j = self.world_to_grid(field_waypoint[0], field_waypoint[1])
            ax.scatter(j,i,c='blue',s=60,label='field wp')
        ax.legend()
        plt.pause(0.01)

    def train_tick(self):
        if self.region is None:
            print(f"[TRAIN_TICK] region=False, running={self.train_running}, "
                  f"queue_len={len(self.train_queue)}, epoch=-1")
            return

        print(f"[TRAIN_TICK] region=True, running={self.train_running}, "
              f"queue_len={len(self.train_queue)}, epoch={self.region.epoch}")

        if self.train_running:
            return  # previous train still going

        with self.train_lock:
            if len(self.train_queue) == 0:
                return
            self.train_running = True     
            world_pts, robot_pose = self.train_queue.pop()
            self.train_queue.clear()

        # ✅ Set BEFORE spawning thread — eliminates race condition
       
        def _do_train():
            try:
                print(f"[TRAIN THREAD] starting feed_region, epoch={self.region.epoch}")
                self.feed_region(world_pts, robot_pose)
                print(f"[TRAIN THREAD] done, epoch={self.region.epoch}")
            except Exception as e:
                self.get_logger().error(f"[TRAIN] failed: {e}")
                import traceback
                traceback.print_exc()
            finally:
                self.train_running = False

        import threading
        t = threading.Thread(target=_do_train, daemon=True)
        t.start()

    def maybe_expand_region(self):
        if self.region is None:
            return

        robot = self.robot_pos[:2]
        bound = self.region.bound.numpy()

        margin = 0.3

        near_edge = (
            abs(robot[0] - bound[0,0]) < margin or
            abs(robot[0] - bound[1,0]) < margin or
            abs(robot[1] - bound[0,1]) < margin or
            abs(robot[1] - bound[1,1]) < margin
        )

        if near_edge:
            new_bound = torch.tensor([
                [bound[0,0]-0.3, bound[0,1]-0.3, -0.1],
                [bound[1,0]+0.3, bound[1,1]+0.3,  0.1],
            ], dtype=torch.float32)

            self.region.set_bound(new_bound)
            self.get_logger().info("[REGION EXPAND]")


    def fit_region_to_lidar(self, world_points_hit: np.ndarray,
                             expand_thresh: float = 0.25,
                             shrink_thresh: float = 0.40,
                             step: float = 0.15,
                             min_hits: int = 8,
                             max_area_mult: float = 3.5):
        """
        Adapt the current region bounding box to fit the observed LiDAR geometry.

        For each of the 4 edges we look at real hits projecting onto that side:

          EXPAND — if >= min_hits points are within expand_thresh of the edge,
                   walls are pressing on the boundary → grow outward by step.

          SHRINK — if no hits fall within shrink_thresh of the edge,
                   there is open space there → pull inward by step.

          LEAVE  — otherwise the boundary is already a reasonable fit.

        Limits: minimum half-size 1.0 m, maximum area max_area_mult × base,
        robot always kept >= 0.5 m inside every edge.
        """
        if self.region is None or self.robot_pos is None:
            return
        if world_points_hit is None or len(world_points_hit) < min_hits:
            return

        b     = self.region.bound.numpy().copy()   # [[xmin,ymin,z],[xmax,ymax,z]]
        pts   = world_points_hit[:, :2]            # (N,2) world XY
        robot = self.robot_pos[:2]

        MIN_HALF  = 1.0
        MAX_AREA  = max_area_mult * (REGION_RADIUS ** 2) * 4
        ROBOT_PAD = 0.5

        def _try_move(axis, side, delta):
            # delta > 0 → expand (edge moves outward)
            # delta < 0 → shrink (edge moves inward)
            new_b = b.copy()
            if side == 0:
                new_b[0, axis] = b[0, axis] - delta   # min edge: expand = move left/down
            else:
                new_b[1, axis] = b[1, axis] + delta   # max edge: expand = move right/up
            if (new_b[1, 0] - new_b[0, 0]) / 2.0 < MIN_HALF:
                return
            if (new_b[1, 1] - new_b[0, 1]) / 2.0 < MIN_HALF:
                return
            if (new_b[1, 0] - new_b[0, 0]) * (new_b[1, 1] - new_b[0, 1]) > MAX_AREA:
                return
            if not (new_b[0, 0] + ROBOT_PAD < robot[0] < new_b[1, 0] - ROBOT_PAD and
                    new_b[0, 1] + ROBOT_PAD < robot[1] < new_b[1, 1] - ROBOT_PAD):
                return
            b[:] = new_b

        # Only count hits whose perpendicular axis falls within the region span
        y_in = (pts[:, 1] >= b[0, 1]) & (pts[:, 1] <= b[1, 1])
        x_in = (pts[:, 0] >= b[0, 0]) & (pts[:, 0] <= b[1, 0])
        pts_for_x_edges = pts[y_in]
        pts_for_y_edges = pts[x_in]

        for axis, side, edge, relevant in [
            (0, 0, b[0, 0], pts_for_x_edges),  # xmin (left)
            (0, 1, b[1, 0], pts_for_x_edges),  # xmax (right)
            (1, 0, b[0, 1], pts_for_y_edges),  # ymin (bottom)
            (1, 1, b[1, 1], pts_for_y_edges),  # ymax (top)
        ]:
            if len(relevant) < min_hits:
                continue
            coords = relevant[:, axis]
            if side == 0:   # min edge
                near_edge  = (coords < edge + expand_thresh).sum()
                has_near   = (coords < edge + shrink_thresh).sum() > 0
            else:           # max edge
                near_edge  = (coords > edge - expand_thresh).sum()
                has_near   = (coords > edge - shrink_thresh).sum() > 0

            if near_edge >= min_hits:
                _try_move(axis, side, step)    # expand
            elif not has_near:
                _try_move(axis, side, -step)   # shrink

        if not np.allclose(b, self.region.bound.numpy(), atol=1e-4):
            new_bound = torch.tensor(
                [[b[0,0], b[0,1], -0.1],
                 [b[1,0], b[1,1],  0.1]], dtype=torch.float32)
            self.region.set_bound(new_bound)
            self.get_logger().info(
                f"[FIT REGION] R{self.region.version} "
                f"({b[0,0]:.2f},{b[0,1]:.2f})→({b[1,0]:.2f},{b[1,1]:.2f})"
            )

    def _get_frontier_world_pts(self, max_pts=200):
        """Return frontier cell centres in world coords as (N,2) array."""
        iy, ix = np.where(self.gridmap.frontier > 0)
        if len(ix) == 0:
            return None
        cell   = self.gridmap.cell_size
        origin = self.gridmap.origin
        half   = self.gridmap.map_size / 2.0
        fx = (origin[0] - half) + (ix + 0.5) * cell
        fy = (origin[1] - half) + (iy + 0.5) * cell
        pts = np.stack([fx, fy], axis=1)
        if len(pts) > max_pts:
            pts = pts[np.random.choice(len(pts), max_pts, replace=False)]
        return pts

    def pick_reachable_frontier(self, max_candidates=300):
        frontier = self.gridmap.frontier
        iy, ix = np.where(frontier > 0)
        if len(ix) == 0:
            return None

        cell   = self.gridmap.cell_size
        origin = self.gridmap.origin
        half   = self.gridmap.map_size / 2.0
        fx = (origin[0] - half) + (ix + 0.5) * cell
        fy = (origin[1] - half) + (iy + 0.5) * cell
        cand = np.stack([fx, fy], axis=1)

        robot = self.robot_pos[:2]
        d = np.linalg.norm(cand - robot[None, :], axis=1)
        valid = d < 2.0
        if valid.sum() == 0:
            return None
        cand = cand[valid]
        d    = d[valid]

        # ✅ FIX 1: Only pick frontiers INSIDE the current region bound
        # The field has no knowledge of anything outside the bound
        if self.region is not None:
            in_bound = np.array([self.region.in_bound(c) for c in cand])
            if in_bound.sum() > 0:
                cand = cand[in_bound]
                d    = d[in_bound]
            # If ALL frontiers are outside the bound, fall back to closest-in-bound
            # rather than sending the robot off a cliff
            else:
                return cand[np.argmin(d)]   # geometric fallback

        if len(cand) > max_candidates:
            cand = cand[np.random.choice(len(cand), max_candidates, replace=False)]

        # No field yet → closest
        if self.region is None or self.region.function is None:
            return cand[np.argmin(d)]

        # Field query (all candidates guaranteed inside bound now)
        x0  = np.tile(np.array([robot[0], robot[1], 0.0]), (len(cand), 1))
        x1  = np.concatenate([cand, np.zeros((len(cand), 1))], axis=1)
        xp  = np.concatenate([x0, x1], axis=1)
        xp_t = torch.tensor(xp, dtype=torch.float32)
        xp_n = self.region.normalize_tensor(xp_t).to(self.region.device)

        with torch.no_grad():
            tt = self.region.function.TravelTimes(xp_n).cpu().numpy().reshape(-1)

        ok = np.isfinite(tt)
        if ok.sum() == 0:
            return cand[np.argmin(d)]

        return cand[ok][np.argmin(tt[ok])]

    def maybe_create_new_region(self):
        """
        Called every update() tick.
        If the robot has moved outside the current region's bound, either:
          (a) switch to an existing region that covers the robot, OR
          (b) spawn a new region centred on the robot.
        Old region is unloaded to disk to free GPU memory.
        """
        if self.region is None or self.robot_pos is None:
            return

        robot = self.robot_pos[:2]
        b = self.region.bound.numpy()

        # Is robot still comfortably inside current bound?
        inside = (
            robot[0] > b[0, 0] + REGION_EXIT_MARGIN and
            robot[0] < b[1, 0] - REGION_EXIT_MARGIN and
            robot[1] > b[0, 1] + REGION_EXIT_MARGIN and
            robot[1] < b[1, 1] - REGION_EXIT_MARGIN
        )
        if inside:
            return  # nothing to do

        # --- Robot has left (or is about to leave) the current region ---

        # (a) Check if any existing region already covers the robot
        for ver, reg in self.all_regions.items():
            if ver == self.region.version:
                continue
            if reg.in_bound(robot):
                if reg.loaded:
                    self._switch_region(reg)
                    return
                else:
                    reg.load_from_disk()
                    self._switch_region(reg)
                    return

        # (b) No existing region covers the robot → create a new one
        #     But only if the area has enough visited free space to train on
        visited = self.gridmap.visited
        cell = self.gridmap.cell_size
        origin = self.gridmap.origin
        half = self.gridmap.map_size / 2.0

        # Count visited cells in the proposed new bound
        def count_visited_in_bound(cx, cy, r):
            ix0 = max(0, int((cx - r - (origin[0] - half)) / cell))
            ix1 = min(self.gridmap.n_cells - 1, int((cx + r - (origin[0] - half)) / cell))
            iy0 = max(0, int((cy - r - (origin[1] - half)) / cell))
            iy1 = min(self.gridmap.n_cells - 1, int((cy + r - (origin[1] - half)) / cell))
            region_area = (ix1 - ix0) * (iy1 - iy0)
            if region_area <= 0:
                return 0.0
            visited_count = visited[iy0:iy1, ix0:ix1].sum()
            return float(visited_count) / float(region_area)

        coverage = count_visited_in_bound(robot[0], robot[1], REGION_RADIUS)
        if coverage < REGION_MIN_OVERLAP:
            # Not enough data yet — keep extending current region temporarily
            # by softly expanding its bound rather than spawning premature region
            self._soft_extend_current_region(robot)
            return

        # Enough data: spawn the new region
        self.get_logger().info(
            f"[NEW REGION] Robot at ({robot[0]:.2f},{robot[1]:.2f}) "
            f"left region {self.region.version}. "
            f"Visited coverage={coverage:.2f}. Spawning new region."
        )
        self._unload_current_region()
        self.init_region(self.robot_pos)

    def maybe_merge_regions(self):
        """
        For each confirmed room that spans multiple regions, merge those
        regions into one with a combined bounding box and retrain.
        """
        # Build room → [region_versions] mapping
        room_to_regions: Dict[int, List[int]] = {}
        for ver, region in self.all_regions.items():
            for room in region.rooms:
                if room.id not in room_to_regions:
                    room_to_regions[room.id] = []
                room_to_regions[room.id].append(ver)

        for room_id, versions in room_to_regions.items():
            if len(versions) <= 1:
                continue  # room already fits in one region

            # Only merge if all regions are confirmed (trained enough)
            regions_to_merge = [self.all_regions[v] for v in versions
                                if v in self.all_regions]
            if any(r.epoch < self.warmup_epochs_needed for r in regions_to_merge):
                continue  # wait until all are trained

            self.get_logger().info(
                f"[MERGE] Room {room_id} spans regions {versions} → merging"
            )
            self._merge_regions(regions_to_merge)


    def _merge_regions(self, regions: List):
        """
        Merge a list of regions into one. The merged region gets:
        - Combined bounding box
        - All rooms from all merged regions
        - All training data concatenated
        - Retraining triggered immediately
        """
        # ── Compute combined bound ────────────────────────────────────
        bounds = [r.bound.numpy() for r in regions]
        new_min = np.array([min(b[0, i] for b in bounds) for i in range(3)])
        new_max = np.array([max(b[1, i] for b in bounds) for i in range(3)])
        new_bound = torch.tensor(
            np.stack([new_min, new_max]), dtype=torch.float32
        )

        # ── Check merged region isn't too large ──────────────────────
        w = new_max[0] - new_min[0]
        h = new_max[1] - new_min[1]
        MAX_MERGE_SIZE = REGION_RADIUS * 2 * 2   # 2× normal region size
        if w > MAX_MERGE_SIZE or h > MAX_MERGE_SIZE:
            self.get_logger().warn(
                f"[MERGE SKIPPED] merged bound too large ({w:.1f}×{h:.1f}m)"
            )
            return

        # ── Create merged region ──────────────────────────────────────
        merged = Region(model=self.model, bound=new_bound,
                        init_network=True, eval=False)
        merged.gridmap = self.gridmap

        # Transfer all rooms
        all_rooms = [room for r in regions for room in r.rooms]
        for room in all_rooms:
            merged.add_room(room)   # also sets room.region = merged

        # Concatenate all_lidar from constituent regions
        lidar_chunks = [r.all_lidar for r in regions
                        if r.all_lidar is not None and len(r.all_lidar) > 0]
        if lidar_chunks:
            merged.all_lidar = np.concatenate(lidar_chunks, axis=0)

        # ── Remove old regions from all_regions and ExplorationGraph ─
        old_versions = [r.version for r in regions]
        for ver in old_versions:
            if ver in self.all_regions:
                try:
                    self.all_regions[ver].unload_to_disk()
                except Exception:
                    pass
                del self.all_regions[ver]
            if ver in self.exploration_graph.records:
                del self.exploration_graph.records[ver]
            if ver in self.exploration_graph.graph:
                self.exploration_graph.graph.remove_node(ver)

        # ── Register merged region ────────────────────────────────────
        self.all_regions[merged.version] = merged
        center = np.array(merged.center[:2], dtype=np.float32)
        self.exploration_graph.ensure_region(merged.version, center)

        # Rebuild spatial edges for merged region
        for v2, reg2 in self.all_regions.items():
            if v2 == merged.version:
                continue
            b1 = new_bound.numpy()
            b2 = reg2.bound.numpy()
            x_gap = max(0.0, max(b1[0,0], b2[0,0]) - min(b1[1,0], b2[1,0]))
            y_gap = max(0.0, max(b1[0,1], b2[0,1]) - min(b1[1,1], b2[1,1]))
            if x_gap < 1.0 and y_gap < 1.0:
                mx = (max(b1[0,0], b2[0,0]) + min(b1[1,0], b2[1,0])) / 2.0
                my = (max(b1[0,1], b2[0,1]) + min(b1[1,1], b2[1,1])) / 2.0
                self.exploration_graph.add_spatial_edge(
                    merged.version, v2, np.array([mx, my], dtype=np.float32)
                )

        # ── If robot was in one of the merged regions, switch to merged ─
        if self.region is not None and self.region.version in old_versions:
            self.region = merged
            self.get_logger().info(
                f"[MERGE] Active region was merged → now region {merged.version}"
            )

        # ── Trigger retraining on merged region ──────────────────────
        self.mode               = "WARMUP"
        self.warmup_start_epoch = int(merged.epoch)
        self.goal_xy            = None
        self.nav_state          = "IDLE"
        self.get_logger().info(
            f"[MERGE] Region {merged.version} created "
            f"(merged {old_versions}) → WARMUP"
        )


    #def _switch_region(self, new_region: 'Region'):
    #    old_ver = self.region.version
#
    #    # keep last scans
    #    if self.region is not None and self.region.all_lidar is not None:
    #        new_region.all_lidar = self.region.all_lidar[-5:].copy()
    #        new_region.lidar_obs = None
    #        new_region.kd_tree = None
#
    #    self._unload_current_region()
    #    self.region = new_region
    #    self.get_logger().info(
    #        f"[REGION SWITCH] {old_ver} → {new_region.version}"
    #    )
    #    self.goal_xy = None
    #    self.last_goal_time = self.get_clock().now()
    #    self.gridmap.frontier[:] = 0
    def _switch_region(self, new_region: 'Region'):
        old_ver = self.region.version

        if self.region is not None and self.region.all_lidar is not None:
            new_region.all_lidar = self.region.all_lidar[-5:].copy()
            new_region.lidar_obs = None
            new_region.kd_tree   = None

        self._unload_current_region()

        # Flush stale training frames from the old region before switching
        with self.train_lock:
            self.train_queue.clear()
        self.train_running = False   # abort any in-flight train for old region

        self.region = new_region
        self.get_logger().info(f"[REGION SWITCH] {old_ver} → {new_region.version}")
        self.goal_xy        = None
        self.last_goal_time = self.get_clock().now()
        self.gridmap.frontier[:] = 0


    def _unload_current_region(self):
        """Save current region weights to disk and free GPU memory."""
        if self.region is None:
            return
        try:
            self.region.save_model()          # saves checkpoint to region folder
            self.region.unload_to_disk()      # releases GPU tensors
            self.get_logger().info(
                f"[REGION {self.region.version} UNLOADED]"
            )
        except Exception as e:
            self.get_logger().error(f"[UNLOAD ERROR] {e}")

    def _soft_extend_current_region(self, robot):
        """
        Softly expand region a bit so we don't immediately spawn a new one,
        BUT stop expanding once region gets too large.
        """

        b = self.region.bound.numpy().copy()
        pad = 0.3

        # --- proposed new bound ---
        new_min_x = min(b[0, 0], robot[0] - REGION_RADIUS - pad)
        new_min_y = min(b[0, 1], robot[1] - REGION_RADIUS - pad)
        new_max_x = max(b[1, 0], robot[0] + REGION_RADIUS + pad)
        new_max_y = max(b[1, 1], robot[1] + REGION_RADIUS + pad)

        # --- current size ---
        cur_w = b[1, 0] - b[0, 0]
        cur_h = b[1, 1] - b[0, 1]

        # --- new size if applied ---
        new_w = new_max_x - new_min_x
        new_h = new_max_y - new_min_y

        # 🧠 IMPORTANT: cap region size
        MAX_SIZE = REGION_RADIUS * 2.0   # ≈ 2.5m if radius=1.0

        if new_w > MAX_SIZE or new_h > MAX_SIZE:
            # Stop expanding → allow new region to spawn
            self.get_logger().info(
                f"[SOFT EXTEND BLOCKED] region already large "
                f"(w={cur_w:.2f}, h={cur_h:.2f})"
            )
            return

        # --- apply extension ---
        b[0, 0] = new_min_x
        b[0, 1] = new_min_y
        b[1, 0] = new_max_x
        b[1, 1] = new_max_y

        new_bound = torch.tensor(b, dtype=torch.float32)
        self.region.set_bound(new_bound)

        self.get_logger().info(
            f"[SOFT EXTEND] new bound: "
            f"({b[0,0]:.2f},{b[0,1]:.2f}) → ({b[1,0]:.2f},{b[1,1]:.2f})"
        )

    def _launch_path_gen(self, robot_xy, goal_xy): #claude diff
        """Offload generate_field_path to a background thread so update() never blocks."""
        if robot_xy is None or goal_xy is None:          # ← add this guard
            return
        if self.region is None or self.region.function is None:
            return
        if getattr(self, '_path_gen_running', False):
            return  # already running, skip
        self._path_gen_running = True
        robot_xy = robot_xy.copy()
        goal_xy  = goal_xy.copy()

        def _worker():
            try:
                raw_path = self.region.generate_field_path(
                    robot_xy, goal_xy, step_size=0.10, max_steps=200
                )
                if raw_path is not None:
                    raw_path = np.asarray(raw_path, dtype=np.float32)
                    if len(raw_path) > 1:
                        decimate = self.decimate_path(raw_path, spacing=0.2)
                        smooth   = self.smooth_path(decimate, window=3)
                        self.latest_candidate_path = np.asarray(smooth, dtype=np.float32)
            except Exception as e:
                self.get_logger().warn(f"[PATH GEN] {e}")
            finally:
                self._path_gen_running = False
        threading.Thread(target=_worker, daemon=True).start() #end claude diff

    def _maybe_create_new_region_guarded(self):
        """
        Only allow region creation when robot is NOT actively navigating.
        """
        if self.region is None:
            self.maybe_create_new_region()
            return

        if self.mode != "EXPLORE":
            self.maybe_create_new_region()
            return

        if self.goal_xy is None:
            self.maybe_create_new_region()
            return

        # Actively navigating → block
        if self.nav_state in ("STABILIZING", "FOLLOWING"):
            robot_xy = self.robot_pos[:2]
            d = np.linalg.norm(robot_xy - self.goal_xy)
            if d < 0.20:
                self.maybe_create_new_region()
            else:
                return  # still en-route, block creation
        else:
            # IDLE or DWELL → both are safe, robot is stationary
            self.maybe_create_new_region()

    def _update_region_graph_via_free_cells(self):
        """
        Build/update ExplorationGraph edges between regions using cells that:
          - are free/traversable (gridmap.data == 0, not occupied)
          - have been visited by the robot (gridmap.visited > 0)
          - fall within the discovered area of BOTH regions simultaneously
            (discovery_map has bits set for both region versions)

        This replaces boundary-midpoint and door-based spatial edges entirely.
        """
        versions = list(self.all_regions.keys())
        if len(versions) < 2:
            return

        # Build a mask of cells that are free AND visited
        free_visited = (self.gridmap.data == 0) & (self.gridmap.visited > 0)
        if not free_visited.any():
            return

        cell_indices = np.argwhere(free_visited)  # Nx2 array of (i, j)
        disc = self.gridmap.discovery_map  # uint64 bitmask per cell

        for v1, v2 in combinations(versions, 2):
            bit1 = np.uint64(1) << np.uint64(v1)
            bit2 = np.uint64(1) << np.uint64(v2)

            # Cells visited by BOTH regions
            shared_mask = free_visited & \
                          ((disc & bit1) != 0) & \
                          ((disc & bit2) != 0)

            shared_cells = np.argwhere(shared_mask)
            if len(shared_cells) < 3:   # need at least a few shared cells
                continue

            # Convert a representative subset to world coords as waypoints
            # Use up to 5 spread-out points via simple stride sampling
            stride = max(1, len(shared_cells) // 5)
            sampled = shared_cells[::stride][:5]
            waypoints = [
                np.array(self.gridmap._grid_to_world(int(ci), int(cj)))
                for ci, cj in sampled
            ]

            self.exploration_graph.add_free_cell_edge(v1, v2, waypoints)

    def _assign_frontiers_to_regions(self, current_version: int):
        """
        Per-cell frontier ownership:
        - Cell discovered only by current region → current region.
        - Cell discovered only by one remote (non-exhausted) region → that region.
        - Cell in overlap with current region → current region (always).
        - Cell in overlap not including current, or remote is exhausted → current region.
        """
        frontier_cells = np.argwhere(self.gridmap.frontier > 0)
        if len(frontier_cells) == 0:
            return

        disc = self.gridmap.discovery_map

        exclusive: dict = {}   # {version: [np.ndarray, ...]}
        current_pts = []

        for ci, cj in frontier_cells:
            cell_bits = disc[ci, cj]
            versions_here = self.gridmap.discover_regions(cell_bits)

            wx, wy = self.gridmap._grid_to_world(int(ci), int(cj))
            pt = np.array([wx, wy], dtype=np.float32)

            if current_version in versions_here or len(versions_here) == 0:
                # Belongs to current (own cell, overlap with current, or undiscovered)
                current_pts.append(pt)
            elif len(versions_here) == 1:
                v = versions_here[0]
                rec = self.exploration_graph.records.get(v)
                if rec is not None and not rec.is_exhausted():
                    exclusive.setdefault(v, []).append(pt)
                else:
                    current_pts.append(pt)   # exhausted remote → current takes it
            else:
                # Multi-region overlap not including current → current takes it
                current_pts.append(pt)

        # Assign exclusive frontiers to non-exhausted remote regions
        for version, pts in exclusive.items():
            region_obj = self.all_regions.get(version)
            if region_obj is None:
                continue
            # Safe center extraction whether bound is a tensor or ndarray
            b = region_obj.bound
            if hasattr(b, 'cpu'):
                b = b.cpu().numpy()
            center = b[:2].mean(axis=0)[:2].astype(np.float32)
            self.exploration_graph.ensure_region(version, center)
            for pt in pts:
                self.exploration_graph.records[version].add_frontier(
                    pt, ExplorationGraph.BLACKLIST_RADIUS
                )

        # All current-region and overlap frontiers → current region
        for pt in current_pts:
            self.exploration_graph.records[current_version].add_frontier(
                pt, ExplorationGraph.BLACKLIST_RADIUS
            )

    def _build_is_frontier_fn(self):
        """Returns a closure over the current frontier grid for bank reconciliation."""
        frontier_cells = np.argwhere(self.gridmap.frontier > 0)
        if len(frontier_cells) == 0:
            def is_frontier_fn(xy, radius):
                return False
            return is_frontier_fn

        from scipy.spatial import cKDTree
        # World coords of all live frontier cells
        world_pts = np.array([
            self.gridmap._grid_to_world(int(ci), int(cj))
            for ci, cj in frontier_cells
        ])
        tree = cKDTree(world_pts)

        def is_frontier_fn(xy: np.ndarray, radius: float) -> bool:
            hits = tree.query_ball_point(xy[:2], r=radius)
            return len(hits) > 0
        return is_frontier_fn

    def _start_goal_dwell(self):
        dwell_frames_needed = int(self.get_parameter('dwell_frames_needed').value)
        if self.region is None or dwell_frames_needed <= 0:
            self.goal_xy   = None
            self.nav_state = "IDLE"
            return

        self.dwell_active       = True
        self.dwell_frames       = []
        self.dwell_frames_count = 0
        self.dwell_phase        = "COLLECT"
        self.nav_state          = "DWELL"
        self.goal_xy            = None
        self.navigator.stop_robot()

        # Flush any in-flight frames from navigation so we get a clean epoch baseline
        with self.train_lock:
            self.train_queue.clear()
        # Epoch baseline set AFTER flush so we don't count pre-dwell epochs
        self.dwell_epoch_start = int(self.region.epoch)

        print(f"[DWELL] Starting — collecting {dwell_frames_needed} frames "
              f"then training {int(self.get_parameter('dwell_epochs').value)} epochs "
              f"(baseline epoch={self.dwell_epoch_start})")

    def _tick_goal_dwell(self, world_points_hit: np.ndarray, robot_xy: np.ndarray):
        dwell_frames_needed = int(self.get_parameter('dwell_frames_needed').value)
        dwell_epochs        = int(self.get_parameter('dwell_epochs').value)

        # If region changed under us (new region created), abort dwell cleanly
        if self.region is not None and int(self.region.epoch) < self.dwell_epoch_start:
            print("[DWELL] Region switched during dwell — ending dwell, entering WARMUP")
            self._end_goal_dwell()
            # Trigger warmup for the new region
            self.mode               = "WARMUP"
            self.warmup_start_epoch = int(self.region.epoch)
            self.warmup_done        = False
            return

        if self.dwell_phase == "COLLECT":
            if self.region is not None and len(world_points_hit) > 5:
                self.dwell_frames.append((
                    world_points_hit.copy(),
                    self.robot_pos.copy() if self.robot_pos is not None else robot_xy
                ))
                self.dwell_frames_count += 1
                print(f"[DWELL] Collected frame {self.dwell_frames_count}/{dwell_frames_needed}")

            if self.dwell_frames_count >= dwell_frames_needed:
                with self.train_lock:
                    for frame in self.dwell_frames:
                        self.train_queue.append(frame)
                self.dwell_epoch_start = int(self.region.epoch)
                self.dwell_phase       = "TRAIN"
                print(f"[DWELL] Collection done — training for {dwell_epochs} epochs "
                      f"from epoch {self.dwell_epoch_start}")

        elif self.dwell_phase == "TRAIN":
            if self.region is None:
                self._end_goal_dwell()
                return
            epochs_done = int(self.region.epoch) - self.dwell_epoch_start
            print(f"[DWELL] Training epoch {epochs_done}/{dwell_epochs}")
            if epochs_done >= dwell_epochs:
                print(f"[DWELL] Training complete — resuming navigation")
                self._end_goal_dwell()


    def _end_goal_dwell(self):
        """Clean up dwell state and return to normal navigation."""
        self.dwell_active  = False
        self.dwell_frames  = []
        self.dwell_phase   = "COLLECT"
        self.nav_state     = "IDLE"
        self.goal_xy       = None

    def update(self):
        now = self.get_clock().now()

        # ── Prerequisites ────────────────────────────────────────────────
        if self.robot_pos is None or self.latest_cloud is None:
            return

        result = self.adapter.update(self.latest_cloud, self.robot_pos)
        if result is None:
            return
        points_lidar, speeds = result
        if len(points_lidar) == 0:
            return

        # ── TF transform ─────────────────────────────────────────────────
        try:
            source_frame = self.latest_cloud.header.frame_id
            stamp        = self.latest_cloud.header.stamp
            if self.tf_buffer.can_transform('odom', source_frame, stamp,
                                            timeout=Duration(seconds=0.1)):
                transform = self.tf_buffer.lookup_transform('odom', source_frame, stamp)
            else:
                return
            t = transform.transform.translation
            q = transform.transform.rotation
            R = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
            world_points = (R @ points_lidar.T).T + np.array([t.x, t.y, t.z])
        except Exception as e:
            self.get_logger().warn(f"TF lookup failed: {e}")
            return

        # ── Bound filter ──────────────────────────────────────────────────
        robot_xy = self.robot_pos[:2]
        half = self.gridmap.map_size / 2
        ox, oy = self.gridmap.origin
        in_bounds = ((world_points[:, 0] > ox - half) & (world_points[:, 0] < ox + half) &
                     (world_points[:, 1] > oy - half) & (world_points[:, 1] < oy + half))
        world_points_filtered = world_points[in_bounds]
        speeds_filtered = speeds[in_bounds]
        if len(world_points_filtered) < 5:
            return

        # Angular downsampling + free-ray synthesis
        # For each of 360 angular bins:
        #   - Real hit bin  → closest point from sensor; hit_dist = measured dist
        #   - Empty bin     → synthetic endpoint at MAX_RANGE; hit_dist = inf
        # The raycast loop later uses hit_dist to decide whether to mark a wall.
        robot = robot_xy
        vecs   = world_points_filtered[:, :2] - robot_xy[None, :]
        dists  = np.linalg.norm(vecs, axis=1) + 1e-9
        angles = np.arctan2(vecs[:, 1], vecs[:, 0])
        bins   = ((angles + np.pi) / (2 * np.pi) * 359).astype(np.int32)

        closest_idx  = {}
        closest_dist = np.full(360, np.inf, dtype=np.float32)
        for k, b in enumerate(bins):
            if dists[k] < closest_dist[b]:
                closest_dist[b] = dists[k]
                closest_idx[b]  = k

        # Real-hit cloud — identical to the original method; used for raycast + gridmap
        keep = np.array(list(closest_idx.values()), dtype=np.int32)
        world_points_hit = world_points_filtered[keep]
        speeds_hit       = speeds_filtered[keep]

        # Full 360-bin cloud — real hits + synthetic free-ray endpoints
        # Used ONLY for training data collection; never touches gridmap/raycast.
        full_endpoints  = []
        full_speeds     = []
        for b in range(360):
            if b in closest_idx:
                idx = closest_idx[b]
                full_endpoints.append(world_points_filtered[idx])
                full_speeds.append(speeds_filtered[idx])
            else:
                theta  = (b / 359.0) * 2.0 * np.pi - np.pi
                dir_xy = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
                free_xy = robot_xy + MAX_RANGE * dir_xy
                free_z  = float(self.robot_pos[2]) if len(self.robot_pos) > 2 else 0.0
                full_endpoints.append(np.array([free_xy[0], free_xy[1], free_z], dtype=np.float32))
                full_speeds.append(0.0)
        world_points_360 = np.asarray(full_endpoints, dtype=np.float32)

        # ── Raycast: update gridmap.data + visited (real hits only) ──────
        cell   = self.gridmap.cell_size
        origin = self.gridmap.origin
        half_  = self.gridmap.map_size / 2

        def world_to_ij(x, y):
            return (int((x - origin[0] + half_) / cell),
                    int((y - origin[1] + half_) / cell))

        for pt3d in world_points_hit:
            pt = pt3d[:2]
            dx, dy = pt[0] - robot_xy[0], pt[1] - robot_xy[1]
            dist = float(np.hypot(dx, dy))
            if dist > MAX_RANGE:
                scale = MAX_RANGE / dist
                dx *= scale; dy *= scale; dist = MAX_RANGE
            steps = int(dist / cell)
            if steps < 1:
                ix, iy = world_to_ij(pt[0], pt[1])
                if 0 <= ix < self.gridmap.n_cells and 0 <= iy < self.gridmap.n_cells:
                    self.gridmap.data[iy, ix] = 1
                    self.gridmap.visited[iy, ix] = 1
                continue
            for s in range(steps - 1):
                x = robot_xy[0] + dx * s / steps
                y = robot_xy[1] + dy * s / steps
                ix, iy = world_to_ij(x, y)
                if not (0 <= ix < self.gridmap.n_cells and
                        0 <= iy < self.gridmap.n_cells):
                    break
                if self.gridmap.data[iy, ix] == 1:
                    break
                self.gridmap.visited[iy, ix] = 1
            ix_end, iy_end = world_to_ij(pt[0], pt[1])
            if (0 <= ix_end < self.gridmap.n_cells and
                    0 <= iy_end < self.gridmap.n_cells):
                self.gridmap.data[iy_end, ix_end] = 1
                self.gridmap.visited[iy_end, ix_end] = 1

        # ── Gridmap update (1 Hz room segmentation) ──────────────────────
        self._map_update_counter += 1
        do_room_update = (self._map_update_counter % 5 == 0)
        self.gridmap.update(
            lidar_scan=world_points_hit[:, :2],
            points=world_points_hit[:, :2],
            speeds=speeds_hit,
            cur_pos=robot_xy,
            region=self.region,
            regions=self.all_regions,
            update_room=do_room_update
        )

        # ── Snapshot live frontiers into ExplorationGraph ────────────────
        if self.region is not None:
            ver = self.region.version
            center = np.array(self.region.center[:2], dtype=np.float32)
            self.exploration_graph.ensure_region(ver, center)
            self._assign_frontiers_to_regions(ver)
            # Collect live frontier points for this region
            #iy, ix = np.where(self.gridmap.frontier > 0)
            #if len(ix) > 0:
            #    cell   = self.gridmap.cell_size
            #    origin = self.gridmap.origin
            #    half   = self.gridmap.map_size / 2.0
            #    fx = (origin[0] - half) + (ix + 0.5) * cell
            #    fy = (origin[1] - half) + (iy + 0.5) * cell
            #    live_frontiers = np.stack([fx, fy], axis=1)
            #    # Downsample to avoid flooding the bank
            #    if len(live_frontiers) > 50:
            #        live_frontiers = live_frontiers[
            #            np.random.choice(len(live_frontiers), 50, replace=False)
            #        ]
            #    self.exploration_graph.snapshot_frontiers(ver, live_frontiers)
        is_frontier_fn = self._build_is_frontier_fn()
        self.exploration_graph.reconcile_all_banks(is_frontier_fn, evict_radius=0.6)

        if do_room_update:
            for u, v, edge_data in self.gridmap.room_graph.edges(data=True):
                room_u = self.gridmap.room_graph.nodes[u].get('room', -1)
                room_v = self.gridmap.room_graph.nodes[v].get('room', -1)

                # Only register edges between two confirmed rooms
                if room_u not in self.gridmap.confirmed_rooms:
                    continue
                if room_v not in self.gridmap.confirmed_rooms:
                    continue

                pos_u = self.gridmap.room_graph.nodes[u].get('pos')
                pos_v = self.gridmap.room_graph.nodes[v].get('pos')
                if pos_u is None or pos_v is None:
                    continue

                door_xy = np.array([(pos_u[0]+pos_v[0])/2,
                                     (pos_u[1]+pos_v[1])/2], dtype=np.float32)

                reg_u = getattr(self.gridmap.confirmed_rooms[room_u], 'region', None)
                reg_v = getattr(self.gridmap.confirmed_rooms[room_v], 'region', None)
                if reg_u is None or reg_v is None:
                    continue
                if reg_u.version == reg_v.version:
                    continue   # same region, spatial edge already covers this

                self.exploration_graph.add_door_edge(reg_u.version, reg_v.version, door_xy)
                self._update_region_graph_via_free_cells()
#            # ── Spatial adjacency edges (intra-room, no door needed) ─────
#            for v1, reg1 in self.all_regions.items():
#                for v2, reg2 in self.all_regions.items():
#                    if v1 >= v2:
#                        continue
#                    #if self.exploration_graph.graph.has_edge(v1, v2):
#                    #    continue  # door edge already exists, skip
#
#                    b1 = reg1.bound.numpy()
#                    b2 = reg2.bound.numpy()
#
#                    # Gap between the two bounding boxes in x and y
#                    x_gap = max(0.0, max(b1[0,0], b2[0,0]) - min(b1[1,0], b2[1,0]))
#                    y_gap = max(0.0, max(b1[0,1], b2[0,1]) - min(b1[1,1], b2[1,1]))
#
#                    if x_gap < 1.0 and y_gap < 1.0:
#                        # Midpoint of the shared/nearest boundary zone
#                        mx = (max(b1[0,0], b2[0,0]) + min(b1[1,0], b2[1,0])) / 2.0
#                        my = (max(b1[0,1], b2[0,1]) + min(b1[1,1], b2[1,1])) / 2.0
#                        midpoint = np.array([mx, my], dtype=np.float32)
#                        #self.exploration_graph.add_spatial_edge(v1, v2, midpoint)
                #self._update_region_graph_via_free_cells()

        # ── Enqueue training frame (ALL modes) ───────────────────────────
        if self.region is not None and bool(self.get_parameter('train_after_region').value):
            self.train_counter += 1
            if self.train_counter % self.train_every_n_frames == 0:
                with self.train_lock:
                    self.train_queue.append((
                        world_points_hit.copy(),
                        self.robot_pos.copy()
                    ))

        # ── Region boundary check ─────────────────────────────────────────
        # ── Fit region bound to observed LiDAR geometry (1 Hz) ─────────
        if do_room_update and self.mode != "BOOTSTRAP":
            self.fit_region_to_lidar(world_points_hit)

        self._maybe_create_new_region_guarded()
        if self._map_update_counter % 10 == 0:
            self.maybe_merge_regions()
        # ── Trajectory ───────────────────────────────────────────────────
        if self.robot_pos is not None:
            self._traj_history.append(self.robot_pos[:2].copy())

        # ── Mode state machine (navigation only, map already updated above) ──

        if self.mode == "BOOTSTRAP":
            self.navigator.stop_robot()
            self.bootstrap_frames += 1
            if self.region is None and self.bootstrap_frames >= self.bootstrap_threshold:
                self.get_logger().info("Initializing first region")
                self.init_region(self.robot_pos)
                self.mode              = "WARMUP"
                self.warmup_start_epoch = int(self.region.epoch)
                self.goal_xy           = None
                self.last_goal_time    = now
                self.get_logger().info("[FLOW] Entering WARMUP phase")

        elif self.mode == "WARMUP":
            # Stop robot, check epochs — but do NOT return; fall through to publish
            self.navigator.stop_robot()
            if self.region is not None:
                cur_epochs = int(self.region.epoch) - int(self.warmup_start_epoch)
                print(f"[WARMUP] epoch {cur_epochs}/{self.warmup_epochs_needed}")
                if cur_epochs >= self.warmup_epochs_needed:
                    self.mode        = "EXPLORE"
                    self.warmup_done = True
                    self.goal_xy     = None
                    self.last_goal_time = now
                    self.get_logger().info("[FLOW] Warmup complete → starting EXPLORE")

        elif self.mode == "EXPLORE":
            robot_xy = np.asarray(self.robot_pos[:2], dtype=np.float32)
            now_sec  = now.nanoseconds * 1e-9

            # ── IDLE: pick a new frontier goal ───────────────────────────
            if self.nav_state == "IDLE":
                current_room = self.gridmap.get_room_at_position(robot_xy)
                rooms        = self.gridmap.get_rooms()
                room_graph   = self.gridmap.get_room_graph()
                g = self.pick_room_aware_goal(current_room, rooms, room_graph)

                if g is None:
                    self.navigator.stop_robot()
                    print("[NO FRONTIER]")
                else:
                    self.goal_xy           = np.asarray(g, dtype=np.float32)
                    self.wait_start_time   = now_sec
                    self._path_gen_running = False
                    print("[GOAL SET]", self.goal_xy)

                    # ── If ExplorationGraph gave us door waypoints, pre-load them ─
                    if self._remote_door_waypoints:
                        print(f"[EXPLORATION GRAPH] Using {len(self._remote_door_waypoints)}"
                              f" door waypoints → region {self._remote_target_version}")
                        # Build composite path: door chain + final frontier goal
                        full_wp = np.array(
                            self._remote_door_waypoints + [self.goal_xy],
                            dtype=np.float32
                        )
                        self.latest_candidate_path = full_wp
                        self._remote_door_waypoints = []   # consumed
                    else:
                        self.latest_candidate_path = None  # normal: path gen will fill this

                    self.nav_state = "STABILIZING"
            # ── STABILIZING: generate path in background, don't block ────
            elif self.nav_state == "STABILIZING":
                self.navigator.stop_robot()
                # kick off path gen if not already running
                self._launch_path_gen(robot_xy, self.goal_xy)

                if now_sec - self.wait_start_time >= self.wait_duration:
                    if self.latest_candidate_path is not None:
                        print("[PATH FROZEN]")
                        self.path = self.latest_candidate_path.copy()
                        self.navigator.reset_path()
                        self.nav_state = "FOLLOWING"
                    else:
                        print("[NO PATH FOUND — retrying]")
                        self.wait_start_time = now_sec  # extend wait

            # ── FOLLOWING: drive, continuously refresh path in background ─
            elif self.nav_state == "FOLLOWING":
                if self.goal_xy is None:
                    self.nav_state = "IDLE"
                    self.path = None
                    return
                if self.goal_xy is not None:
                    d_goal = float(np.linalg.norm(robot_xy - self.goal_xy))
                else:
                    d_goal = None
                #print(f"Distance to frontier: {d_goal:.2f}")
                if d_goal is None:
                    self.nav_state = "IDLE"
                    self.goal_xy = None
                    self.path = None 

                if d_goal is not None and d_goal < 0.25:
                    print("[FRONTIER REACHED]")
                    self.navigator.stop_robot()
                    self.navigator.reset_path()

                    if self.goal_xy is not None:
                        # ── Clear frontier cells around reached goal ──────────────
                        gix = int((self.goal_xy[0] - (self.gridmap.origin[0] - self.gridmap.map_size/2))
                                  / self.gridmap.cell_size)
                        giy = int((self.goal_xy[1] - (self.gridmap.origin[1] - self.gridmap.map_size/2))
                                  / self.gridmap.cell_size)
                        pad = 3
                        for di in range(-pad, pad+1):
                            for dj in range(-pad, pad+1):
                                ii, jj = giy+di, gix+dj
                                if 0 <= ii < self.gridmap.n_cells and 0 <= jj < self.gridmap.n_cells:
                                    self.gridmap.frontier[ii, jj] = 0
                                    self.gridmap.visited[ii, jj]  = 1

                        # ── Update visited blacklist ──────────────────────────────
                        self._visited_goals.append(self.goal_xy.copy())
                        if len(self._visited_goals) > 50:
                            self._visited_goals.pop(0)

                        # ── Notify ExplorationGraph ───────────────────────────────
                        if self.region is not None:
                            self.exploration_graph.mark_goal_reached(
                                self.region.version, self.goal_xy
                            )

                    #self.goal_xy   = None
                    self.path      = None
                    self._remote_door_waypoints = []       # clear any leftover door chain
                    self._remote_target_version = None
                    self._start_goal_dwell()
                    if not self.dwell_active:
                        self.goal_xy = None
                        self.nav_state = "IDLE"

                else:
                    if self.goal_xy is None:       # ← double-check before path gen
                        self.nav_state = "IDLE"
                        return
                    # Continuously refresh the path as the network improves
                    self._launch_path_gen(robot_xy, self.goal_xy)
                    if self.latest_candidate_path is not None:
                        self.path                  = self.latest_candidate_path
                        self.latest_candidate_path = None
                        self.navigator.reset_path()
                    if self.path is not None:
                        self.navigator.drive_path(self.path)
            
            elif self.nav_state == "DWELL":
                self.navigator.stop_robot()
                self._tick_goal_dwell(world_points_hit, robot_xy)

        # ── Composite plot ────────────────────────────────────────────────
        if self._composite_step % 20 == 0:
            try:
                self.save_composite_plot()
                self.save_occupancy_snapshot()
            except Exception as e:
                self.get_logger().warn(f"[COMPOSITE PLOT] {e}")
        self._composite_step += 1

        # ── Always publish (every tick, every mode) ───────────────────────
        self.map_publisher.publish_maps()
        self.markers.publish_goal_marker()
        self.markers.publish_frontier_markers()
        self.publish_field_path(self.path)

    def save_composite_plot(self):
        """
        plot2-style composite:
          - world-scale background (unknown / visited / obstacles)
          - per-region travel-time contours that PERSIST via a numeric cache (no translucent overlays)
          - exploration graph (nodes/edges/doors)
          - ONLY the latest saved frontiers per region (not the entire bank)
        """
        from matplotlib.lines import Line2D
        os.makedirs(self.folder, exist_ok=True)
        out_dir = os.path.join(self.folder, "plots2")
        os.makedirs(out_dir, exist_ok=True)

        # World-map bounds (gridmap is world-sized)
        b_map = [
            [self.gridmap.origin[0] - self.gridmap.map_size / 2,
             self.gridmap.origin[1] - self.gridmap.map_size / 2],
            [self.gridmap.origin[0] + self.gridmap.map_size / 2,
             self.gridmap.origin[1] + self.gridmap.map_size / 2],
        ]

        fig, ax = plt.subplots(1, 1, figsize=(14, 14))

        # ── Background: visited=dark green, obstacle=black, unknown=dark blue ──
        data = self.gridmap.data
        vis  = self.gridmap.visited
        n    = self.gridmap.n_cells
        cell = self.gridmap.cell_size

        bg = np.zeros((n, n, 3), dtype=np.uint8)
        bg[:, :] = [10, 10, 80]        # unknown
        bg[vis == 1]  = [220, 220, 40]     # visited free
        bg[data == 1] = [0, 0, 0]      # obstacles

        extent = [b_map[0][0], b_map[1][0], b_map[0][1], b_map[1][1]]
        ax.imshow(bg, origin='lower', extent=extent, zorder=0, aspect='auto')

        # ── Persistent numeric cache for region fields ─────────────────────────
        # Cache: version -> dict(X_w, Y_w, TT_masked, extent, src_world, step, epoch)
        if not hasattr(self, "_region_field_cache"):
            self._region_field_cache = {}

        # Tuning
        GRID_N = 90                 # resolution of per-region TT grid
        CONTOUR_LEVELS = 40         # number of contour levels
        FRONTIER_LAST_K = 30        # only show last K saved frontiers per region
        CACHE_UPDATE_EVERY = 1      # update active region every step (keep expanding field visible)

        def _build_region_mask(b, grid_n=GRID_N):
            """Mask out cells outside visited/obstacle footprint (same logic as your original)."""
            mask = np.zeros((grid_n, grid_n), dtype=bool)
            for i in range(grid_n):
                for j in range(grid_n):
                    wx = b[0, 0] + (i + 0.5) / grid_n * (b[1, 0] - b[0, 0])
                    wy = b[0, 1] + (j + 0.5) / grid_n * (b[1, 1] - b[0, 1])
                    gx = int((wx - b_map[0][0]) / cell)
                    gy = int((wy - b_map[0][1]) / cell)
                    if 0 <= gx < n and 0 <= gy < n:
                        mask[j, i] = (vis[gy, gx] == 1 or data[gy, gx] == 1)
            return mask

        def _compute_tt_grid(region, src_world, b, mask, grid_n=GRID_N):
            """Compute TT(x0=src, x1=grid) over region bound and apply mask => TT_masked."""
            xmin, ymin = float(b[0, 0]), float(b[0, 1])
            xmax, ymax = float(b[1, 0]), float(b[1, 1])

            xs = np.linspace(xmin, xmax, grid_n)
            ys = np.linspace(ymin, ymax, grid_n)
            X_w, Y_w = np.meshgrid(xs, ys)

            # world -> normalized using region bound
            sx = (xmax - xmin) + 1e-9
            sy = (ymax - ymin) + 1e-9
            X_n = (X_w - xmin) / sx - 0.5
            Y_n = (Y_w - ymin) / sy - 0.5

            src_norm = region.normalize(src_world.reshape(1, 3))[0]  # (3,)

            XP = np.zeros((X_w.size, 2 * region.dim), dtype=np.float32)
            XP[:, :region.dim] = src_norm
            XP[:, region.dim + 0] = X_n.reshape(-1)
            XP[:, region.dim + 1] = Y_n.reshape(-1)
            XP[:, region.dim + 2] = src_norm[2]

            XP_t = torch.tensor(XP, dtype=torch.float32, device=region.device)

            with torch.no_grad():
                TT_t = region.function.TravelTimes(XP_t)
                if TT_t.ndim == 2 and TT_t.shape[1] == 1:
                    TT_t = TT_t[:, 0]
                TT = TT_t.detach().cpu().numpy().reshape(X_w.shape)

            TT_masked = TT.copy()
            TT_masked[~mask] = np.nan

            return X_w, Y_w, TT_masked

        def _pick_src_world(region, b):
            """Stable source for plotting TT contours."""
            # Active region uses robot position if available
            if self.region is not None and region.version == self.region.version and self.robot_pos is not None:
                return np.array([self.robot_pos[0], self.robot_pos[1], 0.0], dtype=np.float32)

            # Inactive: prefer cached source if present, else region center
            cached = self._region_field_cache.get(region.version)
            if cached is not None and "src_world" in cached:
                return cached["src_world"]

            rc = (b[0] + b[1]) / 2.0
            return np.array([float(rc[0]), float(rc[1]), 0.0], dtype=np.float32)

        # ── Update cache for the active region (so its field "expands" live) ──
        if self.region is not None and (self._composite_step % CACHE_UPDATE_EVERY == 0):
            try:
                reg = self.region
                if reg.function is not None:
                    b = reg.bound.detach().cpu().numpy()
                    mask = _build_region_mask(b, grid_n=GRID_N)
                    src_world = _pick_src_world(reg, b)
                    X_w, Y_w, TT_masked = _compute_tt_grid(reg, src_world, b, mask, grid_n=GRID_N)
                    self._region_field_cache[reg.version] = dict(
                        X_w=X_w, Y_w=Y_w, TT=TT_masked,
                        extent=[b[0, 0], b[1, 0], b[0, 1], b[1, 1]],
                        mask=mask, src_world=src_world,
                        step=int(self._composite_step),
                        epoch=int(getattr(reg, "epoch", 0)),
                    )
            except Exception as e:
                self.get_logger().warn(f"[COMPOSITE] active region cache update failed: {e}")

        # ── Ensure each region has at least one cached snapshot (persist) ─────
        for region in self.all_regions.values():
            if region.version in self._region_field_cache:
                continue
            # Only cache if model exists right now; otherwise it will remain missing until loaded once
            if region.function is None:
                continue
            try:
                b = region.bound.detach().cpu().numpy()
                mask = _build_region_mask(b, grid_n=GRID_N)
                src_world = _pick_src_world(region, b)
                X_w, Y_w, TT_masked = _compute_tt_grid(region, src_world, b, mask, grid_n=GRID_N)
                self._region_field_cache[region.version] = dict(
                    X_w=X_w, Y_w=Y_w, TT=TT_masked,
                    extent=[b[0, 0], b[1, 0], b[0, 1], b[1, 1]],
                    mask=mask, src_world=src_world,
                    step=int(self._composite_step),
                    epoch=int(getattr(region, "epoch", 0)),
                )
            except Exception as e:
                self.get_logger().warn(f"[COMPOSITE] region {region.version} initial cache failed: {e}")

        # ── Draw persistent contours for all cached regions ───────────────────
        for region in self.all_regions.values():
            b = region.bound.detach().cpu().numpy()

            cached = self._region_field_cache.get(region.version, None)
            if cached is not None:
                X_w = cached["X_w"]
                Y_w = cached["Y_w"]
                TT  = cached["TT"]

                if np.isfinite(TT).any():
                    ax.contour(
                        X_w, Y_w, TT,
                        levels=CONTOUR_LEVELS,
                        cmap="gist_heat",
                        linewidths=0.65,
                        alpha=0.9,
                        zorder=2
                    )

            # ── Region bounding box (always) ─────────────────────────────────
            rect_w = float(b[1, 0] - b[0, 0])
            rect_h = float(b[1, 1] - b[0, 1])
            is_active = (self.region is not None and region.version == self.region.version)
            edge_color = 'yellow' if is_active else 'gray'
            lw = 2.0 if is_active else 0.8

            rect = plt.Rectangle(
                (float(b[0, 0]), float(b[0, 1])), rect_w, rect_h,
                linewidth=lw, edgecolor=edge_color, facecolor='none', zorder=4
            )
            ax.add_patch(rect)
            ax.text(
                float(b[0, 0]) + 0.02, float(b[1, 1]) - 0.15,
                f"R{region.version}{'*' if is_active else ''}",
                color=edge_color, fontsize=7, zorder=5
            )

        # ── ExplorationGraph overlay: edges + door waypoints ─────────────────
        for v1, v2, edata in self.exploration_graph.graph.edges(data=True):
            rec1 = self.exploration_graph.records.get(v1)
            rec2 = self.exploration_graph.records.get(v2)
            if rec1 is None or rec2 is None:
                continue

            c1 = rec1.center
            c2 = rec2.center

            door_xys = edata.get('door_xys', [edata.get('door_xy')])
            door_xys = [d for d in door_xys if d is not None]

            if door_xys:
                for door_xy in door_xys:
                    xs = [c1[0], door_xy[0], c2[0]]
                    ys = [c1[1], door_xy[1], c2[1]]
                    ax.plot(xs, ys, color='cyan', linewidth=1.2,
                            linestyle='--', alpha=0.7, zorder=6)
                    ax.scatter(door_xy[0], door_xy[1],
                               c='cyan', s=40, marker='D',
                               zorder=7, alpha=0.9)
            else:
                ax.plot([c1[0], c2[0]], [c1[1], c2[1]],
                        color='cyan', linewidth=1.0,
                        linestyle='--', alpha=0.6, zorder=6)

        # ── ExplorationGraph nodes (region centers) ──────────────────────────
        for ver, rec in self.exploration_graph.records.items():
            c = rec.center
            has_f = rec.has_frontiers()
            node_color = 'lime' if has_f else 'red'
            ax.scatter(c[0], c[1], c=node_color, s=120,
                       marker='o', zorder=8, edgecolors='white', linewidths=1.0)
            ax.text(c[0] + 0.05, c[1] + 0.05,
                    f"R{ver}\n({len(rec.saved_frontiers)}f)",
                    color='white', fontsize=6, zorder=9,
                    bbox=dict(boxstyle='round,pad=0.15', fc='black', alpha=0.5))

        # ── Latest saved frontier points per region (NOT all) ─────────────────
        for ver, rec in self.exploration_graph.records.items():
            if not rec.saved_frontiers:
                continue
            pts = rec.saved_frontiers[-FRONTIER_LAST_K:]  # latest K (append order)
            for fp in pts:
                ax.scatter(fp[0], fp[1], c='orange', s=14,
                           marker='.', zorder=7, alpha=0.85)

        # ── Robot, trajectory, goal ─────────────────────────────────────────
        if self.robot_pos is not None:
            ax.scatter(self.robot_pos[0], self.robot_pos[1],
                       c='red', s=100, zorder=11, marker='o', label='Robot')

        if len(self._traj_history) > 1:
            traj = np.array(self._traj_history)
            ax.plot(traj[:, 0], traj[:, 1], 'm-', linewidth=1.5, zorder=9)

        if self.goal_xy is not None:
            ax.scatter(self.goal_xy[0], self.goal_xy[1],
                       c='lime', s=180, zorder=11, marker='*', label='Goal')

        # ── Active inter-region door route ───────────────────────────────────
        if getattr(self, "_remote_door_waypoints", None):
            if len(self._remote_door_waypoints) > 0:
                wps = np.array(self._remote_door_waypoints)
                ax.plot(wps[:, 0], wps[:, 1], color='magenta',
                        linewidth=2.0, linestyle='-', zorder=10, label='Door route')
                ax.scatter(wps[:, 0], wps[:, 1], c='magenta',
                           s=80, marker='D', zorder=11)

        # ── Legend ──────────────────────────────────────────────────────────
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='lime',
                   markersize=8, label='Region (has frontiers)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                   markersize=8, label='Region (exhausted)'),
            Line2D([0], [0], marker='D', color='w', markerfacecolor='cyan',
                   markersize=7, label='Door waypoint'),
            Line2D([0], [0], color='cyan', linestyle='--', label='Graph edge'),
            Line2D([0], [0], marker='.', color='w', markerfacecolor='orange',
                   markersize=6, label=f'Latest {FRONTIER_LAST_K} frontiers'),
            Line2D([0], [0], marker='*', color='w', markerfacecolor='lime',
                   markersize=10, label='Current goal'),
            Line2D([0], [0], color='magenta', label='Active door route'),
        ]
        ax.legend(handles=legend_elements, loc='upper right',
                  fontsize=7, framealpha=0.7)

        ax.set_xlim(b_map[0][0], b_map[1][0])
        ax.set_ylim(b_map[0][1], b_map[1][1])
        ax.set_aspect('equal')
        ax.set_title(
            f"Step {self._composite_step}  |  regions={len(self.all_regions)}"
            f"  doors={len(self.gridmap.doors)}"
            f"  mode={self.mode}/{self.nav_state}"
        )

        fig.savefig(os.path.join(out_dir, f"plot_step_{self._composite_step}.png"),
                    dpi=120, bbox_inches='tight')
        fig.savefig(os.path.join(self.folder, "plot2.png"), bbox_inches='tight')
        plt.close(fig)

    def save_occupancy_snapshot(self, tag=None):
        """
        Saves occupancy grid + visited + frontier as image and numpy.
        """
        out_dir = os.path.join(self.folder, "occupancy_snapshots")
        os.makedirs(out_dir, exist_ok=True)

        if tag is None:
            tag = f"step_{self._composite_step:06d}"

        data = self.gridmap.data.copy()
        visited = self.gridmap.visited.copy()
        frontier = self.gridmap.frontier.copy()

        n = self.gridmap.n_cells

        # ---- Create visualization image ----
        img = np.zeros((n, n, 3), dtype=np.uint8)

        # unknown
        img[:, :] = [10, 10, 80]

        # visited free
        img[visited == 1] = [200, 200, 0]   # yellow (clearer than dark green)

        # occupied
        img[data == 1] = [0, 0, 0]

        # frontier overlay (optional highlight)
        img[frontier == 1] = [255, 140, 0]  # orange

        plt.figure(figsize=(8, 8))
        plt.imshow(img, origin="lower")
        plt.title(f"Occupancy Snapshot {tag}")
        plt.axis("off")
        plt.savefig(os.path.join(out_dir, f"{tag}.png"), dpi=150, bbox_inches="tight")
        plt.close()

        # ---- Save raw arrays ----
        np.savez_compressed(
            os.path.join(out_dir, f"{tag}.npz"),
            occupancy=data,
            visited=visited,
            frontier=frontier
        )

        self.get_logger().info(f"[OCC SNAPSHOT SAVED] {tag}")

def main():
    rclpy.init()
    node = GridmapNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
import rclpy.time
import torch
from scipy.spatial.transform import Rotation
import signal
import threading
from typing import List

import numpy as np
import math
import tf2_ros
from tf2_ros import TransformBroadcaster
import tf2_geometry_msgs

import sensor_msgs_py.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from geometry_msgs.msg import TransformStamped, Twist, Point, PointStamped, PoseStamped, PoseWithCovarianceStamped
from visualization_msgs.msg import Marker

from mntfields.mntfields_ros.sensing.lidar_adapter import Lidar2DAdapter
from mntfields.mntfields_ros.mapping.map_publisher import MapPublisher 
from mntfields.mntfields_ros.mapping.markers import MarkerPublisher
from mntfields.mntfields_ros.mapping.navigation import Navigator
from mntfields.mntfields_ros.mapping.exploration_graph import ExplorationGraph

from mntfields.mntfields_core.mapping.lidar_gridmap import LidarGridmap
from mntfields.mntfields_core.core.region import Region
from mntfields.mntfields_core.core.model_base import Model

from collections import deque
import threading

import os
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg", force=True)
from matplotlib import cm, colors
import matplotlib.pyplot as plt


REGION_RADIUS = 3.5
REGION_EXIT_MARGIN = 0.4 #was 0.4
REGION_MIN_OVERLAP = 0.4 #was 0.4
MAX_RANGE = 3.2
WARMUP_EPOCHS = 6
#EXPLORATION_RANGE = 2.5   # used for raycasting / visited marking

#print("NEW CODE")
class DummyRenderer:
    def __init__(self):
        pass

    def render(self, *args, **kwargs):
        return None

class DummyModel:
    def __init__(self):
        self.folder = "/home/mayank/turtlebot3_ws/mntfields_model"
        self.renderer = "cuda:0"   # IMPORTANT: must be string device
        self.h = 0.0

class FakePCD:
    def __init__(self, pts):
        self.points = pts
        self.normals = np.zeros_like(pts)

class GridmapNode(Node):
    def __init__(self):
        super().__init__('mntfields_gridmap')
        self.folder = os.path.join("/home/mayank/turtlebot3_ws/mntfields_model/mntfields_output")
        os.makedirs(self.folder, exist_ok=True)

        self.train_queue = deque(maxlen=4)   # keep last few frames only
        self.train_lock = threading.Lock()
        self.train_running = False
        self.declare_parameter('region_min_epochs', 10)  # train at least 20 times before 
        self._traj_history = []
        self._composite_step = 0

        # Tune these
        self.declare_parameter('train_timer_sec', 0.5)      # training loop cadence
        self.declare_parameter('train_samples', 5000)       # n_samples to get_data_from_lidar
        self.declare_parameter('train_epoch', 4)            # epoch passed into region.train_core
        self.declare_parameter('train_after_region', True)  # toggle
        self.radius = REGION_RADIUS

        # Start a separate timer JUST for training
        train_dt = float(self.get_parameter('train_timer_sec').value)
        self.create_timer(train_dt, self.train_tick)
        
        self.declare_parameter('n_cells', 400)
        self.declare_parameter('near_dist', 2.0)
        self.declare_parameter('map_size', 40.0)  # Add this

        n_cells = self.get_parameter('n_cells').value
        near_dist = self.get_parameter('near_dist').value
        map_size = self.get_parameter('map_size').value

        self.map_to_odom = None
        self.origin_initialized = False
        self.set_parameters([rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)])
        # --- exploration / control params ---
        self.declare_parameter('goal_reached_dist', 0.20)
        self.declare_parameter('yaw_tol', 0.25)          # rad
        self.declare_parameter('kp_yaw', 0.8)
        self.declare_parameter('kp_lin', 0.25)
        self.declare_parameter('max_lin', 0.15)          # m/s
        self.declare_parameter('max_ang', 0.8)           # rad/s
        self.declare_parameter('frontier_min_cells', 2)
        self.declare_parameter('goal_cooldown_sec', 2.0)

        self.path = None        #these 4 lines are for the path holding for stable paths - latest diff
        self.path_last_update = self.get_clock().now()
        self.path_update_period = 2.0   # seconds
        self.path_index = 0

        self.goal_xy = None
        self.last_goal_time = self.get_clock().now()
        self.mode = "BOOTSTRAP"   # BOOTSTRAP -> EXPLORE
        #--diff
        self.warmup_frames_needed = 5
        self.warmup_epochs_needed = WARMUP_EPOCHS
        self.warmup_done = False
        self.warmup_start_epoch = 0
        self.nav_state = "IDLE"
        self.wait_start_time = None
        self.wait_duration = 3.0
        self.latest_candidate_path = None
        #--
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        self.navigator = Navigator(self)
        self.map_publisher = MapPublisher(self)
        self.markers = MarkerPublisher(self)

        self.robot_pos = None
        self.robot_yaw = 0.0
        self.latest_cloud = None

        self.all_regions: dict = {}
        self.region_initialized = False
        self.bootstrap_frames = 0
        self.bootstrap_threshold = 10 # frames before region starts
        
        self.adapter = Lidar2DAdapter()

        # Traversability map (1 = free, 0 = blocked)
        self.trav_map = np.ones((n_cells, n_cells), dtype=np.uint8)
        self.gridmap = LidarGridmap(
            trav_map=self.trav_map,
            regions=self.all_regions,#{}
            n_cells=n_cells,
            near_dist=2.0,
            map_size=map_size,           # Add this
            origin=(0.0, 0.0), #(origin_x, origin_y)  # Add this
            room_seg_config=(7,2),
        )
        self.origin_initialized = False
        self.declare_parameter('occ_height', 0.3)
        self.occ_height = float(self.get_parameter('occ_height').value)
        self.occ_height_tol = 0.01

        self.create_subscription(PointCloud2,'/lidar3d_points',self.lidar_cb,10)
        #self.create_subscription(PoseWithCovarianceStamped,'/amcl_pose',self.pose_cb,5) - will be used for a built map
        self.create_subscription(Odometry, '/odom', self.odom_cb, 10)

        self.occ_pub = self.create_publisher(OccupancyGrid,'/mntfields/occ_map',5)
        self.frontier_pub = self.create_publisher(OccupancyGrid,'/mntfields/frontier',5)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.goal_marker_pub = self.create_publisher(Marker, '/mntfields/goal_marker', 10)
        self.frontier_marker_pub = self.create_publisher(Marker, '/mntfields/frontier_markers', 10)
        self.field_path_pub = self.create_publisher(Path, '/field_path', 10)

        self.create_timer(0.2, self.update)

        self.model = DummyModel()
        self.region = None

        self.exploration_graph = ExplorationGraph() #graph diff
        self._remote_door_waypoints: List[np.ndarray] = []  # door waypoints for inter-region travel
        self._remote_target_version: Optional[int] = None
        self._goal_blacklist_radius = 0.5 #was 0.3
        self._visited_goals: List[np.ndarray] = []

        self.train_counter = 0
        self.train_every_n_frames = 1

        self._map_update_counter = 0 #claude difff
        self._path_gen_running   = False

        # ── Goal-dwell training burst ─────────────────────────────────────
        self.declare_parameter('dwell_frames_needed', 6)    # frames to collect at goal
        self.declare_parameter('dwell_epochs',        3)    # epochs to train before resuming
        self.dwell_active       = False
        self.dwell_frames       = []                        # accumulated during dwell
        self.dwell_frames_count = 0
        self.dwell_epoch_start  = 0
        self.dwell_phase        = "COLLECT"  # "COLLECT" | "TRAIN" | "DONE"


    def lidar_cb(self, msg):
        self.latest_cloud = msg
        if not hasattr(self, "_last_lidar_log"):
            self._last_lidar_log = self.get_clock().now()
            #self.get_logger().info("Received first LiDAR scan")
            return
        now = self.get_clock().now()
        if (now - self._last_lidar_log).nanoseconds > 2e9:  # 2 seconds
            #self.get_logger().info("Receiving LiDAR scans...")
            self._last_lidar_log = now
    
    def odom_cb(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation

        self.robot_pos = np.array([p.x, p.y, p.z], dtype=np.float32)
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.robot_yaw = np.arctan2(siny_cosp, cosy_cosp)

        # Lock map once, using the *initial* odom pose
        if not self.origin_initialized:
            self.gridmap.origin = np.array([p.x, p.y], dtype=np.float32)
            self.origin_initialized = True

    def init_region(self, robot_pose):
        ''' Initializes a region around the robot and registers it. '''

        center_x = robot_pose[0]
        center_y = robot_pose[1]
        radius = self.radius

        bound = torch.tensor([
            [center_x - radius, center_y - radius, -0.1],
            [center_x + radius, center_y + radius,  0.1],
        ], dtype=torch.float32)

        new_region = Region(
            model=self.model,
            bound=bound,
            init_network=True,
            eval=False,)
        new_region.gridmap = self.gridmap
        self.all_regions[new_region.version] = new_region

        if self.region is not None and self.region.all_lidar is not None:
            last = self.region.all_lidar[-5:]
            new_region.all_lidar = last.copy()
            new_region.lidar_obs = None
            new_region.kd_tree = None
        else:
            new_region.all_lidar = []
        self.region = new_region
        self.region_initialized = True
        self.get_logger().info(
            f"[REGION {new_region.version} INIT] "
            f"center=({center_x:.2f}, {center_y:.2f}) "
            f"bound=[({bound[0,0]:.2f},{bound[0,1]:.2f}) → "
            f"({bound[1,0]:.2f},{bound[1,1]:.2f})]")
        # ── Re-enter WARMUP for every new region ─────────────────────────
        self.mode               = "WARMUP"
        self.warmup_start_epoch = int(self.region.epoch)
        self.goal_xy            = None
        self.path               = None
        self.nav_state          = "IDLE"
        self.latest_candidate_path = None
        self._path_gen_running  = False
        self.get_logger().info(f"[FLOW] Region {new_region.version} → entering WARMUP")

    def start_region_warmup(self):
        self.mode = "WARMUP"
        self.region_warmup_start_epoch = self.region.epoch if self.region else 0
        self.navigator.stop_robot()
        self.get_logger().info(f"[WARMUP] Starting region {self.region.version}")

    def feed_region(self, world_pts, robot_pose):
        if len(world_pts) < 50:
            self.get_logger().warn(f"Too few points: {len(world_pts)}")
            return
        try:
            pts = torch.tensor(world_pts, dtype=torch.float32)
            pos = torch.tensor([robot_pose[0], robot_pose[1], 0.0], dtype=torch.float32)
            ##print("WORLD PTS MIN/MAX:", world_pts.min(), world_pts.max())
            n_samples = int(self.get_parameter('train_samples').value)
            epoch = int(self.get_parameter('train_epoch').value)
            data = self.region.get_data_from_lidar(
                pts.cpu().numpy(),
                pos.cpu().numpy(),
                device='cuda:0',
                n_samples=n_samples
            )
            if data is None:
                self.get_logger().warn("get_data_from_lidar returned None")
                return
            points, closest = data
            if points.shape[0] == 0:
                self.get_logger().warn("get_data_from_lidar returned 0 points")
                return

            if data is None:
                self.get_logger().warn("get_data_from_lidar returned None")
                return
            x0 = points[:, :3]
            x1 = points[:, 3:6]
            cp0 = closest[:, :3]
            cp1 = closest[:, 3:6]

            raw = torch.cat([x0, x1, cp0, cp1], dim=1)   # Nx12
            #print("RAW SHAPE:", raw.shape)

            if raw.ndim != 2 or raw.shape[1] != 12:
                self.get_logger().error(f"[DBG] raw has wrong shape: {tuple(raw.shape)} (expected Nx12)")
                return
            if not torch.isfinite(raw).all():
                bad = (~torch.isfinite(raw)).sum().item()
                self.get_logger().error(f"[DBG] raw has non-finite values: count={bad}")
                return
            with torch.no_grad():
                mn = raw.min(dim=0).values.cpu().numpy()
                mx = raw.max(dim=0).values.cpu().numpy()
            frame_data = self.region.calculate_speed_and_normal(raw)
            #frame_data = raw

            if frame_data.ndim != 2:
                self.get_logger().error(f"[DBG] frame_data not 2D: {tuple(frame_data.shape)}")
                return
            if frame_data.shape[1] < 12:
                self.get_logger().error(f"[DBG] frame_data has too few cols: {frame_data.shape[1]} (expected >=12)")
                return
            if not torch.isfinite(frame_data).all():
                bad = (~torch.isfinite(frame_data)).sum().item()
                self.get_logger().error(f"[DBG] frame_data non-finite: count={bad}")
                return
            with torch.no_grad():
                fd_min = frame_data.min().item()
                fd_max = frame_data.max().item()

            y0_col = frame_data[:, 6]
            y1_col = frame_data[:, 7]

            if frame_data.shape[0] == 0:
                self.get_logger().warn("No valid frame data after speed calculation")
                return
            self.region.last_robot_pose = np.array([robot_pose[0], robot_pose[1], 0.0], dtype=np.float32)

            if self.region.function is not None:
                try:
                    k = min(64, frame_data.shape[0])
                    xp_n = frame_data[:k, :6].clone().to(self.region.device)
                    with torch.no_grad():
                        tt = self.region.function.TravelTimes(xp_n).detach().cpu().numpy().reshape(-1)
                    with torch.enable_grad():
                        sp_t = self.region.function.Speed(xp_n)  # tensor
                    sp = sp_t.detach().cpu().numpy().reshape(-1)

                except Exception as e:
                    self.get_logger().error(f"[DBG MODEL] failed: {e}")
            #print(f"[FEED] frame_data shape={frame_data.shape}, " #UNCOMMENT FOR DEBUG ON FRAME DATA - CORRECT FOR NOW
            #      f"y0 range=[{frame_data[:,6].min():.4f}, {frame_data[:,6].max():.4f}]")
            #print(f"[FEED] calling train_core with {frame_data.shape[0]} rows")      
            success = self.region.train_core(epoch=epoch, frame_data=frame_data)
            loss_val = getattr(self.region, 'last_loss', None)
            try:
                test = frame_data[:128, :6].clone().to(self.region.device)
                #test.requires_grad_(True)
                test_n = self.region.normalize_tensor(test)
                with torch.enable_grad():
                    tau, w, coords_in = self.region.function.network.out(test_n)
                    tt = tau[:, 0]
                    sp = self.region.function.Speed(test_n)

                g = torch.autograd.grad(tt.sum(), coords_in, retain_graph=True)[0]
                g0 = g[:, :3]  # gradient w.r.t x0
                g0_norm = torch.sqrt((g0 * g0).sum(dim=1) + 1e-12)
            except Exception as e:
                print("MODEL DEBUG FAILED:", e)

        except Exception as e:
            self.get_logger().error(f"feed_region error: {e}")
            import traceback
            traceback.print_exc()

    def pick_frontier_goal(self):
        frontier = self.gridmap.frontier
        iy, ix = np.where(frontier > 0)
        min_frontier_cells = int(self.get_parameter('frontier_min_cells').value)

        if len(ix) < min_frontier_cells:
            return None
        # ✅ DIRECTLY call pick_reachable_frontier (it handles everything)
        goal = self.pick_reachable_frontier()
        if goal is not None:
            robot = self.robot_pos[:2]
            dist = np.linalg.norm(goal - robot)
        return goal

    def pick_room_aware_goal(self, current_room, rooms, room_graph):
        robot_xy = self.robot_pos[:2]

        # 1. Try live local frontier (with blacklist)
        local = self._frontier_in_room(current_room, rooms)
        if local is not None:
            if (len(self._visited_goals) == 0 or
                    np.linalg.norm(np.array(self._visited_goals) - local[None,:], axis=1).min()
                    > self._goal_blacklist_radius):
                return local

        # 2. Try global live frontier
        live = self.pick_reachable_frontier()
        if live is not None:
            return live

        # 3. Local region has no live frontiers → check ExplorationGraph bank
        if self.region is not None:
            ver = self.region.version
            result = self.exploration_graph.best_remote_frontier(ver, robot_xy)
            if result is not None:
                target_ver, target_frontier, door_waypoints = result
                print(f"[EXPLORATION GRAPH] Routing to region {target_ver} "
                      f"via {len(door_waypoints)} door(s), "
                      f"frontier={target_frontier}")
                self._remote_door_waypoints = door_waypoints
                self._remote_target_version = target_ver
                return target_frontier

        print("[EXPLORATION GRAPH] Fully explored — no frontiers anywhere")
        return None

    def _frontier_in_room(self, room_id, rooms):
        if room_id is None:
            return None
        if room_id not in rooms:
            return None
        room_cells = rooms[room_id]["cells"]
        frontier_pts = []
        for (i, j) in room_cells:
            if self.gridmap.frontier[i, j] == 1:
                x, y = self.gridmap._grid_to_world(i, j)
                frontier_pts.append([x, y])
        if len(frontier_pts) == 0:
            return None
        frontier_pts = np.array(frontier_pts)
        # choose closest to robot
        robot = self.robot_pos[:2]
        dists = np.linalg.norm(frontier_pts - robot[None, :], axis=1)
        idx = np.argmin(dists)
        return frontier_pts[idx] #END DIFF

    def publish_field_path(self, path_xy, frame_id='odom'):
        """
        path_xy: list[np.ndarray] or np.ndarray of shape (N,2)
        """
        if path_xy is None:
            return
        if isinstance(path_xy, list):
            pts = np.asarray(path_xy, dtype=np.float32)
        else:
            pts = np.asarray(path_xy, dtype=np.float32)

        if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 2:
            return

        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id

        poses = []
        for p in pts:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = float(p[0])
            ps.pose.position.y = float(p[1])
            ps.pose.position.z = 0.0
            ps.pose.orientation.w = 1.0
            poses.append(ps)

        msg.poses = poses
        self.field_path_pub.publish(msg)

    def decimate_path(self, path_xy, spacing=0.4):
        pts = np.asarray(path_xy, dtype=np.float32)

        if pts.ndim == 1:
            pts = pts.reshape(1, 2)

        if len(pts) < 2:
            return pts

        new_pts = [pts[0]]

        for i in range(1, len(pts)):
            if np.linalg.norm(pts[i] - new_pts[-1]) >= spacing:
                new_pts.append(pts[i])

        if np.linalg.norm(new_pts[-1] - pts[-1]) > 0.05:
            new_pts.append(pts[-1])

        return np.asarray(new_pts, dtype=np.float32)

    def smooth_path(self, path_xy, window=5):
        if path_xy is None:
            return None

        pts = np.asarray(path_xy, dtype=np.float32)

        if pts.ndim == 1:
            pts = pts.reshape(1, 2)

        if len(pts) < window:
            return pts

        sm = pts.copy()
        k = window // 2

        for i in range(len(pts)):
            a = max(0, i-k)
            b = min(len(pts), i+k+1)
            sm[i] = pts[a:b].mean(axis=0)

        sm[0] = pts[0]
        sm[-1] = pts[-1]
        return sm 

    def pick_lookahead(self, path, robot_xy, L=0.45):
        pts = np.asarray(path)
        r = np.asarray(robot_xy)

        d = np.linalg.norm(pts - r[None,:], axis=1)
        i0 = np.argmin(d)

        acc = 0.0
        i = i0
        while i < len(pts)-1 and acc < L:
            acc += np.linalg.norm(pts[i+1]-pts[i])
            i += 1

        return pts[i]
       #end diff

    def is_path_significantly_different(self, old_path, new_path, threshold=0.25):

        old_path = np.asarray(old_path)
        new_path = np.asarray(new_path)

        if len(old_path) < 2 or len(new_path) < 2:
            return True

        # compare up to min length
        n = min(len(old_path), len(new_path))

        # compare corresponding waypoints
        diffs = np.linalg.norm(old_path[:n] - new_path[:n], axis=1)

        max_diff = np.max(diffs)

        print("Max waypoint diff:", max_diff)

        return max_diff > threshold

    def debug_plot(self, field_waypoint=None):
        data = self.gridmap.data
        visited = self.gridmap.visited

        im = np.ones((*data.shape, 3), dtype=np.uint8) * 255
        im[data > 0.9] = np.array([0, 0, 0])      # walls
        im[visited == 1] = np.array([0, 255, 0])  # visited

        fig, ax = plt.subplots(1,1,figsize=(6,6))
        ax.imshow(im, origin='lower')

        # robot
        if self.robot_pos is not None:
            i,j = self.world_to_grid(self.robot_pos[0], self.robot_pos[1])
            ax.scatter(j,i,c='magenta',s=60,label='robot')

        # goal
        if self.goal_xy is not None:
            i,j = self.world_to_grid(self.goal_xy[0], self.goal_xy[1])
            ax.scatter(j,i,c='red',s=60,label='goal')

        # field waypoint
        if field_waypoint is not None:
            i,j = self.world_to_grid(field_waypoint[0], field_waypoint[1])
            ax.scatter(j,i,c='blue',s=60,label='field wp')
        ax.legend()
        plt.pause(0.01)

    def train_tick(self):
        if self.region is None:
            print(f"[TRAIN_TICK] region=False, running={self.train_running}, "
                  f"queue_len={len(self.train_queue)}, epoch=-1")
            return

        print(f"[TRAIN_TICK] region=True, running={self.train_running}, "
              f"queue_len={len(self.train_queue)}, epoch={self.region.epoch}")

        if self.train_running:
            return  # previous train still going

        with self.train_lock:
            if len(self.train_queue) == 0:
                return
            self.train_running = True     
            world_pts, robot_pose = self.train_queue.pop()
            self.train_queue.clear()

        # ✅ Set BEFORE spawning thread — eliminates race condition
       
        def _do_train():
            try:
                print(f"[TRAIN THREAD] starting feed_region, epoch={self.region.epoch}")
                self.feed_region(world_pts, robot_pose)
                print(f"[TRAIN THREAD] done, epoch={self.region.epoch}")
            except Exception as e:
                self.get_logger().error(f"[TRAIN] failed: {e}")
                import traceback
                traceback.print_exc()
            finally:
                self.train_running = False

        import threading
        t = threading.Thread(target=_do_train, daemon=True)
        t.start()

    def maybe_expand_region(self):
        if self.region is None:
            return

        robot = self.robot_pos[:2]
        bound = self.region.bound.numpy()

        margin = 0.3

        near_edge = (
            abs(robot[0] - bound[0,0]) < margin or
            abs(robot[0] - bound[1,0]) < margin or
            abs(robot[1] - bound[0,1]) < margin or
            abs(robot[1] - bound[1,1]) < margin
        )

        if near_edge:
            new_bound = torch.tensor([
                [bound[0,0]-0.3, bound[0,1]-0.3, -0.1],
                [bound[1,0]+0.3, bound[1,1]+0.3,  0.1],
            ], dtype=torch.float32)

            self.region.set_bound(new_bound)
            self.get_logger().info("[REGION EXPAND]")


    def fit_region_to_lidar(self,
                             expand_thresh: float = 0.30,
                             shrink_margin: float = 0.50,
                             step: float = 0.15,
                             min_wall_span: float = 1.2,
                             max_area_mult: float = 3.5):
        """
        Fit the current region bounding box to ALL accumulated wall cells
        inside the region, not just the current frame's LiDAR hits.

        Source of truth: gridmap.data == 1 cells (confirmed walls, built up
        over the entire robot trajectory in this region).

        Per-edge logic
        ──────────────
        EXPAND  — wall cells exist within expand_thresh of the edge AND they
                  span at least min_wall_span metres along that edge.
                  The span guard filters doorways (~0.9 m wide) from full
                  room walls (typically >1.5 m), so we only expand when we
                  see a real wall, not a narrow opening.

        SHRINK  — no wall cell falls within shrink_margin of the edge,
                  meaning there is confirmed empty space between the last
                  wall and the boundary → pull inward by step.

        LEAVE   — wall cells are present but not pressing on the edge, and
                  the boundary is already a reasonable fit.

        Hard limits
        ───────────
        - Minimum half-size: 1.0 m on each axis.
        - Maximum area: max_area_mult × (2×REGION_RADIUS)² .
        - Robot must stay >= 0.5 m inside every edge after adjustment.
        - Only moves one step per call (smooth adaptation at ~1 Hz).
        """
        if self.region is None or self.robot_pos is None:
            return

        # ── Collect all confirmed wall cells inside the region bound ─────
        b     = self.region.bound.numpy().copy()
        robot = self.robot_pos[:2]

        iy_w, ix_w = np.where(self.gridmap.data == 1)
        if len(ix_w) == 0:
            return

        cell   = self.gridmap.cell_size
        origin = self.gridmap.origin
        half_m = self.gridmap.map_size / 2.0
        wx = (origin[0] - half_m) + (ix_w + 0.5) * cell
        wy = (origin[1] - half_m) + (iy_w + 0.5) * cell
        wall_pts = np.stack([wx, wy], axis=1)

        # Only walls already inside (or very close to) the current bound
        pad = expand_thresh + 0.05
        inside = (
            (wall_pts[:, 0] >= b[0, 0] - pad) & (wall_pts[:, 0] <= b[1, 0] + pad) &
            (wall_pts[:, 1] >= b[0, 1] - pad) & (wall_pts[:, 1] <= b[1, 1] + pad)
        )
        wall_pts = wall_pts[inside]
        if len(wall_pts) < 4:
            return

        MIN_HALF  = 1.0
        MAX_AREA  = max_area_mult * (REGION_RADIUS * 2) ** 2
        ROBOT_PAD = 0.5

        def _try_move(axis, side, delta):
            new_b = b.copy()
            if side == 0:
                new_b[0, axis] = b[0, axis] - delta
            else:
                new_b[1, axis] = b[1, axis] + delta
            if (new_b[1, 0] - new_b[0, 0]) / 2.0 < MIN_HALF:
                return
            if (new_b[1, 1] - new_b[0, 1]) / 2.0 < MIN_HALF:
                return
            if (new_b[1, 0] - new_b[0, 0]) * (new_b[1, 1] - new_b[0, 1]) > MAX_AREA:
                return
            if not (new_b[0, 0] + ROBOT_PAD < robot[0] < new_b[1, 0] - ROBOT_PAD and
                    new_b[0, 1] + ROBOT_PAD < robot[1] < new_b[1, 1] - ROBOT_PAD):
                return
            b[:] = new_b

        # Per-edge analysis using accumulated wall cells
        # For x-edges: only walls whose Y falls within the region Y span
        # For y-edges: only walls whose X falls within the region X span
        y_in = (wall_pts[:, 1] >= b[0, 1]) & (wall_pts[:, 1] <= b[1, 1])
        x_in = (wall_pts[:, 0] >= b[0, 0]) & (wall_pts[:, 0] <= b[1, 0])

        for axis, side, edge, relevant in [
            (0, 0, b[0, 0], wall_pts[y_in]),   # xmin (left)
            (0, 1, b[1, 0], wall_pts[y_in]),   # xmax (right)
            (1, 0, b[0, 1], wall_pts[x_in]),   # ymin (bottom)
            (1, 1, b[1, 1], wall_pts[x_in]),   # ymax (top)
        ]:
            if len(relevant) == 0:
                continue

            coords      = relevant[:, axis]
            perp_coords = relevant[:, 1 - axis]   # perpendicular axis for span

            if side == 0:
                near_mask = coords < edge + expand_thresh
                any_close = (coords < edge + shrink_margin).any()
            else:
                near_mask = coords > edge - expand_thresh
                any_close = (coords > edge - shrink_margin).any()

            if near_mask.any():
                # Measure span of near-edge walls along the perpendicular axis
                near_perp = perp_coords[near_mask]
                span = float(near_perp.max() - near_perp.min())
                if span >= min_wall_span:
                    _try_move(axis, side, step)   # real wall → expand
                # else: narrow hit cluster (doorway / corner) → leave
            elif not any_close:
                _try_move(axis, side, -step)      # open space → shrink

        if not np.allclose(b, self.region.bound.numpy(), atol=1e-4):
            new_bound = torch.tensor(
                [[b[0, 0], b[0, 1], -0.1],
                 [b[1, 0], b[1, 1],  0.1]], dtype=torch.float32)
            self.region.set_bound(new_bound)
            self.get_logger().info(
                f"[FIT REGION] R{self.region.version} "
                f"({b[0,0]:.2f},{b[0,1]:.2f})→({b[1,0]:.2f},{b[1,1]:.2f})"
            )

    def _get_frontier_world_pts(self, max_pts=200):
        """Return frontier cell centres in world coords as (N,2) array."""
        iy, ix = np.where(self.gridmap.frontier > 0)
        if len(ix) == 0:
            return None
        cell   = self.gridmap.cell_size
        origin = self.gridmap.origin
        half   = self.gridmap.map_size / 2.0
        fx = (origin[0] - half) + (ix + 0.5) * cell
        fy = (origin[1] - half) + (iy + 0.5) * cell
        pts = np.stack([fx, fy], axis=1)
        if len(pts) > max_pts:
            pts = pts[np.random.choice(len(pts), max_pts, replace=False)]
        return pts

    def pick_reachable_frontier(self, max_candidates=300):
        frontier = self.gridmap.frontier
        iy, ix = np.where(frontier > 0)
        if len(ix) == 0:
            return None

        cell   = self.gridmap.cell_size
        origin = self.gridmap.origin
        half   = self.gridmap.map_size / 2.0
        fx = (origin[0] - half) + (ix + 0.5) * cell
        fy = (origin[1] - half) + (iy + 0.5) * cell
        cand = np.stack([fx, fy], axis=1)

        robot = self.robot_pos[:2]
        d = np.linalg.norm(cand - robot[None, :], axis=1)
        valid = d < 2.0
        if valid.sum() == 0:
            return None
        cand = cand[valid]
        d    = d[valid]

        # ✅ FIX 1: Only pick frontiers INSIDE the current region bound
        # The field has no knowledge of anything outside the bound
        if self.region is not None:
            in_bound = np.array([self.region.in_bound(c) for c in cand])
            if in_bound.sum() > 0:
                cand = cand[in_bound]
                d    = d[in_bound]
            # If ALL frontiers are outside the bound, fall back to closest-in-bound
            # rather than sending the robot off a cliff
            else:
                return cand[np.argmin(d)]   # geometric fallback

        if len(cand) > max_candidates:
            cand = cand[np.random.choice(len(cand), max_candidates, replace=False)]

        # No field yet → closest
        if self.region is None or self.region.function is None:
            return cand[np.argmin(d)]

        # Field query (all candidates guaranteed inside bound now)
        x0  = np.tile(np.array([robot[0], robot[1], 0.0]), (len(cand), 1))
        x1  = np.concatenate([cand, np.zeros((len(cand), 1))], axis=1)
        xp  = np.concatenate([x0, x1], axis=1)
        xp_t = torch.tensor(xp, dtype=torch.float32)
        xp_n = self.region.normalize_tensor(xp_t).to(self.region.device)

        with torch.no_grad():
            tt = self.region.function.TravelTimes(xp_n).cpu().numpy().reshape(-1)

        ok = np.isfinite(tt)
        if ok.sum() == 0:
            return cand[np.argmin(d)]

        return cand[ok][np.argmin(tt[ok])]

    def maybe_create_new_region(self):
        """
        Called every update() tick.
        If the robot has moved outside the current region's bound, either:
          (a) switch to an existing region that covers the robot, OR
          (b) spawn a new region centred on the robot.
        Old region is unloaded to disk to free GPU memory.
        """
        if self.region is None or self.robot_pos is None:
            return

        robot = self.robot_pos[:2]
        b = self.region.bound.numpy()

        # Is robot still comfortably inside current bound?
        inside = (
            robot[0] > b[0, 0] + REGION_EXIT_MARGIN and
            robot[0] < b[1, 0] - REGION_EXIT_MARGIN and
            robot[1] > b[0, 1] + REGION_EXIT_MARGIN and
            robot[1] < b[1, 1] - REGION_EXIT_MARGIN
        )
        if inside:
            return  # nothing to do

        # --- Robot has left (or is about to leave) the current region ---

        # (a) Check if any existing region already covers the robot
        for ver, reg in self.all_regions.items():
            if ver == self.region.version:
                continue
            if reg.in_bound(robot):
                if reg.loaded:
                    self._switch_region(reg)
                    return
                else:
                    reg.load_from_disk()
                    self._switch_region(reg)
                    return

        # (b) No existing region covers the robot → create a new one
        #     But only if the area has enough visited free space to train on
        visited = self.gridmap.visited
        cell = self.gridmap.cell_size
        origin = self.gridmap.origin
        half = self.gridmap.map_size / 2.0

        # Count visited cells in the proposed new bound
        def count_visited_in_bound(cx, cy, r):
            ix0 = max(0, int((cx - r - (origin[0] - half)) / cell))
            ix1 = min(self.gridmap.n_cells - 1, int((cx + r - (origin[0] - half)) / cell))
            iy0 = max(0, int((cy - r - (origin[1] - half)) / cell))
            iy1 = min(self.gridmap.n_cells - 1, int((cy + r - (origin[1] - half)) / cell))
            region_area = (ix1 - ix0) * (iy1 - iy0)
            if region_area <= 0:
                return 0.0
            visited_count = visited[iy0:iy1, ix0:ix1].sum()
            return float(visited_count) / float(region_area)

        coverage = count_visited_in_bound(robot[0], robot[1], REGION_RADIUS)
        if coverage < REGION_MIN_OVERLAP:
            # Not enough data yet — keep extending current region temporarily
            # by softly expanding its bound rather than spawning premature region
            self._soft_extend_current_region(robot)
            return

        # Enough data: spawn the new region
        self.get_logger().info(
            f"[NEW REGION] Robot at ({robot[0]:.2f},{robot[1]:.2f}) "
            f"left region {self.region.version}. "
            f"Visited coverage={coverage:.2f}. Spawning new region."
        )
        self._unload_current_region()
        self.init_region(self.robot_pos)

    def maybe_merge_regions(self):
        """
        For each confirmed room that spans multiple regions, merge those
        regions into one with a combined bounding box and retrain.
        """
        # Build room → [region_versions] mapping
        room_to_regions: Dict[int, List[int]] = {}
        for ver, region in self.all_regions.items():
            for room in region.rooms:
                if room.id not in room_to_regions:
                    room_to_regions[room.id] = []
                room_to_regions[room.id].append(ver)

        for room_id, versions in room_to_regions.items():
            if len(versions) <= 1:
                continue  # room already fits in one region

            # Only merge if all regions are confirmed (trained enough)
            regions_to_merge = [self.all_regions[v] for v in versions
                                if v in self.all_regions]
            if any(r.epoch < self.warmup_epochs_needed for r in regions_to_merge):
                continue  # wait until all are trained

            self.get_logger().info(
                f"[MERGE] Room {room_id} spans regions {versions} → merging"
            )
            self._merge_regions(regions_to_merge)


    def _merge_regions(self, regions: List):
        """
        Merge a list of regions into one. The merged region gets:
        - Combined bounding box
        - All rooms from all merged regions
        - All training data concatenated
        - Retraining triggered immediately
        """
        # ── Compute combined bound ────────────────────────────────────
        bounds = [r.bound.numpy() for r in regions]
        new_min = np.array([min(b[0, i] for b in bounds) for i in range(3)])
        new_max = np.array([max(b[1, i] for b in bounds) for i in range(3)])
        new_bound = torch.tensor(
            np.stack([new_min, new_max]), dtype=torch.float32
        )

        # ── Check merged region isn't too large ──────────────────────
        w = new_max[0] - new_min[0]
        h = new_max[1] - new_min[1]
        MAX_MERGE_SIZE = REGION_RADIUS * 2 * 2   # 2× normal region size
        if w > MAX_MERGE_SIZE or h > MAX_MERGE_SIZE:
            self.get_logger().warn(
                f"[MERGE SKIPPED] merged bound too large ({w:.1f}×{h:.1f}m)"
            )
            return

        # ── Create merged region ──────────────────────────────────────
        merged = Region(model=self.model, bound=new_bound,
                        init_network=True, eval=False)
        merged.gridmap = self.gridmap

        # Transfer all rooms
        all_rooms = [room for r in regions for room in r.rooms]
        for room in all_rooms:
            merged.add_room(room)   # also sets room.region = merged

        # Concatenate all_lidar from constituent regions
        lidar_chunks = [r.all_lidar for r in regions
                        if r.all_lidar is not None and len(r.all_lidar) > 0]
        if lidar_chunks:
            merged.all_lidar = np.concatenate(lidar_chunks, axis=0)

        # ── Remove old regions from all_regions and ExplorationGraph ─
        old_versions = [r.version for r in regions]
        for ver in old_versions:
            if ver in self.all_regions:
                try:
                    self.all_regions[ver].unload_to_disk()
                except Exception:
                    pass
                del self.all_regions[ver]
            if ver in self.exploration_graph.records:
                del self.exploration_graph.records[ver]
            if ver in self.exploration_graph.graph:
                self.exploration_graph.graph.remove_node(ver)

        # ── Register merged region ────────────────────────────────────
        self.all_regions[merged.version] = merged
        center = np.array(merged.center[:2], dtype=np.float32)
        self.exploration_graph.ensure_region(merged.version, center)

        # Rebuild spatial edges for merged region
        for v2, reg2 in self.all_regions.items():
            if v2 == merged.version:
                continue
            b1 = new_bound.numpy()
            b2 = reg2.bound.numpy()
            x_gap = max(0.0, max(b1[0,0], b2[0,0]) - min(b1[1,0], b2[1,0]))
            y_gap = max(0.0, max(b1[0,1], b2[0,1]) - min(b1[1,1], b2[1,1]))
            if x_gap < 1.0 and y_gap < 1.0:
                mx = (max(b1[0,0], b2[0,0]) + min(b1[1,0], b2[1,0])) / 2.0
                my = (max(b1[0,1], b2[0,1]) + min(b1[1,1], b2[1,1])) / 2.0
                self.exploration_graph.add_spatial_edge(
                    merged.version, v2, np.array([mx, my], dtype=np.float32)
                )

        # ── If robot was in one of the merged regions, switch to merged ─
        if self.region is not None and self.region.version in old_versions:
            self.region = merged
            self.get_logger().info(
                f"[MERGE] Active region was merged → now region {merged.version}"
            )

        # ── Trigger retraining on merged region ──────────────────────
        self.mode               = "WARMUP"
        self.warmup_start_epoch = int(merged.epoch)
        self.goal_xy            = None
        self.nav_state          = "IDLE"
        self.get_logger().info(
            f"[MERGE] Region {merged.version} created "
            f"(merged {old_versions}) → WARMUP"
        )


    #def _switch_region(self, new_region: 'Region'):
    #    old_ver = self.region.version
#
    #    # keep last scans
    #    if self.region is not None and self.region.all_lidar is not None:
    #        new_region.all_lidar = self.region.all_lidar[-5:].copy()
    #        new_region.lidar_obs = None
    #        new_region.kd_tree = None
#
    #    self._unload_current_region()
    #    self.region = new_region
    #    self.get_logger().info(
    #        f"[REGION SWITCH] {old_ver} → {new_region.version}"
    #    )
    #    self.goal_xy = None
    #    self.last_goal_time = self.get_clock().now()
    #    self.gridmap.frontier[:] = 0
    def _switch_region(self, new_region: 'Region'):
        old_ver = self.region.version

        if self.region is not None and self.region.all_lidar is not None:
            new_region.all_lidar = self.region.all_lidar[-5:].copy()
            new_region.lidar_obs = None
            new_region.kd_tree   = None

        self._unload_current_region()

        # Flush stale training frames from the old region before switching
        with self.train_lock:
            self.train_queue.clear()
        self.train_running = False   # abort any in-flight train for old region

        self.region = new_region
        self.get_logger().info(f"[REGION SWITCH] {old_ver} → {new_region.version}")
        self.goal_xy        = None
        self.last_goal_time = self.get_clock().now()
        self.gridmap.frontier[:] = 0


    def _unload_current_region(self):
        """Save current region weights to disk and free GPU memory."""
        if self.region is None:
            return
        try:
            self.region.save_model()          # saves checkpoint to region folder
            self.region.unload_to_disk()      # releases GPU tensors
            self.get_logger().info(
                f"[REGION {self.region.version} UNLOADED]"
            )
        except Exception as e:
            self.get_logger().error(f"[UNLOAD ERROR] {e}")

    def _soft_extend_current_region(self, robot):
        """
        Softly expand region a bit so we don't immediately spawn a new one,
        BUT stop expanding once region gets too large.
        """

        b = self.region.bound.numpy().copy()
        pad = 0.3

        # --- proposed new bound ---
        new_min_x = min(b[0, 0], robot[0] - REGION_RADIUS - pad)
        new_min_y = min(b[0, 1], robot[1] - REGION_RADIUS - pad)
        new_max_x = max(b[1, 0], robot[0] + REGION_RADIUS + pad)
        new_max_y = max(b[1, 1], robot[1] + REGION_RADIUS + pad)

        # --- current size ---
        cur_w = b[1, 0] - b[0, 0]
        cur_h = b[1, 1] - b[0, 1]

        # --- new size if applied ---
        new_w = new_max_x - new_min_x
        new_h = new_max_y - new_min_y

        # 🧠 IMPORTANT: cap region size
        MAX_SIZE = REGION_RADIUS * 2.0   # ≈ 2.5m if radius=1.0

        if new_w > MAX_SIZE or new_h > MAX_SIZE:
            # Stop expanding → allow new region to spawn
            self.get_logger().info(
                f"[SOFT EXTEND BLOCKED] region already large "
                f"(w={cur_w:.2f}, h={cur_h:.2f})"
            )
            return

        # --- apply extension ---
        b[0, 0] = new_min_x
        b[0, 1] = new_min_y
        b[1, 0] = new_max_x
        b[1, 1] = new_max_y

        new_bound = torch.tensor(b, dtype=torch.float32)
        self.region.set_bound(new_bound)

        self.get_logger().info(
            f"[SOFT EXTEND] new bound: "
            f"({b[0,0]:.2f},{b[0,1]:.2f}) → ({b[1,0]:.2f},{b[1,1]:.2f})"
        )

    def _launch_path_gen(self, robot_xy, goal_xy): #claude diff
        """Offload generate_field_path to a background thread so update() never blocks."""
        if robot_xy is None or goal_xy is None:          # ← add this guard
            return
        if self.region is None or self.region.function is None:
            return
        if getattr(self, '_path_gen_running', False):
            return  # already running, skip
        self._path_gen_running = True
        robot_xy = robot_xy.copy()
        goal_xy  = goal_xy.copy()

        def _worker():
            try:
                raw_path = self.region.generate_field_path(
                    robot_xy, goal_xy, step_size=0.10, max_steps=200
                )
                if raw_path is not None:
                    raw_path = np.asarray(raw_path, dtype=np.float32)
                    if len(raw_path) > 1:
                        decimate = self.decimate_path(raw_path, spacing=0.2)
                        smooth   = self.smooth_path(decimate, window=3)
                        self.latest_candidate_path = np.asarray(smooth, dtype=np.float32)
            except Exception as e:
                self.get_logger().warn(f"[PATH GEN] {e}")
            finally:
                self._path_gen_running = False
        threading.Thread(target=_worker, daemon=True).start() #end claude diff

    def _maybe_create_new_region_guarded(self):
        """
        Only allow region creation when robot is NOT actively navigating.
        """
        if self.region is None:
            self.maybe_create_new_region()
            return

        if self.mode != "EXPLORE":
            self.maybe_create_new_region()
            return

        if self.goal_xy is None:
            self.maybe_create_new_region()
            return

        # Actively navigating → block
        if self.nav_state in ("STABILIZING", "FOLLOWING"):
            robot_xy = self.robot_pos[:2]
            d = np.linalg.norm(robot_xy - self.goal_xy)
            if d < 0.20:
                self.maybe_create_new_region()
            else:
                return  # still en-route, block creation
        else:
            # IDLE or DWELL → both are safe, robot is stationary
            self.maybe_create_new_region()

    def _update_region_graph_via_free_cells(self):
        """
        Build/update ExplorationGraph edges between regions using cells that:
          - are free/traversable (gridmap.data == 0, not occupied)
          - have been visited by the robot (gridmap.visited > 0)
          - fall within the discovered area of BOTH regions simultaneously
            (discovery_map has bits set for both region versions)

        This replaces boundary-midpoint and door-based spatial edges entirely.
        """
        versions = list(self.all_regions.keys())
        if len(versions) < 2:
            return

        # Build a mask of cells that are free AND visited
        free_visited = (self.gridmap.data == 0) & (self.gridmap.visited > 0)
        if not free_visited.any():
            return

        cell_indices = np.argwhere(free_visited)  # Nx2 array of (i, j)
        disc = self.gridmap.discovery_map  # uint64 bitmask per cell

        for v1, v2 in combinations(versions, 2):
            bit1 = np.uint64(1) << np.uint64(v1)
            bit2 = np.uint64(1) << np.uint64(v2)

            # Cells visited by BOTH regions
            shared_mask = free_visited & \
                          ((disc & bit1) != 0) & \
                          ((disc & bit2) != 0)

            shared_cells = np.argwhere(shared_mask)
            if len(shared_cells) < 3:   # need at least a few shared cells
                continue

            # Convert a representative subset to world coords as waypoints
            # Use up to 5 spread-out points via simple stride sampling
            stride = max(1, len(shared_cells) // 5)
            sampled = shared_cells[::stride][:5]
            waypoints = [
                np.array(self.gridmap._grid_to_world(int(ci), int(cj)))
                for ci, cj in sampled
            ]

            self.exploration_graph.add_free_cell_edge(v1, v2, waypoints)

    def _assign_frontiers_to_regions(self, current_version: int):
        """
        Per-cell frontier ownership:
        - Cell discovered only by current region → current region.
        - Cell discovered only by one remote (non-exhausted) region → that region.
        - Cell in overlap with current region → current region (always).
        - Cell in overlap not including current, or remote is exhausted → current region.
        """
        frontier_cells = np.argwhere(self.gridmap.frontier > 0)
        if len(frontier_cells) == 0:
            return

        disc = self.gridmap.discovery_map

        exclusive: dict = {}   # {version: [np.ndarray, ...]}
        current_pts = []

        for ci, cj in frontier_cells:
            cell_bits = disc[ci, cj]
            versions_here = self.gridmap.discover_regions(cell_bits)

            wx, wy = self.gridmap._grid_to_world(int(ci), int(cj))
            pt = np.array([wx, wy], dtype=np.float32)

            if current_version in versions_here or len(versions_here) == 0:
                # Belongs to current (own cell, overlap with current, or undiscovered)
                current_pts.append(pt)
            elif len(versions_here) == 1:
                v = versions_here[0]
                rec = self.exploration_graph.records.get(v)
                if rec is not None and not rec.is_exhausted():
                    exclusive.setdefault(v, []).append(pt)
                else:
                    current_pts.append(pt)   # exhausted remote → current takes it
            else:
                # Multi-region overlap not including current → current takes it
                current_pts.append(pt)

        # Assign exclusive frontiers to non-exhausted remote regions
        for version, pts in exclusive.items():
            region_obj = self.all_regions.get(version)
            if region_obj is None:
                continue
            # Safe center extraction whether bound is a tensor or ndarray
            b = region_obj.bound
            if hasattr(b, 'cpu'):
                b = b.cpu().numpy()
            center = b[:2].mean(axis=0)[:2].astype(np.float32)
            self.exploration_graph.ensure_region(version, center)
            for pt in pts:
                self.exploration_graph.records[version].add_frontier(
                    pt, ExplorationGraph.BLACKLIST_RADIUS
                )

        # All current-region and overlap frontiers → current region
        for pt in current_pts:
            self.exploration_graph.records[current_version].add_frontier(
                pt, ExplorationGraph.BLACKLIST_RADIUS
            )

    def _build_is_frontier_fn(self):
        """Returns a closure over the current frontier grid for bank reconciliation."""
        frontier_cells = np.argwhere(self.gridmap.frontier > 0)
        if len(frontier_cells) == 0:
            def is_frontier_fn(xy, radius):
                return False
            return is_frontier_fn

        from scipy.spatial import cKDTree
        # World coords of all live frontier cells
        world_pts = np.array([
            self.gridmap._grid_to_world(int(ci), int(cj))
            for ci, cj in frontier_cells
        ])
        tree = cKDTree(world_pts)

        def is_frontier_fn(xy: np.ndarray, radius: float) -> bool:
            hits = tree.query_ball_point(xy[:2], r=radius)
            return len(hits) > 0
        return is_frontier_fn

    def _start_goal_dwell(self):
        dwell_frames_needed = int(self.get_parameter('dwell_frames_needed').value)
        if self.region is None or dwell_frames_needed <= 0:
            self.goal_xy   = None
            self.nav_state = "IDLE"
            return

        self.dwell_active       = True
        self.dwell_frames       = []
        self.dwell_frames_count = 0
        self.dwell_phase        = "COLLECT"
        self.nav_state          = "DWELL"
        self.goal_xy            = None
        self.navigator.stop_robot()

        # Flush any in-flight frames from navigation so we get a clean epoch baseline
        with self.train_lock:
            self.train_queue.clear()
        # Epoch baseline set AFTER flush so we don't count pre-dwell epochs
        self.dwell_epoch_start = int(self.region.epoch)

        print(f"[DWELL] Starting — collecting {dwell_frames_needed} frames "
              f"then training {int(self.get_parameter('dwell_epochs').value)} epochs "
              f"(baseline epoch={self.dwell_epoch_start})")

    def _tick_goal_dwell(self, world_points_hit: np.ndarray, robot_xy: np.ndarray):
        dwell_frames_needed = int(self.get_parameter('dwell_frames_needed').value)
        dwell_epochs        = int(self.get_parameter('dwell_epochs').value)

        # If region changed under us (new region created), abort dwell cleanly
        if self.region is not None and int(self.region.epoch) < self.dwell_epoch_start:
            print("[DWELL] Region switched during dwell — ending dwell, entering WARMUP")
            self._end_goal_dwell()
            # Trigger warmup for the new region
            self.mode               = "WARMUP"
            self.warmup_start_epoch = int(self.region.epoch)
            self.warmup_done        = False
            return

        if self.dwell_phase == "COLLECT":
            if self.region is not None and len(world_points_hit) > 5:
                self.dwell_frames.append((
                    world_points_hit.copy(),
                    self.robot_pos.copy() if self.robot_pos is not None else robot_xy
                ))
                self.dwell_frames_count += 1
                print(f"[DWELL] Collected frame {self.dwell_frames_count}/{dwell_frames_needed}")

            if self.dwell_frames_count >= dwell_frames_needed:
                with self.train_lock:
                    for frame in self.dwell_frames:
                        self.train_queue.append(frame)
                self.dwell_epoch_start = int(self.region.epoch)
                self.dwell_phase       = "TRAIN"
                print(f"[DWELL] Collection done — training for {dwell_epochs} epochs "
                      f"from epoch {self.dwell_epoch_start}")

        elif self.dwell_phase == "TRAIN":
            if self.region is None:
                self._end_goal_dwell()
                return
            epochs_done = int(self.region.epoch) - self.dwell_epoch_start
            print(f"[DWELL] Training epoch {epochs_done}/{dwell_epochs}")
            if epochs_done >= dwell_epochs:
                print(f"[DWELL] Training complete — resuming navigation")
                self._end_goal_dwell()


    def _end_goal_dwell(self):
        """Clean up dwell state and return to normal navigation."""
        self.dwell_active  = False
        self.dwell_frames  = []
        self.dwell_phase   = "COLLECT"
        self.nav_state     = "IDLE"
        self.goal_xy       = None

    def update(self):
        now = self.get_clock().now()

        # ── Prerequisites ────────────────────────────────────────────────
        if self.robot_pos is None or self.latest_cloud is None:
            return

        result = self.adapter.update(self.latest_cloud, self.robot_pos)
        if result is None:
            return
        points_lidar, speeds = result
        if len(points_lidar) == 0:
            return

        # ── TF transform ─────────────────────────────────────────────────
        try:
            source_frame = self.latest_cloud.header.frame_id
            stamp        = self.latest_cloud.header.stamp
            if self.tf_buffer.can_transform('odom', source_frame, stamp,
                                            timeout=Duration(seconds=0.1)):
                transform = self.tf_buffer.lookup_transform('odom', source_frame, stamp)
            else:
                return
            t = transform.transform.translation
            q = transform.transform.rotation
            R = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
            world_points = (R @ points_lidar.T).T + np.array([t.x, t.y, t.z])
        except Exception as e:
            self.get_logger().warn(f"TF lookup failed: {e}")
            return

        # ── Bound filter ──────────────────────────────────────────────────
        robot_xy = self.robot_pos[:2]
        half = self.gridmap.map_size / 2
        ox, oy = self.gridmap.origin
        in_bounds = ((world_points[:, 0] > ox - half) & (world_points[:, 0] < ox + half) &
                     (world_points[:, 1] > oy - half) & (world_points[:, 1] < oy + half))
        world_points_filtered = world_points[in_bounds]
        speeds_filtered = speeds[in_bounds]
        if len(world_points_filtered) < 5:
            return

        # Angular downsampling + free-ray synthesis
        # For each of 360 angular bins:
        #   - Real hit bin  → closest point from sensor; hit_dist = measured dist
        #   - Empty bin     → synthetic endpoint at MAX_RANGE; hit_dist = inf
        # The raycast loop later uses hit_dist to decide whether to mark a wall.
        robot = robot_xy
        vecs   = world_points_filtered[:, :2] - robot_xy[None, :]
        dists  = np.linalg.norm(vecs, axis=1) + 1e-9
        angles = np.arctan2(vecs[:, 1], vecs[:, 0])
        bins   = ((angles + np.pi) / (2 * np.pi) * 359).astype(np.int32)

        closest_idx  = {}
        closest_dist = np.full(360, np.inf, dtype=np.float32)
        for k, b in enumerate(bins):
            if dists[k] < closest_dist[b]:
                closest_dist[b] = dists[k]
                closest_idx[b]  = k

        # Real-hit cloud — identical to the original method; used for raycast + gridmap
        keep = np.array(list(closest_idx.values()), dtype=np.int32)
        world_points_hit = world_points_filtered[keep]
        speeds_hit       = speeds_filtered[keep]

        # Full 360-bin cloud — real hits + synthetic free-ray endpoints
        # Used ONLY for training data collection; never touches gridmap/raycast.
        full_endpoints  = []
        full_speeds     = []
        for b in range(360):
            if b in closest_idx:
                idx = closest_idx[b]
                full_endpoints.append(world_points_filtered[idx])
                full_speeds.append(speeds_filtered[idx])
            else:
                theta  = (b / 359.0) * 2.0 * np.pi - np.pi
                dir_xy = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
                free_xy = robot_xy + MAX_RANGE * dir_xy
                free_z  = float(self.robot_pos[2]) if len(self.robot_pos) > 2 else 0.0
                full_endpoints.append(np.array([free_xy[0], free_xy[1], free_z], dtype=np.float32))
                full_speeds.append(0.0)
        world_points_360 = np.asarray(full_endpoints, dtype=np.float32)

        # ── Raycast: update gridmap.data + visited (real hits only) ──────
        cell   = self.gridmap.cell_size
        origin = self.gridmap.origin
        half_  = self.gridmap.map_size / 2

        def world_to_ij(x, y):
            return (int((x - origin[0] + half_) / cell),
                    int((y - origin[1] + half_) / cell))

        for pt3d in world_points_hit:
            pt = pt3d[:2]
            dx, dy = pt[0] - robot_xy[0], pt[1] - robot_xy[1]
            dist = float(np.hypot(dx, dy))
            if dist > MAX_RANGE:
                scale = MAX_RANGE / dist
                dx *= scale; dy *= scale; dist = MAX_RANGE
            steps = int(dist / cell)
            if steps < 1:
                ix, iy = world_to_ij(pt[0], pt[1])
                if 0 <= ix < self.gridmap.n_cells and 0 <= iy < self.gridmap.n_cells:
                    self.gridmap.data[iy, ix] = 1
                    self.gridmap.visited[iy, ix] = 1
                continue
            for s in range(steps - 1):
                x = robot_xy[0] + dx * s / steps
                y = robot_xy[1] + dy * s / steps
                ix, iy = world_to_ij(x, y)
                if not (0 <= ix < self.gridmap.n_cells and
                        0 <= iy < self.gridmap.n_cells):
                    break
                if self.gridmap.data[iy, ix] == 1:
                    break
                self.gridmap.visited[iy, ix] = 1
            ix_end, iy_end = world_to_ij(pt[0], pt[1])
            if (0 <= ix_end < self.gridmap.n_cells and
                    0 <= iy_end < self.gridmap.n_cells):
                self.gridmap.data[iy_end, ix_end] = 1
                self.gridmap.visited[iy_end, ix_end] = 1

        # ── Gridmap update (1 Hz room segmentation) ──────────────────────
        self._map_update_counter += 1
        do_room_update = (self._map_update_counter % 5 == 0)
        self.gridmap.update(
            lidar_scan=world_points_hit[:, :2],
            points=world_points_hit[:, :2],
            speeds=speeds_hit,
            cur_pos=robot_xy,
            region=self.region,
            regions=self.all_regions,
            update_room=do_room_update
        )

        # ── Snapshot live frontiers into ExplorationGraph ────────────────
        if self.region is not None:
            ver = self.region.version
            center = np.array(self.region.center[:2], dtype=np.float32)
            self.exploration_graph.ensure_region(ver, center)
            self._assign_frontiers_to_regions(ver)
        is_frontier_fn = self._build_is_frontier_fn()
        self.exploration_graph.reconcile_all_banks(is_frontier_fn, evict_radius=0.6)

        if do_room_update:
            for u, v, edge_data in self.gridmap.room_graph.edges(data=True):
                room_u = self.gridmap.room_graph.nodes[u].get('room', -1)
                room_v = self.gridmap.room_graph.nodes[v].get('room', -1)

                # Only register edges between two confirmed rooms
                if room_u not in self.gridmap.confirmed_rooms:
                    continue
                if room_v not in self.gridmap.confirmed_rooms:
                    continue

                pos_u = self.gridmap.room_graph.nodes[u].get('pos')
                pos_v = self.gridmap.room_graph.nodes[v].get('pos')
                if pos_u is None or pos_v is None:
                    continue

                door_xy = np.array([(pos_u[0]+pos_v[0])/2,
                                     (pos_u[1]+pos_v[1])/2], dtype=np.float32)

                reg_u = getattr(self.gridmap.confirmed_rooms[room_u], 'region', None)
                reg_v = getattr(self.gridmap.confirmed_rooms[room_v], 'region', None)
                if reg_u is None or reg_v is None:
                    continue
                if reg_u.version == reg_v.version:
                    continue   # same region, spatial edge already covers this

                self.exploration_graph.add_door_edge(reg_u.version, reg_v.version, door_xy)
                self._update_region_graph_via_free_cells()
#            # ── Spatial adjacency edges (intra-room, no door needed) ─────
#            for v1, reg1 in self.all_regions.items():
#                for v2, reg2 in self.all_regions.items():
#                    if v1 >= v2:
#                        continue
#                    #if self.exploration_graph.graph.has_edge(v1, v2):
#                    #    continue  # door edge already exists, skip
#
#                    b1 = reg1.bound.numpy()
#                    b2 = reg2.bound.numpy()
#
#                    # Gap between the two bounding boxes in x and y
#                    x_gap = max(0.0, max(b1[0,0], b2[0,0]) - min(b1[1,0], b2[1,0]))
#                    y_gap = max(0.0, max(b1[0,1], b2[0,1]) - min(b1[1,1], b2[1,1]))
#
#                    if x_gap < 1.0 and y_gap < 1.0:
#                        # Midpoint of the shared/nearest boundary zone
#                        mx = (max(b1[0,0], b2[0,0]) + min(b1[1,0], b2[1,0])) / 2.0
#                        my = (max(b1[0,1], b2[0,1]) + min(b1[1,1], b2[1,1])) / 2.0
#                        midpoint = np.array([mx, my], dtype=np.float32)
#                        #self.exploration_graph.add_spatial_edge(v1, v2, midpoint)
                #self._update_region_graph_via_free_cells()

        # ── Enqueue training frame (ALL modes) ───────────────────────────
        if self.region is not None and bool(self.get_parameter('train_after_region').value):
            self.train_counter += 1
            if self.train_counter % self.train_every_n_frames == 0:
                with self.train_lock:
                    self.train_queue.append((
                        world_points_hit.copy(),
                        self.robot_pos.copy()
                    ))

        # ── Region boundary check ─────────────────────────────────────────
        # ── Fit region bound to observed LiDAR geometry (1 Hz) ─────────
        if do_room_update and self.mode != "BOOTSTRAP":
            self.fit_region_to_lidar()

        self._maybe_create_new_region_guarded()
        if self._map_update_counter % 10 == 0:
            self.maybe_merge_regions()
        # ── Trajectory ───────────────────────────────────────────────────
        if self.robot_pos is not None:
            self._traj_history.append(self.robot_pos[:2].copy())

        # ── Mode state machine (navigation only, map already updated above) ──

        if self.mode == "BOOTSTRAP":
            self.navigator.stop_robot()
            self.bootstrap_frames += 1
            if self.region is None and self.bootstrap_frames >= self.bootstrap_threshold:
                self.get_logger().info("Initializing first region")
                self.init_region(self.robot_pos)
                self.mode              = "WARMUP"
                self.warmup_start_epoch = int(self.region.epoch)
                self.goal_xy           = None
                self.last_goal_time    = now
                self.get_logger().info("[FLOW] Entering WARMUP phase")

        elif self.mode == "WARMUP":
            # Stop robot, check epochs — but do NOT return; fall through to publish
            self.navigator.stop_robot()
            if self.region is not None:
                cur_epochs = int(self.region.epoch) - int(self.warmup_start_epoch)
                print(f"[WARMUP] epoch {cur_epochs}/{self.warmup_epochs_needed}")
                if cur_epochs >= self.warmup_epochs_needed:
                    self.mode        = "EXPLORE"
                    self.warmup_done = True
                    self.goal_xy     = None
                    self.last_goal_time = now
                    self.get_logger().info("[FLOW] Warmup complete → starting EXPLORE")

        elif self.mode == "EXPLORE":
            robot_xy = np.asarray(self.robot_pos[:2], dtype=np.float32)
            now_sec  = now.nanoseconds * 1e-9

            # ── IDLE: pick a new frontier goal ───────────────────────────
            if self.nav_state == "IDLE":
                current_room = self.gridmap.get_room_at_position(robot_xy)
                rooms        = self.gridmap.get_rooms()
                room_graph   = self.gridmap.get_room_graph()
                g = self.pick_room_aware_goal(current_room, rooms, room_graph)

                if g is None:
                    self.navigator.stop_robot()
                    print("[NO FRONTIER]")
                else:
                    self.goal_xy           = np.asarray(g, dtype=np.float32)
                    self.wait_start_time   = now_sec
                    self._path_gen_running = False
                    print("[GOAL SET]", self.goal_xy)

                    # ── If ExplorationGraph gave us door waypoints, pre-load them ─
                    if self._remote_door_waypoints:
                        print(f"[EXPLORATION GRAPH] Using {len(self._remote_door_waypoints)}"
                              f" door waypoints → region {self._remote_target_version}")
                        # Build composite path: door chain + final frontier goal
                        full_wp = np.array(
                            self._remote_door_waypoints + [self.goal_xy],
                            dtype=np.float32
                        )
                        self.latest_candidate_path = full_wp
                        self._remote_door_waypoints = []   # consumed
                    else:
                        self.latest_candidate_path = None  # normal: path gen will fill this

                    self.nav_state = "STABILIZING"
            # ── STABILIZING: generate path in background, don't block ────
            elif self.nav_state == "STABILIZING":
                self.navigator.stop_robot()
                # kick off path gen if not already running
                self._launch_path_gen(robot_xy, self.goal_xy)

                if now_sec - self.wait_start_time >= self.wait_duration:
                    if self.latest_candidate_path is not None:
                        print("[PATH FROZEN]")
                        self.path = self.latest_candidate_path.copy()
                        self.navigator.reset_path()
                        self.nav_state = "FOLLOWING"
                    else:
                        print("[NO PATH FOUND — retrying]")
                        self.wait_start_time = now_sec  # extend wait

            # ── FOLLOWING: drive, continuously refresh path in background ─
            elif self.nav_state == "FOLLOWING":
                if self.goal_xy is None:
                    self.nav_state = "IDLE"
                    self.path = None
                    return
                if self.goal_xy is not None:
                    d_goal = float(np.linalg.norm(robot_xy - self.goal_xy))
                else:
                    d_goal = None
                #print(f"Distance to frontier: {d_goal:.2f}")
                if d_goal is None:
                    self.nav_state = "IDLE"
                    self.goal_xy = None
                    self.path = None 

                if d_goal is not None and d_goal < 0.25:
                    print("[FRONTIER REACHED]")
                    self.navigator.stop_robot()
                    self.navigator.reset_path()

                    if self.goal_xy is not None:
                        # ── Clear frontier cells around reached goal ──────────────
                        gix = int((self.goal_xy[0] - (self.gridmap.origin[0] - self.gridmap.map_size/2))
                                  / self.gridmap.cell_size)
                        giy = int((self.goal_xy[1] - (self.gridmap.origin[1] - self.gridmap.map_size/2))
                                  / self.gridmap.cell_size)
                        pad = 3
                        for di in range(-pad, pad+1):
                            for dj in range(-pad, pad+1):
                                ii, jj = giy+di, gix+dj
                                if 0 <= ii < self.gridmap.n_cells and 0 <= jj < self.gridmap.n_cells:
                                    self.gridmap.frontier[ii, jj] = 0
                                    self.gridmap.visited[ii, jj]  = 1

                        # ── Update visited blacklist ──────────────────────────────
                        self._visited_goals.append(self.goal_xy.copy())
                        if len(self._visited_goals) > 50:
                            self._visited_goals.pop(0)

                        # ── Notify ExplorationGraph ───────────────────────────────
                        if self.region is not None:
                            self.exploration_graph.mark_goal_reached(
                                self.region.version, self.goal_xy
                            )

                    #self.goal_xy   = None
                    self.path      = None
                    self._remote_door_waypoints = []       # clear any leftover door chain
                    self._remote_target_version = None
                    self._start_goal_dwell()
                    if not self.dwell_active:
                        self.goal_xy = None
                        self.nav_state = "IDLE"

                else:
                    if self.goal_xy is None:       # ← double-check before path gen
                        self.nav_state = "IDLE"
                        return
                    # Continuously refresh the path as the network improves
                    self._launch_path_gen(robot_xy, self.goal_xy)
                    if self.latest_candidate_path is not None:
                        self.path                  = self.latest_candidate_path
                        self.latest_candidate_path = None
                        self.navigator.reset_path()
                    if self.path is not None:
                        self.navigator.drive_path(self.path)
            
            elif self.nav_state == "DWELL":
                self.navigator.stop_robot()
                self._tick_goal_dwell(world_points_hit, robot_xy)

        # ── Composite plot ────────────────────────────────────────────────
        if self._composite_step % 20 == 0:
            try:
                self.save_composite_plot()
                self.save_occupancy_snapshot()
            except Exception as e:
                self.get_logger().warn(f"[COMPOSITE PLOT] {e}")
        self._composite_step += 1

        # ── Always publish (every tick, every mode) ───────────────────────
        self.map_publisher.publish_maps()
        self.markers.publish_goal_marker()
        self.markers.publish_frontier_markers()
        self.publish_field_path(self.path)

    def save_composite_plot(self):
        """
        plot2-style composite:
          - world-scale background (unknown / visited / obstacles)
          - per-region travel-time contours that PERSIST via a numeric cache (no translucent overlays)
          - exploration graph (nodes/edges/doors)
          - ONLY the latest saved frontiers per region (not the entire bank)
        """
        from matplotlib.lines import Line2D
        os.makedirs(self.folder, exist_ok=True)
        out_dir = os.path.join(self.folder, "plots2")
        os.makedirs(out_dir, exist_ok=True)

        # World-map bounds (gridmap is world-sized)
        b_map = [
            [self.gridmap.origin[0] - self.gridmap.map_size / 2,
             self.gridmap.origin[1] - self.gridmap.map_size / 2],
            [self.gridmap.origin[0] + self.gridmap.map_size / 2,
             self.gridmap.origin[1] + self.gridmap.map_size / 2],
        ]

        fig, ax = plt.subplots(1, 1, figsize=(14, 14))

        # ── Background: visited=dark green, obstacle=black, unknown=dark blue ──
        data = self.gridmap.data
        vis  = self.gridmap.visited
        n    = self.gridmap.n_cells
        cell = self.gridmap.cell_size

        bg = np.zeros((n, n, 3), dtype=np.uint8)
        bg[:, :] = [10, 10, 80]        # unknown
        bg[vis == 1]  = [220, 220, 40]     # visited free
        bg[data == 1] = [0, 0, 0]      # obstacles

        extent = [b_map[0][0], b_map[1][0], b_map[0][1], b_map[1][1]]
        ax.imshow(bg, origin='lower', extent=extent, zorder=0, aspect='auto')

        # ── Persistent numeric cache for region fields ─────────────────────────
        # Cache: version -> dict(X_w, Y_w, TT_masked, extent, src_world, step, epoch)
        if not hasattr(self, "_region_field_cache"):
            self._region_field_cache = {}

        # Tuning
        GRID_N = 90                 # resolution of per-region TT grid
        CONTOUR_LEVELS = 40         # number of contour levels
        FRONTIER_LAST_K = 30        # only show last K saved frontiers per region
        CACHE_UPDATE_EVERY = 1      # update active region every step (keep expanding field visible)

        def _build_region_mask(b, grid_n=GRID_N):
            """Mask out cells outside visited/obstacle footprint (same logic as your original)."""
            mask = np.zeros((grid_n, grid_n), dtype=bool)
            for i in range(grid_n):
                for j in range(grid_n):
                    wx = b[0, 0] + (i + 0.5) / grid_n * (b[1, 0] - b[0, 0])
                    wy = b[0, 1] + (j + 0.5) / grid_n * (b[1, 1] - b[0, 1])
                    gx = int((wx - b_map[0][0]) / cell)
                    gy = int((wy - b_map[0][1]) / cell)
                    if 0 <= gx < n and 0 <= gy < n:
                        mask[j, i] = (vis[gy, gx] == 1 or data[gy, gx] == 1)
            return mask

        def _compute_tt_grid(region, src_world, b, mask, grid_n=GRID_N):
            """Compute TT(x0=src, x1=grid) over region bound and apply mask => TT_masked."""
            xmin, ymin = float(b[0, 0]), float(b[0, 1])
            xmax, ymax = float(b[1, 0]), float(b[1, 1])

            xs = np.linspace(xmin, xmax, grid_n)
            ys = np.linspace(ymin, ymax, grid_n)
            X_w, Y_w = np.meshgrid(xs, ys)

            # world -> normalized using region bound
            sx = (xmax - xmin) + 1e-9
            sy = (ymax - ymin) + 1e-9
            X_n = (X_w - xmin) / sx - 0.5
            Y_n = (Y_w - ymin) / sy - 0.5

            src_norm = region.normalize(src_world.reshape(1, 3))[0]  # (3,)

            XP = np.zeros((X_w.size, 2 * region.dim), dtype=np.float32)
            XP[:, :region.dim] = src_norm
            XP[:, region.dim + 0] = X_n.reshape(-1)
            XP[:, region.dim + 1] = Y_n.reshape(-1)
            XP[:, region.dim + 2] = src_norm[2]

            XP_t = torch.tensor(XP, dtype=torch.float32, device=region.device)

            with torch.no_grad():
                TT_t = region.function.TravelTimes(XP_t)
                if TT_t.ndim == 2 and TT_t.shape[1] == 1:
                    TT_t = TT_t[:, 0]
                TT = TT_t.detach().cpu().numpy().reshape(X_w.shape)

            TT_masked = TT.copy()
            TT_masked[~mask] = np.nan

            return X_w, Y_w, TT_masked

        def _pick_src_world(region, b):
            """Stable source for plotting TT contours."""
            # Active region uses robot position if available
            if self.region is not None and region.version == self.region.version and self.robot_pos is not None:
                return np.array([self.robot_pos[0], self.robot_pos[1], 0.0], dtype=np.float32)

            # Inactive: prefer cached source if present, else region center
            cached = self._region_field_cache.get(region.version)
            if cached is not None and "src_world" in cached:
                return cached["src_world"]

            rc = (b[0] + b[1]) / 2.0
            return np.array([float(rc[0]), float(rc[1]), 0.0], dtype=np.float32)

        # ── Update cache for the active region (so its field "expands" live) ──
        if self.region is not None and (self._composite_step % CACHE_UPDATE_EVERY == 0):
            try:
                reg = self.region
                if reg.function is not None:
                    b = reg.bound.detach().cpu().numpy()
                    mask = _build_region_mask(b, grid_n=GRID_N)
                    src_world = _pick_src_world(reg, b)
                    X_w, Y_w, TT_masked = _compute_tt_grid(reg, src_world, b, mask, grid_n=GRID_N)
                    self._region_field_cache[reg.version] = dict(
                        X_w=X_w, Y_w=Y_w, TT=TT_masked,
                        extent=[b[0, 0], b[1, 0], b[0, 1], b[1, 1]],
                        mask=mask, src_world=src_world,
                        step=int(self._composite_step),
                        epoch=int(getattr(reg, "epoch", 0)),
                    )
            except Exception as e:
                self.get_logger().warn(f"[COMPOSITE] active region cache update failed: {e}")

        # ── Ensure each region has at least one cached snapshot (persist) ─────
        for region in self.all_regions.values():
            if region.version in self._region_field_cache:
                continue
            # Only cache if model exists right now; otherwise it will remain missing until loaded once
            if region.function is None:
                continue
            try:
                b = region.bound.detach().cpu().numpy()
                mask = _build_region_mask(b, grid_n=GRID_N)
                src_world = _pick_src_world(region, b)
                X_w, Y_w, TT_masked = _compute_tt_grid(region, src_world, b, mask, grid_n=GRID_N)
                self._region_field_cache[region.version] = dict(
                    X_w=X_w, Y_w=Y_w, TT=TT_masked,
                    extent=[b[0, 0], b[1, 0], b[0, 1], b[1, 1]],
                    mask=mask, src_world=src_world,
                    step=int(self._composite_step),
                    epoch=int(getattr(region, "epoch", 0)),
                )
            except Exception as e:
                self.get_logger().warn(f"[COMPOSITE] region {region.version} initial cache failed: {e}")

        # ── Draw persistent contours for all cached regions ───────────────────
        for region in self.all_regions.values():
            b = region.bound.detach().cpu().numpy()

            cached = self._region_field_cache.get(region.version, None)
            if cached is not None:
                X_w = cached["X_w"]
                Y_w = cached["Y_w"]
                TT  = cached["TT"]

                if np.isfinite(TT).any():
                    ax.contour(
                        X_w, Y_w, TT,
                        levels=CONTOUR_LEVELS,
                        cmap="gist_heat",
                        linewidths=0.65,
                        alpha=0.9,
                        zorder=2
                    )

            # ── Region bounding box (always) ─────────────────────────────────
            rect_w = float(b[1, 0] - b[0, 0])
            rect_h = float(b[1, 1] - b[0, 1])
            is_active = (self.region is not None and region.version == self.region.version)
            edge_color = 'yellow' if is_active else 'gray'
            lw = 2.0 if is_active else 0.8

            rect = plt.Rectangle(
                (float(b[0, 0]), float(b[0, 1])), rect_w, rect_h,
                linewidth=lw, edgecolor=edge_color, facecolor='none', zorder=4
            )
            ax.add_patch(rect)
            ax.text(
                float(b[0, 0]) + 0.02, float(b[1, 1]) - 0.15,
                f"R{region.version}{'*' if is_active else ''}",
                color=edge_color, fontsize=7, zorder=5
            )

        # ── ExplorationGraph overlay: edges + door waypoints ─────────────────
        for v1, v2, edata in self.exploration_graph.graph.edges(data=True):
            rec1 = self.exploration_graph.records.get(v1)
            rec2 = self.exploration_graph.records.get(v2)
            if rec1 is None or rec2 is None:
                continue

            c1 = rec1.center
            c2 = rec2.center

            door_xys = edata.get('door_xys', [edata.get('door_xy')])
            door_xys = [d for d in door_xys if d is not None]

            if door_xys:
                for door_xy in door_xys:
                    xs = [c1[0], door_xy[0], c2[0]]
                    ys = [c1[1], door_xy[1], c2[1]]
                    ax.plot(xs, ys, color='cyan', linewidth=1.2,
                            linestyle='--', alpha=0.7, zorder=6)
                    ax.scatter(door_xy[0], door_xy[1],
                               c='cyan', s=40, marker='D',
                               zorder=7, alpha=0.9)
            else:
                ax.plot([c1[0], c2[0]], [c1[1], c2[1]],
                        color='cyan', linewidth=1.0,
                        linestyle='--', alpha=0.6, zorder=6)

        # ── ExplorationGraph nodes (region centers) ──────────────────────────
        for ver, rec in self.exploration_graph.records.items():
            c = rec.center
            has_f = rec.has_frontiers()
            node_color = 'lime' if has_f else 'red'
            ax.scatter(c[0], c[1], c=node_color, s=120,
                       marker='o', zorder=8, edgecolors='white', linewidths=1.0)
            ax.text(c[0] + 0.05, c[1] + 0.05,
                    f"R{ver}\n({len(rec.saved_frontiers)}f)",
                    color='white', fontsize=6, zorder=9,
                    bbox=dict(boxstyle='round,pad=0.15', fc='black', alpha=0.5))

        # ── Latest saved frontier points per region (NOT all) ─────────────────
        for ver, rec in self.exploration_graph.records.items():
            if not rec.saved_frontiers:
                continue
            pts = rec.saved_frontiers[-FRONTIER_LAST_K:]  # latest K (append order)
            for fp in pts:
                ax.scatter(fp[0], fp[1], c='orange', s=14,
                           marker='.', zorder=7, alpha=0.85)

        # ── Robot, trajectory, goal ─────────────────────────────────────────
        if self.robot_pos is not None:
            ax.scatter(self.robot_pos[0], self.robot_pos[1],
                       c='red', s=100, zorder=11, marker='o', label='Robot')

        if len(self._traj_history) > 1:
            traj = np.array(self._traj_history)
            ax.plot(traj[:, 0], traj[:, 1], 'm-', linewidth=1.5, zorder=9)

        if self.goal_xy is not None:
            ax.scatter(self.goal_xy[0], self.goal_xy[1],
                       c='lime', s=180, zorder=11, marker='*', label='Goal')

        # ── Active inter-region door route ───────────────────────────────────
        if getattr(self, "_remote_door_waypoints", None):
            if len(self._remote_door_waypoints) > 0:
                wps = np.array(self._remote_door_waypoints)
                ax.plot(wps[:, 0], wps[:, 1], color='magenta',
                        linewidth=2.0, linestyle='-', zorder=10, label='Door route')
                ax.scatter(wps[:, 0], wps[:, 1], c='magenta',
                           s=80, marker='D', zorder=11)

        # ── Legend ──────────────────────────────────────────────────────────
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='lime',
                   markersize=8, label='Region (has frontiers)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                   markersize=8, label='Region (exhausted)'),
            Line2D([0], [0], marker='D', color='w', markerfacecolor='cyan',
                   markersize=7, label='Door waypoint'),
            Line2D([0], [0], color='cyan', linestyle='--', label='Graph edge'),
            Line2D([0], [0], marker='.', color='w', markerfacecolor='orange',
                   markersize=6, label=f'Latest {FRONTIER_LAST_K} frontiers'),
            Line2D([0], [0], marker='*', color='w', markerfacecolor='lime',
                   markersize=10, label='Current goal'),
            Line2D([0], [0], color='magenta', label='Active door route'),
        ]
        ax.legend(handles=legend_elements, loc='upper right',
                  fontsize=7, framealpha=0.7)

        ax.set_xlim(b_map[0][0], b_map[1][0])
        ax.set_ylim(b_map[0][1], b_map[1][1])
        ax.set_aspect('equal')
        ax.set_title(
            f"Step {self._composite_step}  |  regions={len(self.all_regions)}"
            f"  doors={len(self.gridmap.doors)}"
            f"  mode={self.mode}/{self.nav_state}"
        )

        fig.savefig(os.path.join(out_dir, f"plot_step_{self._composite_step}.png"),
                    dpi=120, bbox_inches='tight')
        fig.savefig(os.path.join(self.folder, "plot2.png"), bbox_inches='tight')
        plt.close(fig)

    def save_occupancy_snapshot(self, tag=None):
        """
        Saves occupancy grid + visited + frontier as image and numpy.
        """
        out_dir = os.path.join(self.folder, "occupancy_snapshots")
        os.makedirs(out_dir, exist_ok=True)

        if tag is None:
            tag = f"step_{self._composite_step:06d}"

        data = self.gridmap.data.copy()
        visited = self.gridmap.visited.copy()
        frontier = self.gridmap.frontier.copy()

        n = self.gridmap.n_cells

        # ---- Create visualization image ----
        img = np.zeros((n, n, 3), dtype=np.uint8)

        # unknown
        img[:, :] = [10, 10, 80]

        # visited free
        img[visited == 1] = [200, 200, 0]   # yellow (clearer than dark green)

        # occupied
        img[data == 1] = [0, 0, 0]

        # frontier overlay (optional highlight)
        img[frontier == 1] = [255, 140, 0]  # orange

        plt.figure(figsize=(8, 8))
        plt.imshow(img, origin="lower")
        plt.title(f"Occupancy Snapshot {tag}")
        plt.axis("off")
        plt.savefig(os.path.join(out_dir, f"{tag}.png"), dpi=150, bbox_inches="tight")
        plt.close()

        # ---- Save raw arrays ----
        np.savez_compressed(
            os.path.join(out_dir, f"{tag}.npz"),
            occupancy=data,
            visited=visited,
            frontier=frontier
        )

        self.get_logger().info(f"[OCC SNAPSHOT SAVED] {tag}")

def main():
    rclpy.init()
    node = GridmapNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()




LATEST
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
import rclpy.time
import torch
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree
import signal
import threading
from typing import List

import numpy as np
import math
import tf2_ros
from tf2_ros import TransformBroadcaster
import tf2_geometry_msgs

import sensor_msgs_py.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from geometry_msgs.msg import TransformStamped, Twist, Point, PointStamped, PoseStamped, PoseWithCovarianceStamped
from visualization_msgs.msg import Marker

from mntfields.mntfields_ros.sensing.lidar_adapter import Lidar2DAdapter
from mntfields.mntfields_ros.mapping.map_publisher import MapPublisher 
from mntfields.mntfields_ros.mapping.markers import MarkerPublisher
from mntfields.mntfields_ros.mapping.navigation import Navigator
from mntfields.mntfields_ros.mapping.exploration_graph import ExplorationGraph

from mntfields.mntfields_core.mapping.lidar_gridmap import LidarGridmap
from mntfields.mntfields_core.core.region import Region
from mntfields.mntfields_core.core.model_base import Model

from collections import deque
import threading

import os
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg", force=True)
from matplotlib import cm, colors
import matplotlib.pyplot as plt


REGION_RADIUS = 2.5
REGION_EXIT_MARGIN = 0.4 #was 0.4
REGION_MIN_OVERLAP = 0.2 #was 0.4
MAX_RANGE = 3.2
WARMUP_EPOCHS = 6

#print("NEW CODE")
class DummyRenderer:
    def __init__(self):
        pass

    def render(self, *args, **kwargs):
        return None

class DummyModel:
    def __init__(self):
        self.folder = "/home/mayank/turtlebot3_ws/mntfields_model"
        self.renderer = "cuda:0"   # IMPORTANT: must be string device
        self.h = 0.0

class FakePCD:
    def __init__(self, pts):
        self.points = pts
        self.normals = np.zeros_like(pts)

class GridmapNode(Node):
    def __init__(self):
        super().__init__('mntfields_gridmap')
        self.folder = os.path.join("/home/mayank/turtlebot3_ws/mntfields_model/mntfields_output")
        os.makedirs(self.folder, exist_ok=True)

        self.train_queue = deque(maxlen=4)   # keep last few frames only
        self.train_lock = threading.Lock()
        self.train_running = False
        self.declare_parameter('region_min_epochs', 10)  # train at least 20 times before 
        self._traj_history = []
        self._composite_step = 0

        # Tune these
        self.declare_parameter('train_timer_sec', 0.5)      # training loop cadence
        self.declare_parameter('train_samples', 5000)       # n_samples to get_data_from_lidar
        self.declare_parameter('train_epoch', 4)            # epoch passed into region.train_core
        self.declare_parameter('train_after_region', True)  # toggle
        self.radius = REGION_RADIUS

        # Start a separate timer JUST for training
        train_dt = float(self.get_parameter('train_timer_sec').value)
        self.create_timer(train_dt, self.train_tick)
        
        self.declare_parameter('n_cells', 400)
        self.declare_parameter('near_dist', 2.0)
        self.declare_parameter('map_size', 40.0)  # Add this

        n_cells = self.get_parameter('n_cells').value
        near_dist = self.get_parameter('near_dist').value
        map_size = self.get_parameter('map_size').value

        self.map_to_odom = None
        self.origin_initialized = False
        self.set_parameters([rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)])
        # --- exploration / control params ---
        self.declare_parameter('goal_reached_dist', 0.20)
        self.declare_parameter('yaw_tol', 0.25)          # rad
        self.declare_parameter('kp_yaw', 0.8)
        self.declare_parameter('kp_lin', 0.25)
        self.declare_parameter('max_lin', 0.15)          # m/s
        self.declare_parameter('max_ang', 0.8)           # rad/s
        self.declare_parameter('frontier_min_cells', 2)
        self.declare_parameter('goal_cooldown_sec', 2.0)

        self.path = None        #these 4 lines are for the path holding for stable paths - latest diff
        self.path_last_update = self.get_clock().now()
        self.path_update_period = 2.0   # seconds
        self.path_index = 0

        self.goal_xy = None
        self.last_goal_time = self.get_clock().now()
        self.mode = "BOOTSTRAP"   # BOOTSTRAP -> EXPLORE
        #--diff
        self.warmup_frames_needed = 5
        self.warmup_epochs_needed = WARMUP_EPOCHS
        self.warmup_done = False
        self.warmup_start_epoch = 0
        self.nav_state = "IDLE"
        self.wait_start_time = None
        self.wait_duration = 3.0
        self.latest_candidate_path = None
        #--
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        self.navigator = Navigator(self)
        self.map_publisher = MapPublisher(self)
        self.markers = MarkerPublisher(self)

        self.robot_pos = None
        self.robot_yaw = 0.0
        self.latest_cloud = None

        self.all_regions: dict = {}
        self.region_initialized = False
        self.bootstrap_frames = 0
        self.bootstrap_threshold = 10 # frames before region starts
        
        self.adapter = Lidar2DAdapter()

        # Traversability map (1 = free, 0 = blocked)
        self.trav_map = np.ones((n_cells, n_cells), dtype=np.uint8)
        self.gridmap = LidarGridmap(
            trav_map=self.trav_map,
            regions=self.all_regions,#{}
            n_cells=n_cells,
            near_dist=2.0,
            map_size=map_size,           # Add this
            origin=(0.0, 0.0), #(origin_x, origin_y)  # Add this
            room_seg_config=(7,2),
        )
        self.origin_initialized = False
        self.declare_parameter('occ_height', 0.3)
        self.occ_height = float(self.get_parameter('occ_height').value)
        self.occ_height_tol = 0.01

        self.create_subscription(PointCloud2,'/lidar3d_points',self.lidar_cb,10)
        #self.create_subscription(PoseWithCovarianceStamped,'/amcl_pose',self.pose_cb,5) - will be used for a built map
        self.create_subscription(Odometry, '/odom', self.odom_cb, 10)

        self.occ_pub = self.create_publisher(OccupancyGrid,'/mntfields/occ_map',5)
        self.frontier_pub = self.create_publisher(OccupancyGrid,'/mntfields/frontier',5)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.goal_marker_pub = self.create_publisher(Marker, '/mntfields/goal_marker', 10)
        self.frontier_marker_pub = self.create_publisher(Marker, '/mntfields/frontier_markers', 10)
        self.field_path_pub = self.create_publisher(Path, '/field_path', 10)

        self.create_timer(0.2, self.update)

        self.model = DummyModel()
        self.region = None

        self.exploration_graph = ExplorationGraph() #graph diff
        self._remote_door_waypoints: List[np.ndarray] = []  # door waypoints for inter-region travel
        self._remote_target_version: Optional[int] = None
        self._goal_blacklist_radius = 0.5 #was 0.3
        self._visited_goals: List[np.ndarray] = []

        self.train_counter = 0
        self.train_every_n_frames = 1

        self._map_update_counter = 0 #claude difff
        self._path_gen_running   = False

        # ── Goal-dwell training burst ─────────────────────────────────────
        self.declare_parameter('dwell_frames_needed', 3)    # frames to collect at goal
        self.declare_parameter('dwell_epochs',        3)    # epochs to train before resuming
        self.dwell_active       = False
        self.dwell_frames       = []                        # accumulated during dwell
        self.dwell_frames_count = 0
        self.dwell_epoch_start  = 0
        self.dwell_phase        = "COLLECT"  # "COLLECT" | "TRAIN" | "DONE"


    def lidar_cb(self, msg):
        self.latest_cloud = msg
        if not hasattr(self, "_last_lidar_log"):
            self._last_lidar_log = self.get_clock().now()
            #self.get_logger().info("Received first LiDAR scan")
            return
        now = self.get_clock().now()
        if (now - self._last_lidar_log).nanoseconds > 2e9:  # 2 seconds
            #self.get_logger().info("Receiving LiDAR scans...")
            self._last_lidar_log = now
    
    def odom_cb(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation

        self.robot_pos = np.array([p.x, p.y, p.z], dtype=np.float32)
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.robot_yaw = np.arctan2(siny_cosp, cosy_cosp)

        # Lock map once, using the *initial* odom pose
        if not self.origin_initialized:
            self.gridmap.origin = np.array([p.x, p.y], dtype=np.float32)
            self.origin_initialized = True

    def init_region(self, robot_pose):
        ''' Initializes a region around the robot and registers it. '''

        center_x = robot_pose[0]
        center_y = robot_pose[1]
        radius = self.radius

        bound = torch.tensor([
            [center_x - radius, center_y - radius, -0.1],
            [center_x + radius, center_y + radius,  0.1],
        ], dtype=torch.float32)

        new_region = Region(
            model=self.model,
            bound=bound,
            init_network=True,
            eval=False,)
        new_region.gridmap = self.gridmap
        self.all_regions[new_region.version] = new_region

        if self.region is not None and self.region.all_lidar is not None:
            last = self.region.all_lidar[-5:]
            new_region.all_lidar = last.copy()
            new_region.lidar_obs = None
            new_region.kd_tree = None
        else:
            new_region.all_lidar = []
        self.region = new_region
        self.region_initialized = True
        self.get_logger().info(
            f"[REGION {new_region.version} INIT] "
            f"center=({center_x:.2f}, {center_y:.2f}) "
            f"bound=[({bound[0,0]:.2f},{bound[0,1]:.2f}) → "
            f"({bound[1,0]:.2f},{bound[1,1]:.2f})]")
        # ── Re-enter WARMUP for every new region ─────────────────────────
        self.mode               = "WARMUP"
        self.warmup_start_epoch = int(self.region.epoch)
        self.goal_xy            = None
        self.path               = None
        self.nav_state          = "IDLE"
        self.latest_candidate_path = None
        self._path_gen_running  = False
        self.get_logger().info(f"[FLOW] Region {new_region.version} → entering WARMUP")

    def start_region_warmup(self):
        self.mode = "WARMUP"
        self.region_warmup_start_epoch = self.region.epoch if self.region else 0
        self.navigator.stop_robot()
        self.get_logger().info(f"[WARMUP] Starting region {self.region.version}")

    def feed_region(self, world_pts, robot_pose):
        if len(world_pts) < 50:
            self.get_logger().warn(f"Too few points: {len(world_pts)}")
            return
        try:
            pts = torch.tensor(world_pts, dtype=torch.float32)
            pos = torch.tensor([robot_pose[0], robot_pose[1], 0.0], dtype=torch.float32)
            ##print("WORLD PTS MIN/MAX:", world_pts.min(), world_pts.max())
            n_samples = int(self.get_parameter('train_samples').value)
            epoch = int(self.get_parameter('train_epoch').value)
            data = self.region.get_data_from_lidar(
                pts.cpu().numpy(),
                pos.cpu().numpy(),
                device='cuda:0',
                n_samples=n_samples
            )
            if data is None:
                self.get_logger().warn("get_data_from_lidar returned None")
                return
            points, closest = data
            if points.shape[0] == 0:
                self.get_logger().warn("get_data_from_lidar returned 0 points")
                return

            if data is None:
                self.get_logger().warn("get_data_from_lidar returned None")
                return
            x0 = points[:, :3]
            x1 = points[:, 3:6]
            cp0 = closest[:, :3]
            cp1 = closest[:, 3:6]

            raw = torch.cat([x0, x1, cp0, cp1], dim=1)   # Nx12
            #print("RAW SHAPE:", raw.shape)

            if raw.ndim != 2 or raw.shape[1] != 12:
                self.get_logger().error(f"[DBG] raw has wrong shape: {tuple(raw.shape)} (expected Nx12)")
                return
            if not torch.isfinite(raw).all():
                bad = (~torch.isfinite(raw)).sum().item()
                self.get_logger().error(f"[DBG] raw has non-finite values: count={bad}")
                return
            with torch.no_grad():
                mn = raw.min(dim=0).values.cpu().numpy()
                mx = raw.max(dim=0).values.cpu().numpy()
            frame_data = self.region.calculate_speed_and_normal(raw)
            #frame_data = raw

            if frame_data.ndim != 2:
                self.get_logger().error(f"[DBG] frame_data not 2D: {tuple(frame_data.shape)}")
                return
            if frame_data.shape[1] < 12:
                self.get_logger().error(f"[DBG] frame_data has too few cols: {frame_data.shape[1]} (expected >=12)")
                return
            if not torch.isfinite(frame_data).all():
                bad = (~torch.isfinite(frame_data)).sum().item()
                self.get_logger().error(f"[DBG] frame_data non-finite: count={bad}")
                return
            with torch.no_grad():
                fd_min = frame_data.min().item()
                fd_max = frame_data.max().item()

            y0_col = frame_data[:, 6]
            y1_col = frame_data[:, 7]

            if frame_data.shape[0] == 0:
                self.get_logger().warn("No valid frame data after speed calculation")
                return
            self.region.last_robot_pose = np.array([robot_pose[0], robot_pose[1], 0.0], dtype=np.float32)

            if self.region.function is not None:
                try:
                    k = min(64, frame_data.shape[0])
                    xp_n = frame_data[:k, :6].clone().to(self.region.device)
                    with torch.no_grad():
                        tt = self.region.function.TravelTimes(xp_n).detach().cpu().numpy().reshape(-1)
                    with torch.enable_grad():
                        sp_t = self.region.function.Speed(xp_n)  # tensor
                    sp = sp_t.detach().cpu().numpy().reshape(-1)

                except Exception as e:
                    self.get_logger().error(f"[DBG MODEL] failed: {e}")
            #print(f"[FEED] frame_data shape={frame_data.shape}, " #UNCOMMENT FOR DEBUG ON FRAME DATA - CORRECT FOR NOW
            #      f"y0 range=[{frame_data[:,6].min():.4f}, {frame_data[:,6].max():.4f}]")
            #print(f"[FEED] calling train_core with {frame_data.shape[0]} rows")      
            success = self.region.train_core(epoch=epoch, frame_data=frame_data)
            loss_val = getattr(self.region, 'last_loss', None)
            try:
                test = frame_data[:128, :6].clone().to(self.region.device)
                #test.requires_grad_(True)
                test_n = self.region.normalize_tensor(test)
                with torch.enable_grad():
                    tau, w, coords_in = self.region.function.network.out(test_n)
                    tt = tau[:, 0]
                    sp = self.region.function.Speed(test_n)

                g = torch.autograd.grad(tt.sum(), coords_in, retain_graph=True)[0]
                g0 = g[:, :3]  # gradient w.r.t x0
                g0_norm = torch.sqrt((g0 * g0).sum(dim=1) + 1e-12)
            except Exception as e:
                print("MODEL DEBUG FAILED:", e)

        except Exception as e:
            self.get_logger().error(f"feed_region error: {e}")
            import traceback
            traceback.print_exc()

    def _frontier_in_room(self, room_id, rooms):
        if room_id is None:
            return None
        if room_id not in rooms:
            return None
        room_cells = rooms[room_id]["cells"]
        frontier_pts = []
        for (i, j) in room_cells:
            if self.gridmap.frontier[i, j] == 1:
                x, y = self.gridmap._grid_to_world(i, j)
                frontier_pts.append([x, y])
        if len(frontier_pts) == 0:
            return None
        frontier_pts = np.array(frontier_pts)
        # choose closest to robot
        robot = self.robot_pos[:2]
        dists = np.linalg.norm(frontier_pts - robot[None, :], axis=1)
        idx = np.argmin(dists)
        return frontier_pts[idx] #END DIFF

    def pick_room_aware_goal(self, current_room, rooms, room_graph):
        robot_xy = self.robot_pos[:2]

        # 1. Try live local frontier (with blacklist)
        local = self._frontier_in_room(current_room, rooms)
        if local is not None:
            if (len(self._visited_goals) == 0 or
                    np.linalg.norm(np.array(self._visited_goals) - local[None,:], axis=1).min()
                    > self._goal_blacklist_radius):
                return local

        # 2. Try global live frontier
        live = self.pick_reachable_frontier()
        if live is not None:
            return live

        # 3. Local region has no live frontiers → check ExplorationGraph bank
        if self.region is not None:
            ver = self.region.version
            result = self.exploration_graph.best_remote_frontier(ver, robot_xy)
            if result is not None:
                target_ver, target_frontier, door_waypoints = result
                print(f"[EXPLORATION GRAPH] Routing to region {target_ver} "
                      f"via {len(door_waypoints)} door(s), "
                      f"frontier={target_frontier}")
                self._remote_door_waypoints = door_waypoints
                self._remote_target_version = target_ver
                return target_frontier

        print("[EXPLORATION GRAPH] Fully explored — no frontiers anywhere")
        return None

    def publish_field_path(self, path_xy, frame_id='odom'):
        """
        path_xy: list[np.ndarray] or np.ndarray of shape (N,2)
        """
        if path_xy is None:
            return
        if isinstance(path_xy, list):
            pts = np.asarray(path_xy, dtype=np.float32)
        else:
            pts = np.asarray(path_xy, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 2:
            return
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        poses = []
        for p in pts:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = float(p[0])
            ps.pose.position.y = float(p[1])
            ps.pose.position.z = 0.0
            ps.pose.orientation.w = 1.0
            poses.append(ps)
        msg.poses = poses
        self.field_path_pub.publish(msg)

    def decimate_path(self, path_xy, spacing=0.4):
        pts = np.asarray(path_xy, dtype=np.float32)
        if pts.ndim == 1:
            pts = pts.reshape(1, 2)
        if len(pts) < 2:
            return pts
        new_pts = [pts[0]]
        for i in range(1, len(pts)):
            if np.linalg.norm(pts[i] - new_pts[-1]) >= spacing:
                new_pts.append(pts[i])
        if np.linalg.norm(new_pts[-1] - pts[-1]) > 0.05:
            new_pts.append(pts[-1])
        return np.asarray(new_pts, dtype=np.float32)

    def smooth_path(self, path_xy, window=5):
        if path_xy is None:
            return None
        pts = np.asarray(path_xy, dtype=np.float32)
        if pts.ndim == 1:
            pts = pts.reshape(1, 2)
        if len(pts) < window:
            return pts
        sm = pts.copy()
        k = window // 2
        for i in range(len(pts)):
            a = max(0, i-k)
            b = min(len(pts), i+k+1)
            sm[i] = pts[a:b].mean(axis=0)
        sm[0] = pts[0]
        sm[-1] = pts[-1]
        return sm 

    def is_path_significantly_different(self, old_path, new_path, threshold=0.25):
        old_path = np.asarray(old_path)
        new_path = np.asarray(new_path)
        if len(old_path) < 2 or len(new_path) < 2:
            return True
        # compare up to min length
        n = min(len(old_path), len(new_path))
        # compare corresponding waypoints
        diffs = np.linalg.norm(old_path[:n] - new_path[:n], axis=1)
        max_diff = np.max(diffs)
        print("Max waypoint diff:", max_diff)
        return max_diff > threshold

    def train_tick(self):
        if self.region is None:
            print(f"[TRAIN_TICK] region=False, running={self.train_running}, "
                  f"queue_len={len(self.train_queue)}, epoch=-1")
            return
        print(f"[TRAIN_TICK] region=True, running={self.train_running}, "
              f"queue_len={len(self.train_queue)}, epoch={self.region.epoch}")
        if self.train_running:
            return  # previous train still going
        with self.train_lock:
            if len(self.train_queue) == 0:
                return
            self.train_running = True     
            world_pts, robot_pose = self.train_queue.pop()
            self.train_queue.clear()
        # ✅ Set BEFORE spawning thread — eliminates race condition
        def _do_train():
            try:
                print(f"[TRAIN THREAD] starting feed_region, epoch={self.region.epoch}")
                self.feed_region(world_pts, robot_pose)
                print(f"[TRAIN THREAD] done, epoch={self.region.epoch}")
            except Exception as e:
                self.get_logger().error(f"[TRAIN] failed: {e}")
                import traceback
                traceback.print_exc()
            finally:
                self.train_running = False
        t = threading.Thread(target=_do_train, daemon=True)
        t.start()

    def maybe_expand_region(self):
        if self.region is None:
            return
        robot = self.robot_pos[:2]
        bound = self.region.bound.numpy()
        margin = 0.3
        near_edge = (
            abs(robot[0] - bound[0,0]) < margin or
            abs(robot[0] - bound[1,0]) < margin or
            abs(robot[1] - bound[0,1]) < margin or
            abs(robot[1] - bound[1,1]) < margin)
        if near_edge:
            new_bound = torch.tensor([
                [bound[0,0]-0.3, bound[0,1]-0.3, -0.1],
                [bound[1,0]+0.3, bound[1,1]+0.3,  0.1],
            ], dtype=torch.float32)
            self.region.set_bound(new_bound)
            self.get_logger().info("[REGION EXPAND]")

    def pick_reachable_frontier(self, max_candidates=300):
        frontier = self.gridmap.frontier
        iy, ix = np.where(frontier > 0)
        if len(ix) == 0:
            return None
        cell   = self.gridmap.cell_size
        origin = self.gridmap.origin
        half   = self.gridmap.map_size / 2.0
        fx = (origin[0] - half) + (ix + 0.5) * cell
        fy = (origin[1] - half) + (iy + 0.5) * cell
        cand = np.stack([fx, fy], axis=1)
        robot = self.robot_pos[:2]
        d = np.linalg.norm(cand - robot[None, :], axis=1)
        valid = d < 2.0
        if valid.sum() == 0:
            return None
        cand = cand[valid]
        d    = d[valid]
        if self.region is not None:
            in_bound = np.array([self.region.in_bound(c) for c in cand])
            if in_bound.sum() > 0:
                cand = cand[in_bound]
                d    = d[in_bound]
            else:
                return cand[np.argmin(d)]   # geometric fallback
        if len(cand) > max_candidates:
            cand = cand[np.random.choice(len(cand), max_candidates, replace=False)]
        # No field yet → closest
        if self.region is None or self.region.function is None:
            return cand[np.argmin(d)]
        # Field query (all candidates guaranteed inside bound now)
        x0  = np.tile(np.array([robot[0], robot[1], 0.0]), (len(cand), 1))
        x1  = np.concatenate([cand, np.zeros((len(cand), 1))], axis=1)
        xp  = np.concatenate([x0, x1], axis=1)
        xp_t = torch.tensor(xp, dtype=torch.float32)
        xp_n = self.region.normalize_tensor(xp_t).to(self.region.device)
        with torch.no_grad():
            tt = self.region.function.TravelTimes(xp_n).cpu().numpy().reshape(-1)
        ok = np.isfinite(tt)
        if ok.sum() == 0:
            return cand[np.argmin(d)]
        return cand[ok][np.argmin(tt[ok])]

    def maybe_create_new_region(self):
        """
        Called every update() tick.
        If the robot has moved outside the current region's bound, either:
          (a) switch to an existing region that covers the robot, OR
          (b) spawn a new region centred on the robot.
        Old region is unloaded to disk to free GPU memory.
        """
        if self.region is None or self.robot_pos is None:
            return

        robot = self.robot_pos[:2]
        b = self.region.bound.numpy()
        # Is robot still comfortably inside current bound?
        inside = (
            robot[0] > b[0, 0] + REGION_EXIT_MARGIN and
            robot[0] < b[1, 0] - REGION_EXIT_MARGIN and
            robot[1] > b[0, 1] + REGION_EXIT_MARGIN and
            robot[1] < b[1, 1] - REGION_EXIT_MARGIN
        )
        if inside:
            return  # nothing to do
        # --- Robot has left (or is about to leave) the current region ---
        # (a) Check if any existing region already covers the robot
        for ver, reg in self.all_regions.items():
            if ver == self.region.version:
                continue
            if reg.in_bound(robot):
                if reg.loaded:
                    self._switch_region(reg)
                    return
                else:
                    reg.load_from_disk()
                    self._switch_region(reg)
                    return
        # (b) No existing region covers the robot → create a new one
        #     But only if the area has enough visited free space to train on
        visited = self.gridmap.visited
        cell = self.gridmap.cell_size
        origin = self.gridmap.origin
        half = self.gridmap.map_size / 2.0
        # Count visited cells in the proposed new bound
        def count_visited_in_bound(cx, cy, r):
            ix0 = max(0, int((cx - r - (origin[0] - half)) / cell))
            ix1 = min(self.gridmap.n_cells - 1, int((cx + r - (origin[0] - half)) / cell))
            iy0 = max(0, int((cy - r - (origin[1] - half)) / cell))
            iy1 = min(self.gridmap.n_cells - 1, int((cy + r - (origin[1] - half)) / cell))
            region_area = (ix1 - ix0) * (iy1 - iy0)
            if region_area <= 0:
                return 0.0
            visited_count = visited[iy0:iy1, ix0:ix1].sum()
            return float(visited_count) / float(region_area)

        coverage = count_visited_in_bound(robot[0], robot[1], REGION_RADIUS)
        if coverage < REGION_MIN_OVERLAP:
            # Not enough data yet — keep extending current region temporarily
            # by softly expanding its bound rather than spawning premature region
            self._soft_extend_current_region(robot)
            return
        # Enough data: spawn the new region
        self.get_logger().info(
            f"[NEW REGION] Robot at ({robot[0]:.2f},{robot[1]:.2f}) "
            f"left region {self.region.version}. "
            f"Visited coverage={coverage:.2f}. Spawning new region."
        )
        self._unload_current_region()
        self.init_region(self.robot_pos)

    def fit_region_to_lidar(self,
                             expand_thresh: float = 0.30,
                             shrink_margin: float = 0.50,
                             step: float = 0.15,
                             min_wall_span: float = 1.2,
                             max_area_mult: float = 3.5):
        """
        Fit the current region bounding box to ALL accumulated wall cells
        inside the region, not just the current frame's LiDAR hits.

        Source of truth: gridmap.data == 1 cells (confirmed walls, built up
        over the entire robot trajectory in this region).

        Per-edge logic
        ──────────────
        EXPAND  — wall cells exist within expand_thresh of the edge AND they
                  span at least min_wall_span metres along that edge.
                  The span guard filters doorways (~0.9 m wide) from full
                  room walls (typically >1.5 m), so we only expand when we
                  see a real wall, not a narrow opening.

        SHRINK  — no wall cell falls within shrink_margin of the edge,
                  meaning there is confirmed empty space between the last
                  wall and the boundary → pull inward by step.

        LEAVE   — wall cells are present but not pressing on the edge, and
                  the boundary is already a reasonable fit.

        Hard limits
        ───────────
        - Minimum half-size: 1.0 m on each axis.
        - Maximum area: max_area_mult × (2×REGION_RADIUS)² .
        - Robot must stay >= 0.5 m inside every edge after adjustment.
        - Only moves one step per call (smooth adaptation at ~1 Hz).
        """
        if self.region is None or self.robot_pos is None:
            return

        # ── Collect all confirmed wall cells inside the region bound ─────
        b     = self.region.bound.numpy().copy()
        robot = self.robot_pos[:2]

        iy_w, ix_w = np.where(self.gridmap.data == 1)
        if len(ix_w) == 0:
            return

        cell   = self.gridmap.cell_size
        origin = self.gridmap.origin
        half_m = self.gridmap.map_size / 2.0
        wx = (origin[0] - half_m) + (ix_w + 0.5) * cell
        wy = (origin[1] - half_m) + (iy_w + 0.5) * cell
        wall_pts = np.stack([wx, wy], axis=1)

        # Only walls already inside (or very close to) the current bound
        pad = expand_thresh + 0.05
        inside = (
            (wall_pts[:, 0] >= b[0, 0] - pad) & (wall_pts[:, 0] <= b[1, 0] + pad) &
            (wall_pts[:, 1] >= b[0, 1] - pad) & (wall_pts[:, 1] <= b[1, 1] + pad)
        )
        wall_pts = wall_pts[inside]
        if len(wall_pts) < 4:
            return

        MIN_HALF  = 1.0
        MAX_AREA  = max_area_mult * (REGION_RADIUS * 2) ** 2
        ROBOT_PAD = 0.5

        def _try_move(axis, side, delta):
            new_b = b.copy()
            if side == 0:
                new_b[0, axis] = b[0, axis] - delta
            else:
                new_b[1, axis] = b[1, axis] + delta
            if (new_b[1, 0] - new_b[0, 0]) / 2.0 < MIN_HALF:
                return
            if (new_b[1, 1] - new_b[0, 1]) / 2.0 < MIN_HALF:
                return
            if (new_b[1, 0] - new_b[0, 0]) * (new_b[1, 1] - new_b[0, 1]) > MAX_AREA:
                return
            if not (new_b[0, 0] + ROBOT_PAD < robot[0] < new_b[1, 0] - ROBOT_PAD and
                    new_b[0, 1] + ROBOT_PAD < robot[1] < new_b[1, 1] - ROBOT_PAD):
                return
            b[:] = new_b

        # Per-edge analysis using accumulated wall cells
        # For x-edges: only walls whose Y falls within the region Y span
        # For y-edges: only walls whose X falls within the region X span
        y_in = (wall_pts[:, 1] >= b[0, 1]) & (wall_pts[:, 1] <= b[1, 1])
        x_in = (wall_pts[:, 0] >= b[0, 0]) & (wall_pts[:, 0] <= b[1, 0])

        for axis, side, edge, relevant in [
            (0, 0, b[0, 0], wall_pts[y_in]),   # xmin (left)
            (0, 1, b[1, 0], wall_pts[y_in]),   # xmax (right)
            (1, 0, b[0, 1], wall_pts[x_in]),   # ymin (bottom)
            (1, 1, b[1, 1], wall_pts[x_in]),   # ymax (top)
        ]:
            if len(relevant) == 0:
                continue

            coords      = relevant[:, axis]
            perp_coords = relevant[:, 1 - axis]   # perpendicular axis for span

            if side == 0:
                near_mask = coords < edge + expand_thresh
                any_close = (coords < edge + shrink_margin).any()
            else:
                near_mask = coords > edge - expand_thresh
                any_close = (coords > edge - shrink_margin).any()

            if near_mask.any():
                # Measure span of near-edge walls along the perpendicular axis
                near_perp = perp_coords[near_mask]
                span = float(near_perp.max() - near_perp.min())
                if span >= min_wall_span:
                    _try_move(axis, side, step)   # real wall → expand
                # else: narrow hit cluster (doorway / corner) → leave
            elif not any_close:
                _try_move(axis, side, -step)      # open space → shrink

        if not np.allclose(b, self.region.bound.numpy(), atol=1e-4):
            new_bound = torch.tensor(
                [[b[0, 0], b[0, 1], -0.1],
                 [b[1, 0], b[1, 1],  0.1]], dtype=torch.float32)
            self.region.set_bound(new_bound)
            self.get_logger().info(
                f"[FIT REGION] R{self.region.version} "
                f"({b[0,0]:.2f},{b[0,1]:.2f})→({b[1,0]:.2f},{b[1,1]:.2f})"
            )

    def maybe_merge_regions(self):
        """
        For each confirmed room that spans multiple regions, merge those
        regions into one with a combined bounding box and retrain.
        """
        # Build room → [region_versions] mapping
        room_to_regions: Dict[int, List[int]] = {}
        for ver, region in self.all_regions.items():
            for room in region.rooms:
                if room.id not in room_to_regions:
                    room_to_regions[room.id] = []
                room_to_regions[room.id].append(ver)
        for room_id, versions in room_to_regions.items():
            if len(versions) <= 1:
                continue  # room already fits in one region
            # Only merge if all regions are confirmed (trained enough)
            regions_to_merge = [self.all_regions[v] for v in versions
                                if v in self.all_regions]
            if any(r.epoch < self.warmup_epochs_needed for r in regions_to_merge):
                continue  # wait until all are trained
            self.get_logger().info(
                f"[MERGE] Room {room_id} spans regions {versions} → merging")
            self._merge_regions(regions_to_merge)

    def _merge_regions(self, regions: List):
        """
        Merge a list of regions into one. The merged region gets:
        - Combined bounding box
        - All rooms from all merged regions
        - All training data concatenated
        - Retraining triggered immediately
        """
        # ── Compute combined bound ────────────────────────────────────
        bounds = [r.bound.numpy() for r in regions]
        new_min = np.array([min(b[0, i] for b in bounds) for i in range(3)])
        new_max = np.array([max(b[1, i] for b in bounds) for i in range(3)])
        new_bound = torch.tensor(
            np.stack([new_min, new_max]), dtype=torch.float32)

        # ── Check merged region isn't too large ──────────────────────
        w = new_max[0] - new_min[0]
        h = new_max[1] - new_min[1]
        MAX_MERGE_SIZE = REGION_RADIUS * 2 * 2   # 2× normal region size
        if w > MAX_MERGE_SIZE or h > MAX_MERGE_SIZE:
            self.get_logger().warn(
                f"[MERGE SKIPPED] merged bound too large ({w:.1f}×{h:.1f}m)")
            return

        # ── Create merged region ──────────────────────────────────────
        merged = Region(model=self.model, bound=new_bound,
                        init_network=True, eval=False)
        merged.gridmap = self.gridmap

        # Transfer all rooms
        all_rooms = [room for r in regions for room in r.rooms]
        for room in all_rooms:
            merged.add_room(room)   # also sets room.region = merged
        # Concatenate all_lidar from constituent regions
        lidar_chunks = [r.all_lidar for r in regions
                        if r.all_lidar is not None and len(r.all_lidar) > 0]
        if lidar_chunks:
            merged.all_lidar = np.concatenate(lidar_chunks, axis=0)
        # ── Remove old regions from all_regions and ExplorationGraph ─
        old_versions = [r.version for r in regions]
        for ver in old_versions:
            if ver in self.all_regions:
                try:
                    self.all_regions[ver].unload_to_disk()
                except Exception:
                    pass
                del self.all_regions[ver]
            if ver in self.exploration_graph.records:
                del self.exploration_graph.records[ver]
            if ver in self.exploration_graph.graph:
                self.exploration_graph.graph.remove_node(ver)

        # ── Register merged region ────────────────────────────────────
        self.all_regions[merged.version] = merged
        center = np.array(merged.center[:2], dtype=np.float32)
        self.exploration_graph.ensure_region(merged.version, center)

        # Rebuild spatial edges for merged region
        for v2, reg2 in self.all_regions.items():
            if v2 == merged.version:
                continue
            b1 = new_bound.numpy()
            b2 = reg2.bound.numpy()
            x_gap = max(0.0, max(b1[0,0], b2[0,0]) - min(b1[1,0], b2[1,0]))
            y_gap = max(0.0, max(b1[0,1], b2[0,1]) - min(b1[1,1], b2[1,1]))
            if x_gap < 1.0 and y_gap < 1.0:
                mx = (max(b1[0,0], b2[0,0]) + min(b1[1,0], b2[1,0])) / 2.0
                my = (max(b1[0,1], b2[0,1]) + min(b1[1,1], b2[1,1])) / 2.0
                self.exploration_graph.add_spatial_edge(
                    merged.version, v2, np.array([mx, my], dtype=np.float32))

        # ── If robot was in one of the merged regions, switch to merged ─
        if self.region is not None and self.region.version in old_versions:
            self.region = merged
            self.get_logger().info(
                f"[MERGE] Active region was merged → now region {merged.version}")

        # ── Trigger retraining on merged region ──────────────────────
        self.mode               = "WARMUP"
        self.warmup_start_epoch = int(merged.epoch)
        self.goal_xy            = None
        self.nav_state          = "IDLE"
        self.get_logger().info(
            f"[MERGE] Region {merged.version} created "
            f"(merged {old_versions}) → WARMUP"
        )

    def _switch_region(self, new_region: 'Region'):
        old_ver = self.region.version
        if self.region is not None and self.region.all_lidar is not None:
            new_region.all_lidar = self.region.all_lidar[-5:].copy()
            new_region.lidar_obs = None
            new_region.kd_tree   = None
        self._unload_current_region()
        # Flush stale training frames from the old region before switching
        with self.train_lock:
            self.train_queue.clear()
        self.train_running = False   # abort any in-flight train for old region
        self.region = new_region
        self.get_logger().info(f"[REGION SWITCH] {old_ver} → {new_region.version}")
        self.goal_xy        = None
        self.last_goal_time = self.get_clock().now()
        self.gridmap.frontier[:] = 0

    def _unload_current_region(self):
        """Save current region weights to disk and free GPU memory."""
        if self.region is None:
            return
        try:
            self.region.save_model()          # saves checkpoint to region folder
            self.region.unload_to_disk()      # releases GPU tensors
            self.get_logger().info(
                f"[REGION {self.region.version} UNLOADED]"
            )
        except Exception as e:
            self.get_logger().error(f"[UNLOAD ERROR] {e}")

    def _soft_extend_current_region(self, robot):
        """
        Softly expand region a bit so we don't immediately spawn a new one,
        BUT stop expanding once region gets too large.
        """
        b = self.region.bound.numpy().copy()
        pad = 0.3
        # --- proposed new bound ---
        new_min_x = min(b[0, 0], robot[0] - REGION_RADIUS - pad)
        new_min_y = min(b[0, 1], robot[1] - REGION_RADIUS - pad)
        new_max_x = max(b[1, 0], robot[0] + REGION_RADIUS + pad)
        new_max_y = max(b[1, 1], robot[1] + REGION_RADIUS + pad)
        # --- current size ---
        cur_w = b[1, 0] - b[0, 0]
        cur_h = b[1, 1] - b[0, 1]
        # --- new size if applied ---
        new_w = new_max_x - new_min_x
        new_h = new_max_y - new_min_y
        # 🧠 IMPORTANT: cap region size
        MAX_SIZE = REGION_RADIUS * 2.0   # ≈ 2.5m if radius=1.0
        if new_w > MAX_SIZE or new_h > MAX_SIZE:
            # Stop expanding → allow new region to spawn
            self.get_logger().info(
                f"[SOFT EXTEND BLOCKED] region already large "
                f"(w={cur_w:.2f}, h={cur_h:.2f})"
            )
            return

        # --- apply extension ---
        b[0, 0] = new_min_x
        b[0, 1] = new_min_y
        b[1, 0] = new_max_x
        b[1, 1] = new_max_y

        new_bound = torch.tensor(b, dtype=torch.float32)
        self.region.set_bound(new_bound)

        self.get_logger().info(
            f"[SOFT EXTEND] new bound: "
            f"({b[0,0]:.2f},{b[0,1]:.2f}) → ({b[1,0]:.2f},{b[1,1]:.2f})"
        )

    def _launch_path_gen(self, robot_xy, goal_xy): #claude diff
        """Offload generate_field_path to a background thread so update() never blocks."""
        if robot_xy is None or goal_xy is None:          # ← add this guard
            return
        if self.region is None or self.region.function is None:
            return
        if getattr(self, '_path_gen_running', False):
            return  # already running, skip
        self._path_gen_running = True
        robot_xy = robot_xy.copy()
        goal_xy  = goal_xy.copy()

        def _worker():
            try:
                raw_path = self.region.generate_field_path(
                    robot_xy, goal_xy, step_size=0.10, max_steps=200
                )
                if raw_path is not None:
                    raw_path = np.asarray(raw_path, dtype=np.float32)
                    if len(raw_path) > 1:
                        decimate = self.decimate_path(raw_path, spacing=0.2)
                        smooth   = self.smooth_path(decimate, window=3)
                        self.latest_candidate_path = np.asarray(smooth, dtype=np.float32)
            except Exception as e:
                self.get_logger().warn(f"[PATH GEN] {e}")
            finally:
                self._path_gen_running = False
        threading.Thread(target=_worker, daemon=True).start() #end claude diff

    def _maybe_create_new_region_guarded(self):
        """
        Only allow region creation when robot is NOT actively navigating.
        """
        if self.region is None:
            self.maybe_create_new_region()
            return

        if self.mode != "EXPLORE":
            self.maybe_create_new_region()
            return

        if self.goal_xy is None:
            self.maybe_create_new_region()
            return

        # Actively navigating → block
        if self.nav_state in ("STABILIZING", "FOLLOWING"):
            robot_xy = self.robot_pos[:2]
            d = np.linalg.norm(robot_xy - self.goal_xy)
            if d < 0.20:
                self.maybe_create_new_region()
            else:
                return  # still en-route, block creation
        else:
            # IDLE or DWELL → both are safe, robot is stationary
            self.maybe_create_new_region()

    def _update_region_graph_via_free_cells(self):
        """
        Build/update ExplorationGraph edges between regions using cells that:
          - are free/traversable (gridmap.data == 0, not occupied)
          - have been visited by the robot (gridmap.visited > 0)
          - fall within the discovered area of BOTH regions simultaneously
            (discovery_map has bits set for both region versions)

        This replaces boundary-midpoint and door-based spatial edges entirely.
        """
        versions = list(self.all_regions.keys())
        if len(versions) < 2:
            return

        # Build a mask of cells that are free AND visited
        free_visited = (self.gridmap.data == 0) & (self.gridmap.visited > 0)
        if not free_visited.any():
            return
        cell_indices = np.argwhere(free_visited)  # Nx2 array of (i, j)
        disc = self.gridmap.discovery_map  # uint64 bitmask per cell
        for v1, v2 in combinations(versions, 2):
            bit1 = np.uint64(1) << np.uint64(v1)
            bit2 = np.uint64(1) << np.uint64(v2)
            # Cells visited by BOTH regions
            shared_mask = free_visited & \
                          ((disc & bit1) != 0) & \
                          ((disc & bit2) != 0)
            shared_cells = np.argwhere(shared_mask)
            if len(shared_cells) < 3:   # need at least a few shared cells
                continue
            # Convert a representative subset to world coords as waypoints
            # Use up to 5 spread-out points via simple stride sampling
            stride = max(1, len(shared_cells) // 5)
            sampled = shared_cells[::stride][:5]
            waypoints = [
                np.array(self.gridmap._grid_to_world(int(ci), int(cj)))
                for ci, cj in sampled
            ]
            self.exploration_graph.add_free_cell_edge(v1, v2, waypoints)

    def _assign_frontiers_to_regions(self, current_version: int):
        """
        Per-cell frontier ownership:
        - Cell discovered only by current region → current region.
        - Cell discovered only by one remote (non-exhausted) region → that region.
        - Cell in overlap with current region → current region (always).
        - Cell in overlap not including current, or remote is exhausted → current region.
        """
        frontier_cells = np.argwhere(self.gridmap.frontier > 0)
        if len(frontier_cells) == 0:
            return
        disc = self.gridmap.discovery_map
        exclusive: dict = {}   # {version: [np.ndarray, ...]}
        current_pts = []
        for ci, cj in frontier_cells:
            cell_bits = disc[ci, cj]
            versions_here = self.gridmap.discover_regions(cell_bits)
            wx, wy = self.gridmap._grid_to_world(int(ci), int(cj))
            pt = np.array([wx, wy], dtype=np.float32)
            if current_version in versions_here or len(versions_here) == 0:
                # Belongs to current (own cell, overlap with current, or undiscovered)
                current_pts.append(pt)
            elif len(versions_here) == 1:
                v = versions_here[0]
                rec = self.exploration_graph.records.get(v)
                if rec is not None and not rec.is_exhausted():
                    exclusive.setdefault(v, []).append(pt)
                else:
                    current_pts.append(pt)   # exhausted remote → current takes it
            else:
                # Multi-region overlap not including current → current takes it
                current_pts.append(pt)
        # Assign exclusive frontiers to non-exhausted remote regions
        for version, pts in exclusive.items():
            region_obj = self.all_regions.get(version)
            if region_obj is None:
                continue
            # Safe center extraction whether bound is a tensor or ndarray
            b = region_obj.bound
            if hasattr(b, 'cpu'):
                b = b.cpu().numpy()
            center = b[:2].mean(axis=0)[:2].astype(np.float32)
            self.exploration_graph.ensure_region(version, center)
            for pt in pts:
                self.exploration_graph.records[version].add_frontier(
                    pt, ExplorationGraph.BLACKLIST_RADIUS
                )
        # All current-region and overlap frontiers → current region
        for pt in current_pts:
            self.exploration_graph.records[current_version].add_frontier(
                pt, ExplorationGraph.BLACKLIST_RADIUS
            )

    def _build_is_frontier_fn(self):
        """Returns a closure over the current frontier grid for bank reconciliation."""
        frontier_cells = np.argwhere(self.gridmap.frontier > 0)
        if len(frontier_cells) == 0:
            def is_frontier_fn(xy, radius):
                return False
            return is_frontier_fn
        # World coords of all live frontier cells
        world_pts = np.array([
            self.gridmap._grid_to_world(int(ci), int(cj))
            for ci, cj in frontier_cells
        ])
        tree = cKDTree(world_pts)
        def is_frontier_fn(xy: np.ndarray, radius: float) -> bool:
            hits = tree.query_ball_point(xy[:2], r=radius)
            return len(hits) > 0
        return is_frontier_fn

    def _start_goal_dwell(self):
        dwell_frames_needed = int(self.get_parameter('dwell_frames_needed').value)
        if self.region is None or dwell_frames_needed <= 0:
            self.goal_xy   = None
            self.nav_state = "IDLE"
            return

        self.dwell_active       = True
        self.dwell_frames       = []
        self.dwell_frames_count = 0
        self.dwell_phase        = "COLLECT"
        self.nav_state          = "DWELL"
        self.goal_xy            = None
        self.navigator.stop_robot()
        # Flush any in-flight frames from navigation so we get a clean epoch baseline
        with self.train_lock:
            self.train_queue.clear()
        # Epoch baseline set AFTER flush so we don't count pre-dwell epochs
        self.dwell_epoch_start = int(self.region.epoch)
        print(f"[DWELL] Starting — collecting {dwell_frames_needed} frames "
              f"then training {int(self.get_parameter('dwell_epochs').value)} epochs "
              f"(baseline epoch={self.dwell_epoch_start})")

    def _tick_goal_dwell(self, world_points_360: np.ndarray, robot_xy: np.ndarray):
        dwell_frames_needed = int(self.get_parameter('dwell_frames_needed').value)
        dwell_epochs        = int(self.get_parameter('dwell_epochs').value)
        # If region changed under us (new region created), abort dwell cleanly
        if self.region is not None and int(self.region.epoch) < self.dwell_epoch_start:
            print("[DWELL] Region switched during dwell — ending dwell, entering WARMUP")
            self._end_goal_dwell()
            # Trigger warmup for the new region
            self.mode               = "WARMUP"
            self.warmup_start_epoch = int(self.region.epoch)
            self.warmup_done        = False
            return
        if self.dwell_phase == "COLLECT":
            if self.region is not None and len(world_points_360) > 5:
                self.dwell_frames.append((
                    world_points_360.copy(),
                    self.robot_pos.copy() if self.robot_pos is not None else robot_xy))
                self.dwell_frames_count += 1
                print(f"[DWELL] Collected frame {self.dwell_frames_count}/{dwell_frames_needed}")

            if self.dwell_frames_count >= dwell_frames_needed:
                with self.train_lock:
                    for frame in self.dwell_frames:
                        self.train_queue.append(frame)
                self.dwell_epoch_start = int(self.region.epoch)
                self.dwell_phase       = "TRAIN"
                print(f"[DWELL] Collection done — training for {dwell_epochs} epochs "
                      f"from epoch {self.dwell_epoch_start}")

        elif self.dwell_phase == "TRAIN":
            if self.region is None:
                self._end_goal_dwell()
                return
            epochs_done = int(self.region.epoch) - self.dwell_epoch_start
            print(f"[DWELL] Training epoch {epochs_done}/{dwell_epochs}")
            if epochs_done >= dwell_epochs:
                print(f"[DWELL] Training complete — resuming navigation")
                self._end_goal_dwell()

    def _end_goal_dwell(self):
        """Clean up dwell state and return to normal navigation."""
        self.dwell_active  = False
        self.dwell_frames  = []
        self.dwell_phase   = "COLLECT"
        self.nav_state     = "IDLE"
        self.goal_xy       = None

    def update(self):
        now = self.get_clock().now()
        # ── Prerequisites ────────────────────────────────────────────────
        if self.robot_pos is None or self.latest_cloud is None:
            return
        result = self.adapter.update(self.latest_cloud, self.robot_pos)
        if result is None:
            return
        points_lidar, speeds = result
        if len(points_lidar) == 0:
            return
        # ── TF transform ─────────────────────────────────────────────────
        try:
            source_frame = self.latest_cloud.header.frame_id
            stamp        = self.latest_cloud.header.stamp
            if self.tf_buffer.can_transform('odom', source_frame, stamp,
                                            timeout=Duration(seconds=0.1)):
                transform = self.tf_buffer.lookup_transform('odom', source_frame, stamp)
            else:
                return
            t = transform.transform.translation
            q = transform.transform.rotation
            R = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
            world_points = (R @ points_lidar.T).T + np.array([t.x, t.y, t.z])
        except Exception as e:
            self.get_logger().warn(f"TF lookup failed: {e}")
            return

        # ── Bound filter ──────────────────────────────────────────────────
        robot_xy = self.robot_pos[:2]
        half = self.gridmap.map_size / 2
        ox, oy = self.gridmap.origin
        in_bounds = ((world_points[:, 0] > ox - half) & (world_points[:, 0] < ox + half) &
                     (world_points[:, 1] > oy - half) & (world_points[:, 1] < oy + half))
        world_points_filtered = world_points[in_bounds]
        speeds_filtered = speeds[in_bounds]
        if len(world_points_filtered) < 5:
            return
        # Angular downsampling + free-ray synthesis
        # For each of 360 angular bins:
        #   - Real hit bin  → closest point from sensor; hit_dist = measured dist
        #   - Empty bin     → synthetic endpoint at MAX_RANGE; hit_dist = inf
        # The raycast loop later uses hit_dist to decide whether to mark a wall.
        robot = robot_xy
        vecs   = world_points_filtered[:, :2] - robot_xy[None, :]
        dists  = np.linalg.norm(vecs, axis=1) + 1e-9
        angles = np.arctan2(vecs[:, 1], vecs[:, 0])
        bins   = ((angles + np.pi) / (2 * np.pi) * 359).astype(np.int32)

        closest_idx  = {}
        closest_dist = np.full(360, np.inf, dtype=np.float32)
        for k, b in enumerate(bins):
            if dists[k] < closest_dist[b]:
                closest_dist[b] = dists[k]
                closest_idx[b]  = k

        # Real-hit cloud — identical to the original method; used for raycast + gridmap
        keep = np.array(list(closest_idx.values()), dtype=np.int32)
        world_points_hit = world_points_filtered[keep]
        speeds_hit       = speeds_filtered[keep]

        # Full 360-bin cloud — real hits + synthetic free-ray endpoints
        # Used ONLY for training data collection; never touches gridmap/raycast.
        full_endpoints  = []
        full_speeds     = []
        for b in range(360):
            if b in closest_idx:
                idx = closest_idx[b]
                full_endpoints.append(world_points_filtered[idx])
                full_speeds.append(speeds_filtered[idx])
            else:
                theta  = (b / 359.0) * 2.0 * np.pi - np.pi
                dir_xy = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
                free_xy = robot_xy + MAX_RANGE * dir_xy
                free_z  = float(self.robot_pos[2]) if len(self.robot_pos) > 2 else 0.0
                full_endpoints.append(np.array([free_xy[0], free_xy[1], free_z], dtype=np.float32))
                full_speeds.append(0.0)
        world_points_360 = np.asarray(full_endpoints, dtype=np.float32)

        # ── Raycast: update gridmap.data + visited (real hits only) ──────
        cell   = self.gridmap.cell_size
        origin = self.gridmap.origin
        half_  = self.gridmap.map_size / 2

        def world_to_ij(x, y):
            return (int((x - origin[0] + half_) / cell),
                    int((y - origin[1] + half_) / cell))

        for pt3d in world_points_hit:
            pt = pt3d[:2]
            dx, dy = pt[0] - robot_xy[0], pt[1] - robot_xy[1]
            dist = float(np.hypot(dx, dy))
            if dist > MAX_RANGE:
                scale = MAX_RANGE / dist
                dx *= scale; dy *= scale; dist = MAX_RANGE
            steps = int(dist / cell)
            if steps < 1:
                ix, iy = world_to_ij(pt[0], pt[1])
                if 0 <= ix < self.gridmap.n_cells and 0 <= iy < self.gridmap.n_cells:
                    self.gridmap.data[iy, ix] = 1
                    self.gridmap.visited[iy, ix] = 1
                continue
            for s in range(steps - 1):
                x = robot_xy[0] + dx * s / steps
                y = robot_xy[1] + dy * s / steps
                ix, iy = world_to_ij(x, y)
                if not (0 <= ix < self.gridmap.n_cells and
                        0 <= iy < self.gridmap.n_cells):
                    break
                if self.gridmap.data[iy, ix] == 1:
                    break
                self.gridmap.visited[iy, ix] = 1
            ix_end, iy_end = world_to_ij(pt[0], pt[1])
            if (0 <= ix_end < self.gridmap.n_cells and
                    0 <= iy_end < self.gridmap.n_cells):
                self.gridmap.data[iy_end, ix_end] = 1
                self.gridmap.visited[iy_end, ix_end] = 1

        # ── Gridmap update (1 Hz room segmentation) ──────────────────────
        self._map_update_counter += 1
        do_room_update = (self._map_update_counter % 5 == 0)
        self.gridmap.update(
            lidar_scan=world_points_hit[:, :2],
            points=world_points_hit[:, :2],
            speeds=speeds_hit,
            cur_pos=robot_xy,
            region=self.region,
            regions=self.all_regions,
            update_room=do_room_update
        )

        # ── Snapshot live frontiers into ExplorationGraph ────────────────
        if self.region is not None:
            ver = self.region.version
            center = np.array(self.region.center[:2], dtype=np.float32)
            self.exploration_graph.ensure_region(ver, center)
            self._assign_frontiers_to_regions(ver)

        is_frontier_fn = self._build_is_frontier_fn()
        self.exploration_graph.reconcile_all_banks(is_frontier_fn, evict_radius=0.6)

        if do_room_update:
            for u, v, edge_data in self.gridmap.room_graph.edges(data=True):
                room_u = self.gridmap.room_graph.nodes[u].get('room', -1)
                room_v = self.gridmap.room_graph.nodes[v].get('room', -1)

                # Only register edges between two confirmed rooms
                if room_u not in self.gridmap.confirmed_rooms:
                    continue
                if room_v not in self.gridmap.confirmed_rooms:
                    continue

                pos_u = self.gridmap.room_graph.nodes[u].get('pos')
                pos_v = self.gridmap.room_graph.nodes[v].get('pos')
                if pos_u is None or pos_v is None:
                    continue

                door_xy = np.array([(pos_u[0]+pos_v[0])/2,
                                     (pos_u[1]+pos_v[1])/2], dtype=np.float32)

                reg_u = getattr(self.gridmap.confirmed_rooms[room_u], 'region', None)
                reg_v = getattr(self.gridmap.confirmed_rooms[room_v], 'region', None)
                if reg_u is None or reg_v is None:
                    continue
                if reg_u.version == reg_v.version:
                    continue   # same region, spatial edge already covers this

                self.exploration_graph.add_door_edge(reg_u.version, reg_v.version, door_xy)
                self._update_region_graph_via_free_cells()

        # ── Enqueue training frame (ALL modes) ───────────────────────────
        if self.region is not None and bool(self.get_parameter('train_after_region').value):
            self.train_counter += 1
            if self.train_counter % self.train_every_n_frames == 0:
                with self.train_lock:
                    self.train_queue.append((
                        world_points_360.copy(),
                        self.robot_pos.copy()
                    ))
        if self._map_update_counter % 3 == 0 and self.mode != "BOOTSTRAP":
            self.fit_region_to_lidar()

        # ── Region boundary check ─────────────────────────────────────────
        self._maybe_create_new_region_guarded()
        if self._map_update_counter % 10 == 0:
            self.maybe_merge_regions()
        # ── Trajectory ───────────────────────────────────────────────────
        if self.robot_pos is not None:
            self._traj_history.append(self.robot_pos[:2].copy())

        # ── Mode state machine (navigation only, map already updated above) ──
        if self.mode == "BOOTSTRAP":
            self.navigator.stop_robot()
            self.bootstrap_frames += 1
            if self.region is None and self.bootstrap_frames >= self.bootstrap_threshold:
                self.get_logger().info("Initializing first region")
                self.init_region(self.robot_pos)
                self.mode              = "WARMUP"
                self.warmup_start_epoch = int(self.region.epoch)
                self.goal_xy           = None
                self.last_goal_time    = now
                self.get_logger().info("[FLOW] Entering WARMUP phase")

        elif self.mode == "WARMUP":
            # Stop robot, check epochs — but do NOT return; fall through to publish
            self.navigator.stop_robot()
            if self.region is not None:
                cur_epochs = int(self.region.epoch) - int(self.warmup_start_epoch)
                print(f"[WARMUP] epoch {cur_epochs}/{self.warmup_epochs_needed}")
                if cur_epochs >= self.warmup_epochs_needed:
                    self.mode        = "EXPLORE"
                    self.warmup_done = True
                    self.goal_xy     = None
                    self.last_goal_time = now
                    self.get_logger().info("[FLOW] Warmup complete → starting EXPLORE")

        elif self.mode == "EXPLORE":
            robot_xy = np.asarray(self.robot_pos[:2], dtype=np.float32)
            now_sec  = now.nanoseconds * 1e-9

            # ── IDLE: pick a new frontier goal ───────────────────────────
            if self.nav_state == "IDLE":
                current_room = self.gridmap.get_room_at_position(robot_xy)
                rooms        = self.gridmap.get_rooms()
                room_graph   = self.gridmap.get_room_graph()
                g = self.pick_room_aware_goal(current_room, rooms, room_graph)

                if g is None:
                    self.navigator.stop_robot()
                    print("[NO FRONTIER]")
                else:
                    self.goal_xy           = np.asarray(g, dtype=np.float32)
                    self.wait_start_time   = now_sec
                    self._path_gen_running = False
                    print("[GOAL SET]", self.goal_xy)

                    # ── If ExplorationGraph gave us door waypoints, pre-load them ─
                    if self._remote_door_waypoints:
                        print(f"[EXPLORATION GRAPH] Using {len(self._remote_door_waypoints)}"
                              f" door waypoints → region {self._remote_target_version}")
                        # Build composite path: door chain + final frontier goal
                        full_wp = np.array(
                            self._remote_door_waypoints + [self.goal_xy],
                            dtype=np.float32
                        )
                        self.latest_candidate_path = full_wp
                        self._remote_door_waypoints = []   # consumed
                    else:
                        self.latest_candidate_path = None  # normal: path gen will fill this

                    self.nav_state = "STABILIZING"
            # ── STABILIZING: generate path in background, don't block ────
            elif self.nav_state == "STABILIZING":
                self.navigator.stop_robot()
                # kick off path gen if not already running
                self._launch_path_gen(robot_xy, self.goal_xy)

                if now_sec - self.wait_start_time >= self.wait_duration:
                    if self.latest_candidate_path is not None:
                        print("[PATH FROZEN]")
                        self.path = self.latest_candidate_path.copy()
                        self.navigator.reset_path()
                        self.nav_state = "FOLLOWING"
                    else:
                        print("[NO PATH FOUND — retrying]")
                        self.wait_start_time = now_sec  # extend wait

            # ── FOLLOWING: drive, continuously refresh path in background ─
            elif self.nav_state == "FOLLOWING":
                if self.goal_xy is None:
                    self.nav_state = "IDLE"
                    self.path = None
                    return
                if self.goal_xy is not None:
                    d_goal = float(np.linalg.norm(robot_xy - self.goal_xy))
                else:
                    d_goal = None
                #print(f"Distance to frontier: {d_goal:.2f}")
                if d_goal is None:
                    self.nav_state = "IDLE"
                    self.goal_xy = None
                    self.path = None 

                if d_goal is not None and d_goal < 0.25:
                    print("[FRONTIER REACHED]")
                    self.navigator.stop_robot()
                    self.navigator.reset_path()

                    if self.goal_xy is not None:
                        # ── Clear frontier cells around reached goal ──────────────
                        gix = int((self.goal_xy[0] - (self.gridmap.origin[0] - self.gridmap.map_size/2))
                                  / self.gridmap.cell_size)
                        giy = int((self.goal_xy[1] - (self.gridmap.origin[1] - self.gridmap.map_size/2))
                                  / self.gridmap.cell_size)
                        pad = 3
                        for di in range(-pad, pad+1):
                            for dj in range(-pad, pad+1):
                                ii, jj = giy+di, gix+dj
                                if 0 <= ii < self.gridmap.n_cells and 0 <= jj < self.gridmap.n_cells:
                                    self.gridmap.frontier[ii, jj] = 0
                                    self.gridmap.visited[ii, jj]  = 1

                        # ── Update visited blacklist ──────────────────────────────
                        self._visited_goals.append(self.goal_xy.copy())
                        if len(self._visited_goals) > 50:
                            self._visited_goals.pop(0)

                        # ── Notify ExplorationGraph ───────────────────────────────
                        if self.region is not None:
                            self.exploration_graph.mark_goal_reached(
                                self.region.version, self.goal_xy
                            )

                    #self.goal_xy   = None
                    self.path      = None
                    self._remote_door_waypoints = []       # clear any leftover door chain
                    self._remote_target_version = None
                    self._start_goal_dwell()
                    if not self.dwell_active:
                        self.goal_xy = None
                        self.nav_state = "IDLE"

                else:
                    if self.goal_xy is None:       # ← double-check before path gen
                        self.nav_state = "IDLE"
                        return
                    # Continuously refresh the path as the network improves
                    self._launch_path_gen(robot_xy, self.goal_xy)
                    if self.latest_candidate_path is not None:
                        self.path                  = self.latest_candidate_path
                        self.latest_candidate_path = None
                        self.navigator.reset_path()
                    if self.path is not None:
                        self.navigator.drive_path(self.path)
            
            elif self.nav_state == "DWELL":
                self.navigator.stop_robot()
                self._tick_goal_dwell(world_points_360, robot_xy)

        # ── Composite plot ────────────────────────────────────────────────
        if self._composite_step % 20 == 0:
            try:
                self.save_composite_plot()
                self.save_occupancy_snapshot()
            except Exception as e:
                self.get_logger().warn(f"[COMPOSITE PLOT] {e}")
        self._composite_step += 1

        # ── Always publish (every tick, every mode) ───────────────────────
        self.map_publisher.publish_maps()
        self.markers.publish_goal_marker()
        self.markers.publish_frontier_markers()
        self.publish_field_path(self.path)

    def save_composite_plot(self):
        """
        plot2-style composite:
          - world-scale background (unknown / visited / obstacles)
          - per-region travel-time contours that PERSIST via a numeric cache (no translucent overlays)
          - exploration graph (nodes/edges/doors)
          - ONLY the latest saved frontiers per region (not the entire bank)
        """
        from matplotlib.lines import Line2D
        os.makedirs(self.folder, exist_ok=True)
        out_dir = os.path.join(self.folder, "plots2")
        os.makedirs(out_dir, exist_ok=True)

        # World-map bounds (gridmap is world-sized)
        b_map = [
            [self.gridmap.origin[0] - self.gridmap.map_size / 2,
             self.gridmap.origin[1] - self.gridmap.map_size / 2],
            [self.gridmap.origin[0] + self.gridmap.map_size / 2,
             self.gridmap.origin[1] + self.gridmap.map_size / 2],
        ]

        fig, ax = plt.subplots(1, 1, figsize=(14, 14))

        # ── Background: visited=dark green, obstacle=black, unknown=dark blue ──
        data = self.gridmap.data
        vis  = self.gridmap.visited
        n    = self.gridmap.n_cells
        cell = self.gridmap.cell_size

        bg = np.zeros((n, n, 3), dtype=np.uint8)
        bg[:, :] = [10, 10, 80]        # unknown
        bg[vis == 1]  = [220, 220, 40]     # visited free
        bg[data == 1] = [0, 0, 0]      # obstacles

        extent = [b_map[0][0], b_map[1][0], b_map[0][1], b_map[1][1]]
        ax.imshow(bg, origin='lower', extent=extent, zorder=0, aspect='auto')
        # ── Persistent numeric cache for region fields ─────────────────────────
        # Cache: version -> dict(X_w, Y_w, TT_masked, extent, src_world, step, epoch)
        if not hasattr(self, "_region_field_cache"):
            self._region_field_cache = {}

        # Tuning
        GRID_N = 90                 # resolution of per-region TT grid
        CONTOUR_LEVELS = np.arange(0, 10, 0.05)        # number of contour levels
        FRONTIER_LAST_K = 30        # only show last K saved frontiers per region
        CACHE_UPDATE_EVERY = 1      # update active region every step (keep expanding field visible)

        def _build_region_mask(b, grid_n=GRID_N):
            """Mask out cells outside visited/obstacle footprint (same logic as your original)."""
            mask = np.zeros((grid_n, grid_n), dtype=bool)
            for i in range(grid_n):
                for j in range(grid_n):
                    wx = b[0, 0] + (i + 0.5) / grid_n * (b[1, 0] - b[0, 0])
                    wy = b[0, 1] + (j + 0.5) / grid_n * (b[1, 1] - b[0, 1])
                    gx = int((wx - b_map[0][0]) / cell)
                    gy = int((wy - b_map[0][1]) / cell)
                    if 0 <= gx < n and 0 <= gy < n:
                        mask[j, i] = (vis[gy, gx] == 1 or data[gy, gx] == 1)
            return mask

        def _compute_tt_grid(region, src_world, b, mask, grid_n=GRID_N):
            """Compute TT(x0=src, x1=grid) over region bound and apply mask => TT_masked."""
            xmin, ymin = float(b[0, 0]), float(b[0, 1])
            xmax, ymax = float(b[1, 0]), float(b[1, 1])
            xs = np.linspace(xmin, xmax, grid_n)
            ys = np.linspace(ymin, ymax, grid_n)
            X_w, Y_w = np.meshgrid(xs, ys)

            # world -> normalized using region bound
            sx = (xmax - xmin) + 1e-9
            sy = (ymax - ymin) + 1e-9
            X_n = (X_w - xmin) / sx - 0.5
            Y_n = (Y_w - ymin) / sy - 0.5

            src_norm = region.normalize(src_world.reshape(1, 3))[0]  # (3,)

            XP = np.zeros((X_w.size, 2 * region.dim), dtype=np.float32)
            XP[:, :region.dim] = src_norm
            XP[:, region.dim + 0] = X_n.reshape(-1)
            XP[:, region.dim + 1] = Y_n.reshape(-1)
            XP[:, region.dim + 2] = src_norm[2]

            XP_t = torch.tensor(XP, dtype=torch.float32, device=region.device)

            with torch.no_grad():
                TT_t = region.function.TravelTimes(XP_t)
                if TT_t.ndim == 2 and TT_t.shape[1] == 1:
                    TT_t = TT_t[:, 0]
                TT = TT_t.detach().cpu().numpy().reshape(X_w.shape)

            TT_masked = TT.copy()
            TT_masked[~mask] = np.nan

            return X_w, Y_w, TT_masked

        def _pick_src_world(region, b):
            """Stable source for plotting TT contours."""
            # Active region uses robot position if available
            if self.region is not None and region.version == self.region.version and self.robot_pos is not None:
                return np.array([self.robot_pos[0], self.robot_pos[1], 0.0], dtype=np.float32)

            # Inactive: prefer cached source if present, else region center
            cached = self._region_field_cache.get(region.version)
            if cached is not None and "src_world" in cached:
                return cached["src_world"]

            rc = (b[0] + b[1]) / 2.0
            return np.array([float(rc[0]), float(rc[1]), 0.0], dtype=np.float32)

        # ── Update cache for the active region (so its field "expands" live) ──
        if self.region is not None and (self._composite_step % CACHE_UPDATE_EVERY == 0):
            try:
                reg = self.region
                if reg.function is not None:
                    b = reg.bound.detach().cpu().numpy()
                    mask = _build_region_mask(b, grid_n=GRID_N)
                    src_world = _pick_src_world(reg, b)
                    X_w, Y_w, TT_masked = _compute_tt_grid(reg, src_world, b, mask, grid_n=GRID_N)
                    self._region_field_cache[reg.version] = dict(
                        X_w=X_w, Y_w=Y_w, TT=TT_masked,
                        extent=[b[0, 0], b[1, 0], b[0, 1], b[1, 1]],
                        mask=mask, src_world=src_world,
                        step=int(self._composite_step),
                        epoch=int(getattr(reg, "epoch", 0)),
                    )
            except Exception as e:
                self.get_logger().warn(f"[COMPOSITE] active region cache update failed: {e}")

        # ── Ensure each region has at least one cached snapshot (persist) ─────
        for region in self.all_regions.values():
            if region.version in self._region_field_cache:
                continue
            # Only cache if model exists right now; otherwise it will remain missing until loaded once
            if region.function is None:
                continue
            try:
                b = region.bound.detach().cpu().numpy()
                mask = _build_region_mask(b, grid_n=GRID_N)
                src_world = _pick_src_world(region, b)
                X_w, Y_w, TT_masked = _compute_tt_grid(region, src_world, b, mask, grid_n=GRID_N)
                self._region_field_cache[region.version] = dict(
                    X_w=X_w, Y_w=Y_w, TT=TT_masked,
                    extent=[b[0, 0], b[1, 0], b[0, 1], b[1, 1]],
                    mask=mask, src_world=src_world,
                    step=int(self._composite_step),
                    epoch=int(getattr(region, "epoch", 0)),
                )
            except Exception as e:
                self.get_logger().warn(f"[COMPOSITE] region {region.version} initial cache failed: {e}")

        # ── Draw persistent contours for all cached regions ───────────────────
        for region in self.all_regions.values():
            b = region.bound.detach().cpu().numpy()

            cached = self._region_field_cache.get(region.version, None)
            if cached is not None:
                X_w = cached["X_w"]
                Y_w = cached["Y_w"]
                TT  = cached["TT"]

                if np.isfinite(TT).any():
                    ax.contour(
                        X_w, Y_w, TT,
                        levels=CONTOUR_LEVELS,
                        cmap="gist_heat",
                        linewidths=0.65,
                        alpha=0.9,
                        zorder=2
                    )

            # ── Region bounding box (always) ─────────────────────────────────
            rect_w = float(b[1, 0] - b[0, 0])
            rect_h = float(b[1, 1] - b[0, 1])
            is_active = (self.region is not None and region.version == self.region.version)
            edge_color = 'yellow' if is_active else 'gray'
            lw = 2.0 if is_active else 0.8

            rect = plt.Rectangle(
                (float(b[0, 0]), float(b[0, 1])), rect_w, rect_h,
                linewidth=lw, edgecolor=edge_color, facecolor='none', zorder=4
            )
            ax.add_patch(rect)
            ax.text(
                float(b[0, 0]) + 0.02, float(b[1, 1]) - 0.15,
                f"R{region.version}{'*' if is_active else ''}",
                color=edge_color, fontsize=7, zorder=5
            )

        # ── ExplorationGraph overlay: edges + door waypoints ─────────────────
        for v1, v2, edata in self.exploration_graph.graph.edges(data=True):
            rec1 = self.exploration_graph.records.get(v1)
            rec2 = self.exploration_graph.records.get(v2)
            if rec1 is None or rec2 is None:
                continue

            c1 = rec1.center
            c2 = rec2.center

            door_xys = edata.get('door_xys', [edata.get('door_xy')])
            door_xys = [d for d in door_xys if d is not None]

            if door_xys:
                for door_xy in door_xys:
                    xs = [c1[0], door_xy[0], c2[0]]
                    ys = [c1[1], door_xy[1], c2[1]]
                    ax.plot(xs, ys, color='cyan', linewidth=1.2,
                            linestyle='--', alpha=0.7, zorder=6)
                    ax.scatter(door_xy[0], door_xy[1],
                               c='cyan', s=40, marker='D',
                               zorder=7, alpha=0.9)
            else:
                ax.plot([c1[0], c2[0]], [c1[1], c2[1]],
                        color='cyan', linewidth=1.0,
                        linestyle='--', alpha=0.6, zorder=6)

        # ── ExplorationGraph nodes (region centers) ──────────────────────────
        for ver, rec in self.exploration_graph.records.items():
            c = rec.center
            has_f = rec.has_frontiers()
            node_color = 'lime' if has_f else 'red'
            ax.scatter(c[0], c[1], c=node_color, s=120,
                       marker='o', zorder=8, edgecolors='white', linewidths=1.0)
            ax.text(c[0] + 0.05, c[1] + 0.05,
                    f"R{ver}\n({len(rec.saved_frontiers)}f)",
                    color='white', fontsize=6, zorder=9,
                    bbox=dict(boxstyle='round,pad=0.15', fc='black', alpha=0.5))

        # ── Latest saved frontier points per region (NOT all) ─────────────────
        for ver, rec in self.exploration_graph.records.items():
            if not rec.saved_frontiers:
                continue
            pts = rec.saved_frontiers[-FRONTIER_LAST_K:]  # latest K (append order)
            for fp in pts:
                ax.scatter(fp[0], fp[1], c='orange', s=14,
                           marker='.', zorder=7, alpha=0.85)

        # ── Robot, trajectory, goal ─────────────────────────────────────────
        if self.robot_pos is not None:
            ax.scatter(self.robot_pos[0], self.robot_pos[1],
                       c='red', s=100, zorder=11, marker='o', label='Robot')

        if len(self._traj_history) > 1:
            traj = np.array(self._traj_history)
            ax.plot(traj[:, 0], traj[:, 1], 'm-', linewidth=1.5, zorder=9)

        if self.goal_xy is not None:
            ax.scatter(self.goal_xy[0], self.goal_xy[1],
                       c='lime', s=180, zorder=11, marker='*', label='Goal')

        # ── Active inter-region door route ───────────────────────────────────
        if getattr(self, "_remote_door_waypoints", None):
            if len(self._remote_door_waypoints) > 0:
                wps = np.array(self._remote_door_waypoints)
                ax.plot(wps[:, 0], wps[:, 1], color='magenta',
                        linewidth=2.0, linestyle='-', zorder=10, label='Door route')
                ax.scatter(wps[:, 0], wps[:, 1], c='magenta',
                           s=80, marker='D', zorder=11)

        # ── Legend ──────────────────────────────────────────────────────────
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='lime',
                   markersize=8, label='Region (has frontiers)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                   markersize=8, label='Region (exhausted)'),
            Line2D([0], [0], marker='D', color='w', markerfacecolor='cyan',
                   markersize=7, label='Door waypoint'),
            Line2D([0], [0], color='cyan', linestyle='--', label='Graph edge'),
            Line2D([0], [0], marker='.', color='w', markerfacecolor='orange',
                   markersize=6, label=f'Latest {FRONTIER_LAST_K} frontiers'),
            Line2D([0], [0], marker='*', color='w', markerfacecolor='lime',
                   markersize=10, label='Current goal'),
            Line2D([0], [0], color='magenta', label='Active door route'),
        ]
        ax.legend(handles=legend_elements, loc='upper right',
                  fontsize=7, framealpha=0.7)

        ax.set_xlim(b_map[0][0], b_map[1][0])
        ax.set_ylim(b_map[0][1], b_map[1][1])
        ax.set_aspect('equal')
        ax.set_title(
            f"Step {self._composite_step}  |  regions={len(self.all_regions)}"
            f"  doors={len(self.gridmap.doors)}"
            f"  mode={self.mode}/{self.nav_state}"
        )

        fig.savefig(os.path.join(out_dir, f"plot_step_{self._composite_step}.png"),
                    dpi=120, bbox_inches='tight')
        fig.savefig(os.path.join(self.folder, "plot2.png"), bbox_inches='tight')
        plt.close(fig)

    def save_occupancy_snapshot(self, tag=None):
        """
        Saves occupancy grid + visited + frontier as image and numpy.
        """
        out_dir = os.path.join(self.folder, "occupancy_snapshots")
        os.makedirs(out_dir, exist_ok=True)

        if tag is None:
            tag = f"step_{self._composite_step:06d}"

        data = self.gridmap.data.copy()
        visited = self.gridmap.visited.copy()
        frontier = self.gridmap.frontier.copy()

        n = self.gridmap.n_cells

        # ---- Create visualization image ----
        img = np.zeros((n, n, 3), dtype=np.uint8)

        # unknown
        img[:, :] = [10, 10, 80]

        # visited free
        img[visited == 1] = [200, 200, 0]   # yellow (clearer than dark green)

        # occupied
        img[data == 1] = [0, 0, 0]

        # frontier overlay (optional highlight)
        img[frontier == 1] = [255, 140, 0]  # orange

        plt.figure(figsize=(8, 8))
        plt.imshow(img, origin="lower")
        plt.title(f"Occupancy Snapshot {tag}")
        plt.axis("off")
        plt.savefig(os.path.join(out_dir, f"{tag}.png"), dpi=150, bbox_inches="tight")
        plt.close()

        # ---- Save raw arrays ----
        np.savez_compressed(
            os.path.join(out_dir, f"{tag}.npz"),
            occupancy=data,
            visited=visited,
            frontier=frontier
        )

        self.get_logger().info(f"[OCC SNAPSHOT SAVED] {tag}")

def main():
    rclpy.init()
    node = GridmapNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
