import numpy as np
import math
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import LaserScan
import sensor_msgs_py.point_cloud2 as pc2

class Lidar2DAdapter:
    def __init__(self,z_slice=0.15,max_range=None,speed_decay=1.5,min_points=30,lidar_height=0.30):
        self.z_slice = z_slice
        self.max_range = max_range  # None = use sensor's own range_max
        self.speed_decay = speed_decay
        self.min_points = min_points
        self.lidar_height = lidar_height

    def pointcloud_to_xy(self, cloud_msg: PointCloud2):
        pts = []
        kept = 0
        for p in pc2.read_points(
            cloud_msg,
            field_names=("x", "y", "z"),
            skip_nans=True
        ):
            x, y, z = p
            # Convert sensor z → world height
            z_world = z + self.lidar_height
            if z_world < 0.05:
                continue
            if z_world > 0.6:
                continue
            dist = math.sqrt(x*x + y*y)
            if self.max_range is not None and dist > self.max_range:
                continue
            if dist < 0.05:
                continue
            pts.append([x, y, z]) #remove z later
            kept += 1
        if kept < self.min_points:
            print("[LIDAR DEBUG] Too few valid wall points → skipping frame")
            return None

        pts = np.array(pts, dtype=np.float32)
        dists = np.linalg.norm(pts, axis=1)
        #print(f"[LIDAR DEBUG] Distance range: {dists.min():.2f} → {dists.max():.2f}")
        return pts

    def scan_to_xy(self, scan_msg: LaserScan):
        pts = []
        if scan_msg.angle_increment == 0.0:
            return None

        angle = scan_msg.angle_min
        rmin = max(0.05, float(scan_msg.range_min))
        rmax = float(scan_msg.range_max) if self.max_range is None else min(self.max_range, float(scan_msg.range_max))

        for rng in scan_msg.ranges:
            if (not math.isfinite(rng)) or rng < rmin or rng > rmax:
                angle += scan_msg.angle_increment
                continue
            x = float(rng) * math.cos(angle)
            y = float(rng) * math.sin(angle)
            # Keep z=0 in scan frame. World z comes from TF translation.
            pts.append([x, y, 0.0])
            angle += scan_msg.angle_increment

        if len(pts) < self.min_points:
            print("[LIDAR DEBUG] Too few valid /scan points → skipping frame")
            return None

        return np.asarray(pts, dtype=np.float32)

    def compute_speed(self, points: np.ndarray):
        dists = np.linalg.norm(points[:, :2], axis=1)
        speeds = np.exp(-self.speed_decay * dists)
        return speeds

    def update(self, lidar_msg, robot_pose):
        if isinstance(lidar_msg, LaserScan):
            pts = self.scan_to_xy(lidar_msg)
        elif isinstance(lidar_msg, PointCloud2):
            pts = self.pointcloud_to_xy(lidar_msg)
        else:
            return None, None
        if pts is None:
            return None, None

        speeds = self.compute_speed(pts)
        return pts, speeds

