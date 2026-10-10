import os
import json
import threading
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
matplotlib.use("Agg", force=True)
from matplotlib import cm, colors

import matplotlib.pyplot as plt
import gc
import torch
import numpy as np
import networkx as nx
import scipy
from scipy.spatial import cKDTree

from typing import Set, Iterable, Optional
from torch import Tensor
import torch_kdtree

# local modules
from mntfields.mntfields_core.mapping.rooms_doors import Door, ConfirmedRoom, UnconfirmedRoom
from mntfields.mntfields_core.ntrl import model_network_metric as model_network
from mntfields.mntfields_core.ntrl import model_function_metric as model_function
from mntfields.mntfields_core.core.model_base import Model

# utils (your merged utils file)
from mntfields.utils.mntfields_utils import (
    percent_overlap,
    np_to_pcd,
    combine_pcd,
    Timer
)
import inspect
import open3d as o3d
from mntfields.mntfields_core.core.model_base import Model as BaseModel
print("Model.__init__ signature:", inspect.signature(BaseModel.__init__))


# Fixed contour levels for TT field plots — 100 levels spanning 0–200 TT units.
# Matplotlib only draws contours where data exists, so this covers any TT range
# without clipping while maintaining even visual spacing (original mNTFields style).
TT_CONTOUR_LEVELS = np.linspace(0, 200, 100)

class Region(Model):
    n_regions = 0

    def __init__(
        self,
        model,
        bound,
        version=-1,
        init_network=True,
        eval=False,
        floor=None,
        ceiling=None,
    ):
        """
        Initialize a Region for ROS2/Gazebo deployment.
        
        Args:
            model: Parent model containing global parameters
            bound: torch.Tensor [2, 3] - [min, max] bounds in world coordinates
            version: int - region ID
            init_network: bool - whether to initialize neural network
            eval: bool - evaluation mode (no training)
            floor: float - floor height (optional)
            ceiling: float - ceiling height (optional)
        """
        # Adjust Z bounds
        bound[0, 2] = -0.1
        bound[1, 2] = 0.1
        
        # Version tracking
        self.alpha = 1
        if version == -1:
            Region.n_regions += 1
            self.version = Region.n_regions
        else:
            self.version = version
        
        self.bound_changed = False
        self.model = model
        
        # Create region-specific folder
        model_path = self.model.folder
        model_path = os.path.join(model_path, f"region_{self.version}")
        
        if not os.path.exists(model_path) and not eval:
            os.makedirs(model_path)
        
        # ✅ Call parent __init__ ONCE with correct signature (NO renderer)
        # Use model-configured device when provided; fallback keeps legacy behavior.
        device = str(getattr(self.model, "renderer", "cuda:0"))
        super().__init__(
            folder=model_path,
            dim=3,
            B_scale=10,
            device=device,
            dim_cells=128,
            init_network=init_network,
            eval=eval,
            lr=1e-4
        )
        
        # Region-specific parameters
        self.bound = bound
        # Near-wall band parameters (real metric, matches original's intent):
        #   minimum: don't sample closer than this to a wall (avoids inside-wall points)
        #   maximum: only x0 within this distance of a wall are used as training anchors
        # Original used 0.01/0.1 in iGibson x10 scale = 0.1/1.0m real.
        # For a 5-8m room, 0.5m gives a well-defined near-wall gradient band.
        self.minimum = 0.05
        self.maximum = 0.35  # tighter band → smaller avg τ → higher exp(-0.5T) weight → stronger wall signal
        
        # Door and room tracking
        self.doors_id: Set[int] = set()
        self.doors: Set[Door] = set()
        
        # LIDAR and mapping
        self.lidar_obs = None
        self.all_lidar = []
        self.kd_tree = None       # legacy CPU tree (kept for compat)
        self.gpu_kd_tree = None   # torch_kdtree GPU tree (rebuilt each scan)
        self.inference_lock = __import__("threading").Lock()  # guards forward passes against concurrent train
        
        # State tracking
        self.saved = False
        self.loaded = True
        self.all_raw_frame_data = []
        self.rooms: Set[ConfirmedRoom] = set()
        self.room_graph = nx.Graph()
        self.fully_explored = False
        self.floor = floor
        self.ceiling = ceiling
        
        # Training state
        self.has_init = False
        self.epoch = 0
        self.region_id = 0
        self.last_robot_pose = None
        self.warmup_complete_epoch: Optional[int] = None
        # Upper bound on retained raw training rows. GridmapNode can override
        # per region via `region_max_raw_samples`.
        self.max_raw_samples = 25_000
        # World-grid bin size for spatially balanced raw-history retention.
        self.raw_memory_cell_size_m = 0.35

        # Semantic object storage: class_name → list of [x, y] world-coordinate observations.
        # Each entry is a distinct sighting; callers should deduplicate before querying.
        self.semantic_objects: dict = {}

    def add_semantic_object(self, class_name: str, world_xy, dedup_radius: float = 0.5):
        """Record a furniture detection in world coordinates.

        Skips the observation if an existing sighting of the same class is already
        within ``dedup_radius`` metres (avoids flooding the list from a stationary robot).

        Args:
            class_name:   furniture label, e.g. "chair", "sofa"
            world_xy:     array-like [x, y] in the global (odom) frame
            dedup_radius: minimum distance (m) from any existing sighting to add a new one
        """
        import numpy as _np
        xy = _np.asarray(world_xy[:2], dtype=np.float32)
        existing = self.semantic_objects.get(class_name, [])
        for prev in existing:
            if _np.linalg.norm(xy - _np.asarray(prev, dtype=np.float32)) < dedup_radius:
                return  # already have a nearby sighting
        existing.append(xy.tolist())
        self.semantic_objects[class_name] = existing

    def destroy(self):
        if self.loaded:
            self.delete_network()
        if hasattr(self, "vis"):
            self.vis.destroy_window()

    def delete_network(self):
        # self.network.cpu()
        if self.B is not None:
            self.B.cpu()
        for state in self.prev_state_queue:
            del state
        self.prev_state_queue = []
        for optimizer in self.prev_optimizer_queue:
            del optimizer
        self.prev_optimizer_queue = []
        # Set to None instead of del — callers guard with "is None",
        # so deleting the attribute causes AttributeError crashes.
        self.B        = None
        self.network  = None
        self.function = None
        self.optimizer = None
        gc.collect()
        torch.cuda.empty_cache()

    def add_room(self, room: ConfirmedRoom, room_graph: nx.Graph=None):
        self.rooms.add(room)
        room.set_region(self)
        if room_graph is not None:
            self.room_graph = room_graph.subgraph(self.rooms)

    def is_connected(self, room: ConfirmedRoom, 
                     room_graph: nx.Graph, 
                     confirmed_rooms: Iterable[ConfirmedRoom],
                     unconfirmed_rooms: Iterable[UnconfirmedRoom]):
        if len(self.rooms) == 0:
            return True
        def u(sets):
            x = set()
            for s in sets:
                x.union(s)
            return x

        owned_doors = u([[d.id for d in room.doors] for room in self.rooms])
        unconfirmed_doors = u([[d.id for d in room.doors] for room in unconfirmed_rooms])
        unconfirmed_doors = self.doors_id.intersection(unconfirmed_doors)
        new_doors = [d.id for d in room.doors]
        sg = room_graph.subgraph(unconfirmed_doors.union(owned_doors).union(new_doors))
        if len(sg.nodes) == 0:
            return True
        return nx.is_connected(sg)

    def _region_paths(self):
        self.network_save_path = os.path.join(self.folder, f'region_{self.version}_network.pt')
        self.B_save_path = os.path.join(self.folder, f'region_{self.version}_B.pt')
        self.data_save_path = os.path.join(self.folder, f'region_{self.version}_data.pth')
        self.meta_save_path = os.path.join(self.folder, f'region_{self.version}_meta.json')
        return self.network_save_path, self.B_save_path, self.data_save_path, self.meta_save_path

    def _write_region_bundle(self):
        self._region_paths()
        torch.save(self.network.state_dict(), self.network_save_path)
        torch.save(self.B, self.B_save_path)
        torch.save({
            'frame_data': self.all_raw_frame_data,
            'all_lidar': self.all_lidar,
            'semantic_objects': self.semantic_objects,
            'epoch': self.epoch,
        }, self.data_save_path)
        meta = {
            'version': self.version,
            'bound': [[float(x) for x in row] for row in self.bound.detach().cpu().tolist()],
            'epoch': int(self.epoch),
            'doors': list(self.doors_id),
            'rooms': [room.id for room in self.rooms],
        }
        with open(self.meta_save_path, 'w') as f:
            json.dump(meta, f)

    def unload_to_disk(self):
        self._write_region_bundle()
        self.delete_network()
        self.network = None
        self.B = None
        self.function = None
        self.optimizer = None
        self.scheduler = None
        self.saved = True
        self.loaded = False
        self.all_raw_frame_data = []
        self.all_lidar = []

    def load_from_disk(self):
        assert self.saved
        self._region_paths()
        self.B = torch.load(self.B_save_path, map_location=self.Params.get('Device', 'cpu'), weights_only=False)
        self.network = model_network.NN(self.Params.get('Device', 'cpu'), self.dim, self.B).to(self.Params.get('Device', 'cpu'))
        self.network.load_state_dict(torch.load(self.network_save_path, map_location=self.Params.get('Device', 'cpu'), weights_only=False))

        try:
            loaded = torch.load(self.data_save_path, weights_only=False)
            if isinstance(loaded, dict):
                self.all_raw_frame_data = loaded.get('frame_data', [])
                self.all_lidar = loaded.get('all_lidar', [])
                self.semantic_objects = loaded.get('semantic_objects', {})
                self.epoch = int(loaded.get('epoch', self.epoch))
            elif isinstance(loaded, (tuple, list)) and len(loaded) >= 2:
                self.all_raw_frame_data = loaded[0]
                self.all_lidar = []
                self.semantic_objects = {}
            else:
                self.all_raw_frame_data = loaded
                self.all_lidar = []
                self.semantic_objects = {}
        except Exception as e:
            print(f"[load_from_disk] could not load data from {self.data_save_path}: {e}")
            self.all_raw_frame_data = []
            self.all_lidar = []
            self.semantic_objects = {}

        if os.path.exists(self.meta_save_path):
            try:
                with open(self.meta_save_path, 'r') as f:
                    meta = json.load(f)
                self.bound = torch.tensor(meta.get('bound', self.bound), dtype=torch.float32)
                self.epoch = int(meta.get('epoch', self.epoch))
            except Exception as e:
                print(f"[load_from_disk] could not load meta from {self.meta_save_path}: {e}")

        self.saved = False
        self.function = model_function.Function(self.folder, self.Params.get('Device', 'cpu'), self.network, self.dim)

        self.Params.setdefault('Training', {})
        lr = self.Params['Training'].get('Learning Rate', 1e-3)
        use_sched = self.Params['Training'].get('Use Scheduler (bool)', False)
        self.optimizer = torch.optim.AdamW(self.network.parameters(), lr=lr, weight_decay=0.1)
        self.scheduler = None
        if bool(use_sched):
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)

        self.loaded = True

    @property
    def center(self):
        # Always compute center on CPU to avoid CUDA→NumPy errors
        b = self.bound.detach().cpu().numpy()
        cx = float((b[0, 0] + b[1, 0]) * 0.5)
        cy = float((b[0, 1] + b[1, 1]) * 0.5)
        cz = float((b[0, 2] + b[1, 2]) * 0.5)
        return np.array([cx, cy, cz], dtype=np.float32)

    def get_data_from_lidar(self, lidar_scan, cur_pos, device='cuda:0', n_samples=4000, non_hit_dirs=None):
        """
        Sample training pairs using the original mntfields idea:
        sample in a local obstacle AABB, then reject pairs using orientation-aware
        signed obstacle distance from the accumulated LiDAR point cloud.
        """
        obs_pts_2d = lidar_scan[:, :2]
        if len(obs_pts_2d) < 20:
            print("[LIDAR KD] too few obstacle points in current scan")
            return None

        # ── 1. Accumulate scan as an oriented point cloud ─────────────────
        cur_lidar_pcd = np_to_pcd(lidar_scan, cur_pos, downsample=True)
        if self.lidar_obs is not None:
            self.lidar_obs = combine_pcd(cur_lidar_pcd, self.lidar_obs)
        else:
            self.lidar_obs = cur_lidar_pcd
        self.all_lidar.append((lidar_scan.copy(), cur_pos.copy()))

        all_obs = np.asarray(self.lidar_obs.points)
        if len(all_obs) < 20:
            print("[LIDAR KD] too few accumulated obstacle points")
            return None
        self.kd_tree = cKDTree(all_obs)   # CPU fallback (used by point_obs_dist)
        self.gpu_kd_tree = torch_kdtree.build_kd_tree(
            torch.from_numpy(all_obs).float(),
            device=torch.device(device),
            squared_distances=False,
        )

        # ── 2. Local AABB: obstacle cloud clipped to current region bound ──
        bb_min = all_obs.min(axis=0).astype(np.float32)
        bb_max = all_obs.max(axis=0).astype(np.float32)
        b = self.bound.detach().cpu().numpy().astype(np.float32)
        bb_min[:2] = np.maximum(bb_min[:2], b[0, :2])
        bb_max[:2] = np.minimum(bb_max[:2], b[1, :2])
        bb_min[2] = -0.05
        bb_max[2] = 0.05
        if np.any(bb_max <= bb_min):
            print("[LIDAR KD] invalid sampling box")
            return None

        # ── 3. Sample x0 uniformly in the bounding box (original mntfields method) ──
        # GPU KD-tree makes the rejection approach fast: oversample uniformly,
        # batch-query nearest wall distance, keep those in the near-wall band.
        # No side filter — outside-wall samples (~50%) have their normals handled
        # correctly by the oriented point cloud in the loss function.
        n_try = n_samples * 6  # oversample; ~20-40% acceptance in near-wall band

        x0_2d_cands = np.random.uniform(
            bb_min[:2], bb_max[:2], (n_try, 2)
        ).astype(np.float32)

        z0 = np.zeros((n_try, 1), dtype=np.float32)
        x0_3d_t = torch.from_numpy(np.hstack([x0_2d_cands, z0])).float().to(device)
        dist0_t, idx0_t = self.gpu_kd_tree.query(x0_3d_t, nr_nns_searches=1)
        dist0 = dist0_t[:, 0].cpu().numpy()
        idx0  = idx0_t[:, 0].cpu().numpy()

        near_wall_mask = (dist0 > float(self.minimum)) & (dist0 < float(self.maximum))
        if near_wall_mask.sum() == 0:
            print("[LIDAR KD] no x0 in near-wall band after uniform sampling")
            return None

        x0_2d   = x0_2d_cands[near_wall_mask].astype(np.float32)
        cp0_arr = all_obs[idx0[near_wall_mask]].astype(np.float32)

        if len(x0_2d) > n_samples:
            sel2 = np.random.choice(len(x0_2d), n_samples, replace=False)
            x0_2d   = x0_2d[sel2]
            cp0_arr = cp0_arr[sel2]

        n = len(x0_2d)
        x0_np = x0_2d
        z_n = np.zeros((n, 1), dtype=np.float32)
        x0 = torch.from_numpy(np.hstack([x0_np, z_n])).float().to(device)
        closest_obs0 = torch.from_numpy(cp0_arr).float().to(device)

        # ── 4. Sample x1 by local random displacement (original mntfields style) ──
        half_extent = max(float(bb_max[0] - bb_min[0]),
                          float(bb_max[1] - bb_min[1])) / 2.0
        half_extent = max(half_extent, 0.5)
        all_angles  = np.random.uniform(0.0, 2.0 * np.pi, n)
        all_lengths = (np.random.uniform(0.0, 1.0, n) + 0.03) * half_extent
        x1_2d = (x0_np +
                 np.stack([np.cos(all_angles), np.sin(all_angles)], axis=-1)
                 * all_lengths[:, None]).astype(np.float32)

        keep = ((x1_2d[:, 0] >= bb_min[0]) & (x1_2d[:, 0] <= bb_max[0]) &
                (x1_2d[:, 1] >= bb_min[1]) & (x1_2d[:, 1] <= bb_max[1]))
        if keep.sum() < 10:
            print("[LIDAR KD] insufficient x1 candidates after displacement sampling")
            return None

        keep_idx     = np.where(keep)[0]
        x0_np        = x0_np[keep_idx]
        x1_2d        = x1_2d[keep_idx]
        closest_obs0 = closest_obs0[keep_idx]

        # ── 5. Look up cp1 from the global KD-tree ───────────────────────────
        x1_3d_q = np.hstack([x1_2d.astype(np.float32),
                              np.zeros((len(x1_2d), 1), dtype=np.float32)])
        x1_3d_t  = torch.from_numpy(x1_3d_q).float().to(device)
        _, idx1_t = self.gpu_kd_tree.query(x1_3d_t, nr_nns_searches=1)
        idx1     = idx1_t[:, 0].cpu().numpy()
        cp1_3d   = all_obs[idx1].astype(np.float32)

        # ── 6. Assemble Nx12 world-coord tensor ──────────────────────────────
        z    = np.zeros((len(x0_np), 1), dtype=np.float32)
        x0_3d = np.hstack([x0_np.astype(np.float32), z])
        x1_3d = np.hstack([x1_2d.astype(np.float32), z])
        cp0_3d = closest_obs0.detach().cpu().numpy().astype(np.float32)

        points         = torch.from_numpy(np.hstack([x0_3d, x1_3d])).float().to(device)
        closest_points = torch.from_numpy(np.hstack([cp0_3d, cp1_3d])).float().to(device)
        return points, closest_points

    def calculate_speed_and_normal(self, data):
        """
        Convert Nx12 world-coord pairs (x0, x1, cp0, cp1) into Nx14 training rows.

        Matches the original mntfields logic:
          - x0 must be in the near-wall band: clearance in (minimum, maximum]
          - x1 can be anywhere: its clearance is clamped to [minimum, maximum]
          - Both normals are unit vectors pointing from nearest obstacle to sample
        """
        x0 = data[:, :3]
        x1 = data[:, 3:6]
        p0 = data[:, 6:9]
        p1 = data[:, 9:12]

        n0 = x0 - p0
        n1 = x1 - p1

        y0 = torch.norm(n0, dim=1, keepdim=True)   # clearance of x0 (m)
        y1 = torch.norm(n1, dim=1, keepdim=True)   # clearance of x1 (m)

        maximum = float(self.maximum)
        minimum = float(self.minimum)

        # Keep rows where x0 is in the near-wall band.
        # x1 is allowed anywhere — its clearance gets clamped below.
        valid = (y0[:, 0] > minimum) & (y0[:, 0] <= maximum)
        x0 = x0[valid];  x1 = x1[valid]
        y0 = y0[valid];  y1 = y1[valid]
        n0 = n0[valid];  n1 = n1[valid]

        if len(x0) == 0:
            return torch.zeros((0, 14), dtype=torch.float32, device=data.device)

        # Unit normals
        n0 = n0 / (y0 + 1e-8)
        n1 = n1 / (y1 + 1e-8)

        # Normalize positions to [-0.5, 0.5] using current bound
        x0_n = self.normalize_tensor(x0)
        x1_n = self.normalize_tensor(x1)

        # Clip clearance to [min, max] then normalize to [0, 1]
        y0 = torch.clamp(y0, min=minimum, max=maximum) / maximum
        y1 = torch.clamp(y1, min=minimum, max=maximum) / maximum

        return torch.cat([x0_n, x1_n, y0, y1, n0, n1], dim=1)

    def randomize_data(self, merged):
        """
        Re-sample training pairs from the accumulated near-wall pool using the
        original mNTFields cross-indexing strategy: x0 and x1 are independently
        drawn from the near-wall pool via separate randperm calls.

        This gives M² combinatorial pair diversity and trains the metric over
        the full spatial range of explored walls — not just short-range neighbours.
        Matches the behaviour of randomize_data() in the original helper.py.

        Input:  merged  — Nx12 world-coord tensor [x0(3), x1(3), cp0(3), cp1(3)]
        Output: Mx14 tensor [x0_norm(3), x1_norm(3), y0(1), y1(1), n0(3), n1(3)]
        """
        minimum = float(self.minimum)
        maximum = float(self.maximum)

        # 1. Extract x0 pool (near-wall only) — same as before
        x0_world  = merged[:, :3]
        cp0_world = merged[:, 6:9]
        n0_vec = x0_world - cp0_world
        y0_raw = torch.norm(n0_vec, dim=1)
        mask = (y0_raw > minimum) & (y0_raw <= maximum)
        if mask.sum() == 0:
            return None
        x0_pool  = x0_world[mask].float()
        cp0_pool = cp0_world[mask].float()
        if x0_pool.device.type != 'cuda':
            x0_pool  = x0_pool.to(self.device)
            cp0_pool = cp0_pool.to(self.device)

        M = len(x0_pool)

        # 2. ORIGINAL APPROACH: independently sample x0 and x1 from the pool.
        # Two separate randperms → any wall point can be paired with any other.
        idx0 = torch.randperm(M, device=self.device)
        idx1 = torch.randperm(M, device=self.device)

        x0_t  = x0_pool[idx0]
        cp0_t = cp0_pool[idx0]
        x1_t  = x0_pool[idx1]   # x1 is an independently sampled near-wall point
        cp1_t = cp0_pool[idx1]

        # 3. Compute speeds and normals for both endpoints
        n0 = x0_t - cp0_t
        n1 = x1_t - cp1_t
        y0 = torch.norm(n0, dim=1, keepdim=True)
        y1 = torch.norm(n1, dim=1, keepdim=True)
        n0 = n0 / (y0 + 1e-8)
        n1 = n1 / (y1 + 1e-8)
        y0 = torch.clamp(y0, min=minimum, max=maximum) / maximum
        y1 = torch.clamp(y1, min=minimum, max=maximum) / maximum

        # 4. Normalize positions to [-0.5, 0.5]
        x0_n = self.normalize_tensor(x0_t.cpu()).to(self.device)
        x1_n = self.normalize_tensor(x1_t.cpu()).to(self.device)

        result = torch.cat([x0_n, x1_n, y0, y1, n0, n1], dim=1)  # (M, 14)

        # 5. Interior x0 augmentation (keep: navigation queries x0 from free space).
        # With cross-indexing, the primary pairs are all wall-to-wall, so the network
        # still needs interior x0 supervision to avoid Euclidean extrapolation in
        # free space. Pair interior x0 positions with random near-wall x1 targets.
        try:
            x1_stored  = merged[:, 3:6].float()
            cp1_stored = merged[:, 9:12].float()
            if x1_stored.device.type != 'cuda':
                x1_stored  = x1_stored.to(self.device)
                cp1_stored = cp1_stored.to(self.device)
            y1_dist  = torch.norm(x1_stored - cp1_stored, dim=1)
            int_mask = y1_dist > maximum          # truly free-space stored x1
            if int_mask.sum() > 0:
                xi_pool  = x1_stored[int_mask]
                cpi_pool = cp1_stored[int_mask]
                # Keep interior augmentation, but cap it lower so wall-to-wall
                # supervision remains dominant in each batch.
                Mi = min(int(int_mask.sum()), 1000)
                if Mi < int(int_mask.sum()):
                    ri = torch.randperm(int(int_mask.sum()), device=self.device)[:Mi]
                    xi_pool  = xi_pool[ri]
                    cpi_pool = cpi_pool[ri]
                idx_tgt  = torch.randint(0, M, (Mi,), device=self.device)
                x1_tgt   = x0_pool[idx_tgt]
                cp1_tgt  = cp0_pool[idx_tgt]
                ni0 = xi_pool - cpi_pool
                yi0 = torch.norm(ni0, dim=1, keepdim=True)
                yi0_norm = torch.clamp(yi0, min=minimum, max=maximum) / maximum
                ni0 = ni0 / (yi0 + 1e-8)
                ni1 = x1_tgt - cp1_tgt
                yi1 = torch.norm(ni1, dim=1, keepdim=True)
                yi1_norm = torch.clamp(yi1, min=minimum, max=maximum) / maximum
                ni1 = ni1 / (yi1 + 1e-8)
                xi0_n = self.normalize_tensor(xi_pool.cpu()).to(self.device)
                xi1_n = self.normalize_tensor(x1_tgt.cpu()).to(self.device)
                interior_rows = torch.cat(
                    [xi0_n, xi1_n, yi0_norm, yi1_norm, ni0, ni1], dim=1)
                result = torch.cat([result, interior_rows], dim=0)
        except Exception:
            pass

        # 6. Cap total
        if len(result) > 7000:
            idx = torch.randperm(len(result), device=self.device)[:7000]
            result = result[idx]

        return result

    def get_next_waypoint(self, robot_xy, goal_xy, step_size=0.35, alpha=0.8, debug=True):
        """
        Return a waypoint one step ahead by following -∇TT (travel time) in the learned field.
        If anything fails, fall back to geometric steering toward goal.
    
        Key fix:
          - Differentiate TT w.r.t. `coords` returned by network.out(), not xp_n.
        """
    
        # --------------------------
        # geometric fallback
        # --------------------------
        dx = float(goal_xy[0] - robot_xy[0])
        dy = float(goal_xy[1] - robot_xy[1])
        dist = float(np.hypot(dx, dy))
    
        if dist < 1e-6:
            return np.array(goal_xy, dtype=np.float32)
    
        geom_dir = np.array([dx, dy], dtype=np.float32) / dist
    
        if self.function is None or getattr(self.function, "network", None) is None:
            #if debug:
            #    print("[GEOMETRIC] function/network not ready")
            step = min(step_size, dist)
            return (np.array(robot_xy, dtype=np.float32) + geom_dir * step).astype(np.float32)
    
        try:
            device = self.device
    
            # build x = [x0, x1]
            x0 = torch.tensor([[robot_xy[0], robot_xy[1], 0.0]], dtype=torch.float32, device=device)
            x1 = torch.tensor([[goal_xy[0],  goal_xy[1],  0.0]], dtype=torch.float32, device=device)
            xp = torch.cat([x0, x1], dim=1)  # (1, 6)
    
            # normalize to [-0.5, 0.5]
            xp_n = self.normalize_tensor(xp.clone())
            xp_n.requires_grad_(True)
    
            # IMPORTANT: use coords returned by out()
            tau, w, coords = self.function.network.out(xp_n)
            tt = tau[:, 0]  # (1,)
    
            #if debug:
            #    print(f"[FIELD DBG] tt.requires_grad={tt.requires_grad}, coords.requires_grad={coords.requires_grad}")
    
            # Gradient of TT w.r.t coords
            grad = torch.autograd.grad(
                outputs=tt.sum(),
                inputs=coords,
                retain_graph=False,
                create_graph=False
            )[0]
    
            if grad is None:
                direction = geom_dir
            else:
                # grad is ∂τ/∂[x0_norm, y0_norm] in normalized [-0.5,0.5] space.
                # Normalization: x_norm = (x_world - bound_start) / bound_len - 0.5
                # So: ∂τ/∂x_world = ∂τ/∂x_norm / bound_len  (chain rule)
                # De-normalize to get world-space gradient direction.
                grad_xy_norm = grad[0, :2].detach().cpu().numpy()
                b_np = self.bound.detach().cpu().numpy()
                bound_len = (b_np[1, :2] - b_np[0, :2]).astype(np.float32)
                grad_xy = grad_xy_norm / (bound_len + 1e-8)
                norm = float(np.linalg.norm(grad_xy))

                if norm < 1e-5:
                    direction = geom_dir
                else:
                    field_dir = (-grad_xy / norm).astype(np.float32)

                    # blend (field + geometric) for stability
                    direction = alpha * field_dir + (1.0 - alpha) * geom_dir
                    direction = direction / (np.linalg.norm(direction) + 1e-8)
    
            step = min(step_size, dist)
            waypoint = np.array(robot_xy, dtype=np.float32) + direction.astype(np.float32) * step
            return waypoint.astype(np.float32)
    
        except Exception as e:
            print("FIELD WAYPOINT ERROR:", e)
            step = min(step_size, dist)
            return (np.array(robot_xy, dtype=np.float32) + geom_dir * step).astype(np.float32)

    def generate_field_path(self, start_xy, goal_xy, step_size=0.15, max_steps=120):
        """
        Follow a bidirectional mNTFields rollout from start_xy to goal_xy.

        Both ends are marched each iteration:
          - start side: x0=start_cur, x1=goal_cur
          - goal side : x0=goal_cur,  x1=start_cur

        This mirrors original predict_trajectory2 behavior more closely and
        reduces drift when one-side gradients become weak.
        """
        cur_s = np.array(start_xy, dtype=np.float32)
        cur_g = np.array(goal_xy, dtype=np.float32)
        pts_start = []
        pts_goal = []
        field_steps = 0

        gridmap = getattr(self, 'gridmap', None)
        b = self.bound.detach().cpu().numpy()

        def _in_wall(xy):
            if gridmap is None:
                return False
            cell   = gridmap.cell_size
            origin = gridmap.origin
            half   = gridmap.map_size / 2.0
            ix = int((xy[0] - (origin[0] - half)) / cell)
            iy = int((xy[1] - (origin[1] - half)) / cell)
            if not (0 <= ix < gridmap.n_cells and 0 <= iy < gridmap.n_cells):
                return True   # out of bounds — treat as wall
            return gridmap.data[iy, ix] == 1

        def _clip_to_bound(xy):
            return np.array([
                np.clip(float(xy[0]), float(b[0, 0]), float(b[1, 0])),
                np.clip(float(xy[1]), float(b[0, 1]), float(b[1, 1])),
            ], dtype=np.float32)

        def _segment_hits_wall(a, c):
            if gridmap is None:
                return False
            d = float(np.linalg.norm(c - a))
            if d < 1e-6:
                return False
            n = max(2, int(np.ceil(d / max(1e-4, 0.5 * gridmap.cell_size))))
            ts = np.linspace(0.0, 1.0, n, dtype=np.float32)
            for t in ts:
                p = a + t * (c - a)
                if _in_wall(p):
                    return True
            return False

        # Wait up to 2s for the training thread to release the lock between
        # epochs.  Training yields every ~200-500ms, so this is sufficient.
        acquired = self.inference_lock.acquire(timeout=2.0)
        if not acquired:
            return None
        try:
            for _ in range(max_steps):
                pts_start.append(cur_s.copy())
                pts_goal.append(cur_g.copy())

                if np.linalg.norm(cur_s - cur_g) <= max(step_size, 0.03):
                    break

                nxt_s = self.get_next_waypoint(cur_s, cur_g, step_size=step_size)
                nxt_g = self.get_next_waypoint(cur_g, cur_s, step_size=step_size)
                if nxt_s is None and nxt_g is None:
                    break

                if nxt_s is not None:
                    nxt_s = _clip_to_bound(np.asarray(nxt_s, dtype=np.float32))
                    if _in_wall(nxt_s) or _segment_hits_wall(cur_s, nxt_s):
                        nxt_s = cur_s.copy()
                    else:
                        field_steps += 1
                else:
                    nxt_s = cur_s.copy()

                if nxt_g is not None:
                    nxt_g = _clip_to_bound(np.asarray(nxt_g, dtype=np.float32))
                    if _in_wall(nxt_g) or _segment_hits_wall(cur_g, nxt_g):
                        nxt_g = cur_g.copy()
                    else:
                        field_steps += 1
                else:
                    nxt_g = cur_g.copy()

                if (np.linalg.norm(nxt_s - cur_s) < 1e-5 and
                        np.linalg.norm(nxt_g - cur_g) < 1e-5):
                    break

                cur_s = nxt_s
                cur_g = nxt_g
        finally:
            self.inference_lock.release()

        pts_start.append(cur_s.copy())
        pts_goal.append(cur_g.copy())

        total_iters = len(pts_start) + len(pts_goal) - 2  # -2 for the final appends
        if total_iters > 0:
            pct = int(100 * field_steps // max(1, total_iters))
            print(f"[PATH] field={field_steps}/{total_iters} ({pct}%)")

        # Stitch start-half with reversed goal-half; interpolate middle gap if needed.
        # But first check for the degenerate case: both rollout sides got stuck
        # immediately (near a wall), yielding a path shorter than one step_size.
        # These 2-3 waypoint paths are accepted (total≤4 bypasses the 40% check),
        # then PATH EXTEND fires every 0.15m → infinite replan loop.
        # Return None so A* handles it properly instead.
        right = list(reversed(pts_goal))
        if not pts_start:
            path = right
        elif not right:
            path = pts_start
        else:
            gap = float(np.linalg.norm(pts_start[-1] - right[0]))
            if gap <= max(step_size, 0.03):
                path = pts_start + right
            else:
                # Goal-side froze in untrained territory. If the start side
                # actually moved (not just clipped in place), return it as a
                # partial path so the caller can redirect goal_xy to the trained
                # boundary. The robot navigates there, dwells, trains, then
                # replans toward the original frontier.
                start_travel = (float(np.linalg.norm(
                    pts_start[-1] - pts_start[0])) if len(pts_start) > 1 else 0.0)
                if start_travel >= step_size * 3:
                    return [np.asarray(p, dtype=np.float32) for p in pts_start]
                return None

        path_arr = [np.asarray(p, dtype=np.float32) for p in path]

        # Reject paths that barely moved from start — both sides were blocked by
        # a wall the field doesn't know about.  Caller falls back to A*.
        goal_dist = float(np.linalg.norm(np.array(start_xy, dtype=np.float32) -
                                         np.array(goal_xy,  dtype=np.float32)))
        if len(path_arr) >= 2:
            total_len = sum(float(np.linalg.norm(path_arr[i+1] - path_arr[i]))
                            for i in range(len(path_arr) - 1))
        else:
            total_len = 0.0
        if total_len < max(step_size * 2, 0.3) and goal_dist > 0.5:
            return None  # too short for a non-trivial goal — let A* route it

        return path_arr

    def generate_field_path_oneway(self, start_xy, goal_xy, step_size=0.15, max_steps=60):
        """
        Single-sided rollout used in mapping:
        march only from start toward fixed goal using get_next_waypoint().
        """
        pts = []
        cur = np.array(start_xy, dtype=np.float32)
        goal = np.array(goal_xy, dtype=np.float32)
        field_steps = 0

        gridmap = getattr(self, 'gridmap', None)

        def _in_wall(xy):
            if gridmap is None:
                return False
            cell = gridmap.cell_size
            origin = gridmap.origin
            half = gridmap.map_size / 2.0
            ix = int((xy[0] - (origin[0] - half)) / cell)
            iy = int((xy[1] - (origin[1] - half)) / cell)
            if not (0 <= ix < gridmap.n_cells and 0 <= iy < gridmap.n_cells):
                return True
            return gridmap.data[iy, ix] == 1

        def _segment_hits_wall(a, c):
            if gridmap is None:
                return False
            d = float(np.linalg.norm(c - a))
            if d < 1e-6:
                return False
            n = max(2, int(np.ceil(d / max(1e-4, 0.5 * gridmap.cell_size))))
            ts = np.linspace(0.0, 1.0, n, dtype=np.float32)
            for t in ts:
                p = a + t * (c - a)
                if _in_wall(p):
                    return True
            return False

        acquired = self.inference_lock.acquire(timeout=2.0)
        if not acquired:
            return None
        try:
            for _ in range(max_steps):
                pts.append(cur.copy())
                if np.linalg.norm(cur - goal) < step_size:
                    break
                nxt = self.get_next_waypoint(cur, goal, step_size=step_size)
                if nxt is None:
                    break

                if _in_wall(nxt) or _segment_hits_wall(cur, nxt):
                    break
                field_steps += 1
                cur = nxt
        finally:
            self.inference_lock.release()

        if np.linalg.norm(cur - goal) <= max(step_size, 0.03):
            pts.append(goal.copy())

        if field_steps > 0:
            print(f"[PATH] field={field_steps}/{field_steps} (100%)")
        if len(pts) < 2:
            return None
        return [np.asarray(p, dtype=np.float32) for p in pts]
           

    def train_core(self, epoch, raw_frame_data=None, frame_data=None, is_one_frame=True):
        """
        Train on accumulated WORLD-COORDINATE data (raw_frame_data, Nx12).

        Storing world coords means the full history is re-normalized against
        the CURRENT bound every training step.  When fit_region_to_lidar
        expands the bound, old frames are correctly remapped — no stale
        normalized coords from the old (smaller) bound polluting training.
        """
        self.has_init = True
        self.alpha = 1
        _lock_acquired = False

        if raw_frame_data is None and (frame_data is None or len(frame_data) == 0):
            return False

        if raw_frame_data is not None:
            if raw_frame_data.shape[1] != 12:
                print(f"[TRAIN] raw_frame_data wrong shape: {raw_frame_data.shape}, expected Nx12")
                return False
            if self.all_raw_frame_data is None:
                self.all_raw_frame_data = []

            raw_frame_data = self._filter_raw_rows_to_bound(raw_frame_data.detach().cpu())
            if len(raw_frame_data) == 0:
                print("[TRAIN] all new raw rows fell outside current bound")
                return False

            # Reservoir-style random subsampling: keeps spatial coverage across full
            # exploration history. NOT strict FIFO — intentional, since strict FIFO
            # would age out early wall data and leave gaps in the trained field.
            # Matches the original helper.py train_core exactly:
            #   all_data = cat(all_raw_framedata); subsample to max_raw; store as [all_data]
            self.all_raw_frame_data.append(raw_frame_data)
            merged = torch.cat(self.all_raw_frame_data, dim=0)
            merged = self._filter_raw_rows_to_bound(merged)
            if len(merged) == 0:
                print("[TRAIN] merged raw history is empty after bound filtering")
                return False
            max_raw = max(1, int(getattr(self, "max_raw_samples", 10_000)))
            if len(merged) > max_raw:
                merged = self._spatially_cap_raw_history(merged, max_raw)
            self.all_raw_frame_data = [merged]   # flatten back to single-element list

            # merged is Nx12 world-coord pairs; randomize_data reads x0/cp0 only
            # and re-samples near-wall x1 every epoch for full gradient signal.
            # Validate that merged has some near-wall rows; if not, bail now.
            _probe = self.randomize_data(merged)
            if _probe is None or len(_probe) == 0:
                print("[TRAIN] randomize_data returned empty data (no near-wall rows)")
                return False
        else:
            if frame_data.shape[1] != 12:
                print(f"[TRAIN] frame_data wrong shape: {frame_data.shape}, expected Nx12")
                return False
            merged = self._filter_raw_rows_to_bound(frame_data.detach().cpu())
            if len(merged) == 0:
                print("[TRAIN] frame_data is empty after bound filtering")
                return False
            _probe = self.randomize_data(merged)
            if _probe is None or len(_probe) == 0:
                print("[TRAIN] randomize_data returned empty data (no near-wall rows)")
                return False

        # Acquire inference lock before the backward pass.
        _lock_acquired = self.inference_lock.acquire(blocking=False)
        if not _lock_acquired:
            return False  # inference running; skip tick (normal, not an error)
        # Use the original-style randomized wall-structured training set.
        wall_data = _probe.to(self.device) if (_probe is not None and len(_probe) > 0) else None

        # LR decay: halve every 150 epochs to prevent catastrophic forgetting in
        # large rooms where the data distribution keeps shifting as the robot explores.
        # 0-149: 5e-4 → 150-299: 2.5e-4 → 300-449: 1.25e-4 → 450+: floored at 5e-5.
        decay_steps = self.epoch // 150
        current_lr = 5e-4 * (0.5 ** decay_steps)
        current_lr = max(5e-5, current_lr)
        self.Params['Training']['Learning Rate'] = current_lr

        # ── Pass Nx14 to base class (x0_n, x1_n, y0, y1, n0, n1) ──
        # Train exclusively on fs_data (calculate_speed_and_normal) — correct x1 displacements
        # covering free space. randomize_data snaps x1→x0 (near-wall) causing degenerate pairs.
        try:
            train_epochs = max(1, int(epoch))
            loss = None
            completed_epochs = 0
            for _ep in range(train_epochs):
                if wall_data is None or len(wall_data) == 0:
                    continue
                all_data = wall_data
                if len(all_data) > 7000:
                    idx = torch.randperm(len(all_data), device=self.device)[:7000]
                    all_data = all_data[idx]
                loss_new, _ = super().train_core(
                    epoch=1, frame_data=all_data, is_one_frame=is_one_frame)
                completed_epochs += 1
                if loss_new is not None:
                    loss = loss_new
                    loss_val = loss_new.item() if isinstance(loss_new, torch.Tensor) else float(loss_new)
                    #print(f"[TRAIN] epoch {self.epoch + completed_epochs:03d} loss={loss_val:.3e}")
                # Yield lock between epochs so path gen can acquire it.
                # The small sleep lets the OS scheduler hand the lock to any
                # waiting path-gen thread before training re-acquires it.
                self.inference_lock.release()
                __import__("time").sleep(0.005)
                self.inference_lock.acquire(blocking=True)
            # Retry on catastrophic None (unloaded mid-training etc.)
            max_retries = 3
            retry_count = 0
            while loss is None and retry_count < max_retries:
                if self.optimizer is None or self.B is None:
                    break
                self.delete_network()
                self.init_network()
                if wall_data is None or len(wall_data) == 0:
                    break
                retry_data = wall_data
                if len(retry_data) > 7000:
                    idx = torch.randperm(len(retry_data), device=self.device)[:7000]
                    retry_data = retry_data[idx]
                loss, _ = super().train_core(
                    epoch=1, frame_data=retry_data, is_one_frame=is_one_frame)
                completed_epochs += 1
                if loss is not None:
                    loss_val = loss.item() if isinstance(loss, torch.Tensor) else float(loss)
                    print(f"[TRAIN] epoch {self.epoch + completed_epochs:03d} loss={loss_val:.3e}")
                retry_count += 1
        finally:
            # Always release — even if delete_network() or init_network() throws.
            self.inference_lock.release()
            _lock_acquired = False

        if loss is None:
            print("⚠️ Training failed after retries")
            return False

        self.epoch += completed_epochs
        loss_val = loss.item() if isinstance(loss, torch.Tensor) else loss
        success = loss_val < 0.005

        import time as _time
        _now_vis = _time.monotonic()
        _vis_interval = 20.0  # seconds between debug plots
        _vis_due = (_now_vis - getattr(self, '_last_vis_wall_t', 0.0)) >= _vis_interval
        if getattr(self, 'enable_debug_vis', False) and _vis_due:
            # Only launch a new vis thread if the previous one finished.
            if not getattr(self, '_vis_running', False):
                self._last_vis_wall_t = _now_vis
                self._vis_running = True
                # Robot-centered: use last_robot_pose as the view center.
                view_src = self.last_robot_pose.copy() if (hasattr(self, 'last_robot_pose') and self.last_robot_pose is not None) else None
                merged_cpu = merged.detach().cpu()  # snapshot before thread starts

                def _run_vis(vs, mc):
                    try:
                        # Robot-centric field debug plot (view center follows robot).
                        if vs is not None:
                            self.save_debug_plot(vs, view_radius=3.5, robot_xy=vs[:2])
                        vis_data = self.randomize_data(mc)
                        if vis_data is not None and len(vis_data) > 0:
                            rxy = vs[:2] if vs is not None else None
                            self.save_training_vis(vis_data[:, :6], vis_data[:, 6:8],
                                                   robot_xy=rxy, view_radius=2.0)
                    except Exception as e:
                        print(f"[VIS] skipped: {e}")
                    finally:
                        self._vis_running = False

                threading.Thread(
                    target=_run_vis,
                    args=(view_src, merged_cpu),
                    daemon=True
                ).start()
        wce = self.warmup_complete_epoch
        if wce is not None:
            # Save every 10 epochs after warmup
            elapsed = self.epoch - wce
            if elapsed > 0 and elapsed % 10 == 0:
                self.save_model()
        elif self.epoch % 50 == 0:
            # Save periodically during warmup (coarser)
            self.save_model()
        return success

    def accumulate_raw_data(self, raw_frame_data: torch.Tensor) -> bool:
        """
        Append raw Nx12 world-coord pairs into this region's data buffer WITHOUT
        running a backward pass.  Used by neighbouring regions to receive wall data
        scanned while the robot was outside this region.  The data will be used
        the next time this region becomes active and calls train_core().
        """
        if raw_frame_data is None or raw_frame_data.shape[0] == 0:
            return False
        if raw_frame_data.shape[1] != 12:
            return False
        if self.all_raw_frame_data is None:
            self.all_raw_frame_data = []
        rows = self._filter_raw_rows_to_bound(raw_frame_data.detach().cpu())
        if len(rows) == 0:
            return False
        self.all_raw_frame_data.append(rows)
        merged = torch.cat(self.all_raw_frame_data, dim=0)
        merged = self._filter_raw_rows_to_bound(merged)
        max_raw = max(1, int(getattr(self, "max_raw_samples", 10_000)))
        if len(merged) > max_raw:
            merged = self._spatially_cap_raw_history(merged, max_raw)
        self.all_raw_frame_data = [merged]
        return True

    def _spatially_cap_raw_history(self, merged: torch.Tensor, max_raw: int) -> torch.Tensor:
        """
        Keep at most `max_raw` rows while preserving spatial coverage.
        Bins are computed from x0 world position in fixed-size world cells.
        """
        if merged is None or len(merged) <= max_raw:
            return merged

        arr = merged.detach().cpu().numpy()
        x0 = arr[:, :2]
        cell = max(0.05, float(getattr(self, "raw_memory_cell_size_m", 0.35)))
        bins = np.floor(x0 / cell).astype(np.int32)

        keys = bins[:, 0].astype(np.int64) * np.int64(1_000_003) + bins[:, 1].astype(np.int64)
        uniq, inv = np.unique(keys, return_inverse=True)
        n_bins = int(len(uniq))
        if n_bins <= 0:
            idx = np.random.choice(len(arr), max_raw, replace=False)
            return merged[idx]

        base_quota = max(1, max_raw // n_bins)
        chosen = []
        chosen_mask = np.zeros(len(arr), dtype=bool)

        for bi in range(n_bins):
            inds = np.where(inv == bi)[0]
            if len(inds) <= base_quota:
                pick = inds
            else:
                pick = np.random.choice(inds, base_quota, replace=False)
            chosen.extend(pick.tolist())
            chosen_mask[pick] = True

        if len(chosen) < max_raw:
            remaining = np.where(~chosen_mask)[0]
            fill = min(max_raw - len(chosen), len(remaining))
            if fill > 0:
                extra = np.random.choice(remaining, fill, replace=False)
                chosen.extend(extra.tolist())

        if len(chosen) > max_raw:
            chosen = np.random.choice(np.asarray(chosen, dtype=np.int64), max_raw, replace=False).tolist()

        idx = torch.as_tensor(np.asarray(chosen, dtype=np.int64), dtype=torch.long)
        return merged[idx]

    def set_bound(self, new_bound: torch.Tensor):
        new_bound[0, 2] = -0.1
        new_bound[1, 2] = 0.1
        # self.all_raw_frame_data = []
        self.bound = new_bound
        self.bound_changed = True

    def in_bound(self, pos):
        # pos: (2,) numpy
        b = self.bound.detach().cpu().numpy()
        return (pos >= b[0, :2]).all() and (pos <= b[1, :2]).all()

    def _xy_in_current_bound(self, pts_xy, margin=0.0):
        """Return a boolean mask for XY world points inside the current region box."""
        pts_xy = np.asarray(pts_xy, dtype=np.float32)
        if pts_xy.size == 0:
            return np.zeros((0,), dtype=bool)
        b = self.bound.detach().cpu().numpy()
        return (
            (pts_xy[:, 0] >= float(b[0, 0]) - margin) &
            (pts_xy[:, 0] <= float(b[1, 0]) + margin) &
            (pts_xy[:, 1] >= float(b[0, 1]) - margin) &
            (pts_xy[:, 1] <= float(b[1, 1]) + margin)
        )

    def _filter_raw_rows_to_bound(self, rows: torch.Tensor) -> torch.Tensor:
        """
        Keep only rows whose sampled x0/x1 stay inside the current region bound.

        Raw history is stored in world coordinates and the region bound can shrink or
        shift over time. Training rows outside the current box normalize outside
        [-0.5, 0.5] and distort the local field the region is supposed to learn.
        """
        if rows is None or len(rows) == 0:
            return rows
        arr = rows.detach().cpu().numpy()
        keep = (
            self._xy_in_current_bound(arr[:, 0:2]) &
            self._xy_in_current_bound(arr[:, 3:5])
        )
        if keep.all():
            return rows
        keep_idx = torch.as_tensor(np.where(keep)[0], dtype=torch.long)
        return rows[keep_idx]

    def normalize_tensor(self, data):
        n_repeat = data.shape[-1] // 3
        original_z = data[..., 2]
        bound_len = (self.bound[1] - self.bound[0]).repeat(n_repeat).to(data)
        bound_start = self.bound[0].repeat(n_repeat).to(data)
        norm_data = (data - bound_start) / bound_len - 0.5
        # norm_data[..., 2] = original_z
        # norm_data[..., 5] = original_z
        return norm_data

    def normalize(self, data):
        if data.shape[-1] == 2:
            data = np.stack([data[..., 0], data[..., 1], np.ones_like(data[..., 1])], axis=-1)
        original_z = data[..., 2]
        bound_len = (self.bound[1] - self.bound[0]).detach().cpu().numpy()
        bound_start = self.bound[0].detach().cpu().numpy()
        norm_data = (data - bound_start) / bound_len - 0.5
        # norm_data[..., 2] = original_z
        return norm_data

    def serialize(self):
        return {
            'version': self.version,
            'doors': [door for door in self.doors_id],
            'rooms': [room.id for room in self.rooms],
            'bound': [[float(x) for x in b] for b in self.bound]
        }
    
    def save_model(self):
        """Save region using the original-style bundle plus semantic objects."""
        if self.network is None:
            return
        self._write_region_bundle()
        print(f"✅ Saved region {self.version} to {self.folder}")

    def load_model(self):
        """Load region from the original-style region bundle."""
        self._region_paths()
        if not (os.path.exists(self.network_save_path) and os.path.exists(self.B_save_path) and os.path.exists(self.data_save_path)):
            print(f"⚠️  Incomplete region bundle in {self.folder}")
            return False
        self.saved = True
        self.load_from_disk()
        print(f"✅ Loaded region {self.version} from epoch {self.epoch}")
        return True

    def _view_grid_world(self, robot_xy, view_radius=2.0, n=160):
        """
        Robot-centered world grid cropped to region bounds.
        Returns X_w, Y_w (world), X_n, Y_n (normalized), and inb (all True).
        This eliminates NaN-masked blank space in plots.
        """
        b = self.bound.detach().cpu().numpy()
        rx, ry = float(robot_xy[0]), float(robot_xy[1])
    
        # Region bounds in world coordinates
        xmin_b, ymin_b = float(b[0, 0]), float(b[0, 1])
        xmax_b, ymax_b = float(b[1, 0]), float(b[1, 1])
    
        # Robot-centered view box
        xmin_v, xmax_v = rx - view_radius, rx + view_radius
        ymin_v, ymax_v = ry - view_radius, ry + view_radius
    
        # ✅ Tight intersection (this is the "field box" you want to plot)
        xmin = max(xmin_b, xmin_v)
        xmax = min(xmax_b, xmax_v)
        ymin = max(ymin_b, ymin_v)
        ymax = min(ymax_b, ymax_v)
    
        # Fallback if robot view is completely outside region
        if not (xmin < xmax and ymin < ymax):
            xs = np.linspace(xmin_v, xmax_v, n)
            ys = np.linspace(ymin_v, ymax_v, n)
            X_w, Y_w = np.meshgrid(xs, ys)
            # Normalize anyway (will be out of bound)
            sx = (xmax_b - xmin_b) + 1e-9
            sy = (ymax_b - ymin_b) + 1e-9
            X_n = (X_w - xmin_b) / sx - 0.5
            Y_n = (Y_w - ymin_b) / sy - 0.5
            inb = np.zeros_like(X_w, dtype=bool)
            return X_w, Y_w, X_n, Y_n, inb
    
        # Sample only within the valid plotting window
        xs = np.linspace(xmin, xmax, n)
        ys = np.linspace(ymin, ymax, n)
        X_w, Y_w = np.meshgrid(xs, ys)
    
        # Convert world -> normalized [-0.5, 0.5] using region bounds
        sx = (xmax_b - xmin_b) + 1e-9
        sy = (ymax_b - ymin_b) + 1e-9
        X_n = (X_w - xmin_b) / sx - 0.5
        Y_n = (Y_w - ymin_b) / sy - 0.5
    
        # Everything is in-bounds by construction
        inb = np.ones_like(X_w, dtype=bool)
        return X_w, Y_w, X_n, Y_n, inb

    def _denorm_points(self, pts_norm):
        """Denormalize Nx3 points from [-0.5,0.5] to world using *region bound*."""
        b = self.bound.detach().cpu().numpy()
        scale = (b[1] - b[0])
        return (pts_norm + 0.5) * scale + b[0]

    def _goal_slice_gradient_overlay(self, goal_xy, sample_xy, stride=6):
        """
        Sparse world-space visualization of -∇TT for the same slice the planner uses:
        x0 = sample point, x1 = fixed goal.
        """
        if self.function is None:
            return None
        sample_xy = np.asarray(sample_xy, dtype=np.float32)
        if sample_xy.ndim != 3 or sample_xy.shape[-1] != 2:
            return None

        xs = sample_xy[::stride, ::stride, 0]
        ys = sample_xy[::stride, ::stride, 1]
        if xs.size == 0:
            return None

        goal_xy = np.asarray(goal_xy[:2], dtype=np.float32)
        z = np.zeros((xs.size, 1), dtype=np.float32)
        x0 = np.stack([xs.reshape(-1), ys.reshape(-1)], axis=1).astype(np.float32)
        x1 = np.repeat(goal_xy.reshape(1, 2), len(x0), axis=0).astype(np.float32)
        xp = np.hstack([x0, z, x1, z])

        xp_t = torch.tensor(xp, dtype=torch.float32, device=self.device)
        xp_n = self.normalize_tensor(xp_t).requires_grad_(True)
        try:
            tau, _, coords = self.function.network.out(xp_n)
            grad = torch.autograd.grad(
                outputs=tau.sum(),
                inputs=coords,
                retain_graph=False,
                create_graph=False,
            )[0]
        except Exception:
            return None

        grad_xy_norm = grad[:, :2].detach().cpu().numpy()
        b = self.bound.detach().cpu().numpy()
        bound_len = (b[1, :2] - b[0, :2]).astype(np.float32)
        grad_xy_world = grad_xy_norm / (bound_len[None, :] + 1e-8)
        flow = -grad_xy_world
        mag = np.linalg.norm(flow, axis=1, keepdims=True)
        good = mag[:, 0] > 1e-6
        flow[good] /= mag[good]
        flow[~good] = 0.0

        return xs, ys, flow[:, 0].reshape(xs.shape), flow[:, 1].reshape(xs.shape)

    def save_debug_plot(self, src_world, view_radius=2.0, n=80, robot_xy=None):
        """
        Speed-field diagnostic (restored from March 11 working version):
          - x0 sweeps the world-coord grid, x1 = fixed goal (NOT x0=robot, x1=grid)
          - Speed = 1/|∂τ/∂x0| gives LOCAL traversal speed at each grid cell
          - No normalisation: raw Speed() passed to pcolormesh(vmin=0, vmax=1)
            → free-space cells (Speed>>1) clamp to yellow, walls (Speed≈0) stay dark
          - Contours: linear + quantile mix for even visual density
          - Grid centered on robot via _view_grid_world (world-coord axes)
        """
        if self.function is None:
            return

        robot_x, robot_y, robot_z = float(src_world[0]), float(src_world[1]), float(src_world[2])
        robot_x_dot = float(robot_xy[0]) if robot_xy is not None else robot_x
        robot_y_dot = float(robot_xy[1]) if robot_xy is not None else robot_y

        src_norm = self.normalize(np.array([[robot_x, robot_y, robot_z]], dtype=np.float32))[0]

        # Goal: use cached latest_goal, then latest_path endpoint, then robot pos
        goal_xy = getattr(self, 'latest_goal', None)
        if goal_xy is not None:
            goal_xy = np.asarray(goal_xy[:2], dtype=np.float32)
        path = getattr(self, 'latest_path', None)
        if goal_xy is None and path is not None and len(path) > 0:
            goal_xy = np.asarray(path[-1][:2], dtype=np.float32)
        if goal_xy is None:
            goal_xy = np.array([robot_x_dot, robot_y_dot], dtype=np.float32)
        goal_norm = self.normalize(np.array([[goal_xy[0], goal_xy[1], robot_z]], dtype=np.float32))[0]

        # Robot-centered world-coord grid cropped to region bounds
        X_w, Y_w, X_n, Y_n, inb = self._view_grid_world([robot_x, robot_y],
                                                          view_radius=view_radius, n=n)
        if inb is None or (hasattr(inb, 'any') and not inb.any()):
            return

        # x0 = grid (variable), x1 = goal (fixed)
        # Speed(x0_grid → x1_goal) = local traversal speed at each grid cell.
        # This gives the room's speed field independent of robot position.
        XP = np.zeros((X_w.size, 2 * self.dim), dtype=np.float32)
        XP[:, :self.dim] = goal_norm      # x0 = goal initially (z,w filled)
        XP[:, self.dim:] = goal_norm      # x1 = goal (fixed)
        XP[:, 0] = X_n.reshape(-1)       # x0_x = grid x
        XP[:, 1] = Y_n.reshape(-1)       # x0_y = grid y
        XP[:, 2] = src_norm[2]           # x0_z = robot z

        if not self.inference_lock.acquire(timeout=2.0):
            return
        try:
            XP_t = torch.tensor(XP, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                TT = self.function.TravelTimes(XP_t).detach().cpu().numpy().reshape(X_w.shape)
            XP_t2 = torch.tensor(XP, dtype=torch.float32,
                                  device=self.device).requires_grad_(True)
            with torch.enable_grad():
                V_t = self.function.Speed(XP_t2)
            V = V_t.detach().cpu().numpy().reshape(X_w.shape)
        finally:
            self.inference_lock.release()

        # No normalisation: Speed() >> 1 in free space → clamped to yellow by vmax=1.
        # Speed() ≈ 0 at walls → dark. pcolormesh clips naturally.

        cmap_speed = plt.get_cmap('viridis').copy()
        cmap_speed.set_bad(alpha=0.0)

        fig, ax = plt.subplots(figsize=(9, 9))
        m0 = ax.pcolormesh(X_w, Y_w, V, vmin=0.0, vmax=1.0, cmap=cmap_speed, shading='auto')

        ax.contour(X_w, Y_w, TT, np.arange(0, 10, 0.03),
                   cmap='gist_heat', linewidths=0.5, alpha=0.8)

        fig.colorbar(m0, ax=ax, pad=0.02)
        ax.set_title(f"Speed field  epoch={self.epoch}")
        ax.scatter(robot_x_dot, robot_y_dot, c='red', s=120, zorder=6, marker='o', label='Robot')

        if path is not None and len(path) > 1:
            path_arr = np.array(path)
            ax.plot(path_arr[:, 0], path_arr[:, 1],
                    color='cyan', linewidth=1.5, zorder=7, linestyle='--', label='Path')
            ax.scatter(goal_xy[0], goal_xy[1], c='lime', s=110, zorder=6,
                       marker='*', label='Goal')

        ax.set_xlim(float(np.min(X_w)), float(np.max(X_w)))
        ax.set_ylim(float(np.min(Y_w)), float(np.max(Y_w)))
        ax.set_aspect('equal')
        ax.legend(fontsize=7)
        fig.suptitle(
            f"Region {self.version} | Epoch {self.epoch} | Robot ({robot_x:.2f}, {robot_y:.2f})",
            fontsize=12
        )

        out_dir = os.path.join(self.folder, "debug")
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(os.path.join(out_dir, f"field_{self.epoch:05d}.png"), dpi=72, bbox_inches='tight')
        fig.savefig(os.path.join(self.folder, "field.png"), dpi=72, bbox_inches='tight')
        plt.close(fig)

    def save_training_vis(self, points, speeds, robot_xy=None, view_radius=2.0):
        if len(points) == 0:
            return

        pts = points.detach().cpu().numpy()   # Nx6 (x0,y0,z0,x1,y1,z1) normalized
        spd = speeds.detach().cpu().numpy()   # Nx2 (y0,y1)

        # Denormalize using region bound (NOT minibatch bounds)
        x0_world = self._denorm_points(pts[:, :3])
        x1_world = self._denorm_points(pts[:, 3:6])

        dists_world = np.linalg.norm(x0_world[:, :2] - x1_world[:, :2], axis=1)

        # Filter out x1 points that are too close to their x0 (< 0.05m) —
        # these look like wall-clustered x0 points in the "Goal x1+y1" subplot,
        # creating a misleading visual overlap.
        far_enough = dists_world >= 0.05

        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        cmap = cm.get_cmap('viridis')
        vmin, vmax = float(spd.min()), float(spd.max())

        ax[0].scatter(x0_world[:, 0], x0_world[:, 1], c=spd[:, 0],
                      cmap=cmap, vmin=vmin, vmax=vmax, s=6, alpha=0.7)
        ax[0].set_title(f"Start x0+y0 (N={len(pts)})")
        ax[0].set_aspect('equal')

        # Plot free-space x1 first (low speed, dark), then near-wall x1 on top
        # (high speed, bright).  Sort by speed ascending so bright dots render
        # last and appear on top — proper z-order without zorder hacks.
        x1_vis   = x1_world[far_enough]
        spd1_vis = spd[far_enough, 1]
        sort_idx = np.argsort(spd1_vis)
        ax[1].scatter(x1_vis[sort_idx, 0], x1_vis[sort_idx, 1], c=spd1_vis[sort_idx],
                      cmap=cmap, vmin=vmin, vmax=vmax, s=6, alpha=0.7)
        ax[1].set_title(f"Goal x1+y1 (N={far_enough.sum()})")
        ax[1].set_aspect('equal')

        ax[2].hist(dists_world, bins=40, alpha=0.8, color='orange', density=True)
        ax[2].axvline(dists_world.mean(), color='red', lw=2,
                      label=f"μ={dists_world.mean():.2f}m")
        ax[2].set_xlabel("World Distance (m)")
        ax[2].set_title("x0→x1 Dist")
        ax[2].legend()

        fig.colorbar(
            cm.ScalarMappable(cmap=cmap, norm=colors.Normalize(vmin=vmin, vmax=vmax)),
            ax=ax, label="Norm Speed", shrink=0.8
        )

        # ✅ Region bounding box is now correct in same coordinate system
        b = self.bound.detach().cpu().numpy()
        for axi in ax[:2]:
            rect = plt.Rectangle((b[0, 0], b[0, 1]),
                                 b[1, 0] - b[0, 0],
                                 b[1, 1] - b[0, 1],
                                 linewidth=2, edgecolor='white', facecolor='none', zorder=10)
            axi.add_patch(rect)

        # ✅ Robot-centered view (optional, but matches your preference)
        #if robot_xy is not None:
        #    rx, ry = float(robot_xy[0]), float(robot_xy[1])
        #    for axi in ax[:2]:
        #        axi.set_xlim(rx - view_radius, rx + view_radius)
        #        axi.set_ylim(ry - view_radius, ry + view_radius)
        b = self.bound.detach().cpu().numpy()
        xmin, ymin = float(b[0, 0]), float(b[0, 1])
        xmax, ymax = float(b[1, 0]), float(b[1, 1])
        
        for axi in ax[:2]:
            axi.set_xlim(xmin, xmax)
            axi.set_ylim(ymin, ymax)

        out = os.path.join(self.folder, f"train_data_{self.epoch:03d}.png")
        fig.savefig(out, dpi=72, bbox_inches='tight')
        plt.close(fig)

        #print(f"[VIS] {out} | x0:{spd[:,0].mean():.3f} x1:{spd[:,1].mean():.3f} "
        #      f"| dist μ:{dists_world.mean():.2f}m σ:{dists_world.std():.2f}m")


    # ============ DEAD CODE (never called) ============

    # Last reviewed as dead: 2026-03-17
    def debug_show_lidar(self):
        if self.lidar_obs is None:
            return
        pts = np.asarray(self.lidar_obs.points)
        if len(pts) < 10:
            return
        # --------------------------------------------------
        # ensure only ONE viewer ever exists
        # --------------------------------------------------
        if not hasattr(self, "_viewer_created"):
            self._viewer_created = False

        if not self._viewer_created:
            self._viewer_created = True
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window("LiDAR debug", width=900, height=700)
            self.vis_pcd = o3d.geometry.PointCloud()
            self.vis.add_geometry(self.vis_pcd)
            render_opt = self.vis.get_render_option()
            render_opt.point_size = 3.0
            render_opt.background_color = np.array([0, 0, 0])
            self._reset_cam = True
        # --------------------------------------------------
        # update geometry
        # --------------------------------------------------
        self.vis_pcd.points = o3d.utility.Vector3dVector(pts)
        z = pts[:, 2]
        z_n = (z - z.min()) / (np.ptp(z) + 1e-8)
        colors = plt.get_cmap("viridis")(z_n)[:, :3]
        self.vis_pcd.colors = o3d.utility.Vector3dVector(colors)
        self.vis.update_geometry(self.vis_pcd)
        if getattr(self, "_reset_cam", False):
            self.vis.reset_view_point(True)
            self._reset_cam = False
        self.vis.poll_events()
        self.vis.update_renderer()

    # Last reviewed as dead: 2026-03-17
    def point_obs_dist(self, points):
        v_obs = torch.from_numpy(np.asarray(self.lidar_obs.points)).to(points.device).float()
        n_obs = torch.from_numpy(np.asarray(self.lidar_obs.normals)).to(points.device).float()
        with Timer(f'kdtree query, n={len(points)}'):
            dists, idxs = self.kd_tree.query(points.cpu().numpy()) # was kd_tree
            dists = torch.from_numpy(dists).to(points.device).float()
            idxs = torch.from_numpy(idxs).to(points.device).long()

        normal = points - v_obs[idxs]
        with Timer(f'normalize'):
            normal = normal / torch.norm(normal, dim=1, keepdim=True)
        lidar_normal = n_obs[idxs]
        with Timer(f'dot product'):
            normal_inner = torch.einsum('ij,ij->i', normal, lidar_normal)
        return dists, v_obs[idxs], normal_inner, normal

    # Last reviewed as dead: 2026-03-17
    def recompute_lidar(self):
        pcds = []
        for lidar_scan, pos in self.all_lidar:
            lidar_scan_pcd = np_to_pcd(self.normalize(lidar_scan), self.normalize(pos))
            pcds.append(lidar_scan_pcd)
        self.lidar_obs = combine_pcd(*pcds)

    # Last reviewed as dead: 2026-03-17
    def predict_tt(self, src, tgt):
        src = src.reshape(3)
        tgt = tgt.reshape(3)
        XP = Tensor(np.concatenate([src, tgt])).reshape(1, 6).to(self.Params['Device'])
        return self.function.TravelTimes(XP)

    # Last reviewed as dead: 2026-03-17
    def add_door(self, door):
        self.doors_id.add(door.id)
        self.doors.add(door)

    # Last reviewed as dead: 2026-03-17
    def get_rect(self):
        bmin = self.bound[0, :2].detach().cpu().numpy()
        bmax = self.bound[1, :2].detach().cpu().numpy()
        return (bmin[0], bmin[1], bmax[0], bmax[1])

    # Last reviewed as dead: 2026-03-17
    def unnormalize_tensor(self, data):
        n_repeat = data.shape[-1] // 3
        bound_len = (self.bound[1] - self.bound[0]).repeat(n_repeat).to(data)
        bound_start = self.bound[0].repeat(n_repeat).to(data)
        unnorm_data = (data + 0.5) * bound_len + bound_start
        # unnorm_data[..., 2] = data[..., 2]
        # unnorm_data[..., -1] = data[..., -1]
        return unnorm_data

    # Last reviewed as dead: 2026-03-17
    def unnormalize(self, data):
        if data.shape[-1] == 2:
            data = np.stack([data[..., 0], data[..., 1], np.ones_like(data[..., 1])], axis=-1)
        bound_len = (self.bound[1] - self.bound[0]).detach().cpu().numpy()
        bound_start = self.bound[0].detach().cpu().numpy()
        unnorm_data = (data + 0.5) * bound_len + bound_start
        # unnorm_data[..., 2] = data[..., 2]
        return unnorm_data

    # Last reviewed as dead: 2026-03-17
    def generate_field_path_to_crossing(
        self,
        start_xy: np.ndarray,
        crossing_xy: np.ndarray,
        step_size: float = 0.15,
        max_steps: int = 80,
    ) -> list:
        """
        Generate a field-guided path from start_xy to crossing_xy using THIS
        region's travel-time field.  Falls back to straight-line if the field
        is unavailable or the crossing is outside this region's bound.
        """
        # Quick geometric fallback
        def _straight(a, b, steps=6):
            ts = np.linspace(0, 1, max(2, steps))
            return [np.array(a) + t * (np.array(b) - np.array(a)) for t in ts]

        if self.function is None:
            return _straight(start_xy, crossing_xy)

        b = self.bound.detach().cpu().numpy()
        # If crossing is outside this region just do straight line
        cx_in = (b[0, 0] <= crossing_xy[0] <= b[1, 0] and
                 b[0, 1] <= crossing_xy[1] <= b[1, 1])
        if not cx_in:
            return _straight(start_xy, crossing_xy)

        return self.generate_field_path(start_xy, crossing_xy,
                                        step_size=step_size, max_steps=max_steps)

    # Last reviewed as dead: 2026-03-17
    def save_path_vis(self, robot_xy, goal_xy, lidar_pts_world=None,
                      frontier_pts=None, step_size=0.25, max_steps=60):
        """
        Simulate the greedy field-gradient path from robot_xy → goal_xy and
        save a composite image showing:
          - Speed field (viridis background)
          - Travel-time contours (gist_heat)
          - LiDAR obstacle points
          - Frontier candidates
          - Simulated field path (blue line)
          - Robot position (red dot) and goal (green star)

        Saved to: <region_folder>/debug/path_<epoch:05d>.png
                   <region_folder>/path_latest.png   (overwritten each call)
        """
        if self.function is None:
            return

        b = self.bound.detach().cpu().numpy()
        rx, ry = float(robot_xy[0]), float(robot_xy[1])
        gx, gy = float(goal_xy[0]), float(goal_xy[1])

        # ── Build normalised grid for background field ────────────────────
        spacing = 1.0 / 80.0
        xs = np.arange(-0.5, 0.5, spacing)
        ys = np.arange(-0.5, 0.5, spacing)
        X, Y = np.meshgrid(xs, ys)

        src_norm = self.normalize(np.array([[rx, ry, 0.0]]))[0]

        goal_norm = self.normalize(np.array([[gx, gy, 0.0]]))[0]
        XP = np.zeros((X.size, 2 * self.dim), dtype=np.float32)
        XP[:, self.dim:]    = goal_norm       # x1 = goal (fixed)
        XP[:, :self.dim]    = goal_norm       # fill x0, then overwrite
        XP[:, 0] = X.flatten()               # x0_x sweeps grid
        XP[:, 1] = Y.flatten()               # x0_y sweeps grid
        XP[:, 2] = goal_norm[2]

        XP_t = torch.tensor(XP, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            TT = self.function.TravelTimes(XP_t).cpu().numpy().reshape(X.shape)

        # Speed needs grad enabled AND requires_grad=True on the input
        XP_t2 = torch.tensor(XP, dtype=torch.float32).to(self.device).requires_grad_(True)
        with torch.enable_grad():
            V = self.function.Speed(XP_t2).detach().cpu().numpy().reshape(X.shape)

        # Unnormalise grid to world coords for axis labels
        X_w = (X + 0.5) * (b[1, 0] - b[0, 0]) + b[0, 0]
        Y_w = (Y + 0.5) * (b[1, 1] - b[0, 1]) + b[0, 1]

        # ── Simulate greedy path by following -∇TT ──────────────────────
        path = [[rx, ry]]
        cx, cy = rx, ry
        for _ in range(max_steps):
            dist_to_goal = np.hypot(cx - gx, cy - gy)
            if dist_to_goal < step_size:
                path.append([gx, gy])
                break

            # Query gradient at current 
            xp = np.array([[cx, cy, 0.0, gx, gy, 0.0]], dtype=np.float32)
            xp_t = torch.tensor(xp).to(self.device)
            xp_n = self.normalize_tensor(xp_t)
            xp_n.requires_grad_(True)

            try:
                tt_val = self.function.TravelTimes(xp_n)
                grad = torch.autograd.grad(tt_val.sum(), xp_n)[0]
                grad_xy = grad[0, :2].detach().cpu().numpy()
                norm = np.linalg.norm(grad_xy) + 1e-8
                # Geometric blended direction (same as get_next_waypoint)
                dx, dy = gx - cx, gy - cy
                geom = np.array([dx, dy]) / (np.hypot(dx, dy) + 1e-8)
                field_dir = -grad_xy / norm
                direction = 0.6 * field_dir + 0.4 * geom
                direction /= np.linalg.norm(direction) + 1e-8
            except Exception:
                dx, dy = gx - cx, gy - cy
                direction = np.array([dx, dy]) / (np.hypot(dx, dy) + 1e-8)

            cx += direction[0] * step_size
            cy += direction[1] * step_size
            path.append([cx, cy])

        path = np.array(path)

        # ── Plot ─────────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(9, 9))

        # Speed field background (disabled for path-centric diagnostics)
        # ax.pcolormesh(X_w, Y_w, V, vmin=0, vmax=1, cmap='viridis', shading='auto', alpha=0.85)

        # Travel-time contours
        ax.contour(X_w, Y_w, TT, levels=TT_CONTOUR_LEVELS,
                   cmap='gist_heat', linewidths=0.4, alpha=0.6)

        # LiDAR obstacle points
        if lidar_pts_world is not None and len(lidar_pts_world) > 0:
            lp = np.asarray(lidar_pts_world)
            ax.scatter(lp[:, 0], lp[:, 1], c='white', s=4, zorder=3,
                       alpha=0.6, label='LiDAR hits')

        # Frontier candidates
        if frontier_pts is not None and len(frontier_pts) > 0:
            fp = np.asarray(frontier_pts)
            ax.scatter(fp[:, 0], fp[:, 1], c='cyan', s=20, marker='^',
                       zorder=4, alpha=0.8, label='Frontiers')

        # Simulated path
        if len(path) > 1:
            ax.plot(path[:, 0], path[:, 1], 'b-o', linewidth=2,
                    markersize=3, zorder=5, label='Field path')

        # Robot and goal
        ax.scatter(rx, ry, c='red',   s=120, zorder=6, marker='o', label='Robot')
        ax.scatter(gx, gy, c='lime',  s=180, zorder=6, marker='*', label='Goal')

        # Region bound box
        rect = plt.Rectangle(
            (b[0, 0], b[0, 1]),
            b[1, 0] - b[0, 0],
            b[1, 1] - b[0, 1],
            linewidth=2, edgecolor='yellow', facecolor='none', zorder=7
        )
        ax.add_patch(rect)

        ax.set_aspect('equal')
        ax.set_title(
            f"Region {self.version} | Epoch {self.epoch} | "
            f"Path steps: {len(path)-1} | "
            f"Goal dist: {np.hypot(rx-gx, ry-gy):.2f}m"
        )
        ax.legend(loc='upper right', fontsize=7)

        out_dir = os.path.join(self.folder, "debug")
        os.makedirs(out_dir, exist_ok=True)
        path_file = os.path.join(out_dir, f"path_{self.epoch:05d}.png")
        fig.savefig(path_file, dpi=120, bbox_inches='tight')
        fig.savefig(os.path.join(self.folder, "path_latest.png"), bbox_inches='tight')
        plt.close(fig)
