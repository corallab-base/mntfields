import numpy as np
import cv2
from scipy.ndimage import convolve
from collections import deque
import open3d as o3d
import time

def connected_components(img):
    img = img.astype(np.uint8)
    threshold = cv2.threshold(img, 0.5, 1, cv2.THRESH_BINARY)[1]
    return cv2.connectedComponents(threshold)[1]

colors = np.array([
    (0, 0, 255),      # Blue
[239,255,191],[191,234,255],[255,250,191],[191,255,255],[255,244,191],[191,212,255],[255,218,191],[191,255,234],[255,191,223],[207,191,255],[191,250,255],[191,202,255],[197,191,255],[255,191,197],[255,191,213],[239,191,255],[191,228,255],[234,255,191],[197,255,191],[255,212,191],[191,223,255],[255,191,191],[191,255,223],[191,239,255],[191,255,218],[244,191,255],[228,191,255],[223,255,191],[191,255,244],[191,191,255],[255,197,191],[255,191,191],[218,255,191],[255,228,191],[255,191,239],[191,255,197],[255,191,250],[255,191,207],[228,255,191],[191,197,255],[202,255,191],[191,255,239],[191,255,213],[255,255,191],[191,207,255],[218,191,255],[255,191,202],[255,223,191],[255,239,191],[255,202,191],[255,191,228],[255,207,191],[191,255,250],[212,191,255],[234,191,255],[191,255,228],[207,255,191],[244,255,191],[191,218,255],[191,244,255],[223,191,255],[255,191,218],[191,255,207],[212,255,191],[255,191,234],[255,191,255],[255,234,191],[191,255,191],[202,191,255],[250,255,191],[191,255,202],[255,191,244],[250,191,255]
])

def mask_center(mask):
    mid_i = int(np.median(np.where(mask)[0]))
    if mask[mid_i].sum() == 0:
        mid_i += 1
    try:
        mid_j = int(np.median(np.where(mask[mid_i])))
    except:
        mid_j = 0
        
    return mid_i, mid_j

def get_neighbor_at_least(bg, fg, n, r=3):
    assert r % 2 == 1
    # get new fg mask s.t. every occupied point in fg has at least n neighbors in radius of r in bg
    bg = bg.astype(int)
    fg = fg.astype(int)
    kernel = np.ones((r, r), dtype=int)
    neighbor_count = convolve(bg, kernel, mode='constant', cval=0)
    neighbor_count[fg == 0] = 0
    valid_idxs = np.where(neighbor_count >= n)
    new_fg = np.zeros_like(fg)
    new_fg[valid_idxs[0], valid_idxs[1]] = 1
    
    return new_fg   

def cc_1d(arr):
    s = 0
    ccs = []
    for i in range(len(arr)):
        if arr[i] != arr[s]:
            if arr[s]:
                ccs.append((arr[s:i], (s+i) // 2))
            s = i
    return ccs

def find_nearest_occupied_cell(occ_map, i, j):
    """
    Find the nearest occupied cell (1) from an arbitrary free cell (i, j).
    
    :param occ_map: 2D numpy array where 1 indicates occupied cells and 0 indicates free space.
    :param i: Row index of the starting cell.
    :param j: Column index of the starting cell.
    :return: (row, col) of the nearest occupied cell or None if no occupied cell exists.
    """
    if occ_map[i, j] == 1:
        return (i, j)  # Already occupied

    rows, cols = occ_map.shape
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    visited = np.zeros_like(occ_map, dtype=bool)
    queue = deque([(i, j)])

    while queue:
        x, y = queue.popleft()
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and not visited[nx, ny]:
                if occ_map[nx, ny] == 1:
                    return (nx, ny)  # Found the nearest occupied cell
                visited[nx, ny] = True
                queue.append((nx, ny))

    return None  # No occupied cell found

def np_to_pcd(arr, cur_pos, downsample=False):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(arr)
    if downsample:
        pcd = downsample_pcd(pcd)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd.orient_normals_towards_camera_location(camera_location=cur_pos)
    return pcd

def downsample_pcd(pcd):
    return pcd.voxel_down_sample(voxel_size=0.001)

def combine_pcd(*pcds):
    points = []
    normals = []
    for pcd in pcds:
        points.append(np.asarray(pcd.points))
        normals.append(np.asarray(pcd.normals))
    points = np.concatenate(points, axis=0)
    normals = np.concatenate(normals, axis=0)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    return downsample_pcd(pcd)

def area(rect):
    return (rect[2] - rect[0]) * (rect[3] - rect[1])

def percent_overlap(rect1, rect2):
    # overlap area / smaller area
    if rect1[2] <= rect2[0] or rect1[0] >= rect2[2] or rect1[3] <= rect2[1] or rect1[1] >= rect2[3]:
        return 0
    
    xmin = max(rect1[0], rect2[0])
    ymin = max(rect1[1], rect2[1])
    xmax = min(rect1[2], rect2[2])
    ymax = min(rect1[3], rect2[3])

    return area((xmin, ymin, xmax, ymax)) / min(area(rect1), area(rect2))

class Timer:
    def __init__(self, text):
        self.text = text
        self.elapsed_time = 0

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, *args):
        self.elapsed_time = time.time() - self.start
        print(f'{self.text} time: {self.elapsed_time}')


# ── Voronoi ownership utilities ───────────────────────────────────────────────

def compute_ownership_map(free, regions, origin, cell_size):
    """Assign each free cell exclusively to the nearest region (Voronoi).

    Parameters
    ----------
    free : np.ndarray [H, W] bool  — True where SLAM reports free space
    regions : dict[int, Region]    — keyed by region.version
    origin : array-like [2]        — world (x, y) of cell (row=0, col=0)
    cell_size : float              — metres per cell

    Returns
    -------
    ownership : np.ndarray [H, W] int32
        Each cell contains the owning region's version number (0 = unowned).
    """
    H, W = free.shape
    ownership = np.zeros((H, W), dtype=np.int32)
    if not regions:
        return ownership

    rows, cols = np.where(free)
    if len(rows) == 0:
        return ownership

    ox, oy = float(origin[0]), float(origin[1])
    world_x = ox + cols * cell_size
    world_y = oy + rows * cell_size
    world_xy = np.stack([world_x, world_y], axis=1)  # (N_free, 2)

    reg_list = list(regions.values())
    reg_versions = np.array([r.version for r in reg_list], dtype=np.int32)

    # Use spawn_center if set, otherwise fall back to AABB midpoint.
    def _center(r):
        if getattr(r, 'spawn_center', None) is not None:
            return r.spawn_center[:2]
        b = r.bound.detach().cpu().numpy()
        return np.array([(b[0, 0] + b[1, 0]) * 0.5,
                         (b[0, 1] + b[1, 1]) * 0.5], dtype=np.float32)

    reg_centers = np.array([_center(r) for r in reg_list], dtype=np.float32)
    reg_bounds = np.array([
        [r.bound[0, 0].item(), r.bound[0, 1].item(),
         r.bound[1, 0].item(), r.bound[1, 1].item()]
        for r in reg_list], dtype=np.float32)  # (N_reg, 4): xmin,ymin,xmax,ymax

    # Containment mask (N_free, N_reg)
    contained = (
        (world_xy[:, 0:1] >= reg_bounds[:, 0]) &
        (world_xy[:, 0:1] <= reg_bounds[:, 2]) &
        (world_xy[:, 1:2] >= reg_bounds[:, 1]) &
        (world_xy[:, 1:2] <= reg_bounds[:, 3])
    )
    # Distance to each region center (N_free, N_reg)
    dist = np.linalg.norm(
        world_xy[:, None, :] - reg_centers[None, :, :], axis=2)
    dist[~contained] = np.inf

    best = np.argmin(dist, axis=1)
    has_owner = np.isfinite(dist[np.arange(len(rows)), best])
    ownership[rows[has_owner], cols[has_owner]] = reg_versions[best[has_owner]]
    return ownership


def compute_crossing_points(ownership, origin, cell_size, occ_grid=None):
    """Return {(ver_a, ver_b): centroid_world_xy} for every ownership boundary.

    Parameters
    ----------
    ownership : np.ndarray [H, W] int32
    origin    : array-like [2]
    cell_size : float
    occ_grid  : np.ndarray [H, W] bool, optional
        True where occupied.  Crossings landing on occupied cells are discarded.

    Returns
    -------
    dict mapping (min_ver, max_ver) -> np.ndarray [2] world xy of boundary centroid
    """
    ox, oy = float(origin[0]), float(origin[1])
    crossings = {}
    # horizontal neighbours (same row, adjacent columns)
    a_h = ownership[:, :-1]
    b_h = ownership[:, 1:]
    mask_h = (a_h != b_h) & (a_h != 0) & (b_h != 0)
    rs_h, cs_h = np.where(mask_h)
    for r, c in zip(rs_h, cs_h):
        a, b = int(a_h[r, c]), int(b_h[r, c])
        key = (min(a, b), max(a, b))
        wx = ox + (c + 0.5) * cell_size
        wy = oy + r * cell_size
        crossings.setdefault(key, []).append([wx, wy])
    # vertical neighbours (same column, adjacent rows)
    a_v = ownership[:-1, :]
    b_v = ownership[1:, :]
    mask_v = (a_v != b_v) & (a_v != 0) & (b_v != 0)
    rs_v, cs_v = np.where(mask_v)
    for r, c in zip(rs_v, cs_v):
        a, b = int(a_v[r, c]), int(b_v[r, c])
        key = (min(a, b), max(a, b))
        wx = ox + c * cell_size
        wy = oy + (r + 0.5) * cell_size
        crossings.setdefault(key, []).append([wx, wy])

    result = {}
    for k, v_list in crossings.items():
        centroid = np.array(v_list, dtype=np.float32).mean(axis=0)
        # Validate crossing is in free space
        if occ_grid is not None:
            ci = int((centroid[0] - ox) / cell_size)
            ri = int((centroid[1] - oy) / cell_size)
            H_o, W_o = occ_grid.shape
            if 0 <= ri < H_o and 0 <= ci < W_o and occ_grid[ri, ci]:
                continue  # crossing lands on wall — discard
        result[k] = centroid
    return result


def straight_line_path(start_xy, goal_xy, step=0.10, collision_fn=None):
    """Walk straight toward goal, stop at first collision.  Returns Nx2 float32."""
    s = np.array(start_xy[:2], dtype=np.float32)
    g = np.array(goal_xy[:2], dtype=np.float32)
    d = float(np.linalg.norm(g - s))
    if d < 1e-4:
        return s.reshape(1, 2)
    direction = (g - s) / d
    n_steps = int(d / step)
    pts = [s.copy()]
    for i in range(1, n_steps + 1):
        pt = s + i * step * direction
        if collision_fn is not None and collision_fn(pt):
            break
        pts.append(pt.astype(np.float32))
    # Append exact goal if we didn't collide and last point is far from goal
    if len(pts) > 1 and float(np.linalg.norm(pts[-1] - g)) > step * 0.5:
        if collision_fn is None or not collision_fn(g):
            pts.append(g.copy())
    return np.array(pts, dtype=np.float32)


def owner_region_for_xy(xy, ownership_map, regions, origin, cell_size):
    """Return the owning Region for world point xy, using the ownership map.

    Falls back to nearest spawn_center if xy is outside the map or unowned.
    """
    if ownership_map is None or not regions:
        return None
    H, W = ownership_map.shape
    ox, oy = float(origin[0]), float(origin[1])
    c = int((float(xy[0]) - ox) / cell_size)
    r = int((float(xy[1]) - oy) / cell_size)
    if 0 <= r < H and 0 <= c < W:
        ver = int(ownership_map[r, c])
        if ver != 0:
            return regions.get(ver)
    # Fallback: nearest center
    def _center(reg):
        if getattr(reg, 'spawn_center', None) is not None:
            return reg.spawn_center[:2]
        b = reg.bound.detach().cpu().numpy()
        return np.array([(b[0, 0] + b[1, 0]) * 0.5,
                         (b[0, 1] + b[1, 1]) * 0.5], dtype=np.float32)

    best = min(
        regions.values(),
        key=lambda reg: float(np.linalg.norm(_center(reg) - np.asarray(xy[:2], dtype=np.float32))),
        default=None)
    return best


def room_id_for_xy(xy, room_map, origin, cell_size):
    """Return confirmed room id at world point xy, or None if unavailable/unlabeled."""
    if room_map is None:
        return None
    H, W = room_map.shape
    ox, oy = float(origin[0]), float(origin[1])
    c = int((float(xy[0]) - ox) / cell_size)
    r = int((float(xy[1]) - oy) / cell_size)
    if 0 <= r < H and 0 <= c < W:
        rid = int(room_map[r, c])
        return rid if rid > 0 else None
    return None


def compute_region_room_ownership(ownership_map, room_map, regions):
    """Aggregate how much of each region's owned support lies in each confirmed room."""
    out = {'by_region': {}, 'by_room': {}}
    if ownership_map is None or room_map is None or not regions:
        return out
    if ownership_map.shape != room_map.shape:
        return out

    own = ownership_map.astype(np.int32, copy=False)
    rooms = room_map.astype(np.int32, copy=False)
    valid = (own > 0) & (rooms > 0)
    if not np.any(valid):
        return out

    reg_ids = own[valid].reshape(-1)
    room_ids = rooms[valid].reshape(-1)
    room_totals = {}
    for rid in np.unique(room_ids):
        room_totals[int(rid)] = int(np.count_nonzero(room_ids == rid))

    for ver, reg in regions.items():
        mask = (reg_ids == int(ver))
        region_cells = int(np.count_nonzero(mask))
        if region_cells <= 0:
            continue
        region_room_ids = room_ids[mask]
        uniq, counts = np.unique(region_room_ids, return_counts=True)
        scores = []
        for room_id, count in zip(uniq.tolist(), counts.tolist()):
            room_total = max(1, int(room_totals.get(int(room_id), count)))
            scores.append({
                'room_id': int(room_id),
                'owned_cells': int(count),
                'region_overlap_ratio': float(count) / float(max(1, region_cells)),
                'room_coverage_ratio': float(count) / float(room_total),
            })
        scores.sort(key=lambda s: (-s['owned_cells'], -s['room_coverage_ratio'], int(s['room_id'])))
        primary = scores[0]['room_id'] if scores else None
        center_xy = None
        if getattr(reg, 'spawn_center', None) is not None:
            center_xy = [float(reg.spawn_center[0]), float(reg.spawn_center[1])]
        else:
            try:
                b = reg.bound.detach().cpu().numpy()
                center_xy = [float((b[0, 0] + b[1, 0]) * 0.5), float((b[0, 1] + b[1, 1]) * 0.5)]
            except Exception:
                center_xy = None
        out['by_region'][int(ver)] = {
            'region_version': int(ver),
            'primary_room_id': (int(primary) if primary is not None else None),
            'owned_cells_in_rooms': int(region_cells),
            'scores': scores,
            'center_xy': center_xy,
        }
        for score in scores:
            room_id = int(score['room_id'])
            out['by_room'].setdefault(room_id, []).append({
                'region_version': int(ver),
                'owned_cells': int(score['owned_cells']),
                'region_overlap_ratio': float(score['region_overlap_ratio']),
                'room_coverage_ratio': float(score['room_coverage_ratio']),
            })

    for room_id, entries in out['by_room'].items():
        entries.sort(key=lambda e: (-e['owned_cells'], -e['room_coverage_ratio'], int(e['region_version'])))
    return out


def select_best_region_for_room(room_id, region_room_ownership, regions, goal_xy=None):
    """Pick the best region for a confirmed room, optionally biased toward a goal point."""
    if room_id is None or region_room_ownership is None or not regions:
        return None
    by_room = region_room_ownership.get('by_room', {}) if isinstance(region_room_ownership, dict) else {}
    entries = by_room.get(int(room_id), [])
    if not entries:
        return None

    goal_xy = None if goal_xy is None else np.asarray(goal_xy[:2], dtype=np.float32)

    def _center(reg):
        if getattr(reg, 'spawn_center', None) is not None:
            return np.asarray(reg.spawn_center[:2], dtype=np.float32)
        b = reg.bound.detach().cpu().numpy()
        return np.array([(b[0, 0] + b[1, 0]) * 0.5, (b[0, 1] + b[1, 1]) * 0.5], dtype=np.float32)

    best = None
    best_key = None
    for entry in entries:
        reg = regions.get(int(entry['region_version']))
        if reg is None:
            continue
        dist = 0.0
        if goal_xy is not None:
            dist = float(np.linalg.norm(_center(reg) - goal_xy))
        key = (
            -float(entry.get('owned_cells', 0)),
            -float(entry.get('room_coverage_ratio', 0.0)),
            -float(entry.get('region_overlap_ratio', 0.0)),
            dist,
            int(entry['region_version']),
        )
        if best_key is None or key < best_key:
            best_key = key
            best = reg
    return best
