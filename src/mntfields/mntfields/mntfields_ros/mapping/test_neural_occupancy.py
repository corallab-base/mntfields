import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from mntfields.mntfields_core.core.region import Region
from mntfields.mntfields_core.core.model_base import Model

# =========================
# CONFIG
# =========================
SNAPSHOT_PATH = "/home/mayank/turtlebot3_ws/mntfields_model/mntfields_output/occupancy_snapshots/step_001140.npz"
REGION_PARENT = "/home/mayank/turtlebot3_ws/mntfields_model"   # contains region_1, region_2, ...

START = np.array([-1.5, -1.8], dtype=np.float32)
GOAL  = np.array([ 2.47,  0.53], dtype=np.float32)

# From your GridmapNode/LidarGridmap init:
ORIGIN   = np.array([0.0, 0.0], dtype=np.float32)  # map center in odom/world
MAP_SIZE = 40.0
N_CELLS  = 400
CELL_SIZE = MAP_SIZE / N_CELLS  # 0.1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# HELPERS: world <-> grid
# =========================
def world_to_grid(x, y):
    half = MAP_SIZE / 2.0
    gx = int((x + half) / CELL_SIZE)
    gy = int((y + half) / CELL_SIZE)
    return gx, gy
    
def in_grid(gx, gy, n):
    return (0 <= gx < n) and (0 <= gy < n)

# =========================
# LOAD SNAPSHOT
# =========================
snap = np.load(SNAPSHOT_PATH)
occ = snap["occupancy"].astype(np.uint8)   # 1=occupied
vis = snap["visited"].astype(np.uint8)     # 1=visited free
n = occ.shape[0]
assert n == N_CELLS, f"Snapshot n={n} but N_CELLS={N_CELLS}. Update N_CELLS or config."

# =========================
# REGION LOADER (checkpoint-based)
# =========================
def load_region(region_dir):
    ckpts = [f for f in os.listdir(region_dir) if f.endswith("_checkpoint.pt")]
    if not ckpts:
        raise RuntimeError(f"No *_checkpoint.pt in {region_dir}")
    ckpt_path = os.path.join(region_dir, ckpts[0])

    ckpt = torch.load(ckpt_path, map_location=DEVICE)

    if "bound" not in ckpt or "B" not in ckpt or "network_state" not in ckpt:
        raise RuntimeError(f"Checkpoint missing required keys: {ckpt.keys()}")

    bound = ckpt["bound"]
    B = ckpt["B"]

    dim = 3
    B_scale = float(torch.norm(B, dim=-1).mean().item())

    model = Model(
        folder=region_dir,
        dim=dim,
        B_scale=B_scale,
        device=str(DEVICE),
        init_network=True,
        eval=True
    )

    # load weights into model.network
    model.network.load_state_dict(ckpt["network_state"])
    model.network.eval()

    region = Region(
        model=model,
        bound=bound,
        init_network=False,
        eval=True
    )

    # CRITICAL: make region.function available to get_next_waypoint()
    region.function = model
    return region

# =========================
# LOAD ALL REGIONS
# =========================
regions = []
for name in sorted(os.listdir(REGION_PARENT)):
    if name.startswith("region_"):
        regions.append(load_region(os.path.join(REGION_PARENT, name)))

print(f"Loaded {len(regions)} regions")

def find_region(pt_xy):
    x, y = float(pt_xy[0]), float(pt_xy[1])
    for r in regions:
        b = r.bound.detach().cpu().numpy()
        if b[0,0] <= x <= b[1,0] and b[0,1] <= y <= b[1,1]:
            return r
    return None

# =========================
# GENERATE NEURAL PATH (multi-region)
# =========================
cur = START.copy()
path = [cur.copy()]

for k in range(600):
    if np.linalg.norm(cur - GOAL) < 0.2:
        break

    r = find_region(cur)
    if r is None:
        print("[STOP] outside all regions at", cur)
        break

    # alpha=1.0 => purely NN direction (except when grad is None/small)
    nxt = r.get_next_waypoint(cur, GOAL, step_size=0.20, alpha=1.0, debug=False)
    if nxt is None:
        print("[STOP] waypoint failed at", cur)
        break

    cur = nxt
    path.append(cur.copy())

path = np.asarray(path, dtype=np.float32)
print("Path points:", len(path))

# =========================
# PLOT OVER OCCUPANCY
# =========================
img = np.zeros((n, n, 3), dtype=np.uint8)
img[:] = [10, 10, 80]          # unknown
img[vis == 1] = [200, 200, 0]  # visited free
img[occ == 1] = [0, 0, 0]      # obstacles

plt.figure(figsize=(9, 9))
plt.imshow(img, origin="lower")

# region bounds
for r in regions:
    b = r.bound.detach().cpu().numpy()
    (gx0, gy0) = world_to_grid(b[0,0], b[0,1])
    (gx1, gy1) = world_to_grid(b[1,0], b[1,1])
    plt.plot([gx0, gx1, gx1, gx0, gx0],
             [gy0, gy0, gy1, gy1, gy0], linewidth=1.2)

# path in grid coords
px, py = [], []
for p in path:
    gx, gy = world_to_grid(p[0], p[1])
    if in_grid(gx, gy, n):
        px.append(gx); py.append(gy)

plt.plot(px, py, "r-", linewidth=2)

sx, sy = world_to_grid(START[0], START[1])
gx, gy = world_to_grid(GOAL[0], GOAL[1])
plt.scatter([sx], [sy], c="lime", s=120, marker="o", label="start")
plt.scatter([gx], [gy], c="cyan", s=120, marker="*", label="goal")

plt.legend()
plt.title("Neural multi-region path over occupancy")
out = "neural_overlay_test.png"
plt.savefig(out, dpi=160, bbox_inches="tight")
plt.close()
print("Saved:", out)