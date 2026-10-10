# test_cross_region_path.py

import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from mntfields.mntfields_core.core.region import Region
from mntfields.mntfields_core.core.model_base import Model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

REGION1_PATH = "/home/mayank/turtlebot3_ws/mntfields_model/region_1"
REGION2_PATH = "/home/mayank/turtlebot3_ws/mntfields_model/region_2"

START = np.array([-1.5, -1.8], dtype=np.float32)
GOAL  = np.array([2.57, 0.68], dtype=np.float32)


# ------------------------------------------------
# PROPER LOADER BASED ON YOUR CHECKPOINT STRUCTURE
# ------------------------------------------------
def load_region(region_path):

    print(f"\nLoading region from: {region_path}")

    # Find checkpoint
    ckpt_file = None
    for f in os.listdir(region_path):
        if f.endswith("_checkpoint.pt"):
            ckpt_file = os.path.join(region_path, f)
            break

    if ckpt_file is None:
        raise RuntimeError("Checkpoint file not found")

    ckpt = torch.load(ckpt_file, map_location=device)

    bound = ckpt["bound"]
    B = ckpt["B"]

    dim = 3
    B_scale = float(torch.norm(B, dim=-1).mean().item())

    print("  Bound:\n", bound)

    # IMPORTANT: init_network=True so model builds its internal net
    model = Model(
        folder=region_path,
        dim=dim,
        B_scale=B_scale,
        device=str(device),
        init_network=True,
        eval=True
    )

    # Load weights into the model directly
    model.network.load_state_dict(ckpt["network_state"])

    region = Region(
        model=model,
        bound=bound,
        init_network=False,
        eval=True
    )
    region.function = model  # <-- CRITICAL LINE
    region.epoch = ckpt["epoch"]

    print("  Loaded region version:", region.version)

    return region


region1 = load_region(REGION1_PATH)
region2 = load_region(REGION2_PATH)


# ------------------------------------------------
# DOOR ESTIMATION
# ------------------------------------------------
def compute_overlap_midpoint(r1, r2):
    b1 = r1.bound.detach().cpu().numpy()
    b2 = r2.bound.detach().cpu().numpy()

    xmin = max(b1[0,0], b2[0,0])
    xmax = min(b1[1,0], b2[1,0])
    ymin = max(b1[0,1], b2[0,1])
    ymax = min(b1[1,1], b2[1,1])

    if xmin < xmax and ymin < ymax:
        return np.array([(xmin+xmax)/2, (ymin+ymax)/2], dtype=np.float32)

    return None


#DOOR = compute_overlap_midpoint(region1, region2)
DOOR = None

if DOOR is None:
    print("No overlap found — using manual door")
    DOOR = np.array([-1.21, -0.39], dtype=np.float32)

print("Door:", DOOR)

print("START in R1:", region1.in_bound(START))
print("GOAL in R2:", region2.in_bound(GOAL))


# ------------------------------------------------
# GENERATE PATH
# ------------------------------------------------
print("\nGenerating path R1 → DOOR")
path1 = region1.generate_field_path(
    START,
    DOOR,
    step_size=0.10,
    max_steps=120
)

if path1 is None:
    raise RuntimeError("Region 1 failed")

path1 = np.asarray(path1, dtype=np.float32)


print("Generating path DOOR → GOAL (R2)")
path2 = region2.generate_field_path(
    DOOR,
    GOAL,
    step_size=0.10,
    max_steps=120
)

if path2 is None:
    raise RuntimeError("Region 2 failed")

path2 = np.asarray(path2, dtype=np.float32)


full_path = np.vstack([path1, path2])

print("Full path length:", len(full_path))


# ==============================================================
# VISUALIZE (Headless-safe)
# ==============================================================

plt.figure(figsize=(8, 6))

# Plot path
plt.plot(full_path[:, 0], full_path[:, 1], '-b', linewidth=2, label="Trajectory")

# Plot regions bounds
def draw_bound(region, color):
    b = region.bound.detach().cpu().numpy()
    rect = plt.Rectangle(
        (b[0, 0], b[0, 1]),
        b[1, 0] - b[0, 0],
        b[1, 1] - b[0, 1],
        linewidth=2,
        edgecolor=color,
        facecolor='none'
    )
    plt.gca().add_patch(rect)

draw_bound(region1, "green")
draw_bound(region2, "orange")

# Mark start / goal / door
plt.scatter(START[0], START[1], c='green', s=120, label="Start")
plt.scatter(GOAL[0], GOAL[1], c='red', s=120, label="Goal")
plt.scatter(DOOR[0], DOOR[1], c='cyan', s=120, label="Door")

plt.axis("equal")
plt.legend()
plt.title("Cross-region trajectory test")

out_path = "cross_region_path.png"
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"\n✅ Saved visualization to: {out_path}")