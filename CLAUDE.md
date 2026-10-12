# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

A ROS 2 colcon workspace (originally Humble/TurtleBot; the Jackal single-field path targets Jazzy via Docker) containing one `ament_python` package, `src/mntfields`: neural metric-field (NTField-style) mapping and navigation for TurtleBot3/TurtleBot4 in Gazebo. The robot explores, trains a small PyTorch time-field network per region (room), saves the regions to disk, then plans paths through the trained fields at navigation time. `models/mntfields_model/` holds a pre-trained model + map (git-ignored).

## Build and run

```bash
source /opt/ros/humble/setup.bash
export TURTLEBOT3_MODEL=waffle_pi          # or burger
colcon build --symlink-install --packages-select mntfields
source install/setup.bash

# Mapping (Gazebo + TB3 + SLAM Toolbox + gridmap_node)
ros2 launch mntfields mntfields_mapping.launch.py

# Navigation with the saved model (starts its own Gazebo/map_server/AMCL — don't run alongside mapping)
ros2 launch mntfields mntfields_navigation.launch.py \
  model_dir:=$PWD/models/mntfields_model map_yaml:=$PWD/models/mntfields_model/map.yaml

# Single node
ros2 run mntfields <entry_point> --ros-args -p key:=value
```

### Jackal (single field, ROS 2 Jazzy) — runs in Docker on the desktop

There is no ROS on the host; everything runs in the `docker/` image (CUDA 12.8 + Jazzy + torch + torch_kdtree). The Jackal only runs its stock Clearpath stack and talks to the desktop over ROS topics — **never deploy code or containers to the robot**.

```bash
cp docker/.env.example docker/.env     # ROS_DOMAIN_ID / RMW / discovery must match robot.yaml
docker compose -f docker/compose.yaml build
docker compose -f docker/compose.yaml run --rm mntfields colcon build --symlink-install --packages-select mntfields
docker compose -f docker/compose.yaml run --rm mntfields \
  ros2 launch mntfields antfields_jackal_mapping.launch.py cmd_vel_topic:=/antfields/cmd_vel_dry   # dry run
```

`antfields_jackal_mapping.launch.py` remaps `/tf`, `/tf_static`, `/scan` onto the Clearpath namespace (`/<ns>/...`) for SLAM Toolbox, the antfields node and RViz, and holds the Jackal defaults (namespace `j100_1639`, `TwistStamped` cmd_vel, Hokuyo UST on `/<ns>/sensors/lidar2d_0/scan`, robot-size params). `lidar_height_m` only matters for PointCloud2 input; LaserScan point height comes from TF. Robot clearance in the field is `wall_offset_m` (distances measured from walls inflated by it), not a larger `wall_min_m`: the slowest field speed is `wall_min_m/wall_max_m`, and raising it makes walls cheap to cut through (paths through walls, "field produced no path"). Simulate in `sim/` (Gazebo, `sim` compose service on an isolated domain) or with `src/mntfields/scripts/sim_jackal_2d.py` (headless; `--map` takes a real SLAM map exported by `sim/map_to_world.py`). Models are saved to `~/mntfields_models/` on the host.

### Notes

- There is no test suite, linter config, or pytest setup. `mntfields_ros/mapping/test_*.py` are ad-hoc scripts with hard-coded `/home/mayank/...` paths, not unit tests.
- **New nodes and launch/config/rviz files must be registered in `setup.py`** (`console_scripts` / explicit `data_files` lists) and rebuilt, or `ros2 run`/`ros2 launch` won't find them. `--symlink-install` makes Python source edits live without a rebuild, but not newly added files.
- `setup.py` registers `gridmap_node_simp`, whose module does not exist; `ros2 run mntfields gridmap_node_simp` will fail.
- Runtime Python deps (torch, torch_kdtree, open3d, scipy, cv2, networkx, matplotlib) are not declared in `package.xml`; they must be installed in the environment.

## Architecture

**Core (ROS-free) — `mntfields/mntfields_core/`**
- `ntrl/model_network_metric.py` (the NN) and `ntrl/model_function_metric.py` (loss/travel-time functions) define the neural field.
- `core/model_base.py` `Model` wraps network + function; `core/region.py` `Region` *holds* a `Model` (composition, not inheritance) and owns one bounded area: `train(...)`, `generate_field_path(start, goal)`, `save_model`/`load_from_disk`. `region_original.py` is the older version kept for reference.
- `mapping/` — room segmentation variants: line-segment (`line_segments.py`, `rooms_doors.py`), watershed, `room_segmenter.py`, plus `room_graph.py` and `lidar_gridmap.py`.

**ROS nodes — `mntfields/mntfields_ros/`** (entry-point name → module mapping lives in `setup.py`; names don't always match files, e.g. `nav_node` → `navigation_node.py`, `line_nav_node` → `nav_node.py`)

Mapping nodes are parallel alternatives, not layers:
- `gridmap_node` — used by `mntfields_mapping.launch.py`; SLAM `/map` + `TopologyExtractor` (Voronoi-style) region spawning.
- `mapping_node` — used by `line_mapping.launch.py`; same loop but with line-segment room segmentation. One active region on GPU at a time; leaving a room saves/unloads its region, re-entering reloads it.
- `antfields_gridmap_node` — single-field mode: one `Region`, no room segmentation (`antfields_*_mapping.launch.py`). Field size via `fixed_region_size_m` / `expand_region_with_map` params. Topics, frames, model dir, lidar range, `cmd_vel_stamped` (TwistStamped for Clearpath Jazzy) and the robot-clearance band (`wall_min_m`/`wall_max_m` → `Region.minimum`/`maximum`) are parameters whose defaults reproduce the TurtleBot behaviour.
- `gridmap_node_original.py` is legacy.

Navigation:
- `nav_node` (`navigation_node.py`) — loads `model_dir/region_*/` and `topology.json` (cross-region routing), takes RViz 2D Goal Pose, plans via `Region.generate_field_path`, publishes `/nav_path`, drives via `/cmd_vel`. State machine `WAITING_FOR_POSE → IDLE ↔ PLANNING → READY → DRIVING`.
- `line_nav_node` (`nav_node.py`) — counterpart for line-mapping output (per-room regions + room graph), also retrains rooms on semantic changes.
- `sdf_replanner_node` — optional (`use_sdf_planner:=true`): refines `/nav_path` into `/sdf_local_path`; nav falls back to the global path if none arrives within 2 s. Requires the external `comp-sdf-dyn-nav` repo (not bundled); its path is hard-coded to `/home/mayank/comp-sdf-dyn-nav` in both the node default (`sdf_repo_dir`) and `mntfields_navigation.launch.py`, with checkpoints expected under `<repo>/ckpts/sdf-object/`.

`tb4_*` launch files/nodes are the TurtleBot4 variants; `sensing/` holds lidar adapters and semantic segmentation/detection nodes.

**Saved model layout (`model_dir`)**: `region_N/{region_N_network.pt, region_N_B.pt, region_N_data.pth, region_N_meta.json}` plus `topology.json`, `room_graph.json`, `room_region_map.json`, `map.yaml/pgm`.

**Other dirs**: `scripts/` — offline retraining/metrics utilities run with plain `python3`. `testing/` — experimental nodes not installed by `setup.py`; per `testing/README.md`, to run one, copy it into `mntfields_ros/<mapping|navigation>/`, add a `console_scripts` entry, rebuild. `TRAINING_PROFILES.txt` records named training-parameter sets (`train_samples`, `train_epoch`, `max_batches`, ...).

## Robot/frames assumptions

Nodes expect `LaserScan` on `/scan`, a `map → base_link`/`base_footprint` TF, and publish `/cmd_vel`. For real robots pass `robot_type:=real` and don't start Gazebo. The navigation launch is deliberately TurtleBot3/Gazebo-specific; see README "Jackal migration notes" before adapting to other platforms (frames/topics overrides: `base_frame`, `cmd_vel_topic`, `lidar_scan_topic`).
