# Testing Scripts

This directory contains test, experimental, and diagnostic scripts that are not part of the production mntfields system.

## Contents

### mapping/
- **tb4_skew_test_node.py** — Tests TurtleBot4 IMU/odometry skew
- **tb4_mapping_probe_node.py** — Probes the mapping system state on TurtleBot4 hardware
- **offline_map_recorder_node.py** — Records offline map data for replay
- **offline_sweep_node.py** — Runs offline sweep exploration sequences
- **simple_Lidar_mapper.py** — Minimal standalone LiDAR mapper (no neural fields)
- **semantic_mapper_test_node.py** — Tests semantic segmentation integration
- **semantic_mapper_node.py** — Full semantic mapper node (experimental)

### navigation/
- **sdf_goal_nav_node.py** — SDF-based goal navigation (standalone, not integrated)
- **field_training_diagnostics.py** — Visualises training progress and field quality

### scripts/
- **move_straight.py** — Commands robot to drive straight for calibration
- **fix_offline_pgms.py** — Fixes .pgm map files recorded offline
- **offline_teleport_train.py** — Trains neural fields by teleporting robot in sim
- **offline_train_sweep.py** — Trains neural fields via sweep pattern in sim

### launch/
- **mntfields_offline_recording.launch.py** — Launch file for offline map recording
- **mntfields_sdf_straightline.launch.py** — Launch file for SDF straight-line nav test

## How to Run

### Nodes (mapping/ and navigation/ scripts)
1. Copy the script to its original location:
   - mapping scripts → `src/mntfields/mntfields/mntfields_ros/mapping/`
   - navigation scripts → `src/mntfields/mntfields/mntfields_ros/navigation/`
2. Add an entry point to `src/mntfields/setup.py` under `console_scripts`, e.g.:
   ```python
   'my_script = mntfields.mntfields_ros.mapping.my_script:main',
   ```
3. Rebuild: `colcon build --packages-select mntfields --symlink-install && source install/setup.bash`
4. Run: `ros2 run mntfields my_script`

### Launch files
1. Copy the launch file to `src/mntfields/launch/`
2. Run: `ros2 launch mntfields <launch_file_name>.launch.py`

### Standalone scripts (scripts/)
Run directly with Python:
```bash
python3 testing/scripts/<script_name>.py --help
```
