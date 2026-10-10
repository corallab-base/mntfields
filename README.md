# MNTFields TurtleBot bundle

This bundle contains the MNTFields ROS 2 Python package and the checked-in
TurtleBot model data needed for the included navigation model. It is intended
for a TurtleBot3 simulation first; the Jackal notes at the end describe the
changes required before using the same stack on a Jackal.

## Contents

- `src/mntfields`: MNTFields mapping, learning, navigation, sensing, launch,
  configuration, and RViz code.
- `models/mntfields_model`: region checkpoints, map, topology, and room graph
  used by the navigation launch.

The SDF local replanner is included in the package, but its external
`comp-sdf-dyn-nav` repository and SDF checkpoints are not copied here. Run the
standard MNTFields navigation first; enable the SDF replanner only after that
repository and its checkpoints are installed separately.

## TurtleBot3 simulation

Use a ROS 2 Humble environment with the TurtleBot3, Gazebo, Nav2, SLAM Toolbox,
and ROS 2 Python dependencies installed. From the extracted bundle directory:

```bash
source /opt/ros/humble/setup.bash
export TURTLEBOT3_MODEL=waffle_pi
colcon build --symlink-install --packages-select mntfields
source install/setup.bash
ros2 launch mntfields mntfields_mapping.launch.py
```

The mapping launch starts Gazebo, the TurtleBot3, SLAM Toolbox, and the
MNTFields grid-map node. For the included saved model and map, use a second
terminal:

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch mntfields mntfields_navigation.launch.py \
  model_dir:=$PWD/models/mntfields_model \
  map_yaml:=$PWD/models/mntfields_model/map.yaml
```

The launch file currently starts its own simulation processes, so normally run
either mapping or navigation, not both at once. Set `TURTLEBOT3_MODEL=burger`
if using the Burger model instead of the default Waffle Pi.

## Single-field mode without room segmentation

MNTFields can run as one continuously trained field instead of splitting the
environment into room regions. Use `antfields_gridmap_node`, not the
room-segmenting `mapping_node` or `gridmap_node` paths:

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
export TURTLEBOT3_MODEL=waffle_pi
ros2 launch mntfields antfields_sim_mapping.launch.py
```

The single-region implementation is:

- `src/mntfields/mntfields/mntfields_ros/mapping/antfields_gridmap_node.py`
- `src/mntfields/launch/antfields_sim_mapping.launch.py`
- `src/mntfields/launch/antfields_tb4_mapping.launch.py`

These paths do not use room segmentation, room ownership, or room-based region
switching. They create one `Region` and train one field network. Avoid
`mntfields_mapping.launch.py`, `line_mapping.launch.py`, `mapping_node`, and
`gridmap_node` when a single field is desired.

For a larger operating area, edit `antfields_gridmap_node.py` and increase the
single-field bounds:

```python
# Jackal/single-field deployment: cover the complete operating environment.
self.declare_parameter("fixed_region_size_m", 50.0)
self.declare_parameter("expand_region_with_map", True)
```

The default is a 20 m fixed region with expansion disabled. A larger field
requires more GPU memory and may train more slowly as the explored area grows.
For the Jackal, also replace the TurtleBot3 driver/description launch, verify
`/scan`, `/cmd_vel`, `base_link`/`base_footprint`, and localization TF, then
retune lidar height, velocity limits, and footprint-related parameters.

## Real TurtleBot3

Do not start Gazebo. Bring up the robot's drivers, laser scanner, odometry,
localization/SLAM, and TF tree first. The MNTFields node expects a LaserScan
on `/scan`, a usable `map -> base_link` or `map -> base_footprint` transform,
and publishes velocity through `/cmd_vel` in its navigation path.

Pass the real map/model paths explicitly and set `robot_type:=real` on the
mapping node or equivalent launch configuration. Start at low velocity and
verify the scan frame and `base_link` TF in RViz before enabling autonomous
motion.

## Jackal migration notes

This code is not automatically Jackal-ready. The following must be adapted:

1. Replace the TurtleBot3 Gazebo launch, URDF, AMCL parameters, and robot
   model with the Jackal ROS 2 description/driver launch.
2. Confirm the Jackal topics and frames. At minimum, check `/scan`,
   `/cmd_vel`, `base_link`, `base_footprint`, `odom`, and the global localization
   frame. Override `base_frame`, `cmd_vel_topic`, and `lidar_scan_topic` where
   necessary.
3. Re-tune `lidar_height_m`, velocity limits, region radius, path commit
   distance, and obstacle-clearance parameters for the larger Jackal chassis.
4. For SDF replanning, provide a Jackal body SDF representation. The optimizer
   needs a Jackal-specific DeepSDF latent code plus its matching normalization
   JSON; a Gazebo `.sdf` file alone is not sufficient. Replace the current
   TurtleBot latent checkpoint and update the SDF node's hard-coded checkpoint
   path before enabling `use_sdf_planner:=true`.
5. Test in Jackal simulation with conservative speeds, then validate the
   footprint against the physical robot including payload, bumpers, and
   permanently mounted sensors.

The included `mntfields_navigation.launch.py` is deliberately TurtleBot3/Gazebo
specific. Treat it as a reference when writing a Jackal launch file rather than
running it unchanged on the Jackal.
