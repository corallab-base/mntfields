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

## Jackal single-field mode (offboard desktop, Docker)

Active single-field exploration on a Clearpath Jackal (ROS 2 Jazzy, 2D lidar).
All computation — SLAM Toolbox, field training, frontier planning, path
following and RViz — runs on a desktop with an NVIDIA GPU inside Docker. The
Jackal only runs its stock Clearpath stack; the two talk over ROS 2 topics only.
Nothing is deployed to the robot.

1. On the robot, read the values the launch needs:

   ```bash
   cat /etc/clearpath/robot.yaml            # namespace, ros2 domain id / rmw / discovery
   ros2 topic list | grep -E "scan|cmd_vel"
   ros2 topic info /<ns>/cmd_vel -v         # TwistStamped -> cmd_vel_stamped:=true
   ```

2. On the desktop (needs Docker + NVIDIA Container Toolkit), from the repo root:

   ```bash
   cp docker/.env.example docker/.env       # set ROS_DOMAIN_ID / RMW / discovery to match the robot
   docker compose -f docker/compose.yaml build
   docker compose -f docker/compose.yaml run --rm mntfields \
     colcon build --symlink-install --packages-select mntfields
   xhost +local:                            # allow RViz to open
   ```

   Keep the desktop and robot clocks in sync (e.g. chrony); scan timestamps
   feed TF and SLAM. Check connectivity with
   `docker compose -f docker/compose.yaml run --rm mntfields ros2 topic hz /<ns>/sensors/lidar2d_0/scan`.

3. Dry run first (velocity goes to a dummy topic, the robot cannot move; the
   start-up spin is disabled because it could never finish), then live:

   ```bash
   docker compose -f docker/compose.yaml run --rm mntfields \
     ros2 launch mntfields antfields_jackal_mapping.launch.py \
       cmd_vel_topic:=/antfields/cmd_vel_dry initial_spin:=false

   docker compose -f docker/compose.yaml run --rm mntfields \
     ros2 launch mntfields antfields_jackal_mapping.launch.py max_lin:=0.15
   ```

   `namespace` defaults to `j100_1639`; pass `namespace:=<ns>` for another
   robot. The launch remaps `/tf`, `/tf_static` and `/scan` onto the Clearpath
   namespace and sends velocity to `/<ns>/cmd_vel` (through the Clearpath
   twist mux, so the joystick still overrides). Robot-size parameters
   (`wall_min_m`, `wall_max_m`, `collision_pad_cells`, ...) default to
   Jackal values; see `--show-args`. The trained field is saved under
   `~/mntfields_models/antfields_jackal/region_1/` on the desktop.

   On this Jackal the Hokuyo faces the **rear** (the power-button side), so
   `/etc/clearpath/robot.yaml` on the robot must give the lidar `rpy: [0.0,
   0.0, 3.1415]` (and its real mount position); otherwise every scan is placed
   180° off. The launch drives rear-first (`drive_reverse:=true`) so the lidar
   leads; set `drive_reverse:=false` for a forward-facing lidar.

   The Hokuyo UST covers ±135°, so a 90° wedge on the side opposite the lidar
   is never scanned. The Jackal launch therefore enables, by default: a 360° spin before
   exploring (`initial_spin`), path validation that only drives through
   SLAM-mapped free space (`enable_path_validation`, `trim_at_unknown`,
   `collision_radius_m`, `collision_check_skip_m`), and a live-scan forward
   safety stop (`safety_stop_enabled`, `robot_front_m`, `robot_half_width_m`,
   `safety_stop_dist_m`, `safety_slow_dist_m`).

   **Controllers and safety** (launch arguments):

   - `controller:=field` (default): steer along the learned field's gradient every
     control tick toward a point ~0.6 m ahead on the SLAM-map route; no
     plan-validate-follow pauses. Where the field disagrees with the route by
     more than 60 degrees (e.g. a passage it has not learned), the route wins.
   - `controller:=nav2`: Nav2 MPPI (`config/nav2_jackal_controller.yaml`) follows
     the NTFields field path (or the map route as fallback).
   - `controller:=path`: the original plan/validate/follow pieces.
   - `safety_mode:=scan` (default): every command is checked by predicting the
     true footprint along its arc against live lidar points (+5 s memory) and the
     SLAM map; it brakes before contact, creeps with a 1 cm margin in tight spots,
     and slows down by side clearance. `safety_mode:=grid` restores the original
     SLAM-grid guards.

   Logs of every run are kept in `~/mntfields_logs` (node log:
   `python3_<pid>_<time>.log`); `[CTRL]` lines say which check changed a command.

   RViz opens with `rviz/antfields_jackal.rviz`: SLAM map, scan, Jackal model,
   field path, frontier/goal markers and the learned travel-time field from the
   robot's current position (`/mntfields/travel_time`, a PointCloud2 whose
   intensity is travel time; `/mntfields/speed` holds the speed field and is
   off by default). Tune with `field_vis_radius_m`, `field_vis_resolution_m`
   and `field_vis_free_only:=false` to show the field outside mapped free space.

### Simulating the Jackal

Both simulators reproduce this Jackal's interfaces (namespaced topics and TF,
`TwistStamped` cmd_vel, rear-mounted Hokuyo facing backwards) and run on an
isolated, localhost-only ROS domain, so they can never reach the real robot.

**Gazebo (Clearpath simulator, with GUI and RViz).** `sim/robot.yaml` is a copy
of the robot's config under the namespace `j100_sim`; `sim/worlds/obstacle_room.sdf`
is an 8 x 8 m room with a 1.5 x 1.5 m box in the centre.

```bash
docker compose -f docker/compose.yaml build sim          # once: adds Gazebo Harmonic
xhost +local:
# terminal 1: Gazebo + the simulated Jackal (optional args: x y yaw)
docker compose -f docker/compose.yaml run --rm sim /ws/sim/run_gazebo.sh
# terminal 2: SLAM + antfields + RViz against the simulated robot
docker compose -f docker/compose.yaml run --rm sim \
  ros2 launch mntfields antfields_jackal_mapping.launch.py \
    namespace:=j100_sim use_sim_time:=true
```

**The real maze.** `sim/map_to_world.py` turns a SLAM map of the real maze into
both a Gazebo world and a 2D-simulator map, so its actual passages can be tested:

```bash
# with a run active (reads /map only); writes sim/worlds/real_maze.sdf and sim/maps/real_maze.npz
docker compose -f docker/compose.yaml run --rm mntfields python3 /ws/sim/map_to_world.py --grab --name real_maze
# Gazebo in the real maze (args: x y yaw world)
docker compose -f docker/compose.yaml run --rm sim /ws/sim/run_gazebo.sh 0.10 1.02 0.63 real_maze
```

**Headless 2D simulator** (`src/mntfields/scripts/sim_jackal_2d.py`): no GUI,
starts in seconds and logs collisions of the robot footprint; useful for quick
debugging. Run it in the same container as the launch, on its own domain:

```bash
docker compose -f docker/compose.yaml run --rm \
  -e ROS_DOMAIN_ID=42 -e ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST mntfields bash -c '
  python3 /ws/src/mntfields/scripts/sim_jackal_2d.py --ns sim &   # add --map /ws/sim/maps/real_maze.npz
  ros2 launch mntfields antfields_jackal_mapping.launch.py namespace:=sim'
```

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
