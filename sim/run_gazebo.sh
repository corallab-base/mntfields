#!/bin/bash
# Start Clearpath's Gazebo simulation of the Jackal in the obstacle room.
# Run through the "sim" compose service (isolated ROS domain, localhost only):
#   docker compose -f docker/compose.yaml run --rm sim /ws/sim/run_gazebo.sh [x y yaw [world]]
#   e.g. the real maze (from sim/map_to_world.py): run_gazebo.sh 0.10 1.02 0.63 real_maze
set -e
X=${1:--2.5}; Y=${2:--2.5}; YAW=${3:-0.0}; WORLD=${4:-obstacle_room}
SIM_DIR=$(cd "$(dirname "$0")" && pwd)
# The Clearpath generator writes its output next to robot.yaml; keep that out of the repo.
SETUP=/tmp/clearpath_sim/
mkdir -p "$SETUP"
cp "$SIM_DIR/robot.yaml" "$SETUP/robot.yaml"
exec ros2 launch "$SIM_DIR/gazebo_obstacle_room.launch.py" \
    setup_path:=$SETUP x:=$X y:=$Y yaw:=$YAW world:=$WORLD
