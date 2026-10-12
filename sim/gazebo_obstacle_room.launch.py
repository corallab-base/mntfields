"""
Clearpath Gazebo simulation of the Jackal in a world from sim/worlds/
(obstacle_room by default; real_maze is generated from the real SLAM map by
sim/map_to_world.py).

clearpath_gz's simulation.launch.py only accepts its bundled worlds, so this
includes its two parts directly: gz_sim.launch.py loads "<world>.sdf" (an
absolute path works) and robot_spawn.launch.py spawns into the world by name.

Usage (via sim/run_gazebo.sh):
    ros2 launch /ws/sim/gazebo_obstacle_room.launch.py setup_path:=/tmp/clearpath_sim/ world:=real_maze
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

WORLDS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "worlds")


def _setup(context):
    launch_dir = os.path.join(get_package_share_directory("clearpath_gz"), "launch")
    setup_path = LaunchConfiguration("setup_path")
    world_name = LaunchConfiguration("world").perform(context)
    world_file_no_ext = os.path.join(WORLDS_DIR, world_name)

    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(launch_dir, "gz_sim.launch.py")),
        launch_arguments=[("world", world_file_no_ext), ("setup_path", setup_path)],
    )
    robot_spawn = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(launch_dir, "robot_spawn.launch.py")),
        launch_arguments=[
            ("use_sim_time", "true"),
            ("setup_path", setup_path),
            ("world", world_name),
            ("rviz", "false"),
            ("x", LaunchConfiguration("x")),
            ("y", LaunchConfiguration("y")),
            ("z", "0.3"),
            ("yaw", LaunchConfiguration("yaw")),
        ],
    )
    return [gz_sim, robot_spawn]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("setup_path", default_value="/tmp/clearpath_sim/"),
        DeclareLaunchArgument("world", default_value="obstacle_room",
                              description="world name in sim/worlds (without .sdf)"),
        DeclareLaunchArgument("x", default_value="-2.5"),
        DeclareLaunchArgument("y", default_value="-2.5"),
        DeclareLaunchArgument("yaw", default_value="0.0"),
        OpaqueFunction(function=_setup),
    ])
