"""
watershed_test.launch.py

Minimal launch for testing watershed room segmentation:
  SLAM Toolbox (online_sync) + keyboard teleop + watershed_room_node

No mapping_node, no nav_node. Drive around with teleop and watch rooms
appear in RViz as the map fills in.

Usage (simulation):
    ros2 launch mntfields watershed_test.launch.py

Usage (real TB4):
    ros2 launch mntfields watershed_test.launch.py use_sim_time:=false robot_type:=real

RViz:
    Add the following displays:
      - Map          → topic: /map
      - OccupancyGrid → topic: /watershed/room_grid        (see room basins)
      - OccupancyGrid → topic: /watershed/distance_field   (see clearance)
      - MarkerArray  → topic: /watershed/rooms             (boxes + labels)
      - MarkerArray  → topic: /watershed/doorways          (doorway cylinders)
      - MarkerArray  → topic: /watershed/peaks             (room seed spheres)
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")
    robot_type   = LaunchConfiguration("robot_type")

    # Pick slam params based on robot type
    # (both use online_sync; the only differences are scan topic + ranges)
    slam_params_sim  = os.path.join(
        get_package_share_directory("mntfields"), "config",
        "slam_toolbox_params.yaml")
    slam_params_real = os.path.join(
        get_package_share_directory("mntfields"), "config",
        "slam_toolbox_tb4_params.yaml")

    # Use real params when robot_type=real, sim params otherwise.
    # LaunchConfiguration can't branch at launch time without conditions —
    # pass the sim params by default and note that for real TB4 you should
    # pass slam_params:= explicitly, or just override robot_type:=real.
    slam_params = LaunchConfiguration("slam_params")

    slam_toolbox = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("slam_toolbox"),
                "launch", "online_sync_launch.py",
            )
        ),
        launch_arguments={
            "slam_params_file": slam_params,
            "use_sim_time":     use_sim_time,
        }.items(),
    )

    watershed_node = Node(
        package="mntfields",
        executable="watershed_room_node",
        name="watershed_room_node",
        output="screen",
        parameters=[{
            "use_sim_time":           use_sim_time,
            "min_peak_clearance_m":   0.40,
            "min_peak_separation_m":  1.5,
            "min_room_area_m2":       0.8,
            "door_min_m":             0.55,
            "door_max_m":             3.50,
            "update_interval_sec":    1.0,
        }],
    )

    # Keyboard teleop (turtlebot3 — works for both waffle_pi and TB4 via /cmd_vel)
    teleop_node = Node(
        package="turtlebot3_teleop",
        executable="teleop_keyboard",
        name="teleop_keyboard",
        output="screen",
        prefix="xterm -e",   # open in its own terminal window
    )

    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("robot_type",   default_value="sim"),
        DeclareLaunchArgument("slam_params",  default_value=slam_params_sim),
        slam_toolbox,
        watershed_node,
        teleop_node,
    ])
