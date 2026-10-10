"""
antfields_tb4_mapping.launch.py

Launch SLAM Toolbox plus the single-region AntFields-style exploration node on
the real TurtleBot.

Usage:
    ros2 launch mntfields antfields_tb4_mapping.launch.py
    ros2 launch mntfields antfields_tb4_mapping.launch.py train_epoch:=30 min_train_epochs_for_nav:=80
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    slam_params = os.path.join(
        get_package_share_directory("mntfields"),
        "config",
        "slam_toolbox_tb4_params.yaml",
    )

    use_sim_time = LaunchConfiguration("use_sim_time")
    train_timer_sec = LaunchConfiguration("train_timer_sec")
    train_samples = LaunchConfiguration("train_samples")
    train_epoch = LaunchConfiguration("train_epoch")
    min_train_epochs_for_nav = LaunchConfiguration("min_train_epochs_for_nav")
    frontier_max_goal_dist_m = LaunchConfiguration("frontier_max_goal_dist_m")
    enable_region_debug_vis = LaunchConfiguration("enable_region_debug_vis")

    slam_toolbox_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("slam_toolbox"),
                "launch",
                "online_sync_launch.py",
            )
        ),
        launch_arguments={
            "slam_params_file": slam_params,
            "use_sim_time": use_sim_time,
        }.items(),
    )

    antfields_node = Node(
        package="mntfields",
        executable="antfields_gridmap_node",
        name="antfields_gridmap",
        output="screen",
        parameters=[
            {
                "use_sim_time": use_sim_time,
                "train_timer_sec": train_timer_sec,
                "train_samples": train_samples,
                "train_epoch": train_epoch,
                "min_train_epochs_for_nav": min_train_epochs_for_nav,
                "frontier_max_goal_dist_m": frontier_max_goal_dist_m,
                "enable_region_debug_vis": enable_region_debug_vis,
            }
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="false"),
            DeclareLaunchArgument("train_timer_sec", default_value="0.4"),
            DeclareLaunchArgument("train_samples", default_value="12000"),
            DeclareLaunchArgument("train_epoch", default_value="20"),
            DeclareLaunchArgument("min_train_epochs_for_nav", default_value="60"),
            DeclareLaunchArgument("frontier_max_goal_dist_m", default_value="6.0"),
            DeclareLaunchArgument("enable_region_debug_vis", default_value="true"),
            slam_toolbox_cmd,
            antfields_node,
        ]
    )
