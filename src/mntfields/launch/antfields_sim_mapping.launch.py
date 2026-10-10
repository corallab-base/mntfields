"""
antfields_sim_mapping.launch.py

Launch Gazebo + TurtleBot3 + SLAM Toolbox + the single-region AntFields-style
exploration node for simulator testing.

Usage:
    ros2 launch mntfields antfields_sim_mapping.launch.py
    ros2 launch mntfields antfields_sim_mapping.launch.py x_pose:=-1.5 y_pose:=-1.8 train_epoch:=30
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    launch_file_dir = os.path.join(
        get_package_share_directory("turtlebot3_gazebo"), "launch"
    )
    pkg_gazebo_ros = get_package_share_directory("gazebo_ros")
    world = os.path.join(
        get_package_share_directory("turtlebot3_gazebo"),
        "worlds",
        "house_main.world",
    )

    use_sim_time = LaunchConfiguration("use_sim_time")
    x_pose = LaunchConfiguration("x_pose")
    y_pose = LaunchConfiguration("y_pose")
    train_timer_sec = LaunchConfiguration("train_timer_sec")
    train_samples = LaunchConfiguration("train_samples")
    train_epoch = LaunchConfiguration("train_epoch")
    min_train_epochs_for_nav = LaunchConfiguration("min_train_epochs_for_nav")
    frontier_max_goal_dist_m = LaunchConfiguration("frontier_max_goal_dist_m")
    enable_region_debug_vis = LaunchConfiguration("enable_region_debug_vis")

    slam_params = os.path.join(
        get_package_share_directory("mntfields"),
        "config",
        "slam_toolbox_params.yaml",
    )

    gzserver_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, "launch", "gzserver.launch.py")
        ),
        launch_arguments={"world": world}.items(),
    )

    gzclient_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, "launch", "gzclient.launch.py")
        )
    )

    robot_state_publisher_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(launch_file_dir, "robot_state_publisher.launch.py")
        ),
        launch_arguments={"use_sim_time": use_sim_time}.items(),
    )

    spawn_turtlebot_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(launch_file_dir, "spawn_turtlebot3.launch.py")
        ),
        launch_arguments={"x_pose": x_pose, "y_pose": y_pose}.items(),
    )

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
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument("x_pose", default_value="-1.5"),
            DeclareLaunchArgument("y_pose", default_value="-1.8"),
            DeclareLaunchArgument("train_timer_sec", default_value="0.4"),
            DeclareLaunchArgument("train_samples", default_value="12000"),
            DeclareLaunchArgument("train_epoch", default_value="20"),
            DeclareLaunchArgument("min_train_epochs_for_nav", default_value="60"),
            DeclareLaunchArgument("frontier_max_goal_dist_m", default_value="6.0"),
            DeclareLaunchArgument("enable_region_debug_vis", default_value="true"),
            gzserver_cmd,
            gzclient_cmd,
            robot_state_publisher_cmd,
            spawn_turtlebot_cmd,
            slam_toolbox_cmd,
            antfields_node,
        ]
    )
