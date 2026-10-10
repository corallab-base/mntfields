"""
tb4_mapping.launch.py

Launches SLAM Toolbox + tb4_node for real-robot mapping.

Usage:
    ros2 launch mntfields tb4_mapping.launch.py

Run RViz separately:
    ros2 run rviz2 rviz2   (set Fixed Frame to "map")
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
        get_package_share_directory('mntfields'), 'config', 'slam_toolbox_tb4_params.yaml')

    slam_toolbox_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('slam_toolbox'), 'launch',
                         'online_sync_launch.py')),
        launch_arguments={
            'slam_params_file': slam_params,
            'use_sim_time': 'false',
        }.items())

    tb4_node = Node(
        package='mntfields',
        executable='tb4_node',
        name='mntfields_gridmap',
        output='screen',
    )

    semantic_detector = Node(
        package='mntfields',
        executable='semantic_detector_node',
        name='semantic_detector',
        output='screen',
        parameters=[{
            'detection_rate_hz': 2.0,
            'score_threshold': 0.45,
            'min_lidar_pts_in_box': 2,
            'confirm_threshold': 5,
            'map_frame': 'map',
        }],
    )

    return LaunchDescription([
        slam_toolbox_cmd,
        tb4_node,
        semantic_detector,
    ])
