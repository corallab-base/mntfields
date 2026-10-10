"""
tb4_line_mapping.launch.py

Launches SLAM Toolbox + mapping_node for real-TurtleBot4 line-based mapping.

Usage:
    ros2 launch mntfields tb4_line_mapping.launch.py

Run RViz separately:
    rviz2 -d ~/turtlebot3_ws/mapping_real.rviz
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
    slam_params = os.path.join(
        get_package_share_directory("mntfields"),
        "config",
        "slam_toolbox_tb4_params.yaml",
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

    mapping_node = Node(
        package="mntfields",
        executable="mapping_node",
        name="mntfields_mapping",
        output="screen",
        parameters=[{
            "use_sim_time": use_sim_time,
            "robot_type": "real",
            "lidar_max_range": 6.0,
            "train_timer_sec": 1.0,    # 0.5s default → 1.0s: halves GPU load, prevents LIDAR TF drops
            "train_epoch": 20,         # 30 default → 20: lighter per-tick GPU burst
        }],
    )

    camera_tf = Node(
        package='tf2_ros', executable='static_transform_publisher',
        name='oakd_camera_tf',
        arguments=[
            '--x', '0.0', '--y', '0.0', '--z', '0.0',
            '--roll', '0.0', '--pitch', '0.0', '--yaw', '0.0',
            '--frame-id', 'oakd_pro_link',
            '--child-frame-id', 'camera_link',
        ])

    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        slam_toolbox_cmd,
        mapping_node,
        camera_tf,
    ])
