"""
tb4_astar_sdf_navigation.launch.py

Lightweight TurtleBot4 demo stack:
  - map_server + AMCL for localization on a saved 2D LiDAR map
  - astar_sdf_nav_node for inflated occupancy-grid A* global planning
  - sdf_replanner_node for local SDF trajectory optimization

Workflow:
  1. Launch this file.
  2. In RViz, set the initial pose with "2D Pose Estimate".
  3. Set a goal with "2D Goal Pose".
  4. The robot follows the A* global path, locally refined by the SDF replanner.
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import EnvironmentVariable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def _expand_path(raw: str) -> str:
    return os.path.abspath(os.path.expandvars(os.path.expanduser(raw.strip())))


def _nav_setup(context):
    resolved_model_dir = _expand_path(LaunchConfiguration("model_dir").perform(context))
    resolved_map = _expand_path(LaunchConfiguration("map").perform(context))
    resolved_use_sim_time = LaunchConfiguration("use_sim_time")
    resolved_max_lin = LaunchConfiguration("max_lin")
    resolved_max_ang = LaunchConfiguration("max_ang")
    resolved_inflation_cells = LaunchConfiguration("inflation_cells")
    resolved_global_path_spacing = LaunchConfiguration("global_path_spacing")
    resolved_smoothing_window = LaunchConfiguration("smoothing_window")
    resolved_launch_rviz = LaunchConfiguration("launch_rviz")

    map_server_node = Node(
        package="nav2_map_server",
        executable="map_server",
        name="map_server",
        output="screen",
        parameters=[{
            "use_sim_time": resolved_use_sim_time,
            "yaml_filename": resolved_map,
        }],
    )

    amcl_node = Node(
        package="nav2_amcl",
        executable="amcl",
        name="amcl",
        output="screen",
        parameters=[{
            "use_sim_time": resolved_use_sim_time,
            "min_particles": 500,
            "max_particles": 3000,
            "robot_model_type": "nav2_amcl::DifferentialMotionModel",
            "alpha1": 0.2,
            "alpha2": 0.2,
            "alpha3": 0.2,
            "alpha4": 0.2,
            "laser_model_type": "likelihood_field",
            "laser_max_range": 12.0,
            "laser_min_range": 0.15,
            "max_beams": 60,
            "sigma_hit": 0.2,
            "z_hit": 0.5,
            "z_rand": 0.5,
            "base_frame_id": "base_link",
            "odom_frame_id": "odom",
            "global_frame_id": "map",
            "scan_topic": "/scan",
            "update_min_d": 0.1,
            "update_min_a": 0.2,
            "resample_interval": 2,
            "recovery_alpha_slow": 0.0,
            "recovery_alpha_fast": 0.0,
            "tf_broadcast": True,
        }],
    )

    lifecycle_manager = Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name="lifecycle_manager_localization",
        output="screen",
        parameters=[{
            "use_sim_time": resolved_use_sim_time,
            "autostart": True,
            "node_names": ["map_server", "amcl"],
        }],
    )

    nav_node = Node(
        package="mntfields",
        executable="astar_sdf_nav_node",
        name="astar_sdf_nav",
        output="screen",
        parameters=[{
            "use_sim_time": resolved_use_sim_time,
            "max_lin": resolved_max_lin,
            "max_ang": resolved_max_ang,
            "global_frame": "map",
            "inflation_cells": resolved_inflation_cells,
            "global_path_spacing": resolved_global_path_spacing,
            "smoothing_window": resolved_smoothing_window,
        }],
    )

    sdf_replanner_node = Node(
        package="mntfields",
        executable="sdf_replanner_node",
        name="sdf_local_replanner",
        output="screen",
        parameters=[{
            "use_sim_time": resolved_use_sim_time,
            "metrics_path": os.path.join(resolved_model_dir, "sdf_replanner_metrics.json"),
            "global_frame": "map",
            "base_frame": "base_link",
        }],
    )

    camera_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="oakd_camera_tf",
        arguments=[
            "--x", "0.0", "--y", "0.0", "--z", "0.0",
            "--roll", "0.0", "--pitch", "0.0", "--yaw", "0.0",
            "--frame-id", "oakd_pro_link",
            "--child-frame-id", "camera_link",
        ],
    )

    rviz_cfg = os.path.join(
        get_package_share_directory("mntfields"),
        "rviz",
        "tb4_astar_sdf_navigation.rviz",
    )
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2_tb4_astar_sdf_navigation",
        arguments=["-d", rviz_cfg],
        output="screen",
        condition=IfCondition(resolved_launch_rviz),
    )

    return [
        map_server_node,
        amcl_node,
        lifecycle_manager,
        camera_tf,
        nav_node,
        sdf_replanner_node,
        rviz_node,
    ]


def generate_launch_description():
    model_dir = LaunchConfiguration("model_dir")
    return LaunchDescription([
        DeclareLaunchArgument(
            "model_dir",
            default_value=PathJoinSubstitution([
                EnvironmentVariable("HOME"),
                "turtlebot3_ws",
                "mntfields_model",
            ]),
        ),
        DeclareLaunchArgument(
            "map",
            default_value=PathJoinSubstitution([model_dir, "map.yaml"]),
        ),
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        DeclareLaunchArgument("max_lin", default_value="0.12"),
        DeclareLaunchArgument("max_ang", default_value="0.6"),
        DeclareLaunchArgument("inflation_cells", default_value="2"),
        DeclareLaunchArgument("global_path_spacing", default_value="0.10"),
        DeclareLaunchArgument("smoothing_window", default_value="7"),
        DeclareLaunchArgument("launch_rviz", default_value="false"),
        OpaqueFunction(function=_nav_setup),
    ])
