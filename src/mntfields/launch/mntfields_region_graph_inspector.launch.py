#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    map_yaml = LaunchConfiguration("map_yaml")
    model_dir = LaunchConfiguration("model_dir")
    step_size = LaunchConfiguration("step_size")
    max_steps = LaunchConfiguration("max_steps")
    adjacency_gap_m = LaunchConfiguration("adjacency_gap_m")
    max_neighbors_per_region = LaunchConfiguration("max_neighbors_per_region")
    inflation_cells = LaunchConfiguration("inflation_cells")
    region_bound_margin_m = LaunchConfiguration("region_bound_margin_m")
    compare_rollouts = LaunchConfiguration("compare_rollouts")
    save_path_overlays = LaunchConfiguration("save_path_overlays")
    path_overlay_grid_n = LaunchConfiguration("path_overlay_grid_n")
    path_overlay_dir = LaunchConfiguration("path_overlay_dir")

    share = get_package_share_directory("mntfields")
    rviz_cfg = os.path.join(share, "rviz", "region_graph_inspector.rviz")

    inspector = Node(
        package="mntfields",
        executable="region_graph_inspector",
        name="region_graph_inspector",
        output="screen",
        parameters=[
            {
                "model_dir": model_dir,
                "map_yaml": map_yaml,
                "step_size": step_size,
                "max_steps": max_steps,
                "adjacency_gap_m": adjacency_gap_m,
                "max_neighbors_per_region": max_neighbors_per_region,
                "inflation_cells": inflation_cells,
                "region_bound_margin_m": region_bound_margin_m,
                "compare_rollouts": compare_rollouts,
                "save_path_overlays": save_path_overlays,
                "path_overlay_grid_n": path_overlay_grid_n,
                "path_overlay_dir": path_overlay_dir,
                "auto_launch_rviz": False,
            }
        ],
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2_region_graph_inspector",
        output="screen",
        arguments=["-d", rviz_cfg],
    )

    ld = LaunchDescription()
    ld.add_action(
        DeclareLaunchArgument(
            "map_yaml",
            default_value="/home/mayank/turtlebot3_ws/mntfields_model/map.yaml",
            description="Absolute path to map YAML",
        )
    )
    ld.add_action(
        DeclareLaunchArgument(
            "model_dir",
            default_value="/home/mayank/turtlebot3_ws/mntfields_model",
            description="Directory with region_N checkpoints and topology.json",
        )
    )
    ld.add_action(DeclareLaunchArgument("step_size", default_value="0.10"))
    ld.add_action(DeclareLaunchArgument("max_steps", default_value="400"))
    ld.add_action(DeclareLaunchArgument("adjacency_gap_m", default_value="0.80"))
    ld.add_action(DeclareLaunchArgument("max_neighbors_per_region", default_value="3"))
    ld.add_action(DeclareLaunchArgument("inflation_cells", default_value="1"))
    ld.add_action(DeclareLaunchArgument("region_bound_margin_m", default_value="0.06"))
    ld.add_action(DeclareLaunchArgument("compare_rollouts", default_value="true"))
    ld.add_action(DeclareLaunchArgument("save_path_overlays", default_value="true"))
    ld.add_action(DeclareLaunchArgument("path_overlay_grid_n", default_value="120"))
    ld.add_action(DeclareLaunchArgument("path_overlay_dir", default_value=""))

    # Publish map as TF root so RViz "Fixed Frame: map" resolves without error.
    # The inspector is offline (no SLAM/localization running), so nothing else
    # broadcasts this frame.
    map_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="map_frame_publisher",
        arguments=["0", "0", "0", "0", "0", "0", "map", "odom"],
    )

    ld.add_action(map_tf)
    ld.add_action(inspector)
    ld.add_action(rviz)
    return ld
