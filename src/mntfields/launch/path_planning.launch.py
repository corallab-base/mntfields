#!/usr/bin/env python3
"""
path_planning.launch.py

Launch the path-planner benchmark (NTFields + A* + RRT* + RRT) with RViz.

Usage (named arg, standard ROS2 syntax):
  ros2 launch mntfields path_planning.launch.py model_dir:=/path/to/mntfields_model

Usage (positional — bare absolute path after the launch file name):
  ros2 launch mntfields path_planning.launch.py /path/to/mntfields_model

Optional overrides (all have sensible defaults derived from model_dir):
  map_yaml:=             path to map YAML (defaults to model_dir/map.yaml)
  step_size:=0.10
  max_steps:=400
  adjacency_gap_m:=0.80
  max_neighbors_per_region:=3
  inflation_cells:=1
  region_bound_margin_m:=0.06
  rrtstar_max_iter:=5000
  rrtstar_step_size:=0.40
  rrtstar_neighborhood_radius:=0.50
  rrtstar_goal_sample_rate:=0.10
  rrtstar_goal_tolerance:=0.35
"""

import os
import sys

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _detect_positional_model_dir() -> str:
    """
    Return the first bare absolute path in sys.argv that isn't a ROS2 flag
    or key:=value pair.  Allows:
      ros2 launch mntfields path_planning.launch.py /path/to/model
    """
    default = "/home/mayank/turtlebot3_ws/mntfields_model"
    for arg in sys.argv[1:]:
        if ":=" in arg or arg.startswith("-"):
            continue
        # Skip package name and launch file name tokens
        if arg.endswith(".launch.py") or arg == "mntfields":
            continue
        if os.path.isabs(arg) or arg.startswith("~"):
            candidate = os.path.expanduser(arg)
            if os.path.isdir(candidate):
                return candidate
    return default


_DEFAULT_MODEL_DIR = _detect_positional_model_dir()


def launch_setup(context, *args, **kwargs):
    model_dir = LaunchConfiguration("model_dir").perform(context)
    map_yaml = LaunchConfiguration("map_yaml").perform(context)
    if not map_yaml:
        map_yaml = os.path.join(model_dir, "map.yaml")

    step_size = LaunchConfiguration("step_size")
    max_steps = LaunchConfiguration("max_steps")
    adjacency_gap_m = LaunchConfiguration("adjacency_gap_m")
    max_neighbors_per_region = LaunchConfiguration("max_neighbors_per_region")
    inflation_cells = LaunchConfiguration("inflation_cells")
    region_bound_margin_m = LaunchConfiguration("region_bound_margin_m")
    rrtstar_max_iter = LaunchConfiguration("rrtstar_max_iter")
    rrtstar_step_size = LaunchConfiguration("rrtstar_step_size")
    rrtstar_neighborhood_radius = LaunchConfiguration("rrtstar_neighborhood_radius")
    rrtstar_goal_sample_rate = LaunchConfiguration("rrtstar_goal_sample_rate")
    rrtstar_goal_tolerance = LaunchConfiguration("rrtstar_goal_tolerance")
    rrt_max_iter = LaunchConfiguration("rrt_max_iter")
    rrt_step_size = LaunchConfiguration("rrt_step_size")
    rrt_goal_sample_rate = LaunchConfiguration("rrt_goal_sample_rate")
    rrt_goal_tolerance = LaunchConfiguration("rrt_goal_tolerance")
    share = get_package_share_directory("mntfields")
    rviz_cfg = os.path.join(share, "rviz", "path_planner_benchmark.rviz")

    benchmark_node = Node(
        package="mntfields",
        executable="path_planner_benchmark",
        name="path_planner_benchmark",
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
                "compare_rollouts": False,
                "save_path_overlays": False,
                "path_overlay_grid_n": 0,
                "path_overlay_dir": "",
                "auto_launch_rviz": False,
                "rrtstar_max_iter": rrtstar_max_iter,
                "rrtstar_step_size": rrtstar_step_size,
                "rrtstar_neighborhood_radius": rrtstar_neighborhood_radius,
                "rrtstar_goal_sample_rate": rrtstar_goal_sample_rate,
                "rrtstar_goal_tolerance": rrtstar_goal_tolerance,
                "rrt_max_iter": rrt_max_iter,
                "rrt_step_size": rrt_step_size,
                "rrt_goal_sample_rate": rrt_goal_sample_rate,
                "rrt_goal_tolerance": rrt_goal_tolerance,
            }
        ],
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2_path_planner_benchmark",
        output="screen",
        arguments=["-d", rviz_cfg],
    )

    map_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="map_frame_publisher",
        arguments=["0", "0", "0", "0", "0", "0", "map", "odom"],
    )

    return [map_tf, benchmark_node, rviz_node]


def generate_launch_description():
    ld = LaunchDescription()

    ld.add_action(
        DeclareLaunchArgument(
            "model_dir",
            default_value=_DEFAULT_MODEL_DIR,
            description="Directory containing region_N/ bundles and topology.json",
        )
    )
    ld.add_action(
        DeclareLaunchArgument(
            "map_yaml",
            default_value="",
            description="Path to map YAML (defaults to model_dir/map.yaml if empty)",
        )
    )
    ld.add_action(DeclareLaunchArgument("step_size", default_value="0.10"))
    ld.add_action(DeclareLaunchArgument("max_steps", default_value="400"))
    ld.add_action(DeclareLaunchArgument("adjacency_gap_m", default_value="0.80"))
    ld.add_action(DeclareLaunchArgument("max_neighbors_per_region", default_value="3"))
    ld.add_action(DeclareLaunchArgument("inflation_cells", default_value="1"))
    ld.add_action(DeclareLaunchArgument("region_bound_margin_m", default_value="0.06"))
    ld.add_action(DeclareLaunchArgument("rrtstar_max_iter", default_value="5000"))
    ld.add_action(DeclareLaunchArgument("rrtstar_step_size", default_value="0.40"))
    ld.add_action(
        DeclareLaunchArgument("rrtstar_neighborhood_radius", default_value="0.50")
    )
    ld.add_action(
        DeclareLaunchArgument("rrtstar_goal_sample_rate", default_value="0.10")
    )
    ld.add_action(
        DeclareLaunchArgument("rrtstar_goal_tolerance", default_value="0.35")
    )
    ld.add_action(DeclareLaunchArgument("rrt_max_iter", default_value="5000"))
    ld.add_action(DeclareLaunchArgument("rrt_step_size", default_value="0.40"))
    ld.add_action(DeclareLaunchArgument("rrt_goal_sample_rate", default_value="0.10"))
    ld.add_action(DeclareLaunchArgument("rrt_goal_tolerance", default_value="0.35"))
    ld.add_action(OpaqueFunction(function=launch_setup))
    return ld
