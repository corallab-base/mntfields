"""
antfields_jackal_mapping.launch.py

Single-region AntFields-style exploration on a Clearpath Jackal (ROS 2 Jazzy).

Runs entirely on the offboard desktop: SLAM Toolbox, the antfields node and
RViz. The robot only runs its stock Clearpath stack; everything is exchanged
over ROS topics. Clearpath namespaces its topics and TF under /<namespace>/,
so /tf, /tf_static and /scan are remapped onto that namespace here.

Usage:
    ros2 launch mntfields antfields_jackal_mapping.launch.py
    # dry run (robot cannot move):
    ros2 launch mntfields antfields_jackal_mapping.launch.py \
        cmd_vel_topic:=/antfields/cmd_vel_dry
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    GroupAction,
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, SetRemap


ANTFIELDS_PARAMS = [
    "model_dir",
    "lidar_height_m",
    "lidar_max_range_m",
    "wall_min_m",
    "wall_max_m",
    "wall_offset_m",
    "collision_pad_cells",
    "goal_reached_dist",
    "path_end_tolerance_m",
    "goal_blacklist_radius_m",
    "abandoned_goal_ttl_sec",
    "max_lin",
    "max_ang",
    "fixed_region_size_m",
    "expand_region_with_map",
    "train_timer_sec",
    "train_samples",
    "train_epoch",
    "min_train_epochs_for_nav",
    "frontier_max_goal_dist_m",
    "enable_region_debug_vis",
    "field_vis_radius_m",
    "field_vis_resolution_m",
    "field_vis_free_only",
    "enable_path_validation",
    "trim_at_unknown",
    "collision_check_skip_m",
    "collision_radius_m",
    "validation_horizon_m",
    "route_subgoal_dist_m",
    "min_frontier_unknown_cells",
    "require_goalward_start",
    "safety_stop_enabled",
    "robot_front_m",
    "robot_rear_m",
    "drive_reverse",
    "robot_half_width_m",
    "safety_stop_dist_m",
    "safety_slow_dist_m",
    "robot_turn_radius_m",
    "initial_spin",
]


def _launch_setup(context, *args, **kwargs):
    def arg(name):
        return LaunchConfiguration(name).perform(context)

    ns = arg("namespace").strip("/")
    prefix = f"/{ns}" if ns else ""
    scan_topic = arg("scan_topic") or f"{prefix}/sensors/lidar2d_0/scan"
    cmd_vel_topic = arg("cmd_vel_topic") or f"{prefix}/cmd_vel"
    use_sim_time = arg("use_sim_time").lower() == "true"

    tf_remaps = [("/tf", f"{prefix}/tf"), ("/tf_static", f"{prefix}/tf_static")]

    slam_params = os.path.join(
        get_package_share_directory("mntfields"),
        "config",
        "slam_toolbox_jackal_params.yaml",
    )
    slam_toolbox = GroupAction(
        [
            SetRemap(src=src, dst=dst)
            for src, dst in tf_remaps + [("/scan", scan_topic)]
        ]
        + [
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(
                        get_package_share_directory("slam_toolbox"),
                        "launch",
                        "online_sync_launch.py",
                    )
                ),
                launch_arguments={
                    "slam_params_file": slam_params,
                    "use_sim_time": arg("use_sim_time"),
                }.items(),
            )
        ]
    )

    params = {
        "use_sim_time": use_sim_time,
        "scan_topic": scan_topic,
        "cmd_vel_topic": cmd_vel_topic,
        "cmd_vel_stamped": arg("cmd_vel_stamped").lower() == "true",
        "base_frame": arg("base_frame"),
    }
    for name in ANTFIELDS_PARAMS:
        params[name] = LaunchConfiguration(name)

    antfields_node = Node(
        package="mntfields",
        executable="antfields_gridmap_node",
        name="antfields_gridmap",
        output="screen",
        parameters=[params],
        remappings=tf_remaps,
    )

    rviz_config = os.path.join(
        get_package_share_directory("mntfields"), "rviz", "antfields_jackal.rviz"
    )
    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", rviz_config],
        parameters=[{"use_sim_time": use_sim_time}],
        # The config uses plain /scan and /robot_description; map them to the robot.
        remappings=tf_remaps
        + [("/scan", scan_topic), ("/robot_description", f"{prefix}/robot_description")],
        condition=IfCondition(LaunchConfiguration("rviz")),
    )

    return [slam_toolbox, antfields_node, rviz]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "namespace", default_value="j100_1639",
                description="Clearpath robot namespace (system.ros2.namespace in /etc/clearpath/robot.yaml)"),
            DeclareLaunchArgument(
                "scan_topic", default_value="",
                description="LaserScan topic (default: /<namespace>/sensors/lidar2d_0/scan)"),
            DeclareLaunchArgument(
                "cmd_vel_topic", default_value="",
                description="Velocity topic (default: /<namespace>/cmd_vel)"),
            DeclareLaunchArgument(
                "cmd_vel_stamped", default_value="true",
                description="Publish TwistStamped (check: ros2 topic info /<ns>/cmd_vel -v)"),
            DeclareLaunchArgument("base_frame", default_value="base_link"),
            DeclareLaunchArgument("use_sim_time", default_value="false"),
            DeclareLaunchArgument("rviz", default_value="true"),
            DeclareLaunchArgument(
                "model_dir", default_value=os.path.expanduser("~/mntfields_models/antfields_jackal")),
            DeclareLaunchArgument(
                "lidar_height_m", default_value="0.30",
                description="Only used for PointCloud2 input; LaserScan height comes from TF"),
            DeclareLaunchArgument("lidar_max_range_m", default_value="8.0"),
            # Field speed band (TurtleBot values: lowest speed wall_min/wall_max = 0.06, so
            # walls strongly repel paths), measured from walls inflated by wall_offset_m.
            # Raising wall_min instead would make walls cheap to cut through.
            DeclareLaunchArgument("wall_min_m", default_value="0.05"),
            DeclareLaunchArgument("wall_max_m", default_value="0.80"),
            DeclareLaunchArgument(
                "wall_offset_m", default_value="0.25",
                description="Field clearance for the Jackal (half-width 0.215 m + margin)"),
            DeclareLaunchArgument(
                "collision_pad_cells", default_value="7",
                description="Square padding; only used when collision_radius_m is negative"),
            DeclareLaunchArgument(
                "collision_radius_m", default_value="0.30",
                description="Reject paths closer than this to a wall (Jackal half-width 0.215 + margin); "
                "0 = no inflation, negative = square collision_pad_cells"),
            DeclareLaunchArgument("goal_reached_dist", default_value="0.40"),
            DeclareLaunchArgument(
                "path_end_tolerance_m", default_value="0.15",
                description="Follower: how close to the end of each committed path piece is done"),
            DeclareLaunchArgument("goal_blacklist_radius_m", default_value="0.8"),
            DeclareLaunchArgument(
                "abandoned_goal_ttl_sec", default_value="60.0",
                description="How long an abandoned (unreached) frontier stays blacklisted"),
            DeclareLaunchArgument("max_lin", default_value="0.20"),
            DeclareLaunchArgument("max_ang", default_value="0.6"),
            DeclareLaunchArgument("fixed_region_size_m", default_value="20.0"),
            DeclareLaunchArgument("expand_region_with_map", default_value="false"),
            DeclareLaunchArgument("train_timer_sec", default_value="0.4"),
            DeclareLaunchArgument("train_samples", default_value="12000"),
            DeclareLaunchArgument("train_epoch", default_value="20"),
            DeclareLaunchArgument("min_train_epochs_for_nav", default_value="60"),
            DeclareLaunchArgument("frontier_max_goal_dist_m", default_value="6.0"),
            DeclareLaunchArgument("enable_region_debug_vis", default_value="true"),
            DeclareLaunchArgument(
                "field_vis_radius_m", default_value="6.0",
                description="Half-width of the travel-time field shown in RViz"),
            DeclareLaunchArgument("field_vis_resolution_m", default_value="0.10"),
            DeclareLaunchArgument(
                "field_vis_free_only", default_value="true",
                description="Only show the field on cells SLAM marks free"),
            # The Hokuyo UST covers +/-135 deg, leaving a 90 deg blind wedge behind the
            # robot that stays unknown in SLAM and untrained in the field. Only drive
            # through mapped-free space, stop on live scan obstacles, and spin once at
            # start-up to map the wedge.
            DeclareLaunchArgument(
                "enable_path_validation", default_value="true",
                description="Check planned paths against the SLAM map before driving"),
            DeclareLaunchArgument(
                "trim_at_unknown", default_value="true",
                description="Cut paths at the first cell SLAM has not seen as free"),
            DeclareLaunchArgument(
                "collision_check_skip_m", default_value="0.35",
                description="Skip SLAM collision checks this close to the robot"),
            DeclareLaunchArgument(
                "validation_horizon_m", default_value="1.5",
                description="Only validate this much of each path (re-checked on replans)"),
            DeclareLaunchArgument(
                "route_subgoal_dist_m", default_value="1.5",
                description="Far frontiers: field plans to this far along the SLAM-map route (0 = off)"),
            DeclareLaunchArgument(
                "min_frontier_unknown_cells", default_value="20",
                description="Ignore frontiers at unknown holes smaller than this (cells) in mapped space"),
            DeclareLaunchArgument(
                "require_goalward_start", default_value="false",
                description="Reject paths whose first step points away from the goal"),
            DeclareLaunchArgument("safety_stop_enabled", default_value="true"),
            # Jackal: base_link at the chassis centre, 0.51 x 0.43 m with fenders.
            DeclareLaunchArgument("robot_front_m", default_value="0.28"),
            DeclareLaunchArgument("robot_rear_m", default_value="0.28"),
            DeclareLaunchArgument(
                "drive_reverse", default_value="true",
                description="Drive rear-first: this Jackal's lidar faces the rear (power-button side)"),
            DeclareLaunchArgument("robot_half_width_m", default_value="0.25"),
            DeclareLaunchArgument(
                "safety_stop_dist_m", default_value="0.25",
                description="Stop forward motion when an obstacle is this close ahead of the bumper"),
            DeclareLaunchArgument(
                "safety_slow_dist_m", default_value="0.60",
                description="Start slowing down at this gap ahead of the bumper"),
            DeclareLaunchArgument(
                "robot_turn_radius_m", default_value="0.34",
                description="Clearance needed to turn on the spot (Jackal corners sweep 0.333 m)"),
            DeclareLaunchArgument("initial_spin", default_value="true"),
            OpaqueFunction(function=_launch_setup),
        ]
    )
