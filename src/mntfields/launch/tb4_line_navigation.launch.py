"""
tb4_line_navigation.launch.py

Launches map_server + AMCL + line_nav_node + sdf_replanner_node for real TurtleBot4.

map_server serves the saved map.pgm/map.yaml from the mapping session.
AMCL localizes the robot against that map and publishes /amcl_pose.
line_nav_node loads saved NTField regions and plans paths via the room graph.
sdf_replanner_node refines the global path using SDF-based local optimization.

Usage:
    ros2 launch mntfields tb4_line_navigation.launch.py

    Optional overrides:
        map:=/path/to/map.yaml
        use_sim_time:=false
        max_lin:=0.12
        max_ang:=0.6

Run RViz separately:
    rviz2 -d ~/turtlebot3_ws/mapping_real.rviz
    (Set 2D Pose Estimate to initialise AMCL, then 2D Goal Pose to send goals.)
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import EnvironmentVariable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def _expand_path(raw: str) -> str:
    return os.path.abspath(os.path.expandvars(os.path.expanduser(raw.strip())))

def _nav_setup(context):
    resolved_model_dir = _expand_path(LaunchConfiguration("model_dir").perform(context))
    resolved_map = _expand_path(LaunchConfiguration("map").perform(context))
    resolved_use_sim_time = LaunchConfiguration("use_sim_time")
    resolved_max_lin = LaunchConfiguration("max_lin")
    resolved_max_ang = LaunchConfiguration("max_ang")

    # ── map_server ────────────────────────────────────────────────────────────
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

    # ── AMCL ──────────────────────────────────────────────────────────────────
    amcl_node = Node(
        package="nav2_amcl",
        executable="amcl",
        name="amcl",
        output="screen",
        parameters=[{
            "use_sim_time": resolved_use_sim_time,
            # Particle filter bounds
            "min_particles": 500,
            "max_particles": 3000,
            # Motion model (differential drive)
            "robot_model_type": "nav2_amcl::DifferentialMotionModel",
            "alpha1": 0.2,   # rotation noise from rotation
            "alpha2": 0.2,   # rotation noise from translation
            "alpha3": 0.2,   # translation noise from translation
            "alpha4": 0.2,   # translation noise from rotation
            # Sensor model
            "laser_model_type": "likelihood_field",
            "laser_max_range": 12.0,
            "laser_min_range": 0.15,
            "max_beams": 60,
            "sigma_hit": 0.2,
            "z_hit": 0.5,
            "z_rand": 0.5,
            # Frames
            "base_frame_id": "base_link",
            "odom_frame_id": "odom",
            "global_frame_id": "map",
            "scan_topic": "/scan",
            # Update thresholds — update on meaningful motion only
            "update_min_d": 0.1,
            "update_min_a": 0.2,
            "resample_interval": 2,
            # Recovery: scatter particles if filter diverges
            "recovery_alpha_slow": 0.0,
            "recovery_alpha_fast": 0.0,
            # Publish tf map→odom
            "tf_broadcast": True,
        }],
    )

    # ── Lifecycle manager — brings map_server + amcl up in order ──────────────
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

    # ── mntfields navigation ───────────────────────────────────────────────────
    nav_node = Node(
        package="mntfields",
        executable="line_nav_node",
        name="mntfields_nav",
        output="screen",
        parameters=[{
            "use_sim_time": resolved_use_sim_time,
            "model_dir": resolved_model_dir,
            "metrics_path": os.path.join(resolved_model_dir, "line_navigation_metrics.json"),
            "max_lin": resolved_max_lin,
            "max_ang": resolved_max_ang,
            "base_frame": "base_link",
            "use_keyboard_confirm": False,
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

    # Completes the TF tree: oakd_pro_link → camera_link
    # (Camera runs on a separate laptop; this TF must be present on the robot side too)
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

    return [
        map_server_node,
        amcl_node,
        lifecycle_manager,
        camera_tf,
        nav_node,
        sdf_replanner_node,
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
        OpaqueFunction(function=_nav_setup),
    ])
