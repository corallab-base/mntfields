"""
mntfields_tb4_sim_navigation.launch.py

Launches Gazebo Sim + TurtleBot4 + map_server + AMCL + line_nav_node for
TB4-style simulation navigation against a saved map.

Usage:
    ros2 launch mntfields mntfields_tb4_sim_navigation.launch.py
    ros2 launch mntfields mntfields_tb4_sim_navigation.launch.py map:=/path/to/map.yaml
    ros2 launch mntfields mntfields_tb4_sim_navigation.launch.py x_pose:=-1.5 y_pose:=-1.8
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction, RegisterEventHandler, SetEnvironmentVariable
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, EnvironmentVariable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


_WORLD_NAME = "default"
_IGN_LIDAR_TOPIC = (
    f"/world/{_WORLD_NAME}/model/turtlebot4"
    f"/link/rplidar_link/sensor/rplidar/scan"
)


def _expand_path(raw: str) -> str:
    return os.path.abspath(os.path.expandvars(os.path.expanduser(raw.strip())))


def _nav_setup(context):
    resolved_model_dir = _expand_path(LaunchConfiguration("model_dir").perform(context))
    resolved_map = _expand_path(LaunchConfiguration("map").perform(context))

    map_server_node = Node(
        package="nav2_map_server",
        executable="map_server",
        name="map_server",
        output="screen",
        parameters=[{
            "use_sim_time": True,
            "yaml_filename": resolved_map,
        }],
    )

    amcl_node = Node(
        package="nav2_amcl",
        executable="amcl",
        name="amcl",
        output="screen",
        parameters=[{
            "use_sim_time": True,
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
            "use_sim_time": True,
            "autostart": True,
            "node_names": ["map_server", "amcl"],
        }],
    )

    nav_node = Node(
        package="mntfields",
        executable="line_nav_node",
        name="mntfields_nav",
        output="screen",
        parameters=[{
            "use_sim_time": True,
            "model_dir": resolved_model_dir,
            "metrics_path": os.path.join(resolved_model_dir, "line_navigation_metrics.json"),
            "max_lin": LaunchConfiguration("max_lin"),
            "max_ang": LaunchConfiguration("max_ang"),
            "base_frame": "base_link",
            "use_keyboard_confirm": False,
        }],
        remappings=[
            ("/cmd_vel", "/diffdrive_controller/cmd_vel_unstamped"),
        ],
    )

    sdf_replanner_node = Node(
        package="mntfields",
        executable="sdf_replanner_node",
        name="sdf_local_replanner",
        output="screen",
        parameters=[{
            "use_sim_time": True,
            "metrics_path": os.path.join(resolved_model_dir, "sdf_replanner_metrics.json"),
            "global_frame": "map",
            "base_frame": "base_link",
        }],
    )

    return [
        map_server_node,
        amcl_node,
        lifecycle_manager,
        nav_node,
        sdf_replanner_node,
    ]


def generate_launch_description():
    pkg_mntfields = get_package_share_directory("mntfields")
    pkg_ros_gz_sim = get_package_share_directory("ros_gz_sim")
    pkg_turtlebot4_description = get_package_share_directory("turtlebot4_description")
    pkg_create3_description = get_package_share_directory("irobot_create_description")

    world = os.path.join(pkg_mntfields, "worlds", "house_main_tb4.world")
    control_params = os.path.join(pkg_mntfields, "config", "tb4_sim_control.yaml")
    tb3_models = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(pkg_mntfields)))),
        "src",
        "turtlebot3_simulations",
        "turtlebot3_gazebo",
        "models",
    )
    gz_resource_path = ":".join([
        os.environ.get("GZ_SIM_RESOURCE_PATH", ""),
        os.path.dirname(pkg_turtlebot4_description),
        os.path.dirname(pkg_create3_description),
        tb3_models,
    ]).strip(":")

    model_dir = LaunchConfiguration("model_dir")
    map_yaml = LaunchConfiguration("map")
    x_pose = LaunchConfiguration("x_pose")
    y_pose = LaunchConfiguration("y_pose")
    max_lin = LaunchConfiguration("max_lin")
    max_ang = LaunchConfiguration("max_ang")

    gz_resource_env = SetEnvironmentVariable(
        name="GZ_SIM_RESOURCE_PATH",
        value=gz_resource_path,
    )

    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, "launch", "gz_sim.launch.py")
        ),
        launch_arguments={
            "gz_args": f"-r {world}",
            "gz_version": "6",
            "on_exit_shutdown": "true",
        }.items(),
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[
            {
                "use_sim_time": True,
                "robot_description": Command(
                    [
                        "xacro ",
                        PathJoinSubstitution(
                            [
                                FindPackageShare("turtlebot4_description"),
                                "urdf",
                                "standard",
                                "turtlebot4.urdf.xacro",
                            ]
                        ),
                        " gazebo:=ignition",
                    ]
                ),
            }
        ],
    )

    spawn_robot = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=[
            "-name", "turtlebot4",
            "-topic", "robot_description",
            "-x", x_pose,
            "-y", y_pose,
            "-z", "0.05",
            "-Y", "0.0",
        ],
        output="screen",
    )

    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", "-c", "controller_manager"],
        output="screen",
    )
    diffdrive_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        parameters=[control_params],
        arguments=["diffdrive_controller", "-c", "controller_manager"],
        output="screen",
    )
    diffdrive_controller_callback = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=joint_state_broadcaster_spawner,
            on_exit=[diffdrive_controller_spawner],
        )
    )

    bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            f"/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
            f"{_IGN_LIDAR_TOPIC}@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan",
        ],
        remappings=[
            (_IGN_LIDAR_TOPIC, "/scan"),
        ],
        output="screen",
    )

    rplidar_frame_bridge = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="rplidar_frame_bridge",
        arguments=[
            "0", "0", "0",
            "0", "0", "0",
            "rplidar_link",
            "turtlebot4/rplidar_link/rplidar",
        ],
        output="screen",
    )

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
        DeclareLaunchArgument("x_pose", default_value="-1.5"),
        DeclareLaunchArgument("y_pose", default_value="-1.8"),
        DeclareLaunchArgument("max_lin", default_value="0.12"),
        DeclareLaunchArgument("max_ang", default_value="0.6"),
        gz_resource_env,
        gz_sim,
        robot_state_publisher,
        spawn_robot,
        joint_state_broadcaster_spawner,
        diffdrive_controller_callback,
        bridge,
        rplidar_frame_bridge,
        OpaqueFunction(function=_nav_setup),
    ])
