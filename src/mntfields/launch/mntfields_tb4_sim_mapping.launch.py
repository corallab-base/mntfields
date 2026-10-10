"""
mntfields_tb4_sim_mapping.launch.py

Launches Ignition Gazebo + TurtleBot4 + mapping_node for MNTFields mapping.

Prerequisites (install once):
    sudo apt install ros-humble-irobot-create-control \
                     ros-humble-irobot-create-gazebo-plugins

Usage:
    ros2 launch mntfields mntfields_tb4_sim_mapping.launch.py
    ros2 launch mntfields mntfields_tb4_sim_mapping.launch.py x_pose:=-1.5 y_pose:=-1.8

Differences vs mntfields_mapping.launch.py (TB3):
  - Uses Ignition Gazebo (ros_gz_sim) instead of classic Gazebo (gazebo_ros)
  - Robot: TurtleBot4 standard  (base_frame = base_link, RPLIDAR A1 12 m range)
  - SLAM params: slam_toolbox_tb4_params.yaml
  - /scan bridged from Ignition sensor via ros_gz_bridge
  - /clock bridged from Ignition so use_sim_time works correctly
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, RegisterEventHandler, SetEnvironmentVariable
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


# Ignition world name — must match the <world name="..."> tag in the .world file.
_WORLD_NAME = "default"

# Ignition topic the RPLIDAR sensor publishes on (Fortress / Gazebo 6 convention).
_IGN_LIDAR_TOPIC = (
    f"/world/{_WORLD_NAME}/model/turtlebot4"
    f"/link/rplidar_link/sensor/rplidar/scan"
)


def generate_launch_description():
    pkg_mntfields = get_package_share_directory("mntfields")
    pkg_ros_gz_sim = get_package_share_directory("ros_gz_sim")
    pkg_turtlebot4_description = get_package_share_directory("turtlebot4_description")
    pkg_create3_description = get_package_share_directory("irobot_create_description")

    world = os.path.join(
        pkg_mntfields,
        "worlds",
        "house_main_tb4.world",
    )
    slam_params = os.path.join(pkg_mntfields, "config", "slam_toolbox_tb4_params.yaml")
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

    x_pose = LaunchConfiguration("x_pose")
    y_pose = LaunchConfiguration("y_pose")

    gz_resource_env = SetEnvironmentVariable(
        name="GZ_SIM_RESOURCE_PATH",
        value=gz_resource_path,
    )

    # ── 1. Ignition Gazebo ────────────────────────────────────────────────────
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

    # ── 2. Robot state publisher (TB4 description, Ignition mode) ─────────────
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

    # ── 3. Spawn TB4 into Ignition ─────────────────────────────────────────────
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

    # Spawn ros2_control controllers required for odom and wheel commands.
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

    # ── 4. ros_gz_bridge: clock + LiDAR scan ──────────────────────────────────
    # odom, cmd_vel and tf are handled directly by ign_ros2_control inside
    # Ignition — they surface as ROS 2 topics without bridging.
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

    # Gazebo Sim publishes the laser scan in a scoped frame name
    # (turtlebot4/rplidar_link/rplidar), while robot_state_publisher exposes the
    # TB4 URDF frame as rplidar_link. Publish an identity static TF so SLAM can
    # resolve incoming scans immediately.
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

    # ── 5. SLAM Toolbox ────────────────────────────────────────────────────────
    slam_toolbox = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("slam_toolbox"),
                "launch",
                "online_sync_launch.py",
            )
        ),
        launch_arguments={
            "slam_params_file": slam_params,
            "use_sim_time": "true",
        }.items(),
    )

    # ── 6. mntfields mapping node ──────────────────────────────────────────────
    mapping_node = Node(
        package="mntfields",
        executable="mapping_node",
        name="mntfields_mapping",
        output="screen",
        parameters=[
            {
                "use_sim_time": True,
                "robot_type": "real",
                "base_frame": "base_link",
                "cmd_vel_topic": "/diffdrive_controller/cmd_vel_unstamped",
                "lidar_max_range": 6.0,
                "train_timer_sec": 1.0,
                "train_epoch": 20,
                "max_lin": 0.28,
                "max_ang": 1.10,
            }
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("x_pose", default_value="-1.5"),
            DeclareLaunchArgument("y_pose", default_value="-1.8"),
            gz_resource_env,
            gz_sim,
            robot_state_publisher,
            spawn_robot,
            joint_state_broadcaster_spawner,
            diffdrive_controller_callback,
            bridge,
            rplidar_frame_bridge,
            slam_toolbox,
            mapping_node,
        ]
    )
