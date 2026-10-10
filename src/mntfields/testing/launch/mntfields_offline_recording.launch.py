"""
mntfields_offline_recording.launch.py

Launch Gazebo + TurtleBot3 + offline map recorder + RViz.

Usage:
  ros2 launch mntfields mntfields_offline_recording.launch.py
  ros2 launch mntfields mntfields_offline_recording.launch.py x_pose:=-1.5 y_pose:=-1.8
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    tb3_launch_dir = os.path.join(get_package_share_directory('turtlebot3_gazebo'), 'launch')
    gazebo_ros_dir = get_package_share_directory('gazebo_ros')
    mntfields_share = get_package_share_directory('mntfields')
    default_world = os.path.join(get_package_share_directory('turtlebot3_gazebo'), 'worlds', 'house_main.world')

    use_sim_time = LaunchConfiguration('use_sim_time')
    world = LaunchConfiguration('world')
    x_pose = LaunchConfiguration('x_pose')
    y_pose = LaunchConfiguration('y_pose')
    output_root = LaunchConfiguration('output_root')
    session_name = LaunchConfiguration('session_name')

    gzserver_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(gazebo_ros_dir, 'launch', 'gzserver.launch.py')),
        launch_arguments={'world': world}.items(),
    )

    gzclient_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(gazebo_ros_dir, 'launch', 'gzclient.launch.py')),
    )

    robot_state_publisher_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(tb3_launch_dir, 'robot_state_publisher.launch.py')),
        launch_arguments={'use_sim_time': use_sim_time}.items(),
    )

    spawn_turtlebot_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(tb3_launch_dir, 'spawn_turtlebot3.launch.py')),
        launch_arguments={'x_pose': x_pose, 'y_pose': y_pose}.items(),
    )

    recorder_node = Node(
        package='mntfields',
        executable='offline_map_recorder_node',
        name='offline_map_recorder',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'output_root': output_root,
            'session_name': session_name,
        }],
    )

    # Offline recorder publishes map in frame "map"; provide identity map->odom
    # so RobotModel/Scan also render under the same fixed frame in RViz.
    map_to_odom_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='offline_map_to_odom_tf',
        output='screen',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom'],
    )

    rviz_cfg = os.path.join(mntfields_share, 'rviz', 'offline.rviz')
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2_offline',
        output='screen',
        arguments=['-d', rviz_cfg],
        parameters=[{'use_sim_time': use_sim_time}],
    )

    ld = LaunchDescription()
    ld.add_action(DeclareLaunchArgument('use_sim_time', default_value='true'))
    ld.add_action(DeclareLaunchArgument('world', default_value=default_world))
    ld.add_action(DeclareLaunchArgument('x_pose', default_value='-1.5'))
    ld.add_action(DeclareLaunchArgument('y_pose', default_value='-1.8'))
    ld.add_action(DeclareLaunchArgument(
        'output_root',
        default_value='/home/mayank/turtlebot3_ws/mntfields_model/offline_datasets',
    ))
    ld.add_action(DeclareLaunchArgument('session_name', default_value=''))

    ld.add_action(gzserver_cmd)
    ld.add_action(gzclient_cmd)
    ld.add_action(robot_state_publisher_cmd)
    ld.add_action(spawn_turtlebot_cmd)
    ld.add_action(recorder_node)
    ld.add_action(map_to_odom_tf)
    ld.add_action(rviz_node)

    return ld
