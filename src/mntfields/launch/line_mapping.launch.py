"""
line_mapping.launch.py

Launches Gazebo + TurtleBot3 + SLAM + mapping_node (line-based room segmentation).

Usage:
    ros2 launch mntfields line_mapping.launch.py
    ros2 launch mntfields line_mapping.launch.py x_pose:=-2.0 y_pose:=-0.5
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    launch_file_dir = os.path.join(
        get_package_share_directory('turtlebot3_gazebo'), 'launch')
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')
    world = os.path.join(
        get_package_share_directory('turtlebot3_gazebo'), 'worlds', 'house_main.world')

    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    x_pose = LaunchConfiguration('x_pose', default='-1.5')
    y_pose = LaunchConfiguration('y_pose', default='-1.8')

    slam_params = os.path.join(
        get_package_share_directory('mntfields'), 'config', 'slam_toolbox_params.yaml')

    gzserver_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gzserver.launch.py')),
        launch_arguments={'world': world}.items())

    gzclient_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gzclient.launch.py')))

    robot_state_publisher_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(launch_file_dir, 'robot_state_publisher.launch.py')),
        launch_arguments={'use_sim_time': use_sim_time}.items())

    spawn_turtlebot_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(launch_file_dir, 'spawn_turtlebot3.launch.py')),
        launch_arguments={'x_pose': x_pose, 'y_pose': y_pose}.items())

    slam_toolbox_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('slam_toolbox'), 'launch',
                         'online_sync_launch.py')),
        launch_arguments={
            'slam_params_file': slam_params,
            'use_sim_time': use_sim_time,
        }.items())

    mapping_node = Node(
        package='mntfields', executable='mapping_node',
        name='mntfields_mapping', output='screen',
        parameters=[{'use_sim_time': True}])

    semantic_detector = Node(
        package='mntfields', executable='semantic_detector_node',
        name='semantic_detector', output='screen',
        parameters=[{'use_sim_time': True}])

    # Completes the TF tree: oakd_pro_link → camera_link
    camera_tf = Node(
        package='tf2_ros', executable='static_transform_publisher',
        name='oakd_camera_tf',
        arguments=[
            '--x', '0.0', '--y', '0.0', '--z', '0.0',
            '--roll', '0.0', '--pitch', '0.0', '--yaw', '0.0',
            '--frame-id', 'oakd_pro_link',
            '--child-frame-id', 'camera_link',
        ])

    ld = LaunchDescription()
    ld.add_action(DeclareLaunchArgument('use_sim_time', default_value='true'))
    ld.add_action(DeclareLaunchArgument('x_pose', default_value='-1.5'))
    ld.add_action(DeclareLaunchArgument('y_pose', default_value='-1.8'))
    ld.add_action(gzserver_cmd)
    ld.add_action(gzclient_cmd)
    ld.add_action(robot_state_publisher_cmd)
    ld.add_action(spawn_turtlebot_cmd)
    ld.add_action(slam_toolbox_cmd)
    ld.add_action(camera_tf)
    ld.add_action(semantic_detector)
    ld.add_action(mapping_node)
    return ld
