"""
Straight-line global planning + local SDF replanning in the default
TurtleBot3 Gazebo world, using a static map->odom transform for RViz/map display.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


TURTLEBOT3_MODEL = os.environ.get('TURTLEBOT3_MODEL', 'waffle_pi')


def generate_launch_description():
    x_pose = LaunchConfiguration('x_pose')
    y_pose = LaunchConfiguration('y_pose')
    max_lin = LaunchConfiguration('max_lin')
    max_ang = LaunchConfiguration('max_ang')
    sdf_repo_dir = LaunchConfiguration('sdf_repo_dir')
    world = LaunchConfiguration('world')
    global_frame = LaunchConfiguration('global_frame')
    map_yaml = LaunchConfiguration('map_yaml')
    sdf_min_hold_sec = LaunchConfiguration('sdf_min_hold_sec')
    sdf_replace_threshold = LaunchConfiguration('sdf_replace_threshold')

    tb3_gazebo_dir = get_package_share_directory('turtlebot3_gazebo')
    tb3_nav2_dir = get_package_share_directory('turtlebot3_navigation2')
    gazebo_ros_dir = get_package_share_directory('gazebo_ros')
    mntfields_share = get_package_share_directory('mntfields')

    rviz_config = os.path.join(mntfields_share, 'rviz', 'sdf.rviz')

    gzserver = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_ros_dir, 'launch', 'gzserver.launch.py')),
        launch_arguments={'world': world}.items())

    gzclient = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_ros_dir, 'launch', 'gzclient.launch.py')))

    robot_state_publisher = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(tb3_gazebo_dir, 'launch', 'robot_state_publisher.launch.py')),
        launch_arguments={'use_sim_time': 'true'}.items())

    spawn_turtlebot = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(tb3_gazebo_dir, 'launch', 'spawn_turtlebot3.launch.py')),
        launch_arguments={'x_pose': x_pose, 'y_pose': y_pose}.items())

    map_server = Node(
        package='nav2_map_server', executable='map_server',
        name='map_server', output='screen',
        parameters=[{'yaml_filename': map_yaml, 'use_sim_time': True}])

    lifecycle_manager = Node(
        package='nav2_lifecycle_manager', executable='lifecycle_manager',
        name='lifecycle_manager_map', output='screen',
        parameters=[{
            'use_sim_time': True,
            'autostart': True,
            'node_names': ['map_server'],
        }])

    map_to_odom = Node(
        package='tf2_ros', executable='static_transform_publisher',
        name='map_to_odom_static_tf', output='screen',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom'])

    sdf_nav = Node(
        package='mntfields', executable='sdf_goal_nav_node',
        name='sdf_goal_nav_node', output='screen',
        parameters=[{
            'max_lin': max_lin,
            'max_ang': max_ang,
            'goal_reached_dist': 0.25,
            'global_path_spacing': 0.10,
            'sdf_valid_sec': 2.0,
            'sdf_min_hold_sec': sdf_min_hold_sec,
            'sdf_replace_threshold': sdf_replace_threshold,
            'control_dt': 0.10,
            'stuck_window': 25,
            'stuck_radius': 0.08,
            'global_frame': global_frame,
            'use_sim_time': True,
        }])

    sdf_replanner = Node(
        package='mntfields', executable='sdf_replanner_node',
        name='sdf_local_replanner', output='screen',
        parameters=[{
            'sdf_repo_dir': sdf_repo_dir,
            'lidar_scan_topic': '/scan',
            'replan_interval_sec': 0.2,
            'max_traj_pts': 8,
            'max_ahead_dist': 2.0,
            'opt_iterations': 12,
            'z_value': 0.225,
            'smoothing': 0.25,
            'trunc_dist_sdf': 0.05,
            'global_frame': global_frame,
            'use_sim_time': True,
        }])

    rviz = Node(
        package='rviz2', executable='rviz2',
        name='rviz2_sdf', output='screen',
        arguments=['-d', rviz_config],
        parameters=[{'use_sim_time': True}])

    ld = LaunchDescription()
    ld.add_action(DeclareLaunchArgument(
        'world',
        default_value=os.path.join(tb3_gazebo_dir, 'worlds', 'turtlebot3_world.world')))
    ld.add_action(DeclareLaunchArgument(
        'map_yaml',
        default_value=os.path.join(tb3_nav2_dir, 'map', 'map.yaml')))
    ld.add_action(DeclareLaunchArgument('global_frame', default_value='map'))
    ld.add_action(DeclareLaunchArgument('x_pose', default_value='-2.0'))
    ld.add_action(DeclareLaunchArgument('y_pose', default_value='-0.5'))
    ld.add_action(DeclareLaunchArgument('max_lin', default_value='0.15'))
    ld.add_action(DeclareLaunchArgument('max_ang', default_value='0.8'))
    ld.add_action(DeclareLaunchArgument('sdf_min_hold_sec', default_value='1.0'))
    ld.add_action(DeclareLaunchArgument('sdf_replace_threshold', default_value='0.20'))
    ld.add_action(DeclareLaunchArgument(
        'sdf_repo_dir', default_value='/home/mayank/comp-sdf-dyn-nav'))

    ld.add_action(gzserver)
    ld.add_action(gzclient)
    ld.add_action(robot_state_publisher)
    ld.add_action(spawn_turtlebot)
    ld.add_action(map_server)
    ld.add_action(lifecycle_manager)
    ld.add_action(map_to_odom)
    ld.add_action(sdf_nav)
    ld.add_action(sdf_replanner)
    ld.add_action(rviz)
    return ld
