"""
mntfields_navigation.launch.py

One-shot launch: Gazebo sim + robot + map/AMCL localization + mntfields nav.

What it starts:
  - Gazebo (house_main.world) + robot state publisher + turtlebot3 spawn
  - map_server  (nav2, lifecycle)
  - amcl        (nav2, lifecycle, auto-activated by lifecycle manager)
  - nav2_lifecycle_manager  → auto-configures + activates map_server & amcl
  - mntfields nav_node
  - sdf_replanner_node (optional, use_sdf_planner:=true)

Arguments:
  map_yaml        path to map.yaml   (default: mntfields_model ros_map)
  model_dir       region checkpoints (default: mntfields_model/)
  x_pose / y_pose robot spawn pose   (default: -1.5 / -1.8)
  max_lin / max_ang  velocity limits
  use_sdf_planner launch SDF replanner (default: false)

Usage:
  export TURTLEBOT3_MODEL=waffle_pi
  ros2 launch mntfields mntfields_navigation.launch.py

  # with SDF replanner:
  ros2 launch mntfields mntfields_navigation.launch.py use_sdf_planner:=true

Jackal porting note:
  This launch file intentionally remains TurtleBot3-specific. For Jackal,
  replace the Gazebo spawn/robot-state-publisher and AMCL sections with the
  Jackal driver/description launch, then verify scan, cmd_vel, and base-frame
  TF names before reusing the MNTFields nodes.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

TURTLEBOT3_MODEL = os.environ.get('TURTLEBOT3_MODEL', 'waffle_pi')


def generate_launch_description():

    # ── Launch configurations ────────────────────────────────────────────────
    map_yaml  = LaunchConfiguration('map_yaml')
    model_dir = LaunchConfiguration('model_dir')
    x_pose    = LaunchConfiguration('x_pose')
    y_pose    = LaunchConfiguration('y_pose')
    max_lin   = LaunchConfiguration('max_lin')
    max_ang   = LaunchConfiguration('max_ang')
    use_sdf   = LaunchConfiguration('use_sdf_planner')

    # ── Package directories ──────────────────────────────────────────────────
    tb3_gazebo_dir   = get_package_share_directory('turtlebot3_gazebo')
    tb3_nav2_dir     = get_package_share_directory('turtlebot3_navigation2')
    gazebo_ros_dir   = get_package_share_directory('gazebo_ros')

    amcl_params = os.path.join(
        tb3_nav2_dir, 'param', 'humble', f'{TURTLEBOT3_MODEL}.yaml')

    # ── Gazebo world (server + client + robot state publisher + spawn) ────────
    world = os.path.join(tb3_gazebo_dir, 'worlds', 'house_main.world')

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

    # ── map_server ────────────────────────────────────────────────────────────
    map_server = Node(
        package='nav2_map_server', executable='map_server',
        name='map_server', output='screen',
        parameters=[{
            'yaml_filename': map_yaml,
            'use_sim_time':  True,
        }])

    # ── AMCL ─────────────────────────────────────────────────────────────────
    amcl = Node(
        package='nav2_amcl', executable='amcl',
        name='amcl', output='screen',
        parameters=[amcl_params, {'use_sim_time': True}])

    # ── Lifecycle manager — auto-activates map_server + amcl ─────────────────
    lifecycle_manager = Node(
        package='nav2_lifecycle_manager', executable='lifecycle_manager',
        name='lifecycle_manager_localization', output='screen',
        parameters=[{
            'use_sim_time':  True,
            'autostart':     True,
            'node_names':    ['map_server', 'amcl'],
        }])

    # ── mntfields navigation node ─────────────────────────────────────────────
    nav_node = Node(
        package='mntfields', executable='nav_node',
        name='navigation_node', output='screen',
        parameters=[{
            'model_dir':         model_dir,
            'map_yaml':          map_yaml,
            'max_lin':           max_lin,
            'max_ang':           max_ang,
            'goal_reached_dist': 0.25,
            'step_size':         0.10,
            'max_steps':         400,
            'replan_at_halfway': True,
            'use_sim_time':      True,
        }])

    # ── SDF local replanner (optional) ───────────────────────────────────────
    sdf_replanner = Node(
        package='mntfields', executable='sdf_replanner_node',
        name='sdf_local_replanner', output='screen',
        condition=IfCondition(use_sdf),
        parameters=[{
            'sdf_repo_dir':         '/home/mayank/comp-sdf-dyn-nav',
            'lidar_scan_topic':     '/scan',
            'replan_interval_sec':  0.5,
            'max_traj_pts':         12,
            'max_ahead_dist':       3.0,
            'opt_iterations':       30,
            'z_value':              0.225,
            'smoothing':            0.25,
            'trunc_dist_sdf':       0.05,
            'use_sim_time':         True,
        }])

    # ── Assemble ─────────────────────────────────────────────────────────────
    ld = LaunchDescription()

    ld.add_action(DeclareLaunchArgument('map_yaml',
        default_value='/home/mayank/turtlebot3_ws/mntfields_model/mntfields_output/ros_map/map.yaml',
        description='Absolute path to map YAML'))
    ld.add_action(DeclareLaunchArgument('model_dir',
        default_value='/home/mayank/turtlebot3_ws/mntfields_model',
        description='Directory with region_N/ checkpoints'))
    ld.add_action(DeclareLaunchArgument('x_pose', default_value='-1.5'))
    ld.add_action(DeclareLaunchArgument('y_pose', default_value='-1.8'))
    ld.add_action(DeclareLaunchArgument('max_lin',  default_value='0.15'))
    ld.add_action(DeclareLaunchArgument('max_ang',  default_value='0.8'))
    ld.add_action(DeclareLaunchArgument('use_sdf_planner', default_value='false',
        description='Also launch SDF local replanner (true/false)'))

    ld.add_action(gzserver)
    ld.add_action(gzclient)
    ld.add_action(robot_state_publisher)
    ld.add_action(spawn_turtlebot)
    ld.add_action(map_server)
    ld.add_action(amcl)
    ld.add_action(lifecycle_manager)
    ld.add_action(nav_node)
    ld.add_action(sdf_replanner)

    return ld
