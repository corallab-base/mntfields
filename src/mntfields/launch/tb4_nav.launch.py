from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'model_dir',
            default_value='/home/mayank/turtlebot3_ws/mntfields_model',
        ),
        DeclareLaunchArgument('max_lin', default_value='0.12'),
        DeclareLaunchArgument('max_ang', default_value='0.6'),
        Node(
            package='mntfields',
            executable='tb4_nav_node',
            name='tb4_nav_node',
            output='screen',
            parameters=[{
                'model_dir':    LaunchConfiguration('model_dir'),
                'global_frame': 'odom',
                'max_lin':      LaunchConfiguration('max_lin'),
                'max_ang':      LaunchConfiguration('max_ang'),
                'use_sim_time': False,
            }],
        ),
    ])
