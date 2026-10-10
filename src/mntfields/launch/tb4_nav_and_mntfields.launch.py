import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    slam = LaunchConfiguration('slam', default='off')
    localization = LaunchConfiguration('localization', default='true')

    nav_bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('turtlebot4_navigation'),
                'launch',
                'nav_bringup.launch.py',
            )
        ),
        launch_arguments={
            'slam': slam,
            'localization': localization,
        }.items(),
    )

    tb4_node = Node(
        package='mntfields',
        executable='tb4_node',
        name='mntfields_gridmap',
        output='screen',
    )

    return LaunchDescription([
        DeclareLaunchArgument('slam', default_value='off'),
        DeclareLaunchArgument('localization', default_value='true'),
        nav_bringup,
        tb4_node,
    ])
