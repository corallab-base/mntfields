from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    test = LaunchConfiguration('test')
    show_window = LaunchConfiguration('show_window')

    realsense_mount_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='tb4_realsense_mount_tf',
        output='screen',
        arguments=[
            '--x', '-0.0596',
            '--y', '0.0',
            '--z', '0.14933',
            '--roll', '0.0',
            '--pitch', '0.0',
            '--yaw', '0.0',
            '--frame-id', 'shell_link',
            '--child-frame-id', 'camera_link',
        ],
    )

    semantic_segmentation = Node(
        package='mntfields',
        executable='semantic_segmentation_rgbd_node',
        name='semantic_segmentation_rgbd',
        output='screen',
        parameters=[{
            'rgb_topic': '/camera/camera/color/image_raw',
            'depth_topic': '/camera/camera/aligned_depth_to_color/image_raw',
            'camera_info_topic': '/camera/camera/aligned_depth_to_color/camera_info',
            'global_frame': 'map',
            'tick_hz': 2.0,
            'max_sync_slop_sec': 0.20,
            'min_depth_m': 0.20,
            'max_depth_m': 6.0,
            'min_component_area_px': 600,
            'summary_stride_px': 4,
            'enable_world_coordinates': PythonExpression(
                ["'", test, "' != 'true'"]),
            'show_window': show_window,
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'test',
            default_value='false',
            description='If true, skip TF/world projection and report camera-frame coordinates only.',
        ),
        DeclareLaunchArgument(
            'show_window',
            default_value='false',
            description='If true, show an OpenCV preview window with the segmentation overlay.',
        ),
        realsense_mount_tf,
        semantic_segmentation,
    ])
