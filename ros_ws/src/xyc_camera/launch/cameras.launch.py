from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, Shutdown
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from xyc.robot.profile import robot_profile


def generate_launch_description():
    package_dir = Path(get_package_share_directory('xyc_camera'))
    profile = robot_profile()
    dds_file = str(package_dir / 'config' / 'fastdds.xml')

    nodes = [DeclareLaunchArgument('preview', default_value='true', choices=['true', 'false'])]
    for camera, settings in profile.cameras.items():
        nodes.append(
            Node(
                package='xyc_camera',
                executable='camera_node',
                name=f'camera_{camera}',
                namespace=f'/camera/{camera}',
                parameters=[settings],
                additional_env={'FASTRTPS_DEFAULT_PROFILES_FILE': dds_file},
                output='screen',
                on_exit=Shutdown(reason=f'{camera} camera process exited'),
            )
        )

    nodes.append(Node(
        package='xyc_camera', executable='camera_preview', name='camera_preview',
        condition=IfCondition(LaunchConfiguration('preview')),
        additional_env={'FASTRTPS_DEFAULT_PROFILES_FILE': dds_file}, output='screen',
        on_exit=Shutdown(reason='camera preview process exited'),
    ))
    return LaunchDescription(nodes)
