# SPDX-License-Identifier: Apache-2.0
"""Read-only ROS view of the already-running unified robot backend."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from xyc.robot.profile import robot_profile


def generate_launch_description():
    package = Path(get_package_share_directory('xyc_observation'))
    camera = Path(get_package_share_directory('xyc_camera'))
    return LaunchDescription([
        DeclareLaunchArgument('calibration_dir', default_value=str(
            robot_profile().robot_root / 'config/calibration')),
        Node(package='xyc_observation', executable='observation_node', name='observation_node',
             namespace='/robot', parameters=[str(package / 'config/observation.yaml'),
                                           {'calibration_dir': LaunchConfiguration('calibration_dir')}],
             additional_env={'FASTRTPS_DEFAULT_PROFILES_FILE': str(camera / 'config/fastdds.xml')},
             output='screen'),
    ])
