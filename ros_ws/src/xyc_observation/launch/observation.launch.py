# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    observation = Path(get_package_share_directory('xyc_observation'))
    camera = Path(get_package_share_directory('xyc_camera'))
    hardware = Path(get_package_share_directory('xyc_hardware'))
    calibration_dir = LaunchConfiguration('calibration_dir')
    return LaunchDescription([
        DeclareLaunchArgument('start_inputs', default_value='false', choices=['true', 'false']),
        DeclareLaunchArgument('calibration_dir',
                              default_value=str(Path.home() / 'pi05_from_zero/config/calibration')),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(camera / 'launch/cameras.launch.py')),
            condition=IfCondition(LaunchConfiguration('start_inputs')),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(hardware / 'launch/motors.launch.py')),
            condition=IfCondition(LaunchConfiguration('start_inputs')),
            launch_arguments={
                'calibrated': 'true', 'calibration_dir': calibration_dir,
                'chassis_mode': 'true',
            }.items(),
        ),
        Node(
            package='xyc_observation', executable='observation_node',
            name='observation_node', namespace='/robot',
            parameters=[str(observation / 'config/observation.yaml'),
                        {'calibration_dir': calibration_dir}],
            # A 4-image frame is about 3.7 MB. Reuse the established 16 MiB
            # FastDDS transport profile; outgoing history is only one frame.
            additional_env={'FASTRTPS_DEFAULT_PROFILES_FILE': str(camera / 'config/fastdds.xml')},
            output='screen',
        ),
    ])
