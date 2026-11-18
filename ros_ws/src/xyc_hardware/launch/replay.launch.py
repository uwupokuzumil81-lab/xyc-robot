"""Exclusive replay backend. Only an explicit PC start request enables torque."""

from xyc.robot.profile import robot_profile
# SPDX-License-Identifier: Apache-2.0

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction, Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution


def acquire_backend(context):
    from launch.event_handlers import OnShutdown
    from xyc.transport.lease import BackendLease
    lease = BackendLease(robot_profile().robot_root)
    lease.acquire(with_motors=True, with_cameras=False, recording=False)
    context.register_event_handler(OnShutdown(on_shutdown=[OpaqueFunction(function=lambda context: lease.close())]))


def generate_launch_description():
    return LaunchDescription([
        OpaqueFunction(function=acquire_backend),
        DeclareLaunchArgument('gripper_torque_limit_raw', default_value='100'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(PathJoinSubstitution([
                FindPackageShare('xyc_hardware'), 'launch', 'motors.launch.py'])),
            launch_arguments={'calibrated': 'true', 'group_mode': 'true', 'auto_hold': 'false',
                              'command_source': 'replay',
                              'chassis_mode': 'true', 'compose_coordinator': 'false',
                              'gripper_torque_limit_raw': LaunchConfiguration(
                                  'gripper_torque_limit_raw')}.items()),
        Node(package='xyc_hardware', executable='motion_replay', name='motion_replay', output='screen',
             on_exit=Shutdown(reason='motion control process exited')),
    ])
