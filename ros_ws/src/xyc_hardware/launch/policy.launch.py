# SPDX-License-Identifier: Apache-2.0
"""Exclusive model-policy backend using the same arm, chassis and safety controllers."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.event_handlers import OnShutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def acquire_backend(context):
    from pathlib import Path
    from xyc_hardware.teleop_transport import BackendLease
    lease = BackendLease(Path.home() / 'pi05_from_zero')
    lease.acquire(with_motors=True, with_cameras=True, recording=False)
    context.register_event_handler(
        OnShutdown(on_shutdown=[OpaqueFunction(function=lambda context: lease.close())]))


def generate_launch_description():
    return LaunchDescription([
        OpaqueFunction(function=acquire_backend),
        DeclareLaunchArgument('gripper_torque_limit_raw', default_value='100'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(PathJoinSubstitution([
                FindPackageShare('xyc_camera'), 'launch', 'cameras.launch.py'])),
            launch_arguments={'preview': 'true'}.items()),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(PathJoinSubstitution([
                FindPackageShare('xyc_hardware'), 'launch', 'motors.launch.py'])),
            launch_arguments={
                'calibrated': 'true', 'group_mode': 'true', 'auto_hold': 'false',
                'command_source': 'policy', 'chassis_mode': 'true',
                'compose_coordinator': 'false',
                'gripper_torque_limit_raw': LaunchConfiguration('gripper_torque_limit_raw'),
            }.items()),
        Node(
            package='xyc_hardware', executable='motion_replay', name='policy_control',
            parameters=[{'command_source': 'policy'}], output='screen'),
    ])
