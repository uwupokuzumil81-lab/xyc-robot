from xyc.robot.profile import robot_profile
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from xyc_hardware.motor_state_node import motor_launch_parameters


def motor_nodes(context):
    calibrated = LaunchConfiguration('calibrated').perform(context) == 'true'
    group_mode = LaunchConfiguration('group_mode').perform(context) == 'true'
    auto_hold = LaunchConfiguration('auto_hold').perform(context) == 'true'
    command_source = LaunchConfiguration('command_source').perform(context)
    compose_coordinator = LaunchConfiguration('compose_coordinator').perform(context) == 'true'
    chassis_mode = LaunchConfiguration('chassis_mode').perform(context) == 'true'
    if group_mode and not calibrated:
        raise ValueError('Group control requires calibrated:=true')
    calibration_dir = Path(LaunchConfiguration('calibration_dir').perform(context)).expanduser()
    nodes = []
    for bus in ('left', 'right'):
        nodes.append(Node(
            package='xyc_hardware',
            executable='motor_state_node',
            name='motor_state',
            namespace=f'/motors/{bus}',
            parameters=[motor_launch_parameters(
                bus, calibration_dir, calibrated=calibrated, group_mode=group_mode,
                chassis_mode=chassis_mode, gripper_torque_limit_raw=int(LaunchConfiguration(
                    'gripper_torque_limit_raw').perform(context)),
            )],
            output='screen',
            on_exit=Shutdown(reason=f'{bus} motor process exited'),
        ))
    if group_mode and not compose_coordinator:
        nodes.append(Node(
            package='xyc_hardware', executable='arm_group_coordinator',
            name='arm_group_coordinator', output='screen',
            parameters=[{'command_source': command_source,
                         'auto_hold': auto_hold}],
            on_exit=Shutdown(reason='arm coordinator process exited'),
        ))
    return nodes


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'calibrated', default_value='false', choices=['true', 'false'],
            description='Validate calibration against hardware and publish per-arm model states',
        ),
        DeclareLaunchArgument('group_mode', default_value='false', choices=['true', 'false']),
        DeclareLaunchArgument('command_source', default_value='teleop'),
        DeclareLaunchArgument('compose_coordinator', default_value='false', choices=['true', 'false']),
        DeclareLaunchArgument('chassis_mode', default_value='false', choices=['true', 'false']),
        DeclareLaunchArgument('auto_hold', default_value='false', choices=['true', 'false']),
        DeclareLaunchArgument(
            'gripper_torque_limit_raw', default_value='100',
            description='Continuous-gripper torque cap, 1..1000 (0.1% increments; default 100 = 10%)',
        ),
        DeclareLaunchArgument(
            'calibration_dir',
            default_value=str(robot_profile().robot_root / 'config/calibration'),
        ),
        OpaqueFunction(function=motor_nodes),
    ])
