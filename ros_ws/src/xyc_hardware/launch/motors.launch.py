# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def motor_nodes(context):
    package = Path(get_package_share_directory('xyc_hardware'))
    parameters = str(package / 'config' / 'motors.yaml')
    calibrated = LaunchConfiguration('calibrated').perform(context) == 'true'
    group_mode = LaunchConfiguration('group_mode').perform(context) == 'true'
    auto_hold = LaunchConfiguration('auto_hold').perform(context) == 'true'
    command_source = LaunchConfiguration('command_source').perform(context)
    compose_coordinator = LaunchConfiguration('compose_coordinator').perform(context) == 'true'
    chassis_mode = LaunchConfiguration('chassis_mode').perform(context) == 'true'
    if group_mode and not calibrated:
        raise ValueError('Group control requires calibrated:=true')
    streaming_control = group_mode and command_source in ('teleop', 'replay', 'policy')
    calibration_dir = Path(LaunchConfiguration('calibration_dir').perform(context)).expanduser()
    nodes = []
    for bus in ('left', 'right'):
        nodes.append(Node(
            package='xyc_hardware',
            executable='motor_state_node',
            name='motor_state',
            namespace=f'/motors/{bus}',
            parameters=[parameters, {
                'calibration_file': str(calibration_dir / f'{bus}.json') if calibrated else '',
                'group_mode': group_mode,
                'chassis_mode': chassis_mode and bus == 'left',
                'gripper_torque_limit_raw': int(LaunchConfiguration(
                    'gripper_torque_limit_raw').perform(context)),
                **({'poll_hz': 60.0, 'read_timeout_s': .014} if group_mode else {}),
            }],
            output='screen',
            on_exit=Shutdown(reason=f'{bus} motor process exited'),
        ))
    if group_mode and not compose_coordinator:
        nodes.append(Node(
            package='xyc_hardware', executable='arm_group_coordinator',
            name='arm_group_coordinator', output='screen',
            parameters=[{'command_source': command_source,
                         'streaming_targets': streaming_control,
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
            default_value=str(Path.home() / 'pi05_from_zero/config/calibration'),
        ),
        OpaqueFunction(function=motor_nodes),
    ])
