from xyc.robot.profile import robot_profile
# SPDX-License-Identifier: Apache-2.0
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription, OpaqueFunction, SetEnvironmentVariable, Shutdown
from launch.event_handlers import OnShutdown
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def acquire_backend(context):
    from xyc.transport.lease import BackendLease
    lease = BackendLease(robot_profile().robot_root)
    lease.acquire(**{name: LaunchConfiguration(name).perform(context) == 'true'
                     for name in ('with_motors', 'with_cameras', 'recording')})
    context.register_event_handler(OnShutdown(on_shutdown=[
        OpaqueFunction(function=lambda context: lease.close())]))


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('side', default_value='right', choices=['left', 'right', 'both']),
        DeclareLaunchArgument('live', default_value='false', choices=['true', 'false']),
        DeclareLaunchArgument('gripper_torque_limit_raw', default_value='100'),
        DeclareLaunchArgument('chassis_mode', default_value='false', choices=['true', 'false']),
        DeclareLaunchArgument('follow_mode', default_value='relative', choices=['relative', 'absolute']),
        DeclareLaunchArgument('with_motors', default_value='false', choices=['true', 'false']),
        DeclareLaunchArgument('with_cameras', default_value='true', choices=['true', 'false']),
        DeclareLaunchArgument('recording', default_value='false', choices=['true', 'false']),
        OpaqueFunction(function=acquire_backend),
        SetEnvironmentVariable('OPENBLAS_NUM_THREADS', '1'),
        SetEnvironmentVariable('OMP_NUM_THREADS', '1'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(PathJoinSubstitution([
                FindPackageShare('xyc_recording'), 'launch', 'recording.launch.py',
            ])),
            condition=IfCondition(LaunchConfiguration('recording')),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(PathJoinSubstitution([
                FindPackageShare('xyc_camera'), 'launch', 'cameras.launch.py',
            ])),
            condition=IfCondition(LaunchConfiguration('with_cameras')),
        ),
        GroupAction(
            condition=IfCondition(LaunchConfiguration('with_motors')),
            actions=[IncludeLaunchDescription(
                PythonLaunchDescriptionSource(PathJoinSubstitution([
                    FindPackageShare('xyc_hardware'), 'launch', 'motors.launch.py',
                ])),
                launch_arguments={
                    'calibrated': 'true',
                    'group_mode': 'true',
                    'command_source': 'teleop',
                    'auto_hold': 'false',
                    'compose_coordinator': 'true',
                    'chassis_mode': LaunchConfiguration('chassis_mode'),
                    'gripper_torque_limit_raw': LaunchConfiguration('gripper_torque_limit_raw'),
                }.items(),
            )],
        ),
        Node(package='xyc_hardware', executable='leader_state_bridge', output='screen',
             parameters=[{'recording': ParameterValue(LaunchConfiguration('recording'), value_type=bool),
                          'chassis_mode': ParameterValue(LaunchConfiguration('chassis_mode'), value_type=bool)}],
             on_exit=Shutdown(reason='leader bridge process exited')),
        Node(package='xyc_hardware', executable='teleop_controller', output='screen', parameters=[{
            'side': LaunchConfiguration('side'),
            'live': ParameterValue(LaunchConfiguration('live'), value_type=bool),
            'recording': ParameterValue(LaunchConfiguration('recording'), value_type=bool),
            'follow_mode': LaunchConfiguration('follow_mode'),
            'start_when_ready': ParameterValue(LaunchConfiguration('with_motors'), value_type=bool),
        }], on_exit=Shutdown(reason='teleop controller process exited')),
    ])
