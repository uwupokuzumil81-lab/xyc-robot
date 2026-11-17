# SPDX-License-Identifier: Apache-2.0
from launch import LaunchDescription
from launch.actions import Shutdown
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([Node(
        package='xyc_recording', executable='recorder', output='screen',
        on_exit=Shutdown(reason='recording process exited'),
    )])
