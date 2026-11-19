# SPDX-License-Identifier: Apache-2.0
from glob import glob
from setuptools import find_packages, setup

package_name = 'xyc_hardware'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'README.md']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools', 'pyserial', 'PyYAML', 'xyc-robot>=0.2'],
    maintainer='pi5',
    maintainer_email='pi5@todo.todo',
    description='XYC robot feedback, calibration and unified teleoperation',
    license='Apache-2.0',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
            'motor_state_node = xyc_hardware.motor_state_node:main',
            'calibrate_arm = xyc.robot.calibration:main',
            'arm_group_coordinator = xyc_hardware.arm_group_coordinator:main',
            'teleop_backend = xyc_hardware.teleop_nodes:main',
            'motion_replay = xyc_hardware.motion_replay:main',
        ],
    },
)
