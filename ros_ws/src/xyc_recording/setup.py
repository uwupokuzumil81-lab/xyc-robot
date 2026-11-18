# SPDX-License-Identifier: Apache-2.0
from glob import glob
from setuptools import find_packages, setup

setup(name='xyc_recording', version='0.1.0', packages=find_packages(exclude=['test']),
      data_files=[('share/ament_index/resource_index/packages', ['resource/xyc_recording']),
                  ('share/xyc_recording', ['package.xml']),
                  ('share/xyc_recording/launch', glob('launch/*.launch.py'))],
      install_requires=['setuptools', 'xyc-robot>=0.2'], license='Apache-2.0',
      description='ROS capture relay and desktop LeRobot v3 writer, with timestamp-paired cameras',
      maintainer='pi5', maintainer_email='pi5@todo.todo',
      entry_points={'console_scripts': ['recorder = xyc_recording.recorder_node:main']})
