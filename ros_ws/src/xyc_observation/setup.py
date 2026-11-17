# SPDX-License-Identifier: Apache-2.0
from glob import glob
from setuptools import find_packages, setup

setup(
    name='xyc_observation', version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/xyc_observation']),
        ('share/xyc_observation', ['package.xml', 'README.md']),
        ('share/xyc_observation/config', glob('config/*.yaml')),
        ('share/xyc_observation/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    maintainer='pi5', maintainer_email='pi5@todo.todo', license='Apache-2.0',
    description='Validated live observations from four cameras and the 18-D robot state',
    extras_require={'test': ['pytest']},
    entry_points={'console_scripts': [
        'observation_node = xyc_observation.observation_node:main',
    ]},
)
