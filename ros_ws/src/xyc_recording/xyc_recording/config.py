# SPDX-License-Identifier: Apache-2.0
"""Recording settings shared by the ROS source and the desktop writer."""

import re


def checked_config(config):
    allowed = {'dataset', 'task', 'fps', 'episode_frames', 'scope', 'resume', 'num_episodes'}
    if not isinstance(config, dict) or set(config) - allowed:
        raise ValueError('Unknown recording setting')
    out = {'dataset': 'xyc/demonstrations', 'task': '', 'fps': 30,
           'episode_frames': 900, 'scope': 'mobile', 'resume': False, 'num_episodes': 0, **config}
    if (not isinstance(out['dataset'], str)
            or not re.fullmatch(r'[A-Za-z0-9_-]+(?:/[A-Za-z0-9_-]+)?', out['dataset'])
            or len(out['dataset']) > 100):
        raise ValueError('dataset must be name or owner/name, using letters, numbers, _ and -')
    if not isinstance(out['task'], str) or not out['task'].strip() or len(out['task'].encode()) > 512:
        raise ValueError('Provide a task instruction of 1..512 UTF-8 bytes')
    if type(out['fps']) is not int or not 1 <= out['fps'] <= 30:
        raise ValueError('fps must be an integer in 1..30')
    if type(out['episode_frames']) is not int or not 2 <= out['episode_frames'] <= 108000:
        raise ValueError('episode_frames must be in 2..108000')
    if out['scope'] not in ('arms', 'mobile') or type(out['resume']) is not bool:
        raise ValueError('scope must be mobile/arms; resume must be boolean')
    if type(out['num_episodes']) is not int or not 0 <= out['num_episodes'] <= 10000:
        raise ValueError('num_episodes must be in 0..10000')
    return out
