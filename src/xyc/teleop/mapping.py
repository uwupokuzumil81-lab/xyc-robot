# SPDX-License-Identifier: Apache-2.0
"""Validate feedback and map leader positions to complete calibrated targets."""

from xyc.robot.calibration import arm_names
from xyc.transport.teleop import MAX_SAMPLE_AGE, SIDES, check_arm_positions


def stamp_ns(msg):
    stamp = msg.header.stamp
    return stamp.sec * 1_000_000_000 + stamp.nanosec


def follower_snapshot(states, now_ns):
    names, positions, calibration_ids, stamps = [], [], [], []
    for side in SIDES:
        msg = states.get(side)
        expected = arm_names(side)
        if (msg is None or msg.arm != side or not msg.complete or not msg.calibration_id
                or tuple(msg.names) != expected or len(msg.valid) != 7 or not all(msg.valid)):
            raise ValueError(f'{side} 从臂位置或标定无效')
        values = list(msg.position_normalized)
        check_arm_positions(side, values)
        stamp = stamp_ns(msg)
        if stamp <= 0 or not 0 <= now_ns - stamp <= 500_000_000:
            raise ValueError(f'{side} 从臂位置过期')
        names.extend(expected)
        positions.extend(values)
        calibration_ids.append(msg.calibration_id)
        stamps.append(stamp)
    if abs(stamps[0] - stamps[1]) > 150_000_000:
        raise ValueError('左右从臂读数时间相差过大')
    return names, positions, calibration_ids


def leader_snapshot(states, sides, now_ns):
    if any(side not in states for side in sides):
        return None
    positions, calibration_ids, frame_ids = {}, [], []
    for side in sides:
        msg = states[side]
        if (msg.arm != side or tuple(msg.names) != arm_names(side)
                or not msg.session_id or msg.sample_id < 1 or not msg.calibration_id):
            raise ValueError(f'{side} 主臂状态身份无效')
        stamp = stamp_ns(msg)
        if stamp <= 0 or not 0 <= now_ns - stamp <= MAX_SAMPLE_AGE * 1e9:
            return None
        # ROS exposes the fixed-length float64 field as NumPy scalars.
        values = [float(value) for value in msg.position_normalized]
        check_arm_positions(side, values)
        positions[side] = values
        calibration_ids.append(msg.calibration_id)
        frame_ids.append((msg.session_id, msg.sample_id, stamp))
    if len(set(frame_ids)) != 1:
        return None  # Wait for the other side of the same acquisition batch.
    session, sample_id, _ = frame_ids[0]
    return session, sample_id, tuple(calibration_ids), positions


class FollowTargets:
    """Keep the complete latest target; transport rate does not alter its value."""

    def __init__(self, leader, follower, *, absolute=False):
        if not leader or any(side not in SIDES for side in leader):
            raise ValueError('Expected a fixed selection of leader arms')
        if len(follower) != 14:
            raise ValueError('Exactly 14 follower positions are required')
        for side, values in leader.items():
            check_arm_positions(side, values)
        for index, side in enumerate(SIDES):
            check_arm_positions(side, follower[index * 7:(index + 1) * 7])
        self.leader_origin = {side: list(values) for side, values in leader.items()}
        self.follower_origin = list(follower)
        self.previous = list(follower)
        self.reference = list(follower)
        self.absolute = absolute

    def update(self, leader):
        if leader.keys() != self.leader_origin.keys():
            raise ValueError('本轮遥操的主臂侧别发生变化')
        targets = list(self.previous)
        for side, values in leader.items():
            check_arm_positions(side, values)
            offset = SIDES.index(side) * 7
            desired = [float(value) for value in values]
            if not self.absolute:
                desired = [self.follower_origin[offset + i] + value - self.leader_origin[side][i]
                           for i, value in enumerate(desired)]
            check_arm_positions(side, desired)  # Reject an invalid goal, never silently clip it.
            targets[offset:offset + 7] = desired
        self.reference = list(targets)
        self.previous = list(targets)
        return targets


# Units are the existing calibrated joint units, including gripper opening.
ALIGNMENT_TOLERANCES = (2., 2., 2., 2., 2., 2., 1.) * len(SIDES)
ALIGNMENT_FRAMES = 3


class AlignedFollow:
    """Engage absolute targets once the leader matches the measured follower."""

    def __init__(self):
        self.planner = None
        self.engaged = False
        self.errors = None
        self.matches = 0
        self.last_sample = None

    def update(self, frame, follower):
        if frame is None:
            self.engaged = False
            self.errors = None
            self.matches = 0
            return None
        if self.planner is None:
            self.planner = FollowTargets(frame.positions, follower, absolute=True)
        targets = self.planner.update(frame.positions)
        if self.engaged:
            return targets

        self.errors = [current - target for current, target in zip(follower, targets)]
        if frame.reused_positions or any(abs(error) > limit
                                         for error, limit in zip(self.errors, ALIGNMENT_TOLERANCES)):
            self.matches = 0
        elif frame.sample_id != self.last_sample:
            self.matches += 1
        self.last_sample = frame.sample_id
        if self.matches >= ALIGNMENT_FRAMES:
            self.engaged = True
            return targets
        return None
