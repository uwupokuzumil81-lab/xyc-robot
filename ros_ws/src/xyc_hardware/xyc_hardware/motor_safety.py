# SPDX-License-Identifier: Apache-2.0
"""Electrical protection shared by arm joints, wheels and the lift axis."""

import math
import time


MOTOR_TEMPERATURE_LIMIT_C = 55
TEMPERATURE_RECHECK_CLEAR_C = 52
TEMPERATURE_RECHECK_READS = 6
TEMPERATURE_RECHECK_INTERVAL_S = .010
MOTOR_CURRENT_RATINGS_A = {2569: (2.2, 9.8), 2825: (1.4, 4.2)}
CURRENT_NEAR_STALL_S = .080
CURRENT_OVERLOAD_S = .650
FEEDBACK_LOSS_GRACE_S = .20


class CurrentProtection:
    """Require consecutive overload samples; isolated peaks do not trip."""

    def __init__(self, model_number, *, gripper=False):
        rated, stall = MOTOR_CURRENT_RATINGS_A[model_number]
        self.limits = (('gripper severe overcurrent', 3.8, .20),) if gripper else (
            ('near-stall current', round(.8 * stall, 3), CURRENT_NEAR_STALL_S),
            ('sustained overcurrent', 2. * rated, CURRENT_OVERLOAD_S),
        )
        self.windows = {}

    def reset(self):
        self.windows.clear()

    def observe(self, current, now):
        current = abs(current)
        for cause, limit, duration in self.limits:
            if current < limit:
                self.windows.pop(cause, None)
                continue
            window = self.windows.get(cause)
            start = now
            if window is not None:
                gap = now - window[1]
                if gap >= FEEDBACK_LOSS_GRACE_S:
                    return (f'{cause} monitoring lost for {gap:.3f} s while '
                            f'current remains above {limit:g} A')
                if gap >= 0:
                    start = window[0]
            self.windows[cause] = start, now
            if now - start >= duration:
                return (f'{cause} for {now - start:.3f} s: '
                        f'current={current:.4f} A (limit {limit:g} A)')
        return ''


def confirm_temperature(sample, *, read_temperature, read_feedback, report):
    """Confirm suspect telemetry using spaced register and full-block reads."""
    reads = []
    for attempt in range(TEMPERATURE_RECHECK_READS):
        if attempt:
            time.sleep(TEMPERATURE_RECHECK_INTERVAL_S)
        temperature = read_temperature() if attempt % 2 == 0 else read_feedback()['temperature_c']
        reads.append(temperature)
        if len(reads) < 3:
            continue
        window = reads[-3:]
        if (all(math.isfinite(value) and 0 <= value <= TEMPERATURE_RECHECK_CLEAR_C
                for value in window) and max(window) - min(window) <= 2):
            report(f"Joint temperature spike cleared: bulk={sample['temperature_c']:.0f} C, direct reads={reads} C")
            return {**sample, 'temperature_c': float(max(window))}
    raise ValueError(f"Joint temperature alarm not cleared by direct reads: "
                     f"bulk={sample['temperature_c']:.0f} C, direct reads={reads} C")
