# SPDX-License-Identifier: Apache-2.0
"""Convert one fresh feedback batch into calibrated model coordinates."""

import hashlib
import json
import math

from xyc_interfaces.msg import ArmState
from xyc_hardware.arm_calibration import (
    FEEDBACK_ENDPOINT_TOLERANCE_TICKS, arm_names, check_hardware, normalize_tick,
)


class ArmStateConverter:
    def __init__(self, document, hardware_settings, spec):
        # Match the saved coordinate system against actual servo EEPROM before
        # publishing model inputs. The caller owns the exclusive serial port.
        check_hardware(document, hardware_settings, spec)
        self.document = document
        self.arm = document['arm']
        self.names = arm_names(self.arm)
        encoded = json.dumps(document, sort_keys=True, separators=(',', ':'),
                             ensure_ascii=False, allow_nan=False).encode()
        self.calibration_id = hashlib.sha256(encoded).hexdigest()

    def convert(self, feedback):
        state = ArmState()
        state.header = feedback.header
        state.arm = self.arm
        state.calibration_id = self.calibration_id
        state.names = list(self.names)
        by_name = {}
        duplicates = set()
        for item in feedback.motors:
            if item.name in by_name:
                duplicates.add(item.name)
            by_name[item.name] = item
        for name in self.names:
            item = by_name.get(name)
            entry = self.document['motors'][name]
            error = ''
            normalized = math.nan
            if name in duplicates:
                error = 'Duplicate motor feedback'
            elif item is None:
                error = 'Motor missing from feedback batch'
            elif (item.bus != self.arm or item.motor_id != entry['id']
                  or item.model_number != entry['model_number']):
                error = 'Motor identity differs from calibration'
            elif not item.valid or item.servo_error or item.error:
                error = item.error or f'Invalid raw feedback; servo_error={item.servo_error}'
            else:
                try:
                    normalized = normalize_tick(
                        entry, item.position_ticks,
                        feedback_tolerance_ticks=FEEDBACK_ENDPOINT_TOLERANCE_TICKS,
                    )
                except ValueError as exc:
                    error = str(exc)
            state.position_normalized.append(normalized)
            state.valid.append(not error)
            state.errors.append(error)
            # Keep the raw per-servo receipt time. An invalid numeric value has
            # no usable sample timestamp even if the batch itself was published.
            stamp = item.header.stamp if not error else type(feedback.header.stamp)()
            state.sample_stamps.append(stamp)
        state.complete = all(state.valid)
        return state
