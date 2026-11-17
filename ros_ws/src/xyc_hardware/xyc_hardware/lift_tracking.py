# SPDX-License-Identifier: Apache-2.0
"""Referenced lift encoder math only; never homes or drives the lift."""
from dataclasses import dataclass
import math

PERIOD = 4096
# Homing accepts a stationary window spanning at most ten encoder ticks.
# Use the same small allowance at a measured boundary; motion targets remain
# strictly inside the verified travel. With the installed lead this is 0.32 mm.
BOUNDARY_TOLERANCE_TICKS = 10


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be finite')
    return float(value)


def _tick(value):
    if type(value) is not int or not 0 <= value < PERIOD:
        raise ValueError('Encoder tick must be an integer in 0..4095')
    return value


@dataclass(frozen=True)
class LiftGeometry:
    """All values require verification for the installed robot."""
    lead_m_per_revolution: float
    direction: int
    max_ticks_s: float
    max_sample_gap_s: float
    min_height_m: float
    max_height_m: float

    def validate(self):
        for name in ('lead_m_per_revolution', 'max_ticks_s', 'max_sample_gap_s'):
            if _number(getattr(self, name), name) <= 0:
                raise ValueError(f'{name} must be positive')
        if type(self.direction) is not int or self.direction not in (-1, 1):
            raise ValueError('direction must be +1 or -1')
        if _number(self.min_height_m, 'min height') >= _number(self.max_height_m, 'max height'):
            raise ValueError('Lift height bounds must be ordered')
        if not self.min_height_m <= 0.0 <= self.max_height_m:
            raise ValueError('Verified bounds must contain the reference height zero')
        if self.max_ticks_s * self.max_sample_gap_s >= PERIOD / 2:
            raise ValueError('Sampling bound must rule out an unseen half-turn')
        return self


class LiftTracker:
    """Track height only from an externally established, live reference.

    A lost sample, implausible speed or backwards clock permanently invalidates
    this tracker. Recovery requires a new verified reference, not reusing a tick.
    """

    def __init__(self, geometry, *, reference_id, zero_tick, reference_time_s):
        self.geometry = geometry.validate()
        if not isinstance(reference_id, str) or not reference_id.strip():
            raise ValueError('A verified session reference ID is required')
        self.reference_id = reference_id
        self.raw_tick = _tick(zero_tick)
        self.time_s = _number(reference_time_s, 'reference_time_s')
        if self.time_s < 0:
            raise ValueError('reference_time_s must not be negative')
        self.extended_delta_ticks = 0
        self.valid = True

    def update(self, raw_tick, timestamp_s):
        if not self.valid:
            raise ValueError('Reference is invalid; establish a new reference')
        try:
            raw_tick = _tick(raw_tick)
            now = _number(timestamp_s, 'timestamp_s')
            dt = now - self.time_s
            if dt <= 0 or dt > self.geometry.max_sample_gap_s:
                raise ValueError('Lift sample time is invalid or feedback was lost')
            delta = (raw_tick - self.raw_tick + PERIOD // 2) % PERIOD - PERIOD // 2
            if abs(delta) == PERIOD // 2:
                raise ValueError('Encoder half-turn is ambiguous')
            # Allow one tick of encoder quantization in a time interval.
            if abs(delta) > self.geometry.max_ticks_s * dt + 1:
                raise ValueError('Encoder speed exceeds verified limit')
            new_delta = self.extended_delta_ticks + delta
            height = self.geometry.direction * new_delta * self.geometry.lead_m_per_revolution / PERIOD
            tolerance = BOUNDARY_TOLERANCE_TICKS * self.geometry.lead_m_per_revolution / PERIOD
            if (height < self.geometry.min_height_m - tolerance - 1e-12
                    or height > self.geometry.max_height_m + tolerance + 1e-12):
                raise ValueError(
                    f'Measured height is outside verified limits: height={height:.6f} m, '
                    f'limits=[{self.geometry.min_height_m:g}, {self.geometry.max_height_m:g}] m, '
                    f'boundary_tolerance={tolerance:.6f} m, tick={raw_tick}, '
                    f'delta_ticks={new_delta}'
                )
        except (ValueError, TypeError):
            self.valid = False
            raise
        self.raw_tick = raw_tick
        self.time_s = now
        self.extended_delta_ticks = new_delta
        # Keep the complete encoder delta above, so jitter cannot ratchet the
        # reference. Only the reported height saturates within the allowance.
        return min(self.geometry.max_height_m, max(self.geometry.min_height_m, height))

    def height_to_delta_ticks(self, height_m):
        if not self.valid:
            raise ValueError('Reference is invalid; establish a new reference')
        height = _number(height_m, 'height_m')
        if not self.geometry.min_height_m <= height <= self.geometry.max_height_m:
            raise ValueError('Target height is outside verified limits')
        return round(self.geometry.direction * height * PERIOD / self.geometry.lead_m_per_revolution)
