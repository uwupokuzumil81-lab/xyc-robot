"""Shared ternary keyboard intent for teleoperation and evaluation preparation."""

import threading
import time

MOTIONS = {'w': (1, 1.), 's': (1, -1.), 'a': (2, 1.), 'd': (2, -1.),
           'j': (3, 1.), 'l': (3, -1.), 'i': (0, 1.), 'k': (0, -1.)}


class MobileKeyboard:
    def __init__(self, *, clock=time.monotonic):
        self.clock = clock
        self.lock = threading.RLock()
        self.reset()

    def reset(self):
        with self.lock:
            self.commands, self.until, self.held = [0.] * 4, [0.] * 4, [None] * 4

    def press(self, key, now=None):
        key = key.lower()
        with self.lock:
            if key in ('h', ' ', 'q', '\x03'):
                self.reset()
            elif key in MOTIONS:
                axis, value = MOTIONS[key]
                self.commands[axis], self.held[axis] = value, key
                self.until[axis] = (self.clock() if now is None else now) + .25

    def poll(self, keys):
        events = [key.lower() for key in keys.read()]
        continued = []
        with self.lock:
            for axis, key in enumerate(self.held):
                if key is None or key in events:
                    continue
                held = keys.is_held(key)
                if held is True:
                    continued.append(key)
                elif held is False:
                    self.until[axis], self.held[axis] = 0., None
        # A fresh reversal or stop must override synthesized key continuation.
        return continued + events

    def values(self, now=None):
        now = self.clock() if now is None else now
        with self.lock:
            return [value if now < expires else 0. for value, expires in zip(self.commands, self.until)]
