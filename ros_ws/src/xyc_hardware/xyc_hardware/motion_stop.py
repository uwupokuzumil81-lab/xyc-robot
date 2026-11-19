# SPDX-License-Identifier: Apache-2.0
"""Request and confirm protective stops on both motor owners."""

import time

from xyc_interfaces.srv import ProtectiveStop


def stop_service(side):
    return f'/motors/{side}/protective_stop'


class MotionFaultStop:
    def __init__(self, node):
        self.clients = {side: node.create_client(ProtectiveStop, stop_service(side))
                        for side in ('left', 'right')}
        self.pending = {}
        self.errors = []
        self.deadline = None

    def begin(self, reason):
        if self.deadline is not None:
            return
        self.deadline = time.monotonic() + 2.
        # Dispatch to both owners before waiting on either bus.
        for side, client in self.clients.items():
            try:
                self.pending[side] = client.call_async(ProtectiveStop.Request(reason=reason))
            except (OSError, RuntimeError, ValueError) as exc:
                self.errors.append(f'{side}: {exc}')

    def poll(self):
        """Return None while waiting, or the errors from the completed stop."""
        expired = time.monotonic() >= self.deadline
        for side, future in list(self.pending.items()):
            if future.done():
                try:
                    result = future.result()
                    if result is None or not result.success:
                        raise RuntimeError(result.message if result is not None else 'empty stop reply')
                except Exception as exc:
                    self.errors.append(f'{side}: {exc}')
            elif expired:
                self.clients[side].remove_pending_request(future)
                self.errors.append(f'{side}: protective stop reply timed out')
            else:
                continue
            del self.pending[side]
        return None if self.pending else tuple(self.errors)

    def wait(self):
        while (errors := self.poll()) is None:
            time.sleep(.001)
        return errors
