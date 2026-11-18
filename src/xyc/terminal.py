"""Nonblocking terminal keys shared by calibration, teleoperation and evaluation."""

import os
import select
import sys


class TerminalKeys:
    """Restore terminal settings on exit; ignore arrow/function-key sequences."""

    def __init__(self, stream=None):
        self.stream = sys.stdin if stream is None else stream
        self.saved = None
        self.escape_mode = 0

    def open(self):
        if not self.stream.isatty():
            raise ValueError('请在 Windows 或 Linux 的交互终端运行')
        if sys.platform == 'win32':
            import ctypes
            from ctypes import wintypes
            self.key_state = ctypes.windll.user32.GetAsyncKeyState
            self.key_state.argtypes, self.key_state.restype = [ctypes.c_int], ctypes.c_short
            self.foreground = ctypes.windll.user32.GetForegroundWindow
            self.foreground.argtypes, self.foreground.restype = [], wintypes.HWND
            self.window = self.foreground()
        else:
            import termios
            import tty
            self.saved = termios.tcgetattr(self.stream.fileno())
            tty.setcbreak(self.stream.fileno())
        return self

    def read(self):
        if sys.platform == 'win32':
            import msvcrt
            result = []
            while msvcrt.kbhit():
                key = msvcrt.getwch()
                if key in ('\x00', '\xe0'):
                    msvcrt.getwch()
                else:
                    if key == '\x03':
                        raise KeyboardInterrupt
                    self.window = self.foreground()
                    result.append(key.lower())
            return result
        if select.select([self.stream], [], [], 0)[0]:
            return self.decode(os.read(self.stream.fileno(), 128).decode(errors='ignore'))
        return []

    def poll(self):
        return self.read()

    def decode(self, text):
        keys = []
        for char in text:
            if self.escape_mode == 1:
                self.escape_mode = 2 if char in '[O' else 0
            elif self.escape_mode == 2:
                if '@' <= char <= '~':
                    self.escape_mode = 0
            elif char == '\x1b':
                self.escape_mode = 1
            elif char == '\x03':
                raise KeyboardInterrupt
            else:
                keys.append(char.lower())
        return keys

    def is_held(self, key):
        if sys.platform != 'win32':
            return None  # Terminals report repeated keys, not key-up events.
        return self.foreground() == self.window and bool(self.key_state(ord(key.upper())) & 0x8000)

    def close(self):
        if self.saved is not None:
            import termios
            termios.tcsetattr(self.stream.fileno(), termios.TCSADRAIN, self.saved)
            self.saved = None

    def __enter__(self):
        return self.open()

    def __exit__(self, *_):
        self.close()
