"""Process-owned file locks, released by the OS even after an abnormal exit."""

import os


def acquire_file_lock(path):
    """Return an owned descriptor; close it to release, leaving the inode intact."""
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.ftruncate(descriptor, 0)
        os.write(descriptor, str(os.getpid()).encode())
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise
