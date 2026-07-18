"""Minimal Windows compatibility shim for nuPlan's optional map-download lock.

nuPlan v1.2 imports :mod:`fcntl` unconditionally, although it uses ``flock``
only when a missing map layer must be downloaded.  This study uses a complete,
pre-downloaded map package and a sequential worker, so no inter-process map
write can occur.  Keeping the no-op explicit makes that constraint auditable.
"""

LOCK_EX = 2
LOCK_UN = 8


def flock(_file, operation: int) -> None:
    """Accept lock/unlock calls for the single-process, read-only map setup."""

    if operation not in (LOCK_EX, LOCK_UN):
        raise ValueError(f"unsupported flock operation on Windows: {operation}")


__all__ = ["LOCK_EX", "LOCK_UN", "flock"]
