"""
Utility decorators: timing, retry, and result caching.
"""

from __future__ import annotations
import functools
import time
from typing import Callable, TypeVar

from utils.logging_utils import get_logger

log = get_logger(__name__)

F = TypeVar("F", bound=Callable)


def timed(func: F) -> F:
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        log.debug("%s completed in %.2fs", func.__qualname__, elapsed)
        return result
    return wrapper  # type: ignore[return-value]


def retry(times: int = 3, delay: float = 2.0, exceptions: tuple = (Exception,)):
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exc: Exception | None = None
            for attempt in range(1, times + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    log.warning(
                        "%s attempt %d/%d failed: %s", func.__qualname__, attempt, times, exc
                    )
                    time.sleep(delay * attempt)
            raise RuntimeError(f"{func.__qualname__} failed after {times} attempts") from last_exc
        return wrapper  # type: ignore[return-value]
    return decorator
