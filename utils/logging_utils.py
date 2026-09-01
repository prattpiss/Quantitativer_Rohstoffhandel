"""
Logging configuration for the project.
Call setup_logging() once at entry-point level.
"""

from __future__ import annotations
import logging
import sys
from pathlib import Path


def setup_logging(log_dir: Path | None = None, level: int = logging.INFO) -> None:
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    stream_handler = logging.StreamHandler(sys.stdout)
    # Force UTF-8 on Windows to handle non-ASCII characters in log messages
    if hasattr(stream_handler.stream, "reconfigure"):
        try:
            stream_handler.stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    handlers: list[logging.Handler] = [stream_handler]
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_dir / "commodities.log", encoding="utf-8"))
    logging.basicConfig(level=level, format=fmt, handlers=handlers, force=True)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
