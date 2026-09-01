"""
I/O helpers: parquet-based caching and result persistence.
"""

from __future__ import annotations
import hashlib
import json
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd

from utils.logging_utils import get_logger

log = get_logger(__name__)


def _cache_path(cache_dir: Path, key: str) -> Path:
    hashed = hashlib.sha256(key.encode()).hexdigest()[:16]
    return cache_dir / f"{hashed}.parquet"


def _meta_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(".json")


def load_cache(cache_dir: Path, key: str, ttl_hours: int = 24) -> pd.DataFrame | None:
    cp = _cache_path(cache_dir, key)
    mp = _meta_path(cp)
    if not cp.exists() or not mp.exists():
        return None
    meta = json.loads(mp.read_text())
    saved_at = datetime.fromisoformat(meta["saved_at"])
    if datetime.utcnow() - saved_at > timedelta(hours=ttl_hours):
        log.debug("Cache expired for key=%s", key)
        return None
    log.debug("Cache hit for key=%s", key)
    return pd.read_parquet(cp)


def save_cache(cache_dir: Path, key: str, df: pd.DataFrame) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cp = _cache_path(cache_dir, key)
    mp = _meta_path(cp)
    df.to_parquet(cp)
    mp.write_text(json.dumps({"key": key, "saved_at": datetime.utcnow().isoformat()}))
    log.debug("Saved cache key=%s", key)


def save_table(output_dir: Path, name: str, df: pd.DataFrame, fmt: str = "csv") -> Path:
    path = output_dir / "tables" / f"{name}.{fmt}"
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "csv":
        df.to_csv(path)
    elif fmt == "parquet":
        df.to_parquet(path)
    elif fmt == "xlsx":
        df.to_excel(path)
    log.info("Saved table -> %s", path)
    return path
