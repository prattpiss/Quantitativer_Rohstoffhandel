"""
Abstract base class for all data loaders.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from config.settings import Settings
from utils.io_utils import load_cache, save_cache
from utils.logging_utils import get_logger

log = get_logger(__name__)


class BaseLoader(ABC):
    """
    Every data loader inherits from this.
    Implements a template method: fetch → validate → cache.
    """

    source_name: str = "base"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._cache_dir = settings.cache_dir / self.source_name

    def load(self, key: str, **kwargs) -> pd.DataFrame:
        cached = load_cache(self._cache_dir, key, self.settings.cache_ttl_hours)
        if cached is not None:
            return cached
        log.info("[%s] Fetching: %s", self.source_name, key)
        df = self._fetch(key, **kwargs)
        df = self._validate(df)
        save_cache(self._cache_dir, key, df)
        return df

    @abstractmethod
    def _fetch(self, key: str, **kwargs) -> pd.DataFrame:
        """Download raw data from the source."""

    def _validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic sanity checks; subclasses may extend."""
        if df.empty:
            raise ValueError(f"[{self.source_name}] Received empty DataFrame.")
        return df

    def invalidate_cache(self, key: str) -> None:
        """Force fresh download on next load()."""
        from utils.io_utils import _cache_path, _meta_path
        cp = _cache_path(self._cache_dir, key)
        mp = _meta_path(cp)
        for p in (cp, mp):
            if p.exists():
                p.unlink()
        log.info("[%s] Cache invalidated for key=%s", self.source_name, key)
