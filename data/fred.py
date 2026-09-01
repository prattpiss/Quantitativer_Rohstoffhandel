"""
FRED (Federal Reserve Economic Data) loader.

Requires either:
  - python-dotenv + FRED_API_KEY environment variable, OR
  - fredapi package with key passed explicitly.

Falls back to direct FRED CSV download if no API key is available.
"""

from __future__ import annotations
import os
from typing import Dict

import pandas as pd

from config.settings import Settings
from config.symbols import FRED_SERIES
from data.base import BaseLoader
from utils.decorators import retry, timed
from utils.logging_utils import get_logger

log = get_logger(__name__)

# FOMC meeting dates (manually maintained for event studies)
FOMC_DATES = [
    "2020-01-29", "2020-03-03", "2020-03-15", "2020-04-29", "2020-06-10",
    "2020-07-29", "2020-09-16", "2020-11-05", "2020-12-16",
    "2021-01-27", "2021-03-17", "2021-04-28", "2021-06-16",
    "2021-07-28", "2021-09-22", "2021-11-03", "2021-12-15",
    "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15",
    "2022-07-27", "2022-09-21", "2022-11-02", "2022-12-14",
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14",
    "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
    "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-11-05", "2025-12-17",
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
    "2026-07-29",
]


class FREDLoader(BaseLoader):
    source_name = "fred"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self._api_key = settings.fred_api_key or os.getenv("FRED_API_KEY", "")
        self._fred = None
        if self._api_key:
            try:
                from fredapi import Fred
                self._fred = Fred(api_key=self._api_key)
                log.info("FRED API client initialised.")
            except ImportError:
                log.warning("fredapi not installed – falling back to direct CSV download.")

    @retry(times=3, delay=2.0)
    def _fetch(self, key: str, **kwargs) -> pd.DataFrame:
        if self._fred is not None:
            series = self._fred.get_series(key)
            return series.rename(key).to_frame()
        return self._fetch_via_csv(key)

    def _fetch_via_csv(self, series_id: str) -> pd.DataFrame:
        """Download FRED series as CSV without API key."""
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        df = pd.read_csv(url, index_col=0, parse_dates=True)
        df.columns = [series_id]
        df.replace(".", float("nan"), inplace=True)
        df[series_id] = pd.to_numeric(df[series_id], errors="coerce")
        return df

    @timed
    def fetch_all(self) -> Dict[str, pd.DataFrame]:
        """Download all configured FRED series."""
        results: Dict[str, pd.DataFrame] = {}
        for series_id, name in FRED_SERIES.items():
            log.info("Fetching FRED series %s (%s)", series_id, name)
            try:
                results[series_id] = self.load(series_id)
            except Exception as exc:
                log.error("Failed to fetch %s: %s", series_id, exc)
        return results

    def get_fomc_dates(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(FOMC_DATES)

    @timed
    def fetch_combined(self) -> pd.DataFrame:
        """Return all FRED series merged into a single DataFrame (monthly/quarterly)."""
        all_series = self.fetch_all()
        if not all_series:
            return pd.DataFrame()
        return pd.concat(all_series.values(), axis=1).sort_index()
