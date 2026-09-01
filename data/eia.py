"""
EIA (Energy Information Administration) data loader.

Uses the public EIA Open Data API v2 (no key needed for basic series).
Falls back to cached CSV download.
"""

from __future__ import annotations
import os
from typing import Dict

import pandas as pd
import requests

from config.settings import Settings
from config.symbols import EIA_SERIES
from data.base import BaseLoader
from utils.decorators import retry, timed
from utils.logging_utils import get_logger

log = get_logger(__name__)

EIA_BASE_URL = "https://api.eia.gov/v2/seriesid/{series_id}"


class EIALoader(BaseLoader):
    source_name = "eia"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        # prefer the key stored on Settings (already resolved from .env)
        self._api_key = settings.eia_api_key or os.getenv("EIA_API_KEY", "")

    @retry(times=3, delay=3.0, exceptions=(requests.RequestException,))
    def _fetch(self, key: str, **kwargs) -> pd.DataFrame:
        """
        Fetch EIA series.  Preferred method: EIA v2 REST API with API key.
        The EIA API requires registration but is free.
        Falls back to a pre-built URL pattern for known public series.
        """
        if self._api_key:
            return self._fetch_via_api(key)
        log.warning("No EIA_API_KEY set; attempting legacy endpoint for %s", key)
        return self._fetch_legacy(key)

    def _fetch_via_api(self, series_id: str) -> pd.DataFrame:
        url = f"https://api.eia.gov/v2/seriesid/{series_id}"
        resp = requests.get(
            url, params={"api_key": self._api_key}, timeout=30
        )
        resp.raise_for_status()
        data = resp.json()
        records = data["response"]["data"]
        df = pd.DataFrame(records)
        df["period"] = pd.to_datetime(df["period"])
        df = df.set_index("period").sort_index()
        df = df[["value"]].rename(columns={"value": series_id})
        df[series_id] = pd.to_numeric(df[series_id], errors="coerce")
        return df

    def _fetch_legacy(self, series_id: str) -> pd.DataFrame:
        """Legacy EIA v1 API – still functional for historical data."""
        url = "https://api.eia.gov/series/"
        params = {"series_id": series_id, "out": "json"}
        if self._api_key:
            params["api_key"] = self._api_key
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        rows = data["series"][0]["data"]  # [[period, value], ...]
        df = pd.DataFrame(rows, columns=["period", series_id])
        df["period"] = pd.to_datetime(df["period"])
        df = df.set_index("period").sort_index()
        df[series_id] = pd.to_numeric(df[series_id], errors="coerce")
        return df

    @timed
    def fetch_all(self) -> Dict[str, pd.DataFrame]:
        results: Dict[str, pd.DataFrame] = {}
        for series_id, name in EIA_SERIES.items():
            log.info("Fetching EIA series %s (%s)", series_id, name)
            try:
                results[series_id] = self.load(series_id)
            except Exception as exc:
                log.error("Failed to fetch EIA %s: %s", series_id, exc)
        return results

    @timed
    def fetch_combined(self) -> pd.DataFrame:
        all_series = self.fetch_all()
        if not all_series:
            return pd.DataFrame()
        return pd.concat(all_series.values(), axis=1).sort_index()
