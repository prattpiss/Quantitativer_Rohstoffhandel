"""
USDA (United States Department of Agriculture) data loader.

Key USDA data releases:
- WASDE (World Agricultural Supply and Demand Estimates) – monthly
- Crop Progress Reports – weekly

Uses the USDA QuickStats API (free, requires registration).
Falls back to scraping public release calendars for event dates.
"""

from __future__ import annotations
import os
from typing import Dict, List

import pandas as pd
import requests

from config.settings import Settings
from data.base import BaseLoader
from utils.decorators import retry, timed
from utils.logging_utils import get_logger

log = get_logger(__name__)

# WASDE release dates 2020-2026 (for event study calendar)
WASDE_DATES = [
    "2020-01-10", "2020-02-11", "2020-03-10", "2020-04-09", "2020-05-12",
    "2020-06-11", "2020-07-10", "2020-08-12", "2020-09-11", "2020-10-09",
    "2020-11-10", "2020-12-10",
    "2021-01-12", "2021-02-09", "2021-03-09", "2021-04-09", "2021-05-12",
    "2021-06-10", "2021-07-09", "2021-08-12", "2021-09-10", "2021-10-08",
    "2021-11-09", "2021-12-09",
    "2022-01-12", "2022-02-09", "2022-03-09", "2022-04-08", "2022-05-12",
    "2022-06-10", "2022-07-12", "2022-08-12", "2022-09-12", "2022-10-12",
    "2022-11-09", "2022-12-09",
    "2023-01-12", "2023-02-08", "2023-03-08", "2023-04-11", "2023-05-11",
    "2023-06-09", "2023-07-12", "2023-08-11", "2023-09-12", "2023-10-11",
    "2023-11-09", "2023-12-08",
    "2024-01-12", "2024-02-08", "2024-03-08", "2024-04-11", "2024-05-10",
    "2024-06-12", "2024-07-11", "2024-08-12", "2024-09-12", "2024-10-11",
    "2024-11-08", "2024-12-10",
    "2025-01-10", "2025-02-11", "2025-03-11", "2025-04-10", "2025-05-12",
    "2025-06-11", "2025-07-11", "2025-08-12",
]


class USDALoader(BaseLoader):
    source_name = "usda"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self._api_key = os.getenv("USDA_API_KEY", "")
        self._base_url = "https://quickstats.nass.usda.gov/api/api_GET/"

    @retry(times=3, delay=2.0, exceptions=(requests.RequestException,))
    def _fetch(self, key: str, **kwargs) -> pd.DataFrame:
        """
        Fetch from USDA NASS QuickStats.
        key format: "commodity|statisticcat|unit"
        """
        if not self._api_key:
            log.warning("No USDA_API_KEY set – WASDE data unavailable via API.")
            return pd.DataFrame()
        parts = key.split("|")
        params = {
            "key": self._api_key,
            "commodity_desc": parts[0] if len(parts) > 0 else "CORN",
            "statisticcat_desc": parts[1] if len(parts) > 1 else "PRODUCTION",
            "unit_desc": parts[2] if len(parts) > 2 else "BU",
            "agg_level_desc": "NATIONAL",
            "format": "JSON",
        }
        resp = requests.get(self._base_url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        df = pd.DataFrame(data.get("data", []))
        if df.empty:
            return df
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        df["Value"] = pd.to_numeric(df["Value"].str.replace(",", ""), errors="coerce")
        df = df[["year", "Value"]].dropna()
        df.index = pd.to_datetime(df["year"].astype(int).astype(str) + "-01-01")
        return df[["Value"]].rename(columns={"Value": key})

    def get_wasde_dates(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(WASDE_DATES)
