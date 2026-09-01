"""
Phase 2: Timezone alignment, trading-calendar harmonisation, and holiday handling.

Design decision:
- All prices are reindexed to a common business-day calendar (NYSE-based).
- Missing observations from calendar mismatches (e.g. commodities trading
  on days equity markets are closed) are forward-filled up to a configurable
  limit so that no look-ahead bias is introduced.
- Timezone: everything is stored as tz-naive UTC-equivalent (market dates).
"""

from __future__ import annotations
from typing import Sequence

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

from config.settings import Settings
from utils.logging_utils import get_logger

log = get_logger(__name__)


class MarketAligner:
    """
    Align multiple price series to a common trading calendar.

    Parameters
    ----------
    calendar_name : str
        pandas_market_calendars exchange name (default: "NYSE").
    max_ffill : int
        Maximum number of business days to forward-fill when a series
        has no observation (e.g. commodity closed but equity market open).
        Kept at 1 to avoid propagating stale prices too far.
    """

    def __init__(
        self,
        settings: Settings,
        calendar_name: str = "NYSE",
        max_ffill: int = 1,
    ) -> None:
        self.settings = settings
        self.max_ffill = max_ffill
        try:
            cal = mcal.get_calendar(calendar_name)
            schedule = cal.schedule(
                start_date=settings.start_date, end_date=settings.end_date
            )
            self.trading_days = mcal.date_range(schedule, frequency="1D").normalize()
            self.trading_days = pd.DatetimeIndex(
                [d.date() for d in self.trading_days], dtype="datetime64[ns]"
            )
        except Exception:
            log.warning("pandas_market_calendars not available – using business-day offset.")
            self.trading_days = pd.bdate_range(settings.start_date, settings.end_date)

    def align(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Reindex prices to the common trading calendar.

        Steps:
          1. Ensure DatetimeIndex is tz-naive.
          2. Reindex to common calendar.
          3. Forward-fill up to max_ffill to handle non-overlapping holidays.
          4. Drop rows where ALL series are NaN (e.g. global closure dates).
        """
        if prices.index.tz is not None:
            prices = prices.tz_convert("UTC").tz_localize(None)

        prices.index = pd.to_datetime(prices.index).normalize()
        prices.index = prices.index.tz_localize(None) if prices.index.tz is not None else prices.index

        # Reindex to union of trading days and series dates, then subset
        combined_idx = self.trading_days.union(prices.index)
        aligned = prices.reindex(combined_idx)
        aligned = aligned.ffill(limit=self.max_ffill)
        aligned = aligned.reindex(self.trading_days)
        aligned = aligned.dropna(how="all")

        log.info(
            "Aligned %d cols x %d dates -> %d rows after alignment.",
            aligned.shape[1], len(self.trading_days), aligned.shape[0],
        )
        return aligned
