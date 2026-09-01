"""
Event calendar: aggregates all macro/commodity release dates.

Sources:
- FOMC dates: hardcoded in data/fred.py
- WASDE dates: hardcoded in data/usda.py
- EIA weekly release day: Wednesdays (computed)
- NFP / CPI / PPI / PMI: loaded from FRED (release dates approximate;
  can be refined with pandas_market_calendars or econdb.com)
"""

from __future__ import annotations

import pandas as pd

from data.fred import FOMC_DATES
from data.usda import WASDE_DATES
from config.settings import Settings
from utils.logging_utils import get_logger

log = get_logger(__name__)


def _weekly_wednesdays(start: str, end: str) -> list[str]:
    """Generate all Wednesdays in a date range (EIA release day)."""
    dates = pd.date_range(start=start, end=end, freq="W-WED")
    return [d.strftime("%Y-%m-%d") for d in dates]


def _first_fridays(start: str, end: str) -> list[str]:
    """Approximate NFP release: first Friday of each month."""
    months = pd.date_range(start=start, end=end, freq="MS")
    result = []
    for m in months:
        fridays = pd.date_range(start=m, end=m + pd.offsets.MonthEnd(), freq="W-FRI")
        if len(fridays) > 0:
            result.append(fridays[0].strftime("%Y-%m-%d"))
    return result


class EventCalendar:
    """
    Centralised event calendar for Phase 7 event studies.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._calendars: dict[str, pd.DatetimeIndex] = {}
        self._build()

    def _build(self) -> None:
        start = self.settings.start_date
        end = self.settings.end_date

        self._calendars["FOMC"] = pd.DatetimeIndex(FOMC_DATES)
        self._calendars["WASDE"] = pd.DatetimeIndex(WASDE_DATES)
        self._calendars["EIA_OIL"] = pd.DatetimeIndex(_weekly_wednesdays(start, end))
        self._calendars["EIA_GAS"] = pd.DatetimeIndex(_weekly_wednesdays(start, end))
        self._calendars["NFP"] = pd.DatetimeIndex(_first_fridays(start, end))

        # CPI / PPI: released mid-month; approximated as 12th of each month
        monthly = pd.date_range(start=start, end=end, freq="MS")
        self._calendars["CPI"] = monthly + pd.offsets.Day(12)
        self._calendars["PPI"] = monthly + pd.offsets.Day(12)
        # ISM PMI: first business day of each month
        self._calendars["ISM_PMI"] = pd.date_range(start=start, end=end, freq="BMS")

        log.info(
            "EventCalendar built: %s",
            {k: len(v) for k, v in self._calendars.items()},
        )

    def get(self, event_type: str) -> pd.DatetimeIndex:
        if event_type not in self._calendars:
            raise KeyError(f"Unknown event type: {event_type}. "
                           f"Available: {list(self._calendars)}")
        return self._calendars[event_type]

    def all_calendars(self) -> dict[str, pd.DatetimeIndex]:
        return dict(self._calendars)
