"""
Phase 7: Event Study Methodology.

Standard event-study procedure:
  1. Define event dates (CPI, PPI, NFP, FOMC, EIA, USDA, PMI releases).
  2. Define estimation window [T1, T2] before the event to fit normal return model.
  3. Define event window [−pre, +post] around event date t=0.
  4. Compute abnormal returns: AR_t = R_t − E[R_t | estimation window].
  5. Aggregate: CAR (cumulative abnormal return) over event window.
  6. Test: t-test on mean CAR across all events of a given type.

Normal return model choices:
  - Market model: R_i = α + β·R_market + ε   (standard, low bias)
  - Constant mean: E[R] = mean over estimation window  (benchmark)
  - Fama-French 3-factor (extended, if data available)

We report results at multiple horizons:
  minutes: [5, 15, 30, 60, 240]  (requires intraday data)
  days:    [1, 3, 5, 10, 20]
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats

from config.settings import Settings
from utils.logging_utils import get_logger

log = get_logger(__name__)


@dataclass
class EventStudyResult:
    event_type: str
    asset: str
    window: int                 # trading days / minutes post-event
    n_events: int
    mean_car: float             # mean CAR across all events
    std_car: float
    t_stat: float
    pvalue: float
    significant: bool
    mean_car_pre: float         # mean CAR in pre-event window (placebo)
    cars: list[float] = None    # individual event CARs

    def to_dict(self) -> dict:
        return {
            "event_type": self.event_type,
            "asset": self.asset,
            "window_days": self.window,
            "n_events": self.n_events,
            "mean_CAR": round(self.mean_car, 6),
            "std_CAR": round(self.std_car, 6),
            "t_stat": round(self.t_stat, 4),
            "pvalue": round(self.pvalue, 6),
            "significant": self.significant,
            "mean_CAR_pre": round(self.mean_car_pre, 6),
        }


class EventStudy:
    """
    Phase 7: Market reaction to macro and commodity-specific events.
    """

    def __init__(
        self,
        settings: Settings,
        estimation_window: int = 120,   # trading days before event
        pre_event_days: int = 5,
    ) -> None:
        self.settings = settings
        self.estimation_window = estimation_window
        self.pre_event_days = pre_event_days

    # ── Abnormal return estimation ────────────────────────────────────────────

    def _market_model_abnormal(
        self,
        asset_returns: pd.Series,
        market_returns: pd.Series,
        event_date: pd.Timestamp,
        post_window: int,
    ) -> tuple[pd.Series, float]:
        """
        Estimate α, β on estimation window; compute AR on event window.
        Returns (AR series, pre-event mean CAR).
        """
        est_end = event_date - pd.tseries.offsets.BDay(1)
        est_start = event_date - pd.tseries.offsets.BDay(self.estimation_window)
        est_idx = asset_returns.loc[est_start:est_end].dropna().index
        est_idx = est_idx.intersection(market_returns.dropna().index)

        if len(est_idx) < 30:
            return pd.Series(dtype=float), float("nan")

        y = asset_returns[est_idx].values
        x = market_returns[est_idx].values
        # OLS
        X = np.column_stack([np.ones(len(x)), x])
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        alpha, beta_mkt = beta

        # Event window returns
        evt_start = event_date - pd.tseries.offsets.BDay(self.pre_event_days)
        evt_end = event_date + pd.tseries.offsets.BDay(post_window)
        evt_idx = asset_returns.loc[evt_start:evt_end].dropna().index
        evt_idx = evt_idx.intersection(market_returns.dropna().index)

        ar = asset_returns[evt_idx] - (alpha + beta_mkt * market_returns[evt_idx])

        # Pre-event CAR (placebo: should be ~0)
        pre_idx = ar.loc[:event_date].iloc[:-1]  # exclude event day itself
        pre_car = float(pre_idx.sum())

        return ar, pre_car

    # ── Core event study ──────────────────────────────────────────────────────

    def run(
        self,
        event_dates: Sequence[pd.Timestamp] | pd.DatetimeIndex,
        asset_returns: pd.Series,
        market_returns: pd.Series,
        event_type: str,
        post_windows: Sequence[int] | None = None,
    ) -> list[EventStudyResult]:
        post_windows = post_windows or self.settings.event_windows_days
        results = []
        for window in post_windows:
            cars = []
            pre_cars = []
            for evt_date in event_dates:
                evt_date = pd.Timestamp(evt_date)
                if evt_date not in asset_returns.index and evt_date not in market_returns.index:
                    # Find next available trading day
                    later_dates = asset_returns.index[asset_returns.index >= evt_date]
                    if len(later_dates) == 0:
                        continue
                    evt_date = later_dates[0]

                ar, pre_car = self._market_model_abnormal(
                    asset_returns, market_returns, evt_date, post_window=window
                )
                if ar.empty:
                    continue
                # CAR = sum of AR from event day (t=0) to t=window
                car_window = ar.loc[evt_date:].iloc[:window + 1]
                cars.append(float(car_window.sum()))
                pre_cars.append(pre_car)

            if len(cars) < 3:
                log.warning("Fewer than 3 events for %s / %s / window=%d", event_type, asset_returns.name, window)
                continue

            cars_arr = np.array(cars)
            t_stat, pvalue = stats.ttest_1samp(cars_arr, 0.0)
            results.append(
                EventStudyResult(
                    event_type=event_type,
                    asset=str(asset_returns.name),
                    window=window,
                    n_events=len(cars),
                    mean_car=float(cars_arr.mean()),
                    std_car=float(cars_arr.std()),
                    t_stat=float(t_stat),
                    pvalue=float(pvalue),
                    significant=pvalue < self.settings.significance_level,
                    mean_car_pre=float(np.nanmean(pre_cars)),
                    cars=cars,
                )
            )
        return results

    def run_all_events(
        self,
        event_calendars: dict[str, Sequence[pd.Timestamp]],
        asset_returns: pd.DataFrame,
        market_col: str = "SPY",
        post_windows: Sequence[int] | None = None,
    ) -> pd.DataFrame:
        """
        Run event studies for all event types × all assets.
        Returns a single consolidated DataFrame.
        """
        all_rows = []
        market_ret = asset_returns[market_col] if market_col in asset_returns else asset_returns.iloc[:, 0]
        for event_type, dates in event_calendars.items():
            for col in asset_returns.columns:
                if col == market_col:
                    continue
                results = self.run(
                    dates, asset_returns[col].rename(col), market_ret, event_type, post_windows
                )
                all_rows.extend(r.to_dict() for r in results)
        return pd.DataFrame(all_rows)
