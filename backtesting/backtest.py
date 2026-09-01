"""
Phase 14: Walk-forward backtesting engine.

Implements strict out-of-sample validation:
  - Training window: fit strategy parameters / signal thresholds.
  - Test window: evaluate strategy without any further optimisation.
  - The test window slides forward and never looks back.

This prevents data snooping and provides a realistic estimate of live performance.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from config.settings import Settings
from utils.logging_utils import get_logger

log = get_logger(__name__)


@dataclass
class WalkForwardResult:
    in_sample_sharpe: float
    out_of_sample_sharpe: float
    out_of_sample_returns: pd.Series
    n_windows: int
    degradation: float      # IS sharpe – OOS sharpe (positive = overfitting)

    def to_dict(self) -> dict:
        return {
            "IS_sharpe": round(self.in_sample_sharpe, 3),
            "OOS_sharpe": round(self.out_of_sample_sharpe, 3),
            "degradation": round(self.degradation, 3),
            "n_windows": self.n_windows,
        }


class WalkForwardBacktester:
    """
    Phase 14: Walk-forward validation framework.

    train_window: number of trading days for in-sample estimation.
    test_window: number of trading days for out-of-sample evaluation.
    step: number of days to advance the window each iteration.
    """

    def __init__(
        self,
        settings: Settings,
        train_window: int = 756,  # 3 years
        test_window: int = 252,   # 1 year
        step: int = 63,           # quarterly refit
    ) -> None:
        self.settings = settings
        self.train_window = train_window
        self.test_window = test_window
        self.step = step

    def run(
        self,
        returns: pd.DataFrame,
        strategy_factory: Callable[[pd.DataFrame], Callable[[pd.DataFrame], pd.Series]],
    ) -> WalkForwardResult:
        """
        strategy_factory(train_data) → strategy_fn
        strategy_fn(test_data)       → daily return series

        The factory fits the strategy on train_data; the returned function
        is then applied to test_data without any further re-fitting.
        """
        all_oos_returns: list[pd.Series] = []
        is_sharpes: list[float] = []
        n = len(returns)
        window = self.train_window + self.test_window

        if n < window:
            raise ValueError(f"Insufficient data ({n} days) for walk-forward (need {window}).")

        start = 0
        while start + window <= n:
            train_data = returns.iloc[start : start + self.train_window]
            test_data = returns.iloc[start + self.train_window : start + window]

            try:
                strategy_fn = strategy_factory(train_data)
                is_ret = strategy_fn(train_data)
                oos_ret = strategy_fn(test_data)

                is_s = (is_ret.mean() * 252) / (is_ret.std() * np.sqrt(252) + 1e-8)
                is_sharpes.append(float(is_s))
                all_oos_returns.append(oos_ret)
            except Exception as exc:
                log.warning("Walk-forward window %d failed: %s", start, exc)

            start += self.step

        if not all_oos_returns:
            raise RuntimeError("No OOS windows completed successfully.")

        combined_oos = pd.concat(all_oos_returns).sort_index()
        oos_s = (combined_oos.mean() * 252) / (combined_oos.std() * np.sqrt(252) + 1e-8)
        is_s_mean = float(np.mean(is_sharpes))

        log.info(
            "Walk-forward: IS Sharpe=%.2f, OOS Sharpe=%.2f, degradation=%.2f, windows=%d",
            is_s_mean, float(oos_s), is_s_mean - float(oos_s), len(is_sharpes),
        )

        return WalkForwardResult(
            in_sample_sharpe=is_s_mean,
            out_of_sample_sharpe=float(oos_s),
            out_of_sample_returns=combined_oos,
            n_windows=len(is_sharpes),
            degradation=is_s_mean - float(oos_s),
        )
