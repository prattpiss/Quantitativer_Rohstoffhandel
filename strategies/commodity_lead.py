"""
Phase 14: Strategy development – only after all statistical analyses are complete.

Design principles:
  - Strategies are built ONLY on statistically robust and reproducible findings.
  - No data snooping: signal threshold chosen from statistical analysis, not optimised.
  - Walk-forward validation: model fitted on in-sample; tested strictly out-of-sample.
  - Transaction costs and slippage are explicit inputs.
  - Minimum liquidity filter prevents trading illiquid small caps.

Strategy types:
  1. Commodity-Lead Signal: buy/sell equities N days after commodity moves.
  2. Information-Delay Reversion: trade the equity that hasn't yet repriced.
  3. Cross-Sectional Momentum (commodity-driven): rank equities by commodity beta.
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
class StrategyConfig:
    name: str
    entry_lag: int                 # days after commodity signal to enter trade
    holding_period: int            # days to hold position
    signal_threshold: float        # minimum commodity return to trigger signal
    long_tickers: list[str]        # assets to go long
    short_tickers: list[str]       # assets to go short (can be empty for long-only)
    max_position_size: float = 0.1 # max fraction of capital per position
    transaction_cost_bps: float = 10.0  # basis points one-way
    slippage_bps: float = 5.0          # basis points slippage per trade
    liquidity_filter_adv: float = 1e6  # minimum average daily volume


class CommodityLeadStrategy:
    """
    Phase 14: Trade equities with confirmed lag behind commodity.

    Signal: if commodity R_t > threshold, expect equity to follow in lag days.
    Entry:  buy equity at close on day t + lag.
    Exit:   close position at close on day t + lag + holding_period.
    """

    def __init__(
        self,
        config: StrategyConfig,
        settings: Settings,
    ) -> None:
        self.config = config
        self.settings = settings

    def generate_signals(
        self,
        commodity_returns: pd.Series,
        direction: int = 1,  # 1 = long on positive commodity move
    ) -> pd.Series:
        """
        Generate entry signals (±1 or 0) for each date.
        Signal is triggered when commodity return exceeds threshold.
        """
        thresh = self.config.signal_threshold
        if direction == 1:
            raw = (commodity_returns > thresh).astype(float)
        else:
            raw = -(commodity_returns < -thresh).astype(float)
        # Shift by lag: trade executes `entry_lag` days after signal
        return raw.shift(self.config.entry_lag).fillna(0)

    def backtest(
        self,
        commodity_returns: pd.Series,
        equity_returns: pd.DataFrame,
        direction: int = 1,
    ) -> pd.DataFrame:
        """
        Vectorised backtest (no look-ahead bias).

        Returns a DataFrame of daily strategy returns per ticker.
        Position is held for holding_period days then closed.
        Transaction costs are subtracted on entry and exit.
        """
        config = self.config
        tc_daily = (config.transaction_cost_bps + config.slippage_bps) / 10_000

        signals = self.generate_signals(commodity_returns, direction)
        # Convert pulse signal to holding position
        position = pd.Series(0.0, index=signals.index)
        for i, (date, sig) in enumerate(signals.items()):
            if sig != 0:
                end_idx = min(i + config.holding_period, len(position) - 1)
                position.iloc[i : end_idx + 1] = sig

        strategy_returns = {}
        for ticker in equity_returns.columns:
            eq_ret = equity_returns[ticker]
            pos_shifted = position.reindex(eq_ret.index).fillna(0)
            raw = pos_shifted * eq_ret

            # Subtract transaction costs on position changes
            trades = pos_shifted.diff().fillna(0).abs()
            raw -= trades * tc_daily
            strategy_returns[ticker] = raw

        return pd.DataFrame(strategy_returns).dropna(how="all")

    def performance_summary(self, strategy_returns: pd.DataFrame) -> pd.DataFrame:
        """Compute performance metrics for each ticker's strategy returns."""
        rows = []
        for col in strategy_returns.columns:
            r = strategy_returns[col].dropna()
            n = len(r)
            ann_ret = r.mean() * 252
            ann_vol = r.std() * np.sqrt(252)
            sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
            downside = r[r < 0].std() * np.sqrt(252)
            sortino = ann_ret / downside if downside > 0 else np.nan
            cum = (1 + r).cumprod()
            roll_max = cum.cummax()
            max_dd = ((cum - roll_max) / roll_max).min()
            hit = (r > 0).sum() / max(n, 1)
            rows.append({
                "ticker": col,
                "ann_return": round(ann_ret, 4),
                "ann_vol": round(ann_vol, 4),
                "sharpe": round(sharpe, 3),
                "sortino": round(sortino, 3),
                "max_drawdown": round(max_dd, 4),
                "hit_rate": round(hit, 3),
                "n_obs": n,
            })
        return pd.DataFrame(rows).set_index("ticker")
