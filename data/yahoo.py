"""
Yahoo Finance data loader.

Uses yfinance under the hood.  Returns adjusted close prices by default,
but can optionally return full OHLCV data.
"""

from __future__ import annotations
from typing import Sequence

import pandas as pd
import yfinance as yf

from config.settings import Settings
from data.base import BaseLoader
from utils.decorators import retry, timed
from utils.logging_utils import get_logger

log = get_logger(__name__)


class YahooLoader(BaseLoader):
    source_name = "yahoo"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)

    @retry(times=3, delay=2.0, exceptions=(Exception,))
    def _fetch(self, key: str, **kwargs) -> pd.DataFrame:
        # key encodes tickers + column; not used directly here
        return pd.DataFrame()  # placeholder; use fetch_prices instead

    @timed
    def fetch_prices(
        self,
        tickers: Sequence[str],
        start: str | None = None,
        end: str | None = None,
        column: str = "Close",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Download close prices for a list of tickers.
        Primary: yfinance Ticker.history() (per-ticker, robust to tz issues).
        Returns a DataFrame with tickers as columns and DatetimeIndex.
        """
        start = start or self.settings.start_date
        end = end or self.settings.end_date
        cache_key = f"prices_{'_'.join(sorted(tickers))}_{start}_{end}_{column}"

        if use_cache:
            from utils.io_utils import load_cache, save_cache
            cached = load_cache(self._cache_dir, cache_key, self.settings.cache_ttl_hours)
            if cached is not None:
                return cached

        log.info("Downloading %d tickers from Yahoo Finance …", len(tickers))
        df = self._fetch_per_ticker(list(tickers), start, end)

        if df.empty:
            raise ValueError("yfinance returned empty data for all tickers.")

        if use_cache:
            from utils.io_utils import save_cache
            save_cache(self._cache_dir, cache_key, df)

        return df

    def _fetch_per_ticker(
        self,
        tickers: list[str],
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """
        Download each ticker individually via Ticker.history().
        More robust than batch download for futures and index symbols.
        """
        frames: dict[str, pd.Series] = {}
        for t in tickers:
            try:
                hist = yf.Ticker(t).history(start=start, end=end, auto_adjust=True)
                if hist.empty:
                    log.warning("No data returned for %s", t)
                    continue
                col = "Close" if "Close" in hist.columns else hist.columns[0]
                s = hist[col].copy()
                # yfinance 1.x returns tz-aware index; strip tz for uniformity
                if hasattr(s.index, "tz") and s.index.tz is not None:
                    s.index = s.index.tz_localize(None)
                else:
                    s.index = pd.to_datetime(s.index)
                frames[t] = s
                log.debug("Downloaded %s: %d rows", t, len(s))
            except Exception as exc:
                log.warning("Failed to download %s: %s", t, exc)

        if not frames:
            return pd.DataFrame()
        return pd.DataFrame(frames).sort_index()

    @timed
    def fetch_ohlcv(
        self,
        ticker: str,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        """Full OHLCV for a single ticker."""
        start = start or self.settings.start_date
        end = end or self.settings.end_date
        cache_key = f"ohlcv_{ticker}_{start}_{end}"

        from utils.io_utils import load_cache, save_cache
        cached = load_cache(self._cache_dir, cache_key, self.settings.cache_ttl_hours)
        if cached is not None:
            return cached

        raw = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)
        if hasattr(raw.index, "tz") and raw.index.tz is not None:
            raw.index = raw.index.tz_localize(None)
        else:
            raw.index = pd.to_datetime(raw.index)
        save_cache(self._cache_dir, cache_key, raw)
        return raw

    @timed
    def fetch_intraday(
        self,
        tickers: Sequence[str],
        period: str = "1mo",
        interval: str = "5m",
    ) -> pd.DataFrame:
        """
        Intraday data via yfinance.
        Note: yfinance limits intraday history to 60 days for 1m, less for longer bars.
        """
        log.info("Downloading intraday (%s) data for %d tickers", interval, len(tickers))
        frames = {}
        for t in tickers:
            raw = yf.download(
                tickers=t,
                period=period,
                interval=interval,
                auto_adjust=True,
                progress=False,
            )
            if not raw.empty:
                raw.index = pd.to_datetime(raw.index).tz_localize(None)
                frames[t] = raw["Close"]
        return pd.DataFrame(frames)
