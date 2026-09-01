"""Daten-Layer: yfinance Einzel-Ticker-Download mit lokalem CSV-Cache."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data_cache" / "yahoo"
CACHE_MAX_AGE_H = 12.0

_MEM: dict[str, pd.DataFrame] = {}


def _safe(ticker: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in ticker)


def _normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    idx = pd.to_datetime(df.index, errors="coerce")
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    df = df.copy()
    df.index = idx.normalize()
    df = df[~df.index.isna()]
    return df[~df.index.duplicated(keep="last")].sort_index()


def _fetch(ticker: str) -> pd.DataFrame:
    import yfinance as yf
    for attempt in range(3):
        try:
            raw = yf.Ticker(ticker).history(period="max", auto_adjust=True)
            if raw is not None and len(raw) > 0:
                return _normalize_index(raw)
        except Exception as exc:  # noqa: BLE001
            if attempt == 2:
                print(f"    ! {ticker}: {type(exc).__name__} {exc}")
        time.sleep(1.5 * (attempt + 1))
    return pd.DataFrame()


def ohlcv(ticker: str, force: bool = False) -> pd.DataFrame:
    """OHLCV-Frame (auto_adjust) mit tz-naivem, normalisiertem DatetimeIndex."""
    if ticker in _MEM and not force:
        return _MEM[ticker]
    CACHE.mkdir(parents=True, exist_ok=True)
    fp = CACHE / f"{_safe(ticker)}.csv"
    fresh = fp.exists() and (time.time() - fp.stat().st_mtime) < CACHE_MAX_AGE_H * 3600
    df = pd.DataFrame()
    if fresh and not force:
        try:
            df = _normalize_index(pd.read_csv(fp, index_col=0))
        except Exception:  # noqa: BLE001
            df = pd.DataFrame()
    if df.empty:
        df = _fetch(ticker)
        if not df.empty:
            df.to_csv(fp)
        elif fp.exists():  # Netzwerkfehler -> alten Cache verwenden
            try:
                df = _normalize_index(pd.read_csv(fp, index_col=0))
            except Exception:  # noqa: BLE001
                df = pd.DataFrame()
    _MEM[ticker] = df
    return df


def close(ticker: str) -> pd.Series:
    df = ohlcv(ticker)
    if df.empty or "Close" not in df.columns:
        return pd.Series(dtype=float, name=ticker)
    return pd.to_numeric(df["Close"], errors="coerce").dropna().rename(ticker)


def volume(ticker: str) -> pd.Series:
    df = ohlcv(ticker)
    if df.empty or "Volume" not in df.columns:
        return pd.Series(dtype=float, name=ticker)
    return pd.to_numeric(df["Volume"], errors="coerce").rename(f"{ticker}_Vol")


def close_panel(tickers: list[str], min_obs: int = 250,
                verbose: bool = True) -> pd.DataFrame:
    """Close-Panel; Ticker mit zu wenig Historie werden verworfen."""
    cols, dropped = {}, []
    for t in tickers:
        s = close(t)
        if len(s) >= min_obs:
            cols[t] = s
        else:
            dropped.append(t)
    if verbose and dropped:
        print(f"    verworfen (<{min_obs} Obs): {', '.join(dropped)}")
    if not cols:
        return pd.DataFrame()
    return pd.DataFrame(cols).sort_index()


def align(panel: pd.DataFrame, how: str = "inner") -> pd.DataFrame:
    """Auf gemeinsamen Handelstagen ausrichten; Lücken vorwärts füllen."""
    if panel.empty:
        return panel
    if how == "inner":
        return panel.dropna(how="any")
    return panel.ffill().dropna(how="all")


def log_returns(panel: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Log-Renditen; nicht-positive Preise (z. B. WTI im April 2020) werden maskiert."""
    return np.log(panel.where(panel > 0)).diff().replace([np.inf, -np.inf], np.nan)


def availability(tickers: list[str]) -> pd.DataFrame:
    """Datenverfügbarkeits-Übersicht (für Look-Ahead-/Survivorship-Diskussion)."""
    rows = []
    for t in tickers:
        s = close(t)
        if len(s) == 0:
            rows.append({"Ticker": t, "Start": "—", "Ende": "—", "Obs": 0, "Jahre": 0.0})
            continue
        rows.append({
            "Ticker": t,
            "Start": s.index[0].date().isoformat(),
            "Ende": s.index[-1].date().isoformat(),
            "Obs": int(len(s)),
            "Jahre": round((s.index[-1] - s.index[0]).days / 365.25, 1),
        })
    return pd.DataFrame(rows)
