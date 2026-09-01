"""Mehrbein-Simulationsengine für Long/Short-, Pair- und Spread-Strukturen.

Konventionen
------------
* Signal aus dem Schluss von Tag t-1 bestimmt die Position, die die Rendite
  von Tag t vereinnahmt (kein Look-Ahead).
* Zwei-Bein-Strukturen werden auf Brutto-Exposure 1 normiert (je 0.5).
* Stop-Loss wirkt auf die kumulierte Trade-Rendite je Exposure-Einheit und
  wird zum Schlusskurs ausgeführt. Das weicht bewusst vom intraday-Stop in
  Report 00 ab, weil ein Tagestief für eine Zwei-Bein-Position nicht
  definiert ist; der Vergleich innerhalb dieses Reports bleibt konsistent.
* Nach einem Stop bleibt die Strategie flach, bis der Signalzustand wechselt.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .stats_tools import perf_metrics

# Zielgewichte (Bein A = JETS, Bein B = Öl) für Signalzustand s = +1.
STRUCTURES: dict[str, tuple[float, float]] = {
    "Long-only": (1.0, 0.0),
    "Long/Short": (1.0, 0.0),
    "Pair (Long JETS / Short Öl)": (0.5, -0.5),
    "Spread (beide Long)": (0.5, 0.5),
}
# Long-only unterdrückt den Short-Zustand, alle anderen lassen ihn zu.
ALLOW_SHORT = {"Long-only": False, "Long/Short": True,
               "Pair (Long JETS / Short Öl)": True, "Spread (beide Long)": True}

STOPS = {"kein Stop": ("none", 0.0),
         "fest 8 %": ("fixed", 0.08),
         "Trailing 10 %": ("trailing", 0.10)}

SIZINGS = ("fest 1.0", "Vol-Target 15 %", "Stress-skaliert")

COSTS = {"ohne Kosten": 0.0, "10 bp je Umsatz": 10.0}


@dataclass
class SimResult:
    rets: pd.Series
    equity: pd.Series
    exposure: pd.Series
    trades: pd.DataFrame

    @property
    def n_trades(self) -> int:
        return len(self.trades)


def target_weights(signal: pd.Series, structure: str) -> tuple[np.ndarray, np.ndarray]:
    s = signal.to_numpy(float)
    if not ALLOW_SHORT[structure]:
        s = np.clip(s, 0.0, None)
    wa, wb = STRUCTURES[structure]
    return s * wa, s * wb


def vol_target_size(rets_a: pd.Series, target: float = 0.15,
                    window: int = 20, lo: float = 0.25, hi: float = 1.5) -> pd.Series:
    rv = rets_a.rolling(window).std() * np.sqrt(252)
    return (target / rv.where(rv > 0)).clip(lo, hi).shift(1).fillna(1.0)


def stress_size(stress_pct: pd.Series, index: pd.Index) -> pd.Series:
    """Skaliert linear mit dem Stress-Perzentil: 0 % Stress -> 1.0, 100 % -> 0."""
    s = stress_pct.reindex(index).ffill().fillna(50.0) / 100.0
    return (1.0 - s).clip(0.05, 1.0).shift(1).fillna(1.0)


def simulate(rets_a: pd.Series, rets_b: pd.Series, signal: pd.Series,
             structure: str = "Long-only", stop: str = "none",
             stop_level: float = 0.0, size: pd.Series | float = 1.0,
             cost_bps: float = 0.0,
             price_a: pd.Series | None = None) -> SimResult:
    idx = rets_a.index
    ra = rets_a.to_numpy(float)
    rb = rets_b.reindex(idx).fillna(0.0).to_numpy(float)
    wa, wb = target_weights(signal.reindex(idx).fillna(0.0), structure)
    k = (size.reindex(idx).ffill().fillna(1.0).to_numpy(float)
         if isinstance(size, pd.Series) else np.full(len(idx), float(size)))
    wa, wb = wa * k, wb * k
    pa = (price_a.reindex(idx).to_numpy(float) if price_a is not None
          else np.full(len(idx), np.nan))
    state = np.sign(np.where(np.abs(wa) > 0, wa, wb))

    n = len(idx)
    out = np.zeros(n)
    expo = np.zeros(n)
    pos_a = pos_b = 0.0
    cum = peak = 1.0
    entry_i = -1
    blocked = 0.0          # Signalzustand, der nach einem Stop gesperrt bleibt
    trades: list[dict] = []

    def close_trade(i: int, reason: str) -> None:
        nonlocal pos_a, pos_b, cum, peak, entry_i
        trades.append({
            "Entry": idx[entry_i], "Exit": idx[i], "Tage": int(i - entry_i),
            "EntryPx": pa[entry_i], "ExitPx": pa[i], "Return": cum - 1.0,
            "Size": abs(pos_a) + abs(pos_b), "Richtung": "Long" if state[entry_i] > 0 else "Short",
            "Grund": reason,
        })
        pos_a = pos_b = 0.0
        cum = peak = 1.0
        entry_i = -1

    for t in range(1, n):
        gross = abs(pos_a) + abs(pos_b)
        if gross > 0:
            r = pos_a * ra[t] + pos_b * rb[t]
            out[t] = r
            expo[t] = gross
            cum *= 1.0 + r / gross
            peak = max(peak, cum)

        if gross > 0 and stop != "none":
            drop = (cum - 1.0) if stop == "fixed" else (cum / peak - 1.0)
            if drop <= -stop_level:
                blocked = state[t]
                close_trade(t, "Stop-Loss")
                gross = 0.0

        ta, tb = wa[t], wb[t]
        if blocked != 0.0:
            if state[t] == blocked:
                ta = tb = 0.0
            else:
                blocked = 0.0

        turnover = abs(ta - pos_a) + abs(tb - pos_b)
        if turnover > 1e-12:
            out[t] -= turnover * cost_bps / 1e4
            if entry_i >= 0 and ta == 0.0 and tb == 0.0:
                close_trade(t, "Signal-Exit")
            elif entry_i >= 0 and state[t] != state[entry_i]:
                close_trade(t, "Signal-Wechsel")
            if entry_i < 0 and (ta != 0.0 or tb != 0.0):
                entry_i = t
                cum = peak = 1.0
            pos_a, pos_b = ta, tb

    if entry_i >= 0:
        close_trade(n - 1, "offen")

    r = pd.Series(out, index=idx, name="rets")
    return SimResult(r, (1 + r).cumprod(), pd.Series(expo, index=idx, name="exposure"),
                     pd.DataFrame(trades))


def trade_quality(trades: pd.DataFrame, lo: pd.Timestamp | None = None,
                  hi: pd.Timestamp | None = None) -> dict:
    if trades is None or trades.empty:
        return {"Trades": 0, "WinRate": np.nan, "ProfitFaktor": np.nan,
                "Ø Return": np.nan, "Ø Gewinn": np.nan, "Ø Verlust": np.nan,
                "Ø Tage": np.nan, "Max Konsek. Verluste": 0}
    t = trades
    if lo is not None:
        t = t[t["Exit"] >= lo]
    if hi is not None:
        t = t[t["Exit"] <= hi]
    if t.empty:
        return trade_quality(None)
    r = t["Return"]
    gains, losses = r[r > 0], -r[r < 0]
    streak = mx = 0
    for v in r:
        streak = streak + 1 if v <= 0 else 0
        mx = max(mx, streak)
    return {"Trades": len(t), "WinRate": float((r > 0).mean()),
            "ProfitFaktor": float(gains.sum() / losses.sum()) if losses.sum() > 0 else np.nan,
            "Ø Return": float(r.mean()),
            "Ø Gewinn": float(gains.mean()) if len(gains) else np.nan,
            "Ø Verlust": float(-losses.mean()) if len(losses) else np.nan,
            "Ø Tage": float(t["Tage"].mean()),
            "Max Konsek. Verluste": int(mx)}


def period_metrics(res: SimResult, periods: dict[str, tuple[pd.Timestamp, pd.Timestamp]]) -> dict:
    """Kennzahlen je Zeitfenster aus einer einzigen durchgehenden Simulation."""
    row: dict = {}
    for name, (a, b) in periods.items():
        seg = res.rets.loc[a:b]
        m = perf_metrics(seg) if len(seg) > 20 else {}
        q = trade_quality(res.trades, a, b)
        row[f"{name} Sharpe"] = m.get("Sharpe", np.nan)
        row[f"{name} CAGR"] = m.get("CAGR", np.nan)
        row[f"{name} MaxDD"] = m.get("MaxDD", np.nan)
        row[f"{name} Calmar"] = m.get("Calmar", np.nan)
        row[f"{name} Sortino"] = m.get("Sortino", np.nan)
        row[f"{name} Trades"] = q["Trades"]
        row[f"{name} WinRate"] = q["WinRate"]
        row[f"{name} ProfitFaktor"] = q["ProfitFaktor"]
    return row
