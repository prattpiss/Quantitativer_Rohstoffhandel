"""Baseline-Strategie: JETS Long, gesteuert vom Öl-/Energie-Basket-Signal.

Konventionen (Look-Ahead-Vermeidung):
  * Ein Signal, das aus dem Schluss von Tag t berechnet wird, führt zu einer
    Position, die frühestens die Rendite von Tag t+1 vereinnahmt.
  * Stop-Loss wird intraday am Tagestief geprüft und zum Stop-Kurs ausgeführt.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import data as dat
from .stats_tools import perf_metrics

# ── Baseline-Parameter (aus CONTINUATION_PROMPT §2.1) ───────────────────────
SIGNAL_BASKET = ["CL=F", "BZ=F", "XLE", "XOM", "CVX"]
TARGET = "JETS"
SIGNAL_WINDOW = 20
VIX_MAX = 25.0
STOP_LOSS = 0.08
BASE_SIZE = 1.0

# sign = +1 entspricht der Wortlaut-Spezifikation ("Basket-Mittel > 0 -> Long").
# sign = -1 ist die in Report 00 validierte Richtung (fallende Energiepreise ->
# sinkende Treibstoffkosten -> Long Airlines) und dient als Arbeits-Baseline.
SPEC_SIGN = 1
BASELINE_SIGN = -1


@dataclass
class StrategyResult:
    rets: pd.Series
    equity: pd.Series
    trades: pd.DataFrame
    exposure: pd.Series
    metrics: dict = field(default_factory=dict)

    @property
    def n_trades(self) -> int:
        return len(self.trades)


def basket_signal(window: int = SIGNAL_WINDOW, vix_max: float = VIX_MAX,
                  basket: list[str] | None = None,
                  sign: int = BASELINE_SIGN) -> pd.DataFrame:
    """Rohsignal-Frame: basket_mean, vix, signal (bool) auf gemeinsamem Index."""
    basket = basket or SIGNAL_BASKET
    px = dat.close_panel(basket + [TARGET, "^VIX"], min_obs=250)
    if px.empty or TARGET not in px.columns:
        return pd.DataFrame()
    px = px.ffill()
    rets = px[[c for c in basket if c in px.columns]].pct_change()
    basket_mean = rets.mean(axis=1).rolling(window).mean()
    vix = px["^VIX"] if "^VIX" in px.columns else pd.Series(np.nan, index=px.index)
    out = pd.DataFrame({
        "basket": basket_mean,
        "vix": vix,
        TARGET: px[TARGET],
    }).dropna(subset=[TARGET])
    out["signal"] = (sign * out["basket"] > 0) & (out["vix"] < vix_max)
    return out.dropna(subset=["basket"])


def rsi(s: pd.Series, window: int = 14) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / window, adjust=False).mean()
    return (100 - 100 / (1 + up / dn.replace(0, np.nan))).fillna(50.0)


def rsi_signal(leader: str = "CL=F", window: int = 14, threshold: float = 70.0,
               index: pd.Index | None = None) -> pd.Series:
    """Ursprüngliche Framework-Variante: Long JETS solange RSI(Leader) < 70."""
    px = dat.close(leader)
    if index is not None:
        px = px.reindex(index).ffill()
    return (rsi(px, window) < threshold).rename("rsi_signal")


def run_strategy(price: pd.Series, signal: pd.Series,
                 stop_loss: float | pd.Series = STOP_LOSS,
                 size: float | pd.Series = BASE_SIZE,
                 low: pd.Series | None = None,
                 cooldown: int = 1,
                 cost_bps: float = 0.0) -> StrategyResult:
    """Ereignisgetriebene Simulation Long-Only mit Stop-Loss.

    stop_loss / size dürfen Serien sein (Hedge-Schicht: dynamisch pro Tag).
    """
    px = pd.to_numeric(price, errors="coerce").dropna()
    idx = px.index
    sig = signal.reindex(idx).fillna(False).astype(bool)
    lo = (low.reindex(idx) if low is not None else px).ffill()
    sl = (stop_loss.reindex(idx).ffill() if isinstance(stop_loss, pd.Series)
          else pd.Series(float(stop_loss), index=idx))
    sz = (size.reindex(idx).ffill().fillna(0.0) if isinstance(size, pd.Series)
          else pd.Series(float(size), index=idx))

    p = px.to_numpy(float)
    l = lo.to_numpy(float)
    s = sig.to_numpy(bool)
    slv = sl.to_numpy(float)
    szv = np.nan_to_num(sz.to_numpy(float))

    n = len(p)
    rets = np.zeros(n)
    expo = np.zeros(n)
    trades: list[dict] = []
    in_pos = False
    entry_px = np.nan
    entry_i = -1
    entry_sz = 0.0
    cool = 0

    for i in range(1, n):
        if in_pos:
            stop_px = entry_px * (1.0 - slv[i])
            hit = l[i] <= stop_px
            raw = (stop_px / p[i - 1] - 1.0) if hit else (p[i] / p[i - 1] - 1.0)
            rets[i] = entry_sz * raw
            expo[i] = entry_sz
            if hit:
                exit_px = stop_px
                reason = "Stop-Loss"
            elif not s[i]:
                exit_px = p[i]
                reason = "Signal-Exit"
            else:
                continue
            trades.append({
                "Entry": idx[entry_i], "Exit": idx[i],
                "Tage": int(i - entry_i), "EntryPx": entry_px, "ExitPx": exit_px,
                "Return": exit_px / entry_px - 1.0, "Size": entry_sz, "Grund": reason,
            })
            rets[i] -= cost_bps / 1e4 * entry_sz
            in_pos = False
            cool = cooldown if hit else 0
            continue

        if cool > 0:
            cool -= 1
            continue
        if s[i] and szv[i] > 0:
            in_pos = True
            entry_px = p[i]
            entry_i = i
            entry_sz = szv[i]
            rets[i] -= cost_bps / 1e4 * entry_sz

    if in_pos:
        trades.append({
            "Entry": idx[entry_i], "Exit": idx[n - 1], "Tage": int(n - 1 - entry_i),
            "EntryPx": entry_px, "ExitPx": p[n - 1],
            "Return": p[n - 1] / entry_px - 1.0, "Size": entry_sz, "Grund": "offen",
        })

    r = pd.Series(rets, index=idx, name="strategy")
    tr = pd.DataFrame(trades)
    return StrategyResult(
        rets=r, equity=(1 + r).cumprod(), trades=tr,
        exposure=pd.Series(expo, index=idx, name="exposure"),
        metrics=perf_metrics(r),
    )


def baseline(sig_df: pd.DataFrame | None = None, **kw) -> tuple[StrategyResult, pd.DataFrame]:
    """Referenzlauf mit den Parametern aus dem Continuation-Prompt."""
    sig_df = basket_signal() if sig_df is None else sig_df
    if sig_df.empty:
        return StrategyResult(pd.Series(dtype=float), pd.Series(dtype=float),
                              pd.DataFrame(), pd.Series(dtype=float)), sig_df
    low = dat.ohlcv(TARGET).get("Low")
    res = run_strategy(sig_df[TARGET], sig_df["signal"], low=low, **kw)
    return res, sig_df


def buy_hold(price: pd.Series) -> StrategyResult:
    r = pd.to_numeric(price, errors="coerce").pct_change().fillna(0.0)
    return StrategyResult(r, (1 + r).cumprod(), pd.DataFrame(),
                          pd.Series(1.0, index=r.index), perf_metrics(r))


def trade_stats(trades: pd.DataFrame) -> dict:
    if trades is None or trades.empty:
        return {"n": 0, "win": np.nan, "avg": np.nan, "avg_days": np.nan,
                "best": np.nan, "worst": np.nan, "pf": np.nan, "stops": 0}
    r = trades["Return"]
    gains, losses = r[r > 0].sum(), -r[r < 0].sum()
    return {
        "n": len(trades), "win": float((r > 0).mean()), "avg": float(r.mean()),
        "avg_days": float(trades["Tage"].mean()), "best": float(r.max()),
        "worst": float(r.min()),
        "pf": float(gains / losses) if losses > 0 else np.nan,
        "stops": int((trades["Grund"] == "Stop-Loss").sum()),
    }


def metrics_table(named: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for name, r in named.items():
        m = perf_metrics(r)
        rows.append({
            "Variante": name,
            "Ann. Return": m["CAGR"], "Volatilität": m["Vol"], "Sharpe": m["Sharpe"],
            "Sortino": m["Sortino"], "Calmar": m["Calmar"], "Max DD": m["MaxDD"],
            "Trefferquote (Tage)": m["WinRate"],
        })
    return pd.DataFrame(rows).set_index("Variante")


def walk_forward(price: pd.Series, signal_fn, params: list, low: pd.Series | None = None,
                 train_months: int = 60, test_months: int = 12) -> pd.DataFrame:
    """Rollierende Walk-Forward-Validierung.

    signal_fn(param) -> bool-Series. In jedem Fenster wird der im Trainings-
    abschnitt beste Parameter (Sharpe) out-of-sample im Testabschnitt geprüft.
    """
    idx = price.index
    if len(idx) < 500:
        return pd.DataFrame()
    starts = pd.date_range(idx[0], idx[-1], freq="MS")
    rows = []
    for k in range(len(starts)):
        tr0 = starts[k]
        tr1 = tr0 + pd.DateOffset(months=train_months)
        te1 = tr1 + pd.DateOffset(months=test_months)
        if te1 > idx[-1]:
            break
        best, best_sh = None, -np.inf
        for prm in params:
            sig = signal_fn(prm)
            m = run_strategy(price.loc[tr0:tr1], sig.loc[tr0:tr1], low=low).metrics
            if np.isfinite(m["Sharpe"]) and m["Sharpe"] > best_sh:
                best, best_sh = prm, m["Sharpe"]
        if best is None:
            continue
        sig = signal_fn(best)
        oos = run_strategy(price.loc[tr1:te1], sig.loc[tr1:te1], low=low)
        rows.append({"Test-Start": tr1.date().isoformat(), "Test-Ende": te1.date().isoformat(),
                     "Bester Parameter (IS)": str(best), "Sharpe IS": best_sh,
                     "Sharpe OOS": oos.metrics["Sharpe"], "Return OOS": oos.metrics["CAGR"],
                     "MaxDD OOS": oos.metrics["MaxDD"]})
    return pd.DataFrame(rows)
