"""Signal-Bausteine: zwei Basissignale und drei Filter -> 16 Kombinationen.

Alle Bausteine liefern Zustände in {+1, 0, -1}:
  +1  bullish für JETS (fallende bzw. nicht überhitzte Energiepreise)
  -1  bearish für JETS
   0  kein Handel (Filter blockiert)

Die Filter werden multiplikativ angewandt; ein blockierender Filter setzt den
Zustand auf 0. Der Saisonalitätsfilter wird ausschließlich auf dem
In-Sample-Zeitraum kalibriert, damit er im Out-of-Sample look-ahead-frei ist.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from . import data as dat
from . import strategy as st

BASE_LABELS = {"Basket": "Basket-20d", "RSI<70": "RSI(CL=F,14) < 70"}
FILTER_LABELS = {"S": "Saisonalität (gute Monate, IS-kalibriert)",
                 "V": "VIX < 25", "T": "TNX-Trend fallend"}


def base_panel(tickers: list[str] | None = None) -> pd.DataFrame:
    """Alle Rohreihen, die für Signale und Strukturen benötigt werden."""
    tk = tickers or (st.SIGNAL_BASKET + [st.TARGET, "^VIX", "^TNX"])
    return dat.close_panel(sorted(set(tk)), min_obs=250).ffill()


def basket_state(px: pd.DataFrame, window: int = 20) -> pd.Series:
    cols = [c for c in st.SIGNAL_BASKET if c in px.columns]
    m = px[cols].pct_change().mean(axis=1).rolling(window).mean()
    # Fallende Energiepreise -> sinkende Treibstoffkosten -> bullish für Airlines.
    return pd.Series(np.where(m < 0, 1.0, np.where(m > 0, -1.0, 0.0)),
                     index=px.index, name="Basket").where(m.notna(), 0.0)


def rsi_state(px: pd.DataFrame, leader: str = "CL=F", window: int = 14,
              threshold: float = 70.0) -> pd.Series:
    if leader not in px.columns:
        return pd.Series(0.0, index=px.index, name="RSI")
    r = st.rsi(px[leader], window)
    return pd.Series(np.where(r < threshold, 1.0, -1.0), index=px.index, name="RSI<70")


def seasonal_filter(px: pd.DataFrame, is_end: pd.Timestamp,
                    target: str = st.TARGET) -> tuple[pd.Series, list[int]]:
    """Monate mit positiver Durchschnittsrendite — nur aus dem IS-Fenster."""
    r = px[target].pct_change()
    is_r = r.loc[:is_end]
    good = sorted(int(m) for m, v in is_r.groupby(is_r.index.month).mean().items() if v > 0)
    return pd.Series(px.index.month.isin(good), index=px.index, name="S"), good


def vix_filter(px: pd.DataFrame, cap: float = 25.0) -> pd.Series:
    if "^VIX" not in px.columns:
        return pd.Series(True, index=px.index, name="V")
    return (px["^VIX"] < cap).rename("V")


def tnx_filter(px: pd.DataFrame, window: int = 20) -> pd.Series:
    if "^TNX" not in px.columns:
        return pd.Series(True, index=px.index, name="T")
    return (px["^TNX"].diff(window) < 0).rename("T")


def build_signals(px: pd.DataFrame, is_end: pd.Timestamp) -> tuple[dict[str, pd.Series], dict]:
    """Alle 16 Kombinationen aus zwei Basissignalen und drei Filtern."""
    bases = {"Basket": basket_state(px), "RSI<70": rsi_state(px)}
    seas, good_months = seasonal_filter(px, is_end)
    filters = {"S": seas, "V": vix_filter(px), "T": tnx_filter(px)}

    combos: dict[str, pd.Series] = {}
    for bname, b in bases.items():
        for k in range(4):
            for sel in itertools.combinations("SVT", k):
                mask = pd.Series(True, index=px.index)
                for f in sel:
                    mask &= filters[f].reindex(px.index).fillna(False)
                name = bname + ("".join(f"+{f}" for f in sel))
                combos[name] = (b * mask.astype(float)).rename(name)
    meta = {"gute Monate (IS)": good_months,
            "Filter": {k: FILTER_LABELS[k] for k in filters},
            "Anteil aktiv": {n: float((s != 0).mean()) for n, s in combos.items()}}
    return combos, meta


def adaptive_signal(combos: dict[str, pd.Series], vix: pd.Series) -> pd.Series:
    """Meta-Strategie: Signalwahl nach VIX-Regime (dokumentierte Logik)."""
    idx = next(iter(combos.values())).index
    v = vix.reindex(idx).ffill()
    out = pd.Series(0.0, index=idx, name="Adaptiv")
    out = out.mask(v < 20, combos["Basket"])
    out = out.mask((v >= 20) & (v < 25), combos["RSI<70+S+V"])
    return out.mask(v >= 25, 0.0).fillna(0.0)
