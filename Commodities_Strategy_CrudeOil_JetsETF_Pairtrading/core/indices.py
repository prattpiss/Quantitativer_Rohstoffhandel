"""Zusammengesetzte Risiko-Indizes: CSI, CPI, PPI, CGR.

Alle Indizes werden ausschließlich aus Informationen gebildet, die am Schluss
des jeweiligen Handelstages verfügbar sind (rollierende Fenster, keine
zentrierten Glättungen) — damit ist das Look-Ahead-Risiko konstruktionsbedingt
ausgeschlossen.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as dat
from .stats_tools import prank, zscore

# ── CSI: Crash Stress Index (0–100) ─────────────────────────────────────────
CSI_WEIGHTS = {"C1": 0.30, "C2": 0.18, "C3": 0.30, "C4": 0.10, "C5": 0.12, "C6": 0.15}
CSI_LABELS = {
    "C1": "VIX-Niveau (Perzentil)",
    "C2": "VIX 5-Tage-Spike",
    "C3": "Credit Spread −log(HYG/IEF)",
    "C4": "|DXY 5d| Safe-Haven-Nachfrage",
    "C5": "JETS Volumen-Anomalie",
    "C6": "VIX-Terminstruktur-Inversion",
}


def _align(series: dict[str, pd.Series], min_obs: int = 500) -> tuple[pd.DatetimeIndex, dict]:
    base = [s for s in series.values() if len(s) > min_obs]
    if not base:
        return pd.DatetimeIndex([]), {}
    idx = base[0].index
    for s in base[1:]:
        idx = idx.intersection(s.index)
    idx = idx.sort_values()
    out = {k: (s.reindex(idx).ffill().bfill() if len(s) > 50
               else pd.Series(np.nan, index=idx)) for k, s in series.items()}
    return idx, out


def csi_components(window: int = 252) -> pd.DataFrame:
    """Sechs Stress-Komponenten als rollierende Perzentile (0–100)."""
    jets = dat.ohlcv("JETS")
    raw = {
        "vix": dat.close("^VIX"), "vix9d": dat.close("^VIX9D"),
        "hyg": dat.close("HYG"), "ief": dat.close("IEF"),
        "dxy": dat.close("DX-Y.NYB"),
        "jets": dat.close("JETS"),
    }
    idx, a = _align(raw)
    if len(idx) == 0:
        return pd.DataFrame()
    jvol = (pd.to_numeric(jets["Volume"], errors="coerce").reindex(idx).ffill()
            if "Volume" in jets.columns else pd.Series(np.nan, index=idx))

    c1 = prank(a["vix"], window)
    c2 = prank((a["vix"] / a["vix"].shift(5) - 1).clip(lower=0).fillna(0.0), window)
    credit = -np.log((a["hyg"] / a["ief"].replace(0, np.nan)).replace(0, np.nan))
    c3 = prank(credit.fillna(0.0), window)
    c4 = (prank(a["dxy"].pct_change(5).abs().fillna(0.0), window)
          if a["dxy"].notna().sum() > 200 else pd.Series(50.0, index=idx))
    vr = (jvol / jvol.rolling(20).mean().replace(0, np.nan) - 1).clip(lower=0).fillna(0.0)
    c5 = prank(vr, window) if jvol.notna().sum() > 200 else pd.Series(50.0, index=idx)
    has_ts = a["vix9d"].notna().sum() > 100
    c6 = (prank(-(a["vix9d"] - a["vix"]).fillna(0.0), window) if has_ts
          else pd.Series(50.0, index=idx))

    df = pd.DataFrame({"C1": c1, "C2": c2, "C3": c3, "C4": c4, "C5": c5, "C6": c6},
                      index=idx)
    df["JETS"] = a["jets"]
    df["VIX"] = a["vix"]
    df["credit_raw"] = credit
    df.attrs["has_term_structure"] = bool(has_ts)
    return df


def csi_from_components(comp: pd.DataFrame, weights: dict[str, float] | None = None,
                        smooth: int = 3) -> pd.Series:
    w = dict(CSI_WEIGHTS if weights is None else weights)
    tot = sum(w.values())
    if tot <= 0:
        return pd.Series(dtype=float)
    w = {k: v / tot for k, v in w.items()}
    raw = sum(comp[k] * v for k, v in w.items() if k in comp.columns)
    return raw.rolling(smooth).mean().rename("CSI")


# ── CPI: Composite Predictivity Index (Z-Skala) ─────────────────────────────
CPI_WEIGHTS = {"VIX": 0.30, "Credit": 0.25, "Curve": 0.20, "Gold": 0.15, "Defense": 0.10}


def cpi_components() -> pd.DataFrame:
    raw = {
        "vix": dat.close("^VIX"), "tnx": dat.close("^TNX"), "irx": dat.close("^IRX"),
        "hyg": dat.close("HYG"), "ief": dat.close("IEF"), "gld": dat.close("GLD"),
        "ita": dat.close("ITA"), "spy": dat.close("SPY"),
    }
    idx, a = _align(raw)
    if len(idx) == 0:
        return pd.DataFrame()
    curve = a["tnx"] - a["irx"]
    credit = -np.log((a["hyg"] / a["ief"].replace(0, np.nan)).replace(0, np.nan))
    gold = a["gld"].pct_change(20)
    defense = (a["ita"] / a["spy"].replace(0, np.nan)).pct_change(20)
    return pd.DataFrame({
        "VIX": zscore(a["vix"]),
        "Credit": zscore(credit.fillna(0.0)),
        "Curve": -zscore(curve.fillna(0.0)),
        "Gold": zscore(gold.fillna(0.0)),
        "Defense": zscore(defense.fillna(0.0)),
    }, index=idx)


def cpi_index(comp: pd.DataFrame | None = None,
              weights: dict[str, float] | None = None) -> pd.Series:
    comp = cpi_components() if comp is None else comp
    if comp.empty:
        return pd.Series(dtype=float)
    w = weights or CPI_WEIGHTS
    return sum(comp[k] * v for k, v in w.items() if k in comp.columns).rename("CPI")


# ── PPI: Pandemic Proxy Index (Z-Skala) ─────────────────────────────────────
PPI_WEIGHTS = {"VIXspike": 0.25, "AirVol": 0.20, "Pharma": 0.20,
               "AirWeak": 0.20, "Credit": 0.15}
PPI_LABELS = {
    "VIXspike": "VIX-Sprung (5d)",
    "AirVol": "Airline-Volumenanomalie (JETS/DAL/UAL)",
    "Pharma": "Pharma/Biotech relative Stärke vs. SPY",
    "AirWeak": "Airline-Schwäche relativ zum Markt",
    "Credit": "Ausweitung des Credit Spreads",
}


def ppi_components(mode: str = "standard") -> pd.DataFrame:
    """PPI-Komponenten.

    mode="standard": volle Signalbreite ab JETS-Auflegung (2015).
    mode="long":     historisch weit zurückreichende Variante (ab ca. 2002) ohne
                     JETS/HYG — nötig, um SARS 2003, H1N1 2009, MERS 2012 und
                     Ebola 2014 überhaupt validieren zu können.
    """
    long_mode = mode == "long"
    air_tickers = ("LUV", "DAL", "UAL") if long_mode else ("JETS", "DAL", "UAL", "LUV")
    raw = {"vix": dat.close("^VIX"), "spy": dat.close("SPY"),
           "ief": dat.close("IEF"), "xlv": dat.close("XLV"), "pfe": dat.close("PFE"),
           "luv": dat.close("LUV")}
    raw["credit_ref"] = dat.close("LQD") if long_mode else dat.close("HYG")
    if long_mode:
        raw["dal"] = dat.close("DAL")
        raw["ual"] = dat.close("UAL")
    else:
        raw |= {"jets": dat.close("JETS"), "dal": dat.close("DAL"),
                "ual": dat.close("UAL"), "ibb": dat.close("IBB")}
    idx, a = _align(raw, min_obs=300)
    if len(idx) == 0:
        return pd.DataFrame()

    vol_frames = []
    for t in air_tickers:
        df = dat.ohlcv(t)
        if "Volume" in df.columns:
            v = pd.to_numeric(df["Volume"], errors="coerce").reindex(idx).ffill()
            vol_frames.append((v / v.rolling(20).mean().replace(0, np.nan) - 1)
                              .clip(lower=0))
    air_vol = (pd.concat(vol_frames, axis=1).mean(axis=1) if vol_frames
               else pd.Series(0.0, index=idx))

    air_keys = [k for k in ("jets", "dal", "ual", "luv")
                if k in a and a[k].notna().sum() > 200]
    airline_px = pd.concat([a[k] for k in air_keys], axis=1).mean(axis=1)
    air_rel = (airline_px / a["spy"].replace(0, np.nan)).pct_change(20)

    ph_keys = [k for k in ("ibb", "xlv", "pfe") if k in a and a[k].notna().sum() > 200]
    pharma = pd.concat([a[k] for k in ph_keys], axis=1).mean(axis=1)
    pharma_rel = (pharma / a["spy"].replace(0, np.nan)).pct_change(20)
    credit = -np.log((a["credit_ref"] / a["ief"].replace(0, np.nan)).replace(0, np.nan))

    out = pd.DataFrame({
        "VIXspike": zscore((a["vix"] / a["vix"].shift(5) - 1).clip(lower=0).fillna(0.0)),
        "AirVol": zscore(air_vol.fillna(0.0)),
        "Pharma": zscore(pharma_rel.fillna(0.0)),
        "AirWeak": -zscore(air_rel.fillna(0.0)),
        "Credit": zscore(credit.diff(20).fillna(0.0)),
    }, index=idx)
    out.attrs["mode"] = mode
    out.attrs["airlines"] = list(air_keys)
    return out



def ppi_index(comp: pd.DataFrame | None = None,
              weights: dict[str, float] | None = None) -> pd.Series:
    comp = ppi_components() if comp is None else comp
    if comp.empty:
        return pd.Series(dtype=float)
    w = weights or PPI_WEIGHTS
    s = sum(comp[k] * v for k, v in w.items() if k in comp.columns)
    return s.rolling(3).mean().rename("PPI")


# ── CGR: Composite Geopolitical Risk (Z-Skala) ──────────────────────────────
CGR_WEIGHTS = {"Defense": 0.30, "BrentWTI": 0.20, "Gold": 0.20,
               "Dollar": 0.15, "TermStruct": 0.15}


def cgr_components() -> pd.DataFrame:
    raw = {"lmt": dat.close("LMT"), "rtx": dat.close("RTX"), "noc": dat.close("NOC"),
           "spy": dat.close("SPY"), "brent": dat.close("BZ=F"), "wti": dat.close("CL=F"),
           "gld": dat.close("GLD"), "dxy": dat.close("DX-Y.NYB"),
           "vix": dat.close("^VIX"), "vix9d": dat.close("^VIX9D")}
    idx, a = _align(raw, min_obs=300)
    if len(idx) == 0:
        return pd.DataFrame()
    defense = pd.concat([a[k] for k in ("lmt", "rtx", "noc")
                         if a[k].notna().sum() > 200], axis=1).mean(axis=1)
    def_rel = (defense / a["spy"].replace(0, np.nan)).pct_change(20)
    spread = a["brent"] - a["wti"]
    ts = -(a["vix9d"] - a["vix"]) if a["vix9d"].notna().sum() > 100 \
        else pd.Series(0.0, index=idx)
    return pd.DataFrame({
        "Defense": zscore(def_rel.fillna(0.0)),
        "BrentWTI": zscore(spread.fillna(0.0)),
        "Gold": zscore(a["gld"].pct_change(20).fillna(0.0)),
        "Dollar": zscore(a["dxy"].pct_change(20).fillna(0.0)),
        "TermStruct": zscore(ts.fillna(0.0)),
    }, index=idx)


def cgr_index(comp: pd.DataFrame | None = None) -> pd.Series:
    comp = cgr_components() if comp is None else comp
    if comp.empty:
        return pd.Series(dtype=float)
    return sum(comp[k] * v for k, v in CGR_WEIGHTS.items()
               if k in comp.columns).rolling(3).mean().rename("CGR")
