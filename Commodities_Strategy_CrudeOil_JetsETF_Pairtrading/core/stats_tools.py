"""Statistik-Werkzeuge: Stationarität, Kausalität, Multiple Testing, Resampling,
Klassifikations-Gütemaße, Strukturbrüche, Survival, Faktoren.

Bewusst ohne sklearn — PCA via np.linalg.svd, Metriken direkt implementiert.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

SEED = 42


# ── Transformationen ────────────────────────────────────────────────────────
def zscore(s: pd.Series, window: int = 252, min_periods: int = 63) -> pd.Series:
    mu = s.rolling(window, min_periods=min_periods).mean()
    sd = s.rolling(window, min_periods=min_periods).std().replace(0, np.nan)
    return ((s - mu) / sd).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def prank(s: pd.Series, window: int = 252, min_periods: int = 63) -> pd.Series:
    """Rollierendes Perzentil (0–100) des aktuellen Werts im Fenster."""
    return (s.fillna(0.0).rolling(window, min_periods=min_periods)
            .rank(pct=True) * 100).astype(float)


# ── Stationarität ───────────────────────────────────────────────────────────
def adf(s: pd.Series, regression: str = "c") -> dict:
    from statsmodels.tsa.stattools import adfuller
    x = pd.to_numeric(s, errors="coerce").dropna().to_numpy(dtype=float)
    if len(x) < 30:
        return {"stat": np.nan, "p": np.nan, "lags": 0, "n": len(x),
                "crit": {}, "stationary": None}
    st, p, lags, nobs, crit, _ = adfuller(x, autolag="AIC", regression=regression,
                                          result_object=False)
    return {"stat": float(st), "p": float(p), "lags": int(lags), "n": int(nobs),
            "crit": {k: float(v) for k, v in crit.items()},
            "stationary": bool(p < 0.05)}


def kpss(s: pd.Series, regression: str = "c") -> dict:
    from statsmodels.tsa.stattools import kpss as _kpss
    x = pd.to_numeric(s, errors="coerce").dropna().to_numpy(dtype=float)
    if len(x) < 30:
        return {"stat": np.nan, "p": np.nan, "crit": {}, "stationary": None}
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        st, p, _, crit = _kpss(x, regression=regression, nlags="auto")
    return {"stat": float(st), "p": float(p),
            "crit": {k: float(v) for k, v in crit.items()},
            "stationary": bool(p > 0.05)}


def stationarity_table(panel: pd.DataFrame, label: str = "") -> pd.DataFrame:
    """ADF + KPSS kombiniert. Konsens: beide einig -> klar, sonst 'uneindeutig'."""
    rows = []
    for c in panel.columns:
        a, k = adf(panel[c]), kpss(panel[c])
        if a["stationary"] is None:
            verdict = "zu wenig Daten"
        elif a["stationary"] and k["stationary"]:
            verdict = "I(0) — stationär"
        elif (not a["stationary"]) and (not k["stationary"]):
            verdict = "I(1) — differenzieren"
        else:
            verdict = "uneindeutig"
        rows.append({
            "Serie": f"{label}{c}",
            "ADF-Stat": round(a["stat"], 3), "ADF-p": round(a["p"], 4),
            "ADF-krit(5%)": round(a["crit"].get("5%", np.nan), 3),
            "KPSS-Stat": round(k["stat"], 3), "KPSS-p": round(k["p"], 4),
            "Lags": a["lags"], "Schluss": verdict,
        })
    return pd.DataFrame(rows)


# ── Multiple Testing ────────────────────────────────────────────────────────
def bonferroni(pvals, alpha: float = 0.05) -> tuple[np.ndarray, float]:
    p = np.asarray(pvals, dtype=float)
    n = max(int(np.isfinite(p).sum()), 1)
    return (p < alpha / n), alpha / n


def benjamini_hochberg(pvals, alpha: float = 0.05) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    ok = np.isfinite(p)
    out = np.zeros_like(p, dtype=bool)
    idx = np.where(ok)[0]
    if idx.size == 0:
        return out
    order = idx[np.argsort(p[idx])]
    m = order.size
    thresh = alpha * (np.arange(1, m + 1) / m)
    passed = p[order] <= thresh
    if passed.any():
        kmax = np.max(np.where(passed)[0])
        out[order[:kmax + 1]] = True
    return out


# ── Granger-Kausalität ──────────────────────────────────────────────────────
def granger_pvalue(cause: pd.Series, effect: pd.Series, max_lag: int = 10
                   ) -> tuple[float, int]:
    """AIC-optimierte Lag-Wahl über VAR, dann F-Test. Rückgabe (p, lag)."""
    from statsmodels.tsa.stattools import grangercausalitytests
    from statsmodels.tsa.api import VAR
    df = pd.concat([effect.rename("y"), cause.rename("x")], axis=1).dropna()
    if len(df) < 20 * max_lag:
        return np.nan, 0
    try:
        sel = VAR(df.to_numpy(dtype=float)).select_order(maxlags=max_lag)
        lag = int(sel.aic) if sel.aic and sel.aic > 0 else 1
    except Exception:  # noqa: BLE001
        lag = 1
    lag = max(1, min(lag, max_lag))
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = grangercausalitytests(df.to_numpy(dtype=float), maxlag=[lag])
        return float(res[lag][0]["ssr_ftest"][1]), lag
    except Exception:  # noqa: BLE001
        return np.nan, lag


# ── Korrelation & Regime ────────────────────────────────────────────────────
def fisher_z(r: float, n: int) -> tuple[float, float]:
    r = float(np.clip(r, -0.999999, 0.999999))
    return np.arctanh(r), 1.0 / np.sqrt(max(n - 3, 1))


def fisher_z_test(r1: float, n1: int, r2: float, n2: int) -> tuple[float, float]:
    """H0: rho1 == rho2. Rückgabe (z-Statistik, zweiseitiger p-Wert)."""
    if not (np.isfinite(r1) and np.isfinite(r2)) or min(n1, n2) < 5:
        return np.nan, np.nan
    z1, _ = fisher_z(r1, n1)
    z2, _ = fisher_z(r2, n2)
    se = np.sqrt(1.0 / max(n1 - 3, 1) + 1.0 / max(n2 - 3, 1))
    z = (z1 - z2) / se
    return float(z), float(2 * (1 - sps.norm.cdf(abs(z))))


def corr_ci(r: float, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if not np.isfinite(r) or n < 6:
        return np.nan, np.nan
    z, se = fisher_z(r, n)
    q = sps.norm.ppf(1 - alpha / 2)
    return float(np.tanh(z - q * se)), float(np.tanh(z + q * se))


def ewma_corr(x: pd.Series, y: pd.Series, lam: float = 0.94) -> pd.Series:
    """DCC-GARCH-Approximation: EWMA-Kovarianz auf standardisierten Residuen.

    Schritt 1 (Devolatilisierung): eps_t = r_t / sigma_t mit EWMA-Volatilität.
    Schritt 2 (dynamische Korrelation): EWMA-Korrelation der eps.
    Entspricht dem DCC(1,1)-Integrated-Spezialfall (a = 1 - lam, b = lam).
    """
    df = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    if len(df) < 60:
        return pd.Series(dtype=float)
    a = 1.0 - lam
    vx = df["x"].pow(2).ewm(alpha=a, adjust=False).mean().pow(0.5).replace(0, np.nan)
    vy = df["y"].pow(2).ewm(alpha=a, adjust=False).mean().pow(0.5).replace(0, np.nan)
    ex, ey = df["x"] / vx, df["y"] / vy
    qxy = (ex * ey).ewm(alpha=a, adjust=False).mean()
    qxx = ex.pow(2).ewm(alpha=a, adjust=False).mean()
    qyy = ey.pow(2).ewm(alpha=a, adjust=False).mean()
    return (qxy / (qxx.pow(0.5) * qyy.pow(0.5))).clip(-1, 1).dropna()


# ── PCA (SVD, ohne sklearn) ─────────────────────────────────────────────────
def pca_svd(panel: pd.DataFrame, standardize: bool = True) -> dict:
    X = panel.dropna(how="any").to_numpy(dtype=float)
    if X.shape[0] < 30 or X.shape[1] < 2:
        return {}
    mu = X.mean(axis=0)
    Xc = X - mu
    if standardize:
        sd = Xc.std(axis=0, ddof=1)
        sd[sd == 0] = 1.0
        Xc = Xc / sd
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = (S ** 2) / max(Xc.shape[0] - 1, 1)
    ratio = var / var.sum()
    scores = U * S
    idx = panel.dropna(how="any").index
    return {
        "explained": ratio,
        "cum": np.cumsum(ratio),
        "loadings": pd.DataFrame(Vt.T, index=panel.columns,
                                 columns=[f"PC{i+1}" for i in range(Vt.shape[0])]),
        "scores": pd.DataFrame(scores, index=idx,
                               columns=[f"PC{i+1}" for i in range(scores.shape[1])]),
        "eigenvalues": var,
    }


# ── Performance-Metriken & Bootstrap ────────────────────────────────────────
def perf_metrics(rets: pd.Series, rf: float = 0.0, periods: int = 252) -> dict:
    r = pd.to_numeric(rets, errors="coerce").dropna()
    if len(r) < 20:
        return {k: np.nan for k in
                ("CAGR", "Vol", "Sharpe", "Sortino", "Calmar", "MaxDD", "WinRate")}
    eq = (1 + r).cumprod()
    yrs = len(r) / periods
    cagr = eq.iloc[-1] ** (1 / yrs) - 1 if eq.iloc[-1] > 0 else -1.0
    vol = r.std(ddof=1) * np.sqrt(periods)
    ex = r - rf / periods
    sharpe = ex.mean() / r.std(ddof=1) * np.sqrt(periods) if r.std(ddof=1) > 0 else np.nan
    dn = r[r < 0].std(ddof=1)
    sortino = ex.mean() / dn * np.sqrt(periods) if dn and dn > 0 else np.nan
    dd = (eq / eq.cummax() - 1).min()
    return {
        "CAGR": float(cagr), "Vol": float(vol), "Sharpe": float(sharpe),
        "Sortino": float(sortino),
        "Calmar": float(cagr / abs(dd)) if dd < 0 else np.nan,
        "MaxDD": float(dd), "WinRate": float((r > 0).mean()),
    }


def block_bootstrap_ci(rets: pd.Series, metric: str = "Sharpe", n: int = 1000,
                       block: int = 21, alpha: float = 0.05,
                       seed: int = SEED) -> tuple[float, float, np.ndarray]:
    """Stationärer Block-Bootstrap — erhält Autokorrelation der Renditen."""
    r = pd.to_numeric(rets, errors="coerce").dropna().to_numpy(dtype=float)
    if len(r) < 60:
        return np.nan, np.nan, np.array([])
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(len(r) / block))
    out = np.empty(n)
    for i in range(n):
        starts = rng.integers(0, len(r) - block, size=nb)
        samp = np.concatenate([r[s:s + block] for s in starts])[:len(r)]
        out[i] = perf_metrics(pd.Series(samp))[metric]
    out = out[np.isfinite(out)]
    if out.size == 0:
        return np.nan, np.nan, out
    return (float(np.quantile(out, alpha / 2)),
            float(np.quantile(out, 1 - alpha / 2)), out)


def sharpe_diff_test(r_a: pd.Series, r_b: pd.Series, n: int = 1000,
                     block: int = 21, seed: int = SEED) -> dict:
    """Gepaarter Block-Bootstrap auf Sharpe(A) − Sharpe(B)."""
    df = pd.concat([r_a.rename("a"), r_b.rename("b")], axis=1).dropna()
    if len(df) < 60:
        return {"diff": np.nan, "lo": np.nan, "hi": np.nan, "p": np.nan}
    A, B = df["a"].to_numpy(float), df["b"].to_numpy(float)
    base = perf_metrics(df["a"])["Sharpe"] - perf_metrics(df["b"])["Sharpe"]
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(len(A) / block))
    diffs = np.empty(n)
    for i in range(n):
        st = rng.integers(0, len(A) - block, size=nb)
        ia = np.concatenate([np.arange(s, s + block) for s in st])[:len(A)]
        diffs[i] = (perf_metrics(pd.Series(A[ia]))["Sharpe"]
                    - perf_metrics(pd.Series(B[ia]))["Sharpe"])
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        return {"diff": base, "lo": np.nan, "hi": np.nan, "p": np.nan}
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {"diff": float(base), "lo": float(np.quantile(diffs, 0.025)),
            "hi": float(np.quantile(diffs, 0.975)), "p": float(min(p, 1.0))}


# ── Klassifikations-Gütemaße ────────────────────────────────────────────────
def _thin(*arrays, max_points: int = 1500):
    n = len(arrays[0])
    if n <= max_points:
        return arrays
    keep = np.unique(np.concatenate([
        np.linspace(0, n - 1, max_points).astype(int), [n - 1]]))
    return tuple(a[keep] for a in arrays)


def roc_curve(score: np.ndarray, label: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    s = np.asarray(score, dtype=float)
    y = np.asarray(label).astype(bool)
    ok = np.isfinite(s)
    s, y = s[ok], y[ok]
    if y.sum() == 0 or (~y).sum() == 0:
        return np.array([0, 1.0]), np.array([0, 1.0]), np.array([np.nan, np.nan])
    order = np.argsort(-s)
    s, y = s[order], y[order]
    tp = np.cumsum(y)
    fp = np.cumsum(~y)
    return _thin(fp / (~y).sum(), tp / y.sum(), s)


def auc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    return float(np.trapezoid(tpr, fpr)) if len(fpr) > 1 else np.nan


def pr_curve(score: np.ndarray, label: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    s = np.asarray(score, dtype=float)
    y = np.asarray(label).astype(bool)
    ok = np.isfinite(s)
    s, y = s[ok], y[ok]
    if y.sum() == 0:
        return np.array([0.0]), np.array([0.0]), np.array([np.nan])
    order = np.argsort(-s)
    s, y = s[order], y[order]
    tp = np.cumsum(y)
    prec = tp / np.arange(1, len(y) + 1)
    rec = tp / y.sum()
    return _thin(rec, prec, s)


def average_precision(rec: np.ndarray, prec: np.ndarray) -> float:
    if len(rec) < 2:
        return np.nan
    return float(np.sum(np.diff(np.concatenate([[0.0], rec])) * prec))


def youden(score: np.ndarray, label: np.ndarray) -> dict:
    fpr, tpr, thr = roc_curve(score, label)
    if not np.isfinite(thr).any():
        return {"thr": np.nan, "j": np.nan, "tpr": np.nan, "fpr": np.nan}
    j = tpr - fpr
    k = int(np.nanargmax(j))
    return {"thr": float(thr[k]), "j": float(j[k]),
            "tpr": float(tpr[k]), "fpr": float(fpr[k])}


# ── Strukturbrüche & Extremwerte ────────────────────────────────────────────
def cusum(s: pd.Series, k: float = 0.5, h: float = 5.0) -> pd.DataFrame:
    """Zweiseitiges CUSUM-Kontrolldiagramm auf standardisierter Serie."""
    x = pd.to_numeric(s, errors="coerce").dropna()
    if len(x) < 60:
        return pd.DataFrame()
    z = (x - x.expanding(60).mean()) / x.expanding(60).std().replace(0, np.nan)
    z = z.fillna(0.0)
    hi = np.zeros(len(z))
    lo = np.zeros(len(z))
    v = z.to_numpy(float)
    for i in range(1, len(v)):
        hi[i] = max(0.0, hi[i - 1] + v[i] - k)
        lo[i] = min(0.0, lo[i - 1] + v[i] + k)
    return pd.DataFrame({"C+": hi, "C-": lo, "alarm": (hi > h) | (lo < -h)},
                        index=z.index)


def gpd_tail(losses: pd.Series, q: float = 0.95) -> dict:
    """Peaks-over-Threshold: GPD-Fit auf Verlust-Exzedenzen (Extremwerttheorie)."""
    x = pd.to_numeric(losses, errors="coerce").dropna()
    x = x[x > 0]
    if len(x) < 200:
        return {}
    u = float(np.quantile(x, q))
    exc = (x[x > u] - u).to_numpy(float)
    if exc.size < 20:
        return {}
    xi, _, beta = sps.genpareto.fit(exc, floc=0)
    zeta = exc.size / len(x)

    def var_level(p: float) -> float:
        return u + beta / xi * (((1 - p) / zeta) ** (-xi) - 1) if abs(xi) > 1e-8 \
            else u + beta * np.log(zeta / (1 - p))

    v99 = var_level(0.99)
    es99 = (v99 + beta - xi * u) / (1 - xi) if xi < 1 else np.nan
    return {"u": u, "xi": float(xi), "beta": float(beta), "n_exc": int(exc.size),
            "VaR99": float(v99), "ES99": float(es99)}


# ── Survival: Kaplan-Meier ──────────────────────────────────────────────────
def kaplan_meier(durations, events) -> pd.DataFrame:
    d = np.asarray(durations, dtype=float)
    e = np.asarray(events).astype(bool)
    ok = np.isfinite(d)
    d, e = d[ok], e[ok]
    if d.size == 0:
        return pd.DataFrame()
    rows, S, var = [], 1.0, 0.0
    for t in np.unique(d):
        at_risk = int((d >= t).sum())
        died = int(((d == t) & e).sum())
        if at_risk == 0:
            continue
        if died:
            S *= (1 - died / at_risk)
            var += died / (at_risk * max(at_risk - died, 1))
        se = S * np.sqrt(var)
        rows.append({"t": float(t), "at_risk": at_risk, "events": died,
                     "S": S, "lo": max(S - 1.96 * se, 0.0),
                     "hi": min(S + 1.96 * se, 1.0)})
    return pd.DataFrame(rows)


# ── Formatierung ────────────────────────────────────────────────────────────
def pfmt(p: float) -> str:
    if not np.isfinite(p):
        return "—"
    return "&lt;0.0001" if p < 1e-4 else f"{p:.4f}"


def pct(x: float, digits: int = 2) -> str:
    return "—" if not np.isfinite(x) else f"{x * 100:.{digits}f}%"


def num(x: float, digits: int = 3) -> str:
    return "—" if not np.isfinite(x) else f"{x:.{digits}f}"
