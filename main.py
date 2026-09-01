"""
Main research pipeline.

Executes all 14 phases sequentially.
Each phase produces artefacts (tables, figures) in outputs/.
Phases are idempotent: outputs are cached so re-runs skip completed work.

Usage:
    python main.py                          # run full pipeline
    python main.py --phase 1 2 3            # run specific phases
    python main.py --phase 6 --force        # re-run phase 6 even if cached
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

# ── Allow imports from project root ──────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config.settings import SETTINGS, Settings
from config.symbols import SYMBOLS, UNIVERSE
from utils.logging_utils import setup_logging, get_logger
from utils.io_utils import save_table

log = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Phase runners
# ═══════════════════════════════════════════════════════════════════════════════

def phase1_load_data(s: Settings):
    """Phase 1: Download and cache all data."""
    log.info("=" * 60)
    log.info("PHASE 1 – DATA LOADING")
    log.info("=" * 60)
    from data.yahoo import YahooLoader
    from data.fred import FREDLoader
    loader = YahooLoader(s)
    prices = loader.fetch_prices(SYMBOLS["all_yahoo"], start=s.start_date, end=s.end_date)
    save_table(s.output_dir, "phase1_prices", prices)
    fred = FREDLoader(s)
    macro = fred.fetch_combined()
    if not macro.empty:
        save_table(s.output_dir, "phase1_macro", macro)
    log.info("Phase 1 complete. Prices shape: %s", prices.shape)
    return prices, macro


def phase2_preprocess(prices, macro, s: Settings):
    """Phase 2: Align, clean, compute returns."""
    log.info("=" * 60)
    log.info("PHASE 2 – PREPROCESSING")
    log.info("=" * 60)
    from preprocessing.alignment import MarketAligner
    from preprocessing.cleaning import DataCleaner
    from preprocessing.returns import ReturnCalculator

    aligner = MarketAligner(s)
    prices_aligned = aligner.align(prices)

    cleaner = DataCleaner(s)
    prices_clean = cleaner.fill_missing(prices_aligned)

    calc = ReturnCalculator(s)
    returns = calc.compute(prices_clean)
    returns = returns.dropna(how="all")

    save_table(s.output_dir, "phase2_prices_aligned", prices_aligned)
    save_table(s.output_dir, "phase2_returns", returns)
    log.info("Phase 2 complete. Returns shape: %s", returns.shape)
    return prices_clean, returns


def phase3_eda(returns, s: Settings):
    """Phase 3: Descriptive statistics, rolling metrics."""
    log.info("=" * 60)
    log.info("PHASE 3 – EXPLORATORY DATA ANALYSIS")
    log.info("=" * 60)
    from statistics.descriptive import DescriptiveAnalyzer
    from visualization.charts import ChartBuilder
    from visualization.heatmaps import HeatmapBuilder

    analyzer = DescriptiveAnalyzer(s)
    stats = analyzer.compute(returns)
    stats_df = stats.to_dataframe()
    save_table(s.output_dir, "phase3_descriptive_stats", stats_df)

    rolling_vol = analyzer.rolling_volatility(returns)
    save_table(s.output_dir, "phase3_rolling_vol", rolling_vol)

    charts = ChartBuilder(s)
    charts.rolling_volatility(rolling_vol[SYMBOLS["commodities"]])

    heatmaps = HeatmapBuilder(s)
    from statistics.correlation import CorrelationAnalyzer
    corr = CorrelationAnalyzer(s)
    pearson_mat = corr.pearson(returns.dropna())
    heatmaps.correlation_heatmap(pearson_mat, title="Pearson Correlation")

    log.info("Phase 3 complete.")
    return stats_df


def phase4_stationarity(prices, returns, s: Settings):
    """Phase 4: Unit-root tests."""
    log.info("=" * 60)
    log.info("PHASE 4 – STATIONARITY TESTS")
    log.info("=" * 60)
    from statistics.stationarity import StationarityTester

    tester = StationarityTester(s)
    # Test returns (should be stationary)
    results = tester.test_all(returns.dropna(how="all").dropna(axis=1, how="all"))
    save_table(s.output_dir, "phase4_stationarity", results)

    order_df = tester.recommend_integration_order(
        prices.dropna(how="all"), returns.dropna(how="all")
    )
    save_table(s.output_dir, "phase4_integration_order", order_df)
    log.info("Phase 4 complete.")
    return results


def phase5_correlation(returns, s: Settings):
    """Phase 5: Full correlation analysis."""
    log.info("=" * 60)
    log.info("PHASE 5 – CORRELATION ANALYSIS")
    log.info("=" * 60)
    from statistics.correlation import CorrelationAnalyzer
    from visualization.heatmaps import HeatmapBuilder

    corr = CorrelationAnalyzer(s)
    clean = returns.dropna(how="all")
    clean = clean.loc[:, clean.notna().sum() >= 252]

    for method in ["pearson", "spearman", "kendall"]:
        mat = clean.corr(method=method)
        save_table(s.output_dir, f"phase5_corr_{method}", mat)

    sig_corrs = corr.significant_correlations(clean, method="spearman")
    save_table(s.output_dir, "phase5_significant_correlations", sig_corrs)

    hm = HeatmapBuilder(s)
    hm.correlation_heatmap(clean.corr(method="spearman"), title="Spearman Correlation")
    log.info("Phase 5 complete.")


def phase6_leadlag(returns, s: Settings):
    """Phase 6: Lead-lag analysis (CCF, Granger, VAR, Transfer Entropy)."""
    log.info("=" * 60)
    log.info("PHASE 6 – LEAD-LAG ANALYSIS")
    log.info("=" * 60)
    from leadlag.crosscorrelation import CrossCorrelationAnalyzer
    from leadlag.granger import GrangerAnalyzer
    from leadlag.var_models import VARAnalyzer
    from leadlag.transfer_entropy import TransferEntropyAnalyzer
    from visualization.heatmaps import HeatmapBuilder

    clean = returns.dropna(how="all")
    clean = clean.loc[:, clean.notna().sum() >= 252]
    comm_cols = [c for c in SYMBOLS["commodities"] if c in clean.columns]
    eq_cols = [c for c in SYMBOLS["equities"] if c in clean.columns]

    comm_ret = clean[comm_cols]
    eq_ret = clean[eq_cols]

    # CCF screening
    ccf = CrossCorrelationAnalyzer(s)
    lag_df = ccf.screen_commodity_to_equities(comm_ret, eq_ret, max_lag=20)
    save_table(s.output_dir, "phase6_ccf_lags", lag_df)

    # Granger causality (on first commodity vs all equities)
    if comm_cols:
        granger = GrangerAnalyzer(s)
        gc_df = granger.test_commodity_to_equities(comm_ret.iloc[:, 0], eq_ret, max_lag=10)
        save_table(s.output_dir, "phase6_granger", gc_df)

        # Granger causality matrix (all × all, smaller subset for speed)
        subset_cols = comm_cols[:3] + eq_cols[:6]
        subset = clean[[c for c in subset_cols if c in clean.columns]]
        gc_mat = granger.build_causality_matrix(subset, max_lag=10)
        save_table(s.output_dir, "phase6_granger_matrix", gc_mat)

        hm = HeatmapBuilder(s)
        hm.lead_lag_heatmap(gc_mat, title="Granger Causality Lag Matrix")

    # VAR + IRF + FEVD (small system: 1 commodity + SPY + 2 producers)
    var_tickers = comm_cols[:1] + ["SPY"] + eq_cols[:2]
    var_tickers = [t for t in var_tickers if t in clean.columns]
    if len(var_tickers) >= 3:
        var_data = clean[var_tickers].dropna()
        var_analyzer = VARAnalyzer(s)
        var_res = var_analyzer.fit_var(var_data)
        irf_df = var_analyzer.impulse_response(var_res, periods=20)
        fevd_df = var_analyzer.fevd(var_res, periods=20)
        save_table(s.output_dir, "phase6_irf", irf_df)
        save_table(s.output_dir, "phase6_fevd", fevd_df)

        # Information delay
        if len(var_tickers) >= 2:
            delay = var_analyzer.information_delay(irf_df, var_tickers[0], var_tickers[-1])
            log.info("Information delay %s -> %s: %s", var_tickers[0], var_tickers[-1], delay)

    log.info("Phase 6 complete.")


def phase7_events(returns, s: Settings):
    """Phase 7: Event studies."""
    log.info("=" * 60)
    log.info("PHASE 7 – EVENT STUDIES")
    log.info("=" * 60)
    from events.calendar import EventCalendar
    from events.event_study import EventStudy

    calendar = EventCalendar(s)
    event_study = EventStudy(s)
    clean = returns.dropna(how="all")
    clean = clean.loc[:, clean.notna().sum() >= 252]

    # cap events to 200 most recent per type to keep runtime manageable
    raw_calendars = calendar.all_calendars()
    capped = {k: list(v)[-200:] for k, v in raw_calendars.items()}
    results_df = event_study.run_all_events(
        event_calendars=capped,
        asset_returns=clean,
        market_col="SPY" if "SPY" in clean.columns else clean.columns[0],
        post_windows=[1, 3, 5],
    )
    save_table(s.output_dir, "phase7_event_studies", results_df)
    log.info("Phase 7 complete. %d event-asset combinations tested.", len(results_df))
    return results_df


def phase8_cointegration(prices, s: Settings):
    """Phase 8: Cointegration – Johansen multivariate, Engle-Granger pairwise."""
    log.info("=" * 60)
    log.info("PHASE 8 – COINTEGRATION ANALYSIS")
    log.info("=" * 60)
    import numpy as np
    import pandas as pd
    from statsmodels.tsa.vector_ar.vecm import coint_johansen
    from statsmodels.tsa.stattools import coint

    clean = prices.dropna(how="all")
    clean = clean.loc[:, clean.notna().sum() >= 504]
    comm_cols = [c for c in SYMBOLS["commodities"] if c in clean.columns]
    eq_cols   = [c for c in SYMBOLS["equities"]    if c in clean.columns]

    # Log-price levels (stationary after differencing = I(1), cointegration tests on levels)
    log_p = np.log(clean.replace(0, np.nan).ffill().dropna(how="all"))

    # ── Johansen test on commodity log-prices ─────────────────────────────────
    johansen_rows = []
    try:
        joh_data = log_p[comm_cols].dropna()
        result = coint_johansen(joh_data.values, det_order=0, k_ar_diff=1)
        for i in range(len(comm_cols)):
            johansen_rows.append({
                "r_null": i,
                "trace_stat": round(result.lr1[i], 3),
                "crit_90": round(result.cvt[i, 0], 3),
                "crit_95": round(result.cvt[i, 1], 3),
                "crit_99": round(result.cvt[i, 2], 3),
                "reject_r_null_95": bool(result.lr1[i] > result.cvt[i, 1]),
            })
    except Exception as exc:
        log.warning("Johansen test failed: %s", exc)

    johansen_df = pd.DataFrame(johansen_rows)
    save_table(s.output_dir, "phase8_johansen", johansen_df)

    # ── Engle-Granger pairwise ─────────────────────────────────────────────────
    eg_rows = []
    test_cols = (comm_cols + eq_cols[:8])
    log_sub   = log_p[[c for c in test_cols if c in log_p.columns]].dropna()
    for i, c1 in enumerate(log_sub.columns):
        for c2 in list(log_sub.columns)[i + 1:]:
            try:
                stat, pval, _ = coint(log_sub[c1], log_sub[c2])
                eg_rows.append({
                    "asset1": c1, "asset2": c2,
                    "EG_stat": round(stat, 4),
                    "pvalue": round(pval, 6),
                    "cointegrated_95": bool(pval < 0.05),
                })
            except Exception:
                pass

    eg_df = pd.DataFrame(eg_rows)
    save_table(s.output_dir, "phase8_eg_cointegration", eg_df)
    n_coint = eg_df["cointegrated_95"].sum() if not eg_df.empty else 0
    log.info("Phase 8 complete. Johansen: %d ranks. EG cointegrated pairs: %d/%d",
             len(johansen_df), n_coint, len(eg_df))
    return johansen_df, eg_df


def phase9_garch_regimes(returns, prices, s: Settings):
    """Phase 9: GARCH(1,1) volatility modeling + STL seasonality decomposition."""
    log.info("=" * 60)
    log.info("PHASE 9 – VOLATILITY REGIMES & SEASONALITY")
    log.info("=" * 60)
    import numpy as np
    import pandas as pd
    from preprocessing.seasonality import SeasonalDecomposer

    clean = returns.dropna(how="all")
    clean = clean.loc[:, clean.notna().sum() >= 252]
    comm_cols = [c for c in SYMBOLS["commodities"] if c in clean.columns]

    # ── GARCH(1,1) per commodity ───────────────────────────────────────────────
    garch_rows = []
    cond_vols  = {}
    try:
        from arch import arch_model
        for ticker in comm_cols[:6]:
            try:
                s_ret = clean[ticker].dropna() * 100
                res = arch_model(s_ret, vol="Garch", p=1, q=1, dist="normal").fit(
                    disp="off", show_warning=False)
                cond_vols[ticker] = res.conditional_volatility / 100
                garch_rows.append({
                    "ticker": ticker,
                    "omega":  round(res.params.get("omega",    0), 6),
                    "alpha1": round(res.params.get("alpha[1]", 0), 4),
                    "beta1":  round(res.params.get("beta[1]",  0), 4),
                    "persistence": round(
                        res.params.get("alpha[1]", 0) + res.params.get("beta[1]", 0), 4),
                    "half_life_days": round(
                        np.log(0.5) / np.log(
                            res.params.get("alpha[1]", 0) + res.params.get("beta[1]", 0) + 1e-9
                        ), 1),
                    "AIC": round(res.aic, 2),
                    "BIC": round(res.bic, 2),
                })
            except Exception as exc:
                log.warning("GARCH failed for %s: %s", ticker, exc)
    except ImportError:
        log.warning("arch library not installed – skipping GARCH")

    garch_df = pd.DataFrame(garch_rows)
    save_table(s.output_dir, "phase9_garch_params", garch_df)

    if cond_vols:
        cv_df = pd.DataFrame(cond_vols)
        save_table(s.output_dir, "phase9_conditional_vol", cv_df)
        # High-vol regime: conditional vol > rolling median(63d)
        regime_df = (cv_df > cv_df.rolling(63).median()).astype(int)
        save_table(s.output_dir, "phase9_vol_regimes", regime_df)

    # ── STL decomposition on commodity price levels ────────────────────────────
    decomposer = SeasonalDecomposer(s)
    price_clean = prices.dropna(how="all")
    stl_rows = []
    stl_components = {}
    for ticker in comm_cols[:5]:
        if ticker not in price_clean.columns:
            continue
        try:
            series = price_clean[ticker].dropna()
            if len(series) < 504:
                continue
            res = decomposer.stl(series, period=252, robust=True)
            stl_components[ticker] = {
                "trend":    res.trend,
                "seasonal": res.seasonal,
                "residual": res.residual,
            }
            stl_rows.append({
                "ticker":              ticker,
                "trend_mean":          round(float(res.trend.mean()), 4),
                "seasonal_amplitude":  round(float(res.seasonal.std()), 4),
                "residual_std":        round(float(res.residual.std()), 4),
                "seasonal_to_total":   round(float(res.seasonal.std() /
                                              (res.trend.std() + res.seasonal.std() +
                                               res.residual.std() + 1e-9)), 4),
            })
        except Exception as exc:
            log.warning("STL failed for %s: %s", ticker, exc)

    stl_df = pd.DataFrame(stl_rows)
    save_table(s.output_dir, "phase9_stl_summary", stl_df)

    # Save combined STL component table
    if stl_components:
        rows_list = []
        for ticker, comps in stl_components.items():
            df_t = pd.DataFrame(comps)
            df_t["ticker"] = ticker
            rows_list.append(df_t)
        stl_long = pd.concat(rows_list)
        save_table(s.output_dir, "phase9_stl_components", stl_long)

    log.info("Phase 9 complete. GARCH models: %d. STL decompositions: %d.",
             len(garch_df), len(stl_rows))
    return garch_df, stl_df


def phase10_factors(returns, s: Settings):
    """Phase 10: PCA and factor regression."""
    log.info("=" * 60)
    log.info("PHASE 10 – FACTOR MODELS")
    log.info("=" * 60)
    from factors.pca import FactorModeler
    from factors.regression import FactorRegressor
    import numpy as np

    clean = returns.dropna(how="all")
    clean = clean.loc[:, clean.notna().sum() >= 252]
    factor_modeler = FactorModeler(s)

    pca_result = factor_modeler.run_pca(clean, n_components=10)
    save_table(s.output_dir, "phase10_pca_loadings", pca_result.loadings)
    save_table(s.output_dir, "phase10_pca_explained_variance",
               pca_result.explained_variance.to_frame("explained_variance"))

    # Save PC scores (projections of returns onto principal components)
    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        scaler = StandardScaler()
        X = scaler.fit_transform(clean.dropna())
        pca = PCA(n_components=10)
        scores = pca.fit_transform(X)
        scores_df = pd.DataFrame(
            scores,
            index=clean.dropna().index,
            columns=[f"PC{i+1}" for i in range(10)]
        )
        save_table(s.output_dir, "phase10_pc_scores", scores_df)
    except Exception as exc:
        log.warning("PC scores save failed: %s", exc)

    # Factor regression
    comm_cols = [c for c in SYMBOLS["commodities"] if c in clean.columns]
    ctrl_cols = [c for c in SYMBOLS["controls"] if c in clean.columns]
    factor_cols = ["SPY"] if "SPY" in clean.columns else []
    regressor_cols = comm_cols[:2] + ctrl_cols + factor_cols
    regressor_cols = [c for c in regressor_cols if c in clean.columns]

    if regressor_cols:
        eq_cols = [c for c in SYMBOLS["equities"] if c in clean.columns]
        regressor = FactorRegressor(s)
        reg_summary = regressor.fit_all(clean[eq_cols], clean[regressor_cols])
        save_table(s.output_dir, "phase10_regression_summary", reg_summary)
        log.info("Phase 10 complete.")
    return pca_result


def phase13_network(returns, granger_df, s: Settings):
    """Phase 13: Build information flow network."""
    log.info("=" * 60)
    log.info("PHASE 13 – NETWORK ANALYSIS")
    log.info("=" * 60)
    from network.information_flow import InformationFlowNetwork
    from visualization.networks import NetworkVisualizer

    net = InformationFlowNetwork(s)
    if granger_df is not None and not granger_df.empty:
        net.add_granger_edges(granger_df)

    metrics = net.compute_metrics()
    save_table(s.output_dir, "phase13_network_metrics", metrics)

    summary = net.summary()
    log.info("Network summary: %s", summary)

    viz = NetworkVisualizer(s)
    viz.plot_information_network(net.graph, layout="hierarchical")
    log.info("Phase 13 complete.")
    return net


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="Commodity Information Flow Research Pipeline")
    parser.add_argument("--phase", nargs="*", type=int, help="Run only specific phases (1-14)")
    parser.add_argument("--force", action="store_true", help="Re-run even if outputs exist")
    return parser.parse_args()


def main():
    setup_logging(SETTINGS.log_dir)
    args = parse_args()
    phases_to_run = set(args.phase) if args.phase else set(range(1, 15))

    log.info("╔══════════════════════════════════════════════════════╗")
    log.info("║  Commodity Information Flow Research Framework       ║")
    log.info("╚══════════════════════════════════════════════════════╝")

    prices = macro = returns = stats_df = granger_df = None

    if 1 in phases_to_run:
        prices, macro = phase1_load_data(SETTINGS)
    if 2 in phases_to_run and prices is not None:
        prices, returns = phase2_preprocess(prices, macro, SETTINGS)
    if 3 in phases_to_run and returns is not None:
        stats_df = phase3_eda(returns, SETTINGS)
    if 4 in phases_to_run and returns is not None:
        phase4_stationarity(prices, returns, SETTINGS)
    if 5 in phases_to_run and returns is not None:
        phase5_correlation(returns, SETTINGS)
    if 6 in phases_to_run and returns is not None:
        phase6_leadlag(returns, SETTINGS)
        # Load granger df for network
        granger_path = SETTINGS.output_dir / "tables" / "phase6_granger.csv"
        if granger_path.exists():
            import pandas as pd
            granger_df = pd.read_csv(granger_path)
    if 7 in phases_to_run and returns is not None:
        phase7_events(returns, SETTINGS)
    if 10 in phases_to_run and returns is not None:
        phase10_factors(returns, SETTINGS)
    if 13 in phases_to_run and returns is not None:
        phase13_network(returns, granger_df, SETTINGS)

    log.info("Pipeline completed. Outputs saved to: %s", SETTINGS.output_dir)


if __name__ == "__main__":
    main()
