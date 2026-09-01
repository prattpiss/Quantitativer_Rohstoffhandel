"""
Full pipeline runner – all phases + HTML report generation.
"""
import sys
sys.path.insert(0, '.')

from utils.logging_utils import setup_logging
from config.settings import SETTINGS

setup_logging(SETTINGS.log_dir)

import pandas as pd
import main as pipeline
from reports.report_builder import build_all_reports

# ── Phase 1: Data Loading ────────────────────────────────────────────────────
prices, macro = pipeline.phase1_load_data(SETTINGS)

# ── Phase 2: Preprocessing ───────────────────────────────────────────────────
prices_clean, returns = pipeline.phase2_preprocess(prices, macro, SETTINGS)

# ── Phase 3: EDA ─────────────────────────────────────────────────────────────
pipeline.phase3_eda(returns, SETTINGS)

# ── Phase 4: Stationarity ────────────────────────────────────────────────────
pipeline.phase4_stationarity(prices_clean, returns, SETTINGS)

# ── Phase 5: Correlation ──────────────────────────────────────────────────────
pipeline.phase5_correlation(returns, SETTINGS)

# ── Phase 6: Lead-Lag ─────────────────────────────────────────────────────────
pipeline.phase6_leadlag(returns, SETTINGS)

# ── Phase 7: Event Studies ───────────────────────────────────────────────────
pipeline.phase7_events(returns, SETTINGS)

# ── Phase 8: Cointegration ───────────────────────────────────────────────────
pipeline.phase8_cointegration(prices_clean, SETTINGS)

# ── Phase 9: GARCH Volatility Regimes + STL ──────────────────────────────────
pipeline.phase9_garch_regimes(returns, prices_clean, SETTINGS)

# ── Phase 10: Factor Models ───────────────────────────────────────────────────
pipeline.phase10_factors(returns, SETTINGS)

# ── Phase 13: Network Analysis ───────────────────────────────────────────────
granger_path = SETTINGS.output_dir / "tables" / "phase6_granger.csv"
granger_df = pd.read_csv(granger_path) if granger_path.exists() else None
pipeline.phase13_network(returns, granger_df, SETTINGS)

# ── HTML Reports ─────────────────────────────────────────────────────────────
build_all_reports(SETTINGS.output_dir)

print("\nAll phases completed. Reports at:", SETTINGS.output_dir / "reports" / "index.html")
