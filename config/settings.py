"""
Global project configuration.
All parameters are centralised here so that analyses are fully reproducible.
"""

from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")  # load API keys before Settings is instantiated


@dataclass
class Settings:
    # ── Paths ─────────────────────────────────────────────────────────────────
    root: Path = ROOT
    cache_dir: Path = ROOT / "data_cache"
    output_dir: Path = ROOT / "outputs"
    log_dir: Path = ROOT / "logs"

    # ── Data ──────────────────────────────────────────────────────────────────
    start_date: str = "2000-01-01"
    end_date: str = "2026-08-30"          # current date
    default_freq: str = "B"               # business-day frequency
    cache_ttl_hours: int = 24

    # ── Returns ───────────────────────────────────────────────────────────────
    return_type: str = "log"              # "log" | "simple"
    winsorise_quantile: float = 0.005     # two-sided → 1 % total

    # ── Rolling windows ───────────────────────────────────────────────────────
    rolling_short: int = 21               # ≈ 1 month
    rolling_medium: int = 63              # ≈ 3 months
    rolling_long: int = 252              # ≈ 1 year

    # ── Lead-lag ──────────────────────────────────────────────────────────────
    max_lag: int = 60                     # trading days
    granger_max_lag: int = 20
    var_max_lag: int = 10
    significance_level: float = 0.05

    # ── Bootstrap / Monte Carlo ───────────────────────────────────────────────
    n_bootstrap: int = 2_000
    n_montecarlo: int = 5_000
    random_seed: int = 42

    # ── Event studies ─────────────────────────────────────────────────────────
    event_windows_minutes: list[int] = field(
        default_factory=lambda: [5, 15, 30, 60, 240]
    )
    event_windows_days: list[int] = field(
        default_factory=lambda: [1, 3, 5]
    )
    pre_event_days: int = 20
    post_event_days: int = 20

    # ── Factor models ─────────────────────────────────────────────────────────
    n_pca_components: int = 10
    factor_regression_lags: int = 5

    # ── Network ───────────────────────────────────────────────────────────────
    transfer_entropy_lag: int = 1
    network_threshold: float = 0.05      # p-value threshold for edges

    # ── API Keys (loaded from .env via load_dotenv above) ────────────────────
    fred_api_key: str = field(default_factory=lambda: os.getenv("FRED_API_KEY", ""))
    eia_api_key: str = field(default_factory=lambda: os.getenv("EIA_API_KEY", ""))

    def __post_init__(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "figures").mkdir(exist_ok=True)
        (self.output_dir / "tables").mkdir(exist_ok=True)
        (self.output_dir / "reports").mkdir(exist_ok=True)


# Singleton used throughout the project
SETTINGS = Settings()
