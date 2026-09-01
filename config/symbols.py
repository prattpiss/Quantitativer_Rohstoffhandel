"""
Central universe definition.

Symbols are grouped by asset class and market-cap tier.
The proximity_level attribute encodes distance to the raw commodity:
  0 = commodity itself
  1 = direct producer (E&P, miner)
  2 = service / royalty company
  3 = downstream processing
  4 = industrial consumer
  5 = end-consumer sector
  6 = broad market index
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class AssetMeta:
    ticker: str
    name: str
    asset_class: str          # "commodity" | "etf" | "equity" | "index" | "control"
    sector: str = ""
    cap_tier: str = ""        # "mega" | "large" | "mid" | "small" | ""
    proximity_level: int = -1  # 0-6, see module docstring; -1 = not applicable
    commodities: List[str] = field(default_factory=list)  # related commodity tickers


# ── Commodity Futures ─────────────────────────────────────────────────────────
COMMODITIES: Dict[str, AssetMeta] = {
    "CL=F": AssetMeta("CL=F", "WTI Crude Oil", "commodity", "Energy", proximity_level=0),
    "BZ=F": AssetMeta("BZ=F", "Brent Crude Oil", "commodity", "Energy", proximity_level=0),
    "NG=F": AssetMeta("NG=F", "Natural Gas", "commodity", "Energy", proximity_level=0),
    "GC=F": AssetMeta("GC=F", "Gold", "commodity", "Metals", proximity_level=0),
    "SI=F": AssetMeta("SI=F", "Silver", "commodity", "Metals", proximity_level=0),
    "HG=F": AssetMeta("HG=F", "Copper", "commodity", "Metals", proximity_level=0),
    "ZC=F": AssetMeta("ZC=F", "Corn", "commodity", "Agriculture", proximity_level=0),
    "ZW=F": AssetMeta("ZW=F", "Wheat", "commodity", "Agriculture", proximity_level=0),
    "ZS=F": AssetMeta("ZS=F", "Soybeans", "commodity", "Agriculture", proximity_level=0),
}

# ── Sector ETFs ───────────────────────────────────────────────────────────────
ETFS: Dict[str, AssetMeta] = {
    "XLE": AssetMeta("XLE", "Energy Select Sector SPDR", "etf", "Energy", proximity_level=1,
                     commodities=["CL=F", "NG=F", "BZ=F"]),
    "XLB": AssetMeta("XLB", "Materials Select Sector SPDR", "etf", "Materials", proximity_level=2,
                     commodities=["GC=F", "SI=F", "HG=F"]),
    "XLI": AssetMeta("XLI", "Industrial Select Sector SPDR", "etf", "Industrials", proximity_level=4),
    "GDX": AssetMeta("GDX", "VanEck Gold Miners ETF", "etf", "Metals", proximity_level=1,
                     commodities=["GC=F"]),
    "SIL": AssetMeta("SIL", "Global X Silver Miners ETF", "etf", "Metals", proximity_level=1,
                     commodities=["SI=F"]),
    "JETS": AssetMeta("JETS", "US Global Jets ETF", "etf", "Aviation", proximity_level=5,
                      commodities=["CL=F", "BZ=F"]),
    "IYT": AssetMeta("IYT", "iShares Transportation Average ETF", "etf", "Transportation",
                     proximity_level=4, commodities=["CL=F"]),
}

# ── Equities – Mega Cap Producers ─────────────────────────────────────────────
MEGA_CAPS: Dict[str, AssetMeta] = {
    "XOM": AssetMeta("XOM", "Exxon Mobil", "equity", "Energy", "mega", 1,
                     ["CL=F", "NG=F", "BZ=F"]),
    "CVX": AssetMeta("CVX", "Chevron", "equity", "Energy", "mega", 1,
                     ["CL=F", "NG=F", "BZ=F"]),
    "FCX": AssetMeta("FCX", "Freeport-McMoRan", "equity", "Metals", "mega", 1,
                     ["HG=F", "GC=F"]),
    "NEM": AssetMeta("NEM", "Newmont", "equity", "Metals", "mega", 1,
                     ["GC=F"]),
}

# ── Equities – Mid Cap Producers ─────────────────────────────────────────────
MID_CAPS: Dict[str, AssetMeta] = {
    "APA": AssetMeta("APA", "APA Corp", "equity", "Energy", "mid", 1,
                     ["CL=F", "NG=F"]),
    "OXY": AssetMeta("OXY", "Occidental Petroleum", "equity", "Energy", "mid", 1,
                     ["CL=F", "NG=F"]),
    "TECK": AssetMeta("TECK", "Teck Resources", "equity", "Metals", "mid", 1,
                      ["HG=F", "ZC=F"]),
}

# ── Equities – Small Cap Producers ───────────────────────────────────────────
SMALL_CAPS: Dict[str, AssetMeta] = {
    "SM": AssetMeta("SM", "SM Energy", "equity", "Energy", "small", 1,
                    ["CL=F", "NG=F"]),
    "TGB": AssetMeta("TGB", "Taseko Mines", "equity", "Metals", "small", 1,
                     ["HG=F"]),
    "GORO": AssetMeta("GORO", "Gold Resource Corp", "equity", "Metals", "small", 1,
                      ["GC=F", "SI=F"]),
}

# ── Market Indices ────────────────────────────────────────────────────────────
MARKET_INDICES: Dict[str, AssetMeta] = {
    "SPY": AssetMeta("SPY", "S&P 500 ETF", "index", "Broad Market", "mega", 6),
    "QQQ": AssetMeta("QQQ", "NASDAQ-100 ETF", "index", "Tech", "mega", 6),
    "IWM": AssetMeta("IWM", "Russell 2000 ETF", "index", "Broad Market", "small", 6),
    "IJH": AssetMeta("IJH", "S&P 400 Mid Cap ETF", "index", "Broad Market", "mid", 6),
    "MGC": AssetMeta("MGC", "Vanguard Mega Cap ETF", "index", "Broad Market", "mega", 6),
}

# ── Control Variables ─────────────────────────────────────────────────────────
CONTROLS: Dict[str, AssetMeta] = {
    "^VIX": AssetMeta("^VIX", "CBOE Volatility Index", "control", "Volatility"),
    "DX-Y.NYB": AssetMeta("DX-Y.NYB", "US Dollar Index", "control", "FX"),
    "^TNX": AssetMeta("^TNX", "10-Year Treasury Yield", "control", "Rates"),
}

# ── External / Alternative Factors ────────────────────────────────────────────
EXTERNAL_FACTORS: Dict[str, AssetMeta] = {
    "CNY=X":    AssetMeta("CNY=X",    "USD/CNY (China Yuan)", "external", "FX"),
    "AUDUSD=X": AssetMeta("AUDUSD=X", "AUD/USD (Australien-Dollar)", "external", "FX"),
    "BRL=X":    AssetMeta("BRL=X",    "USD/BRL (Brasilianischer Real)", "external", "FX"),
    "SBLK":     AssetMeta("SBLK",     "Star Bulk Carriers (BDI-Proxy)", "external", "Shipping"),
    "LIT":      AssetMeta("LIT",      "Global X Lithium ETF (EV-Nachfrage)", "external", "Technology"),
    "URA":      AssetMeta("URA",      "Global X Uranium ETF", "external", "Energy"),
    "REMX":     AssetMeta("REMX",     "VanEck Rare Earth ETF", "external", "Materials"),
    "PDBC":     AssetMeta("PDBC",     "Invesco Optimum Yield Commodity ETF", "external", "Commodity"),
}

# ── FRED Macro Series ─────────────────────────────────────────────────────────
FRED_SERIES: Dict[str, str] = {
    "CPIAUCSL": "CPI All Items",
    "PPIACO": "PPI All Commodities",
    "GDP": "Gross Domestic Product",
    "UNRATE": "Unemployment Rate",
    "PAYEMS": "Nonfarm Payrolls",
    "MANEMP": "Manufacturing Employment",
    "INDPRO": "Industrial Production Index",
    "HOUST": "Housing Starts",
    "RSXFS": "Retail Sales Ex Food Services",
    "FEDFUNDS": "Federal Funds Effective Rate",
    "DFF": "Fed Funds Rate (Daily)",
    # External drivers (new)
    "T5YIE": "5-Year Breakeven Inflation Rate (TIPS)",
    "T10YIE": "10-Year Breakeven Inflation Rate (TIPS)",
    "BAMLH0A0HYM2": "US High Yield OAS (Credit Risk)",
    "UMCSENT": "Univ. Michigan Consumer Sentiment",
    "DCOILBRENTEU": "Brent Crude Spot (EIA/FRED)",
    "DHHNGSP": "Henry Hub Natural Gas Spot",
}

# ── EIA Series ────────────────────────────────────────────────────────────────
EIA_SERIES: Dict[str, str] = {
    "PET.WCRSTUS1.W": "Weekly Crude Oil Stocks (US)",
    "NG.NW2_EPG0_SWO_R48_BCF.W": "Weekly Natural Gas Stocks",
}

# ── Combined universe ─────────────────────────────────────────────────────────
UNIVERSE: Dict[str, AssetMeta] = {
    **COMMODITIES,
    **ETFS,
    **MEGA_CAPS,
    **MID_CAPS,
    **SMALL_CAPS,
    **MARKET_INDICES,
    **CONTROLS,
    **EXTERNAL_FACTORS,
}

SYMBOLS: Dict[str, list[str]] = {
    "commodities": list(COMMODITIES),
    "etfs": list(ETFS),
    "mega_caps": list(MEGA_CAPS),
    "mid_caps": list(MID_CAPS),
    "small_caps": list(SMALL_CAPS),
    "market_indices": list(MARKET_INDICES),
    "controls": list(CONTROLS),
    "external_factors": list(EXTERNAL_FACTORS),
    "all_yahoo": list(UNIVERSE),
    "producers": list(MEGA_CAPS) + list(MID_CAPS) + list(SMALL_CAPS),
    "equities": list(MEGA_CAPS) + list(MID_CAPS) + list(SMALL_CAPS) + list(ETFS),
}
