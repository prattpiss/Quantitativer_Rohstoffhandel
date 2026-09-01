"""Ticker-Universum nach CONTINUATION_PROMPT §4/§5/§6.

Auswahl-Logik (Market-Cap-Konvergenz): Aufnahme absteigend nach Marktkapitalisierung,
bis der marginale Beitrag einer Aktie unter 1 % der kumulierten Sektor-Marktkapitalisierung
fällt bzw. die Aktie unter 500 Mio USD liegt. Delistete Werte (SAVE, PDCE, CHK, ...)
werden bewusst mitgeführt und erst zur Laufzeit anhand der Datenverfügbarkeit verworfen —
so bleibt die Verwerfung dokumentiert statt still (Survivorship-Bias-Transparenz).
"""
from __future__ import annotations

AIRLINES = ["DAL", "UAL", "AAL", "LUV", "JBLU", "ALGT", "SAVE", "HA", "ULCC",
            "RYAAY", "ICAGY", "DLAKY", "AZUL", "GOL", "CPA", "VLRS"]

ENERGY = ["XOM", "CVX", "COP", "EOG", "MPC", "PSX", "VLO", "HES", "DVN", "FANG",
          "OXY", "SLB", "HAL", "BKR", "MRO", "APA", "CTRA", "SM", "RRC", "AR",
          "EQT", "CRC", "MTDR"]

MATERIALS_COMMODITY = ["FCX", "NEM", "GOLD", "AEM", "AGI", "WPM", "KGC", "PAAS",
                       "VALE", "RIO", "BHP", "AA", "X", "NUE", "STLD", "CLF",
                       "MP", "SCCO", "CENX", "CSTM"]

MATERIALS_CHEMICAL = ["LIN", "APD", "ECL", "SHW", "PPG", "DD", "DOW", "EMN", "ALB",
                      "AMCR", "AVY", "IP", "PKG", "CF", "MOS", "NTR", "FMC", "IFF",
                      "RPM", "FUL"]

AEROSPACE_DEFENSE = ["BA", "LMT", "RTX", "NOC", "GD", "HII", "LHX", "LDOS", "CACI",
                     "BWXT", "TDG", "HEI", "CW", "MOG-A", "AXON"]

SECTOR_ETFS = ["XLK", "XLV", "XLF", "XLU", "XLP", "XLY", "XLI", "XLE", "XLB",
               "XLRE", "XLC"]

COMMODITY_ETFS = ["GLD", "SLV", "GDX", "GDXJ", "COPX", "SIL", "REMX", "PALL", "PPLT"]

THEME_ETFS = ["ITA", "XAR", "PPA", "VAW", "JETS", "IYT", "IBB", "XBI"]

FUTURES = ["CL=F", "BZ=F", "NG=F", "GC=F", "SI=F", "HG=F"]

MACRO = ["^VIX", "^VIX9D", "^TNX", "^IRX", "SPY", "HYG", "IEF", "TLT",
         "DX-Y.NYB", "LQD", "^GSPC"]

PHARMA = ["PFE", "MRNA", "BNTX", "JNJ", "GILD", "REGN", "LLY", "ABBV"]

DEFENSIVE = ["XLU", "GLD", "TLT", "XLP"]

# Sektor-Zuordnung für Färbung / Gruppierung in den Reports
GROUPS: dict[str, list[str]] = {
    "Airlines": AIRLINES,
    "Energie": ENERGY,
    "Materials (Rohstoff)": MATERIALS_COMMODITY,
    "Materials (Chemie)": MATERIALS_CHEMICAL,
    "Aerospace & Defense": AEROSPACE_DEFENSE,
    "Sektor-ETFs": SECTOR_ETFS,
    "Rohstoff-ETFs": COMMODITY_ETFS,
    "Themen-ETFs": THEME_ETFS,
    "Futures": FUTURES,
    "Makro": MACRO,
}

GROUP_COLOURS = {
    "Airlines": "#ffa657", "Energie": "#d29922", "Materials (Rohstoff)": "#bc8cff",
    "Materials (Chemie)": "#7ee787", "Aerospace & Defense": "#ff7b72",
    "Sektor-ETFs": "#58a6ff", "Rohstoff-ETFs": "#d2a8ff", "Themen-ETFs": "#39d353",
    "Futures": "#e3b341", "Makro": "#8b949e",
}

SUBSECTORS = {
    "Precious Metals": ["GLD", "SLV", "GDX", "NEM", "GOLD", "AEM", "WPM", "KGC", "PAAS"],
    "Base Metals": ["COPX", "FCX", "VALE", "SCCO", "RIO", "BHP", "X", "NUE", "STLD", "AA"],
    "Energy Complex": ["CL=F", "BZ=F", "NG=F", "XLE", "XOM", "CVX", "VLO", "MPC"],
    "Chemicals": ["LIN", "APD", "ECL", "SHW", "PPG", "DD", "DOW", "EMN"],
    "Fertilizer": ["CF", "MOS", "NTR", "FMC"],
}


def group_of(ticker: str) -> str:
    for g, members in GROUPS.items():
        if ticker in members:
            return g
    return "Sonstige"


def sector_rotation_universe() -> list[str]:
    seen, out = set(), []
    for lst in (SECTOR_ETFS, THEME_ETFS, COMMODITY_ETFS, FUTURES, AIRLINES, ENERGY,
                MATERIALS_COMMODITY, MATERIALS_CHEMICAL, AEROSPACE_DEFENSE, ["SPY", "^VIX"]):
        for t in lst:
            if t not in seen:
                seen.add(t)
                out.append(t)
    return out
