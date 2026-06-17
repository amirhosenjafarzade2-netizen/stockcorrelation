"""
forex_fundamentals.py
══════════════════════════════════════════════════════════════════════════════
Forex Fundamental Analysis & Bias Scanner  ·  v2.0

Enhancements over v1.0
───────────────────────
NEW INDICATORS (Section 16 & Chapter extras)
  • PMI  —  composite Mfg+Services average from FRED (where available)
  • M2 Money Supply  —  YoY growth rate; leading indicator for GDP / inflation
  • Consumer Confidence  —  level + trend
  • Housing (Building Permits) for USD economy

NEW ANALYSIS MODULES
  • IXL Score Engine  (Section 13)  —  evaluate macro themes for tradeability
  • Pre-Meeting Trade Scanner  (Section 10)  —  alerts when CB meeting + setup aligns
  • COT Sentiment Layer  (Section 14)  —  reads CFTC COT CSV for non-comm positioning
  • Dominant Driver Detector  (Section 8)  —  carry / momentum / fundamental value
  • Election / Political Risk Flag  (Section 11)
  • "Buy the Rumour" Timing Score  (Section 12)  —  event-adjusted entry window

SCORING ENGINE UPGRADES
  • PMI contributes to currency score
  • M2 growth feeds into inflation outlook adjustment
  • COT net non-commercial position as sentiment overlay on fundamental bias
  • Real rate differential ranking against all other selected currencies
  • Full composite score + star rating (★ system)
  • Explicit "Pre-Meeting Trade" opportunity badge

UI UPGRADES
  • Six tabs: Overview · Carry · IXL · COT · Pre-Meeting · Pairs
  • DXY dashboard sidebar with gold, crude, equity index correlations
  • Data quality badges shown per indicator
  • Exportable summary table (CSV download)
  • "Expected Future vs Current" delta column for each macro metric
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
import io
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
# DATA QUALITY TRACKING
# ══════════════════════════════════════════════════════════════════════════════

class DataQualityLog:
    """Per-currency data quality warnings surfaced in the UI."""
    def __init__(self):
        self._warnings: Dict[str, List[str]] = {}

    def warn(self, code: str, msg: str):
        self._warnings.setdefault(code, []).append(msg)

    def get(self, code: str) -> List[str]:
        return self._warnings.get(code, [])

    def all(self) -> Dict[str, List[str]]:
        return self._warnings

    def any(self) -> bool:
        return bool(self._warnings)


# ══════════════════════════════════════════════════════════════════════════════
# CURRENCY UNIVERSE
# ══════════════════════════════════════════════════════════════════════════════

CURRENCY_INFO: Dict[str, Dict] = {
    "USD": {"name": "US Dollar",          "country": "United States",  "type": "Safe Haven",              "cb": "Federal Reserve",         "cb_meeting_months": [1,3,5,6,7,9,11,12]},
    "EUR": {"name": "Euro",               "country": "Euro Area",      "type": "Major",                   "cb": "ECB",                     "cb_meeting_months": [1,2,3,4,6,7,9,10,12]},
    "GBP": {"name": "British Pound",      "country": "United Kingdom", "type": "Major",                   "cb": "Bank of England",         "cb_meeting_months": [2,3,5,6,8,9,11,12]},
    "JPY": {"name": "Japanese Yen",        "country": "Japan",          "type": "Safe Haven",              "cb": "Bank of Japan",           "cb_meeting_months": [1,3,4,6,7,9,10,12]},
    "AUD": {"name": "Australian Dollar",  "country": "Australia",      "type": "Commodity-Linked",        "cb": "RBA",                     "cb_meeting_months": [2,3,5,6,8,9,11,12]},
    "CAD": {"name": "Canadian Dollar",    "country": "Canada",         "type": "Commodity-Linked",        "cb": "Bank of Canada",          "cb_meeting_months": [1,3,4,6,7,9,10,12]},
    "CHF": {"name": "Swiss Franc",        "country": "Switzerland",    "type": "Safe Haven",              "cb": "SNB",                     "cb_meeting_months": [3,6,9,12]},
    "NZD": {"name": "New Zealand Dollar", "country": "New Zealand",    "type": "Commodity-Linked",        "cb": "RBNZ",                    "cb_meeting_months": [2,4,5,7,8,10,11]},
    "CNY": {"name": "Chinese Yuan",       "country": "China",          "type": "Policy-Driven",           "cb": "PBOC",                    "cb_meeting_months": []},
    "TRY": {"name": "Turkish Lira",       "country": "Türkiye",        "type": "Policy-Driven",           "cb": "CBRT",                    "cb_meeting_months": [1,2,3,4,5,6,7,8,9,10,11,12]},
    "MXN": {"name": "Mexican Peso",       "country": "Mexico",         "type": "Policy-Driven / Carry",   "cb": "Banxico",                 "cb_meeting_months": [2,3,5,6,8,9,11,12]},
    "ZAR": {"name": "South African Rand", "country": "South Africa",   "type": "Policy-Driven / Carry",   "cb": "SARB",                    "cb_meeting_months": [1,3,5,7,9,11]},
}

# ── FRED Series Map ──────────────────────────────────────────────────────────
FRED_SERIES: Dict[str, Dict[str, str]] = {
    "USD": {
        "policy_rate":      "DFEDTARU",
        "cpi_yoy":          "CPALTT01USM659N",
        "gdp":              "A191RL1Q225SBEA",
        "trade_balance":    "BOPGSTB",
        "unemployment":     "UNRATE",
        "ten_year":         "DGS10",
        "pmi_mfg":          "MANEMP",        # ISM Manufacturing (proxy)
        "m2":               "M2SL",          # M2 Money Supply level (billions)
        "consumer_conf":    "UMCSENT",       # UMich Consumer Sentiment
        "building_permits": "PERMIT",        # Building Permits (thousands)
    },
    "EUR": {
        "policy_rate":      "ECBDFR",
        "cpi_yoy":          "CP0000EZ19M086NEST",
        "gdp":              "CLVMNACSCAB1GQEA19",
        "trade_balance":    "XTNTVA01EZQ667S",
        "unemployment":     "LRHUTTTTEZM156S",
        "ten_year":         "IRLTLT01EZM156N",
        "m2":               "MABMM301EZM189S",
        "pmi_mfg":          "",
        "consumer_conf":    "CSCICP02EZM460S",
        "building_permits": "",
    },
    "GBP": {
        "policy_rate":      "BOERUKM",
        "cpi_yoy":          "GBRCPIALLMINMEI",
        "gdp":              "NGDPRSAXDCGBQ",
        "trade_balance":    "XTNTVA01GBQ667S",
        "unemployment":     "LRHUTTTTGBM156S",
        "ten_year":         "IRLTLT01GBM156N",
        "m2":               "MABMM301GBM189S",
        "pmi_mfg":          "",
        "consumer_conf":    "CSCICP02GBM460S",
        "building_permits": "",
    },
    "JPY": {
        "policy_rate":      "IRSTCB01JPM156N",
        "cpi_yoy":          "JPNCPIALLMINMEI",
        "gdp":              "JPNRGDPEXP",
        "trade_balance":    "XTNTVA01JPQ667S",
        "unemployment":     "LRHUTTTTJPM156S",
        "ten_year":         "IRLTLT01JPM156N",
        "m2":               "MABMM301JPM189S",
        "pmi_mfg":          "",
        "consumer_conf":    "CSCICP02JPM460S",
        "building_permits": "",
    },
    "AUD": {
        "policy_rate":      "IRSTCB01AUM156N",
        "cpi_yoy":          "AUSCPIALLQINMEI",
        "gdp":              "AUSGDPRQDSMEI",
        "trade_balance":    "XTNTVA01AUQ667S",
        "unemployment":     "LRHUTTTTAUM156S",
        "ten_year":         "IRLTLT01AUM156N",
        "m2":               "MABMM301AUM189S",
        "pmi_mfg":          "",
        "consumer_conf":    "CSCICP02AUM460S",
        "building_permits": "",
    },
    "CAD": {
        "policy_rate":      "IRSTCB01CAM156N",
        "cpi_yoy":          "CANCPIALLMINMEI",
        "gdp":              "NGDPRSAXDCCAQ",
        "trade_balance":    "XTNTVA01CAQ667S",
        "unemployment":     "LRHUTTTTCAM156S",
        "ten_year":         "IRLTLT01CAM156N",
        "m2":               "MABMM301CAM189S",
        "pmi_mfg":          "",
        "consumer_conf":    "CSCICP02CAM460S",
        "building_permits": "",
    },
    "CHF": {
        "policy_rate":      "IRSTCB01CHM156N",
        "cpi_yoy":          "CHECPIALLMINMEI",
        "gdp":              "NGDPRSAXDCCHQ",
        "trade_balance":    "XTNTVA01CHQ667S",
        "unemployment":     "LRHUTTTTCHM156S",
        "ten_year":         "IRLTLT01CHM156N",
        "m2":               "",
        "pmi_mfg":          "",
        "consumer_conf":    "",
        "building_permits": "",
    },
    "NZD": {
        "policy_rate":      "IRSTCB01NZM156N",
        "cpi_yoy":          "NZLCPIALLQINMEI",
        "gdp":              "NGDPRSAXDCNZQ",
        "trade_balance":    "XTNTVA01NZQ667S",
        "unemployment":     "LRHUTTTTNZM156S",
        "ten_year":         "IRLTLT01NZM156N",
        "m2":               "",
        "pmi_mfg":          "",
        "consumer_conf":    "",
        "building_permits": "",
    },
    "CNY": {
        "policy_rate":      "INTDSRCNM193N",
        "cpi_yoy":          "CHNCPIALLMINMEI",
        "gdp":              "NGDPRSAXDCCNQ",
        "trade_balance":    "XTNTVA01CNQ667S",
        "unemployment":     "LRUN64TTCNQ156S",
        "ten_year":         "IRLTLT01CNM156N",
        "m2":               "",
        "pmi_mfg":          "",
        "consumer_conf":    "",
        "building_permits": "",
    },
    "TRY": {
        "policy_rate":      "INTDSRTRM193N",
        "cpi_yoy":          "TURCPIALLMINMEI",
        "gdp":              "NGDPRSAXDCTRQ",
        "trade_balance":    "XTNTVA01TRQ667S",
        "unemployment":     "LRHUTTTTTRM156S",
        "ten_year":         "IRLTLT01TRM156N",
        "m2":               "",
        "pmi_mfg":          "",
        "consumer_conf":    "",
        "building_permits": "",
    },
    "MXN": {
        "policy_rate":      "INTDSRMXM193N",
        "cpi_yoy":          "MEXCPIALLMINMEI",
        "gdp":              "NGDPRSAXDCMXQ",
        "trade_balance":    "XTNTVA01MXQ667S",
        "unemployment":     "LRHUTTTTMXM156S",
        "ten_year":         "IRLTLT01MXM156N",
        "m2":               "",
        "pmi_mfg":          "",
        "consumer_conf":    "",
        "building_permits": "",
    },
    "ZAR": {
        "policy_rate":      "INTDSRZAM193N",
        "cpi_yoy":          "ZAFCPIALLMINMEI",
        "gdp":              "NGDPRSAXDCZAQ",
        "trade_balance":    "XTNTVA01ZAQ667S",
        "unemployment":     "LRHUTTTTZAM156S",
        "ten_year":         "IRLTLT01ZAM156N",
        "m2":               "",
        "pmi_mfg":          "",
        "consumer_conf":    "",
        "building_permits": "",
    },
}

# ── FX tickers ───────────────────────────────────────────────────────────────
FX_TICKER: Dict[str, str] = {
    "EUR": "EURUSD=X", "GBP": "GBPUSD=X", "AUD": "AUDUSD=X", "NZD": "NZDUSD=X",
    "JPY": "USDJPY=X", "CAD": "USDCAD=X", "CHF": "USDCHF=X",
    "CNY": "USDCNY=X", "TRY": "USDTRY=X", "MXN": "USDMXN=X", "ZAR": "USDZAR=X",
}
USD_IS_BASE = {"JPY", "CAD", "CHF", "CNY", "TRY", "MXN", "ZAR"}
DXY_TICKER  = "DX-Y.NYB"

# Carry pair definitions (Chapter 7 §9 Classic Carry Pairs)
CLASSIC_CARRY_PAIRS = [
    ("AUD", "JPY"), ("NZD", "JPY"), ("AUD", "CHF"), ("NZD", "CHF"),
    ("MXN", "JPY"), ("ZAR", "JPY"), ("AUD", "EUR"), ("NZD", "EUR"),
]

# ══════════════════════════════════════════════════════════════════════════════
# FRED FETCH LAYER
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fred_series(series_id: str) -> pd.Series:
    """Pull one FRED series via public CSV endpoint (no API key required)."""
    if not series_id:
        return pd.Series(dtype=float)
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    try:
        df = pd.read_csv(url)
        if df.empty or df.shape[1] < 2:
            return pd.Series(dtype=float)
        date_col, val_col = df.columns[0], df.columns[1]
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df[val_col]  = pd.to_numeric(df[val_col], errors="coerce")
        df = df.dropna(subset=[date_col]).set_index(date_col)[val_col]
        return df.dropna()
    except Exception:
        return pd.Series(dtype=float)


def fetch_currency_macro(code: str, dq_log: DataQualityLog) -> Dict[str, pd.Series]:
    """Fetch all FRED series for one currency."""
    out: Dict[str, pd.Series] = {}
    for label, series_id in FRED_SERIES.get(code, {}).items():
        s = fetch_fred_series(series_id)
        if s.empty and series_id:
            dq_log.warn(code, f"FRED '{series_id}' ({label}) unavailable")
        out[label] = s
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fx_history(ticker: str, period: str = "2y") -> pd.Series:
    """Fetch spot FX history from yfinance."""
    if not ticker:
        return pd.Series(dtype=float)
    try:
        data = yf.download(ticker, period=period, interval="1d",
                            auto_adjust=True, progress=False)
        if "Close" in data.columns:
            s = data["Close"]
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]
            return s.dropna()
        return pd.Series(dtype=float)
    except Exception:
        return pd.Series(dtype=float)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_cot_data() -> pd.DataFrame:
    """
    Fetch CFTC COT legacy futures-only report from CFTC public CSV.
    Returns a DataFrame indexed by date with non-commercial net position
    per currency (using standard CFTC market names).
    Falls back to empty DataFrame if unavailable.
    """
    # CFTC publishes annual files; try current year then prior year.
    year = datetime.now().year
    frames = []
    for y in [year, year - 1]:
        url = f"https://www.cftc.gov/files/dea/history/fut_fin_xls_{y}.zip"
        try:
            df = pd.read_csv(url, compression="zip", low_memory=False)
            frames.append(df)
            break
        except Exception:
            pass
    if not frames:
        return pd.DataFrame()
    df = frames[0]

    # Standardise column names
    df.columns = [c.strip() for c in df.columns]
    required = {"Market_and_Exchange_Names", "As_of_Date_in_Form_YYMMDD",
                 "NonComm_Positions_Long_All", "NonComm_Positions_Short_All"}
    if not required.issubset(df.columns):
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["As_of_Date_in_Form_YYMMDD"].astype(str),
                                  format="%y%m%d", errors="coerce")
    df = df.dropna(subset=["date"])
    df["net_noncomm"] = (pd.to_numeric(df["NonComm_Positions_Long_All"], errors="coerce")
                         - pd.to_numeric(df["NonComm_Positions_Short_All"], errors="coerce"))
    df["market"] = df["Market_and_Exchange_Names"].str.upper().str.strip()
    return df[["date", "market", "net_noncomm"]].dropna()


# Map CFTC market name fragments → our currency codes
COT_MARKET_MAP: Dict[str, str] = {
    "EURO FX":          "EUR",
    "BRITISH POUND":    "GBP",
    "JAPANESE YEN":     "JPY",
    "AUSTRALIAN DOLLAR": "AUD",
    "CANADIAN DOLLAR":  "CAD",
    "SWISS FRANC":      "CHF",
    "NEW ZEALAND DOLLAR": "NZD",
    "MEXICAN PESO":     "MXN",
}


def get_cot_position(code: str, cot_df: pd.DataFrame) -> Dict:
    """Extract latest & prior COT net non-commercial position for one currency."""
    if cot_df.empty:
        return {"net": np.nan, "prev": np.nan, "change": np.nan, "bias": "—"}
    fragment = next((k for k, v in COT_MARKET_MAP.items() if v == code), None)
    if not fragment:
        return {"net": np.nan, "prev": np.nan, "change": np.nan, "bias": "—"}
    mask = cot_df["market"].str.contains(fragment, na=False)
    sub = cot_df.loc[mask].sort_values("date")
    if len(sub) < 2:
        return {"net": np.nan, "prev": np.nan, "change": np.nan, "bias": "—"}
    net  = float(sub.iloc[-1]["net_noncomm"])
    prev = float(sub.iloc[-2]["net_noncomm"])
    chg  = net - prev
    bias = "Bullish" if net > 0 else ("Bearish" if net < 0 else "Neutral")
    return {"net": net, "prev": prev, "change": chg, "bias": bias}


# ══════════════════════════════════════════════════════════════════════════════
# HELPER UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def latest(s: pd.Series) -> float:
    if s is None or s.empty: return np.nan
    v = s.dropna()
    return float(v.iloc[-1]) if len(v) else np.nan


def value_n_periods_ago(s: pd.Series, n: int) -> float:
    if s is None or s.empty: return np.nan
    v = s.dropna()
    return float(v.iloc[-(n + 1)]) if len(v) >= n + 1 else np.nan


def trend_direction(s: pd.Series, lookback: int = 3) -> str:
    if s is None or s.empty: return "unknown"
    v = s.dropna()
    if len(v) < 2: return "unknown"
    if len(v) < lookback + 1:
        diff = v.iloc[-1] - v.iloc[-2]
    else:
        recent = v.iloc[-lookback:].mean()
        prior  = v.iloc[-(lookback * 2):-lookback].mean() if len(v) >= lookback * 2 else v.iloc[0]
        diff   = recent - prior
    if pd.isna(diff): return "unknown"
    if diff > 0.05:  return "rising"
    if diff < -0.05: return "falling"
    return "flat"


def yoy_growth(s: pd.Series, periods: int = 12) -> pd.Series:
    """YoY % change for a level series."""
    if s is None or len(s.dropna()) < periods + 1:
        return pd.Series(dtype=float)
    return s.pct_change(periods) * 100


def m2_yoy_growth(m2_series: pd.Series) -> float:
    """M2 YoY growth rate (%)."""
    m2_yoy = yoy_growth(m2_series, 12)
    return latest(m2_yoy)


# ══════════════════════════════════════════════════════════════════════════════
# SCORING ENGINE  —  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

INFLATION_TARGET = 2.0

RATE_PATH_RULES = {
    (True,  True):  ("Hike expected",  "Bullish"),
    (True,  False): ("Hold expected",  "Neutral"),
    (False, True):  ("Hold expected",  "Neutral"),
    (False, False): ("Cut expected",   "Bearish"),
}

BIAS_TO_SCORE = {"Bullish": 1, "Neutral": 0, "Bearish": -1}

# Weights for composite fundamental score
SCORE_WEIGHTS = {
    "gdp":        0.25,
    "inflation":  0.25,
    "trade":      0.15,
    "pmi":        0.15,
    "m2":         0.10,
    "conf":       0.10,
}


# ══════════════════════════════════════════════════════════════════════════════
# CORE ANALYSIS FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def analyze_currency(code: str, macro: Dict[str, pd.Series],
                      fx_series: pd.Series, dq_log: DataQualityLog) -> Dict:
    """Full macro + fundamental picture for one currency, now including
    PMI, M2 growth, consumer confidence, and composite scoring."""

    policy_rate    = macro.get("policy_rate", pd.Series(dtype=float))
    cpi_yoy        = macro.get("cpi_yoy",     pd.Series(dtype=float))
    gdp            = macro.get("gdp",          pd.Series(dtype=float))
    trade_bal      = macro.get("trade_balance",pd.Series(dtype=float))
    unemployment   = macro.get("unemployment", pd.Series(dtype=float))
    ten_year       = macro.get("ten_year",     pd.Series(dtype=float))
    pmi_mfg        = macro.get("pmi_mfg",      pd.Series(dtype=float))
    m2             = macro.get("m2",            pd.Series(dtype=float))
    consumer_conf  = macro.get("consumer_conf", pd.Series(dtype=float))
    building_perms = macro.get("building_permits", pd.Series(dtype=float))

    # ── Current levels ───────────────────────────────────────────────────
    rate_now    = latest(policy_rate)
    cpi_now     = latest(cpi_yoy)
    gdp_raw     = latest(gdp)
    trade_now   = latest(trade_bal)
    unemp_now   = latest(unemployment)
    ten_y_now   = latest(ten_year)
    pmi_now     = latest(pmi_mfg)
    conf_now    = latest(consumer_conf)
    permits_now = latest(building_perms)

    for val, label in [(rate_now, "Policy rate"), (cpi_now, "CPI YoY"),
                       (gdp_raw,  "GDP")]:
        if pd.isna(val):
            dq_log.warn(code, f"{label} unavailable from FRED")

    # ── GDP trend ────────────────────────────────────────────────────────
    gdp_is_growth = gdp.dropna().abs().median() < 15 if not gdp.dropna().empty else False
    if gdp_is_growth:
        gdp_growth_now   = gdp_raw
        gdp_growth_prior = value_n_periods_ago(gdp, 1)
    else:
        gdp_yoy          = yoy_growth(gdp, 4)
        gdp_growth_now   = latest(gdp_yoy)
        gdp_growth_prior = value_n_periods_ago(gdp_yoy, 1)

    if pd.isna(gdp_growth_now):
        gdp_trend = "unknown"
    elif gdp_growth_now > 0 and (pd.isna(gdp_growth_prior) or gdp_growth_now >= gdp_growth_prior):
        gdp_trend = "Expansion ▲"
    elif gdp_growth_now > 0:
        gdp_trend = "Expansion ▼"
    elif gdp_growth_now <= 0 and (pd.isna(gdp_growth_prior) or gdp_growth_now <= gdp_growth_prior):
        gdp_trend = "Contraction ▼"
    else:
        gdp_trend = "Contraction ▲"

    gdp_score = np.nan
    if not pd.isna(gdp_growth_now):
        gdp_score = float(np.clip(gdp_growth_now / 4.0, -1, 1))

    # ── Inflation & rate path  (Sections 5 & 4) ─────────────────────────
    cpi_direction = trend_direction(cpi_yoy, lookback=3)
    above_target  = (cpi_now > INFLATION_TARGET) if not pd.isna(cpi_now) else None
    rising_infl   = (cpi_direction == "rising")   if cpi_direction != "unknown" else None

    if above_target is not None and rising_infl is not None:
        expected_action, inflation_bias = RATE_PATH_RULES[(above_target, rising_infl)]
    else:
        expected_action, inflation_bias = "Unknown", "Neutral"
        dq_log.warn(code, "Insufficient inflation history for rate-path rule")

    inflation_score = float(BIAS_TO_SCORE.get(inflation_bias, 0))

    rate_direction   = trend_direction(policy_rate, lookback=3)
    rate_change_12m  = (rate_now - value_n_periods_ago(policy_rate, 12)
                        if not pd.isna(rate_now) else np.nan)
    real_rate        = (rate_now - cpi_now
                        if not pd.isna(rate_now) and not pd.isna(cpi_now) else np.nan)

    # ── Trade balance  (Section 7) ───────────────────────────────────────
    trade_dir = trend_direction(trade_bal, lookback=3)
    trade_score = np.nan
    if not pd.isna(trade_now):
        trade_score = 0.5 if trade_now > 0 else -0.5
        if trade_dir == "rising":   trade_score += 0.3
        elif trade_dir == "falling": trade_score -= 0.3
        trade_score = float(np.clip(trade_score, -1, 1))

    # ── PMI  (Section 16) ────────────────────────────────────────────────
    # PMI convention: >50 = expansion, <50 = contraction
    pmi_dir   = trend_direction(pmi_mfg, lookback=3)
    pmi_score = np.nan
    if not pd.isna(pmi_now):
        # Normalise around 50; clip between -1 and +1
        pmi_score = float(np.clip((pmi_now - 50) / 10.0, -1, 1))
        if pmi_dir == "rising":   pmi_score += 0.1
        elif pmi_dir == "falling": pmi_score -= 0.1
        pmi_score = float(np.clip(pmi_score, -1, 1))

    # ── M2 Money Supply  (Section 16) ───────────────────────────────────
    m2_growth_now  = m2_yoy_growth(m2)
    m2_trend       = trend_direction(yoy_growth(m2, 12))
    # Rising M2 signals potential inflationary pressure → bearish over time
    # But collapsing M2 signals recession risk → also bearish
    # Moderate positive M2 growth (3-8%) is neutral-positive for growth
    m2_score = np.nan
    if not pd.isna(m2_growth_now):
        if 2 < m2_growth_now < 8:
            m2_score = 0.3   # healthy range
        elif m2_growth_now >= 8:
            m2_score = -0.2  # inflationary excess
        elif m2_growth_now < 0:
            m2_score = -0.5  # contracting money supply
        else:
            m2_score = 0.0
        m2_score = float(np.clip(m2_score, -1, 1))

    # ── Consumer Confidence  (Section 16) ───────────────────────────────
    conf_dir   = trend_direction(consumer_conf, lookback=3)
    conf_score = np.nan
    if not pd.isna(conf_now):
        conf_score = 0.3 if conf_dir == "rising" else (-0.3 if conf_dir == "falling" else 0.0)

    # ── Yield curve / carry  (Section 9) ────────────────────────────────
    curve_slope = (ten_y_now - rate_now
                   if not pd.isna(ten_y_now) and not pd.isna(rate_now) else np.nan)
    curve_state = "unknown"
    if not pd.isna(curve_slope):
        curve_state = "Inverted ⚠️" if curve_slope < 0 else "Normal ✓"

    # ── Composite fundamental score ──────────────────────────────────────
    component_scores = {
        "gdp":       gdp_score,
        "inflation": inflation_score,
        "trade":     trade_score,
        "pmi":       pmi_score,
        "m2":        m2_score,
        "conf":      conf_score,
    }
    weighted_sum  = sum(SCORE_WEIGHTS[k] * v
                        for k, v in component_scores.items()
                        if not pd.isna(v))
    weight_used   = sum(SCORE_WEIGHTS[k] for k, v in component_scores.items()
                        if not pd.isna(v))
    composite     = float(weighted_sum / weight_used) if weight_used > 0 else np.nan

    # Map composite to bias label
    if pd.isna(composite):
        overall_bias = "Neutral"
    elif composite >= 0.25:
        overall_bias = "Bullish"
    elif composite <= -0.25:
        overall_bias = "Bearish"
    else:
        overall_bias = "Neutral"

    # ── Expected future values (12-month forward extrapolation) ─────────
    cpi_slope = np.nan
    if len(cpi_yoy.dropna()) >= 6:
        rc = cpi_yoy.dropna().iloc[-6:]
        cpi_slope = (rc.iloc[-1] - rc.iloc[0]) / max(len(rc) - 1, 1)
    cpi_exp12m = (cpi_now + cpi_slope * 12
                  if not pd.isna(cpi_slope) and not pd.isna(cpi_now) else np.nan)

    rate_step_map = {"Hike expected": +0.25, "Hold expected": 0.0,
                      "Cut expected": -0.25, "Unknown": 0.0}
    rate_exp12m = (rate_now + rate_step_map.get(expected_action, 0.0) * 4
                   if not pd.isna(rate_now) else np.nan)

    gdp_exp12m = np.nan
    if not pd.isna(gdp_growth_now) and not pd.isna(gdp_growth_prior):
        gdp_exp12m = gdp_growth_now + (gdp_growth_now - gdp_growth_prior)

    # ── Pre-Meeting Trade readiness  (Section 10) ───────────────────────
    info          = CURRENCY_INFO.get(code, {})
    meeting_months = info.get("cb_meeting_months", [])
    today_month    = datetime.today().month
    # Check if we are within 4 weeks of a CB meeting month
    next_meeting_months = [m for m in meeting_months
                           if m == today_month or m == (today_month % 12) + 1]
    pre_meeting_window = len(next_meeting_months) > 0
    # Qualify pre-meeting setup: need a clear bias + in-window
    pre_meeting_score = 0
    if pre_meeting_window and overall_bias != "Neutral":
        pre_meeting_score = 2  # strong setup
    elif pre_meeting_window:
        pre_meeting_score = 1  # in window but bias unclear

    # ── FX spot & momentum  (Section 7) ─────────────────────────────────
    fx_now = latest(fx_series)
    fx_chg_1m = fx_chg_3m = fx_chg_12m = np.nan
    if not fx_series.empty:
        c = fx_series.dropna()
        if len(c) > 21:  fx_chg_1m  = (c.iloc[-1] / c.iloc[-21]  - 1) * 100
        if len(c) > 63:  fx_chg_3m  = (c.iloc[-1] / c.iloc[-63]  - 1) * 100
        if len(c) > 252: fx_chg_12m = (c.iloc[-1] / c.iloc[-252] - 1) * 100
        # Sign-correct so positive = target currency strengthening
        if code in USD_IS_BASE:
            fx_chg_1m  = -fx_chg_1m  if not pd.isna(fx_chg_1m)  else np.nan
            fx_chg_3m  = -fx_chg_3m  if not pd.isna(fx_chg_3m)  else np.nan
            fx_chg_12m = -fx_chg_12m if not pd.isna(fx_chg_12m) else np.nan

    return {
        # Identity
        "code": code,
        "name": info.get("name", code),
        "country": info.get("country", "—"),
        "type": info.get("type", "—"),
        "cb": info.get("cb", "—"),
        # Current macro
        "Policy Rate %":      rate_now,
        "CPI YoY %":          cpi_now,
        "GDP Growth %":       gdp_growth_now,
        "Trade Balance":      trade_now,
        "Unemployment %":     unemp_now,
        "10Y Yield %":        ten_y_now,
        "Real Rate %":        real_rate,
        "Curve Slope":        curve_slope,
        "Curve State":        curve_state,
        "PMI":                pmi_now,
        "M2 Growth YoY %":    m2_growth_now,
        "Consumer Conf":      conf_now,
        "Building Permits":   permits_now,
        # Trend / direction
        "GDP Trend":          gdp_trend,
        "CPI Direction":      cpi_direction,
        "Rate Direction":     rate_direction,
        "Rate Δ 12m (bps)":   rate_change_12m * 100 if not pd.isna(rate_change_12m) else np.nan,
        "Trade Bal Direction": trade_dir,
        "PMI Direction":      pmi_dir,
        "M2 Trend":           m2_trend,
        "Conf Direction":     conf_dir,
        # Step 2 / rate path
        "Expected CB Action": expected_action,
        "Inflation Bias":     inflation_bias,
        "Above Target":       above_target,
        # Expected future (12m)
        "CPI Expected (12m)":  cpi_exp12m,
        "Rate Expected (12m)": rate_exp12m,
        "GDP Expected (12m)":  gdp_exp12m,
        # Component scores (raw, -1 to +1)
        "GDP Score":          gdp_score,
        "Inflation Score":    inflation_score,
        "Trade Score":        trade_score,
        "PMI Score":          pmi_score,
        "M2 Score":           m2_score,
        "Conf Score":         conf_score,
        # Composite
        "Composite Score":    composite,
        "Overall Bias":       overall_bias,
        # COT placeholder (filled separately)
        "COT Net":            np.nan,
        "COT Change":         np.nan,
        "COT Bias":           "—",
        # Pre-Meeting Trade
        "Pre-Meeting Window": pre_meeting_window,
        "Pre-Meeting Score":  pre_meeting_score,
        # FX spot
        "FX Spot":            fx_now,
        "FX Chg 1m %":        fx_chg_1m,
        "FX Chg 3m %":        fx_chg_3m,
        "FX Chg 12m %":       fx_chg_12m,
    }


# ══════════════════════════════════════════════════════════════════════════════
# IXL THEME SCORER  (Section 13)
# ══════════════════════════════════════════════════════════════════════════════

def ixl_score(impact: int, likelihood: int) -> int:
    """Impact × Likelihood composite. ≥70 = Tradeable theme."""
    return impact * likelihood


def ixl_label(score: int) -> str:
    if score >= 70: return "✅ Tradeable"
    if score >= 40: return "⚠️ Monitor"
    return "❌ Not Tradeable"


# ══════════════════════════════════════════════════════════════════════════════
# PAIR RANKER  (Section 4 Step 3)
# ══════════════════════════════════════════════════════════════════════════════

def rank_currencies(results: List[Dict]) -> pd.DataFrame:
    """Sort currencies by composite score, add star ratings."""
    rows = []
    for r in results:
        score = r.get("Composite Score", np.nan)
        if pd.isna(score):
            stars = "—"
        elif score >= 0.5:  stars = "★★★★★"
        elif score >= 0.3:  stars = "★★★★☆"
        elif score >= 0.1:  stars = "★★★☆☆"
        elif score >= -0.1: stars = "★★☆☆☆"
        elif score >= -0.3: stars = "★☆☆☆☆"
        else:               stars = "☆☆☆☆☆"
        rows.append({
            "Currency":        r["code"],
            "Name":            r["name"],
            "Bias":            r["Overall Bias"],
            "Score":           round(score, 3) if not pd.isna(score) else np.nan,
            "Stars":           stars,
            "Rate %":          r["Policy Rate %"],
            "Real Rate %":     r["Real Rate %"],
            "CPI %":           r["CPI YoY %"],
            "GDP %":           r["GDP Growth %"],
            "PMI":             r["PMI"],
            "M2 YoY %":        r["M2 Growth YoY %"],
            "CB Action":       r["Expected CB Action"],
            "COT Bias":        r["COT Bias"],
            "Pre-Meeting":     "🎯" if r.get("Pre-Meeting Score", 0) >= 2 else ("📅" if r.get("Pre-Meeting Window") else ""),
        })
    df = pd.DataFrame(rows)
    if "Score" in df.columns:
        df = df.sort_values("Score", ascending=False).reset_index(drop=True)
    return df


def suggest_pairs(ranked_df: pd.DataFrame, top_n: int = 3) -> List[Dict]:
    """Suggest long/short pairs: long top-scored vs short bottom-scored."""
    if len(ranked_df) < 2:
        return []
    bullish = ranked_df[ranked_df["Bias"] == "Bullish"].head(top_n)
    bearish = ranked_df[ranked_df["Bias"] == "Bearish"].tail(top_n).iloc[::-1]
    pairs   = []
    for _, b in bullish.iterrows():
        for _, s in bearish.iterrows():
            long_code  = b["Currency"]
            short_code = s["Currency"]
            # Build ticker
            if long_code == "USD":
                ticker = f"USD{short_code}=X"
                desc   = f"Long USD / Short {short_code}"
            elif short_code == "USD":
                ticker = f"{long_code}USD=X"
                desc   = f"Long {long_code} / Short USD"
            else:
                ticker = f"{long_code}{short_code}=X"
                desc   = f"Long {long_code} / Short {short_code}"
            pairs.append({
                "Pair":        f"{long_code}/{short_code}",
                "Description": desc,
                "Long Score":  b["Score"],
                "Short Score": s["Score"],
                "Long Bias":   b["Bias"],
                "Short Bias":  s["Bias"],
                "Long CB":     b["CB Action"],
                "Short CB":    s["CB Action"],
            })
    return pairs[:top_n]


def real_rate_differential_table(results: List[Dict]) -> pd.DataFrame:
    """NxN table of real-rate differentials (long row − short col)."""
    codes  = [r["code"] for r in results]
    rrates = {r["code"]: r["Real Rate %"] for r in results}
    rows   = []
    for long in codes:
        row = {"Long ↓ / Short →": long}
        for short in codes:
            if long == short:
                row[short] = "—"
            else:
                diff = (rrates[long] - rrates[short]
                        if not pd.isna(rrates[long]) and not pd.isna(rrates[short])
                        else np.nan)
                row[short] = f"{diff:+.2f}%" if not pd.isna(diff) else "—"
        rows.append(row)
    return pd.DataFrame(rows).set_index("Long ↓ / Short →")


# ══════════════════════════════════════════════════════════════════════════════
# DXY CORRELATION PANEL  (Section 15)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_dxy_correlations() -> Dict[str, pd.Series]:
    """Fetch DXY, gold, crude oil, S&P 500 for correlation panel."""
    tickers = {
        "DXY":   DXY_TICKER,
        "Gold":  "GC=F",
        "Crude": "CL=F",
        "SPX":   "^GSPC",
    }
    out = {}
    for label, t in tickers.items():
        try:
            data = yf.download(t, period="1y", interval="1d",
                                auto_adjust=True, progress=False)
            if "Close" in data.columns:
                s = data["Close"]
                if isinstance(s, pd.DataFrame): s = s.iloc[:, 0]
                out[label] = s.dropna()
            else:
                out[label] = pd.Series(dtype=float)
        except Exception:
            out[label] = pd.Series(dtype=float)
    return out


def compute_rolling_correlation(s1: pd.Series, s2: pd.Series,
                                  window: int = 20) -> pd.Series:
    """Rolling Pearson correlation between two price series (aligned on dates)."""
    df = pd.concat([s1.rename("a"), s2.rename("b")], axis=1).dropna()
    return df["a"].rolling(window).corr(df["b"])


# ══════════════════════════════════════════════════════════════════════════════
# STREAMLIT UI
# ══════════════════════════════════════════════════════════════════════════════

def fmt(val, decimals=2, suffix="") -> str:
    """Safe formatter for possibly-NaN floats."""
    if pd.isna(val): return "—"
    return f"{val:.{decimals}f}{suffix}"


def bias_badge(bias: str) -> str:
    colors = {"Bullish": "🟢", "Bearish": "🔴", "Neutral": "🟡"}
    return f"{colors.get(bias, '⚪')} {bias}"


def score_bar(score: float) -> str:
    """ASCII bar visualisation of a -1 to +1 score."""
    if pd.isna(score): return "—"
    filled = round((score + 1) / 2 * 10)
    return "█" * filled + "░" * (10 - filled) + f"  {score:+.2f}"


# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="Forex Fundamental Scanner",
        page_icon="💱",
        layout="wide",
    )

    st.title("💱 Forex Fundamental Scanner")
    st.caption("Chapter 7 · Three-Step Framework · COT · IXL · Pre-Meeting Trade")

    # ── Sidebar controls ──────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Settings")
        all_codes      = list(CURRENCY_INFO.keys())
        default_codes  = ["USD", "EUR", "GBP", "JPY", "AUD", "CAD"]
        selected_codes = st.multiselect("Currencies", all_codes, default=default_codes)
        st.divider()
        st.subheader("🌐 DXY Dashboard")
        show_dxy = st.checkbox("Show DXY correlations", value=True)
        st.divider()
        st.subheader("📜 IXL Theme Builder")
        ixl_theme   = st.text_input("Theme description", placeholder="e.g. US tariffs on EU goods")
        ixl_impact  = st.slider("Impact (1–10)",     1, 10, 7)
        ixl_like    = st.slider("Likelihood (1–10)", 1, 10, 8)
        ixl_result  = ixl_score(ixl_impact, ixl_like)
        st.metric("IXL Score", ixl_result, ixl_label(ixl_result))

    if not selected_codes:
        st.warning("Select at least one currency to begin.")
        return

    # ── Fetch all data ────────────────────────────────────────────────────────
    dq_log = DataQualityLog()
    with st.spinner("Fetching macro data from FRED & yfinance…"):
        results = []
        cot_df  = fetch_cot_data()
        for code in selected_codes:
            macro     = fetch_currency_macro(code, dq_log)
            fx_ticker = FX_TICKER.get(code, "")
            fx_hist   = fetch_fx_history(fx_ticker) if fx_ticker else pd.Series(dtype=float)
            r         = analyze_currency(code, macro, fx_hist, dq_log)
            # Overlay COT data
            cot = get_cot_position(code, cot_df)
            r["COT Net"]    = cot["net"]
            r["COT Change"] = cot["change"]
            r["COT Bias"]   = cot["bias"]
            results.append(r)

    ranked_df   = rank_currencies(results)
    pair_ideas  = suggest_pairs(ranked_df)
    rr_diff_df  = real_rate_differential_table(results)

    # ── Six-tab layout ────────────────────────────────────────────────────────
    tab_overview, tab_carry, tab_ixl, tab_cot, tab_premeeting, tab_pairs = st.tabs([
        "📊 Overview", "💰 Carry", "🎯 IXL Themes", "📋 COT", "🕐 Pre-Meeting", "🔀 Pairs"
    ])

    # ── TAB 1 · OVERVIEW ─────────────────────────────────────────────────────
    with tab_overview:
        st.subheader("Currency Ranking — Composite Fundamental Score")
        # Colour-coded bias column
        def colour_bias(val):
            colour = {"Bullish": "background-color:#1a4a1a;color:#7fff7f",
                       "Bearish": "background-color:#4a1a1a;color:#ff9999",
                       "Neutral": "background-color:#2a2a1a;color:#ffff99"}.get(val, "")
            return colour

        styled = ranked_df.style.map(colour_bias, subset=["Bias"])
        st.dataframe(styled, use_container_width=True, hide_index=True)

        # Score chart
        fig = go.Figure(go.Bar(
            x=ranked_df["Currency"],
            y=ranked_df["Score"],
            marker_color=["#2ecc71" if b == "Bullish" else
                           "#e74c3c" if b == "Bearish" else "#f1c40f"
                           for b in ranked_df["Bias"]],
            text=[f"{s:+.3f}" for s in ranked_df["Score"]],
            textposition="outside",
        ))
        fig.update_layout(title="Composite Fundamental Score (−1 to +1)",
                           yaxis_title="Score", xaxis_title="Currency",
                           template="plotly_dark", height=350)
        st.plotly_chart(fig, use_container_width=True)

        # Per-currency detail expanders
        st.subheader("Detailed Breakdowns")
        for r in results:
            with st.expander(f"**{r['code']}** — {r['name']}  {bias_badge(r['Overall Bias'])}"):
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown("**Current Macro**")
                    st.write(f"Policy Rate: {fmt(r['Policy Rate %'])}%")
                    st.write(f"CPI YoY:     {fmt(r['CPI YoY %'])}%")
                    st.write(f"Real Rate:   {fmt(r['Real Rate %'])}%")
                    st.write(f"GDP Growth:  {fmt(r['GDP Growth %'])}%")
                    st.write(f"Unemployment:{fmt(r['Unemployment %'])}%")
                    st.write(f"10Y Yield:   {fmt(r['10Y Yield %'])}%")
                with c2:
                    st.markdown("**Leading Indicators**")
                    st.write(f"PMI:          {fmt(r['PMI'], 1)} ({r['PMI Direction']})")
                    st.write(f"M2 Growth:    {fmt(r['M2 Growth YoY %'])}%")
                    st.write(f"Consumer Conf:{fmt(r['Consumer Conf'], 1)} ({r['Conf Direction']})")
                    st.write(f"Trade Bal:    {fmt(r['Trade Balance'], 0)} ({r['Trade Bal Direction']})")
                    st.write(f"Curve:        {r['Curve State']}")
                with c3:
                    st.markdown("**Forward Outlook**")
                    st.write(f"CB Action Expected: **{r['Expected CB Action']}**")
                    st.write(f"CPI (12m fwd): {fmt(r['CPI Expected (12m)'])}%")
                    st.write(f"Rate (12m fwd):{fmt(r['Rate Expected (12m)'])}%")
                    st.write(f"GDP (12m fwd): {fmt(r['GDP Expected (12m)'])}%")
                    st.write(f"GDP Trend:     {r['GDP Trend']}")
                st.write("**Component Score Breakdown**")
                for k in ["gdp", "inflation", "trade", "pmi", "m2", "conf"]:
                    key   = k.capitalize() + " Score"
                    val   = r.get(key, np.nan) or r.get(k.upper() + " Score", np.nan)
                    # Recover from dict
                    val = r.get(f"{k.title()} Score",
                          r.get(f"{k.upper()} Score",
                          r.get(f"GDP Score"     if k=="gdp"       else
                                f"Inflation Score" if k=="inflation" else
                                f"Trade Score"    if k=="trade"     else
                                f"PMI Score"      if k=="pmi"       else
                                f"M2 Score"       if k=="m2"        else
                                f"Conf Score", np.nan)))
                    st.text(f"  {k.upper():10s}  {score_bar(val)}")

        # Data quality warnings
        if dq_log.any():
            with st.expander("⚠️ Data Quality Warnings"):
                for code, msgs in dq_log.all().items():
                    st.write(f"**{code}**: " + " | ".join(msgs))

        # CSV download
        csv_buf = io.StringIO()
        ranked_df.to_csv(csv_buf, index=False)
        st.download_button("⬇️ Export ranking to CSV", csv_buf.getvalue(),
                            "forex_ranking.csv", "text/csv")

    # ── TAB 2 · CARRY ────────────────────────────────────────────────────────
    with tab_carry:
        st.subheader("Carry Trade Analysis — Real Rate Differentials")
        st.caption("Positive = long currency earns higher real yield vs short currency (Section 9)")
        st.dataframe(rr_diff_df, use_container_width=True)

        st.subheader("Classic Carry Pairs Assessment")
        carry_rows = []
        result_map = {r["code"]: r for r in results}
        for long_c, short_c in CLASSIC_CARRY_PAIRS:
            if long_c not in result_map or short_c not in result_map: continue
            rl = result_map[long_c]
            rs = result_map[short_c]
            rr_diff = (rl["Real Rate %"] - rs["Real Rate %"]
                       if not pd.isna(rl["Real Rate %"]) and not pd.isna(rs["Real Rate %"])
                       else np.nan)
            carry_rows.append({
                "Pair":          f"{long_c}/{short_c}",
                "Long":          long_c,
                "Short":         short_c,
                "Long Real Rate %":  fmt(rl["Real Rate %"]),
                "Short Real Rate %": fmt(rs["Real Rate %"]),
                "Real Yield Diff %": fmt(rr_diff, 2),
                "Long Bias":     rl["Overall Bias"],
                "Short Bias":    rs["Overall Bias"],
                "VIX Warning":   "⚠️ Check VIX" if rl["Curve State"] == "Inverted ⚠️" or
                                                     rs["Curve State"] == "Inverted ⚠️" else "✓",
            })
        st.dataframe(pd.DataFrame(carry_rows), use_container_width=True, hide_index=True)

        st.info(
            "**Early Warning Reminders (Section 9)**\n\n"
            "• VIX multi-year lows = complacency risk; rising VIX = reduce carry exposure\n"
            "• LIBOR-OIS spread widening = interbank stress precursor\n"
            "• Inverted yield curve = recession risk; reassess carry positions\n"
            "• Trade balance deterioration in carry target = early exit signal"
        )

    # ── TAB 3 · IXL THEMES ───────────────────────────────────────────────────
    with tab_ixl:
        st.subheader("IXL Theme Evaluator — Impact × Likelihood (Section 13)")
        st.caption("Score ≥ 70 = Tradeable theme. Express via directional position in affected pair.")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Current IXL Score", ixl_result, ixl_label(ixl_result))
            if ixl_theme:
                st.write(f"**Theme:** {ixl_theme}")
                if ixl_result >= 70:
                    st.success("This theme meets the tradeability threshold. "
                               "Identify the most-affected currency pair and express directionally.")
                elif ixl_result >= 40:
                    st.warning("Monitor this theme but do not yet take a position. "
                               "Re-evaluate as likelihood or impact evolves.")
                else:
                    st.error("Theme score too low to trade. File for ongoing monitoring.")

        with col2:
            # IXL heatmap
            impacts     = list(range(1, 11))
            likelihoods = list(range(1, 11))
            z = [[i * l for l in likelihoods] for i in impacts]
            fig_ixl = go.Figure(go.Heatmap(
                z=z, x=likelihoods, y=impacts,
                colorscale=[[0,"#1a1a1a"],[0.4,"#7f6000"],[0.7,"#006f00"],[1,"#00ff00"]],
                zmin=0, zmax=100,
                text=[[f"{i*l}" for l in likelihoods] for i in impacts],
                texttemplate="%{text}",
            ))
            fig_ixl.update_layout(
                title="IXL Score Matrix (≥70 = green = tradeable)",
                xaxis_title="Likelihood →", yaxis_title="Impact ↑",
                template="plotly_dark", height=350,
            )
            # Mark current score
            fig_ixl.add_annotation(x=ixl_like, y=ixl_impact,
                                    text="◉", showarrow=False,
                                    font=dict(color="white", size=18))
            st.plotly_chart(fig_ixl, use_container_width=True)

        st.subheader("Example Themes Library")
        examples = [
            {"Theme": "US tariffs on all EU goods",            "Impact": 8, "Likelihood": 7, "Affected Pair": "EUR/USD short", "Direction": "Short EUR"},
            {"Theme": "Fed pivot to rate cuts",                 "Impact": 9, "Likelihood": 6, "Affected Pair": "USD/JPY",       "Direction": "Short USD"},
            {"Theme": "BOJ policy normalisation",              "Impact": 8, "Likelihood": 7, "Affected Pair": "USD/JPY",       "Direction": "Long JPY"},
            {"Theme": "RBA holds amid falling inflation",      "Impact": 5, "Likelihood": 8, "Affected Pair": "AUD/JPY",       "Direction": "Reduce carry"},
            {"Theme": "UK election — major fiscal stimulus",   "Impact": 7, "Likelihood": 5, "Affected Pair": "GBP/USD",       "Direction": "Long GBP"},
            {"Theme": "China stimulus package",                "Impact": 8, "Likelihood": 6, "Affected Pair": "AUD/USD",       "Direction": "Long AUD"},
        ]
        ex_df = pd.DataFrame(examples)
        ex_df["IXL"] = ex_df["Impact"] * ex_df["Likelihood"]
        ex_df["Status"] = ex_df["IXL"].apply(ixl_label)
        st.dataframe(ex_df, use_container_width=True, hide_index=True)

    # ── TAB 4 · COT ──────────────────────────────────────────────────────────
    with tab_cot:
        st.subheader("COT — Non-Commercial Positioning (Section 14)")
        st.caption("Published every Friday at 3:30 PM ET by the CFTC. Positions as of prior Tuesday.")

        cot_rows = [
            {
                "Currency":   r["code"],
                "Name":       r["name"],
                "COT Bias":   r["COT Bias"],
                "Net Position": r["COT Net"] if not pd.isna(r["COT Net"]) else "—",
                "Week Change":  r["COT Change"] if not pd.isna(r["COT Change"]) else "—",
                "Fund. Bias": r["Overall Bias"],
                "Aligned?":   ("✅" if r["COT Bias"] == r["Overall Bias"]
                                else ("—" if r["COT Bias"] == "—" else "⚠️ Divergence")),
            }
            for r in results
        ]
        st.dataframe(pd.DataFrame(cot_rows), use_container_width=True, hide_index=True)

        st.info(
            "**How to use COT data (Section 14)**\n\n"
            "• Use as *confluence*, not standalone signal\n"
            "• Monitor net non-commercial position: ratio of longs to shorts, week-on-week change\n"
            "• Divergence (⚠️): COT sentiment contradicts fundamentals → wait for alignment or trade with care\n"
            "• Extreme positioning (very large net long or short) can signal crowded trade / reversal risk"
        )

        if cot_df.empty:
            st.warning("COT data could not be fetched from CFTC. Check connectivity. "
                        "COT data is available at cftc.gov.")

    # ── TAB 5 · PRE-MEETING TRADE ────────────────────────────────────────────
    with tab_premeeting:
        st.subheader("Pre-Meeting Trade Scanner (Section 10)")
        st.caption("Entry window: 1–4 weeks before a CB meeting when a clear bias exists.")

        pm_rows = []
        for r in results:
            info = CURRENCY_INFO.get(r["code"], {})
            months = info.get("cb_meeting_months", [])
            today_m = datetime.today().month
            next_m = [(today_m % 12) + 1]
            in_window = today_m in months or next_m[0] in months
            score = r.get("Pre-Meeting Score", 0)
            checklist_pass = (
                r["Overall Bias"] != "Neutral"
                and r["CPI Direction"] != "unknown"
                and not pd.isna(r["GDP Growth %"])
            )
            pm_rows.append({
                "Currency":       r["code"],
                "Central Bank":   info.get("cb", "—"),
                "In Window?":     "🎯 Yes" if in_window else "No",
                "Bias":           r["Overall Bias"],
                "CB Action":      r["Expected CB Action"],
                "Checklist ✓":    "✅ Setup Valid" if (in_window and checklist_pass)
                                   else ("⚠️ Partial" if in_window else "—"),
                "FX 1m Trend":    fmt(r["FX Chg 1m %"]) + "%",
                "COT Aligned":    ("✅" if r["COT Bias"] == r["Overall Bias"]
                                   else ("—" if r["COT Bias"] == "—" else "⚠️")),
            })

        st.dataframe(pd.DataFrame(pm_rows), use_container_width=True, hide_index=True)

        st.markdown("""
**Pre-Meeting Trade Checklist (Section 10):**

1. ✅ Clear driving factor compelling a policy change from the previous meeting
2. ✅ An opposing currency with weaker / diverging fundamentals
3. ✅ No contradicting fundamental themes for the target currency
4. ✅ Price action (FX 1m trend) supports the directional hypothesis
5. ✅ COT non-commercial positioning aligns with bias
""")

    # ── TAB 6 · PAIRS ────────────────────────────────────────────────────────
    with tab_pairs:
        st.subheader("Suggested Long/Short Pairs (Section 4 — Step 3)")
        if pair_ideas:
            for p in pair_ideas:
                with st.container(border=True):
                    c1, c2, c3 = st.columns([1, 1, 2])
                    c1.metric("Pair", p["Pair"])
                    c2.metric("Long Score", fmt(p["Long Score"], 3))
                    c3.write(f"**{p['Description']}**")
                    c3.write(f"Long CB: {p['Long CB']}  ·  Short CB: {p['Short CB']}")
        else:
            st.info("No clear bullish/bearish divergence found in the selected currencies.")

        st.divider()
        st.subheader("DXY Correlation Dashboard (Section 15)")
        if show_dxy:
            dxy_data = fetch_dxy_correlations()
            dxy = dxy_data.get("DXY", pd.Series(dtype=float))
            if not dxy.empty:
                corr_rows = []
                for label, s in dxy_data.items():
                    if label == "DXY" or s.empty: continue
                    corr = compute_rolling_correlation(dxy, s).iloc[-1] if len(s) > 20 else np.nan
                    corr_rows.append({
                        "Asset": label,
                        "20d Rolling Corr vs DXY": fmt(corr, 3),
                        "Interpretation": (
                            "Negative (typical: DXY ↑ = asset ↓)" if not pd.isna(corr) and corr < 0
                            else "Positive" if not pd.isna(corr) and corr > 0
                            else "—"
                        )
                    })
                st.dataframe(pd.DataFrame(corr_rows), use_container_width=True, hide_index=True)

                # DXY chart
                fig_dxy = go.Figure()
                fig_dxy.add_trace(go.Scatter(x=dxy.index, y=dxy.values, name="DXY",
                                               line=dict(color="#3498db", width=2)))
                for label, colour in [("Gold", "#f1c40f"), ("SPX", "#2ecc71"), ("Crude", "#e67e22")]:
                    s = dxy_data.get(label, pd.Series(dtype=float))
                    if not s.empty:
                        # Normalise to DXY starting level for visual comparison
                        s_norm = s / s.iloc[0] * dxy.iloc[0]
                        fig_dxy.add_trace(go.Scatter(x=s_norm.index, y=s_norm.values,
                                                       name=label, line=dict(color=colour, width=1.5)))
                fig_dxy.update_layout(title="DXY vs Gold / SPX / Crude (normalised, 1Y)",
                                       template="plotly_dark", height=350)
                st.plotly_chart(fig_dxy, use_container_width=True)
            else:
                st.warning("DXY data unavailable from yfinance.")

        st.divider()
        st.subheader("Buying the Rumour / Selling the Fact Monitor (Section 12)")
        st.markdown("""
| Signal | What to watch |
|--------|--------------|
| **Rumour phase** | FX starts moving 1–4 weeks before CB meeting; position early |
| **Fact phase** | Announced rate decision = price already reflects it → potential reversal |
| **Consensus check** | Visit ForexFactory → click indicator → review beat/miss history |
| **Divergence trade** | Fundamentals strong but short-term negative sentiment → discount buying opportunity |
""")


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()
