import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
import io
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
# DATA QUALITY TRACKING
# ══════════════════════════════════════════════════════════════════════════════

class DataQualityLog:
    def __init__(self):
        self._w: Dict[str, List[str]] = {}

    def warn(self, code: str, msg: str):
        self._w.setdefault(code, []).append(msg)

    def get(self, code: str) -> List[str]:
        return self._w.get(code, [])

    def all(self) -> Dict[str, List[str]]:
        return self._w

    def any(self) -> bool:
        return bool(self._w)

    def count(self) -> int:
        return sum(len(v) for v in self._w.values())


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS & UNIVERSE
# ══════════════════════════════════════════════════════════════════════════════

INFLATION_TARGET = 2.0
HIGH_INFLATION_CURRENCIES = {"TRY", "ZAR", "MXN", "CNY"}

# Liquidity tiers for pair filtering
# Tier 1 = most liquid / most tradeable
PAIR_LIQUIDITY_TIERS: Dict[Tuple[str, str], int] = {
    ("EUR", "USD"): 1, ("USD", "JPY"): 1, ("GBP", "USD"): 1,
    ("USD", "CHF"): 1, ("AUD", "USD"): 1, ("USD", "CAD"): 1, ("NZD", "USD"): 1,
    ("EUR", "JPY"): 2, ("GBP", "JPY"): 2, ("AUD", "JPY"): 2, ("NZD", "JPY"): 2,
    ("EUR", "GBP"): 2, ("EUR", "AUD"): 2, ("EUR", "CAD"): 2, ("EUR", "CHF"): 2,
    ("AUD", "CAD"): 2, ("AUD", "NZD"): 2, ("AUD", "CHF"): 2, ("NZD", "CHF"): 2,
    ("USD", "CNY"): 2, ("USD", "MXN"): 3, ("USD", "ZAR"): 3, ("USD", "TRY"): 3,
    ("EUR", "TRY"): 3, ("MXN", "JPY"): 3, ("ZAR", "JPY"): 3,
}

def pair_liquidity_tier(a: str, b: str) -> int:
    return PAIR_LIQUIDITY_TIERS.get((a, b), PAIR_LIQUIDITY_TIERS.get((b, a), 4))

CURRENCY_INFO: Dict[str, Dict] = {
    "USD": {"name": "US Dollar",          "country": "United States",  "type": "Safe Haven",            "cb": "Federal Reserve", "cb_meeting_months": [1,3,5,6,7,9,11,12]},
    "EUR": {"name": "Euro",               "country": "Euro Area",      "type": "Major",                 "cb": "ECB",             "cb_meeting_months": [1,2,3,4,6,7,9,10,12]},
    "GBP": {"name": "British Pound",      "country": "United Kingdom", "type": "Major",                 "cb": "Bank of England", "cb_meeting_months": [2,3,5,6,8,9,11,12]},
    "JPY": {"name": "Japanese Yen",        "country": "Japan",          "type": "Safe Haven",            "cb": "Bank of Japan",   "cb_meeting_months": [1,3,4,6,7,9,10,12]},
    "AUD": {"name": "Australian Dollar",  "country": "Australia",      "type": "Commodity-Linked",      "cb": "RBA",             "cb_meeting_months": [2,3,5,6,8,9,11,12]},
    "CAD": {"name": "Canadian Dollar",    "country": "Canada",         "type": "Commodity-Linked",      "cb": "Bank of Canada",  "cb_meeting_months": [1,3,4,6,7,9,10,12]},
    "CHF": {"name": "Swiss Franc",        "country": "Switzerland",    "type": "Safe Haven",            "cb": "SNB",             "cb_meeting_months": [3,6,9,12]},
    "NZD": {"name": "New Zealand Dollar", "country": "New Zealand",    "type": "Commodity-Linked",      "cb": "RBNZ",            "cb_meeting_months": [2,4,5,7,8,10,11]},
    "CNY": {"name": "Chinese Yuan",       "country": "China",          "type": "Policy-Driven",         "cb": "PBOC",            "cb_meeting_months": []},
    "TRY": {"name": "Turkish Lira",       "country": "Türkiye",        "type": "Policy-Driven",         "cb": "CBRT",            "cb_meeting_months": [1,2,3,4,5,6,7,8,9,10,11,12]},
    "MXN": {"name": "Mexican Peso",       "country": "Mexico",         "type": "Policy-Driven / Carry", "cb": "Banxico",         "cb_meeting_months": [2,3,5,6,8,9,11,12]},
    "ZAR": {"name": "South African Rand", "country": "South Africa",   "type": "Policy-Driven / Carry", "cb": "SARB",            "cb_meeting_months": [1,3,5,7,9,11]},
}

FRED_SERIES: Dict[str, Dict[str, str]] = {
    "USD": {
        "policy_rate": "DFEDTARU", "cpi_yoy": "CPALTT01USM659N",
        "gdp": "A191RL1Q225SBEA", "trade_balance": "BOPGSTB",
        "unemployment": "UNRATE", "ten_year": "DGS10",
        "pmi_mfg": "", "m2": "M2SL",
        "consumer_conf": "UMCSENT", "building_permits": "PERMIT",
    },
    "EUR": {
        "policy_rate": "ECBDFR", "cpi_yoy": "CP0000EZ19M086NEST",
        "gdp": "CLVMNACSCAB1GQEA19", "trade_balance": "XTNTVA01EZQ667S",
        "unemployment": "LRHUTTTTEZM156S", "ten_year": "IRLTLT01EZM156N",
        "pmi_mfg": "", "m2": "MABMM301EZM189S",
        "consumer_conf": "CSCICP02EZM460S", "building_permits": "",
    },
    "GBP": {
        "policy_rate": "BOERUKM", "cpi_yoy": "GBRCPIALLMINMEI",
        "gdp": "NGDPRSAXDCGBQ", "trade_balance": "XTNTVA01GBQ667S",
        "unemployment": "LRHUTTTTGBM156S", "ten_year": "IRLTLT01GBM156N",
        "pmi_mfg": "", "m2": "MABMM301GBM189S",
        "consumer_conf": "CSCICP02GBM460S", "building_permits": "",
    },
    "JPY": {
        "policy_rate": "IRSTCB01JPM156N", "cpi_yoy": "JPNCPIALLMINMEI",
        "gdp": "JPNRGDPEXP", "trade_balance": "XTNTVA01JPQ667S",
        "unemployment": "LRHUTTTTJPM156S", "ten_year": "IRLTLT01JPM156N",
        "pmi_mfg": "", "m2": "MABMM301JPM189S",
        "consumer_conf": "CSCICP02JPM460S", "building_permits": "",
    },
    "AUD": {
        "policy_rate": "IRSTCB01AUM156N", "cpi_yoy": "AUSCPIALLQINMEI",
        "gdp": "AUSGDPRQDSMEI", "trade_balance": "XTNTVA01AUQ667S",
        "unemployment": "LRHUTTTTAUM156S", "ten_year": "IRLTLT01AUM156N",
        "pmi_mfg": "", "m2": "MABMM301AUM189S",
        "consumer_conf": "CSCICP02AUM460S", "building_permits": "",
    },
    "CAD": {
        "policy_rate": "IRSTCB01CAM156N", "cpi_yoy": "CANCPIALLMINMEI",
        "gdp": "NGDPRSAXDCCAQ", "trade_balance": "XTNTVA01CAQ667S",
        "unemployment": "LRHUTTTTCAM156S", "ten_year": "IRLTLT01CAM156N",
        "pmi_mfg": "", "m2": "MABMM301CAM189S",
        "consumer_conf": "CSCICP02CAM460S", "building_permits": "",
    },
    "CHF": {
        "policy_rate": "IRSTCB01CHM156N", "cpi_yoy": "CHECPIALLMINMEI",
        "gdp": "NGDPRSAXDCCHQ", "trade_balance": "XTNTVA01CHQ667S",
        "unemployment": "LRHUTTTTCHM156S", "ten_year": "IRLTLT01CHM156N",
        "pmi_mfg": "", "m2": "", "consumer_conf": "", "building_permits": "",
    },
    "NZD": {
        "policy_rate": "IRSTCB01NZM156N", "cpi_yoy": "NZLCPIALLQINMEI",
        "gdp": "NGDPRSAXDCNZQ", "trade_balance": "XTNTVA01NZQ667S",
        "unemployment": "LRHUTTTTNZM156S", "ten_year": "IRLTLT01NZM156N",
        "pmi_mfg": "", "m2": "", "consumer_conf": "", "building_permits": "",
    },
    "CNY": {
        "policy_rate": "INTDSRCNM193N", "cpi_yoy": "CHNCPIALLMINMEI",
        "gdp": "NGDPRSAXDCCNQ", "trade_balance": "XTNTVA01CNQ667S",
        "unemployment": "LRUN64TTCNQ156S", "ten_year": "IRLTLT01CNM156N",
        "pmi_mfg": "", "m2": "", "consumer_conf": "", "building_permits": "",
    },
    "TRY": {
        "policy_rate": "INTDSRTRM193N", "cpi_yoy": "TURCPIALLMINMEI",
        "gdp": "NGDPRSAXDCTRQ", "trade_balance": "XTNTVA01TRQ667S",
        "unemployment": "LRHUTTTTTRM156S", "ten_year": "IRLTLT01TRM156N",
        "pmi_mfg": "", "m2": "", "consumer_conf": "", "building_permits": "",
    },
    "MXN": {
        "policy_rate": "INTDSRMXM193N", "cpi_yoy": "MEXCPIALLMINMEI",
        "gdp": "NGDPRSAXDCMXQ", "trade_balance": "XTNTVA01MXQ667S",
        "unemployment": "LRHUTTTTMXM156S", "ten_year": "IRLTLT01MXM156N",
        "pmi_mfg": "", "m2": "", "consumer_conf": "", "building_permits": "",
    },
    "ZAR": {
        "policy_rate": "INTDSRZAM193N", "cpi_yoy": "ZAFCPIALLMINMEI",
        "gdp": "NGDPRSAXDCZAQ", "trade_balance": "XTNTVA01ZAQ667S",
        "unemployment": "LRHUTTTTZAM156S", "ten_year": "IRLTLT01ZAM156N",
        "pmi_mfg": "", "m2": "", "consumer_conf": "", "building_permits": "",
    },
}

FX_TICKER: Dict[str, str] = {
    "EUR": "EURUSD=X", "GBP": "GBPUSD=X", "AUD": "AUDUSD=X", "NZD": "NZDUSD=X",
    "JPY": "USDJPY=X", "CAD": "USDCAD=X", "CHF": "USDCHF=X",
    "CNY": "USDCNY=X", "TRY": "USDTRY=X", "MXN": "USDMXN=X", "ZAR": "USDZAR=X",
}
USD_IS_BASE = {"JPY", "CAD", "CHF", "CNY", "TRY", "MXN", "ZAR"}
DXY_TICKER  = "DX-Y.NYB"

CLASSIC_CARRY_PAIRS = [
    ("AUD","JPY"),("NZD","JPY"),("AUD","CHF"),("NZD","CHF"),
    ("MXN","JPY"),("ZAR","JPY"),("AUD","EUR"),("NZD","EUR"),
]

SCORE_WEIGHTS = {
    "gdp": 0.25, "inflation": 0.20, "cb_taylor": 0.20,
    "trade": 0.15, "m2": 0.10, "conf": 0.10,
}


# ══════════════════════════════════════════════════════════════════════════════
# FRED FETCH
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fred_series(series_id: str) -> pd.Series:
    if not series_id:
        return pd.Series(dtype=float)
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    try:
        df = pd.read_csv(url)
        if df.empty or df.shape[1] < 2:
            return pd.Series(dtype=float)
        dc, vc = df.columns[0], df.columns[1]
        df[dc] = pd.to_datetime(df[dc], errors="coerce")
        df[vc] = pd.to_numeric(df[vc], errors="coerce")
        return df.dropna(subset=[dc]).set_index(dc)[vc].dropna()
    except Exception:
        return pd.Series(dtype=float)


def normalise_rate_series(s: pd.Series, label: str = "", code: str = "",
                           dq_log: Optional["DataQualityLog"] = None) -> pd.Series:
    """
    No-op pass-through for rate/CPI series.

    Every FRED series referenced in FRED_SERIES for policy_rate, ten_year,
    and cpi_yoy is documented by FRED/source as already being in "Percent"
    units (e.g. DFEDTARU, ECBDFR, IRLTLT01..., CPALTT01..., etc.) -- none
    of them ship as a 0-1 decimal fraction. A prior version of this
    function tried to auto-detect decimal-encoded values from a magnitude
    heuristic (median < 0.20 -> multiply by 100), but that heuristic has
    no way to tell "0.10 meaning 0.10%" (a real, low-but-valid policy
    rate for JPY/CHF, or CPI near zero) apart from "0.0010 meaning
    0.10% encoded as a fraction" -- both have the same magnitude. That
    ambiguity caused real bugs: genuinely low rates got wrongly
    multiplied by 100, and the resulting mixed units flowed into
    real-rate subtraction (10Y yield − CPI) producing nonsensical results
    like several-hundred-percent "real rates".

    Trusting FRED's documented units (which this app's own series
    selection controls) is more reliable than guessing from data shape.
    If a specific series is ever found to ship as a decimal fraction,
    add it explicitly to DECIMAL_ENCODED_SERIES below rather than
    re-introducing a magnitude guess.
    """
    return s


# Series IDs (if any) that are known from FRED's own documentation to be
# decimal-encoded rather than already in percent. Add explicit IDs here
# only after confirming via the FRED series page ("Units:" field) --
# do not re-introduce magnitude-based guessing.
DECIMAL_ENCODED_SERIES: set = set()


def fetch_currency_macro(code: str, dq_log: DataQualityLog) -> Dict[str, pd.Series]:
    out: Dict[str, pd.Series] = {}
    rate_fields = {"policy_rate", "ten_year", "cpi_yoy"}
    for label, sid in FRED_SERIES.get(code, {}).items():
        s = fetch_fred_series(sid)
        if s.empty and sid:
            dq_log.warn(code, f"FRED '{sid}' ({label}) unavailable")
        if label in rate_fields and sid in DECIMAL_ENCODED_SERIES:
            s = s * 100
            dq_log.warn(code, f"{label} ('{sid}'): converted from decimal to percent per known series unit.")
        out[label] = s
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fx_history(ticker: str, period: str = "2y") -> pd.Series:
    if not ticker:
        return pd.Series(dtype=float)
    try:
        data = yf.download(ticker, period=period, interval="1d",
                            auto_adjust=True, progress=False)
        if data.empty:
            return pd.Series(dtype=float)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
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
    year = datetime.now().year
    for y in [year, year - 1]:
        for url in [
            f"https://www.cftc.gov/files/dea/history/fut_fin_xls_{y}.zip",
            f"https://www.cftc.gov/files/dea/history/dea_fut_xls_{y}.zip",
        ]:
            try:
                df = pd.read_csv(url, compression="zip", low_memory=False)
                if df.empty:
                    continue
                df.columns = [c.strip() for c in df.columns]
                required = {"Market_and_Exchange_Names", "As_of_Date_in_Form_YYMMDD",
                             "NonComm_Positions_Long_All", "NonComm_Positions_Short_All"}
                if not required.issubset(df.columns):
                    continue
                df["date"] = pd.to_datetime(
                    df["As_of_Date_in_Form_YYMMDD"].astype(str),
                    format="%y%m%d", errors="coerce")
                df = df.dropna(subset=["date"])
                df["net_noncomm"] = (
                    pd.to_numeric(df["NonComm_Positions_Long_All"],  errors="coerce") -
                    pd.to_numeric(df["NonComm_Positions_Short_All"], errors="coerce"))
                df["market"] = df["Market_and_Exchange_Names"].str.upper().str.strip()
                return df[["date", "market", "net_noncomm"]].dropna()
            except Exception:
                continue
    return pd.DataFrame()


COT_MARKET_MAP = {
    "EURO FX": "EUR", "BRITISH POUND": "GBP", "JAPANESE YEN": "JPY",
    "AUSTRALIAN DOLLAR": "AUD", "CANADIAN DOLLAR": "CAD",
    "SWISS FRANC": "CHF", "NEW ZEALAND DOLLAR": "NZD", "MEXICAN PESO": "MXN",
}


def get_cot_position(code: str, cot_df: pd.DataFrame) -> Dict:
    empty = {"net": np.nan, "prev": np.nan, "change": np.nan, "bias": "—"}
    if cot_df.empty:
        return empty
    fragment = next((k for k, v in COT_MARKET_MAP.items() if v == code), None)
    if not fragment:
        return empty
    sub = cot_df[cot_df["market"].str.contains(fragment, na=False)].sort_values("date")
    if len(sub) < 2:
        return empty
    net  = float(sub.iloc[-1]["net_noncomm"])
    prev = float(sub.iloc[-2]["net_noncomm"])
    return {"net": net, "prev": prev, "change": net - prev,
            "bias": "Bullish" if net > 0 else ("Bearish" if net < 0 else "Neutral")}


# ══════════════════════════════════════════════════════════════════════════════
# HELPER UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def latest(s: pd.Series) -> float:
    if s is None or s.empty: return np.nan
    v = s.dropna()
    return float(v.iloc[-1]) if len(v) else np.nan

def value_n_ago(s: pd.Series, n: int) -> float:
    if s is None or s.empty: return np.nan
    v = s.dropna()
    return float(v.iloc[-(n+1)]) if len(v) >= n+1 else np.nan

def trend_direction(s: pd.Series, lookback: int = 3) -> str:
    if s is None or s.empty: return "unknown"
    v = s.dropna()
    if len(v) < 2: return "unknown"
    if len(v) < lookback + 1:
        diff = v.iloc[-1] - v.iloc[-2]
    else:
        recent = v.iloc[-lookback:].mean()
        prior  = v.iloc[-(lookback*2):-lookback].mean() if len(v) >= lookback*2 else v.iloc[0]
        diff   = recent - prior
    if pd.isna(diff): return "unknown"
    if diff > 0.05:   return "rising"
    if diff < -0.05:  return "falling"
    return "flat"

def yoy_growth(s: pd.Series, periods: int = 12) -> pd.Series:
    if s is None or len(s.dropna()) < periods + 1:
        return pd.Series(dtype=float)
    return s.pct_change(periods) * 100

def zscore_current(s: pd.Series, window: int = 60) -> float:
    """Z-score of most recent value vs trailing window."""
    v = s.dropna()
    if len(v) < 10: return np.nan
    hist = v.iloc[-window:] if len(v) >= window else v
    mu, sigma = hist.mean(), hist.std()
    if sigma < 1e-9: return 0.0
    return float((v.iloc[-1] - mu) / sigma)

def is_in_cb_window(code: str) -> bool:
    months = CURRENCY_INFO.get(code, {}).get("cb_meeting_months", [])
    today_m = datetime.today().month
    return today_m in months or (today_m % 12) + 1 in months


# ══════════════════════════════════════════════════════════════════════════════
# TAYLOR RULE CB PREDICTOR  (replaces the binary above/below-target rule)
# ══════════════════════════════════════════════════════════════════════════════

def taylor_rule_score(
    cpi_now:      float,
    cpi_3m_ago:   float,
    unemp_now:    float,
    unemp_prior:  float,
    gdp_growth:   float,
    rate_now:     float,
    rate_6m_ago:  float,
    code:         str,
) -> Tuple[float, str, str]:
    """
    Returns (score -1..+1, expected_action_label, inflation_bias_label).

    Taylor Rule intuition:
      - Inflation above target AND rising          → pressure to hike
      - Inflation above target but falling fast    → hold or consider cuts
      - Unemployment rising                        → easing pressure
      - GDP contracting                            → easing pressure
      - Rate already rising (momentum)             → likely to continue
      - Rate already falling                       → likely to continue

    Score > 0 → hawkish (Bullish for currency)
    Score < 0 → dovish  (Bearish for currency)
    """
    score = 0.0
    factors = 0

    # 1. Inflation gap: how far above/below 2% target
    #    Divisor tightened from 3.0 -> 2.0: a CPI of 5% (3pt gap) now maps
    #    to tanh(1.5)=0.90 instead of tanh(1.0)=0.76, so a clearly
    #    above-target print actually pushes the score toward the edges
    #    instead of staying compressed near the centre.
    if not pd.isna(cpi_now):
        infl_gap = cpi_now - INFLATION_TARGET
        infl_score = np.tanh(infl_gap / 2.0)
        score += infl_score * 0.35
        factors += 0.35

    # 2. Inflation momentum: recent trend (3m change)
    #    Divisor tightened from 2.0 -> 1.2.
    if not pd.isna(cpi_now) and not pd.isna(cpi_3m_ago):
        infl_chg = cpi_now - cpi_3m_ago
        mom_score = np.tanh(infl_chg / 1.2)
        score += mom_score * 0.20
        factors += 0.20

    # 3. Labour market: rising unemployment = dovish pressure
    #    Divisor tightened from 1.0 -> 0.6 (a 0.3pt unemployment move is a
    #    meaningful labour-market signal, not a rounding blip).
    if not pd.isna(unemp_now) and not pd.isna(unemp_prior):
        unemp_chg = unemp_now - unemp_prior
        unemp_score = -np.tanh(unemp_chg / 0.6)
        score += unemp_score * 0.20
        factors += 0.20

    # 4. GDP momentum
    #    Divisor tightened from 3.0 -> 2.0.
    if not pd.isna(gdp_growth):
        gdp_score_t = np.tanh(gdp_growth / 2.0)
        score += gdp_score_t * 0.15
        factors += 0.15

    # 5. Rate momentum: is the CB already moving in a direction?
    #    Divisor tightened from 1.0 -> 0.75.
    if not pd.isna(rate_now) and not pd.isna(rate_6m_ago):
        rate_chg = rate_now - rate_6m_ago
        rate_mom = np.tanh(rate_chg / 0.75)
        score += rate_mom * 0.10
        factors += 0.10

    if factors > 0:
        score = score / factors  # normalise to -1..+1
    else:
        score = 0.0

    score = float(np.clip(score, -1, 1))

    # Map to action label
    if score >= 0.4:
        action = "Strong Hike Expected"
        bias   = "Bullish"
    elif score >= 0.15:
        action = "Hike Expected"
        bias   = "Bullish"
    elif score <= -0.4:
        action = "Strong Cut Expected"
        bias   = "Bearish"
    elif score <= -0.15:
        action = "Cut Expected"
        bias   = "Bearish"
    else:
        action = "Hold Expected"
        bias   = "Neutral"

    return score, action, bias


# ══════════════════════════════════════════════════════════════════════════════
# MACRO REGIME CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════

REGIME_TRADE_BIAS: Dict[str, Dict] = {
    "Risk-On": {
        "colour": "#2ecc71",
        "long":   ["AUD", "NZD", "MXN", "ZAR"],
        "short":  ["JPY", "CHF", "USD"],
        "note":   "Favour high-beta / commodity currencies. Reduce safe havens.",
    },
    "Risk-Off": {
        "colour": "#e74c3c",
        "long":   ["JPY", "CHF", "USD"],
        "short":  ["AUD", "NZD", "MXN", "ZAR"],
        "note":   "Safe havens bid. Reduce carry, reduce commodity exposure.",
    },
    "Inflation Shock": {
        "colour": "#e67e22",
        "long":   ["USD", "CAD", "AUD"],
        "short":  ["JPY", "EUR"],
        "note":   "Hawkish CBs favoured. Energy exporters (CAD, AUD) benefit.",
    },
    "Growth Shock": {
        "colour": "#9b59b6",
        "long":   ["USD", "JPY"],
        "short":  ["AUD", "NZD", "GBP"],
        "note":   "Growth-sensitive currencies sold. Defensive positioning.",
    },
    "Carry Environment": {
        "colour": "#3498db",
        "long":   ["AUD", "NZD", "MXN"],
        "short":  ["JPY", "CHF"],
        "note":   "Low volatility. Real yield differentials drive returns.",
    },
    "Stagflation": {
        "colour": "#c0392b",
        "long":   ["CHF", "JPY"],
        "short":  ["GBP", "EUR", "AUD"],
        "note":   "Worst of both worlds. Reduce risk, avoid rate-sensitive currencies.",
    },
    "Neutral": {
        "colour": "#95a5a6",
        "long":   [],
        "short":  [],
        "note":   "No dominant regime. Rely on individual currency fundamentals.",
    },
}


def classify_regime(results: List[Dict]) -> Tuple[str, str]:
    """
    Classify macro regime from the aggregate of currency analysis results.
    Returns (regime_name, description).
    """
    if not results:
        return "Neutral", "Insufficient data."

    # Pull cross-currency averages
    cpis     = [r["CPI YoY %"]    for r in results if not pd.isna(r["CPI YoY %"])]
    gdps     = [r["GDP Growth %"] for r in results if not pd.isna(r["GDP Growth %"])]
    unemps   = [r["Unemployment %"] for r in results if not pd.isna(r["Unemployment %"])]
    cb_scores= [r["CB Taylor Score"] for r in results if not pd.isna(r.get("CB Taylor Score", np.nan))]

    avg_cpi   = np.mean(cpis)   if cpis   else np.nan
    avg_gdp   = np.mean(gdps)   if gdps   else np.nan
    avg_unemp = np.mean(unemps) if unemps else np.nan
    avg_cb    = np.mean(cb_scores) if cb_scores else np.nan

    # CPI direction: count currencies with rising inflation
    cpi_rising_count = sum(1 for r in results if r.get("CPI Direction") == "rising")
    cpi_falling_count= sum(1 for r in results if r.get("CPI Direction") == "falling")

    # GDP direction
    gdp_contracting = sum(1 for r in results if "Contraction" in r.get("GDP Trend",""))
    n = len(results)

    high_inflation  = not pd.isna(avg_cpi)  and avg_cpi  > 3.5
    low_growth      = not pd.isna(avg_gdp)  and avg_gdp  < 1.0
    rising_unemp    = not pd.isna(avg_unemp)
    hawkish_bias    = not pd.isna(avg_cb)   and avg_cb   > 0.15
    dovish_bias     = not pd.isna(avg_cb)   and avg_cb   < -0.15
    broad_contraction = gdp_contracting / n > 0.5 if n > 0 else False

    # Classification logic
    if high_inflation and low_growth:
        return "Stagflation", f"Avg CPI {avg_cpi:.1f}% with GDP {avg_gdp:.1f}% — classic stagflation mix."
    if high_inflation and hawkish_bias:
        return "Inflation Shock", f"Avg CPI {avg_cpi:.1f}%, majority of CBs signalling hikes."
    if broad_contraction and dovish_bias:
        return "Growth Shock", f"{gdp_contracting}/{n} currencies in contraction. CBs easing."
    if not high_inflation and not broad_contraction and dovish_bias:
        return "Carry Environment", "Low inflation, stable growth, dovish CBs — ideal carry conditions."
    if not high_inflation and not broad_contraction and not hawkish_bias:
        return "Risk-On", "Benign macro backdrop — risk appetite supported."
    if broad_contraction or (not pd.isna(avg_gdp) and avg_gdp < 0):
        return "Risk-Off", "Broad growth deterioration signals risk-off positioning."
    return "Neutral", "Mixed signals — no dominant regime."


# ══════════════════════════════════════════════════════════════════════════════
# CORE ANALYSIS ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def analyze_currency(code: str, macro: Dict[str, pd.Series],
                      fx_series: pd.Series, dq_log: DataQualityLog) -> Dict:

    policy_rate    = macro.get("policy_rate",     pd.Series(dtype=float))
    cpi_yoy        = macro.get("cpi_yoy",         pd.Series(dtype=float))
    gdp            = macro.get("gdp",             pd.Series(dtype=float))
    trade_bal      = macro.get("trade_balance",   pd.Series(dtype=float))
    unemployment   = macro.get("unemployment",    pd.Series(dtype=float))
    ten_year       = macro.get("ten_year",        pd.Series(dtype=float))
    pmi_mfg        = macro.get("pmi_mfg",         pd.Series(dtype=float))
    m2             = macro.get("m2",              pd.Series(dtype=float))
    consumer_conf  = macro.get("consumer_conf",   pd.Series(dtype=float))
    building_perms = macro.get("building_permits",pd.Series(dtype=float))

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

    for val, label in [(rate_now,"Policy rate"),(cpi_now,"CPI YoY"),(gdp_raw,"GDP")]:
        if pd.isna(val):
            dq_log.warn(code, f"{label} unavailable from FRED")

    # ── Real rates ───────────────────────────────────────────────────────
    real_rate_10y    = ten_y_now - cpi_now if not pd.isna(ten_y_now) and not pd.isna(cpi_now) else np.nan
    real_rate_policy = rate_now  - cpi_now if not pd.isna(rate_now)  and not pd.isna(cpi_now) else np.nan
    real_rate        = real_rate_10y if not pd.isna(real_rate_10y) else real_rate_policy

    # ── Yield curve ──────────────────────────────────────────────────────
    curve_slope = ten_y_now - rate_now if not pd.isna(ten_y_now) and not pd.isna(rate_now) else np.nan
    curve_state = "unknown"
    if not pd.isna(curve_slope):
        curve_state = "Inverted ⚠️" if curve_slope < 0 else "Normal ✓"

    # ── GDP — z-score relative to own history ────────────────────────────
    gdp_is_growth = gdp.dropna().abs().median() < 15 if not gdp.dropna().empty else False
    if gdp_is_growth:
        gdp_growth_now   = gdp_raw
        gdp_growth_prior = value_n_ago(gdp, 1)
        gdp_series_for_z = gdp
    else:
        gdp_yoy = yoy_growth(gdp, 4)
        gdp_growth_now   = latest(gdp_yoy)
        gdp_growth_prior = value_n_ago(gdp_yoy, 1)
        gdp_series_for_z = gdp_yoy

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

    # Z-score GDP: relative to own 5-year history (20 quarters)
    gdp_z = zscore_current(gdp_series_for_z, window=20)
    gdp_score = float(np.clip(gdp_z / 2.0, -1, 1)) if not pd.isna(gdp_z) else (
        float(np.clip(gdp_growth_now / 4.0, -1, 1)) if not pd.isna(gdp_growth_now) else np.nan
    )

    # ── Taylor Rule CB Score (replaces old binary rule) ──────────────────
    cpi_direction = trend_direction(cpi_yoy, lookback=3)
    cpi_3m_ago    = value_n_ago(cpi_yoy, 3)
    unemp_prior   = value_n_ago(unemployment, 3)
    rate_6m_ago   = value_n_ago(policy_rate, 6)
    rate_direction= trend_direction(policy_rate, lookback=3)
    rate_change_12m = rate_now - value_n_ago(policy_rate, 12) if not pd.isna(rate_now) else np.nan

    cb_taylor_score, expected_action, inflation_bias = taylor_rule_score(
        cpi_now, cpi_3m_ago, unemp_now, unemp_prior,
        gdp_growth_now, rate_now, rate_6m_ago, code
    )
    inflation_score = cb_taylor_score  # CB score IS the inflation/policy signal

    # ── Trade balance ────────────────────────────────────────────────────
    # IMPORTANT: FRED trade-balance series are NOT in comparable units
    # across countries. USD uses BOPGSTB (raw USD millions). Most other
    # currencies use OECD "XTNTVA01..." series, which are an OECD index
    # unit (CXMLSA), not USD millions. Comparing raw levels side-by-side
    # (e.g. "USD: -50,000" vs "EUR: -12.4") is comparing different units
    # and produces numbers that look arbitrary/wrong -- because they are
    # not on the same scale. We therefore:
    #   1. Score trade direction/sign using the country's OWN series only
    #      (sign + trend within one unit system is still meaningful).
    #   2. Compute a z-score vs the country's own trailing history so the
    #      Overview can show "how strong vs normal for this country" --
    #      directly comparable across currencies, the same pattern already
    #      used for GDP -- instead of showing incomparable raw levels.
    trade_dir = trend_direction(trade_bal, lookback=3)
    trade_score = np.nan
    if not pd.isna(trade_now):
        base = 0.5 if trade_now > 0 else -0.5
        mom  = 0.3 if trade_dir == "rising" else (-0.3 if trade_dir == "falling" else 0.0)
        trade_score = float(np.clip(base + mom, -1, 1))
    trade_z = zscore_current(trade_bal, window=20)
    # Unit label so the UI never implies cross-currency comparability of
    # the raw figure.
    trade_unit = "$M (raw)" if code == "USD" else "OECD index (own units)"

    # ── M2 ───────────────────────────────────────────────────────────────
    m2_growth_now = latest(yoy_growth(m2, 12))
    m2_trend      = trend_direction(yoy_growth(m2, 12))
    m2_score      = np.nan
    if not pd.isna(m2_growth_now):
        if code in HIGH_INFLATION_CURRENCIES:
            m2_yoy_s = yoy_growth(m2, 12).dropna()
            if len(m2_yoy_s) >= 12:
                med = m2_yoy_s.rolling(24, min_periods=6).median().iloc[-1]
                m2_score = float(np.clip(-(m2_growth_now - med) / 10.0, -1, 1))
            else:
                m2_score = 0.0
        else:
            if 2 < m2_growth_now < 8:   m2_score = 0.3
            elif m2_growth_now >= 8:    m2_score = -0.2
            elif m2_growth_now < 0:     m2_score = -0.5
            else:                       m2_score = 0.0
        m2_score = float(np.clip(m2_score, -1, 1))

    # ── Consumer Confidence ──────────────────────────────────────────────
    conf_dir   = trend_direction(consumer_conf, lookback=3)
    conf_score = np.nan
    if not pd.isna(conf_now):
        conf_score = 0.3 if conf_dir == "rising" else (-0.3 if conf_dir == "falling" else 0.0)

    # ── PMI (placeholder — no free FRED series) ──────────────────────────
    pmi_dir   = trend_direction(pmi_mfg, lookback=3)
    pmi_score = np.nan
    if not pd.isna(pmi_now):
        pmi_score = float(np.clip((pmi_now - 50) / 10.0 +
                                   (0.1 if pmi_dir=="rising" else -0.1 if pmi_dir=="falling" else 0), -1, 1))

    # ── Composite score ──────────────────────────────────────────────────
    component_scores = {
        "gdp": gdp_score, "inflation": inflation_score,
        "cb_taylor": cb_taylor_score,  # separate weight from inflation_score
        "trade": trade_score, "m2": m2_score, "conf": conf_score,
    }
    # Deduplicate: inflation_score and cb_taylor_score are the same value.
    # We only want to count the CB signal once in composite (via 'inflation' slot).
    # Use inflation only once; cb_taylor weight goes to zero here to avoid double-counting.
    effective_weights = {**SCORE_WEIGHTS, "cb_taylor": 0.0}
    weighted_sum = sum(effective_weights[k] * v for k, v in component_scores.items() if not pd.isna(v))
    weight_used  = sum(effective_weights[k] for k, v in component_scores.items() if not pd.isna(v))
    composite    = float(weighted_sum / weight_used) if weight_used > 0 else np.nan

    # Bullish/Bearish band. Composite is a weighted blend of several -1..+1
    # component scores, so it rarely reaches the extremes even when one
    # driver (e.g. CB policy) is screaming in one direction -- that's
    # expected dampening, not a bug. The 0.25 cutoff is calibrated against
    # the tanh divisors in taylor_rule_score(); if those divisors change,
    # this threshold should be re-checked against realistic scenarios.
    if pd.isna(composite):        overall_bias = "Neutral"
    elif composite >= 0.25:       overall_bias = "Bullish"
    elif composite <= -0.25:      overall_bias = "Bearish"
    else:                         overall_bias = "Neutral"

    # ── CPI forward: mean-reversion blend (not linear extrapolation) ─────
    # Expected CPI = 60% current + 40% target, capped at ±3% from current
    cpi_exp12m = np.nan
    if not pd.isna(cpi_now):
        raw_exp = 0.60 * cpi_now + 0.40 * INFLATION_TARGET
        cpi_exp12m = float(np.clip(raw_exp, cpi_now - 3.0, cpi_now + 3.0))

    rate_step_map = {"Strong Hike Expected": +0.75, "Hike Expected": +0.25,
                      "Hold Expected": 0.0, "Cut Expected": -0.25, "Strong Cut Expected": -0.75}
    rate_exp12m = rate_now + rate_step_map.get(expected_action, 0.0) if not pd.isna(rate_now) else np.nan

    gdp_exp12m = np.nan
    if not pd.isna(gdp_growth_now) and not pd.isna(gdp_growth_prior):
        # Mean-revert toward 2% rather than extrapolating the trend
        raw = gdp_growth_now + 0.5 * (gdp_growth_now - gdp_growth_prior)
        gdp_exp12m = float(0.7 * raw + 0.3 * 2.0)

    # ── Pre-meeting window ───────────────────────────────────────────────
    pre_meeting_window = is_in_cb_window(code)
    pre_meeting_score  = 2 if (pre_meeting_window and overall_bias != "Neutral") else (1 if pre_meeting_window else 0)

    # ── FX momentum ──────────────────────────────────────────────────────
    fx_now = latest(fx_series)
    fx_chg_1m = fx_chg_3m = fx_chg_12m = np.nan
    if not fx_series.empty:
        c = fx_series.dropna()
        if len(c) > 21:  fx_chg_1m  = (c.iloc[-1]/c.iloc[-21]  - 1)*100
        if len(c) > 63:  fx_chg_3m  = (c.iloc[-1]/c.iloc[-63]  - 1)*100
        if len(c) > 252: fx_chg_12m = (c.iloc[-1]/c.iloc[-252] - 1)*100
        if code in USD_IS_BASE:
            fx_chg_1m  = -fx_chg_1m  if not pd.isna(fx_chg_1m)  else np.nan
            fx_chg_3m  = -fx_chg_3m  if not pd.isna(fx_chg_3m)  else np.nan
            fx_chg_12m = -fx_chg_12m if not pd.isna(fx_chg_12m) else np.nan

    info = CURRENCY_INFO.get(code, {})
    return {
        "code": code, "name": info.get("name", code),
        "country": info.get("country", "—"), "type": info.get("type", "—"), "cb": info.get("cb","—"),
        "Policy Rate %":       rate_now,   "CPI YoY %": cpi_now,
        "GDP Growth %":        gdp_growth_now, "GDP Z-Score": gdp_z,
        "Trade Balance":       trade_now,  "Unemployment %": unemp_now,
        "Trade Z-Score":       trade_z,    "Trade Unit": trade_unit,
        "10Y Yield %":         ten_y_now,  "Real Rate % (10Y)": real_rate_10y,
        "Real Rate % (Policy)":real_rate_policy, "Real Rate %": real_rate,
        "Curve Slope":         curve_slope,"Curve State": curve_state,
        "PMI":                 pmi_now,    "M2 Growth YoY %": latest(yoy_growth(m2,12)),
        "Consumer Conf":       conf_now,   "Building Permits": permits_now,
        "GDP Trend":           gdp_trend,  "CPI Direction": cpi_direction,
        "Rate Direction":      rate_direction,
        "Rate Δ 12m (bps)":    rate_change_12m*100 if not pd.isna(rate_change_12m) else np.nan,
        "Trade Bal Direction": trade_dir,  "M2 Trend": m2_trend, "Conf Direction": conf_dir,
        "CB Taylor Score":     cb_taylor_score,
        "Expected CB Action":  expected_action, "Inflation Bias": inflation_bias,
        "CPI Expected (12m)":  cpi_exp12m, "Rate Expected (12m)": rate_exp12m,
        "GDP Expected (12m)":  gdp_exp12m,
        "GDP Score":           gdp_score,  "Inflation Score": inflation_score,
        "Trade Score":         trade_score,"M2 Score": m2_score, "Conf Score": conf_score,
        "Composite Score":     composite,  "Overall Bias": overall_bias,
        "COT Net": np.nan, "COT Change": np.nan, "COT Bias": "—",
        "Pre-Meeting Window":  pre_meeting_window, "Pre-Meeting Score": pre_meeting_score,
        "FX Spot":             fx_now, "FX Chg 1m %": fx_chg_1m,
        "FX Chg 3m %":         fx_chg_3m, "FX Chg 12m %": fx_chg_12m,
    }


# ══════════════════════════════════════════════════════════════════════════════
# PAIR SCORING ENGINE  (relative analysis)
# ══════════════════════════════════════════════════════════════════════════════

def build_pair_matrix(results: List[Dict], max_tier: int = 3) -> pd.DataFrame:
    """
    Build a scored pair matrix. For every A/B combination:
      pair_score = A_composite - B_composite
    Filtered to liquidity tier ≤ max_tier. Sorted by |score| descending.
    """
    rows = []
    for i, ra in enumerate(results):
        for j, rb in enumerate(results):
            if i >= j: continue
            a, b = ra["code"], rb["code"]
            tier = pair_liquidity_tier(a, b)
            if tier > max_tier: continue
            sa = ra.get("Composite Score", np.nan)
            sb = rb.get("Composite Score", np.nan)
            if pd.isna(sa) or pd.isna(sb): continue
            pair_score = sa - sb
            # Direction: positive = long A / short B is the trade
            long_code  = a if pair_score >= 0 else b
            short_code = b if pair_score >= 0 else a
            abs_score  = abs(pair_score)
            rows.append({
                "Pair":          f"{long_code}/{short_code}",
                "Pair Score":    round(pair_score, 3),
                "Abs Score":     round(abs_score, 3),
                "Long":          long_code,
                "Short":         short_code,
                "Long Score":    round(sa if pair_score >= 0 else sb, 3),
                "Short Score":   round(sb if pair_score >= 0 else sa, 3),
                "Long Bias":     ra["Overall Bias"] if pair_score >= 0 else rb["Overall Bias"],
                "Short Bias":    rb["Overall Bias"] if pair_score >= 0 else ra["Overall Bias"],
                "Long CB":       ra["Expected CB Action"] if pair_score >= 0 else rb["Expected CB Action"],
                "Short CB":      rb["Expected CB Action"] if pair_score >= 0 else ra["Expected CB Action"],
                "Liquidity Tier":tier,
                "Conviction":    ("★★★" if abs_score >= 0.4 else ("★★" if abs_score >= 0.2 else "★")),
            })
    df = pd.DataFrame(rows).sort_values("Abs Score", ascending=False).reset_index(drop=True)
    return df


def rank_currencies(results: List[Dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        score = r.get("Composite Score", np.nan)
        stars = ("★★★★★" if score >= 0.5 else "★★★★☆" if score >= 0.3 else
                 "★★★☆☆" if score >= 0.1 else "★★☆☆☆" if score >= -0.1 else
                 "★☆☆☆☆" if score >= -0.3 else "☆☆☆☆☆") if not pd.isna(score) else "—"
        tb = r["Trade Balance"]
        tb_z = r.get("Trade Z-Score", np.nan)
        # Show the z-score (comparable across currencies) as the primary
        # signal, with the raw print + its native unit in parentheses so
        # it's clear the raw numbers are NOT directly comparable across
        # rows of this table.
        if not pd.isna(tb_z):
            tb_label = f"{'▲' if tb_z > 0 else '▼'} z={tb_z:+.2f}"
        elif not pd.isna(tb):
            tb_label = f"{'▲' if tb > 0 else '▼'} {tb:,.0f} ({r.get('Trade Unit','—')})"
        else:
            tb_label = "—"
        rows.append({
            "Currency":     r["code"],     "Name": r["name"],
            "Bias":         r["Overall Bias"], "Score": round(score,3) if not pd.isna(score) else np.nan,
            "Stars":        stars,
            "Rate %":       r["Policy Rate %"], "CPI YoY %": r["CPI YoY %"],
            "Real Rate %":  r["Real Rate %"],   "GDP %": r["GDP Growth %"],
            "GDP Z":        round(r["GDP Z-Score"],2) if not pd.isna(r.get("GDP Z-Score",np.nan)) else np.nan,
            "Trade Balance":tb_label,
            "CB Action":    r["Expected CB Action"], "CB Score": round(r.get("CB Taylor Score",np.nan),2) if not pd.isna(r.get("CB Taylor Score",np.nan)) else np.nan,
            "COT Bias":     r["COT Bias"],
            "FX Mom 1m %":  round(r["FX Chg 1m %"],2) if not pd.isna(r["FX Chg 1m %"]) else np.nan,
            "Pre-Meeting":  "🎯" if r.get("Pre-Meeting Score",0) >= 2 else ("📅" if r.get("Pre-Meeting Window") else ""),
        })
    df = pd.DataFrame(rows)
    return df.sort_values("Score", ascending=False).reset_index(drop=True) if "Score" in df.columns else df


def real_rate_differential_table(results: List[Dict]) -> pd.DataFrame:
    codes  = [r["code"] for r in results]
    rrates = {r["code"]: r["Real Rate %"] for r in results}
    rows   = []
    for long in codes:
        row = {"Long ↓ / Short →": long}
        for short in codes:
            if long == short:
                row[short] = "—"
            else:
                v1, v2 = rrates.get(long, np.nan), rrates.get(short, np.nan)
                diff = v1 - v2 if not pd.isna(v1) and not pd.isna(v2) else np.nan
                row[short] = f"{diff:+.2f}%" if not pd.isna(diff) else "—"
        rows.append(row)
    return pd.DataFrame(rows).set_index("Long ↓ / Short →")


def cpi_actual_vs_expected(cpi_series: pd.Series, months: int = 24) -> pd.DataFrame:
    """
    Build a trailing actual-vs-model-expected CPI table for one currency.

    "Expected" reuses the SAME mean-reversion blend already used elsewhere
    in this app for the forward 12m CPI forecast (60% persistence /
    40% reversion to the 2% inflation target), but applied retrospectively:
    for each past month, "expected" = what the model would have forecast
    GIVEN the prior month's actual reading. This lets the user see how well
    the simple mean-reversion heuristic has been tracking reality, month
    by month, rather than only seeing a single forward-looking number.

    Note this is a one-step-ahead model check, not a real economist
    forecast -- it's intended to show the model's recent error pattern,
    not to be a forecasting authority.
    """
    s = cpi_series.dropna()
    if s.empty:
        return pd.DataFrame(columns=["Date", "Actual CPI YoY %", "Model-Expected %", "Surprise (Actual − Expected)"])

    s = s.iloc[-(months + 1):]  # need one extra point to seed "prior" for the first row
    rows = []
    for i in range(1, len(s)):
        date        = s.index[i]
        actual      = float(s.iloc[i])
        prior       = float(s.iloc[i - 1])
        expected    = 0.60 * prior + 0.40 * INFLATION_TARGET
        expected    = float(np.clip(expected, prior - 3.0, prior + 3.0))
        surprise    = actual - expected
        rows.append({
            "Date": date, "Actual CPI YoY %": round(actual, 2),
            "Model-Expected %": round(expected, 2),
            "Surprise (Actual − Expected)": round(surprise, 2),
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# IXL THEME SCORER
# ══════════════════════════════════════════════════════════════════════════════

def ixl_score(impact: int, likelihood: int) -> int:
    return impact * likelihood

def ixl_label(score: int) -> str:
    if score >= 70: return "✅ Tradeable"
    if score >= 40: return "⚠️ Monitor"
    return "❌ Not Tradeable"


# ══════════════════════════════════════════════════════════════════════════════
# DXY CORRELATIONS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_dxy_correlations() -> Dict[str, pd.Series]:
    tickers = {"DXY": DXY_TICKER, "Gold": "GC=F", "Crude": "CL=F", "SPX": "^GSPC"}
    out = {}
    for label, t in tickers.items():
        try:
            data = yf.download(t, period="1y", interval="1d",
                                auto_adjust=True, progress=False)
            if data.empty:
                out[label] = pd.Series(dtype=float); continue
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            s = data["Close"] if "Close" in data.columns else pd.Series(dtype=float)
            if isinstance(s, pd.DataFrame): s = s.iloc[:, 0]
            out[label] = s.dropna()
        except Exception:
            out[label] = pd.Series(dtype=float)
    return out

def compute_rolling_corr(s1: pd.Series, s2: pd.Series, window: int = 20) -> pd.Series:
    df = pd.concat([s1.rename("a"), s2.rename("b")], axis=1).dropna()
    return df["a"].rolling(window).corr(df["b"])


# ══════════════════════════════════════════════════════════════════════════════
# UI HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def fmt(val, decimals=2, suffix="") -> str:
    return "—" if pd.isna(val) else f"{val:.{decimals}f}{suffix}"

def bias_badge(bias: str) -> str:
    return {"Bullish":"🟢","Bearish":"🔴","Neutral":"🟡"}.get(bias,"⚪") + f" {bias}"

def score_bar(score: float) -> str:
    if pd.isna(score): return "—"
    filled = round((score + 1) / 2 * 10)
    return "█" * filled + "░" * (10 - filled) + f"  {score:+.2f}"

def colour_bias_style(val):
    return {"Bullish": "background-color:#1a4a1a;color:#7fff7f",
            "Bearish": "background-color:#4a1a1a;color:#ff9999",
            "Neutral": "background-color:#2a2a1a;color:#ffff99"}.get(val, "")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(page_title="Forex Fundamental Scanner",
                        page_icon="💱", layout="wide")
    st.title("💱 Forex Fundamental Scanner  v3.0")
    st.caption("Taylor Rule CB Prediction · GDP Z-Score · Pair Matrix · Regime Classifier · Heat Map")

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Settings")
        all_codes      = list(CURRENCY_INFO.keys())
        default_codes  = ["USD","EUR","GBP","JPY","AUD","CAD","TRY"]
        selected_codes = st.multiselect("Currencies", all_codes, default=default_codes)
        st.divider()
        max_tier = st.selectbox("Pair liquidity filter",
                                 options=[1,2,3,4],
                                 format_func=lambda x: {1:"G7 majors only",2:"G10 + crosses",
                                                         3:"Include liquid EMs",4:"All pairs"}[x],
                                 index=2)
        st.divider()
        show_dxy = st.checkbox("Show DXY correlations", value=True)
        st.divider()
        st.subheader("📜 IXL Theme Builder")
        ixl_theme  = st.text_input("Theme", placeholder="e.g. US tariffs on EU goods")
        ixl_impact = st.slider("Impact (1–10)",     1, 10, 7)
        ixl_like   = st.slider("Likelihood (1–10)", 1, 10, 8)
        ixl_res    = ixl_score(ixl_impact, ixl_like)
        st.metric("IXL Score", ixl_res, ixl_label(ixl_res))

    if not selected_codes:
        st.warning("Select at least one currency.")
        return

    # ── Fetch & analyse ───────────────────────────────────────────────────────
    dq_log = DataQualityLog()
    cpi_history: Dict[str, pd.Series] = {}
    with st.spinner("Fetching macro data…"):
        results, cot_df = [], fetch_cot_data()
        for code in selected_codes:
            macro   = fetch_currency_macro(code, dq_log)
            fx_tick = FX_TICKER.get(code, "")
            fx_hist = fetch_fx_history(fx_tick) if fx_tick else pd.Series(dtype=float)
            r       = analyze_currency(code, macro, fx_hist, dq_log)
            cot     = get_cot_position(code, cot_df)
            r["COT Net"] = cot["net"]; r["COT Change"] = cot["change"]; r["COT Bias"] = cot["bias"]
            results.append(r)
            cpi_history[code] = macro.get("cpi_yoy", pd.Series(dtype=float))

    ranked_df  = rank_currencies(results)
    pair_df    = build_pair_matrix(results, max_tier=max_tier)
    rr_diff_df = real_rate_differential_table(results)
    regime, regime_desc = classify_regime(results)

    with st.sidebar:
        if dq_log.any():
            st.warning(f"⚠️ {dq_log.count()} data issue(s)")
        regime_info = REGIME_TRADE_BIAS.get(regime, REGIME_TRADE_BIAS["Neutral"])
        st.divider()
        st.subheader("🌐 Macro Regime")
        st.markdown(f"<span style='color:{regime_info['colour']};font-size:1.2em;font-weight:bold'>{regime}</span>",
                    unsafe_allow_html=True)
        st.caption(regime_desc)

    # ── Eight tabs ────────────────────────────────────────────────────────────
    (tab_overview, tab_heatmap, tab_pairs_tab,
     tab_carry, tab_ixl, tab_cot, tab_premeeting, tab_cpi) = st.tabs([
        "📊 Overview", "🌡️ Heat Map & Regime", "🔀 Pairs",
        "💰 Carry", "🎯 IXL", "📋 COT", "🕐 Pre-Meeting", "📈 CPI Actual vs Expected"
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 · OVERVIEW
    # ══════════════════════════════════════════════════════════════════════════
    with tab_overview:
        st.subheader("Currency Ranking")
        st.caption(
            "GDP Z-Score = GDP growth vs own 5-year history (not a fixed divisor). "
            "CB Score = Taylor Rule estimate (−1 dovish … +1 hawkish). "
            "Real Rate uses 10Y yield where available. "
            "Trade Balance column shows a z-score vs each country's own history (comparable across "
            "rows) — raw FRED trade-balance levels use different units per country (USD = $ millions, "
            "most others = OECD index units) and are NOT directly comparable, so we don't show them side-by-side."
        )
        styled = ranked_df.style.map(colour_bias_style, subset=["Bias"])
        st.dataframe(styled, use_container_width=True, hide_index=True)

        fig = go.Figure(go.Bar(
            x=ranked_df["Currency"], y=ranked_df["Score"],
            marker_color=["#2ecc71" if b=="Bullish" else "#e74c3c" if b=="Bearish" else "#f1c40f"
                           for b in ranked_df["Bias"]],
            text=[fmt(s,3) for s in ranked_df["Score"]], textposition="outside",
        ))
        fig.update_layout(title="Composite Fundamental Score",
                           yaxis_title="Score (−1 to +1)",
                           template="plotly_dark", height=340)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Detailed Breakdowns")
        for r in results:
            with st.expander(f"**{r['code']}** — {r['name']}  {bias_badge(r['Overall Bias'])}"):
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown("**Macro Levels**")
                    st.write(f"Policy Rate:      {fmt(r['Policy Rate %'])}%")
                    st.write(f"CPI YoY:          {fmt(r['CPI YoY %'])}% ({r['CPI Direction']})")
                    st.write(f"Real Rate (10Y):  {fmt(r['Real Rate % (10Y)'])}%")
                    st.write(f"Real Rate (Policy):{fmt(r['Real Rate % (Policy)'])}%")
                    st.write(f"GDP Growth:       {fmt(r['GDP Growth %'])}%  Z={fmt(r.get('GDP Z-Score',np.nan))}")
                    st.write(f"Unemployment:     {fmt(r['Unemployment %'])}%")
                    st.write(f"10Y Yield:        {fmt(r['10Y Yield %'])}%")
                    st.write(f"Yield Curve:      {r['Curve State']}")
                with c2:
                    st.markdown("**Leading Indicators**")
                    st.write(f"Trade Balance:    {fmt(r['Trade Balance'],2)} {r.get('Trade Unit','')}  "
                             f"(z={fmt(r.get('Trade Z-Score',np.nan),2)} vs own history, {r['Trade Bal Direction']})")
                    st.write(f"M2 Growth:        {fmt(r['M2 Growth YoY %'])}%  ({r['M2 Trend']})")
                    st.write(f"Consumer Conf:    {fmt(r['Consumer Conf'],1)}  ({r['Conf Direction']})")
                    pmi_lbl = fmt(r['PMI'],1) if not pd.isna(r['PMI']) else "N/A (ISM not on FRED)"
                    st.write(f"PMI:              {pmi_lbl}")
                    st.write(f"Building Permits: {fmt(r['Building Permits'],0)}")
                with c3:
                    st.markdown("**CB & Forward Outlook**")
                    st.write(f"Taylor CB Score:  {fmt(r.get('CB Taylor Score',np.nan))} ← key signal")
                    st.write(f"CB Action:        **{r['Expected CB Action']}**")
                    st.write(f"CPI (12m fwd):    {fmt(r['CPI Expected (12m)'])}%")
                    st.write(f"Rate (12m fwd):   {fmt(r['Rate Expected (12m)'])}%")
                    st.write(f"GDP (12m fwd):    {fmt(r['GDP Expected (12m)'])}%")
                    st.write(f"FX 1m/3m/12m:     {fmt(r['FX Chg 1m %'])}% / {fmt(r['FX Chg 3m %'])}% / {fmt(r['FX Chg 12m %'])}%")

                st.write("**Component Scores**")
                for label, key in [("GDP","GDP Score"),("INFLATION","Inflation Score"),
                                    ("TRADE","Trade Score"),("M2","M2 Score"),("CONF","Conf Score")]:
                    st.text(f"  {label:10s}  {score_bar(r.get(key, np.nan))}")

        if dq_log.any():
            with st.expander("⚠️ Data Quality Warnings"):
                for code, msgs in dq_log.all().items():
                    st.write(f"**{code}**: " + " | ".join(msgs))

        csv_buf = io.StringIO()
        ranked_df.to_csv(csv_buf, index=False)
        st.download_button("⬇️ Export CSV", csv_buf.getvalue(), "forex_ranking.csv", "text/csv")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 · HEAT MAP & REGIME
    # ══════════════════════════════════════════════════════════════════════════
    with tab_heatmap:
        # ── Currency strength heat map ────────────────────────────────────
        st.subheader("🌡️ Currency Strength Heat Map")
        hm_codes  = ranked_df["Currency"].tolist()
        hm_scores = ranked_df["Score"].tolist()
        bar_colors = ["#2ecc71" if s > 0.25 else "#e74c3c" if s < -0.25 else "#f1c40f"
                       for s in hm_scores]
        fig_hm = go.Figure(go.Bar(
            x=hm_scores, y=hm_codes, orientation="h",
            marker_color=bar_colors,
            text=[f"{fmt(s,3)}  {ranked_df.loc[ranked_df['Currency']==c,'Stars'].values[0]}"
                   for c, s in zip(hm_codes, hm_scores)],
            textposition="outside",
        ))
        fig_hm.update_layout(
            title="Strongest → Weakest (Composite Score)",
            xaxis_title="Score", xaxis=dict(range=[-1.1, 1.1]),
            yaxis=dict(autorange="reversed"),
            template="plotly_dark", height=max(300, len(hm_codes)*50),
        )
        st.plotly_chart(fig_hm, use_container_width=True)

        # ── Component radar chart per currency ───────────────────────────
        st.subheader("Component Breakdown — Radar")
        radar_currencies = st.multiselect("Select currencies for radar",
                                           hm_codes, default=hm_codes[:4])
        categories = ["GDP", "Inflation", "Trade", "M2", "Conf"]
        score_keys  = ["GDP Score","Inflation Score","Trade Score","M2 Score","Conf Score"]
        fig_radar = go.Figure()
        for r in results:
            if r["code"] not in radar_currencies: continue
            vals = [r.get(k, 0) or 0 for k in score_keys]
            vals += [vals[0]]  # close polygon
            fig_radar.add_trace(go.Scatterpolar(
                r=vals, theta=categories + [categories[0]],
                fill="toself", name=r["code"],
                line=dict(width=2),
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[-1,1])),
            template="plotly_dark", height=450,
            title="Factor Radar (−1 = bearish, +1 = bullish)",
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # ── Regime panel ──────────────────────────────────────────────────
        st.divider()
        st.subheader("🌐 Macro Regime")
        ri = REGIME_TRADE_BIAS.get(regime, REGIME_TRADE_BIAS["Neutral"])
        r1, r2 = st.columns([1, 2])
        with r1:
            st.markdown(f"<div style='background:{ri['colour']}22;border-left:4px solid "
                        f"{ri['colour']};padding:12px;border-radius:6px'>"
                        f"<h2 style='color:{ri['colour']};margin:0'>{regime}</h2>"
                        f"<p style='margin:4px 0 0'>{regime_desc}</p></div>",
                        unsafe_allow_html=True)
        with r2:
            st.markdown(f"**Trade implication:** {ri['note']}")
            if ri["long"]:
                long_str  = "  ".join([f"🟢 **{c}**" for c in ri["long"]  if c in selected_codes])
                short_str = "  ".join([f"🔴 **{c}**" for c in ri["short"] if c in selected_codes])
                if long_str:  st.markdown(f"Favour Long: {long_str}")
                if short_str: st.markdown(f"Favour Short: {short_str}")
            st.info("Regime is inferred from aggregate macro signals across selected currencies. "
                    "It is a qualitative guide — not a precise classification.")

        # ── CB Taylor Score bar ───────────────────────────────────────────
        st.divider()
        st.subheader("Central Bank Taylor Rule Scores")
        st.caption("Derived from: inflation gap + inflation momentum + unemployment trend + GDP + rate momentum. "
                    "Positive = hawkish bias (bullish for currency), negative = dovish.")
        cb_codes  = [r["code"] for r in results]
        cb_scores = [r.get("CB Taylor Score", 0) or 0 for r in results]
        cb_actions= [r["Expected CB Action"] for r in results]
        fig_cb = go.Figure(go.Bar(
            x=cb_codes, y=cb_scores,
            marker_color=["#2ecc71" if s > 0 else "#e74c3c" for s in cb_scores],
            text=cb_actions, textposition="outside",
        ))
        fig_cb.update_layout(title="Taylor Rule CB Score (−1 to +1)",
                              yaxis=dict(range=[-1.1,1.1]),
                              template="plotly_dark", height=320)
        st.plotly_chart(fig_cb, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3 · PAIRS
    # ══════════════════════════════════════════════════════════════════════════
    with tab_pairs_tab:
        st.subheader("Pair Score Matrix — Relative Fundamental Analysis")
        st.caption(
            "Pair Score = Long currency composite − Short currency composite. "
            "Higher absolute score = greater fundamental divergence. "
            "Liquidity tier filter applied (sidebar). ★★★ = high conviction."
        )

        if pair_df.empty:
            st.info("No qualifying pairs with current liquidity filter. Try expanding to Tier 3 or 4.")
        else:
            top_df = pair_df[["Pair","Pair Score","Conviction","Long","Long Score",
                               "Short","Short Score","Long CB","Short CB","Liquidity Tier"]].head(15)
            styled_pairs = top_df.style.background_gradient(
                subset=["Pair Score"], cmap="RdYlGn", vmin=-1, vmax=1)
            st.dataframe(styled_pairs, use_container_width=True, hide_index=True)

            # Conviction top 3
            st.subheader("Top Conviction Trades")
            top3 = pair_df.head(3)
            for _, p in top3.iterrows():
                with st.container(border=True):
                    col1, col2, col3 = st.columns([1,1,2])
                    col1.metric("Pair", p["Pair"])
                    col1.write(p["Conviction"])
                    col2.metric("Pair Score", fmt(p["Pair Score"],3))
                    col2.write(f"Tier {int(p['Liquidity Tier'])}")
                    col3.write(f"**Long {p['Long']}**: score {fmt(p['Long Score'],3)}  ·  CB: {p['Long CB']}")
                    col3.write(f"**Short {p['Short']}**: score {fmt(p['Short Score'],3)}  ·  CB: {p['Short CB']}")

        # Full NxN matrix
        st.divider()
        st.subheader("Full Score Differential Matrix")
        st.caption("Cell = row currency score − column currency score. "
                    "Positive = row currency is fundamentally stronger.")
        codes_in  = [r["code"] for r in results]
        scores_in = {r["code"]: r.get("Composite Score", np.nan) for r in results}
        matrix_data = []
        for a in codes_in:
            row = {}
            for b in codes_in:
                if a == b: row[b] = "—"
                else:
                    v1, v2 = scores_in.get(a, np.nan), scores_in.get(b, np.nan)
                    row[b] = f"{v1-v2:+.3f}" if not pd.isna(v1) and not pd.isna(v2) else "—"
            matrix_data.append({"":a, **row})
        matrix_df = pd.DataFrame(matrix_data).set_index("")
        st.dataframe(matrix_df, use_container_width=True)

        st.divider()
        st.subheader("DXY Correlation Dashboard")
        if show_dxy:
            dxy_data = fetch_dxy_correlations()
            dxy = dxy_data.get("DXY", pd.Series(dtype=float))
            if not dxy.empty:
                corr_rows = []
                for label, s in dxy_data.items():
                    if label == "DXY" or s.empty: continue
                    corr = compute_rolling_corr(dxy, s).iloc[-1] if len(s) > 20 else np.nan
                    corr_rows.append({"Asset": label,
                                       "20d Corr vs DXY": fmt(corr,3),
                                       "Interpretation": ("Negative (typical)" if not pd.isna(corr) and corr<0
                                                           else "Positive" if not pd.isna(corr) and corr>0 else "—")})
                st.dataframe(pd.DataFrame(corr_rows), use_container_width=True, hide_index=True)
                fig_dxy = go.Figure()
                fig_dxy.add_trace(go.Scatter(x=dxy.index, y=dxy.values, name="DXY",
                                               line=dict(color="#3498db", width=2)))
                for label, col in [("Gold","#f1c40f"),("SPX","#2ecc71"),("Crude","#e67e22")]:
                    s = dxy_data.get(label, pd.Series(dtype=float))
                    if not s.empty:
                        s_n = s / s.iloc[0] * dxy.iloc[0]
                        fig_dxy.add_trace(go.Scatter(x=s_n.index, y=s_n.values, name=label,
                                                       line=dict(color=col, width=1.5)))
                fig_dxy.update_layout(title="DXY vs Gold / SPX / Crude (normalised)",
                                       template="plotly_dark", height=320)
                st.plotly_chart(fig_dxy, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 4 · CARRY
    # ══════════════════════════════════════════════════════════════════════════
    with tab_carry:
        st.subheader("Real Rate Differentials (10Y-based)")
        st.dataframe(rr_diff_df, use_container_width=True)

        st.subheader("Classic Carry Pairs")
        result_map = {r["code"]: r for r in results}
        carry_rows = []
        for lc, sc in CLASSIC_CARRY_PAIRS:
            if lc not in result_map or sc not in result_map: continue
            rl, rs = result_map[lc], result_map[sc]
            rrd = (rl["Real Rate %"] - rs["Real Rate %"]
                   if not pd.isna(rl["Real Rate %"]) and not pd.isna(rs["Real Rate %"]) else np.nan)
            carry_rows.append({
                "Pair": f"{lc}/{sc}",
                "Long Real Rate %":  fmt(rl["Real Rate %"]),
                "Short Real Rate %": fmt(rs["Real Rate %"]),
                "Real Yield Diff %": fmt(rrd),
                "Long Bias":  rl["Overall Bias"], "Short Bias": rs["Overall Bias"],
                "Long Policy Rate %":  fmt(rl["Policy Rate %"]),
                "Short Policy Rate %": fmt(rs["Policy Rate %"]),
                "VIX Check": "⚠️" if (rl["Curve State"]=="Inverted ⚠️" or rs["Curve State"]=="Inverted ⚠️") else "✓",
            })
        st.dataframe(pd.DataFrame(carry_rows), use_container_width=True, hide_index=True)
        st.info("**Early warnings:** VIX multi-year lows = complacency; rising VIX = reduce carry. "
                "Inverted yield curve = recession risk. LIBOR-OIS widening = interbank stress.")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 5 · IXL
    # ══════════════════════════════════════════════════════════════════════════
    with tab_ixl:
        st.subheader("IXL Theme Evaluator — Impact × Likelihood")
        st.caption("Score ≥ 70 = Tradeable. Express via directional position in most-affected pair.")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("IXL Score", ixl_res, ixl_label(ixl_res))
            if ixl_theme:
                st.write(f"**Theme:** {ixl_theme}")
                if ixl_res >= 70:   st.success("Meets threshold. Identify affected pair and express directionally.")
                elif ixl_res >= 40: st.warning("Monitor. Re-evaluate as conditions evolve.")
                else:               st.error("Score too low. File for monitoring.")
        with col2:
            z = [[i*l for l in range(1,11)] for i in range(1,11)]
            fig_ixl = go.Figure(go.Heatmap(
                z=z, x=list(range(1,11)), y=list(range(1,11)),
                colorscale=[[0,"#1a1a1a"],[0.4,"#7f6000"],[0.7,"#006f00"],[1,"#00ff00"]],
                zmin=0, zmax=100,
                text=[[f"{i*l}" for l in range(1,11)] for i in range(1,11)],
                texttemplate="%{text}",
            ))
            fig_ixl.add_annotation(x=ixl_like, y=ixl_impact, text="◉",
                                    showarrow=False, font=dict(color="white",size=18))
            fig_ixl.update_layout(xaxis_title="Likelihood →", yaxis_title="Impact ↑",
                                   template="plotly_dark", height=340,
                                   title="IXL Matrix (≥70 = tradeable)")
            st.plotly_chart(fig_ixl, use_container_width=True)

        st.subheader("Example Themes")
        examples = [
            {"Theme":"US tariffs on EU goods",        "Impact":8,"Likelihood":7,"Pair":"EUR/USD","Direction":"Short EUR"},
            {"Theme":"Fed pivot to cuts",              "Impact":9,"Likelihood":6,"Pair":"USD/JPY","Direction":"Short USD"},
            {"Theme":"BOJ normalisation",             "Impact":8,"Likelihood":7,"Pair":"USD/JPY","Direction":"Long JPY"},
            {"Theme":"CBRT cuts rates (TRY)",          "Impact":8,"Likelihood":7,"Pair":"USD/TRY","Direction":"Short TRY"},
            {"Theme":"RBA holds, inflation falling",   "Impact":5,"Likelihood":8,"Pair":"AUD/JPY","Direction":"Reduce carry"},
            {"Theme":"China stimulus",                 "Impact":8,"Likelihood":6,"Pair":"AUD/USD","Direction":"Long AUD"},
        ]
        ex_df = pd.DataFrame(examples)
        ex_df["IXL"] = ex_df["Impact"] * ex_df["Likelihood"]
        ex_df["Status"] = ex_df["IXL"].apply(ixl_label)
        st.dataframe(ex_df, use_container_width=True, hide_index=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 6 · COT
    # ══════════════════════════════════════════════════════════════════════════
    with tab_cot:
        st.subheader("COT — Non-Commercial Positioning")
        st.caption("CFTC weekly report. Positions as of prior Tuesday.")
        cot_rows = [{
            "Currency":     r["code"],  "Name": r["name"],
            "COT Bias":     r["COT Bias"],
            "Net Position": f"{r['COT Net']:,.0f}" if not pd.isna(r["COT Net"]) else "—",
            "Week Change":  f"{r['COT Change']:+,.0f}" if not pd.isna(r["COT Change"]) else "—",
            "Fund. Bias":   r["Overall Bias"],
            "Aligned?":     ("✅" if r["COT Bias"]==r["Overall Bias"]
                              else ("—" if r["COT Bias"]=="—" else "⚠️ Divergence")),
        } for r in results]
        st.dataframe(pd.DataFrame(cot_rows), use_container_width=True, hide_index=True)
        st.info("⚠️ Divergence = COT contradicts fundamentals → wait for alignment or trade with caution. "
                "Extreme net positioning = crowded trade / reversal risk.")
        if cot_df.empty:
            st.warning("COT data unavailable from CFTC. Check connectivity.")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 7 · PRE-MEETING
    # ══════════════════════════════════════════════════════════════════════════
    with tab_premeeting:
        st.subheader("Pre-Meeting Trade Scanner")
        st.caption("Entry window: 1–4 weeks before CB meeting when a clear bias exists.")
        pm_rows = [{
            "Currency":     r["code"],
            "Central Bank": CURRENCY_INFO.get(r["code"],{}).get("cb","—"),
            "In Window?":   "🎯 Yes" if r["Pre-Meeting Window"] else "No",
            "Bias":         r["Overall Bias"],
            "CB Score":     fmt(r.get("CB Taylor Score",np.nan)),
            "CB Action":    r["Expected CB Action"],
            "Checklist ✓":  ("✅ Valid" if r["Pre-Meeting Window"] and r["Overall Bias"]!="Neutral"
                               and r["CPI Direction"]!="unknown" and not pd.isna(r["GDP Growth %"])
                               else ("⚠️ Partial" if r["Pre-Meeting Window"] else "—")),
            "FX 1m %":      fmt(r["FX Chg 1m %"])+"%",
            "COT Aligned":  ("✅" if r["COT Bias"]==r["Overall Bias"]
                              else ("—" if r["COT Bias"]=="—" else "⚠️")),
        } for r in results]
        st.dataframe(pd.DataFrame(pm_rows), use_container_width=True, hide_index=True)
        st.markdown("""
**Pre-Meeting Checklist:**
1. ✅ Clear driving factor compelling a policy change from prior meeting
2. ✅ Opposing currency with weaker / diverging fundamentals
3. ✅ No contradicting themes for target currency
4. ✅ FX 1m trend supports directional hypothesis
5. ✅ COT positioning aligned with bias
6. ✅ Taylor CB Score confirms expected action direction
""")
        st.subheader("Buy the Rumour / Sell the Fact (Section 12)")
        st.markdown("""
| Signal | Watch for |
|--------|-----------|
| **Rumour phase** | FX moves 1–4 weeks before meeting; position early |
| **Fact phase** | Decision already priced → potential reversal at announcement |
| **Consensus check** | ForexFactory → indicator → beat/miss history |
| **Divergence trade** | Strong fundamentals + short-term negative sentiment → discount entry |
""")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 8 · CPI ACTUAL VS EXPECTED
    # ══════════════════════════════════════════════════════════════════════════
    with tab_cpi:
        st.subheader("CPI YoY — Actual vs. Model-Expected, Trailing Months")
        st.caption(
            "“Expected” is generated retrospectively by the same mean-reversion model this app uses "
            "for its forward 12-month CPI forecast (60% persistence of the prior reading + 40% reversion "
            "toward the 2% inflation target). This is a simple heuristic, not an economist consensus "
            "forecast — use it to see where the model's assumption has been running hot or cold vs. "
            "actual prints, not as a forecasting authority."
        )

        months_back = st.slider("Months of history to show", min_value=6, max_value=36, value=24, step=1)
        cpi_codes_available = [c for c in selected_codes if not cpi_history.get(c, pd.Series(dtype=float)).dropna().empty]

        if not cpi_codes_available:
            st.warning("No CPI history available for the selected currencies.")
        else:
            cpi_tab_currencies = st.multiselect(
                "Currencies to display", cpi_codes_available,
                default=cpi_codes_available[:4], key="cpi_tab_select")

            for code in cpi_tab_currencies:
                series = cpi_history.get(code, pd.Series(dtype=float))
                cpi_df = cpi_actual_vs_expected(series, months=months_back)
                if cpi_df.empty:
                    st.info(f"{code}: not enough CPI history to build a trail.")
                    continue

                info = CURRENCY_INFO.get(code, {})
                st.markdown(f"**{code}** — {info.get('name', code)}")

                fig_cpi = go.Figure()
                fig_cpi.add_trace(go.Scatter(
                    x=cpi_df["Date"], y=cpi_df["Actual CPI YoY %"],
                    name="Actual CPI YoY %", mode="lines+markers",
                    line=dict(color="#3498db", width=2)))
                fig_cpi.add_trace(go.Scatter(
                    x=cpi_df["Date"], y=cpi_df["Model-Expected %"],
                    name="Model-Expected %", mode="lines",
                    line=dict(color="#f1c40f", width=2, dash="dash")))
                fig_cpi.add_hline(y=INFLATION_TARGET, line_dash="dot", line_color="#7f8c8d",
                                   annotation_text="2% target", annotation_position="bottom right")
                fig_cpi.update_layout(
                    template="plotly_dark", height=300,
                    yaxis_title="CPI YoY %",
                    margin=dict(t=10, b=10),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                )
                st.plotly_chart(fig_cpi, use_container_width=True)

                # Surprise bar chart -- where the model under/over-shot
                fig_surprise = go.Figure(go.Bar(
                    x=cpi_df["Date"], y=cpi_df["Surprise (Actual − Expected)"],
                    marker_color=["#2ecc71" if v >= 0 else "#e74c3c"
                                  for v in cpi_df["Surprise (Actual − Expected)"]],
                ))
                fig_surprise.update_layout(
                    template="plotly_dark", height=180,
                    yaxis_title="Surprise (pp)",
                    title="Actual − Expected (positive = inflation came in hotter than the model assumed)",
                    margin=dict(t=30, b=10),
                )
                st.plotly_chart(fig_surprise, use_container_width=True)

                with st.expander(f"Show data table — {code}"):
                    show_df = cpi_df.copy()
                    show_df["Date"] = pd.to_datetime(show_df["Date"]).dt.strftime("%Y-%m")
                    st.dataframe(show_df, use_container_width=True, hide_index=True)

                avg_surprise = cpi_df["Surprise (Actual − Expected)"].mean()
                st.caption(f"Average surprise over period: {avg_surprise:+.2f}pp "
                           f"({'inflation has been running hotter than the model expects' if avg_surprise > 0.2 else 'inflation has been running cooler than the model expects' if avg_surprise < -0.2 else 'model has tracked reasonably well'})")
                st.divider()


if __name__ == "__main__":
    main()
