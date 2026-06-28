import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import io
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
# DATA QUALITY TRACKING
# ══════════════════════════════════════════════════════════════════════════════

class DataQualityLog:
    """Collects non-fatal data issues (missing series, implausible values,
    disabled fields) so they can be surfaced to the user instead of silently
    producing blank/misleading numbers."""

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

# Economic weight tier used by the regime classifier to down-weight EM
# outliers when computing cross-currency averages.
# Tier 1 = largest / most systemically important economies.
ECONOMY_WEIGHT: Dict[str, float] = {
    "USD": 1.0, "EUR": 0.9, "JPY": 0.7, "GBP": 0.6,
    "CAD": 0.5, "AUD": 0.5, "CHF": 0.4, "NZD": 0.3,
    "CNY": 0.6,  # large economy but policy-driven; moderate weight
    "TRY": 0.15, "MXN": 0.2, "ZAR": 0.15,
}

# Liquidity tiers for pair filtering
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


# ══════════════════════════════════════════════════════════════════════════════
# FRED SERIES MAP
# ══════════════════════════════════════════════════════════════════════════════
#
# v3.2 NOTE ON PMI:
# v3.0/v3.1 wired USD's "pmi_mfg" to a FRED series ID "ISMMAN". That ID does
# not correspond to a real, publicly-hosted FRED series — ISM's PMI is
# licensed/proprietary data and FRED does not republish it under that or any
# other free series ID. Treating it as "fixed" was itself a bug: it would
# have silently failed (empty series, same as the other currencies) while a
# comment claimed it worked. Rather than guess at a replacement ID we can't
# verify here, PMI has been removed entirely from the data model. It was
# already excluded from SCORE_WEIGHTS in prior versions, so removing it does
# not change any score — it only removes a field that looked live but
# wasn't. If you have a paid ISM/Trading Economics/Investing.com feed, wire
# it in as a new field and give it an explicit weight in SCORE_WEIGHTS.

FRED_SERIES: Dict[str, Dict[str, str]] = {
    "USD": {
        "policy_rate": "DFEDTARU", "cpi_yoy": "CPALTT01USM659N",
        "gdp": "A191RL1Q225SBEA", "trade_balance": "BOPGSTB",
        "unemployment": "UNRATE", "ten_year": "DGS10",
        "m2": "M2SL", "consumer_conf": "UMCSENT", "building_permits": "PERMIT",
    },
    "EUR": {
        "policy_rate": "ECBDFR", "cpi_yoy": "CPHPTT01EZM659N",
        "gdp": "CLVMNACSCAB1GQEA19", "trade_balance": "XTNTVA01EZQ667S",
        "unemployment": "LRHUTTTTEZM156S", "ten_year": "IRLTLT01EZM156N",
        "m2": "MABMM301EZM189S", "consumer_conf": "CSCICP02EZM460S", "building_permits": "",
    },
    "GBP": {
        "policy_rate": "BOERUKM", "cpi_yoy": "CPALTT01GBM659N",
        "gdp": "NGDPRSAXDCGBQ", "trade_balance": "XTNTVA01GBQ667S",
        "unemployment": "LRHUTTTTGBM156S", "ten_year": "IRLTLT01GBM156N",
        "m2": "MABMM301GBM189S", "consumer_conf": "CSCICP02GBM460S", "building_permits": "",
    },
    "JPY": {
        "policy_rate": "IRSTCB01JPM156N", "cpi_yoy": "CPALTT01JPM659N",
        "gdp": "JPNRGDPEXP", "trade_balance": "XTNTVA01JPQ667S",
        "unemployment": "LRHUTTTTJPM156S", "ten_year": "IRLTLT01JPM156N",
        "m2": "MABMM301JPM189S", "consumer_conf": "CSCICP02JPM460S", "building_permits": "",
    },
    "AUD": {
        "policy_rate": "IRSTCB01AUM156N", "cpi_yoy": "CPALTT01AUQ659N",
        "gdp": "AUSGDPRQDSMEI", "trade_balance": "XTNTVA01AUQ667S",
        "unemployment": "LRHUTTTTAUM156S", "ten_year": "IRLTLT01AUM156N",
        "m2": "MABMM301AUM189S", "consumer_conf": "CSCICP02AUM460S", "building_permits": "",
    },
    "CAD": {
        "policy_rate": "IRSTCB01CAM156N", "cpi_yoy": "CPALTT01CAM659N",
        "gdp": "NGDPRSAXDCCAQ", "trade_balance": "XTNTVA01CAQ667S",
        "unemployment": "LRHUTTTTCAM156S", "ten_year": "IRLTLT01CAM156N",
        "m2": "MABMM301CAM189S", "consumer_conf": "CSCICP02CAM460S", "building_permits": "",
    },
    "CHF": {
        "policy_rate": "IRSTCB01CHM156N", "cpi_yoy": "CPALTT01CHM659N",
        "gdp": "NGDPRSAXDCCHQ", "trade_balance": "XTNTVA01CHQ667S",
        "unemployment": "LRHUTTTTCHM156S", "ten_year": "IRLTLT01CHM156N",
        "m2": "", "consumer_conf": "", "building_permits": "",
    },
    "NZD": {
        "policy_rate": "IRSTCB01NZM156N", "cpi_yoy": "CPALTT01NZQ659N",
        "gdp": "NGDPRSAXDCNZQ", "trade_balance": "XTNTVA01NZQ667S",
        "unemployment": "LRHUTTTTNZM156S", "ten_year": "IRLTLT01NZM156N",
        "m2": "", "consumer_conf": "", "building_permits": "",
    },
    "CNY": {
        "policy_rate": "INTDSRCNM193N", "cpi_yoy": "CPALTT01CNM659N",
        "gdp": "NGDPRSAXDCCNQ", "trade_balance": "XTNTVA01CNQ667S",
        "unemployment": "LRUN64TTCNQ156S", "ten_year": "IRLTLT01CNM156N",
        "m2": "", "consumer_conf": "", "building_permits": "",
    },
    "TRY": {
        "policy_rate": "INTDSRTRM193N", "cpi_yoy": "CPALTT01TRM659N",
        "gdp": "NGDPRSAXDCTRQ", "trade_balance": "XTNTVA01TRQ667S",
        "unemployment": "LRHUTTTTTRM156S", "ten_year": "IRLTLT01TRM156N",
        "m2": "", "consumer_conf": "", "building_permits": "",
    },
    "MXN": {
        "policy_rate": "INTDSRMXM193N", "cpi_yoy": "CPALTT01MXM659N",
        "gdp": "NGDPRSAXDCMXQ", "trade_balance": "XTNTVA01MXQ667S",
        "unemployment": "LRHUTTTTMXM156S", "ten_year": "IRLTLT01MXM156N",
        "m2": "", "consumer_conf": "", "building_permits": "",
    },
    "ZAR": {
        "policy_rate": "INTDSRZAM193N", "cpi_yoy": "ZAFCPIALLMINMEI",  # broken; disabled below
        "gdp": "NGDPRSAXDCZAQ", "trade_balance": "XTNTVA01ZAQ667S",
        "unemployment": "LRHUTTTTZAM156S", "ten_year": "IRLTLT01ZAM156N",
        "m2": "", "consumer_conf": "", "building_permits": "",
    },
}

BROKEN_CPI_SERIES: Dict[str, str] = {
    "ZAR": ("ZAFCPIALLMINMEI is an OECD CPI INDEX LEVEL (Units: 'Index 2015=100'), "
            "not a YoY growth rate. No verified replacement found on FRED. "
            "cpi_yoy, real rate, and CB Taylor score for ZAR disabled until confirmed."),
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

# CB reaction-function score (cb_taylor) and "inflation level vs target" score
# are now two genuinely distinct inputs (see analyze_currency) instead of the
# same number counted under two labels. Weights re-balanced accordingly.
SCORE_WEIGHTS = {
    "gdp": 0.22, "inflation": 0.13, "cb_taylor": 0.30,
    "trade": 0.13, "m2": 0.10, "conf": 0.12,
}

# Annualised vol lookback in trading days for carry adjustment.
VOL_LOOKBACK_DAYS = 63   # ~3 months
VOL_ANNUALISE     = 252  # trading days per year

# Network/IO tuning
MAX_FETCH_WORKERS = 8
HTTP_TIMEOUT_SECONDS = 12


# ══════════════════════════════════════════════════════════════════════════════
# FRED FETCH
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fred_series(series_id: str) -> pd.Series:
    if not series_id:
        return pd.Series(dtype=float)
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    for _attempt in range(2):  # one retry — FRED's CSV endpoint occasionally
                                # times out or 5xx's transiently under load.
        try:
            df = pd.read_csv(url, storage_options={"timeout": HTTP_TIMEOUT_SECONDS})
            if df.empty or df.shape[1] < 2:
                return pd.Series(dtype=float)
            dc, vc = df.columns[0], df.columns[1]
            df[dc] = pd.to_datetime(df[dc], errors="coerce")
            df[vc] = pd.to_numeric(df[vc], errors="coerce")
            return df.dropna(subset=[dc]).set_index(dc)[vc].dropna()
        except Exception:
            continue
    return pd.Series(dtype=float)


# DECIMAL_ENCODED_SERIES: add series IDs here (after verifying on FRED) if
# a specific series ships as a 0–1 decimal fraction rather than percent.
# Do NOT re-introduce a magnitude heuristic — that caused real bugs.
DECIMAL_ENCODED_SERIES: set = set()

CPI_YOY_PLAUSIBILITY_LIMIT = 150.0


def fetch_currency_macro(code: str, dq_log: "DataQualityLog") -> Dict[str, pd.Series]:
    out: Dict[str, pd.Series] = {}
    rate_fields = {"policy_rate", "ten_year", "cpi_yoy"}
    for label, sid in FRED_SERIES.get(code, {}).items():
        if label == "cpi_yoy" and code in BROKEN_CPI_SERIES:
            dq_log.warn(code, f"cpi_yoy disabled — {BROKEN_CPI_SERIES[code]}")
            out[label] = pd.Series(dtype=float)
            continue

        if not sid:
            out[label] = pd.Series(dtype=float)
            continue

        s = fetch_fred_series(sid)
        if s.empty:
            dq_log.warn(code, f"FRED '{sid}' ({label}) unavailable")

        if label in rate_fields and sid in DECIMAL_ENCODED_SERIES:
            s = s * 100
            dq_log.warn(code, f"{label} ('{sid}'): converted from decimal to percent.")

        if label == "cpi_yoy" and not s.empty:
            non_null = s.dropna()
            if len(non_null) and abs(float(non_null.iloc[-1])) > CPI_YOY_PLAUSIBILITY_LIMIT:
                dq_log.warn(
                    code,
                    f"cpi_yoy ('{sid}') latest value {float(non_null.iloc[-1]):.1f} is implausible "
                    f"for a YoY %% rate — likely an index-level series. Verify FRED 'Units:' field."
                )

        out[label] = s
    return out


def fetch_all_macro(codes: List[str], dq_log: "DataQualityLog") -> Dict[str, Dict[str, pd.Series]]:
    """
    Fetch macro data for all selected currencies concurrently.

    The original version fetched currencies sequentially: with ~9 FRED series
    per currency, 7 default currencies meant ~60+ sequential HTTP round trips
    on every cold cache. Each fetch is I/O-bound (network wait, not CPU), so
    a thread pool gives a large wall-clock speedup with no change in results.
    st.cache_data's underlying cache is safe for concurrent access, so the
    *only* change here is parallel scheduling — caching behaviour is
    unaffected (this still uses fetch_currency_macro -> fetch_fred_series,
    which is itself cached as before).
    """
    out: Dict[str, Dict[str, pd.Series]] = {}
    with ThreadPoolExecutor(max_workers=MAX_FETCH_WORKERS) as ex:
        futures = {ex.submit(fetch_currency_macro, code, dq_log): code for code in codes}
        for fut in as_completed(futures):
            code = futures[fut]
            try:
                out[code] = fut.result()
            except Exception as e:
                dq_log.warn(code, f"Macro fetch failed unexpectedly: {e}")
                out[code] = {}
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fx_history(ticker: str, period: str = "2y") -> pd.Series:
    if not ticker:
        return pd.Series(dtype=float)
    try:
        data = yf.download(ticker, period=period, interval="1d",
                            auto_adjust=True, progress=False, timeout=HTTP_TIMEOUT_SECONDS)
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


def fetch_all_fx(codes: List[str]) -> Dict[str, pd.Series]:
    """Concurrently fetch FX history for all selected currencies' tickers."""
    tickers = {code: FX_TICKER[code] for code in codes if code in FX_TICKER}
    out: Dict[str, pd.Series] = {c: pd.Series(dtype=float) for c in codes}
    if not tickers:
        return out
    with ThreadPoolExecutor(max_workers=MAX_FETCH_WORKERS) as ex:
        futures = {ex.submit(fetch_fx_history, t): code for code, t in tickers.items()}
        for fut in as_completed(futures):
            code = futures[fut]
            try:
                out[code] = fut.result()
            except Exception:
                out[code] = pd.Series(dtype=float)
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_cot_data() -> pd.DataFrame:
    year = datetime.now().year
    for y in [year, year - 1]:
        for url in [
            f"https://www.cftc.gov/files/dea/history/fut_fin_xls_{y}.zip",
            f"https://www.cftc.gov/files/dea/history/dea_fut_xls_{y}.zip",
        ]:
            try:
                df = pd.read_csv(url, compression="zip", low_memory=False,
                                  storage_options={"timeout": HTTP_TIMEOUT_SECONDS})
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
# REALIZED VOLATILITY FOR CARRY ADJUSTMENT
# ══════════════════════════════════════════════════════════════════════════════
#
# v3.2: realized vol is now computed FROM the same 2y FX history that
# fetch_fx_history() already retrieves, instead of issuing a second,
# independent yf.download() call per ticker. The old code fetched every FX
# ticker twice per run (once in fetch_fx_history for the price chart, once
# in fetch_realized_vol for the carry tab) — same data, same cost, double
# the network calls and double the chance of a transient failure on one of
# the two going empty while the other succeeds. Passing the series in
# directly removes that redundancy and keeps the two views consistent.

def realized_vol_from_series(close: pd.Series, lookback: int = VOL_LOOKBACK_DAYS) -> float:
    """
    Annualised realized volatility from daily log-returns over `lookback` days.
    Used in vol-adjusted carry = real_rate_differential / annualised_vol.
    Returns NaN if insufficient data.
    """
    if close is None or close.empty:
        return np.nan
    c = close.dropna()
    if len(c) < lookback + 5:
        return np.nan
    log_ret = np.log(c / c.shift(1)).dropna()
    recent = log_ret.iloc[-lookback:]
    return float(recent.std() * np.sqrt(VOL_ANNUALISE) * 100)  # in % annualised


def pair_realized_vol(code_long: str, code_short: str, fx_history: Dict[str, pd.Series]) -> float:
    """
    Approximate annualised realized vol for a cross pair.

    For pairs involving USD (one leg IS USD), use the non-USD currency's own
    vol vs USD directly. For crosses (neither leg is USD), approximate:
        vol(A/B) ≈ sqrt(vol(A/USD)^2 + vol(B/USD)^2)
    assuming ~zero correlation between the two legs. Most currencies are
    *positively* correlated against USD, and for two positively-correlated
    legs the true cross vol is LOWER than this independent-sum estimate
    (sqrt(va^2+vb^2-2*rho*va*vb) < sqrt(va^2+vb^2) when rho>0). So this is a
    conservative over-estimate that errs on the side of penalising crosses
    more, not less — confirmed numerically, not just asserted.
    """
    def _vol(c: str) -> float:
        if c == "USD":
            return 0.0
        return realized_vol_from_series(fx_history.get(c, pd.Series(dtype=float)))

    va = _vol(code_long)
    vb = _vol(code_short)

    if code_long == "USD":
        return vb
    if code_short == "USD":
        return va
    if pd.isna(va) or pd.isna(vb):
        return np.nan
    return float(np.sqrt(va**2 + vb**2))


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

def weighted_trimmed_mean(values: List[float], weights: List[float],
                           trim_pct: float = 0.1) -> float:
    """
    Weighted mean after trimming the top and bottom `trim_pct` fraction
    of values (by value, not by weight). Used by classify_regime() to
    prevent EM outliers (e.g. TRY CPI of 32%) from dominating the
    cross-currency average that drives the regime label.
    """
    if not values:
        return np.nan
    pairs = sorted(zip(values, weights), key=lambda x: x[0])
    n = len(pairs)
    cut = max(1, int(n * trim_pct))
    trimmed = pairs[cut:-cut] if n > 2 * cut else pairs
    vals_t, wts_t = zip(*trimmed)
    total_w = sum(wts_t)
    if total_w == 0:
        return np.nan
    return sum(v * w for v, w in zip(vals_t, wts_t)) / total_w


# ══════════════════════════════════════════════════════════════════════════════
# TAYLOR RULE CB PREDICTOR
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
    Score > 0 -> hawkish (Bullish for currency). Score < 0 -> dovish (Bearish).
    This is the central-bank REACTION FUNCTION score: it blends the level
    and momentum of inflation, the labour market, growth, and recent rate
    moves into one estimate of which way the central bank leans next.
    """
    score = 0.0
    factors = 0

    if not pd.isna(cpi_now):
        infl_gap   = cpi_now - INFLATION_TARGET
        infl_score = np.tanh(infl_gap / 2.0)
        score += infl_score * 0.35
        factors += 0.35

    if not pd.isna(cpi_now) and not pd.isna(cpi_3m_ago):
        infl_chg  = cpi_now - cpi_3m_ago
        mom_score = np.tanh(infl_chg / 1.2)
        score += mom_score * 0.20
        factors += 0.20

    if not pd.isna(unemp_now) and not pd.isna(unemp_prior):
        unemp_chg   = unemp_now - unemp_prior
        unemp_score = -np.tanh(unemp_chg / 0.6)
        score += unemp_score * 0.20
        factors += 0.20

    if not pd.isna(gdp_growth):
        gdp_score_t = np.tanh(gdp_growth / 2.0)
        score += gdp_score_t * 0.15
        factors += 0.15

    if not pd.isna(rate_now) and not pd.isna(rate_6m_ago):
        rate_chg = rate_now - rate_6m_ago
        rate_mom = np.tanh(rate_chg / 0.75)
        score += rate_mom * 0.10
        factors += 0.10

    score = float(np.clip(score / factors if factors > 0 else 0.0, -1, 1))

    if score >= 0.4:    action, bias = "Strong Hike Expected", "Bullish"
    elif score >= 0.15: action, bias = "Hike Expected",        "Bullish"
    elif score <= -0.4: action, bias = "Strong Cut Expected",  "Bearish"
    elif score <= -0.15:action, bias = "Cut Expected",         "Bearish"
    else:               action, bias = "Hold Expected",        "Neutral"

    return score, action, bias


def inflation_level_score(cpi_now: float, code: str) -> float:
    """
    v3.2: a genuinely separate "inflation level vs target" signal, distinct
    from the CB Taylor score above.

    Prior versions set `Inflation Score` directly equal to the CB Taylor
    score and then summed both into the composite with separate weights —
    i.e. the same number was counted twice under two labels, while the
    composite's "cb_taylor" weight was zeroed out at the last second to
    paper over the double-count. That made the composite opaque (its named
    "inflation" component wasn't really about inflation; it was the whole
    Taylor blend again) and meant the displayed component-score radar chart
    was showing two identical bars without saying so.

    This function instead scores ONLY where current inflation sits relative
    to a sensible target, with the opposite-sign convention used elsewhere
    (positive = currency-supportive). For DM currencies, far-above-target
    inflation is currency-negative in real terms (purchasing power erosion)
    even though it may also be tightening-supportive in the Taylor score —
    these two effects are related but not identical, which is exactly why
    they deserve separate, smaller weights rather than one weight twice.
    For HIGH_INFLATION_CURRENCIES, target is relaxed since 2% is not a
    realistic anchor for these economies (same reasoning as
    forward_cpi_estimate's EM anchor below).
    """
    if pd.isna(cpi_now):
        return np.nan
    target = 10.0 if code in HIGH_INFLATION_CURRENCIES else INFLATION_TARGET
    gap = cpi_now - target
    # Moderate above-target inflation is mildly negative (erodes real
    # returns); deflation/very-low inflation is also mildly negative
    # (growth-risk signal); scored as a downward-opening curve around target.
    return float(np.clip(-abs(gap) / 6.0 + 0.15, -1, 1))


# ══════════════════════════════════════════════════════════════════════════════
# MACRO REGIME CLASSIFIER  (weighted trimmed mean, G10 vs EM split)
# ══════════════════════════════════════════════════════════════════════════════

# Currencies treated as "core" (G10-like) for regime classification.
# EM currencies are weighted down via ECONOMY_WEIGHT and trimmed-mean logic,
# so a TRY CPI of 32% no longer forces an "Inflation Shock" label when the
# G10 bloc is sitting at 2-3%.
G10_CODES = {"USD","EUR","GBP","JPY","AUD","CAD","CHF","NZD"}

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
    Classify macro regime using economy-weighted trimmed means.

    - Cross-currency averages computed with weighted_trimmed_mean() using
      ECONOMY_WEIGHT, trimming top/bottom 10% of values. This prevents a
      single EM outlier (e.g. TRY CPI ~32%) from dragging the aggregate into
      "Inflation Shock" when G10 inflation is benign.
    - G10-only sub-aggregate computed separately for the hawkish/dovish CB
      bias signal, since EM CBs often move for idiosyncratic reasons
      (political pressure, FX defence) that don't reflect G10 policy cycles.
    """
    if not results:
        return "Neutral", "Insufficient data."

    def _collect(field, codes=None):
        vals, wts = [], []
        for r in results:
            if codes and r["code"] not in codes:
                continue
            v = r.get(field, np.nan)
            if not pd.isna(v):
                vals.append(float(v))
                wts.append(ECONOMY_WEIGHT.get(r["code"], 0.3))
        return vals, wts

    cpi_vals,  cpi_wts  = _collect("CPI YoY %")
    gdp_vals,  gdp_wts  = _collect("GDP Growth %")
    cb_vals_g10, cb_wts_g10 = _collect("CB Taylor Score", codes=G10_CODES)

    avg_cpi = weighted_trimmed_mean(cpi_vals, cpi_wts)
    avg_gdp = weighted_trimmed_mean(gdp_vals, gdp_wts)
    total_cbw = sum(cb_wts_g10)
    avg_cb = (sum(v * w for v, w in zip(cb_vals_g10, cb_wts_g10)) / total_cbw
              if total_cbw > 0 else np.nan)

    n = len(results)
    gdp_contracting = sum(1 for r in results if "Contraction" in r.get("GDP Trend", ""))

    high_inflation     = not pd.isna(avg_cpi) and avg_cpi > 3.5
    low_growth         = not pd.isna(avg_gdp) and avg_gdp < 1.0
    hawkish_bias       = not pd.isna(avg_cb)  and avg_cb  > 0.15
    dovish_bias        = not pd.isna(avg_cb)  and avg_cb  < -0.15
    broad_contraction  = gdp_contracting / n > 0.5 if n > 0 else False

    cpi_str = f"{avg_cpi:.1f}%" if not pd.isna(avg_cpi) else "n/a"
    gdp_str = f"{avg_gdp:.1f}%" if not pd.isna(avg_gdp) else "n/a"

    if high_inflation and low_growth:
        return ("Stagflation",
                f"Wtd-trimmed avg CPI {cpi_str}, GDP {gdp_str} (G10/EM weighted).")
    if high_inflation and hawkish_bias:
        return ("Inflation Shock",
                f"Wtd-trimmed avg CPI {cpi_str}; G10 CBs signalling hikes.")
    if broad_contraction and dovish_bias:
        return ("Growth Shock",
                f"{gdp_contracting}/{n} currencies contracting; G10 CBs easing.")
    if not high_inflation and not broad_contraction and dovish_bias:
        return ("Carry Environment",
                "Low inflation, stable growth, dovish G10 CBs — ideal carry conditions.")
    if not high_inflation and not broad_contraction and not hawkish_bias:
        return ("Risk-On",
                "Benign macro backdrop — risk appetite supported.")
    if broad_contraction or (not pd.isna(avg_gdp) and avg_gdp < 0):
        return ("Risk-Off",
                "Broad growth deterioration signals risk-off positioning.")
    return "Neutral", "Mixed signals — no dominant regime."


# ══════════════════════════════════════════════════════════════════════════════
# FORWARD CPI MODEL  (EM-aware mean reversion)
# ══════════════════════════════════════════════════════════════════════════════

def forward_cpi_estimate(code: str, cpi_now: float,
                          cpi_history: pd.Series) -> float:
    """
    12-month forward CPI estimate.

    For HIGH_INFLATION_CURRENCIES (TRY, ZAR, MXN, CNY): anchor toward the
    currency's own trailing 5-year median CPI (if >= 12 months of history
    available), capped so the forecast can't deviate > +-5pp from current.
    A flat 2% DM anchor would forecast TRY at 32% reverting to ~20% in a
    year, which has no basis — TRY has not been near 2% in the modern era.
    For standard currencies: 60/40 blend toward 2%, capped +-3pp.
    If insufficient history for the EM anchor, fall back to 80% persistence
    (less pull toward any anchor) rather than the DM 2% anchor.
    """
    if pd.isna(cpi_now):
        return np.nan

    if code in HIGH_INFLATION_CURRENCIES:
        hist = cpi_history.dropna()
        em_anchor = float(hist.iloc[-60:].median()) if len(hist) >= 12 else np.nan
        if not pd.isna(em_anchor):
            raw = 0.65 * cpi_now + 0.35 * em_anchor
            return float(np.clip(raw, cpi_now - 5.0, cpi_now + 5.0))
        else:
            return float(np.clip(0.80 * cpi_now, cpi_now - 5.0, cpi_now + 5.0))
    else:
        raw = 0.60 * cpi_now + 0.40 * INFLATION_TARGET
        return float(np.clip(raw, cpi_now - 3.0, cpi_now + 3.0))


def cpi_actual_vs_expected(code: str, cpi_series: pd.Series,
                            months: int = 24) -> pd.DataFrame:
    """
    Retrospective actual-vs-model CPI table. One-step-ahead: for each month,
    "expected" = what the model would have forecast given the prior month's
    actual reading, using only history available up to that point (no
    lookahead bias).
    """
    s = cpi_series.dropna()
    if s.empty:
        return pd.DataFrame(columns=["Date","Actual CPI YoY %",
                                      "Model-Expected %","Surprise (Actual − Expected)"])

    s = s.iloc[-(months + 1):]
    rows = []
    for i in range(1, len(s)):
        date     = s.index[i]
        actual   = float(s.iloc[i])
        prior    = float(s.iloc[i - 1])
        hist_up_to_prior = s.iloc[:i]
        expected = forward_cpi_estimate(code, prior, hist_up_to_prior)
        if pd.isna(expected):
            continue
        rows.append({
            "Date": date,
            "Actual CPI YoY %":            round(actual,   2),
            "Model-Expected %":            round(expected, 2),
            "Surprise (Actual − Expected)": round(actual - expected, 2),
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# FORWARD GDP MODEL  (v3.2 — same EM-aware fix applied to GDP)
# ══════════════════════════════════════════════════════════════════════════════

# Countries whose structural trend growth is far from a generic "2%" anchor.
# Without this, the GDP forward estimate pulled every currency toward 2% --
# the same bug class that forward_cpi_estimate already fixed for inflation,
# just left unfixed for GDP. Japan and Switzerland trend well under 2%;
# the high-inflation/EM bloc and a couple of commodity economies often run
# above it. These are rough structural anchors, not official trend-growth
# estimates -- intended to be less wrong than a flat 2% for every country,
# not to be precise.
GDP_TREND_ANCHOR: Dict[str, float] = {
    "USD": 2.0, "EUR": 1.2, "GBP": 1.5, "JPY": 0.7,
    "AUD": 2.3, "CAD": 1.8, "CHF": 1.5, "NZD": 2.2,
    "CNY": 4.5, "TRY": 3.5, "MXN": 2.0, "ZAR": 1.0,
}


def forward_gdp_estimate(code: str, gdp_growth_now: float,
                          gdp_growth_prior: float) -> float:
    """
    12-month forward GDP growth estimate, anchored to each currency's own
    structural trend rather than a flat 2% for every economy.
    """
    if pd.isna(gdp_growth_now):
        return np.nan
    anchor = GDP_TREND_ANCHOR.get(code, INFLATION_TARGET)
    if not pd.isna(gdp_growth_prior):
        momentum = gdp_growth_now + 0.5 * (gdp_growth_now - gdp_growth_prior)
    else:
        momentum = gdp_growth_now
    raw = 0.7 * momentum + 0.3 * anchor
    return float(np.clip(raw, gdp_growth_now - 3.0, gdp_growth_now + 3.0))


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
    m2             = macro.get("m2",              pd.Series(dtype=float))
    consumer_conf  = macro.get("consumer_conf",   pd.Series(dtype=float))
    building_perms = macro.get("building_permits",pd.Series(dtype=float))

    rate_now    = latest(policy_rate)
    cpi_now     = latest(cpi_yoy)
    gdp_raw     = latest(gdp)
    trade_now   = latest(trade_bal)
    unemp_now   = latest(unemployment)
    ten_y_now   = latest(ten_year)
    conf_now    = latest(consumer_conf)
    permits_now = latest(building_perms)

    for val, label in [(rate_now,"Policy rate"),(cpi_now,"CPI YoY"),(gdp_raw,"GDP")]:
        if pd.isna(val):
            dq_log.warn(code, f"{label} unavailable from FRED")

    real_rate_10y    = ten_y_now - cpi_now if not pd.isna(ten_y_now) and not pd.isna(cpi_now) else np.nan
    real_rate_policy = rate_now  - cpi_now if not pd.isna(rate_now)  and not pd.isna(cpi_now) else np.nan
    real_rate        = real_rate_10y if not pd.isna(real_rate_10y) else real_rate_policy

    curve_slope = ten_y_now - rate_now if not pd.isna(ten_y_now) and not pd.isna(rate_now) else np.nan
    curve_state = ("Inverted ⚠️" if not pd.isna(curve_slope) and curve_slope < 0
                   else "Normal ✓" if not pd.isna(curve_slope) else "unknown")

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

    gdp_z = zscore_current(gdp_series_for_z, window=20)
    gdp_score = float(np.clip(gdp_z / 2.0, -1, 1)) if not pd.isna(gdp_z) else (
        float(np.clip(gdp_growth_now / 4.0, -1, 1)) if not pd.isna(gdp_growth_now) else np.nan
    )

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
    # v3.2: this is now a genuinely distinct signal from cb_taylor_score —
    # see inflation_level_score() docstring for why the two were merged
    # under one weight before and are now separated.
    infl_level_score = inflation_level_score(cpi_now, code)

    trade_dir   = trend_direction(trade_bal, lookback=3)
    trade_score = np.nan
    if not pd.isna(trade_now):
        base = 0.5 if trade_now > 0 else -0.5
        mom  = 0.3 if trade_dir == "rising" else (-0.3 if trade_dir == "falling" else 0.0)
        trade_score = float(np.clip(base + mom, -1, 1))
    trade_z = zscore_current(trade_bal, window=20)
    trade_unit = "$M (raw)" if code == "USD" else "OECD index (own units)"

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

    conf_dir   = trend_direction(consumer_conf, lookback=3)
    conf_score = np.nan
    if not pd.isna(conf_now):
        conf_score = 0.3 if conf_dir == "rising" else (-0.3 if conf_dir == "falling" else 0.0)

    component_scores = {
        "gdp": gdp_score, "inflation": infl_level_score,
        "cb_taylor": cb_taylor_score,
        "trade": trade_score, "m2": m2_score, "conf": conf_score,
    }
    weighted_sum = sum(SCORE_WEIGHTS[k] * v for k, v in component_scores.items() if not pd.isna(v))
    weight_used  = sum(SCORE_WEIGHTS[k] for k, v in component_scores.items() if not pd.isna(v))
    composite    = float(weighted_sum / weight_used) if weight_used > 0 else np.nan

    if pd.isna(composite):        overall_bias = "Neutral"
    elif composite >= 0.25:       overall_bias = "Bullish"
    elif composite <= -0.25:      overall_bias = "Bearish"
    else:                         overall_bias = "Neutral"

    # Forward estimates
    cpi_exp12m  = forward_cpi_estimate(code, cpi_now, cpi_yoy)

    rate_step_map = {"Strong Hike Expected": +0.75, "Hike Expected": +0.25,
                      "Hold Expected": 0.0, "Cut Expected": -0.25, "Strong Cut Expected": -0.75}
    rate_exp12m = rate_now + rate_step_map.get(expected_action, 0.0) if not pd.isna(rate_now) else np.nan

    gdp_exp12m = forward_gdp_estimate(code, gdp_growth_now, gdp_growth_prior)

    pre_meeting_window = is_in_cb_window(code)
    pre_meeting_score  = 2 if (pre_meeting_window and overall_bias != "Neutral") else (1 if pre_meeting_window else 0)

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

    # v3.2: derived from the already-fetched fx_series instead of a second
    # independent yf.download() call for the same ticker (see note above
    # realized_vol_from_series).
    fx_realized_vol = realized_vol_from_series(fx_series)

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
        "M2 Growth YoY %": latest(yoy_growth(m2,12)),
        "Consumer Conf":       conf_now,   "Building Permits": permits_now,
        "GDP Trend":           gdp_trend,  "CPI Direction": cpi_direction,
        "Rate Direction":      rate_direction,
        "Rate Δ 12m (bps)":    rate_change_12m*100 if not pd.isna(rate_change_12m) else np.nan,
        "Trade Bal Direction": trade_dir,  "M2 Trend": m2_trend, "Conf Direction": conf_dir,
        "CB Taylor Score":     cb_taylor_score,
        "Expected CB Action":  expected_action, "Inflation Bias": inflation_bias,
        "CPI Expected (12m)":  cpi_exp12m, "Rate Expected (12m)": rate_exp12m,
        "GDP Expected (12m)":  gdp_exp12m,
        "GDP Score":           gdp_score,  "Inflation Score": infl_level_score,
        "Trade Score":         trade_score,"M2 Score": m2_score, "Conf Score": conf_score,
        "Composite Score":     composite,  "Overall Bias": overall_bias,
        "COT Net": np.nan, "COT Change": np.nan, "COT Bias": "—",
        "Pre-Meeting Window":  pre_meeting_window, "Pre-Meeting Score": pre_meeting_score,
        "FX Spot":             fx_now, "FX Chg 1m %": fx_chg_1m,
        "FX Chg 3m %":         fx_chg_3m, "FX Chg 12m %": fx_chg_12m,
        "FX Realized Vol %":   fx_realized_vol,
    }


# ══════════════════════════════════════════════════════════════════════════════
# PAIR SCORING ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def build_pair_matrix(results: List[Dict], max_tier: int = 3) -> pd.DataFrame:
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


def vol_adjusted_carry_table(results: List[Dict], fx_history: Dict[str, pd.Series]) -> pd.DataFrame:
    """
    Build carry analysis table with vol-adjusted carry (Sharpe-like ratio).

    Vol-adjusted carry = real_rate_differential / pair_annualised_vol (%).
    A ratio > 0.20 is considered attractive; < 0.10 is marginal.

    Raw carry alone overstates EM carry attractiveness vs G10 carry because
    it ignores that FX vol can erase the carry in a single bad session.
    Vol is computed from 63-day realized volatility of the FX rate vs USD
    for each leg (reusing already-fetched history), then combined as
    sqrt(va^2 + vb^2) for crosses.
    """
    result_map = {r["code"]: r for r in results}
    rows = []
    for lc, sc in CLASSIC_CARRY_PAIRS:
        if lc not in result_map or sc not in result_map:
            continue
        rl, rs = result_map[lc], result_map[sc]
        rrd = (rl["Real Rate %"] - rs["Real Rate %"]
               if not pd.isna(rl["Real Rate %"]) and not pd.isna(rs["Real Rate %"]) else np.nan)

        pair_vol = pair_realized_vol(lc, sc, fx_history)
        if not pd.isna(rrd) and not pd.isna(pair_vol) and pair_vol > 0:
            adj_carry = rrd / pair_vol
        else:
            adj_carry = np.nan

        if not pd.isna(adj_carry):
            if adj_carry >= 0.20:   carry_rating = "✅ Attractive"
            elif adj_carry >= 0.10: carry_rating = "⚠️ Marginal"
            elif adj_carry >= 0:    carry_rating = "❌ Unattractive"
            else:                   carry_rating = "🔴 Negative"
        else:
            carry_rating = "—"

        rows.append({
            "Pair":                f"{lc}/{sc}",
            "Long":                lc,
            "Short":               sc,
            "Long Real Rate %":    fmt(rl["Real Rate %"]),
            "Short Real Rate %":   fmt(rs["Real Rate %"]),
            "Real Yield Diff %":   fmt(rrd),
            "Pair Vol % (Ann.)":   fmt(pair_vol) if not pd.isna(pair_vol) else "—",
            "Vol-Adj Carry":       f"{adj_carry:.3f}" if not pd.isna(adj_carry) else "—",
            "Carry Rating":        carry_rating,
            "Long Policy Rate %":  fmt(rl["Policy Rate %"]),
            "Short Policy Rate %": fmt(rs["Policy Rate %"]),
            "Long Bias":           rl["Overall Bias"],
            "Short Bias":          rs["Overall Bias"],
            "Curve Warning":       ("⚠️" if (rl["Curve State"]=="Inverted ⚠️"
                                             or rs["Curve State"]=="Inverted ⚠️") else "✓"),
        })
    return pd.DataFrame(rows)


def fx_returns_correlation_matrix(fx_history: Dict[str, pd.Series], codes: List[str],
                                   window: int = 90) -> pd.DataFrame:
    """
    New in v3.2. Pearson correlation of daily FX returns (vs USD) over the
    trailing `window` days for each selected currency.

    Why this matters: the Pairs tab ranks trades by fundamental divergence,
    but two "different" top pairs can be near-duplicate risk if their legs
    are highly correlated (e.g. AUD/USD and NZD/USD often move together).
    This view helps size a basket of trades without unknowingly doubling up
    on the same underlying risk factor.
    """
    series = {}
    for c in codes:
        s = fx_history.get(c, pd.Series(dtype=float)).dropna()
        if len(s) > window:
            series[c] = np.log(s / s.shift(1)).dropna().iloc[-window:]
    if len(series) < 2:
        return pd.DataFrame()
    df = pd.DataFrame(series).dropna()
    if df.empty or df.shape[0] < 10:
        return pd.DataFrame()
    return df.corr()


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
                                auto_adjust=True, progress=False, timeout=HTTP_TIMEOUT_SECONDS)
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
    st.title("💱 Forex Fundamental Scanner  v3.2")
    st.caption(
        "Taylor Rule CB Prediction · GDP Z-Score · Vol-Adjusted Carry · "
        "Weighted-Trimmed Regime Classifier · EM-Aware CPI & GDP Forward Models"
    )

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
        ixl_affected = st.multiselect("Affected currencies (optional)", all_codes,
                                       help="Pick the currencies this theme would move. "
                                            "Used in the IXL tab to cross-check against "
                                            "current fundamentals.")

    if not selected_codes:
        st.warning("Select at least one currency.")
        return

    dq_log = DataQualityLog()
    with st.spinner("Fetching macro & FX data…"):
        # v3.2: macro and FX data are now fetched concurrently across all
        # selected currencies (see fetch_all_macro / fetch_all_fx) instead of
        # one sequential currency-by-currency loop. Same cached functions
        # underneath, same results — just scheduled in parallel so a cold
        # cache doesn't mean waiting on ~10 sequential HTTP calls per
        # currency one at a time.
        macro_by_code = fetch_all_macro(selected_codes, dq_log)
        fx_by_code    = fetch_all_fx(selected_codes)
        cot_df        = fetch_cot_data()

        results = []
        for code in selected_codes:
            macro   = macro_by_code.get(code, {})
            fx_hist = fx_by_code.get(code, pd.Series(dtype=float))
            r       = analyze_currency(code, macro, fx_hist, dq_log)
            cot     = get_cot_position(code, cot_df)
            r["COT Net"] = cot["net"]; r["COT Change"] = cot["change"]; r["COT Bias"] = cot["bias"]
            results.append(r)

    cpi_history = {code: macro_by_code.get(code, {}).get("cpi_yoy", pd.Series(dtype=float))
                   for code in selected_codes}

    ranked_df  = rank_currencies(results)
    pair_df    = build_pair_matrix(results, max_tier=max_tier)
    rr_diff_df = real_rate_differential_table(results)
    carry_df   = vol_adjusted_carry_table(results, fx_by_code)
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
            "GDP Z-Score = GDP growth vs own 5-year history. "
            "CB Score = Taylor Rule estimate (−1 dovish … +1 hawkish). "
            "Real Rate uses 10Y yield where available. "
            "Trade Balance shows z-score vs each country's own history — raw levels "
            "are not cross-country comparable (USD = $M, others = OECD index)."
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
                    st.write(f"Policy Rate:       {fmt(r['Policy Rate %'])}%")
                    st.write(f"CPI YoY:           {fmt(r['CPI YoY %'])}% ({r['CPI Direction']})")
                    st.write(f"Real Rate (10Y):   {fmt(r['Real Rate % (10Y)'])}%")
                    st.write(f"Real Rate (Policy):{fmt(r['Real Rate % (Policy)'])}%")
                    st.write(f"GDP Growth:        {fmt(r['GDP Growth %'])}%  Z={fmt(r.get('GDP Z-Score',np.nan))}")
                    st.write(f"Unemployment:      {fmt(r['Unemployment %'])}%")
                    st.write(f"10Y Yield:         {fmt(r['10Y Yield %'])}%")
                    st.write(f"Yield Curve:       {r['Curve State']}")
                    st.write(f"FX Realized Vol:   {fmt(r.get('FX Realized Vol %', np.nan))}% (ann.)")
                with c2:
                    st.markdown("**Leading Indicators**")
                    st.write(f"Trade Balance:  {fmt(r['Trade Balance'],2)} {r.get('Trade Unit','')}  "
                             f"(z={fmt(r.get('Trade Z-Score',np.nan),2)} vs own history, {r['Trade Bal Direction']})")
                    st.write(f"M2 Growth:      {fmt(r['M2 Growth YoY %'])}%  ({r['M2 Trend']})")
                    st.write(f"Consumer Conf:  {fmt(r['Consumer Conf'],1)}  ({r['Conf Direction']})")
                    st.write(f"Building Permits:{fmt(r['Building Permits'],0)}")
                with c3:
                    st.markdown("**CB & Forward Outlook**")
                    st.write(f"Taylor CB Score: {fmt(r.get('CB Taylor Score',np.nan))} ← key signal")
                    st.write(f"CB Action:       **{r['Expected CB Action']}**")
                    st.write(f"CPI (12m fwd):   {fmt(r['CPI Expected (12m)'])}%")
                    st.write(f"Rate (12m fwd):  {fmt(r['Rate Expected (12m)'])}%")
                    st.write(f"GDP (12m fwd):   {fmt(r['GDP Expected (12m)'])}%")
                    st.write(f"FX 1m/3m/12m:    {fmt(r['FX Chg 1m %'])}% / {fmt(r['FX Chg 3m %'])}% / {fmt(r['FX Chg 12m %'])}%")

                st.write("**Component Scores**")
                for label, key in [("GDP","GDP Score"),("INFLATION LEVEL","Inflation Score"),
                                    ("CB TAYLOR","CB Taylor Score"),
                                    ("TRADE","Trade Score"),("M2","M2 Score"),("CONF","Conf Score")]:
                    st.text(f"  {label:16s}  {score_bar(r.get(key, np.nan))}")
                st.caption(
                    "Inflation Level and CB Taylor are now distinct inputs: Inflation Level "
                    "scores how far CPI sits from a sensible target; CB Taylor scores the "
                    "full hike/cut reaction function (inflation momentum + labour market + "
                    "growth + recent rate moves). Composite weights: "
                    + ", ".join(f"{k}={v:.0%}" for k, v in SCORE_WEIGHTS.items())
                )

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

        st.subheader("Component Breakdown — Radar")
        radar_currencies = st.multiselect("Select currencies for radar",
                                           hm_codes, default=hm_codes[:4])
        categories = ["GDP", "Inflation Level", "CB Taylor", "Trade", "M2", "Conf"]
        score_keys  = ["GDP Score","Inflation Score","CB Taylor Score","Trade Score","M2 Score","Conf Score"]
        fig_radar = go.Figure()
        for r in results:
            if r["code"] not in radar_currencies: continue
            vals = [r.get(k, 0) or 0 for k in score_keys]
            vals += [vals[0]]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals, theta=categories + [categories[0]],
                fill="toself", name=r["code"], line=dict(width=2),
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[-1,1])),
            template="plotly_dark", height=450,
            title="Factor Radar (−1 = bearish, +1 = bullish)",
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        st.divider()
        st.subheader("🌐 Macro Regime")
        st.caption(
            "Regime is computed from economy-weighted trimmed means "
            "(ECONOMY_WEIGHT × trimmed 10%), so EM outliers no longer dominate "
            "the G10 aggregate. G10 CB bias is computed separately."
        )
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
            st.info("Regime is inferred from aggregate macro signals. It is a qualitative guide.")

        st.divider()
        st.subheader("Central Bank Taylor Rule Scores")
        st.caption("Positive = hawkish bias (bullish for currency), negative = dovish.")
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

        st.divider()
        st.subheader("FX Returns Correlation Matrix (90d, vs USD)")
        st.caption(
            "New: helps avoid stacking near-duplicate risk. Two pairs that look "
            "different in the Pairs tab can still carry the same underlying bet "
            "if their legs are highly correlated against USD (e.g. AUD and NZD "
            "often move together). Values close to ±1 mean the two currencies' "
            "USD-leg returns have been moving in lockstep (+1) or in mirror-image (−1) "
            "over the last 90 trading days; values near 0 mean they've been moving "
            "largely independently."
        )
        corr_codes = [c for c in selected_codes if c != "USD"]
        corr_mat = fx_returns_correlation_matrix(fx_by_code, corr_codes, window=90)
        if corr_mat.empty:
            st.info("Not enough overlapping FX history to compute correlations "
                    "(need 2+ non-USD currencies with sufficient price history).")
        else:
            fig_corr = go.Figure(go.Heatmap(
                z=corr_mat.values, x=corr_mat.columns.tolist(), y=corr_mat.index.tolist(),
                colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
                text=[[f"{v:.2f}" for v in row] for row in corr_mat.values],
                texttemplate="%{text}",
            ))
            fig_corr.update_layout(template="plotly_dark", height=max(300, len(corr_mat)*45),
                                    title="90-Day FX Return Correlation (vs USD leg)")
            st.plotly_chart(fig_corr, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3 · PAIRS
    # ══════════════════════════════════════════════════════════════════════════
    with tab_pairs_tab:
        st.subheader("Pair Score Matrix — Relative Fundamental Analysis")
        st.caption(
            "Pair Score = Long composite − Short composite. "
            "Higher absolute score = greater fundamental divergence. "
            "★★★ = high conviction."
        )

        if pair_df.empty:
            st.info("No qualifying pairs with current liquidity filter.")
        else:
            top_df = pair_df[["Pair","Pair Score","Conviction","Long","Long Score",
                               "Short","Short Score","Long CB","Short CB","Liquidity Tier"]].head(15)
            styled_pairs = top_df.style.background_gradient(
                subset=["Pair Score"], cmap="RdYlGn", vmin=-1, vmax=1)
            st.dataframe(styled_pairs, use_container_width=True, hide_index=True)

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

        st.divider()
        st.subheader("Full Score Differential Matrix")
        st.caption("Cell = row currency score − column currency score.")
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
            else:
                st.info("DXY data unavailable right now (yfinance fetch returned empty).")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 4 · CARRY  (vol-adjusted carry)
    # ══════════════════════════════════════════════════════════════════════════
    with tab_carry:
        st.subheader("Vol-Adjusted Carry Analysis")
        st.caption(
            "**Vol-Adj Carry = Real Yield Differential ÷ Pair Annualised Vol.**  "
            "A ratio ≥ 0.20 is considered attractive; < 0.10 is marginal. "
            "This is a Sharpe-like measure: a 5% carry in TRY/JPY (vol ~25%) "
            "scores the same as a 1.6% carry in AUD/JPY (vol ~8%), because "
            "TRY's FX vol can erase the carry in a single bad session. "
            "Raw carry differentials alone overstate EM carry attractiveness. "
            "Vol = 63-day realized (annualised) from daily FX returns, derived "
            "from the same price history shown elsewhere in the app."
        )

        if carry_df.empty:
            st.info("No classic carry pairs found in selected currencies.")
        else:
            def colour_carry(val):
                if "Attractive" in str(val): return "background-color:#1a4a1a;color:#7fff7f"
                if "Marginal"   in str(val): return "background-color:#2a2a1a;color:#ffff99"
                if "Negative"   in str(val): return "background-color:#4a1a1a;color:#ff9999"
                if "Unattract" in str(val):  return "background-color:#3a2a1a;color:#ffaa55"
                return ""

            styled_carry = carry_df.style.map(colour_carry, subset=["Carry Rating"])
            st.dataframe(styled_carry, use_container_width=True, hide_index=True)

            plot_rows = carry_df[carry_df["Vol-Adj Carry"] != "—"].copy()
            if not plot_rows.empty:
                plot_rows["_adj"] = pd.to_numeric(plot_rows["Vol-Adj Carry"], errors="coerce")
                plot_rows = plot_rows.dropna(subset=["_adj"])
                fig_carry = go.Figure(go.Bar(
                    x=plot_rows["Pair"],
                    y=plot_rows["_adj"],
                    marker_color=["#2ecc71" if v >= 0.20 else "#f1c40f" if v >= 0.10
                                  else "#e74c3c" for v in plot_rows["_adj"]],
                    text=[f"{v:.3f}" for v in plot_rows["_adj"]],
                    textposition="outside",
                ))
                fig_carry.add_hline(y=0.20, line_dash="dash", line_color="#2ecc71",
                                    annotation_text="Attractive threshold (0.20)")
                fig_carry.add_hline(y=0.10, line_dash="dot", line_color="#f1c40f",
                                    annotation_text="Marginal threshold (0.10)")
                fig_carry.update_layout(
                    title="Vol-Adjusted Carry Ratio by Pair",
                    yaxis_title="Real Yield Diff / Pair Vol",
                    template="plotly_dark", height=340,
                )
                st.plotly_chart(fig_carry, use_container_width=True)

        st.divider()
        st.subheader("Real Rate Differentials (10Y-based)")
        st.caption("Raw differentials shown for reference. Use the vol-adjusted table above for sizing.")
        st.dataframe(rr_diff_df, use_container_width=True)

        st.info(
            "**Early warnings:** VIX multi-year lows = complacency; rising VIX = reduce carry. "
            "Inverted yield curve = recession risk. LIBOR-OIS widening = interbank stress. "
            "High FX vol (Pair Vol % column) = carry unwind risk even when rate diff is wide."
        )

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

        # New: cross-check the theme's affected currencies (picked in the
        # sidebar) against current fundamentals, so the IXL tool isn't a
        # disconnected slider exercise — it actually tells you whether the
        # theme would be reinforcing or fighting the existing macro picture.
        if ixl_affected:
            st.divider()
            st.subheader("Theme vs. Current Fundamentals")
            st.caption(
                "Cross-check: does this theme reinforce or fight what the fundamental "
                "model already shows for the currencies you flagged as affected?"
            )
            result_map = {r["code"]: r for r in results}
            rows = []
            for c in ixl_affected:
                if c not in result_map:
                    rows.append({"Currency": c, "Note": "Not in selected currency universe — add it in the sidebar to see fundamentals."})
                    continue
                r = result_map[c]
                rows.append({
                    "Currency":        c,
                    "Current Bias":    r["Overall Bias"],
                    "Composite Score": fmt(r.get("Composite Score", np.nan), 3),
                    "CB Action":       r["Expected CB Action"],
                    "FX 1m Mom %":     fmt(r["FX Chg 1m %"]),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            if ixl_res >= 40:
                st.info(
                    "If the theme's expected direction for a currency matches its 'Current "
                    "Bias' above, the theme reinforces the existing fundamental trade. If it "
                    "contradicts it, that's a genuine conflict to resolve before sizing a "
                    "position — not something this tool can resolve for you automatically."
                )

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
        st.info("⚠️ Divergence = COT contradicts fundamentals → wait for alignment or trade with caution.")
        if cot_df.empty:
            st.warning("COT data unavailable from CFTC. Check connectivity.")
        no_cot_codes = [r["code"] for r in results if r["COT Bias"] == "—"]
        if no_cot_codes and not cot_df.empty:
            st.caption(
                f"No CFTC non-commercial futures contract is tracked for: {', '.join(no_cot_codes)} "
                "(USD has no standalone contract in this dataset; TRY/ZAR/CNY are not listed CME "
                "futures in the CFTC financial futures report)."
            )

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
        st.subheader("Buy the Rumour / Sell the Fact")
        st.markdown("""
| Signal | Watch for |
|--------|-----------|
| **Rumour phase** | FX moves 1–4 weeks before meeting; position early |
| **Fact phase** | Decision already priced → potential reversal at announcement |
| **Consensus check** | ForexFactory → indicator → beat/miss history |
| **Divergence trade** | Strong fundamentals + short-term negative sentiment → discount entry |
""")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 8 · CPI ACTUAL VS EXPECTED  (EM-aware model)
    # ══════════════════════════════════════════════════════════════════════════
    with tab_cpi:
        st.subheader("CPI YoY — Actual vs. Model-Expected, Trailing Months")
        st.caption(
            "HIGH_INFLATION_CURRENCIES (TRY, ZAR, MXN, CNY) use an EM-aware anchor: "
            "the model reverts toward the currency's own 5-year median CPI rather than "
            "the 2% DM target. This avoids a chronic negative-surprise artefact that a "
            "flat 2% anchor would otherwise produce for a 32% CPI currency."
        )

        months_back = st.slider("Months of history to show", min_value=6, max_value=36, value=24, step=1)
        cpi_codes_available = [c for c in selected_codes
                                if not cpi_history.get(c, pd.Series(dtype=float)).dropna().empty]

        if not cpi_codes_available:
            st.warning("No CPI history available for the selected currencies.")
        else:
            cpi_tab_currencies = st.multiselect(
                "Currencies to display", cpi_codes_available,
                default=cpi_codes_available[:4], key="cpi_tab_select")

            for code in cpi_tab_currencies:
                series = cpi_history.get(code, pd.Series(dtype=float))
                cpi_df = cpi_actual_vs_expected(code, series, months=months_back)
                if cpi_df.empty:
                    st.info(f"{code}: not enough CPI history to build a trail.")
                    continue

                info = CURRENCY_INFO.get(code, {})
                em_note = " (EM anchor: own 5yr median)" if code in HIGH_INFLATION_CURRENCIES else " (DM anchor: 2% target)"
                st.markdown(f"**{code}** — {info.get('name', code)}{em_note}")

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
                                   annotation_text="2% DM target", annotation_position="bottom right")
                fig_cpi.update_layout(
                    template="plotly_dark", height=300,
                    yaxis_title="CPI YoY %",
                    margin=dict(t=10, b=10),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                )
                st.plotly_chart(fig_cpi, use_container_width=True)

                fig_surprise = go.Figure(go.Bar(
                    x=cpi_df["Date"], y=cpi_df["Surprise (Actual − Expected)"],
                    marker_color=["#2ecc71" if v >= 0 else "#e74c3c"
                                  for v in cpi_df["Surprise (Actual − Expected)"]],
                ))
                fig_surprise.update_layout(
                    template="plotly_dark", height=180,
                    yaxis_title="Surprise (pp)",
                    title="Actual − Expected (positive = hotter than model predicted)",
                    margin=dict(t=30, b=10),
                )
                st.plotly_chart(fig_surprise, use_container_width=True)

                with st.expander(f"Show data table — {code}"):
                    show_df = cpi_df.copy()
                    show_df["Date"] = pd.to_datetime(show_df["Date"]).dt.strftime("%Y-%m")
                    st.dataframe(show_df, use_container_width=True, hide_index=True)

                avg_surprise = cpi_df["Surprise (Actual − Expected)"].mean()
                st.caption(
                    f"Average surprise over period: {avg_surprise:+.2f}pp "
                    f"({'inflation running hotter than model' if avg_surprise > 0.2 else 'inflation running cooler than model' if avg_surprise < -0.2 else 'model tracking well'})"
                )
                st.divider()


if __name__ == "__main__":
    main()
