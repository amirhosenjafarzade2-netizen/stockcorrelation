import streamlit as st
import requests
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

COINGECKO_BASE  = "https://api.coingecko.com/api/v3"
BLOCKCHAIN_BASE = "https://api.blockchain.info"
FEAR_GREED_URL  = "https://api.alternative.me/fng/?limit=30"
BINANCE_FR_URL  = "https://fapi.binance.com/fapi/v1/fundingRate?symbol=BTCUSDT&limit=8"

REQUEST_TIMEOUT = 10
HEADERS = {"User-Agent": "Mozilla/5.0 CryptoScanner/3.2"}

# Block reward halving schedule (epoch start height -> reward). Used only to
# estimate current BTC issuance per day for supply / mining-cost context.
BLOCK_REWARD_BTC   = 3.125          # post-April-2024 halving reward
BLOCKS_PER_DAY_AVG = 144            # ~10 min/block target
NEXT_HALVING_YEAR  = 2028

# Macro tickers / FRED series for the new Macro Liquidity signal.
DXY_TICKER = "DX-Y.NYB"
FRED_FED_FUNDS   = "DFEDTARU"          # Fed funds target rate (upper bound)
FRED_CPI_USD_YOY = "CPALTT01USM659N"   # US CPI YoY %
FRED_UNRATE      = "UNRATE"            # US unemployment rate
FRED_GDP_GROWTH  = "A191RL1Q225SBEA"   # US real GDP growth, annualized
FRED_10Y         = "DGS10"             # 10Y Treasury yield
INFLATION_TARGET = 2.0

# Stablecoin tickers tracked via CoinGecko market data for the new
# Stablecoin Supply Growth signal — a liquidity/dry-powder proxy that is
# largely independent of spot price action (unlike the correlated group).
STABLECOIN_IDS = {
    "USDT": "tether",
    "USDC": "usd-coin",
    "DAI":  "dai",
}


# ══════════════════════════════════════════════════════════════════════════════
# SCORE WEIGHTS — analyst priors, NOT backtested coefficients.
# Surfaced here (rather than buried) so it's visible these are a starting
# point for discussion, not a validated model. Editable via the UI, and the
# UI auto-rebalances all weights to sum to 100% whenever one is changed.
#
# v3.2 note on defaults: adding "Macro Liquidity" to the correlated
# price-action group (see CORRELATED_SIGNAL_GROUP below) would inflate that
# group's combined weight if simply appended on top of v3.1's weights. To
# avoid that, the four pre-existing correlated signals were trimmed down
# slightly so the five-signal correlated group's total share (~45%) is
# roughly unchanged from v3.1's four-signal total (~46%). "Stablecoin
# Supply Growth" is NOT part of the correlated group — it is a liquidity/
# demand proxy that does not mechanically move with price the way the
# others do — so its weight was carved out of the broader on-chain budget
# rather than from the correlated group.
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_WEIGHTS = {
    "Fear & Greed":            12,   # was 14 — trimmed to make room for Macro Liquidity
    "Hash Rate":                11,   # was 12
    "Active Addresses":        11,   # was 12
    "Funding Rate":             10,   # was 12 — trimmed (correlated group)
    "Price vs MA200":          10,   # was 12 — trimmed (correlated group)
    "30d Momentum":              7,   # was 8  — trimmed (correlated group)
    "BTC Dominance":             4,   # was 5
    "NVT Ratio":                 9,   # was 10
    "Puell Multiple":            5,   # was 6
    "Volume Confirmation":       8,   # was 9
    "Macro Liquidity":           8,   # NEW — correlated group (price-action driver)
    "Stablecoin Supply Growth":  5,   # NEW — independent liquidity/demand proxy
}
# Weights above sum to 100 by construction; the UI re-derives fractions
# (weight / 100) at render time regardless, so this is a convenience, not
# a hard requirement.


# ══════════════════════════════════════════════════════════════════════════════
# DATA FETCHERS  (all cached 15 min)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=900, show_spinner=False)
def fetch_fear_greed() -> Dict:
    """Fear & Greed Index — current value + 30-day history."""
    try:
        r = requests.get(FEAR_GREED_URL, timeout=REQUEST_TIMEOUT, headers=HEADERS)
        data = r.json().get("data", [])
        if not data:
            return {}
        current = data[0]
        history = [
            {
                "date":  datetime.fromtimestamp(int(d["timestamp"])).strftime("%Y-%m-%d"),
                "value": int(d["value"]),
                "label": d["value_classification"],
            }
            for d in data
        ]
        return {
            "value":   int(current["value"]),
            "label":   current["value_classification"],
            "history": history,
        }
    except Exception:
        return {}


@st.cache_data(ttl=900, show_spinner=False)
def fetch_coingecko_btc() -> Dict:
    """BTC market data from CoinGecko (price, dominance, volume, changes)."""
    try:
        r = requests.get(
            f"{COINGECKO_BASE}/coins/bitcoin",
            params={
                "localization": "false",
                "tickers": "false",
                "community_data": "false",
                "developer_data": "false",
            },
            timeout=REQUEST_TIMEOUT,
            headers=HEADERS,
        )
        d = r.json()
        md = d.get("market_data", {})

        g = requests.get(
            f"{COINGECKO_BASE}/global",
            timeout=REQUEST_TIMEOUT,
            headers=HEADERS,
        ).json().get("data", {})

        dominance = g.get("market_cap_percentage", {}).get("btc", np.nan)

        return {
            "price":           md.get("current_price", {}).get("usd", np.nan),
            "market_cap":      md.get("market_cap", {}).get("usd", np.nan),
            "volume_24h":      md.get("total_volume", {}).get("usd", np.nan),
            "chg_24h":         md.get("price_change_percentage_24h", np.nan),
            "chg_7d":          md.get("price_change_percentage_7d", np.nan),
            "chg_30d":         md.get("price_change_percentage_30d", np.nan),
            "ath":             md.get("ath", {}).get("usd", np.nan),
            "ath_chg":         md.get("ath_change_percentage", {}).get("usd", np.nan),
            "dominance":       dominance,
            "circulating":     md.get("circulating_supply", np.nan),
            "max_supply":      md.get("max_supply", np.nan),
        }
    except Exception:
        return {}


@st.cache_data(ttl=900, show_spinner=False)
def fetch_blockchain_info() -> Dict:
    """Hash rate, active addresses, tx volume, mempool, miner revenue,
    difficulty, and USD on-chain transfer volume from blockchain.info."""
    out = {}
    try:
        s = requests.get(f"{BLOCKCHAIN_BASE}/stats", timeout=REQUEST_TIMEOUT, headers=HEADERS).json()
        out["hash_rate_eh"]      = s.get("hash_rate", np.nan) / 1e9   # convert to EH/s
        out["n_tx_24h"]          = s.get("n_tx", np.nan)
        out["total_btc_sent"]    = s.get("total_btc_sent", np.nan) / 1e8
        out["mempool_size"]      = s.get("mempool_size", np.nan)
        out["blocks_mined_24h"]  = s.get("n_blocks_mined", np.nan)
        out["difficulty"]        = s.get("difficulty", np.nan)
        out["minutes_between_blocks"] = s.get("minutes_between_blocks", np.nan)
        out["miners_revenue_usd_24h"] = s.get("miners_revenue_usd", np.nan)
        out["total_fees_btc_24h"]     = s.get("total_fees_btc", np.nan) / 1e8 if s.get("total_fees_btc") else np.nan
    except Exception:
        pass

    try:
        hr = requests.get(
            f"{BLOCKCHAIN_BASE}/charts/hash-rate",
            params={"timespan": "30days", "format": "json", "sampled": "true"},
            timeout=REQUEST_TIMEOUT,
            headers=HEADERS,
        ).json()
        if "values" in hr:
            out["hash_rate_history"] = [
                {"date": datetime.fromtimestamp(v["x"]).strftime("%Y-%m-%d"),
                 "value": v["y"] / 1e9}
                for v in hr["values"]
            ]
    except Exception:
        out["hash_rate_history"] = []

    try:
        aa = requests.get(
            f"{BLOCKCHAIN_BASE}/charts/n-unique-addresses",
            params={"timespan": "30days", "format": "json", "sampled": "true"},
            timeout=REQUEST_TIMEOUT,
            headers=HEADERS,
        ).json()
        if "values" in aa:
            out["active_addr_history"] = [
                {"date": datetime.fromtimestamp(v["x"]).strftime("%Y-%m-%d"),
                 "value": v["y"]}
                for v in aa["values"]
            ]
            vals = [v["value"] for v in out["active_addr_history"]]
            out["active_addr_now"]  = vals[-1]  if vals else np.nan
            out["active_addr_30d"]  = vals[0]   if vals else np.nan
    except Exception:
        out["active_addr_history"] = []
        out["active_addr_now"]     = np.nan
        out["active_addr_30d"]     = np.nan

    try:
        tv = requests.get(
            f"{BLOCKCHAIN_BASE}/charts/estimated-transaction-volume-usd",
            params={"timespan": "90days", "format": "json", "sampled": "true"},
            timeout=REQUEST_TIMEOUT,
            headers=HEADERS,
        ).json()
        if "values" in tv:
            out["tx_volume_usd_history"] = [
                {"date": datetime.fromtimestamp(v["x"]).strftime("%Y-%m-%d"),
                 "value": v["y"]}
                for v in tv["values"]
            ]
            vals = [v["value"] for v in out["tx_volume_usd_history"]]
            out["tx_volume_usd_now"] = vals[-1] if vals else np.nan
            out["tx_volume_usd_avg30"] = float(np.mean(vals[-30:])) if len(vals) >= 1 else np.nan
    except Exception:
        out["tx_volume_usd_history"] = []
        out["tx_volume_usd_now"]     = np.nan
        out["tx_volume_usd_avg30"]   = np.nan

    try:
        mr = requests.get(
            f"{BLOCKCHAIN_BASE}/charts/miners-revenue",
            params={"timespan": "1year", "format": "json", "sampled": "true"},
            timeout=REQUEST_TIMEOUT,
            headers=HEADERS,
        ).json()
        if "values" in mr:
            out["miner_revenue_history"] = [
                {"date": datetime.fromtimestamp(v["x"]).strftime("%Y-%m-%d"),
                 "value": v["y"]}
                for v in mr["values"]
            ]
    except Exception:
        out["miner_revenue_history"] = []

    return out


@st.cache_data(ttl=900, show_spinner=False)
def fetch_funding_rate() -> Dict:
    """Latest BTC perpetual funding rate from Binance Futures."""
    try:
        r = requests.get(BINANCE_FR_URL, timeout=REQUEST_TIMEOUT, headers=HEADERS)
        data = r.json()
        if not data or not isinstance(data, list):
            return {}
        rates = [float(d["fundingRate"]) * 100 for d in data]  # express as %
        if not rates:
            return {}
        return {
            "latest":  rates[-1],
            "avg_8":   float(np.mean(rates)),
            "history": rates,
        }
    except Exception:
        return {}


@st.cache_data(ttl=900, show_spinner=False)
def fetch_btc_ohlcv() -> pd.DataFrame:
    """BTC/USD daily OHLCV (1 year) via yfinance for MA, momentum & VPA."""
    try:
        data = yf.download("BTC-USD", period="1y", interval="1d",
                            auto_adjust=True, progress=False)
        if "Close" in data.columns:
            return data[["Open", "High", "Low", "Close", "Volume"]].dropna()
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=900, show_spinner=False)
def fetch_btc_marketcap_history(days: int = 90) -> pd.DataFrame:
    """
    Daily BTC market cap history from CoinGecko, used to build a TRUE NVT
    series (real mcap ÷ real on-chain volume per day) instead of holding
    market cap constant at today's value across history.
    """
    try:
        r = requests.get(
            f"{COINGECKO_BASE}/coins/bitcoin/market_chart",
            params={"vs_currency": "usd", "days": str(days), "interval": "daily"},
            timeout=REQUEST_TIMEOUT,
            headers=HEADERS,
        )
        d = r.json()
        caps = d.get("market_caps", [])
        if not caps:
            return pd.DataFrame()
        df = pd.DataFrame(caps, columns=["ts", "market_cap"])
        df["date"] = pd.to_datetime(df["ts"], unit="ms").dt.strftime("%Y-%m-%d")
        return df[["date", "market_cap"]].drop_duplicates(subset="date", keep="last")
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fred_series(series_id: str) -> pd.Series:
    """Generic FRED CSV fetch, shared by the Macro Liquidity signal."""
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


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_dxy_history() -> pd.Series:
    """DXY (US Dollar Index) daily history via yfinance, used for the
    Macro Liquidity signal's dollar-strength component."""
    try:
        data = yf.download(DXY_TICKER, period="6mo", interval="1d",
                            auto_adjust=True, progress=False)
        if data.empty:
            return pd.Series(dtype=float)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        s = data["Close"] if "Close" in data.columns else pd.Series(dtype=float)
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        return s.dropna()
    except Exception:
        return pd.Series(dtype=float)


@st.cache_data(ttl=900, show_spinner=False)
def fetch_stablecoin_supply() -> Dict:
    """
    Aggregate market cap (≈ circulating supply, since stablecoins are
    pegged ~1:1) for USDT, USDC, and DAI from CoinGecko, plus 30-day
    history per coin so we can measure supply GROWTH rather than just
    a snapshot level.

    Rationale: net stablecoin issuance is a reasonable proxy for "dry
    powder" entering the crypto ecosystem ahead of deployment into BTC
    or other assets. It is largely mechanical (driven by issuer minting/
    redemption decisions and exchange demand for on/off ramps) rather
    than a restatement of BTC's own price action — which is what makes
    it a genuinely different vote from the correlated price-action
    group, unlike Macro Liquidity (which IS expected to correlate with
    price action, since that's precisely the channel macro liquidity
    operates through).
    """
    total_now  = 0.0
    total_30d  = 0.0
    any_ok     = False
    per_coin   = {}

    for symbol, coingecko_id in STABLECOIN_IDS.items():
        try:
            r = requests.get(
                f"{COINGECKO_BASE}/coins/{coingecko_id}/market_chart",
                params={"vs_currency": "usd", "days": "30", "interval": "daily"},
                timeout=REQUEST_TIMEOUT,
                headers=HEADERS,
            )
            d = r.json()
            caps = d.get("market_caps", [])
            if not caps:
                continue
            vals = [c[1] for c in caps]
            now_v, ago_v = vals[-1], vals[0]
            total_now += now_v
            total_30d += ago_v
            per_coin[symbol] = {"now": now_v, "ago_30d": ago_v}
            any_ok = True
        except Exception:
            continue

    if not any_ok or total_30d <= 0:
        return {}

    growth_pct = (total_now - total_30d) / total_30d * 100
    return {
        "total_now":   total_now,
        "total_30d":   total_30d,
        "growth_pct":  growth_pct,
        "per_coin":    per_coin,
    }


# ══════════════════════════════════════════════════════════════════════════════
# MACRO LIQUIDITY  (NEW in v3.2)
# ══════════════════════════════════════════════════════════════════════════════

def trend_pct_change(s: pd.Series, lookback: int = 20) -> float:
    """% change of the trailing mean over `lookback` days vs the
    prior `lookback`-day window. Shared helper for DXY / yield trend."""
    v = s.dropna()
    if len(v) < lookback * 2:
        return np.nan
    recent = v.iloc[-lookback:].mean()
    prior  = v.iloc[-(lookback*2):-lookback].mean()
    if prior == 0 or pd.isna(prior):
        return np.nan
    return float((recent - prior) / abs(prior) * 100)


def taylor_rule_score_us(
    cpi_now: float, cpi_3m_ago: float,
    unemp_now: float, unemp_prior: float,
    gdp_growth: float,
    rate_now: float, rate_6m_ago: float,
) -> Tuple[float, str]:
    """
    US-only Taylor Rule estimate, ported from the forex fundamental
    scanner's taylor_rule_score(). Returns (score -1..+1, action label).
    Score > 0 = hawkish Fed (typically a headwind for risk assets/BTC);
    score < 0 = dovish Fed (typically a tailwind).
    """
    score, factors = 0.0, 0.0

    if not pd.isna(cpi_now):
        score += np.tanh((cpi_now - INFLATION_TARGET) / 2.0) * 0.35
        factors += 0.35
    if not pd.isna(cpi_now) and not pd.isna(cpi_3m_ago):
        score += np.tanh((cpi_now - cpi_3m_ago) / 1.2) * 0.20
        factors += 0.20
    if not pd.isna(unemp_now) and not pd.isna(unemp_prior):
        score += -np.tanh((unemp_now - unemp_prior) / 0.6) * 0.20
        factors += 0.20
    if not pd.isna(gdp_growth):
        score += np.tanh(gdp_growth / 2.0) * 0.15
        factors += 0.15
    if not pd.isna(rate_now) and not pd.isna(rate_6m_ago):
        score += np.tanh((rate_now - rate_6m_ago) / 0.75) * 0.10
        factors += 0.10

    score = float(np.clip(score / factors if factors > 0 else 0.0, -1, 1))
    if score >= 0.4:     action = "Strong Hike Expected"
    elif score >= 0.15:  action = "Hike Expected"
    elif score <= -0.4:  action = "Strong Cut Expected"
    elif score <= -0.15: action = "Cut Expected"
    else:                action = "Hold Expected"
    return score, action


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_macro_liquidity_inputs() -> Dict:
    """Pulls all raw series needed for the Macro Liquidity signal in one
    cached call: Fed policy data (FRED) + DXY + 10Y real-yield trend."""
    fed_rate = fetch_fred_series(FRED_FED_FUNDS)
    cpi      = fetch_fred_series(FRED_CPI_USD_YOY)
    unemp    = fetch_fred_series(FRED_UNRATE)
    gdp      = fetch_fred_series(FRED_GDP_GROWTH)
    ten_y    = fetch_fred_series(FRED_10Y)
    dxy      = fetch_dxy_history()

    def _latest(s):
        v = s.dropna()
        return float(v.iloc[-1]) if len(v) else np.nan

    def _n_ago(s, n):
        v = s.dropna()
        return float(v.iloc[-(n+1)]) if len(v) >= n + 1 else np.nan

    return {
        "fed_rate_now": _latest(fed_rate), "fed_rate_6m_ago": _n_ago(fed_rate, 6),
        "cpi_now": _latest(cpi), "cpi_3m_ago": _n_ago(cpi, 3),
        "unemp_now": _latest(unemp), "unemp_3m_ago": _n_ago(unemp, 3),
        "gdp_growth": _latest(gdp),
        "ten_y_now": _latest(ten_y),
        "real_yield_now": (_latest(ten_y) - _latest(cpi))
                            if not pd.isna(_latest(ten_y)) and not pd.isna(_latest(cpi)) else np.nan,
        "ten_y_series": ten_y, "cpi_series": cpi,
        "dxy_series": dxy,
        "dxy_now": _latest(dxy),
        "dxy_trend_pct": trend_pct_change(dxy, lookback=20),
    }


def compute_macro_liquidity(inputs: Dict) -> Dict:
    """
    Blends three sub-components into one Macro Liquidity reading:
      1. Fed Taylor score (hawkish/dovish bias) — weight 0.5
      2. DXY 20d trend (dollar strengthening = liquidity headwind) — weight 0.3
      3. Real 10Y yield level (higher real yields = risk-asset headwind) — weight 0.2

    All three feed from the same underlying liquidity regime, which is why
    this signal is grouped with the correlated price-action signals rather
    than treated as independent evidence — it does NOT measure something
    structurally different from Price vs MA200 / Funding Rate the way
    Stablecoin Supply Growth does. It earns its own scorecard row because
    it is forward-looking (policy trajectory) rather than backward-looking
    (price that has already happened), which is still useful, but the
    correlation tag should not be removed.
    """
    fed_score, fed_action = taylor_rule_score_us(
        inputs.get("cpi_now", np.nan), inputs.get("cpi_3m_ago", np.nan),
        inputs.get("unemp_now", np.nan), inputs.get("unemp_3m_ago", np.nan),
        inputs.get("gdp_growth", np.nan),
        inputs.get("fed_rate_now", np.nan), inputs.get("fed_rate_6m_ago", np.nan),
    )
    # Fed hawkishness is a headwind for risk assets like BTC, so we invert
    # sign here: hawkish Fed (fed_score > 0) -> negative contribution to
    # the bullish/bearish scale used throughout this scorecard.
    fed_component = -fed_score

    dxy_trend = inputs.get("dxy_trend_pct", np.nan)
    dxy_component = np.nan
    if not pd.isna(dxy_trend):
        # Rising DXY = tightening dollar liquidity = bearish for BTC.
        dxy_component = float(np.clip(-dxy_trend / 4.0, -1, 1))

    real_yield = inputs.get("real_yield_now", np.nan)
    yield_component = np.nan
    if not pd.isna(real_yield):
        # Real yield around 0-1% treated as neutral; materially positive
        # real yields are a headwind, materially negative are a tailwind.
        yield_component = float(np.clip(-(real_yield - 0.5) / 2.0, -1, 1))

    parts, weights_local = [], []
    if not pd.isna(fed_component):   parts.append(fed_component);   weights_local.append(0.5)
    if not pd.isna(dxy_component):   parts.append(dxy_component);   weights_local.append(0.3)
    if not pd.isna(yield_component): parts.append(yield_component); weights_local.append(0.2)

    if not parts:
        composite = np.nan
    else:
        w_sum = sum(weights_local)
        composite = float(sum(p * w for p, w in zip(parts, weights_local)) / w_sum)

    return {
        "score": composite,
        "fed_score": fed_score, "fed_action": fed_action, "fed_component": fed_component,
        "dxy_trend_pct": dxy_trend, "dxy_component": dxy_component,
        "real_yield_now": real_yield, "yield_component": yield_component,
    }


def score_macro_liquidity(macro: Dict) -> Tuple[float, str]:
    score = macro.get("score", np.nan)
    if pd.isna(score):
        return 0.0, "Insufficient macro data"

    fed_action = macro.get("fed_action", "—")
    dxy_trend  = macro.get("dxy_trend_pct", np.nan)
    real_yield = macro.get("real_yield_now", np.nan)

    dxy_note = f"DXY {'rising' if not pd.isna(dxy_trend) and dxy_trend > 0 else 'falling'} " \
               f"{abs(dxy_trend):.1f}% (20d)" if not pd.isna(dxy_trend) else "DXY n/a"
    yield_note = f"real 10Y yield {real_yield:+.1f}%" if not pd.isna(real_yield) else "real yield n/a"

    note = f"Fed: {fed_action}; {dxy_note}; {yield_note}"
    if score > 0.3:
        note += " — liquidity tailwind, BULLISH for risk assets"
    elif score < -0.3:
        note += " — liquidity headwind, BEARISH for risk assets"
    else:
        note += " — liquidity roughly neutral"
    return score, note


# ══════════════════════════════════════════════════════════════════════════════
# STABLECOIN SUPPLY GROWTH SCORING  (NEW in v3.2)
# ══════════════════════════════════════════════════════════════════════════════

def score_stablecoin_growth(sc: Dict) -> Tuple[float, str]:
    """
    Net stablecoin supply growth as a dry-powder / liquidity-inflow proxy.

    This is intentionally NOT part of CORRELATED_SIGNAL_GROUP: stablecoin
    minting/redemption is driven by issuer and exchange-side decisions
    (new fiat on-ramping, redemptions for off-ramping) that lead or lag
    price moves rather than mechanically restating them, unlike Funding
    Rate or 30d Momentum, which are direct transformations of price/order-
    book data. Treat it as one of the more independent votes in the
    scorecard.

    Caveat (disclosed in UI): stablecoin supply also grows for reasons
    unrelated to crypto risk appetite — e.g. trading desks parking
    treasury cash, cross-border settlement use, or yield-farming flows
    on DeFi protocols that never touch BTC. It is a noisy proxy, not a
    confirmed "money about to buy BTC" signal.
    """
    growth = sc.get("growth_pct", np.nan)
    if pd.isna(growth):
        return 0.0, "No stablecoin supply data"

    if growth > 8:    return +1.0, f"Stablecoin supply +{growth:.1f}% (30d) — strong inflow, dry powder building — BULLISH"
    if growth > 3:    return +0.5, f"Stablecoin supply +{growth:.1f}% (30d) — mild inflow"
    if growth > -3:   return  0.0, f"Stablecoin supply {growth:+.1f}% (30d) — roughly flat"
    if growth > -8:   return -0.5, f"Stablecoin supply {growth:.1f}% (30d) — mild outflow / redemptions"
    return -1.0,                 f"Stablecoin supply {growth:.1f}% (30d) — sharp outflow, capital leaving — BEARISH"


# ══════════════════════════════════════════════════════════════════════════════
# DERIVED METRICS  (mining cost, NVT, Puell, supply, VPA)
# ══════════════════════════════════════════════════════════════════════════════

def compute_mining_cost(bc: Dict, cg: Dict, electricity_usd_kwh: float,
                         fleet_efficiency_j_th: float) -> Dict:
    """
    Estimate an all-in ELECTRICITY-ONLY cost per BTC mined.

    Deliberately excludes ASIC capex/depreciation, cooling overhead, and pool
    fees — those are real costs but vary too widely by miner to model
    generically. For many operations, hardware depreciation is comparable to
    or larger than electricity cost, so treat this number as a soft floor,
    not a true breakeven. It tells you "below this, even the cheapest-power
    miners are burning cash on electricity alone" — a useful capitulation
    threshold, but not full miner P&L.
    """
    hash_rate_eh = bc.get("hash_rate_eh", np.nan)
    blocks_24h   = bc.get("blocks_mined_24h", np.nan)
    price        = cg.get("price", np.nan)

    if pd.isna(hash_rate_eh) or hash_rate_eh <= 0:
        return {}

    hash_rate_hs = hash_rate_eh * 1e18
    efficiency_j_per_h = fleet_efficiency_j_th / 1e12

    watts        = hash_rate_hs * efficiency_j_per_h
    network_mw   = watts / 1e6
    kwh_per_day  = watts * 24 / 1000
    cost_per_day = kwh_per_day * electricity_usd_kwh

    blocks = blocks_24h if not pd.isna(blocks_24h) and blocks_24h > 0 else BLOCKS_PER_DAY_AVG
    btc_issued_per_day = BLOCK_REWARD_BTC * blocks

    breakeven_per_btc = cost_per_day / btc_issued_per_day if btc_issued_per_day > 0 else np.nan

    margin_pct = np.nan
    if not pd.isna(price) and not pd.isna(breakeven_per_btc) and breakeven_per_btc > 0:
        margin_pct = (price - breakeven_per_btc) / breakeven_per_btc * 100

    return {
        "network_power_mw":      network_mw,
        "network_power_gw":      network_mw / 1000,
        "daily_energy_cost_usd": cost_per_day,
        "btc_issued_per_day":    btc_issued_per_day,
        "breakeven_per_btc":     breakeven_per_btc,
        "spot_price":            price,
        "margin_pct":            margin_pct,
    }


def compute_nvt(bc: Dict, cg: Dict, mcap_hist: pd.DataFrame) -> Dict:
    """
    NVT Ratio = Market Cap / Daily On-Chain Transaction Volume (USD), 30d
    smoothed to reduce single-day noise (standard "NVT Signal" adjustment).

    TREND component: uses real daily market-cap history (fetched from
    CoinGecko) divided by real daily on-chain volume, so the trend
    reflects actual NVT movement, not just the volume component of it.
    If market-cap history isn't available, trend falls back to volume-only
    drift and is EXPLICITLY labeled as such in the UI.

    Caveat carried into the UI: static NVT thresholds were calibrated years
    ago. As more BTC activity moves to exchanges, Lightning, and custodial
    rails that never touch the base chain, on-chain transfer volume has
    structurally drifted down relative to market cap — so "elevated" NVT
    today may not mean what it meant in 2018. Treat NVT as one input, not
    a standalone valuation call.
    """
    mcap       = cg.get("market_cap", np.nan)
    tx_vol30   = bc.get("tx_volume_usd_avg30", np.nan)
    tx_vol_now = bc.get("tx_volume_usd_now", np.nan)
    tv_hist    = bc.get("tx_volume_usd_history", [])

    nvt_smoothed = mcap / tx_vol30 if (not pd.isna(mcap) and not pd.isna(tx_vol30) and tx_vol30 > 0) else np.nan
    nvt_daily    = mcap / tx_vol_now if (not pd.isna(mcap) and not pd.isna(tx_vol_now) and tx_vol_now > 0) else np.nan

    nvt_trend_pct = np.nan
    nvt_trend_is_true = False

    if len(tv_hist) >= 35:
        df_vol = pd.DataFrame(tv_hist)[["date", "value"]].rename(columns={"value": "tx_vol"})

        if mcap_hist is not None and not mcap_hist.empty:
            merged = df_vol.merge(mcap_hist, on="date", how="inner")
            merged = merged.dropna(subset=["tx_vol", "market_cap"])
            merged = merged[merged["tx_vol"] > 0]
            if len(merged) >= 35:
                merged["nvt"] = merged["market_cap"] / merged["tx_vol"]
                nvt_series = merged["nvt"].tail(35).tolist()
                recent_avg = np.nanmean(nvt_series[-5:])
                prior_avg  = np.nanmean(nvt_series[-35:-30])
                if prior_avg and not pd.isna(prior_avg) and prior_avg != 0:
                    nvt_trend_pct = (recent_avg - prior_avg) / prior_avg * 100
                    nvt_trend_is_true = True

        if not nvt_trend_is_true:
            vals = [v["value"] for v in tv_hist]
            nvt_proxy_series = [mcap / v if (v and v > 0 and not pd.isna(mcap)) else np.nan for v in vals[-35:]]
            recent_avg = np.nanmean(nvt_proxy_series[-5:])
            prior_avg  = np.nanmean(nvt_proxy_series[-35:-30])
            if prior_avg and not pd.isna(prior_avg) and prior_avg != 0:
                nvt_trend_pct = (recent_avg - prior_avg) / prior_avg * 100

    return {
        "nvt_smoothed": nvt_smoothed,
        "nvt_daily": nvt_daily,
        "nvt_trend_pct": nvt_trend_pct,
        "nvt_trend_is_true": nvt_trend_is_true,
    }


def compute_puell_multiple(bc: Dict) -> Dict:
    """
    Puell Multiple proxy = Daily Miner Revenue (USD) / trailing average of
    Daily Miner Revenue (USD, up to 1y history as a proxy for the standard
    365d MA).

    Also returns a TREND component (5d vs prior 5d, on the multiple itself).
    Note: unlike NVT, this trend is NOT subject to the constant-denominator
    bug, because `ma` (the trailing average) is a single fixed number for
    the whole series by construction.
    """
    hist = bc.get("miner_revenue_history", [])
    if len(hist) < 30:
        return {"value": np.nan, "trend_pct": np.nan}

    vals = [v["value"] for v in hist]
    ma = np.mean(vals)
    if ma == 0 or pd.isna(ma):
        return {"value": np.nan, "trend_pct": np.nan}

    puell_now = vals[-1] / ma

    trend_pct = np.nan
    if len(vals) >= 40:
        puell_series = [v / ma for v in vals]
        recent_avg = np.mean(puell_series[-5:])
        prior_avg  = np.mean(puell_series[-35:-30]) if len(puell_series) >= 35 else np.nan
        if not pd.isna(prior_avg) and prior_avg != 0:
            trend_pct = (recent_avg - prior_avg) / prior_avg * 100

    return {"value": puell_now, "trend_pct": trend_pct}


def compute_vpa(ohlcv: pd.DataFrame) -> Dict:
    """
    Volume-Price Analysis. On-chain metrics describe network fundamentals
    on a weeks-to-months horizon; VPA describes order-book participation
    on a days-to-weeks horizon.

    Computes:
      - OBV (On-Balance Volume) and its 20d trend
      - Price 20d trend, for divergence comparison against OBV
      - 20d Volume-Weighted Average Price (rolling)
      - Volume-confirmed breakout/breakdown flag
    """
    if ohlcv.empty or len(ohlcv) < 25:
        return {}

    close = ohlcv["Close"]
    vol   = ohlcv["Volume"]
    if isinstance(close, pd.DataFrame): close = close.iloc[:, 0]
    if isinstance(vol, pd.DataFrame):   vol = vol.iloc[:, 0]

    direction = np.sign(close.diff().fillna(0))
    obv = (direction * vol).cumsum()

    obv_recent = float(obv.iloc[-5:].mean())
    obv_prior  = float(obv.iloc[-25:-20].mean())
    obv_trend_pct = ((obv_recent - obv_prior) / abs(obv_prior) * 100) if obv_prior != 0 else np.nan

    price_recent = float(close.iloc[-5:].mean())
    price_prior  = float(close.iloc[-25:-20].mean())
    price_trend_pct = ((price_recent - price_prior) / price_prior * 100) if price_prior != 0 else np.nan

    vwap20 = (close * vol).rolling(20).sum() / vol.rolling(20).sum()
    vwap20_now = float(vwap20.iloc[-1]) if not pd.isna(vwap20.iloc[-1]) else np.nan
    price_now = float(close.iloc[-1])

    avg_vol_20 = float(vol.iloc[-20:].mean())
    recent_vol_5 = float(vol.iloc[-5:].mean())
    vol_ratio = (recent_vol_5 / avg_vol_20) if avg_vol_20 > 0 else np.nan

    return {
        "obv_series": obv,
        "obv_trend_pct": obv_trend_pct,
        "price_trend_pct": price_trend_pct,
        "vwap20_series": vwap20,
        "vwap20_now": vwap20_now,
        "price_now": price_now,
        "vol_ratio_5_over_20": vol_ratio,
        "avg_vol_20": avg_vol_20,
        "recent_vol_5": recent_vol_5,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SCORING ENGINE  (existing signals, unchanged logic from v3.1)
# ══════════════════════════════════════════════════════════════════════════════

def score_fear_greed(fg: Dict) -> Tuple[float, str]:
    val = fg.get("value", np.nan)
    if pd.isna(val):
        return 0.0, "No data"
    if val <= 20:   return +1.0, f"Extreme Fear ({val}) — contrarian BULLISH zone"
    if val <= 35:   return +0.5, f"Fear ({val}) — mild bullish lean"
    if val <= 55:   return  0.0, f"Neutral ({val})"
    if val <= 75:   return -0.5, f"Greed ({val}) — mild bearish lean"
    return -1.0,               f"Extreme Greed ({val}) — contrarian BEARISH zone"


def score_hash_rate(bc: Dict) -> Tuple[float, str]:
    """
    Trend measured via linear-regression slope over the full 30-day
    history rather than comparing only the first 7 days to the last 7
    days, so the middle of the window counts and single-day noise at
    either edge matters less.
    """
    hist = bc.get("hash_rate_history", [])
    if len(hist) < 10:
        return 0.0, "Insufficient history"
    vals = np.array([v["value"] for v in hist], dtype=float)
    x = np.arange(len(vals))
    slope, intercept = np.polyfit(x, vals, 1)
    mean_val = np.mean(vals)
    if mean_val == 0:
        return 0.0, "No baseline"
    chg_pct = (slope * (len(vals) - 1)) / mean_val * 100
    if chg_pct > 5:    return +1.0, f"Hash rate trend +{chg_pct:.1f}% (30d, regression) — BULLISH"
    if chg_pct > 1:    return +0.5, f"Hash rate trend slightly rising +{chg_pct:.1f}% — mild bullish"
    if chg_pct > -1:   return  0.0, f"Hash rate trend flat ({chg_pct:+.1f}%) — Neutral"
    if chg_pct > -5:   return -0.5, f"Hash rate trend slightly falling {chg_pct:.1f}% — mild bearish"
    return -1.0,               f"Hash rate trend falling {chg_pct:.1f}% — BEARISH (miner stress)"


def score_active_addresses(bc: Dict) -> Tuple[float, str]:
    """
    Caveat: active-address counts are a directionally useful but noisy
    proxy for real usage — distorted by exchange custodial-wallet
    batching, UTXO consolidation sweeps, address reuse patterns, and
    activity migrating to Layer 2 / Lightning. Treat as one weak vote.
    """
    now = bc.get("active_addr_now", np.nan)
    ago = bc.get("active_addr_30d", np.nan)
    if pd.isna(now) or pd.isna(ago) or ago == 0:
        return 0.0, "No address data"
    chg = (now - ago) / ago * 100
    if chg > 10:   return +1.0, f"Active addresses +{chg:.1f}% (30d) — usage rising (noisy proxy; see guide)"
    if chg > 3:    return +0.5, f"Active addresses +{chg:.1f}% — mild rise (noisy proxy; see guide)"
    if chg > -3:   return  0.0, f"Active addresses flat ({chg:+.1f}%)"
    if chg > -10:  return -0.5, f"Active addresses {chg:.1f}% — mild decline (noisy proxy; see guide)"
    return -1.0,               f"Active addresses {chg:.1f}% — sharp drop (noisy proxy; see guide)"


def score_funding_rate(fr: Dict) -> Tuple[float, str]:
    """
    Thresholds loosened: moderately positive funding is common and
    unremarkable during genuine bull trends — the contrarian read works
    best at real extremes, not at the first sign of positive funding.
    """
    avg = fr.get("avg_8", np.nan)
    if pd.isna(avg):
        return 0.0, "No funding data"
    if avg > 0.08:    return -1.0, f"Funding rate very high ({avg:+.4f}%) — crowded longs — BEARISH"
    if avg > 0.03:    return -0.5, f"Funding rate elevated ({avg:+.4f}%) — mild bearish lean"
    if avg > -0.02:   return  0.0, f"Funding rate near-neutral ({avg:+.4f}%) — normal in either trend"
    if avg > -0.06:   return +0.5, f"Funding rate negative ({avg:+.4f}%) — shorts dominant — contrarian bullish"
    return +1.0,               f"Funding rate very negative ({avg:+.4f}%) — short squeeze setup — BULLISH"


def score_btc_dominance(cg: Dict) -> Tuple[float, str]:
    """
    Dominance is NOT a reliable directional signal for BTC itself — the
    same reading can mean opposite things depending on regime, so this
    is scored as a small regime-flag rather than a directional vote.
    """
    dom = cg.get("dominance", np.nan)
    if pd.isna(dom):
        return 0.0, "No dominance data"
    if dom > 60:   return +0.3, f"BTC dominance {dom:.1f}% — high; capital concentrated in BTC (regime flag, not a directional call)"
    if dom > 52:   return +0.15, f"BTC dominance {dom:.1f}% — elevated (regime flag, not a directional call)"
    if dom > 45:   return  0.0, f"BTC dominance {dom:.1f}% — balanced regime"
    if dom > 38:   return -0.15, f"BTC dominance {dom:.1f}% — alt-rotation regime (regime flag, not a directional call)"
    return -0.3,               f"BTC dominance {dom:.1f}% — low; heavy alt-rotation (regime flag, not a directional call)"


def score_price_vs_ma(ohlcv: pd.DataFrame) -> Tuple[float, str]:
    if ohlcv.empty or len(ohlcv) < 50:
        return 0.0, "Insufficient price history"
    close = ohlcv["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.dropna()
    price = float(close.iloc[-1])
    ma200 = float(close.rolling(min(200, len(close))).mean().iloc[-1])
    ma50  = float(close.rolling(50).mean().iloc[-1])
    pct_above_200 = (price - ma200) / ma200 * 100

    if price > ma200 and price > ma50:
        return +1.0, f"Price ${price:,.0f} above both MA50 & MA200 — strong uptrend — BULLISH"
    if price > ma200:
        return +0.5, "Price above MA200 but below MA50 — cautious bullish"
    if price < ma200 and pct_above_200 > -10:
        return -0.5, f"Price {pct_above_200:.1f}% below MA200 — bearish but near support"
    return -1.0,     f"Price {pct_above_200:.1f}% below MA200 — strong downtrend — BEARISH"


def score_30d_momentum(cg: Dict) -> Tuple[float, str]:
    chg = cg.get("chg_30d", np.nan)
    if pd.isna(chg):
        return 0.0, "No 30d data"
    if chg > 20:   return +1.0, f"30d return +{chg:.1f}% — strong momentum BULLISH"
    if chg > 5:    return +0.5, f"30d return +{chg:.1f}% — positive momentum"
    if chg > -5:   return  0.0, f"30d return {chg:+.1f}% — flat / consolidating"
    if chg > -20:  return -0.5, f"30d return {chg:.1f}% — negative momentum"
    return -1.0,               f"30d return {chg:.1f}% — strong selling pressure — BEARISH"


def score_nvt(nvt: Dict) -> Tuple[float, str]:
    """
    Combines LEVEL (is it cheap/expensive vs history) with TREND (is it
    getting cheaper or more expensive right now).
    """
    val = nvt.get("nvt_smoothed", np.nan)
    trend = nvt.get("nvt_trend_pct", np.nan)
    is_true_trend = nvt.get("nvt_trend_is_true", False)
    if pd.isna(val):
        return 0.0, "No NVT data"

    if val < 40:   base, label = +1.0, "usage far outpaces valuation — BULLISH (undervalued)"
    elif val < 65: base, label = +0.5, "usage healthy vs valuation"
    elif val < 90: base, label = 0.0,  "neutral valuation zone"
    elif val < 130: base, label = -0.5, "valuation stretched vs usage"
    else:          base, label = -1.0, "valuation far outpacing usage — BEARISH (overvalued)"

    trend_note = ""
    if not pd.isna(trend):
        nudge = np.clip(-trend / 100, -0.3, 0.3)
        base = float(np.clip(base + nudge, -1.0, 1.0))
        direction = "up" if trend > 0 else "down"
        if is_true_trend:
            trend_note = f", trending {direction} {abs(trend):.0f}% (30d, true NVT)"
        else:
            trend_note = f", volume-only proxy trending {direction} {abs(trend):.0f}% (30d — mcap history unavailable, NOT true NVT trend)"

    return base, f"NVT {val:.0f}{trend_note} — {label}"


def score_puell(puell: Dict) -> Tuple[float, str]:
    val = puell.get("value", np.nan)
    trend = puell.get("trend_pct", np.nan)
    if pd.isna(val):
        return 0.0, "No miner revenue history"

    if val < 0.5:   base, label = +1.0, "miner capitulation zone — contrarian BULLISH"
    elif val < 0.8: base, label = +0.5, "below-average miner revenue — mild bullish"
    elif val < 2.0: base, label = 0.0,  "normal range"
    elif val < 4.0: base, label = -0.5, "elevated miner revenue — mild bearish"
    else:           base, label = -1.0, "miner euphoria zone — contrarian BEARISH"

    trend_note = ""
    if not pd.isna(trend):
        nudge = np.clip(-trend / 100, -0.25, 0.25)
        base = float(np.clip(base + nudge, -1.0, 1.0))
        trend_note = f", {'rising' if trend > 0 else 'falling'} {abs(trend):.0f}% recently"

    return base, f"Puell {val:.2f}{trend_note} — {label}"


def score_volume_confirmation(vpa: Dict) -> Tuple[float, str]:
    """
    Does volume confirm the price trend, or diverge from it?
    """
    if not vpa:
        return 0.0, "No volume data"

    price_trend = vpa.get("price_trend_pct", np.nan)
    obv_trend   = vpa.get("obv_trend_pct", np.nan)
    vol_ratio   = vpa.get("vol_ratio_5_over_20", np.nan)

    if pd.isna(price_trend) or pd.isna(obv_trend):
        return 0.0, "Insufficient volume history"

    both_up   = price_trend > 1 and obv_trend > 1
    both_down = price_trend < -1 and obv_trend < -1
    bull_div  = price_trend < -1 and obv_trend > 1
    bear_div  = price_trend > 1 and obv_trend < -1

    vol_note = ""
    if not pd.isna(vol_ratio):
        if vol_ratio > 1.3:
            vol_note = f", recent volume {vol_ratio:.1f}× the 20d average — move has participation"
        elif vol_ratio < 0.7:
            vol_note = f", recent volume only {vol_ratio:.1f}× the 20d average — thin, low-conviction move"

    if both_up:
        return +1.0, f"Price and OBV both rising{vol_note} — volume confirms uptrend — BULLISH"
    if both_down:
        return -1.0, f"Price and OBV both falling{vol_note} — volume confirms downtrend — BEARISH"
    if bull_div:
        return +0.5, f"Price falling but OBV rising{vol_note} — possible accumulation under weakness"
    if bear_div:
        return -0.5, f"Price rising but OBV falling{vol_note} — possible distribution into strength — caution"
    return 0.0, f"Price/volume trend mixed or flat{vol_note}"


# ── Composite Scorer ─────────────────────────────────────────────────────────

# Signals that tend to move together because they're all reading the same
# underlying price action / liquidity backdrop from different angles.
# "Macro Liquidity" belongs here: it operates through the same channel
# (risk-asset appetite) that drives Price vs MA200 / Funding Rate / 30d
# Momentum / Fear & Greed, even though its inputs (Fed policy, DXY, real
# yields) are different data sources. "Stablecoin Supply Growth" is
# deliberately NOT in this group — see its scoring function for why.
CORRELATED_SIGNAL_GROUP = ["30d Momentum", "Price vs MA200", "Funding Rate", "Fear & Greed", "Macro Liquidity"]


def compute_composite(scores: Dict[str, float], weights: Dict[str, float]) -> float:
    total_w = sum(weights[k] for k in scores if k in weights and not pd.isna(scores[k]))
    if total_w == 0:
        return np.nan
    weighted = sum(weights[k] * scores[k]
                   for k in scores if k in weights and not pd.isna(scores[k]))
    return weighted / total_w


def composite_to_verdict(score: float) -> Tuple[str, str, str]:
    if pd.isna(score):
        return "Neutral", "★★★☆☆", "#f1c40f"
    if score >= 0.50:   return "Strong Bullish",  "★★★★★", "#00e676"
    if score >= 0.20:   return "Bullish",          "★★★★☆", "#69f0ae"
    if score >= -0.10:  return "Neutral",           "★★★☆☆", "#f1c40f"
    if score >= -0.40:  return "Bearish",           "★★☆☆☆", "#ff5252"
    return                "Strong Bearish",         "★☆☆☆☆", "#d50000"


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def fmt(val, decimals=2, prefix="", suffix="") -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "—"
    return f"{prefix}{val:,.{decimals}f}{suffix}"


def score_bar(score: float, width: int = 12) -> str:
    if pd.isna(score): return "—"
    filled = round((score + 1) / 2 * width)
    return "█" * filled + "░" * (width - filled) + f"  {score:+.2f}"


def delta_colour(val) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)): return "off"
    return "normal" if val >= 0 else "inverse"


# ══════════════════════════════════════════════════════════════════════════════
# WEIGHT SLIDER UI — auto-rebalancing  (NEW in v3.2)
# ══════════════════════════════════════════════════════════════════════════════

def _rebalance_weight(changed_key: str, total_cap: float = 60.0) -> None:
    """
    on_change callback for a single weight slider. Redistributes the
    delta across all OTHER weights in proportion to their current share
    of the remaining total, so relative balance among untouched sliders
    is preserved (rather than flattening toward equal weight over
    repeated edits, which a naive equal-split would do).
    """
    state = st.session_state.weight_state
    new_val = float(st.session_state[f"w_{changed_key}"])
    new_val = max(0.0, min(total_cap, new_val))
    old_val = state[changed_key]
    delta = new_val - old_val

    others = [k for k in state if k != changed_key]
    others_sum = sum(state[k] for k in others)

    state[changed_key] = new_val
    if others_sum > 0:
        for k in others:
            share = state[k] / others_sum
            state[k] = max(0.0, state[k] - delta * share)
    else:
        # All other sliders are at 0 — split the remainder evenly.
        even_share = max(0.0, (100.0 - new_val) / len(others)) if others else 0.0
        for k in others:
            state[k] = even_share

    # Floating point drift correction — force exact sum to 100.
    drift = 100.0 - sum(state.values())
    if abs(drift) > 1e-9 and others:
        state[others[0]] = max(0.0, state[others[0]] + drift)

    for k in state:
        st.session_state[f"w_{k}"] = state[k]


def render_weight_sliders(default_weights: Dict[str, float],
                           correlated_group: set) -> Dict[str, float]:
    """
    Renders one slider per signal. Moving any slider auto-rebalances all
    others (proportionally, not equally) so the total always sums to
    100%. State persists in st.session_state across reruns so successive
    drags compound correctly instead of resetting to defaults each time.
    Returns a dict of {signal_name: fraction} where fractions sum to 1.0,
    suitable for direct use in compute_composite().
    """
    if "weight_state" not in st.session_state:
        st.session_state.weight_state = dict(default_weights)
        for k, v in default_weights.items():
            st.session_state[f"w_{k}"] = v

    state = st.session_state.weight_state

    st.caption(
        "Drag any slider — the others automatically rebalance, proportionally, "
        "so the total always stays at 100%. This keeps the relative weighting "
        "you set elsewhere from silently drifting, the way post-hoc renormalization can."
    )

    reset_col, total_col = st.columns([1, 3])
    with reset_col:
        if st.button("↺ Reset to defaults", key="reset_weights_btn"):
            st.session_state.weight_state = dict(default_weights)
            for k, v in default_weights.items():
                st.session_state[f"w_{k}"] = v
            st.rerun()

    cols = st.columns(3)
    for i, name in enumerate(state.keys()):
        with cols[i % 3]:
            tag = " 🔗" if name in correlated_group else ""
            st.slider(
                name + tag, min_value=0.0, max_value=60.0,
                value=float(state[name]), step=1.0,
                key=f"w_{name}",
                on_change=_rebalance_weight, args=(name,),
            )

    total = sum(state.values())
    with total_col:
        st.caption(f"🔗 = correlated price-action / liquidity group · **Total: {total:.1f}%**")

    return {k: v / 100.0 for k, v in state.items()}


# ══════════════════════════════════════════════════════════════════════════════
# MAIN RENDER FUNCTION  (called by app.py)
# ══════════════════════════════════════════════════════════════════════════════

def render_crypto_onchain():
    st.header("₿ Bitcoin On-Chain Fundamental Scanner")
    st.caption("Chapter 9 Framework · Fear & Greed · Hash Rate · Active Addresses · Funding Rate · "
               "Dominance · NVT (trend-aware) · Puell (trend-aware) · Volume-Price Analysis · "
               "Mining Economics · Macro Liquidity · Stablecoin Supply Growth")

    with st.expander("⚙️ Mining Cost Assumptions (adjust for your estimate)", expanded=False):
        col_a, col_b = st.columns(2)
        with col_a:
            electricity_cost = st.slider(
                "Electricity cost ($/kWh)", min_value=0.02, max_value=0.20,
                value=0.06, step=0.01,
                help="Global average industrial mining electricity cost is roughly $0.04–$0.08/kWh"
            )
        with col_b:
            fleet_efficiency = st.slider(
                "Fleet efficiency (J/TH)", min_value=15.0, max_value=40.0,
                value=21.5, step=0.5,
                help="Modern ASICs (e.g. S21 class) run ~16-18 J/TH; network-wide average "
                     "(mix of older + newer rigs) is typically 20-25 J/TH"
            )
        st.warning(
            "⚠️ This is an **electricity-only** estimate. It excludes ASIC hardware depreciation, "
            "cooling overhead, and pool fees — for many miners, hardware depreciation is comparable to "
            "or larger than power cost. Treat the breakeven number below as a soft capitulation floor, "
            "not a true all-in cost.",
            icon="⚠️",
        )

    with st.expander("🎛️ Composite Score Weights (analyst priors — not backtested)", expanded=False):
        st.caption(
            "These weights are a reasonable starting point reflecting Chapter 9's framework plus two "
            "macro/liquidity additions, **not coefficients derived from a backtest**. Adjust them if "
            "you weight certain signals differently, or to stress-test how sensitive the verdict is "
            "to the weighting scheme."
        )
        st.info(
            "📐 **Correlation note:** *30d Momentum*, *Price vs MA200*, *Funding Rate*, *Fear & Greed*, "
            "and *Macro Liquidity* mostly read the same underlying price-action / liquidity backdrop "
            "from different angles and tend to move together. Summing their weights as if they were "
            "independent votes double-counts that shared information. *Stablecoin Supply Growth* is "
            "kept separate from this group — issuer-driven minting/redemption flows lead or lag price "
            "rather than mechanically restating it — but it is still a noisy proxy, not confirmed "
            "demand. Treat the composite below as a **structured checklist**, not a statistically "
            "validated predictive score.",
            icon="📐",
        )
        weights = render_weight_sliders(DEFAULT_WEIGHTS, set(CORRELATED_SIGNAL_GROUP))

    with st.spinner("Fetching on-chain & market data…"):
        fg          = fetch_fear_greed()
        cg          = fetch_coingecko_btc()
        bc          = fetch_blockchain_info()
        fr          = fetch_funding_rate()
        ohlcv       = fetch_btc_ohlcv()
        mcap_hist   = fetch_btc_marketcap_history(days=90)
        macro_in    = fetch_macro_liquidity_inputs()
        stablecoins = fetch_stablecoin_supply()

    mining = compute_mining_cost(bc, cg, electricity_cost, fleet_efficiency)
    nvt    = compute_nvt(bc, cg, mcap_hist)
    puell  = compute_puell_multiple(bc)
    vpa    = compute_vpa(ohlcv)
    macro  = compute_macro_liquidity(macro_in)

    fg_score,    fg_note    = score_fear_greed(fg)
    hr_score,    hr_note    = score_hash_rate(bc)
    aa_score,    aa_note    = score_active_addresses(bc)
    fund_score,  fund_note  = score_funding_rate(fr)
    dom_score,   dom_note   = score_btc_dominance(cg)
    ma_score,    ma_note    = score_price_vs_ma(ohlcv)
    mom_score,   mom_note   = score_30d_momentum(cg)
    nvt_score,   nvt_note   = score_nvt(nvt)
    puell_score, puell_note = score_puell(puell)
    vol_score,   vol_note   = score_volume_confirmation(vpa)
    macro_score, macro_note = score_macro_liquidity(macro)
    sc_score,    sc_note    = score_stablecoin_growth(stablecoins)

    raw_scores = {
        "Fear & Greed":               fg_score,
        "Hash Rate":                  hr_score,
        "Active Addresses":           aa_score,
        "Funding Rate":               fund_score,
        "Price vs MA200":             ma_score,
        "30d Momentum":               mom_score,
        "BTC Dominance":              dom_score,
        "NVT Ratio":                  nvt_score,
        "Puell Multiple":             puell_score,
        "Volume Confirmation":        vol_score,
        "Macro Liquidity":            macro_score,
        "Stablecoin Supply Growth":   sc_score,
    }
    notes = {
        "Fear & Greed":               fg_note,
        "Hash Rate":                  hr_note,
        "Active Addresses":           aa_note,
        "Funding Rate":               fund_note,
        "Price vs MA200":             ma_note,
        "30d Momentum":               mom_note,
        "BTC Dominance":              dom_note,
        "NVT Ratio":                  nvt_note,
        "Puell Multiple":             puell_note,
        "Volume Confirmation":        vol_note,
        "Macro Liquidity":            macro_note,
        "Stablecoin Supply Growth":   sc_note,
    }

    composite              = compute_composite(raw_scores, weights)
    verdict, stars, colour = composite_to_verdict(composite)

    data_coverage = sum(1 for v in raw_scores.values() if not pd.isna(v) and v != 0.0)
    missing = [k for k, v in raw_scores.items() if pd.isna(v)]

    # ══════════════════════════════════════════════════════════════════════════
    # VERDICT BANNER
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, {colour}22, {colour}11);
            border: 2px solid {colour};
            border-radius: 16px;
            padding: 24px 32px;
            margin-bottom: 24px;
            text-align: center;
        ">
            <div style="font-size: 2.4rem; font-weight: 800; color: {colour};">
                {verdict}
            </div>
            <div style="font-size: 1.4rem; margin: 6px 0;">{stars}</div>
            <div style="font-size: 1rem; color: #aaa;">
                Composite On-Chain Score: <b style="color:{colour}">{composite:+.3f}</b>
                &nbsp;|&nbsp; BTC Price: <b>${cg.get('price', 0):,.0f}</b>
                &nbsp;|&nbsp; Updated: {datetime.now().strftime('%H:%M UTC')}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if missing:
        st.caption(f"ℹ️ No data available for: {', '.join(missing)} — composite reweighted across remaining signals.")

    # ══════════════════════════════════════════════════════════════════════════
    # TOP METRICS ROW
    # ══════════════════════════════════════════════════════════════════════════
    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    c1.metric("BTC Price",     fmt(cg.get("price"), 0, "$"),
              fmt(cg.get("chg_24h"), 2, suffix="%"),
              delta_color=delta_colour(cg.get("chg_24h")))
    c2.metric("Fear & Greed",  f"{fg.get('value', '—')} — {fg.get('label', '—')}")
    c3.metric("BTC Dominance", fmt(cg.get("dominance"), 1, suffix="%"))
    c4.metric("Funding Rate",  fmt(fr.get("latest"), 4, suffix="%"))
    c5.metric("Hash Rate",     fmt(bc.get("hash_rate_eh"), 1, suffix=" EH/s"))
    c6.metric("NVT Ratio",     fmt(nvt.get("nvt_smoothed"), 1))
    c7.metric("Fed Bias",      macro.get("fed_action", "—"))

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # TABS
    # ══════════════════════════════════════════════════════════════════════════
    (tab_score, tab_price, tab_vpa, tab_onchain,
     tab_mining, tab_macro, tab_sentiment, tab_guide) = st.tabs([
        "📊 Score Breakdown", "📈 Price & Trend", "📐 Volume-Price Analysis", "🔗 On-Chain Data",
        "⛏️ Mining Economics", "🌐 Macro Liquidity", "😨 Sentiment", "📖 Signal Guide"
    ])

    # ── TAB 1 · SCORE BREAKDOWN ───────────────────────────────────────────────
    with tab_score:
        st.subheader("Signal Scorecard")

        rows = []
        for name, score in raw_scores.items():
            weight = weights.get(name, 0)
            contrib = score * weight if not pd.isna(score) else np.nan
            tag = " 🔗" if name in CORRELATED_SIGNAL_GROUP else ""
            rows.append({
                "Signal":       name + tag,
                "Score (−1→+1)": score_bar(score),
                "Weight":       f"{weight*100:.0f}%",
                "Contribution": f"{contrib:+.3f}" if not pd.isna(contrib) else "—",
                "Interpretation": notes[name],
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption("🔗 = correlated price-action / liquidity group (30d Momentum, Price vs MA200, "
                   "Funding Rate, Fear & Greed, Macro Liquidity) — these tend to move together, so "
                   "their combined weight overstates how much independent evidence they provide. "
                   "Stablecoin Supply Growth is deliberately kept outside this group.")

        fig = go.Figure()
        categories = list(raw_scores.keys())
        values     = [raw_scores[k] for k in categories]
        colours    = ["#00e676" if v > 0.1 else "#ff5252" if v < -0.1 else "#f1c40f"
                      for v in values]

        fig.add_trace(go.Bar(
            x=categories, y=values, marker_color=colours,
            text=[f"{v:+.2f}" for v in values], textposition="outside",
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.4)
        fig.add_hrect(y0=0.2, y1=1.0,  fillcolor="#00e676", opacity=0.05, line_width=0)
        fig.add_hrect(y0=-1.0, y1=-0.2, fillcolor="#ff5252", opacity=0.05, line_width=0)
        fig.update_layout(
            title="On-Chain, Macro & Volume Signal Scores (−1 Bearish → +1 Bullish)",
            yaxis=dict(range=[-1.2, 1.2], title="Score"),
            template="plotly_dark", height=420,
            xaxis=dict(tickangle=-20),
        )
        st.plotly_chart(fig, use_container_width=True)

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=composite if not pd.isna(composite) else 0,
            delta={"reference": 0, "valueformat": ".3f"},
            title={"text": f"Composite Score → <b>{verdict}</b>", "font": {"size": 18}},
            gauge={
                "axis": {"range": [-1, 1], "tickvals": [-1, -0.5, 0, 0.5, 1]},
                "bar":  {"color": colour},
                "steps": [
                    {"range": [-1, -0.4], "color": "#4a1a1a"},
                    {"range": [-0.4, -0.1], "color": "#3a2a1a"},
                    {"range": [-0.1, 0.2], "color": "#2a2a1a"},
                    {"range": [0.2, 0.5], "color": "#1a3a1a"},
                    {"range": [0.5, 1],   "color": "#1a4a1a"},
                ],
                "threshold": {"line": {"color": "white", "width": 3}, "value": composite or 0},
            },
            number={"valueformat": ".3f"},
        ))
        fig_gauge.update_layout(template="plotly_dark", height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)

    # ── TAB 2 · PRICE & TREND ─────────────────────────────────────────────────
    with tab_price:
        st.subheader("BTC/USD Price with Moving Averages")
        if not ohlcv.empty:
            close = ohlcv["Close"]
            if isinstance(close, pd.DataFrame): close = close.iloc[:, 0]
            close = close.dropna()

            ma50  = close.rolling(50).mean()
            ma200 = close.rolling(min(200, len(close))).mean()

            fig_p = go.Figure()
            fig_p.add_trace(go.Scatter(x=close.index, y=close, name="BTC/USD",
                                        line=dict(color="#f7931a", width=2)))
            fig_p.add_trace(go.Scatter(x=ma50.index, y=ma50, name="MA 50",
                                        line=dict(color="#3498db", width=1.5, dash="dot")))
            fig_p.add_trace(go.Scatter(x=ma200.index, y=ma200, name="MA 200",
                                        line=dict(color="#9b59b6", width=1.5, dash="dash")))
            fig_p.update_layout(title="BTC/USD — 1 Year", template="plotly_dark",
                                 height=400, yaxis_title="Price (USD)")
            st.plotly_chart(fig_p, use_container_width=True)

            price_now = float(close.iloc[-1])
            ma50_now  = float(ma50.dropna().iloc[-1])
            ma200_now = float(ma200.dropna().iloc[-1])
            col1, col2, col3 = st.columns(3)
            col1.metric("Current Price", f"${price_now:,.0f}")
            col2.metric("MA 50", f"${ma50_now:,.0f}",
                        f"{(price_now/ma50_now-1)*100:+.1f}% vs price",
                        delta_color=delta_colour(price_now - ma50_now))
            col3.metric("MA 200", f"${ma200_now:,.0f}",
                        f"{(price_now/ma200_now-1)*100:+.1f}% vs price",
                        delta_color=delta_colour(price_now - ma200_now))

            st.subheader("Return Summary")
            ret_data = {
                "Period": ["24h", "7 days", "30 days"],
                "Return": [
                    fmt(cg.get("chg_24h"), 2, suffix="%"),
                    fmt(cg.get("chg_7d"), 2, suffix="%"),
                    fmt(cg.get("chg_30d"), 2, suffix="%"),
                ],
            }
            st.dataframe(pd.DataFrame(ret_data), use_container_width=True, hide_index=True)
        else:
            st.warning("Price data unavailable from yfinance.")

    # ── TAB 3 · VOLUME-PRICE ANALYSIS ────────────────────────────────────────
    with tab_vpa:
        st.subheader("📐 Volume-Price Analysis")
        st.caption(
            "On-chain metrics describe network fundamentals over weeks-to-months. VPA describes "
            "exchange order-book participation over days-to-weeks — whether the current price move "
            "is backed by real volume, or is a thin move likely to fail."
        )

        if vpa:
            col1, col2, col3 = st.columns(3)
            col1.metric("5d Price Trend", fmt(vpa.get("price_trend_pct"), 1, suffix="%"))
            col2.metric("5d OBV Trend", fmt(vpa.get("obv_trend_pct"), 1, suffix="%"))
            col3.metric("Vol (5d avg ÷ 20d avg)", fmt(vpa.get("vol_ratio_5_over_20"), 2, suffix="×"))

            st.markdown(f"**Read:** {vol_note}")

            close = ohlcv["Close"]
            vol_ = ohlcv["Volume"]
            if isinstance(close, pd.DataFrame): close = close.iloc[:, 0]
            if isinstance(vol_, pd.DataFrame): vol_ = vol_.iloc[:, 0]
            close = close.dropna()
            obv_series = vpa["obv_series"]
            vwap_series = vpa["vwap20_series"]

            fig_vpa = go.Figure()
            fig_vpa.add_trace(go.Scatter(x=close.index, y=close, name="BTC/USD",
                                          line=dict(color="#f7931a", width=2)))
            fig_vpa.add_trace(go.Scatter(x=vwap_series.index, y=vwap_series, name="20d VWAP",
                                          line=dict(color="#00bcd4", width=1.5, dash="dot")))
            fig_vpa.update_layout(title="Price vs 20-Day Volume-Weighted Average Price",
                                   template="plotly_dark", height=350, yaxis_title="USD")
            st.plotly_chart(fig_vpa, use_container_width=True)

            fig_obv = go.Figure()
            fig_obv.add_trace(go.Scatter(x=obv_series.index, y=obv_series, name="OBV",
                                          line=dict(color="#69f0ae", width=2), fill="tozeroy",
                                          fillcolor="rgba(105,240,174,0.08)"))
            fig_obv.update_layout(title="On-Balance Volume (OBV) — cumulative buy/sell pressure",
                                   template="plotly_dark", height=280, yaxis_title="OBV (BTC-equivalent volume units)")
            st.plotly_chart(fig_obv, use_container_width=True)

            fig_vol = go.Figure()
            fig_vol.add_trace(go.Bar(x=vol_.index[-60:], y=vol_.iloc[-60:], name="Volume",
                                      marker_color="#7f8c8d"))
            fig_vol.add_hline(y=vpa.get("avg_vol_20", 0), line_dash="dash", line_color="#f1c40f",
                               annotation_text="20d avg volume")
            fig_vol.update_layout(title="Daily Volume — Last 60 Days",
                                   template="plotly_dark", height=250)
            st.plotly_chart(fig_vol, use_container_width=True)

            st.caption(
                "**OBV** adds the day's volume when price closes up and subtracts it when price closes "
                "down, so it tracks cumulative buying/selling pressure independent of price itself. "
                "When OBV and price move together, the trend has real participation behind it. When they "
                "diverge, that's historically a higher-risk setup. **20d VWAP** acts as a dynamic "
                "support/resistance reference many traders watch for mean-reversion entries. Note: this "
                "uses exchange-reported spot volume via yfinance, which varies in quality/coverage across "
                "venues — treat OBV as directional, not exact."
            )
        else:
            st.warning("Insufficient price/volume history to compute VPA metrics.")

    # ── TAB 4 · ON-CHAIN DATA ─────────────────────────────────────────────────
    with tab_onchain:
        col_l, col_r = st.columns(2)

        with col_l:
            st.subheader("Network Health")
            st.metric("Hash Rate",        fmt(bc.get("hash_rate_eh"), 2, suffix=" EH/s"))
            st.metric("Difficulty",       fmt(bc.get("difficulty"), 0))
            st.metric("Transactions 24h", fmt(bc.get("n_tx_24h"), 0))
            st.metric("BTC Sent 24h",     fmt(bc.get("total_btc_sent"), 1, suffix=" BTC"))
            st.metric("Blocks Mined 24h", fmt(bc.get("blocks_mined_24h"), 0))
            st.metric("Mempool Size",     fmt(bc.get("mempool_size"), 0, suffix=" bytes"))
            st.metric("Active Addresses", fmt(bc.get("active_addr_now"), 0))

        with col_r:
            st.subheader("Market & Valuation")
            st.metric("Market Cap",       fmt(cg.get("market_cap"), 0, "$"))
            st.metric("24h Volume",       fmt(cg.get("volume_24h"), 0, "$"))
            st.metric("On-Chain Volume (USD, 24h)", fmt(bc.get("tx_volume_usd_now"), 0, "$"))
            st.metric("NVT Ratio (30d-smoothed)",   fmt(nvt.get("nvt_smoothed"), 1))
            nvt_trend_label = "NVT 30d Trend" if nvt.get("nvt_trend_is_true") else "Vol-only NVT proxy trend*"
            st.metric(nvt_trend_label, fmt(nvt.get("nvt_trend_pct"), 1, suffix="%"))
            st.metric("BTC Dominance",    fmt(cg.get("dominance"), 2, suffix="%"))
            st.metric("Circulating Supply", fmt(cg.get("circulating"), 0, suffix=" BTC"))
            if not nvt.get("nvt_trend_is_true", False) and not pd.isna(nvt.get("nvt_trend_pct", np.nan)):
                st.caption("*Market-cap history fetch failed — this trend reflects on-chain volume drift only, not true NVT.")

        st.markdown("---")
        st.caption(
            "**NVT Ratio** = Market Cap ÷ on-chain USD transfer volume (30-day average, smoothed). "
            "Think of it as an on-chain 'P/E ratio.' The trend is computed from real daily market-cap "
            "history when available; if that fetch fails, the app falls back to a volume-only proxy "
            "and labels it as such rather than presenting it as true NVT trend."
        )

        hr_hist = bc.get("hash_rate_history", [])
        if hr_hist:
            df_hr = pd.DataFrame(hr_hist)
            df_hr["date"] = pd.to_datetime(df_hr["date"])
            fig_hr = px.area(df_hr, x="date", y="value",
                              labels={"value": "EH/s", "date": ""},
                              title="Hash Rate — 30 Days (EH/s)",
                              color_discrete_sequence=["#f7931a"])
            fig_hr.update_layout(template="plotly_dark", height=280)
            st.plotly_chart(fig_hr, use_container_width=True)

        aa_hist = bc.get("active_addr_history", [])
        if aa_hist:
            df_aa = pd.DataFrame(aa_hist)
            df_aa["date"] = pd.to_datetime(df_aa["date"])
            fig_aa = px.area(df_aa, x="date", y="value",
                              labels={"value": "Addresses", "date": ""},
                              title="Active Addresses — 30 Days",
                              color_discrete_sequence=["#3498db"])
            fig_aa.update_layout(template="plotly_dark", height=280)
            st.plotly_chart(fig_aa, use_container_width=True)

        tv_hist = bc.get("tx_volume_usd_history", [])
        if tv_hist:
            df_tv = pd.DataFrame(tv_hist)
            df_tv["date"] = pd.to_datetime(df_tv["date"])
            fig_tv = px.area(df_tv, x="date", y="value",
                              labels={"value": "USD", "date": ""},
                              title="Estimated On-Chain Transfer Volume — 90 Days (USD)",
                              color_discrete_sequence=["#9b59b6"])
            fig_tv.update_layout(template="plotly_dark", height=280)
            st.plotly_chart(fig_tv, use_container_width=True)

    # ── TAB 5 · MINING ECONOMICS ───────────────────────────────────────────
    with tab_mining:
        st.subheader("⛏️ Mining Cost & Network Economics")
        st.caption(
            "Electricity-only model. Excludes ASIC depreciation, cooling, and pool fees — read the "
            "breakeven figure as a soft capitulation floor, not full miner P&L."
        )

        if mining:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Network Power Draw", fmt(mining.get("network_power_gw"), 2, suffix=" GW"))
            col2.metric("Est. Daily Energy Cost", fmt(mining.get("daily_energy_cost_usd"), 0, "$"))
            col3.metric("Est. Breakeven Cost / BTC", fmt(mining.get("breakeven_per_btc"), 0, "$"))
            col4.metric(
                "Spot vs Breakeven",
                fmt(mining.get("margin_pct"), 1, suffix="%"),
                delta_color=delta_colour(mining.get("margin_pct"))
            )

            margin = mining.get("margin_pct", np.nan)
            if not pd.isna(margin):
                if margin > 50:
                    st.success(
                        f"Spot price is **{margin:.0f}% above** estimated electricity-only breakeven — "
                        "wide margin, historically supportive of continued hash rate growth and less "
                        "forced-selling pressure."
                    )
                elif margin > 0:
                    st.info(
                        f"Spot price is **{margin:.0f}% above** estimated breakeven — profitable but "
                        "moderate margins. Watch for hash rate deceleration if price weakens further."
                    )
                else:
                    st.warning(
                        f"Spot price is **{abs(margin):.0f}% below** estimated electricity-only breakeven — "
                        "a level that historically pressures less-efficient miners toward capitulation."
                    )

            st.markdown("---")
            st.markdown(f"""
**Model inputs & assumptions:**
- Network hash rate: **{fmt(bc.get('hash_rate_eh'), 1)} EH/s**
- Assumed fleet efficiency: **{fleet_efficiency:.1f} J/TH**
- Assumed electricity cost: **${electricity_cost:.2f}/kWh**
- Estimated BTC issued per day: **{fmt(mining.get('btc_issued_per_day'), 1)} BTC**
- *(Electricity-only — excludes capex/depreciation, cooling, pool fees, curtailment strategies.)*
""")
        else:
            st.warning("Mining cost model unavailable — hash rate data could not be fetched.")

        st.markdown("---")
        st.subheader("Miner Revenue & Puell Multiple")

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Daily Miner Revenue (USD)", fmt(bc.get("miners_revenue_usd_24h"), 0, "$"))
        col_b.metric("Puell Multiple (proxy)", fmt(puell.get("value"), 2))
        col_c.metric("Puell Trend (recent)", fmt(puell.get("trend_pct"), 1, suffix="%"))

        mr_hist = bc.get("miner_revenue_history", [])
        if mr_hist:
            df_mr = pd.DataFrame(mr_hist)
            df_mr["date"] = pd.to_datetime(df_mr["date"])
            fig_mr = px.area(df_mr, x="date", y="value",
                              labels={"value": "USD", "date": ""},
                              title="Daily Miner Revenue — 1 Year (USD)",
                              color_discrete_sequence=["#f1c40f"])
            fig_mr.update_layout(template="plotly_dark", height=300)
            st.plotly_chart(fig_mr, use_container_width=True)

        st.caption(
            "**Puell Multiple** = daily miner revenue ÷ its trailing average. Historically, readings "
            "**below ~0.5** coincide with miner-capitulation bottoms; readings **above ~4** coincide with "
            "cycle-top euphoria. Unlike NVT's trend, this one is not affected by a constant-denominator "
            "issue, since both the ratio and its trailing average are drawn from the same revenue series."
        )

        st.markdown("---")
        st.subheader("Supply Context")
        circ = cg.get("circulating", np.nan)
        max_supply = cg.get("max_supply", 21_000_000)
        issued_per_day = mining.get("btc_issued_per_day", np.nan) if mining else np.nan
        col_x, col_y, col_z = st.columns(3)
        col_x.metric("Circulating Supply", fmt(circ, 0, suffix=" BTC"))
        col_y.metric("Remaining to Mine", fmt((max_supply or 21_000_000) - (circ or 0), 0, suffix=" BTC"))
        col_z.metric("Est. New Supply / Day", fmt(issued_per_day, 1, suffix=" BTC"))
        st.caption(f"Next halving expected around **{NEXT_HALVING_YEAR}**, cutting new issuance roughly in half again.")

    # ── TAB 6 · MACRO LIQUIDITY  (NEW) ────────────────────────────────────────
    with tab_macro:
        st.subheader("🌐 Macro Liquidity")
        st.caption(
            "BTC behaves like a high-beta risk asset most of the time, and macro liquidity conditions "
            "can override on-chain signals in the short term. This tab makes that channel explicit "
            "instead of leaving it as an unmeasured caveat: it blends a US Fed Taylor Rule estimate "
            "(ported from the forex fundamental scanner), the 20-day DXY trend, and the real 10Y yield "
            "level into one liquidity reading."
        )

        col1, col2, col3 = st.columns(3)
        col1.metric("Fed Bias", macro.get("fed_action", "—"),
                    fmt(macro.get("fed_score"), 2))
        col2.metric("DXY 20d Trend", fmt(macro.get("dxy_trend_pct"), 1, suffix="%"))
        col3.metric("Real 10Y Yield", fmt(macro.get("real_yield_now"), 2, suffix="%"))

        st.markdown(f"**Read:** {macro_note}")

        st.markdown("---")
        st.markdown("**Sub-component contributions to the Macro Liquidity score**")
        sub_df = pd.DataFrame([
            {"Component": "Fed Taylor bias (inverted: hawkish = headwind)",
             "Weight in blend": "50%",
             "Score": fmt(macro.get("fed_component"), 2)},
            {"Component": "DXY 20d trend (rising = tightening)",
             "Weight in blend": "30%",
             "Score": fmt(macro.get("dxy_component"), 2)},
            {"Component": "Real 10Y yield level (higher = headwind)",
             "Weight in blend": "20%",
             "Score": fmt(macro.get("yield_component"), 2)},
        ])
        st.dataframe(sub_df, use_container_width=True, hide_index=True)
        st.caption(
            "These three sub-component weights (50/30/20) are internal to the Macro Liquidity signal "
            "and are separate from the overall scorecard weight this signal receives in the composite — "
            "adjust the scorecard weight in the '🎛️ Composite Score Weights' panel above; these "
            "internal proportions are fixed in code."
        )

        dxy_series = macro_in.get("dxy_series", pd.Series(dtype=float))
        if not dxy_series.empty:
            fig_dxy = go.Figure()
            fig_dxy.add_trace(go.Scatter(x=dxy_series.index, y=dxy_series.values, name="DXY",
                                          line=dict(color="#3498db", width=2)))
            fig_dxy.update_layout(title="US Dollar Index (DXY) — 6 Months",
                                   template="plotly_dark", height=300, yaxis_title="Index")
            st.plotly_chart(fig_dxy, use_container_width=True)

        ten_y_series = macro_in.get("ten_y_series", pd.Series(dtype=float))
        cpi_series   = macro_in.get("cpi_series", pd.Series(dtype=float))
        if not ten_y_series.empty and not cpi_series.empty:
            merged = pd.concat([ten_y_series.rename("ten_y"), cpi_series.rename("cpi")], axis=1).dropna()
            merged["real_yield"] = merged["ten_y"] - merged["cpi"]
            fig_ry = go.Figure()
            fig_ry.add_trace(go.Scatter(x=merged.index, y=merged["real_yield"], name="Real 10Y Yield",
                                         line=dict(color="#e67e22", width=2), fill="tozeroy",
                                         fillcolor="rgba(230,126,34,0.08)"))
            fig_ry.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.4)
            fig_ry.update_layout(title="US Real 10Y Yield (10Y nominal − CPI YoY)",
                                  template="plotly_dark", height=280, yaxis_title="%")
            st.plotly_chart(fig_ry, use_container_width=True)

        st.markdown("---")
        st.subheader("💵 Stablecoin Supply Growth")
        st.caption(
            "Aggregate USDT + USDC + DAI market cap (≈ circulating supply, pegged ~1:1) as a dry-powder "
            "proxy. Tracked separately from the Macro Liquidity blend above because it is driven by "
            "issuer/exchange minting and redemption decisions rather than Fed policy or FX markets — "
            "a genuinely different vote, not the same liquidity story from another angle."
        )
        if stablecoins:
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Total Stablecoin Mcap", fmt(stablecoins.get("total_now"), 0, "$"))
            col_b.metric("30d Ago", fmt(stablecoins.get("total_30d"), 0, "$"))
            col_c.metric("30d Growth", fmt(stablecoins.get("growth_pct"), 1, suffix="%"),
                         delta_color=delta_colour(stablecoins.get("growth_pct")))
            st.markdown(f"**Read:** {sc_note}")

            per_coin = stablecoins.get("per_coin", {})
            if per_coin:
                pc_rows = [{
                    "Stablecoin": sym,
                    "Market Cap Now": fmt(vals["now"], 0, "$"),
                    "Market Cap 30d Ago": fmt(vals["ago_30d"], 0, "$"),
                    "30d Growth": fmt((vals["now"] - vals["ago_30d"]) / vals["ago_30d"] * 100, 1, suffix="%")
                                  if vals["ago_30d"] else "—",
                } for sym, vals in per_coin.items()]
                st.dataframe(pd.DataFrame(pc_rows), use_container_width=True, hide_index=True)
        else:
            st.warning("Stablecoin supply data unavailable from CoinGecko.")

        st.info(
            "**Caveats:** Fed Taylor Rule scores are a heuristic estimate ported from the forex "
            "fundamental scanner, not a forecast verified against FOMC outcomes. DXY and real-yield "
            "components are simple trend/level reads, not full rates-market models. Stablecoin supply "
            "also grows for reasons unrelated to crypto risk appetite — e.g. treasury cash parking or "
            "cross-border settlement use — so treat it as a noisy proxy, not a confirmed 'money about "
            "to buy BTC' signal."
        )

    # ── TAB 7 · SENTIMENT ─────────────────────────────────────────────────────
    with tab_sentiment:
        st.subheader("Fear & Greed Index")
        fg_val   = fg.get("value", 50)
        fg_label = fg.get("label", "—")

        fg_colour = (
            "#00e676" if fg_val <= 25 else
            "#69f0ae" if fg_val <= 45 else
            "#f1c40f" if fg_val <= 55 else
            "#ff7043" if fg_val <= 75 else
            "#d50000"
        )
        fig_fg = go.Figure(go.Indicator(
            mode="gauge+number",
            value=fg_val,
            title={"text": f"<b>{fg_label}</b>", "font": {"size": 20}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar":  {"color": fg_colour},
                "steps": [
                    {"range": [0, 25], "color": "#1a4a1a"},
                    {"range": [25, 45], "color": "#1a3a1a"},
                    {"range": [45, 55], "color": "#2a2a1a"},
                    {"range": [55, 75], "color": "#3a2a1a"},
                    {"range": [75, 100], "color": "#4a1a1a"},
                ],
                "threshold": {"line": {"color": "white", "width": 3}, "value": fg_val},
            },
        ))
        fig_fg.update_layout(template="plotly_dark", height=300)
        st.plotly_chart(fig_fg, use_container_width=True)

        fg_hist = fg.get("history", [])
        if fg_hist:
            df_fg = pd.DataFrame(fg_hist)
            df_fg["date"]  = pd.to_datetime(df_fg["date"])
            df_fg["value"] = df_fg["value"].astype(int)
            fig_fgh = px.bar(df_fg, x="date", y="value",
                              color="value", color_continuous_scale=[
                                  [0.0, "#d50000"], [0.25, "#ff5252"],
                                  [0.5, "#f1c40f"],
                                  [0.75, "#69f0ae"], [1.0, "#00e676"]
                              ],
                              title="Fear & Greed — 30 Day History",
                              labels={"value": "Index", "date": ""})
            fig_fgh.add_hline(y=20, line_dash="dot", line_color="#00e676",
                               annotation_text="Extreme Fear")
            fig_fgh.add_hline(y=80, line_dash="dot", line_color="#d50000",
                               annotation_text="Extreme Greed")
            fig_fgh.update_layout(template="plotly_dark", height=300,
                                   coloraxis_showscale=False)
            st.plotly_chart(fig_fgh, use_container_width=True)

        st.subheader("Funding Rate (Binance BTCUSDT Perpetual)")
        fr_hist = fr.get("history", [])
        if fr_hist:
            df_fr = pd.DataFrame({"period": range(len(fr_hist)), "rate": fr_hist})
            fig_fr = px.bar(df_fr, x="period", y="rate",
                             color="rate",
                             color_continuous_scale=["#d50000", "#f1c40f", "#00e676"],
                             title="Funding Rate History (last 8 periods)",
                             labels={"rate": "Rate %", "period": "Period"})
            fig_fr.add_hline(y=0, line_color="white", opacity=0.4)
            fig_fr.update_layout(template="plotly_dark", height=250,
                                  coloraxis_showscale=False)
            st.plotly_chart(fig_fr, use_container_width=True)
        else:
            st.info("Funding rate history unavailable (Binance API may be temporarily unreachable).")

        col1, col2 = st.columns(2)
        col1.metric("Latest Funding Rate", fmt(fr.get("latest"), 4, suffix="%"))
        col2.metric("8-Period Avg Rate",   fmt(fr.get("avg_8"),  4, suffix="%"))

    # ── TAB 8 · SIGNAL GUIDE ─────────────────────────────────────────────────
    with tab_guide:
        st.subheader("Chapter 9 Signal Reference Guide")

        st.markdown("""
**How the composite score works:**

| Score Range | Verdict | Meaning |
|---|---|---|
| +0.50 to +1.00 | ★★★★★ Strong Bullish | Multiple signals aligned positively |
| +0.20 to +0.49 | ★★★★☆ Bullish | Majority of signals positive |
| −0.10 to +0.19 | ★★★☆☆ Neutral | Mixed signals — wait for clarity |
| −0.40 to −0.11 | ★★☆☆☆ Bearish | Majority of signals negative |
| −1.00 to −0.41 | ★☆☆☆☆ Strong Bearish | Multiple signals negative |
""")

        st.markdown("""
---
**Individual Signal Interpretation:**

**Fear & Greed Index** 🔗 — Contrarian signal. Extreme Fear (<20) historically marks accumulation zones;
Extreme Greed (>80) marks distribution zones. Never trade on this alone — confirm with on-chain and volume.

**Hash Rate** — Measured via a 30-day linear-regression slope rather than comparing only the first
and last week, so the middle of the window actually counts and single noisy days at either edge matter
less. Rising hash rate signals miner confidence; falling hash rate can signal miner stress or upcoming
capitulation.

**Active Addresses** — A *directionally* useful but noisy proxy for real usage, not a clean adoption
read. It's distorted by exchange custodial-wallet batching, UTXO consolidation sweeps, address reuse,
and activity migrating off-chain to Layer 2 / Lightning. Weight it as one vote among many, not
confirmation on its own.

**Funding Rate** 🔗 — Persistently very positive = overcrowded longs vulnerable to a liquidation
cascade. Very negative = short squeeze potential. Thresholds favor real extremes over the first sign
of positive funding, since moderately positive funding is common and unremarkable during genuine
bull trends.

**Price vs MA200** 🔗 — The macro trend filter. Above MA200 = bull market structure; below = bear
market structure.

**30d Momentum** 🔗 — Trend direction over a meaningful period, avoiding daily noise.

**BTC Dominance** — *Not* a reliable directional signal for BTC on its own. Rising dominance shows up
in early bull markets (capital flowing into BTC first) **and** in bear markets (capital fleeing
riskier alts faster than BTC). Falling dominance shows up in "altseason" **and** can simply mean alts
are getting hit less hard. Scored as a small regime-flag (±0.3 max) rather than a directional vote,
with a reduced weight in the default mix.

**NVT Ratio (trend-aware)** — On-chain "valuation multiple." Low = usage justifies valuation (cheap);
high = valuation has detached from usage (expensive). The trend calculation uses real daily market-cap
history so the trend reflects genuine NVT movement; if market-cap history can't be fetched, the app
falls back to the volume-only proxy and says so explicitly rather than presenting it as true NVT
trend. Static NVT thresholds were calibrated years ago — as activity moves off-chain, the "normal"
range likely drifts — so treat NVT as one input among several, not a standalone call.

**Puell Multiple (trend-aware)** — Compares today's miner revenue to its trailing average. Extreme
lows have marked capitulation bottoms; extreme highs have marked euphoric tops. This trend calculation
does not share NVT's old constant-denominator issue, since both the ratio and its baseline come from
the same revenue series.

**Volume-Price Analysis** — Does trading volume confirm or contradict the price trend? Price and OBV
moving together = real participation. Divergence is historically a higher-risk setup, regardless of
what on-chain fundamentals say — this is the tactical layer that sits on top of the slower-moving
fundamental picture. Of all the signals here, this is one of the more directly defensible, since it's
standard technical-analysis logic rather than a crypto-specific heuristic — though exchange-reported
volume quality varies, so treat it as directional.

**Macro Liquidity** 🔗 *(new)* — Blends a US Fed Taylor Rule estimate (ported from the forex
fundamental scanner), the 20-day DXY trend, and the real 10Y yield level into one liquidity reading.
Hawkish Fed / rising dollar / high real yields = headwind for BTC as a risk asset; the reverse = a
tailwind. This is tagged as part of the correlated price-action/liquidity group because it operates
through the same risk-appetite channel as Price vs MA200, Funding Rate, and Fear & Greed, even though
its raw inputs (rates, FX) are different data sources from theirs. It is forward-looking (policy
trajectory) rather than backward-looking (price that already happened), which is its main value-add
over the other four — but it should not be treated as a fifth independent vote.

**Stablecoin Supply Growth** *(new)* — Aggregate USDT + USDC + DAI market cap growth over 30 days, as
a dry-powder / liquidity-inflow proxy. Deliberately kept OUTSIDE the correlated group: minting and
redemption decisions are driven by issuers and exchanges responding to fiat on/off-ramp demand, which
can lead or lag price rather than mechanically restating it. Still a noisy proxy — supply also grows
for reasons unrelated to crypto risk appetite, like treasury cash parking or cross-border settlement —
so treat it as one vote, not confirmed incoming demand.

**Mining Cost / Breakeven** — Electricity-only estimate of network production cost per BTC. Excludes
capex/depreciation — read it as a soft floor, not full miner P&L. When spot falls toward or below this
line, more miners historically approach capitulation.

**Composite score / linear weighting** 🔗 — The score sums weight × signal across all twelve
components as if each were independent evidence. In reality, the 🔗-tagged signals (Fear & Greed,
Funding Rate, Price vs MA200, 30d Momentum, Macro Liquidity) substantially correlate with each other
because they're all reading the same underlying price-action / liquidity backdrop from different
angles — so the composite partially double-counts that one piece of information. Treat the composite
as a **structured checklist** that organizes the inputs you should be looking at, not a statistically
validated predictive model.

**Editable weights** — Every weight in the '🎛️ Composite Score Weights' panel can be adjusted. Moving
any single slider automatically rebalances all the others proportionally so the total always sums to
100% — this preserves the relative balance among the signals you didn't touch, rather than flattening
everything toward equal weight over repeated edits. Use the reset button to return to the defaults
shown above at any time.
""")

        st.info(
            "**Key reminders:**\n\n"
            "• No single signal is sufficient — use this composite as a confluence tool, not a trading signal\n"
            "• On-chain data is most informative at extremes; trend matters as much as level\n"
            "• Volume-price analysis adds a tactical timing layer on top of the slower fundamental picture\n"
            "• Macro liquidity conditions can override on-chain signals in the short term — now measured "
            "explicitly via the Macro Liquidity signal instead of being an unmeasured caveat\n"
            "• Score weights here are analyst priors, not backtested coefficients — adjust and stress-test them\n"
            "• 🔗-tagged signals correlate with each other (price-action / liquidity based) — don't read the "
            "composite as twelve independent votes\n"
            "• BTC dominance is a regime flag, not a reliable directional call on BTC itself\n"
            "• Active-address counts are a noisy adoption proxy, not a clean usage read\n"
            "• Stablecoin supply growth is a noisy dry-powder proxy, not confirmed incoming demand\n"
            "• Mining cost estimates are electricity-only approximations, not exact miner P&L\n"
            "• For deeper metrics (MVRV, NUPL, realised price, exchange reserves) consider Glassnode or CryptoQuant\n\n"
            "**This tool is for research and education — it is not financial advice.**"
        )
