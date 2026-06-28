import streamlit as st
import requests
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, List
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
HEADERS = {"User-Agent": "Mozilla/5.0 CryptoScanner/3.3"}

BLOCK_REWARD_BTC   = 3.125
BLOCKS_PER_DAY_AVG = 144
NEXT_HALVING_YEAR  = 2028

DXY_TICKER = "DX-Y.NYB"
FRED_FED_FUNDS   = "DFEDTARU"
FRED_CPI_USD_YOY = "CPALTT01USM659N"
FRED_UNRATE      = "UNRATE"
FRED_GDP_GROWTH  = "A191RL1Q225SBEA"
FRED_10Y         = "DGS10"
INFLATION_TARGET = 2.0

STABLECOIN_IDS = {
    "USDT": "tether",
    "USDC": "usd-coin",
    "DAI":  "dai",
}

# ── Liquidity Dashboard FRED series (v3.4) ──────────────────────────────────
# All confirmed live FRED tickers. Net liquidity proxy follows the standard
# market convention: Fed Balance Sheet − Reverse Repo − Treasury General
# Account (see e.g. the widely-cited WALCL-RRPONTSYD-WTREGEN formula).
FRED_FED_BALANCE_SHEET = "WALCL"       # Fed total assets, weekly, $mm
FRED_REVERSE_REPO      = "RRPONTSYD"   # ON RRP usage, daily, $bn
FRED_TGA               = "WTREGEN"     # Treasury General Account, weekly, $mm
FRED_M2                = "M2SL"        # M2 money stock, monthly, $bn
FRED_TOTAL_DEBT        = "GFDEBTN"     # Total US federal debt outstanding, quarterly, $mm
                                        # (proxy for Treasury issuance trend — FRED has no
                                        # single clean "daily net issuance" series; GFDEBTN's
                                        # quarter-over-quarter change is used as that proxy
                                        # and is explicitly labeled as such in the UI)

# Global liquidity is approximated as Fed + ECB balance sheets, FX-converted
# to USD — a simplified version of the standard "Global Net Liquidity"
# community proxy (e.g. as popularized by Michael Howell / CrossBorder
# Capital and widely replicated on TradingView, which typically also adds
# BoJ + PBoC legs). It is NOT an official series from any single source,
# and is clearly labeled as an approximation in the UI.
FRED_ECB_ASSETS  = "ECBASSETSW"   # ECB total assets, weekly, EUR mm
ECB_FX_TICKER    = "EURUSD=X"     # EUR/USD for conversion
# No reliable free daily BoJ/PBoC series via FRED/yfinance — those legs are
# OMITTED from the v3.4 global liquidity proxy rather than estimated; see
# the Liquidity tab caveat box for the explicit disclosure.


# ── ETF Dashboard (v3.4) ────────────────────────────────────────────────────
# IMPORTANT: there is no free, public, programmatic API for daily spot BTC
# ETF flows. Farside Investors and SoSoValue publish the numbers people
# actually cite, but both are scrape-only HTML tables with no stable free
# API and no confirmed redistribution license. Rather than fabricate flow
# figures (these numbers move markets and people act on them), this
# dashboard:
#   1. Computes what IS independently verifiable from a real, fetchable
#      source: aggregate AUM / shares-outstanding trend for the largest
#      spot BTC ETFs via Yahoo Finance (yfinance), which reliably provides
#      price and volume data for these tickers.
#   2. Clearly labels which numbers are auto-fetched market data vs.
#      DERIVED ESTIMATES (e.g. implied flow ≈ AUM change adjusted for BTC
#      price change), and never presents an estimate as a reported flow.
#   3. Provides a manual-entry override so a user who has today's real
#      Farside/SoSoValue flow numbers can paste them in for an accurate
#      read, visually distinguished from the auto-fetched estimate.
ETF_TICKERS = {
    "IBIT": "iShares Bitcoin Trust (BlackRock)",
    "FBTC": "Fidelity Wise Origin Bitcoin Fund",
    "GBTC": "Grayscale Bitcoin Trust",
    "ARKB": "ARK 21Shares Bitcoin ETF",
    "BITB": "Bitwise Bitcoin ETF",
}


# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL GROUPS  (v3.3 reorganization)
#
# Every signal now belongs to exactly one of four groups. This replaces the
# old single "correlated vs not" flag with an explicit functional taxonomy:
#
#   VALUATION   — is BTC cheap or expensive relative to its own usage/cost?
#   TREND       — which direction is price actually moving, and is that
#                 move confirmed?
#   SENTIMENT   — how positioned/emotional are participants right now?
#                 (this is what lets us estimate trend STAGE, not just
#                 direction — see compute_trend_stage() below)
#   CONTEXT     — signals that are real information but are NOT a clean
#                 directional vote on BTC's own price (regime flags,
#                 noisy usage proxies, external liquidity drivers, flow
#                 proxies). Kept separate so they don't get silently
#                 averaged in with signals that ARE direct votes.
#
# Composite score weighting is unchanged in spirit (the UI sliders still
# rebalance to 100%) — what's new is that the Score Breakdown tab now
# renders three/four distinct scorecards instead of one flat list, and a
# derived "Trend Stage" readout sits on top, explicitly built from TREND +
# VALUATION + SENTIMENT together (see compute_trend_stage()).
# ══════════════════════════════════════════════════════════════════════════════

GROUP_VALUATION = "Valuation"
GROUP_TREND     = "Trend Direction"
GROUP_SENTIMENT = "Sentiment / Positioning"
GROUP_CONTEXT   = "Context / Flow"

SIGNAL_GROUPS = {
    "NVT Ratio":                  GROUP_VALUATION,
    "Puell Multiple":             GROUP_VALUATION,
    "Mining Cost Margin":         GROUP_VALUATION,
    "Percent Supply in Profit":   GROUP_VALUATION,   # NEW v3.4

    "Price vs MA200":             GROUP_TREND,
    "30d Momentum":               GROUP_TREND,
    "Hash Rate":                  GROUP_TREND,
    "Volume Confirmation":        GROUP_TREND,

    "Fear & Greed":                GROUP_SENTIMENT,
    "Funding Rate":                GROUP_SENTIMENT,

    "BTC Dominance":               GROUP_CONTEXT,
    "Active Addresses":            GROUP_CONTEXT,
    "Stablecoin Supply Growth":    GROUP_CONTEXT,
    "Macro Liquidity":             GROUP_CONTEXT,
    "Fed Net Liquidity":           GROUP_CONTEXT,   # NEW v3.4 — Liquidity Dashboard headline metric
    "ETF Flow Pressure":           GROUP_CONTEXT,   # NEW v3.4 — ETF Dashboard headline metric
}

GROUP_ORDER = [GROUP_VALUATION, GROUP_TREND, GROUP_SENTIMENT, GROUP_CONTEXT]

GROUP_COLOURS = {
    GROUP_VALUATION: "#9b59b6",
    GROUP_TREND:     "#3498db",
    GROUP_SENTIMENT: "#e67e22",
    GROUP_CONTEXT:   "#7f8c8d",
}

GROUP_ICONS = {
    GROUP_VALUATION: "💰",
    GROUP_TREND:     "📈",
    GROUP_SENTIMENT: "🎭",
    GROUP_CONTEXT:   "🌐",
}

GROUP_BLURB = {
    GROUP_VALUATION: "Is BTC cheap or expensive relative to its own usage and production cost?",
    GROUP_TREND:     "Which direction is price actually moving, and is that move confirmed by hash rate / volume?",
    GROUP_SENTIMENT: "How positioned and emotional are participants right now? Feeds trend-stage maturity.",
    GROUP_CONTEXT:   "Real information, but not a clean directional vote on BTC's own price — regime flags, noisy proxies, and external liquidity drivers.",
}


# ══════════════════════════════════════════════════════════════════════════════
# SCORE WEIGHTS — analyst priors, NOT backtested coefficients.
# Unchanged numerically from v3.2 — only the grouping/labeling above changes.
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_WEIGHTS = {
    # v3.4 note: two new weighted signals were added (Percent Supply in
    # Profit, ETF Flow Pressure). To make room without re-deriving the
    # whole scheme, every pre-existing weight below was trimmed by
    # roughly the same proportion it was trimmed in the v3.2 rebalance
    # (i.e. the trims are spread across all prior signals rather than
    # taken from one group), so relative balance among v3.1/v3.2/v3.3
    # signals is preserved. Sum is still 100 by construction; the UI
    # re-derives fractions at render time regardless.
    "Fear & Greed":              10,   # was 12
    "Hash Rate":                  9,   # was 11
    "Active Addresses":           9,   # was 11
    "Funding Rate":                8,   # was 10
    "Price vs MA200":              8,   # was 10
    "30d Momentum":                6,   # was 7
    "BTC Dominance":                3,   # was 4
    "NVT Ratio":                    7,   # was 9
    "Puell Multiple":               4,   # was 5
    "Volume Confirmation":          7,   # was 8
    "Macro Liquidity":              7,   # was 8
    "Stablecoin Supply Growth":     4,   # was 5
    "Percent Supply in Profit":     9,   # NEW v3.4 — strong valuation-extremity signal,
                                          # historically one of the cleanest cycle markers
    "ETF Flow Pressure":            9,   # NEW v3.4 — flows now dominate marginal BTC demand
}
# Weights above sum to 100 by construction.

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
        out["hash_rate_eh"]      = s.get("hash_rate", np.nan) / 1e9
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
        rates = [float(d["fundingRate"]) * 100 for d in data]
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
    """Daily BTC market cap history from CoinGecko, used to build a TRUE NVT
    series (real mcap ÷ real on-chain volume per day)."""
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
    """DXY (US Dollar Index) daily history via yfinance."""
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
    """Aggregate market cap (≈ circulating supply) for USDT, USDC, DAI from
    CoinGecko, plus 30-day history per coin to measure supply GROWTH."""
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
# LIQUIDITY DASHBOARD DATA  (NEW in v3.4)
#
# Everything below is fetched from FRED (already have fetch_fred_series) plus
# yfinance for the EUR/USD cross used to convert ECB assets to USD. All
# series are real, confirmed-live FRED tickers (see constants section).
# Net liquidity and global liquidity are clearly-labeled APPROXIMATIONS
# using the standard community formula, not an official published series.
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_liquidity_dashboard_data() -> Dict:
    """Pulls every series needed for the Liquidity Dashboard tab in one
    cached call. Returns raw series (for charting) plus latest/trend
    scalars (for the scorecard)."""
    fed_bs   = fetch_fred_series(FRED_FED_BALANCE_SHEET)   # $mm, weekly
    rrp      = fetch_fred_series(FRED_REVERSE_REPO)        # $bn, daily
    tga      = fetch_fred_series(FRED_TGA)                 # $mm, weekly
    m2       = fetch_fred_series(FRED_M2)                  # $bn, monthly
    debt     = fetch_fred_series(FRED_TOTAL_DEBT)           # $mm, quarterly
    ecb      = fetch_fred_series(FRED_ECB_ASSETS)           # EUR mm, weekly

    eurusd = pd.Series(dtype=float)
    try:
        fx = yf.download(ECB_FX_TICKER, period="2y", interval="1d",
                          auto_adjust=True, progress=False)
        if not fx.empty:
            if isinstance(fx.columns, pd.MultiIndex):
                fx.columns = fx.columns.get_level_values(0)
            s = fx["Close"] if "Close" in fx.columns else pd.Series(dtype=float)
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]
            eurusd = s.dropna()
    except Exception:
        pass

    def _latest(s):
        v = s.dropna()
        return float(v.iloc[-1]) if len(v) else np.nan

    def _n_ago_periods(s, n):
        v = s.dropna()
        return float(v.iloc[-(n+1)]) if len(v) >= n + 1 else np.nan

    # Fed Net Liquidity (USD bn) = WALCL($mm) - RRP($bn*1000... careful with units)
    # Units check: WALCL & TGA are reported in $millions; RRPONTSYD is in $billions.
    # Convert everything to $ billions for the net-liquidity series.
    net_liq_series = pd.Series(dtype=float)
    if not fed_bs.empty and not rrp.empty and not tga.empty:
        df = pd.concat([
            (fed_bs / 1000.0).rename("fed_bn"),
            rrp.rename("rrp_bn"),
            (tga / 1000.0).rename("tga_bn"),
        ], axis=1).ffill().dropna()
        if not df.empty:
            net_liq_series = (df["fed_bn"] - df["rrp_bn"] - df["tga_bn"])

    fed_bs_now_bn = _latest(fed_bs) / 1000.0 if not pd.isna(_latest(fed_bs)) else np.nan
    rrp_now_bn    = _latest(rrp)
    tga_now_bn    = _latest(tga) / 1000.0 if not pd.isna(_latest(tga)) else np.nan
    net_liq_now   = float(net_liq_series.iloc[-1]) if len(net_liq_series) else np.nan
    net_liq_4w_ago = float(net_liq_series.iloc[-29]) if len(net_liq_series) >= 29 else np.nan
    net_liq_trend_pct = (
        (net_liq_now - net_liq_4w_ago) / abs(net_liq_4w_ago) * 100
        if not pd.isna(net_liq_now) and not pd.isna(net_liq_4w_ago) and net_liq_4w_ago != 0
        else np.nan
    )

    m2_now = _latest(m2)
    m2_yoy = np.nan
    m2v = m2.dropna()
    if len(m2v) >= 13:
        m2_yoy = float((m2v.iloc[-1] - m2v.iloc[-13]) / abs(m2v.iloc[-13]) * 100)

    debt_now = _latest(debt) / 1000.0 if not pd.isna(_latest(debt)) else np.nan  # -> $bn
    debt_qoq_pct = np.nan
    dv = debt.dropna()
    if len(dv) >= 2:
        debt_qoq_pct = float((dv.iloc[-1] - dv.iloc[-2]) / abs(dv.iloc[-2]) * 100)

    # Global liquidity proxy: Fed + ECB (USD-converted). BoJ/PBoC omitted —
    # see constants section + UI caveat for why.
    global_liq_series = pd.Series(dtype=float)
    if not fed_bs.empty and not ecb.empty and len(eurusd):
        ecb_usd = (ecb * eurusd.reindex(ecb.index, method="ffill")) / 1000.0  # EUR mm * rate -> USD mm -> /1000 = USD bn... 
        # ecb is in EUR millions; eurusd gives USD per EUR; product is USD millions; /1000 -> USD bn
        df_g = pd.concat([(fed_bs / 1000.0).rename("fed_bn"), ecb_usd.rename("ecb_bn")], axis=1).ffill().dropna()
        if not df_g.empty:
            global_liq_series = df_g["fed_bn"] + df_g["ecb_bn"]

    global_liq_now = float(global_liq_series.iloc[-1]) if len(global_liq_series) else np.nan
    global_liq_trend_pct = np.nan
    if len(global_liq_series) >= 29:
        prior = float(global_liq_series.iloc[-29])
        if prior != 0:
            global_liq_trend_pct = float((global_liq_series.iloc[-1] - prior) / abs(prior) * 100)

    return {
        "fed_bs_series": fed_bs, "fed_bs_now_bn": fed_bs_now_bn,
        "rrp_series": rrp, "rrp_now_bn": rrp_now_bn,
        "tga_series": tga, "tga_now_bn": tga_now_bn,
        "m2_series": m2, "m2_now_bn": m2_now, "m2_yoy_pct": m2_yoy,
        "debt_series": debt, "debt_now_bn": debt_now, "debt_qoq_pct": debt_qoq_pct,
        "net_liq_series": net_liq_series, "net_liq_now_bn": net_liq_now,
        "net_liq_trend_pct": net_liq_trend_pct,
        "global_liq_series": global_liq_series, "global_liq_now_bn": global_liq_now,
        "global_liq_trend_pct": global_liq_trend_pct,
        "ecb_series": ecb, "eurusd_series": eurusd,
    }


def score_fed_net_liquidity(liq: Dict) -> Tuple[float, str]:
    """
    Scores the Fed Net Liquidity TREND (4-week % change), not the level —
    level alone isn't comparable across time given the balance sheet's
    long-run size changes. Rising net liquidity (QE-like / RRP & TGA
    draining) is a tailwind for risk assets including BTC; falling net
    liquidity (QT-like / RRP & TGA refilling) is a headwind. This is the
    same channel as Macro Liquidity (Context group) but a different,
    independently-sourced read — Macro Liquidity uses Fed policy RATE +
    DXY + real yields, this uses the Fed's actual BALANCE SHEET mechanics.
    """
    trend = liq.get("net_liq_trend_pct", np.nan)
    now_bn = liq.get("net_liq_now_bn", np.nan)
    if pd.isna(trend):
        return 0.0, "Insufficient Fed liquidity data"

    level_note = f"Net liquidity ≈ ${now_bn:,.0f}bn" if not pd.isna(now_bn) else "Net liquidity level n/a"

    if trend > 4:    base, label = +1.0, "rising sharply (4w) — strong liquidity tailwind — BULLISH"
    elif trend > 1:   base, label = +0.5, "rising (4w) — mild liquidity tailwind"
    elif trend > -1:  base, label = 0.0,  "roughly flat (4w) — neutral"
    elif trend > -4:  base, label = -0.5, "falling (4w) — mild liquidity headwind"
    else:             base, label = -1.0, "falling sharply (4w) — strong liquidity headwind — BEARISH"

    return base, f"{level_note}, {trend:+.1f}% (4w) — {label}"


# ══════════════════════════════════════════════════════════════════════════════
# ETF DASHBOARD DATA  (NEW in v3.4)
#
# See ETF_TICKERS constant comment for the data-availability caveat. This
# fetcher pulls real price/volume data via yfinance for each ETF ticker,
# then derives an APPROXIMATE flow-pressure read from volume + price
# action — explicitly NOT the same as a reported creation/redemption flow
# number. A manual-override path lets the user paste real numbers from
# Farside/SoSoValue/the issuers' own daily disclosures for an accurate read.
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=900, show_spinner=False)
def fetch_etf_market_data() -> Dict:
    """Real, fetchable market data (price, volume, shares) for the major
    spot BTC ETF tickers via yfinance. This is NOT flow data — see
    compute_etf_flow_proxy() for how (and how cautiously) it's used."""
    out = {}
    for ticker in ETF_TICKERS:
        try:
            data = yf.download(ticker, period="3mo", interval="1d",
                                auto_adjust=True, progress=False)
            if data.empty:
                continue
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            close = data["Close"] if "Close" in data.columns else pd.Series(dtype=float)
            vol = data["Volume"] if "Volume" in data.columns else pd.Series(dtype=float)
            if isinstance(close, pd.DataFrame): close = close.iloc[:, 0]
            if isinstance(vol, pd.DataFrame): vol = vol.iloc[:, 0]
            out[ticker] = {"close": close.dropna(), "volume": vol.dropna()}
        except Exception:
            continue
    return out


def compute_etf_flow_proxy(etf_data: Dict, btc_chg_24h: float) -> Dict:
    """
    Builds an APPROXIMATE daily/weekly/30d "flow pressure" read from dollar
    volume traded in each ETF, explicitly distinct from real net
    creation/redemption flows (which require authorized-participant data
    these issuers/Farside/SoSoValue publish but don't expose via a free
    API). Dollar volume is a liquidity/attention proxy, not a flow number:
    high volume can mean strong net buying, strong net selling, or just
    heavy two-way trading with no net flow at all.

    What this CAN tell you directionally: rising aggregate dollar volume
    alongside a rising BTC price has historically coincided with net
    creation periods (more cited inflow days); rising volume alongside a
    falling price has historically coincided with net redemption periods.
    That correlation is the basis for the "pressure" sign below — it is a
    heuristic tilt, not a substitute for the real number.
    """
    if not etf_data:
        return {}

    total_dollar_vol = None
    per_ticker = {}
    for ticker, d in etf_data.items():
        close, vol = d["close"], d["volume"]
        aligned = pd.concat([close.rename("close"), vol.rename("volume")], axis=1).dropna()
        if aligned.empty:
            continue
        dollar_vol = aligned["close"] * aligned["volume"]
        per_ticker[ticker] = dollar_vol
        total_dollar_vol = dollar_vol if total_dollar_vol is None else total_dollar_vol.add(dollar_vol, fill_value=0)

    if total_dollar_vol is None or total_dollar_vol.empty:
        return {}

    daily_now = float(total_dollar_vol.iloc[-1])
    weekly_now = float(total_dollar_vol.iloc[-5:].sum()) if len(total_dollar_vol) >= 5 else np.nan
    avg30 = float(total_dollar_vol.iloc[-30:].mean()) if len(total_dollar_vol) >= 1 else np.nan
    trend_30d_pct = np.nan
    if len(total_dollar_vol) >= 35:
        recent = total_dollar_vol.iloc[-5:].mean()
        prior = total_dollar_vol.iloc[-35:-30].mean()
        if prior and prior != 0:
            trend_30d_pct = float((recent - prior) / prior * 100)

    # Directional tilt: same-sign volume trend + price change = net-flow-like
    # pressure in that direction; sign flips if volume is rising while price
    # falls (distribution-like) or volume is falling while price rises
    # (low-conviction rally, not flow-driven).
    pressure_sign = 0.0
    if not pd.isna(trend_30d_pct) and not pd.isna(btc_chg_24h):
        if trend_30d_pct > 5 and btc_chg_24h > 0:
            pressure_sign = +1.0
        elif trend_30d_pct > 5 and btc_chg_24h < 0:
            pressure_sign = -1.0
        elif trend_30d_pct < -5:
            pressure_sign = 0.0  # thinning volume = fading conviction either way, treated as neutral

    return {
        "daily_dollar_volume": daily_now,
        "weekly_dollar_volume": weekly_now,
        "avg30_dollar_volume": avg30,
        "trend_30d_pct": trend_30d_pct,
        "pressure_sign": pressure_sign,
        "per_ticker_dollar_volume": per_ticker,
        "total_dollar_volume_series": total_dollar_vol,
    }


def score_etf_flow_pressure(etf_proxy: Dict, manual_override: Optional[Dict] = None) -> Tuple[float, str]:
    """
    If the user has supplied real flow numbers via the manual-override
    panel (today's reported net flow in $mm, e.g. copied from Farside),
    THOSE are used and clearly labeled as user-supplied real data. If not,
    falls back to the auto-fetched volume-based proxy with an explicit
    "estimate, not reported flow" label so the distinction is never lost
    downstream in the scorecard.
    """
    if manual_override and not pd.isna(manual_override.get("net_flow_usd_mm", np.nan)):
        flow = manual_override["net_flow_usd_mm"]
        note_prefix = f"User-supplied net flow: ${flow:+,.0f}mm (reported, not estimated)"
        if flow > 500:    return +1.0, f"{note_prefix} — strong net inflow — BULLISH"
        if flow > 100:    return +0.5, f"{note_prefix} — moderate net inflow"
        if flow > -100:   return  0.0, f"{note_prefix} — roughly flat"
        if flow > -500:   return -0.5, f"{note_prefix} — moderate net outflow"
        return -1.0,                  f"{note_prefix} — strong net outflow — BEARISH"

    if not etf_proxy:
        return 0.0, "No ETF market data available"

    pressure = etf_proxy.get("pressure_sign", 0.0)
    trend = etf_proxy.get("trend_30d_pct", np.nan)
    trend_note = f"ETF dollar volume {'rising' if not pd.isna(trend) and trend > 0 else 'falling'} " \
                 f"{abs(trend):.0f}% (30d)" if not pd.isna(trend) else "ETF volume trend n/a"

    note = f"{trend_note} [ESTIMATE from trading volume, NOT a reported flow figure]"
    if pressure > 0.3:
        note += " — volume+price pattern resembles net-inflow periods — lean BULLISH"
    elif pressure < -0.3:
        note += " — volume+price pattern resembles net-outflow periods — lean BEARISH"
    else:
        note += " — no clear directional tilt"
    return pressure, note

# ══════════════════════════════════════════════════════════════════════════════
# MACRO LIQUIDITY
# ══════════════════════════════════════════════════════════════════════════════

def trend_pct_change(s: pd.Series, lookback: int = 20) -> float:
    """% change of the trailing mean over `lookback` days vs the
    prior `lookback`-day window."""
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
    """US-only Taylor Rule estimate. Returns (score -1..+1, action label).
    Score > 0 = hawkish Fed (headwind for risk assets/BTC); score < 0 = dovish."""
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
    """Blends Fed Taylor score, DXY 20d trend, and real 10Y yield level into
    one Macro Liquidity reading. Grouped under CONTEXT, not TREND or
    VALUATION, because it's an external driver of price action rather than
    a direct measurement of BTC's own current trend or valuation."""
    fed_score, fed_action = taylor_rule_score_us(
        inputs.get("cpi_now", np.nan), inputs.get("cpi_3m_ago", np.nan),
        inputs.get("unemp_now", np.nan), inputs.get("unemp_3m_ago", np.nan),
        inputs.get("gdp_growth", np.nan),
        inputs.get("fed_rate_now", np.nan), inputs.get("fed_rate_6m_ago", np.nan),
    )
    fed_component = -fed_score

    dxy_trend = inputs.get("dxy_trend_pct", np.nan)
    dxy_component = np.nan
    if not pd.isna(dxy_trend):
        dxy_component = float(np.clip(-dxy_trend / 4.0, -1, 1))

    real_yield = inputs.get("real_yield_now", np.nan)
    yield_component = np.nan
    if not pd.isna(real_yield):
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
# STABLECOIN SUPPLY GROWTH SCORING
# ══════════════════════════════════════════════════════════════════════════════

def score_stablecoin_growth(sc: Dict) -> Tuple[float, str]:
    """Net stablecoin supply growth as a dry-powder / liquidity-inflow proxy.
    Grouped under CONTEXT: minting/redemption is issuer/exchange-driven and
    can lead or lag price rather than being a direct vote on BTC's own
    trend or valuation. Noisy — see caveat in UI."""
    growth = sc.get("growth_pct", np.nan)
    if pd.isna(growth):
        return 0.0, "No stablecoin supply data"

    if growth > 8:    return +1.0, f"Stablecoin supply +{growth:.1f}% (30d) — strong inflow, dry powder building — BULLISH"
    if growth > 3:    return +0.5, f"Stablecoin supply +{growth:.1f}% (30d) — mild inflow"
    if growth > -3:   return  0.0, f"Stablecoin supply {growth:+.1f}% (30d) — roughly flat"
    if growth > -8:   return -0.5, f"Stablecoin supply {growth:.1f}% (30d) — mild outflow / redemptions"
    return -1.0,                 f"Stablecoin supply {growth:.1f}% (30d) — sharp outflow, capital leaving — BEARISH"

# ══════════════════════════════════════════════════════════════════════════════
# DERIVED METRICS  (mining cost, NVT, Puell, VPA)
# ══════════════════════════════════════════════════════════════════════════════

def compute_mining_cost(bc: Dict, cg: Dict, electricity_usd_kwh: float,
                         fleet_efficiency_j_th: float) -> Dict:
    """Estimate an all-in ELECTRICITY-ONLY cost per BTC mined. Excludes ASIC
    capex/depreciation, cooling, and pool fees — treat as a soft floor, not
    full miner P&L."""
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
    """NVT Ratio = Market Cap / Daily On-Chain Transaction Volume (USD),
    30d smoothed. TREND uses real daily market-cap history when available,
    falling back to a volume-only proxy (explicitly labeled) otherwise."""
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
    """Puell Multiple proxy = Daily Miner Revenue (USD) / trailing average."""
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


# ══════════════════════════════════════════════════════════════════════════════
# PERCENT SUPPLY IN PROFIT  (NEW in v3.4)
#
# The "real" version of this metric (as published by Glassnode/CryptoQuant)
# requires UTXO-level cost-basis data — the price at which every coin last
# moved on-chain — which isn't available from any free API used in this
# app (blockchain.info doesn't expose realized-price-by-cohort data).
#
# What's built here is an explicitly-labeled PROXY: it approximates the
# distribution of holder cost basis using the BTC/USD daily close history
# itself (1 year, from fetch_btc_ohlcv) volume-weighted by that day's
# traded volume as a rough stand-in for how much supply "changed hands"
# at each price level. Days with more volume contribute more weight to
# the implied cost-basis distribution. The % of that weighted distribution
# below the current price is the proxy "% supply in profit" figure.
#
# This is a meaningfully cruder estimate than the real on-chain version:
#   - It only sees 1 year of trading, not full UTXO age history (coins
#     held >1 year and untouched are invisible to this proxy)
#   - It uses exchange volume as a stand-in for on-chain coin movement,
#     which conflates trading activity with actual ownership transfer
#   - It cannot distinguish a coin moving between two cold wallets from a
#     coin actually changing economic ownership
#
# Despite those gaps, the basic shape (cheap accumulation low vs. expensive
# euphoric high) is usually directionally right even on a 1-year window,
# which is why it's kept in the Valuation group rather than discarded —
# but every UI surface for this metric carries the "proxy, not on-chain
# realized-price-based" caveat explicitly.
# ══════════════════════════════════════════════════════════════════════════════

def compute_percent_supply_in_profit(ohlcv: pd.DataFrame) -> Dict:
    if ohlcv.empty or len(ohlcv) < 30:
        return {}

    close = ohlcv["Close"]
    vol = ohlcv["Volume"]
    if isinstance(close, pd.DataFrame): close = close.iloc[:, 0]
    if isinstance(vol, pd.DataFrame): vol = vol.iloc[:, 0]
    close = close.dropna()
    vol = vol.reindex(close.index).fillna(0)

    price_now = float(close.iloc[-1])
    total_vol = float(vol.sum())
    if total_vol <= 0:
        return {}

    vol_below = float(vol[close < price_now].sum())
    pct_in_profit = (vol_below / total_vol) * 100

    # 30d-ago comparison for a trend read.
    pct_30d_ago = np.nan
    if len(close) > 30:
        close_30 = close.iloc[:-30]
        vol_30 = vol.iloc[:-30]
        price_30_ago = float(close_30.iloc[-1]) if len(close_30) else np.nan
        if not pd.isna(price_30_ago) and float(vol_30.sum()) > 0:
            pct_30d_ago = float(vol_30[close_30 < price_30_ago].sum()) / float(vol_30.sum()) * 100

    trend_pp = pct_in_profit - pct_30d_ago if not pd.isna(pct_30d_ago) else np.nan

    return {
        "pct_in_profit": pct_in_profit,
        "pct_in_profit_30d_ago": pct_30d_ago,
        "trend_pp": trend_pp,   # change in percentage points, not %
        "price_now": price_now,
        "lookback_days": len(close),
    }


def compute_vpa(ohlcv: pd.DataFrame) -> Dict:
    """Volume-Price Analysis: OBV + trend, price trend, 20d VWAP, and a
    volume-confirmation ratio."""
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
# SCORING ENGINE  (individual signal scores, logic unchanged from v3.2)
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
    """Trend via linear-regression slope over the full 30-day history."""
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
    """Caveat: noisy proxy — distorted by exchange batching, UTXO sweeps,
    address reuse, and L2/Lightning migration. One weak vote."""
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
    """Thresholds favor real extremes over the first sign of positive funding."""
    avg = fr.get("avg_8", np.nan)
    if pd.isna(avg):
        return 0.0, "No funding data"
    if avg > 0.08:    return -1.0, f"Funding rate very high ({avg:+.4f}%) — crowded longs — BEARISH"
    if avg > 0.03:    return -0.5, f"Funding rate elevated ({avg:+.4f}%) — mild bearish lean"
    if avg > -0.02:   return  0.0, f"Funding rate near-neutral ({avg:+.4f}%) — normal in either trend"
    if avg > -0.06:   return +0.5, f"Funding rate negative ({avg:+.4f}%) — shorts dominant — contrarian bullish"
    return +1.0,               f"Funding rate very negative ({avg:+.4f}%) — short squeeze setup — BULLISH"


def score_btc_dominance(cg: Dict) -> Tuple[float, str]:
    """Not a reliable directional signal — scored as a small regime-flag."""
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
    """Combines LEVEL (cheap/expensive vs history) with TREND."""
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


def score_percent_supply_in_profit(sip: Dict) -> Tuple[float, str]:
    """
    Contrarian valuation read: when almost all (traded) supply sits in
    profit, that's historically euphoric-zone behavior, since there's
    little overhead resistance from underwater holders wanting to "get
    back to even" and a lot of latent profit-taking incentive. When most
    supply is underwater, that's historically closer to capitulation —
    sellers willing to realize a loss have largely already done so.

    Reminder (also surfaced in the UI): this is a 1-year volume-weighted
    PROXY, not the on-chain realized-price-based metric Glassnode/
    CryptoQuant publish — see compute_percent_supply_in_profit() docstring.
    """
    pct = sip.get("pct_in_profit", np.nan)
    trend = sip.get("trend_pp", np.nan)
    if pd.isna(pct):
        return 0.0, "No supply-in-profit data"

    if pct > 95:    base, label = -1.0, "almost all traded supply in profit — euphoria zone — contrarian BEARISH"
    elif pct > 85:  base, label = -0.5, "most traded supply in profit — elevated, watch for profit-taking"
    elif pct > 50:  base, label = 0.0,  "majority of traded supply in profit — normal bull-market range"
    elif pct > 25:  base, label = +0.5, "minority of traded supply in profit — stress building, mild bullish (contrarian)"
    else:           base, label = +1.0, "most traded supply underwater — capitulation-zone — contrarian BULLISH"

    trend_note = ""
    if not pd.isna(trend):
        direction = "up" if trend > 0 else "down"
        trend_note = f", {direction} {abs(trend):.1f}pp (30d)"

    return base, f"~{pct:.0f}% of traded supply in profit (1y volume-weighted PROXY, not on-chain realized price){trend_note} — {label}"

def score_mining_cost_margin(mining: Dict) -> Tuple[float, str]:
    """
    NEW scored wrapper around the existing mining-cost model so it can sit
    inside the Valuation group with a comparable -1..+1 scale, rather than
    being purely descriptive (as it was in v3.2's Mining tab). Wide margin
    above electricity breakeven = historically supportive of continued
    hash rate growth (bullish-leaning valuation read); price near/below
    breakeven = capitulation-risk zone (bearish-leaning).

    Deliberately NOT added to DEFAULT_WEIGHTS / the composite score: the
    electricity-only model has wide assumption error bars (user-adjustable
    sliders for cost/efficiency), so it stays informational-only in the
    Valuation group rather than a weighted vote. Shown for context.
    """
    margin = mining.get("margin_pct", np.nan)
    if pd.isna(margin):
        return 0.0, "No mining cost data"

    if margin > 100:  return +1.0, f"Spot {margin:.0f}% above est. breakeven — wide margin, healthy miner economics"
    if margin > 30:    return +0.5, f"Spot {margin:.0f}% above est. breakeven — comfortable margin"
    if margin > 0:     return  0.0, f"Spot {margin:.0f}% above est. breakeven — thin margin, watch for stress"
    if margin > -20:   return -0.5, f"Spot {abs(margin):.0f}% below est. breakeven — miner stress building"
    return -1.0,                 f"Spot {abs(margin):.0f}% below est. breakeven — capitulation-risk zone"


def score_volume_confirmation(vpa: Dict) -> Tuple[float, str]:
    """Does volume confirm the price trend, or diverge from it?"""
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

def compute_composite(scores: Dict[str, float], weights: Dict[str, float]) -> float:
    total_w = sum(weights[k] for k in scores if k in weights and not pd.isna(scores[k]))
    if total_w == 0:
        return np.nan
    weighted = sum(weights[k] * scores[k]
                   for k in scores if k in weights and not pd.isna(scores[k]))
    return weighted / total_w


def compute_group_composite(scores: Dict[str, float], group: str) -> float:
    """Simple unweighted mean of every signal belonging to `group`, ignoring
    NaNs. Used for the per-group sub-scores (Valuation score, Trend score,
    Sentiment score) shown above the main scorecards — these are diagnostic
    summaries, separate from the main weighted composite."""
    members = [scores[k] for k, g in SIGNAL_GROUPS.items() if g == group and k in scores and not pd.isna(scores[k])]
    if not members:
        return np.nan
    return float(np.mean(members))


def composite_to_verdict(score: float) -> Tuple[str, str, str]:
    if pd.isna(score):
        return "Neutral", "★★★☆☆", "#f1c40f"
    if score >= 0.50:   return "Strong Bullish",  "★★★★★", "#00e676"
    if score >= 0.20:   return "Bullish",          "★★★★☆", "#69f0ae"
    if score >= -0.10:  return "Neutral",           "★★★☆☆", "#f1c40f"
    if score >= -0.40:  return "Bearish",           "★★☆☆☆", "#ff5252"
    return                "Strong Bearish",         "★☆☆☆☆", "#d50000"


# ══════════════════════════════════════════════════════════════════════════════
# TREND STAGE  (NEW in v3.3)
#
# Direction alone (Trend group) doesn't tell you WHERE in a move you are.
# Two markets can both be "above MA200 with positive momentum" while one
# just broke out of a bottom and the other is six months into a blow-off
# top — those are very different risk profiles even though Trend Direction
# scores them the same.
#
# This function explicitly combines:
#   - DIRECTION (from the Trend group: Price vs MA200 + 30d Momentum) to
#     decide Uptrend / Downtrend / Range in the first place
#   - MATURITY (from Valuation + Sentiment groups: NVT, Puell, Fear&Greed,
#     Funding Rate) to decide Early / Mid / Late within that direction
#
# It is a heuristic overlay, not a new data source — every input already
# exists elsewhere in the scorecard. The value-add is making the
# early/mid/late read explicit and showing exactly which inputs drove it,
# rather than leaving the reader to mentally combine twelve numbers.
# ══════════════════════════════════════════════════════════════════════════════

def compute_trend_stage(raw_scores: Dict[str, float], nvt: Dict, puell: Dict,
                         fg: Dict, fr: Dict, ohlcv: pd.DataFrame) -> Dict:
    ma_score  = raw_scores.get("Price vs MA200", np.nan)
    mom_score = raw_scores.get("30d Momentum", np.nan)

    direction_inputs = [s for s in [ma_score, mom_score] if not pd.isna(s)]
    if not direction_inputs:
        return {"direction": "Unknown", "stage": "—", "maturity_score": np.nan,
                "label": "Unknown", "inputs_used": [], "note": "Insufficient trend data."}

    direction_score = float(np.mean(direction_inputs))
    if direction_score >= 0.25:
        direction = "Uptrend"
    elif direction_score <= -0.25:
        direction = "Downtrend"
    else:
        direction = "Range"

    # Maturity inputs: how stretched is sentiment/valuation in the direction
    # we're already moving? High Puell + Extreme Greed + very positive
    # funding while in an uptrend = late-stage. The mirror image in a
    # downtrend = late-stage capitulation, which is "late" in duration but
    # actually the more BULLISH place to be a contrarian buyer — the label
    # reflects that explicitly below.
    maturity_parts = []
    inputs_used = []

    nvt_val = nvt.get("nvt_smoothed", np.nan)
    if not pd.isna(nvt_val):
        nvt_extremity = float(np.clip((nvt_val - 65) / 65, -1, 1))
        maturity_parts.append(nvt_extremity)
        inputs_used.append(f"NVT {nvt_val:.0f}")

    puell_val = puell.get("value", np.nan)
    if not pd.isna(puell_val):
        puell_extremity = float(np.clip((puell_val - 1.2) / 1.5, -1, 1))
        maturity_parts.append(puell_extremity)
        inputs_used.append(f"Puell {puell_val:.2f}")

    fg_val = fg.get("value", np.nan)
    if not pd.isna(fg_val):
        fg_extremity = float(np.clip((fg_val - 50) / 40, -1, 1))
        maturity_parts.append(fg_extremity)
        inputs_used.append(f"Fear&Greed {fg_val}")

    fr_val = fr.get("avg_8", np.nan)
    if not pd.isna(fr_val):
        fr_extremity = float(np.clip(fr_val / 0.08, -1, 1))
        maturity_parts.append(fr_extremity)
        inputs_used.append(f"Funding {fr_val:+.4f}%")

    if not maturity_parts:
        maturity_score = np.nan
    else:
        maturity_score = float(np.mean(maturity_parts))

    # Combine direction + maturity into a single stage label.
    if direction == "Range":
        if pd.isna(maturity_score):
            stage, label = "—", "Range — no clear stage (insufficient sentiment/valuation data)"
        elif maturity_score > 0.3:
            stage, label = "Range (top-heavy)", "Range-bound, but sentiment/valuation skew toward the expensive/greedy side — watch for breakdown"
        elif maturity_score < -0.3:
            stage, label = "Range (base-building)", "Range-bound, with sentiment/valuation skew toward the cheap/fearful side — watch for breakout"
        else:
            stage, label = "Range (balanced)", "Range-bound with no strong sentiment/valuation skew either way"
    elif direction == "Uptrend":
        if pd.isna(maturity_score):
            stage, label = "Mid", "Uptrend confirmed, stage unclear (insufficient sentiment/valuation data)"
        elif maturity_score < -0.15:
            stage, label = "Early", "Early-stage uptrend — direction has turned up but valuation/sentiment haven't caught up yet"
        elif maturity_score < 0.35:
            stage, label = "Mid", "Mid-stage uptrend — direction and sentiment/valuation are reasonably in sync"
        else:
            stage, label = "Late", "Late-stage uptrend — valuation stretched and/or sentiment euphoric; more vulnerable to a sharp reversal"
    else:  # Downtrend
        if pd.isna(maturity_score):
            stage, label = "Mid", "Downtrend confirmed, stage unclear (insufficient sentiment/valuation data)"
        elif maturity_score > 0.15:
            stage, label = "Early", "Early-stage downtrend — direction has turned down but valuation/sentiment haven't caught up yet"
        elif maturity_score > -0.35:
            stage, label = "Mid", "Mid-stage downtrend — direction and sentiment/valuation are reasonably in sync"
        else:
            stage, label = "Late", "Late-stage downtrend — capitulation-type readings (cheap valuation, fearful sentiment); historically closer to a bottom than a top, even though duration-wise it's 'late'"

    return {
        "direction": direction,
        "direction_score": direction_score,
        "stage": stage,
        "maturity_score": maturity_score,
        "label": label,
        "inputs_used": inputs_used,
    }

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
# WEIGHT SLIDER UI — auto-rebalancing  (unchanged from v3.2)
# ══════════════════════════════════════════════════════════════════════════════

def _rebalance_weight(changed_key: str, total_cap: float = 60.0) -> None:
    """on_change callback for a single weight slider. Redistributes the
    delta across all OTHER weights proportionally, so relative balance
    among untouched sliders is preserved."""
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
        even_share = max(0.0, (100.0 - new_val) / len(others)) if others else 0.0
        for k in others:
            state[k] = even_share

    drift = 100.0 - sum(state.values())
    if abs(drift) > 1e-9 and others:
        state[others[0]] = max(0.0, state[others[0]] + drift)

    for k in state:
        st.session_state[f"w_{k}"] = state[k]


def render_weight_sliders(default_weights: Dict[str, float],
                           signal_groups: Dict[str, str]) -> Dict[str, float]:
    """Renders one slider per signal, organized into the four functional
    groups (Valuation / Trend / Sentiment / Context) instead of a flat
    list, with a 🔗-equivalent group icon next to each name. Moving any
    slider auto-rebalances all others proportionally so the total always
    sums to 100%."""
    if "weight_state" not in st.session_state:
        st.session_state.weight_state = dict(default_weights)
        for k, v in default_weights.items():
            st.session_state[f"w_{k}"] = v

    state = st.session_state.weight_state

    st.caption(
        "Drag any slider — the others automatically rebalance, proportionally, "
        "so the total always stays at 100%. Signals are organized below by what "
        "kind of question they answer, not just listed flat."
    )

    reset_col, total_col = st.columns([1, 3])
    with reset_col:
        if st.button("↺ Reset to defaults", key="reset_weights_btn"):
            st.session_state.weight_state = dict(default_weights)
            for k, v in default_weights.items():
                st.session_state[f"w_{k}"] = v
            st.rerun()

    for group in GROUP_ORDER:
        members = [k for k in state.keys() if signal_groups.get(k) == group]
        if not members:
            continue
        st.markdown(f"**{GROUP_ICONS[group]} {group}**")
        cols = st.columns(min(3, len(members)))
        for i, name in enumerate(members):
            with cols[i % len(cols)]:
                st.slider(
                    name, min_value=0.0, max_value=60.0,
                    value=float(state[name]), step=1.0,
                    key=f"w_{name}",
                    on_change=_rebalance_weight, args=(name,),
                )

    total = sum(state.values())
    with total_col:
        st.caption(f"**Total: {total:.1f}%**")

    return {k: v / 100.0 for k, v in state.items()}

# ══════════════════════════════════════════════════════════════════════════════
# MAIN RENDER FUNCTION  (called by app.py)
# ══════════════════════════════════════════════════════════════════════════════

def render_crypto_onchain():
    st.header("₿ Bitcoin On-Chain Fundamental Scanner")
    st.caption("v3.3 · Signals grouped by Valuation · Trend Direction · Sentiment/Positioning · Context — "
               "plus a derived Trend Stage readout (Early / Mid / Late)")

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
            "These weights are a reasonable starting point, **not coefficients derived from a "
            "backtest**. Adjust them if you weight certain signals differently, or to stress-test "
            "how sensitive the verdict is to the weighting scheme."
        )
        st.info(
            "📐 **Why signals are grouped:** *30d Momentum*, *Price vs MA200*, *Funding Rate*, and "
            "*Fear & Greed* mostly read the same underlying price-action backdrop from different "
            "angles and tend to move together — summing their weights as if independent double-counts "
            "that shared information. Grouping by **what question each signal answers** (Valuation / "
            "Trend / Sentiment / Context) makes that overlap visible instead of hiding it in one flat "
            "list. Treat the composite below as a **structured checklist**, not a statistically "
            "validated predictive score.",
            icon="📐",
        )
        weights = render_weight_sliders(DEFAULT_WEIGHTS, SIGNAL_GROUPS)

    with st.expander("💵 ETF Flow Override (paste real numbers — optional)", expanded=False):
        st.caption(
            "There is no free public API for actual daily BTC ETF creation/redemption flows. "
            "The auto-fetched ETF signal below is an ESTIMATE built from trading volume, not a "
            "reported flow figure. If you have today's real net flow (e.g. from Farside Investors "
            "or SoSoValue), paste it here for an accurate score instead of the volume-based proxy."
        )
        use_manual = st.checkbox("Use manual net flow instead of the auto-estimate", value=False, key="etf_manual_toggle")
        if use_manual:
            manual_flow = st.number_input(
                "Today's reported net flow, all spot BTC ETFs combined ($ millions, negative = outflow)",
                min_value=-5000.0, max_value=5000.0, value=0.0, step=10.0, key="etf_manual_flow_input",
            )
            st.session_state["etf_manual_override"] = {"net_flow_usd_mm": manual_flow}
            st.caption(f"Using manual figure: ${manual_flow:+,.0f}mm")
        else:
            st.session_state["etf_manual_override"] = None

    with st.spinner("Fetching on-chain & market data…"):
        fg          = fetch_fear_greed()
        cg          = fetch_coingecko_btc()
        bc          = fetch_blockchain_info()
        fr          = fetch_funding_rate()
        ohlcv       = fetch_btc_ohlcv()
        mcap_hist   = fetch_btc_marketcap_history(days=90)
        macro_in    = fetch_macro_liquidity_inputs()
        stablecoins = fetch_stablecoin_supply()
        liquidity   = fetch_liquidity_dashboard_data()
        etf_market  = fetch_etf_market_data()

    mining = compute_mining_cost(bc, cg, electricity_cost, fleet_efficiency)
    nvt    = compute_nvt(bc, cg, mcap_hist)
    puell  = compute_puell_multiple(bc)
    vpa    = compute_vpa(ohlcv)
    macro  = compute_macro_liquidity(macro_in)
    sip    = compute_percent_supply_in_profit(ohlcv)
    etf_proxy = compute_etf_flow_proxy(etf_market, cg.get("chg_24h", np.nan))

    # Manual ETF flow override, set via the expander rendered below the
    # tabs setup (st.session_state persists it across reruns).
    etf_manual_override = st.session_state.get("etf_manual_override", None)

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
    mining_score, mining_note = score_mining_cost_margin(mining)
    sip_score,   sip_note   = score_percent_supply_in_profit(sip)
    fedliq_score, fedliq_note = score_fed_net_liquidity(liquidity)
    etf_score,   etf_note   = score_etf_flow_pressure(etf_proxy, etf_manual_override)

    # Signals included in the WEIGHTED COMPOSITE.
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
        "Percent Supply in Profit":   sip_score,
        "ETF Flow Pressure":          etf_score,
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
        "Percent Supply in Profit":   sip_note,
        "ETF Flow Pressure":          etf_note,
    }

    # Display-only scores (shown in their group's scorecard, but not part
    # of the weighted composite / pie of 100%). Fed Net Liquidity is kept
    # display-only for the same reason Mining Cost Margin is: it's a
    # genuinely useful trend read, but the 4-week-trend thresholds are
    # analyst priors on a metric most users haven't calibrated intuition
    # for yet, vs. ETF Flow Pressure and Percent Supply in Profit which
    # are common enough framings (and have a manual-override path for ETF)
    # that they're included as full weighted votes instead.
    display_only_scores = {"Mining Cost Margin": mining_score, "Fed Net Liquidity": fedliq_score}
    display_only_notes  = {"Mining Cost Margin": mining_note, "Fed Net Liquidity": fedliq_note}

    all_scores_for_groups = {**raw_scores, **display_only_scores}
    all_notes_for_groups  = {**notes, **display_only_notes}

    composite              = compute_composite(raw_scores, weights)
    verdict, stars, colour = composite_to_verdict(composite)

    val_group_score  = compute_group_composite(all_scores_for_groups, GROUP_VALUATION)
    trend_group_score = compute_group_composite(all_scores_for_groups, GROUP_TREND)
    sent_group_score  = compute_group_composite(all_scores_for_groups, GROUP_SENTIMENT)
    ctx_group_score   = compute_group_composite(all_scores_for_groups, GROUP_CONTEXT)

    trend_stage = compute_trend_stage(raw_scores, nvt, puell, fg, fr, ohlcv)

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
    # GROUP SUMMARY STRIP — the headline addition: four group scores +
    # the derived trend-stage label, all visible before drilling into tabs.
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("##### At a glance")
    g1, g2, g3, g4, g5 = st.columns(5)

    def _group_metric(col, label, icon, score):
        v, s, c = composite_to_verdict(score) if not pd.isna(score) else (None, None, None)
        col.metric(f"{icon} {label}", fmt(score, 2, prefix=("+" if (not pd.isna(score) and score >= 0) else "")) if not pd.isna(score) else "—")

    _group_metric(g1, "Valuation", GROUP_ICONS[GROUP_VALUATION], val_group_score)
    _group_metric(g2, "Trend", GROUP_ICONS[GROUP_TREND], trend_group_score)
    _group_metric(g3, "Sentiment", GROUP_ICONS[GROUP_SENTIMENT], sent_group_score)
    _group_metric(g4, "Context", GROUP_ICONS[GROUP_CONTEXT], ctx_group_score)
    g5.metric("🧭 Trend Stage", f"{trend_stage['direction']} · {trend_stage['stage']}")

    st.caption(
        f"**Trend Stage read:** {trend_stage['label']}" +
        (f"  *(inputs: {', '.join(trend_stage['inputs_used'])})*" if trend_stage['inputs_used'] else "")
    )

    st.markdown("---")

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
    (tab_score, tab_stage, tab_price, tab_vpa, tab_onchain,
     tab_mining, tab_macro, tab_liquidity, tab_etf, tab_sentiment, tab_guide) = st.tabs([
        "📊 Score Breakdown", "🧭 Trend Stage", "📈 Price & Trend", "📐 Volume-Price Analysis",
        "🔗 On-Chain Data", "⛏️ Mining Economics", "🌐 Macro Liquidity", "💧 Liquidity Dashboard",
        "🏦 ETF Dashboard", "😨 Sentiment", "📖 Signal Guide"
    ])

    # ── TAB 1 · SCORE BREAKDOWN (now four grouped scorecards) ────────────────
    with tab_score:
        st.subheader("Signal Scorecard — by Group")
        st.caption(
            "Each group answers a different question. Composite weight still drives the overall "
            "verdict at the top of the page; the group score below is a simple unweighted average "
            "of that group's members, shown for diagnostic context."
        )

        group_scores_map = {
            GROUP_VALUATION: val_group_score,
            GROUP_TREND: trend_group_score,
            GROUP_SENTIMENT: sent_group_score,
            GROUP_CONTEXT: ctx_group_score,
        }

        for group in GROUP_ORDER:
            members = [k for k, g in SIGNAL_GROUPS.items() if g == group]
            gscore = group_scores_map[group]
            gcolour = GROUP_COLOURS[group]

            st.markdown(
                f"""<div style="border-left: 4px solid {gcolour}; padding-left: 12px; margin: 14px 0 6px 0;">
                <span style="font-size:1.15rem; font-weight:700;">{GROUP_ICONS[group]} {group}</span>
                <span style="color:#aaa; font-size:0.9rem;"> — {GROUP_BLURB[group]}</span>
                </div>""",
                unsafe_allow_html=True,
            )

            rows = []
            for name in members:
                score = all_scores_for_groups.get(name, np.nan)
                weight = weights.get(name, np.nan)
                weight_label = f"{weight*100:.0f}%" if not pd.isna(weight) else "— (display only)"
                contrib = score * weight if (not pd.isna(score) and not pd.isna(weight)) else np.nan
                rows.append({
                    "Signal":       name,
                    "Score (−1→+1)": score_bar(score),
                    "Composite Weight": weight_label,
                    "Contribution": f"{contrib:+.3f}" if not pd.isna(contrib) else "—",
                    "Interpretation": all_notes_for_groups.get(name, "—"),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.caption(f"Group average score: **{fmt(gscore, 2, prefix=('+' if (not pd.isna(gscore) and gscore >= 0) else ''))}**")

        st.markdown("---")
        fig = go.Figure()
        categories = list(raw_scores.keys()) + list(display_only_scores.keys())
        values     = [all_scores_for_groups[k] for k in categories]
        bar_colours = [GROUP_COLOURS[SIGNAL_GROUPS[k]] for k in categories]

        fig.add_trace(go.Bar(
            x=categories, y=values, marker_color=bar_colours,
            text=[f"{v:+.2f}" if not pd.isna(v) else "—" for v in values], textposition="outside",
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.4)
        fig.update_layout(
            title="All Signal Scores, Coloured by Group (−1 Bearish → +1 Bullish)",
            yaxis=dict(range=[-1.2, 1.2], title="Score"),
            template="plotly_dark", height=440,
            xaxis=dict(tickangle=-25),
        )
        st.plotly_chart(fig, use_container_width=True)
        legend_bits = "  ·  ".join(f"<span style='color:{GROUP_COLOURS[g]}'>●</span> {g}" for g in GROUP_ORDER)
        st.markdown(legend_bits, unsafe_allow_html=True)

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=composite if not pd.isna(composite) else 0,
            delta={"reference": 0, "valueformat": ".3f"},
            title={"text": f"Weighted Composite → <b>{verdict}</b>", "font": {"size": 18}},
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

    # ── TAB 2 · TREND STAGE  (NEW) ─────────────────────────────────────────────
    with tab_stage:
        st.subheader("🧭 Trend Stage: Direction + Maturity")
        st.caption(
            "Direction alone doesn't say where in a move you are. This view combines the Trend group "
            "(Price vs MA200, 30d Momentum) with extremity readings from Valuation (NVT, Puell) and "
            "Sentiment (Fear & Greed, Funding Rate) to estimate whether the current direction looks "
            "early, mid, or late. This is a heuristic overlay — every input already exists elsewhere "
            "in the scorecard; nothing here is a new data source."
        )

        stage_colour = {
            "Early": "#69f0ae", "Mid": "#f1c40f", "Late": "#ff5252", "—": "#7f8c8d",
        }.get(trend_stage["stage"].split(" ")[0] if trend_stage["stage"] != "—" else "—", "#f1c40f")

        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, {stage_colour}22, {stage_colour}11);
                border: 2px solid {stage_colour};
                border-radius: 16px;
                padding: 20px 28px;
                margin-bottom: 18px;
                text-align: center;
            ">
                <div style="font-size: 1.1rem; color: #aaa;">Direction</div>
                <div style="font-size: 1.8rem; font-weight: 800;">{trend_stage['direction']}</div>
                <div style="font-size: 1.1rem; color: #aaa; margin-top: 10px;">Stage</div>
                <div style="font-size: 1.8rem; font-weight: 800; color: {stage_colour};">{trend_stage['stage']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(f"**Read:** {trend_stage['label']}")

        col1, col2 = st.columns(2)
        col1.metric("Direction score (Trend group)", fmt(trend_stage.get("direction_score"), 2))
        col2.metric("Maturity score (Valuation + Sentiment extremity)", fmt(trend_stage.get("maturity_score"), 2))

        if trend_stage["inputs_used"]:
            st.caption("Maturity inputs used: " + ", ".join(trend_stage["inputs_used"]))
        else:
            st.caption("No maturity inputs available — stage defaulted to 'Mid' / unclear.")

        st.markdown("---")
        st.markdown("**How direction and stage are derived**")
        st.markdown("""
| Step | Inputs | What it decides |
|---|---|---|
| 1. Direction | Price vs MA200 + 30d Momentum (averaged) | Uptrend (≥ +0.25) / Downtrend (≤ −0.25) / Range (between) |
| 2. Maturity | NVT level, Puell level, Fear & Greed level, Funding Rate level — each rescaled to roughly −1..+1 around its "neutral" zone, then averaged | How stretched valuation/sentiment is *in the direction already underway* |
| 3. Stage | Direction × Maturity together | Early / Mid / Late (or a Range sub-label) |
""")

        st.info(
            "**Reading the stage labels correctly:**\n\n"
            "- **Late-stage uptrend** = stretched valuation + euphoric sentiment *while price is still "
            "rising* — historically the more fragile place to be a buyer, not the most bullish.\n"
            "- **Late-stage downtrend** = cheap valuation + fearful sentiment *while price is still "
            "falling* — historically closer to a capitulation bottom than a continuation, even though "
            "it's labeled 'late' the same way late-uptrend is.\n"
            "- **Early stage** (either direction) means the price direction has just turned, but "
            "valuation/sentiment haven't caught up yet — i.e. there's more room left in the move if it "
            "continues.\n"
            "- This is a heuristic blend of existing signals, not a backtested regime classifier. "
            "Treat it as a framing device for *where to look harder*, not a timing signal on its own."
        )

        st.markdown("---")
        st.markdown("**Maturity components (detail)**")
        detail_rows = []
        nvt_val = nvt.get("nvt_smoothed", np.nan)
        if not pd.isna(nvt_val):
            detail_rows.append({"Component": "NVT Ratio", "Value": f"{nvt_val:.0f}",
                                 "Read": "high = expensive/stretched, low = cheap"})
        puell_val = puell.get("value", np.nan)
        if not pd.isna(puell_val):
            detail_rows.append({"Component": "Puell Multiple", "Value": f"{puell_val:.2f}",
                                 "Read": "high = miner euphoria, low = miner capitulation"})
        fg_val = fg.get("value", np.nan)
        if not pd.isna(fg_val):
            detail_rows.append({"Component": "Fear & Greed", "Value": f"{fg_val}",
                                 "Read": "high = greedy, low = fearful"})
        fr_val = fr.get("avg_8", np.nan)
        if not pd.isna(fr_val):
            detail_rows.append({"Component": "Funding Rate (8-period avg)", "Value": f"{fr_val:+.4f}%",
                                 "Read": "high = crowded longs, low/negative = crowded shorts"})
        if detail_rows:
            st.dataframe(pd.DataFrame(detail_rows), use_container_width=True, hide_index=True)
        else:
            st.warning("No maturity component data available right now.")

    # ── TAB 3 · PRICE & TREND ─────────────────────────────────────────────────
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

            st.caption("Price vs MA200 and 30d Momentum (Return Summary below) are the two **Trend "
                       "Direction** group members — see the 🧭 Trend Stage tab for how they combine "
                       "with valuation/sentiment to estimate stage.")

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

    # ── TAB 4 · VOLUME-PRICE ANALYSIS ────────────────────────────────────────
    with tab_vpa:
        st.subheader("📐 Volume-Price Analysis")
        st.caption(
            "On-chain metrics describe network fundamentals over weeks-to-months. VPA describes "
            "exchange order-book participation over days-to-weeks — whether the current price move "
            "is backed by real volume, or is a thin move likely to fail. This is part of the "
            "**Trend Direction** group: it confirms or contradicts the direction read elsewhere."
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

    # ── TAB 5 · ON-CHAIN DATA ─────────────────────────────────────────────────
    with tab_onchain:
        st.caption(
            "Hash Rate lives in the **Trend Direction** group. NVT Ratio lives in **Valuation**. "
            "Active Addresses lives in **Context** (noisy usage proxy, not a clean directional vote)."
        )
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

    # ── TAB 6 · MINING ECONOMICS ───────────────────────────────────────────
    with tab_mining:
        st.subheader("⛏️ Mining Cost & Network Economics")
        st.caption(
            "Electricity-only model. Excludes ASIC depreciation, cooling, and pool fees — read the "
            "breakeven figure as a soft capitulation floor, not full miner P&L. Margin vs breakeven "
            "is part of the **Valuation** group (shown informationally; not in the weighted composite "
            "— see the Signal Guide for why)."
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

            st.markdown(f"**Valuation read:** {mining_note}")

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
        st.caption("Puell Multiple is the other **Valuation** group member alongside NVT and Mining Cost Margin.")

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

    # ── TAB 7 · MACRO LIQUIDITY ────────────────────────────────────────────────
    with tab_macro:
        st.subheader("🌐 Macro Liquidity")
        st.caption(
            "Both signals on this tab live in the **Context / Flow** group: real information, but "
            "external to BTC's own price/valuation mechanics rather than a direct vote on them. "
            "BTC behaves like a high-beta risk asset most of the time, and macro liquidity conditions "
            "can override on-chain signals in the short term — this tab blends a US Fed Taylor Rule "
            "estimate, the 20-day DXY trend, and the real 10Y yield level into one liquidity reading."
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
            "proxy. Also **Context / Flow** group — driven by issuer/exchange minting and redemption "
            "decisions rather than Fed policy, FX markets, or BTC's own price action."
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
            "**Caveats:** Fed Taylor Rule scores are a heuristic estimate ported from a forex "
            "fundamental scanner, not a forecast verified against FOMC outcomes. DXY and real-yield "
            "components are simple trend/level reads, not full rates-market models. Stablecoin supply "
            "also grows for reasons unrelated to crypto risk appetite — e.g. treasury cash parking or "
            "cross-border settlement use — so treat it as a noisy proxy, not a confirmed 'money about "
            "to buy BTC' signal."
        )

    # ── TAB · LIQUIDITY DASHBOARD  (NEW in v3.4) ──────────────────────────────
    with tab_liquidity:
        st.subheader("💧 Liquidity Dashboard")
        st.caption(
            "Macro liquidity conditions matter more to BTC every cycle as it's grown into a "
            "macro-sensitive asset. This tab tracks the plumbing directly — Fed balance sheet, "
            "Reverse Repo, Treasury General Account, M2, Treasury debt issuance trend, and a "
            "global liquidity approximation — rather than the policy-rate/DXY framing used in the "
            "🌐 Macro Liquidity tab. Both live in the **Context / Flow** group; they're independently "
            "sourced reads on a related but distinct channel."
        )

        net_liq_now = liquidity.get("net_liq_now_bn", np.nan)
        net_liq_trend = liquidity.get("net_liq_trend_pct", np.nan)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Fed Net Liquidity", fmt(net_liq_now, 0, "$", "bn"),
                   fmt(net_liq_trend, 1, suffix="% (4w)"),
                   delta_color=delta_colour(net_liq_trend))
        c2.metric("Fed Balance Sheet", fmt(liquidity.get("fed_bs_now_bn"), 0, "$", "bn"))
        c3.metric("Reverse Repo (ON RRP)", fmt(liquidity.get("rrp_now_bn"), 0, "$", "bn"))
        c4.metric("Treasury General Acct", fmt(liquidity.get("tga_now_bn"), 0, "$", "bn"))

        st.markdown(f"**Fed Net Liquidity read:** {fedliq_note}")
        st.caption(
            "**Fed Net Liquidity = Fed Balance Sheet − Reverse Repo − Treasury General Account.** "
            "This is the standard community formula popularized by liquidity analysts (e.g. Michael "
            "Howell / CrossBorder Capital) and widely replicated on TradingView — it is not an "
            "official published Fed series. Rising net liquidity = more cash available to flow into "
            "risk assets (RRP/TGA draining, or the balance sheet expanding); falling = a headwind "
            "(RRP/TGA refilling, or QT shrinking the balance sheet). Shown as display-only context "
            "in the scorecard, not a weighted composite vote — see Signal Guide for why."
        )

        net_liq_series = liquidity.get("net_liq_series", pd.Series(dtype=float))
        if not net_liq_series.empty:
            fig_nl = go.Figure()
            fig_nl.add_trace(go.Scatter(x=net_liq_series.index, y=net_liq_series.values,
                                         name="Fed Net Liquidity", line=dict(color="#00bcd4", width=2),
                                         fill="tozeroy", fillcolor="rgba(0,188,212,0.08)"))
            fig_nl.update_layout(title="Fed Net Liquidity — WALCL − RRP − TGA ($bn)",
                                  template="plotly_dark", height=320, yaxis_title="$ Billions")
            st.plotly_chart(fig_nl, use_container_width=True)
        else:
            st.warning("Fed Net Liquidity series unavailable — one or more FRED series failed to fetch.")

        st.markdown("---")
        st.markdown("**Components**")

        col_a, col_b = st.columns(2)
        with col_a:
            fed_bs_series = liquidity.get("fed_bs_series", pd.Series(dtype=float))
            if not fed_bs_series.empty:
                fig_fb = go.Figure()
                fig_fb.add_trace(go.Scatter(x=fed_bs_series.index, y=fed_bs_series.values / 1000.0,
                                             name="Fed Balance Sheet", line=dict(color="#9b59b6", width=2)))
                fig_fb.update_layout(title="Fed Balance Sheet — Total Assets (WALCL, $bn)",
                                      template="plotly_dark", height=280, yaxis_title="$ Billions")
                st.plotly_chart(fig_fb, use_container_width=True)

            tga_series = liquidity.get("tga_series", pd.Series(dtype=float))
            if not tga_series.empty:
                fig_tga = go.Figure()
                fig_tga.add_trace(go.Scatter(x=tga_series.index, y=tga_series.values / 1000.0,
                                              name="TGA", line=dict(color="#e67e22", width=2)))
                fig_tga.update_layout(title="Treasury General Account (WTREGEN, $bn)",
                                       template="plotly_dark", height=280, yaxis_title="$ Billions")
                st.plotly_chart(fig_tga, use_container_width=True)

        with col_b:
            rrp_series = liquidity.get("rrp_series", pd.Series(dtype=float))
            if not rrp_series.empty:
                fig_rrp = go.Figure()
                fig_rrp.add_trace(go.Scatter(x=rrp_series.index, y=rrp_series.values,
                                              name="Reverse Repo", line=dict(color="#ff5252", width=2)))
                fig_rrp.update_layout(title="Overnight Reverse Repo Usage (RRPONTSYD, $bn)",
                                       template="plotly_dark", height=280, yaxis_title="$ Billions")
                st.plotly_chart(fig_rrp, use_container_width=True)

            m2_series = liquidity.get("m2_series", pd.Series(dtype=float))
            if not m2_series.empty:
                fig_m2 = go.Figure()
                fig_m2.add_trace(go.Scatter(x=m2_series.index, y=m2_series.values,
                                             name="M2", line=dict(color="#69f0ae", width=2)))
                fig_m2.update_layout(title="M2 Money Stock (M2SL, $bn, monthly)",
                                      template="plotly_dark", height=280, yaxis_title="$ Billions")
                st.plotly_chart(fig_m2, use_container_width=True)

        st.markdown("---")
        col_m2, col_debt = st.columns(2)
        col_m2.metric("M2 Money Stock", fmt(liquidity.get("m2_now_bn"), 0, "$", "bn"),
                      fmt(liquidity.get("m2_yoy_pct"), 1, suffix="% YoY"),
                      delta_color=delta_colour(liquidity.get("m2_yoy_pct")))
        col_debt.metric("Total Federal Debt", fmt(liquidity.get("debt_now_bn"), 0, "$", "bn"),
                        fmt(liquidity.get("debt_qoq_pct"), 1, suffix="% QoQ"),
                        delta_color=delta_colour(liquidity.get("debt_qoq_pct")))
        st.caption(
            "**Treasury issuance proxy:** FRED has no single clean daily/weekly 'net Treasury "
            "issuance' series. **Total Federal Debt Outstanding (GFDEBTN)**, reported quarterly, is "
            "used here as a trend proxy — its quarter-over-quarter change reflects net new issuance "
            "after accounting for maturities. Heavier issuance (faster debt growth) competes with "
            "other assets for the same pool of dollar liquidity and is generally viewed as a "
            "liquidity-draining force at the margin; lighter issuance is the reverse."
        )

        debt_series = liquidity.get("debt_series", pd.Series(dtype=float))
        if not debt_series.empty:
            fig_debt = go.Figure()
            fig_debt.add_trace(go.Bar(x=debt_series.index, y=debt_series.values / 1000.0,
                                       name="Total Federal Debt", marker_color="#7f8c8d"))
            fig_debt.update_layout(title="Total US Federal Debt Outstanding (GFDEBTN, $bn, quarterly)",
                                    template="plotly_dark", height=280, yaxis_title="$ Billions")
            st.plotly_chart(fig_debt, use_container_width=True)

        st.markdown("---")
        st.subheader("🌍 Global Liquidity (approximation)")
        global_liq_now = liquidity.get("global_liq_now_bn", np.nan)
        global_liq_trend = liquidity.get("global_liq_trend_pct", np.nan)
        col_g1, col_g2 = st.columns(2)
        col_g1.metric("Global Liquidity Proxy (Fed + ECB, USD)", fmt(global_liq_now, 0, "$", "bn"))
        col_g2.metric("4-Week Trend", fmt(global_liq_trend, 1, suffix="%"),
                      delta_color=delta_colour(global_liq_trend))

        global_liq_series = liquidity.get("global_liq_series", pd.Series(dtype=float))
        if not global_liq_series.empty:
            fig_gl = go.Figure()
            fig_gl.add_trace(go.Scatter(x=global_liq_series.index, y=global_liq_series.values,
                                         name="Global Liquidity (Fed+ECB)", line=dict(color="#3498db", width=2),
                                         fill="tozeroy", fillcolor="rgba(52,152,219,0.08)"))
            fig_gl.update_layout(title="Global Liquidity Proxy — Fed + ECB Balance Sheets (USD bn)",
                                  template="plotly_dark", height=300, yaxis_title="$ Billions")
            st.plotly_chart(fig_gl, use_container_width=True)
        else:
            st.warning("Global liquidity series unavailable — ECB series or EUR/USD fetch may have failed.")

        st.warning(
            "⚠️ **This global liquidity figure is a simplified approximation, not an official series.** "
            "It sums the Fed and ECB balance sheets (ECB converted to USD via the EUR/USD rate). The "
            "community version of this metric popularized by liquidity analysts typically also adds "
            "the Bank of Japan and PBoC balance sheets — those legs are **omitted here** because "
            "there's no reliable free daily series for them via FRED or yfinance, not because they're "
            "unimportant. Treat this as directionally useful (Fed+ECB tend to co-move with the fuller "
            "version) but meaningfully incomplete in absolute level.",
            icon="⚠️",
        )

    # ── TAB · ETF DASHBOARD  (NEW in v3.4) ────────────────────────────────────
    with tab_etf:
        st.subheader("🏦 ETF Dashboard")
        st.caption(
            "Since spot BTC ETF flows now dominate marginal demand for many traders' mental models "
            "of price action, this tab tracks them explicitly. **Read the data-availability note "
            "below before trusting any number on this tab — it matters more here than on any other "
            "tab in this app.**"
        )

        st.error(
            "🚫 **No free public API exists for real daily BTC ETF creation/redemption flows.** "
            "Farside Investors and SoSoValue publish the numbers people actually cite, but both are "
            "scrape-only HTML pages with no stable free API and no confirmed license for "
            "redistribution here. **The 'Estimated Flow Pressure' figures below are derived from ETF "
            "trading volume, NOT reported net flows** — they are a liquidity/attention proxy that can "
            "diverge significantly from the real number, especially on high-volume, low-net-flow "
            "(two-way trading) days. If you need the real number: check Farside Investors, SoSoValue, "
            "or the issuers' own daily disclosures directly, or paste it into the **'💵 ETF Flow "
            "Override'** panel above the data section for an accurate score.",
            icon="🚫",
        )

        using_manual = etf_manual_override is not None and not pd.isna(etf_manual_override.get("net_flow_usd_mm", np.nan))
        if using_manual:
            st.success(f"✅ Using your manually-entered net flow: **${etf_manual_override['net_flow_usd_mm']:+,.0f}mm** — this is real data, not an estimate.")
        else:
            st.info("ℹ️ No manual override set — all figures below are volume-based estimates. See the panel above the data section to enter real numbers.")

        st.markdown(f"**ETF Flow Pressure read:** {etf_note}")

        if etf_proxy:
            col1, col2, col3 = st.columns(3)
            col1.metric("Daily $ Volume (all tracked ETFs)", fmt(etf_proxy.get("daily_dollar_volume"), 0, "$"))
            col2.metric("Weekly $ Volume (5 trading days)", fmt(etf_proxy.get("weekly_dollar_volume"), 0, "$"))
            col3.metric("30d Volume Trend", fmt(etf_proxy.get("trend_30d_pct"), 1, suffix="%"),
                       delta_color=delta_colour(etf_proxy.get("trend_30d_pct")))
            st.caption(
                "These are **dollar trading volumes** (price × shares traded), an ESTIMATE input, not "
                "reported creation/redemption flows. Cumulative holdings (AUM / shares outstanding "
                "growth) would be the more direct proxy for that, but reliable shares-outstanding "
                "history isn't exposed via yfinance for these tickers — only price and volume are."
            )

            total_vol_series = etf_proxy.get("total_dollar_volume_series", pd.Series(dtype=float))
            if not total_vol_series.empty:
                fig_etf_vol = go.Figure()
                fig_etf_vol.add_trace(go.Bar(x=total_vol_series.index[-60:], y=total_vol_series.iloc[-60:],
                                              name="Aggregate $ Volume", marker_color="#f7931a"))
                fig_etf_vol.update_layout(
                    title="Aggregate Spot BTC ETF Dollar Volume — Last 60 Days (estimate input, not flow)",
                    template="plotly_dark", height=320, yaxis_title="USD")
                st.plotly_chart(fig_etf_vol, use_container_width=True)

            st.markdown("---")
            st.markdown("**Per-ETF dollar volume (last value)**")
            per_ticker = etf_proxy.get("per_ticker_dollar_volume", {})
            if per_ticker:
                rows = []
                for ticker, series in per_ticker.items():
                    if series.empty:
                        continue
                    rows.append({
                        "Ticker": ticker,
                        "Fund": ETF_TICKERS.get(ticker, ticker),
                        "Latest $ Volume": fmt(float(series.iloc[-1]), 0, "$"),
                        "30d Avg $ Volume": fmt(float(series.iloc[-30:].mean()), 0, "$") if len(series) >= 1 else "—",
                    })
                if rows:
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                fig_pie = go.Figure(data=[go.Pie(
                    labels=[r["Ticker"] for r in rows],
                    values=[float(per_ticker[r["Ticker"]].iloc[-1]) for r in rows],
                    hole=0.4,
                )])
                fig_pie.update_layout(title="Share of Latest Daily $ Volume by Ticker",
                                       template="plotly_dark", height=320)
                st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.warning("ETF market data unavailable — yfinance fetch may have failed for the tracked tickers.")

        st.markdown("---")
        st.caption(
            "**Tickers tracked:** " + ", ".join(f"{t} ({name})" for t, name in ETF_TICKERS.items()) +
            ". This covers the largest spot BTC ETFs by AUM as of when this app was built, but the "
            "list is hardcoded — newly-launched or since-closed funds won't automatically appear or "
            "disappear."
        )

    # ── TAB 8 · SENTIMENT ─────────────────────────────────────────────────────
    with tab_sentiment:
        st.caption("Both signals here are **Sentiment / Positioning** group members and are the primary "
                   "inputs (alongside Valuation) to the 🧭 Trend Stage maturity read.")
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

    # ── TAB 9 · SIGNAL GUIDE ─────────────────────────────────────────────────
    with tab_guide:
        st.subheader("Signal Reference Guide")

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
### Why signals are grouped (v3.3)

Every signal answers one of four different questions:

| Group | Question it answers | Members |
|---|---|---|
| 💰 **Valuation** | Is BTC cheap or expensive relative to its own usage/cost? | NVT Ratio, Puell Multiple, Mining Cost Margin |
| 📈 **Trend Direction** | Which way is price actually moving, and is the move confirmed? | Price vs MA200, 30d Momentum, Hash Rate, Volume Confirmation |
| 🎭 **Sentiment / Positioning** | How emotional/positioned are participants right now? | Fear & Greed, Funding Rate |
| 🌐 **Context / Flow** | Real information, but not a clean directional vote on BTC itself | BTC Dominance, Active Addresses, Stablecoin Supply Growth, Macro Liquidity |

This replaces the old single "correlated vs. independent" tag with an explicit taxonomy. The old tag
was trying to say two things at once — *what does this signal measure* and *does it double-count with
others* — and conflating them made it easy to miss that, e.g., Fear & Greed and Funding Rate aren't
really "trend" signals at all; they're sentiment signals that happen to often move with trend.

**Mining Cost Margin** is new to the scorecard view in v3.3 (it existed before as a purely descriptive
metric in the Mining tab). It's shown in the Valuation group for context but is **excluded from the
weighted composite** — the electricity-only model has wide, user-adjustable assumption error bars
(cost per kWh, fleet efficiency), so it's display-only rather than a backtested vote.

---
### Trend Stage — a new derived layer on top of the groups

The four groups above answer "what kind of evidence is this," but none of them alone answer
**"where in the current move are we?"** Two markets can both show "price above MA200, positive
momentum" (Trend Direction = bullish) while one just broke out of a multi-month base and the other
is six months into a blow-off top with NVT stretched and Fear & Greed at 90 — same Trend Direction
score, very different risk.

The 🧭 **Trend Stage** tab combines:
1. **Direction** — Price vs MA200 + 30d Momentum → Uptrend / Downtrend / Range
2. **Maturity** — how extreme NVT, Puell, Fear & Greed, and Funding Rate are *in the direction
   already underway* → Early / Mid / Late

It is a heuristic overlay, not a new data source: every input already exists in the scorecard above.
The value-add is making the early/mid/late read explicit and showing exactly which inputs drove it.

Important asymmetry to keep in mind: "late-stage uptrend" (stretched + euphoric while still rising)
is historically a more *fragile* place to be a buyer, while "late-stage downtrend" (cheap + fearful
while still falling) is historically *closer to a bottom* than a continuation — both get labeled
"Late" for consistency of language, but they mean opposite things for risk.
""")

        st.markdown("""
---
**Individual Signal Interpretation:**

**💰 Valuation Group**

**NVT Ratio (trend-aware)** — On-chain "valuation multiple." Low = usage justifies valuation (cheap);
high = valuation has detached from usage (expensive). The trend calculation uses real daily market-cap
history so the trend reflects genuine NVT movement; if market-cap history can't be fetched, the app
falls back to the volume-only proxy and says so explicitly. Static NVT thresholds were calibrated
years ago — as activity moves off-chain, the "normal" range likely drifts — so treat NVT as one input
among several, not a standalone call.

**Puell Multiple (trend-aware)** — Compares today's miner revenue to its trailing average. Extreme
lows have marked capitulation bottoms; extreme highs have marked euphoric tops. This trend calculation
does not share NVT's old constant-denominator issue, since both the ratio and its baseline come from
the same revenue series.

**Mining Cost Margin** *(display only — not in composite)* — Spot price vs. an electricity-only
breakeven estimate. Wide margin above breakeven has historically supported continued hash rate growth;
price near/below breakeven is a capitulation-risk zone for less-efficient miners. Excluded from the
weighted score because the underlying cost assumptions are user-adjustable and the model omits
hardware depreciation — treat it as a soft floor, not full miner P&L.

---
**📈 Trend Direction Group**

**Price vs MA200** — The macro trend filter. Above MA200 = bull market structure; below = bear market
structure.

**30d Momentum** — Trend direction over a meaningful period, avoiding daily noise.

**Hash Rate** — Measured via a 30-day linear-regression slope rather than comparing only the first
and last week, so the middle of the window actually counts and single noisy days at either edge matter
less. Rising hash rate signals miner confidence; falling hash rate can signal miner stress.

**Volume Confirmation (VPA)** — Does trading volume confirm or contradict the price trend? Price and
OBV moving together = real participation; divergence is historically a higher-risk setup. Of all the
signals here, this is one of the more directly defensible, since it's standard technical-analysis
logic rather than a crypto-specific heuristic — though exchange-reported volume quality varies, so
treat it as directional.

---
**🎭 Sentiment / Positioning Group**

**Fear & Greed Index** — Contrarian signal. Extreme Fear (<20) historically marks accumulation zones;
Extreme Greed (>80) marks distribution zones. Never trade on this alone — confirm with on-chain and
volume. Also a primary input to Trend Stage maturity.

**Funding Rate** — Persistently very positive = overcrowded longs vulnerable to a liquidation
cascade. Very negative = short squeeze potential. Thresholds favor real extremes over the first sign
of positive funding, since moderately positive funding is common and unremarkable during genuine bull
trends.

---
**🌐 Context / Flow Group**

**BTC Dominance** — *Not* a reliable directional signal for BTC on its own. Rising dominance shows up
in early bull markets (capital flowing into BTC first) **and** in bear markets (capital fleeing
riskier alts faster than BTC). Falling dominance shows up in "altseason" **and** can simply mean alts
are getting hit less hard. Scored as a small regime-flag (±0.3 max) rather than a directional vote.

**Active Addresses** — A *directionally* useful but noisy proxy for real usage, not a clean adoption
read. It's distorted by exchange custodial-wallet batching, UTXO consolidation sweeps, address reuse,
and activity migrating off-chain to Layer 2 / Lightning. Weight it as one vote among many.

**Macro Liquidity** — Blends a US Fed Taylor Rule estimate, the 20-day DXY trend, and the real 10Y
yield level into one liquidity reading. Hawkish Fed / rising dollar / high real yields = headwind for
BTC as a risk asset; the reverse = a tailwind. Kept in Context rather than Trend Direction because
it's an external driver of price action, not a direct measurement of BTC's own current trend.

**Stablecoin Supply Growth** — Aggregate USDT + USDC + DAI market cap growth over 30 days, as a
dry-powder / liquidity-inflow proxy. Minting and redemption decisions are driven by issuers and
exchanges responding to fiat on/off-ramp demand, which can lead or lag price rather than mechanically
restating it. Still a noisy proxy — supply also grows for reasons unrelated to crypto risk appetite,
like treasury cash parking or cross-border settlement — so treat it as one vote, not confirmed
incoming demand.

---
**Composite score / weighting** — The composite still sums weight × signal across the eleven weighted
components as if each were independent evidence. In reality, several Trend Direction and Sentiment
members substantially correlate with each other because they're reading the same underlying
price-action backdrop from different angles — the group view above is designed to make that overlap
visible rather than averaging it away silently. Treat the composite as a **structured checklist**
that organizes the inputs you should be looking at, not a statistically validated predictive model.

**Editable weights** — Every weighted signal in the '🎛️ Composite Score Weights' panel can be
adjusted, now organized by group rather than listed flat. Moving any single slider automatically
rebalances all the others proportionally so the total always sums to 100%. Use the reset button to
return to the defaults at any time.
""")

        st.info(
            "**Key reminders:**\n\n"
            "• No single signal is sufficient — use this composite as a confluence tool, not a trading signal\n"
            "• On-chain data is most informative at extremes; trend matters as much as level\n"
            "• The four groups (Valuation / Trend / Sentiment / Context) answer different questions — "
            "don't read them as twelve identical independent votes\n"
            "• Trend Stage (🧭 tab) is a heuristic overlay combining existing signals, not a backtested "
            "regime classifier or a timing signal on its own\n"
            "• 'Late-stage downtrend' and 'late-stage uptrend' mean opposite things for risk even though "
            "both are labeled 'Late' — see the Trend Stage tab for the asymmetry\n"
            "• BTC dominance is a regime flag, not a reliable directional call on BTC itself\n"
            "• Active-address counts are a noisy adoption proxy, not a clean usage read\n"
            "• Stablecoin supply growth is a noisy dry-powder proxy, not confirmed incoming demand\n"
            "• Mining Cost Margin is shown for context but excluded from the weighted composite\n"
            "• Score weights here are analyst priors, not backtested coefficients — adjust and stress-test them\n"
            "• For deeper metrics (MVRV, NUPL, realised price, exchange reserves) consider Glassnode or CryptoQuant\n\n"
            "**This tool is for research and education — it is not financial advice.**"
        )
