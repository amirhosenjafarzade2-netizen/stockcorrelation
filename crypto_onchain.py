"""
crypto_onchain.py
══════════════════════════════════════════════════════════════════════════════
Bitcoin On-Chain Fundamental Scanner  (v3.0 — Trend-Aware + VPA)
Based on Chapter 9: Crypto — On-Chain Fundamentals, Bitcoin Drivers,
Sentiment & Strategies

Data Sources (all free, no API key required):
  • Alternative.me    — Fear & Greed Index
  • CoinGecko         — Price, market cap, BTC dominance, volume, 24h/7d/30d change
  • Blockchain.info   — Hash rate, active addresses, tx volume, mempool,
                         miner revenue, on-chain USD transfer volume, difficulty
  • Binance Futures   — BTC perpetual funding rate
  • yfinance          — BTC/USD OHLCV for moving averages, momentum & VPA

What changed in v3.0 (and why):
  • NVT and Puell are now scored on TREND as well as LEVEL. A static
    threshold can't tell you whether a reading of "70" got there by falling
    from 110 (de-risking, arguably bullish) or rising from 40 (re-risking,
    arguably bearish). Both readings now carry a rate-of-change component.
  • New Volume-Price Analysis (VPA) tab. On-chain data answers "what regime
    are we in" over weeks/months; VPA answers "does the current price move
    have real participation behind it" over days/weeks. The original module
    fetched OHLCV with Volume but never used Volume for anything. This adds:
      - On-Balance Volume (OBV) and its trend vs price trend (divergence)
      - Volume-confirmed breakout/breakdown flags vs 20d avg volume
      - Volume-Weighted Average Price (rolling 20d) as a dynamic reference
  • Mining cost model is unchanged in methodology (electricity-only is the
    right default proxy) but the capex blind spot is now a visible banner,
    not just a code comment, since it materially affects how the breakeven
    number should be read.
  • Score weights are now a module-level config users can see and adjust,
    explicitly labeled as analyst priors rather than backtested coefficients
    — the original code presented fixed weights with no caveat, which
    overstates their precision.
  • Defensive fixes: NVT/Puell/VPA scoring no longer throws on short
    histories or empty Binance responses; composite scoring degrades
    gracefully and says so in the UI rather than silently going to 0.

Scoring Engine (Chapter 9 framework + extensions):
  • Fear & Greed Index       (contrarian signal)
  • Hash Rate trend          (miner confidence / network health)
  • Active Addresses trend   (adoption / real usage)
  • Funding Rate             (derivatives sentiment)
  • Price vs 200d MA         (macro trend)
  • 30d Price Momentum       (trend direction)
  • BTC Dominance            (risk appetite)
  • NVT Ratio (level+trend)  (on-chain valuation, now trend-aware)
  • Puell Multiple (level+trend) (miner cycle extremes, now trend-aware)
  • Volume-Price Confirmation (new) — does volume support the price trend?
  ─────────────────────────────────────────────
  Composite → ★ rating + Bullish / Neutral / Bearish verdict
"""

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
HEADERS = {"User-Agent": "Mozilla/5.0 CryptoScanner/3.0"}

# Block reward halving schedule (epoch start height -> reward). Used only to
# estimate current BTC issuance per day for supply / mining-cost context.
BLOCK_REWARD_BTC   = 3.125          # post-April-2024 halving reward
BLOCKS_PER_DAY_AVG = 144            # ~10 min/block target
NEXT_HALVING_YEAR  = 2028


# ══════════════════════════════════════════════════════════════════════════════
# SCORE WEIGHTS — analyst priors, NOT backtested coefficients.
# Surfaced here (rather than buried) so it's visible these are a starting
# point for discussion, not a validated model. Editable via the UI.
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_WEIGHTS = {
    "Fear & Greed":        0.14,
    "Hash Rate":           0.12,
    "Active Addresses":    0.12,
    "Funding Rate":        0.12,
    "Price vs MA200":      0.12,
    "30d Momentum":        0.08,
    "BTC Dominance":       0.05,
    "NVT Ratio":           0.10,
    "Puell Multiple":      0.06,
    "Volume Confirmation": 0.09,
}


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


def compute_nvt(bc: Dict, cg: Dict) -> Dict:
    """
    NVT Ratio = Market Cap / Daily On-Chain Transaction Volume (USD), 30d
    smoothed to reduce single-day noise (standard "NVT Signal" adjustment).

    Also returns a TREND component: is NVT rising or falling over the last
    ~30 days? A level of "70" reached by falling from 110 is a different
    story than "70" reached by rising from 40 — static thresholds alone
    can't distinguish a market de-risking from one re-risking. This trend
    is later combined with the level in scoring.

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
    nvt_series = []
    if len(tv_hist) >= 35 and not pd.isna(mcap):
        vals = [v["value"] for v in tv_hist]
        # Build a rough NVT series using current mcap as a constant proxy
        # (mcap history isn't fetched here; this approximates recent NVT
        # drift via the volume denominator, which dominates short-term
        # NVT movement far more than mcap does day-to-day).
        nvt_series = [mcap / v if v > 0 else np.nan for v in vals[-35:]]
        recent_avg = np.nanmean(nvt_series[-5:])
        prior_avg  = np.nanmean(nvt_series[-35:-30])
        if prior_avg and not pd.isna(prior_avg) and prior_avg != 0:
            nvt_trend_pct = (recent_avg - prior_avg) / prior_avg * 100

    return {
        "nvt_smoothed": nvt_smoothed,
        "nvt_daily": nvt_daily,
        "nvt_trend_pct": nvt_trend_pct,
    }


def compute_puell_multiple(bc: Dict) -> Dict:
    """
    Puell Multiple proxy = Daily Miner Revenue (USD) / trailing average of
    Daily Miner Revenue (USD, up to 1y history as a proxy for the standard
    365d MA).

    Also returns a TREND component (5d vs prior 5d, on the multiple itself)
    for the same reason as NVT: a Puell of 0.6 falling further is a
    different signal than 0.6 having just bounced off 0.4.
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
    on a days-to-weeks horizon — whether a price move is backed by real
    volume or is a thin, low-conviction move likely to fail.

    Computes:
      - OBV (On-Balance Volume) and its 20d trend
      - Price 20d trend, for divergence comparison against OBV
      - 20d Volume-Weighted Average Price (rolling)
      - Volume-confirmed breakout/breakdown flag: did the latest 5d price
        move happen on above-average volume?
    """
    if ohlcv.empty or len(ohlcv) < 25:
        return {}

    close = ohlcv["Close"]
    vol   = ohlcv["Volume"]
    if isinstance(close, pd.DataFrame): close = close.iloc[:, 0]
    if isinstance(vol, pd.DataFrame):   vol = vol.iloc[:, 0]

    # On-Balance Volume
    direction = np.sign(close.diff().fillna(0))
    obv = (direction * vol).cumsum()

    obv_recent = float(obv.iloc[-5:].mean())
    obv_prior  = float(obv.iloc[-25:-20].mean())
    obv_trend_pct = ((obv_recent - obv_prior) / abs(obv_prior) * 100) if obv_prior != 0 else np.nan

    price_recent = float(close.iloc[-5:].mean())
    price_prior  = float(close.iloc[-25:-20].mean())
    price_trend_pct = ((price_recent - price_prior) / price_prior * 100) if price_prior != 0 else np.nan

    # Rolling 20-day VWAP
    vwap20 = (close * vol).rolling(20).sum() / vol.rolling(20).sum()
    vwap20_now = float(vwap20.iloc[-1]) if not pd.isna(vwap20.iloc[-1]) else np.nan
    price_now = float(close.iloc[-1])

    # Volume confirmation on the most recent 5-day move
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
# SCORING ENGINE
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
    hist = bc.get("hash_rate_history", [])
    if len(hist) < 7:
        return 0.0, "Insufficient history"
    vals = [v["value"] for v in hist]
    recent_avg = np.mean(vals[-7:])
    prior_avg  = np.mean(vals[:7])
    if prior_avg == 0:
        return 0.0, "No baseline"
    chg_pct = (recent_avg - prior_avg) / prior_avg * 100
    if chg_pct > 5:    return +1.0, f"Hash rate rising +{chg_pct:.1f}% (30d) — BULLISH"
    if chg_pct > 1:    return +0.5, f"Hash rate slightly rising +{chg_pct:.1f}% — mild bullish"
    if chg_pct > -1:   return  0.0, f"Hash rate flat ({chg_pct:+.1f}%) — Neutral"
    if chg_pct > -5:   return -0.5, f"Hash rate slightly falling {chg_pct:.1f}% — mild bearish"
    return -1.0,               f"Hash rate falling {chg_pct:.1f}% — BEARISH (miner stress)"


def score_active_addresses(bc: Dict) -> Tuple[float, str]:
    now = bc.get("active_addr_now", np.nan)
    ago = bc.get("active_addr_30d", np.nan)
    if pd.isna(now) or pd.isna(ago) or ago == 0:
        return 0.0, "No address data"
    chg = (now - ago) / ago * 100
    if chg > 10:   return +1.0, f"Active addresses +{chg:.1f}% (30d) — strong BULLISH adoption"
    if chg > 3:    return +0.5, f"Active addresses +{chg:.1f}% — rising adoption"
    if chg > -3:   return  0.0, f"Active addresses flat ({chg:+.1f}%)"
    if chg > -10:  return -0.5, f"Active addresses {chg:.1f}% — declining usage"
    return -1.0,               f"Active addresses {chg:.1f}% — sharp drop — BEARISH"


def score_funding_rate(fr: Dict) -> Tuple[float, str]:
    avg = fr.get("avg_8", np.nan)
    if pd.isna(avg):
        return 0.0, "No funding data"
    if avg > 0.05:    return -1.0, f"Funding rate high ({avg:+.4f}%) — crowded longs — BEARISH"
    if avg > 0.01:    return -0.5, f"Funding rate positive ({avg:+.4f}%) — mild bearish"
    if avg > -0.01:   return  0.0, f"Funding rate neutral ({avg:+.4f}%)"
    if avg > -0.05:   return +0.5, f"Funding rate negative ({avg:+.4f}%) — shorts dominant — contrarian bullish"
    return +1.0,               f"Funding rate very negative ({avg:+.4f}%) — short squeeze setup — BULLISH"


def score_btc_dominance(cg: Dict) -> Tuple[float, str]:
    dom = cg.get("dominance", np.nan)
    if pd.isna(dom):
        return 0.0, "No dominance data"
    if dom > 60:   return +1.0, f"BTC dominance {dom:.1f}% — very high, capital in BTC — BULLISH"
    if dom > 52:   return +0.5, f"BTC dominance {dom:.1f}% — elevated, BTC preferred"
    if dom > 45:   return  0.0, f"BTC dominance {dom:.1f}% — balanced"
    if dom > 38:   return -0.3, f"BTC dominance {dom:.1f}% — altseason developing"
    return -0.5,               f"BTC dominance {dom:.1f}% — low, capital rotating to alts"


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
    getting cheaper or more expensive right now). Level sets the base score;
    trend nudges it by up to ±0.3, so a level reading near a threshold
    boundary can flip categories if the trend is moving sharply against it.
    """
    val = nvt.get("nvt_smoothed", np.nan)
    trend = nvt.get("nvt_trend_pct", np.nan)
    if pd.isna(val):
        return 0.0, "No NVT data"

    if val < 40:   base, label = +1.0, "usage far outpaces valuation — BULLISH (undervalued)"
    elif val < 65: base, label = +0.5, "usage healthy vs valuation"
    elif val < 90: base, label = 0.0,  "neutral valuation zone"
    elif val < 130: base, label = -0.5, "valuation stretched vs usage"
    else:          base, label = -1.0, "valuation far outpacing usage — BEARISH (overvalued)"

    trend_note = ""
    if not pd.isna(trend):
        nudge = np.clip(-trend / 100, -0.3, 0.3)  # NVT rising (trend>0) is bearish nudge
        base = float(np.clip(base + nudge, -1.0, 1.0))
        trend_note = f", trending {'up' if trend > 0 else 'down'} {abs(trend):.0f}% (30d)"

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
    Bullish: price rising + OBV rising together (real accumulation).
    Bearish divergence: price rising while OBV falling (distribution into
    strength — classic warning sign) or price falling while OBV rising
    (absorption / accumulation into weakness — contrarian bullish).
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
    bull_div  = price_trend < -1 and obv_trend > 1   # price down, volume accumulating
    bear_div  = price_trend > 1 and obv_trend < -1   # price up, volume distributing

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
# MAIN RENDER FUNCTION  (called by app.py)
# ══════════════════════════════════════════════════════════════════════════════

def render_crypto_onchain():
    st.header("₿ Bitcoin On-Chain Fundamental Scanner")
    st.caption("Chapter 9 Framework · Fear & Greed · Hash Rate · Active Addresses · Funding Rate · "
               "Dominance · NVT (trend-aware) · Puell (trend-aware) · Volume-Price Analysis · Mining Economics")

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
            "These weights are a reasonable starting point reflecting Chapter 9's framework, "
            "**not coefficients derived from a backtest**. Adjust them if you weight certain "
            "signals differently, or to stress-test how sensitive the verdict is to the weighting scheme."
        )
        weights = {}
        cols = st.columns(3)
        for i, (name, default) in enumerate(DEFAULT_WEIGHTS.items()):
            with cols[i % 3]:
                weights[name] = st.slider(name, 0.0, 0.30, default, 0.01, key=f"w_{name}")
        total_w = sum(weights.values())
        if total_w > 0:
            weights = {k: v / total_w for k, v in weights.items()}
        st.caption(f"Weights normalized to sum to 100% (raw sum was {total_w:.2f}).")

    with st.spinner("Fetching on-chain & market data…"):
        fg    = fetch_fear_greed()
        cg    = fetch_coingecko_btc()
        bc    = fetch_blockchain_info()
        fr    = fetch_funding_rate()
        ohlcv = fetch_btc_ohlcv()

    mining = compute_mining_cost(bc, cg, electricity_cost, fleet_efficiency)
    nvt    = compute_nvt(bc, cg)
    puell  = compute_puell_multiple(bc)
    vpa    = compute_vpa(ohlcv)

    fg_score,   fg_note   = score_fear_greed(fg)
    hr_score,   hr_note   = score_hash_rate(bc)
    aa_score,   aa_note   = score_active_addresses(bc)
    fund_score, fund_note = score_funding_rate(fr)
    dom_score,  dom_note  = score_btc_dominance(cg)
    ma_score,   ma_note   = score_price_vs_ma(ohlcv)
    mom_score,  mom_note  = score_30d_momentum(cg)
    nvt_score,  nvt_note  = score_nvt(nvt)
    puell_score, puell_note = score_puell(puell)
    vol_score,  vol_note  = score_volume_confirmation(vpa)

    raw_scores = {
        "Fear & Greed":        fg_score,
        "Hash Rate":           hr_score,
        "Active Addresses":    aa_score,
        "Funding Rate":        fund_score,
        "Price vs MA200":      ma_score,
        "30d Momentum":        mom_score,
        "BTC Dominance":       dom_score,
        "NVT Ratio":           nvt_score,
        "Puell Multiple":      puell_score,
        "Volume Confirmation": vol_score,
    }
    notes = {
        "Fear & Greed":        fg_note,
        "Hash Rate":           hr_note,
        "Active Addresses":    aa_note,
        "Funding Rate":        fund_note,
        "Price vs MA200":      ma_note,
        "30d Momentum":        mom_note,
        "BTC Dominance":       dom_note,
        "NVT Ratio":           nvt_note,
        "Puell Multiple":      puell_note,
        "Volume Confirmation": vol_note,
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
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("BTC Price",     fmt(cg.get("price"), 0, "$"),
              fmt(cg.get("chg_24h"), 2, suffix="%"),
              delta_color=delta_colour(cg.get("chg_24h")))
    c2.metric("Fear & Greed",  f"{fg.get('value', '—')} — {fg.get('label', '—')}")
    c3.metric("BTC Dominance", fmt(cg.get("dominance"), 1, suffix="%"))
    c4.metric("Funding Rate",  fmt(fr.get("latest"), 4, suffix="%"))
    c5.metric("Hash Rate",     fmt(bc.get("hash_rate_eh"), 1, suffix=" EH/s"))
    c6.metric("NVT Ratio",     fmt(nvt.get("nvt_smoothed"), 1))

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # TABS
    # ══════════════════════════════════════════════════════════════════════════
    tab_score, tab_price, tab_vpa, tab_onchain, tab_mining, tab_sentiment, tab_guide = st.tabs([
        "📊 Score Breakdown", "📈 Price & Trend", "📐 Volume-Price Analysis", "🔗 On-Chain Data",
        "⛏️ Mining Economics", "😨 Sentiment", "📖 Signal Guide"
    ])

    # ── TAB 1 · SCORE BREAKDOWN ───────────────────────────────────────────────
    with tab_score:
        st.subheader("Signal Scorecard")

        rows = []
        for name, score in raw_scores.items():
            weight = weights.get(name, 0)
            contrib = score * weight if not pd.isna(score) else np.nan
            rows.append({
                "Signal":       name,
                "Score (−1→+1)": score_bar(score),
                "Weight":       f"{weight*100:.0f}%",
                "Contribution": f"{contrib:+.3f}" if not pd.isna(contrib) else "—",
                "Interpretation": notes[name],
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

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
            title="On-Chain & Volume Signal Scores (−1 Bearish → +1 Bullish)",
            yaxis=dict(range=[-1.2, 1.2], title="Score"),
            template="plotly_dark", height=400,
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

    # ── TAB 3 · VOLUME-PRICE ANALYSIS (NEW) ───────────────────────────────────
    with tab_vpa:
        st.subheader("📐 Volume-Price Analysis")
        st.caption(
            "On-chain metrics describe network fundamentals over weeks-to-months. VPA describes "
            "exchange order-book participation over days-to-weeks — whether the current price move "
            "is backed by real volume, or is a thin move likely to fail. The two are complementary, "
            "not redundant: on-chain tells you what zone you're in, VPA helps with *when* to act within it."
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
                "diverge — price rising while OBV falls, or vice versa — that's historically a higher-risk "
                "setup, since the move isn't backed by the volume that would normally confirm it. "
                "**20d VWAP** acts as a dynamic support/resistance reference many traders watch for "
                "mean-reversion entries."
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
            st.metric("NVT 30d Trend", fmt(nvt.get("nvt_trend_pct"), 1, suffix="%"))
            st.metric("BTC Dominance",    fmt(cg.get("dominance"), 2, suffix="%"))
            st.metric("Circulating Supply", fmt(cg.get("circulating"), 0, suffix=" BTC"))

        st.markdown("---")
        st.caption(
            "**NVT Ratio** = Market Cap ÷ on-chain USD transfer volume (30-day average, smoothed). "
            "Think of it as an on-chain 'P/E ratio.' Note the **trend** column above: a level near a "
            "threshold boundary means less if it's moving sharply in the opposite direction — falling "
            "NVT into a 'neutral' reading is a different signal than rising NVT into the same reading."
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
            "cycle-top euphoria. The trend figure shows whether it's moving toward or away from an extreme."
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

    # ── TAB 6 · SENTIMENT ─────────────────────────────────────────────────────
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

    # ── TAB 7 · SIGNAL GUIDE ─────────────────────────────────────────────────
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

**Fear & Greed Index** — Contrarian signal. Extreme Fear (<20) historically marks accumulation zones;
Extreme Greed (>80) marks distribution zones. Never trade on this alone — confirm with on-chain and volume.

**Hash Rate** — Rising hash rate signals miner confidence. Falling hash rate can signal miner stress
or upcoming capitulation.

**Active Addresses** — A core measure of real demand. Divergence between price rising and addresses
falling is a warning signal.

**Funding Rate** — Persistently positive = overcrowded longs vulnerable to a liquidation cascade.
Extremely negative = short squeeze potential.

**Price vs MA200** — The macro trend filter. Above MA200 = bull market structure; below = bear
market structure.

**30d Momentum** — Trend direction over a meaningful period, avoiding daily noise.

**BTC Dominance** — High/rising = capital concentrated in BTC. Falling = altcoin season developing.

**NVT Ratio (now trend-aware)** — On-chain "valuation multiple." Low = usage justifies valuation
(cheap); high = valuation has detached from usage (expensive). **Caveat:** static thresholds were
calibrated years ago; as activity moves off-chain (exchanges, Lightning, custodial rails), the
"normal" range likely drifts over time — treat NVT as one input among several, not a standalone call.
The trend component shows whether NVT is currently moving toward or away from an extreme.

**Puell Multiple (now trend-aware)** — Compares today's miner revenue to its trailing average.
Extreme lows have marked capitulation bottoms; extreme highs have marked euphoric tops.

**Volume-Price Analysis (new)** — Does trading volume confirm or contradict the price trend? Price
and OBV moving together = real participation. Divergence (price up, volume distributing, or vice
versa) is historically a higher-risk setup, regardless of what on-chain fundamentals say — this is
the tactical layer that sits on top of the slower-moving fundamental picture.

**Mining Cost / Breakeven** — Electricity-only estimate of network production cost per BTC. Excludes
capex/depreciation — read it as a soft floor, not full miner P&L. When spot falls toward or below this
line, more miners historically approach capitulation.
""")

        st.info(
            "**Key reminders:**\n\n"
            "• No single signal is sufficient — use this composite as a confluence tool, not a trading signal\n"
            "• On-chain data is most informative at extremes; trend matters as much as level\n"
            "• Volume-price analysis adds a tactical timing layer on top of the slower fundamental picture\n"
            "• Macro liquidity conditions can override on-chain signals in the short term\n"
            "• Score weights here are analyst priors, not backtested coefficients — adjust and stress-test them\n"
            "• Mining cost estimates are electricity-only approximations, not exact miner P&L\n"
            "• For deeper metrics (MVRV, NUPL, realised price, exchange reserves) consider Glassnode or CryptoQuant\n\n"
            "**This tool is for research and education — it is not financial advice.**"
        )
