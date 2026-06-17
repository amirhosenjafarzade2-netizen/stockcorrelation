"""
crypto_onchain.py
══════════════════════════════════════════════════════════════════════════════
Bitcoin On-Chain Fundamental Scanner
Based on Chapter 9: Crypto — On-Chain Fundamentals, Bitcoin Drivers,
Sentiment & Strategies

Data Sources (all free, no API key required):
  • Alternative.me    — Fear & Greed Index
  • CoinGecko         — Price, market cap, BTC dominance, volume, 24h/7d/30d change
  • Blockchain.info   — Hash rate, active addresses, transaction volume, mempool
  • Binance Futures   — BTC perpetual funding rate
  • yfinance          — BTC/USD OHLCV for moving averages & momentum

Scoring Engine (Chapter 9 framework):
  • Fear & Greed Index     (contrarian signal)
  • Hash Rate trend        (miner confidence / network health)
  • Active Addresses trend (adoption / real usage)
  • Transaction Volume     (on-chain demand)
  • BTC Dominance          (risk appetite)
  • Funding Rate           (derivatives sentiment)
  • Price vs 200d MA       (macro trend)
  • 30d Price Momentum     (trend direction)
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
from typing import Dict, Optional
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
HEADERS = {"User-Agent": "Mozilla/5.0 CryptoScanner/1.0"}


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
        # Market data
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

        # Global dominance
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
    """Hash rate, active addresses, tx volume, mempool from blockchain.info."""
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
    except Exception:
        pass

    # Historical hash rate for trend (30-day chart)
    try:
        end_ts   = int(datetime.now().timestamp())
        start_ts = end_ts - 30 * 86400
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

    # Active addresses (30-day)
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

    return out


@st.cache_data(ttl=900, show_spinner=False)
def fetch_funding_rate() -> Dict:
    """Latest BTC perpetual funding rate from Binance Futures."""
    try:
        r = requests.get(BINANCE_FR_URL, timeout=REQUEST_TIMEOUT, headers=HEADERS)
        data = r.json()
        if not data:
            return {}
        rates = [float(d["fundingRate"]) * 100 for d in data]  # express as %
        return {
            "latest":  rates[-1] if rates else np.nan,
            "avg_8":   float(np.mean(rates)) if rates else np.nan,
            "history": rates,
        }
    except Exception:
        return {}


@st.cache_data(ttl=900, show_spinner=False)
def fetch_btc_ohlcv() -> pd.DataFrame:
    """BTC/USD daily OHLCV (1 year) via yfinance for MA + momentum."""
    try:
        data = yf.download("BTC-USD", period="1y", interval="1d",
                            auto_adjust=True, progress=False)
        if "Close" in data.columns:
            s = data["Close"]
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]
            return data[["Open","High","Low","Close","Volume"]].dropna()
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# SCORING ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def score_fear_greed(fg: Dict) -> tuple[float, str]:
    """
    Contrarian signal (Chapter 9 §3):
    Extreme Fear (<20)  = accumulation zone → Bullish score
    Extreme Greed (>80) = distribution zone → Bearish score
    """
    val = fg.get("value", np.nan)
    if pd.isna(val):
        return 0.0, "No data"
    if val <= 20:   return +1.0, f"Extreme Fear ({val}) — contrarian BULLISH zone"
    if val <= 35:   return +0.5, f"Fear ({val}) — mild bullish lean"
    if val <= 55:   return  0.0, f"Neutral ({val})"
    if val <= 75:   return -0.5, f"Greed ({val}) — mild bearish lean"
    return -1.0,               f"Extreme Greed ({val}) — contrarian BEARISH zone"


def score_hash_rate(bc: Dict) -> tuple[float, str]:
    """Rising hash rate = miner confidence = Bullish (Chapter 9 §1)."""
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


def score_active_addresses(bc: Dict) -> tuple[float, str]:
    """Rising active addresses = adoption growth (Chapter 9 §1)."""
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


def score_funding_rate(fr: Dict) -> tuple[float, str]:
    """
    Funding rate (Chapter 9 §1, §3):
    Persistently positive = overcrowded longs (bearish)
    Negative = bearish bias or short squeeze setup (contrarian bullish)
    """
    avg = fr.get("avg_8", np.nan)
    if pd.isna(avg):
        return 0.0, "No funding data"
    if avg > 0.05:    return -1.0, f"Funding rate high ({avg:+.4f}%) — crowded longs — BEARISH"
    if avg > 0.01:    return -0.5, f"Funding rate positive ({avg:+.4f}%) — mild bearish"
    if avg > -0.01:   return  0.0, f"Funding rate neutral ({avg:+.4f}%)"
    if avg > -0.05:   return +0.5, f"Funding rate negative ({avg:+.4f}%) — shorts dominant — contrarian bullish"
    return +1.0,               f"Funding rate very negative ({avg:+.4f}%) — short squeeze setup — BULLISH"


def score_btc_dominance(cg: Dict) -> tuple[float, str]:
    """
    BTC dominance (Chapter 9 §1):
    High / rising dominance = capital flowing into BTC = risk-off but BTC-specific bullish
    Very high dominance (>60%) = strong BTC preference
    """
    dom = cg.get("dominance", np.nan)
    if pd.isna(dom):
        return 0.0, "No dominance data"
    if dom > 60:   return +1.0, f"BTC dominance {dom:.1f}% — very high, capital in BTC — BULLISH"
    if dom > 52:   return +0.5, f"BTC dominance {dom:.1f}% — elevated, BTC preferred"
    if dom > 45:   return  0.0, f"BTC dominance {dom:.1f}% — balanced"
    if dom > 38:   return -0.3, f"BTC dominance {dom:.1f}% — altseason developing"
    return -0.5,               f"BTC dominance {dom:.1f}% — low, capital rotating to alts"


def score_price_vs_ma(ohlcv: pd.DataFrame) -> tuple[float, str]:
    """Price vs 200-day MA — macro trend filter."""
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
        return +0.5, f"Price above MA200 but below MA50 — cautious bullish"
    if price < ma200 and pct_above_200 > -10:
        return -0.5, f"Price {pct_above_200:.1f}% below MA200 — bearish but near support"
    return -1.0,     f"Price {pct_above_200:.1f}% below MA200 — strong downtrend — BEARISH"


def score_30d_momentum(cg: Dict) -> tuple[float, str]:
    """30-day price momentum (trend direction)."""
    chg = cg.get("chg_30d", np.nan)
    if pd.isna(chg):
        return 0.0, "No 30d data"
    if chg > 20:   return +1.0, f"30d return +{chg:.1f}% — strong momentum BULLISH"
    if chg > 5:    return +0.5, f"30d return +{chg:.1f}% — positive momentum"
    if chg > -5:   return  0.0, f"30d return {chg:+.1f}% — flat / consolidating"
    if chg > -20:  return -0.5, f"30d return {chg:.1f}% — negative momentum"
    return -1.0,               f"30d return {chg:.1f}% — strong selling pressure — BEARISH"


# ── Composite Scorer ─────────────────────────────────────────────────────────

SCORE_WEIGHTS = {
    "Fear & Greed":       0.20,
    "Hash Rate":          0.18,
    "Active Addresses":   0.15,
    "Funding Rate":       0.15,
    "Price vs MA200":     0.15,
    "30d Momentum":       0.10,
    "BTC Dominance":      0.07,
}


def compute_composite(scores: Dict[str, float]) -> float:
    total_w = sum(SCORE_WEIGHTS[k] for k in scores if not pd.isna(scores[k]))
    if total_w == 0:
        return np.nan
    weighted = sum(SCORE_WEIGHTS[k] * scores[k]
                   for k in scores if not pd.isna(scores[k]))
    return weighted / total_w


def composite_to_verdict(score: float) -> tuple[str, str, str]:
    """Returns (verdict, stars, colour)."""
    if pd.isna(score):
        return "Neutral", "★★★☆☆", "#f1c40f"
    if score >= 0.50:   return "Strong Bullish",  "★★★★★", "#00e676"
    if score >= 0.20:   return "Bullish",          "★★★★☆", "#69f0ae"
    if score >= -0.10:  return "Neutral",           "★★★☆☆", "#f1c40f"
    if score >= -0.40:  return "Bearish",           "★★☆☆☆", "#ff5252"
    return               "Strong Bearish",          "★☆☆☆☆", "#d50000"


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
    st.caption("Chapter 9 Framework · Fear & Greed · Hash Rate · Active Addresses · Funding Rate · Dominance · Price Trend")

    # ── Fetch all data ────────────────────────────────────────────────────────
    with st.spinner("Fetching on-chain & market data…"):
        fg    = fetch_fear_greed()
        cg    = fetch_coingecko_btc()
        bc    = fetch_blockchain_info()
        fr    = fetch_funding_rate()
        ohlcv = fetch_btc_ohlcv()

    # ── Run scoring ───────────────────────────────────────────────────────────
    fg_score,   fg_note   = score_fear_greed(fg)
    hr_score,   hr_note   = score_hash_rate(bc)
    aa_score,   aa_note   = score_active_addresses(bc)
    fund_score, fund_note = score_funding_rate(fr)
    dom_score,  dom_note  = score_btc_dominance(cg)
    ma_score,   ma_note   = score_price_vs_ma(ohlcv)
    mom_score,  mom_note  = score_30d_momentum(cg)

    raw_scores = {
        "Fear & Greed":     fg_score,
        "Hash Rate":        hr_score,
        "Active Addresses": aa_score,
        "Funding Rate":     fund_score,
        "Price vs MA200":   ma_score,
        "30d Momentum":     mom_score,
        "BTC Dominance":    dom_score,
    }
    notes = {
        "Fear & Greed":     fg_note,
        "Hash Rate":        hr_note,
        "Active Addresses": aa_note,
        "Funding Rate":     fund_note,
        "Price vs MA200":   ma_note,
        "30d Momentum":     mom_note,
        "BTC Dominance":    dom_note,
    }

    composite             = compute_composite(raw_scores)
    verdict, stars, colour = composite_to_verdict(composite)

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

    # ══════════════════════════════════════════════════════════════════════════
    # TOP METRICS ROW
    # ══════════════════════════════════════════════════════════════════════════
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("BTC Price",     fmt(cg.get("price"), 0, "$"),
              fmt(cg.get("chg_24h"), 2, suffix="%"),
              delta_color=delta_colour(cg.get("chg_24h")))
    c2.metric("Fear & Greed",  f"{fg.get('value', '—')} — {fg.get('label', '—')}")
    c3.metric("BTC Dominance", fmt(cg.get("dominance"), 1, suffix="%"))
    c4.metric("Funding Rate",  fmt(fr.get("latest"), 4, suffix="%"))
    c5.metric("Hash Rate",     fmt(bc.get("hash_rate_eh"), 1, suffix=" EH/s"))

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # TABS
    # ══════════════════════════════════════════════════════════════════════════
    tab_score, tab_price, tab_onchain, tab_sentiment, tab_guide = st.tabs([
        "📊 Score Breakdown", "📈 Price & Trend", "🔗 On-Chain Data",
        "😨 Sentiment", "📖 Signal Guide"
    ])

    # ── TAB 1 · SCORE BREAKDOWN ───────────────────────────────────────────────
    with tab_score:
        st.subheader("Signal Scorecard")

        # Score table
        rows = []
        for name, score in raw_scores.items():
            weight = SCORE_WEIGHTS.get(name, 0)
            contrib = score * weight if not pd.isna(score) else np.nan
            rows.append({
                "Signal":       name,
                "Score (−1→+1)": score_bar(score),
                "Weight":       f"{weight*100:.0f}%",
                "Contribution": f"{contrib:+.3f}" if not pd.isna(contrib) else "—",
                "Interpretation": notes[name],
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Radar / bar chart
        fig = go.Figure()
        categories = list(raw_scores.keys())
        values     = [raw_scores[k] for k in categories]
        colours    = ["#00e676" if v > 0.1 else "#ff5252" if v < -0.1 else "#f1c40f"
                      for v in values]

        fig.add_trace(go.Bar(
            x=categories,
            y=values,
            marker_color=colours,
            text=[f"{v:+.2f}" for v in values],
            textposition="outside",
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.4)
        fig.add_hrect(y0=0.2, y1=1.0,  fillcolor="#00e676", opacity=0.05, line_width=0)
        fig.add_hrect(y0=-1.0, y1=-0.2, fillcolor="#ff5252", opacity=0.05, line_width=0)
        fig.update_layout(
            title="On-Chain Signal Scores (−1 Bearish → +1 Bullish)",
            yaxis=dict(range=[-1.2, 1.2], title="Score"),
            template="plotly_dark",
            height=380,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Composite gauge
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
            fig_p.add_trace(go.Scatter(x=close.index, y=close,     name="BTC/USD",
                                        line=dict(color="#f7931a", width=2)))
            fig_p.add_trace(go.Scatter(x=ma50.index,  y=ma50,      name="MA 50",
                                        line=dict(color="#3498db", width=1.5, dash="dot")))
            fig_p.add_trace(go.Scatter(x=ma200.index, y=ma200,     name="MA 200",
                                        line=dict(color="#9b59b6", width=1.5, dash="dash")))
            fig_p.update_layout(title="BTC/USD — 1 Year", template="plotly_dark",
                                  height=400, yaxis_title="Price (USD)")
            st.plotly_chart(fig_p, use_container_width=True)

            # Key price levels
            price_now = float(close.iloc[-1])
            ma50_now  = float(ma50.dropna().iloc[-1])
            ma200_now = float(ma200.dropna().iloc[-1])
            col1, col2, col3 = st.columns(3)
            col1.metric("Current Price", f"${price_now:,.0f}")
            col2.metric("MA 50",  f"${ma50_now:,.0f}",
                        f"{(price_now/ma50_now-1)*100:+.1f}% vs price",
                        delta_color=delta_colour(price_now - ma50_now))
            col3.metric("MA 200", f"${ma200_now:,.0f}",
                        f"{(price_now/ma200_now-1)*100:+.1f}% vs price",
                        delta_color=delta_colour(price_now - ma200_now))

            # Returns table
            st.subheader("Return Summary")
            ret_data = {
                "Period": ["24h", "7 days", "30 days"],
                "Return": [
                    fmt(cg.get("chg_24h"),  2, suffix="%"),
                    fmt(cg.get("chg_7d"),   2, suffix="%"),
                    fmt(cg.get("chg_30d"),  2, suffix="%"),
                ],
            }
            st.dataframe(pd.DataFrame(ret_data), use_container_width=True, hide_index=True)
        else:
            st.warning("Price data unavailable from yfinance.")

    # ── TAB 3 · ON-CHAIN DATA ─────────────────────────────────────────────────
    with tab_onchain:
        col_l, col_r = st.columns(2)

        with col_l:
            st.subheader("Network Health")
            st.metric("Hash Rate",        fmt(bc.get("hash_rate_eh"), 2, suffix=" EH/s"))
            st.metric("Transactions 24h", fmt(bc.get("n_tx_24h"), 0))
            st.metric("BTC Sent 24h",     fmt(bc.get("total_btc_sent"), 1, suffix=" BTC"))
            st.metric("Blocks Mined 24h", fmt(bc.get("blocks_mined_24h"), 0))
            st.metric("Mempool Size",     fmt(bc.get("mempool_size"), 0, suffix=" bytes"))
            st.metric("Active Addresses", fmt(bc.get("active_addr_now"), 0))

        with col_r:
            st.subheader("Market Data")
            st.metric("Market Cap",       fmt(cg.get("market_cap"), 0, "$"))
            st.metric("24h Volume",       fmt(cg.get("volume_24h"), 0, "$"))
            st.metric("BTC Dominance",    fmt(cg.get("dominance"), 2, suffix="%"))
            st.metric("Circulating Supply", fmt(cg.get("circulating"), 0, suffix=" BTC"))
            st.metric("% of Max Supply",
                      fmt((cg.get("circulating") or 0) / 21_000_000 * 100, 2, suffix="%"))
            st.metric("ATH Distance",     fmt(cg.get("ath_chg"), 1, suffix="%"))

        # Hash rate chart
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

        # Active addresses chart
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

    # ── TAB 4 · SENTIMENT ─────────────────────────────────────────────────────
    with tab_sentiment:
        st.subheader("Fear & Greed Index")
        fg_val   = fg.get("value", 50)
        fg_label = fg.get("label", "—")

        # Gauge
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
                    {"range": [0,  25], "color": "#1a4a1a"},   # Extreme Fear
                    {"range": [25, 45], "color": "#1a3a1a"},   # Fear
                    {"range": [45, 55], "color": "#2a2a1a"},   # Neutral
                    {"range": [55, 75], "color": "#3a2a1a"},   # Greed
                    {"range": [75, 100], "color": "#4a1a1a"},  # Extreme Greed
                ],
                "threshold": {"line": {"color": "white", "width": 3}, "value": fg_val},
            },
        ))
        fig_fg.update_layout(template="plotly_dark", height=300)
        st.plotly_chart(fig_fg, use_container_width=True)

        # History chart
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

        # Funding rate history
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

        col1, col2 = st.columns(2)
        col1.metric("Latest Funding Rate", fmt(fr.get("latest"), 4, suffix="%"))
        col2.metric("8-Period Avg Rate",   fmt(fr.get("avg_8"),  4, suffix="%"))

    # ── TAB 5 · SIGNAL GUIDE ─────────────────────────────────────────────────
    with tab_guide:
        st.subheader("Chapter 9 Signal Reference Guide")

        st.markdown("""
**How the composite score works:**

| Score Range | Verdict | Meaning |
|---|---|---|
| +0.50 to +1.00 | ★★★★★ Strong Bullish | Multiple on-chain signals aligned positively |
| +0.20 to +0.49 | ★★★★☆ Bullish | Majority of signals positive |
| −0.10 to +0.19 | ★★★☆☆ Neutral | Mixed signals — wait for clarity |
| −0.40 to −0.11 | ★★☆☆☆ Bearish | Majority of signals negative |
| −1.00 to −0.41 | ★☆☆☆☆ Strong Bearish | Multiple on-chain signals negative |
""")

        st.markdown("""
---
**Individual Signal Interpretation (Chapter 9 §1–§3):**

**Fear & Greed Index** — Contrarian signal. Extreme Fear (<20) historically marks accumulation zones.
Extreme Greed (>80) marks distribution zones. Never trade fear or greed alone — confirm with on-chain.

**Hash Rate** — Rising hash rate signals miner confidence in network health and future profitability.
Falling hash rate can signal miner stress or upcoming capitulation (watch Hash Ribbon crossovers).

**Active Addresses** — The truest measure of real demand. Rising addresses = growing adoption.
Divergence between price rising and addresses falling = warning signal.

**Funding Rate** — Reflects how much leveraged traders are paying to hold positions.
Persistently positive = overcrowded longs vulnerable to liquidation cascade.
Extremely negative = short squeeze potential.

**Price vs MA200** — The macro trend filter. Above MA200 = bull market structure.
Below MA200 = bear market structure. Do not fight the trend.

**30d Momentum** — Trend direction over a meaningful period. Avoids daily noise.

**BTC Dominance** — High and rising dominance = capital concentrated in BTC (risk-off for alts but BTC-positive).
Falling dominance = altcoin season developing, risk appetite rising broadly.
""")

        st.info(
            "**Chapter 9 Key Reminders:**\n\n"
            "• No single signal is sufficient — use this composite as a confluence tool\n"
            "• On-chain data is most powerful at extremes (MVRV >3 or <1, extreme fear/greed)\n"
            "• Macro liquidity conditions override on-chain in the short term\n"
            "• BTC halvings (next: 2028) structurally reduce supply — bullish on 12–18 month horizon\n"
            "• For deeper metrics (MVRV, NUPL, realised price, exchange reserves) visit Glassnode or CryptoQuant"
        )
