import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
import random
from datetime import datetime
from typing import Optional, Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# ── Configuration ──────────────────────────────────────────────────────────

# Expanded list of popular ETFs by category
ETF_UNIVERSE = {
    "Equity": ["SPY", "VOO", "IVV", "VTI", "SCHB", "ITOT", "SPTM"],
    "International": ["VXUS", "VEA", "IEFA", "VWO", "IEMG", "EFA", "IXUS"],
    "Sector - Technology": ["XLK", "VGT", "IYW", "FTEC", "QQQ", "SOXX"],
    "Sector - Healthcare": ["XLV", "VHT", "IYH", "FHLC"],
    "Sector - Finance": ["XLF", "VFH", "IYF", "KBE"],
    "Sector - Energy": ["XLE", "VDE", "IYE", "XOP"],
    "Sector - Consumer": ["XLY", "XLP", "VCR", "VDC"],
    "Bond": ["AGG", "BND", "BNDX", "TLT", "SHY", "LQD", "HYG"],
    "Commodity": ["GLD", "SLV", "USO", "DBA", "IAU"],
    "Real Estate": ["VNQ", "IYR", "XLRE", "RWR"],
    "ESG": ["ESGU", "ESGV", "SUSL", "USSG", "DSI"],
}

# Retry behaviour tuned for Yahoo Finance's rate limiting on shared-IP hosts
# like Streamlit Community Cloud. Keep attempts low and backoff meaningful —
# hammering a rate limiter with fast retries makes the block worse, not better.
MAX_RETRIES = 3
BASE_BACKOFF_SECONDS = 1.5

# ── Resilient fetch helpers ───────────────────────────────────────────────


def _is_rate_limit_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(
        token in msg
        for token in ["rate limit", "429", "too many requests", "jsondecodeerror"]
    )


def _with_retries(fn, *args, **kwargs):
    """Call fn with exponential backoff + jitter on rate-limit-shaped errors.

    Returns (result, error). Non-rate-limit exceptions are not retried —
    they're returned immediately so the caller can decide what to do
    (e.g. an unknown ticker should fail fast, not retry 3 times).
    """
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            return fn(*args, **kwargs), None
        except Exception as e:  # noqa: BLE001 - yfinance raises many error types
            last_err = e
            if not _is_rate_limit_error(e) or attempt == MAX_RETRIES - 1:
                return None, e
            sleep_for = BASE_BACKOFF_SECONDS * (2 ** attempt) + random.uniform(0, 0.5)
            time.sleep(sleep_for)
    return None, last_err


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_etf_data(ticker: str) -> Optional[Dict]:
    """Fetch ETF data with graceful degradation.

    Strategy, cheapest/most-reliable first:
    1. Try the lightweight `fast_info` endpoint — fewer fields, but far less
       prone to Yahoo's rate limiting than the full `.info` scrape.
    2. Try the full `.info` scrape for the richer field set (expense ratio,
       category, AUM, dividend yield, etc).
    3. Merge whatever succeeded. If BOTH fail, return a dict with an
       explicit "Error" key instead of None, so the UI can tell the
       difference between "ticker doesn't exist" and "Yahoo is blocking us
       right now" and show the right message.
    """
    ticker = ticker.upper().strip()
    etf = yf.Ticker(ticker)

    data: Dict = {"Ticker": ticker, "_data_quality": "full"}

    # Step 1: fast_info (lightweight, more reliable)
    fast = None
    fast_result, fast_err = _with_retries(lambda: dict(etf.fast_info))
    if fast_result:
        fast = fast_result

    # Step 2: full info (rich, more fragile)
    info = None
    info_result, info_err = _with_retries(lambda: etf.info)
    if info_result and info_result.get("regularMarketPrice") is not None:
        info = info_result
    elif info_result and fast is None:
        # info came back but looks empty/invalid, and fast_info also failed
        info = None

    if not fast and not info:
        # Both endpoints failed. Distinguish rate-limit from "ticker invalid".
        err = info_err or fast_err
        if err and _is_rate_limit_error(err):
            return {
                "Ticker": ticker,
                "Error": "rate_limited",
                "_data_quality": "none",
            }
        return {
            "Ticker": ticker,
            "Error": "not_found",
            "_data_quality": "none",
        }

    if not info:
        data["_data_quality"] = "partial"

    # Price: prefer info, fall back to fast_info
    price = (info or {}).get("regularMarketPrice")
    if price is None and fast:
        price = fast.get("last_price")
    price = price or (info or {}).get("previousClose") or 0

    name = (info or {}).get("longName") or (info or {}).get("shortName") or ticker

    er = (info or {}).get("annualReportExpenseRatio")
    expense_ratio = (er * 100) if er is not None else np.nan

    aum = (info or {}).get("totalAssets")
    if aum is None and fast:
        aum = fast.get("market_cap")
    aum_b = (aum or 0) / 1e9

    volume = (info or {}).get("volume")
    if volume is None and fast:
        volume = fast.get("last_volume")
    volume = volume or 0

    avg_vol = (info or {}).get("averageDailyVolume10Day")
    if avg_vol is None and fast:
        avg_vol = fast.get("three_month_average_volume")
    avg_vol = avg_vol or 0

    high_52 = (info or {}).get("fiftyTwoWeekHigh")
    if high_52 is None and fast:
        high_52 = fast.get("year_high")

    low_52 = (info or {}).get("fiftyTwoWeekLow")
    if low_52 is None and fast:
        low_52 = fast.get("year_low")

    nav = (info or {}).get("navPrice") or price

    data.update(
        {
            "Name": name,
            "Price": price,
            "Expense Ratio (%)": expense_ratio,
            "AUM (B)": aum_b,
            "Volume": volume,
            "Avg Daily Volume": avg_vol,
            "Beta": (info or {}).get("beta"),
            "52W High": high_52 or 0,
            "52W Low": low_52 or 0,
            "Category": (info or {}).get("category", "N/A"),
            "Inception Date": (info or {}).get("inceptionDate"),
            "Dividend Yield (%)": ((info or {}).get("yield") or 0) * 100,
            "NAV": nav,
            "52W Change (%)": ((info or {}).get("52WeekChange") or 0) * 100,
            "YTD Return (%)": ((info or {}).get("ytdReturn") or 0) * 100,
        }
    )

    if nav and price and nav > 0:
        data["Premium/Discount (%)"] = ((price - nav) / nav) * 100
    else:
        data["Premium/Discount (%)"] = 0

    # Holdings via the correct yfinance API. fund_holdings.holdings is the
    # real property name; the previous get_holdings() call doesn't exist on
    # yfinance Ticker objects and would always throw.
    top_holdings = "N/A"
    top_weight = 0
    try:
        holdings_df = etf.funds_data.top_holdings
        if isinstance(holdings_df, pd.DataFrame) and not holdings_df.empty:
            name_col = (
                "Name"
                if "Name" in holdings_df.columns
                else holdings_df.columns[0]
            )
            weight_col = next(
                (c for c in holdings_df.columns if "hold" in c.lower() or "weight" in c.lower()),
                holdings_df.columns[-1],
            )
            top_holdings = ", ".join(str(n) for n in holdings_df[name_col].head(5))
            top_weight = float(holdings_df[weight_col].iloc[0]) * 100
    except Exception:
        pass

    data["Top Holdings"] = top_holdings
    data["Top Holding Weight (%)"] = top_weight

    return data


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_historical_data(ticker: str, period: str = "1y") -> Optional[pd.DataFrame]:
    """Fetch historical price data for charts, with retry on rate limiting."""
    def _do_fetch():
        etf = yf.Ticker(ticker)
        hist = etf.history(period=period)
        if hist is None or hist.empty:
            raise ValueError("empty history")
        return hist

    result, err = _with_retries(_do_fetch)
    return result


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_batch_history(tickers: Tuple[str, ...], period: str = "1y") -> Optional[pd.DataFrame]:
    """Batch-download close prices for multiple tickers in ONE request.

    This is both faster and far gentler on Yahoo's rate limiter than calling
    .history() once per ticker in a loop, since yf.download batches the
    underlying HTTP calls. Used for correlation/normalized-return comparison.
    """
    def _do_fetch():
        df = yf.download(
            list(tickers), period=period, group_by="ticker",
            auto_adjust=True, progress=False, threads=True,
        )
        if df is None or df.empty:
            raise ValueError("empty batch download")
        return df

    result, err = _with_retries(_do_fetch)
    return result


def get_error_message(ticker: str, error_code: str) -> str:
    if error_code == "rate_limited":
        return (
            f"⏳ Yahoo Finance is rate-limiting requests for **{ticker}** right now. "
            "This happens often on shared cloud hosting. Wait a minute and try again, "
            "or try fewer tickers at once."
        )
    return f"❌ Could not find data for **{ticker}**. Double check the ticker symbol."


# ── Scoring ────────────────────────────────────────────────────────────────


def calculate_etf_score(etf_data: Dict) -> Tuple[float, Dict]:
    """
    Enhanced scoring with detailed breakdown
    Returns: (total_score, breakdown_dict)
    """
    breakdown = {
        "Expense Ratio": 0,
        "Size (AUM)": 0,
        "Liquidity": 0,
        "Tracking": 0,
        "Diversification": 0,
    }

    weights = {
        "Expense Ratio": 30,
        "Size (AUM)": 25,
        "Liquidity": 20,
        "Tracking": 15,
        "Diversification": 10,
    }

    # 1. Expense Ratio (lower = better)
    er = etf_data.get("Expense Ratio (%)", np.nan)
    if pd.isna(er):
        breakdown["Expense Ratio"] = 50
    elif er <= 0.05:
        breakdown["Expense Ratio"] = 100
    elif er <= 0.10:
        breakdown["Expense Ratio"] = 90
    elif er <= 0.20:
        breakdown["Expense Ratio"] = 75
    elif er <= 0.50:
        breakdown["Expense Ratio"] = 50
    elif er <= 1.00:
        breakdown["Expense Ratio"] = 25
    else:
        breakdown["Expense Ratio"] = 10

    # 2. AUM (higher = better)
    aum = etf_data.get("AUM (B)", 0)
    if aum >= 50:
        breakdown["Size (AUM)"] = 100
    elif aum >= 10:
        breakdown["Size (AUM)"] = 90
    elif aum >= 5:
        breakdown["Size (AUM)"] = 75
    elif aum >= 1:
        breakdown["Size (AUM)"] = 60
    elif aum >= 0.5:
        breakdown["Size (AUM)"] = 40
    else:
        breakdown["Size (AUM)"] = 20

    # 3. Liquidity (Volume)
    vol = etf_data.get("Avg Daily Volume", 0)
    if vol >= 10_000_000:
        breakdown["Liquidity"] = 100
    elif vol >= 5_000_000:
        breakdown["Liquidity"] = 85
    elif vol >= 1_000_000:
        breakdown["Liquidity"] = 70
    elif vol >= 500_000:
        breakdown["Liquidity"] = 50
    elif vol >= 100_000:
        breakdown["Liquidity"] = 30
    else:
        breakdown["Liquidity"] = 15

    # 4. Tracking / Premium-Discount
    pdisc = abs(etf_data.get("Premium/Discount (%)", 0))
    if pdisc < 0.1:
        breakdown["Tracking"] = 100
    elif pdisc < 0.25:
        breakdown["Tracking"] = 85
    elif pdisc < 0.5:
        breakdown["Tracking"] = 70
    elif pdisc < 1.0:
        breakdown["Tracking"] = 50
    else:
        breakdown["Tracking"] = 25

    # 5. Diversification (based on top holding concentration)
    top_weight = etf_data.get("Top Holding Weight (%)", 0)
    if top_weight < 5:
        breakdown["Diversification"] = 100
    elif top_weight < 10:
        breakdown["Diversification"] = 85
    elif top_weight < 15:
        breakdown["Diversification"] = 70
    elif top_weight < 25:
        breakdown["Diversification"] = 50
    else:
        breakdown["Diversification"] = 30

    total_score = sum(breakdown[k] * weights[k] / 100 for k in breakdown.keys())

    return round(total_score, 1), breakdown


# ── Risk metrics (new) ────────────────────────────────────────────────────


def calculate_risk_metrics(hist_data: pd.DataFrame, risk_free_rate: float = 0.04) -> Dict:
    """Compute volatility, Sharpe ratio, max drawdown, and CAGR from price history."""
    if hist_data is None or hist_data.empty or len(hist_data) < 2:
        return {}

    closes = hist_data["Close"].dropna()
    returns = closes.pct_change().dropna()

    if returns.empty:
        return {}

    trading_days = 252
    ann_vol = returns.std() * np.sqrt(trading_days)
    ann_return = returns.mean() * trading_days

    sharpe = (ann_return - risk_free_rate) / ann_vol if ann_vol > 0 else np.nan

    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()

    n_years = len(closes) / trading_days
    cagr = (closes.iloc[-1] / closes.iloc[0]) ** (1 / n_years) - 1 if n_years > 0 and closes.iloc[0] > 0 else np.nan

    return {
        "Annualized Volatility (%)": ann_vol * 100,
        "Annualized Return (%)": ann_return * 100,
        "Sharpe Ratio": sharpe,
        "Max Drawdown (%)": max_drawdown * 100,
        "CAGR (%)": cagr * 100 if pd.notna(cagr) else np.nan,
    }


# ── Charts ─────────────────────────────────────────────────────────────────


def create_price_chart(ticker: str, hist_data: pd.DataFrame, show_ma: bool = True) -> go.Figure:
    """Create an interactive price chart with optional moving averages."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=hist_data.index,
        y=hist_data['Close'],
        mode='lines',
        name='Price',
        line=dict(color='#1f77b4', width=2),
        fill='tonexty',
        fillcolor='rgba(31, 119, 180, 0.1)'
    ))

    if show_ma and len(hist_data) >= 50:
        ma50 = hist_data['Close'].rolling(window=50).mean()
        fig.add_trace(go.Scatter(
            x=hist_data.index, y=ma50, mode='lines', name='50-day MA',
            line=dict(color='#ff7f0e', width=1.5, dash='dash')
        ))
    if show_ma and len(hist_data) >= 200:
        ma200 = hist_data['Close'].rolling(window=200).mean()
        fig.add_trace(go.Scatter(
            x=hist_data.index, y=ma200, mode='lines', name='200-day MA',
            line=dict(color='#9467bd', width=1.5, dash='dot')
        ))

    fig.update_layout(
        title=f"{ticker} Price History",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        hovermode='x unified',
        template='plotly_white',
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig


def create_score_chart(breakdown: Dict) -> go.Figure:
    """Create a radar chart for score breakdown"""
    categories = list(breakdown.keys())
    values = list(breakdown.values())

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Score',
        line=dict(color='#2ecc71', width=2),
        fillcolor='rgba(46, 204, 113, 0.3)'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        showlegend=False,
        height=400,
        title="ETF Quality Breakdown"
    )

    return fig


def create_comparison_chart(comparison_df: pd.DataFrame, metric: str) -> go.Figure:
    """Create a comparison bar chart"""
    fig = px.bar(
        comparison_df,
        x='Ticker',
        y=metric,
        color='Ticker',
        title=f"Comparison: {metric}",
        text_auto='.2f'
    )

    fig.update_layout(
        showlegend=False,
        template='plotly_white',
        height=350
    )

    return fig


def create_normalized_return_chart(batch_df: pd.DataFrame, tickers: List[str]) -> Optional[go.Figure]:
    """Plot all tickers' returns normalized to 100 at the start, for fair comparison."""
    fig = go.Figure()
    plotted = 0
    for ticker in tickers:
        try:
            if len(tickers) == 1:
                closes = batch_df['Close']
            else:
                closes = batch_df[ticker]['Close']
            closes = closes.dropna()
            if closes.empty:
                continue
            normalized = (closes / closes.iloc[0]) * 100
            fig.add_trace(go.Scatter(x=normalized.index, y=normalized, mode='lines', name=ticker))
            plotted += 1
        except Exception:
            continue

    if plotted == 0:
        return None

    fig.update_layout(
        title="Normalized Return Comparison (Base = 100)",
        xaxis_title="Date",
        yaxis_title="Indexed Value",
        template='plotly_white',
        height=400,
        hovermode='x unified',
    )
    return fig


def create_correlation_heatmap(batch_df: pd.DataFrame, tickers: List[str]) -> Optional[go.Figure]:
    """Correlation matrix of daily returns across the selected tickers."""
    returns = {}
    for ticker in tickers:
        try:
            closes = batch_df['Close'] if len(tickers) == 1 else batch_df[ticker]['Close']
            returns[ticker] = closes.pct_change().dropna()
        except Exception:
            continue

    if len(returns) < 2:
        return None

    ret_df = pd.DataFrame(returns).dropna()
    if ret_df.empty:
        return None

    corr = ret_df.corr()

    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale='RdBu',
        zmid=0,
        zmin=-1,
        zmax=1,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        colorbar=dict(title="Correlation"),
    ))
    fig.update_layout(title="Return Correlation Matrix", height=400, template='plotly_white')
    return fig


# ── UI Components ──────────────────────────────────────────────────────────


def render_etf_card(data: Dict, score: float, breakdown: Dict):
    """Render a detailed ETF information card"""

    if data.get("_data_quality") == "partial":
        st.info(
            "ℹ️ Yahoo's detailed data feed was unavailable, so some fields below "
            "(expense ratio, category, dividend yield) use lighter-weight backup "
            "data and may show as N/A. Price and volume are still live."
        )

    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"### {data['Name']}")
        st.caption(f"**Ticker:** {data['Ticker']} | **Category:** {data.get('Category', 'N/A')}")
    with col2:
        if score >= 85:
            rating = "Excellent"
        elif score >= 70:
            rating = "Good"
        elif score >= 50:
            rating = "Fair"
        else:
            rating = "Poor"

        st.metric("Quality Score", f"{score}/100", delta=rating)

    st.divider()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Price", f"${data['Price']:.2f}")
        st.metric("52W High", f"${data.get('52W High', 0):.2f}")

    with col2:
        er = data['Expense Ratio (%)']
        try:
            if er is None or (isinstance(er, float) and (np.isnan(er) or pd.isna(er))):
                er_display = "N/A"
            else:
                er_display = f"{float(er):.3f}%"
        except (ValueError, TypeError):
            er_display = "N/A"
        st.metric("Expense Ratio", er_display)
        st.metric("52W Low", f"${data.get('52W Low', 0):.2f}")

    with col3:
        st.metric("AUM", f"${data['AUM (B)']:.2f}B")
        st.metric("Dividend Yield", f"{data['Dividend Yield (%)']:.2f}%")

    with col4:
        vol = data['Avg Daily Volume']
        vol_str = f"{vol/1e6:.1f}M" if vol >= 1e6 else f"{vol/1e3:.0f}K"
        st.metric("Avg Volume", vol_str)
        st.metric("Beta", f"{data.get('Beta', 0):.2f}" if data.get('Beta') else "N/A")

    if data.get('Top Holdings') != "N/A":
        st.caption(f"**Top 5 Holdings:** {data['Top Holdings']}")
        if data.get('Top Holding Weight (%)'):
            st.caption(f"**Top Holding Weight:** {data['Top Holding Weight (%)']:.2f}%")

    with st.expander("📊 Additional Metrics"):
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Premium/Discount:** {data.get('Premium/Discount (%)', 0):.3f}%")
            st.write(f"**52W Change:** {data.get('52W Change (%)', 0):.2f}%")
        with col2:
            st.write(f"**Volume:** {data['Volume']:,}")
            if data.get('Inception Date'):
                st.write(f"**Inception Date:** {data['Inception Date']}")


def render_risk_metrics(risk: Dict):
    """Render the risk metrics row — new addition."""
    if not risk:
        st.warning("Not enough price history to compute risk metrics.")
        return

    st.markdown("### 📉 Risk & Return Metrics")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("CAGR", f"{risk.get('CAGR (%)', float('nan')):.2f}%")
    with col2:
        st.metric("Ann. Volatility", f"{risk.get('Annualized Volatility (%)', float('nan')):.2f}%")
    with col3:
        sharpe = risk.get("Sharpe Ratio", float('nan'))
        st.metric("Sharpe Ratio", f"{sharpe:.2f}" if pd.notna(sharpe) else "N/A")
    with col4:
        st.metric("Max Drawdown", f"{risk.get('Max Drawdown (%)', float('nan')):.2f}%")
    with col5:
        st.metric("Ann. Return", f"{risk.get('Annualized Return (%)', float('nan')):.2f}%")
    st.caption("Sharpe ratio assumes a 4% annual risk-free rate. All figures computed from the selected chart period's price history.")


# ── Main Application ───────────────────────────────────────────────────────


def render_etf_analyzer():
    st.title("📊 ETF Analyzer & Screener Pro")
    st.markdown("Advanced ETF analysis with quality scoring, screening, and comparison tools")

    with st.sidebar:
        st.header("⚙️ Settings")
        chart_period = st.selectbox(
            "Chart Period",
            ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
            index=3
        )
        show_ma = st.checkbox("Show moving averages", value=True)

        st.divider()
        st.markdown("### 📖 About")
        st.caption("""
        This tool analyzes ETFs based on:
        - **Expense Ratio** (30%): Lower is better
        - **Size/AUM** (25%): Larger funds are more stable
        - **Liquidity** (20%): Higher volume = easier trading
        - **Tracking** (15%): How well it tracks NAV
        - **Diversification** (10%): Lower concentration = better
        """)
        st.divider()
        st.caption(
            "⚠️ Data comes from Yahoo Finance via `yfinance`. On shared cloud "
            "hosting, Yahoo occasionally rate-limits requests — if a fetch "
            "fails, wait a moment and retry. Data is cached for 1 hour to "
            "reduce repeat calls."
        )

    mode = st.radio(
        "Select Mode",
        ["🔍 Single ETF Analysis", "📋 ETF Screener", "⚖️ Compare ETFs", "🌍 Browse by Category"],
        horizontal=True
    )

    st.divider()

    # ── Single ETF Analysis ────────────────────────────────────────────────
    if mode == "🔍 Single ETF Analysis":
        col1, col2 = st.columns([3, 1])
        with col1:
            ticker = st.text_input("Enter ETF Ticker", "SPY", help="E.g., SPY, QQQ, VTI").upper().strip()
        with col2:
            analyze_btn = st.button("🔎 Analyze", type="primary", use_container_width=True)

        if analyze_btn and ticker:
            with st.spinner(f"Analyzing {ticker}..."):
                data = fetch_etf_data(ticker)

                if not data or data.get("Error"):
                    msg = get_error_message(ticker, data.get("Error", "not_found") if data else "not_found")
                    st.error(msg)
                    return

                score, breakdown = calculate_etf_score(data)
                render_etf_card(data, score, breakdown)

                st.divider()
                col1, col2 = st.columns(2)

                hist_data = fetch_historical_data(ticker, chart_period)
                with col1:
                    if hist_data is not None:
                        st.plotly_chart(create_price_chart(ticker, hist_data, show_ma), use_container_width=True, key=f"price_chart_{ticker}")
                    else:
                        st.warning("Price history not available right now (Yahoo may be rate-limiting). Try again shortly.")

                with col2:
                    st.plotly_chart(create_score_chart(breakdown), use_container_width=True, key=f"score_chart_{ticker}")

                # Risk metrics (new)
                st.divider()
                risk = calculate_risk_metrics(hist_data) if hist_data is not None else {}
                render_risk_metrics(risk)

                st.divider()
                st.markdown("### 📊 Quality Score Breakdown")

                if pd.isna(data['Expense Ratio (%)']):
                    st.warning("⚠️ Expense ratio data not available from Yahoo Finance for this ETF. Using neutral score (50/100) for this metric.")

                breakdown_df = pd.DataFrame([
                    {"Metric": k, "Score": f"{v}/100", "Weight": f"{w}%", "Weighted": f"{v*w/100:.1f}"}
                    for k, v, w in zip(
                        breakdown.keys(),
                        breakdown.values(),
                        [30, 25, 20, 15, 10]
                    )
                ])
                st.dataframe(breakdown_df, use_container_width=True, hide_index=True)

    # ── ETF Screener ───────────────────────────────────────────────────────
    elif mode == "📋 ETF Screener":
        st.markdown("### 🔍 Screen ETFs by Criteria")

        col1, col2, col3 = st.columns(3)

        with col1:
            categories = ["All"] + list(ETF_UNIVERSE.keys())
            selected_category = st.selectbox("Category", categories)

        with col2:
            min_aum = st.number_input("Min AUM ($B)", value=0.5, step=0.5, min_value=0.0)

        with col3:
            max_expense = st.number_input("Max Expense Ratio (%)", value=0.50, step=0.05, min_value=0.0)

        col1, col2, col3 = st.columns(3)

        with col1:
            min_volume = st.number_input("Min Avg Volume", value=100_000, step=100_000, min_value=0)

        with col2:
            min_score = st.slider("Min Quality Score", 0, 100, 50)

        with col3:
            sort_by = st.selectbox("Sort By", ["Score", "AUM (B)", "Expense (%)", "Volume"])

        if st.button("🚀 Run Screener", type="primary"):
            if selected_category == "All":
                etf_list = [etf for etfs in ETF_UNIVERSE.values() for etf in etfs]
            else:
                etf_list = ETF_UNIVERSE.get(selected_category, [])

            results = []
            failed = []
            progress_bar = st.progress(0)
            status = st.empty()

            for idx, ticker in enumerate(etf_list):
                status.caption(f"Fetching {ticker}... ({idx + 1}/{len(etf_list)})")
                progress_bar.progress((idx + 1) / len(etf_list))
                data = fetch_etf_data(ticker)

                if not data or data.get("Error"):
                    failed.append((ticker, data.get("Error", "not_found") if data else "not_found"))
                    continue

                score, _ = calculate_etf_score(data)

                er = data["Expense Ratio (%)"]
                er_check = pd.notna(er) and er <= max_expense

                if (data["AUM (B)"] >= min_aum and
                        er_check and
                        data["Avg Daily Volume"] >= min_volume and
                        score >= min_score):

                    results.append({
                        "Ticker": ticker,
                        "Name": data["Name"][:40] + "..." if len(data["Name"]) > 40 else data["Name"],
                        "Score": score,
                        "AUM (B)": data["AUM (B)"],
                        "Expense (%)": data["Expense Ratio (%)"],
                        "Volume": data["Avg Daily Volume"],
                        "Yield (%)": data["Dividend Yield (%)"],
                        "Category": data.get("Category", "N/A")
                    })

            progress_bar.empty()
            status.empty()

            if failed:
                rate_limited = [t for t, e in failed if e == "rate_limited"]
                not_found = [t for t, e in failed if e == "not_found"]
                if rate_limited:
                    st.warning(f"⏳ Rate-limited by Yahoo Finance for: {', '.join(rate_limited)}. Try re-running the screener in a minute.")
                if not_found:
                    st.caption(f"Could not find data for: {', '.join(not_found)}")

            if results:
                df = pd.DataFrame(results)

                sort_col = {
                    "Score": "Score",
                    "AUM (B)": "AUM (B)",
                    "Expense (%)": "Expense (%)",
                    "Volume": "Volume"
                }[sort_by]

                ascending = sort_by == "Expense (%)"
                df = df.sort_values(sort_col, ascending=ascending)

                st.success(f"✅ Found {len(df)} ETFs matching your criteria")

                st.dataframe(
                    df.style.format({
                        "AUM (B)": "{:.2f}",
                        "Expense (%)": lambda x: f"{x:.3f}" if x is not None and pd.notna(x) else "N/A",
                        "Score": "{:.1f}",
                        "Yield (%)": "{:.2f}",
                        "Volume": "{:,.0f}"
                    }).background_gradient(subset=["Score"], cmap="RdYlGn", vmin=0, vmax=100),
                    use_container_width=True,
                    height=500
                )

                csv = df.to_csv(index=False)
                st.download_button(
                    "📥 Download Results (CSV)",
                    csv,
                    "etf_screener_results.csv",
                    "text/csv"
                )
            elif not failed:
                st.warning("⚠️ No ETFs matched your filters. Try adjusting your criteria.")

    # ── Compare ETFs ───────────────────────────────────────────────────────
    elif mode == "⚖️ Compare ETFs":
        st.markdown("### ⚖️ Side-by-Side ETF Comparison")
        st.caption("Compare up to 5 ETFs at once, including correlation and normalized returns.")

        default_tickers = ["SPY", "QQQ", "", "", ""]
        cols = st.columns(5)
        tickers_input = []
        for i, col in enumerate(cols):
            with col:
                label = f"ETF {i + 1}" + (" (Optional)" if i >= 2 else "")
                tickers_input.append(st.text_input(label, default_tickers[i], key=f"etf_compare_{i}").upper().strip())

        if st.button("🔄 Compare", type="primary"):
            etfs_to_compare = [etf for etf in tickers_input if etf]

            if len(etfs_to_compare) < 2:
                st.warning("Please enter at least 2 ETFs to compare")
                return

            with st.spinner("Fetching comparison data..."):
                comparison_data = []
                scores = []
                failed = []

                for ticker in etfs_to_compare:
                    data = fetch_etf_data(ticker)
                    if data and not data.get("Error"):
                        score, breakdown = calculate_etf_score(data)
                        comparison_data.append(data)
                        scores.append((ticker, score, breakdown))
                    else:
                        failed.append((ticker, data.get("Error", "not_found") if data else "not_found"))

                for ticker, err in failed:
                    st.error(get_error_message(ticker, err))

                if len(comparison_data) >= 2:
                    comparison_df = pd.DataFrame(comparison_data)
                    comparison_df = comparison_df.set_index('Ticker')

                    metrics_to_show = [
                        'Name', 'Price', 'Expense Ratio (%)', 'AUM (B)',
                        'Avg Daily Volume', 'Dividend Yield (%)', 'Beta',
                        '52W High', '52W Low', 'Premium/Discount (%)'
                    ]

                    def format_value(x):
                        if x is None or pd.isna(x):
                            return "N/A"
                        elif isinstance(x, (int, float)):
                            return f"{x:.3f}"
                        else:
                            return str(x)

                    styled_df = comparison_df[metrics_to_show].T.style.format(
                        format_value,
                        subset=pd.IndexSlice[:, :]
                    )

                    st.dataframe(styled_df, use_container_width=True)

                    st.divider()

                    st.markdown("### 📊 Quality Score Comparison")

                    score_df = pd.DataFrame([
                        {'Ticker': ticker, 'Score': score}
                        for ticker, score, _ in scores
                    ])

                    fig = px.bar(
                        score_df, x='Ticker', y='Score', color='Ticker',
                        title="Overall Quality Scores", text_auto='.1f',
                        color_discrete_sequence=px.colors.qualitative.Set2
                    )
                    fig.update_layout(showlegend=False, template='plotly_white', height=400)
                    fig.update_yaxes(range=[0, 100])
                    st.plotly_chart(fig, use_container_width=True, key="comparison_overall_scores")

                    st.markdown("### 🎯 Detailed Score Breakdown")

                    cols = st.columns(len(scores))
                    for idx, (ticker, score, breakdown) in enumerate(scores):
                        with cols[idx]:
                            st.markdown(f"**{ticker}**")
                            st.plotly_chart(
                                create_score_chart(breakdown),
                                use_container_width=True,
                                config={'displayModeBar': False},
                                key=f"comparison_radar_{ticker}_{idx}"
                            )

                    st.markdown("### 📈 Metric Comparisons")

                    col1, col2 = st.columns(2)

                    with col1:
                        fig = create_comparison_chart(
                            comparison_df.reset_index()[['Ticker', 'Expense Ratio (%)']],
                            'Expense Ratio (%)'
                        )
                        st.plotly_chart(fig, use_container_width=True, key="comparison_expense_ratio")

                    with col2:
                        fig = create_comparison_chart(
                            comparison_df.reset_index()[['Ticker', 'AUM (B)']],
                            'AUM (B)'
                        )
                        st.plotly_chart(fig, use_container_width=True, key="comparison_aum")

                    # New: normalized returns + correlation, fetched as ONE
                    # batched call rather than N separate .history() calls.
                    st.divider()
                    st.markdown("### 📉 Historical Performance Comparison")
                    valid_tickers = [t for t in etfs_to_compare if t not in [f for f, _ in failed]]
                    batch_hist = fetch_batch_history(tuple(valid_tickers), chart_period)

                    if batch_hist is not None:
                        norm_fig = create_normalized_return_chart(batch_hist, valid_tickers)
                        if norm_fig:
                            st.plotly_chart(norm_fig, use_container_width=True, key="normalized_returns")
                        else:
                            st.caption("Could not compute normalized returns from the batch data.")

                        if len(valid_tickers) >= 2:
                            corr_fig = create_correlation_heatmap(batch_hist, valid_tickers)
                            if corr_fig:
                                st.plotly_chart(corr_fig, use_container_width=True, key="correlation_heatmap")
                                st.caption("Closer to +1 = moves together; closer to -1 = moves oppositely. Useful for spotting redundant holdings.")
                    else:
                        st.warning("Could not fetch historical price data for the comparison charts right now (Yahoo may be rate-limiting batch downloads). Try again shortly.")

    # ── Browse by Category ─────────────────────────────────────────────────
    else:  # Browse by Category
        st.markdown("### 🌍 Browse ETFs by Category")

        for category, etfs in ETF_UNIVERSE.items():
            with st.expander(f"**{category}** ({len(etfs)} ETFs)", expanded=False):
                st.write(", ".join(etfs))

        st.divider()

        selected_cat = st.selectbox("Select a category to analyze", list(ETF_UNIVERSE.keys()))

        if st.button("📊 Analyze Category", type="primary"):
            etf_list = ETF_UNIVERSE[selected_cat]
            results = []
            failed = []

            progress_bar = st.progress(0)
            status = st.empty()

            for idx, ticker in enumerate(etf_list):
                status.caption(f"Fetching {ticker}... ({idx + 1}/{len(etf_list)})")
                progress_bar.progress((idx + 1) / len(etf_list))
                data = fetch_etf_data(ticker)

                if not data or data.get("Error"):
                    failed.append((ticker, data.get("Error", "not_found") if data else "not_found"))
                    continue

                score, _ = calculate_etf_score(data)
                results.append({
                    "Ticker": ticker,
                    "Name": data["Name"][:40] + "..." if len(data["Name"]) > 40 else data["Name"],
                    "Score": score,
                    "Price": data["Price"],
                    "AUM (B)": data["AUM (B)"],
                    "Expense (%)": data["Expense Ratio (%)"],
                    "Yield (%)": data["Dividend Yield (%)"]
                })

            progress_bar.empty()
            status.empty()

            if failed:
                rate_limited = [t for t, e in failed if e == "rate_limited"]
                if rate_limited:
                    st.warning(f"⏳ Rate-limited by Yahoo Finance for: {', '.join(rate_limited)}. Try again in a minute.")

            if results:
                df = pd.DataFrame(results).sort_values("Score", ascending=False)

                st.success(f"✅ Analyzed {len(df)} ETFs in {selected_cat}")

                st.dataframe(
                    df.style.format({
                        "Price": "${:.2f}",
                        "AUM (B)": "{:.2f}",
                        "Expense (%)": lambda x: f"{x:.3f}" if x is not None and pd.notna(x) else "N/A",
                        "Score": "{:.1f}",
                        "Yield (%)": "{:.2f}"
                    }).background_gradient(subset=["Score"], cmap="RdYlGn", vmin=0, vmax=100),
                    use_container_width=True,
                    height=500
                )

                # New: one-click batched performance chart for the whole category
                st.divider()
                if st.checkbox("📉 Show normalized performance chart for this category", value=False):
                    batch_hist = fetch_batch_history(tuple(df["Ticker"].tolist()), chart_period)
                    if batch_hist is not None:
                        norm_fig = create_normalized_return_chart(batch_hist, df["Ticker"].tolist())
                        if norm_fig:
                            st.plotly_chart(norm_fig, use_container_width=True, key="category_normalized_returns")
                    else:
                        st.warning("Could not fetch batch history right now. Try again shortly.")


# ── Entry Point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    st.set_page_config(
        page_title="ETF Analyzer Pro",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    render_etf_analyzer()
