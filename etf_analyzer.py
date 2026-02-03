import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import warnings

warnings.filterwarnings('ignore')

# ── Configuration ──────────────────────────────────────────────────────────
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
    "ESG": ["ESGU", "ESGV", "SUSL", "USSG", "DSI"]
}

# ── Helper Functions ───────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_etf_data(ticker: str) -> Optional[Dict]:
    """Fetch comprehensive ETF data with improved expense ratio & holdings"""
    try:
        etf = yf.Ticker(ticker)
        info = etf.info

        if not info or info.get('regularMarketPrice') is None:
            return None

        data = {
            "Ticker": ticker,
            "Name": info.get("longName") or info.get("shortName", ticker),
            "Price": info.get("regularMarketPrice") or info.get("previousClose", 0),
            "AUM (B)": (info.get("totalAssets") or 0) / 1e9,
            "Volume": info.get("volume", 0),
            "Avg Daily Volume": info.get("averageDailyVolume10Day", 0) or info.get("averageVolume", 0),
            "Beta": info.get("beta"),
            "52W High": info.get("fiftyTwoWeekHigh"),
            "52W Low": info.get("fiftyTwoWeekLow"),
            "Category": info.get("category", "N/A"),
            "Inception Date": info.get("inceptionDate"),
            "Dividend Yield (%)": (info.get("yield") or info.get("trailingAnnualDividendYield") or 0) * 100,
            "NAV": info.get("navPrice") or info.get("regularMarketPrice", 0),
            "52W Change (%)": info.get("52WeekChange", 0) * 100,
            "YTD Return (%)": info.get("ytdReturn", 0) * 100,
        }

        # ── Expense Ratio (main fix) ───────────────────────────────────────
        expense_ratio = np.nan
        try:
            if hasattr(etf, 'funds_data') and etf.funds_data is not None:
                fd = etf.funds_data
                er_raw = getattr(fd, 'annualReportExpenseRatio', None)
                if er_raw is not None:
                    expense_ratio = float(er_raw) * 100
        except:
            pass

        data["Expense Ratio (%)"] = expense_ratio

        # ── Top Holdings (main fix) ────────────────────────────────────────
        try:
            if hasattr(etf, 'funds_data') and etf.funds_data is not None:
                holdings_df = etf.funds_data.top_holdings
                if isinstance(holdings_df, pd.DataFrame) and not holdings_df.empty:
                    # Usually index = holding name, columns include % weight
                    data["Top Holdings"] = ", ".join(holdings_df.index[:5].astype(str))
                    # Find percentage column (names vary: % Holding, percent, etc.)
                    weight_col = next(
                        (c for c in holdings_df.columns if any(k in c.lower() for k in ["percent", "%", "weight"])),
                        holdings_df.columns[-1]  # fallback to last column
                    )
                    top_weight = holdings_df.iloc[0][weight_col]
                    data["Top Holding Weight (%)"] = float(top_weight) if pd.notna(top_weight) else 0.0
                else:
                    raise ValueError
            else:
                raise ValueError
        except:
            data["Top Holdings"] = "N/A"
            data["Top Holding Weight (%)"] = 0.0

        # Premium/Discount
        nav = info.get("navPrice")
        price = info.get("regularMarketPrice")
        if nav and price and nav > 0:
            data["Premium/Discount (%)"] = ((price - nav) / nav) * 100
        else:
            data["Premium/Discount (%)"] = 0.0

        return data

    except Exception as e:
        st.warning(f"Error fetching {ticker}: {str(e)}")
        return None


@st.cache_data(ttl=3600)
def fetch_historical_data(ticker: str, period: str = "1y") -> Optional[pd.DataFrame]:
    """Fetch historical price data for charts"""
    try:
        etf = yf.Ticker(ticker)
        hist = etf.history(period=period)
        return hist if not hist.empty else None
    except:
        return None


def calculate_etf_score(etf_data: Dict) -> tuple[float, Dict]:
    """
    Enhanced scoring with detailed breakdown
    Returns: (total_score, breakdown_dict)
    """
    breakdown = {
        "Expense Ratio": 0,
        "Size (AUM)": 0,
        "Liquidity": 0,
        "Tracking": 0,
        "Diversification": 0
    }

    weights = {
        "Expense Ratio": 30,
        "Size (AUM)": 25,
        "Liquidity": 20,
        "Tracking": 15,
        "Diversification": 10
    }

    # 1. Expense Ratio (lower = better)
    er = etf_data.get("Expense Ratio (%)", np.nan)
    if pd.isna(er):
        breakdown["Expense Ratio"] = 50      # neutral when missing
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

    # 5. Diversification (based on top holding concentration) — now works!
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

    # Calculate weighted total
    total_score = sum(breakdown[k] * weights[k] / 100 for k in breakdown)

    return round(total_score, 1), breakdown


def create_price_chart(ticker: str, hist_data: pd.DataFrame) -> go.Figure:
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

    fig.update_layout(
        title=f"{ticker} Price History",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        hovermode='x unified',
        template='plotly_white',
        height=400
    )

    return fig


def create_score_chart(breakdown: Dict) -> go.Figure:
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


# ── UI Components ──────────────────────────────────────────────────────────
def render_etf_card(data: Dict, score: float, breakdown: Dict):
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"### {data['Name']}")
        st.caption(f"**Ticker:** {data['Ticker']} | **Category:** {data.get('Category', 'N/A')}")
    with col2:
        if score >= 85:
            color, rating = "🟢 Excellent", "Excellent"
        elif score >= 70:
            color, rating = "🟡 Good", "Good"
        elif score >= 50:
            color, rating = "🟠 Fair", "Fair"
        else:
            color, rating = "🔴 Poor", "Poor"

        st.metric("Quality Score", f"{score}/100", delta=rating)

    st.divider()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Price", f"${data['Price']:.2f}")
        st.metric("52W High", f"${data.get('52W High', 0):.2f}")

    with col2:
        er = data['Expense Ratio (%)']
        er_display = f"{er:.3f}%" if pd.notna(er) else "N/A"
        st.metric("Expense Ratio", er_display)
        st.metric("52W Low", f"${data.get('52W Low', 0):.2f}")

    with col3:
        st.metric("AUM", f"${data['AUM (B)']:.2f}B")
        st.metric("Dividend Yield", f"{data['Dividend Yield (%)']:.2f}%")

    with col4:
        vol = data['Avg Daily Volume']
        vol_str = f"{vol/1e6:.1f}M" if vol >= 1e6 else f"{vol/1e3:.0f}K"
        st.metric("Avg Volume", vol_str)
        st.metric("Beta", f"{data.get('Beta', 'N/A'):.2f}" if data.get('Beta') else "N/A")

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
                try:
                    dt = datetime.fromtimestamp(data['Inception Date'])
                    st.write(f"**Inception:** {dt.strftime('%Y-%m')}")
                except:
                    st.write(f"**Inception:** {data['Inception Date']}")


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

    mode = st.radio(
        "Select Mode",
        ["🔍 Single ETF Analysis", "📋 ETF Screener", "⚖️ Compare ETFs", "🌍 Browse by Category"],
        horizontal=True
    )

    st.divider()

    if mode == "🔍 Single ETF Analysis":
        col1, col2 = st.columns([3, 1])
        with col1:
            ticker = st.text_input("Enter ETF Ticker", "SPY", help="E.g., SPY, QQQ, VTI").upper().strip()
        with col2:
            analyze_btn = st.button("🔎 Analyze", type="primary", use_container_width=True)

        if analyze_btn and ticker:
            with st.spinner(f"Analyzing {ticker}..."):
                data = fetch_etf_data(ticker)

                if not data:
                    st.error(f"❌ Could not fetch data for {ticker}. Please check the ticker symbol.")
                    return

                score, breakdown = calculate_etf_score(data)

                render_etf_card(data, score, breakdown)

                st.divider()
                col1, col2 = st.columns(2)

                with col1:
                    hist_data = fetch_historical_data(ticker, chart_period)
                    if hist_data is not None:
                        st.plotly_chart(create_price_chart(ticker, hist_data), use_container_width=True)
                    else:
                        st.warning("Price history not available")

                with col2:
                    st.plotly_chart(create_score_chart(breakdown), use_container_width=True)

                st.markdown("### 📊 Quality Score Breakdown")

                if pd.isna(data['Expense Ratio (%)']):
                    st.warning("⚠️ Expense ratio data not available from Yahoo Finance for this ETF.")

                breakdown_df = pd.DataFrame([
                    {"Metric": k, "Score": f"{v}/100", "Weight": f"{w}%", "Weighted": f"{v * w / 100:.1f}"}
                    for k, v, w in zip(breakdown.keys(), breakdown.values(), [30, 25, 20, 15, 10])
                ])
                st.dataframe(breakdown_df, use_container_width=True, hide_index=True)

    # ── (The rest of the modes — Screener, Compare, Browse — remain unchanged)
    # You can keep your original code for those sections or let me know if you want them adjusted too.

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
            with st.spinner("Screening ETFs..."):
                if selected_category == "All":
                    etf_list = [etf for etfs in ETF_UNIVERSE.values() for etf in etfs]
                else:
                    etf_list = ETF_UNIVERSE.get(selected_category, [])

                results = []
                progress_bar = st.progress(0)

                for idx, tkr in enumerate(etf_list):
                    progress_bar.progress((idx + 1) / len(etf_list))
                    data = fetch_etf_data(tkr)
                    if data:
                        score, _ = calculate_etf_score(data)
                        er = data["Expense Ratio (%)"]
                        if (data["AUM (B)"] >= min_aum and
                            (pd.isna(er) or er <= max_expense) and
                            data["Avg Daily Volume"] >= min_volume and
                            score >= min_score):
                            results.append({
                                "Ticker": tkr,
                                "Name": data["Name"][:40] + "..." if len(data["Name"]) > 40 else data["Name"],
                                "Score": score,
                                "AUM (B)": data["AUM (B)"],
                                "Expense (%)": data["Expense Ratio (%)"],
                                "Volume": data["Avg Daily Volume"],
                                "Yield (%)": data["Dividend Yield (%)"],
                                "Category": data.get("Category", "N/A")
                            })

                progress_bar.empty()

                if results:
                    df = pd.DataFrame(results)
                    sort_col = {"Score": "Score", "AUM (B)": "AUM (B)", "Expense (%)": "Expense (%)", "Volume": "Volume"}[sort_by]
                    df = df.sort_values(sort_col, ascending=(sort_by == "Expense (%)"))

                    st.success(f"✅ Found {len(df)} ETFs matching your criteria")

                    st.dataframe(
                        df.style.format({
                            "AUM (B)": "{:.2f}",
                            "Expense (%)": lambda x: f"{x:.3f}" if pd.notna(x) else "N/A",
                            "Score": "{:.1f}",
                            "Yield (%)": "{:.2f}",
                            "Volume": "{:,.0f}"
                        }).background_gradient(subset=["Score"], cmap="RdYlGn", vmin=0, vmax=100),
                        use_container_width=True,
                        height=500
                    )

                    st.download_button(
                        "📥 Download Results (CSV)",
                        df.to_csv(index=False),
                        "etf_screener_results.csv",
                        "text/csv"
                    )
                else:
                    st.warning("⚠️ No ETFs matched your filters. Try adjusting criteria.")

    elif mode == "⚖️ Compare ETFs":
        st.markdown("### ⚖️ Side-by-Side ETF Comparison")

        col1, col2, col3 = st.columns(3)
        with col1:
            etf1 = st.text_input("ETF 1", "SPY").upper().strip()
        with col2:
            etf2 = st.text_input("ETF 2", "QQQ").upper().strip()
        with col3:
            etf3 = st.text_input("ETF 3 (Optional)", "").upper().strip()

        if st.button("🔄 Compare", type="primary"):
            etfs = [e for e in [etf1, etf2, etf3] if e]
            if len(etfs) < 2:
                st.warning("Please enter at least 2 ETFs to compare")
                return

            with st.spinner("Fetching comparison data..."):
                comparison_data = []
                scores = []

                for tkr in etfs:
                    data = fetch_etf_data(tkr)
                    if data:
                        score, breakdown = calculate_etf_score(data)
                        comparison_data.append(data)
                        scores.append((tkr, score, breakdown))
                    else:
                        st.error(f"Could not fetch data for {tkr}")

                if len(comparison_data) >= 2:
                    df = pd.DataFrame(comparison_data).set_index('Ticker')

                    metrics = [
                        'Name', 'Price', 'Expense Ratio (%)', 'AUM (B)',
                        'Avg Daily Volume', 'Dividend Yield (%)', 'Beta',
                        '52W High', '52W Low', 'Premium/Discount (%)'
                    ]

                    def fmt(x):
                        if pd.isna(x) or x is None: return "N/A"
                        if isinstance(x, (int, float)): return f"{x:.3f}"
                        return str(x)

                    st.dataframe(
                        df[metrics].T.style.format(fmt),
                        use_container_width=True
                    )

                    st.divider()
                    st.markdown("### 📊 Quality Score Comparison")

                    score_df = pd.DataFrame([{'Ticker': t, 'Score': s} for t, s, _ in scores])
                    fig = px.bar(score_df, x='Ticker', y='Score', color='Ticker',
                                 title="Overall Quality Scores", text_auto='.1f')
                    fig.update_layout(showlegend=False, template='plotly_white', height=400)
                    fig.update_yaxes(range=[0, 100])
                    st.plotly_chart(fig, use_container_width=True)

                    st.markdown("### 🎯 Detailed Score Breakdown")
                    cols = st.columns(len(scores))
                    for i, (tkr, _, brk) in enumerate(scores):
                        with cols[i]:
                            st.markdown(f"**{tkr}**")
                            st.plotly_chart(create_score_chart(brk), use_container_width=True)

    else:  # Browse by Category
        st.markdown("### 🌍 Browse ETFs by Category")

        for cat, tickers in ETF_UNIVERSE.items():
            with st.expander(f"**{cat}** ({len(tickers)} ETFs)"):
                st.write(", ".join(tickers))

        st.divider()
        selected_cat = st.selectbox("Select a category to analyze", list(ETF_UNIVERSE.keys()))

        if st.button("📊 Analyze Category", type="primary"):
            with st.spinner(f"Analyzing {selected_cat} ETFs..."):
                etfs = ETF_UNIVERSE[selected_cat]
                results = []
                progress = st.progress(0)

                for i, tkr in enumerate(etfs):
                    progress.progress((i + 1) / len(etfs))
                    data = fetch_etf_data(tkr)
                    if data:
                        score, _ = calculate_etf_score(data)
                        results.append({
                            "Ticker": tkr,
                            "Name": data["Name"][:40] + "..." if len(data["Name"]) > 40 else data["Name"],
                            "Score": score,
                            "Price": data["Price"],
                            "AUM (B)": data["AUM (B)"],
                            "Expense (%)": data["Expense Ratio (%)"],
                            "Yield (%)": data["Dividend Yield (%)"]
                        })

                progress.empty()

                if results:
                    df = pd.DataFrame(results).sort_values("Score", ascending=False)
                    st.success(f"✅ Analyzed {len(df)} ETFs in {selected_cat}")
                    st.dataframe(
                        df.style.format({
                            "Price": "${:.2f}",
                            "AUM (B)": "{:.2f}",
                            "Expense (%)": lambda x: f"{x:.3f}" if pd.notna(x) else "N/A",
                            "Score": "{:.1f}",
                            "Yield (%)": "{:.2f}"
                        }).background_gradient(subset=["Score"], cmap="RdYlGn"),
                        use_container_width=True,
                        height=500
                    )


if __name__ == "__main__":
    st.set_page_config(
        page_title="ETF Analyzer Pro",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    render_etf_analyzer()
