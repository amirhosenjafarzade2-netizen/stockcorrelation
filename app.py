# app.py
# Stock / Asset Analyzer - Modern Streamlit Multi-Module Application
# Features: Correlation, Metrics, Heatmap, Excel Export, Grapher, Screener
# Designed with clean sidebar navigation, theming awareness, caching, and user-friendly layout

import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import date, timedelta
import plotly.express as px

# ────────────────────────────────────────────────
#   Import all modules (flat structure)
# ────────────────────────────────────────────────
from additional_metrics import render_additional_metrics
from correlation_finder import render_correlation_finder
from excel_export import render_excel_export
from grapher import render_grapher
from screener import render_screener

# Optional: try to import price_chart if you created it later
# from price_chart import render_price_chart

# ────────────────────────────────────────────────
#   Page config & theming
# ────────────────────────────────────────────────
st.set_page_config(
    page_title="Asset Analyzer Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/asset-analyzer',
        'Report a bug': "https://github.com/yourusername/asset-analyzer/issues",
        'About': "Asset Analyzer • Multi-asset analysis tool powered by yfinance & Streamlit"
    }
)

# Custom CSS for modern look (light/dark mode friendly)
st.markdown("""
    <style>
    /* Main container padding */
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        padding-left: 1.5rem;
        padding-right: 1.5rem;
    }
    
    /* Sidebar improvements */
    section[data-testid="stSidebar"] {
        width: 320px !important;
        background-color: rgba(240, 242, 246, 0.05);
    }
    
    /* Headers & titles */
    h1, h2, h3 {
        margin-bottom: 1rem !important;
    }
    
    /* Card-like containers */
    .stExpander, .stTabs [data-baseweb="tab-panel"] {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.2rem;
    }
    
    /* Buttons */
    .stButton > button {
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        font-weight: 500;
    }
    
    /* Success / info boxes */
    .stSuccess, .stInfo, .stWarning, .stError {
        border-radius: 10px;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────
#   SHARED SIDEBAR – Common Controls
# ────────────────────────────────────────────────
with st.sidebar:
    st.title("Asset Analyzer")
    st.caption("v1.0 • Istanbul, 2026")

    # ── Common inputs (used by most modules) ────────────────────────────────
    st.subheader("Common Settings")

    tickers_input = st.text_input(
        "Tickers (comma-separated)",
        value="AAPL,MSFT",
        placeholder="AAPL,MSFT,GC=F,BTC-USD,EURUSD=X",
        help="Supports stocks, ETFs, commodities, forex, crypto, indices (^GSPC)"
    )
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=date(2020, 1, 1),
            max_value=date.today()
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=date.today(),
            min_value=start_date
        )

    # Cache shared data fetch
    @st.cache_data(ttl=3600, show_spinner="Fetching market data...")
    def fetch_prices(tickers_list, s_date, e_date):
        if not tickers_list:
            return pd.DataFrame()
        try:
            data = yf.download(
                tickers_list,
                start=s_date,
                end=e_date,
                auto_adjust=True,
                progress=False,
                threads=True
            )["Close"]
            if len(tickers_list) == 1:
                data = data.to_frame(name=tickers_list[0])
            return data.dropna(how="all")
        except:
            return pd.DataFrame()

    df_prices = fetch_prices(tickers, start_date, end_date)

    if df_prices.empty and tickers:
        st.warning("No price data loaded. Check tickers or date range.")

    st.markdown("---")

    # Navigation radio
    st.subheader("Analysis Mode")

    pages = {
        "📊 Additional Metrics": render_additional_metrics,
        "🔗 Correlation Finder": render_correlation_finder,
        "📈 Grapher (Fundamentals)": render_grapher,
        "🔍 Stock Screener": render_screener,
        "📥 Excel Export": render_excel_export,
        # Add more when ready:
        # "📉 Price Chart": render_price_chart,
        # "🧪 Backtester": render_backtester,
    }

    selected_mode = st.radio(
        "Choose analysis",
        options=list(pages.keys()),
        index=0,
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.caption("Data from Yahoo Finance • Not financial advice")
    st.caption(f"Last updated: {date.today().strftime('%Y-%m-%d')}")

# ────────────────────────────────────────────────
#   MAIN CONTENT AREA
# ────────────────────────────────────────────────
st.title(f"{selected_mode}")

# Show selected tickers & period at top
if tickers:
    st.caption(f"Active tickers: **{', '.join(tickers)}**  •  Period: **{start_date}** → **{end_date}**  •  {len(df_prices)} trading days")

# Execute the selected module
if selected_mode in pages:
    try:
        if selected_mode in ["📊 Additional Metrics", "🔗 Correlation Finder"]:
            # These use shared df_prices
            if df_prices.empty:
                st.info("Please enter valid tickers and date range in the sidebar.")
            else:
                pages[selected_mode](df_prices, tickers)
        elif selected_mode == "📈 Grapher (Fundamentals)":
            # Grapher uses single ticker → take first one or prompt
            if len(tickers) == 0:
                st.info("Enter at least one ticker in sidebar.")
            else:
                # Grapher has own inputs → just call it
                pages[selected_mode]()
        elif selected_mode in ["🔍 Stock Screener", "📥 Excel Export"]:
            # These have self-contained inputs
            pages[selected_mode]()
        else:
            # Fallback for future modules
            pages[selected_mode](df_prices, tickers)
    except Exception as e:
        st.error(f"Module error: {str(e)}")
        st.info("Try refreshing the page or checking your inputs.")
else:
    st.info("Select an analysis mode from the sidebar.")

# Footer
st.markdown("---")
col_left, col_right = st.columns([3,1])
with col_left:
    st.caption("Built with ❤️ in Istanbul • Powered by Streamlit & yfinance")
with col_right:
    if st.button("Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
