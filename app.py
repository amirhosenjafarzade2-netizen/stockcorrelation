# app.py
# Stock / Asset Analyzer - Enhanced Multi-Module Application
# Version 2.0 - Refactored with better architecture, error handling, and UX

import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import date, timedelta
from typing import List, Tuple, Optional, Dict
import warnings
warnings.filterwarnings('ignore')

# ────────────────────────────────────────────────
#   Configuration
# ────────────────────────────────────────────────
class Config:
    APP_TITLE = "Asset Analyzer Pro"
    APP_VERSION = "2.0"
    DEFAULT_TICKERS = ["AAPL", "MSFT"]
    CACHE_TTL = 3600  # 1 hour
    MAX_TICKERS = 25
    
    TICKER_PRESETS = {
        "Custom": "",
        "Tech Giants": "AAPL,MSFT,GOOGL,AMZN,META",
        "Crypto Top 5": "BTC-USD,ETH-USD,BNB-USD,XRP-USD,SOL-USD",
        "Major Indices": "^GSPC,^DJI,^IXIC,^FTSE,^N225",
        "Commodities": "GC=F,SI=F,CL=F,NG=F",
        "Forex Majors": "EURUSD=X,GBPUSD=X,USDJPY=X,AUDUSD=X"
    }
    
    DATE_PRESETS = {
        "Custom": None,
        "1 Month": 30,
        "3 Months": 90,
        "6 Months": 180,
        "1 Year": 365,
        "2 Years": 730,
        "5 Years": 1825,
    }

# ────────────────────────────────────────────────
#   Analysis Context (standardized data container)
# ────────────────────────────────────────────────
class AnalysisContext:
    """Standardized context object passed to all modules"""
    def __init__(self, df_prices: pd.DataFrame, tickers: List[str], 
                 start_date: date, end_date: date):
        self.df_prices = df_prices
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.metadata = {
            'trading_days': len(df_prices),
            'date_range': f"{start_date} → {end_date}"
        }

# ────────────────────────────────────────────────
#   Data Provider (centralized data fetching)
# ────────────────────────────────────────────────
class DataProvider:
    """Handles all data fetching with caching and error handling"""
    
    @staticmethod
    @st.cache_data(ttl=Config.CACHE_TTL, show_spinner=False)
    def fetch_prices(tickers_list: List[str], s_date: date, e_date: date) -> Tuple[pd.DataFrame, List[str]]:
        """
        Fetch price data from Yahoo Finance
        Returns: (DataFrame, list of failed tickers)
        """
        if not tickers_list:
            return pd.DataFrame(), []
        
        failed_tickers = []
        
        try:
            data = yf.download(
                tickers_list,
                start=s_date,
                end=e_date,
                auto_adjust=True,
                progress=False,
                threads=True
            )
            
            # Handle single ticker case
            if len(tickers_list) == 1:
                if "Close" in data.columns:
                    data = data["Close"].to_frame(name=tickers_list[0])
                else:
                    return pd.DataFrame(), tickers_list
            else:
                data = data["Close"] if "Close" in data.columns else data
            
            # Check which tickers failed
            if isinstance(data, pd.DataFrame):
                for ticker in tickers_list:
                    if ticker not in data.columns or data[ticker].isna().all():
                        failed_tickers.append(ticker)
                        if ticker in data.columns:
                            data = data.drop(columns=[ticker])
            
            return data.dropna(how="all"), failed_tickers
            
        except Exception as e:
            st.error(f"❌ Data fetch error: {str(e)}")
            return pd.DataFrame(), tickers_list
    
    @staticmethod
    def validate_data_quality(df: pd.DataFrame) -> List[str]:
        """Check data quality and return list of issues"""
        issues = []
        
        if df.empty:
            return ["No data available"]
        
        # Check for missing data
        missing_pct = df.isna().sum() / len(df) * 100
        for col in missing_pct[missing_pct > 10].index:
            issues.append(f"⚠️ {col}: {missing_pct[col]:.1f}% missing data")
        
        # Check for very short time series
        if len(df) < 20:
            issues.append(f"⚠️ Only {len(df)} trading days available (may affect analysis)")
        
        return issues

# ────────────────────────────────────────────────
#   Input Validation
# ────────────────────────────────────────────────
def validate_tickers(tickers_list: List[str]) -> List[str]:
    """Validate ticker format and return clean list"""
    valid = []
    
    for t in tickers_list:
        # Check length
        if len(t) > 15:
            st.sidebar.warning(f"⚠️ '{t}' looks too long for a ticker symbol")
            continue
        
        # Check for valid characters (alphanumeric + special chars for forex/futures)
        if not all(c.isalnum() or c in ['-', '=', '^', '.'] for c in t):
            st.sidebar.warning(f"⚠️ '{t}' contains invalid characters")
            continue
        
        valid.append(t)
    
    # Check max limit
    if len(valid) > Config.MAX_TICKERS:
        st.sidebar.error(f"⚠️ Maximum {Config.MAX_TICKERS} tickers allowed. Using first {Config.MAX_TICKERS}.")
        return valid[:Config.MAX_TICKERS]
    
    return valid

# ────────────────────────────────────────────────
#   Module Registry
# ────────────────────────────────────────────────
def get_available_modules() -> Dict[str, dict]:
    """Return available modules with metadata"""
    modules = {}
    
    try:
        from additional_metrics import render_additional_metrics
        modules["📊 Additional Metrics"] = {
            "func": render_additional_metrics,
            "desc": "Calculate returns, volatility, Sharpe ratios, and drawdowns",
            "uses_context": True
        }
    except ImportError:
        pass
    
    try:
        from correlation_finder import render_correlation_finder
        modules["🔗 Correlation Finder"] = {
            "func": render_correlation_finder,
            "desc": "Discover correlations between assets over time",
            "uses_context": True
        }
    except ImportError:
        pass
    
    try:
        from grapher import render_grapher
        modules["📈 Grapher (Fundamentals)"] = {
            "func": render_grapher,
            "desc": "Visualize fundamental metrics and financial data",
            "uses_context": False
        }
    except ImportError:
        pass
    
    try:
        from screener import render_screener
        modules["🔍 Stock Screener"] = {
            "func": render_screener,
            "desc": "Advanced stock screening with technical & fundamental filters",
            "uses_context": False
        }
    except ImportError:
        pass
    
    try:
        from excel_export import render_excel_export
        modules["📤 Excel Export"] = {
            "func": render_excel_export,
            "desc": "Download price data and analytics in formatted Excel",
            "uses_context": False
        }
    except ImportError:
        pass
    
    # ── NEW MODULE ────────────────────────────────────────────────────────
    try:
        from intrinsic_value import render_intrinsic_value
        modules["💎 Intrinsic Value"] = {
            "func": render_intrinsic_value,
            "desc": "Calculate intrinsic value using multiple methods and screen for overvalued stocks",
            "uses_context": False
        }
    except ImportError:
        pass
    
    return modules

# ────────────────────────────────────────────────
#   SIDEBAR - Controls & Inputs
# ────────────────────────────────────────────────
with st.sidebar:
    st.title(f"{Config.APP_TITLE}")
    st.caption(f"v{Config.APP_VERSION}")
    
    st.markdown("### 🎯 Select Assets")
    
    preset = st.selectbox("Quick Presets", options=list(Config.TICKER_PRESETS.keys()))
    
    if preset != "Custom":
        default_value = Config.TICKER_PRESETS[preset]
    else:
        default_value = ", ".join(st.session_state.get('preferences', {}).get('last_tickers', Config.DEFAULT_TICKERS))
    
    tickers_input = st.text_input(
        "Tickers (comma-separated)",
        value=default_value,
        help="e.g. AAPL, MSFT, TSLA or BTC-USD, ETH-USD"
    )
    
    # Parse and validate tickers
    raw_tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    tickers = validate_tickers(raw_tickers)
    
    if tickers:
        st.caption(f"✅ {len(tickers)} valid ticker(s)")
        st.session_state.preferences = st.session_state.get('preferences', {})
        st.session_state.preferences['last_tickers'] = tickers
    
    st.markdown("---")
    
    # ── Date Range ──
    st.subheader("📅 Date Range")
    
    date_preset = st.selectbox(
        "Date Preset",
        options=list(Config.DATE_PRESETS.keys()),
        index=0
    )
    
    # Calculate dates based on preset
    if date_preset == "Custom":
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start",
                value=date(2020, 1, 1),
                max_value=date.today()
            )
        with col2:
            end_date = st.date_input(
                "End",
                value=date.today(),
                min_value=start_date
            )
    else:
        days_back = Config.DATE_PRESETS[date_preset]
        end_date = date.today()
        start_date = end_date - timedelta(days=days_back)
        st.caption(f"📆 {start_date} → {end_date}")
    
    st.markdown("---")
    
    # ── Fetch Data Button ──
    fetch_button = st.button("🔄 Fetch Data", type="primary", use_container_width=True)
    
    # Fetch data (on button or if cache is empty)
    df_prices = None
    failed_tickers = []
    
    cache_key = f"{','.join(tickers)}_{start_date}_{end_date}"
    cache_valid = st.session_state.get('data_cache', {}).get('date_range') == cache_key
    
    if fetch_button or not cache_valid:
        if tickers:
            with st.spinner("📡 Fetching market data..."):
                df_prices, failed_tickers = DataProvider.fetch_prices(tickers, start_date, end_date)
                
                # Update cache
                if 'data_cache' not in st.session_state:
                    st.session_state.data_cache = {}
                st.session_state.data_cache['df_prices'] = df_prices
                st.session_state.data_cache['tickers'] = tickers
                st.session_state.data_cache['date_range'] = cache_key
                
            if not df_prices.empty:
                st.success(f"✅ Loaded {len(df_prices)} trading days")
            else:
                st.error("❌ No data loaded")
        else:
            st.warning("⚠️ Please enter at least one ticker")
    else:
        # Use cached data
        df_prices = st.session_state.data_cache.get('df_prices', pd.DataFrame())
    
    # Show failed tickers
    if failed_tickers:
        with st.expander("⚠️ Failed Tickers", expanded=True):
            st.error(f"Could not fetch: **{', '.join(failed_tickers)}**")
            st.caption("Check ticker symbols or try again later")
    
    # Data quality check
    if df_prices is not None and not df_prices.empty:
        quality_issues = DataProvider.validate_data_quality(df_prices)
        if quality_issues:
            with st.expander("⚠️ Data Quality Issues"):
                for issue in quality_issues:
                    st.warning(issue)
    
    st.markdown("---")
    
    # ── Module Selection ──
    st.subheader("🧭 Analysis Mode")
    
    available_modules = get_available_modules()
    
    if not available_modules:
        st.error("❌ No modules found. Please check module files.")
    else:
        module_names = list(available_modules.keys())
        
        selected_mode = st.radio(
            "Choose analysis",
            options=module_names,
            index=0,
            label_visibility="collapsed"
        )
        
        # Show module description
        if selected_mode in available_modules:
            st.info(available_modules[selected_mode]["desc"])
    
    st.markdown("---")
    
    # ── Footer ──
    st.caption("💾 **Data from Yahoo Finance**")
    st.caption("⚠️ Not financial advice")
    st.caption(f"🕒 Last updated: {date.today().strftime('%Y-%m-%d')}")
    
    # Clear cache button
    if st.button("🗑️ Clear Cache", use_container_width=True):
        st.cache_data.clear()
        st.session_state.data_cache = {'df_prices': None, 'tickers': None, 'date_range': None}
        st.success("Cache cleared!")
        st.rerun()

# ────────────────────────────────────────────────
#   MAIN CONTENT AREA
# ────────────────────────────────────────────────
st.title(selected_mode)

# Show active configuration
if tickers and df_prices is not None and not df_prices.empty:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Active Tickers", len(df_prices.columns), delta=None)
    with col2:
        st.metric("Trading Days", len(df_prices), delta=None)
    with col3:
        st.metric("Date Range", f"{(end_date - start_date).days} days", delta=None)
    
    st.caption(f"**Tickers:** {', '.join(df_prices.columns.tolist())}  •  **Period:** {start_date} → {end_date}")
    st.markdown("---")

# Execute selected module
if available_modules and selected_mode in available_modules:
    module_info = available_modules[selected_mode]
    
    try:
        if df_prices is None or df_prices.empty:
            if module_info["uses_context"]:
                st.info("👈 **Please fetch data from the sidebar to begin analysis**")
                st.markdown("""
                ### Getting Started
                1. Enter ticker symbols (or select a preset)
                2. Choose your date range
                3. Click **Fetch Data**
                4. Results will appear here automatically
                """)
            else:
                # Modules that don't need price data can run anyway
                pass
        
        # Call module with appropriate parameters
        if module_info["uses_context"]:
            # Create context object
            context = AnalysisContext(df_prices, tickers, start_date, end_date)
            module_info["func"](context.df_prices, context.tickers)
        else:
            # Module has its own inputs / doesn't need shared data
            module_info["func"]()
                
    except Exception as e:
        st.error(f"❌ **Module Error:** {str(e)}")
        with st.expander("🔍 Error Details"):
            st.code(str(e))
        st.info("💡 Try refreshing the page or checking your inputs")
else:
    st.warning("⚠️ No modules available. Please check module files.")

# ────────────────────────────────────────────────
#   Footer
# ────────────────────────────────────────────────
st.markdown("---")
col_left, col_right = st.columns([3, 1])
with col_left:
    st.caption(f"Built with ❤️ in Istanbul • Powered by Streamlit & yfinance • {Config.APP_TITLE} v{Config.APP_VERSION}")
with col_right:
    if st.button("↻ Refresh", use_container_width=True):
        st.rerun()
