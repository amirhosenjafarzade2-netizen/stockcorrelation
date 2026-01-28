# forex_professional.py - Professional Forex Analysis with Real Economic Forecasts
# Uses Trading Economics API for forecasts, FinViz for economic calendar, and Yahoo Finance for price data
# Shows ACTUAL vs EXPECTED vs FORECAST economic data for interest rates, inflation, GDP, etc.

import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# Try to import finvizfinance for economic calendar
try:
    from finvizfinance.calendar import Calendar
    FINVIZ_AVAILABLE = True
except ImportError:
    FINVIZ_AVAILABLE = False
    st.warning("⚠️ Install finvizfinance for economic calendar: pip install finvizfinance")

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

# API Keys
TRADING_ECONOMICS_API_KEY = "your_trading_economics_key_here"  # Get from https://tradingeconomics.com/
FRED_API_KEY = "your_fred_api_key_here"  # For historical data

# API URLs
TE_BASE_URL = "https://api.tradingeconomics.com"
FRED_API_URL = "https://api.stlouisfed.org/fred/series/observations"

# Currency Coverage
MAJOR_CURRENCIES = ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD"]
MINOR_CURRENCIES = ["SEK", "NOK", "DKK", "SGD", "HKD", "KRW", "MXN", "ZAR", "TRY", "BRL", "INR", "CNY"]
ALL_CURRENCIES = MAJOR_CURRENCIES + MINOR_CURRENCIES

# Currency to Country Mapping
CURRENCY_COUNTRIES = {
    "USD": "United States", "EUR": "Euro Area", "GBP": "United Kingdom", "JPY": "Japan",
    "AUD": "Australia", "CAD": "Canada", "CHF": "Switzerland", "NZD": "New Zealand",
    "SEK": "Sweden", "NOK": "Norway", "DKK": "Denmark", "SGD": "Singapore",
    "HKD": "Hong Kong", "KRW": "South Korea", "MXN": "Mexico", "ZAR": "South Africa",
    "TRY": "Turkey", "BRL": "Brazil", "INR": "India", "CNY": "China"
}

# Economic Indicators to Track
ECONOMIC_INDICATORS = {
    "Interest Rate": "Interest Rate",
    "Inflation Rate": "Inflation Rate",
    "GDP Growth Rate": "GDP Growth Rate",
    "Unemployment Rate": "Unemployment Rate",
    "Trade Balance": "Balance of Trade",
    "Current Account": "Current Account",
    "Government Debt": "Government Debt to GDP",
    "Consumer Confidence": "Consumer Confidence"
}

# Popular Currency Pairs
POPULAR_PAIRS = {
    "Major Pairs": ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"],
    "Minor Pairs": ["EURGBP", "EURJPY", "GBPJPY", "EURCHF", "EURAUD", "EURCAD", "AUDJPY"],
    "Exotic Pairs": ["USDTRY", "USDZAR", "USDMXN", "USDBRL", "USDINR", "USDSGD"]
}

# ============================================================================
# TRADING ECONOMICS API FUNCTIONS
# ============================================================================

@st.cache_data(ttl=3600)
def fetch_te_historical(country: str, indicator: str, api_key: str) -> pd.DataFrame:
    """
    Fetch historical economic data from Trading Economics.
    """
    if not api_key or api_key == "your_trading_economics_key_here":
        return pd.DataFrame()
    
    try:
        url = f"{TE_BASE_URL}/historical/country/{country}/indicator/{indicator}"
        params = {'c': api_key, 'f': 'json'}
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if data:
            df = pd.DataFrame(data)
            if 'DateTime' in df.columns and 'Value' in df.columns:
                df['DateTime'] = pd.to_datetime(df['DateTime'])
                df = df.set_index('DateTime').sort_index()
                return df[['Value', 'Country', 'Category']]
        
        return pd.DataFrame()
    except Exception as e:
        return pd.DataFrame()

@st.cache_data(ttl=1800)
def fetch_te_forecast(country: str, indicator: str, api_key: str) -> pd.DataFrame:
    """
    Fetch ACTUAL FORECAST data from Trading Economics.
    This shows what economists predict for future economic indicators.
    """
    if not api_key or api_key == "your_trading_economics_key_here":
        return pd.DataFrame()
    
    try:
        url = f"{TE_BASE_URL}/forecast/country/{country}/indicator/{indicator}"
        params = {'c': api_key, 'f': 'json'}
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if data:
            df = pd.DataFrame(data)
            # TE forecasts include: q1, q2, q3, q4 (quarterly forecasts) and years ahead
            return df
        
        return pd.DataFrame()
    except Exception as e:
        return pd.DataFrame()

@st.cache_data(ttl=900)  # Cache for 15 minutes
def fetch_te_calendar(country: str, api_key: str) -> pd.DataFrame:
    """
    Fetch upcoming economic calendar events from Trading Economics.
    Shows: Date, Event, Actual, Forecast, Previous, Importance
    """
    if not api_key or api_key == "your_trading_economics_key_here":
        return pd.DataFrame()
    
    try:
        # Get calendar for specific country
        url = f"{TE_BASE_URL}/calendar/country/{country}"
        params = {'c': api_key, 'f': 'json'}
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if data:
            df = pd.DataFrame(data)
            # Filter for key indicators
            key_events = ['Interest Rate', 'Inflation', 'GDP', 'Employment', 'Trade']
            if 'Event' in df.columns:
                df = df[df['Event'].str.contains('|'.join(key_events), case=False, na=False)]
            return df
        
        return pd.DataFrame()
    except Exception as e:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def fetch_te_indicators_snapshot(country: str, api_key: str) -> pd.DataFrame:
    """
    Get current snapshot of all economic indicators for a country.
    Includes latest value, previous value, and forecasts.
    """
    if not api_key or api_key == "your_trading_economics_key_here":
        return pd.DataFrame()
    
    try:
        url = f"{TE_BASE_URL}/country/{country}"
        params = {'c': api_key, 'f': 'json'}
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if data:
            df = pd.DataFrame(data)
            return df
        
        return pd.DataFrame()
    except Exception as e:
        return pd.DataFrame()

# ============================================================================
# FINVIZ ECONOMIC CALENDAR FUNCTIONS
# ============================================================================

@st.cache_data(ttl=900)
def fetch_finviz_calendar() -> pd.DataFrame:
    """
    Fetch economic calendar from FinViz (free, no API key needed).
    Shows upcoming economic releases with Expected, Actual, and Previous values.
    """
    if not FINVIZ_AVAILABLE:
        return pd.DataFrame()
    
    try:
        calendar = Calendar()
        df = calendar.calendar()
        
        if df is not None and not df.empty:
            # FinViz provides: Datetime, Release, Impact, For, Actual, Expected, Prior
            return df
        
        return pd.DataFrame()
    except Exception as e:
        return pd.DataFrame()

# ============================================================================
# YAHOO FINANCE DATA FUNCTIONS
# ============================================================================

@st.cache_data(ttl=1800)
def fetch_fx_data(pair: str, start_date, end_date, interval='1d') -> pd.DataFrame:
    """Fetch FX rate data from Yahoo Finance."""
    try:
        ticker = f"{pair}=X"
        data = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False)
        
        if data.empty:
            return pd.DataFrame()
        
        df = pd.DataFrame()
        df['Rate'] = data['Close'] if 'Close' in data.columns else data['Adj Close']
        
        if 'Open' in data.columns:
            df['Open'] = data['Open']
        if 'High' in data.columns:
            df['High'] = data['High']
        if 'Low' in data.columns:
            df['Low'] = data['Low']
        if 'Volume' in data.columns:
            df['Volume'] = data['Volume']
            
        return df.dropna()
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def fetch_fred_data(series_id: str, start_date: str, end_date: str, api_key: str) -> pd.DataFrame:
    """Fetch historical economic data from FRED (backup source)."""
    if not api_key or api_key == "your_fred_api_key_here":
        return pd.DataFrame()

    params = {
        'series_id': series_id,
        'api_key': api_key,
        'file_type': 'json',
        'observation_start': start_date,
        'observation_end': end_date
    }

    try:
        response = requests.get(FRED_API_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if 'observations' in data and data['observations']:
            df = pd.DataFrame(data['observations'])
            df['date'] = pd.to_datetime(df['date'])
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            df = df.dropna(subset=['value']).sort_values('date').set_index('date')
            return df[['value']]
        
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

# ============================================================================
# TECHNICAL ANALYSIS FUNCTIONS
# ============================================================================

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators for FX data."""
    if df.empty or 'Rate' not in df.columns:
        return df
    
    result = df.copy()
    
    # Moving Averages
    result['SMA_20'] = result['Rate'].rolling(window=20).mean()
    result['SMA_50'] = result['Rate'].rolling(window=50).mean()
    result['SMA_200'] = result['Rate'].rolling(window=200).mean()
    
    # RSI
    delta = result['Rate'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    result['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    result['BB_Middle'] = result['Rate'].rolling(window=20).mean()
    bb_std = result['Rate'].rolling(window=20).std()
    result['BB_Upper'] = result['BB_Middle'] + (bb_std * 2)
    result['BB_Lower'] = result['BB_Middle'] - (bb_std * 2)
    
    # Volatility
    result['Returns'] = result['Rate'].pct_change()
    result['Volatility'] = result['Returns'].rolling(window=20).std() * np.sqrt(252) * 100
    
    return result

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_price_chart(df: pd.DataFrame, pair: str) -> go.Figure:
    """Create price chart with technical indicators."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.05, row_heights=[0.7, 0.3])
    
    # Price
    fig.add_trace(go.Scatter(x=df.index, y=df['Rate'], name='Price', line=dict(color='blue')), row=1, col=1)
    
    # Moving Averages
    if 'SMA_20' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20', 
                                line=dict(dash='dash', color='orange')), row=1, col=1)
    if 'SMA_50' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50', 
                                line=dict(dash='dash', color='red')), row=1, col=1)
    
    # Bollinger Bands
    if all(col in df.columns for col in ['BB_Upper', 'BB_Lower']):
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], name='BB Upper', 
                                line=dict(color='gray', dash='dot')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], name='BB Lower', 
                                line=dict(color='gray', dash='dot'), fill='tonexty'), row=1, col=1)
    
    # RSI
    if 'RSI' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple')), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    fig.update_layout(title=f"{pair} Technical Analysis", height=700, template='plotly_white')
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1)
    
    return fig

def create_forecast_comparison_chart(historical: pd.DataFrame, forecast: pd.DataFrame, 
                                     indicator: str, country: str) -> go.Figure:
    """
    Create chart comparing historical data with actual economist forecasts.
    """
    fig = go.Figure()
    
    # Historical data
    if not historical.empty and 'Value' in historical.columns:
        fig.add_trace(go.Scatter(
            x=historical.index,
            y=historical['Value'],
            name='Historical',
            line=dict(color='blue', width=2)
        ))
    
    # Forecast data (quarterly projections)
    if not forecast.empty:
        # Trading Economics provides quarterly forecasts (q1, q2, q3, q4)
        quarters = ['q1', 'q2', 'q3', 'q4']
        forecast_dates = []
        forecast_values = []
        
        if 'q1' in forecast.columns:
            # Create future dates for quarterly forecasts
            last_date = historical.index[-1] if not historical.empty else datetime.now()
            for i, q in enumerate(quarters):
                if q in forecast.columns:
                    forecast_dates.append(last_date + timedelta(days=90*(i+1)))
                    forecast_values.append(forecast[q].iloc[0])
        
        if forecast_dates and forecast_values:
            fig.add_trace(go.Scatter(
                x=forecast_dates,
                y=forecast_values,
                name='Economist Forecast',
                line=dict(color='red', width=2, dash='dash'),
                mode='lines+markers',
                marker=dict(size=10)
            ))
    
    fig.update_layout(
        title=f"{indicator} - {country}: Historical vs Forecast",
        xaxis_title="Date",
        yaxis_title=indicator,
        height=500,
        template='plotly_white',
        hovermode='x unified'
    )
    
    return fig

def create_calendar_table(calendar_df: pd.DataFrame) -> go.Figure:
    """Create interactive table of upcoming economic events."""
    if calendar_df.empty:
        return go.Figure()
    
    # Select key columns
    display_cols = []
    for col in ['Date', 'Event', 'Actual', 'Forecast', 'Previous', 'Importance', 'Expected', 'Prior']:
        if col in calendar_df.columns:
            display_cols.append(col)
    
    if not display_cols:
        return go.Figure()
    
    df_display = calendar_df[display_cols].head(20)  # Show top 20 events
    
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=list(df_display.columns),
            fill_color='paleturquoise',
            align='left',
            font=dict(size=12, color='black')
        ),
        cells=dict(
            values=[df_display[col] for col in df_display.columns],
            fill_color='lavender',
            align='left',
            font=dict(size=11)
        )
    )])
    
    fig.update_layout(title="Upcoming Economic Releases", height=600)
    
    return fig

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def forex_module(analysis_context=None):
    """Main Forex Analysis Module with Real Economic Forecasts."""
    
    st.set_page_config(page_title="Forex Analysis Pro", page_icon="📈", layout="wide")
    
    # Header
    st.markdown("# 📈 Professional Forex Analysis")
    st.markdown("### Real Economic Forecasts & Technical Analysis")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Keys
        te_api_key = st.text_input(
            "Trading Economics API Key",
            value=TRADING_ECONOMICS_API_KEY,
            type="password",
            help="Get free trial at https://tradingeconomics.com/"
        )
        
        fred_api_key = st.text_input(
            "FRED API Key (Optional)",
            value=FRED_API_KEY,
            type="password",
            help="For additional historical data"
        )
        
        st.markdown("---")
        
        # Currency Pair Selection
        selection_mode = st.radio(
            "Selection Mode",
            ["Quick Select", "Build Your Own"],
            help="Choose how to select currency pairs"
        )
        
        selected_pairs = []
        
        if selection_mode == "Quick Select":
            category = st.selectbox("Category", list(POPULAR_PAIRS.keys()))
            selected_pairs = st.multiselect(
                "Select Pairs",
                POPULAR_PAIRS[category],
                default=POPULAR_PAIRS[category][:2]
            )
        else:
            num_pairs = st.number_input("Number of pairs", 1, 5, 2)
            for i in range(num_pairs):
                col1, col2 = st.columns(2)
                with col1:
                    base = st.selectbox(f"Base {i+1}", ALL_CURRENCIES, key=f"base_{i}")
                with col2:
                    quote = st.selectbox(f"Quote {i+1}", ALL_CURRENCIES, key=f"quote_{i}", index=1)
                if base != quote:
                    selected_pairs.append(f"{base}{quote}")
        
        st.markdown("---")
        
        # Date Range
        period = st.selectbox("Period", ["3 Months", "6 Months", "1 Year", "2 Years"], index=2)
        period_map = {"3 Months": 90, "6 Months": 180, "1 Year": 365, "2 Years": 730}
        
        end_date = datetime.today().date()
        start_date = end_date - timedelta(days=period_map[period])
        
        st.markdown("---")
        
        # Analysis Options
        st.subheader("Analysis Options")
        show_technical = st.checkbox("Technical Analysis", value=True)
        show_economic = st.checkbox("Economic Indicators & Forecasts", value=True)
        show_calendar = st.checkbox("Economic Calendar", value=True)
    
    # Main Content
    if not selected_pairs:
        st.info("👈 Select currency pairs from the sidebar to begin")
        
        # Show info about forecasts
        with st.expander("ℹ️ About Economic Forecasts"):
            st.markdown("""
            This module displays **REAL ECONOMIC FORECASTS** from professional economists and institutions:
            
            **Data Sources:**
            - 📊 **Trading Economics**: Quarterly forecasts for interest rates, inflation, GDP, unemployment
            - 📅 **FinViz Economic Calendar**: Upcoming releases with consensus forecasts
            - 📈 **Yahoo Finance**: Historical FX price data
            - 🏦 **FRED**: Backup historical economic data
            
            **What You'll See:**
            - **Historical Data**: Actual economic indicators over time
            - **Economist Forecasts**: Projected values from consensus estimates
            - **Economic Calendar**: Upcoming releases with expected vs actual values
            - **Interest Rate Decisions**: Central bank forecasts and market expectations
            - **Consensus Estimates**: What the market expects for key indicators
            
            **Not ML Predictions:**
            This is NOT machine learning or algorithmic forecasting - these are actual predictions
            from economists, analysts, and institutions used by professional traders.
            """)
        
        return
    
    # Analysis Button
    analyze_button = st.button("🚀 Run Analysis", type="primary", use_container_width=True)
    
    if not analyze_button:
        st.info("👆 Click 'Run Analysis' to start")
        return
    
    # ========================================================================
    # DATA LOADING
    # ========================================================================
    
    st.markdown("---")
    st.header("Loading Data...")
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Fetch FX Data
    status_text.text("Fetching currency price data...")
    fx_data = {}
    for pair in selected_pairs:
        df = fetch_fx_data(pair, start_date, end_date)
        if not df.empty:
            fx_data[pair] = calculate_technical_indicators(df)
    progress_bar.progress(0.3)
    
    # Fetch Economic Data & Forecasts
    econ_data = {}
    forecast_data = {}
    calendar_data = {}
    
    if show_economic and te_api_key and te_api_key != "your_trading_economics_key_here":
        status_text.text("Fetching economic indicators and forecasts...")
        
        for pair in selected_pairs:
            base_curr, quote_curr = pair[:3], pair[3:]
            base_country = CURRENCY_COUNTRIES.get(base_curr, "")
            quote_country = CURRENCY_COUNTRIES.get(quote_curr, "")
            
            econ_data[pair] = {}
            forecast_data[pair] = {}
            
            for country, curr in [(base_country, base_curr), (quote_country, quote_curr)]:
                if country:
                    # Fetch current indicators snapshot
                    snapshot = fetch_te_indicators_snapshot(country, te_api_key)
                    if not snapshot.empty:
                        econ_data[pair][f"snapshot_{curr}"] = snapshot
                    
                    # Fetch historical and forecast for key indicators
                    for indicator_name, indicator_te in ECONOMIC_INDICATORS.items():
                        # Historical
                        hist = fetch_te_historical(country, indicator_te, te_api_key)
                        if not hist.empty:
                            econ_data[pair][f"{indicator_name}_{curr}_hist"] = hist
                        
                        # FORECAST (this is the key part!)
                        fcst = fetch_te_forecast(country, indicator_te, te_api_key)
                        if not fcst.empty:
                            forecast_data[pair][f"{indicator_name}_{curr}_forecast"] = fcst
        
        progress_bar.progress(0.6)
    
    # Fetch Economic Calendar
    if show_calendar:
        status_text.text("Fetching economic calendar...")
        
        # Try FinViz calendar (free)
        finviz_cal = fetch_finviz_calendar()
        if not finviz_cal.empty:
            calendar_data['finviz'] = finviz_cal
        
        # Try Trading Economics calendar (if API key available)
        if te_api_key and te_api_key != "your_trading_economics_key_here":
            for pair in selected_pairs:
                base_curr, quote_curr = pair[:3], pair[3:]
                base_country = CURRENCY_COUNTRIES.get(base_curr, "")
                quote_country = CURRENCY_COUNTRIES.get(quote_curr, "")
                
                for country, curr in [(base_country, base_curr), (quote_country, quote_curr)]:
                    if country:
                        cal = fetch_te_calendar(country, te_api_key)
                        if not cal.empty:
                            calendar_data[f"{curr}_calendar"] = cal
        
        progress_bar.progress(0.9)
    
    progress_bar.progress(1.0)
    status_text.text("✅ Data loading complete!")
    
    # ========================================================================
    # DISPLAY RESULTS
    # ========================================================================
    
    st.markdown("---")
    
    tabs = ["📊 Price Analysis", "🌍 Economic Indicators & Forecasts", "📅 Economic Calendar", "💾 Export"]
    tab_objects = st.tabs(tabs)
    
    # ========================================================================
    # TAB 1: PRICE ANALYSIS
    # ========================================================================
    
    with tab_objects[0]:
        st.header("📊 FX Price Analysis")
        
        for pair in selected_pairs:
            if pair not in fx_data:
                continue
            
            st.subheader(f"{pair}")
            
            df = fx_data[pair]
            
            # Current metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                current = df['Rate'].iloc[-1]
                st.metric("Current Rate", f"{current:.5f}")
            
            with col2:
                change_20d = ((df['Rate'].iloc[-1] / df['Rate'].iloc[-20]) - 1) * 100 if len(df) >= 20 else 0
                st.metric("20D Change", f"{change_20d:+.2f}%")
            
            with col3:
                vol = df['Volatility'].iloc[-1] if 'Volatility' in df.columns else 0
                st.metric("Volatility", f"{vol:.2f}%")
            
            with col4:
                rsi = df['RSI'].iloc[-1] if 'RSI' in df.columns else 50
                st.metric("RSI", f"{rsi:.1f}")
            
            # Price chart
            if show_technical:
                fig = create_price_chart(df, pair)
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
    
    # ========================================================================
    # TAB 2: ECONOMIC INDICATORS & FORECASTS
    # ========================================================================
    
    with tab_objects[1]:
        st.header("🌍 Economic Indicators & Real Forecasts")
        
        if not show_economic:
            st.info("Enable 'Economic Indicators & Forecasts' in the sidebar")
        elif not te_api_key or te_api_key == "your_trading_economics_key_here":
            st.warning("⚠️ Trading Economics API key required for economic forecasts")
            st.markdown("Get a free trial at: https://tradingeconomics.com/")
        elif not econ_data and not forecast_data:
            st.warning("No economic data available. Check your API key and try again.")
        else:
            for pair in selected_pairs:
                if pair not in econ_data and pair not in forecast_data:
                    continue
                
                st.subheader(f"{pair} Economic Analysis")
                
                base_curr, quote_curr = pair[:3], pair[3:]
                
                # Display economic snapshots
                if pair in econ_data:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"#### {base_curr} Economic Snapshot")
                        snapshot_key = f"snapshot_{base_curr}"
                        if snapshot_key in econ_data[pair]:
                            snapshot = econ_data[pair][snapshot_key]
                            if not snapshot.empty and 'Category' in snapshot.columns:
                                # Display key indicators
                                for idx, row in snapshot.head(10).iterrows():
                                    if 'Category' in row and 'LatestValue' in row:
                                        st.metric(
                                            row['Category'],
                                            f"{row['LatestValue']}" + (f" {row.get('Unit', '')}" if 'Unit' in row else "")
                                        )
                    
                    with col2:
                        st.markdown(f"#### {quote_curr} Economic Snapshot")
                        snapshot_key = f"snapshot_{quote_curr}"
                        if snapshot_key in econ_data[pair]:
                            snapshot = econ_data[pair][snapshot_key]
                            if not snapshot.empty and 'Category' in snapshot.columns:
                                for idx, row in snapshot.head(10).iterrows():
                                    if 'Category' in row and 'LatestValue' in row:
                                        st.metric(
                                            row['Category'],
                                            f"{row['LatestValue']}" + (f" {row.get('Unit', '')}" if 'Unit' in row else "")
                                        )
                
                st.markdown("---")
                
                # Display forecasts for key indicators
                if pair in forecast_data:
                    st.markdown(f"#### Economist Forecasts for {pair}")
                    
                    for indicator_name in ECONOMIC_INDICATORS.keys():
                        # Check if we have historical and forecast data
                        hist_key_base = f"{indicator_name}_{base_curr}_hist"
                        fcst_key_base = f"{indicator_name}_{base_curr}_forecast"
                        
                        hist_key_quote = f"{indicator_name}_{quote_curr}_hist"
                        fcst_key_quote = f"{indicator_name}_{quote_curr}_forecast"
                        
                        # Base currency
                        if hist_key_base in econ_data.get(pair, {}) or fcst_key_base in forecast_data.get(pair, {}):
                            with st.expander(f"📊 {indicator_name} ({base_curr})"):
                                hist_df = econ_data.get(pair, {}).get(hist_key_base, pd.DataFrame())
                                fcst_df = forecast_data.get(pair, {}).get(fcst_key_base, pd.DataFrame())
                                
                                if not hist_df.empty or not fcst_df.empty:
                                    fig = create_forecast_comparison_chart(
                                        hist_df, fcst_df, indicator_name, base_curr
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Show forecast values
                                    if not fcst_df.empty:
                                        st.markdown("**Quarterly Forecasts:**")
                                        quarters = ['q1', 'q2', 'q3', 'q4']
                                        cols = st.columns(4)
                                        for i, q in enumerate(quarters):
                                            if q in fcst_df.columns:
                                                with cols[i]:
                                                    st.metric(q.upper(), f"{fcst_df[q].iloc[0]:.2f}")
                        
                        # Quote currency
                        if hist_key_quote in econ_data.get(pair, {}) or fcst_key_quote in forecast_data.get(pair, {}):
                            with st.expander(f"📊 {indicator_name} ({quote_curr})"):
                                hist_df = econ_data.get(pair, {}).get(hist_key_quote, pd.DataFrame())
                                fcst_df = forecast_data.get(pair, {}).get(fcst_key_quote, pd.DataFrame())
                                
                                if not hist_df.empty or not fcst_df.empty:
                                    fig = create_forecast_comparison_chart(
                                        hist_df, fcst_df, indicator_name, quote_curr
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    if not fcst_df.empty:
                                        st.markdown("**Quarterly Forecasts:**")
                                        quarters = ['q1', 'q2', 'q3', 'q4']
                                        cols = st.columns(4)
                                        for i, q in enumerate(quarters):
                                            if q in fcst_df.columns:
                                                with cols[i]:
                                                    st.metric(q.upper(), f"{fcst_df[q].iloc[0]:.2f}")
                
                st.markdown("---")
    
    # ========================================================================
    # TAB 3: ECONOMIC CALENDAR
    # ========================================================================
    
    with tab_objects[2]:
        st.header("📅 Economic Calendar - Upcoming Releases")
        
        if not show_calendar:
            st.info("Enable 'Economic Calendar' in the sidebar")
        elif not calendar_data:
            st.warning("No calendar data available")
        else:
            st.markdown("""
            **Economic Calendar** shows upcoming economic data releases with:
            - 📅 **Date & Time** of release
            - 📊 **Expected Value** (consensus forecast)
            - 📈 **Previous Value** (last release)
            - 🎯 **Actual Value** (after release)
            - ⚠️ **Importance** (impact on markets)
            """)
            
            st.markdown("---")
            
            # Display FinViz calendar if available
            if 'finviz' in calendar_data:
                st.subheader("📊 FinViz Economic Calendar (All Countries)")
                finviz_cal = calendar_data['finviz']
                
                # Filter for high impact events
                if 'Impact' in finviz_cal.columns:
                    high_impact = finviz_cal[finviz_cal['Impact'].isin(['high', 'High', '3'])]
                    if not high_impact.empty:
                        st.markdown("**High Impact Events:**")
                        st.dataframe(high_impact, use_container_width=True)
                
                st.markdown("**All Upcoming Events:**")
                st.dataframe(finviz_cal, use_container_width=True)
            
            # Display Trading Economics calendars by currency
            for key, cal_df in calendar_data.items():
                if key != 'finviz' and not cal_df.empty:
                    currency = key.replace('_calendar', '')
                    st.subheader(f"📅 {currency} Economic Calendar")
                    
                    # Create interactive table
                    fig = create_calendar_table(cal_df)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show raw data
                    with st.expander("View Raw Calendar Data"):
                        st.dataframe(cal_df, use_container_width=True)
    
    # ========================================================================
    # TAB 4: EXPORT
    # ========================================================================
    
    with tab_objects[3]:
        st.header("💾 Export Data")
        
        st.markdown("Download analysis data for further processing")
        
        # Export FX Data
        st.subheader("💱 Currency Price Data")
        for pair, df in fx_data.items():
            csv = df.to_csv()
            st.download_button(
                label=f"📥 Download {pair} Price Data",
                data=csv,
                file_name=f"{pair}_prices_{start_date}_{end_date}.csv",
                mime="text/csv",
                key=f"fx_{pair}"
            )
        
        # Export Economic Data
        if econ_data:
            st.subheader("🌍 Economic Indicators")
            for pair, indicators in econ_data.items():
                with st.expander(f"{pair} Economic Data"):
                    for ind_name, df in indicators.items():
                        if isinstance(df, pd.DataFrame) and not df.empty:
                            csv = df.to_csv()
                            safe_name = ind_name.replace(" ", "_").replace("(", "").replace(")", "")
                            st.download_button(
                                label=f"📥 {ind_name}",
                                data=csv,
                                file_name=f"{pair}_{safe_name}.csv",
                                mime="text/csv",
                                key=f"econ_{pair}_{safe_name}"
                            )
        
        # Export Forecasts
        if forecast_data:
            st.subheader("🔮 Economic Forecasts")
            for pair, forecasts in forecast_data.items():
                with st.expander(f"{pair} Forecasts"):
                    for fcst_name, df in forecasts.items():
                        if isinstance(df, pd.DataFrame) and not df.empty:
                            csv = df.to_csv()
                            safe_name = fcst_name.replace(" ", "_").replace("(", "").replace(")", "")
                            st.download_button(
                                label=f"📥 {fcst_name}",
                                data=csv,
                                file_name=f"{pair}_{safe_name}_forecast.csv",
                                mime="text/csv",
                                key=f"fcst_{pair}_{safe_name}"
                            )
        
        # Export Calendar Data
        if calendar_data:
            st.subheader("📅 Economic Calendar")
            for cal_name, df in calendar_data.items():
                if isinstance(df, pd.DataFrame) and not df.empty:
                    csv = df.to_csv()
                    st.download_button(
                        label=f"📥 {cal_name} Calendar",
                        data=csv,
                        file_name=f"{cal_name}_calendar.csv",
                        mime="text/csv",
                        key=f"cal_{cal_name}"
                    )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>📈 Professional Forex Analysis | Data: Trading Economics, FinViz, Yahoo Finance</p>
        <p style='font-size: 0.8em;'>Forecasts are consensus estimates from professional economists.</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    forex_module()
