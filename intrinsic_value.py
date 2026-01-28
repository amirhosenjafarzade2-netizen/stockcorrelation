# intrinsic_value.py - Stock Intrinsic Value Calculator and Screener
# Version 2.0 - Enhanced with better UX, visualizations, and error handling
# Calculates intrinsic value using multiple methods: Graham, Lynch (PEG-based), DCF, PE, Forward PE
# Screens for overvalued/undervalued stocks with filtering options
# Note: Intrinsic value calculations are estimates and involve assumptions. Not financial advice.

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import time
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Constants ──────────────────────────────────────────────────────────────
DEFAULT_DISCOUNT_RATE = 0.10  # 10% discount rate for DCF
DEFAULT_GROWTH_RATE = 0.05    # 5% perpetual growth for DCF
DEFAULT_TERMINAL_MULTIPLE = 15.0  # Terminal P/E for DCF
DEFAULT_RISK_FREE_RATE = 0.04  # 4% risk-free rate for Graham
DEFAULT_EPS_GROWTH_YEARS = 5.0   # For Lynch PEG

# ── Helper Functions ───────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def get_finviz_data(ticker: str) -> Dict:
    """Scrape additional data from Finviz with better error handling"""
    data = {}
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    url = f"https://finviz.com/quote.ashx?t={ticker}"
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find the table with fundamentals
        table = soup.find('table', {'class': 'snapshot-table2'})
        if table:
            rows = table.find_all('tr')
            for row in rows:
                cells = row.find_all('td')
                for i in range(0, len(cells), 2):
                    if i + 1 < len(cells):
                        key = cells[i].text.strip()
                        value = cells[i+1].text.strip()
                        data[key] = value
        
        return data
    except Exception as e:
        # Silent fail for finviz data - we have yfinance as backup
        return {}

@st.cache_data(ttl=3600)
def fetch_stock_data(ticker: str) -> Optional[Dict]:
    """Fetch stock data from yfinance and Finviz with improved error handling"""
    try:
        yf_ticker = yf.Ticker(ticker)
        info = yf_ticker.info
        
        if not info or 'currentPrice' not in info:
            return None
        
        finviz_data = get_finviz_data(ticker)
        
        data = {
            "Symbol": ticker,
            "Price": info.get("currentPrice", np.nan),
            "Market Cap (B)": info.get("marketCap", np.nan) / 1e9 if info.get("marketCap") else np.nan,
            "Trailing P/E": info.get("trailingPE", np.nan),
            "Forward P/E": info.get("forwardPE", np.nan),
            "PEG Ratio": info.get("pegRatio", np.nan),
            "Book Value": info.get("bookValue", np.nan),
            "EPS (TTM)": info.get("trailingEps", np.nan),
            "Forward EPS": info.get("forwardEps", np.nan),
            "Beta": info.get("beta", np.nan),
            "Dividend Yield (%)": info.get("dividendYield", 0) * 100 if info.get("dividendYield") else 0,
            "ROE (%)": float(finviz_data.get("ROE", "0%").rstrip('%')) if finviz_data.get("ROE") else np.nan,
            "Revenue Growth (%)": float(finviz_data.get("Sales Y/Y", "0%").rstrip('%')) if finviz_data.get("Sales Y/Y") else np.nan,
            "Earnings Growth (%)": float(finviz_data.get("EPS Y/Y", "0%").rstrip('%')) if finviz_data.get("EPS Y/Y") else np.nan,
            "Free Cash Flow (B)": info.get("freeCashflow", np.nan) / 1e9 if info.get("freeCashflow") else np.nan,
            "Debt to Equity": info.get("debtToEquity", np.nan) / 100 if info.get("debtToEquity") else np.nan,
            "Current Ratio": info.get("currentRatio", np.nan),
            "Sector": info.get("sector", "N/A"),
            "Industry": info.get("industry", "N/A"),
        }
        
        # Fetch historical EPS growth if available
        try:
            income = yf_ticker.get_income_stmt()
            if not income.empty:
                if "BasicEPS" in income.index:
                    eps_series = income.loc["BasicEPS"]
                elif "DilutedEPS" in income.index:
                    eps_series = income.loc["DilutedEPS"]
                else:
                    eps_series = None
                    
                if eps_series is not None and len(eps_series) > 1:
                    eps_growth = eps_series.pct_change().mean() * 100
                    data["Hist EPS Growth (%)"] = eps_growth
                else:
                    data["Hist EPS Growth (%)"] = data["Earnings Growth (%)"]
        except:
            data["Hist EPS Growth (%)"] = data["Earnings Growth (%)"]
        
        return data
    except Exception as e:
        return None

def calculate_graham_intrinsic(data: Dict, risk_free_rate: float = DEFAULT_RISK_FREE_RATE) -> Tuple[float, str]:
    """Graham Formula: sqrt(22.5 * EPS * BVPS) adjusted for risk-free rate"""
    eps = data.get("EPS (TTM)", 0)
    bvps = data.get("Book Value", 0)
    
    if eps <= 0 or bvps <= 0:
        return np.nan, "Missing or invalid EPS/Book Value"
    
    adjustment = (4.4 / risk_free_rate) if risk_free_rate > 0 else 1
    value = np.sqrt(22.5 * eps * bvps) * adjustment
    return value, "Success"

def calculate_lynch_intrinsic(data: Dict, growth_years: int = DEFAULT_EPS_GROWTH_YEARS) -> Tuple[float, str]:
    """Lynch PEG-based: Fair PE = Growth Rate, then Price = EPS * Fair PE"""
    eps = data.get("EPS (TTM)", 0)
    growth = data.get("Hist EPS Growth (%)", 0)
    
    if eps <= 0:
        return np.nan, "Missing or invalid EPS"
    if growth <= 0:
        return np.nan, "Missing or negative growth rate"
    
    # Lynch's fair value PE = growth rate
    fair_pe = growth
    value = eps * fair_pe
    return value, "Success"

def calculate_dcf_intrinsic(data: Dict, discount_rate: float = DEFAULT_DISCOUNT_RATE,
                            growth_rate: float = DEFAULT_GROWTH_RATE,
                            terminal_multiple: float = DEFAULT_TERMINAL_MULTIPLE) -> Tuple[float, str]:
    """Simple DCF: Project FCF for 5 years, then terminal value"""
    fcf = data.get("Free Cash Flow (B)", 0) * 1e9  # Convert back to full value
    eps_growth = data.get("Hist EPS Growth (%)", 0) / 100
    
    if fcf <= 0:
        return np.nan, "Missing or negative FCF"
    
    if discount_rate <= growth_rate:
        return np.nan, "Discount rate must be > growth rate"
    
    # Project FCF for 5 years using EPS growth as proxy
    projected_fcf = [fcf * (1 + eps_growth) ** y for y in range(1, 6)]
    
    # Discount projected FCF
    pv_fcf = sum(p / (1 + discount_rate) ** y for y, p in enumerate(projected_fcf, 1))
    
    # Terminal value
    terminal_fcf = projected_fcf[-1] * (1 + growth_rate)
    terminal_value = terminal_fcf / (discount_rate - growth_rate)
    pv_terminal = terminal_value / (1 + discount_rate) ** 5
    
    # Total value / shares (approx using market cap / price for shares outstanding)
    price = data.get("Price", 1)
    mc = data.get("Market Cap (B)", 0) * 1e9
    
    if price <= 0 or mc <= 0:
        return np.nan, "Missing market cap or price data"
    
    shares = mc / price
    total_value = pv_fcf + pv_terminal
    value = total_value / shares
    
    return value, "Success"

def calculate_pe_intrinsic(data: Dict) -> Tuple[float, str]:
    """PE-based: Current Price (assumes current PE is fair value)"""
    price = data.get("Price", 0)
    if price <= 0:
        return np.nan, "Missing price data"
    return price, "Success"

def calculate_forward_pe_intrinsic(data: Dict) -> Tuple[float, str]:
    """Forward PE: Forward EPS * Forward PE"""
    forward_eps = data.get("Forward EPS", 0)
    forward_pe = data.get("Forward P/E", np.nan)
    
    if forward_eps <= 0 or np.isnan(forward_pe):
        return np.nan, "Missing Forward EPS or PE"
    
    value = forward_eps * forward_pe
    return value, "Success"

def calculate_all_intrinsic_values(data: Dict, discount_rate: float, growth_rate: float,
                                   terminal_multiple: float, risk_free_rate: float,
                                   growth_years: int) -> Dict:
    """Calculate all intrinsic values and return detailed breakdown"""
    results = {}
    
    graham_val, graham_msg = calculate_graham_intrinsic(data, risk_free_rate)
    results["Graham"] = {"value": graham_val, "status": graham_msg}
    
    lynch_val, lynch_msg = calculate_lynch_intrinsic(data, int(growth_years))
    results["Lynch"] = {"value": lynch_val, "status": lynch_msg}
    
    dcf_val, dcf_msg = calculate_dcf_intrinsic(data, discount_rate, growth_rate, terminal_multiple)
    results["DCF"] = {"value": dcf_val, "status": dcf_msg}
    
    pe_val, pe_msg = calculate_pe_intrinsic(data)
    results["PE"] = {"value": pe_val, "status": pe_msg}
    
    fwd_pe_val, fwd_pe_msg = calculate_forward_pe_intrinsic(data)
    results["Forward PE"] = {"value": fwd_pe_val, "status": fwd_pe_msg}
    
    return results

def calculate_average_intrinsic(data: Dict, methods: List[str],
                                discount_rate: float, growth_rate: float,
                                terminal_multiple: float, risk_free_rate: float,
                                growth_years: int) -> float:
    """Average of selected methods"""
    all_results = calculate_all_intrinsic_values(
        data, discount_rate, growth_rate, terminal_multiple, risk_free_rate, growth_years
    )
    
    values = []
    for method in methods:
        if method in all_results and not np.isnan(all_results[method]["value"]):
            values.append(all_results[method]["value"])
    
    return np.mean(values) if values else np.nan

# ── Visualization Functions ────────────────────────────────────────────────

def create_valuation_chart(data: Dict, intrinsic_values: Dict, current_price: float):
    """Create a bar chart comparing different valuation methods"""
    methods = []
    values = []
    colors = []
    
    for method, result in intrinsic_values.items():
        if not np.isnan(result["value"]):
            methods.append(method)
            values.append(result["value"])
            # Color based on over/undervaluation
            if result["value"] > current_price:
                colors.append('green')
            else:
                colors.append('red')
    
    if not methods:
        return None
    
    fig = go.Figure()
    
    # Add intrinsic value bars
    fig.add_trace(go.Bar(
        x=methods,
        y=values,
        name='Intrinsic Value',
        marker_color=colors,
        text=[f'${v:.2f}' for v in values],
        textposition='outside'
    ))
    
    # Add current price line
    fig.add_hline(
        y=current_price,
        line_dash="dash",
        line_color="blue",
        annotation_text=f"Current Price: ${current_price:.2f}",
        annotation_position="right"
    )
    
    fig.update_layout(
        title="Intrinsic Value Comparison",
        xaxis_title="Valuation Method",
        yaxis_title="Price ($)",
        showlegend=False,
        height=400
    )
    
    return fig

def create_screening_scatter(df: pd.DataFrame):
    """Create scatter plot for screening results"""
    fig = px.scatter(
        df,
        x="Market Cap (B)",
        y="Premium %",
        size="Market Cap (B)",
        color="Premium %",
        hover_data=["Symbol", "Price", "Intrinsic", "Trailing P/E"],
        color_continuous_scale=["green", "yellow", "red"],
        title="Valuation Overview: Premium vs Market Cap"
    )
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(height=500)
    
    return fig

# ── Universe Fetchers ──────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def get_finviz_tickers(sector: str) -> List[str]:
    """Fetch all tickers from a Finviz sector"""
    sector_map = {
        "Technology": "sec_technology",
        "Healthcare": "sec_healthcare",
        "Financials": "sec_financial",
        "Energy": "sec_energy",
        "Consumer Discretionary": "sec_consumercyclical",
        "Consumer Staples": "sec_consumernoncyclical",
        "Industrials": "sec_industrials",
        "Basic Materials": "sec_basicmaterials",
        "Communication Services": "sec_communicationservices",
        "Utilities": "sec_utilities",
        "Real Estate": "sec_realestate"
    }
    
    if sector not in sector_map:
        return []
    
    tickers = []
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    for page_num in range(1, 51):
        start = (page_num - 1) * 20 + 1
        url = f"https://finviz.com/screener.ashx?v=111&f={sector_map[sector]}&r={start}"
        
        try:
            response = requests.get(url, headers=headers, timeout=15)
            soup = BeautifulSoup(response.text, 'html.parser')
            found = False
            for link in soup.find_all('a', {'class': 'tab-link'}):
                ticker = link.text.strip()
                if ticker and len(ticker) <= 5:
                    tickers.append(ticker)
                    found = True
            if not found:
                break
            time.sleep(0.3)
        except:
            break
    
    return list(set(tickers))

@st.cache_data(ttl=3600)
def get_sp500_tickers() -> List[str]:
    """Fetch S&P 500 tickers"""
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        df = tables[0]
        return df["Symbol"].str.replace(".", "-", regex=False).tolist()
    except:
        return []

@st.cache_data(ttl=3600)
def get_nasdaq100_tickers() -> List[str]:
    """Fetch NASDAQ 100 tickers"""
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/Nasdaq-100")
        for table in tables:
            if 'Ticker' in table.columns:
                return table['Ticker'].tolist()
        return []
    except:
        return []

# ── Main Render Function ───────────────────────────────────────────────────

def render_intrinsic_value():
    st.title("📊 Intrinsic Value Calculator & Stock Screener")
    st.markdown("""
    Calculate fair value estimates using multiple valuation methods and screen for 
    mis-priced opportunities. Choose between analyzing a single stock or screening multiple stocks.
    """)
    
    # Disclaimer
    with st.expander("⚠️ Important Disclaimer"):
        st.warning("""
        **This tool is for educational purposes only and does not constitute financial advice.**
        
        - Intrinsic value calculations are estimates based on assumptions and historical data
        - Past performance does not guarantee future results
        - All investments carry risk - consult a financial advisor before making investment decisions
        - Data accuracy depends on external sources and may contain errors
        """)

    # ── Mode Selection ─────────────────────────────────────────────────────
    mode = st.radio(
        "Select Mode",
        ["🔍 Single Stock Calculator", "📈 Multi-Stock Screener"],
        index=0,
        horizontal=True
    )

    # ── Common Parameters ──────────────────────────────────────────────────
    with st.expander("⚙️ Valuation Assumptions", expanded=False):
        st.markdown("**DCF Parameters**")
        col1, col2, col3 = st.columns(3)
        with col1:
            discount_rate = st.number_input(
                "Discount Rate (WACC) %",
                value=DEFAULT_DISCOUNT_RATE * 100,
                min_value=0.0,
                max_value=20.0,
                step=0.5,
                help="Weighted Average Cost of Capital - typical range 8-12%"
            ) / 100
        with col2:
            growth_rate = st.number_input(
                "Terminal Growth %",
                value=DEFAULT_GROWTH_RATE * 100,
                min_value=0.0,
                max_value=10.0,
                step=0.5,
                help="Long-term growth rate - typically 2-5% (GDP growth)"
            ) / 100
        with col3:
            terminal_multiple = st.number_input(
                "Terminal Multiple",
                value=DEFAULT_TERMINAL_MULTIPLE,
                min_value=5.0,
                max_value=30.0,
                step=1.0,
                help="Terminal P/E multiple - typical range 12-18"
            )
        
        st.markdown("**Other Parameters**")
        col4, col5 = st.columns(2)
        with col4:
            risk_free_rate = st.number_input(
                "Risk-Free Rate %",
                value=DEFAULT_RISK_FREE_RATE * 100,
                min_value=0.0,
                max_value=10.0,
                step=0.5,
                help="10-year Treasury yield - used in Graham formula"
            ) / 100
        with col5:
            growth_years = st.number_input(
                "Growth Years (Lynch)",
                value=DEFAULT_EPS_GROWTH_YEARS,
                min_value=1.0,
                max_value=10.0,
                step=1.0,
                help="Years to project growth - Lynch uses 5 years typically"
            )

    methods = st.multiselect(
        "Select Valuation Methods",
        ["Graham", "Lynch", "DCF", "PE", "Forward PE"],
        default=["Graham", "Lynch", "DCF"],
        help="Multiple methods provide a range of estimates"
    )
    
    if not methods:
        st.warning("⚠️ Please select at least one valuation method")
        return

    # ── Single Stock Mode ──────────────────────────────────────────────────
    if "Single" in mode:
        col1, col2 = st.columns([3, 1])
        with col1:
            ticker = st.text_input("Enter Stock Ticker Symbol", value="AAPL", max_chars=10).upper().strip()
        with col2:
            st.write("")
            st.write("")
            calculate_btn = st.button("📊 Calculate Value", type="primary", use_container_width=True)
        
        if calculate_btn and ticker:
            with st.spinner(f"Analyzing {ticker}..."):
                data = fetch_stock_data(ticker)
                
                if not data:
                    st.error(f"❌ Could not fetch data for {ticker}. Please verify the ticker symbol.")
                    return
                
                # Calculate all values
                intrinsic_results = calculate_all_intrinsic_values(
                    data, discount_rate, growth_rate,
                    terminal_multiple, risk_free_rate, int(growth_years)
                )
                
                # Calculate average of selected methods
                intrinsic_avg = calculate_average_intrinsic(
                    data, methods, discount_rate, growth_rate,
                    terminal_multiple, risk_free_rate, int(growth_years)
                )
                
                price = data["Price"]
                
                # Display Results
                st.markdown("---")
                st.subheader(f"📈 {ticker} Valuation Analysis")
                
                # Key Metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Current Price", f"${price:.2f}")
                with col2:
                    if not np.isnan(intrinsic_avg):
                        st.metric("Avg Intrinsic Value", f"${intrinsic_avg:.2f}")
                    else:
                        st.metric("Avg Intrinsic Value", "N/A")
                with col3:
                    if not np.isnan(intrinsic_avg) and intrinsic_avg > 0:
                        premium = ((price - intrinsic_avg) / intrinsic_avg * 100)
                        delta_color = "inverse" if premium > 0 else "normal"
                        st.metric("Valuation", f"{premium:+.1f}%", delta=None)
                    else:
                        st.metric("Valuation", "N/A")
                with col4:
                    if not np.isnan(intrinsic_avg):
                        if price < intrinsic_avg * 0.9:
                            st.success("🟢 Undervalued")
                        elif price > intrinsic_avg * 1.1:
                            st.error("🔴 Overvalued")
                        else:
                            st.info("🟡 Fair Value")
                
                # Detailed Breakdown
                st.markdown("### 📊 Method Breakdown")
                method_cols = st.columns(len(methods))
                for idx, method in enumerate(methods):
                    with method_cols[idx]:
                        if method in intrinsic_results:
                            result = intrinsic_results[method]
                            if not np.isnan(result["value"]):
                                st.metric(
                                    method,
                                    f"${result['value']:.2f}",
                                    delta=f"{((price - result['value']) / result['value'] * 100):+.1f}%"
                                )
                            else:
                                st.metric(method, "N/A")
                                st.caption(f"⚠️ {result['status']}")
                
                # Visualization
                if not np.isnan(intrinsic_avg):
                    fig = create_valuation_chart(data, intrinsic_results, price)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                
                # Company Information
                st.markdown("### 🏢 Company Information")
                info_col1, info_col2, info_col3 = st.columns(3)
                with info_col1:
                    st.write(f"**Sector:** {data.get('Sector', 'N/A')}")
                    st.write(f"**Industry:** {data.get('Industry', 'N/A')}")
                    st.write(f"**Market Cap:** ${data.get('Market Cap (B)', 0):.2f}B")
                with info_col2:
                    st.write(f"**P/E Ratio:** {data.get('Trailing P/E', np.nan):.2f}" if not np.isnan(data.get('Trailing P/E', np.nan)) else "**P/E Ratio:** N/A")
                    st.write(f"**PEG Ratio:** {data.get('PEG Ratio', np.nan):.2f}" if not np.isnan(data.get('PEG Ratio', np.nan)) else "**PEG Ratio:** N/A")
                    st.write(f"**Beta:** {data.get('Beta', np.nan):.2f}" if not np.isnan(data.get('Beta', np.nan)) else "**Beta:** N/A")
                with info_col3:
                    st.write(f"**ROE:** {data.get('ROE (%)', np.nan):.1f}%" if not np.isnan(data.get('ROE (%)', np.nan)) else "**ROE:** N/A")
                    st.write(f"**Debt/Equity:** {data.get('Debt to Equity', np.nan):.2f}" if not np.isnan(data.get('Debt to Equity', np.nan)) else "**Debt/Equity:** N/A")
                    st.write(f"**Div Yield:** {data.get('Dividend Yield (%)', 0):.2f}%")
                
                # Raw Data
                with st.expander("📋 View Raw Data"):
                    st.json(data)

    # ── Screener Mode ──────────────────────────────────────────────────────
    else:
        with st.form("screener_form"):
            st.subheader("🔍 Screener Configuration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                universe_type = st.selectbox(
                    "Stock Universe",
                    ["S&P 500", "NASDAQ 100", "Sector", "Custom List"]
                )
                
                universe = []
                if universe_type == "S&P 500":
                    universe = get_sp500_tickers()
                elif universe_type == "NASDAQ 100":
                    universe = get_nasdaq100_tickers()
                elif universe_type == "Sector":
                    sector = st.selectbox(
                        "Select Sector",
                        ["Technology", "Healthcare", "Financials", "Energy", "Consumer Discretionary",
                         "Consumer Staples", "Industrials", "Basic Materials", "Communication Services",
                         "Utilities", "Real Estate"]
                    )
                    universe = get_finviz_tickers(sector)
                elif universe_type == "Custom List":
                    custom_tickers = st.text_area(
                        "Enter Tickers (comma-separated)",
                        value="AAPL,MSFT,GOOGL,AMZN,TSLA",
                        height=100
                    )
                    universe = [t.strip().upper() for t in custom_tickers.split(",") if t.strip()]
                
                max_stocks = st.slider(
                    "Max Stocks to Screen",
                    min_value=10,
                    max_value=500,
                    value=50,
                    step=10,
                    help="More stocks = longer processing time"
                )
            
            with col2:
                screen_type = st.radio(
                    "Screen For",
                    ["Overvalued (Price > Intrinsic)", "Undervalued (Price < Intrinsic)", "All Stocks"],
                    help="Filter results by valuation"
                )
                
                mc_min = st.number_input(
                    "Min Market Cap (B)",
                    value=1.0,
                    min_value=0.0,
                    help="Filter out small-cap stocks"
                )
                
                mc_max = st.number_input(
                    "Max Market Cap (B)",
                    value=None,
                    min_value=0.0,
                    help="Optional: Filter out mega-cap stocks"
                )
                
                sort_by = st.selectbox(
                    "Sort By",
                    ["Premium %", "Market Cap (B)", "Intrinsic Value", "Price"],
                    index=0
                )
                
                ascending = st.checkbox("Ascending Order", value=False)
            
            run = st.form_submit_button("🚀 RUN SCREENER", type="primary", use_container_width=True)

        if not run:
            return

        # Limit universe
        universe = universe[:max_stocks]
        
        if not universe:
            st.error("❌ No stocks found in selected universe")
            return

        st.markdown("---")
        st.markdown(f"### 📈 Screening {len(universe)} Stocks...")
        
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        error_count = 0
        
        for i, symbol in enumerate(universe):
            progress_bar.progress((i + 1) / len(universe))
            status_text.text(f"Analyzing {symbol}... ({i+1}/{len(universe)}) | Errors: {error_count}")
            
            data = fetch_stock_data(symbol)
            if data:
                # Apply market cap filter early
                mc = data["Market Cap (B)"]
                if np.isnan(mc) or (mc_min and mc < mc_min) or (mc_max and mc > mc_max):
                    continue
                
                intrinsic = calculate_average_intrinsic(
                    data, methods, discount_rate, growth_rate,
                    terminal_multiple, risk_free_rate, int(growth_years)
                )
                price = data["Price"]
                
                if not np.isnan(intrinsic) and intrinsic > 0:
                    premium = (price - intrinsic) / intrinsic * 100
                    
                    # Apply screening filter
                    include = False
                    if "Overvalued" in screen_type and premium > 0:
                        include = True
                    elif "Undervalued" in screen_type and premium < 0:
                        include = True
                    elif "All" in screen_type:
                        include = True
                    
                    if include:
                        result = {
                            "Symbol": symbol,
                            "Price": price,
                            "Intrinsic": intrinsic,
                            "Premium %": premium,
                            "Market Cap (B)": mc,
                            "Trailing P/E": data.get("Trailing P/E"),
                            "PEG Ratio": data.get("PEG Ratio"),
                            "ROE (%)": data.get("ROE (%)"),
                            "EPS Growth (%)": data.get("Hist EPS Growth (%)"),
                            "Sector": data.get("Sector", "N/A")
                        }
                        results.append(result)
            else:
                error_count += 1
            
            # Rate limiting
            if (i + 1) % 10 == 0:
                time.sleep(0.5)

        progress_bar.empty()
        status_text.empty()

        if not results:
            st.error(f"❌ No stocks found matching criteria. ({error_count} errors)")
            return

        df = pd.DataFrame(results)
        df = df.sort_values(sort_by, ascending=ascending, na_position='last')

        # Display Summary
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Stocks Found", len(df))
        with col2:
            st.metric("Avg Premium", f"{df['Premium %'].mean():.1f}%")
        with col3:
            overvalued = len(df[df['Premium %'] > 0])
            st.metric("Overvalued", overvalued)
        with col4:
            undervalued = len(df[df['Premium %'] < 0])
            st.metric("Undervalued", undervalued)

        # Visualization
        if len(df) > 0:
            fig = create_screening_scatter(df)
            st.plotly_chart(fig, use_container_width=True)

        # Results Table
        st.markdown("### 📊 Detailed Results")
        st.dataframe(
            df.style.format({
                "Price": "${:.2f}",
                "Intrinsic": "${:.2f}",
                "Premium %": "{:.1f}%",
                "Market Cap (B)": "{:.1f}",
                "Trailing P/E": "{:.2f}",
                "PEG Ratio": "{:.2f}",
                "ROE (%)": "{:.1f}",
                "EPS Growth (%)": "{:.1f}"
            }).background_gradient(subset=['Premium %'], cmap='RdYlGn_r'),
            use_container_width=True,
            height=400
        )

        # Download
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name=f"stock_screener_{screen_type.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True
        )

# For local testing
if __name__ == "__main__":
    st.set_page_config(
        page_title="Intrinsic Value Calculator",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    render_intrinsic_value()
