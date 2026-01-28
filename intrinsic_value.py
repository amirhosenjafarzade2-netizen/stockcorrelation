# intrinsic_value.py - Stock Intrinsic Value Calculator and Screener
# Version 1.0 - Integrated with yfinance and Finviz scraping
# Calculates intrinsic value using multiple methods: Graham, Lynch (PEG-based), DCF, PE, Forward PE
# Screens for overvalued stocks (actual price > intrinsic value)
# Note: Intrinsic value calculations are estimates and involve assumptions. Not financial advice.

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import time
from typing import List, Dict, Optional
from datetime import datetime

# ── Constants ──────────────────────────────────────────────────────────────
DEFAULT_DISCOUNT_RATE = 0.10  # 10% discount rate for DCF
DEFAULT_GROWTH_RATE = 0.05    # 5% perpetual growth for DCF
DEFAULT_TERMINAL_MULTIPLE = 15  # Terminal P/E for DCF
DEFAULT_RISK_FREE_RATE = 0.04  # 4% risk-free rate for Graham
DEFAULT_EPS_GROWTH_YEARS = 5   # For Lynch PEG

# ── Helper Functions ───────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def get_finviz_data(ticker: str) -> Dict:
    """Scrape additional data from Finviz"""
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
    except Exception:
        return {}

@st.cache_data(ttl=3600)
def fetch_stock_data(ticker: str) -> Optional[Dict]:
    """Fetch stock data from yfinance and Finviz"""
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
            "ROE (%)": float(finviz_data.get("ROE", "0%").rstrip('%')) if finviz_data.get("ROE") else np.nan,
            "Revenue Growth (%)": float(finviz_data.get("Sales Y/Y", "0%").rstrip('%')) if finviz_data.get("Sales Y/Y") else np.nan,
            "Earnings Growth (%)": float(finviz_data.get("EPS Y/Y", "0%").rstrip('%')) if finviz_data.get("EPS Y/Y") else np.nan,
            "Free Cash Flow (B)": info.get("freeCashflow", np.nan) / 1e9 if info.get("freeCashflow") else np.nan,
        }
        
        # Fetch historical EPS growth if available
        try:
            income = yf_ticker.get_income_stmt()
            if not income.empty:
                eps_series = income.loc["BasicEPS"] if "BasicEPS" in income.index else income.loc["DilutedEPS"]
                eps_growth = eps_series.pct_change().mean() * 100 if len(eps_series) > 1 else data["Earnings Growth (%)"]
                data["Hist EPS Growth (%)"] = eps_growth
        except:
            data["Hist EPS Growth (%)"] = data["Earnings Growth (%)"]
        
        return data
    except Exception:
        return None

def calculate_graham_intrinsic(data: Dict, risk_free_rate: float = DEFAULT_RISK_FREE_RATE) -> float:
    """Graham Formula: sqrt(22.5 * EPS * BVPS) adjusted for risk-free rate"""
    eps = data.get("EPS (TTM)", 0)
    bvps = data.get("Book Value", 0)
    if eps <= 0 or bvps <= 0:
        return np.nan
    adjustment = (4.4 / risk_free_rate) if risk_free_rate > 0 else 1
    return np.sqrt(22.5 * eps * bvps) * adjustment

def calculate_lynch_intrinsic(data: Dict, growth_years: int = DEFAULT_EPS_GROWTH_YEARS) -> float:
    """Lynch PEG-based: EPS * (Growth Rate + Dividend Yield) / PE"""
    eps = data.get("EPS (TTM)", 0)
    growth = data.get("Hist EPS Growth (%)", 0) / 100
    div_yield = float(data.get("Dividend Yield (%)", 0)) / 100 if "Dividend Yield (%)" in data else 0
    pe = data.get("Trailing P/E", np.nan)
    if eps <= 0 or pe <= 0:
        return np.nan
    expected_pe = growth * 100 + div_yield * 100  # Simplified Lynch
    return eps * expected_pe

def calculate_dcf_intrinsic(data: Dict, discount_rate: float = DEFAULT_DISCOUNT_RATE,
                            growth_rate: float = DEFAULT_GROWTH_RATE,
                            terminal_multiple: float = DEFAULT_TERMINAL_MULTIPLE) -> float:
    """Simple DCF: Project FCF for 5 years, then terminal value"""
    fcf = data.get("Free Cash Flow (B)", 0) * 1e9  # Convert back to full value
    eps_growth = data.get("Hist EPS Growth (%)", 0) / 100
    if fcf <= 0:
        return np.nan
    
    # Project FCF for 5 years using EPS growth as proxy
    projected_fcf = [fcf * (1 + eps_growth) ** y for y in range(1, 6)]
    
    # Discount projected FCF
    pv_fcf = sum(p / (1 + discount_rate) ** y for y, p in enumerate(projected_fcf, 1))
    
    # Terminal value
    terminal_fcf = projected_fcf[-1] * (1 + growth_rate)
    terminal_value = terminal_fcf / (discount_rate - growth_rate) if discount_rate > growth_rate else 0
    pv_terminal = terminal_value / (1 + discount_rate) ** 5
    
    # Total value / shares (approx using market cap / price for shares outstanding)
    shares = data.get("Market Cap (B)", 0) * 1e9 / data.get("Price", 1) if data.get("Price", 1) > 0 else np.nan
    if np.isnan(shares):
        return np.nan
    
    total_value = pv_fcf + pv_terminal
    return total_value / shares

def calculate_pe_intrinsic(data: Dict) -> float:
    """PE-based: EPS * Industry Avg PE (using trailing PE as proxy if available)"""
    eps = data.get("EPS (TTM)", 0)
    pe = data.get("Trailing P/E", np.nan)
    if eps <= 0 or np.isnan(pe):
        return np.nan
    return eps * pe  # Simple historical PE multiple

def calculate_forward_pe_intrinsic(data: Dict) -> float:
    """Forward PE: Forward EPS * Forward PE"""
    forward_eps = data.get("Forward EPS", 0)
    forward_pe = data.get("Forward P/E", np.nan)
    if forward_eps <= 0 or np.isnan(forward_pe):
        return np.nan
    return forward_eps * forward_pe

def calculate_average_intrinsic(data: Dict, methods: List[str],
                                discount_rate: float, growth_rate: float,
                                terminal_multiple: float, risk_free_rate: float,
                                growth_years: int) -> float:
    """Average of selected methods"""
    values = []
    if "Graham" in methods:
        values.append(calculate_graham_intrinsic(data, risk_free_rate))
    if "Lynch" in methods:
        values.append(calculate_lynch_intrinsic(data, growth_years))
    if "DCF" in methods:
        values.append(calculate_dcf_intrinsic(data, discount_rate, growth_rate, terminal_multiple))
    if "PE" in methods:
        values.append(calculate_pe_intrinsic(data))
    if "Forward PE" in methods:
        values.append(calculate_forward_pe_intrinsic(data))
    
    valid_values = [v for v in values if not np.isnan(v) and v > 0]
    return np.mean(valid_values) if valid_values else np.nan

# ── Universe Fetchers (from screener.py) ───────────────────────────────────

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
        return []  # Fallback to empty if fails

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
    st.subheader("Intrinsic Value Calculator & Screener")
    st.caption("Calculate intrinsic value using multiple methods and screen for overvalued stocks (Price > Intrinsic)")

    # ── Mode Selection ─────────────────────────────────────────────────────
    mode = st.radio("Mode", ["Single Stock Calculator", "Multi-Stock Screener"], index=1)

    # ── Common Parameters ──────────────────────────────────────────────────
    with st.expander("⚙️ Valuation Assumptions", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            discount_rate = st.number_input("Discount Rate (%)", value=DEFAULT_DISCOUNT_RATE * 100, min_value=0.0, max_value=20.0, step=0.5) / 100
            growth_rate = st.number_input("Perpetual Growth (%)", value=DEFAULT_GROWTH_RATE * 100, min_value=0.0, max_value=10.0, step=0.5) / 100
        with col2:
            terminal_multiple = st.number_input("Terminal Multiple", value=float(DEFAULT_TERMINAL_MULTIPLE), min_value=5.0, max_value=30.0, step=1.0)
            risk_free_rate = st.number_input("Risk-Free Rate (%)", value=DEFAULT_RISK_FREE_RATE * 100, min_value=0.0, max_value=10.0, step=0.5) / 100
        with col3:
            growth_years = int(st.number_input("Growth Years (Lynch)", value=float(DEFAULT_EPS_GROWTH_YEARS), min_value=1.0, max_value=10.0, step=1.0))

    methods = st.multiselect(
        "Valuation Methods",
        ["Graham", "Lynch", "DCF", "PE", "Forward PE"],
        default=["Graham", "DCF"]
    )
    if not methods:
        st.warning("Select at least one method")
        return

    if mode == "Single Stock Calculator":
        ticker = st.text_input("Ticker Symbol", value="AAPL").upper()
        if st.button("Calculate Intrinsic Value", type="primary"):
            with st.spinner(f"Fetching data for {ticker}..."):
                data = fetch_stock_data(ticker)
                if not data:
                    st.error(f"Could not fetch data for {ticker}")
                    return
                
                intrinsic = calculate_average_intrinsic(
                    data, methods, discount_rate, growth_rate,
                    terminal_multiple, risk_free_rate, growth_years
                )
                
                st.success(f"**Intrinsic Value Estimate:** ${intrinsic:.2f}")
                price = data["Price"]
                if not np.isnan(intrinsic):
                    premium = ((price - intrinsic) / intrinsic * 100) if intrinsic > 0 else np.nan
                    st.metric("Current Price", f"${price:.2f}")
                    st.metric("Valuation Premium/Discount", f"{premium:.1f}%", delta=None)
                    if premium > 0:
                        st.warning("Overvalued (Price > Intrinsic)")
                    elif premium < 0:
                        st.success("Undervalued (Price < Intrinsic)")
                    else:
                        st.info("Fairly Valued")
                
                with st.expander("Detailed Data"):
                    st.json(data)

    else:  # Screener Mode
        with st.form("screener_form"):
            st.subheader("Screener Configuration")
            
            universe_type = st.selectbox(
                "Universe",
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
                custom_tickers = st.text_area("Custom Tickers (comma-separated)", value="AAPL,MSFT,GOOGL")
                universe = [t.strip().upper() for t in custom_tickers.split(",") if t.strip()]
            
            max_stocks = st.slider("Max Stocks to Screen", min_value=10, max_value=500, value=100, step=10)
            universe = universe[:max_stocks]
            
            mc_min = st.number_input("Min Market Cap (B)", value=1.0, min_value=0.0)
            mc_max = st.number_input("Max Market Cap (B)", value=None, min_value=0.0)
            
            sort_by = st.selectbox("Sort By", ["Market Cap (B)", "Premium %", "Intrinsic Value"], index=0)
            ascending = st.checkbox("Ascending", value=False)
            
            run = st.form_submit_button("🚀 RUN SCREENER", type="primary")

        if not run:
            return

        st.markdown("### 📈 Screening Results")
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, symbol in enumerate(universe):
            progress_bar.progress((i + 1) / len(universe))
            status_text.text(f"Analyzing {symbol}... ({i+1}/{len(universe)})")
            
            data = fetch_stock_data(symbol)
            if data:
                # Apply market cap filter early
                mc = data["Market Cap (B)"]
                if (mc_min and mc < mc_min) or (mc_max and mc > mc_max):
                    continue
                
                intrinsic = calculate_average_intrinsic(
                    data, methods, discount_rate, growth_rate,
                    terminal_multiple, risk_free_rate, growth_years
                )
                price = data["Price"]
                
                if not np.isnan(intrinsic) and intrinsic > 0:
                    premium = (price - intrinsic) / intrinsic * 100
                    if premium > 0:  # Overvalued: Price > Intrinsic
                        result = {
                            "Symbol": symbol,
                            "Price": price,
                            "Intrinsic": intrinsic,
                            "Premium %": premium,
                            "Market Cap (B)": mc,
                            "Trailing P/E": data.get("Trailing P/E"),
                            "PEG Ratio": data.get("PEG Ratio"),
                            "ROE (%)": data.get("ROE (%)"),
                            "EPS Growth (%)": data.get("Hist EPS Growth (%)")
                        }
                        results.append(result)
            
            if (i + 1) % 10 == 0:
                time.sleep(0.5)

        progress_bar.empty()
        status_text.empty()

        if not results:
            st.error("❌ No overvalued stocks found matching criteria.")
            return

        df = pd.DataFrame(results)
        df = df.sort_values(sort_by, ascending=ascending, na_position='last')

        st.success(f"✅ Found {len(df)} overvalued stocks (Price > Intrinsic)")

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
            }),
            use_container_width=True
        )

        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results",
            data=csv,
            file_name=f"overvalued_stocks_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

# For local testing
if __name__ == "__main__":
    st.set_page_config(page_title="Intrinsic Value Calculator", layout="wide")
    render_intrinsic_value()
