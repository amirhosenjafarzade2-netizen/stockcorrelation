# screener.py - FIXED VERSION (January 2026)

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import time
from typing import List, Dict, Optional
from datetime import datetime, timedelta

# Try importing pandas_ta, fallback to manual RSI calculation
try:
    import pandas_ta as ta
    HAS_PANDAS_TA = True
except ImportError:
    HAS_PANDAS_TA = False
    st.warning("pandas_ta not installed. RSI calculations will use fallback method.")

def calculate_rsi_manual(prices: pd.Series, period: int = 14) -> Optional[float]:
    """Manual RSI calculation if pandas_ta is not available"""
    if len(prices) < period + 1:
        return None
    
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else None

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_finviz_tickers(sector: str) -> List[str]:
    """Fetch all tickers from a Finviz sector with caching"""
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
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Connection': 'keep-alive',
    }
    
    for page_num in range(1, 51):  # Max 50 pages
        start = (page_num - 1) * 20 + 1
        url = f"https://finviz.com/screener.ashx?v=111&f={sector_map[sector]}&r={start}"
        
        try:
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find all ticker links in the screener table
            found_tickers = False
            for link in soup.find_all('a', {'class': 'tab-link'}):
                ticker = link.text.strip()
                if ticker and len(ticker) <= 5:  # Valid ticker length
                    tickers.append(ticker)
                    found_tickers = True
            
            # If no tickers found, we've reached the end
            if not found_tickers:
                break
            
            time.sleep(0.3)  # Be nice to the server
            
        except Exception as e:
            # Silent fail for pagination end
            break
    
    return list(set(tickers))  # Remove duplicates

@st.cache_data(ttl=3600)
def get_sp500_tickers() -> List[str]:
    """Fetch S&P 500 tickers with multiple fallback methods"""
    
    # Method 1: Try Wikipedia with pandas
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        df = tables[0]
        tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()
        if len(tickers) > 400:  # Should have ~500
            return tickers
    except:
        pass
    
    # Method 2: Try direct Wikipedia scraping
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            headers=headers,
            timeout=15
        )
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', {'id': 'constituents'})
        if not table:
            table = soup.find('table', {'class': 'wikitable'})
        
        tickers = []
        for row in table.find_all('tr')[1:]:
            cells = row.find_all('td')
            if cells:
                ticker = cells[0].text.strip().replace('.', '-')
                if ticker:
                    tickers.append(ticker)
        
        if len(tickers) > 400:
            return tickers
    except:
        pass
    
    # Method 3: Use yfinance Ticker for ^GSPC components
    try:
        sp500 = yf.Ticker("^GSPC")
        # This doesn't work reliably, but worth a try
    except:
        pass
    
    # Fallback: Return major S&P 500 stocks with a warning
    st.warning("⚠️ Could not fetch full S&P 500 list. Using top 50 stocks as fallback.")
    return [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK.B", 
        "UNH", "XOM", "JNJ", "JPM", "V", "PG", "MA", "HD", "CVX", "MRK", 
        "ABBV", "PEP", "COST", "AVGO", "KO", "WMT", "MCD", "DIS", "ADBE",
        "CRM", "NFLX", "AMD", "CSCO", "ACN", "TMO", "ORCL", "ABT", "DHR",
        "CMCSA", "VZ", "TXN", "INTC", "NEE", "PM", "HON", "UPS", "IBM",
        "QCOM", "INTU", "LOW", "AMGN", "RTX"
    ]

@st.cache_data(ttl=3600)
def get_nasdaq100_tickers() -> List[str]:
    """Fetch NASDAQ 100 tickers with multiple fallback methods"""
    
    # Method 1: Try Wikipedia with pandas
    try:
        url = "https://en.wikipedia.org/wiki/Nasdaq-100"
        tables = pd.read_html(url)
        for table in tables:
            if 'Ticker' in table.columns:
                tickers = table['Ticker'].tolist()
                tickers = [str(t).strip() for t in tickers if isinstance(t, str) and len(str(t)) <= 5]
                if len(tickers) > 50:
                    return tickers
            elif 'Symbol' in table.columns:
                tickers = table['Symbol'].tolist()
                tickers = [str(t).strip() for t in tickers if isinstance(t, str) and len(str(t)) <= 5]
                if len(tickers) > 50:
                    return tickers
    except:
        pass
    
    # Method 2: Try direct scraping
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=15)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        tickers = []
        for table in soup.find_all('table', {'class': 'wikitable'}):
            for row in table.find_all('tr')[1:]:
                cells = row.find_all('td')
                if len(cells) >= 2:
                    for cell in cells[:2]:
                        ticker = cell.text.strip()
                        if ticker and 1 <= len(ticker) <= 5 and ticker.replace('.', '').isalpha():
                            tickers.append(ticker)
                            break
        
        if len(tickers) > 50:
            return list(set(tickers))
    except:
        pass
    
    # Fallback: Return major NASDAQ 100 stocks
    st.warning("⚠️ Could not fetch full NASDAQ 100 list. Using top 50 stocks as fallback.")
    return [
        "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "NVDA", "META", "TSLA", 
        "AVGO", "COST", "NFLX", "AMD", "PEP", "ADBE", "CSCO", "TMUS", 
        "CMCSA", "INTC", "TXN", "QCOM", "INTU", "AMGN", "HON", "AMAT", 
        "SBUX", "ISRG", "BKNG", "GILD", "ADP", "VRTX", "ADI", "REGN",
        "PANW", "MU", "LRCX", "MDLZ", "MELI", "PYPL", "KLAC", "SNPS",
        "CDNS", "CRWD", "MAR", "MRVL", "ORLY", "CSX", "ADSK", "NXPI",
        "ABNB", "WDAY"
    ]

def fetch_stock_data(symbol: str, include_rsi: bool = True) -> Optional[Dict]:
    """Fetch all data for a single stock with error handling"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        # Validate we got data
        if not info or not info.get("symbol"):
            return None
        
        row = {
            "Symbol": symbol,
            "Name": info.get("longName") or info.get("shortName", symbol),
            "Sector": info.get("sector", "N/A"),
            "Price": info.get("currentPrice") or info.get("regularMarketPrice"),
            "Market Cap (B)": info.get("marketCap", 0) / 1e9 if info.get("marketCap") else None,
            "Trailing P/E": info.get("trailingPE"),
            "Forward P/E": info.get("forwardPE"),
            "P/B": info.get("priceToBook"),
            "P/S": info.get("priceToSalesTrailing12Months"),
            "ROE (%)": info.get("returnOnEquity") * 100 if info.get("returnOnEquity") else None,
            "ROA (%)": info.get("returnOnAssets") * 100 if info.get("returnOnAssets") else None,
            "Debt/Equity": info.get("debtToEquity"),
            "Dividend Yield (%)": info.get("dividendYield") * 100 if info.get("dividendYield") else None,
            "Beta": info.get("beta"),
        }
        
        # Fetch RSI if requested
        if include_rsi:
            try:
                hist = ticker.history(period="3mo", auto_adjust=True)
                if len(hist) >= 30 and 'Close' in hist.columns:
                    if HAS_PANDAS_TA:
                        rsi_series = ta.rsi(hist["Close"], length=14)
                        row["RSI (14)"] = rsi_series.iloc[-1] if not pd.isna(rsi_series.iloc[-1]) else None
                    else:
                        row["RSI (14)"] = calculate_rsi_manual(hist["Close"])
            except:
                row["RSI (14)"] = None
        
        return row
        
    except Exception as e:
        return None

def render_screener() -> None:
    st.subheader("🔍 Ultimate Stock Screener")

    # Universe Selection
    with st.expander("📊 1. Choose Universe", expanded=True):
        option = st.radio("Source", [
            "S&P 500", 
            "NASDAQ 100", 
            "Finviz Sector", 
            "Custom List"
        ], horizontal=True)

        universe = []

        if option == "S&P 500":
            if st.button("Load S&P 500"):
                with st.spinner("Loading S&P 500..."):
                    universe = get_sp500_tickers()
                    if universe:
                        st.session_state['universe'] = universe
                        st.success(f"✅ Loaded {len(universe)} stocks")

        elif option == "NASDAQ 100":
            if st.button("Load NASDAQ 100"):
                with st.spinner("Loading NASDAQ 100..."):
                    universe = get_nasdaq100_tickers()
                    if universe:
                        st.session_state['universe'] = universe
                        st.success(f"✅ Loaded {len(universe)} stocks")

        elif option == "Finviz Sector":
            sector = st.selectbox("Sector", [
                "Technology", "Healthcare", "Financials", "Energy", 
                "Consumer Discretionary", "Consumer Staples", "Industrials", 
                "Basic Materials", "Communication Services", "Utilities", "Real Estate"
            ])
            if st.button("Load Sector"):
                with st.spinner(f"Loading {sector} stocks from Finviz..."):
                    universe = get_finviz_tickers(sector)
                    if universe:
                        st.session_state['universe'] = universe
                        st.success(f"✅ Loaded {len(universe)} stocks from {sector}")
                    else:
                        st.error("Failed to load sector data")

        elif option == "Custom List":
            text = st.text_area("Enter tickers (comma or space separated)", height=100)
            universe = [t.strip().upper() for t in text.replace(",", " ").split() if t.strip()]
            if universe:
                st.session_state['universe'] = universe
                st.info(f"📝 {len(universe)} tickers entered")

        # Use session state universe
        if 'universe' in st.session_state:
            universe = st.session_state['universe']
            max_n = st.slider("Max stocks to screen", 10, min(500, len(universe)), min(100, len(universe)))
            universe = universe[:max_n]

    if not universe:
        st.info("👆 Select a universe and click the load button to begin")
        st.stop()

    # Filters
    with st.form("filters"):
        st.markdown("### ⚙️ 2. Set Filters (leave blank to ignore)")

        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Valuation**")
            mc_min = st.number_input("Market Cap (B) ≥", value=None, format="%.2f")
            mc_max = st.number_input("Market Cap (B) ≤", value=None, format="%.2f")
            pe_min = st.number_input("P/E ≥", value=None, format="%.2f")
            pe_max = st.number_input("P/E ≤", value=None, format="%.2f")
            
        with col2:
            st.markdown("**Profitability**")
            roe_min = st.number_input("ROE (%) ≥", value=None, format="%.2f")
            roe_max = st.number_input("ROE (%) ≤", value=None, format="%.2f")
            dy_min = st.number_input("Div Yield (%) ≥", value=None, format="%.2f")
            dy_max = st.number_input("Div Yield (%) ≤", value=None, format="%.2f")
            
        with col3:
            st.markdown("**Technical**")
            rsi_min = st.number_input("RSI (14) ≥", value=None, min_value=0.0, max_value=100.0, format="%.1f")
            rsi_max = st.number_input("RSI (14) ≤", value=None, min_value=0.0, max_value=100.0, format="%.1f")
            beta_min = st.number_input("Beta ≥", value=None, format="%.2f")
            beta_max = st.number_input("Beta ≤", value=None, format="%.2f")

        st.markdown("---")
        col_sort1, col_sort2 = st.columns(2)
        with col_sort1:
            sort_by = st.selectbox("Sort by", [
                "None", "Market Cap (B)", "Price", "Trailing P/E", "ROE (%)", 
                "Dividend Yield (%)", "RSI (14)", "P/B"
            ])
        with col_sort2:
            ascending = st.checkbox("Ascending", value=False)

        run = st.form_submit_button("🚀 RUN SCREENER", type="primary", use_container_width=True)

    if not run:
        return

    # Screening
    st.markdown("### 📈 Screening Results")
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    include_rsi = rsi_min is not None or rsi_max is not None
    
    for i, symbol in enumerate(universe):
        progress_bar.progress((i + 1) / len(universe))
        status_text.text(f"Screening {symbol}... ({i+1}/{len(universe)})")
        
        data = fetch_stock_data(symbol, include_rsi)
        if data:
            results.append(data)
        
        # Rate limiting
        if (i + 1) % 10 == 0:
            time.sleep(0.5)

    progress_bar.empty()
    status_text.empty()

    if not results:
        st.error("❌ No data retrieved. Check your internet connection or try a smaller universe.")
        return

    df = pd.DataFrame(results)

    # Apply filters
    mask = pd.Series([True] * len(df))
    
    if mc_min is not None:
        mask &= df["Market Cap (B)"] >= mc_min
    if mc_max is not None:
        mask &= df["Market Cap (B)"] <= mc_max
    if pe_min is not None:
        mask &= df["Trailing P/E"] >= pe_min
    if pe_max is not None:
        mask &= df["Trailing P/E"] <= pe_max
    if roe_min is not None:
        mask &= df["ROE (%)"] >= roe_min
    if roe_max is not None:
        mask &= df["ROE (%)"] <= roe_max
    if dy_min is not None:
        mask &= df["Dividend Yield (%)"] >= dy_min
    if dy_max is not None:
        mask &= df["Dividend Yield (%)"] <= dy_max
    if rsi_min is not None and "RSI (14)" in df.columns:
        mask &= df["RSI (14)"] >= rsi_min
    if rsi_max is not None and "RSI (14)" in df.columns:
        mask &= df["RSI (14)"] <= rsi_max
    if beta_min is not None:
        mask &= df["Beta"] >= beta_min
    if beta_max is not None:
        mask &= df["Beta"] <= beta_max

    final = df[mask].copy()

    if final.empty:
        st.warning("⚠️ No stocks match your filters. Try relaxing your criteria.")
        return

    # Sort
    if sort_by != "None" and sort_by in final.columns:
        final = final.sort_values(sort_by, ascending=ascending, na_position='last')

    st.success(f"✅ **{len(final)} stocks** passed your screen out of {len(df)} analyzed!")

    # Display with formatting
    display_df = final.copy()
    for col in display_df.select_dtypes(include=[np.number]).columns:
        if col != "Symbol":
            display_df[col] = display_df[col].round(2)
    
    st.dataframe(display_df, height=600, use_container_width=True)

    # Download
    csv = final.to_csv(index=False)
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv,
        file_name=f"screener_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

# Main entry point
if __name__ == "__main__":
    st.set_page_config(
        page_title="Ultimate Stock Screener",
        page_icon="📊",
        layout="wide"
    )
    render_screener()
