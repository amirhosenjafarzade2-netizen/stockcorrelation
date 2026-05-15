"""
screener.py — Ultimate Stock Screener (merged)

Combines:
- New screener: multi-universe, rich fundamental filters, RSI, technical tab
- Original screener: valuation model integration (intrinsic value, undervaluation %,
  safe buy price, verdict, score) via valuation_models.py + validate_inputs + PDF export
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import time
import io
from typing import List, Dict, Optional
from datetime import datetime

# Internal module imports (available when running inside the main app)
try:
    from valuation_models import calculate_valuation
    from utils import validate_inputs, generate_pdf_report
    from data_fetch import fetch_stock_data as fetch_valuation_inputs
    HAS_VALUATION = True
except ImportError:
    HAS_VALUATION = False

# Optional: pandas_ta for RSI
try:
    import pandas_ta as ta
    HAS_PANDAS_TA = True
except ImportError:
    HAS_PANDAS_TA = False


# ---------------------------------------------------------------------------
# RSI helpers
# ---------------------------------------------------------------------------

def calculate_rsi_manual(prices: pd.Series, period: int = 14) -> Optional[float]:
    """Fallback RSI calculation when pandas_ta is unavailable."""
    if len(prices) < period + 1:
        return None
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    val = rsi.iloc[-1]
    return None if pd.isna(val) else float(val)


# ---------------------------------------------------------------------------
# Universe loaders
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_sp500_tickers() -> pd.DataFrame:
    """
    Return a DataFrame with columns [Symbol, Security, GICS Sector].
    Tries pandas read_html first, then BeautifulSoup scraping, then a fallback list.
    """
    # Method 1: pandas read_html
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        df = tables[0][["Symbol", "Security", "GICS Sector"]].copy()
        df["Symbol"] = df["Symbol"].str.replace(".", "-", regex=False)
        if len(df) > 400:
            return df
    except Exception:
        pass

    # Method 2: BeautifulSoup
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            headers=headers, timeout=15
        )
        soup = BeautifulSoup(response.text, "html.parser")
        table = soup.find("table", {"id": "constituents"}) or soup.find("table", {"class": "wikitable"})
        rows = []
        for row in table.find_all("tr")[1:]:
            cells = row.find_all("td")
            if len(cells) >= 4:
                rows.append({
                    "Symbol": cells[0].text.strip().replace(".", "-"),
                    "Security": cells[1].text.strip(),
                    "GICS Sector": cells[3].text.strip(),
                })
        df = pd.DataFrame(rows)
        if len(df) > 400:
            return df
    except Exception:
        pass

    # Fallback
    st.warning("⚠️ Could not fetch full S&P 500 list. Using a top-50 fallback.")
    return pd.DataFrame({
        "Symbol": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B",
                   "UNH", "XOM", "JNJ", "JPM", "V", "PG", "MA", "HD", "CVX", "MRK",
                   "ABBV", "PEP", "COST", "AVGO", "KO", "WMT", "MCD", "DIS", "ADBE",
                   "CRM", "NFLX", "AMD", "CSCO", "ACN", "TMO", "ORCL", "ABT", "DHR",
                   "CMCSA", "VZ", "TXN", "INTC", "NEE", "PM", "HON", "UPS", "IBM",
                   "QCOM", "INTU", "LOW", "AMGN", "RTX"],
        "Security": ["Apple", "Microsoft", "Alphabet A", "Amazon", "NVIDIA", "Meta",
                     "Tesla", "Berkshire B", "UnitedHealth", "ExxonMobil", "J&J",
                     "JPMorgan", "Visa", "P&G", "Mastercard", "Home Depot", "Chevron",
                     "Merck", "AbbVie", "PepsiCo", "Costco", "Broadcom", "Coca-Cola",
                     "Walmart", "McDonald's", "Disney", "Adobe", "Salesforce", "Netflix",
                     "AMD", "Cisco", "Accenture", "Thermo Fisher", "Oracle", "Abbott",
                     "Danaher", "Comcast", "Verizon", "TI", "Intel", "NextEra", "Philip Morris",
                     "Honeywell", "UPS", "IBM", "Qualcomm", "Intuit", "Lowe's", "Amgen", "RTX"],
        "GICS Sector": ["Information Technology"] * 10 + ["Health Care"] * 5 +
                       ["Financials"] * 5 + ["Consumer Staples"] * 5 +
                       ["Consumer Discretionary"] * 5 + ["Industrials"] * 5 +
                       ["Communication Services"] * 5 + ["Energy"] * 5 +
                       ["Information Technology"] * 5,
    })


@st.cache_data(ttl=3600)
def get_nasdaq100_tickers() -> List[str]:
    """Return NASDAQ 100 ticker list with fallback."""
    try:
        url = "https://en.wikipedia.org/wiki/Nasdaq-100"
        tables = pd.read_html(url)
        for table in tables:
            for col in ("Ticker", "Symbol"):
                if col in table.columns:
                    tickers = [str(t).strip() for t in table[col] if isinstance(t, str) and 1 <= len(str(t).strip()) <= 5]
                    if len(tickers) > 50:
                        return tickers
    except Exception:
        pass

    st.warning("⚠️ Could not fetch NASDAQ 100 list. Using top-50 fallback.")
    return [
        "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "NVDA", "META", "TSLA",
        "AVGO", "COST", "NFLX", "AMD", "PEP", "ADBE", "CSCO", "TMUS",
        "CMCSA", "INTC", "TXN", "QCOM", "INTU", "AMGN", "HON", "AMAT",
        "SBUX", "ISRG", "BKNG", "GILD", "ADP", "VRTX", "ADI", "REGN",
        "PANW", "MU", "LRCX", "MDLZ", "MELI", "PYPL", "KLAC", "SNPS",
        "CDNS", "CRWD", "MAR", "MRVL", "ORLY", "CSX", "ADSK", "NXPI",
        "ABNB", "WDAY",
    ]


@st.cache_data(ttl=3600)
def get_finviz_tickers(sector: str) -> List[str]:
    """Fetch tickers for a sector from Finviz."""
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
        "Real Estate": "sec_realestate",
    }
    if sector not in sector_map:
        return []

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept-Language": "en-US,en;q=0.5",
    }
    tickers = []
    for page_num in range(1, 51):
        start = (page_num - 1) * 20 + 1
        url = f"https://finviz.com/screener.ashx?v=111&f={sector_map[sector]}&r={start}"
        try:
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            found = False
            for link in soup.find_all("a", {"class": "tab-link"}):
                ticker = link.text.strip()
                if ticker and len(ticker) <= 5:
                    tickers.append(ticker)
                    found = True
            if not found:
                break
            time.sleep(0.3)
        except Exception:
            break

    return list(set(tickers))


# ---------------------------------------------------------------------------
# Per-stock data fetch: market + fundamentals + RSI
# ---------------------------------------------------------------------------

def fetch_market_data(symbol: str, include_rsi: bool = True) -> Optional[Dict]:
    """
    Fetch market + fundamental data for a single stock via yfinance.
    Returns a flat dict of screener columns, or None on failure.
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info

        if not info or not info.get("symbol"):
            return None

        row = {
            "Symbol": symbol,
            "Name": info.get("longName") or info.get("shortName", symbol),
            "Sector": info.get("sector", "N/A"),
            "Price": info.get("currentPrice") or info.get("regularMarketPrice"),

            # Size & Valuation
            "Market Cap (B)": info.get("marketCap", 0) / 1e9 if info.get("marketCap") else None,
            "Enterprise Value (B)": info.get("enterpriseValue", 0) / 1e9 if info.get("enterpriseValue") else None,
            "Trailing P/E": info.get("trailingPE"),
            "Forward P/E": info.get("forwardPE"),
            "PEG Ratio": info.get("pegRatio"),
            "P/B": info.get("priceToBook"),
            "P/S": info.get("priceToSalesTrailing12Months"),
            "EV/EBITDA": info.get("enterpriseToEbitda"),

            # Profitability
            "ROE (%)": info.get("returnOnEquity") * 100 if info.get("returnOnEquity") else None,
            "ROA (%)": info.get("returnOnAssets") * 100 if info.get("returnOnAssets") else None,
            "Profit Margin (%)": info.get("profitMargins") * 100 if info.get("profitMargins") else None,
            "Operating Margin (%)": info.get("operatingMargins") * 100 if info.get("operatingMargins") else None,
            "Gross Margin (%)": info.get("grossMargins") * 100 if info.get("grossMargins") else None,

            # Growth
            "Revenue Growth (%)": info.get("revenueGrowth") * 100 if info.get("revenueGrowth") else None,
            "Earnings Growth (%)": info.get("earningsGrowth") * 100 if info.get("earningsGrowth") else None,
            "EPS (TTM)": info.get("trailingEps"),

            # Financial Health
            "Debt/Equity": info.get("debtToEquity"),
            "Current Ratio": info.get("currentRatio"),
            "Quick Ratio": info.get("quickRatio"),

            # Cash Flow
            "Free Cash Flow (B)": info.get("freeCashflow", 0) / 1e9 if info.get("freeCashflow") else None,
            "Operating Cash Flow (B)": info.get("operatingCashflow", 0) / 1e9 if info.get("operatingCashflow") else None,

            # Dividends
            "Dividend Yield (%)": info.get("dividendYield") * 100 if info.get("dividendYield") else None,
            "Payout Ratio (%)": info.get("payoutRatio") * 100 if info.get("payoutRatio") else None,

            # Risk
            "Beta": info.get("beta"),
            "52W High": info.get("fiftyTwoWeekHigh"),
            "52W Low": info.get("fiftyTwoWeekLow"),
        }

        # RSI
        if include_rsi:
            try:
                hist = ticker.history(period="3mo", auto_adjust=True)
                if len(hist) >= 30 and "Close" in hist.columns:
                    if HAS_PANDAS_TA:
                        rsi_s = ta.rsi(hist["Close"], length=14)
                        val = rsi_s.iloc[-1]
                        row["RSI (14)"] = None if pd.isna(val) else float(val)
                    else:
                        row["RSI (14)"] = calculate_rsi_manual(hist["Close"])
            except Exception:
                row["RSI (14)"] = None

        return row

    except Exception:
        return None


def fetch_valuation_data(symbol: str, model: str, mos: float) -> Optional[Dict]:
    """
    Fetch valuation data (intrinsic value, undervaluation %, verdict, score)
    using the app's existing valuation_models pipeline.
    Returns a dict with keys: Intrinsic Value, Safe Buy Price, Undervaluation %, Verdict, Score.
    """
    if not HAS_VALUATION:
        return None
    try:
        data = fetch_valuation_inputs(symbol)
        data["model"] = model
        data["core_mos"] = mos
        data["dividend_mos"] = mos
        data["dcf_mos"] = mos
        data["ri_mos"] = mos

        if not validate_inputs(data):
            return None

        res = calculate_valuation(data)
        return {
            "Intrinsic Value ($)": round(res.get("intrinsic_value", 0), 2),
            "Safe Buy Price ($)": round(res.get("safe_buy_price", 0), 2),
            "Undervaluation (%)": round(res.get("undervaluation", 0), 2),
            "Verdict": res.get("verdict", "-"),
            "Score (0-100)": round(res.get("score", 0), 1),
            "Lynch Value ($)": round(res.get("lynch_value", 0), 2),
            "DCF Value ($)": round(res.get("dcf_value", 0), 2),
            "Graham Value ($)": round(res.get("graham_value", 0), 2),
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main Streamlit UI
# ---------------------------------------------------------------------------

def display_screener() -> None:
    st.subheader("🔍 Ultimate Stock Screener")
    st.caption("Combines live market data with full valuation-model analysis. Not financial advice.")

    # ── 1. Universe ──────────────────────────────────────────────────────────
    with st.expander("📊 1. Choose Universe", expanded=True):
        option = st.radio(
            "Source",
            ["S&P 500", "NASDAQ 100", "Finviz Sector", "Custom List"],
            horizontal=True,
            key="screener_universe_option",
        )

        universe_symbols: List[str] = []
        sp500_df: Optional[pd.DataFrame] = None  # Only set for S&P 500

        if option == "S&P 500":
            sector_options = []
            if st.button("Load S&P 500", key="load_sp500"):
                with st.spinner("Loading S&P 500…"):
                    sp500_df = get_sp500_tickers()
                    st.session_state["sp500_df"] = sp500_df
                    st.session_state["universe_symbols"] = sp500_df["Symbol"].tolist()
                    st.success(f"✅ Loaded {len(sp500_df)} stocks")

            if "sp500_df" in st.session_state:
                sp500_df = st.session_state["sp500_df"]
                sector_options = sorted(sp500_df["GICS Sector"].unique().tolist())
                selected_sectors = st.multiselect(
                    "Filter by GICS Sector(s)",
                    sector_options,
                    default=sector_options,
                    key="screener_sectors",
                    help="Leave all selected to screen the full index.",
                )
                filtered_df = sp500_df[sp500_df["GICS Sector"].isin(selected_sectors)]
                st.session_state["universe_symbols"] = filtered_df["Symbol"].tolist()

        elif option == "NASDAQ 100":
            if st.button("Load NASDAQ 100", key="load_ndq"):
                with st.spinner("Loading NASDAQ 100…"):
                    tickers = get_nasdaq100_tickers()
                    st.session_state["universe_symbols"] = tickers
                    st.success(f"✅ Loaded {len(tickers)} stocks")

        elif option == "Finviz Sector":
            sector = st.selectbox(
                "Sector",
                ["Technology", "Healthcare", "Financials", "Energy",
                 "Consumer Discretionary", "Consumer Staples", "Industrials",
                 "Basic Materials", "Communication Services", "Utilities", "Real Estate"],
                key="screener_finviz_sector",
            )
            if st.button("Load Sector", key="load_finviz"):
                with st.spinner(f"Loading {sector} from Finviz…"):
                    tickers = get_finviz_tickers(sector)
                    if tickers:
                        st.session_state["universe_symbols"] = tickers
                        st.success(f"✅ Loaded {len(tickers)} stocks")
                    else:
                        st.error("Failed to load sector data from Finviz.")

        elif option == "Custom List":
            text = st.text_area("Enter tickers (comma or space separated)", height=80, key="screener_custom_tickers")
            tickers = [t.strip().upper() for t in text.replace(",", " ").split() if t.strip()]
            if tickers:
                st.session_state["universe_symbols"] = tickers
                st.info(f"📝 {len(tickers)} tickers entered")

        universe_symbols = st.session_state.get("universe_symbols", [])

        if universe_symbols:
            max_n = st.slider(
                "Max stocks to screen",
                10, min(500, len(universe_symbols)),
                min(100, len(universe_symbols)),
                key="screener_max_n",
            )
            universe_symbols = universe_symbols[:max_n]
            st.caption(f"Will screen **{len(universe_symbols)}** stocks.")

    if not universe_symbols:
        st.info("👆 Select a universe and click Load to begin.")
        return

    # ── 2. Valuation Model (from original app) ────────────────────────────
    with st.expander("🏦 2. Valuation Model (optional)", expanded=HAS_VALUATION):
        if not HAS_VALUATION:
            st.warning("Valuation modules not found. Intrinsic value columns will be skipped.")
            run_valuation = False
            val_model = "Core Valuation (Excel)"
            val_mos = 25.0
        else:
            run_valuation = st.checkbox(
                "Run valuation models (slower — fetches extra data per stock)",
                value=False,
                key="screener_run_valuation",
            )
            val_model = st.selectbox(
                "Valuation Model",
                ["Core Valuation (Excel)", "Lynch Method", "Discounted Cash Flow (DCF)",
                 "Dividend Discount Model (DDM)", "Two-Stage DCF", "Residual Income (RI)",
                 "Reverse DCF", "Graham Intrinsic Value"],
                key="screener_val_model",
                help="Select which model computes the intrinsic value column.",
                disabled=not run_valuation,
            )
            val_mos = st.number_input(
                "Margin of Safety (%)",
                min_value=0.0, max_value=100.0, value=25.0,
                key="screener_val_mos",
                disabled=not run_valuation,
                help="Applied to all models uniformly.",
            )
            min_undervaluation = st.number_input(
                "Minimum Undervaluation % (filter after valuation)",
                min_value=0.0, max_value=200.0, value=0.0,
                key="screener_min_underval",
                disabled=not run_valuation,
            )

    # ── 3. Filters ────────────────────────────────────────────────────────
    with st.form("screener_filters"):
        st.markdown("### ⚙️ 3. Set Filters *(leave blank to ignore)*")

        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["📊 Valuation", "💰 Profitability", "📈 Growth", "🏦 Financial Health", "📉 Technical"]
        )

        with tab1:
            st.markdown("**Valuation Metrics**")
            c1, c2 = st.columns(2)
            with c1:
                mc_min = st.number_input("Market Cap (B) ≥", value=None, format="%.2f")
                pe_min = st.number_input("P/E ≥", value=None, format="%.2f")
                peg_min = st.number_input("PEG Ratio ≥", value=None, format="%.2f")
                pb_min = st.number_input("P/B ≥", value=None, format="%.2f")
                ps_min = st.number_input("P/S ≥", value=None, format="%.2f")
                ev_ebitda_min = st.number_input("EV/EBITDA ≥", value=None, format="%.2f")
            with c2:
                mc_max = st.number_input("Market Cap (B) ≤", value=None, format="%.2f")
                pe_max = st.number_input("P/E ≤", value=None, format="%.2f")
                peg_max = st.number_input("PEG Ratio ≤", value=None, format="%.2f")
                pb_max = st.number_input("P/B ≤", value=None, format="%.2f")
                ps_max = st.number_input("P/S ≤", value=None, format="%.2f")
                ev_ebitda_max = st.number_input("EV/EBITDA ≤", value=None, format="%.2f")

        with tab2:
            st.markdown("**Profitability Metrics**")
            c1, c2 = st.columns(2)
            with c1:
                roe_min = st.number_input("ROE (%) ≥", value=None, format="%.2f")
                roa_min = st.number_input("ROA (%) ≥", value=None, format="%.2f")
                profit_margin_min = st.number_input("Profit Margin (%) ≥", value=None, format="%.2f")
                operating_margin_min = st.number_input("Operating Margin (%) ≥", value=None, format="%.2f")
                eps_min = st.number_input("EPS (TTM) ≥", value=None, format="%.2f")
            with c2:
                roe_max = st.number_input("ROE (%) ≤", value=None, format="%.2f")
                roa_max = st.number_input("ROA (%) ≤", value=None, format="%.2f")
                profit_margin_max = st.number_input("Profit Margin (%) ≤", value=None, format="%.2f")
                operating_margin_max = st.number_input("Operating Margin (%) ≤", value=None, format="%.2f")
                eps_max = st.number_input("EPS (TTM) ≤", value=None, format="%.2f")

        with tab3:
            st.markdown("**Growth Metrics**")
            c1, c2 = st.columns(2)
            with c1:
                rev_growth_min = st.number_input("Revenue Growth (%) ≥", value=None, format="%.2f")
                earn_growth_min = st.number_input("Earnings Growth (%) ≥", value=None, format="%.2f")
            with c2:
                rev_growth_max = st.number_input("Revenue Growth (%) ≤", value=None, format="%.2f")
                earn_growth_max = st.number_input("Earnings Growth (%) ≤", value=None, format="%.2f")

        with tab4:
            st.markdown("**Financial Health**")
            c1, c2 = st.columns(2)
            with c1:
                de_min = st.number_input("Debt/Equity ≥", value=None, format="%.2f")
                current_ratio_min = st.number_input("Current Ratio ≥", value=None, format="%.2f")
                quick_ratio_min = st.number_input("Quick Ratio ≥", value=None, format="%.2f")
                fcf_min = st.number_input("Free Cash Flow (B) ≥", value=None, format="%.2f")
                dy_min = st.number_input("Dividend Yield (%) ≥", value=None, format="%.2f")
                payout_min = st.number_input("Payout Ratio (%) ≥", value=None, format="%.2f")
            with c2:
                de_max = st.number_input("Debt/Equity ≤", value=None, format="%.2f")
                current_ratio_max = st.number_input("Current Ratio ≤", value=None, format="%.2f")
                quick_ratio_max = st.number_input("Quick Ratio ≤", value=None, format="%.2f")
                fcf_max = st.number_input("Free Cash Flow (B) ≤", value=None, format="%.2f")
                dy_max = st.number_input("Dividend Yield (%) ≤", value=None, format="%.2f")
                payout_max = st.number_input("Payout Ratio (%) ≤", value=None, format="%.2f")

        with tab5:
            st.markdown("**Technical Indicators & Risk**")
            c1, c2 = st.columns(2)
            with c1:
                rsi_min = st.number_input("RSI (14) ≥", value=None, min_value=0.0, max_value=100.0, format="%.1f")
                beta_min = st.number_input("Beta ≥", value=None, format="%.2f")
            with c2:
                rsi_max = st.number_input("RSI (14) ≤", value=None, min_value=0.0, max_value=100.0, format="%.1f")
                beta_max = st.number_input("Beta ≤", value=None, format="%.2f")

        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            sort_by = st.selectbox(
                "Sort by",
                ["None", "Market Cap (B)", "Price", "Trailing P/E", "PEG Ratio", "ROE (%)",
                 "Profit Margin (%)", "Revenue Growth (%)", "Dividend Yield (%)",
                 "RSI (14)", "P/B", "EV/EBITDA", "Free Cash Flow (B)",
                 "Undervaluation (%)", "Score (0-100)"],
            )
        with c2:
            ascending = st.checkbox("Ascending order", value=False)

        run = st.form_submit_button("🚀 RUN SCREENER", type="primary", use_container_width=True)

    if not run:
        return

    # ── 4. Screening loop ─────────────────────────────────────────────────
    st.markdown("### 📈 Screening in progress…")
    include_rsi = rsi_min is not None or rsi_max is not None

    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    failed = 0

    for i, symbol in enumerate(universe_symbols):
        progress_bar.progress((i + 1) / len(universe_symbols))
        status_text.text(f"Screening {symbol}… ({i+1}/{len(universe_symbols)})")

        row = fetch_market_data(symbol, include_rsi)
        if row is None:
            failed += 1
            continue

        # Valuation model columns (optional, from original app)
        if run_valuation and HAS_VALUATION:
            val = fetch_valuation_data(symbol, val_model, val_mos)
            if val:
                row.update(val)

        results.append(row)

        # Rate limiting
        if (i + 1) % 10 == 0:
            time.sleep(0.5)

    progress_bar.empty()
    status_text.empty()

    if not results:
        st.error("❌ No data retrieved. Check your internet connection or try a smaller universe.")
        return

    df = pd.DataFrame(results)

    # ── 5. Apply filters ─────────────────────────────────────────────────
    mask = pd.Series([True] * len(df))

    def apply(col, lo, hi):
        nonlocal mask
        if col not in df.columns:
            return
        if lo is not None:
            mask &= df[col].fillna(-np.inf) >= lo
        if hi is not None:
            mask &= df[col].fillna(np.inf) <= hi

    apply("Market Cap (B)", mc_min, mc_max)
    apply("Trailing P/E", pe_min, pe_max)
    apply("PEG Ratio", peg_min, peg_max)
    apply("P/B", pb_min, pb_max)
    apply("P/S", ps_min, ps_max)
    apply("EV/EBITDA", ev_ebitda_min, ev_ebitda_max)
    apply("ROE (%)", roe_min, roe_max)
    apply("ROA (%)", roa_min, roa_max)
    apply("Profit Margin (%)", profit_margin_min, profit_margin_max)
    apply("Operating Margin (%)", operating_margin_min, operating_margin_max)
    apply("EPS (TTM)", eps_min, eps_max)
    apply("Revenue Growth (%)", rev_growth_min, rev_growth_max)
    apply("Earnings Growth (%)", earn_growth_min, earn_growth_max)
    apply("Debt/Equity", de_min, de_max)
    apply("Current Ratio", current_ratio_min, current_ratio_max)
    apply("Quick Ratio", quick_ratio_min, quick_ratio_max)
    apply("Free Cash Flow (B)", fcf_min, fcf_max)
    apply("Dividend Yield (%)", dy_min, dy_max)
    apply("Payout Ratio (%)", payout_min, payout_max)
    apply("RSI (14)", rsi_min, rsi_max)
    apply("Beta", beta_min, beta_max)

    # Valuation undervaluation threshold filter (from original app)
    if run_valuation and HAS_VALUATION and "Undervaluation (%)" in df.columns:
        mask &= df["Undervaluation (%)"].fillna(-np.inf) >= min_undervaluation

    final = df[mask].copy()

    if final.empty:
        st.warning("⚠️ No stocks match your filters. Try relaxing your criteria.")
        return

    # ── 6. Sort ───────────────────────────────────────────────────────────
    if sort_by != "None" and sort_by in final.columns:
        final = final.sort_values(sort_by, ascending=ascending, na_position="last")

    # ── 7. Results display ────────────────────────────────────────────────
    st.success(
        f"✅ **{len(final)} stocks** passed your screen out of {len(df)} analyzed."
        + (f" ({failed} tickers skipped due to data errors)" if failed else "")
    )

    # Verdict colour legend (if valuation was run)
    if run_valuation and HAS_VALUATION and "Verdict" in final.columns:
        st.caption("🟢 Strong Buy  |  🟡 Buy  |  🟠 Hold  |  🔴 Sell")

    # Round numeric columns for display
    display_df = final.copy()
    for col in display_df.select_dtypes(include=[np.number]).columns:
        display_df[col] = display_df[col].round(2)

    st.dataframe(display_df, height=600, use_container_width=True)

    # ── 8. Downloads ──────────────────────────────────────────────────────
    st.markdown("#### 📥 Export Results")
    col_a, col_b = st.columns(2)

    with col_a:
        csv = final.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"screener_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col_b:
        # PDF report (from original app) — only if valuation ran and module exists
        if run_valuation and HAS_VALUATION and "Verdict" in final.columns:
            try:
                pdf_cols = ["Symbol", "Intrinsic Value ($)", "Undervaluation (%)", "Verdict", "Beta"]
                pdf_cols = [c for c in pdf_cols if c in final.columns]
                pdf_portfolio = final[pdf_cols].rename(columns={
                    "Symbol": "Ticker",
                    "Intrinsic Value ($)": "Intrinsic Value",
                    "Undervaluation (%)": "Undervaluation %",
                })
                # Use the first row's results as the "primary" result for the PDF header
                first = final.iloc[0]
                pdf_results = {
                    "model": val_model,
                    "current_price": first.get("Price", 0),
                    "intrinsic_value": first.get("Intrinsic Value ($)", 0),
                    "safe_buy_price": first.get("Safe Buy Price ($)", 0),
                    "undervaluation": first.get("Undervaluation (%)", 0),
                    "verdict": first.get("Verdict", "-"),
                    "peg_ratio": first.get("PEG Ratio", 0),
                    "score": first.get("Score (0-100)", 0),
                    "lynch_value": first.get("Lynch Value ($)", 0),
                    "dcf_value": first.get("DCF Value ($)", 0),
                    "graham_value": first.get("Graham Value ($)", 0),
                }
                pdf_bytes = generate_pdf_report(pdf_results, pdf_portfolio)
                st.download_button(
                    label="Download PDF Report",
                    data=pdf_bytes,
                    file_name=f"screener_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            except Exception as e:
                st.caption(f"PDF export unavailable: {e}")
        else:
            st.caption("Enable valuation models above to unlock PDF report export.")


# ---------------------------------------------------------------------------
# Stand-alone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    st.set_page_config(
        page_title="Ultimate Stock Screener",
        page_icon="📊",
        layout="wide",
    )
    display_screener()
