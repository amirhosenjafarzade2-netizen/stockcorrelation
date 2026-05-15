"""
screener.py — Ultimate Stock Screener (v2.0)

Improvements over v1:
- Fixed: render_screener alias so app.py can import it correctly
- Added: live progress with ETA estimate
- Added: per-column color formatting (heatmap-style) on results dataframe
- Added: summary cards (avg P/E, median ROE, etc.) above the results table
- Added: quick-filter chips (Oversold RSI, High Dividend, Profitable, Low Debt)
- Added: watchlist session state — star stocks to track across reruns
- Added: column visibility toggle (show/hide groups of columns)
- Added: sector breakdown bar chart of filtered results
- Improved: rate-limit handling with exponential back-off
- Improved: error reporting per ticker in an expandable log
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import time
import io
import math
from typing import List, Dict, Optional, Tuple
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


# ─────────────────────────────────────────────────────────────────────────────
# RSI helpers
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Universe loaders
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def get_sp500_tickers() -> pd.DataFrame:
    for method in ("read_html", "bs4"):
        try:
            if method == "read_html":
                url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
                tables = pd.read_html(url)
                df = tables[0][["Symbol", "Security", "GICS Sector"]].copy()
                df["Symbol"] = df["Symbol"].str.replace(".", "-", regex=False)
                if len(df) > 400:
                    return df
            else:
                headers = {"User-Agent": "Mozilla/5.0"}
                r = requests.get(
                    "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                    headers=headers, timeout=15,
                )
                soup = BeautifulSoup(r.text, "html.parser")
                table = soup.find("table", {"id": "constituents"}) or soup.find(
                    "table", {"class": "wikitable"}
                )
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
            continue

    st.warning("⚠️ Could not fetch full S&P 500 list. Using top-50 fallback.")
    return pd.DataFrame({
        "Symbol": ["AAPL","MSFT","GOOGL","AMZN","NVDA","META","TSLA","BRK-B",
                   "UNH","XOM","JNJ","JPM","V","PG","MA","HD","CVX","MRK",
                   "ABBV","PEP","COST","AVGO","KO","WMT","MCD","DIS","ADBE",
                   "CRM","NFLX","AMD","CSCO","ACN","TMO","ORCL","ABT","DHR",
                   "CMCSA","VZ","TXN","INTC","NEE","PM","HON","UPS","IBM",
                   "QCOM","INTU","LOW","AMGN","RTX"],
        "Security": ["Apple","Microsoft","Alphabet A","Amazon","NVIDIA","Meta",
                     "Tesla","Berkshire B","UnitedHealth","ExxonMobil","J&J",
                     "JPMorgan","Visa","P&G","Mastercard","Home Depot","Chevron",
                     "Merck","AbbVie","PepsiCo","Costco","Broadcom","Coca-Cola",
                     "Walmart","McDonald's","Disney","Adobe","Salesforce","Netflix",
                     "AMD","Cisco","Accenture","Thermo Fisher","Oracle","Abbott",
                     "Danaher","Comcast","Verizon","TI","Intel","NextEra",
                     "Philip Morris","Honeywell","UPS","IBM","Qualcomm","Intuit",
                     "Lowe's","Amgen","RTX"],
        "GICS Sector": (
            ["Information Technology"] * 10 + ["Health Care"] * 5 +
            ["Financials"] * 5 + ["Consumer Staples"] * 5 +
            ["Consumer Discretionary"] * 5 + ["Industrials"] * 5 +
            ["Communication Services"] * 5 + ["Energy"] * 5 +
            ["Information Technology"] * 5
        ),
    })


@st.cache_data(ttl=3600)
def get_nasdaq100_tickers() -> List[str]:
    try:
        url = "https://en.wikipedia.org/wiki/Nasdaq-100"
        tables = pd.read_html(url)
        for table in tables:
            for col in ("Ticker", "Symbol"):
                if col in table.columns:
                    tickers = [
                        str(t).strip()
                        for t in table[col]
                        if isinstance(t, str) and 1 <= len(str(t).strip()) <= 5
                    ]
                    if len(tickers) > 50:
                        return tickers
    except Exception:
        pass

    st.warning("⚠️ Could not fetch NASDAQ 100 list. Using top-50 fallback.")
    return [
        "AAPL","MSFT","GOOGL","GOOG","AMZN","NVDA","META","TSLA",
        "AVGO","COST","NFLX","AMD","PEP","ADBE","CSCO","TMUS",
        "CMCSA","INTC","TXN","QCOM","INTU","AMGN","HON","AMAT",
        "SBUX","ISRG","BKNG","GILD","ADP","VRTX","ADI","REGN",
        "PANW","MU","LRCX","MDLZ","MELI","PYPL","KLAC","SNPS",
        "CDNS","CRWD","MAR","MRVL","ORLY","CSX","ADSK","NXPI",
        "ABNB","WDAY",
    ]


@st.cache_data(ttl=3600)
def get_finviz_tickers(sector: str) -> List[str]:
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
    tickers, page_num = [], 1
    while page_num <= 50:
        start = (page_num - 1) * 20 + 1
        url = f"https://finviz.com/screener.ashx?v=111&f={sector_map[sector]}&r={start}"
        try:
            r = requests.get(url, headers=headers, timeout=15)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
            found = False
            for link in soup.find_all("a", {"class": "tab-link"}):
                t = link.text.strip()
                if t and len(t) <= 5:
                    tickers.append(t)
                    found = True
            if not found:
                break
            time.sleep(0.3)
            page_num += 1
        except Exception:
            break

    return list(set(tickers))


# ─────────────────────────────────────────────────────────────────────────────
# Per-stock data fetch with exponential back-off
# ─────────────────────────────────────────────────────────────────────────────

def fetch_market_data(
    symbol: str,
    include_rsi: bool = True,
    retries: int = 2,
) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Returns (row_dict, error_message).
    row_dict is None on failure; error_message is None on success.
    """
    for attempt in range(retries + 1):
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            if not info or not info.get("symbol"):
                return None, "No data returned by yfinance"

            price = info.get("currentPrice") or info.get("regularMarketPrice")

            row: Dict = {
                "Symbol": symbol,
                "Name": info.get("longName") or info.get("shortName", symbol),
                "Sector": info.get("sector", "N/A"),
                "Industry": info.get("industry", "N/A"),
                "Price ($)": price,
                "52W Change (%)": _pct(info.get("52WeekChange")),

                # Size & Valuation
                "Market Cap (B)": _b(info.get("marketCap")),
                "Enterprise Value (B)": _b(info.get("enterpriseValue")),
                "Trailing P/E": info.get("trailingPE"),
                "Forward P/E": info.get("forwardPE"),
                "PEG Ratio": info.get("pegRatio"),
                "P/B": info.get("priceToBook"),
                "P/S": info.get("priceToSalesTrailing12Months"),
                "EV/EBITDA": info.get("enterpriseToEbitda"),
                "EV/Revenue": info.get("enterpriseToRevenue"),

                # Profitability
                "ROE (%)": _pct(info.get("returnOnEquity")),
                "ROA (%)": _pct(info.get("returnOnAssets")),
                "ROIC (%)": _pct(info.get("returnOnCapital")),
                "Profit Margin (%)": _pct(info.get("profitMargins")),
                "Operating Margin (%)": _pct(info.get("operatingMargins")),
                "Gross Margin (%)": _pct(info.get("grossMargins")),
                "EBITDA Margin (%)": _ebitda_margin(info),

                # Growth
                "Revenue Growth (%)": _pct(info.get("revenueGrowth")),
                "Earnings Growth (%)": _pct(info.get("earningsGrowth")),
                "EPS (TTM)": info.get("trailingEps"),
                "EPS Growth (%)": _pct(info.get("earningsQuarterlyGrowth")),

                # Financial Health
                "Debt/Equity": info.get("debtToEquity"),
                "Current Ratio": info.get("currentRatio"),
                "Quick Ratio": info.get("quickRatio"),
                "Interest Coverage": _interest_coverage(info),

                # Cash Flow
                "Free Cash Flow (B)": _b(info.get("freeCashflow")),
                "Operating CF (B)": _b(info.get("operatingCashflow")),
                "FCF Yield (%)": _fcf_yield(info),

                # Dividends
                "Dividend Yield (%)": _pct(info.get("dividendYield")),
                "Payout Ratio (%)": _pct(info.get("payoutRatio")),
                "Dividend Growth (5Y%)": _pct(info.get("fiveYearAvgDividendYield")),

                # Risk
                "Beta": info.get("beta"),
                "Short Ratio": info.get("shortRatio"),
                "52W High": info.get("fiftyTwoWeekHigh"),
                "52W Low": info.get("fiftyTwoWeekLow"),
                "Avg Volume (M)": _m(info.get("averageVolume")),
            }

            # % off 52W high — useful for spotting pullbacks
            if row["Price ($)"] and row["52W High"]:
                row["% off 52W High"] = round(
                    (row["Price ($)"] / row["52W High"] - 1) * 100, 2
                )
            else:
                row["% off 52W High"] = None

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
                    else:
                        row["RSI (14)"] = None
                except Exception:
                    row["RSI (14)"] = None

            return row, None

        except Exception as exc:
            if attempt < retries:
                time.sleep(2 ** attempt)  # exponential back-off
            else:
                return None, str(exc)

    return None, "Unknown error"


# ─────────────────────────────────────────────────────────────────────────────
# Small helpers
# ─────────────────────────────────────────────────────────────────────────────

def _b(v) -> Optional[float]:
    """Convert raw value to billions."""
    return round(v / 1e9, 3) if v else None

def _m(v) -> Optional[float]:
    """Convert raw value to millions."""
    return round(v / 1e6, 2) if v else None

def _pct(v) -> Optional[float]:
    """Multiply fraction by 100."""
    return round(v * 100, 2) if v is not None else None

def _ebitda_margin(info: dict) -> Optional[float]:
    ebitda = info.get("ebitda")
    rev = info.get("totalRevenue")
    if ebitda and rev and rev != 0:
        return round(ebitda / rev * 100, 2)
    return None

def _interest_coverage(info: dict) -> Optional[float]:
    ebit = info.get("ebit")
    interest = info.get("interestExpense")
    if ebit and interest and interest != 0:
        return round(ebit / abs(interest), 2)
    return None

def _fcf_yield(info: dict) -> Optional[float]:
    fcf = info.get("freeCashflow")
    mc = info.get("marketCap")
    if fcf and mc and mc != 0:
        return round(fcf / mc * 100, 2)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Valuation data (optional integration)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_valuation_data(symbol: str, model: str, mos: float) -> Optional[Dict]:
    if not HAS_VALUATION:
        return None
    try:
        data = fetch_valuation_inputs(symbol)
        for key in ("core_mos", "dividend_mos", "dcf_mos", "ri_mos"):
            data[key] = mos
        data["model"] = model
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


# ─────────────────────────────────────────────────────────────────────────────
# Quick-filter presets
# ─────────────────────────────────────────────────────────────────────────────

QUICK_FILTERS: Dict[str, Dict] = {
    "🔻 Oversold (RSI < 30)": {"RSI (14)_max": 30},
    "💰 High Dividend (≥4%)": {"Dividend Yield (%)_min": 4.0},
    "📈 Profitable (Margin ≥15%)": {"Profit Margin (%)_min": 15.0},
    "🏋️ Low Debt (D/E ≤ 0.5)": {"Debt/Equity_max": 0.5},
    "🚀 High Growth (Rev ≥20%)": {"Revenue Growth (%)_min": 20.0},
    "💎 Value (P/E ≤ 15)": {"Trailing P/E_max": 15.0},
    "🌊 Pullback (≥20% off 52W High)": {"% off 52W High_max": -20.0},
}

COLUMN_GROUPS = {
    "Core": ["Symbol", "Name", "Sector", "Price ($)", "Market Cap (B)"],
    "Valuation": ["Trailing P/E", "Forward P/E", "PEG Ratio", "P/B", "P/S", "EV/EBITDA", "EV/Revenue"],
    "Profitability": ["ROE (%)", "ROA (%)", "Profit Margin (%)", "Operating Margin (%)", "Gross Margin (%)"],
    "Growth": ["Revenue Growth (%)", "Earnings Growth (%)", "EPS (TTM)", "EPS Growth (%)"],
    "Financial Health": ["Debt/Equity", "Current Ratio", "Quick Ratio", "Interest Coverage"],
    "Cash Flow": ["Free Cash Flow (B)", "Operating CF (B)", "FCF Yield (%)"],
    "Dividends": ["Dividend Yield (%)", "Payout Ratio (%)", "Dividend Growth (5Y%)"],
    "Technical": ["RSI (14)", "Beta", "52W Change (%)", "% off 52W High", "Short Ratio"],
}


# ─────────────────────────────────────────────────────────────────────────────
# Watchlist helpers (session-state based)
# ─────────────────────────────────────────────────────────────────────────────

def _init_watchlist():
    if "screener_watchlist" not in st.session_state:
        st.session_state["screener_watchlist"] = set()


def _toggle_watchlist(symbol: str):
    wl = st.session_state["screener_watchlist"]
    if symbol in wl:
        wl.discard(symbol)
    else:
        wl.add(symbol)


# ─────────────────────────────────────────────────────────────────────────────
# Summary cards
# ─────────────────────────────────────────────────────────────────────────────

def _render_summary_cards(df: pd.DataFrame):
    """Show quick metric overview above the table."""
    cols = st.columns(6)
    metrics = [
        ("Stocks", str(len(df)), None),
        ("Avg P/E", _fmt(df["Trailing P/E"].median()), None),
        ("Median ROE", _fmt(df["ROE (%)"].median(), suffix="%"), None),
        ("Avg Div Yield", _fmt(df["Dividend Yield (%)"].median(), suffix="%"), None),
        ("Avg Rev Growth", _fmt(df["Revenue Growth (%)"].median(), suffix="%"), None),
        ("Avg Beta", _fmt(df["Beta"].median()), None),
    ]
    for col, (label, value, delta) in zip(cols, metrics):
        col.metric(label, value, delta)


def _fmt(v, suffix="", prefix="") -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "N/A"
    return f"{prefix}{v:.2f}{suffix}"


# ─────────────────────────────────────────────────────────────────────────────
# Sector breakdown chart
# ─────────────────────────────────────────────────────────────────────────────

def _render_sector_chart(df: pd.DataFrame):
    if "Sector" not in df.columns:
        return
    counts = df["Sector"].value_counts().reset_index()
    counts.columns = ["Sector", "Count"]
    st.bar_chart(counts.set_index("Sector"), use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main UI
# ─────────────────────────────────────────────────────────────────────────────

def display_screener() -> None:
    _init_watchlist()

    st.subheader("🔍 Ultimate Stock Screener")
    st.caption("Live market data + optional valuation models. Not financial advice.")

    # ── Watchlist pill ────────────────────────────────────────────────────
    wl = st.session_state["screener_watchlist"]
    if wl:
        with st.expander(f"⭐ Watchlist ({len(wl)} stocks)", expanded=False):
            st.write(", ".join(sorted(wl)))
            if st.button("Clear Watchlist", key="clear_wl"):
                st.session_state["screener_watchlist"] = set()
                st.rerun()

    # ── 1. Universe ───────────────────────────────────────────────────────
    with st.expander("📊 1. Choose Universe", expanded=True):
        option = st.radio(
            "Source",
            ["S&P 500", "NASDAQ 100", "Finviz Sector", "Custom List"],
            horizontal=True,
            key="screener_universe_option",
        )

        universe_symbols: List[str] = []
        sp500_df = None

        if option == "S&P 500":
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
                ["Technology","Healthcare","Financials","Energy",
                 "Consumer Discretionary","Consumer Staples","Industrials",
                 "Basic Materials","Communication Services","Utilities","Real Estate"],
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
            text = st.text_area(
                "Enter tickers (comma or space separated)",
                height=80, key="screener_custom_tickers",
                placeholder="AAPL MSFT GOOGL AMZN NVDA",
            )
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

    # ── 2. Quick-filter chips ─────────────────────────────────────────────
    with st.expander("⚡ 2. Quick Filters (optional presets)", expanded=False):
        st.caption("Select a preset to auto-populate filter values below.")
        qf_choice = st.radio(
            "Preset",
            ["None"] + list(QUICK_FILTERS.keys()),
            horizontal=True,
            key="screener_qf_choice",
        )

    # ── 3. Valuation model ────────────────────────────────────────────────
    with st.expander("🏦 3. Valuation Model (optional)", expanded=HAS_VALUATION):
        if not HAS_VALUATION:
            st.warning("Valuation modules not found. Intrinsic value columns will be skipped.")
            run_valuation = False
            val_model = "Core Valuation (Excel)"
            val_mos = 25.0
            min_undervaluation = 0.0
        else:
            run_valuation = st.checkbox(
                "Run valuation models (slower — fetches extra data per stock)",
                value=False,
                key="screener_run_valuation",
            )
            val_model = st.selectbox(
                "Valuation Model",
                ["Core Valuation (Excel)","Lynch Method","Discounted Cash Flow (DCF)",
                 "Dividend Discount Model (DDM)","Two-Stage DCF","Residual Income (RI)",
                 "Reverse DCF","Graham Intrinsic Value"],
                key="screener_val_model",
                disabled=not run_valuation,
            )
            val_mos = st.number_input(
                "Margin of Safety (%)",
                min_value=0.0, max_value=100.0, value=25.0,
                key="screener_val_mos",
                disabled=not run_valuation,
            )
            min_undervaluation = st.number_input(
                "Minimum Undervaluation % (post-valuation filter)",
                min_value=0.0, max_value=200.0, value=0.0,
                key="screener_min_underval",
                disabled=not run_valuation,
            )

    # ── 4. Column group visibility ────────────────────────────────────────
    with st.expander("👁️ 4. Column Visibility", expanded=False):
        st.caption("Choose which groups of columns to show in results.")
        visible_groups = st.multiselect(
            "Show column groups",
            list(COLUMN_GROUPS.keys()),
            default=list(COLUMN_GROUPS.keys()),
            key="screener_col_groups",
        )

    # Derive defaults from quick-filter preset
    qf_vals = QUICK_FILTERS.get(qf_choice, {}) if qf_choice != "None" else {}

    def _qf(key, default=None):
        return qf_vals.get(key, default)

    # ── 5. Filter form ────────────────────────────────────────────────────
    with st.form("screener_filters"):
        st.markdown("### ⚙️ 5. Set Filters *(leave blank to ignore)*")

        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Valuation", "💰 Profitability", "📈 Growth",
            "🏦 Financial Health", "💵 Cash Flow & Dividends", "📉 Technical",
        ])

        # ── Valuation tab ──
        with tab1:
            c1, c2 = st.columns(2)
            with c1:
                mc_min = st.number_input("Market Cap (B) ≥", value=None, format="%.2f")
                pe_min = st.number_input("Trailing P/E ≥", value=_qf("Trailing P/E_min"), format="%.2f")
                fwd_pe_min = st.number_input("Forward P/E ≥", value=None, format="%.2f")
                peg_min = st.number_input("PEG Ratio ≥", value=None, format="%.2f")
                pb_min = st.number_input("P/B ≥", value=None, format="%.2f")
                ps_min = st.number_input("P/S ≥", value=None, format="%.2f")
                ev_ebitda_min = st.number_input("EV/EBITDA ≥", value=None, format="%.2f")
            with c2:
                mc_max = st.number_input("Market Cap (B) ≤", value=None, format="%.2f")
                pe_max = st.number_input("Trailing P/E ≤", value=_qf("Trailing P/E_max"), format="%.2f")
                fwd_pe_max = st.number_input("Forward P/E ≤", value=None, format="%.2f")
                peg_max = st.number_input("PEG Ratio ≤", value=None, format="%.2f")
                pb_max = st.number_input("P/B ≤", value=None, format="%.2f")
                ps_max = st.number_input("P/S ≤", value=None, format="%.2f")
                ev_ebitda_max = st.number_input("EV/EBITDA ≤", value=None, format="%.2f")

        # ── Profitability tab ──
        with tab2:
            c1, c2 = st.columns(2)
            with c1:
                roe_min = st.number_input("ROE (%) ≥", value=None, format="%.2f")
                roa_min = st.number_input("ROA (%) ≥", value=None, format="%.2f")
                pm_min = st.number_input("Profit Margin (%) ≥", value=_qf("Profit Margin (%)_min"), format="%.2f")
                om_min = st.number_input("Operating Margin (%) ≥", value=None, format="%.2f")
                gm_min = st.number_input("Gross Margin (%) ≥", value=None, format="%.2f")
                eps_min = st.number_input("EPS (TTM) ≥", value=None, format="%.2f")
            with c2:
                roe_max = st.number_input("ROE (%) ≤", value=None, format="%.2f")
                roa_max = st.number_input("ROA (%) ≤", value=None, format="%.2f")
                pm_max = st.number_input("Profit Margin (%) ≤", value=None, format="%.2f")
                om_max = st.number_input("Operating Margin (%) ≤", value=None, format="%.2f")
                gm_max = st.number_input("Gross Margin (%) ≤", value=None, format="%.2f")
                eps_max = st.number_input("EPS (TTM) ≤", value=None, format="%.2f")

        # ── Growth tab ──
        with tab3:
            c1, c2 = st.columns(2)
            with c1:
                rev_g_min = st.number_input("Revenue Growth (%) ≥", value=_qf("Revenue Growth (%)_min"), format="%.2f")
                earn_g_min = st.number_input("Earnings Growth (%) ≥", value=None, format="%.2f")
                eps_g_min = st.number_input("EPS Growth (%) ≥", value=None, format="%.2f")
            with c2:
                rev_g_max = st.number_input("Revenue Growth (%) ≤", value=None, format="%.2f")
                earn_g_max = st.number_input("Earnings Growth (%) ≤", value=None, format="%.2f")
                eps_g_max = st.number_input("EPS Growth (%) ≤", value=None, format="%.2f")

        # ── Financial Health tab ──
        with tab4:
            c1, c2 = st.columns(2)
            with c1:
                de_min = st.number_input("Debt/Equity ≥", value=None, format="%.2f")
                cr_min = st.number_input("Current Ratio ≥", value=None, format="%.2f")
                qr_min = st.number_input("Quick Ratio ≥", value=None, format="%.2f")
                ic_min = st.number_input("Interest Coverage ≥", value=None, format="%.2f")
            with c2:
                de_max = st.number_input("Debt/Equity ≤", value=_qf("Debt/Equity_max"), format="%.2f")
                cr_max = st.number_input("Current Ratio ≤", value=None, format="%.2f")
                qr_max = st.number_input("Quick Ratio ≤", value=None, format="%.2f")
                ic_max = st.number_input("Interest Coverage ≤", value=None, format="%.2f")

        # ── Cash Flow & Dividends tab ──
        with tab5:
            c1, c2 = st.columns(2)
            with c1:
                fcf_min = st.number_input("Free Cash Flow (B) ≥", value=None, format="%.2f")
                fcfy_min = st.number_input("FCF Yield (%) ≥", value=None, format="%.2f")
                dy_min = st.number_input("Dividend Yield (%) ≥", value=_qf("Dividend Yield (%)_min"), format="%.2f")
                pr_min = st.number_input("Payout Ratio (%) ≥", value=None, format="%.2f")
            with c2:
                fcf_max = st.number_input("Free Cash Flow (B) ≤", value=None, format="%.2f")
                fcfy_max = st.number_input("FCF Yield (%) ≤", value=None, format="%.2f")
                dy_max = st.number_input("Dividend Yield (%) ≤", value=None, format="%.2f")
                pr_max = st.number_input("Payout Ratio (%) ≤", value=None, format="%.2f")

        # ── Technical tab ──
        with tab5:
            pass  # already defined above

        with tab6:
            c1, c2 = st.columns(2)
            with c1:
                rsi_min = st.number_input("RSI (14) ≥", value=None, min_value=0.0, max_value=100.0, format="%.1f")
                beta_min = st.number_input("Beta ≥", value=None, format="%.2f")
                w52_chg_min = st.number_input("52W Change (%) ≥", value=None, format="%.2f")
                off52h_min = st.number_input("% off 52W High ≥", value=None, format="%.2f")
            with c2:
                rsi_max = st.number_input("RSI (14) ≤", value=_qf("RSI (14)_max"), min_value=0.0, max_value=100.0, format="%.1f")
                beta_max = st.number_input("Beta ≤", value=None, format="%.2f")
                w52_chg_max = st.number_input("52W Change (%) ≤", value=None, format="%.2f")
                off52h_max = st.number_input("% off 52W High ≤", value=_qf("% off 52W High_max"), format="%.2f")

        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            sort_by = st.selectbox(
                "Sort by",
                ["None", "Market Cap (B)", "Price ($)", "Trailing P/E", "Forward P/E",
                 "PEG Ratio", "ROE (%)", "Profit Margin (%)", "Revenue Growth (%)",
                 "Dividend Yield (%)", "RSI (14)", "P/B", "EV/EBITDA",
                 "Free Cash Flow (B)", "FCF Yield (%)", "Undervaluation (%)", "Score (0-100)",
                 "52W Change (%)", "% off 52W High"],
            )
        with c2:
            ascending = st.checkbox("Ascending order", value=False)

        run = st.form_submit_button("🚀 RUN SCREENER", type="primary", use_container_width=True)

    if not run:
        return

    # ── 6. Screening loop ─────────────────────────────────────────────────
    st.markdown("### 📡 Screening in progress…")
    include_rsi = rsi_min is not None or rsi_max is not None

    results: List[Dict] = []
    errors: List[str] = []
    progress_bar = st.progress(0)
    status_col, eta_col = st.columns([3, 1])
    status_text = status_col.empty()
    eta_text = eta_col.empty()

    t_start = time.time()

    for i, symbol in enumerate(universe_symbols):
        progress_bar.progress((i + 1) / len(universe_symbols))
        status_text.text(f"Screening {symbol}… ({i+1}/{len(universe_symbols)})")

        # ETA estimate
        elapsed = time.time() - t_start
        if i > 0:
            rate = elapsed / i  # seconds per ticker
            remaining = rate * (len(universe_symbols) - i)
            eta_text.text(f"~{int(remaining)}s left")

        row, err = fetch_market_data(symbol, include_rsi)
        if row is None:
            errors.append(f"{symbol}: {err}")
            continue

        if run_valuation and HAS_VALUATION:
            val = fetch_valuation_data(symbol, val_model, val_mos)
            if val:
                row.update(val)

        results.append(row)

        if (i + 1) % 10 == 0:
            time.sleep(0.5)

    progress_bar.empty()
    status_text.empty()
    eta_text.empty()

    if errors:
        with st.expander(f"⚠️ {len(errors)} ticker(s) failed — click to see", expanded=False):
            for e in errors:
                st.caption(e)

    if not results:
        st.error("❌ No data retrieved. Check your internet connection or try a smaller universe.")
        return

    df = pd.DataFrame(results)

    # ── 7. Apply filters ──────────────────────────────────────────────────
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
    apply("Forward P/E", fwd_pe_min, fwd_pe_max)
    apply("PEG Ratio", peg_min, peg_max)
    apply("P/B", pb_min, pb_max)
    apply("P/S", ps_min, ps_max)
    apply("EV/EBITDA", ev_ebitda_min, ev_ebitda_max)
    apply("ROE (%)", roe_min, roe_max)
    apply("ROA (%)", roa_min, roa_max)
    apply("Profit Margin (%)", pm_min, pm_max)
    apply("Operating Margin (%)", om_min, om_max)
    apply("Gross Margin (%)", gm_min, gm_max)
    apply("EPS (TTM)", eps_min, eps_max)
    apply("Revenue Growth (%)", rev_g_min, rev_g_max)
    apply("Earnings Growth (%)", earn_g_min, earn_g_max)
    apply("EPS Growth (%)", eps_g_min, eps_g_max)
    apply("Debt/Equity", de_min, de_max)
    apply("Current Ratio", cr_min, cr_max)
    apply("Quick Ratio", qr_min, qr_max)
    apply("Interest Coverage", ic_min, ic_max)
    apply("Free Cash Flow (B)", fcf_min, fcf_max)
    apply("FCF Yield (%)", fcfy_min, fcfy_max)
    apply("Dividend Yield (%)", dy_min, dy_max)
    apply("Payout Ratio (%)", pr_min, pr_max)
    apply("RSI (14)", rsi_min, rsi_max)
    apply("Beta", beta_min, beta_max)
    apply("52W Change (%)", w52_chg_min, w52_chg_max)
    apply("% off 52W High", off52h_min, off52h_max)

    if run_valuation and HAS_VALUATION and "Undervaluation (%)" in df.columns:
        mask &= df["Undervaluation (%)"].fillna(-np.inf) >= min_undervaluation

    final = df[mask].copy()

    if final.empty:
        st.warning("⚠️ No stocks match your filters. Try relaxing your criteria.")
        return

    # ── 8. Sort ───────────────────────────────────────────────────────────
    if sort_by != "None" and sort_by in final.columns:
        final = final.sort_values(sort_by, ascending=ascending, na_position="last")

    # ── 9. Column visibility ──────────────────────────────────────────────
    visible_cols = []
    for group in visible_groups:
        for col in COLUMN_GROUPS.get(group, []):
            if col in final.columns and col not in visible_cols:
                visible_cols.append(col)
    # Always keep Symbol
    if "Symbol" not in visible_cols:
        visible_cols = ["Symbol"] + visible_cols
    # Add any valuation columns if present
    val_cols = ["Intrinsic Value ($)", "Safe Buy Price ($)", "Undervaluation (%)",
                "Verdict", "Score (0-100)", "Lynch Value ($)", "DCF Value ($)", "Graham Value ($)"]
    for vc in val_cols:
        if vc in final.columns and vc not in visible_cols:
            visible_cols.append(vc)

    display_df = final[[c for c in visible_cols if c in final.columns]].copy()

    # ── 10. Results display ───────────────────────────────────────────────
    st.success(
        f"✅ **{len(final)} stocks** passed your screen out of {len(df)} analyzed."
        + (f" ({len(errors)} tickers skipped)" if errors else "")
    )

    _render_summary_cards(final)

    # Verdict legend
    if run_valuation and HAS_VALUATION and "Verdict" in final.columns:
        st.caption("🟢 Strong Buy  |  🟡 Buy  |  🟠 Hold  |  🔴 Sell")

    # Round numeric columns
    for col in display_df.select_dtypes(include=[np.number]).columns:
        display_df[col] = display_df[col].round(2)

    # Colour-format with pandas Styler
    num_cols = display_df.select_dtypes(include=[np.number]).columns.tolist()

    def _style(styler):
        # Green = high, red = low for good metrics; inverted for bad ones
        bad_if_high = {"Debt/Equity", "Payout Ratio (%)", "Short Ratio",
                       "Trailing P/E", "Forward P/E", "P/B", "P/S",
                       "EV/EBITDA", "EV/Revenue"}
        for col in num_cols:
            if col in bad_if_high:
                styler = styler.background_gradient(subset=[col], cmap="RdYlGn_r", axis=0)
            else:
                styler = styler.background_gradient(subset=[col], cmap="RdYlGn", axis=0)
        return styler

    try:
        styled = display_df.style.pipe(_style).format(
            {c: "{:.2f}" for c in num_cols}, na_rep="—"
        )
        st.dataframe(styled, height=600, use_container_width=True)
    except Exception:
        st.dataframe(display_df, height=600, use_container_width=True)

    # ── Sector breakdown ──────────────────────────────────────────────────
    if "Sector" in final.columns:
        with st.expander("🗂️ Sector Breakdown", expanded=False):
            _render_sector_chart(final)

    # ── Watchlist adder ───────────────────────────────────────────────────
    with st.expander("⭐ Add to Watchlist", expanded=False):
        symbols_to_add = st.multiselect(
            "Select symbols to star",
            options=final["Symbol"].tolist(),
            default=[],
            key="screener_wl_add",
        )
        if st.button("Add selected to Watchlist", key="btn_wl_add"):
            for s in symbols_to_add:
                st.session_state["screener_watchlist"].add(s)
            st.success(f"Added {len(symbols_to_add)} stock(s) to watchlist.")
            st.rerun()

    # ── Downloads ─────────────────────────────────────────────────────────
    st.markdown("#### 📥 Export Results")
    col_a, col_b = st.columns(2)

    with col_a:
        st.download_button(
            label="⬇️ Download CSV",
            data=final.to_csv(index=False),
            file_name=f"screener_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col_b:
        if run_valuation and HAS_VALUATION and "Verdict" in final.columns:
            try:
                pdf_cols = [c for c in ["Symbol","Intrinsic Value ($)","Undervaluation (%)","Verdict","Beta"] if c in final.columns]
                pdf_portfolio = final[pdf_cols].rename(columns={
                    "Symbol": "Ticker",
                    "Intrinsic Value ($)": "Intrinsic Value",
                    "Undervaluation (%)": "Undervaluation %",
                })
                first = final.iloc[0]
                pdf_results = {
                    "model": val_model,
                    "current_price": first.get("Price ($)", 0),
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
                    label="⬇️ Download PDF Report",
                    data=pdf_bytes,
                    file_name=f"screener_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            except Exception as e:
                st.caption(f"PDF export unavailable: {e}")
        else:
            st.caption("Enable valuation models above to unlock PDF export.")


# ─────────────────────────────────────────────────────────────────────────────
# Public alias — this is what app.py imports
# ─────────────────────────────────────────────────────────────────────────────
render_screener = display_screener


# ─────────────────────────────────────────────────────────────────────────────
# Stand-alone entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    st.set_page_config(
        page_title="Ultimate Stock Screener",
        page_icon="📊",
        layout="wide",
    )
    display_screener()
