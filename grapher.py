"""
Financial Grapher Pro — Improved
- Fetches maximum available periods by combining new + legacy yfinance APIs
- Fixed FCF capex sign handling
- Full Ratios tab: P/E, EV/EBITDA, Debt/Equity, Current Ratio, ROA, ROCE
- New EPS & Per-Share tab
- Analyst tab with visualizations instead of raw tables
- Debug expander hidden by default (toggle in sidebar)
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, timedelta
import numpy as np
import time


# ══════════════════════════════════════════════════════════════════════════════
# DATA FETCHING  — tries new API first, falls back to legacy, picks the longer
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600)
def fetch_price_data(ticker: str, start_date: date, end_date: date) -> pd.Series:
    max_retries = 3
    for attempt in range(max_retries):
        try:
            prices = yf.download(
                ticker, start=start_date, end=end_date,
                progress=False, auto_adjust=True
            )
            if isinstance(prices, pd.DataFrame) and not prices.empty:
                if isinstance(prices.columns, pd.MultiIndex):
                    prices.columns = prices.columns.get_level_values(0)
                series = prices["Close"] if "Close" in prices.columns else prices.iloc[:, 0]
                if isinstance(series, pd.DataFrame):
                    series = series.squeeze()
                return series.dropna()
            return pd.Series(dtype=float)
        except Exception:
            if attempt == max_retries - 1:
                return pd.Series(dtype=float)
            time.sleep(1)
    return pd.Series(dtype=float)


def _merge_statements(new_df: pd.DataFrame, legacy_df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine two statement DataFrames (same row labels, potentially different column sets).
    Columns are Timestamps; we take the union, preferring `new_df` on conflicts.
    """
    if new_df is None or new_df.empty:
        return legacy_df if legacy_df is not None else pd.DataFrame()
    if legacy_df is None or legacy_df.empty:
        return new_df

    # Normalize column names to date only (strip time component)
    def normalize_cols(df):
        df = df.copy()
        df.columns = pd.to_datetime(df.columns).normalize()
        return df

    new_df = normalize_cols(new_df)
    legacy_df = normalize_cols(legacy_df)

    # Align rows (union of index)
    combined = new_df.combine_first(legacy_df)
    combined = combined.sort_index(axis=1)  # chronological columns
    return combined


@st.cache_data(ttl=3600)
def fetch_fundamentals(ticker: str, frequency: str):
    """
    Fetch income, balance, cashflow for the given frequency.
    Tries the new pretty API and the legacy API, then merges both to
    maximise the number of historical periods returned.
    """
    try:
        t = yf.Ticker(ticker)
        annual = (frequency == "Annual")
        freq_str = "yearly" if annual else "quarterly"

        # ── New API ──────────────────────────────────────────────────────────
        try:
            inc_new  = t.get_income_stmt(pretty=True, freq=freq_str)
            bal_new  = t.get_balance_sheet(pretty=True, freq=freq_str)
            cf_new   = t.get_cash_flow(pretty=True, freq=freq_str)
        except Exception:
            inc_new = bal_new = cf_new = pd.DataFrame()

        # ── Legacy API ───────────────────────────────────────────────────────
        if annual:
            inc_leg = t.financials
            bal_leg = t.balance_sheet
            cf_leg  = t.cashflow
        else:
            inc_leg = t.quarterly_financials
            bal_leg = t.quarterly_balance_sheet
            cf_leg  = t.quarterly_cashflow

        # ── Merge (maximise periods) ─────────────────────────────────────────
        income   = _merge_statements(inc_new, inc_leg)
        balance  = _merge_statements(bal_new, bal_leg)
        cashflow = _merge_statements(cf_new, cf_leg)

        info = t.info
        return income, balance, cashflow, info

    except Exception as e:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}


@st.cache_data(ttl=3600)
def fetch_analyst_data(ticker: str) -> dict:
    try:
        t = yf.Ticker(ticker)
        return {
            "recommendations":   getattr(t, "recommendations", None),
            "price_target":      getattr(t, "analyst_price_targets", None),
            "earnings_forecasts":getattr(t, "earnings_forecasts", None),
            "revenue_forecasts": getattr(t, "revenue_estimate", None),
            "earnings_trend":    getattr(t, "earnings_trend", None),
            "upgrades_downgrades": getattr(t, "upgrades_downgrades", None),
        }
    except Exception:
        return {}


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def safe_get(df: pd.DataFrame, key: str, default=None) -> pd.Series:
    """Return a chronologically sorted Series from a statement DataFrame."""
    if df is None or df.empty:
        return pd.Series(dtype=float)

    def _extract(idx_key):
        s = df.loc[idx_key]
        if isinstance(s, pd.DataFrame):
            s = s.squeeze()
        s.index = pd.to_datetime(s.index).normalize()
        return s.sort_index().dropna()

    if key in df.index:
        return _extract(key)

    # Case-insensitive partial match
    for idx in df.index:
        if isinstance(idx, str) and key.lower() in idx.lower():
            return _extract(idx)

    # Curated aliases
    ALIASES = {
        "Total Revenue":                ["TotalRevenue", "Total Revenues", "Revenue"],
        "Gross Profit":                 ["GrossProfit", "Gross Income"],
        "Operating Income":             ["OperatingIncome", "EBIT", "Operating Revenue", "Ebit"],
        "Net Income":                   ["NetIncome", "Net Income Common Stockholders", "Net Income Applicable To Common Shares"],
        "Operating Cash Flow":          ["OperatingCashFlow", "Total Cash From Operating Activities", "Cash Flow From Continuing Operating Activities"],
        "Capital Expenditure":          ["CapitalExpenditure", "Capital Expenditures", "Purchase Of PPE", "Capital Expenditures Reported"],
        "Stock Based Compensation":     ["StockBasedCompensation", "Stock Based Compensation"],
        "Basic Average Shares":         ["BasicAverageShares", "Ordinary Shares Number", "Basic Shares Outstanding"],
        "Diluted Average Shares":       ["DilutedAverageShares", "Diluted Shares Outstanding"],
        "Diluted EPS":                  ["DilutedEPS", "Diluted Eps", "Basic EPS", "BasicEPS"],
        "Basic EPS":                    ["BasicEPS", "Basic Eps", "Diluted EPS"],
        "Total Assets":                 ["TotalAssets"],
        "Total Debt":                   ["TotalDebt", "Long Term Debt", "LongTermDebt"],
        "Current Assets":               ["CurrentAssets", "Total Current Assets"],
        "Current Liabilities":          ["CurrentLiabilities", "Total Current Liabilities"],
        "Cash And Cash Equivalents":    ["CashAndCashEquivalents", "Cash Cash Equivalents And Short Term Investments", "Cash And Short Term Investments"],
        "Tax Provision":                ["TaxProvision", "Income Tax Expense"],
        "Research Development":         ["ResearchAndDevelopment", "Research Development", "Research And Development"],
        "Selling General Administrative": ["SellingGeneralAndAdministrative", "Selling General Administrative"],
        "Total Liabilities":            ["TotalLiabilities", "Total Liabilities Net Minority Interest"],
        "Stockholders Equity":          ["StockholdersEquity", "Total Equity Gross Minority Interest", "Common Stock Equity"],
        "EBITDA":                       ["EBITDA", "Normalized EBITDA"],
        "Interest Expense":             ["InterestExpense", "Interest Expense Non Operating"],
        "Inventory":                    ["Inventory", "Inventories"],
    }

    for alt in ALIASES.get(key, []):
        if alt in df.index:
            return _extract(alt)
        for idx in df.index:
            if isinstance(idx, str) and alt.lower() == idx.lower():
                return _extract(idx)

    if default is not None:
        cols = pd.to_datetime(df.columns).normalize().sort_values() if not df.empty else pd.Index([])
        return pd.Series(default, index=cols)
    return pd.Series(dtype=float)


def fmt(num: float) -> str:
    if pd.isna(num):
        return "N/A"
    sign = "-" if num < 0 else ""
    a = abs(num)
    if a >= 1e12:
        return f"{sign}${a/1e12:.2f}T"
    if a >= 1e9:
        return f"{sign}${a/1e9:.2f}B"
    if a >= 1e6:
        return f"{sign}${a/1e6:.2f}M"
    if a >= 1e3:
        return f"{sign}${a/1e3:.2f}K"
    return f"{sign}${a:.2f}"


def cagr(s: pd.Series) -> float:
    c = s.dropna()
    if len(c) < 2 or c.iloc[0] <= 0 or c.iloc[-1] <= 0:
        return np.nan
    years = (c.index[-1] - c.index[0]).days / 365.25
    return (pow(c.iloc[-1] / c.iloc[0], 1 / years) - 1) * 100 if years > 0 else np.nan


def pct_change_yoy(s: pd.Series) -> pd.Series:
    return s.pct_change() * 100


# ══════════════════════════════════════════════════════════════════════════════
# PLOT HELPERS  — consistent styling throughout
# ══════════════════════════════════════════════════════════════════════════════

PALETTE = ["#2196F3", "#4CAF50", "#FF5722", "#9C27B0", "#FF9800", "#00BCD4", "#F44336"]

def _cagr_subtitle(s: pd.Series) -> str:
    c = cagr(s)
    return f"<br><sup>CAGR: {c:.1f}%</sup>" if not np.isnan(c) else ""


def plot_line(data: pd.Series, title: str, yaxis: str = "Value",
              color: str = None, show_yoy: bool = False):
    if isinstance(data, pd.DataFrame):
        data = data.squeeze()
    data = data.dropna()
    if data.empty:
        st.info(f"📊 {title}: No data available")
        return

    fig = px.line(
        x=data.index, y=data, title=title + _cagr_subtitle(data),
        markers=True, color_discrete_sequence=[color or PALETTE[0]]
    )
    fig.update_traces(line_width=2.5)

    if show_yoy and len(data) > 1:
        yoy = pct_change_yoy(data).dropna()
        if not yoy.empty:
            fig.add_bar(x=yoy.index, y=yoy, name="YoY %", yaxis="y2",
                        marker_color="rgba(150,150,150,0.4)")
            fig.update_layout(
                yaxis2=dict(title="YoY Growth %", overlaying="y", side="right"),
                showlegend=True
            )

    fig.update_layout(yaxis_title=yaxis, hovermode="x unified", height=420,
                      margin=dict(t=60))
    st.plotly_chart(fig, use_container_width=True)


def plot_bar(data: pd.Series, title: str, yaxis: str = "Value",
             color: str = None, show_yoy: bool = False):
    if isinstance(data, pd.DataFrame):
        data = data.squeeze()
    data = data.dropna()
    if data.empty:
        st.info(f"📊 {title}: No data available")
        return

    fig = px.bar(
        x=data.index, y=data, title=title + _cagr_subtitle(data),
        color_discrete_sequence=[color or PALETTE[1]]
    )

    if show_yoy and len(data) > 1:
        yoy = pct_change_yoy(data).dropna()
        if not yoy.empty:
            fig.add_scatter(x=yoy.index, y=yoy, name="YoY %", yaxis="y2",
                            mode="lines+markers",
                            line=dict(color="gray", dash="dot"))
            fig.update_layout(
                yaxis2=dict(title="YoY %", overlaying="y", side="right"),
                showlegend=True
            )

    fig.update_layout(yaxis_title=yaxis, height=420, margin=dict(t=60))
    st.plotly_chart(fig, use_container_width=True)


def plot_multi(df: pd.DataFrame, title: str, yaxis: str = "Value",
               colors: list = None):
    if df.empty or df.isna().all().all():
        st.info(f"📊 {title}: No data available")
        return

    fig = go.Figure()
    cols_used = 0
    for i, col in enumerate(df.columns):
        s = df[col].dropna()
        if s.empty:
            continue
        fig.add_trace(go.Scatter(
            x=s.index, y=s, mode="lines+markers", name=str(col),
            line=dict(color=(colors or PALETTE)[i % len(colors or PALETTE)],
                      width=2.5)
        ))
        cols_used += 1

    if cols_used == 0:
        st.info(f"📊 {title}: No data available")
        return

    fig.update_layout(title=title, yaxis_title=yaxis, hovermode="x unified",
                      height=420, showlegend=True, margin=dict(t=60))
    st.plotly_chart(fig, use_container_width=True)


def plot_waterfall(categories: list, values: list, title: str):
    fig = go.Figure(go.Waterfall(
        name="", orientation="v",
        x=categories, y=values,
        connector={"line": {"color": "rgb(63,63,63)"}},
        increasing={"marker": {"color": "#4CAF50"}},
        decreasing={"marker": {"color": "#F44336"}},
        totals={"marker": {"color": "#2196F3"}},
    ))
    fig.update_layout(title=title, height=400, margin=dict(t=60))
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════════

def render_grapher():
    st.set_page_config(
        page_title="Financial Grapher Pro",
        page_icon="📈",
        layout="wide"
    )
    st.markdown("""
        <style>
        [data-testid="stMetric"] {
            background: #f7f9fc;
            border-radius: 8px;
            padding: 12px 16px;
            border-left: 4px solid #2196F3;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("📈 Financial Grapher Pro")
    st.caption("Comprehensive financial analysis — price history, fundamentals, ratios & analyst forecasts")

    # ── Sidebar controls ────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Settings")
        show_debug = st.checkbox("Show debug info", value=False)
        st.markdown("---")
        st.markdown("**Data Notes**")
        st.markdown(
            "Yahoo Finance provides ~4 years of annual data and ~8 quarters of quarterly data. "
            "This app combines the new and legacy APIs to maximise coverage."
        )

    # ── Controls ─────────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns([3, 2, 2])
    with col1:
        ticker = st.text_input("Ticker Symbol", value="AAPL",
                               placeholder="e.g. AAPL, MSFT, TSLA").strip().upper()
    with col2:
        frequency = st.selectbox("Reporting Frequency", ["Annual", "Quarterly"], index=0)
    with col3:
        price_years = st.slider("Price History (Years)", 1, 30, 10)

    if not ticker:
        st.info("👆 Enter a ticker symbol to get started")
        return

    end_date   = date.today()
    start_date = end_date - timedelta(days=365 * price_years)

    if not st.button("🚀 Load & Analyse", type="primary", use_container_width=True):
        return

    # ── Fetch ────────────────────────────────────────────────────────────────
    with st.spinner(f"Fetching data for **{ticker}** …"):
        income, balance, cashflow, info = fetch_fundamentals(ticker, frequency)
        prices                           = fetch_price_data(ticker, start_date, end_date)
        analyst_data                     = fetch_analyst_data(ticker)

    if prices.empty and income.empty:
        st.error(f"❌  No data found for **{ticker}**. Check the ticker symbol.")
        return

    # ── Company header ───────────────────────────────────────────────────────
    st.markdown("### 📋 Company Overview")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Company", info.get("longName", ticker))
    with c2:
        st.metric("Sector", info.get("sector", "N/A"))
    with c3:
        st.metric("Industry", info.get("industry", "N/A"))
    with c4:
        mc = info.get("marketCap")
        st.metric("Market Cap", fmt(mc) if mc else "N/A")
    with c5:
        curr = info.get("currency", "USD")
        exch = info.get("exchange", "N/A")
        st.metric("Exchange / CCY", f"{exch} / {curr}")

    # ── Data availability banner ─────────────────────────────────────────────
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        if not prices.empty:
            st.success(f"✅ Price data: **{len(prices)}** trading days")
        else:
            st.warning("⚠️ Price data unavailable")
    with c2:
        if not income.empty:
            n = len(income.columns)
            st.success(f"✅ Fundamentals: **{n}** {frequency.lower()} periods")
        else:
            st.warning("⚠️ Fundamental data unavailable")
    with c3:
        has_analyst = any(
            v is not None and (
                (isinstance(v, pd.DataFrame) and not v.empty) or
                (isinstance(v, dict) and bool(v))
            )
            for v in analyst_data.values()
        )
        if has_analyst:
            st.success("✅ Analyst forecasts available")
        else:
            st.info("ℹ️ Analyst data limited or unavailable")

    # ── Debug ────────────────────────────────────────────────────────────────
    if show_debug:
        with st.expander("🔍 Debug: raw index labels"):
            st.write("**Income index:**",  list(income.index)   if not income.empty   else "EMPTY")
            st.write("**Income columns:**",list(income.columns) if not income.empty   else "EMPTY")
            st.write("**Cashflow index:**",list(cashflow.index) if not cashflow.empty else "EMPTY")
            st.write("**Balance index:**", list(balance.index)  if not balance.empty  else "EMPTY")

    # ── Extract all metrics ──────────────────────────────────────────────────
    revenue        = safe_get(income, "Total Revenue")
    gross_profit   = safe_get(income, "Gross Profit")
    op_income      = safe_get(income, "Operating Income")
    net_income     = safe_get(income, "Net Income")
    rd_expense     = safe_get(income, "Research Development")
    sga_expense    = safe_get(income, "Selling General Administrative")
    interest_exp   = safe_get(income, "Interest Expense")
    tax_provision  = safe_get(income, "Tax Provision")
    ebitda_raw     = safe_get(income, "EBITDA")

    ocf            = safe_get(cashflow, "Operating Cash Flow")
    capex_raw      = safe_get(cashflow, "Capital Expenditure")
    sbc            = safe_get(cashflow, "Stock Based Compensation")

    total_assets   = safe_get(balance, "Total Assets")
    total_liab     = safe_get(balance, "Total Liabilities")
    equity         = safe_get(balance, "Stockholders Equity")
    total_debt     = safe_get(balance, "Total Debt", 0)
    cash           = safe_get(balance, "Cash And Cash Equivalents", 0)
    current_assets = safe_get(balance, "Current Assets")
    current_liab   = safe_get(balance, "Current Liabilities")
    inventory      = safe_get(balance, "Inventory")

    shares_basic   = safe_get(income, "Basic Average Shares")
    shares_diluted = safe_get(income, "Diluted Average Shares")
    eps_diluted    = safe_get(income, "Diluted EPS")
    eps_basic      = safe_get(income, "Basic EPS")

    # Fix capex: yfinance returns it as negative; FCF = OCF + capex
    # Guard against unexpected positive capex from some tickers
    if not capex_raw.empty:
        capex = -capex_raw.abs()    # ensure it's always stored as negative
    else:
        capex = pd.Series(dtype=float)

    # Derived
    fcf = (ocf + capex_raw) if not ocf.empty and not capex_raw.empty else pd.Series(dtype=float)

    # ── TABS ─────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 📊 Comprehensive Analysis")

    tab_labels = [
        "📈 Price & Revenue",
        "💰 Profitability",
        "💵 Cash Flow",
        "📊 Balance Sheet",
        "📐 Ratios & Multiples",
        "🔢 EPS & Per-Share",
        "🔮 Analyst Forecasts",
    ]
    tabs = st.tabs(tab_labels)

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 1 · Price & Revenue
    # ─────────────────────────────────────────────────────────────────────────
    with tabs[0]:
        st.markdown("### Stock Price History")
        if not prices.empty:
            plot_line(prices, f"{ticker} Stock Price", "Price (USD)", PALETTE[0])

            c1, c2, c3, c4, c5 = st.columns(5)
            p_last  = float(prices.iloc[-1])
            p_first = float(prices.iloc[0])
            with c1:
                st.metric("Current Price", f"${p_last:.2f}")
            with c2:
                st.metric(f"{price_years}Y High", f"${float(prices.max()):.2f}")
            with c3:
                st.metric(f"{price_years}Y Low",  f"${float(prices.min()):.2f}")
            with c4:
                total_ret = ((p_last / p_first) - 1) * 100
                st.metric(f"{price_years}Y Return", f"{total_ret:.1f}%")
            with c5:
                price_cagr = cagr(prices)
                st.metric("CAGR", f"{price_cagr:.1f}%" if not np.isnan(price_cagr) else "N/A")

            # Moving averages overlay
            st.markdown("#### Price + 50/200-day Moving Averages")
            fig_ma = go.Figure()
            fig_ma.add_trace(go.Scatter(x=prices.index, y=prices, name="Price",
                                        line=dict(color=PALETTE[0], width=1.5)))
            if len(prices) >= 50:
                ma50 = prices.rolling(50).mean()
                fig_ma.add_trace(go.Scatter(x=ma50.index, y=ma50, name="50-day MA",
                                            line=dict(color=PALETTE[2], dash="dot")))
            if len(prices) >= 200:
                ma200 = prices.rolling(200).mean()
                fig_ma.add_trace(go.Scatter(x=ma200.index, y=ma200, name="200-day MA",
                                            line=dict(color=PALETTE[3], dash="dash")))
            fig_ma.update_layout(hovermode="x unified", height=420, showlegend=True,
                                 yaxis_title="Price (USD)")
            st.plotly_chart(fig_ma, use_container_width=True)

        st.markdown("### Revenue")
        plot_bar(revenue, "Total Revenue", "USD", PALETTE[1], show_yoy=True)

        if not rd_expense.empty or not sga_expense.empty:
            st.markdown("### Operating Expenses Breakdown")
            opex_df = pd.DataFrame({"R&D": rd_expense, "SG&A": sga_expense}).dropna(how="all")
            if not opex_df.empty:
                plot_multi(opex_df, "Operating Expenses", "USD", [PALETTE[4], PALETTE[5]])

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 2 · Profitability
    # ─────────────────────────────────────────────────────────────────────────
    with tabs[1]:
        st.markdown("### Profit Margins")
        if not revenue.empty:
            margins = {}
            if not gross_profit.empty:
                margins["Gross Margin %"] = (gross_profit / revenue * 100)
            if not op_income.empty:
                margins["Operating Margin %"] = (op_income / revenue * 100)
            if not net_income.empty:
                margins["Net Margin %"] = (net_income / revenue * 100)
            if margins:
                plot_multi(pd.DataFrame(margins), "Margins (%)", "Margin (%)",
                           [PALETTE[1], PALETTE[0], PALETTE[2]])

        st.markdown("### Absolute Profit")
        profit_df = pd.DataFrame({
            "Gross Profit":     gross_profit,
            "Operating Income": op_income,
            "Net Income":       net_income,
        }).dropna(how="all")
        if not profit_df.empty:
            plot_multi(profit_df, "Profit Lines", "USD",
                       [PALETTE[1], PALETTE[0], PALETTE[2]])

        # Latest-period income waterfall
        if not income.empty and not revenue.empty:
            latest_col = income.columns[-1]
            try:
                _rev  = float(revenue.iloc[-1])
                _gp   = float(gross_profit.iloc[-1]) if not gross_profit.empty else np.nan
                _oi   = float(op_income.iloc[-1])    if not op_income.empty    else np.nan
                _ni   = float(net_income.iloc[-1])   if not net_income.empty   else np.nan

                if not any(np.isnan(v) for v in [_rev, _gp, _oi, _ni]):
                    st.markdown(f"### Income Waterfall — {pd.Timestamp(latest_col).strftime('%Y-%m-%d')}")
                    plot_waterfall(
                        ["Revenue", "→ Gross Profit", "→ Operating Income", "→ Net Income"],
                        [_rev, _gp - _rev, _oi - _gp, _ni - _oi],
                        "From Revenue to Net Income (most recent period)"
                    )
            except Exception:
                pass

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 3 · Cash Flow
    # ─────────────────────────────────────────────────────────────────────────
    with tabs[2]:
        st.markdown("### Operating Cash Flow")
        plot_line(ocf, "Operating Cash Flow", "USD", PALETTE[0], show_yoy=True)

        st.markdown("### Capital Expenditure (absolute)")
        if not capex_raw.empty:
            plot_bar(capex_raw.abs(), "CapEx (absolute)", "USD", PALETTE[2])

        st.markdown("### Free Cash Flow  (OCF − CapEx)")
        if not fcf.empty:
            plot_bar(fcf, "Free Cash Flow", "USD", PALETTE[1], show_yoy=True)

        # OCF vs FCF overlay
        if not ocf.empty and not fcf.empty:
            st.markdown("### OCF vs FCF vs SBC")
            cf_df = pd.DataFrame({"Operating CF": ocf, "Free CF": fcf})
            if not sbc.empty:
                cf_df["Stock-Based Comp"] = sbc
            plot_multi(cf_df, "Cash Flow Comparison", "USD",
                       [PALETTE[0], PALETTE[1], PALETTE[4]])

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 4 · Balance Sheet
    # ─────────────────────────────────────────────────────────────────────────
    with tabs[3]:
        st.markdown("### Assets, Liabilities & Equity")
        if not total_assets.empty:
            derived_equity = (total_assets - total_liab) if not total_liab.empty else equity
            bal_df = pd.DataFrame({
                "Total Assets":      total_assets,
                "Total Liabilities": total_liab,
                "Equity":            equity if not equity.empty else derived_equity,
            }).dropna(how="all")
            plot_multi(bal_df, "Balance Sheet", "USD",
                       [PALETTE[1], PALETTE[2], PALETTE[0]])

        st.markdown("### Debt vs Cash")
        if not total_debt.empty or not cash.empty:
            dc_df = pd.DataFrame({
                "Total Debt": total_debt if not isinstance(total_debt, int) else pd.Series(),
                "Cash & Equivalents": cash if not isinstance(cash, int) else pd.Series(),
            }).dropna(how="all")
            if not dc_df.empty:
                plot_multi(dc_df, "Debt vs Cash", "USD", [PALETTE[2], PALETTE[1]])

        # Net debt
        if not isinstance(total_debt, int) and not isinstance(cash, int):
            net_debt = total_debt - cash
            if not net_debt.empty:
                st.markdown("### Net Debt  (Total Debt − Cash)")
                plot_bar(net_debt, "Net Debt", "USD", PALETTE[3])

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 5 · Ratios & Multiples
    # ─────────────────────────────────────────────────────────────────────────
    with tabs[4]:
        # Current snapshot from info
        st.markdown("### 📌 Current Key Ratios (from latest data)")
        snap_metrics = {
            "P/E (trailing)":   info.get("trailingPE"),
            "P/E (forward)":    info.get("forwardPE"),
            "P/B":              info.get("priceToBook"),
            "EV/EBITDA":        info.get("enterpriseToEbitda"),
            "Debt/Equity":      info.get("debtToEquity"),
            "Current Ratio":    info.get("currentRatio"),
            "Quick Ratio":      info.get("quickRatio"),
            "ROE (ttm)":        info.get("returnOnEquity"),
            "ROA (ttm)":        info.get("returnOnAssets"),
            "Gross Margin":     info.get("grossMargins"),
            "Op. Margin":       info.get("operatingMargins"),
            "Net Margin":       info.get("profitMargins"),
        }
        snap_cols = st.columns(4)
        for i, (label, val) in enumerate(snap_metrics.items()):
            with snap_cols[i % 4]:
                if val is not None:
                    if "Margin" in label or "ROE" in label or "ROA" in label:
                        st.metric(label, f"{val*100:.1f}%")
                    else:
                        st.metric(label, f"{val:.2f}")
                else:
                    st.metric(label, "N/A")

        st.markdown("---")

        # Historical ratio charts (computed from statement data)
        st.markdown("### Historical Return Metrics")

        if not net_income.empty and not equity.empty:
            roe = (net_income / equity.shift(1) * 100).replace([np.inf, -np.inf], np.nan)
            plot_line(roe.dropna(), "Return on Equity (ROE)", "ROE (%)", PALETTE[0])

        if not net_income.empty and not total_assets.empty:
            roa = (net_income / total_assets.shift(1) * 100).replace([np.inf, -np.inf], np.nan)
            plot_line(roa.dropna(), "Return on Assets (ROA)", "ROA (%)", PALETTE[1])

        if not op_income.empty and not total_assets.empty and not total_liab.empty:
            capital_employed = total_assets - current_liab
            roce = (op_income / capital_employed.shift(1) * 100).replace([np.inf, -np.inf], np.nan)
            plot_line(roce.dropna(), "Return on Capital Employed (ROCE)", "ROCE (%)", PALETTE[4])

        st.markdown("### Leverage & Liquidity")

        if not isinstance(total_debt, int) and not equity.empty:
            de_ratio = (total_debt / equity).replace([np.inf, -np.inf], np.nan)
            plot_line(de_ratio.dropna(), "Debt / Equity Ratio", "D/E", PALETTE[2])

        if not current_assets.empty and not current_liab.empty:
            cr = (current_assets / current_liab).replace([np.inf, -np.inf], np.nan)
            plot_line(cr.dropna(), "Current Ratio", "Ratio", PALETTE[5])

        # Historical P/E using price and EPS
        if not prices.empty and not eps_diluted.empty:
            st.markdown("### Historical P/E  (year-end price ÷ Diluted EPS)")
            try:
                # Resample price to annual/quarterly year-end to align with EPS
                price_resampled = prices.resample("YE").last() if frequency == "Annual" \
                                  else prices.resample("QE").last()
                price_resampled.index = price_resampled.index.normalize()

                eps_aligned = eps_diluted.copy()
                eps_aligned.index = pd.DatetimeIndex(eps_aligned.index).normalize()

                common_idx = price_resampled.index.intersection(eps_aligned.index)
                if len(common_idx) >= 2:
                    hist_pe = (price_resampled.loc[common_idx] /
                               eps_aligned.loc[common_idx]).replace([np.inf, -np.inf], np.nan)
                    plot_line(hist_pe.dropna(), "Historical P/E Ratio", "P/E", PALETTE[3])
            except Exception:
                pass

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 6 · EPS & Per-Share
    # ─────────────────────────────────────────────────────────────────────────
    with tabs[5]:
        st.markdown("### Earnings Per Share")

        if not eps_diluted.empty or not eps_basic.empty:
            eps_df = pd.DataFrame({
                "Diluted EPS": eps_diluted,
                "Basic EPS":   eps_basic,
            }).dropna(how="all")
            plot_multi(eps_df, "EPS — Diluted vs Basic", "EPS (USD)",
                       [PALETTE[0], PALETTE[4]])
        elif not net_income.empty and not shares_diluted.empty:
            # Compute from statements
            calc_eps = (net_income / shares_diluted).dropna()
            if not calc_eps.empty:
                plot_line(calc_eps, "EPS (calculated)", "EPS (USD)", PALETTE[0])

        st.markdown("### Share Count Trend")
        if not shares_basic.empty or not shares_diluted.empty:
            sh_df = pd.DataFrame({
                "Basic Shares":   shares_basic,
                "Diluted Shares": shares_diluted,
            }).dropna(how="all")
            plot_multi(sh_df, "Shares Outstanding", "Shares", [PALETTE[1], PALETTE[2]])

        st.markdown("### Free Cash Flow Per Share")
        if not fcf.empty and not shares_diluted.empty:
            fcf_per_share = (fcf / shares_diluted).dropna()
            if not fcf_per_share.empty:
                plot_bar(fcf_per_share, "FCF Per Share", "FCF/Share (USD)", PALETTE[1])

        st.markdown("### Revenue Per Share")
        if not revenue.empty and not shares_diluted.empty:
            rev_per_share = (revenue / shares_diluted).dropna()
            if not rev_per_share.empty:
                plot_line(rev_per_share, "Revenue Per Share", "Revenue/Share (USD)",
                          PALETTE[5], show_yoy=True)

        # SBC as % of net income — dilution quality
        if not sbc.empty and not net_income.empty:
            st.markdown("### SBC as % of Net Income  (dilution indicator)")
            sbc_pct = (sbc / net_income.abs() * 100).replace([np.inf, -np.inf], np.nan)
            plot_bar(sbc_pct.dropna(), "Stock-Based Compensation / Net Income",
                     "SBC %", PALETTE[3])

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 7 · Analyst Forecasts
    # ─────────────────────────────────────────────────────────────────────────
    with tabs[6]:
        st.markdown("### 🔮 Analyst Forecasts & Recommendations")

        # Price targets
        pt = analyst_data.get("price_target")
        if isinstance(pt, dict) and pt:
            current_price = float(prices.iloc[-1]) if not prices.empty else None
            st.markdown("#### 🎯 Analyst Price Targets")
            c1, c2, c3, c4, c5 = st.columns(5)
            mean   = pt.get("mean", 0)
            low_t  = pt.get("low",  0)
            high_t = pt.get("high", 0)
            curr_t = pt.get("current", current_price or 0)
            with c1:
                st.metric("Current Price", f"${curr_t:.2f}" if curr_t else "N/A")
            with c2:
                upside = ((mean / curr_t) - 1) * 100 if curr_t and mean else None
                st.metric("Mean Target", f"${mean:.2f}",
                          delta=f"{upside:+.1f}%" if upside else None)
            with c3:
                st.metric("Low Target",  f"${low_t:.2f}")
            with c4:
                st.metric("High Target", f"${high_t:.2f}")
            with c5:
                range_pct = ((high_t - low_t) / curr_t * 100) if curr_t and high_t else None
                st.metric("Target Range", f"{range_pct:.1f}%" if range_pct else "N/A")

            # Visual: current price vs target range
            if curr_t and low_t and mean and high_t:
                fig_pt = go.Figure()
                fig_pt.add_shape(type="rect",
                                 x0=0, x1=1, y0=low_t, y1=high_t,
                                 fillcolor="rgba(33,150,243,0.15)",
                                 line=dict(width=0))
                fig_pt.add_hline(y=mean,   line=dict(color=PALETTE[0], dash="dash"),
                                 annotation_text="Mean target")
                fig_pt.add_hline(y=curr_t, line=dict(color=PALETTE[2], width=2),
                                 annotation_text="Current price")
                fig_pt.update_layout(
                    title="Price vs Analyst Target Range",
                    yaxis_title="Price (USD)",
                    height=280,
                    xaxis_visible=False
                )
                st.plotly_chart(fig_pt, use_container_width=True)

        # Recommendations summary
        recs = analyst_data.get("recommendations")
        if recs is not None and isinstance(recs, pd.DataFrame) and not recs.empty:
            st.markdown("#### 📊 Recent Analyst Recommendations")

            # Try to pivot into Buy / Hold / Sell counts per period
            try:
                if "period" in recs.columns:
                    summary_cols = [c for c in ["strongBuy", "buy", "hold", "sell", "strongSell"] if c in recs.columns]
                    if summary_cols:
                        rec_plot = recs.set_index("period")[summary_cols].iloc[::-1].head(8)
                        fig_rec = px.bar(
                            rec_plot, barmode="stack",
                            color_discrete_map={
                                "strongBuy": "#1B5E20", "buy": "#4CAF50",
                                "hold": "#FF9800",
                                "sell": "#F44336",  "strongSell": "#B71C1C"
                            },
                            title="Analyst Recommendations by Period",
                            labels={"value": "# Analysts", "period": "Period"}
                        )
                        fig_rec.update_layout(height=380)
                        st.plotly_chart(fig_rec, use_container_width=True)
                    else:
                        st.dataframe(recs.tail(12), use_container_width=True)
                else:
                    st.dataframe(recs.tail(12), use_container_width=True)
            except Exception:
                st.dataframe(recs.tail(12), use_container_width=True)

        # Revenue / Earnings forecasts
        rev_fc = analyst_data.get("revenue_forecasts")
        if rev_fc is not None and isinstance(rev_fc, pd.DataFrame) and not rev_fc.empty:
            st.markdown("#### 💰 Revenue Estimates")
            st.dataframe(rev_fc, use_container_width=True)

        earn_fc = analyst_data.get("earnings_forecasts")
        if earn_fc is not None and isinstance(earn_fc, pd.DataFrame) and not earn_fc.empty:
            st.markdown("#### 📈 Earnings Estimates")
            st.dataframe(earn_fc, use_container_width=True)

        earn_trend = analyst_data.get("earnings_trend")
        if earn_trend is not None and isinstance(earn_trend, pd.DataFrame) and not earn_trend.empty:
            st.markdown("#### 📉 Earnings Trend")
            st.dataframe(earn_trend, use_container_width=True)

        upgrades = analyst_data.get("upgrades_downgrades")
        if upgrades is not None and isinstance(upgrades, pd.DataFrame) and not upgrades.empty:
            st.markdown("#### ⬆️⬇️ Recent Upgrades / Downgrades")
            # colour-code Action column if present
            try:
                display_df = upgrades.tail(20).copy()
                st.dataframe(display_df, use_container_width=True)
            except Exception:
                st.dataframe(upgrades.tail(20), use_container_width=True)

        if not has_analyst:
            st.warning("⚠️ No analyst data could be retrieved for this ticker.")

    # ── Footer ───────────────────────────────────────────────────────────────
    st.markdown("---")
    periods = len(income.columns) if not income.empty else 0
    st.success(
        f"✅ Analysis complete — **{periods}** {frequency.lower()} fundamental periods "
        f"| **{len(prices)}** price data points"
    )
    st.caption(
        "Data sourced from Yahoo Finance via yfinance. "
        "Fundamental data is limited to ~4 annual periods or ~8 quarterly periods by Yahoo Finance. "
        "Not financial advice."
    )


# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    render_grapher()
