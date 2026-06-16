import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════════════════════════════════
# DATA LAYER
# ══════════════════════════════════════════════════════════════════════════════

def fetch_single_ticker(ticker: str, is_annual: bool) -> Tuple[str, Optional[Dict]]:
    """Fetch all financial data for a single ticker. Returns (ticker, data_dict | None)."""
    try:
        obj = yf.Ticker(ticker)
        if is_annual:
            income   = obj.income_stmt
            balance  = obj.balance_sheet
            cashflow = obj.cashflow
        else:
            income   = obj.quarterly_income_stmt
            balance  = obj.quarterly_balance_sheet
            cashflow = obj.quarterly_cashflow

        if income.empty or balance.empty or cashflow.empty:
            return ticker, None

        try:
            info = obj.info
            if not info or not isinstance(info, dict):
                info = {}
        except Exception:
            info = {}

        return ticker, {
            "income":   income,
            "balance":  balance,
            "cashflow": cashflow,
            "info":     info,
        }
    except Exception:
        return ticker, None


def fetch_all_tickers(tickers: List[str], is_annual: bool, max_workers: int = 8) -> Tuple[Dict, List[str]]:
    """Fetch data for all tickers in parallel. Returns (data_dict, failed_list)."""
    all_data: Dict     = {}
    failed:   List[str] = []

    progress = st.progress(0)
    status   = st.empty()
    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_single_ticker, t, is_annual): t for t in tickers}
        for future in as_completed(futures):
            ticker, data = future.result()
            completed += 1
            progress.progress(completed / len(tickers))
            status.text(f"Loading data… ({completed}/{len(tickers)})")
            if data is not None:
                all_data[ticker] = data
            else:
                failed.append(ticker)

    progress.empty()
    status.empty()
    return all_data, failed


# ══════════════════════════════════════════════════════════════════════════════
# HELPER UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

_FIELD_ALIASES: Dict[str, List[str]] = {
    "Total Revenue":              ["TotalRevenue", "Total Revenues", "Revenue"],
    "Gross Profit":               ["GrossProfit", "Gross Income"],
    "Operating Income":           ["OperatingIncome", "EBIT", "Operating Income Loss"],
    "Net Income":                 ["NetIncome", "Net Income Common Stockholders", "Net Income Loss"],
    "EBITDA":                     ["EBITDA", "Normalized EBITDA"],
    "Operating Cash Flow":        ["OperatingCashFlow", "Total Cash From Operating Activities",
                                   "Cash Flow From Continuing Operating Activities"],
    "Capital Expenditure":        ["CapitalExpenditure", "Capital Expenditures"],
    "Free Cash Flow":             ["FreeCashFlow", "Free Cash Flow"],
    "Total Assets":               ["TotalAssets"],
    "Total Debt":                 ["TotalDebt", "Long Term Debt", "Total Debt And Capital Lease Obligation"],
    "Cash And Cash Equivalents":  ["CashAndCashEquivalents", "Cash",
                                   "Cash Cash Equivalents And Short Term Investments"],
    "Total Stockholder Equity":   ["TotalStockholderEquity", "Stockholders Equity",
                                   "Total Equity Gross Minority Interest"],
    "Diluted Average Shares":     ["DilutedAverageShares", "Average Shares Diluted"],
    "Basic Average Shares":       ["BasicAverageShares", "Ordinary Shares Number"],
    "Current Assets":             ["CurrentAssets", "Total Current Assets"],
    "Current Liabilities":        ["CurrentLiabilities", "Total Current Liabilities"],
    "Total Liabilities":          ["TotalLiabilities", "Total Liabilities Net Minority Interest"],
    "Inventory":                  ["Inventory", "Inventories"],
    "Dividends Paid":             ["DividendsPaid", "Common Stock Dividends Paid",
                                   "Payment Of Dividends"],
}


def safe_get(df: pd.DataFrame, key: str, default=None) -> pd.Series:
    """Return a row from df by primary key or known aliases."""
    if df.empty:
        return pd.Series(dtype=float)
    if key in df.index:
        return df.loc[key]
    for alt in _FIELD_ALIASES.get(key, []):
        if alt in df.index:
            return df.loc[alt]
    if default is not None:
        return pd.Series(default, index=df.columns if not df.empty else [])
    return pd.Series(dtype=float)


def latest(series: pd.Series) -> float:
    """Return most-recent non-null value, or NaN."""
    if series.empty:
        return np.nan
    valid = series.dropna()
    return float(valid.iloc[-1]) if len(valid) > 0 else np.nan


def sdiv(num, den) -> float:
    """Safe division — returns NaN on zero or missing."""
    if pd.isna(num) or pd.isna(den) or den == 0:
        return np.nan
    return float(num) / float(den)


def avg_growth(series: pd.Series) -> float:
    """Average period-over-period growth rate (%) from a time series."""
    valid = series.dropna()
    if len(valid) < 2:
        return np.nan
    return float(valid.pct_change().mean() * 100)


def cagr(series: pd.Series) -> float:
    """Compound annual growth rate (%) across available periods."""
    valid = series.dropna()
    if len(valid) < 2:
        return np.nan
    start, end = float(valid.iloc[0]), float(valid.iloc[-1])
    if start <= 0:
        return np.nan
    n = len(valid) - 1
    return ((end / start) ** (1 / n) - 1) * 100


# ══════════════════════════════════════════════════════════════════════════════
# METRICS ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def calculate_metrics(ticker_data: Dict) -> Dict:
    """Compute comprehensive fundamental metrics for one ticker."""
    income   = ticker_data["income"]
    balance  = ticker_data["balance"]
    cashflow = ticker_data["cashflow"]
    info     = ticker_data["info"]

    # ── Raw series ───────────────────────────────────────────────────────────
    revenue    = safe_get(income,   "Total Revenue")
    gross_pft  = safe_get(income,   "Gross Profit")
    oper_inc   = safe_get(income,   "Operating Income")
    net_inc    = safe_get(income,   "Net Income")
    ebitda     = safe_get(income,   "EBITDA")
    ocf        = safe_get(cashflow, "Operating Cash Flow")
    capex      = safe_get(cashflow, "Capital Expenditure", 0)
    fcf_series = ocf + capex          # capex is typically negative

    total_assets  = safe_get(balance, "Total Assets")
    curr_assets   = safe_get(balance, "Current Assets")
    curr_liab     = safe_get(balance, "Current Liabilities")
    total_liab    = safe_get(balance, "Total Liabilities")
    total_debt    = safe_get(balance, "Total Debt", 0)
    cash          = safe_get(balance, "Cash And Cash Equivalents", 0)
    equity        = safe_get(balance, "Total Stockholder Equity")
    inventory     = safe_get(balance, "Inventory", 0)
    div_paid      = safe_get(cashflow, "Dividends Paid", 0)

    shares = safe_get(income, "Diluted Average Shares")
    if shares.empty:
        shares = safe_get(income, "Basic Average Shares")

    # ── Latest scalar values ─────────────────────────────────────────────────
    rev_l    = latest(revenue)
    gp_l     = latest(gross_pft)
    oi_l     = latest(oper_inc)
    ni_l     = latest(net_inc)
    ebitda_l = latest(ebitda)
    ocf_l    = latest(ocf)
    fcf_l    = latest(fcf_series)
    assets_l = latest(total_assets)
    ca_l     = latest(curr_assets)
    cl_l     = latest(curr_liab)
    tl_l     = latest(total_liab)
    debt_l   = latest(total_debt)
    cash_l   = latest(cash)
    eq_l     = latest(equity)
    inv_l    = latest(inventory)
    sh_l     = latest(shares)
    div_l    = abs(latest(div_paid)) if not pd.isna(latest(div_paid)) else np.nan

    # ── Market / price data ──────────────────────────────────────────────────
    mkt_cap  = info.get("marketCap", np.nan)
    ev       = info.get("enterpriseValue", np.nan)
    price    = info.get("currentPrice",
               info.get("regularMarketPrice",
               info.get("previousClose", np.nan)))

    # Reconstruct EV if missing
    if pd.isna(ev) or ev <= 0:
        if not any(pd.isna(x) for x in [mkt_cap, debt_l, cash_l]):
            ev = mkt_cap + debt_l - cash_l

    # ── Profitability ────────────────────────────────────────────────────────
    gross_margin   = sdiv(gp_l,     rev_l) * 100
    oper_margin    = sdiv(oi_l,     rev_l) * 100
    net_margin     = sdiv(ni_l,     rev_l) * 100
    ebitda_margin  = sdiv(ebitda_l, rev_l) * 100
    roe            = sdiv(ni_l, eq_l)   * 100
    roa            = sdiv(ni_l, assets_l) * 100
    roic           = sdiv(oi_l * 0.65, (eq_l or 0) + (debt_l or 0)) * 100  # approx NOPAT/IC

    # ── Liquidity & solvency ─────────────────────────────────────────────────
    current_ratio  = sdiv(ca_l, cl_l)
    # Accurate quick ratio: exclude inventory from current assets
    quick_ratio    = sdiv(ca_l - (inv_l or 0), cl_l)
    debt_to_equity = sdiv(debt_l, eq_l)
    debt_to_assets = sdiv(debt_l, assets_l)
    equity_ratio   = sdiv(eq_l,   assets_l)
    net_debt       = (debt_l or 0) - (cash_l or 0)
    nd_to_ebitda   = sdiv(net_debt, ebitda_l) if ebitda_l and ebitda_l > 0 else np.nan

    # ── Per-share ────────────────────────────────────────────────────────────
    eps                = sdiv(ni_l,  sh_l)
    revenue_per_share  = sdiv(rev_l, sh_l)
    fcf_per_share      = sdiv(fcf_l, sh_l)
    bvps               = sdiv(eq_l,  sh_l)

    # ── Valuation multiples ──────────────────────────────────────────────────
    # P/E — prefer trailing, fall back to forward, then calculate
    pe = info.get("trailingPE", info.get("forwardPE", np.nan))
    if pd.isna(pe) or pe <= 0 or pe > 1000:
        if not pd.isna(price) and not pd.isna(eps) and eps > 0:
            pe = price / eps

    pb = info.get("priceToBook", np.nan)
    if pd.isna(pb) or pb <= 0:
        if not pd.isna(mkt_cap) and not pd.isna(eq_l) and eq_l > 0:
            pb = mkt_cap / eq_l

    ps = info.get("priceToSalesTrailing12Months", np.nan)
    if pd.isna(ps) or ps <= 0:
        if not pd.isna(mkt_cap) and not pd.isna(rev_l) and rev_l > 0:
            ps = mkt_cap / rev_l

    price_to_fcf = sdiv(mkt_cap, fcf_l) if not pd.isna(mkt_cap) and not pd.isna(fcf_l) and fcf_l > 0 else np.nan
    ev_to_rev    = sdiv(ev, rev_l)    if not pd.isna(ev) and not pd.isna(rev_l)    and rev_l    > 0 else np.nan
    ev_to_ebitda = info.get("enterpriseToEbitda", np.nan)
    if pd.isna(ev_to_ebitda) or ev_to_ebitda <= 0 or ev_to_ebitda > 200:
        ev_to_ebitda = sdiv(ev, ebitda_l) if not pd.isna(ev) and not pd.isna(ebitda_l) and ebitda_l > 0 else np.nan

    # ── Growth ───────────────────────────────────────────────────────────────
    rev_growth   = avg_growth(revenue)
    rev_cagr     = cagr(revenue)
    ni_growth    = avg_growth(net_inc)
    ni_cagr      = cagr(net_inc)
    fcf_growth   = avg_growth(fcf_series)
    eps_series   = net_inc / shares if not shares.empty and not net_inc.empty else pd.Series(dtype=float)
    eps_growth   = avg_growth(eps_series)

    # PEG
    peg = np.nan
    if not pd.isna(pe) and not pd.isna(rev_growth) and rev_growth > 0:
        peg = pe / rev_growth

    # ── Cash flow quality ────────────────────────────────────────────────────
    fcf_to_ni  = sdiv(fcf_l, ni_l)  if not pd.isna(ni_l)  and ni_l  > 0 else np.nan
    ocf_to_ni  = sdiv(ocf_l, ni_l)  if not pd.isna(ni_l)  and ni_l  > 0 else np.nan
    fcf_margin = sdiv(fcf_l, rev_l) * 100

    # ── Efficiency ───────────────────────────────────────────────────────────
    asset_turnover = sdiv(rev_l, assets_l)

    # ── Dividends ────────────────────────────────────────────────────────────
    div_yield   = info.get("dividendYield", np.nan)
    if not pd.isna(div_yield):
        div_yield *= 100
    payout_ratio = sdiv(div_l, ni_l) * 100 if not pd.isna(div_l) and not pd.isna(ni_l) and ni_l > 0 else np.nan
    div_coverage = sdiv(ni_l, div_l)  if not pd.isna(div_l) and div_l > 0 else np.nan
    div_5yr_growth = avg_growth(abs(div_paid) if not div_paid.empty else pd.Series(dtype=float))

    # ── Simple 2-stage DCF estimate ──────────────────────────────────────────
    dcf_intrinsic = np.nan
    if not any(pd.isna(x) for x in [fcf_l, sh_l, rev_growth]) and fcf_l > 0 and sh_l > 0:
        try:
            wacc           = 0.09
            terminal_g     = 0.03
            stage1_g       = min(max(rev_growth / 100, -0.1), 0.25)   # cap at ±25%
            stage2_g       = min(stage1_g * 0.5, 0.08)                # fade
            pv             = 0.0
            cf             = fcf_l
            for yr in range(1, 6):
                cf  *= (1 + stage1_g)
                pv  += cf / (1 + wacc) ** yr
            for yr in range(6, 11):
                cf  *= (1 + stage2_g)
                pv  += cf / (1 + wacc) ** yr
            terminal_value = cf * (1 + terminal_g) / (wacc - terminal_g)
            pv            += terminal_value / (1 + wacc) ** 10
            dcf_intrinsic  = pv / sh_l
        except Exception:
            dcf_intrinsic = np.nan

    # ── Time series for trend charts ─────────────────────────────────────────
    ts_gross_margin = (gross_pft / revenue * 100).replace([np.inf, -np.inf], np.nan) \
        if not revenue.empty and not gross_pft.empty else pd.Series(dtype=float)
    ts_oper_margin  = (oper_inc  / revenue * 100).replace([np.inf, -np.inf], np.nan) \
        if not revenue.empty and not oper_inc.empty  else pd.Series(dtype=float)
    ts_net_margin   = (net_inc   / revenue * 100).replace([np.inf, -np.inf], np.nan) \
        if not revenue.empty and not net_inc.empty   else pd.Series(dtype=float)
    ts_roe          = (net_inc   / equity  * 100).replace([np.inf, -np.inf], np.nan) \
        if not equity.empty   and not net_inc.empty  else pd.Series(dtype=float)

    return {
        # Core financials
        "Revenue":          rev_l,
        "Gross Profit":     gp_l,
        "Operating Income": oi_l,
        "Net Income":       ni_l,
        "EBITDA":           ebitda_l,
        "Operating CF":     ocf_l,
        "Free Cash Flow":   fcf_l,
        "Total Assets":     assets_l,
        "Total Debt":       debt_l,
        "Net Debt":         net_debt,
        "Cash":             cash_l,
        "Equity":           eq_l,
        "Market Cap":       mkt_cap,
        # Profitability
        "Gross Margin %":     gross_margin,
        "Operating Margin %": oper_margin,
        "Net Margin %":       net_margin,
        "EBITDA Margin %":    ebitda_margin,
        "ROE %":              roe,
        "ROA %":              roa,
        "ROIC %":             roic,
        # Liquidity & solvency
        "Current Ratio":   current_ratio,
        "Quick Ratio":     quick_ratio,
        "Debt/Equity":     debt_to_equity,
        "Debt/Assets":     debt_to_assets,
        "Equity Ratio":    equity_ratio,
        "Net Debt/EBITDA": nd_to_ebitda,
        # Per share
        "EPS":             eps,
        "Revenue/Share":   revenue_per_share,
        "FCF/Share":       fcf_per_share,
        "Book Value/Share": bvps,
        # Valuation
        "P/E":         pe,
        "P/B":         pb,
        "P/S":         ps,
        "Price/FCF":   price_to_fcf,
        "EV/Revenue":  ev_to_rev,
        "EV/EBITDA":   ev_to_ebitda,
        "PEG":         peg,
        "DCF Estimate": dcf_intrinsic,
        # Efficiency
        "Asset Turnover": asset_turnover,
        # Cash flow quality
        "FCF/NI":      fcf_to_ni,
        "OCF/NI":      ocf_to_ni,
        "FCF Margin %": fcf_margin,
        # Growth
        "Revenue Growth %": rev_growth,
        "Revenue CAGR %":   rev_cagr,
        "NI Growth %":      ni_growth,
        "NI CAGR %":        ni_cagr,
        "FCF Growth %":     fcf_growth,
        "EPS Growth %":     eps_growth,
        # Dividends
        "Dividend Yield %":     div_yield,
        "Payout Ratio %":       payout_ratio,
        "Dividend Coverage":    div_coverage,
        "Dividend Growth 5Y %": div_5yr_growth,
        # Time series (prefixed with _ — excluded from export/scoring)
        "_revenue_ts":        revenue,
        "_net_income_ts":     net_inc,
        "_ocf_ts":            ocf,
        "_fcf_ts":            fcf_series,
        "_gross_margin_ts":   ts_gross_margin,
        "_operating_margin_ts": ts_oper_margin,
        "_net_margin_ts":     ts_net_margin,
        "_roe_ts":            ts_roe,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SCORING ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def calculate_fundamental_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Score stocks 0-100 across profitability, growth, financial health, and
    valuation. Weights are normalised to sum to 1.0 before use.
    """
    CRITERIA = {
        # Profitability
        "ROE %":              {"raw_weight": 0.10, "higher_better": True},
        "Net Margin %":       {"raw_weight": 0.10, "higher_better": True},
        "Operating Margin %": {"raw_weight": 0.10, "higher_better": True},
        "ROIC %":             {"raw_weight": 0.10, "higher_better": True},
        # Growth
        "Revenue Growth %":   {"raw_weight": 0.10, "higher_better": True},
        "NI Growth %":        {"raw_weight": 0.08, "higher_better": True},
        "EPS Growth %":       {"raw_weight": 0.07, "higher_better": True},
        # Financial health
        "Current Ratio":      {"raw_weight": 0.05, "higher_better": True},
        "Debt/Equity":        {"raw_weight": 0.08, "higher_better": False,
                               "clip_high": 10},
        "FCF/NI":             {"raw_weight": 0.07, "higher_better": True},
        # Valuation (lower = better; optional — may be NaN for many tickers)
        "P/E":                {"raw_weight": 0.05, "higher_better": False,
                               "clip_high": 100,  "optional": True},
        "PEG":                {"raw_weight": 0.05, "higher_better": False,
                               "clip_high": 5,    "optional": True},
        "EV/EBITDA":          {"raw_weight": 0.05, "higher_better": False,
                               "clip_high": 50,   "optional": True},
    }

    # Drop optional criteria where fewer than 2 tickers have valid data
    active = {}
    for metric, params in CRITERIA.items():
        if metric not in df.columns:
            continue
        n_valid = df[metric].replace([np.inf, -np.inf], np.nan).notna().sum()
        if params.get("optional") and n_valid < 2:
            continue
        active[metric] = params

    # Normalise weights so they always sum to 1.0
    total_w = sum(p["raw_weight"] for p in active.values())
    for metric in active:
        active[metric]["weight"] = active[metric]["raw_weight"] / total_w

    scores = pd.DataFrame(index=df.index)

    for metric, params in active.items():
        values = df[metric].replace([np.inf, -np.inf], np.nan).copy()

        # Clip extreme outliers before normalising
        if not params["higher_better"] and "clip_high" in params:
            values = values.clip(upper=params["clip_high"])

        if values.notna().sum() < 1:
            scores[metric] = 0.0
            continue

        mn, mx = values.min(), values.max()
        if mn == mx:
            norm = pd.Series(50.0, index=values.index)
        elif params["higher_better"]:
            norm = (values - mn) / (mx - mn) * 100
        else:
            norm = (mx - values) / (mx - mn) * 100

        # Tickers with NaN get the median of the normalised column
        norm = norm.fillna(norm.median())
        scores[metric] = norm * params["weight"]

    scores["Total Score"] = scores.sum(axis=1)

    def grade(s: float) -> str:
        if pd.isna(s): return "N/A"
        thresholds = [(80, "A+"), (75, "A"), (70, "A-"), (65, "B+"),
                      (60, "B"), (55, "B-"), (50, "C+"), (45, "C"),
                      (40, "C-"), (35, "D+"), (30, "D")]
        for threshold, letter in thresholds:
            if s >= threshold:
                return letter
        return "F"

    scores["Grade"] = scores["Total Score"].apply(grade)
    scores["Rank"]  = scores["Total Score"].rank(ascending=False, method="min").astype(int)
    return scores


# ══════════════════════════════════════════════════════════════════════════════
# CHART HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def bar_chart(data: pd.Series, ylabel: str, title: str = "", color: str = "steelblue",
              fmt: str = "{:.2f}", hline: Optional[float] = None) -> go.Figure:
    clean = data.replace([np.inf, -np.inf], np.nan).dropna()
    fig = go.Figure(go.Bar(
        x=clean.index, y=clean.values,
        marker_color=color,
        text=[fmt.format(v) for v in clean.values],
        textposition="outside",
    ))
    fig.update_layout(yaxis_title=ylabel, title=title, height=380, showlegend=False)
    if hline is not None:
        fig.add_hline(y=hline, line_dash="dash", line_color="gray",
                      annotation_text=f"Ref: {hline}")
    return fig


def grouped_bar(df: pd.DataFrame, cols: List[str], ylabel: str, title: str = "",
                hline: Optional[float] = None) -> go.Figure:
    fig = go.Figure()
    for col in cols:
        if col not in df.columns:
            continue
        valid = df[col].replace([np.inf, -np.inf], np.nan).dropna()
        if not valid.empty:
            fig.add_trace(go.Bar(name=col, x=valid.index, y=valid.values))
    fig.update_layout(barmode="group", yaxis_title=ylabel, title=title, height=400)
    if hline is not None:
        fig.add_hline(y=hline, line_dash="dash", line_color="green",
                      annotation_text=f"Ref: {hline}")
    return fig


def trend_chart(data_dict: Dict[str, pd.Series], ylabel: str, title: str = "") -> go.Figure:
    fig = go.Figure()
    for ticker, series in data_dict.items():
        valid = series.dropna()
        if not valid.empty:
            fig.add_trace(go.Scatter(
                x=valid.index, y=valid.values, name=ticker,
                mode="lines+markers", line=dict(width=2), marker=dict(size=6),
            ))
    fig.update_layout(
        title=title, yaxis_title=ylabel, height=400,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def color_valuation(val):
    """Cell-level colour for valuation tables (used with Styler.map)."""
    try:
        v = float(val)
        if v < 0:    return "background-color: #ffcccc"
        if v > 100:  return "background-color: #ffcccc"
        if v < 15:   return "background-color: #ccffcc"
        if v < 25:   return "background-color: #ffffcc"
        return "background-color: #ffddcc"
    except Exception:
        return ""


# ══════════════════════════════════════════════════════════════════════════════
# MAIN RENDER FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def render_fundamental_comparison(tickers: List[str] = None) -> None:
    """
    Streamlit page: Advanced Fundamental Analysis & Comparison.
    Call from your app with an optional pre-set ticker list.
    """
    st.markdown("# 📊 Advanced Fundamental Analysis & Comparison")
    st.markdown("Compare financial metrics, analyse trends, and rank stocks by fundamental strength.")

    # ── Configuration ─────────────────────────────────────────────────────────
    with st.expander("⚙️ Configuration", expanded=True):
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            default = ", ".join(tickers) if tickers else "AAPL, MSFT, GOOGL, AMZN, META"
            ticker_input = st.text_input(
                "Stock tickers (comma-separated)", value=default,
                help="Enter 2–15 ticker symbols"
            )
            tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
        with col2:
            frequency = st.selectbox("Data frequency", ["Annual", "Quarterly"])
        with col3:
            analyze = st.button("🚀 Analyze", type="primary", use_container_width=True)

    if len(tickers) < 2:
        st.info("📌 Enter at least 2 tickers to compare fundamental metrics.")
        return
    if len(tickers) > 15:
        st.warning("⚠️ Maximum 15 tickers. Using first 15.")
        tickers = tickers[:15]

    if not analyze:
        st.info("👆 Click 'Analyze' to load and compare fundamental data.")
        return

    # ── Data fetching (parallel + session_state cache) ────────────────────────
    is_annual  = frequency == "Annual"
    cache_key  = f"fund_{'_'.join(sorted(tickers))}_{frequency}"

    if cache_key not in st.session_state:
        with st.spinner(f"📥 Fetching data for {len(tickers)} stocks in parallel…"):
            data, failed = fetch_all_tickers(tickers, is_annual)
        st.session_state[cache_key] = (data, failed)
    else:
        st.info("ℹ️ Using cached data. Click Analyze again to refresh.")
        data, failed = st.session_state[cache_key]

    if not data:
        st.error("❌ No data retrieved. Check ticker symbols and try again.")
        return

    tickers = list(data.keys())

    c1, c2 = st.columns([3, 1])
    with c1:
        st.success(f"✅ Loaded **{len(tickers)} stocks**: {', '.join(tickers)}")
    with c2:
        if failed:
            st.error(f"❌ Failed: {', '.join(failed)}")

    # ── Metric calculation ─────────────────────────────────────────────────────
    with st.spinner("📊 Calculating metrics…"):
        metrics = {t: calculate_metrics(data[t]) for t in tickers}
        df = pd.DataFrame(metrics).T

    scores_df = calculate_fundamental_score(df)

    # Columns safe for display / export (exclude internal time series)
    export_cols = [c for c in df.columns if not c.startswith("_")]

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_labels = ["🏆 Rankings", "📊 Overview", "💰 Profitability",
                  "💵 Cash Flow", "🏦 Financial Health", "📈 Valuation",
                  "💸 Dividends", "📉 Trends", "📥 Export"]
    tabs = st.tabs(tab_labels)

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 1 — RANKINGS
    # ═════════════════════════════════════════════════════════════════════════
    with tabs[0]:
        st.markdown("## 🏆 Fundamental Strength Rankings")
        st.caption("Scoring based on profitability, growth, financial health, and valuation. "
                   "Weights normalised to sum to 100%.")

        ranking = pd.DataFrame({
            "Rank":               scores_df["Rank"],
            "Score":              scores_df["Total Score"],
            "Grade":              scores_df["Grade"],
            "ROE %":              df["ROE %"],
            "Net Margin %":       df["Net Margin %"],
            "Revenue Growth %":   df["Revenue Growth %"],
            "Debt/Equity":        df["Debt/Equity"],
            "P/E":                df["P/E"],
            "Market Cap":         df["Market Cap"],
        }).sort_values("Rank")

        def highlight_rank(row):
            colours = {1: "#FFD700", 2: "#C0C0C0", 3: "#CD7F32"}
            bg = colours.get(row["Rank"], "")
            return [f"background-color: {bg}" if bg else ""] * len(row)

        st.dataframe(
            ranking.style
            .format({
                "Score":            "{:.1f}",
                "ROE %":            "{:.2f}",
                "Net Margin %":     "{:.2f}",
                "Revenue Growth %": "{:.2f}",
                "Debt/Equity":      "{:.2f}",
                "P/E":              "{:.2f}",
                "Market Cap":       "${:,.0f}",
            }, na_rep="-")
            .background_gradient(cmap="RdYlGn", subset=["Score"], vmin=0, vmax=100)
            .apply(highlight_rank, axis=1),
            use_container_width=True,
            height=min(600, (len(ranking) + 1) * 35 + 3),
        )

        # Score bar + grade pie
        c1, c2 = st.columns(2)
        with c1:
            colours = ["gold" if r == 1 else "silver" if r == 2 else "#CD7F32" if r == 3
                       else "steelblue" for r in ranking["Rank"]]
            fig = go.Figure(go.Bar(
                x=ranking.index, y=ranking["Score"],
                marker_color=colours,
                text=ranking["Grade"], textposition="outside",
                hovertemplate="<b>%{x}</b><br>Score: %{y:.1f}<br>Grade: %{text}<extra></extra>",
            ))
            fig.update_layout(yaxis_title="Score", yaxis_range=[0, 110], height=380,
                               showlegend=False, title="Fundamental scores")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            gc = ranking["Grade"].value_counts()
            fig = go.Figure(go.Pie(
                labels=gc.index, values=gc.values, hole=0.4,
                marker_colors=px.colors.diverging.RdYlGn[::-1],
            ))
            fig.update_layout(height=380, title="Grade distribution")
            st.plotly_chart(fig, use_container_width=True)

        # Radar — top 5
        st.markdown("---")
        st.markdown("### Category breakdown (top 5)")

        cat_raw = pd.DataFrame({
            "Profitability": df[["ROE %", "Net Margin %", "Operating Margin %", "ROIC %"]].mean(axis=1),
            "Growth":        df[["Revenue Growth %", "NI Growth %", "EPS Growth %"]].mean(axis=1),
            "Health":        df["Current Ratio"] * 10 - df["Debt/Equity"].fillna(0) * 5,
            "Valuation":     50 - df[["P/E", "PEG", "EV/EBITDA"]].mean(axis=1),
        })
        for col in cat_raw.columns:
            mn, mx = cat_raw[col].min(), cat_raw[col].max()
            if mn != mx:
                cat_raw[col] = (cat_raw[col] - mn) / (mx - mn) * 100

        fig = go.Figure()
        for ticker in ranking.head(5).index:
            if ticker in cat_raw.index:
                vals = cat_raw.loc[ticker].tolist()
                vals.append(vals[0])
                fig.add_trace(go.Scatterpolar(
                    r=vals,
                    theta=list(cat_raw.columns) + [cat_raw.columns[0]],
                    name=ticker, fill="toself",
                ))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                          height=480, title="Top 5 — category radar")
        st.plotly_chart(fig, use_container_width=True)

        # Category leaders
        st.markdown("---")
        st.markdown("### Category leaders")
        cats = {
            "💰 Profitability":    ("ROE %",            False),
            "📈 Growth":           ("Revenue Growth %", False),
            "🏦 Financial Health": ("Current Ratio",    False),
            "💎 Valuation":        ("PEG",              True),   # lower is better
        }
        cols = st.columns(4)
        for col, (label, (metric, lower_better)) in zip(cols, cats.items()):
            with col:
                valid = df[metric].replace([np.inf, -np.inf], np.nan).dropna()
                if not valid.empty:
                    best = valid.idxmin() if lower_better else valid.idxmax()
                    st.metric(label, best, f"{valid[best]:.2f}")

        # Score component breakdown
        st.markdown("---")
        with st.expander("🔍 Score component detail"):
            comp_cols = [c for c in scores_df.columns if c not in ("Total Score", "Grade", "Rank")]
            comp_disp = scores_df[comp_cols + ["Total Score"]].sort_values("Total Score", ascending=False)
            st.dataframe(
                comp_disp.style.format("{:.2f}").background_gradient(cmap="RdYlGn", vmin=0),
                use_container_width=True,
            )

        # Summary text
        st.markdown("---")
        st.markdown("### Analysis summary")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**✅ Strongest fundamentals**")
            for i, (ticker, row) in enumerate(ranking.head(3).iterrows(), 1):
                st.markdown(f"{i}. **{ticker}** — Grade {row['Grade']} (Score: {row['Score']:.1f})")
                s = []
                med = ranking["ROE %"].median()
                if not pd.isna(row["ROE %"]) and row["ROE %"] > med: s.append("Strong ROE")
                med = ranking["Revenue Growth %"].median()
                if not pd.isna(row["Revenue Growth %"]) and row["Revenue Growth %"] > med: s.append("High growth")
                med = ranking["Debt/Equity"].median()
                if not pd.isna(row["Debt/Equity"]) and row["Debt/Equity"] < med: s.append("Low debt")
                if s: st.caption("Key strengths: " + ", ".join(s))
        with c2:
            st.markdown("**⚠️ Weaker fundamentals**")
            bottom = ranking.tail(3)
            offset = len(ranking) - len(bottom)
            for i, (ticker, row) in enumerate(bottom.iterrows(), 1):
                st.markdown(f"{offset + i}. **{ticker}** — Grade {row['Grade']} (Score: {row['Score']:.1f})")
                c = []
                med = ranking["ROE %"].median()
                if not pd.isna(row["ROE %"]) and row["ROE %"] < med: c.append("Low ROE")
                if not pd.isna(row["Revenue Growth %"]) and row["Revenue Growth %"] < 0: c.append("Negative growth")
                med = ranking["Debt/Equity"].median()
                if not pd.isna(row["Debt/Equity"]) and row["Debt/Equity"] > med: c.append("High debt")
                if c: st.caption("Concerns: " + ", ".join(c))

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 2 — OVERVIEW
    # ═════════════════════════════════════════════════════════════════════════
    with tabs[1]:
        st.markdown("### Key metrics overview")
        ov_cols = ["Market Cap", "Revenue", "Net Income", "Free Cash Flow",
                   "EPS", "P/E", "P/B", "ROE %", "Net Margin %", "Revenue Growth %"]
        ov = df[ov_cols].copy()
        st.dataframe(
            ov.style.format({
                "Market Cap":       "${:,.0f}",
                "Revenue":          "${:,.0f}",
                "Net Income":       "${:,.0f}",
                "Free Cash Flow":   "${:,.0f}",
                "EPS":              "${:.2f}",
                "P/E":              "{:.2f}",
                "P/B":              "{:.2f}",
                "ROE %":            "{:.2f}",
                "Net Margin %":     "{:.2f}",
                "Revenue Growth %": "{:.2f}",
            }, na_rep="-")
            .background_gradient(cmap="RdYlGn", subset=["ROE %", "Net Margin %", "Revenue Growth %"]),
            use_container_width=True,
        )

        c1, c2 = st.columns(2)
        with c1:
            mcap = ov["Market Cap"].dropna()
            if not mcap.empty:
                st.plotly_chart(
                    bar_chart(mcap, "Market Cap ($)", "Market capitalisation",
                              fmt="${:,.0f}"),
                    use_container_width=True,
                )
        with c2:
            rev = ov["Revenue"].dropna()
            if not rev.empty:
                fig = px.pie(values=rev.values, names=rev.index, title="Revenue share")
                fig.update_layout(height=380)
                st.plotly_chart(fig, use_container_width=True)

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 3 — PROFITABILITY
    # ═════════════════════════════════════════════════════════════════════════
    with tabs[2]:
        st.markdown("### Profitability metrics")
        prof_cols = ["Gross Margin %", "Operating Margin %", "Net Margin %",
                     "EBITDA Margin %", "ROE %", "ROA %", "ROIC %", "Revenue/Share", "EPS"]
        prof = df[prof_cols].copy()
        st.dataframe(
            prof.style.format({c: "{:.2f}" for c in prof_cols[:7]}
                              | {"Revenue/Share": "${:.2f}", "EPS": "${:.2f}"},
                              na_rep="-")
            .background_gradient(cmap="RdYlGn"),
            use_container_width=True,
        )

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                grouped_bar(prof, ["Gross Margin %", "Operating Margin %", "Net Margin %"],
                            "Margin %", "Margin comparison"),
                use_container_width=True,
            )
        with c2:
            st.plotly_chart(
                grouped_bar(prof, ["ROE %", "ROA %", "ROIC %"],
                            "Return %", "Return metrics"),
                use_container_width=True,
            )

        scatter = prof[["Net Margin %", "ROE %"]].dropna()
        if len(scatter) > 1:
            fig = px.scatter(scatter, x="Net Margin %", y="ROE %",
                             text=scatter.index, size=[100] * len(scatter),
                             height=420, title="Profitability positioning")
            fig.update_traces(textposition="top center")
            fig.add_hline(y=scatter["ROE %"].median(), line_dash="dash", line_color="gray")
            fig.add_vline(x=scatter["Net Margin %"].median(), line_dash="dash", line_color="gray")
            st.plotly_chart(fig, use_container_width=True)

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 4 — CASH FLOW
    # ═════════════════════════════════════════════════════════════════════════
    with tabs[3]:
        st.markdown("### Cash flow analysis")
        cf_cols = ["Operating CF", "Free Cash Flow", "FCF/Share",
                   "FCF Margin %", "FCF/NI", "OCF/NI", "Net Income", "Revenue"]
        cf = df[cf_cols].copy()
        st.dataframe(
            cf.style.format({
                "Operating CF":   "${:,.0f}",
                "Free Cash Flow": "${:,.0f}",
                "FCF/Share":      "${:.2f}",
                "FCF Margin %":   "{:.2f}",
                "FCF/NI":         "{:.2f}",
                "OCF/NI":         "{:.2f}",
                "Net Income":     "${:,.0f}",
                "Revenue":        "${:,.0f}",
            }, na_rep="-")
            .background_gradient(cmap="RdYlGn", subset=["FCF Margin %", "FCF/NI", "OCF/NI"]),
            use_container_width=True,
        )

        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            ni  = cf["Net Income"].dropna()
            fcf = cf["Free Cash Flow"].dropna()
            if not ni.empty:
                fig.add_trace(go.Bar(name="Net Income", x=ni.index, y=ni.values, marker_color="lightcoral"))
            if not fcf.empty:
                fig.add_trace(go.Bar(name="Free Cash Flow", x=fcf.index, y=fcf.values, marker_color="lightgreen"))
            fig.update_layout(barmode="group", yaxis_title="Amount ($)", height=400,
                               title="FCF vs net income")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.plotly_chart(
                bar_chart(cf["FCF Margin %"].dropna(), "FCF Margin %",
                          "FCF margin", "steelblue", fmt="{:.1f}%"),
                use_container_width=True,
            )

        st.plotly_chart(
            grouped_bar(cf, ["FCF/NI", "OCF/NI"], "Ratio",
                        "Cash flow quality (>1.0 = strong)", hline=1.0),
            use_container_width=True,
        )

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 5 — FINANCIAL HEALTH
    # ═════════════════════════════════════════════════════════════════════════
    with tabs[4]:
        st.markdown("### Financial health & solvency")
        health_cols = ["Total Assets", "Total Debt", "Net Debt", "Cash", "Equity",
                       "Current Ratio", "Quick Ratio", "Debt/Equity",
                       "Debt/Assets", "Equity Ratio", "Net Debt/EBITDA"]
        health = df[health_cols].copy()
        st.dataframe(
            health.style.format({
                "Total Assets":    "${:,.0f}",
                "Total Debt":      "${:,.0f}",
                "Net Debt":        "${:,.0f}",
                "Cash":            "${:,.0f}",
                "Equity":          "${:,.0f}",
                "Current Ratio":   "{:.2f}",
                "Quick Ratio":     "{:.2f}",
                "Debt/Equity":     "{:.2f}",
                "Debt/Assets":     "{:.2f}",
                "Equity Ratio":    "{:.2f}",
                "Net Debt/EBITDA": "{:.2f}",
            }, na_rep="-")
            .background_gradient(cmap="RdYlGn",   subset=["Current Ratio", "Quick Ratio", "Equity Ratio"])
            .background_gradient(cmap="RdYlGn_r", subset=["Debt/Equity", "Debt/Assets"]),
            use_container_width=True,
        )

        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            c = health["Cash"].dropna()
            d = health["Total Debt"].dropna()
            if not c.empty:
                fig.add_trace(go.Bar(name="Cash", x=c.index, y=c.values, marker_color="lightgreen"))
            if not d.empty:
                fig.add_trace(go.Bar(name="Total Debt", x=d.index, y=d.values, marker_color="lightcoral"))
            fig.update_layout(barmode="group", yaxis_title="Amount ($)",
                               height=400, title="Debt vs cash")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.plotly_chart(
                grouped_bar(health, ["Current Ratio", "Quick Ratio"],
                            "Ratio", "Liquidity ratios", hline=1.0),
                use_container_width=True,
            )

        st.plotly_chart(
            bar_chart(health["Debt/Equity"].dropna(), "Debt / Equity",
                      "Debt/Equity (lower = safer)", "indianred", fmt="{:.2f}"),
            use_container_width=True,
        )

        nd_ebitda = health["Net Debt/EBITDA"].dropna()
        if not nd_ebitda.empty:
            st.plotly_chart(
                bar_chart(nd_ebitda, "Net Debt / EBITDA",
                          "Net Debt/EBITDA (<3× generally healthy)", "darkorange",
                          fmt="{:.2f}", hline=3.0),
                use_container_width=True,
            )

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 6 — VALUATION
    # ═════════════════════════════════════════════════════════════════════════
    with tabs[5]:
        st.markdown("### Valuation metrics")
        val_cols = ["P/E", "P/B", "P/S", "Price/FCF",
                    "EV/Revenue", "EV/EBITDA", "PEG",
                    "DCF Estimate", "Market Cap", "Revenue Growth %"]
        val = df[val_cols].copy()

        primary_metrics = ["P/E", "P/B", "P/S", "EV/EBITDA"]
        valid_cnt = val[primary_metrics].replace([np.inf, -np.inf], np.nan).notna().sum().sum()
        if valid_cnt == 0:
            st.warning("⚠️ No valuation multiples available from yfinance. "
                       "Tickers may have negative earnings or temporary data gaps.")
        else:
            st.caption(f"📊 {valid_cnt} / {len(val) * len(primary_metrics)} valuation data points available")

        st.dataframe(
            val.style.format({
                "P/E":              "{:.2f}",
                "P/B":              "{:.2f}",
                "P/S":              "{:.2f}",
                "Price/FCF":        "{:.2f}",
                "EV/Revenue":       "{:.2f}",
                "EV/EBITDA":        "{:.2f}",
                "PEG":              "{:.2f}",
                "DCF Estimate":     "${:.2f}",
                "Market Cap":       "${:,.0f}",
                "Revenue Growth %": "{:.2f}",
            }, na_rep="-")
            .map(color_valuation, subset=["P/E", "P/B", "P/S", "EV/EBITDA", "PEG"]),
            use_container_width=True,
        )
        st.caption("💡 Green < 15, yellow 15–25, orange > 25, red < 0 or > 100. "
                   "Lower multiples generally indicate cheaper valuation.")

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                grouped_bar(val, ["P/E", "P/B", "P/S", "EV/EBITDA"],
                            "Multiple", "Valuation multiples"),
                use_container_width=True,
            )
        with c2:
            peg = val["PEG"].replace([np.inf, -np.inf], np.nan).dropna()
            if not peg.empty:
                st.plotly_chart(
                    bar_chart(peg, "PEG Ratio",
                              "PEG ratio (<1.0 = potentially undervalued)",
                              "purple", fmt="{:.2f}", hline=1.0),
                    use_container_width=True,
                )

        # DCF vs current price scatter
        dcf_vals = val["DCF Estimate"].dropna()
        if not dcf_vals.empty:
            st.markdown("**DCF intrinsic value estimate vs market price**")
            st.caption("⚠️ Simple 2-stage DCF using historical FCF growth. "
                       "Treat as directional, not precise.")
            prices = {t: data[t]["info"].get("currentPrice",
                          data[t]["info"].get("regularMarketPrice",
                          data[t]["info"].get("previousClose", np.nan)))
                      for t in tickers}
            price_s = pd.Series(prices, name="Current Price").dropna()
            dcf_compare = pd.DataFrame({
                "DCF Estimate":  dcf_vals,
                "Current Price": price_s,
            }).dropna()
            if len(dcf_compare) > 0:
                fig = go.Figure()
                fig.add_trace(go.Bar(name="DCF Estimate", x=dcf_compare.index,
                                     y=dcf_compare["DCF Estimate"], marker_color="steelblue"))
                fig.add_trace(go.Bar(name="Current Price", x=dcf_compare.index,
                                     y=dcf_compare["Current Price"], marker_color="lightcoral"))
                fig.update_layout(barmode="group", yaxis_title="Price ($)", height=400,
                                   title="DCF estimate vs current price")
                st.plotly_chart(fig, use_container_width=True)

        # P/E vs growth scatter
        pe_g = val[["P/E", "Revenue Growth %"]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(pe_g) > 1:
            fig = px.scatter(pe_g, x="Revenue Growth %", y="P/E",
                             text=pe_g.index, size=[100] * len(pe_g),
                             height=420, title="Growth vs valuation")
            fig.update_traces(textposition="top center")
            fig.add_hline(y=pe_g["P/E"].median(), line_dash="dash", line_color="gray")
            fig.add_vline(x=pe_g["Revenue Growth %"].median(), line_dash="dash", line_color="gray")
            st.plotly_chart(fig, use_container_width=True)

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 7 — DIVIDENDS
    # ═════════════════════════════════════════════════════════════════════════
    with tabs[6]:
        st.markdown("### Dividend analysis")
        div_cols = ["Dividend Yield %", "Payout Ratio %",
                    "Dividend Coverage", "Dividend Growth 5Y %",
                    "Free Cash Flow", "Net Income"]
        div = df[div_cols].copy()

        payers = div["Dividend Yield %"].dropna()
        if payers.empty:
            st.info("📌 None of the selected stocks appear to pay dividends "
                    "(or dividend data is unavailable).")
        else:
            st.caption(f"📊 {len(payers)} of {len(tickers)} stocks pay dividends.")
            st.dataframe(
                div.style.format({
                    "Dividend Yield %":     "{:.2f}",
                    "Payout Ratio %":       "{:.2f}",
                    "Dividend Coverage":    "{:.2f}",
                    "Dividend Growth 5Y %": "{:.2f}",
                    "Free Cash Flow":       "${:,.0f}",
                    "Net Income":           "${:,.0f}",
                }, na_rep="-")
                .background_gradient(cmap="RdYlGn", subset=["Dividend Yield %"]),
                use_container_width=True,
            )

            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(
                    bar_chart(payers, "Dividend Yield %",
                              "Dividend yield", "mediumseagreen", fmt="{:.2f}%"),
                    use_container_width=True,
                )
            with c2:
                pr = div["Payout Ratio %"].dropna()
                if not pr.empty:
                    st.plotly_chart(
                        bar_chart(pr, "Payout Ratio %",
                                  "Payout ratio (<60% sustainable)", "goldenrod",
                                  fmt="{:.1f}%", hline=60.0),
                        use_container_width=True,
                    )

            cov = div["Dividend Coverage"].dropna()
            if not cov.empty:
                st.plotly_chart(
                    bar_chart(cov, "Coverage ratio",
                              "Dividend coverage (>2× comfortable)", "steelblue",
                              fmt="{:.2f}×", hline=2.0),
                    use_container_width=True,
                )

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 8 — TRENDS
    # ═════════════════════════════════════════════════════════════════════════
    with tabs[7]:
        st.markdown("### Historical trends")

        def ts_dict(key: str) -> Dict[str, pd.Series]:
            return {t: df.loc[t, key] for t in tickers
                    if not df.loc[t, key].empty}

        st.plotly_chart(
            trend_chart(ts_dict("_revenue_ts"), "Revenue ($)", "Revenue over time"),
            use_container_width=True,
        )
        st.plotly_chart(
            trend_chart(ts_dict("_net_income_ts"), "Net Income ($)", "Net income over time"),
            use_container_width=True,
        )

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                trend_chart(ts_dict("_net_margin_ts"), "Net Margin %", "Net margin % trend"),
                use_container_width=True,
            )
        with c2:
            st.plotly_chart(
                trend_chart(ts_dict("_roe_ts"), "ROE %", "ROE % trend"),
                use_container_width=True,
            )

        st.plotly_chart(
            trend_chart(ts_dict("_fcf_ts"), "Free Cash Flow ($)", "Free cash flow over time"),
            use_container_width=True,
        )

        growth_cols = ["Revenue Growth %", "NI Growth %", "FCF Growth %", "EPS Growth %"]
        growth_df = df[growth_cols].replace([np.inf, -np.inf], np.nan).dropna(how="all")
        if not growth_df.empty:
            st.plotly_chart(
                grouped_bar(growth_df, growth_cols, "Growth %",
                            "Average period-over-period growth rates", hline=0),
                use_container_width=True,
            )

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 9 — EXPORT
    # ═════════════════════════════════════════════════════════════════════════
    with tabs[8]:
        st.markdown("### Export data")

        export_df   = df[export_cols].copy()
        full_export = pd.concat([scores_df[["Rank", "Total Score", "Grade"]], export_df], axis=1)
        key_export  = export_df[["Market Cap", "Revenue", "Net Income", "ROE %",
                                 "P/E", "Revenue Growth %", "Debt/Equity"]].copy()

        date_str = datetime.now().strftime("%Y%m%d")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**📊 Complete analysis**")
            st.download_button("Download full dataset (CSV)", full_export.to_csv(),
                               f"fundamental_analysis_{date_str}.csv", "text/csv",
                               use_container_width=True)
        with c2:
            st.markdown("**🏆 Rankings only**")
            st.download_button("Download rankings (CSV)", ranking.to_csv(),
                               f"stock_rankings_{date_str}.csv", "text/csv",
                               use_container_width=True)
        with c3:
            st.markdown("**📈 Key metrics**")
            st.download_button("Download key metrics (CSV)", key_export.to_csv(),
                               f"key_metrics_{date_str}.csv", "text/csv",
                               use_container_width=True)

        st.markdown("---")
        preview = st.selectbox("Preview dataset",
                               ["Complete analysis", "Rankings", "Key metrics"])
        preview_map = {
            "Complete analysis": full_export,
            "Rankings":          ranking,
            "Key metrics":       key_export,
        }
        st.dataframe(preview_map[preview], use_container_width=True, height=400)


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    st.set_page_config(
        page_title="Advanced Fundamental Analysis",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    render_fundamental_comparison()
