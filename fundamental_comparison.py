import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════════════════════════════════
# DATA QUALITY TRACKING
# ══════════════════════════════════════════════════════════════════════════════

class DataQualityLog:
    """Collects per-ticker data quality warnings for surface in the UI."""
    def __init__(self):
        self._warnings: Dict[str, List[str]] = {}

    def warn(self, ticker: str, msg: str):
        self._warnings.setdefault(ticker, []).append(msg)

    def get(self, ticker: str) -> List[str]:
        return self._warnings.get(ticker, [])

    def all(self) -> Dict[str, List[str]]:
        return self._warnings

    def any(self) -> bool:
        return bool(self._warnings)


# ══════════════════════════════════════════════════════════════════════════════
# DATA LAYER
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_single_ticker_cached(ticker: str, is_annual: bool) -> Tuple[str, Optional[Dict]]:
    """Fetch all financial data for a single ticker (cross-session cache, 1hr TTL)."""
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

        # ── Analyst / forward data ────────────────────────────────────────────
        try:
            analyst_targets = obj.analyst_price_targets
            if analyst_targets is None or (hasattr(analyst_targets, 'empty') and analyst_targets.empty):
                analyst_targets = {}
            elif isinstance(analyst_targets, pd.DataFrame):
                analyst_targets = analyst_targets.to_dict()
        except Exception:
            analyst_targets = {}

        try:
            earnings_est = obj.earnings_estimate
            if earnings_est is None or (hasattr(earnings_est, 'empty') and earnings_est.empty):
                earnings_est = pd.DataFrame()
        except Exception:
            earnings_est = pd.DataFrame()

        try:
            revenue_est = obj.revenue_estimate
            if revenue_est is None or (hasattr(revenue_est, 'empty') and revenue_est.empty):
                revenue_est = pd.DataFrame()
        except Exception:
            revenue_est = pd.DataFrame()

        # ── Price history for trend overlay ──────────────────────────────────
        try:
            price_hist = obj.history(period="5y", interval="1mo", auto_adjust=True)["Close"]
        except Exception:
            price_hist = pd.Series(dtype=float)

        return ticker, {
            "income":          income,
            "balance":         balance,
            "cashflow":        cashflow,
            "info":            info,
            "analyst_targets": analyst_targets,
            "earnings_est":    earnings_est,
            "revenue_est":     revenue_est,
            "price_hist":      price_hist,
        }
    except Exception:
        return ticker, None


def fetch_all_tickers(tickers: List[str], is_annual: bool, max_workers: int = 8) -> Tuple[Dict, List[str]]:
    """Fetch data for all tickers in parallel."""
    all_data: Dict     = {}
    failed:   List[str] = []

    progress  = st.progress(0)
    status    = st.empty()
    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_single_ticker_cached, t, is_annual): t for t in tickers}
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
# SECTOR BENCHMARK DATA
# ══════════════════════════════════════════════════════════════════════════════

# Typical median benchmarks by sector for key metrics.
# Source: broad market observations — directionally accurate, not live data.
SECTOR_BENCHMARKS: Dict[str, Dict[str, float]] = {
    "Technology": {
        "Gross Margin %": 55.0, "Operating Margin %": 20.0, "Net Margin %": 18.0,
        "ROE %": 28.0, "ROA %": 12.0, "Revenue Growth %": 12.0,
        "Current Ratio": 2.0, "Debt/Equity": 0.5, "P/E": 28.0, "EV/EBITDA": 20.0,
    },
    "Healthcare": {
        "Gross Margin %": 55.0, "Operating Margin %": 15.0, "Net Margin %": 12.0,
        "ROE %": 18.0, "ROA %": 8.0, "Revenue Growth %": 8.0,
        "Current Ratio": 2.2, "Debt/Equity": 0.6, "P/E": 22.0, "EV/EBITDA": 15.0,
    },
    "Financial Services": {
        "Gross Margin %": 60.0, "Operating Margin %": 28.0, "Net Margin %": 22.0,
        "ROE %": 12.0, "ROA %": 1.2, "Revenue Growth %": 5.0,
        "Current Ratio": 1.2, "Debt/Equity": 3.0, "P/E": 12.0, "EV/EBITDA": 10.0,
    },
    "Consumer Cyclical": {
        "Gross Margin %": 35.0, "Operating Margin %": 8.0, "Net Margin %": 6.0,
        "ROE %": 20.0, "ROA %": 7.0, "Revenue Growth %": 5.0,
        "Current Ratio": 1.5, "Debt/Equity": 1.0, "P/E": 18.0, "EV/EBITDA": 12.0,
    },
    "Consumer Defensive": {
        "Gross Margin %": 30.0, "Operating Margin %": 10.0, "Net Margin %": 7.0,
        "ROE %": 22.0, "ROA %": 8.0, "Revenue Growth %": 3.0,
        "Current Ratio": 1.2, "Debt/Equity": 1.2, "P/E": 20.0, "EV/EBITDA": 14.0,
    },
    "Industrials": {
        "Gross Margin %": 32.0, "Operating Margin %": 12.0, "Net Margin %": 8.0,
        "ROE %": 16.0, "ROA %": 6.0, "Revenue Growth %": 5.0,
        "Current Ratio": 1.6, "Debt/Equity": 0.8, "P/E": 18.0, "EV/EBITDA": 13.0,
    },
    "Energy": {
        "Gross Margin %": 40.0, "Operating Margin %": 15.0, "Net Margin %": 10.0,
        "ROE %": 14.0, "ROA %": 6.0, "Revenue Growth %": 3.0,
        "Current Ratio": 1.3, "Debt/Equity": 0.7, "P/E": 14.0, "EV/EBITDA": 7.0,
    },
    "Communication Services": {
        "Gross Margin %": 50.0, "Operating Margin %": 18.0, "Net Margin %": 14.0,
        "ROE %": 22.0, "ROA %": 8.0, "Revenue Growth %": 8.0,
        "Current Ratio": 1.5, "Debt/Equity": 0.9, "P/E": 22.0, "EV/EBITDA": 14.0,
    },
    "Basic Materials": {
        "Gross Margin %": 28.0, "Operating Margin %": 12.0, "Net Margin %": 8.0,
        "ROE %": 14.0, "ROA %": 6.0, "Revenue Growth %": 3.0,
        "Current Ratio": 1.6, "Debt/Equity": 0.7, "P/E": 14.0, "EV/EBITDA": 9.0,
    },
    "Real Estate": {
        "Gross Margin %": 55.0, "Operating Margin %": 22.0, "Net Margin %": 15.0,
        "ROE %": 8.0, "ROA %": 3.0, "Revenue Growth %": 4.0,
        "Current Ratio": 1.1, "Debt/Equity": 2.0, "P/E": 30.0, "EV/EBITDA": 18.0,
    },
    "Utilities": {
        "Gross Margin %": 35.0, "Operating Margin %": 18.0, "Net Margin %": 12.0,
        "ROE %": 10.0, "ROA %": 3.5, "Revenue Growth %": 2.0,
        "Current Ratio": 0.8, "Debt/Equity": 2.5, "P/E": 18.0, "EV/EBITDA": 12.0,
    },
}
DEFAULT_BENCHMARK = {
    "Gross Margin %": 40.0, "Operating Margin %": 12.0, "Net Margin %": 8.0,
    "ROE %": 15.0, "ROA %": 6.0, "Revenue Growth %": 5.0,
    "Current Ratio": 1.5, "Debt/Equity": 1.0, "P/E": 18.0, "EV/EBITDA": 13.0,
}

HIGHER_BETTER = {"Gross Margin %", "Operating Margin %", "Net Margin %", "ROE %",
                 "ROA %", "Revenue Growth %", "Current Ratio"}
LOWER_BETTER  = {"Debt/Equity", "P/E", "EV/EBITDA"}


def sector_vs_benchmark(value: float, metric: str, sector: str) -> Optional[float]:
    """
    Returns a delta vs. sector benchmark.
    Positive = above benchmark (good for higher-is-better metrics),
    sign-flipped for lower-is-better so that positive always = outperforming.
    Returns None if metric not in benchmarks.
    """
    bench = SECTOR_BENCHMARKS.get(sector, DEFAULT_BENCHMARK)
    if metric not in bench or pd.isna(value):
        return None
    delta = value - bench[metric]
    if metric in LOWER_BETTER:
        delta = -delta
    return delta


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
    if series.empty:
        return np.nan
    valid = series.dropna()
    return float(valid.iloc[-1]) if len(valid) > 0 else np.nan


def sdiv(num, den, dq_log: DataQualityLog = None, ticker: str = "", label: str = "") -> float:
    """Safe division with optional data-quality logging."""
    if pd.isna(num) or pd.isna(den) or den == 0:
        if dq_log and ticker and label and not (pd.isna(num) and pd.isna(den)):
            dq_log.warn(ticker, f"{label} unavailable (denominator is zero or missing)")
        return np.nan
    return float(num) / float(den)


def avg_growth(series: pd.Series) -> float:
    valid = series.dropna()
    if len(valid) < 2:
        return np.nan
    return float(valid.pct_change().mean() * 100)


def cagr(series: pd.Series) -> float:
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

def calculate_metrics(ticker: str, ticker_data: Dict,
                      dq_log: DataQualityLog) -> Dict:
    """Compute comprehensive fundamental metrics — with per-metric guards."""
    income   = ticker_data["income"]
    balance  = ticker_data["balance"]
    cashflow = ticker_data["cashflow"]
    info     = ticker_data["info"]

    # ── Raw series ─────────────────────────────────────────────────────────
    revenue    = safe_get(income,   "Total Revenue")
    gross_pft  = safe_get(income,   "Gross Profit")
    oper_inc   = safe_get(income,   "Operating Income")
    net_inc    = safe_get(income,   "Net Income")
    ebitda     = safe_get(income,   "EBITDA")
    ocf        = safe_get(cashflow, "Operating Cash Flow")
    capex_raw  = safe_get(cashflow, "Capital Expenditure")
    capex_missing = capex_raw.empty or capex_raw.dropna().empty

    if capex_missing:
        capex = pd.Series(0, index=ocf.index if not ocf.empty else [])
        dq_log.warn(ticker, "CapEx not found — FCF treated as equal to Operating CF (may be overstated)")
    else:
        capex = capex_raw

    fcf_series = ocf + capex   # capex is typically negative

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
    if shares.empty or shares.dropna().empty:
        shares = safe_get(income, "Basic Average Shares")
        if not shares.empty and not shares.dropna().empty:
            dq_log.warn(ticker, "Using basic share count (diluted not available) — EPS may be slightly understated")

    # ── Latest scalar values ───────────────────────────────────────────────
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

    if pd.isna(rev_l):  dq_log.warn(ticker, "Revenue not found in income statement")
    if pd.isna(ni_l):   dq_log.warn(ticker, "Net Income not found in income statement")
    if pd.isna(eq_l):   dq_log.warn(ticker, "Stockholder Equity not found — ROE/ROIC/D/E unavailable")
    if pd.isna(sh_l):   dq_log.warn(ticker, "Share count not found — EPS and per-share metrics unavailable")

    # ── Market / price data ────────────────────────────────────────────────
    mkt_cap = info.get("marketCap", np.nan)
    ev      = info.get("enterpriseValue", np.nan)
    price   = info.get("currentPrice",
              info.get("regularMarketPrice",
              info.get("previousClose", np.nan)))
    sector  = info.get("sector", "Unknown")
    industry = info.get("industry", "Unknown")

    if pd.isna(ev) or ev <= 0:
        if not any(pd.isna(x) for x in [mkt_cap, debt_l, cash_l]):
            ev = mkt_cap + debt_l - cash_l
            dq_log.warn(ticker, "Enterprise Value reconstructed from MarketCap + Debt − Cash")

    # ── Profitability (guarded) ────────────────────────────────────────────
    gross_margin  = sdiv(gp_l, rev_l,   dq_log, ticker, "Gross Margin") * 100
    oper_margin   = sdiv(oi_l, rev_l,   dq_log, ticker, "Operating Margin") * 100
    net_margin    = sdiv(ni_l, rev_l,   dq_log, ticker, "Net Margin") * 100
    ebitda_margin = sdiv(ebitda_l, rev_l, dq_log, ticker, "EBITDA Margin") * 100
    roe           = sdiv(ni_l, eq_l,    dq_log, ticker, "ROE") * 100
    roa           = sdiv(ni_l, assets_l, dq_log, ticker, "ROA") * 100

    # ROIC guard: avoid divide-by-zero when both equity and debt are zero
    ic = (eq_l or 0) + (debt_l or 0)
    if ic == 0:
        roic = np.nan
        dq_log.warn(ticker, "ROIC unavailable — Invested Capital (Equity + Debt) is zero")
    else:
        nopat = oi_l * 0.65 if not pd.isna(oi_l) else np.nan
        roic  = sdiv(nopat, ic, dq_log, ticker, "ROIC") * 100

    # ── Liquidity & solvency (guarded) ────────────────────────────────────
    current_ratio  = sdiv(ca_l, cl_l, dq_log, ticker, "Current Ratio")
    quick_ratio    = sdiv(ca_l - (inv_l or 0), cl_l, dq_log, ticker, "Quick Ratio")
    debt_to_equity = sdiv(debt_l, eq_l, dq_log, ticker, "Debt/Equity")
    debt_to_assets = sdiv(debt_l, assets_l, dq_log, ticker, "Debt/Assets")
    equity_ratio   = sdiv(eq_l,   assets_l, dq_log, ticker, "Equity Ratio")
    net_debt       = (debt_l or 0) - (cash_l or 0)
    nd_to_ebitda   = sdiv(net_debt, ebitda_l, dq_log, ticker, "Net Debt/EBITDA") \
                     if (not pd.isna(ebitda_l) and ebitda_l > 0) else np.nan

    # ── Per-share (guarded) ───────────────────────────────────────────────
    eps               = sdiv(ni_l,  sh_l, dq_log, ticker, "EPS")
    revenue_per_share = sdiv(rev_l, sh_l, dq_log, ticker, "Revenue/Share")
    fcf_per_share     = sdiv(fcf_l, sh_l, dq_log, ticker, "FCF/Share")
    bvps              = sdiv(eq_l,  sh_l, dq_log, ticker, "Book Value/Share")

    # ── Valuation multiples (guarded) ─────────────────────────────────────
    pe = info.get("trailingPE", info.get("forwardPE", np.nan))
    pe_source = "yfinance"
    if pd.isna(pe) or pe <= 0 or pe > 1000:
        if not pd.isna(price) and not pd.isna(eps) and eps > 0:
            pe = price / eps
            pe_source = "calculated (price/EPS)"
        else:
            pe = np.nan
            dq_log.warn(ticker, "P/E unavailable — no trailing/forward PE from yfinance and EPS is negative or missing")

    if pe_source != "yfinance" and not pd.isna(pe):
        dq_log.warn(ticker, f"P/E {pe_source}")

    pb = info.get("priceToBook", np.nan)
    if pd.isna(pb) or pb <= 0:
        if not pd.isna(mkt_cap) and not pd.isna(eq_l) and eq_l > 0:
            pb = mkt_cap / eq_l
        else:
            pb = np.nan

    ps = info.get("priceToSalesTrailing12Months", np.nan)
    if pd.isna(ps) or ps <= 0:
        if not pd.isna(mkt_cap) and not pd.isna(rev_l) and rev_l > 0:
            ps = mkt_cap / rev_l
        else:
            ps = np.nan

    price_to_fcf = sdiv(mkt_cap, fcf_l) \
                   if (not pd.isna(mkt_cap) and not pd.isna(fcf_l) and fcf_l > 0) else np.nan
    ev_to_rev    = sdiv(ev, rev_l) \
                   if (not pd.isna(ev) and not pd.isna(rev_l) and rev_l > 0) else np.nan
    ev_to_ebitda = info.get("enterpriseToEbitda", np.nan)
    if pd.isna(ev_to_ebitda) or ev_to_ebitda <= 0 or ev_to_ebitda > 200:
        ev_to_ebitda = sdiv(ev, ebitda_l) \
                       if (not pd.isna(ev) and not pd.isna(ebitda_l) and ebitda_l > 0) else np.nan

    # ── Growth ────────────────────────────────────────────────────────────
    rev_growth  = avg_growth(revenue)
    rev_cagr    = cagr(revenue)
    ni_growth   = avg_growth(net_inc)
    ni_cagr     = cagr(net_inc)
    fcf_growth  = avg_growth(fcf_series)
    eps_series  = net_inc / shares \
                  if (not shares.empty and not net_inc.empty and not shares.dropna().empty) \
                  else pd.Series(dtype=float)
    eps_growth  = avg_growth(eps_series)

    peg = np.nan
    if not pd.isna(pe) and not pd.isna(rev_growth) and rev_growth > 0:
        peg = pe / rev_growth

    # ── Cash flow quality ─────────────────────────────────────────────────
    fcf_to_ni  = sdiv(fcf_l, ni_l)  if (not pd.isna(ni_l) and ni_l > 0) else np.nan
    ocf_to_ni  = sdiv(ocf_l, ni_l)  if (not pd.isna(ni_l) and ni_l > 0) else np.nan
    fcf_margin = sdiv(fcf_l, rev_l) * 100

    # ── Efficiency ────────────────────────────────────────────────────────
    asset_turnover = sdiv(rev_l, assets_l, dq_log, ticker, "Asset Turnover")

    # ── Dividends ─────────────────────────────────────────────────────────
    div_yield = info.get("dividendYield", np.nan)
    if not pd.isna(div_yield):
        div_yield *= 100
    payout_ratio = sdiv(div_l, ni_l) * 100 \
                   if (not pd.isna(div_l) and not pd.isna(ni_l) and ni_l > 0) else np.nan
    div_coverage = sdiv(ni_l, div_l) if (not pd.isna(div_l) and div_l > 0) else np.nan
    div_5yr_growth = avg_growth(abs(div_paid) if not div_paid.empty else pd.Series(dtype=float))

    # ── Analyst & forward data ────────────────────────────────────────────
    earnings_est = ticker_data.get("earnings_est", pd.DataFrame())
    revenue_est  = ticker_data.get("revenue_est",  pd.DataFrame())
    analyst_targets = ticker_data.get("analyst_targets", {})

    # Forward EPS estimates (0q = current quarter, +1q = next, 0y = current year, +1y = next year)
    fwd_eps_cy  = np.nan  # current year estimate
    fwd_eps_ny  = np.nan  # next year estimate
    eps_revision = np.nan # analyst revision direction

    if isinstance(earnings_est, pd.DataFrame) and not earnings_est.empty:
        try:
            if "0y" in earnings_est.index:
                row = earnings_est.loc["0y"]
                fwd_eps_cy = float(row.get("avg", np.nan)) if hasattr(row, "get") else np.nan
            if "+1y" in earnings_est.index:
                row = earnings_est.loc["+1y"]
                fwd_eps_ny = float(row.get("avg", np.nan)) if hasattr(row, "get") else np.nan
        except Exception:
            pass

    # Forward P/E from info or estimate
    fwd_pe = info.get("forwardPE", np.nan)
    if (pd.isna(fwd_pe) or fwd_pe <= 0) and not pd.isna(price) and not pd.isna(fwd_eps_cy) and fwd_eps_cy > 0:
        fwd_pe = price / fwd_eps_cy

    # Revenue estimate next year
    fwd_rev_ny = np.nan
    if isinstance(revenue_est, pd.DataFrame) and not revenue_est.empty:
        try:
            if "+1y" in revenue_est.index:
                row = revenue_est.loc["+1y"]
                fwd_rev_ny = float(row.get("avg", np.nan)) if hasattr(row, "get") else np.nan
        except Exception:
            pass

    # Analyst price targets
    target_mean = np.nan
    target_high = np.nan
    target_low  = np.nan
    target_upside = np.nan
    n_analysts  = np.nan

    if isinstance(analyst_targets, dict) and analyst_targets:
        target_mean = analyst_targets.get("mean", np.nan)
        target_high = analyst_targets.get("high", np.nan)
        target_low  = analyst_targets.get("low",  np.nan)
        if not pd.isna(target_mean) and not pd.isna(price) and price > 0:
            target_upside = (target_mean / price - 1) * 100

    # Also try info dict
    if pd.isna(target_mean):
        target_mean   = info.get("targetMeanPrice",   np.nan)
        target_high   = info.get("targetHighPrice",   np.nan)
        target_low    = info.get("targetLowPrice",    np.nan)
        n_analysts    = info.get("numberOfAnalystOpinions", np.nan)
        if not pd.isna(target_mean) and not pd.isna(price) and price > 0:
            target_upside = (target_mean / price - 1) * 100

    recommendation = info.get("recommendationKey", "").upper().replace("_", " ")

    # ── DCF estimate ──────────────────────────────────────────────────────
    dcf_intrinsic = np.nan
    if not any(pd.isna(x) for x in [fcf_l, sh_l, rev_growth]) and fcf_l > 0 and sh_l > 0:
        try:
            wacc       = 0.09
            terminal_g = 0.03
            stage1_g   = min(max(rev_growth / 100, -0.1), 0.25)
            stage2_g   = min(stage1_g * 0.5, 0.08)
            pv, cf     = 0.0, fcf_l
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

    # ── Time series ───────────────────────────────────────────────────────
    ts_gross_margin = (gross_pft / revenue * 100).replace([np.inf, -np.inf], np.nan) \
        if (not revenue.empty and not gross_pft.empty) else pd.Series(dtype=float)
    ts_oper_margin  = (oper_inc  / revenue * 100).replace([np.inf, -np.inf], np.nan) \
        if (not revenue.empty and not oper_inc.empty)  else pd.Series(dtype=float)
    ts_net_margin   = (net_inc   / revenue * 100).replace([np.inf, -np.inf], np.nan) \
        if (not revenue.empty and not net_inc.empty)   else pd.Series(dtype=float)
    ts_roe          = (net_inc   / equity  * 100).replace([np.inf, -np.inf], np.nan) \
        if (not equity.empty   and not net_inc.empty)  else pd.Series(dtype=float)

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
        # Metadata
        "Sector":           sector,
        "Industry":         industry,
        "Current Price":    price,
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
        "Fwd P/E":     fwd_pe,
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
        # Analyst / forward
        "Fwd EPS (CY)":    fwd_eps_cy,
        "Fwd EPS (NY)":    fwd_eps_ny,
        "Fwd Rev (NY)":    fwd_rev_ny,
        "Price Target":    target_mean,
        "Target High":     target_high,
        "Target Low":      target_low,
        "Analyst Upside %": target_upside,
        "# Analysts":      n_analysts,
        "Recommendation":  recommendation,
        # Time series (prefixed _ — excluded from export/scoring)
        "_revenue_ts":          revenue,
        "_net_income_ts":       net_inc,
        "_ocf_ts":              ocf,
        "_fcf_ts":              fcf_series,
        "_gross_margin_ts":     ts_gross_margin,
        "_operating_margin_ts": ts_oper_margin,
        "_net_margin_ts":       ts_net_margin,
        "_roe_ts":              ts_roe,
        "_price_hist":          ticker_data.get("price_hist", pd.Series(dtype=float)),
    }


# ══════════════════════════════════════════════════════════════════════════════
# SCORING ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def calculate_fundamental_score(df: pd.DataFrame) -> pd.DataFrame:
    CRITERIA = {
        "ROE %":              {"raw_weight": 0.10, "higher_better": True},
        "Net Margin %":       {"raw_weight": 0.10, "higher_better": True},
        "Operating Margin %": {"raw_weight": 0.10, "higher_better": True},
        "ROIC %":             {"raw_weight": 0.10, "higher_better": True},
        "Revenue Growth %":   {"raw_weight": 0.10, "higher_better": True},
        "NI Growth %":        {"raw_weight": 0.08, "higher_better": True},
        "EPS Growth %":       {"raw_weight": 0.07, "higher_better": True},
        "Current Ratio":      {"raw_weight": 0.05, "higher_better": True},
        "Debt/Equity":        {"raw_weight": 0.08, "higher_better": False, "clip_high": 10},
        "FCF/NI":             {"raw_weight": 0.07, "higher_better": True},
        "P/E":                {"raw_weight": 0.05, "higher_better": False, "clip_high": 100, "optional": True},
        "PEG":                {"raw_weight": 0.05, "higher_better": False, "clip_high": 5,   "optional": True},
        "EV/EBITDA":          {"raw_weight": 0.05, "higher_better": False, "clip_high": 50,  "optional": True},
    }

    active = {}
    for metric, params in CRITERIA.items():
        if metric not in df.columns:
            continue
        n_valid = df[metric].replace([np.inf, -np.inf], np.nan).notna().sum()
        if params.get("optional") and n_valid < 2:
            continue
        active[metric] = params

    total_w = sum(p["raw_weight"] for p in active.values())
    for metric in active:
        active[metric]["weight"] = active[metric]["raw_weight"] / total_w

    scores = pd.DataFrame(index=df.index)
    for metric, params in active.items():
        values = df[metric].replace([np.inf, -np.inf], np.nan).copy()
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
        norm = norm.fillna(norm.median())
        scores[metric] = norm * params["weight"]

    scores["Total Score"] = scores.sum(axis=1)

    def grade(s: float) -> str:
        if pd.isna(s): return "N/A"
        for threshold, letter in [(80, "A+"), (75, "A"), (70, "A-"), (65, "B+"),
                                   (60, "B"), (55, "B-"), (50, "C+"), (45, "C"),
                                   (40, "C-"), (35, "D+"), (30, "D")]:
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
        x=clean.index, y=clean.values, marker_color=color,
        text=[fmt.format(v) for v in clean.values], textposition="outside",
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
        title=title, yaxis_title=ylabel, height=400, hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def price_overlay_chart(price_data: Dict[str, pd.Series],
                        fundamental_data: Dict[str, pd.Series],
                        fundamental_label: str,
                        title: str) -> go.Figure:
    """Dual-axis chart: normalised price + a fundamental metric over time."""
    fig = go.Figure()
    colors = px.colors.qualitative.Plotly

    for i, (ticker, price_series) in enumerate(price_data.items()):
        ps = price_series.dropna()
        if ps.empty:
            continue
        norm = ps / ps.iloc[0] * 100  # normalise to 100
        fig.add_trace(go.Scatter(
            x=norm.index, y=norm.values, name=f"{ticker} Price (norm.)",
            mode="lines", line=dict(width=2, color=colors[i % len(colors)]),
            yaxis="y1",
        ))

    for i, (ticker, fund_series) in enumerate(fundamental_data.items()):
        fs = fund_series.dropna()
        if fs.empty:
            continue
        fig.add_trace(go.Scatter(
            x=fs.index, y=fs.values, name=f"{ticker} {fundamental_label}",
            mode="lines+markers", line=dict(width=1.5, dash="dot",
                                             color=colors[i % len(colors)]),
            marker=dict(size=5), yaxis="y2",
        ))

    fig.update_layout(
        title=title, height=460, hovermode="x unified",
        yaxis  = dict(title="Price (normalised to 100)", side="left"),
        yaxis2 = dict(title=fundamental_label, side="right", overlaying="y"),
        legend = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def color_valuation(val):
    try:
        v = float(val)
        if v < 0:   return "background-color: #ffcccc"
        if v > 100: return "background-color: #ffcccc"
        if v < 15:  return "background-color: #ccffcc"
        if v < 25:  return "background-color: #ffffcc"
        return "background-color: #ffddcc"
    except Exception:
        return ""


def color_upside(val):
    try:
        v = float(val)
        if v > 20:  return "background-color: #ccffcc"
        if v > 5:   return "background-color: #ffffcc"
        if v < -10: return "background-color: #ffcccc"
        return ""
    except Exception:
        return ""


# ══════════════════════════════════════════════════════════════════════════════
# MAIN RENDER FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def render_fundamental_comparison(tickers: List[str] = None) -> None:
    st.markdown("# 📊 Advanced Fundamental Analysis & Comparison")
    st.markdown("Compare financial metrics, analyse trends, and rank stocks by fundamental strength.")

    # ── Configuration ──────────────────────────────────────────────────────
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

    is_annual = frequency == "Annual"
    cache_key = f"fund_{'_'.join(sorted(tickers))}_{frequency}"

    if cache_key not in st.session_state:
        with st.spinner(f"📥 Fetching data for {len(tickers)} stocks…"):
            data, failed = fetch_all_tickers(tickers, is_annual)
        st.session_state[cache_key] = (data, failed)
    else:
        st.info("ℹ️ Using cached data (≤1 hr old). Click Analyze again to refresh.")
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

    # ── Metric calculation ─────────────────────────────────────────────────
    dq_log = DataQualityLog()
    with st.spinner("📊 Calculating metrics…"):
        metrics = {t: calculate_metrics(t, data[t], dq_log) for t in tickers}
        df = pd.DataFrame(metrics).T

    scores_df = calculate_fundamental_score(df)
    export_cols = [c for c in df.columns if not c.startswith("_")]

    # ── Data quality banner ────────────────────────────────────────────────
    if dq_log.any():
        with st.expander("⚠️ Data quality notices", expanded=False):
            for ticker, warnings in dq_log.all().items():
                st.markdown(f"**{ticker}**")
                for w in warnings:
                    st.caption(f"  • {w}")

    # ── Tabs ──────────────────────────────────────────────────────────────
    tab_labels = ["🏆 Rankings", "📊 Overview", "💰 Profitability",
                  "💵 Cash Flow", "🏦 Financial Health", "📈 Valuation",
                  "🔭 Analyst & Forward", "💸 Dividends", "📉 Trends", "📥 Export"]
    tabs = st.tabs(tab_labels)

    # ═════════════════════════════════════════════════════════════════════
    # TAB 1 — RANKINGS
    # ═════════════════════════════════════════════════════════════════════
    with tabs[0]:
        st.markdown("## 🏆 Fundamental Strength Rankings")
        st.caption("Scoring based on profitability, growth, financial health, and valuation. "
                   "Weights normalised to sum to 100%.")

        ranking = pd.DataFrame({
            "Rank":               scores_df["Rank"],
            "Score":              scores_df["Total Score"],
            "Grade":              scores_df["Grade"],
            "Sector":             df["Sector"],
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

        # ── Sector-relative view ──────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 🏭 Sector-relative performance")
        st.caption("Delta vs. sector benchmark medians. Green = outperforming sector, "
                   "red = underperforming. Positive always means better (sign-adjusted for "
                   "lower-is-better metrics like P/E).")

        sector_metrics = ["Gross Margin %", "Operating Margin %", "Net Margin %",
                          "ROE %", "ROA %", "Revenue Growth %", "Current Ratio",
                          "Debt/Equity", "P/E", "EV/EBITDA"]

        sector_delta = {}
        for t in tickers:
            sector = df.loc[t, "Sector"]
            row = {}
            for m in sector_metrics:
                val = df.loc[t, m] if m in df.columns else np.nan
                row[m] = sector_vs_benchmark(val, m, sector)
            sector_delta[t] = row

        delta_df = pd.DataFrame(sector_delta).T

        def color_delta(val):
            try:
                v = float(val)
                if v > 5:   return "background-color: #c6efce; color: #276221"
                if v > 0:   return "background-color: #e2efda"
                if v < -5:  return "background-color: #ffc7ce; color: #9c0006"
                return "background-color: #ffe0e0"
            except Exception:
                return ""

        sector_display = delta_df.copy()
        for col in sector_display.columns:
            sector_display[col] = sector_display[col].map(
                lambda x: f"+{x:.1f}" if not pd.isna(x) and x > 0
                else (f"{x:.1f}" if not pd.isna(x) else "-")
            )
        # Add sector label
        sector_display.insert(0, "Sector", df["Sector"])

        st.dataframe(
            delta_df.style.map(color_delta).format("{:+.1f}", na_rep="-"),
            use_container_width=True,
        )
        st.caption("Benchmark source: sector median estimates. Values shown as delta from benchmark.")

        # Score bar + grade pie
        c1, c2 = st.columns(2)
        with c1:
            colours = ["gold" if r == 1 else "silver" if r == 2 else "#CD7F32" if r == 3
                       else "steelblue" for r in ranking["Rank"]]
            fig = go.Figure(go.Bar(
                x=ranking.index, y=ranking["Score"], marker_color=colours,
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
                    r=vals, theta=list(cat_raw.columns) + [cat_raw.columns[0]],
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
            "💎 Valuation":        ("PEG",              True),
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
                st.markdown(f"{i}. **{ticker}** ({row['Sector']}) — Grade {row['Grade']} ({row['Score']:.1f})")
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
                st.markdown(f"{offset + i}. **{ticker}** ({row['Sector']}) — Grade {row['Grade']} ({row['Score']:.1f})")
                c = []
                med = ranking["ROE %"].median()
                if not pd.isna(row["ROE %"]) and row["ROE %"] < med: c.append("Low ROE")
                if not pd.isna(row["Revenue Growth %"]) and row["Revenue Growth %"] < 0: c.append("Negative growth")
                med = ranking["Debt/Equity"].median()
                if not pd.isna(row["Debt/Equity"]) and row["Debt/Equity"] > med: c.append("High debt")
                if c: st.caption("Concerns: " + ", ".join(c))

    # ═════════════════════════════════════════════════════════════════════
    # TAB 2 — OVERVIEW
    # ═════════════════════════════════════════════════════════════════════
    with tabs[1]:
        st.markdown("### Key metrics overview")
        ov_cols = ["Sector", "Industry", "Market Cap", "Revenue", "Net Income",
                   "Free Cash Flow", "EPS", "P/E", "P/B", "ROE %",
                   "Net Margin %", "Revenue Growth %"]
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
            .background_gradient(cmap="RdYlGn",
                                  subset=["ROE %", "Net Margin %", "Revenue Growth %"]),
            use_container_width=True,
        )

        c1, c2 = st.columns(2)
        with c1:
            mcap = ov["Market Cap"].dropna()
            if not mcap.empty:
                st.plotly_chart(
                    bar_chart(mcap, "Market Cap ($)", "Market capitalisation", fmt="${:,.0f}"),
                    use_container_width=True,
                )
        with c2:
            rev = ov["Revenue"].dropna()
            if not rev.empty:
                fig = px.pie(values=rev.values, names=rev.index, title="Revenue share")
                fig.update_layout(height=380)
                st.plotly_chart(fig, use_container_width=True)

    # ═════════════════════════════════════════════════════════════════════
    # TAB 3 — PROFITABILITY
    # ═════════════════════════════════════════════════════════════════════
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

    # ═════════════════════════════════════════════════════════════════════
    # TAB 4 — CASH FLOW
    # ═════════════════════════════════════════════════════════════════════
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
            .background_gradient(cmap="RdYlGn",
                                  subset=["FCF Margin %", "FCF/NI", "OCF/NI"]),
            use_container_width=True,
        )

        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            ni  = cf["Net Income"].dropna()
            fcf = cf["Free Cash Flow"].dropna()
            if not ni.empty:
                fig.add_trace(go.Bar(name="Net Income", x=ni.index,  y=ni.values,
                                     marker_color="lightcoral"))
            if not fcf.empty:
                fig.add_trace(go.Bar(name="Free Cash Flow", x=fcf.index, y=fcf.values,
                                     marker_color="lightgreen"))
            fig.update_layout(barmode="group", yaxis_title="Amount ($)",
                               height=400, title="FCF vs net income")
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

    # ═════════════════════════════════════════════════════════════════════
    # TAB 5 — FINANCIAL HEALTH
    # ═════════════════════════════════════════════════════════════════════
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
            .background_gradient(cmap="RdYlGn",
                                  subset=["Current Ratio", "Quick Ratio", "Equity Ratio"])
            .background_gradient(cmap="RdYlGn_r",
                                  subset=["Debt/Equity", "Debt/Assets"]),
            use_container_width=True,
        )

        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            c = health["Cash"].dropna()
            d = health["Total Debt"].dropna()
            if not c.empty:
                fig.add_trace(go.Bar(name="Cash", x=c.index, y=c.values,
                                     marker_color="lightgreen"))
            if not d.empty:
                fig.add_trace(go.Bar(name="Total Debt", x=d.index, y=d.values,
                                     marker_color="lightcoral"))
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

    # ═════════════════════════════════════════════════════════════════════
    # TAB 6 — VALUATION
    # ═════════════════════════════════════════════════════════════════════
    with tabs[5]:
        st.markdown("### Valuation metrics")
        val_cols = ["P/E", "Fwd P/E", "P/B", "P/S", "Price/FCF",
                    "EV/Revenue", "EV/EBITDA", "PEG",
                    "DCF Estimate", "Current Price", "Market Cap", "Revenue Growth %"]
        val = df[[c for c in val_cols if c in df.columns]].copy()

        primary_metrics = ["P/E", "P/B", "P/S", "EV/EBITDA"]
        valid_cnt = val[primary_metrics].replace([np.inf, -np.inf], np.nan).notna().sum().sum()
        if valid_cnt == 0:
            st.warning("⚠️ No valuation multiples available from yfinance.")
        else:
            st.caption(f"📊 {valid_cnt} / {len(val) * len(primary_metrics)} valuation data points available")

        fmt_map = {
            "P/E": "{:.2f}", "Fwd P/E": "{:.2f}", "P/B": "{:.2f}", "P/S": "{:.2f}",
            "Price/FCF": "{:.2f}", "EV/Revenue": "{:.2f}", "EV/EBITDA": "{:.2f}",
            "PEG": "{:.2f}", "DCF Estimate": "${:.2f}", "Current Price": "${:.2f}",
            "Market Cap": "${:,.0f}", "Revenue Growth %": "{:.2f}",
        }
        style_subset = [c for c in ["P/E", "Fwd P/E", "P/B", "P/S", "EV/EBITDA", "PEG"]
                        if c in val.columns]
        st.dataframe(
            val.style.format(
                {k: v for k, v in fmt_map.items() if k in val.columns}, na_rep="-"
            ).map(color_valuation, subset=style_subset),
            use_container_width=True,
        )
        st.caption("💡 Green < 15, yellow 15–25, orange > 25, red < 0 or > 100.")

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                grouped_bar(val, ["P/E", "Fwd P/E", "P/B", "P/S", "EV/EBITDA"],
                            "Multiple", "Valuation multiples (trailing vs forward P/E)"),
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

        # DCF vs price
        dcf_vals = val["DCF Estimate"].dropna()
        if not dcf_vals.empty:
            st.markdown("**DCF intrinsic value estimate vs market price**")
            st.caption("⚠️ Simple 2-stage DCF using historical FCF growth. Directional only.")
            prices = {t: df.loc[t, "Current Price"] for t in tickers
                      if not pd.isna(df.loc[t, "Current Price"])}
            price_s = pd.Series(prices, name="Current Price")
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
                fig.update_layout(barmode="group", yaxis_title="Price ($)",
                                   height=400, title="DCF estimate vs current price")
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

    # ═════════════════════════════════════════════════════════════════════
    # TAB 7 — ANALYST & FORWARD
    # ═════════════════════════════════════════════════════════════════════
    with tabs[6]:
        st.markdown("## 🔭 Analyst Estimates & Forward Metrics")
        st.caption("Forward-looking data from analyst consensus. May be unavailable for "
                   "smaller-cap or thinly-covered stocks.")

        analyst_cols = ["Current Price", "Price Target", "Target High", "Target Low",
                        "Analyst Upside %", "# Analysts", "Recommendation",
                        "Fwd P/E", "Fwd EPS (CY)", "Fwd EPS (NY)", "Fwd Rev (NY)"]
        analyst_df = df[[c for c in analyst_cols if c in df.columns]].copy()

        # Check coverage
        has_targets = analyst_df["Price Target"].replace([np.inf, -np.inf], np.nan).dropna()
        if has_targets.empty:
            st.info("📌 No analyst price targets available for the selected stocks.")
        else:
            st.markdown("### Price targets & recommendations")
            style_cols = [c for c in ["Analyst Upside %"] if c in analyst_df.columns]
            fmt_analyst = {
                "Current Price":    "${:.2f}",
                "Price Target":     "${:.2f}",
                "Target High":      "${:.2f}",
                "Target Low":       "${:.2f}",
                "Analyst Upside %": "{:+.1f}%",
                "# Analysts":       "{:.0f}",
                "Fwd P/E":          "{:.2f}",
                "Fwd EPS (CY)":     "${:.2f}",
                "Fwd EPS (NY)":     "${:.2f}",
                "Fwd Rev (NY)":     "${:,.0f}",
            }
            sty = analyst_df.style.format(
                {k: v for k, v in fmt_analyst.items() if k in analyst_df.columns}, na_rep="-"
            )
            if style_cols:
                sty = sty.map(color_upside, subset=style_cols)
            st.dataframe(sty, use_container_width=True)

            # Price target waterfall chart
            pt_data = analyst_df[["Current Price", "Target Low", "Price Target", "Target High"]].dropna(
                subset=["Price Target"])
            if not pt_data.empty:
                fig = go.Figure()
                for ticker in pt_data.index:
                    row = pt_data.loc[ticker]
                    fig.add_trace(go.Box(
                        name=ticker,
                        q1=[float(row["Target Low"])],
                        median=[float(row["Price Target"])],
                        q3=[float(row["Target High"])],
                        lowerfence=[float(row["Target Low"])],
                        upperfence=[float(row["Target High"])],
                        mean=[float(row["Price Target"])],
                        showlegend=False,
                    ))
                    # Overlay current price
                    fig.add_shape(type="line",
                                  x0=ticker, x1=ticker,
                                  y0=float(row["Current Price"]) * 0.999,
                                  y1=float(row["Current Price"]) * 1.001,
                                  line=dict(color="red", width=4))
                fig.update_layout(
                    title="Price target ranges (red line = current price)",
                    yaxis_title="Price ($)", height=420,
                )
                st.plotly_chart(fig, use_container_width=True)

            # Upside bar
            upside = analyst_df["Analyst Upside %"].dropna()
            if not upside.empty:
                colours = ["green" if v >= 0 else "red" for v in upside.values]
                fig = go.Figure(go.Bar(
                    x=upside.index, y=upside.values,
                    marker_color=colours,
                    text=[f"{v:+.1f}%" for v in upside.values],
                    textposition="outside",
                ))
                fig.add_hline(y=0, line_color="gray")
                fig.update_layout(
                    title="Analyst consensus upside/downside vs current price",
                    yaxis_title="Upside %", height=380,
                )
                st.plotly_chart(fig, use_container_width=True)

        # Forward EPS comparison
        st.markdown("---")
        st.markdown("### Forward EPS estimates")
        fwd_eps = analyst_df[["Fwd EPS (CY)", "Fwd EPS (NY)"]].dropna(how="all")
        if not fwd_eps.empty:
            st.plotly_chart(
                grouped_bar(fwd_eps, ["Fwd EPS (CY)", "Fwd EPS (NY)"],
                            "EPS ($)", "Current-year vs next-year EPS estimates"),
                use_container_width=True,
            )
        else:
            st.info("📌 Forward EPS estimates not available.")

        # Trailing vs forward P/E
        st.markdown("---")
        st.markdown("### Trailing vs Forward P/E")
        pe_comp = df[["P/E", "Fwd P/E"]].dropna(how="all")
        if not pe_comp.empty:
            st.plotly_chart(
                grouped_bar(pe_comp, ["P/E", "Fwd P/E"],
                            "P/E Multiple", "P/E expansion / contraction"),
                use_container_width=True,
            )
        else:
            st.info("📌 P/E data not available.")

    # ═════════════════════════════════════════════════════════════════════
    # TAB 8 — DIVIDENDS
    # ═════════════════════════════════════════════════════════════════════
    with tabs[7]:
        st.markdown("### Dividend analysis")
        div_cols = ["Dividend Yield %", "Payout Ratio %",
                    "Dividend Coverage", "Dividend Growth 5Y %",
                    "Free Cash Flow", "Net Income"]
        div = df[div_cols].copy()
        payers = div["Dividend Yield %"].dropna()
        if payers.empty:
            st.info("📌 None of the selected stocks appear to pay dividends.")
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

    # ═════════════════════════════════════════════════════════════════════
    # TAB 9 — TRENDS  (with price overlay)
    # ═════════════════════════════════════════════════════════════════════
    with tabs[8]:
        st.markdown("### Historical trends")

        def ts_dict(key: str) -> Dict[str, pd.Series]:
            return {t: df.loc[t, key] for t in tickers
                    if key in df.columns and not df.loc[t, key].empty}

        # Price data dict
        price_hist_dict = {
            t: df.loc[t, "_price_hist"] for t in tickers
            if "_price_hist" in df.columns and not df.loc[t, "_price_hist"].empty
        }

        st.plotly_chart(
            trend_chart(ts_dict("_revenue_ts"), "Revenue ($)", "Revenue over time"),
            use_container_width=True,
        )
        st.plotly_chart(
            trend_chart(ts_dict("_net_income_ts"), "Net Income ($)", "Net income over time"),
            use_container_width=True,
        )

        # Price + margin overlay
        st.markdown("---")
        st.markdown("### 📉 Price vs fundamentals overlay")
        st.caption("Normalised price (left axis, base 100) vs a selected fundamental metric "
                   "(right axis, dotted). Helps answer: *has the market already priced this in?*")

        overlay_choice = st.selectbox(
            "Fundamental metric to overlay",
            ["Net Margin %", "ROE (approx) %", "FCF (scaled)", "Revenue Growth %"],
        )

        overlay_map = {
            "Net Margin %":      "_net_margin_ts",
            "ROE (approx) %":    "_roe_ts",
            "FCF (scaled)":      "_fcf_ts",
            "Revenue Growth %":  "_revenue_ts",
        }
        fund_key = overlay_map[overlay_choice]
        fund_data = ts_dict(fund_key)

        if price_hist_dict and fund_data:
            # Align monthly price to annual/quarterly dates where possible
            st.plotly_chart(
                price_overlay_chart(price_hist_dict, fund_data,
                                    overlay_choice,
                                    f"Price (normalised) vs {overlay_choice}"),
                use_container_width=True,
            )
        else:
            st.info("📌 Price history or fundamental time series not available for overlay.")

        st.markdown("---")
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

        # Normalised price chart standalone
        if price_hist_dict:
            st.markdown("---")
            st.markdown("### Price performance (normalised to 100)")
            norm_fig = go.Figure()
            colors = px.colors.qualitative.Plotly
            for i, (t, ps) in enumerate(price_hist_dict.items()):
                ps = ps.dropna()
                if not ps.empty:
                    norm = ps / ps.iloc[0] * 100
                    norm_fig.add_trace(go.Scatter(
                        x=norm.index, y=norm.values, name=t,
                        mode="lines", line=dict(width=2, color=colors[i % len(colors)]),
                    ))
            norm_fig.add_hline(y=100, line_dash="dash", line_color="gray",
                                annotation_text="Start (100)")
            norm_fig.update_layout(
                title="5-year normalised price performance",
                yaxis_title="Indexed price (100 = start)",
                height=420, hovermode="x unified",
            )
            st.plotly_chart(norm_fig, use_container_width=True)

        growth_cols = ["Revenue Growth %", "NI Growth %", "FCF Growth %", "EPS Growth %"]
        growth_df = df[growth_cols].replace([np.inf, -np.inf], np.nan).dropna(how="all")
        if not growth_df.empty:
            st.plotly_chart(
                grouped_bar(growth_df, growth_cols, "Growth %",
                            "Average period-over-period growth rates", hline=0),
                use_container_width=True,
            )

    # ═════════════════════════════════════════════════════════════════════
    # TAB 10 — EXPORT
    # ═════════════════════════════════════════════════════════════════════
    with tabs[9]:
        st.markdown("### Export data")

        export_df   = df[export_cols].copy()
        full_export = pd.concat([scores_df[["Rank", "Total Score", "Grade"]], export_df], axis=1)
        key_export  = export_df[["Market Cap", "Revenue", "Net Income", "ROE %",
                                 "P/E", "Fwd P/E", "Revenue Growth %", "Debt/Equity",
                                 "Analyst Upside %", "Recommendation"]].copy()

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
