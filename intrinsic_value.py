# intrinsic_value.py
# ============================================================
# Intrinsic Value Pro — Unified Edition
# ============================================================
# Combines:
#   - Multi-model intrinsic value calculator
#   - S&P 500 / NASDAQ 100 / Sector / Custom screener
#   - Rich result table (sortable, heatmap-styled)
#   - Watchlist, export (CSV), scenario DCF, sensitivity
#
# Educational purposes only. Not financial advice.
# ============================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import time
import math

from bs4 import BeautifulSoup
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import plotly.graph_objects as go
import plotly.express as px

# ── Optional RSI library ─────────────────────────────────────────────────────
try:
    import pandas_ta as ta
    HAS_PANDAS_TA = True
except ImportError:
    HAS_PANDAS_TA = False

# ============================================================
# CONSTANTS
# ============================================================

DEFAULT_MOS        = 25.0
DCF_YEARS          = 5
BASE_DISCOUNT      = 0.10
BASE_TERMINAL      = 0.03
MAX_GROWTH         = 0.30
MIN_GROWTH         = 0.02
OUTLIER_MULTIPLE   = 15
MAX_THREADS        = 6

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.google.com/",
}

ALL_MODELS = [
    "Graham Number",
    "Lynch Method",
    "Two-Stage DCF",
    "Dividend Discount Model (DDM)",
    "Residual Income",
    "Owner Earnings (Buffett)",
    "Core Valuation (PE-based)",
    "Asset-Based (NAV)",
]

COLUMN_GROUPS = {
    "Core":            ["Symbol", "Name", "Sector", "Price ($)", "Market Cap (B)"],
    "Valuation":       ["Intrinsic ($)", "Safe Buy ($)", "Premium/Discount (%)", "Verdict"],
    "Multiples":       ["Trailing P/E", "Forward P/E", "PEG", "P/B", "P/S", "EV/EBITDA"],
    "Profitability":   ["ROE (%)", "ROA (%)", "Profit Margin (%)", "Op. Margin (%)", "Gross Margin (%)"],
    "Growth":          ["Rev. Growth (%)", "EPS Growth (%)", "EPS (TTM)"],
    "Financial Health":["Debt/Equity", "Current Ratio", "Interest Coverage"],
    "Cash Flow":       ["FCF (B)", "FCF Yield (%)"],
    "Dividends":       ["Div. Yield (%)", "Payout Ratio (%)"],
    "Technical":       ["RSI (14)", "Beta", "52W Change (%)", "% off 52W High"],
}

VERDICT_EMOJI = {
    "Strong Buy": "🟢",
    "Buy":        "🟡",
    "Hold":       "🟠",
    "Sell":       "🔴",
}


# ============================================================
# HELPERS
# ============================================================

def safe_float(v) -> float:
    try:
        if v is None:
            return np.nan
        if isinstance(v, str):
            v = v.replace("%","").replace(",","").replace("$","").strip()
            if v in ["", "-", "N/A"]:
                return np.nan
        return float(v)
    except Exception:
        return np.nan


def clamp_growth(g: float) -> float:
    if np.isnan(g):
        return 0.06
    return float(np.clip(g, MIN_GROWTH, MAX_GROWTH))


def pct(v) -> Optional[float]:
    return round(v * 100, 2) if v is not None else None


def to_b(v) -> Optional[float]:
    return round(v / 1e9, 3) if v else None


def rsi_manual(prices: pd.Series, period: int = 14) -> Optional[float]:
    if len(prices) < period + 1:
        return None
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    val = rsi.iloc[-1]
    return None if pd.isna(val) else float(val)


def verdict(undervaluation_pct: float) -> str:
    if undervaluation_pct > 30:
        return "Strong Buy"
    if undervaluation_pct > 10:
        return "Buy"
    if undervaluation_pct > -10:
        return "Hold"
    return "Sell"


# ============================================================
# DATA SOURCES
# ============================================================

@st.cache_data(ttl=3600)
def get_finviz_data(ticker: str) -> dict:
    url = f"https://finviz.com/quote.ashx?t={ticker}"
    try:
        time.sleep(0.4)
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            return {}
        soup = BeautifulSoup(r.text, "html.parser")
        table = soup.find("table", {"class": "snapshot-table2"})
        if not table:
            return {}
        data = {}
        for row in table.find_all("tr"):
            cells = row.find_all("td")
            for i in range(0, len(cells) - 1, 2):
                k = cells[i].text.strip()
                v = cells[i + 1].text.strip()
                data[k] = v
        return data
    except Exception:
        return {}


@st.cache_data(ttl=86400)
def get_sp500() -> pd.DataFrame:
    try:
        tables = pd.read_html(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        )
        df = tables[0][["Symbol", "Security", "GICS Sector"]].copy()
        df["Symbol"] = df["Symbol"].str.replace(".", "-", regex=False)
        if len(df) > 400:
            return df
    except Exception:
        pass
    return pd.DataFrame({
        "Symbol": ["AAPL","MSFT","GOOGL","AMZN","NVDA","META","TSLA","JPM",
                   "JNJ","V","PG","XOM","UNH","HD","CVX","MRK","ABBV","PEP",
                   "COST","KO","WMT","MCD","DIS","ADBE","CRM","NFLX","AMD",
                   "CSCO","ACN","TMO","ORCL","ABT","NEE","HON","IBM","QCOM"],
        "Security": ["Apple","Microsoft","Alphabet","Amazon","NVIDIA","Meta",
                     "Tesla","JPMorgan","J&J","Visa","P&G","ExxonMobil",
                     "UnitedHealth","Home Depot","Chevron","Merck","AbbVie",
                     "PepsiCo","Costco","Coca-Cola","Walmart","McDonald's",
                     "Disney","Adobe","Salesforce","Netflix","AMD","Cisco",
                     "Accenture","Thermo Fisher","Oracle","Abbott","NextEra",
                     "Honeywell","IBM","Qualcomm"],
        "GICS Sector": (["Information Technology"]*8 + ["Health Care"]*5 +
                        ["Financials"]*4 + ["Consumer Staples"]*5 +
                        ["Consumer Discretionary"]*4 + ["Communication Services"]*4 +
                        ["Industrials"]*3 + ["Energy"]*3),
    })


@st.cache_data(ttl=86400)
def get_nasdaq100() -> List[str]:
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/Nasdaq-100")
        for tbl in tables:
            for col in ("Ticker", "Symbol"):
                if col in tbl.columns:
                    tickers = [
                        str(t).strip()
                        for t in tbl[col]
                        if isinstance(t, str) and 1 <= len(str(t).strip()) <= 5
                    ]
                    if len(tickers) > 50:
                        return tickers
    except Exception:
        pass
    return ["AAPL","MSFT","NVDA","AMZN","META","GOOGL","TSLA","AVGO",
            "COST","NFLX","AMD","ADBE","CSCO","TMUS","TXN","QCOM","INTU",
            "AMGN","ISRG","BKNG","GILD","ADP","VRTX","ADI","REGN","PANW",
            "MU","LRCX","KLAC","SNPS","CDNS","MELI","PYPL","CRWD","ORLY",
            "MRVL","WDAY","ABNB","NXPI","MAR","ADSK","CSX","MNST","FTNT"]


@st.cache_data(ttl=3600)
def get_finviz_sector_tickers(sector: str) -> List[str]:
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
    tickers, page = [], 1
    while page <= 50:
        start = (page - 1) * 20 + 1
        url = f"https://finviz.com/screener.ashx?v=111&f={sector_map[sector]}&r={start}"
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
            found = False
            for link in soup.find_all("a", {"class": "tab-link"}):
                t = link.text.strip()
                if t and 1 <= len(t) <= 5:
                    tickers.append(t)
                    found = True
            if not found:
                break
            time.sleep(0.3)
            page += 1
        except Exception:
            break
    return list(set(tickers))


# ============================================================
# STOCK DATA FETCH (rich, hybrid yfinance + Finviz)
# ============================================================

@st.cache_data(ttl=1800)
def fetch_stock_data(ticker: str) -> Optional[Dict]:
    try:
        stock = yf.Ticker(ticker)
        info  = stock.info or {}

        price = safe_float(info.get("currentPrice") or info.get("regularMarketPrice"))
        if np.isnan(price) or price <= 0:
            return None

        finviz = get_finviz_data(ticker)

        # ── Historical EPS growth ───────────────────────────
        growth = 0.06
        try:
            income = stock.get_income_stmt()
            for field in ("DilutedEPS", "BasicEPS"):
                if field in income.index:
                    eps_s = income.loc[field].dropna()[::-1]
                    if len(eps_s) >= 2 and eps_s.iloc[0] > 0 and eps_s.iloc[-1] > 0:
                        yrs = len(eps_s) - 1
                        growth = (eps_s.iloc[-1] / eps_s.iloc[0]) ** (1 / yrs) - 1
                    break
        except Exception:
            pass
        growth = clamp_growth(growth)

        # ── RSI ─────────────────────────────────────────────
        rsi_val = None
        try:
            hist = stock.history(period="3mo", auto_adjust=True)
            if len(hist) >= 30 and "Close" in hist.columns:
                if HAS_PANDAS_TA:
                    s = ta.rsi(hist["Close"], length=14)
                    v = s.iloc[-1]
                    rsi_val = None if pd.isna(v) else float(v)
                else:
                    rsi_val = rsi_manual(hist["Close"])
        except Exception:
            pass

        # ── Convenience derived fields ────────────────────────
        mc   = safe_float(info.get("marketCap"))
        fcf  = safe_float(info.get("freeCashflow"))
        rev  = safe_float(info.get("totalRevenue"))
        ebit = safe_float(info.get("ebit"))
        interest = safe_float(info.get("interestExpense"))
        ebitda = safe_float(info.get("ebitda"))
        hi52   = safe_float(info.get("fiftyTwoWeekHigh"))
        div_rate = safe_float(info.get("dividendRate")) or safe_float(info.get("trailingAnnualDividendRate")) or 0.0
        shares = safe_float(info.get("sharesOutstanding"))
        book_v = safe_float(info.get("bookValue"))
        fwd_pe = safe_float(info.get("forwardPE")) or 15.0
        hist_pe = safe_float(info.get("trailingPE")) or fwd_pe
        div_yield = safe_float(info.get("dividendYield") or 0) * 100
        roe = safe_float(info.get("returnOnEquity") or 0) * 100
        roa = safe_float(info.get("returnOnAssets") or 0) * 100
        gross_m = safe_float(info.get("grossMargins") or 0) * 100
        op_m    = safe_float(info.get("operatingMargins") or 0) * 100
        profit_m = safe_float(info.get("profitMargins") or 0) * 100
        rev_g   = safe_float(info.get("revenueGrowth") or 0) * 100
        eps_g   = safe_float(info.get("earningsQuarterlyGrowth") or 0) * 100
        de      = safe_float(info.get("debtToEquity"))
        cr      = safe_float(info.get("currentRatio"))
        beta    = safe_float(info.get("beta"))
        peg     = safe_float(info.get("pegRatio"))
        pb      = safe_float(info.get("priceToBook"))
        ps      = safe_float(info.get("priceToSalesTrailing12Months"))
        ev_ebitda = safe_float(info.get("enterpriseToEbitda"))
        ev      = safe_float(info.get("enterpriseValue"))
        w52chg  = safe_float(info.get("52WeekChange") or 0) * 100

        # Finviz overrides / supplements
        if finviz:
            roe      = safe_float(finviz.get("ROE", roe))
            hist_pe  = safe_float(finviz.get("P/E", hist_pe))
            beta     = safe_float(finviz.get("Beta", beta))

        interest_cov = None
        if not np.isnan(ebit) and not np.isnan(interest) and interest != 0:
            interest_cov = round(ebit / abs(interest), 2)

        fcf_yield = None
        if not np.isnan(fcf) and not np.isnan(mc) and mc > 0:
            fcf_yield = round(fcf / mc * 100, 2)

        pct_off_hi = None
        if not np.isnan(hi52) and hi52 > 0:
            pct_off_hi = round((price / hi52 - 1) * 100, 2)

        ebitda_margin = None
        if not np.isnan(ebitda) and not np.isnan(rev) and rev > 0:
            ebitda_margin = round(ebitda / rev * 100, 2)

        eps_ttm   = safe_float(info.get("trailingEps"))
        fwd_eps   = max(safe_float(info.get("forwardEps") or eps_ttm or 1.0), 0.01)
        debt      = safe_float(info.get("totalDebt")) or 0
        cash      = safe_float(info.get("totalCash")) or 0
        total_assets = safe_float(info.get("totalAssets")) or 0
        invested_cap = total_assets - safe_float(info.get("totalCurrentLiabilities") or 0)
        nopat        = safe_float(info.get("operatingCashflow") or 0) * 0.75

        return {
            # identifiers
            "Symbol":       ticker,
            "Name":         info.get("longName") or info.get("shortName", ticker),
            "Sector":       info.get("sector", "N/A"),
            "Industry":     info.get("industry", "N/A"),
            # price
            "Price ($)":    price,
            "52W High":     hi52,
            "52W Low":      safe_float(info.get("fiftyTwoWeekLow")),
            "52W Change (%)": round(w52chg, 2),
            "% off 52W High": pct_off_hi,
            # size
            "Market Cap (B)": to_b(mc),
            "Enterprise Value (B)": to_b(ev),
            # multiples
            "Trailing P/E": round(hist_pe, 2) if not np.isnan(hist_pe) else None,
            "Forward P/E":  round(fwd_pe, 2)  if not np.isnan(fwd_pe)  else None,
            "PEG":          round(peg, 2)      if not np.isnan(peg)     else None,
            "P/B":          round(pb, 2)       if not np.isnan(pb)      else None,
            "P/S":          round(ps, 2)       if not np.isnan(ps)      else None,
            "EV/EBITDA":    round(ev_ebitda,2) if not np.isnan(ev_ebitda) else None,
            # profitability
            "ROE (%)":           round(roe, 2)     if not np.isnan(roe)     else None,
            "ROA (%)":           round(roa, 2)     if not np.isnan(roa)     else None,
            "Gross Margin (%)":  round(gross_m, 2) if not np.isnan(gross_m) else None,
            "Op. Margin (%)":    round(op_m, 2)    if not np.isnan(op_m)    else None,
            "Profit Margin (%)": round(profit_m,2) if not np.isnan(profit_m) else None,
            "EBITDA Margin (%)": ebitda_margin,
            # growth
            "EPS (TTM)":       round(eps_ttm, 2) if not np.isnan(eps_ttm) else None,
            "Forward EPS":     round(fwd_eps, 2),
            "Rev. Growth (%)": round(rev_g, 2)   if not np.isnan(rev_g)   else None,
            "EPS Growth (%)":  round(eps_g, 2)   if not np.isnan(eps_g)   else None,
            "Historical Growth": growth,
            # health
            "Debt/Equity":        round(de, 2)           if not np.isnan(de)  else None,
            "Current Ratio":      round(cr, 2)           if not np.isnan(cr)  else None,
            "Interest Coverage":  interest_cov,
            # cash flow
            "FCF (B)":     to_b(fcf),
            "FCF Yield (%)": fcf_yield,
            # dividends
            "Div. Yield (%)":  round(div_yield, 2),
            "Payout Ratio (%)": round(safe_float(info.get("payoutRatio") or 0) * 100, 2),
            "Div. Per Share":   div_rate,
            # technical
            "RSI (14)": round(rsi_val, 1) if rsi_val is not None else None,
            "Beta":     round(beta, 2)   if not np.isnan(beta)   else None,
            # raw valuation inputs
            "_eps_ttm":       eps_ttm,
            "_fwd_eps":       fwd_eps,
            "_book_value":    book_v,
            "_shares":        shares,
            "_fcf":           fcf,
            "_debt":          debt,
            "_cash":          cash,
            "_hist_pe":       hist_pe,
            "_fwd_pe":        fwd_pe,
            "_roe":           roe,
            "_div_rate":      div_rate,
            "_mc":            mc,
            "_invested_cap":  invested_cap,
            "_nopat":         nopat,
        }

    except Exception:
        return None


# ============================================================
# VALUATION MODELS  (all return a single float: intrinsic per share)
# ============================================================

def _graham_number(d: dict) -> float:
    """√(22.5 × EPS × Book Value) — Benjamin Graham"""
    eps = d["_eps_ttm"]
    bv  = d["_book_value"]
    if any(np.isnan(x) or x <= 0 for x in [eps, bv]):
        return np.nan
    return math.sqrt(22.5 * eps * bv)


def _lynch_method(d: dict, growth_override: float = None) -> float:
    """Peter Lynch: EPS × (Growth% + Div%)  →  Fair PE × EPS"""
    eps  = d["_eps_ttm"]
    g    = (growth_override * 100) if growth_override else (d["Historical Growth"] * 100)
    div  = d["Div. Yield (%)"]
    if np.isnan(eps) or eps <= 0:
        return np.nan
    fair_pe = float(np.clip(g + div, 8, 40))
    return max(eps * fair_pe, 0)


def _two_stage_dcf(d: dict, discount: float = 0.10, terminal: float = 0.03,
                   years: int = DCF_YEARS, growth_override: float = None) -> float:
    """Two-Stage DCF on free cash flow"""
    fcf    = d["_fcf"]
    shares = d["_shares"]
    debt   = d["_debt"]
    cash   = d["_cash"]
    g      = growth_override if growth_override else d["Historical Growth"]
    g      = clamp_growth(g)

    if any(np.isnan(x) or x <= 0 for x in [fcf, shares]) or discount <= terminal:
        return np.nan

    pv = 0.0
    cf = fcf
    for yr in range(1, years + 1):
        cf  *= (1 + g)
        pv  += cf / (1 + discount) ** yr

    terminal_value = (cf * (1 + terminal)) / (discount - terminal)
    pv_terminal    = terminal_value / (1 + discount) ** years

    equity = pv + pv_terminal - (debt or 0) + (cash or 0)
    return max(equity / shares, 0)


def _ddm(d: dict, growth_override: float = None) -> float:
    """Gordon Growth Dividend Discount Model"""
    div  = d["_div_rate"]
    g    = growth_override if growth_override else min(d["Historical Growth"], 0.08)
    r    = 0.10  # required return (fixed; users can't break it)
    if np.isnan(div) or div <= 0 or r <= g:
        return np.nan
    return max(div * (1 + g) / (r - g), 0)


def _residual_income(d: dict, growth_override: float = None) -> float:
    """Residual Income Model — growth capped below COE to keep model finite."""
    bv  = d["_book_value"]
    roe = d["_roe"] / 100
    g   = growth_override if growth_override else d["Historical Growth"]
    coe = 0.10  # cost of equity
    # Ensure spread is positive; cap growth 0.5pp below COE
    g   = min(clamp_growth(g), coe - 0.005)
    if np.isnan(bv) or bv <= 0:
        return np.nan
    ri  = bv * (roe - coe)
    pv_ri = sum(ri * (1 + g) ** t / (1 + coe) ** t for t in range(1, 11))
    return max(bv + pv_ri, 0)


def _owner_earnings(d: dict, growth_override: float = None) -> float:
    """Owner Earnings (Buffett): FCF per share, discounted. Growth capped below r."""
    fcf    = d["_fcf"]
    shares = d["_shares"]
    r      = 0.10
    g      = growth_override if growth_override else d["Historical Growth"]
    g      = min(clamp_growth(g), r - 0.005)  # keep terminal value finite
    if any(np.isnan(x) or x <= 0 for x in [fcf, shares]):
        return np.nan
    oe = fcf / shares
    pv = sum(oe * (1 + g) ** t / (1 + r) ** t for t in range(1, 11))
    tv = (oe * (1 + g) ** 10 * (1 + g)) / (r - g) / (1 + r) ** 10
    return max(pv + tv, 0)


def _core_pe_valuation(d: dict, growth_override: float = None) -> float:
    """PE-based: project Forward EPS × historical PE, discount back"""
    fwd_eps = d["_fwd_eps"]
    hist_pe = d["_hist_pe"]
    g       = growth_override if growth_override else d["Historical Growth"]
    g       = clamp_growth(g)
    r       = 0.10
    years   = 5
    if np.isnan(fwd_eps) or fwd_eps <= 0 or np.isnan(hist_pe) or hist_pe <= 0:
        return np.nan
    future_eps   = fwd_eps * (1 + g) ** years
    future_price = future_eps * hist_pe
    return max(future_price / (1 + r) ** years, 0)


def _nav(d: dict) -> float:
    """Net Asset Value — book value with 30% discount"""
    bv = d["_book_value"]
    if np.isnan(bv) or bv <= 0:
        return np.nan
    return bv * 0.70


MODEL_FN = {
    "Graham Number":           _graham_number,
    "Lynch Method":            _lynch_method,
    "Two-Stage DCF":           _two_stage_dcf,
    "Dividend Discount Model (DDM)": _ddm,
    "Residual Income":         _residual_income,
    "Owner Earnings (Buffett)": _owner_earnings,
    "Core Valuation (PE-based)": _core_pe_valuation,
    "Asset-Based (NAV)":       _nav,
}


def run_valuation(
    data: dict,
    selected_models: List[str],
    mos_pct: float = DEFAULT_MOS,
    growth_override: float = None,
) -> Dict:
    """
    Run each selected model, remove outliers, take median.
    Returns per-model values + composite intrinsic + safe buy price.
    """
    per_model = {}
    for m in selected_models:
        fn = MODEL_FN.get(m)
        if fn is None:
            continue
        try:
            if m in ("Graham Number", "Asset-Based (NAV)"):
                val = fn(data)
            else:
                val = fn(data, growth_override)
        except Exception:
            val = np.nan
        per_model[m] = val

    # Remove outliers: must be > 0 and < price × OUTLIER_MULTIPLE
    price = data["Price ($)"]
    clean = [
        v for v in per_model.values()
        if not np.isnan(v) and v > 0 and v < price * OUTLIER_MULTIPLE
    ]

    if not clean:
        intrinsic = np.nan
    else:
        intrinsic = float(np.median(clean))

    safe_buy   = intrinsic * (1 - mos_pct / 100) if not np.isnan(intrinsic) else np.nan
    premium    = ((price - intrinsic) / intrinsic * 100) if not np.isnan(intrinsic) and intrinsic > 0 else np.nan
    under      = -premium if not np.isnan(premium) else np.nan

    return {
        "per_model":  per_model,
        "intrinsic":  intrinsic,
        "safe_buy":   safe_buy,
        "premium":    premium,       # positive = overvalued
        "undervaluation": under,     # positive = undervalued
        "verdict":    verdict(under) if not np.isnan(under) else "N/A",
    }


# ============================================================
# SCENARIO DCF
# ============================================================

def scenario_dcf(data: dict) -> Dict[str, float]:
    scenarios = {
        "Bear (12% disc, 2% term)": (0.12, 0.02),
        "Base (10% disc, 3% term)": (0.10, 0.03),
        "Bull (8% disc, 4% term)":  (0.08, 0.04),
    }
    return {
        name: _two_stage_dcf(data, discount=d, terminal=t)
        for name, (d, t) in scenarios.items()
    }


# ============================================================
# SENSITIVITY ANALYSIS
# ============================================================

def sensitivity_table(data: dict, steps: int = 7) -> pd.DataFrame:
    """Vary discount rate (rows) vs terminal growth (cols) for DCF."""
    discounts  = np.linspace(0.07, 0.14, steps)
    terminals  = np.linspace(0.01, 0.05, steps)
    matrix = []
    for d in discounts:
        row = []
        for t in terminals:
            if d > t:
                row.append(_two_stage_dcf(data, discount=d, terminal=t))
            else:
                row.append(np.nan)
        matrix.append(row)
    return pd.DataFrame(
        matrix,
        index=[f"{d*100:.0f}%" for d in discounts],
        columns=[f"{t*100:.0f}%" for t in terminals],
    )


# ============================================================
# CHARTS
# ============================================================

def valuation_bar_chart(price: float, per_model: Dict[str, float]) -> go.Figure:
    names, vals, colors = [], [], []
    for k, v in per_model.items():
        if not np.isnan(v) and v > 0:
            names.append(k)
            vals.append(v)
            colors.append("#2ecc71" if v > price else "#e74c3c")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=names, y=vals,
        marker_color=colors,
        text=[f"${x:.2f}" for x in vals],
        textposition="outside",
    ))
    fig.add_hline(
        y=price, line_dash="dash", line_color="#3498db",
        annotation_text=f"Market Price ${price:.2f}",
        annotation_position="top right",
    )
    fig.update_layout(
        title="Intrinsic Value by Model vs. Market Price",
        yaxis_title="Price ($)",
        height=420,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12),
    )
    return fig


def scenario_chart(price: float, scenarios: Dict[str, float]) -> go.Figure:
    names  = list(scenarios.keys())
    vals   = [v if not np.isnan(v) else 0 for v in scenarios.values()]
    colors = ["#e74c3c", "#f39c12", "#2ecc71"]

    fig = go.Figure(go.Bar(
        x=names, y=vals,
        marker_color=colors,
        text=[f"${v:.2f}" if v > 0 else "N/A" for v in vals],
        textposition="outside",
    ))
    fig.add_hline(y=price, line_dash="dot", line_color="#3498db",
                  annotation_text=f"Price ${price:.2f}")
    fig.update_layout(title="DCF Scenario Analysis", height=360,
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    return fig


def scatter_screener_chart(df: pd.DataFrame) -> go.Figure:
    needed = {"Symbol", "Rev. Growth (%)", "Premium/Discount (%)", "Market Cap (B)"}
    if not needed.issubset(df.columns):
        return go.Figure()

    plot_df = df.dropna(subset=["Rev. Growth (%)", "Premium/Discount (%)"])

    fig = px.scatter(
        plot_df,
        x="Rev. Growth (%)",
        y="Premium/Discount (%)",
        size=plot_df["Market Cap (B)"].clip(lower=0.1).fillna(1),
        color="Premium/Discount (%)",
        color_continuous_scale="RdYlGn_r",
        hover_name="Symbol",
        hover_data={"Trailing P/E": True, "ROE (%)": True, "Verdict": True},
        title="Revenue Growth vs. Premium/Discount  (bubble = market cap)",
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(height=450, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    return fig


# ============================================================
# SINGLE STOCK UI
# ============================================================

def single_stock_ui():
    st.header("🔍 Single Stock Valuation")

    col_l, col_r = st.columns([1, 3])

    with col_l:
        st.markdown("### ⚙️ Settings")
        ticker = st.text_input("Ticker Symbol", "AAPL").upper().strip()

        selected_models = st.multiselect(
            "Valuation Models",
            ALL_MODELS,
            default=["Graham Number", "Lynch Method", "Two-Stage DCF", "Core Valuation (PE-based)"],
        )

        mos = st.slider("Margin of Safety (%)", 0, 50, 25, 5)

        with st.expander("🛠️ Override Parameters (optional)"):
            override_growth = st.number_input(
                "Custom Growth Rate (%)",
                min_value=0.0, max_value=50.0, value=0.0, step=0.5,
                help="Leave at 0 to use historical EPS CAGR",
            )
            custom_discount = st.number_input("DCF Discount Rate (%)", 5.0, 20.0, 10.0, 0.5)
            custom_terminal = st.number_input("DCF Terminal Growth (%)", 0.5, 6.0, 3.0, 0.5)

        analyze = st.button("Analyze", type="primary", use_container_width=True)

    with col_r:
        if not analyze:
            st.info("Configure settings on the left and click **Analyze**.")
            return

        if not selected_models:
            st.warning("Select at least one valuation model.")
            return

        with st.spinner(f"Fetching data for {ticker}…"):
            data = fetch_stock_data(ticker)

        if not data:
            st.error(f"Could not fetch data for **{ticker}**. Check the ticker and try again.")
            return

        g_override = (override_growth / 100) if override_growth > 0 else None

        # patch DCF params into data for the models that need them
        data["_custom_discount"] = custom_discount / 100
        data["_custom_terminal"] = custom_terminal / 100

        result = run_valuation(data, selected_models, mos_pct=mos, growth_override=g_override)

        price     = data["Price ($)"]
        intrinsic = result["intrinsic"]
        premium   = result["premium"]
        safe_buy  = result["safe_buy"]
        verd      = result["verdict"]

        # ── Header ───────────────────────────────────────────
        st.markdown(f"## {data['Name']} · `{ticker}`")
        st.caption(f"**Sector:** {data['Sector']}  |  **Industry:** {data['Industry']}")

        if not np.isnan(intrinsic):
            emoji = VERDICT_EMOJI.get(verd, "⚪")
            st.markdown(f"### {emoji} {verd}")

        # ── Key metrics ───────────────────────────────────────
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Current Price",  f"${price:.2f}")
        c2.metric("Intrinsic Value",f"${intrinsic:.2f}" if not np.isnan(intrinsic) else "N/A")
        c3.metric("Safe Buy Price", f"${safe_buy:.2f}" if not np.isnan(safe_buy) else "N/A")
        premium_str = f"{premium:+.1f}%" if not np.isnan(premium) else "N/A"
        delta_color = "inverse" if not np.isnan(premium) and premium > 0 else "normal"
        c4.metric("Premium/Discount", premium_str)
        c5.metric("Margin of Safety", f"{mos}%")

        st.divider()

        # ── Valuation bar chart ───────────────────────────────
        st.plotly_chart(
            valuation_bar_chart(price, result["per_model"]),
            use_container_width=True,
        )

        # ── Per-model table ───────────────────────────────────
        model_df = pd.DataFrame([
            {
                "Model": m,
                "Intrinsic ($)": f"${v:.2f}" if not np.isnan(v) else "N/A",
                "vs. Price": f"{((v - price)/price*100):+.1f}%" if not np.isnan(v) else "N/A",
                "Used in Median": "✅" if not np.isnan(v) and v > 0 and v < price * OUTLIER_MULTIPLE else "❌ (outlier removed)",
            }
            for m, v in result["per_model"].items()
        ])
        st.subheader("Model Breakdown")
        st.dataframe(model_df, use_container_width=True, hide_index=True)

        st.divider()

        # ── Scenario DCF ──────────────────────────────────────
        st.subheader("📊 DCF Scenario Analysis")
        scens = scenario_dcf(data)
        st.plotly_chart(scenario_chart(price, scens), use_container_width=True)

        scen_df = pd.DataFrame([
            {
                "Scenario": k,
                "DCF Value ($)": f"${v:.2f}" if not np.isnan(v) else "N/A",
                "vs. Price": f"{((v-price)/price*100):+.1f}%" if not np.isnan(v) else "N/A",
            }
            for k, v in scens.items()
        ])
        st.dataframe(scen_df, use_container_width=True, hide_index=True)

        # ── Sensitivity heatmap ───────────────────────────────
        st.subheader("🌡️ DCF Sensitivity (Discount Rate × Terminal Growth)")
        st.caption("Rows = discount rate · Columns = terminal growth rate")
        sens = sensitivity_table(data)
        price_mask = sens.map(lambda v: not np.isnan(v) and v > price)
        styled = (
            sens.style
            .background_gradient(cmap="RdYlGn", axis=None, vmin=price*0.4, vmax=price*2)
            .format("${:.2f}", na_rep="—")
        )
        st.dataframe(styled, use_container_width=True)
        st.caption("🟢 Green = intrinsic > market price (potentially undervalued)")

        # ── Fundamentals snapshot ─────────────────────────────
        with st.expander("📋 Full Fundamentals Snapshot"):
            snapshot_fields = [
                "Price ($)", "Market Cap (B)", "Trailing P/E", "Forward P/E", "PEG",
                "P/B", "P/S", "EV/EBITDA", "Beta",
                "ROE (%)", "ROA (%)", "Gross Margin (%)", "Op. Margin (%)", "Profit Margin (%)",
                "Rev. Growth (%)", "EPS Growth (%)", "EPS (TTM)", "Forward EPS",
                "Debt/Equity", "Current Ratio", "Interest Coverage",
                "FCF (B)", "FCF Yield (%)",
                "Div. Yield (%)", "Payout Ratio (%)",
                "RSI (14)", "52W Change (%)", "% off 52W High",
                "Historical Growth",
            ]
            snap = {k: data.get(k) for k in snapshot_fields if k in data}
            snap_df = pd.DataFrame(snap.items(), columns=["Metric", "Value"])
            snap_df["Value"] = snap_df["Value"].apply(
                lambda v: f"{v:.2f}" if isinstance(v, float) and not np.isnan(v) else (str(v) if v is not None else "N/A")
            )
            st.dataframe(snap_df, use_container_width=True, hide_index=True)


# ============================================================
# SCREENER
# ============================================================

def _process_screener_row(
    symbol: str,
    selected_models: List[str],
    mos_pct: float,
    growth_override: Optional[float],
) -> Tuple[Optional[Dict], Optional[str]]:
    data = fetch_stock_data(symbol)
    if not data:
        return None, "No data"
    try:
        # Base row — always populated from live market data, no valuation required
        row = {
            "Symbol":            data["Symbol"],
            "Name":              data["Name"],
            "Sector":            data["Sector"],
            "Price ($)":         data["Price ($)"],
            "Market Cap (B)":    data["Market Cap (B)"],
            "Trailing P/E":      data["Trailing P/E"],
            "Forward P/E":       data["Forward P/E"],
            "PEG":               data["PEG"],
            "P/B":               data["P/B"],
            "P/S":               data["P/S"],
            "EV/EBITDA":         data["EV/EBITDA"],
            "ROE (%)":           data["ROE (%)"],
            "ROA (%)":           data["ROA (%)"],
            "Gross Margin (%)":  data["Gross Margin (%)"],
            "Op. Margin (%)":    data["Op. Margin (%)"],
            "Profit Margin (%)": data["Profit Margin (%)"],
            "EPS (TTM)":         data["EPS (TTM)"],
            "Rev. Growth (%)":   data["Rev. Growth (%)"],
            "EPS Growth (%)":    data["EPS Growth (%)"],
            "Debt/Equity":       data["Debt/Equity"],
            "Current Ratio":     data["Current Ratio"],
            "Interest Coverage": data["Interest Coverage"],
            "FCF (B)":           data["FCF (B)"],
            "FCF Yield (%)":     data["FCF Yield (%)"],
            "Div. Yield (%)":    data["Div. Yield (%)"],
            "Payout Ratio (%)":  data["Payout Ratio (%)"],
            "Beta":              data["Beta"],
            "RSI (14)":          data["RSI (14)"],
            "52W Change (%)":    data["52W Change (%)"],
            "% off 52W High":    data["% off 52W High"],
        }
        # Valuation columns — only computed when the user selected models
        if selected_models:
            result = run_valuation(data, selected_models, mos_pct=mos_pct, growth_override=growth_override)
            row["Intrinsic ($)"]        = round(result["intrinsic"], 2)      if not np.isnan(result["intrinsic"])       else None
            row["Safe Buy ($)"]         = round(result["safe_buy"],  2)      if not np.isnan(result["safe_buy"])        else None
            row["Premium/Discount (%)"] = round(result["premium"],   1)      if not np.isnan(result["premium"])         else None
            row["Undervaluation (%)"]   = round(result["undervaluation"], 1) if not np.isnan(result["undervaluation"])  else None
            row["Verdict"]              = result["verdict"]
        return row, None
    except Exception as e:
        return None, str(e)


def screener_ui():
    st.header("📡 Multi-Stock Screener")
    st.caption("Screen hundreds of stocks with fundamental filters + intrinsic value. Not financial advice.")

    # ── Watchlist ──────────────────────────────────────────────────────────
    if "watchlist" not in st.session_state:
        st.session_state["watchlist"] = set()
    wl = st.session_state["watchlist"]
    if wl:
        with st.expander(f"⭐ Watchlist ({len(wl)})", expanded=False):
            st.write(", ".join(sorted(wl)))
            if st.button("Clear watchlist"):
                st.session_state["watchlist"] = set()
                st.rerun()

    # ── 1. Universe ────────────────────────────────────────────────────────
    with st.expander("📊 1. Universe", expanded=True):
        universe_opt = st.radio(
            "Source",
            ["S&P 500", "NASDAQ 100", "Finviz Sector", "Custom"],
            horizontal=True,
        )

        universe_tickers: List[str] = []
        sp500_df: Optional[pd.DataFrame] = None

        if universe_opt == "S&P 500":
            if st.button("Load S&P 500"):
                with st.spinner("Loading…"):
                    sp500_df = get_sp500()
                    st.session_state["_sp500_df"] = sp500_df
                    st.session_state["_universe"] = sp500_df["Symbol"].tolist()
                    st.success(f"✅ {len(sp500_df)} stocks loaded")

            if "_sp500_df" in st.session_state:
                sp500_df = st.session_state["_sp500_df"]
                sectors = sorted(sp500_df["GICS Sector"].unique().tolist())
                sel_sectors = st.multiselect("Filter sectors", sectors, default=sectors)
                filtered = sp500_df[sp500_df["GICS Sector"].isin(sel_sectors)]
                st.session_state["_universe"] = filtered["Symbol"].tolist()

        elif universe_opt == "NASDAQ 100":
            if st.button("Load NASDAQ 100"):
                with st.spinner("Loading…"):
                    tickers = get_nasdaq100()
                    st.session_state["_universe"] = tickers
                    st.success(f"✅ {len(tickers)} stocks loaded")

        elif universe_opt == "Finviz Sector":
            sector = st.selectbox("Sector", [
                "Technology","Healthcare","Financials","Energy",
                "Consumer Discretionary","Consumer Staples","Industrials",
                "Basic Materials","Communication Services","Utilities","Real Estate",
            ])
            if st.button("Load Sector"):
                with st.spinner(f"Scraping Finviz for {sector}…"):
                    tickers = get_finviz_sector_tickers(sector)
                    if tickers:
                        st.session_state["_universe"] = tickers
                        st.success(f"✅ {len(tickers)} stocks found")
                    else:
                        st.error("Could not load sector tickers. Finviz may be rate-limiting.")

        else:  # Custom
            raw = st.text_area(
                "Enter tickers (comma or space separated)",
                placeholder="AAPL MSFT GOOGL AMZN NVDA",
                height=80,
            )
            tickers = [t.strip().upper() for t in raw.replace(",", " ").split() if t.strip()]
            if tickers:
                st.session_state["_universe"] = tickers
                st.info(f"{len(tickers)} tickers entered")

        universe_tickers = st.session_state.get("_universe", [])
        if universe_tickers:
            max_n = st.slider("Max stocks to screen", 10, min(500, len(universe_tickers)),
                              min(100, len(universe_tickers)))
            universe_tickers = universe_tickers[:max_n]
            st.caption(f"Will screen **{len(universe_tickers)}** stocks")

    if not universe_tickers:
        st.info("👆 Choose a universe above and click Load.")
        return

    # ── 2. Valuation ───────────────────────────────────────────────────────
    with st.expander("💡 2. Valuation Settings (optional)", expanded=False):
        st.caption(
            "Enable this to add intrinsic value columns (Intrinsic $, Safe Buy, Verdict). "
            "You can also screen purely on fundamentals — no models required."
        )
        enable_valuation = st.checkbox("Enable intrinsic value calculation", value=False)
        if enable_valuation:
            selected_models = st.multiselect(
                "Models to use",
                ALL_MODELS,
                default=["Graham Number", "Lynch Method", "Two-Stage DCF", "Core Valuation (PE-based)"],
            )
            mos = st.slider("Margin of Safety (%)", 0, 50, 25, 5)
            override_g = st.number_input(
                "Custom Growth Override (% — leave 0 to use historical)",
                min_value=0.0, max_value=50.0, value=0.0, step=0.5,
            )
        else:
            selected_models = []
            mos = 25
            override_g = 0.0

    # ── 3. Filters ─────────────────────────────────────────────────────────
    with st.expander("⚙️ 3. Filters", expanded=False):
        st.caption("Leave fields blank to skip that filter.")
        fc1, fc2 = st.columns(2)
        with fc1:
            f_verdict  = st.multiselect("Verdict (requires valuation enabled)", ["Strong Buy","Buy","Hold","Sell"], default=[])
            f_pe_max   = st.number_input("P/E ≤", value=None, format="%.1f")
            f_roe_min  = st.number_input("ROE (%) ≥", value=None, format="%.1f")
            f_de_max   = st.number_input("Debt/Equity ≤", value=None, format="%.2f")
            f_dy_min   = st.number_input("Div. Yield (%) ≥", value=None, format="%.2f")
        with fc2:
            f_mc_min   = st.number_input("Market Cap (B) ≥", value=None, format="%.2f")
            f_pm_min   = st.number_input("Profit Margin (%) ≥", value=None, format="%.1f")
            f_rg_min   = st.number_input("Rev. Growth (%) ≥", value=None, format="%.1f")
            f_rsi_max  = st.number_input("RSI (14) ≤", value=None, min_value=0.0, max_value=100.0, format="%.0f")
            f_under_min = st.number_input("Undervaluation (%) ≥", value=None, format="%.1f")

    # ── 4. Columns ─────────────────────────────────────────────────────────
    with st.expander("👁️ 4. Visible Column Groups", expanded=False):
        visible_groups = st.multiselect(
            "Show groups",
            list(COLUMN_GROUPS.keys()),
            default=["Core", "Valuation", "Multiples", "Profitability", "Growth", "Technical"],
        )

    # ── Run ─────────────────────────────────────────────────────────────────
    if not st.button("🚀 Run Screener", type="primary"):
        return

    g_override = (override_g / 100) if override_g > 0 else None

    results: List[Dict] = []
    errors: List[str]   = []

    prog    = st.progress(0)
    stat    = st.empty()
    t_start = time.time()

    with ThreadPoolExecutor(max_workers=MAX_THREADS) as ex:
        futures = {
            ex.submit(_process_screener_row, sym, selected_models, mos, g_override): sym
            for sym in universe_tickers
        }
        done = 0
        for fut in as_completed(futures):
            done += 1
            prog.progress(done / len(universe_tickers))
            sym = futures[fut]
            elapsed  = time.time() - t_start
            eta      = int(elapsed / done * (len(universe_tickers) - done)) if done > 1 else "?"
            stat.text(f"Screening {sym}… ({done}/{len(universe_tickers)})  ETA ~{eta}s")
            try:
                row, err = fut.result()
                if row:
                    results.append(row)
                elif err:
                    errors.append(f"{sym}: {err}")
            except Exception as e:
                errors.append(f"{sym}: {e}")

    prog.empty()
    stat.empty()

    if errors:
        with st.expander(f"⚠️ {len(errors)} ticker(s) failed"):
            for e in errors:
                st.caption(e)

    if not results:
        st.error("No data retrieved. Try fewer stocks or check your connection.")
        return

    df = pd.DataFrame(results)

    # ── Apply filters ───────────────────────────────────────────────────────
    mask = pd.Series([True] * len(df))

    # Verdict only available when valuation models were run
    if f_verdict and "Verdict" in df.columns:
        mask &= df["Verdict"].isin(f_verdict)

    def apply_filter(col, lo=None, hi=None):
        nonlocal mask
        if col not in df.columns:
            return
        if lo is not None:
            mask &= df[col].fillna(-np.inf) >= lo
        if hi is not None:
            mask &= df[col].fillna(np.inf)  <= hi

    apply_filter("Market Cap (B)",    lo=f_mc_min)
    apply_filter("Trailing P/E",      hi=f_pe_max)
    apply_filter("ROE (%)",           lo=f_roe_min)
    apply_filter("Profit Margin (%)", lo=f_pm_min)
    apply_filter("Rev. Growth (%)",   lo=f_rg_min)
    apply_filter("Debt/Equity",       hi=f_de_max)
    apply_filter("Div. Yield (%)",    lo=f_dy_min)
    apply_filter("RSI (14)",          hi=f_rsi_max)
    apply_filter("Undervaluation (%)",lo=f_under_min)

    final = df[mask].copy()

    if final.empty:
        st.warning("No stocks pass the current filters. Try relaxing criteria.")
        return

    # ── Summary cards ──────────────────────────────────────────────────────
    st.success(f"✅ **{len(final)} stocks** matched out of {len(df)} screened")

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Matched",       str(len(final)))
    m2.metric("Avg P/E",       f"{final['Trailing P/E'].median():.1f}" if "Trailing P/E" in final else "N/A")
    m3.metric("Median ROE",    f"{final['ROE (%)'].median():.1f}%"     if "ROE (%)" in final else "N/A")
    m4.metric("Avg Rev Growth",f"{final['Rev. Growth (%)'].median():.1f}%" if "Rev. Growth (%)" in final else "N/A")
    m5.metric("Avg Div Yield", f"{final['Div. Yield (%)'].median():.1f}%" if "Div. Yield (%)" in final else "N/A")
    m6.metric("Avg Beta",      f"{final['Beta'].median():.2f}"         if "Beta" in final else "N/A")

    # ── Column selection ───────────────────────────────────────────────────
    visible_cols: List[str] = []
    for grp in visible_groups:
        for col in COLUMN_GROUPS.get(grp, []):
            if col in final.columns and col not in visible_cols:
                visible_cols.append(col)
    # Always include Symbol
    if "Symbol" not in visible_cols:
        visible_cols = ["Symbol"] + visible_cols
    display_df = final[[c for c in visible_cols if c in final.columns]].copy()

    # ── Sort ───────────────────────────────────────────────────────────────
    sort_col = st.selectbox(
        "Sort by",
        ["Premium/Discount (%)"] + [c for c in display_df.columns if c != "Symbol"],
        index=0,
    )
    asc = st.checkbox("Ascending", value=True)
    if sort_col in display_df.columns:
        display_df = display_df.sort_values(sort_col, ascending=asc, na_position="last")

    # ── Heatmap styling ────────────────────────────────────────────────────
    num_cols = display_df.select_dtypes(include=[np.number]).columns.tolist()
    bad_if_high = {
        "Trailing P/E","Forward P/E","P/B","P/S","EV/EBITDA","Debt/Equity",
        "Payout Ratio (%)","Beta","Premium/Discount (%)",
    }

    def _style(styler):
        for col in num_cols:
            cmap = "RdYlGn_r" if col in bad_if_high else "RdYlGn"
            try:
                styler = styler.background_gradient(subset=[col], cmap=cmap, axis=0)
            except Exception:
                pass
        return styler

    try:
        styled = display_df.style.pipe(_style).format(
            {c: "{:.2f}" for c in num_cols}, na_rep="—"
        )
        st.dataframe(styled, height=560, use_container_width=True)
    except Exception:
        st.dataframe(display_df, height=560, use_container_width=True)

    # ── Scatter chart ──────────────────────────────────────────────────────
    with st.expander("📈 Scatter: Growth vs. Premium/Discount", expanded=True):
        st.plotly_chart(scatter_screener_chart(final), use_container_width=True)

    # ── Sector breakdown ───────────────────────────────────────────────────
    with st.expander("🗂️ Sector Breakdown"):
        if "Sector" in final.columns:
            counts = final["Sector"].value_counts().reset_index()
            counts.columns = ["Sector","Count"]
            fig = px.bar(counts, x="Sector", y="Count", color="Count",
                         color_continuous_scale="Blues")
            st.plotly_chart(fig, use_container_width=True)

    # ── Watchlist ──────────────────────────────────────────────────────────
    with st.expander("⭐ Add to Watchlist"):
        to_add = st.multiselect("Select symbols", final["Symbol"].tolist())
        if st.button("Add to Watchlist"):
            for s in to_add:
                st.session_state["watchlist"].add(s)
            st.success(f"Added {len(to_add)} stock(s).")
            st.rerun()

    # ── Export ─────────────────────────────────────────────────────────────
    st.markdown("#### 📥 Export")
    st.download_button(
        "⬇️ Download CSV",
        data=final.to_csv(index=False),
        file_name=f"screener_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )


# ============================================================
# COMPARE MODE
# ============================================================

def compare_ui():
    st.header("⚖️ Side-by-Side Stock Comparison")
    raw = st.text_input("Enter 2–5 tickers (comma-separated)", "AAPL, MSFT, GOOGL")
    tickers = [t.strip().upper() for t in raw.split(",") if t.strip()]

    selected_models = st.multiselect(
        "Models",
        ALL_MODELS,
        default=["Graham Number", "Lynch Method", "Two-Stage DCF"],
    )
    mos = st.slider("Margin of Safety (%)", 0, 50, 25, 5)

    if not st.button("Compare", type="primary"):
        return
    if len(tickers) < 2:
        st.warning("Enter at least 2 tickers.")
        return
    if not selected_models:
        st.warning("Select at least one model.")
        return

    rows = []
    for t in tickers[:5]:
        with st.spinner(f"Fetching {t}…"):
            data = fetch_stock_data(t)
        if not data:
            st.warning(f"Could not fetch {t}")
            continue
        res = run_valuation(data, selected_models, mos_pct=mos)
        row = {
            "Ticker":            t,
            "Name":              data["Name"],
            "Sector":            data["Sector"],
            "Price ($)":         data["Price ($)"],
            "Intrinsic ($)":     round(res["intrinsic"], 2) if not np.isnan(res["intrinsic"]) else None,
            "Premium/Discount":  f"{res['premium']:+.1f}%" if not np.isnan(res["premium"]) else "N/A",
            "Verdict":           res["verdict"],
            "Market Cap (B)":    data["Market Cap (B)"],
            "Trailing P/E":      data["Trailing P/E"],
            "PEG":               data["PEG"],
            "ROE (%)":           data["ROE (%)"],
            "Profit Margin (%)": data["Profit Margin (%)"],
            "Rev. Growth (%)":   data["Rev. Growth (%)"],
            "Debt/Equity":       data["Debt/Equity"],
            "Div. Yield (%)":    data["Div. Yield (%)"],
            "Beta":              data["Beta"],
            "RSI (14)":          data["RSI (14)"],
        }
        rows.append(row)

    if not rows:
        st.error("No valid data.")
        return

    cmp_df = pd.DataFrame(rows).set_index("Ticker")
    st.dataframe(cmp_df.T, use_container_width=True)

    # radar chart
    metrics = ["ROE (%)","Profit Margin (%)","Rev. Growth (%)","Div. Yield (%)"]
    fig = go.Figure()
    for _, row in pd.DataFrame(rows).iterrows():
        vals = [row.get(m) or 0 for m in metrics]
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=metrics + [metrics[0]],
            fill="toself",
            name=row["Ticker"],
        ))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True)), title="Fundamentals Radar")
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# MAIN
# ============================================================

def main():
    st.set_page_config(
        page_title="Intrinsic Value Pro",
        page_icon="📈",
        layout="wide",
    )

    st.title("📈 Intrinsic Value Pro")
    st.markdown(
        "Multi-model intrinsic value calculator · Advanced screener · Side-by-side comparison  \n"
        "*Educational purposes only — not financial advice.*"
    )

    mode = st.radio(
        "Mode",
        ["Single Stock", "Screener", "Compare"],
        horizontal=True,
    )

    if mode == "Single Stock":
        single_stock_ui()
    elif mode == "Screener":
        screener_ui()
    else:
        compare_ui()


# ── Public alias ─────────────────────────────────────────────────────────────
render_intrinsic_value = main

if __name__ == "__main__":
    main()
