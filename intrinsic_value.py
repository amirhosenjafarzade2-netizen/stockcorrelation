import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import time
import math
import json
import os

from bs4 import BeautifulSoup
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import plotly.graph_objects as go
import plotly.express as px

# ── Optional RSI library ──────────────────────────────────────────────────────
try:
    import pandas_ta as ta
    HAS_PANDAS_TA = True
except ImportError:
    HAS_PANDAS_TA = False

# ============================================================
# CONSTANTS
# ============================================================

DEFAULT_MOS      = 25.0
DCF_YEARS        = 5
MAX_GROWTH       = 0.30
MIN_GROWTH       = 0.02
OUTLIER_MULTIPLE = 15
MAX_THREADS      = 6
WATCHLIST_FILE   = os.path.join(os.path.dirname(__file__), "watchlist.json")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.google.com/",
}

# Sectors where certain models are unreliable
FINANCIAL_SECTORS = {"Financial Services", "Banks", "Financials", "Insurance"}
# Models that need FCF > 0
FCF_MODELS        = {"Two-Stage DCF", "Owner Earnings (Buffett)"}
# Models that need dividends
DIV_MODELS        = {"Dividend Discount Model (DDM)"}
# Models that need positive EPS
EPS_MODELS        = {"Graham Number", "Lynch Method", "Core Valuation (PE-based)"}
# Models that need positive book value
BOOK_MODELS       = {"Graham Number", "Residual Income", "Asset-Based (NAV)"}

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
    "Core":             ["Symbol", "Name", "Sector", "Price ($)", "Market Cap (B)"],
    "Valuation":        ["Intrinsic ($)", "Safe Buy ($)", "Premium/Discount (%)", "Undervaluation (%)", "Verdict"],
    "Multiples":        ["Trailing P/E", "Forward P/E", "PEG", "P/B", "P/S", "EV/EBITDA"],
    "Profitability":    ["ROE (%)", "ROA (%)", "Profit Margin (%)", "Op. Margin (%)", "Gross Margin (%)"],
    "Growth":           ["Rev. Growth (%)", "EPS Growth (%)", "Analyst Growth (%)", "EPS (TTM)"],
    "Financial Health": ["Debt/Equity", "Current Ratio", "Interest Coverage"],
    "Cash Flow":        ["FCF (B)", "FCF Yield (%)"],
    "Dividends":        ["Div. Yield (%)", "Payout Ratio (%)"],
    "Technical":        ["RSI (14)", "Beta", "52W Change (%)", "% off 52W High"],
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

def safe_float(v, default=np.nan) -> float:
    try:
        if v is None:
            return default
        if isinstance(v, str):
            v = v.replace("%", "").replace(",", "").replace("$", "").strip()
            if v in ["", "-", "N/A", "—"]:
                return default
        f = float(v)
        return default if (math.isnan(f) or math.isinf(f)) else f
    except Exception:
        return default


def clamp_growth(g: float) -> float:
    if math.isnan(g):
        return 0.06
    return float(np.clip(g, MIN_GROWTH, MAX_GROWTH))


def to_b(v) -> Optional[float]:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return round(v / 1e9, 3) if v else None


def fmt(v, prefix="", suffix="", decimals=2) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "N/A"
    return f"{prefix}{v:.{decimals}f}{suffix}"


def verdict_from_underval(u: float) -> str:
    if u > 30:  return "Strong Buy"
    if u > 10:  return "Buy"
    if u > -10: return "Hold"
    return "Sell"


# ============================================================
# PERSISTENT WATCHLIST
# ============================================================

def load_watchlist() -> set:
    try:
        if os.path.exists(WATCHLIST_FILE):
            with open(WATCHLIST_FILE) as f:
                return set(json.load(f))
    except Exception:
        pass
    return set()


def save_watchlist(wl: set):
    try:
        with open(WATCHLIST_FILE, "w") as f:
            json.dump(sorted(wl), f)
    except Exception:
        pass


# ============================================================
# NETWORK — retrying fetch
# ============================================================

def _get(url: str, retries: int = 2, delay: float = 0.4) -> Optional[requests.Response]:
    for attempt in range(retries + 1):
        try:
            time.sleep(delay)
            r = requests.get(url, headers=HEADERS, timeout=15)
            if r.status_code == 200:
                return r
        except Exception:
            pass
        if attempt < retries:
            time.sleep(2 ** attempt)
    return None


# ============================================================
# DATA: UNIVERSE LOADERS
# ============================================================

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
    # Fallback top-50
    return pd.DataFrame({
        "Symbol": ["AAPL","MSFT","GOOGL","AMZN","NVDA","META","TSLA","JPM",
                   "JNJ","V","PG","XOM","UNH","HD","CVX","MRK","ABBV","PEP",
                   "COST","KO","WMT","MCD","DIS","ADBE","CRM","NFLX","AMD",
                   "CSCO","ACN","TMO","ORCL","ABT","NEE","HON","IBM","QCOM",
                   "INTC","TXN","AMGN","GILD","MDT","CAT","DE","MMM","GS",
                   "MS","BAC","WFC","C","RTX"],
        "Security": ["Apple","Microsoft","Alphabet","Amazon","NVIDIA","Meta",
                     "Tesla","JPMorgan","J&J","Visa","P&G","ExxonMobil",
                     "UnitedHealth","Home Depot","Chevron","Merck","AbbVie",
                     "PepsiCo","Costco","Coca-Cola","Walmart","McDonald's",
                     "Disney","Adobe","Salesforce","Netflix","AMD","Cisco",
                     "Accenture","Thermo Fisher","Oracle","Abbott","NextEra",
                     "Honeywell","IBM","Qualcomm","Intel","TI","Amgen","Gilead",
                     "Medtronic","Caterpillar","Deere","3M","Goldman","Morgan S",
                     "BofA","Wells F","Citigroup","RTX"],
        "GICS Sector": (
            ["Information Technology"] * 10 + ["Health Care"] * 6 +
            ["Financials"] * 5 + ["Consumer Staples"] * 5 +
            ["Consumer Discretionary"] * 4 + ["Communication Services"] * 3 +
            ["Industrials"] * 5 + ["Energy"] * 2 + ["Financials"] * 10
        ),
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
        "Technology": "sec_technology", "Healthcare": "sec_healthcare",
        "Financials": "sec_financial", "Energy": "sec_energy",
        "Consumer Discretionary": "sec_consumercyclical",
        "Consumer Staples": "sec_consumernoncyclical",
        "Industrials": "sec_industrials", "Basic Materials": "sec_basicmaterials",
        "Communication Services": "sec_communicationservices",
        "Utilities": "sec_utilities", "Real Estate": "sec_realestate",
    }
    if sector not in sector_map:
        return []
    tickers, page = [], 1
    while page <= 50:
        start = (page - 1) * 20 + 1
        url = f"https://finviz.com/screener.ashx?v=111&f={sector_map[sector]}&r={start}"
        r = _get(url, delay=0.3)
        if not r:
            break
        soup = BeautifulSoup(r.text, "html.parser")
        found = False
        for link in soup.find_all("a", {"class": "tab-link"}):
            t = link.text.strip()
            if t and 1 <= len(t) <= 5:
                tickers.append(t)
                found = True
        if not found:
            break
        page += 1
    return list(set(tickers))


# ============================================================
# DATA: FUNDAMENTALS  (cached 30 min)
# ============================================================

@st.cache_data(ttl=1800)
def fetch_fundamentals(ticker: str) -> Optional[Dict]:
    """
    Fetch all fundamental data for a ticker.
    Returns a dict with all fields needed for valuation + display,
    plus a '_quality' sub-dict recording the source of each key input.
    """
    try:
        stock = yf.Ticker(ticker)
        info  = stock.info or {}

        price = safe_float(info.get("currentPrice") or info.get("regularMarketPrice"))
        if math.isnan(price) or price <= 0:
            return None

        quality: Dict[str, str] = {}   # field -> "yfinance" | "finviz" | "estimated"

        # ── Finviz supplement ────────────────────────────────
        fvz: Dict = {}
        try:
            r = _get(f"https://finviz.com/quote.ashx?t={ticker}", delay=0.4)
            if r:
                soup = BeautifulSoup(r.text, "html.parser")
                tbl  = soup.find("table", {"class": "snapshot-table2"})
                if tbl:
                    for row in tbl.find_all("tr"):
                        cells = row.find_all("td")
                        for i in range(0, len(cells) - 1, 2):
                            fvz[cells[i].text.strip()] = cells[i + 1].text.strip()
        except Exception:
            pass

        def yf_or_fvz(yf_key, fvz_key=None, scale=1.0, as_pct=False):
            """Get a value preferring yfinance, falling back to Finviz."""
            v = safe_float(info.get(yf_key))
            src = "yfinance"
            if math.isnan(v) and fvz_key and fvz_key in fvz:
                v   = safe_float(fvz.get(fvz_key))
                src = "finviz"
            if not math.isnan(v):
                v = v * scale
                if as_pct:
                    v = v * 100
            return v, src

        # ── Growth rates: analyst forward vs historical ───────
        analyst_growth_raw = safe_float(info.get("earningsGrowth"))  # forward-looking
        historical_growth  = 0.06
        growth_source      = "estimated (6% default)"

        try:
            income = stock.get_income_stmt()
            for field in ("DilutedEPS", "BasicEPS"):
                if field in income.index:
                    eps_s = income.loc[field].dropna()[::-1]
                    if len(eps_s) >= 2 and eps_s.iloc[0] > 0 and eps_s.iloc[-1] > 0:
                        yrs = len(eps_s) - 1
                        historical_growth = (eps_s.iloc[-1] / eps_s.iloc[0]) ** (1 / yrs) - 1
                        growth_source = "historical EPS CAGR"
                    break
        except Exception:
            pass
        historical_growth = clamp_growth(historical_growth)

        # Analyst consensus (forward-looking) — prefer for forward models
        if not math.isnan(analyst_growth_raw) and analyst_growth_raw > 0:
            analyst_growth = clamp_growth(analyst_growth_raw)
            quality["analyst_growth"] = "yfinance (analyst consensus)"
        else:
            analyst_growth = historical_growth
            quality["analyst_growth"] = f"fallback: {growth_source}"

        # ── Core fields ───────────────────────────────────────
        mc        = safe_float(info.get("marketCap"))
        fcf       = safe_float(info.get("freeCashflow"))
        rev       = safe_float(info.get("totalRevenue"))
        ebit      = safe_float(info.get("ebit"))
        interest  = safe_float(info.get("interestExpense"))
        ebitda    = safe_float(info.get("ebitda"))
        hi52      = safe_float(info.get("fiftyTwoWeekHigh"))
        shares    = safe_float(info.get("sharesOutstanding"))
        book_v    = safe_float(info.get("bookValue"))
        eps_ttm   = safe_float(info.get("trailingEps"))
        fwd_eps   = safe_float(info.get("forwardEps"))
        debt      = safe_float(info.get("totalDebt"), default=0.0)
        cash      = safe_float(info.get("totalCash"), default=0.0)

        div_rate  = safe_float(info.get("dividendRate") or
                               info.get("trailingAnnualDividendRate"), default=0.0)
        div_yield = safe_float(info.get("dividendYield"), default=0.0) * 100

        fwd_pe  = safe_float(info.get("forwardPE"))
        hist_pe = safe_float(info.get("trailingPE"))

        # Finviz overrides for fields yfinance often gets wrong
        if "P/E" in fvz:
            v = safe_float(fvz["P/E"])
            if not math.isnan(v):
                hist_pe = v
                quality["hist_pe"] = "finviz"
        if "Beta" in fvz:
            v = safe_float(fvz["Beta"])
            if not math.isnan(v):
                quality["beta"] = "finviz"

        if math.isnan(hist_pe):
            hist_pe = fwd_pe if not math.isnan(fwd_pe) else 15.0
        if math.isnan(fwd_pe):
            fwd_pe = hist_pe
        if math.isnan(fwd_eps) or fwd_eps <= 0:
            fwd_eps = eps_ttm if not math.isnan(eps_ttm) else 1.0

        roe      = safe_float(info.get("returnOnEquity"), default=0.0) * 100
        roa      = safe_float(info.get("returnOnAssets"), default=0.0) * 100
        gross_m  = safe_float(info.get("grossMargins"), default=0.0) * 100
        op_m     = safe_float(info.get("operatingMargins"), default=0.0) * 100
        profit_m = safe_float(info.get("profitMargins"), default=0.0) * 100
        rev_g    = safe_float(info.get("revenueGrowth"), default=0.0) * 100
        eps_g    = safe_float(info.get("earningsQuarterlyGrowth"), default=0.0) * 100
        de       = safe_float(info.get("debtToEquity"))
        cr       = safe_float(info.get("currentRatio"))
        beta     = safe_float(info.get("beta"))
        peg      = safe_float(info.get("pegRatio"))
        pb       = safe_float(info.get("priceToBook"))
        ps       = safe_float(info.get("priceToSalesTrailing12Months"))
        ev_ebitda = safe_float(info.get("enterpriseToEbitda"))
        ev        = safe_float(info.get("enterpriseValue"))
        w52chg    = safe_float(info.get("52WeekChange"), default=0.0) * 100
        pay_ratio = safe_float(info.get("payoutRatio"), default=0.0) * 100

        # Derived
        interest_cov = None
        if not math.isnan(ebit) and not math.isnan(interest) and interest != 0:
            interest_cov = round(ebit / abs(interest), 2)

        fcf_yield = None
        if not math.isnan(fcf) and not math.isnan(mc) and mc > 0:
            fcf_yield = round(fcf / mc * 100, 2)

        pct_off_hi = None
        if not math.isnan(hi52) and hi52 > 0:
            pct_off_hi = round((price / hi52 - 1) * 100, 2)

        ebitda_margin = None
        if not math.isnan(ebitda) and not math.isnan(rev) and rev > 0:
            ebitda_margin = round(ebitda / rev * 100, 2)

        total_assets = safe_float(info.get("totalAssets"), default=0.0)
        invested_cap = total_assets - safe_float(info.get("totalCurrentLiabilities"), default=0.0)

        sector = info.get("sector", "N/A")

        return {
            # ── identifiers
            "Symbol":   ticker,
            "Name":     info.get("longName") or info.get("shortName", ticker),
            "Sector":   sector,
            "Industry": info.get("industry", "N/A"),
            # ── price
            "Price ($)":      price,
            "52W High":       hi52,
            "52W Low":        safe_float(info.get("fiftyTwoWeekLow")),
            "52W Change (%)": round(w52chg, 2),
            "% off 52W High": pct_off_hi,
            # ── size
            "Market Cap (B)":        to_b(mc),
            "Enterprise Value (B)":  to_b(ev),
            # ── multiples
            "Trailing P/E": round(hist_pe, 2) if not math.isnan(hist_pe) else None,
            "Forward P/E":  round(fwd_pe, 2)  if not math.isnan(fwd_pe)  else None,
            "PEG":          round(peg, 2)      if not math.isnan(peg)     else None,
            "P/B":          round(pb, 2)       if not math.isnan(pb)      else None,
            "P/S":          round(ps, 2)       if not math.isnan(ps)      else None,
            "EV/EBITDA":    round(ev_ebitda,2) if not math.isnan(ev_ebitda) else None,
            # ── profitability
            "ROE (%)":           round(roe, 2)      if not math.isnan(roe)      else None,
            "ROA (%)":           round(roa, 2)      if not math.isnan(roa)      else None,
            "Gross Margin (%)":  round(gross_m, 2)  if not math.isnan(gross_m)  else None,
            "Op. Margin (%)":    round(op_m, 2)     if not math.isnan(op_m)     else None,
            "Profit Margin (%)": round(profit_m, 2) if not math.isnan(profit_m) else None,
            "EBITDA Margin (%)": ebitda_margin,
            # ── growth
            "EPS (TTM)":          round(eps_ttm, 2) if not math.isnan(eps_ttm) else None,
            "Forward EPS":        round(fwd_eps, 2),
            "Rev. Growth (%)":    round(rev_g, 2)   if not math.isnan(rev_g)   else None,
            "EPS Growth (%)":     round(eps_g, 2)   if not math.isnan(eps_g)   else None,
            "Analyst Growth (%)": round(analyst_growth * 100, 2),
            "Historical Growth":  historical_growth,
            "Analyst Growth":     analyst_growth,
            "Growth Source":      quality.get("analyst_growth", growth_source),
            # ── health
            "Debt/Equity":       round(de, 2)          if not math.isnan(de) else None,
            "Current Ratio":     round(cr, 2)          if not math.isnan(cr) else None,
            "Interest Coverage": interest_cov,
            # ── cash flow
            "FCF (B)":      to_b(fcf),
            "FCF Yield (%)": fcf_yield,
            # ── dividends
            "Div. Yield (%)":  round(div_yield, 2),
            "Payout Ratio (%)": round(pay_ratio, 2),
            "Div. Per Share":   div_rate,
            # ── technical
            "Beta":     round(beta, 2) if not math.isnan(beta) else None,
            # ── raw valuation inputs (prefixed _)
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
            "_quality":       quality,
        }

    except Exception:
        return None


# ============================================================
# DATA: TECHNICAL  (cached 5 min — faster moving)
# ============================================================

@st.cache_data(ttl=300)
def fetch_technical(ticker: str) -> Dict:
    """Fetch RSI and recent price history. Separate cache from fundamentals."""
    result = {"RSI (14)": None, "price_history": None}
    try:
        stock = yf.Ticker(ticker)
        hist  = stock.history(period="5y", auto_adjust=True)
        if len(hist) < 15 or "Close" not in hist.columns:
            return result

        # RSI
        if len(hist) >= 30:
            if HAS_PANDAS_TA:
                s = ta.rsi(hist["Close"], length=14)
                v = s.iloc[-1]
                result["RSI (14)"] = None if pd.isna(v) else round(float(v), 1)
            else:
                delta = hist["Close"].diff()
                gain  = delta.where(delta > 0, 0).rolling(14).mean()
                loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs    = gain / loss
                rsi   = 100 - (100 / (1 + rs))
                v     = rsi.iloc[-1]
                result["RSI (14)"] = None if pd.isna(v) else round(float(v), 1)

        # Price history for historical PE chart
        result["price_history"] = hist

    except Exception:
        pass
    return result


# ============================================================
# DATA: HISTORICAL P/E  (needs both price + EPS history)
# ============================================================

@st.cache_data(ttl=3600)
def fetch_historical_pe(ticker: str) -> Optional[pd.Series]:
    """Return a Series of trailing P/E over the past 4 years (quarterly)."""
    try:
        stock  = yf.Ticker(ticker)
        income = stock.get_income_stmt()
        hist   = stock.history(period="4y", auto_adjust=True)
        if income is None or hist is None or hist.empty:
            return None

        for field in ("DilutedEPS", "BasicEPS", "NetIncome"):
            if field in income.index:
                eps_s = income.loc[field].dropna()
                if len(eps_s) < 2:
                    continue
                # Build rolling TTM EPS approximation
                records = []
                for date, price_row in hist["Close"].resample("QE").last().items():
                    # Find the most recent annual EPS before this date
                    past = eps_s[[c for c in eps_s.index if c <= date]]
                    if past.empty:
                        continue
                    eps_val = past.iloc[0]
                    if field == "NetIncome":
                        shares = safe_float(stock.info.get("sharesOutstanding"))
                        if math.isnan(shares) or shares <= 0:
                            continue
                        eps_val = eps_val / shares
                    if eps_val > 0:
                        records.append((date, price_row / eps_val))
                if records:
                    return pd.Series(
                        [v for _, v in records],
                        index=[d for d, _ in records],
                        name="Trailing P/E"
                    )
    except Exception:
        pass
    return None


# ============================================================
# SECTOR-AWARE MODEL SELECTION
# ============================================================

def get_applicable_models(data: dict) -> Tuple[List[str], Dict[str, str]]:
    """
    Return (applicable_models, skip_reasons) for a given stock.
    skip_reasons maps model name -> human-readable reason it was excluded.
    """
    sector   = data.get("Sector", "")
    eps      = data.get("_eps_ttm", np.nan)
    bv       = data.get("_book_value", np.nan)
    fcf      = data.get("_fcf", np.nan)
    div      = data.get("_div_rate", 0.0)
    fcf_ok   = not math.isnan(fcf) and fcf > 0
    eps_ok   = not math.isnan(eps) and eps > 0
    bv_ok    = not math.isnan(bv) and bv > 0
    div_ok   = div > 0
    is_fin   = sector in FINANCIAL_SECTORS

    applicable, skipped = [], {}

    for m in ALL_MODELS:
        reason = None
        if m in FCF_MODELS and not fcf_ok:
            reason = "No positive FCF"
        elif m in FCF_MODELS and is_fin:
            reason = "DCF unreliable for financial sector"
        elif m in DIV_MODELS and not div_ok:
            reason = "No dividend paid"
        elif m in EPS_MODELS and not eps_ok:
            reason = "Negative or missing EPS"
        elif m in BOOK_MODELS and not bv_ok:
            reason = "Negative or missing book value"

        if reason:
            skipped[m] = reason
        else:
            applicable.append(m)

    return applicable, skipped


# ============================================================
# VALUATION MODELS
# ============================================================

def _graham_number(d: dict, **_) -> float:
    eps = d["_eps_ttm"]
    bv  = d["_book_value"]
    if any(math.isnan(x) or x <= 0 for x in [eps, bv]):
        return np.nan
    return math.sqrt(22.5 * eps * bv)


def _lynch_method(d: dict, growth=None, **_) -> float:
    eps = d["_eps_ttm"]
    g   = (growth * 100) if growth else (d["Analyst Growth"] * 100)
    div = d["Div. Yield (%)"]
    if math.isnan(eps) or eps <= 0:
        return np.nan
    fair_pe = float(np.clip(g + div, 8, 40))
    return max(eps * fair_pe, 0)


def _two_stage_dcf(d: dict, discount=0.10, terminal=0.03,
                   years=DCF_YEARS, growth=None, **_) -> float:
    fcf    = d["_fcf"]
    shares = d["_shares"]
    debt   = d["_debt"]
    cash   = d["_cash"]
    # Use analyst growth for forward-looking DCF
    g = growth if growth else d["Analyst Growth"]
    g = clamp_growth(g)

    if any(math.isnan(x) or x <= 0 for x in [fcf, shares]):
        return np.nan
    if discount <= terminal or (discount - terminal) < 0.01:
        return np.nan

    pv, cf = 0.0, fcf
    for yr in range(1, years + 1):
        cf  *= (1 + g)
        pv  += cf / (1 + discount) ** yr

    terminal_value = (cf * (1 + terminal)) / (discount - terminal)
    pv_terminal    = terminal_value / (1 + discount) ** years
    equity         = pv + pv_terminal - (debt or 0) + (cash or 0)
    return max(equity / shares, 0)


def _ddm(d: dict, growth=None, **_) -> float:
    div = d["_div_rate"]
    g   = growth if growth else min(d["Historical Growth"], 0.08)
    r   = 0.10
    if math.isnan(div) or div <= 0:
        return np.nan
    g = min(clamp_growth(g), r - 0.005)
    return max(div * (1 + g) / (r - g), 0)


def _residual_income(d: dict, growth=None, **_) -> float:
    bv  = d["_book_value"]
    roe = d["_roe"] / 100
    g   = growth if growth else d["Historical Growth"]
    coe = 0.10
    g   = min(clamp_growth(g), coe - 0.005)
    if math.isnan(bv) or bv <= 0:
        return np.nan
    ri    = bv * (roe - coe)
    pv_ri = sum(ri * (1 + g) ** t / (1 + coe) ** t for t in range(1, 11))
    return max(bv + pv_ri, 0)


def _owner_earnings(d: dict, growth=None, **_) -> float:
    fcf, shares = d["_fcf"], d["_shares"]
    r   = 0.10
    g   = growth if growth else d["Analyst Growth"]
    g   = min(clamp_growth(g), r - 0.005)
    if any(math.isnan(x) or x <= 0 for x in [fcf, shares]):
        return np.nan
    oe = fcf / shares
    pv = sum(oe * (1 + g) ** t / (1 + r) ** t for t in range(1, 11))
    tv = (oe * (1 + g) ** 10 * (1 + g)) / (r - g) / (1 + r) ** 10
    return max(pv + tv, 0)


def _core_pe_valuation(d: dict, growth=None, **_) -> float:
    fwd_eps = d["_fwd_eps"]
    hist_pe = d["_hist_pe"]
    g       = growth if growth else d["Analyst Growth"]
    g       = clamp_growth(g)
    r, yrs  = 0.10, 5
    if math.isnan(fwd_eps) or fwd_eps <= 0 or math.isnan(hist_pe) or hist_pe <= 0:
        return np.nan
    future_price = fwd_eps * (1 + g) ** yrs * hist_pe
    return max(future_price / (1 + r) ** yrs, 0)


def _nav(d: dict, **_) -> float:
    bv = d["_book_value"]
    if math.isnan(bv) or bv <= 0:
        return np.nan
    return bv * 0.70


MODEL_FN = {
    "Graham Number":              _graham_number,
    "Lynch Method":               _lynch_method,
    "Two-Stage DCF":              _two_stage_dcf,
    "Dividend Discount Model (DDM)": _ddm,
    "Residual Income":            _residual_income,
    "Owner Earnings (Buffett)":   _owner_earnings,
    "Core Valuation (PE-based)":  _core_pe_valuation,
    "Asset-Based (NAV)":          _nav,
}


def run_valuation(
    data: dict,
    selected_models: List[str],
    mos_pct: float = DEFAULT_MOS,
    growth_override: float = None,
    discount_rate: float = 0.10,
    terminal_growth: float = 0.03,
) -> Dict:
    """
    Run each selected model, filter outliers, return composite median.
    Also returns per-model skip reasons for display.
    """
    # Sector-aware filtering
    applicable, skipped_auto = get_applicable_models(data)

    per_model:    Dict[str, float] = {}
    skip_reasons: Dict[str, str]   = dict(skipped_auto)

    price = data["Price ($)"]

    for m in selected_models:
        if m in skipped_auto:
            continue
        fn = MODEL_FN.get(m)
        if fn is None:
            continue
        try:
            val = fn(
                data,
                growth=growth_override,
                discount=discount_rate,
                terminal=terminal_growth,
            )
        except Exception as e:
            val = np.nan
            skip_reasons[m] = f"Error: {e}"
        per_model[m] = val

    # Remove outliers
    clean = [
        v for v in per_model.values()
        if not math.isnan(v) and v > 0 and v < price * OUTLIER_MULTIPLE
    ]
    # Note models filtered as outliers
    for m, v in per_model.items():
        if not math.isnan(v) and v > 0 and v >= price * OUTLIER_MULTIPLE:
            skip_reasons[m] = skip_reasons.get(m, "") + " (outlier — excluded from median)"

    intrinsic = float(np.median(clean)) if clean else np.nan

    safe_buy  = intrinsic * (1 - mos_pct / 100) if not math.isnan(intrinsic) else np.nan
    premium   = ((price - intrinsic) / intrinsic * 100) if not math.isnan(intrinsic) and intrinsic > 0 else np.nan
    under     = -premium if not math.isnan(premium) else np.nan

    return {
        "per_model":      per_model,
        "skip_reasons":   skip_reasons,
        "intrinsic":      intrinsic,
        "safe_buy":       safe_buy,
        "premium":        premium,
        "undervaluation": under,
        "verdict":        verdict_from_underval(under) if not math.isnan(under) else "N/A",
        "models_used":    clean,
    }


# ============================================================
# REVERSE DCF — implied growth at current price
# ============================================================

def reverse_dcf(data: dict, discount: float = 0.10, terminal: float = 0.03,
                years: int = DCF_YEARS) -> Optional[float]:
    """
    Binary-search for the FCF growth rate that makes DCF intrinsic == current price.
    Returns implied growth rate (as a fraction) or None if unsolvable.
    """
    price  = data["Price ($)"]
    fcf    = data["_fcf"]
    shares = data["_shares"]
    debt   = data["_debt"]
    cash   = data["_cash"]

    if math.isnan(fcf) or fcf <= 0 or math.isnan(shares) or shares <= 0:
        return None

    def dcf_at_g(g):
        pv, cf = 0.0, fcf
        for yr in range(1, years + 1):
            cf  *= (1 + g)
            pv  += cf / (1 + discount) ** yr
        tv = (cf * (1 + terminal)) / (discount - terminal)
        pv += tv / (1 + discount) ** years
        return (pv - debt + cash) / shares

    lo, hi = -0.20, 0.50
    for _ in range(60):
        mid = (lo + hi) / 2
        val = dcf_at_g(mid)
        if abs(val - price) < 0.01:
            return mid
        if val < price:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# ============================================================
# SCENARIO + SENSITIVITY
# ============================================================

def scenario_dcf(data: dict) -> Dict[str, float]:
    return {
        "Bear (12% disc, 2% term)": _two_stage_dcf(data, discount=0.12, terminal=0.02),
        "Base (10% disc, 3% term)": _two_stage_dcf(data, discount=0.10, terminal=0.03),
        "Bull  (8% disc, 4% term)": _two_stage_dcf(data, discount=0.08, terminal=0.04),
    }


def sensitivity_table(data: dict, steps: int = 7) -> pd.DataFrame:
    discounts = np.linspace(0.07, 0.14, steps)
    terminals = np.linspace(0.01, 0.05, steps)
    matrix = []
    for d in discounts:
        row = []
        for t in terminals:
            v = _two_stage_dcf(data, discount=d, terminal=t) if d > t else np.nan
            row.append(v)
        matrix.append(row)
    return pd.DataFrame(
        matrix,
        index=[f"{d*100:.0f}%" for d in discounts],
        columns=[f"{t*100:.0f}%" for t in terminals],
    )


# ============================================================
# CHARTS
# ============================================================

def valuation_bar_chart(price: float, per_model: Dict[str, float],
                        skip_reasons: Dict[str, str]) -> go.Figure:
    names, vals, colors, notes = [], [], [], []
    for k, v in per_model.items():
        if not math.isnan(v) and v > 0 and v < price * OUTLIER_MULTIPLE:
            names.append(k)
            vals.append(v)
            colors.append("#2ecc71" if v > price else "#e74c3c")
            notes.append(f"${v:.2f}")

    fig = go.Figure()
    if names:
        fig.add_trace(go.Bar(
            x=names, y=vals, marker_color=colors,
            text=notes, textposition="outside",
        ))
    fig.add_hline(
        y=price, line_dash="dash", line_color="#3498db",
        annotation_text=f"Market ${price:.2f}", annotation_position="top right",
    )
    fig.update_layout(
        title="Intrinsic Value by Model vs. Market Price",
        yaxis_title="Price ($)", height=400,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def historical_pe_chart(pe_series: pd.Series, current_pe: Optional[float]) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pe_series.index, y=pe_series.values,
        mode="lines", name="Trailing P/E",
        line=dict(color="#3498db", width=2),
    ))
    mean_pe = pe_series.mean()
    fig.add_hline(y=mean_pe, line_dash="dot", line_color="#95a5a6",
                  annotation_text=f"4Y Mean: {mean_pe:.1f}x")
    if current_pe:
        fig.add_hline(y=current_pe, line_dash="dash", line_color="#e74c3c",
                      annotation_text=f"Current: {current_pe:.1f}x")
    fig.update_layout(
        title="Historical Trailing P/E (4 Years)",
        yaxis_title="P/E Ratio", height=300,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def scenario_chart(price: float, scenarios: Dict[str, float]) -> go.Figure:
    names  = list(scenarios.keys())
    vals   = [v if not math.isnan(v) else 0 for v in scenarios.values()]
    colors = ["#e74c3c", "#f39c12", "#2ecc71"]
    fig = go.Figure(go.Bar(
        x=names, y=vals, marker_color=colors,
        text=[f"${v:.2f}" if v > 0 else "N/A" for v in vals],
        textposition="outside",
    ))
    fig.add_hline(y=price, line_dash="dot", line_color="#3498db",
                  annotation_text=f"Price ${price:.2f}")
    fig.update_layout(title="DCF Scenario Analysis", height=340,
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    return fig


def scatter_screener_chart(df: pd.DataFrame) -> go.Figure:
    x_col = "Rev. Growth (%)"
    y_col = "Premium/Discount (%)" if "Premium/Discount (%)" in df.columns else "Trailing P/E"
    needed = {"Symbol", x_col, y_col, "Market Cap (B)"}
    if not needed.issubset(df.columns):
        return go.Figure()
    plot_df = df.dropna(subset=[x_col, y_col]).copy()
    if plot_df.empty:
        return go.Figure()
    fig = px.scatter(
        plot_df, x=x_col, y=y_col,
        size=plot_df["Market Cap (B)"].clip(lower=0.1).fillna(1),
        color=y_col,
        color_continuous_scale="RdYlGn_r" if "Premium" in y_col else "RdYlGn",
        hover_name="Symbol",
        hover_data={c: True for c in ["Trailing P/E", "ROE (%)", "Verdict", "Sector"]
                    if c in plot_df.columns},
        title=f"{x_col} vs. {y_col}  (bubble = market cap)",
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
            help="Sector-inapplicable models will be auto-excluded with reasons shown.",
        )
        mos = st.slider("Margin of Safety (%)", 0, 50, 25, 5,
                        help="Applied on top of the composite intrinsic value.")

        with st.expander("🛠️ Override Parameters"):
            override_growth = st.number_input(
                "Custom Growth Rate (%)",
                min_value=0.0, max_value=50.0, value=0.0, step=0.5,
                help="0 = use analyst consensus (falling back to historical CAGR)",
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

        with st.spinner(f"Fetching fundamentals for {ticker}…"):
            data = fetch_fundamentals(ticker)

        if not data:
            st.error(f"Could not fetch data for **{ticker}**. Check the ticker symbol.")
            return

        g_override = (override_growth / 100) if override_growth > 0 else None

        result = run_valuation(
            data, selected_models,
            mos_pct=mos,
            growth_override=g_override,
            discount_rate=custom_discount / 100,
            terminal_growth=custom_terminal / 100,
        )

        price     = data["Price ($)"]
        intrinsic = result["intrinsic"]
        safe_buy  = result["safe_buy"]
        premium   = result["premium"]
        verd      = result["verdict"]

        # ── Company header ────────────────────────────────────
        st.markdown(f"## {data['Name']} · `{ticker}`")
        st.caption(
            f"**Sector:** {data['Sector']}  |  **Industry:** {data['Industry']}  |  "
            f"**Growth source:** {data['Growth Source']}"
        )

        emoji = VERDICT_EMOJI.get(verd, "⚪")
        if not math.isnan(intrinsic):
            st.markdown(f"### {emoji} {verd}")

        # ── Key metrics ───────────────────────────────────────
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Market Price",   f"${price:.2f}")
        c2.metric("Intrinsic Value", fmt(intrinsic, "$") if not math.isnan(intrinsic) else "N/A")
        c3.metric("Safe Buy Price",  fmt(safe_buy, "$")  if not math.isnan(safe_buy)  else "N/A")
        c4.metric("Premium/Discount", fmt(premium, suffix="%", decimals=1) if not math.isnan(premium) else "N/A",
                  delta="overvalued" if not math.isnan(premium) and premium > 0 else "undervalued",
                  delta_color="inverse" if not math.isnan(premium) and premium > 0 else "normal")
        c5.metric("Margin of Safety", f"{mos}%")

        st.divider()

        # ── Reverse DCF — surfaced prominently ───────────────
        st.subheader("🔄 Reverse DCF — What Is the Market Pricing In?")
        implied_g = reverse_dcf(data, discount=custom_discount/100, terminal=custom_terminal/100)
        if implied_g is not None:
            ra, rb, rc = st.columns(3)
            ra.metric("Implied FCF Growth Rate", f"{implied_g*100:.1f}%/yr",
                      help="The FCF growth rate baked into the current market price.")
            rb.metric("Your Growth Assumption",
                      f"{(g_override or data['Analyst Growth'])*100:.1f}%/yr",
                      delta=f"{((g_override or data['Analyst Growth']) - implied_g)*100:+.1f}pp vs implied",
                      delta_color="normal" if (g_override or data['Analyst Growth']) > implied_g else "inverse")
            rc.metric("Analyst Consensus Growth", f"{data['Analyst Growth (%)']:.1f}%/yr")
            if implied_g > 0.20:
                st.warning(f"⚠️ The market is pricing in **{implied_g*100:.1f}% annual FCF growth** — a very high bar to clear.")
            elif implied_g < 0.03:
                st.success(f"✅ The market implies only **{implied_g*100:.1f}% FCF growth** — a low hurdle if the business is sound.")
        else:
            st.info("Reverse DCF unavailable — no positive FCF data.")

        st.divider()

        # ── Valuation bar chart ───────────────────────────────
        st.plotly_chart(
            valuation_bar_chart(price, result["per_model"], result["skip_reasons"]),
            use_container_width=True,
        )

        # ── Model breakdown table ─────────────────────────────
        model_rows = []
        for m in ALL_MODELS:
            if m in result["per_model"]:
                v = result["per_model"][m]
                vs_price = f"{((v - price)/price*100):+.1f}%" if not math.isnan(v) else "—"
                used     = "✅" if (not math.isnan(v) and v > 0 and v < price * OUTLIER_MULTIPLE) else "❌"
                reason   = result["skip_reasons"].get(m, "")
                model_rows.append({"Model": m, "Value ($)": fmt(v, "$") if not math.isnan(v) else "N/A",
                                   "vs. Price": vs_price, "In Median": used, "Note": reason})
            elif m in result["skip_reasons"]:
                model_rows.append({"Model": m, "Value ($)": "—", "vs. Price": "—",
                                   "In Median": "⏭️ skipped", "Note": result["skip_reasons"][m]})

        st.subheader("Model Breakdown")
        st.dataframe(pd.DataFrame(model_rows), use_container_width=True, hide_index=True)

        st.divider()

        # ── Scenario DCF ──────────────────────────────────────
        st.subheader("📊 DCF Scenario Analysis")
        scens = scenario_dcf(data)
        st.plotly_chart(scenario_chart(price, scens), use_container_width=True)

        # ── Sensitivity heatmap ───────────────────────────────
        st.subheader("🌡️ DCF Sensitivity (Discount × Terminal Growth)")
        st.caption("Rows = discount rate · Columns = terminal growth · 🟢 = intrinsic above market price")
        sens = sensitivity_table(data)
        styled_sens = (
            sens.style
            .background_gradient(cmap="RdYlGn", axis=None, vmin=price * 0.3, vmax=price * 2.5)
            .format("${:.2f}", na_rep="—")
        )
        st.dataframe(styled_sens, use_container_width=True)

        st.divider()

        # ── Historical P/E chart ──────────────────────────────
        st.subheader("📈 Historical P/E Context")
        with st.spinner("Loading price history…"):
            pe_series = fetch_historical_pe(ticker)
        if pe_series is not None and len(pe_series) > 2:
            st.plotly_chart(
                historical_pe_chart(pe_series, data.get("Trailing P/E")),
                use_container_width=True,
            )
            mean_pe = pe_series.mean()
            curr_pe = data.get("Trailing P/E")
            if curr_pe:
                diff = curr_pe - mean_pe
                if diff > 3:
                    st.warning(f"Current P/E ({curr_pe:.1f}x) is **{diff:.1f} turns above** its 4-year average ({mean_pe:.1f}x).")
                elif diff < -3:
                    st.success(f"Current P/E ({curr_pe:.1f}x) is **{abs(diff):.1f} turns below** its 4-year average ({mean_pe:.1f}x).")
                else:
                    st.info(f"Current P/E ({curr_pe:.1f}x) is near its 4-year average ({mean_pe:.1f}x).")
        else:
            st.info("Historical P/E data unavailable for this ticker.")

        st.divider()

        # ── Fundamentals snapshot ─────────────────────────────
        with st.expander("📋 Full Fundamentals Snapshot"):
            fields = [
                "Price ($)", "Market Cap (B)", "Trailing P/E", "Forward P/E", "PEG",
                "P/B", "P/S", "EV/EBITDA", "Beta",
                "ROE (%)", "ROA (%)", "Gross Margin (%)", "Op. Margin (%)", "Profit Margin (%)",
                "Rev. Growth (%)", "EPS Growth (%)", "Analyst Growth (%)", "EPS (TTM)", "Forward EPS",
                "Debt/Equity", "Current Ratio", "Interest Coverage",
                "FCF (B)", "FCF Yield (%)",
                "Div. Yield (%)", "Payout Ratio (%)",
                "Beta", "52W Change (%)", "% off 52W High",
            ]
            snap = {k: data.get(k) for k in fields if k in data}
            snap_df = pd.DataFrame(snap.items(), columns=["Metric", "Value"])
            snap_df["Value"] = snap_df["Value"].apply(
                lambda v: f"{v:.2f}" if isinstance(v, float) and not math.isnan(v)
                else (str(v) if v is not None else "N/A")
            )
            st.dataframe(snap_df, use_container_width=True, hide_index=True)

        # ── Watchlist ─────────────────────────────────────────
        wl = load_watchlist()
        in_wl = ticker in wl
        col_w1, col_w2 = st.columns([1, 4])
        with col_w1:
            if in_wl:
                if st.button("⭐ Remove from Watchlist"):
                    wl.discard(ticker)
                    save_watchlist(wl)
                    st.rerun()
            else:
                if st.button("☆ Add to Watchlist"):
                    wl.add(ticker)
                    save_watchlist(wl)
                    st.success(f"{ticker} added to watchlist.")


# ============================================================
# SCREENER
# ============================================================

def _screener_row(
    symbol: str,
    selected_models: List[str],
    mos_pct: float,
    growth_override: Optional[float],
    include_rsi: bool,
) -> Tuple[Optional[Dict], Optional[str]]:
    data = fetch_fundamentals(symbol)
    if not data:
        return None, "No fundamental data"
    try:
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
            "Analyst Growth (%)": data["Analyst Growth (%)"],
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
            "52W Change (%)":    data["52W Change (%)"],
            "% off 52W High":    data["% off 52W High"],
            "RSI (14)":          None,  # filled below if requested
        }

        if include_rsi:
            tech = fetch_technical(symbol)
            row["RSI (14)"] = tech.get("RSI (14)")

        if selected_models:
            res = run_valuation(data, selected_models, mos_pct=mos_pct,
                                growth_override=growth_override)
            row["Intrinsic ($)"]        = round(res["intrinsic"], 2)       if not math.isnan(res["intrinsic"])      else None
            row["Safe Buy ($)"]         = round(res["safe_buy"],  2)       if not math.isnan(res["safe_buy"])       else None
            row["Premium/Discount (%)"] = round(res["premium"],   1)       if not math.isnan(res["premium"])        else None
            row["Undervaluation (%)"]   = round(res["undervaluation"], 1)  if not math.isnan(res["undervaluation"]) else None
            row["Verdict"]              = res["verdict"]

        return row, None
    except Exception as e:
        return None, str(e)


def screener_ui():
    st.header("📡 Multi-Stock Screener")
    st.caption("Filters apply instantly to cached results — re-screen only when you change the universe or models.")

    # ── Persistent watchlist ───────────────────────────────────────────────
    wl = load_watchlist()
    if wl:
        with st.expander(f"⭐ Watchlist ({len(wl)} stocks)", expanded=False):
            cols_wl = st.columns(6)
            for i, sym in enumerate(sorted(wl)):
                cols_wl[i % 6].write(sym)
            if st.button("Clear Watchlist"):
                save_watchlist(set())
                st.rerun()

    # ── 1. Universe ────────────────────────────────────────────────────────
    with st.expander("📊 1. Universe", expanded=True):
        universe_opt = st.radio(
            "Source", ["S&P 500", "NASDAQ 100", "Finviz Sector", "Custom"],
            horizontal=True, key="scr_universe_opt",
        )
        universe_tickers: List[str] = []

        if universe_opt == "S&P 500":
            if st.button("Load S&P 500", key="load_sp500"):
                with st.spinner("Loading…"):
                    sp500 = get_sp500()
                    st.session_state["_sp500_df"] = sp500
                    st.session_state["_raw_universe"] = sp500["Symbol"].tolist()
                    st.session_state["_screener_cache"] = None  # invalidate
                    st.success(f"✅ {len(sp500)} stocks")

            if "_sp500_df" in st.session_state:
                sp500 = st.session_state["_sp500_df"]
                sectors = sorted(sp500["GICS Sector"].unique().tolist())
                sel_sectors = st.multiselect("Filter sectors", sectors, default=sectors, key="scr_sectors")
                filtered = sp500[sp500["GICS Sector"].isin(sel_sectors)]
                st.session_state["_raw_universe"] = filtered["Symbol"].tolist()

        elif universe_opt == "NASDAQ 100":
            if st.button("Load NASDAQ 100", key="load_ndq"):
                with st.spinner("Loading…"):
                    tickers = get_nasdaq100()
                    st.session_state["_raw_universe"] = tickers
                    st.session_state["_screener_cache"] = None
                    st.success(f"✅ {len(tickers)} stocks")

        elif universe_opt == "Finviz Sector":
            fsector = st.selectbox("Sector", [
                "Technology","Healthcare","Financials","Energy",
                "Consumer Discretionary","Consumer Staples","Industrials",
                "Basic Materials","Communication Services","Utilities","Real Estate",
            ], key="scr_finviz_sector")
            if st.button("Load Sector", key="load_finviz"):
                with st.spinner("Scraping Finviz…"):
                    tickers = get_finviz_sector_tickers(fsector)
                    if tickers:
                        st.session_state["_raw_universe"] = tickers
                        st.session_state["_screener_cache"] = None
                        st.success(f"✅ {len(tickers)} stocks")
                    else:
                        st.error("Could not load sector tickers. Finviz may be rate-limiting.")

        else:
            raw = st.text_area("Tickers (comma or space separated)",
                               placeholder="AAPL MSFT GOOGL AMZN NVDA", height=80, key="scr_custom")
            tickers = [t.strip().upper() for t in raw.replace(",", " ").split() if t.strip()]
            if tickers:
                st.session_state["_raw_universe"] = tickers
                st.info(f"{len(tickers)} tickers entered")

        universe_tickers = st.session_state.get("_raw_universe", [])
        if universe_tickers:
            max_n = st.slider("Max stocks to screen", 10,
                              min(500, len(universe_tickers)),
                              min(100, len(universe_tickers)), key="scr_max_n")
            universe_tickers = universe_tickers[:max_n]
            st.caption(f"**{len(universe_tickers)}** stocks queued")

    if not universe_tickers:
        st.info("👆 Choose a universe above and click Load.")
        return

    # ── 2. Valuation (optional) ────────────────────────────────────────────
    with st.expander("💡 2. Valuation (optional — adds Intrinsic/Verdict columns)", expanded=False):
        enable_val = st.checkbox("Enable intrinsic value calculation", value=False, key="scr_enable_val")
        if enable_val:
            selected_models = st.multiselect(
                "Models", ALL_MODELS,
                default=["Graham Number", "Lynch Method", "Two-Stage DCF", "Core Valuation (PE-based)"],
                key="scr_models",
            )
            mos = st.slider("Margin of Safety (%)", 0, 50, 25, 5, key="scr_mos")
            override_g = st.number_input(
                "Growth Override (0 = use analyst consensus)",
                min_value=0.0, max_value=50.0, value=0.0, step=0.5, key="scr_g",
            )
        else:
            selected_models, mos, override_g = [], 25, 0.0

        include_rsi = st.checkbox(
            "Fetch RSI (slower — one extra API call per stock)", value=False, key="scr_rsi",
            help="Skipping RSI roughly halves screener runtime on large universes.",
        )

    # ── 3. Filters ─────────────────────────────────────────────────────────
    with st.expander("⚙️ 3. Filters (apply instantly after screening)", expanded=False):
        st.caption("All filters apply to the cached results — no re-fetch needed.")
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            st.markdown("**Valuation**")
            f_verdict    = st.multiselect("Verdict", ["Strong Buy","Buy","Hold","Sell"],
                                          default=[], key="scr_f_verdict")
            f_pe_max     = st.number_input("P/E ≤",         value=None, format="%.1f", key="f_pe_max")
            f_pb_max     = st.number_input("P/B ≤",         value=None, format="%.2f", key="f_pb_max")
            f_under_min  = st.number_input("Underval. (%) ≥", value=None, format="%.1f", key="f_under")
        with fc2:
            st.markdown("**Quality**")
            f_roe_min    = st.number_input("ROE (%) ≥",      value=None, format="%.1f", key="f_roe")
            f_pm_min     = st.number_input("Profit Margin ≥",value=None, format="%.1f", key="f_pm")
            f_rg_min     = st.number_input("Rev. Growth ≥",  value=None, format="%.1f", key="f_rg")
            f_de_max     = st.number_input("Debt/Equity ≤",  value=None, format="%.2f", key="f_de")
            f_mc_min     = st.number_input("Mkt Cap (B) ≥",  value=None, format="%.2f", key="f_mc")
        with fc3:
            st.markdown("**Technical**")
            f_dy_min     = st.number_input("Div. Yield ≥",   value=None, format="%.2f", key="f_dy")
            f_rsi_max    = st.number_input("RSI ≤",          value=None, min_value=0.0,
                                           max_value=100.0, format="%.0f", key="f_rsi")
            f_beta_max   = st.number_input("Beta ≤",         value=None, format="%.2f", key="f_beta")
            f_fcfy_min   = st.number_input("FCF Yield ≥",    value=None, format="%.2f", key="f_fcfy")

    # ── 4. Columns ─────────────────────────────────────────────────────────
    with st.expander("👁️ 4. Column Groups", expanded=False):
        visible_groups = st.multiselect(
            "Show", list(COLUMN_GROUPS.keys()),
            default=["Core", "Valuation", "Multiples", "Profitability", "Growth", "Technical"],
            key="scr_col_groups",
        )

    # ── Run / Re-screen ────────────────────────────────────────────────────
    cache_key = f"{sorted(universe_tickers)}|{sorted(selected_models)}|{mos}|{override_g}|{include_rsi}"
    cached    = st.session_state.get("_screener_cache")
    cache_hit = (cached is not None and
                 st.session_state.get("_screener_cache_key") == cache_key)

    col_run1, col_run2 = st.columns([2, 1])
    with col_run1:
        run_btn = st.button("🚀 Run Screener", type="primary")
    with col_run2:
        if cache_hit:
            if st.button("🔄 Re-screen (clear cache)"):
                st.session_state["_screener_cache"] = None
                st.rerun()

    if cache_hit and not run_btn:
        df = cached
        st.info(f"Showing cached results for {len(df)} stocks — filters apply instantly below.")
    elif run_btn or (not cache_hit and cached is not None):
        g_override = (override_g / 100) if override_g > 0 else None
        results: List[Dict] = []
        errors:  List[str]  = []
        prog   = st.progress(0)
        stat   = st.empty()
        t0     = time.time()

        with ThreadPoolExecutor(max_workers=MAX_THREADS) as ex:
            futures = {
                ex.submit(_screener_row, sym, selected_models, mos, g_override, include_rsi): sym
                for sym in universe_tickers
            }
            done = 0
            for fut in as_completed(futures):
                done += 1
                prog.progress(done / len(universe_tickers))
                sym     = futures[fut]
                elapsed = time.time() - t0
                eta     = int(elapsed / done * (len(universe_tickers) - done)) if done > 1 else "?"
                stat.text(f"Screening {sym}… ({done}/{len(universe_tickers)})  ~{eta}s left")
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
            st.error("No data retrieved.")
            return

        df = pd.DataFrame(results)
        st.session_state["_screener_cache"]     = df
        st.session_state["_screener_cache_key"] = cache_key
    else:
        st.info("👆 Click **Run Screener** to start.")
        return

    # ── Apply filters (instant — client side) ──────────────────────────────
    mask = pd.Series([True] * len(df))

    def flt(col, lo=None, hi=None):
        nonlocal mask
        if col not in df.columns:
            return
        if lo is not None:
            mask &= df[col].fillna(-np.inf) >= lo
        if hi is not None:
            mask &= df[col].fillna(np.inf) <= hi

    if f_verdict and "Verdict" in df.columns:
        mask &= df["Verdict"].isin(f_verdict)

    flt("Trailing P/E",      hi=f_pe_max)
    flt("P/B",               hi=f_pb_max)
    flt("Undervaluation (%)",lo=f_under_min)
    flt("ROE (%)",           lo=f_roe_min)
    flt("Profit Margin (%)", lo=f_pm_min)
    flt("Rev. Growth (%)",   lo=f_rg_min)
    flt("Debt/Equity",       hi=f_de_max)
    flt("Market Cap (B)",    lo=f_mc_min)
    flt("Div. Yield (%)",    lo=f_dy_min)
    flt("RSI (14)",          hi=f_rsi_max)
    flt("Beta",              hi=f_beta_max)
    flt("FCF Yield (%)",     lo=f_fcfy_min)

    final = df[mask].copy()
    if final.empty:
        st.warning("No stocks pass the current filters. Try relaxing criteria.")
        return

    # ── Summary cards ──────────────────────────────────────────────────────
    st.success(f"✅ **{len(final)} stocks** matched out of {len(df)} screened")
    s1, s2, s3, s4, s5, s6 = st.columns(6)
    def _med(col): return f"{final[col].median():.1f}" if col in final.columns and not final[col].isna().all() else "—"
    s1.metric("Matched",        str(len(final)))
    s2.metric("Median P/E",     _med("Trailing P/E"))
    s3.metric("Median ROE %",   _med("ROE (%)"))
    s4.metric("Median Rev G %", _med("Rev. Growth (%)"))
    s5.metric("Median FCF Y %", _med("FCF Yield (%)"))
    s6.metric("Median Beta",    _med("Beta"))

    # ── Column selection ───────────────────────────────────────────────────
    visible_cols: List[str] = []
    for grp in visible_groups:
        for col in COLUMN_GROUPS.get(grp, []):
            if col in final.columns and col not in visible_cols:
                visible_cols.append(col)
    if "Symbol" not in visible_cols:
        visible_cols = ["Symbol"] + visible_cols
    display_df = final[[c for c in visible_cols if c in final.columns]].copy()

    # ── Sort ───────────────────────────────────────────────────────────────
    sortable = [c for c in display_df.columns if c != "Symbol"]
    default_sort = "Premium/Discount (%)" if "Premium/Discount (%)" in sortable else sortable[0] if sortable else None
    sort_col = st.selectbox("Sort by", sortable, index=sortable.index(default_sort) if default_sort in sortable else 0, key="scr_sort")
    asc = st.checkbox("Ascending", value=True, key="scr_asc")
    if sort_col and sort_col in display_df.columns:
        display_df = display_df.sort_values(sort_col, ascending=asc, na_position="last")

    # ── Heatmap table ──────────────────────────────────────────────────────
    num_cols = display_df.select_dtypes(include=[np.number]).columns.tolist()
    bad_if_high = {"Trailing P/E","Forward P/E","P/B","P/S","EV/EBITDA",
                   "Debt/Equity","Payout Ratio (%)","Beta","Premium/Discount (%)"}

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

    # ── Charts ─────────────────────────────────────────────────────────────
    with st.expander("📈 Scatter Chart", expanded=True):
        st.plotly_chart(scatter_screener_chart(final), use_container_width=True)

    with st.expander("🗂️ Sector Breakdown"):
        if "Sector" in final.columns:
            counts = final["Sector"].value_counts().reset_index()
            counts.columns = ["Sector", "Count"]
            fig = px.bar(counts, x="Sector", y="Count", color="Count",
                         color_continuous_scale="Blues")
            st.plotly_chart(fig, use_container_width=True)

    # ── Watchlist ──────────────────────────────────────────────────────────
    with st.expander("⭐ Add to Watchlist"):
        to_add = st.multiselect("Select symbols", final["Symbol"].tolist(), key="scr_wl_add")
        if st.button("Add selected", key="scr_wl_btn"):
            wl = load_watchlist()
            for s in to_add:
                wl.add(s)
            save_watchlist(wl)
            st.success(f"Added {len(to_add)} stock(s) to watchlist.")
            st.rerun()

    # ── Export ─────────────────────────────────────────────────────────────
    st.download_button(
        "⬇️ Download CSV",
        data=final.to_csv(index=False),
        file_name=f"screener_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )


# ============================================================
# COMPARE UI
# ============================================================

def compare_ui():
    st.header("⚖️ Side-by-Side Comparison")
    raw = st.text_input("Tickers (2–5, comma-separated)", "AAPL, MSFT, GOOGL")
    tickers = [t.strip().upper() for t in raw.split(",") if t.strip()][:5]

    selected_models = st.multiselect(
        "Valuation Models", ALL_MODELS,
        default=["Graham Number", "Lynch Method", "Two-Stage DCF"],
        key="cmp_models",
    )
    mos = st.slider("Margin of Safety (%)", 0, 50, 25, 5, key="cmp_mos")

    if not st.button("Compare", type="primary"):
        return
    if len(tickers) < 2:
        st.warning("Enter at least 2 tickers.")
        return

    rows = []
    for t in tickers:
        with st.spinner(f"Fetching {t}…"):
            data = fetch_fundamentals(t)
        if not data:
            st.warning(f"Could not fetch {t}.")
            continue

        res = run_valuation(data, selected_models, mos_pct=mos) if selected_models else {}
        implied_g = reverse_dcf(data) if data["_fcf"] and data["_fcf"] > 0 else None

        row: Dict = {
            "Ticker":             t,
            "Name":               data["Name"],
            "Sector":             data["Sector"],
            "Price ($)":          data["Price ($)"],
            "Market Cap (B)":     data["Market Cap (B)"],
        }
        if res:
            row["Intrinsic ($)"]       = round(res["intrinsic"], 2) if not math.isnan(res["intrinsic"]) else None
            row["Premium/Discount (%)"]= round(res["premium"], 1)   if not math.isnan(res["premium"])   else None
            row["Verdict"]             = res["verdict"]
        row.update({
            "Trailing P/E":       data["Trailing P/E"],
            "Forward P/E":        data["Forward P/E"],
            "PEG":                data["PEG"],
            "P/B":                data["P/B"],
            "ROE (%)":            data["ROE (%)"],
            "Profit Margin (%)":  data["Profit Margin (%)"],
            "Rev. Growth (%)":    data["Rev. Growth (%)"],
            "Analyst Growth (%)": data["Analyst Growth (%)"],
            "Debt/Equity":        data["Debt/Equity"],
            "FCF Yield (%)":      data["FCF Yield (%)"],
            "Div. Yield (%)":     data["Div. Yield (%)"],
            "Beta":               data["Beta"],
            "Implied FCF Growth": f"{implied_g*100:.1f}%" if implied_g is not None else "N/A",
        })
        rows.append(row)

    if not rows:
        st.error("No valid data.")
        return

    cmp_df = pd.DataFrame(rows).set_index("Ticker")
    st.dataframe(cmp_df.T, use_container_width=True)

    # Radar chart — normalise each metric to 0–1 for fair comparison
    radar_metrics = ["ROE (%)", "Profit Margin (%)", "Rev. Growth (%)",
                     "Div. Yield (%)", "FCF Yield (%)"]
    radar_df = pd.DataFrame(rows)[["Ticker"] + radar_metrics].set_index("Ticker")
    # Min-max normalise
    normalised = (radar_df - radar_df.min()) / (radar_df.max() - radar_df.min() + 1e-9)

    fig = go.Figure()
    for ticker_name, row_vals in normalised.iterrows():
        vals = row_vals.tolist()
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=radar_metrics + [radar_metrics[0]],
            fill="toself", name=ticker_name,
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="Normalised Fundamentals Radar (0 = worst in set, 1 = best)",
        height=450,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Per-model detail
    if selected_models:
        st.subheader("Model-by-Model Valuation")
        model_rows = []
        for row in rows:
            data = fetch_fundamentals(row["Ticker"])
            if not data:
                continue
            res = run_valuation(data, selected_models, mos_pct=mos)
            for m, v in res["per_model"].items():
                model_rows.append({
                    "Ticker": row["Ticker"],
                    "Model": m,
                    "Value ($)": round(v, 2) if not math.isnan(v) else None,
                    "vs. Price": round((v - data["Price ($)"]) / data["Price ($)"] * 100, 1)
                                 if not math.isnan(v) else None,
                })
        if model_rows:
            st.dataframe(pd.DataFrame(model_rows), use_container_width=True, hide_index=True)


# ============================================================
# WATCHLIST UI
# ============================================================

def watchlist_ui():
    st.header("⭐ Watchlist")
    wl = load_watchlist()
    if not wl:
        st.info("Your watchlist is empty. Add stocks from Single Stock or Screener.")
        return

    st.caption(f"{len(wl)} stocks saved to `watchlist.json` next to this file.")

    selected_models = st.multiselect(
        "Quick-value with models", ALL_MODELS,
        default=["Graham Number", "Core Valuation (PE-based)"],
        key="wl_models",
    )
    mos = st.slider("MOS (%)", 0, 50, 25, 5, key="wl_mos")

    if st.button("Refresh Watchlist Data", type="primary"):
        rows = []
        prog = st.progress(0)
        tickers = sorted(wl)
        for i, sym in enumerate(tickers):
            prog.progress((i + 1) / len(tickers))
            data = fetch_fundamentals(sym)
            if not data:
                continue
            row = {
                "Symbol": sym, "Name": data["Name"], "Sector": data["Sector"],
                "Price ($)": data["Price ($)"],
                "Market Cap (B)": data["Market Cap (B)"],
                "Trailing P/E": data["Trailing P/E"],
                "ROE (%)": data["ROE (%)"],
                "Rev. Growth (%)": data["Rev. Growth (%)"],
                "Div. Yield (%)": data["Div. Yield (%)"],
                "Beta": data["Beta"],
            }
            if selected_models:
                res = run_valuation(data, selected_models, mos_pct=mos)
                row["Intrinsic ($)"]       = round(res["intrinsic"], 2) if not math.isnan(res["intrinsic"]) else None
                row["Premium/Discount (%)"]= round(res["premium"], 1)   if not math.isnan(res["premium"])   else None
                row["Verdict"]             = res["verdict"]
            rows.append(row)
        prog.empty()
        if rows:
            st.session_state["_wl_df"] = pd.DataFrame(rows)

    if "_wl_df" in st.session_state:
        st.dataframe(st.session_state["_wl_df"], use_container_width=True, hide_index=True)
        # Remove button
        to_remove = st.multiselect("Remove from watchlist", sorted(wl), key="wl_remove")
        if st.button("Remove selected"):
            for s in to_remove:
                wl.discard(s)
            save_watchlist(wl)
            st.session_state.pop("_wl_df", None)
            st.rerun()
    else:
        st.info("Click **Refresh Watchlist Data** to load current prices and valuations.")


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
        "Multi-model intrinsic value calculator · Advanced screener · "
        "Side-by-side comparison · Persistent watchlist  \n"
        "*Educational purposes only — not financial advice.*"
    )

    mode = st.radio(
        "Mode",
        ["Single Stock", "Screener", "Compare", "Watchlist"],
        horizontal=True,
    )

    if mode == "Single Stock":
        single_stock_ui()
    elif mode == "Screener":
        screener_ui()
    elif mode == "Compare":
        compare_ui()
    else:
        watchlist_ui()


render_intrinsic_value = main

if __name__ == "__main__":
    main()
ENDOFFILE
echo "Written: $(wc -l < /mnt/user-data/outputs/intrinsic_value.py) lines"
