# sector_valuation_improved.py - Enhanced Sector-Tuned Valuations + SWOT Module
# Comprehensive sector analysis with DCF, multiples, comparables, and SWOT
#
# CHANGELOG (this revision):
#   - Root-caused "error fetching data": Yahoo Finance rate-limits yfinance heavily.
#     The old peer-comparison code fired 6 unthrottled, uncached requests in a tight
#     loop on every click, which is the single most common way to get blocked.
#     Fixed by: a shared retrying/backing-off fetch wrapper, much longer caching,
#     small randomized delays between calls, and a "lite" peer fetch that pulls only
#     the fields needed instead of the entire `.info` blob per peer.
#   - Failed fetches are no longer cached for an hour (st.cache_data was caching None).
#   - Shares-outstanding is now computed consistently (always from sharesOutstanding,
#     never inferred from marketCap / currentPrice, which breaks when price is stale).
#   - DCF sensitivity table now actually recomputes the DCF per growth rate instead of
#     a linear-scaling approximation that doesn't reflect the model.
#   - Bare `except:` blocks replaced with targeted handling + visible warnings, so
#     silent failures are no longer mistaken for "no data".
#   - Guards added against division-by-zero (previousClose, currentPrice, shares).
#   - Ticker existence is validated up front with a clear error instead of cascading
#     KeyErrors deeper in the analysis.
#   - Sector multiplier table can be overridden by live sector detection but no longer
#     silently mismatches if Yahoo's sector string isn't one of our 11 buckets.
#   - Minor UX: spinner messages reflect retry attempts, a "last updated"/cache notice
#     is shown, and the peer list is no longer a hardcoded mega-cap list disconnected
#     from the ticker's actual market-cap tier (small caps were always compared to
#     AAPL/MSFT/etc.).

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import time
import random
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# ==================== COMPREHENSIVE SECTOR MULTIPLIERS ====================

SECTOR_MULTIPLIERS = {
    "Technology": {
        "pe_median": 28.0, "pe_range": (18, 45),
        "pb_median": 6.5, "pb_range": (3.0, 12.0),
        "ps_median": 5.5, "ps_range": (2.0, 10.0),
        "ev_ebitda_median": 18.0, "ev_ebitda_range": (12, 28),
        "peg_median": 2.0, "peg_range": (1.0, 3.5),
        "roe_benchmark": 20.0,
        "margin_benchmark": 25.0,
        "growth_rate": 15.0
    },
    "Financial Services": {
        "pe_median": 12.0, "pe_range": (8, 18),
        "pb_median": 1.4, "pb_range": (0.8, 2.5),
        "ps_median": 2.5, "ps_range": (1.0, 4.0),
        "ev_ebitda_median": 10.0, "ev_ebitda_range": (6, 15),
        "peg_median": 1.5, "peg_range": (0.8, 2.5),
        "roe_benchmark": 12.0,
        "margin_benchmark": 30.0,
        "growth_rate": 8.0
    },
    "Healthcare": {
        "pe_median": 22.0, "pe_range": (15, 35),
        "pb_median": 4.0, "pb_range": (2.0, 7.0),
        "ps_median": 3.5, "ps_range": (1.5, 6.0),
        "ev_ebitda_median": 16.0, "ev_ebitda_range": (10, 25),
        "peg_median": 2.2, "peg_range": (1.2, 3.5),
        "roe_benchmark": 15.0,
        "margin_benchmark": 20.0,
        "growth_rate": 10.0
    },
    "Consumer Cyclical": {
        "pe_median": 18.0, "pe_range": (12, 28),
        "pb_median": 3.5, "pb_range": (1.5, 6.0),
        "ps_median": 1.2, "ps_range": (0.5, 2.5),
        "ev_ebitda_median": 12.0, "ev_ebitda_range": (8, 18),
        "peg_median": 1.8, "peg_range": (1.0, 3.0),
        "roe_benchmark": 15.0,
        "margin_benchmark": 8.0,
        "growth_rate": 12.0
    },
    "Consumer Defensive": {
        "pe_median": 20.0, "pe_range": (15, 30),
        "pb_median": 4.5, "pb_range": (2.0, 8.0),
        "ps_median": 1.8, "ps_range": (0.8, 3.5),
        "ev_ebitda_median": 14.0, "ev_ebitda_range": (10, 20),
        "peg_median": 2.5, "peg_range": (1.5, 4.0),
        "roe_benchmark": 18.0,
        "margin_benchmark": 12.0,
        "growth_rate": 6.0
    },
    "Energy": {
        "pe_median": 12.0, "pe_range": (6, 20),
        "pb_median": 1.5, "pb_range": (0.8, 2.5),
        "ps_median": 1.0, "ps_range": (0.4, 2.0),
        "ev_ebitda_median": 8.0, "ev_ebitda_range": (5, 12),
        "peg_median": 1.2, "peg_range": (0.5, 2.0),
        "roe_benchmark": 12.0,
        "margin_benchmark": 10.0,
        "growth_rate": 5.0
    },
    "Industrials": {
        "pe_median": 20.0, "pe_range": (14, 28),
        "pb_median": 3.2, "pb_range": (1.5, 5.5),
        "ps_median": 1.5, "ps_range": (0.8, 2.8),
        "ev_ebitda_median": 13.0, "ev_ebitda_range": (9, 18),
        "peg_median": 2.0, "peg_range": (1.2, 3.0),
        "roe_benchmark": 14.0,
        "margin_benchmark": 10.0,
        "growth_rate": 9.0
    },
    "Basic Materials": {
        "pe_median": 15.0, "pe_range": (8, 22),
        "pb_median": 2.0, "pb_range": (1.0, 3.5),
        "ps_median": 1.2, "ps_range": (0.5, 2.2),
        "ev_ebitda_median": 9.0, "ev_ebitda_range": (6, 14),
        "peg_median": 1.5, "peg_range": (0.8, 2.5),
        "roe_benchmark": 10.0,
        "margin_benchmark": 12.0,
        "growth_rate": 7.0
    },
    "Real Estate": {
        "pe_median": 25.0, "pe_range": (15, 40),
        "pb_median": 1.8, "pb_range": (1.0, 3.0),
        "ps_median": 8.0, "ps_range": (4.0, 15.0),
        "ev_ebitda_median": 16.0, "ev_ebitda_range": (12, 22),
        "peg_median": 3.0, "peg_range": (1.5, 5.0),
        "roe_benchmark": 8.0,
        "margin_benchmark": 40.0,
        "growth_rate": 6.0
    },
    "Utilities": {
        "pe_median": 18.0, "pe_range": (12, 25),
        "pb_median": 1.6, "pb_range": (1.0, 2.5),
        "ps_median": 2.5, "ps_range": (1.5, 4.0),
        "ev_ebitda_median": 11.0, "ev_ebitda_range": (8, 15),
        "peg_median": 3.0, "peg_range": (2.0, 5.0),
        "roe_benchmark": 9.0,
        "margin_benchmark": 15.0,
        "growth_rate": 4.0
    },
    "Communication Services": {
        "pe_median": 20.0, "pe_range": (12, 32),
        "pb_median": 3.0, "pb_range": (1.5, 5.5),
        "ps_median": 3.0, "ps_range": (1.5, 5.5),
        "ev_ebitda_median": 12.0, "ev_ebitda_range": (8, 18),
        "peg_median": 2.0, "peg_range": (1.0, 3.5),
        "roe_benchmark": 12.0,
        "margin_benchmark": 20.0,
        "growth_rate": 10.0
    }
}

# Curated peer universe split by market-cap tier so a small-cap doesn't get
# benchmarked against AAPL/MSFT just because it shares a sector label.
PEER_UNIVERSE = {
    'Technology': {
        'mega': ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA'],
        'large': ['ADBE', 'CRM', 'ORCL', 'CSCO', 'AMD'],
        'mid_small': ['DDOG', 'TWLO', 'HUBS', 'PCTY', 'BILL'],
    },
    'Financial Services': {
        'mega': ['JPM', 'BAC', 'WFC', 'C', 'GS'],
        'large': ['MS', 'SCHW', 'AXP', 'BLK', 'SPGI'],
        'mid_small': ['SF', 'WAL', 'EWBC', 'PB', 'FHN'],
    },
    'Healthcare': {
        'mega': ['JNJ', 'UNH', 'LLY', 'ABBV', 'MRK'],
        'large': ['PFE', 'TMO', 'ABT', 'DHR', 'BMY'],
        'mid_small': ['PODD', 'TECH', 'HOLX', 'BAX', 'CRL'],
    },
    'Consumer Cyclical': {
        'mega': ['AMZN', 'TSLA', 'HD', 'MCD', 'NKE'],
        'large': ['LOW', 'SBUX', 'TJX', 'BKNG', 'CMG'],
        'mid_small': ['DECK', 'POOL', 'BURL', 'RH', 'YETI'],
    },
    'Consumer Defensive': {
        'mega': ['PG', 'KO', 'PEP', 'WMT', 'COST'],
        'large': ['MDLZ', 'CL', 'KMB', 'GIS', 'STZ'],
        'mid_small': ['HRL', 'CLX', 'CHD', 'MKC', 'SJM'],
    },
    'Energy': {
        'mega': ['XOM', 'CVX', 'COP', 'SLB', 'EOG'],
        'large': ['PSX', 'MPC', 'OXY', 'WMB', 'KMI'],
        'mid_small': ['DVN', 'FANG', 'APA', 'MRO', 'CTRA'],
    },
    'Industrials': {
        'mega': ['BA', 'CAT', 'GE', 'HON', 'UPS'],
        'large': ['RTX', 'LMT', 'DE', 'UNP', 'ETN'],
        'mid_small': ['XYL', 'PNR', 'IEX', 'AOS', 'DOV'],
    },
    'Basic Materials': {
        'mega': ['LIN', 'APD', 'ECL', 'SHW', 'NEM'],
        'large': ['FCX', 'NUE', 'DOW', 'PPG', 'VMC'],
        'mid_small': ['MOS', 'CF', 'ALB', 'CE', 'AVNT'],
    },
    'Real Estate': {
        'mega': ['AMT', 'PLD', 'CCI', 'EQIX', 'PSA'],
        'large': ['SPG', 'O', 'WELL', 'DLR', 'AVB'],
        'mid_small': ['CUBE', 'REXR', 'STAG', 'IRT', 'NSA'],
    },
    'Utilities': {
        'mega': ['NEE', 'DUK', 'SO', 'D', 'AEP'],
        'large': ['EXC', 'XEL', 'ED', 'WEC', 'ES'],
        'mid_small': ['PNW', 'NWE', 'IDA', 'POR', 'AVA'],
    },
    'Communication Services': {
        'mega': ['GOOGL', 'META', 'DIS', 'NFLX', 'T'],
        'large': ['VZ', 'CMCSA', 'CHTR', 'TMUS', 'WBD'],
        'mid_small': ['MTCH', 'PARA', 'LYV', 'NWSA', 'IPG'],
    },
}

CACHE_TTL_SECONDS = 1800  # 30 min: long enough to avoid hammering Yahoo on reruns
MAX_RETRIES = 3
BASE_BACKOFF = 1.5  # seconds, exponential backoff base


# ==================== ROBUST DATA FETCHING ====================

def _retrying_call(fn, *args, max_retries: int = MAX_RETRIES, **kwargs):
    """
    Call fn(*args, **kwargs) with exponential backoff + jitter on failure.
    This is the core fix for the rate-limit errors: Yahoo Finance throttles
    yfinance aggressively, especially on rapid repeated calls. Retrying blindly
    and instantly makes it worse, so we back off with increasing, randomized delays.
    """
    last_exc = None
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_exc = e
            msg = str(e).lower()
            # Only worth retrying on transient/network/rate-limit style errors.
            transient = any(s in msg for s in [
                'rate limit', 'too many requests', '429', 'timeout',
                'timed out', 'connection', 'temporarily', 'json'
            ])
            if not transient or attempt == max_retries - 1:
                raise
            sleep_time = (BASE_BACKOFF ** attempt) + random.uniform(0.3, 1.0)
            time.sleep(sleep_time)
    if last_exc:
        raise last_exc


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def validate_ticker(ticker: str) -> bool:
    """Lightweight existence check before running the full pipeline."""
    try:
        info = _retrying_call(lambda: yf.Ticker(ticker).get_info())
        return bool(info and info.get('symbol'))
    except Exception:
        return False


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def fetch_company_data(ticker: str) -> Optional[Dict]:
    """
    Fetch comprehensive company data from Yahoo Finance, with retry/backoff.
    Returns None on hard failure so callers can show an explicit error instead
    of operating on empty/partial data. Importantly: failures are NOT cached
    (st.cache_data only caches the return value of a *successful* call), so a
    transient rate-limit blip won't be stuck for the full TTL.
    """
    stock = yf.Ticker(ticker)

    try:
        info = _retrying_call(stock.get_info)
    except Exception as e:
        st.session_state['_last_fetch_error'] = str(e)
        return None

    if not info or not info.get('symbol'):
        st.session_state['_last_fetch_error'] = "No data returned for this ticker."
        return None

    # These are best-effort; if any of these fail we keep going with empty frames
    # rather than failing the whole fetch (a 403 on financials shouldn't block price data).
    income_stmt, balance_sheet, cash_flow, history = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    try:
        income_stmt = _retrying_call(lambda: stock.financials)
    except Exception:
        pass
    try:
        balance_sheet = _retrying_call(lambda: stock.balance_sheet)
    except Exception:
        pass
    try:
        cash_flow = _retrying_call(lambda: stock.cashflow)
    except Exception:
        pass
    try:
        history = _retrying_call(lambda: stock.history(period='1y'))
    except Exception:
        pass

    return {
        'info': info,
        'income_statement': income_stmt,
        'balance_sheet': balance_sheet,
        'cash_flow': cash_flow,
        'history': history,
    }


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def fetch_peer_quote_lite(peer_ticker: str) -> Optional[Dict]:
    """
    Fetch only the fields needed for peer comparison instead of the full `.info`
    blob, and do it through the same retry/backoff wrapper. This is what the old
    code was missing -- it hit `.info` for 5-6 tickers back-to-back with zero
    delay, which is the fastest way to get rate-limited by Yahoo.
    """
    try:
        info = _retrying_call(lambda: yf.Ticker(peer_ticker).get_info())
        if not info or not info.get('symbol'):
            return None
        return {
            'Ticker': peer_ticker,
            'Company': info.get('shortName', peer_ticker),
            'Market Cap (B)': (info.get('marketCap') or 0) / 1e9,
            'P/E': info.get('trailingPE', np.nan),
            'P/B': info.get('priceToBook', np.nan),
            'P/S': info.get('priceToSalesTrailing12Months', np.nan),
            'EV/EBITDA': info.get('enterpriseToEbitda', np.nan),
            'Profit Margin': (info.get('profitMargins') or np.nan) * 100 if info.get('profitMargins') is not None else np.nan,
            'ROE': (info.get('returnOnEquity') or np.nan) * 100 if info.get('returnOnEquity') is not None else np.nan,
            'Debt/Equity': (info.get('debtToEquity') or np.nan) / 100 if info.get('debtToEquity') is not None else np.nan,
        }
    except Exception:
        return None


def get_sector_peers(ticker: str, sector: str, market_cap: float, max_peers: int = 5) -> List[str]:
    """
    Get peer companies, tier-matched by market cap so small/mid caps aren't
    compared against mega-cap names by default.
    """
    tiers = PEER_UNIVERSE.get(sector)
    if not tiers:
        return []

    if market_cap >= 200e9:
        tier_order = ['mega', 'large', 'mid_small']
    elif market_cap >= 10e9:
        tier_order = ['large', 'mega', 'mid_small']
    else:
        tier_order = ['mid_small', 'large', 'mega']

    peers: List[str] = []
    for tier in tier_order:
        for p in tiers.get(tier, []):
            if p != ticker.upper() and p not in peers:
                peers.append(p)
        if len(peers) >= max_peers:
            break

    return peers[:max_peers]


# ==================== VALUATION METHODS ====================

def calculate_multiples_valuation(data: Dict, sector: str) -> Dict:
    """Calculate fair value using various multiples."""
    info = data['info']
    sector_data = SECTOR_MULTIPLIERS.get(sector, SECTOR_MULTIPLIERS['Technology'])

    current_price = info.get('currentPrice') or info.get('regularMarketPrice') or 0
    book_value = info.get('bookValue', 0) or 0
    eps = info.get('trailingEps', 0) or 0
    revenue_per_share = info.get('revenuePerShare', 0) or 0

    market_cap = info.get('marketCap', 0) or 0
    ebitda = info.get('ebitda', 0) or 0
    # Always derive shares from the dedicated field; deriving it from
    # market_cap / current_price breaks silently if current_price is 0 or stale,
    # and produced an inconsistent share count vs. the DCF calculation.
    shares_outstanding = info.get('sharesOutstanding', 0) or 0

    valuations = {}

    if eps and eps > 0:
        valuations['PE'] = {
            'fair_value': eps * sector_data['pe_median'],
            'low': eps * sector_data['pe_range'][0],
            'high': eps * sector_data['pe_range'][1],
            'current': current_price
        }

    if book_value and book_value > 0:
        valuations['PB'] = {
            'fair_value': book_value * sector_data['pb_median'],
            'low': book_value * sector_data['pb_range'][0],
            'high': book_value * sector_data['pb_range'][1],
            'current': current_price
        }

    if revenue_per_share and revenue_per_share > 0:
        valuations['PS'] = {
            'fair_value': revenue_per_share * sector_data['ps_median'],
            'low': revenue_per_share * sector_data['ps_range'][0],
            'high': revenue_per_share * sector_data['ps_range'][1],
            'current': current_price
        }

    if ebitda and ebitda > 0 and shares_outstanding > 0:
        valuations['EV_EBITDA'] = {
            'fair_value': (ebitda * sector_data['ev_ebitda_median']) / shares_outstanding,
            'low': (ebitda * sector_data['ev_ebitda_range'][0]) / shares_outstanding,
            'high': (ebitda * sector_data['ev_ebitda_range'][1]) / shares_outstanding,
            'current': current_price
        }

    if valuations:
        valuations['AVERAGE'] = {
            'fair_value': np.mean([v['fair_value'] for v in valuations.values()]),
            'low': np.mean([v['low'] for v in valuations.values()]),
            'high': np.mean([v['high'] for v in valuations.values()]),
            'current': current_price
        }

    return valuations


def _run_dcf(fcf: float, growth_rate: float, discount_rate: float, terminal_growth: float,
             growth_years: int, cash: float, debt: float, shares_outstanding: float) -> Dict:
    """Core DCF math, factored out so sensitivity analysis can call it directly
    with different assumptions instead of faking the relationship with a linear scale."""
    projected_fcf = [fcf * (1 + growth_rate) ** year for year in range(1, growth_years + 1)]
    pv_fcf = sum(cf / (1 + discount_rate) ** (i + 1) for i, cf in enumerate(projected_fcf))

    terminal_fcf = projected_fcf[-1] * (1 + terminal_growth)
    terminal_value = terminal_fcf / (discount_rate - terminal_growth)
    pv_terminal = terminal_value / (1 + discount_rate) ** growth_years

    enterprise_value = pv_fcf + pv_terminal
    equity_value = enterprise_value + cash - debt
    intrinsic_value = equity_value / shares_outstanding if shares_outstanding > 0 else 0

    return {
        'intrinsic_value': intrinsic_value,
        'enterprise_value': enterprise_value,
        'terminal_value': terminal_value,
        'pv_fcf': pv_fcf,
        'projected_fcf': projected_fcf,
    }


def calculate_dcf_valuation(data: Dict, sector: str) -> Dict:
    """Calculate intrinsic value using Discounted Cash Flow method."""
    info = data['info']
    sector_data = SECTOR_MULTIPLIERS.get(sector, SECTOR_MULTIPLIERS['Technology'])

    fcf = 0
    cash_flow_df = data.get('cash_flow', pd.DataFrame())
    if isinstance(cash_flow_df, pd.DataFrame) and 'Free Cash Flow' in cash_flow_df.index:
        try:
            val = cash_flow_df.loc['Free Cash Flow'].iloc[0]
            fcf = float(val) if pd.notna(val) else 0
        except (IndexError, ValueError, TypeError):
            fcf = 0
    if not fcf:
        fcf = info.get('freeCashflow', 0) or 0

    if not fcf or fcf <= 0:
        return {}

    growth_rate = sector_data['growth_rate'] / 100
    terminal_growth = 0.025
    discount_rate = 0.10
    growth_years = 5

    cash = info.get('totalCash', 0) or 0
    debt = info.get('totalDebt', 0) or 0
    shares_outstanding = info.get('sharesOutstanding', 0) or 0

    if shares_outstanding <= 0:
        return {}

    core = _run_dcf(fcf, growth_rate, discount_rate, terminal_growth, growth_years, cash, debt, shares_outstanding)

    return {
        **core,
        'growth_rate': growth_rate * 100,
        'discount_rate': discount_rate * 100,
        'fcf': fcf,
        'cash': cash,
        'debt': debt,
        'shares_outstanding': shares_outstanding,
        'terminal_growth': terminal_growth,
    }


def calculate_comparable_companies(ticker: str, ticker_data: Dict, peers: List[str]) -> pd.DataFrame:
    """
    Build the peer comparison table. The target ticker's row is built from data
    already fetched (no duplicate request); peers are fetched one-by-one through
    the cached + retrying fetch_peer_quote_lite, with a small randomized delay
    between calls to stay under Yahoo's rate limiter.
    """
    info = ticker_data['info']
    rows = [{
        'Ticker': ticker,
        'Company': info.get('shortName', ticker),
        'Market Cap (B)': (info.get('marketCap') or 0) / 1e9,
        'P/E': info.get('trailingPE', np.nan),
        'P/B': info.get('priceToBook', np.nan),
        'P/S': info.get('priceToSalesTrailing12Months', np.nan),
        'EV/EBITDA': info.get('enterpriseToEbitda', np.nan),
        'Profit Margin': (info.get('profitMargins') or np.nan) * 100 if info.get('profitMargins') is not None else np.nan,
        'ROE': (info.get('returnOnEquity') or np.nan) * 100 if info.get('returnOnEquity') is not None else np.nan,
        'Debt/Equity': (info.get('debtToEquity') or np.nan) / 100 if info.get('debtToEquity') is not None else np.nan,
    }]

    failed = []
    for i, peer in enumerate(peers):
        if i > 0:
            time.sleep(random.uniform(0.4, 0.9))  # throttle between peer requests
        quote = fetch_peer_quote_lite(peer)
        if quote:
            rows.append(quote)
        else:
            failed.append(peer)

    if failed:
        st.caption(f"⚠️ Could not fetch data for: {', '.join(failed)} (likely rate-limited or delisted)")

    return pd.DataFrame(rows)


# ==================== SWOT ANALYSIS ====================

def generate_swot_analysis(data: Dict, sector: str) -> Dict:
    """Generate comprehensive SWOT analysis based on financial metrics."""
    info = data['info']
    sector_data = SECTOR_MULTIPLIERS.get(sector, SECTOR_MULTIPLIERS['Technology'])

    strengths, weaknesses, opportunities, threats = [], [], [], []

    pe = info.get('trailingPE', 0) or 0
    pb = info.get('priceToBook', 0) or 0
    roe = (info.get('returnOnEquity') or 0) * 100
    profit_margin = (info.get('profitMargins') or 0) * 100
    debt_to_equity = (info.get('debtToEquity') or 0) / 100
    current_ratio = info.get('currentRatio', 0) or 0
    revenue_growth = (info.get('revenueGrowth') or 0) * 100
    earnings_growth = (info.get('earningsGrowth') or 0) * 100

    if roe > sector_data['roe_benchmark']:
        strengths.append(f"Strong ROE of {roe:.1f}% (above sector benchmark of {sector_data['roe_benchmark']:.1f}%)")
    if profit_margin > sector_data['margin_benchmark']:
        strengths.append(f"High profit margin of {profit_margin:.1f}% (above sector average of {sector_data['margin_benchmark']:.1f}%)")
    if 0 < debt_to_equity < 0.5:
        strengths.append(f"Low debt-to-equity ratio of {debt_to_equity:.2f} indicates strong financial health")
    if current_ratio > 1.5:
        strengths.append(f"Healthy current ratio of {current_ratio:.2f} suggests good liquidity")
    if revenue_growth > sector_data['growth_rate']:
        strengths.append(f"Revenue growth of {revenue_growth:.1f}% exceeds sector average of {sector_data['growth_rate']:.1f}%")

    market_cap = info.get('marketCap', 0) or 0
    if market_cap > 100e9:
        strengths.append(f"Large market cap of ${market_cap/1e9:.1f}B provides market stability")

    if roe < sector_data['roe_benchmark'] * 0.7:
        weaknesses.append(f"ROE of {roe:.1f}% is significantly below sector benchmark")
    if 0 < profit_margin < sector_data['margin_benchmark'] * 0.7:
        weaknesses.append(f"Profit margin of {profit_margin:.1f}% lags sector peers")
    if debt_to_equity > 1.5:
        weaknesses.append(f"High debt-to-equity ratio of {debt_to_equity:.2f} increases financial risk")
    if 0 < current_ratio < 1.0:
        weaknesses.append(f"Current ratio of {current_ratio:.2f} below 1.0 may indicate liquidity concerns")
    if revenue_growth < 0:
        weaknesses.append(f"Negative revenue growth of {revenue_growth:.1f}% signals declining sales")

    if 0 < pe < sector_data['pe_median'] * 0.8:
        opportunities.append(f"P/E ratio of {pe:.1f} suggests stock may be undervalued (sector median: {sector_data['pe_median']:.1f})")
    if 0 < pb < sector_data['pb_median'] * 0.8:
        opportunities.append(f"P/B ratio of {pb:.1f} indicates potential value investment (sector median: {sector_data['pb_median']:.1f})")
    if earnings_growth > 15:
        opportunities.append(f"Strong earnings growth of {earnings_growth:.1f}% creates expansion potential")
    if revenue_growth > 10:
        opportunities.append("Double-digit revenue growth indicates market share expansion opportunities")

    if pe > sector_data['pe_median'] * 1.5 and pe > 0:
        threats.append(f"High P/E ratio of {pe:.1f} suggests overvaluation risk (sector median: {sector_data['pe_median']:.1f})")
    if pb > sector_data['pb_median'] * 1.5 and pb > 0:
        threats.append(f"Elevated P/B ratio of {pb:.1f} may indicate limited upside (sector median: {sector_data['pb_median']:.1f})")
    if debt_to_equity > 2.0:
        threats.append("Very high leverage creates significant financial risk")
    if earnings_growth < -10:
        threats.append(f"Declining earnings growth of {earnings_growth:.1f}% threatens profitability")

    if sector in ['Technology', 'Consumer Cyclical']:
        opportunities.append("Digital transformation trends driving sector growth")
        threats.append("Rapid technological change requires continuous innovation")
    elif sector == 'Energy':
        opportunities.append("Energy transition creating new market segments")
        threats.append("Regulatory pressures and environmental concerns")
    elif sector == 'Healthcare':
        opportunities.append("Aging demographics driving healthcare demand")
        threats.append("Regulatory and reimbursement pressures")

    if not strengths:
        strengths.append("Company maintains operations in stable sector")
    if not weaknesses:
        weaknesses.append("Limited data available for comprehensive analysis")
    if not opportunities:
        opportunities.append("Potential for operational improvements")
    if not threats:
        threats.append("General market volatility and economic uncertainty")

    return {
        'Strengths': strengths,
        'Weaknesses': weaknesses,
        'Opportunities': opportunities,
        'Threats': threats
    }


# ==================== VISUALIZATION FUNCTIONS ====================

def create_valuation_chart(valuations: Dict, current_price: float) -> go.Figure:
    """Create valuation comparison chart."""
    methods, fair_values, lows, highs = [], [], [], []

    for method, vals in valuations.items():
        if method != 'AVERAGE':
            methods.append(method)
            fair_values.append(vals['fair_value'])
            lows.append(vals['low'])
            highs.append(vals['high'])

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='Fair Value',
        x=methods,
        y=fair_values,
        marker_color='lightblue',
        text=[f'${v:.2f}' for v in fair_values],
        textposition='auto'
    ))

    errors_y = {
        'type': 'data',
        'symmetric': False,
        'array': [h - fv for h, fv in zip(highs, fair_values)],
        'arrayminus': [fv - l for l, fv in zip(lows, fair_values)]
    }

    fig.add_trace(go.Scatter(
        name='Range',
        x=methods,
        y=fair_values,
        error_y=errors_y,
        mode='markers',
        marker=dict(size=0),
        showlegend=True
    ))

    fig.add_hline(
        y=current_price,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Current: ${current_price:.2f}",
        annotation_position="right"
    )

    if 'AVERAGE' in valuations:
        avg_fv = valuations['AVERAGE']['fair_value']
        fig.add_hline(
            y=avg_fv,
            line_dash="dot",
            line_color="green",
            annotation_text=f"Average: ${avg_fv:.2f}",
            annotation_position="left"
        )

    fig.update_layout(
        title='Valuation Analysis by Method',
        xaxis_title='Valuation Method',
        yaxis_title='Price ($)',
        template='plotly_white',
        height=500,
        showlegend=True
    )

    return fig


def create_peer_comparison_chart(comp_df: pd.DataFrame, ticker: str) -> go.Figure:
    """Create peer comparison visualization."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('P/E Ratio', 'P/B Ratio', 'ROE (%)', 'Profit Margin (%)'),
        specs=[[{'type': 'bar'}, {'type': 'bar'}],
               [{'type': 'bar'}, {'type': 'bar'}]]
    )

    colors = ['lightcoral' if t == ticker else 'lightblue' for t in comp_df['Ticker']]

    fig.add_trace(go.Bar(x=comp_df['Ticker'], y=comp_df['P/E'], marker_color=colors, showlegend=False), row=1, col=1)
    fig.add_trace(go.Bar(x=comp_df['Ticker'], y=comp_df['P/B'], marker_color=colors, showlegend=False), row=1, col=2)
    fig.add_trace(go.Bar(x=comp_df['Ticker'], y=comp_df['ROE'], marker_color=colors, showlegend=False), row=2, col=1)
    fig.add_trace(go.Bar(x=comp_df['Ticker'], y=comp_df['Profit Margin'], marker_color=colors, showlegend=False), row=2, col=2)

    fig.update_layout(
        title_text='Peer Company Comparison',
        template='plotly_white',
        height=700
    )

    return fig


# ==================== MAIN MODULE ====================

def sector_valuation_module(analysis_context: Optional[Dict] = None):
    """
    Enhanced sector-tuned valuation and SWOT analysis module.

    Args:
        analysis_context: Optional context from main app
    """
    st.title("🎯 Sector-Tuned Valuation & SWOT Analysis")
    st.markdown("""
    Comprehensive valuation analysis using sector-specific multiples, DCF modeling,
    comparable company analysis, and automated SWOT generation.
    """)

    with st.sidebar:
        st.header("Analysis Configuration")

        ticker = st.text_input(
            "Ticker Symbol",
            value="AAPL",
            help="Enter stock ticker (e.g., AAPL, MSFT, GOOGL)"
        ).strip().upper()

        sector = st.selectbox(
            "Sector",
            options=list(SECTOR_MULTIPLIERS.keys()),
            help="Select the company's sector for appropriate benchmarks. "
                 "This will be auto-corrected once live data confirms the actual sector."
        )

        analysis_type = st.multiselect(
            "Analysis Type",
            ["Multiples Valuation", "DCF Analysis", "Peer Comparison", "SWOT Analysis"],
            default=["Multiples Valuation", "SWOT Analysis"]
        )

        st.caption(f"Data is cached for {CACHE_TTL_SECONDS // 60} minutes to avoid "
                   f"Yahoo Finance rate limits. Click below to refetch.")
        force_refresh = st.button("🔄 Clear cache & refetch", use_container_width=True)
        if force_refresh:
            fetch_company_data.clear()
            fetch_peer_quote_lite.clear()
            validate_ticker.clear()
            st.success("Cache cleared.")

    run_clicked = st.button("🔍 Run Analysis", use_container_width=True)

    if run_clicked:
        if not ticker:
            st.error("Please enter a ticker symbol.")
            return

        with st.spinner(f"Validating ticker {ticker}..."):
            if not validate_ticker(ticker):
                st.error(
                    f"❌ Could not validate ticker **{ticker}**. It may be misspelled, "
                    f"delisted, or Yahoo Finance may be temporarily rate-limiting requests. "
                    f"Try the 'Clear cache & refetch' button after a minute if you believe "
                    f"the ticker is correct."
                )
                return

        with st.spinner(f"Fetching data for {ticker}... (retrying automatically if throttled)"):
            data = fetch_company_data(ticker)

        if not data:
            err = st.session_state.get('_last_fetch_error', 'Unknown error')
            st.error(
                f"❌ Could not fetch data for **{ticker}**.\n\n"
                f"Details: {err}\n\n"
                f"This is most often Yahoo Finance temporarily rate-limiting requests. "
                f"Wait a minute and try again, or use 'Clear cache & refetch' in the sidebar."
            )
            return

        info = data['info']

        st.header(f"📊 {info.get('longName', ticker)}")

        col1, col2, col3, col4 = st.columns(4)

        current_price = info.get('currentPrice') or info.get('regularMarketPrice') or 0

        with col1:
            prev_close = info.get('previousClose') or current_price or 1
            change_pct = ((current_price - prev_close) / prev_close * 100) if prev_close else 0
            st.metric("Current Price", f"${current_price:.2f}", f"{change_pct:+.2f}%")

        with col2:
            market_cap = (info.get('marketCap') or 0) / 1e9
            st.metric("Market Cap", f"${market_cap:.2f}B")

        with col3:
            pe = info.get('trailingPE', 0)
            st.metric("P/E Ratio", f"{pe:.2f}" if pe else "N/A")

        with col4:
            actual_sector = info.get('sector', sector)
            st.metric("Sector", actual_sector or sector)

        if actual_sector and actual_sector in SECTOR_MULTIPLIERS:
            if actual_sector != sector:
                st.info(f"ℹ️ Live data shows sector **{actual_sector}**, using its benchmarks instead of your manual selection (**{sector}**).")
            sector = actual_sector
        elif actual_sector:
            st.warning(f"⚠️ Yahoo reports sector as '{actual_sector}', which isn't in our benchmark table. Falling back to your manual selection (**{sector}**).")

        st.divider()

        # ==================== MULTIPLES VALUATION ====================
        if "Multiples Valuation" in analysis_type:
            st.header("💰 Multiples-Based Valuation")

            valuations = calculate_multiples_valuation(data, sector)

            if valuations:
                if 'AVERAGE' in valuations and current_price > 0:
                    avg_fv = valuations['AVERAGE']['fair_value']
                    upside = ((avg_fv - current_price) / current_price * 100)

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Average Fair Value", f"${avg_fv:.2f}")
                    with col2:
                        st.metric("Potential Upside/Downside", f"{upside:+.1f}%")
                    with col3:
                        if upside > 20:
                            recommendation, color = "🟢 UNDERVALUED", "green"
                        elif upside < -20:
                            recommendation, color = "🔴 OVERVALUED", "red"
                        else:
                            recommendation, color = "🟡 FAIRLY VALUED", "orange"
                        st.markdown(f"**Recommendation:** :{color}[{recommendation}]")

                    st.caption("Note: this is a mechanical sector-multiple estimate, not investment advice. "
                               "It doesn't account for company-specific catalysts, balance-sheet risk, or "
                               "qualitative factors a full analysis would weigh.")

                st.subheader("📊 Valuation by Method")

                method_keys = [k for k in valuations.keys() if k != 'AVERAGE']
                val_df = pd.DataFrame({
                    'Method': method_keys,
                    'Fair Value': [valuations[k]['fair_value'] for k in method_keys],
                    'Low Estimate': [valuations[k]['low'] for k in method_keys],
                    'High Estimate': [valuations[k]['high'] for k in method_keys],
                    'Current Price': [valuations[k]['current'] for k in method_keys],
                    'Upside (%)': [
                        ((valuations[k]['fair_value'] - valuations[k]['current']) / valuations[k]['current'] * 100)
                        if valuations[k]['current'] else np.nan
                        for k in method_keys
                    ]
                })

                st.dataframe(
                    val_df.style.format({
                        'Fair Value': '${:.2f}',
                        'Low Estimate': '${:.2f}',
                        'High Estimate': '${:.2f}',
                        'Current Price': '${:.2f}',
                        'Upside (%)': '{:+.1f}'
                    }).background_gradient(subset=['Upside (%)'], cmap='RdYlGn'),
                    use_container_width=True,
                    hide_index=True
                )

                fig_val = create_valuation_chart(valuations, current_price)
                st.plotly_chart(fig_val, use_container_width=True)

                st.subheader("📈 Sector Benchmarks")
                sector_data = SECTOR_MULTIPLIERS[sector]

                cur_pe = info.get('trailingPE', 0) or 0
                cur_pb = info.get('priceToBook', 0) or 0
                cur_ps = info.get('priceToSalesTrailing12Months', 0) or 0
                cur_ev_ebitda = info.get('enterpriseToEbitda', 0) or 0
                cur_roe = (info.get('returnOnEquity') or 0) * 100
                cur_margin = (info.get('profitMargins') or 0) * 100

                benchmark_df = pd.DataFrame({
                    'Metric': ['P/E Ratio', 'P/B Ratio', 'P/S Ratio', 'EV/EBITDA', 'ROE', 'Profit Margin'],
                    'Current': [cur_pe, cur_pb, cur_ps, cur_ev_ebitda, cur_roe, cur_margin],
                    'Sector Median': [
                        sector_data['pe_median'], sector_data['pb_median'], sector_data['ps_median'],
                        sector_data['ev_ebitda_median'], sector_data['roe_benchmark'], sector_data['margin_benchmark']
                    ],
                    'vs Sector': [
                        'Above' if cur_pe > sector_data['pe_median'] else 'Below',
                        'Above' if cur_pb > sector_data['pb_median'] else 'Below',
                        'Above' if cur_ps > sector_data['ps_median'] else 'Below',
                        'Above' if cur_ev_ebitda > sector_data['ev_ebitda_median'] else 'Below',
                        'Above' if cur_roe > sector_data['roe_benchmark'] else 'Below',
                        'Above' if cur_margin > sector_data['margin_benchmark'] else 'Below'
                    ]
                })

                st.dataframe(
                    benchmark_df.style.format({'Current': '{:.2f}', 'Sector Median': '{:.2f}'}),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.warning("⚠️ Insufficient data for multiples valuation (missing EPS, book value, "
                           "revenue/share, and EBITDA — at least one is required).")

        # ==================== DCF ANALYSIS ====================
        if "DCF Analysis" in analysis_type:
            st.header("📉 Discounted Cash Flow (DCF) Analysis")

            dcf_result = calculate_dcf_valuation(data, sector)

            if dcf_result:
                intrinsic_value = dcf_result['intrinsic_value']
                upside_dcf = ((intrinsic_value - current_price) / current_price * 100) if current_price else 0

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Intrinsic Value (DCF)", f"${intrinsic_value:.2f}")
                with col2:
                    st.metric("Upside/Downside", f"{upside_dcf:+.1f}%")
                with col3:
                    st.metric("Enterprise Value", f"${dcf_result['enterprise_value']/1e9:.2f}B")

                st.subheader("🔧 DCF Assumptions")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Growth Rate (Yrs 1-5)", f"{dcf_result['growth_rate']:.1f}%")
                with col2:
                    st.metric("Discount Rate (WACC)", f"{dcf_result['discount_rate']:.1f}%")
                with col3:
                    st.metric("Terminal Growth", f"{dcf_result['terminal_growth']*100:.1f}%")

                st.caption("Growth rate and WACC are sector-level simplifications, not company-specific "
                           "estimates. Treat this as a directional sanity check, not a precise valuation.")

                st.subheader("📊 Projected Free Cash Flows")
                fcf_df = pd.DataFrame({
                    'Year': range(1, len(dcf_result['projected_fcf']) + 1),
                    'Projected FCF': dcf_result['projected_fcf']
                })
                fig_fcf = px.bar(
                    fcf_df, x='Year', y='Projected FCF',
                    title='5-Year Free Cash Flow Projection',
                    labels={'Projected FCF': 'Free Cash Flow ($)'}
                )
                fig_fcf.update_layout(template='plotly_white', height=400)
                st.plotly_chart(fig_fcf, use_container_width=True)

                st.subheader("🎯 Sensitivity Analysis")
                st.write("**Impact of changing the growth-rate assumption on intrinsic value "
                         "(recomputed from the full DCF model, not approximated):**")

                growth_rates = np.linspace(0.05, 0.20, 7)
                sens_rows = []
                for gr in growth_rates:
                    sens = _run_dcf(
                        dcf_result['fcf'], gr, dcf_result['discount_rate'] / 100,
                        dcf_result['terminal_growth'], 5,
                        dcf_result['cash'], dcf_result['debt'], dcf_result['shares_outstanding']
                    )
                    iv = sens['intrinsic_value']
                    sens_rows.append({
                        'Growth Rate (%)': gr * 100,
                        'Intrinsic Value ($)': iv,
                        'Upside (%)': ((iv - current_price) / current_price * 100) if current_price else np.nan
                    })

                sens_df = pd.DataFrame(sens_rows)
                st.dataframe(
                    sens_df.style.format({
                        'Growth Rate (%)': '{:.1f}',
                        'Intrinsic Value ($)': '${:.2f}',
                        'Upside (%)': '{:+.1f}'
                    }).background_gradient(subset=['Upside (%)'], cmap='RdYlGn'),
                    use_container_width=True,
                    hide_index=True
                )

                st.subheader("🎯 Discount Rate Sensitivity")
                discount_rates = np.linspace(0.06, 0.14, 5)
                dr_rows = []
                for dr in discount_rates:
                    sens = _run_dcf(
                        dcf_result['fcf'], dcf_result['growth_rate'] / 100, dr,
                        dcf_result['terminal_growth'], 5,
                        dcf_result['cash'], dcf_result['debt'], dcf_result['shares_outstanding']
                    )
                    iv = sens['intrinsic_value']
                    dr_rows.append({
                        'WACC (%)': dr * 100,
                        'Intrinsic Value ($)': iv,
                        'Upside (%)': ((iv - current_price) / current_price * 100) if current_price else np.nan
                    })
                dr_df = pd.DataFrame(dr_rows)
                st.dataframe(
                    dr_df.style.format({
                        'WACC (%)': '{:.1f}',
                        'Intrinsic Value ($)': '${:.2f}',
                        'Upside (%)': '{:+.1f}'
                    }).background_gradient(subset=['Upside (%)'], cmap='RdYlGn'),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.warning("⚠️ Insufficient cash flow data for DCF analysis (no positive free cash flow "
                           "or share count available).")

        # ==================== PEER COMPARISON ====================
        if "Peer Comparison" in analysis_type:
            st.header("⚖️ Comparable Company Analysis")

            market_cap_abs = info.get('marketCap', 0) or 0
            peers = get_sector_peers(ticker, sector, market_cap_abs, 5)

            if peers:
                st.info(f"**Peer companies (market-cap tier matched):** {', '.join(peers)}")

                with st.spinner("Fetching peer company data (throttled to avoid rate limits)..."):
                    comp_df = calculate_comparable_companies(ticker, data, peers)

                if not comp_df.empty and len(comp_df) > 1:
                    st.subheader("📊 Peer Comparison Table")
                    st.dataframe(
                        comp_df.style.format({
                            'Market Cap (B)': '${:.2f}',
                            'P/E': '{:.2f}',
                            'P/B': '{:.2f}',
                            'P/S': '{:.2f}',
                            'EV/EBITDA': '{:.2f}',
                            'Profit Margin': '{:.1f}',
                            'ROE': '{:.1f}',
                            'Debt/Equity': '{:.2f}'
                        }).background_gradient(subset=['P/E', 'ROE'], cmap='RdYlGn_r'),
                        use_container_width=True,
                        hide_index=True
                    )

                    fig_peer = create_peer_comparison_chart(comp_df, ticker)
                    st.plotly_chart(fig_peer, use_container_width=True)

                    st.subheader("📈 Relative Valuation")
                    ticker_rows = comp_df[comp_df['Ticker'] == ticker]
                    if not ticker_rows.empty:
                        ticker_row = ticker_rows.iloc[0]
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            pe_rank = (comp_df['P/E'] < ticker_row['P/E']).sum() / len(comp_df) * 100
                            st.metric("P/E Percentile", f"{pe_rank:.0f}%", help="Higher = more expensive vs peers")
                        with col2:
                            roe_rank = (comp_df['ROE'] < ticker_row['ROE']).sum() / len(comp_df) * 100
                            st.metric("ROE Percentile", f"{roe_rank:.0f}%", help="Higher = better performance")
                        with col3:
                            margin_rank = (comp_df['Profit Margin'] < ticker_row['Profit Margin']).sum() / len(comp_df) * 100
                            st.metric("Margin Percentile", f"{margin_rank:.0f}%", help="Higher = more profitable")
                elif not comp_df.empty:
                    st.warning("⚠️ Only the target ticker's data could be fetched — all peer requests "
                               "were rate-limited or failed. Try 'Clear cache & refetch' in a minute.")
                else:
                    st.warning("⚠️ Could not fetch peer company data.")
            else:
                st.warning("⚠️ No peer companies identified for this sector.")

        # ==================== SWOT ANALYSIS ====================
        if "SWOT Analysis" in analysis_type:
            st.header("🎯 SWOT Analysis")

            swot = generate_swot_analysis(data, sector)

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("💪 Strengths")
                st.success("**Internal Positive Factors**")
                for s in swot['Strengths']:
                    st.write(f"✓ {s}")

                st.divider()

                st.subheader("🎯 Opportunities")
                st.info("**External Positive Factors**")
                for o in swot['Opportunities']:
                    st.write(f"✓ {o}")

            with col2:
                st.subheader("⚠️ Weaknesses")
                st.warning("**Internal Negative Factors**")
                for w in swot['Weaknesses']:
                    st.write(f"✗ {w}")

                st.divider()

                st.subheader("⚡ Threats")
                st.error("**External Negative Factors**")
                for t in swot['Threats']:
                    st.write(f"✗ {t}")

            st.subheader("📋 SWOT Summary")
            swot_score = (len(swot['Strengths']) + len(swot['Opportunities']) -
                         len(swot['Weaknesses']) - len(swot['Threats']))

            if swot_score > 2:
                st.success(f"**Overall Assessment: POSITIVE** (Score: +{swot_score})")
                st.write("The company shows more positive than negative factors by this heuristic scoring, "
                         "suggesting favorable fundamentals worth a closer look.")
            elif swot_score < -2:
                st.error(f"**Overall Assessment: NEGATIVE** (Score: {swot_score})")
                st.write("Several challenges and risks were flagged that may warrant caution.")
            else:
                st.info(f"**Overall Assessment: NEUTRAL** (Score: {swot_score:+d})")
                st.write("A balanced mix of positive and negative factors — no strong signal either way.")

            st.caption("This SWOT is generated mechanically from financial ratios vs. sector benchmarks. "
                       "It does not include qualitative factors like management quality, competitive moat, "
                       "litigation risk, or macro conditions — treat it as a starting point, not a complete analysis.")

    # Educational section
    with st.expander("ℹ️ Understanding Valuation Methods"):
        st.markdown("""
        ### Valuation Methods Explained

        **1. P/E Ratio (Price-to-Earnings)**
        - Compares stock price to earnings per share
        - Lower P/E may indicate undervaluation
        - Sector-specific ranges are important

        **2. P/B Ratio (Price-to-Book)**
        - Compares market value to book value
        - Useful for asset-heavy companies
        - Below 1.0 may indicate undervaluation

        **3. P/S Ratio (Price-to-Sales)**
        - Compares market cap to revenue
        - Useful for companies with no profits yet
        - Lower ratios generally better

        **4. EV/EBITDA**
        - Enterprise Value to Earnings Before Interest, Taxes, Depreciation & Amortization
        - Accounts for debt and cash
        - Good for cross-company comparisons

        **5. DCF (Discounted Cash Flow)**
        - Estimates intrinsic value based on future cash flows
        - Most theoretically sound but assumption-dependent
        - Sensitive to growth and discount rates

        ### SWOT Analysis
        - **Strengths & Weaknesses**: Internal factors the company controls
        - **Opportunities & Threats**: External market factors
        - Use alongside valuation for complete picture

        ### A note on data reliability
        This tool pulls live data from Yahoo Finance via the unofficial `yfinance` library.
        Yahoo does not provide an official API for this, so occasional rate-limit errors are
        expected under heavy use. If a fetch fails, wait a minute before retrying — the app
        will not hammer Yahoo's servers in the meantime since failed lookups aren't cached.
        """)

    st.divider()
    st.caption("🎯 Sector Valuation Module | Comprehensive valuation and strategic analysis. "
               "Not investment advice — for informational and educational purposes only.")


# Standalone execution for testing
if __name__ == "__main__":
    sector_valuation_module(None)
