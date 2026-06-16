# economics_enhanced.py - Advanced Macro/Economic Context Module
# Features: FRED API integration, Yield Curve Analysis, Economic Cycle Detection, Recession Indicators

import streamlit as st
import pandas as pd
import requests
from io import StringIO
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple
import numpy as np
from scipy import signal
from sklearn.preprocessing import StandardScaler

# FRED API Configuration
FRED_API_KEY = "your_fred_api_key_here"  # Get free key from https://fred.stlouisfed.org/docs/api/api_key.html
FRED_API_URL = "https://api.stlouisfed.org/fred/series/observations"

# Enhanced Metric definitions with working FRED series
METRIC_MAPPING = {
    "GDP": {
        "fred_code": "GDP",
        "unit": "Billions USD",
        "description": "Gross Domestic Product",
        "inverse": False,
        "category": "output",
        "frequency": "quarterly"
    },
    "GDP_Growth": {
        "fred_code": "A191RL1Q225SBEA",
        "unit": "Percent",
        "description": "Real GDP Growth Rate",
        "inverse": False,
        "category": "output",
        "frequency": "quarterly"
    },
    "Inflation_CPI": {
        "fred_code": "CPIAUCSL",
        "unit": "Index (1982-84=100)",
        "description": "Consumer Price Index",
        "inverse": True,
        "category": "prices",
        "frequency": "monthly"
    },
    "Core_Inflation": {
        "fred_code": "CPILFESL",
        "unit": "Index (1982-84=100)",
        "description": "Core CPI (excl. Food & Energy)",
        "inverse": True,
        "category": "prices",
        "frequency": "monthly"
    },
    "PCE_Inflation": {
        "fred_code": "PCEPI",
        "unit": "Index (2017=100)",
        "description": "PCE Price Index",
        "inverse": True,
        "category": "prices",
        "frequency": "monthly"
    },
    "Unemployment": {
        "fred_code": "UNRATE",
        "unit": "Percentage",
        "description": "Unemployment Rate",
        "inverse": True,
        "category": "labor",
        "frequency": "monthly"
    },
    "Initial_Claims": {
        "fred_code": "ICSA",
        "unit": "Thousands",
        "description": "Initial Jobless Claims",
        "inverse": True,
        "category": "labor",
        "frequency": "weekly"
    },
    "Payrolls": {
        "fred_code": "PAYEMS",
        "unit": "Thousands",
        "description": "Total Nonfarm Payrolls",
        "inverse": False,
        "category": "labor",
        "frequency": "monthly"
    },
    "Fed_Funds_Rate": {
        "fred_code": "FEDFUNDS",
        "unit": "Percentage",
        "description": "Federal Funds Effective Rate",
        "inverse": False,
        "category": "rates",
        "frequency": "monthly"
    },
    "10Y_Treasury": {
        "fred_code": "DGS10",
        "unit": "Percentage",
        "description": "10-Year Treasury Rate",
        "inverse": False,
        "category": "rates",
        "frequency": "daily"
    },
    "2Y_Treasury": {
        "fred_code": "DGS2",
        "unit": "Percentage",
        "description": "2-Year Treasury Rate",
        "inverse": False,
        "category": "rates",
        "frequency": "daily"
    },
    "3M_Treasury": {
        "fred_code": "DGS3MO",
        "unit": "Percentage",
        "description": "3-Month Treasury Rate",
        "inverse": False,
        "category": "rates",
        "frequency": "daily"
    },
    "5Y_Treasury": {
        "fred_code": "DGS5",
        "unit": "Percentage",
        "description": "5-Year Treasury Rate",
        "inverse": False,
        "category": "rates",
        "frequency": "daily"
    },
    "30Y_Treasury": {
        "fred_code": "DGS30",
        "unit": "Percentage",
        "description": "30-Year Treasury Rate",
        "inverse": False,
        "category": "rates",
        "frequency": "daily"
    },
    "Consumer_Sentiment": {
        "fred_code": "UMCSENT",
        "unit": "Index (1966:Q1=100)",
        "description": "University of Michigan Consumer Sentiment",
        "inverse": False,
        "category": "confidence",
        "frequency": "monthly"
    },
    "Industrial_Production": {
        "fred_code": "INDPRO",
        "unit": "Index (2017=100)",
        "description": "Industrial Production Index",
        "inverse": False,
        "category": "output",
        "frequency": "monthly"
    },
    "Capacity_Utilization": {
        "fred_code": "TCU",
        "unit": "Percentage",
        "description": "Capacity Utilization",
        "inverse": False,
        "category": "output",
        "frequency": "monthly"
    },
    "Housing_Starts": {
        "fred_code": "HOUST",
        "unit": "Thousands of Units",
        "description": "Housing Starts",
        "inverse": False,
        "category": "housing",
        "frequency": "monthly"
    },
    "Retail_Sales": {
        "fred_code": "RSXFS",
        "unit": "Millions of Dollars",
        "description": "Advance Retail Sales",
        "inverse": False,
        "category": "consumption",
        "frequency": "monthly"
    },
    "Personal_Spending": {
        "fred_code": "PCE",
        "unit": "Billions of Dollars",
        "description": "Personal Consumption Expenditures",
        "inverse": False,
        "category": "consumption",
        "frequency": "monthly"
    },
    "Leading_Index": {
        "fred_code": "USSLIND",
        "unit": "Index (2016=100)",
        "description": "Leading Economic Index",
        "inverse": False,
        "category": "composite",
        "frequency": "monthly"
    }
}

# Yield curve maturities (in order)
YIELD_CURVE_RATES = {
    "1M": "DGS1MO",
    "3M": "DGS3MO",
    "6M": "DGS6MO",
    "1Y": "DGS1",
    "2Y": "DGS2",
    "3Y": "DGS3",
    "5Y": "DGS5",
    "7Y": "DGS7",
    "10Y": "DGS10",
    "20Y": "DGS20",
    "30Y": "DGS30"
}

# Minimum recommended date range (years) for cycle detection
MIN_YEARS_FOR_CYCLE = 5

# ==================== CYCLE INTERPRETATION DATA ====================

CYCLE_PLAYBOOK = {
    "Early Expansion": {
        "description": "Economy emerging from trough. Growth accelerating, unemployment falling, credit loosening.",
        "typical_duration": "12–18 months",
        "next_phase": "Late Expansion",
        "next_phase_signals": [
            "Unemployment approaches multi-decade lows",
            "Inflation begins rising above 3%",
            "Fed starts hiking rates",
            "Yield curve flattening",
            "Consumer sentiment near highs"
        ],
        "macro_backdrop": "Strong earnings growth, rising revenues, credit expansion. Risk appetite high. Commodities start rallying.",
        "asset_recommendations": {
            "Equities": {
                "rating": "⭐⭐⭐⭐⭐ STRONG BUY",
                "sectors": ["Financials 🏦", "Consumer Discretionary 🛒", "Industrials 🏭", "Materials ⛏️", "Small Caps 📈"],
                "rationale": "Cyclicals outperform as growth accelerates. Small caps benefit from improved credit conditions."
            },
            "Fixed Income": {
                "rating": "⭐⭐ UNDERWEIGHT",
                "sectors": ["High Yield Bonds", "Short Duration"],
                "rationale": "Rising rates hurt long bonds. HY spreads tighten as default risk falls."
            },
            "Real Assets": {
                "rating": "⭐⭐⭐⭐ BUY",
                "sectors": ["Commodities", "Real Estate (REITs)", "Energy"],
                "rationale": "Demand recovery boosts commodity prices. REITs benefit from economic recovery."
            },
            "Cash": {
                "rating": "⭐ AVOID",
                "rationale": "Opportunity cost too high — deploy into risk assets."
            },
            "Alternatives": {
                "rating": "⭐⭐⭐ NEUTRAL",
                "sectors": ["Private Equity", "Infrastructure"],
                "rationale": "Long-term commitments payoff well when entered at cycle bottom."
            }
        },
        "avoid": ["Long-duration Treasuries", "Utilities", "Defensive Consumer Staples"],
        "color": "#2ecc71",
        "emoji": "🚀"
    },
    "Late Expansion": {
        "description": "Peak growth. Labor markets tight, inflation rising, Fed tightening. Economy running hot.",
        "typical_duration": "12–24 months",
        "next_phase": "Early Contraction",
        "next_phase_signals": [
            "Yield curve inverts (10Y-2Y < 0)",
            "Leading Economic Index turns negative",
            "Consumer confidence peaks and turns",
            "Fed pauses or pivots",
            "Credit spreads begin widening",
            "ISM Manufacturing below 50"
        ],
        "macro_backdrop": "Earnings still positive but growth slowing. Margins squeezed by wages and input costs. Valuation stretched.",
        "asset_recommendations": {
            "Equities": {
                "rating": "⭐⭐⭐ NEUTRAL → REDUCE",
                "sectors": ["Energy ⚡", "Healthcare 🏥", "Consumer Staples 🛒", "Dividend Quality 💰"],
                "rationale": "Shift to quality and pricing power. Avoid rate-sensitive growth stocks. Energy benefits from tight supply."
            },
            "Fixed Income": {
                "rating": "⭐⭐⭐ NEUTRAL",
                "sectors": ["TIPS (Inflation-Protected)", "Short-to-Medium Duration", "Investment Grade"],
                "rationale": "TIPS hedge inflation risk. Avoid long duration as rates still rising. IG over HY as credit cycle matures."
            },
            "Real Assets": {
                "rating": "⭐⭐⭐⭐ BUY",
                "sectors": ["Commodities (Energy, Metals)", "Gold 🥇", "Infrastructure"],
                "rationale": "Inflation hedge. Commodities in super-cycle. Gold rallies as real rates peak."
            },
            "Cash": {
                "rating": "⭐⭐⭐ BUILD POSITION",
                "rationale": "Money market yields attractive. Dry powder for the coming contraction."
            },
            "Alternatives": {
                "rating": "⭐⭐⭐ NEUTRAL",
                "sectors": ["Hedge Funds (Market Neutral)", "Commodities CTAs"],
                "rationale": "Volatility strategies and trend-following perform well as trends shift."
            }
        },
        "avoid": ["High-growth tech (rate sensitive)", "Speculative small caps", "Highly leveraged companies"],
        "color": "#f39c12",
        "emoji": "⚠️"
    },
    "Early Contraction": {
        "description": "Growth decelerating sharply. Credit tightening, layoffs beginning. Possible recession ahead.",
        "typical_duration": "6–12 months",
        "next_phase": "Late Contraction",
        "next_phase_signals": [
            "GDP growth turns negative",
            "Unemployment rising > 1% from cycle low",
            "Sahm Rule triggered",
            "Fed begins cutting rates",
            "Credit spreads spike > 400 bps",
            "Consumer sentiment at multi-year lows"
        ],
        "macro_backdrop": "Earnings revisions negative. Revenue growth stalls. Credit defaults rising. Risk-off sentiment dominates.",
        "asset_recommendations": {
            "Equities": {
                "rating": "⭐⭐ UNDERWEIGHT",
                "sectors": ["Utilities 💡", "Consumer Staples 🥫", "Healthcare 🏥", "Low Volatility Factors"],
                "rationale": "Defensive sectors outperform. Avoid cyclicals and growth. Quality over momentum."
            },
            "Fixed Income": {
                "rating": "⭐⭐⭐⭐⭐ STRONG BUY",
                "sectors": ["Long-Duration Treasuries 🏛️", "Government Bonds", "High-Grade IG Corporates"],
                "rationale": "Flight to safety drives Treasury prices up. Duration pays off as rates fall. Avoid HY — spreads widen."
            },
            "Real Assets": {
                "rating": "⭐⭐ REDUCE",
                "sectors": ["Gold 🥇 (safe haven)"],
                "rationale": "Commodities typically fall on demand destruction. Gold holds value as crisis hedge."
            },
            "Cash": {
                "rating": "⭐⭐⭐⭐ OVERWEIGHT",
                "rationale": "Capital preservation priority. High cash allocation protects against drawdown."
            },
            "Alternatives": {
                "rating": "⭐⭐⭐ BUY",
                "sectors": ["Long/Short Equity", "Global Macro", "Managed Futures"],
                "rationale": "Absolute return strategies shine in volatile, trending markets."
            }
        },
        "avoid": ["Cyclicals", "Financials (credit risk)", "High Yield bonds", "Leveraged loans", "Emerging Markets"],
        "color": "#e74c3c",
        "emoji": "📉"
    },
    "Late Contraction": {
        "description": "Recession underway. Maximum pessimism. Unemployment high, GDP negative, but leading indicators may be bottoming.",
        "typical_duration": "6–12 months",
        "next_phase": "Early Expansion",
        "next_phase_signals": [
            "LEI starts recovering for 2+ consecutive months",
            "Initial jobless claims peak and decline",
            "ISM Manufacturing stabilizes above 45",
            "Yield curve steepens sharply",
            "Fed cutting aggressively / QE begins",
            "Housing activity bottoms"
        ],
        "macro_backdrop": "Maximum fear and uncertainty. Forced selling. Value opportunities emerging. Smart money begins accumulating.",
        "asset_recommendations": {
            "Equities": {
                "rating": "⭐⭐⭐ BEGIN ACCUMULATING",
                "sectors": ["Value Stocks 💎", "Beaten-down Cyclicals", "Financials (cautiously)", "Small Caps (for recovery)"],
                "rationale": "Best long-term entry points. Contrarian accumulation of quality names at discounted valuations."
            },
            "Fixed Income": {
                "rating": "⭐⭐⭐⭐ OVERWEIGHT",
                "sectors": ["Treasuries (begin shortening duration)", "High Yield (selectively)", "Convertibles"],
                "rationale": "Rates bottoming — start reducing duration. HY at peak spreads offer asymmetric reward for recovery."
            },
            "Real Assets": {
                "rating": "⭐⭐⭐ NEUTRAL → BUILD",
                "sectors": ["Commodities (for recovery)", "Real Estate (distressed)"],
                "rationale": "Bottom of commodity cycle. Early accumulation pays off in next expansion."
            },
            "Cash": {
                "rating": "⭐⭐⭐ DEPLOY GRADUALLY",
                "rationale": "Deploy cash in tranches into beaten-down assets. Don't wait for 'all clear'."
            },
            "Alternatives": {
                "rating": "⭐⭐⭐⭐ BUY",
                "sectors": ["Distressed Debt", "Private Credit", "Special Situations"],
                "rationale": "Distressed cycles create exceptional opportunities for patient capital."
            }
        },
        "avoid": ["Highly leveraged companies at risk of default", "Sectors with structural headwinds"],
        "color": "#8e44ad",
        "emoji": "🔴"
    },
    "Expansion": {
        "description": "Economic expansion phase. GDP growing, unemployment falling, positive business conditions.",
        "typical_duration": "24–48 months",
        "next_phase": "Contraction",
        "next_phase_signals": [
            "Yield curve inversion",
            "Leading indicators turning negative",
            "Unemployment bottoming",
            "Fed policy tightening",
        ],
        "macro_backdrop": "Growth positive, credit available, confidence elevated.",
        "asset_recommendations": {
            "Equities": {"rating": "⭐⭐⭐⭐ BUY", "sectors": ["Broad market", "Cyclicals", "Growth"], "rationale": "Risk-on environment favors equities."},
            "Fixed Income": {"rating": "⭐⭐ UNDERWEIGHT", "sectors": ["Short duration"], "rationale": "Rising rate environment."},
            "Real Assets": {"rating": "⭐⭐⭐ NEUTRAL", "sectors": ["Commodities", "REITs"], "rationale": "Demand growth supports prices."},
            "Cash": {"rating": "⭐ LOW", "rationale": "Deploy into risk assets."},
            "Alternatives": {"rating": "⭐⭐⭐ NEUTRAL", "sectors": ["Private Equity"], "rationale": "Leverage benefits from growth."}
        },
        "avoid": ["Long-duration bonds", "Extreme defensives"],
        "color": "#27ae60",
        "emoji": "📈"
    },
    "Contraction": {
        "description": "Economic contraction phase. Growth slowing or negative, rising unemployment, tightening financial conditions.",
        "typical_duration": "6–18 months",
        "next_phase": "Expansion",
        "next_phase_signals": [
            "Leading indicators bottoming",
            "Yield curve steepening",
            "Fed cutting rates",
            "Jobless claims peaking"
        ],
        "macro_backdrop": "Risk-off. Capital preservation priority. Opportunities emerging at cycle trough.",
        "asset_recommendations": {
            "Equities": {"rating": "⭐⭐ UNDERWEIGHT", "sectors": ["Defensives", "Quality"], "rationale": "Capital preservation."},
            "Fixed Income": {"rating": "⭐⭐⭐⭐⭐ BUY", "sectors": ["Treasuries", "IG Bonds"], "rationale": "Safe haven demand."},
            "Real Assets": {"rating": "⭐⭐ REDUCE", "sectors": ["Gold"], "rationale": "Gold as crisis hedge."},
            "Cash": {"rating": "⭐⭐⭐⭐ OVERWEIGHT", "rationale": "Capital preservation."},
            "Alternatives": {"rating": "⭐⭐⭐ BUY", "sectors": ["Hedge Funds", "Managed Futures"], "rationale": "Absolute return in volatile markets."}
        },
        "avoid": ["Cyclicals", "High Yield", "Leveraged plays"],
        "color": "#c0392b",
        "emoji": "📉"
    }
}


# ==================== DATA FETCHING ====================

@st.cache_data(ttl=3600)
def get_fred_data_api(series_id: str, start_date: str, end_date: str, api_key: Optional[str] = None) -> pd.DataFrame:
    """Fetch data from FRED API (requires API key)."""
    if not api_key or api_key == "your_fred_api_key_here":
        return pd.DataFrame()

    try:
        params = {
            'series_id': series_id,
            'api_key': api_key,
            'file_type': 'json',
            'observation_start': start_date,
            'observation_end': end_date
        }

        response = requests.get(FRED_API_URL, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()

        if 'observations' in data:
            df = pd.DataFrame(data['observations'])
            df['date'] = pd.to_datetime(df['date'])
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            df = df.set_index('date')[['value']]
            df.columns = [series_id]
            return df.dropna()

        return pd.DataFrame()

    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def get_fred_data_csv(series_id: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch data from FRED CSV download (no API key required)."""
    try:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"

        response = requests.get(url, timeout=30)
        response.raise_for_status()

        df = pd.read_csv(StringIO(response.text), index_col=0, parse_dates=True)
        df.columns = [series_id]

        df = df.loc[start_date:end_date]
        df[series_id] = pd.to_numeric(df[series_id], errors='coerce')
        return df.dropna()

    except Exception:
        return pd.DataFrame()


def infer_frequency(series: pd.Series) -> str:
    """Infer frequency of a time series."""
    if len(series) < 2:
        return 'monthly'
    median_days = pd.Series(series.index).diff().dt.days.median()
    if median_days <= 2:
        return 'daily'
    elif median_days <= 10:
        return 'weekly'
    elif median_days <= 40:
        return 'monthly'
    else:
        return 'quarterly'


def resample_to_monthly(series: pd.Series) -> pd.Series:
    """Resample any frequency series to monthly for alignment."""
    freq = infer_frequency(series)
    if freq == 'daily' or freq == 'weekly':
        return series.resample('ME').mean()
    elif freq == 'quarterly':
        # Forward-fill quarterly data to monthly
        return series.resample('ME').ffill()
    return series.resample('ME').last()


# ==================== ANALYTICS ====================

def calculate_growth_rate(data: pd.Series, periods: int = 4) -> pd.Series:
    """Calculate year-over-year growth rate."""
    return ((data / data.shift(periods)) - 1) * 100


def calculate_statistics(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate key statistics for the data."""
    if data.empty:
        return pd.DataFrame()

    series = data.iloc[:, 0]
    stats = {
        'Current': series.iloc[-1] if len(series) > 0 else np.nan,
        'Mean': series.mean(),
        'Median': series.median(),
        'Std Dev': series.std(),
        'Min': series.min(),
        'Max': series.max(),
        '1Y Change (%)': ((series.iloc[-1] - series.iloc[-13]) / abs(series.iloc[-13]) * 100) if len(series) >= 13 else np.nan
    }
    return pd.DataFrame(stats, index=['Value']).T


def calculate_correlations(data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Calculate correlations between different metrics."""
    if len(data_dict) < 2:
        return pd.DataFrame()

    combined = pd.DataFrame()
    for metric, df in data_dict.items():
        if not df.empty and len(df.columns) > 0:
            s = df.iloc[:, 0]
            # Resample to monthly for alignment
            combined[metric] = resample_to_monthly(s)

    if not combined.empty and len(combined.columns) > 1:
        combined = combined.dropna()
        if len(combined) > 10:
            return combined.corr()

    return pd.DataFrame()


# ==================== ECONOMIC HEALTH SCORECARD ====================

def compute_health_scorecard(data_dict: Dict[str, pd.DataFrame], spreads: pd.DataFrame) -> Dict:
    """Compute an economic health score across multiple dimensions."""
    scores = {}

    # GDP / Growth (0-100)
    if 'GDP_Growth' in data_dict and not data_dict['GDP_Growth'].empty:
        g = data_dict['GDP_Growth'].iloc[-1, 0]
        scores['Growth'] = min(100, max(0, (g + 2) / 6 * 100))
    elif 'Industrial_Production' in data_dict and not data_dict['Industrial_Production'].empty:
        ip = data_dict['Industrial_Production'].iloc[:, 0]
        g = calculate_growth_rate(ip, 12).iloc[-1] if len(ip) > 12 else 0
        scores['Growth'] = min(100, max(0, (g + 3) / 8 * 100))

    # Labor (0-100)
    if 'Unemployment' in data_dict and not data_dict['Unemployment'].empty:
        u = data_dict['Unemployment'].iloc[-1, 0]
        scores['Labor'] = min(100, max(0, (10 - u) / 7 * 100))

    # Inflation (0-100 — sweet spot is 1.5-2.5%)
    for m in ['Core_Inflation', 'Inflation_CPI', 'PCE_Inflation']:
        if m in data_dict and not data_dict[m].empty:
            inf = data_dict[m].iloc[:, 0]
            yoy = calculate_growth_rate(inf, 12).iloc[-1] if len(inf) > 12 else 3
            # Score peaks at 2%, falls off for high or low inflation
            scores['Inflation'] = min(100, max(0, 100 - abs(yoy - 2) * 18))
            break

    # Yield curve health (0-100)
    if not spreads.empty and '10Y-2Y' in spreads.columns:
        sp = spreads['10Y-2Y'].iloc[-1]
        # Positive steep curve = good, flat/inverted = bad
        scores['Yield Curve'] = min(100, max(0, (sp + 1) / 3 * 100))

    # Consumer confidence (0-100)
    if 'Consumer_Sentiment' in data_dict and not data_dict['Consumer_Sentiment'].empty:
        sent = data_dict['Consumer_Sentiment'].iloc[:, 0]
        current = sent.iloc[-1]
        hist_mean = sent.mean()
        scores['Confidence'] = min(100, max(0, current / hist_mean * 50))

    # Leading indicators (0-100)
    if 'Leading_Index' in data_dict and not data_dict['Leading_Index'].empty:
        lei = data_dict['Leading_Index'].iloc[:, 0]
        if len(lei) > 6:
            change = lei.pct_change(6).iloc[-1] * 100
            scores['Leading'] = min(100, max(0, (change + 5) / 10 * 100))

    overall = np.mean(list(scores.values())) if scores else 50
    scores['Overall'] = overall

    return scores


def plot_health_scorecard(scores: Dict) -> go.Figure:
    """Render a radar chart for economic health."""
    dims = [k for k in scores if k != 'Overall']
    vals = [scores[k] for k in dims]

    # Close the polygon
    dims_closed = dims + [dims[0]]
    vals_closed = vals + [vals[0]]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=vals_closed,
        theta=dims_closed,
        fill='toself',
        name='Health Score',
        fillcolor='rgba(46, 204, 113, 0.25)',
        line=dict(color='#2ecc71', width=2),
        marker=dict(size=7)
    ))

    # Reference ring at 50
    ref_vals = [50] * len(dims_closed)
    fig.add_trace(go.Scatterpolar(
        r=ref_vals,
        theta=dims_closed,
        name='Neutral (50)',
        line=dict(color='gray', width=1, dash='dot'),
        showlegend=True,
        fill=None
    ))

    overall = scores.get('Overall', 50)
    if overall >= 65:
        color = '#2ecc71'
        label = 'Healthy'
    elif overall >= 40:
        color = '#f39c12'
        label = 'Mixed'
    else:
        color = '#e74c3c'
        label = 'Stressed'

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=9))),
        showlegend=True,
        height=380,
        title=dict(text=f"Economic Health: <b style='color:{color}'>{overall:.0f}/100 — {label}</b>", x=0.5),
        template='plotly_white',
        margin=dict(t=60, b=20)
    )

    return fig


# ==================== YIELD CURVE ANALYSIS ====================

def get_yield_curve_data(date: datetime, use_api: bool = False, api_key: Optional[str] = None) -> pd.Series:
    """Fetch yield curve data for a specific date."""
    yields = {}
    date_str = date.strftime("%Y-%m-%d")

    for maturity, fred_code in YIELD_CURVE_RATES.items():
        if use_api and api_key:
            df = get_fred_data_api(fred_code, date_str, date_str, api_key)
        else:
            df = get_fred_data_csv(fred_code, date_str, date_str)

        if not df.empty:
            yields[maturity] = df.iloc[-1, 0]

    return pd.Series(yields) if yields else pd.Series()


def get_yield_curve_historical(start_date: str, end_date: str, use_api: bool = False, api_key: Optional[str] = None) -> pd.DataFrame:
    """Fetch historical yield curve data."""
    all_curves = pd.DataFrame()

    for maturity, fred_code in YIELD_CURVE_RATES.items():
        if use_api and api_key:
            df = get_fred_data_api(fred_code, start_date, end_date, api_key)
        else:
            df = get_fred_data_csv(fred_code, start_date, end_date)

        if not df.empty:
            all_curves[maturity] = df.iloc[:, 0]

    return all_curves


def calculate_yield_spreads(yield_curve_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate key yield spreads."""
    spreads = pd.DataFrame(index=yield_curve_df.index)

    if '10Y' in yield_curve_df.columns and '2Y' in yield_curve_df.columns:
        spreads['10Y-2Y'] = yield_curve_df['10Y'] - yield_curve_df['2Y']

    if '10Y' in yield_curve_df.columns and '3M' in yield_curve_df.columns:
        spreads['10Y-3M'] = yield_curve_df['10Y'] - yield_curve_df['3M']

    if '2Y' in yield_curve_df.columns and '3M' in yield_curve_df.columns:
        spreads['2Y-3M'] = yield_curve_df['2Y'] - yield_curve_df['3M']

    if '30Y' in yield_curve_df.columns and '5Y' in yield_curve_df.columns:
        spreads['30Y-5Y'] = yield_curve_df['30Y'] - yield_curve_df['5Y']

    return spreads


def detect_inversions(spreads: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Detect yield curve inversions and their durations."""
    inversions = {}

    for spread_name in spreads.columns:
        spread_data = spreads[spread_name].dropna()
        is_inverted = spread_data < 0
        inversion_changes = is_inverted.astype(int).diff()
        starts = spread_data.index[inversion_changes == 1]
        ends = spread_data.index[inversion_changes == -1]

        if len(starts) > 0 or len(ends) > 0:
            if len(starts) > 0 and (len(ends) == 0 or starts[0] < ends[0]):
                if is_inverted.iloc[0]:
                    starts = spread_data.index[[0]].append(starts)

            if len(ends) > 0 and (len(starts) == 0 or ends[-1] > starts[-1]):
                if is_inverted.iloc[-1]:
                    ends = ends.append(spread_data.index[[-1]])

            min_len = min(len(starts), len(ends))
            if min_len > 0:
                inversions[spread_name] = pd.DataFrame({
                    'Start': starts[:min_len],
                    'End': ends[:min_len],
                    'Duration_Days': [(ends[i] - starts[i]).days for i in range(min_len)],
                    'Max_Inversion': [spread_data.loc[starts[i]:ends[i]].min() for i in range(min_len)]
                })

    return inversions


def plot_yield_curve_current(yields: pd.Series) -> go.Figure:
    """Plot current yield curve."""
    fig = go.Figure()

    maturity_map = {'1M': 0.083, '3M': 0.25, '6M': 0.5, '1Y': 1, '2Y': 2,
                    '3Y': 3, '5Y': 5, '7Y': 7, '10Y': 10, '20Y': 20, '30Y': 30}

    x_values = [maturity_map[m] for m in yields.index if m in maturity_map]
    y_values = [yields[m] for m in yields.index if m in maturity_map]
    labels = [m for m in yields.index if m in maturity_map]

    is_inverted = any(yields.diff() < 0)
    color = '#d62728' if is_inverted else '#2ca02c'

    fig.add_trace(go.Scatter(
        x=x_values,
        y=y_values,
        mode='lines+markers',
        name='Yield Curve',
        line=dict(color=color, width=3),
        marker=dict(size=8),
        hovertemplate='%{text}: %{y:.2f}%<extra></extra>',
        text=labels
    ))

    fig.update_layout(
        title='Current Treasury Yield Curve' + (' — INVERTED ⚠️' if is_inverted else ' — Normal'),
        xaxis_title='Maturity (Years)',
        yaxis_title='Yield (%)',
        template='plotly_white',
        hovermode='x unified',
        height=420,
        xaxis=dict(type='log', tickvals=[0.083, 0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30],
                   ticktext=['1M', '3M', '6M', '1Y', '2Y', '3Y', '5Y', '7Y', '10Y', '20Y', '30Y'])
    )

    return fig


def plot_yield_spreads(spreads: pd.DataFrame, inversions: Dict) -> go.Figure:
    """Plot yield spreads over time with inversion highlights."""
    fig = go.Figure()

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for idx, spread_name in enumerate(spreads.columns):
        fig.add_trace(go.Scatter(
            x=spreads.index,
            y=spreads[spread_name],
            name=spread_name,
            line=dict(color=colors[idx % len(colors)], width=2),
            hovertemplate='%{y:.2f}%<extra></extra>'
        ))

        if spread_name in inversions and not inversions[spread_name].empty:
            for _, inv in inversions[spread_name].iterrows():
                fig.add_vrect(
                    x0=inv['Start'], x1=inv['End'],
                    fillcolor=colors[idx % len(colors)], opacity=0.1,
                    layer="below", line_width=0,
                )

    fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Inversion Threshold")

    fig.update_layout(
        title='Treasury Yield Spreads Over Time',
        xaxis_title='Date',
        yaxis_title='Spread (%)',
        template='plotly_white',
        hovermode='x unified',
        height=450,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    return fig


# ==================== ECONOMIC CYCLE DETECTION ====================

def detect_economic_cycle(data_dict: Dict[str, pd.DataFrame]) -> Tuple[pd.Series, Dict]:
    """
    Detect economic cycle phases using multiple indicators.

    Key fix: All series are resampled to monthly frequency before alignment,
    avoiding the 'only 11 data points' problem caused by quarterly GDP data
    dominating the intersection of date ranges.
    """

    indicators = pd.DataFrame()
    available_indicators = []

    # --- Resample everything to monthly first ---

    def add_monthly(name: str, series: pd.Series):
        """Add a series to indicators after resampling to monthly."""
        monthly = resample_to_monthly(series).dropna()
        if len(monthly) > 0:
            indicators[name] = monthly
            available_indicators.append(name)

    # GDP Growth — quarterly → monthly via ffill
    if 'GDP_Growth' in data_dict and not data_dict['GDP_Growth'].empty:
        add_monthly('GDP_Growth', data_dict['GDP_Growth'].iloc[:, 0])
    elif 'GDP' in data_dict and not data_dict['GDP'].empty:
        gdp = resample_to_monthly(data_dict['GDP'].iloc[:, 0])
        gdp_growth = calculate_growth_rate(gdp, 12)
        if len(gdp_growth.dropna()) > 0:
            add_monthly('GDP_Growth', gdp_growth.dropna())

    # Industrial Production (monthly, YoY growth)
    if 'Industrial_Production' in data_dict and not data_dict['Industrial_Production'].empty:
        ip = resample_to_monthly(data_dict['Industrial_Production'].iloc[:, 0])
        ip_growth = calculate_growth_rate(ip, 12).dropna()
        if len(ip_growth) > 0:
            add_monthly('IP_Growth', ip_growth)

    # Unemployment (invert so higher = better economy)
    if 'Unemployment' in data_dict and not data_dict['Unemployment'].empty:
        add_monthly('Unemployment_Inv', -data_dict['Unemployment'].iloc[:, 0])

    # Consumer Sentiment
    if 'Consumer_Sentiment' in data_dict and not data_dict['Consumer_Sentiment'].empty:
        add_monthly('Sentiment', data_dict['Consumer_Sentiment'].iloc[:, 0])

    # Leading Economic Index (YoY growth)
    if 'Leading_Index' in data_dict and not data_dict['Leading_Index'].empty:
        lei = resample_to_monthly(data_dict['Leading_Index'].iloc[:, 0])
        lei_growth = calculate_growth_rate(lei, 12).dropna()
        if len(lei_growth) > 0:
            add_monthly('LEI_Growth', lei_growth)
        else:
            # Fallback: use level if we don't have enough for YoY
            add_monthly('LEI_Level', lei)

    # Payrolls (YoY growth)
    if 'Payrolls' in data_dict and not data_dict['Payrolls'].empty:
        pay = resample_to_monthly(data_dict['Payrolls'].iloc[:, 0])
        pay_growth = calculate_growth_rate(pay, 12).dropna()
        if len(pay_growth) > 0:
            add_monthly('Payrolls_Growth', pay_growth)

    # Retail Sales (YoY growth)
    if 'Retail_Sales' in data_dict and not data_dict['Retail_Sales'].empty:
        retail = resample_to_monthly(data_dict['Retail_Sales'].iloc[:, 0])
        retail_growth = calculate_growth_rate(retail, 12).dropna()
        if len(retail_growth) > 0:
            add_monthly('Retail_Growth', retail_growth)

    # Capacity Utilization (level)
    if 'Capacity_Utilization' in data_dict and not data_dict['Capacity_Utilization'].empty:
        add_monthly('Cap_Util', data_dict['Capacity_Utilization'].iloc[:, 0])

    # --- Validate ---

    if indicators.empty:
        return pd.Series(), {
            'error': 'no_indicators',
            'message': 'No indicators loaded. Please select at least 2 indicators from the sidebar.',
            'available_indicators': available_indicators
        }

    if len(indicators.columns) < 2:
        return pd.Series(), {
            'error': 'insufficient_indicators',
            'message': f'Only {len(indicators.columns)} indicator(s) available after processing. Need at least 2.',
            'available_indicators': available_indicators
        }

    # Align on common dates
    indicators = indicators.dropna()

    MIN_POINTS = 24  # 2 years of monthly data
    if len(indicators) < MIN_POINTS:
        years_available = len(indicators) / 12
        return pd.Series(), {
            'error': 'insufficient_length',
            'message': (
                f"Only {len(indicators)} monthly data points available after aligning all indicators "
                f"({years_available:.1f} years). Extend your date range to at least {MIN_YEARS_FOR_CYCLE} years "
                f"for meaningful cycle detection."
            ),
            'data_points': len(indicators),
            'min_required': MIN_POINTS,
            'available_indicators': available_indicators
        }

    # --- Standardise and build composite ---

    scaler = StandardScaler()
    indicators_scaled = pd.DataFrame(
        scaler.fit_transform(indicators),
        index=indicators.index,
        columns=indicators.columns
    )

    composite = indicators_scaled.mean(axis=1)

    # Adaptive smoothing: 3 months min, 6 months max
    window_size = min(6, max(3, len(composite) // 20))
    composite_smooth = composite.rolling(window=window_size, center=True).mean().dropna()

    if len(composite_smooth) < MIN_POINTS:
        return pd.Series(), {
            'error': 'insufficient_after_smoothing',
            'message': f'Only {len(composite_smooth)} data points remain after smoothing. Extend date range.',
            'data_points': len(composite_smooth),
            'min_required': MIN_POINTS,
            'available_indicators': available_indicators
        }

    # --- Peak/trough detection ---

    min_distance = max(6, len(composite_smooth) // 15)
    min_prominence = max(0.2, composite_smooth.std() * 0.4)

    peaks, _ = signal.find_peaks(composite_smooth, distance=min_distance, prominence=min_prominence)
    troughs, _ = signal.find_peaks(-composite_smooth, distance=min_distance, prominence=min_prominence)

    # --- Phase labelling ---

    phases = pd.Series('Unknown', index=composite.index)
    peak_dates = composite_smooth.index[peaks]
    trough_dates = composite_smooth.index[troughs]
    all_turns = sorted(list(peak_dates) + list(trough_dates))

    for i in range(len(all_turns) - 1):
        start_dt = all_turns[i]
        end_dt = all_turns[i + 1]
        phases.loc[start_dt:end_dt] = 'Contraction' if start_dt in peak_dates else 'Expansion'

    if all_turns:
        phases.loc[:all_turns[0]] = 'Expansion' if all_turns[0] in peak_dates else 'Contraction'
        phases.loc[all_turns[-1]:] = 'Contraction' if all_turns[-1] in peak_dates else 'Expansion'

    # --- Refine into early/late sub-phases ---

    composite_median = composite.median()
    refined = phases.copy()
    for i in range(len(refined)):
        if phases.iloc[i] == 'Expansion':
            refined.iloc[i] = 'Early Expansion' if composite.iloc[i] < composite_median else 'Late Expansion'
        elif phases.iloc[i] == 'Contraction':
            refined.iloc[i] = 'Early Contraction' if composite.iloc[i] > -abs(composite_median) else 'Late Contraction'

    current_phase = refined.iloc[-1] if len(refined) > 0 else 'Unknown'
    phase_run = refined[::-1]
    phase_duration = (phase_run == phase_run.iloc[0]).cumprod().sum()

    analysis = {
        'current_phase': current_phase,
        'composite_index': composite,
        'composite_smooth': composite_smooth,
        'peaks': peak_dates,
        'troughs': trough_dates,
        'indicators': indicators,
        'indicators_scaled': indicators_scaled,
        'phase_duration_months': int(phase_duration),
        'available_indicators': available_indicators,
        'n_data_points': len(indicators)
    }

    return refined, analysis


def plot_economic_cycle(phases: pd.Series, analysis: Dict) -> go.Figure:
    """Plot economic cycle with composite index."""

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Composite Economic Index', 'Economic Cycle Phases'),
        vertical_spacing=0.15,
        row_heights=[0.6, 0.4]
    )

    composite = analysis['composite_smooth']

    fig.add_trace(
        go.Scatter(
            x=composite.index,
            y=composite.values,
            name='Composite Index',
            line=dict(color='#1f77b4', width=2),
            hovertemplate='%{y:.2f}<extra></extra>'
        ),
        row=1, col=1
    )

    if len(analysis['peaks']) > 0:
        peak_values = composite.reindex(analysis['peaks'], method='nearest')
        fig.add_trace(
            go.Scatter(
                x=analysis['peaks'],
                y=peak_values.values,
                mode='markers',
                name='Peaks',
                marker=dict(color='red', size=10, symbol='triangle-down'),
                hovertemplate='Peak: %{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )

    if len(analysis['troughs']) > 0:
        trough_values = composite.reindex(analysis['troughs'], method='nearest')
        fig.add_trace(
            go.Scatter(
                x=analysis['troughs'],
                y=trough_values.values,
                mode='markers',
                name='Troughs',
                marker=dict(color='green', size=10, symbol='triangle-up'),
                hovertemplate='Trough: %{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )

    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)

    phase_colors = {
        'Early Expansion': '#90EE90',
        'Late Expansion': '#228B22',
        'Early Contraction': '#FFB6C1',
        'Late Contraction': '#DC143C',
        'Expansion': '#32CD32',
        'Contraction': '#FF6347',
        'Unknown': '#D3D3D3'
    }

    # Draw filled phase segments
    current_phase = None
    segment_start = None
    composite_full = analysis['composite_index']

    for date, phase in phases.items():
        if phase != current_phase:
            if current_phase is not None and segment_start is not None:
                seg = composite_full.loc[segment_start:date]
                fig.add_trace(go.Scatter(
                    x=seg.index, y=seg.values,
                    fill='tozeroy',
                    fillcolor=phase_colors.get(current_phase, '#D3D3D3'),
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip'
                ), row=2, col=1)
            current_phase = phase
            segment_start = date

    if current_phase is not None and segment_start is not None:
        seg = composite_full.loc[segment_start:]
        fig.add_trace(go.Scatter(
            x=seg.index, y=seg.values,
            fill='tozeroy',
            fillcolor=phase_colors.get(current_phase, '#D3D3D3'),
            line=dict(width=0),
            name=current_phase,
            hovertemplate=f'{current_phase}<extra></extra>'
        ), row=2, col=1)

    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Composite (Std. Dev.)", row=1, col=1)
    fig.update_yaxes(title_text="Phase", showticklabels=False, row=2, col=1)

    fig.update_layout(
        height=700,
        template='plotly_white',
        hovermode='x unified',
        title_text='Economic Cycle Analysis'
    )

    return fig


# ==================== RECESSION INDICATORS ====================

def get_recession_indicators(data_dict: Dict[str, pd.DataFrame], spreads: pd.DataFrame) -> pd.DataFrame:
    """Calculate various recession indicators."""

    indicators = pd.DataFrame()

    if 'Unemployment' in data_dict and not data_dict['Unemployment'].empty:
        unemp = data_dict['Unemployment'].iloc[:, 0]
        rolling_min = unemp.rolling(window=3).min()
        sahm = unemp - rolling_min
        indicators['Sahm_Rule'] = (sahm >= 0.5).astype(int)

    if not spreads.empty and '10Y-2Y' in spreads.columns:
        spread_data = spreads['10Y-2Y'].dropna()
        if len(spread_data) > 0:
            indicators['Yield_Curve_Inverted'] = (spread_data < 0).astype(int)

    if not spreads.empty and '10Y-3M' in spreads.columns:
        spread_data = spreads['10Y-3M'].dropna()
        if len(spread_data) > 0:
            indicators['Yield_10Y3M_Inverted'] = (spread_data < 0).astype(int)

    if 'Leading_Index' in data_dict and not data_dict['Leading_Index'].empty:
        lei = data_dict['Leading_Index'].iloc[:, 0]
        if len(lei) > 6:
            lei_6m_change = lei.pct_change(periods=6) * 100
            indicators['LEI_Declining'] = (lei_6m_change < -2).astype(int)
        if len(lei) > 3:
            lei_3m_change = lei.pct_change(periods=3) * 100
            indicators['LEI_3M_Declining'] = (lei_3m_change < -1).astype(int)

    if 'Consumer_Sentiment' in data_dict and not data_dict['Consumer_Sentiment'].empty:
        sent = data_dict['Consumer_Sentiment'].iloc[:, 0]
        if len(sent) > 12:
            sent_change = sent.pct_change(periods=12) * 100
            indicators['Sentiment_Collapse'] = (sent_change < -15).astype(int)
        if len(sent) > 60:
            sent_mean = sent.rolling(window=60).mean()
            indicators['Sentiment_Low'] = (sent < sent_mean * 0.85).astype(int)

    if 'Industrial_Production' in data_dict and not data_dict['Industrial_Production'].empty:
        ip = data_dict['Industrial_Production'].iloc[:, 0]
        if len(ip) > 6:
            ip_6m_change = ip.pct_change(periods=6) * 100
            indicators['IP_Declining'] = (ip_6m_change < -2).astype(int)
        if len(ip) > 3:
            ip_3m_change = ip.pct_change(periods=3) * 100
            indicators['IP_3M_Declining'] = (ip_3m_change < -1).astype(int)

    if 'Unemployment' in data_dict and not data_dict['Unemployment'].empty:
        unemp = data_dict['Unemployment'].iloc[:, 0]
        if len(unemp) > 6:
            unemp_6m_change = unemp.diff(periods=6)
            indicators['Unemployment_Rising'] = (unemp_6m_change > 0.5).astype(int)

    if 'Initial_Claims' in data_dict and not data_dict['Initial_Claims'].empty:
        claims = data_dict['Initial_Claims'].iloc[:, 0]
        if len(claims) > 52:
            claims_ma = claims.rolling(window=4).mean()
            claims_threshold = claims.rolling(window=52).quantile(0.75)
            indicators['Claims_Elevated'] = (claims_ma > claims_threshold).astype(int)

    if 'Payrolls' in data_dict and not data_dict['Payrolls'].empty:
        payrolls = data_dict['Payrolls'].iloc[:, 0]
        if len(payrolls) > 6:
            payrolls_6m_change = payrolls.pct_change(periods=6) * 100
            indicators['Payrolls_Declining'] = (payrolls_6m_change < -1).astype(int)

    if 'GDP' in data_dict and not data_dict['GDP'].empty:
        gdp = data_dict['GDP'].iloc[:, 0]
        if len(gdp) > 2:
            gdp_change = gdp.pct_change(periods=2) * 100
            indicators['GDP_Declining'] = (gdp_change < 0).astype(int)

    if 'GDP_Growth' in data_dict and not data_dict['GDP_Growth'].empty:
        gdp_growth = data_dict['GDP_Growth'].iloc[:, 0]
        indicators['GDP_Growth_Negative'] = (gdp_growth < 0).astype(int)

    if 'Retail_Sales' in data_dict and not data_dict['Retail_Sales'].empty:
        retail = data_dict['Retail_Sales'].iloc[:, 0]
        if len(retail) > 6:
            retail_6m_change = retail.pct_change(periods=6) * 100
            indicators['Retail_Declining'] = (retail_6m_change < -2).astype(int)

    if 'Housing_Starts' in data_dict and not data_dict['Housing_Starts'].empty:
        housing = data_dict['Housing_Starts'].iloc[:, 0]
        if len(housing) > 6:
            housing_6m_change = housing.pct_change(periods=6) * 100
            indicators['Housing_Declining'] = (housing_6m_change < -10).astype(int)

    if not indicators.empty:
        weights = {}
        high_weight = ['Sahm_Rule', 'Yield_Curve_Inverted', 'GDP_Declining', 'GDP_Growth_Negative']
        medium_weight = ['LEI_Declining', 'IP_Declining', 'Unemployment_Rising', 'Payrolls_Declining']
        low_weight = ['Sentiment_Collapse', 'Retail_Declining', 'Housing_Declining', 'Claims_Elevated',
                      'LEI_3M_Declining', 'IP_3M_Declining', 'Sentiment_Low', 'Yield_10Y3M_Inverted']

        for col in indicators.columns:
            weights[col] = 3.0 if col in high_weight else 2.0 if col in medium_weight else 1.0

       # AFTER (fixed):
weighted_sum = None
total_weight = 0.0

for col in indicators.columns:
    w = weights.get(col, 1.0)
    weighted_col = indicators[col] * w
    if weighted_sum is None:
        weighted_sum = weighted_col
    else:
        weighted_sum = weighted_sum.add(weighted_col, fill_value=0)
    total_weight += w

    if weighted_sum is not None and total_weight > 0:
    indicators['Recession_Probability'] = (weighted_sum / total_weight * 100).clip(0, 100)

    return indicators


# ==================== AI INTERPRETATION ENGINE ====================

def generate_interpretation(
    data_dict: Dict[str, pd.DataFrame],
    current_phase: str,
    cycle_analysis: Dict,
    recession_indicators: pd.DataFrame,
    spreads: pd.DataFrame
) -> Dict:
    """Generate comprehensive human-readable interpretation of the current economic environment."""

    interpretation = {
        'phase': current_phase,
        'headline': '',
        'summary': '',
        'key_findings': [],
        'warnings': [],
        'positives': [],
        'next_phase': '',
        'watch_signals': [],
        'confidence': 'Medium',
        'data_sources_used': []
    }

    playbook = CYCLE_PLAYBOOK.get(current_phase, CYCLE_PLAYBOOK.get('Expansion', {}))
    interpretation['next_phase'] = playbook.get('next_phase', 'Unknown')
    interpretation['watch_signals'] = playbook.get('next_phase_signals', [])

    # --- Recession probability ---
    if not recession_indicators.empty and 'Recession_Probability' in recession_indicators.columns:
        recession_prob = recession_indicators['Recession_Probability'].iloc[-1]
        indicator_cols = [c for c in recession_indicators.columns if c != 'Recession_Probability']
        active_signals = sum(recession_indicators[col].iloc[-1] == 1 for col in indicator_cols)

        if recession_prob >= 60:
            interpretation['warnings'].append(
                f"🔴 **High Recession Risk ({recession_prob:.0f}%)**: {active_signals} of {len(indicator_cols)} indicators flashing warning signs."
            )
        elif recession_prob >= 40:
            interpretation['warnings'].append(
                f"🟠 **Elevated Recession Risk ({recession_prob:.0f}%)**: Multiple indicators signaling stress. Caution warranted."
            )
        elif recession_prob >= 20:
            interpretation['key_findings'].append(
                f"🟡 **Moderate Recession Risk ({recession_prob:.0f}%)**: Some warning signs but not widespread."
            )
        else:
            interpretation['positives'].append(
                f"🟢 **Low Recession Risk ({recession_prob:.0f}%)**: Most indicators remain healthy."
            )

        if 'Sahm_Rule' in recession_indicators.columns and recession_indicators['Sahm_Rule'].iloc[-1] == 1:
            interpretation['warnings'].append("⚡ **Sahm Rule Triggered**: Unemployment has risen 0.5%+ above its 3-month low — historically a recession signal.")
        if 'Yield_Curve_Inverted' in recession_indicators.columns and recession_indicators['Yield_Curve_Inverted'].iloc[-1] == 1:
            interpretation['warnings'].append("📉 **Yield Curve Inverted (10Y-2Y)**: Inverted curve has preceded every US recession since 1955 — typically 6–24 month lag.")
        if 'LEI_Declining' in recession_indicators.columns and recession_indicators['LEI_Declining'].iloc[-1] == 1:
            interpretation['warnings'].append("🔻 **Leading Economic Index Declining**: LEI fell >2% over 6 months, signaling forward weakness.")
        if 'GDP_Growth_Negative' in recession_indicators.columns and recession_indicators['GDP_Growth_Negative'].iloc[-1] == 1:
            interpretation['warnings'].append("📊 **Negative GDP Growth**: Real GDP growth has turned negative — technical recession territory.")

    # --- GDP / Growth ---
    if 'GDP_Growth' in data_dict and not data_dict['GDP_Growth'].empty:
        g = data_dict['GDP_Growth'].iloc[-1, 0]
        interpretation['data_sources_used'].append('GDP Growth')
        if g > 3:
            interpretation['positives'].append(f"💪 **Strong GDP Growth** at {g:.1f}% — economy running above potential.")
        elif g > 1.5:
            interpretation['positives'].append(f"✅ **Solid GDP Growth** at {g:.1f}% — healthy expansion.")
        elif g > 0:
            interpretation['key_findings'].append(f"📉 **Slowing GDP Growth** at {g:.1f}% — positive but losing momentum.")
        else:
            interpretation['warnings'].append(f"🔴 **Negative GDP Growth** at {g:.1f}% — output contracting.")

    # --- Inflation ---
    for inf_metric in ['Core_Inflation', 'Inflation_CPI', 'PCE_Inflation']:
        if inf_metric in data_dict and not data_dict[inf_metric].empty:
            inf_data = data_dict[inf_metric].iloc[:, 0]
            if len(inf_data) > 12:
                inf_yoy = calculate_growth_rate(inf_data, 12).iloc[-1]
                interpretation['data_sources_used'].append(inf_metric.replace('_', ' '))
                if inf_yoy > 5:
                    interpretation['warnings'].append(f"🔥 **High Inflation** ({inf_metric.replace('_', ' ')}): {inf_yoy:.1f}% YoY — well above Fed target.")
                elif inf_yoy > 3:
                    interpretation['key_findings'].append(f"⚠️ **Above-Target Inflation** ({inf_metric.replace('_', ' ')}): {inf_yoy:.1f}% YoY — Fed likely to stay restrictive.")
                elif inf_yoy > 1.5:
                    interpretation['positives'].append(f"✅ **Inflation Near Target** ({inf_metric.replace('_', ' ')}): {inf_yoy:.1f}% YoY — price stability improving.")
                else:
                    interpretation['key_findings'].append(f"❄️ **Low Inflation** ({inf_metric.replace('_', ' ')}): {inf_yoy:.1f}% YoY — possible deflationary pressure.")
            break

    # --- Unemployment ---
    if 'Unemployment' in data_dict and not data_dict['Unemployment'].empty:
        unemp = data_dict['Unemployment'].iloc[:, 0]
        current_unemp = unemp.iloc[-1]
        prev_unemp = unemp.iloc[-13] if len(unemp) > 13 else unemp.iloc[0]
        unemp_change = current_unemp - prev_unemp
        interpretation['data_sources_used'].append('Unemployment')
        if current_unemp < 4.5 and unemp_change <= 0:
            interpretation['positives'].append(f"💼 **Tight Labor Market**: Unemployment at {current_unemp:.1f}% and stable — strong consumer backdrop.")
        elif current_unemp < 4.5 and unemp_change > 0.3:
            interpretation['key_findings'].append(f"💼 **Labor Market Cooling**: Unemployment at {current_unemp:.1f}%, up {unemp_change:.1f}pp YoY — watch closely.")
        elif current_unemp >= 5 and unemp_change > 0.5:
            interpretation['warnings'].append(f"📊 **Rising Unemployment**: {current_unemp:.1f}% (+{unemp_change:.1f}pp YoY) — labor market deteriorating.")

    # --- Yield curve ---
    if not spreads.empty and '10Y-2Y' in spreads.columns:
        current_spread = spreads['10Y-2Y'].dropna().iloc[-1]
        interpretation['data_sources_used'].append('Yield Curve')
        if current_spread < -0.5:
            interpretation['warnings'].append(f"📉 **Deeply Inverted Yield Curve**: 10Y-2Y at {current_spread:.2f}% — strong recession signal.")
        elif current_spread < 0:
            interpretation['warnings'].append(f"⚠️ **Inverted Yield Curve**: 10Y-2Y at {current_spread:.2f}% — historical recession precursor.")
        elif current_spread < 0.5:
            interpretation['key_findings'].append(f"📊 **Flat Yield Curve**: 10Y-2Y at {current_spread:.2f}% — watch for inversion.")
        else:
            interpretation['positives'].append(f"✅ **Normal Yield Curve**: 10Y-2Y at {current_spread:.2f}% — healthy term premium.")

    # --- Consumer sentiment ---
    if 'Consumer_Sentiment' in data_dict and not data_dict['Consumer_Sentiment'].empty:
        sent = data_dict['Consumer_Sentiment'].iloc[:, 0]
        current_sent = sent.iloc[-1]
        hist_mean = sent.mean()
        interpretation['data_sources_used'].append('Consumer Sentiment')
        if current_sent > hist_mean * 1.1:
            interpretation['positives'].append(f"😊 **High Consumer Confidence**: Sentiment at {current_sent:.0f} — well above avg of {hist_mean:.0f}.")
        elif current_sent > hist_mean * 0.9:
            interpretation['key_findings'].append(f"😐 **Average Consumer Confidence**: Sentiment at {current_sent:.0f} vs. avg {hist_mean:.0f}.")
        else:
            interpretation['warnings'].append(f"😟 **Low Consumer Confidence**: Sentiment at {current_sent:.0f} — below avg of {hist_mean:.0f}.")

    # --- Fed Funds ---
    if 'Fed_Funds_Rate' in data_dict and not data_dict['Fed_Funds_Rate'].empty:
        ffr = data_dict['Fed_Funds_Rate'].iloc[:, 0]
        current_ffr = ffr.iloc[-1]
        prev_ffr = ffr.iloc[-13] if len(ffr) > 13 else ffr.iloc[0]
        ffr_change = current_ffr - prev_ffr
        interpretation['data_sources_used'].append('Fed Funds Rate')
        if ffr_change > 1:
            interpretation['key_findings'].append(f"🏦 **Fed Tightening Cycle**: Fed Funds at {current_ffr:.2f}%, up {ffr_change:.2f}pp YoY — restrictive policy.")
        elif ffr_change < -0.5:
            interpretation['positives'].append(f"🏦 **Fed Easing Cycle**: Fed Funds at {current_ffr:.2f}%, down {abs(ffr_change):.2f}pp YoY — monetary tailwind.")
        else:
            interpretation['key_findings'].append(f"🏦 **Fed on Hold**: Fed Funds at {current_ffr:.2f}% — stable monetary policy.")

    # --- Leading Index ---
    if 'Leading_Index' in data_dict and not data_dict['Leading_Index'].empty:
        lei = data_dict['Leading_Index'].iloc[:, 0]
        if len(lei) > 6:
            lei_6m = lei.pct_change(6).iloc[-1] * 100
            interpretation['data_sources_used'].append('Leading Economic Index')
            if lei_6m > 2:
                interpretation['positives'].append(f"📈 **LEI Improving**: Leading Index up {lei_6m:.1f}% over 6 months — continued expansion ahead.")
            elif lei_6m > 0:
                interpretation['key_findings'].append(f"📊 **LEI Flat to Slightly Positive**: Leading Index up {lei_6m:.1f}%.")
            elif lei_6m > -2:
                interpretation['warnings'].append(f"⚠️ **LEI Weakening**: Leading Index down {abs(lei_6m):.1f}% over 6 months.")
            else:
                interpretation['warnings'].append(f"🔻 **LEI Declining Sharply**: Down {abs(lei_6m):.1f}% — strong warning of slowdown ahead.")

    # --- Confidence level ---
    n_warnings = len(interpretation['warnings'])
    n_positives = len(interpretation['positives'])
    if n_warnings == 0 and n_positives >= 2:
        interpretation['confidence'] = 'High (bullish)'
    elif n_warnings >= 3:
        interpretation['confidence'] = 'High (bearish)'
    elif n_warnings >= 2:
        interpretation['confidence'] = 'Medium (cautious)'
    else:
        interpretation['confidence'] = 'Medium'

    # --- Headline + summary ---
    phase_emoji = CYCLE_PLAYBOOK.get(current_phase, {}).get('emoji', '📊')
    interpretation['headline'] = f"{phase_emoji} Currently in **{current_phase}** — {CYCLE_PLAYBOOK.get(current_phase, {}).get('description', '')}"

    if n_warnings == 0 and n_positives > 0:
        tone = "The data paints a broadly constructive picture."
    elif n_warnings > n_positives:
        tone = "Warning signals outnumber positives — a defensive posture is prudent."
    elif n_positives > n_warnings:
        tone = "Positive signals dominate, though some risks remain on the horizon."
    else:
        tone = "Mixed signals call for a balanced, selective approach."

    interpretation['summary'] = (
        f"Based on {len(interpretation['data_sources_used'])} economic indicators, "
        f"the US economy appears to be in the **{current_phase}** phase of the business cycle. "
        f"{tone} The next likely transition is toward **{interpretation['next_phase']}**."
    )

    return interpretation


def render_interpretation_section(
    data_dict: Dict,
    current_phase: str,
    cycle_analysis: Dict,
    recession_indicators: pd.DataFrame,
    spreads: pd.DataFrame
):
    """Render the full AI interpretation and investment playbook section in Streamlit."""

    st.header("🧠 Economic Interpretation & Investment Playbook")

    interpretation = generate_interpretation(
        data_dict, current_phase, cycle_analysis, recession_indicators, spreads
    )

    playbook = CYCLE_PLAYBOOK.get(current_phase, CYCLE_PLAYBOOK.get('Expansion', {}))

    phase_color = playbook.get('color', '#888888')
    phase_emoji = playbook.get('emoji', '📊')

    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {phase_color}22, {phase_color}44);
                border-left: 6px solid {phase_color};
                border-radius: 8px;
                padding: 20px 24px;
                margin-bottom: 16px;">
        <h2 style="margin:0; color: {phase_color};">{phase_emoji} {current_phase}</h2>
        <p style="margin: 8px 0 0 0; font-size: 1rem; color: #333;">
            {playbook.get('description', '')}
        </p>
        <p style="margin: 6px 0 0 0; font-size: 0.85rem; color: #666;">
            📅 Typical duration: {playbook.get('typical_duration', 'N/A')} &nbsp;|&nbsp;
            ➡️ Next likely phase: <strong>{interpretation['next_phase']}</strong> &nbsp;|&nbsp;
            🎯 Confidence: {interpretation['confidence']}
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(interpretation['summary'])

    findings_tab, next_tab, assets_tab, signals_tab = st.tabs([
        "📋 Key Findings", "🔮 What to Watch Next", "💰 Asset Allocation", "📡 Signal Dashboard"
    ])

    with findings_tab:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("⚠️ Warning Signs")
            if interpretation['warnings']:
                for w in interpretation['warnings']:
                    st.warning(w)
            else:
                st.success("No major warning signals detected.")

        with col2:
            st.subheader("✅ Positives")
            if interpretation['positives']:
                for p in interpretation['positives']:
                    st.success(p)
            else:
                st.info("No strong positive signals at this time.")

        if interpretation['key_findings']:
            st.subheader("🔍 Neutral / Mixed Observations")
            for f in interpretation['key_findings']:
                st.info(f)

        st.markdown(f"**📖 Macro Backdrop:** {playbook.get('macro_backdrop', '')}")

    with next_tab:
        st.subheader(f"🔮 Transition to {interpretation['next_phase']}: What to Watch")
        st.markdown(
            f"Currently in **{current_phase}**. Historical patterns suggest the next transition "
            f"will be toward **{interpretation['next_phase']}**. Watch for:"
        )

        signals = interpretation['watch_signals']
        if signals:
            for i, sig in enumerate(signals, 1):
                st.markdown(f"**{i}.** {sig}")
        else:
            st.info("Transition signals not available for current phase.")

        # Cycle wheel
        phase_order = ['Early Expansion', 'Late Expansion', 'Early Contraction', 'Late Contraction']
        phase_colors_list = ['#2ecc71', '#f39c12', '#e67e22', '#c0392b']

        fig_cycle = go.Figure()
        for i, (phase, color) in enumerate(zip(phase_order, phase_colors_list)):
            is_current = (phase == current_phase or
                          (current_phase == 'Expansion' and phase == 'Early Expansion') or
                          (current_phase == 'Contraction' and phase == 'Early Contraction'))
            fig_cycle.add_trace(go.Scatterpolar(
                r=[1], theta=[i * 90],
                mode='markers+text',
                marker=dict(size=40 if is_current else 25, color=color,
                            opacity=1.0 if is_current else 0.4,
                            line=dict(width=3 if is_current else 1, color='white')),
                text=[f"{'► ' if is_current else ''}{phase.replace(' ', '<br>')}"],
                textposition='middle center',
                name=phase,
                textfont=dict(size=9 if not is_current else 10, color='white')
            ))

        fig_cycle.update_layout(
            polar=dict(radialaxis=dict(visible=False, range=[0, 2]),
                       angularaxis=dict(visible=False)),
            showlegend=True, height=380,
            title="Business Cycle Position", template='plotly_white'
        )
        st.plotly_chart(fig_cycle, use_container_width=True)

    with assets_tab:
        st.subheader(f"💰 Investment Playbook for {current_phase}")
        st.markdown(f"*Based on historical performance during **{current_phase}**.*")

        asset_recs = playbook.get('asset_recommendations', {})

        for asset_class, details in asset_recs.items():
            rating = details.get('rating', 'N/A')
            rationale = details.get('rationale', '')
            sectors = details.get('sectors', [])

            stars = rating.count('⭐')
            if stars >= 4:
                box_color, border_color = '#d4edda', '#28a745'
            elif stars == 3:
                box_color, border_color = '#fff3cd', '#ffc107'
            else:
                box_color, border_color = '#f8d7da', '#dc3545'

            sectors_html = ' &nbsp;·&nbsp; '.join(sectors) if sectors else ''

            st.markdown(f"""
            <div style="background:{box_color}; border-left: 4px solid {border_color};
                        border-radius:6px; padding:14px 18px; margin-bottom:10px;">
                <strong style="font-size:1rem;">{asset_class}</strong>
                &nbsp;&nbsp;<span style="font-size:0.9rem;">{rating}</span><br>
                <span style="font-size:0.85rem; color:#555;">{rationale}</span>
                {'<br><span style="font-size:0.82rem; color:#333; margin-top:4px; display:block;">📌 ' + sectors_html + '</span>' if sectors_html else ''}
            </div>
            """, unsafe_allow_html=True)

        avoid_items = playbook.get('avoid', [])
        if avoid_items:
            st.markdown("---")
            st.markdown(f"**🚫 Avoid / Underweight:** {' &nbsp;·&nbsp; '.join(avoid_items)}", unsafe_allow_html=True)

        st.caption(
            "⚠️ *Framework based on historical business cycle patterns, not financial advice. "
            "Consult a qualified financial advisor before making investment decisions.*"
        )

    with signals_tab:
        st.subheader("📡 Current Signal Dashboard")

        signal_data = []
        if not recession_indicators.empty:
            indicator_cols = [c for c in recession_indicators.columns if c != 'Recession_Probability']
            for col in indicator_cols:
                is_on = recession_indicators[col].iloc[-1] == 1
                high_weight = ['Sahm_Rule', 'Yield_Curve_Inverted', 'GDP_Declining', 'GDP_Growth_Negative']
                medium_weight = ['LEI_Declining', 'IP_Declining', 'Unemployment_Rising', 'Payrolls_Declining']
                importance = "🔴 High" if col in high_weight else "🟠 Medium" if col in medium_weight else "🟡 Supporting"
                signal_data.append({
                    'Indicator': col.replace('_', ' '),
                    'Status': '🔴 SIGNALING' if is_on else '🟢 Clear',
                    'Importance': importance
                })

        if signal_data:
            sig_df = pd.DataFrame(signal_data)
            st.dataframe(sig_df, use_container_width=True, hide_index=True)

            total = len(signal_data)
            active = sum(1 for s in signal_data if '🔴' in s['Status'])

            fig_signals = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=active,
                delta={'reference': 0},
                title={'text': f"Active Warning Signals ({active} of {total})"},
                gauge={
                    'axis': {'range': [0, total]},
                    'bar': {'color': '#e74c3c' if active > total * 0.5 else '#f39c12' if active > total * 0.25 else '#2ecc71'},
                    'steps': [
                        {'range': [0, total * 0.25], 'color': '#d5f5e3'},
                        {'range': [total * 0.25, total * 0.5], 'color': '#fef9e7'},
                        {'range': [total * 0.5, total], 'color': '#fadbd8'}
                    ]
                }
            ))
            fig_signals.update_layout(height=280)
            st.plotly_chart(fig_signals, use_container_width=True)
        else:
            st.info("Load economic indicators to populate the signal dashboard.")

        st.markdown("---")
        st.caption(f"**Data used:** {', '.join(interpretation['data_sources_used']) if interpretation['data_sources_used'] else 'None loaded yet'}")


# ==================== DASHBOARD ====================

def create_economic_dashboard(data_dict: Dict[str, pd.DataFrame]) -> go.Figure:
    """Create a comprehensive dashboard view."""
    non_empty = {m: d for m, d in data_dict.items() if not d.empty}
    if not non_empty:
        return None

    n_cols = 3
    n_rows = (len(non_empty) + n_cols - 1) // n_cols
    metric_names = list(non_empty.keys())

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[METRIC_MAPPING.get(m, {}).get('description', m.replace('_', ' ')) for m in metric_names],
        vertical_spacing=0.1, horizontal_spacing=0.08
    )

    colors = px.colors.qualitative.Set3

    for idx, (metric, df) in enumerate(non_empty.items()):
        row, col = divmod(idx, n_cols)
        series = df.iloc[:, 0]
        fig.add_trace(
            go.Scatter(x=series.index, y=series.values, name=metric,
                       line=dict(color=colors[idx % len(colors)], width=2),
                       mode='lines', showlegend=False,
                       hovertemplate='%{y:.2f}<extra></extra>'),
            row=row + 1, col=col + 1
        )
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray', row=row + 1, col=col + 1)
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray', row=row + 1, col=col + 1)

    fig.update_layout(
        height=350 * n_rows,
        title_text="Economic Indicators Dashboard",
        template='plotly_white',
        hovermode='x unified'
    )
    return fig


# ==================== MAIN ====================

def economics_module(analysis_context: Optional[Dict] = None):
    """Main enhanced economics module."""

    st.set_page_config(
        page_title="Advanced Economic Analysis",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    ) if __name__ == "__main__" else None

    st.title("📊 Advanced Macro/Economic Context Analysis")

    st.markdown("""
    Comprehensive macroeconomic analysis featuring **real-time FRED data**, yield curve analysis,
    economic cycle detection, recession probability scoring, and an AI-driven investment playbook.
    """)

    with st.expander("🚀 Quick Start Guide", expanded=False):
        st.markdown(f"""
        ### Getting Started

        **1. Optional: Get a FREE FRED API Key**
        Visit [FRED API Key Registration](https://fred.stlouisfed.org/docs/api/api_key.html) and enter your email. Paste the key in the sidebar.

        **2. Select Analysis Modules** in the sidebar — all are on by default.

        **3. Choose Indicators** — defaults are optimised for all features.

        **4. Set Date Range** — for cycle detection, use **at least {MIN_YEARS_FOR_CYCLE} years**
        (10+ years recommended for best results).

        **5. Explore the Results!**
        The *Interpretation & Playbook* section at the bottom synthesises all loaded data automatically.
        """)

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")

        use_api = st.checkbox("Use FRED API (faster, requires free key)", value=False)
        api_key = None

        if use_api:
            api_key = st.text_input("FRED API Key", type="password",
                                    help="Get a free API key at https://fred.stlouisfed.org/docs/api/api_key.html")
            if api_key:
                st.success("✅ API key configured")
        else:
            st.info("💡 Using CSV download (no key needed, slightly slower)")

        st.subheader("📊 Analysis Modules")
        show_indicators = st.checkbox("Economic Indicators", value=True)
        show_health = st.checkbox("Economic Health Scorecard", value=True)
        show_yield_curve = st.checkbox("Yield Curve Analysis", value=True)
        show_cycle = st.checkbox("Economic Cycle Detection", value=True)
        show_recession = st.checkbox("Recession Indicators", value=True)
        show_interpretation = st.checkbox("🧠 Interpretation & Playbook", value=True)

        st.subheader("📈 Select Indicators")

        categories = {
            'output': 'Output & Production',
            'labor': 'Labor Market',
            'prices': 'Inflation & Prices',
            'rates': 'Interest Rates',
            'consumption': 'Consumption',
            'housing': 'Housing',
            'confidence': 'Confidence',
            'composite': 'Composite Indices'
        }

        selected_metrics = []
        default_selections = [
            'GDP', 'GDP_Growth', 'Unemployment', 'Inflation_CPI',
            'Fed_Funds_Rate', '10Y_Treasury', '2Y_Treasury',
            'Consumer_Sentiment', 'Industrial_Production', 'Leading_Index'
        ]

        for cat_key, cat_name in categories.items():
            with st.expander(cat_name):
                cat_metrics = [k for k, v in METRIC_MAPPING.items() if v.get('category') == cat_key]
                for metric in cat_metrics:
                    if st.checkbox(
                        METRIC_MAPPING[metric]['description'],
                        value=metric in default_selections,
                        key=f"metric_{metric}"
                    ):
                        selected_metrics.append(metric)

        st.subheader("📅 Date Range")

        # Warn if date range is too short for cycle detection
        default_start = datetime.now() - timedelta(days=365 * 15)

        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start", value=default_start, max_value=datetime.now())
        with col2:
            end_date = st.date_input("End", value=datetime.now(), max_value=datetime.now())

        years_selected = (end_date - start_date).days / 365
        if show_cycle and years_selected < MIN_YEARS_FOR_CYCLE:
            st.warning(
                f"⚠️ Cycle detection works best with ≥ {MIN_YEARS_FOR_CYCLE} years of data. "
                f"You've selected {years_selected:.1f} years."
            )

        st.subheader("📊 Display Options")
        show_growth = st.checkbox("Overlay YoY growth rates", value=False)
        normalize_data = st.checkbox("Normalize to 100 (indexed)", value=False)

    # Validation
    if start_date >= end_date:
        st.error("❌ Start date must be before end date.")
        return

    # Fetch data
    data_dict = {}
    failed_series = []

    if show_indicators and selected_metrics:
        with st.spinner("Fetching economic data from FRED..."):
            progress_bar = st.progress(0)
            status_text = st.empty()

            for idx, metric in enumerate(selected_metrics):
                metric_info = METRIC_MAPPING[metric]
                fred_code = metric_info['fred_code']
                status_text.text(f"Loading {metric_info['description']}...")

                if use_api and api_key:
                    df = get_fred_data_api(fred_code, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"), api_key)
                else:
                    df = get_fred_data_csv(fred_code, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))

                if not df.empty:
                    data_dict[metric] = df
                else:
                    failed_series.append(metric)

                progress_bar.progress((idx + 1) / len(selected_metrics))

            progress_bar.empty()
            status_text.empty()

    if failed_series:
        st.warning(f"⚠️ Could not fetch: {', '.join(failed_series)}. Check your connection or try the CSV method.")

    # ===================== ECONOMIC INDICATORS =====================

    if show_indicators and data_dict:
        st.header("📊 Economic Indicators")

        # Snapshot metrics
        n_display = min(len(data_dict), 4)
        snapshot_metrics = list(data_dict.items())
        cols = st.columns(n_display)

        for idx, (metric, df) in enumerate(snapshot_metrics[:n_display]):
            with cols[idx]:
                if not df.empty and len(df) > 1:
                    current_val = df.iloc[-1, 0]
                    prev_val = df.iloc[-13, 0] if len(df) > 13 else df.iloc[0, 0]
                    change = current_val - prev_val
                    metric_info = METRIC_MAPPING[metric]
                    st.metric(
                        label=metric.replace('_', ' '),
                        value=f"{current_val:.2f}",
                        delta=f"{change:+.2f}",
                        delta_color="inverse" if metric_info.get('inverse', False) else "normal"
                    )

        dashboard_fig = create_economic_dashboard(data_dict)
        if dashboard_fig:
            st.plotly_chart(dashboard_fig, use_container_width=True)

        with st.expander("📈 Detailed Charts & Statistics"):
            for metric, df in data_dict.items():
                if df.empty:
                    continue

                metric_info = METRIC_MAPPING[metric]
                st.markdown(f"### {metric_info['description']}")
                series = df.iloc[:, 0]

                if normalize_data and series.iloc[0] != 0:
                    series = (series / series.iloc[0]) * 100

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=series.index, y=series.values,
                    name=metric_info['description'],
                    line=dict(color='#1f77b4', width=2),
                    mode='lines', hovertemplate='%{y:.2f}<extra></extra>'
                ))

                if show_growth and len(series) > 12:
                    growth = calculate_growth_rate(series, 12)
                    fig.add_trace(go.Scatter(
                        x=growth.index, y=growth.values, name='YoY Growth %',
                        line=dict(color='#ff7f0e', width=2, dash='dash'),
                        yaxis='y2', hovertemplate='%{y:.2f}%<extra></extra>'
                    ))

                layout_cfg = dict(
                    title=f"{metric_info['description']} ({metric_info['unit']})",
                    xaxis_title="Date", yaxis_title=metric_info['unit'],
                    template='plotly_white', hovermode='x unified', height=400
                )
                if show_growth and len(series) > 12:
                    layout_cfg['yaxis2'] = dict(title='YoY Growth %', overlaying='y', side='right')
                fig.update_layout(**layout_cfg)
                st.plotly_chart(fig, use_container_width=True)

                stats = calculate_statistics(df)
                if not stats.empty:
                    st.dataframe(stats.style.format("{:.2f}"), use_container_width=True)

    # ===================== HEALTH SCORECARD =====================

    if show_health and (data_dict or not (spreads := pd.DataFrame()).empty):
        st.header("🏥 Economic Health Scorecard")
        # We'll compute after yield curve is available — moved below

    # ===================== YIELD CURVE =====================

    yield_curves = pd.DataFrame()
    spreads = pd.DataFrame()

    if show_yield_curve:
        st.header("📈 Yield Curve Analysis")

        with st.spinner("Fetching yield curve data..."):
            yield_curves = get_yield_curve_historical(
                start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"),
                use_api, api_key
            )

            if not yield_curves.empty:
                spreads = calculate_yield_spreads(yield_curves)
                inversions = detect_inversions(spreads)

                current_yields = yield_curves.iloc[-1].dropna()
                if not current_yields.empty:
                    st.subheader("Current Yield Curve")
                    st.plotly_chart(plot_yield_curve_current(current_yields), use_container_width=True)

                    if not spreads.empty:
                        st.subheader("Current Spreads")
                        spread_cols = st.columns(min(len(spreads.columns), 4))
                        for idx, sname in enumerate(spreads.columns):
                            with spread_cols[idx % 4]:
                                cur_sp = spreads[sname].iloc[-1]
                                prev_sp = spreads[sname].iloc[-30] if len(spreads) > 30 else cur_sp
                                st.metric(
                                    label=sname,
                                    value=f"{cur_sp:.2f}%",
                                    delta=f"{cur_sp - prev_sp:+.2f}%",
                                    delta_color="inverse" if cur_sp < 0 else "normal"
                                )
                                if cur_sp < 0:
                                    st.error("⚠️ INVERTED")

                st.subheader("Historical Yield Spreads")
                st.plotly_chart(plot_yield_spreads(spreads, inversions), use_container_width=True)

                if inversions:
                    st.subheader("Inversion History")
                    for sname, inv_df in inversions.items():
                        if not inv_df.empty:
                            with st.expander(f"{sname} — {len(inv_df)} inversion period(s)"):
                                display_inv = inv_df.copy()
                                display_inv['Start'] = pd.to_datetime(display_inv['Start']).dt.strftime('%Y-%m-%d')
                                display_inv['End'] = pd.to_datetime(display_inv['End']).dt.strftime('%Y-%m-%d')
                                st.dataframe(
                                    display_inv.style.format({'Duration_Days': '{:.0f}', 'Max_Inversion': '{:.2f}%'})
                                    .background_gradient(subset=['Max_Inversion'], cmap='Reds'),
                                    use_container_width=True
                                )

                with st.expander("🔍 3D Yield Curve Evolution"):
                    sample_dates = yield_curves.resample('ME').last().dropna(how='all').tail(36)
                    if not sample_dates.empty:
                        maturity_map = {'1M': 0.083, '3M': 0.25, '6M': 0.5, '1Y': 1, '2Y': 2,
                                        '3Y': 3, '5Y': 5, '7Y': 7, '10Y': 10, '20Y': 20, '30Y': 30}
                        fig_3d = go.Figure()
                        for date, yields_row in sample_dates.iterrows():
                            yields_clean = yields_row.dropna()
                            xv = [maturity_map[m] for m in yields_clean.index if m in maturity_map]
                            yv = [yields_clean[m] for m in yields_clean.index if m in maturity_map]
                            fig_3d.add_trace(go.Scatter3d(
                                x=xv, y=[date] * len(xv), z=yv,
                                mode='lines+markers', name=date.strftime('%Y-%m'),
                                line=dict(width=2), marker=dict(size=3), showlegend=False
                            ))
                        fig_3d.update_layout(
                            title='Yield Curve Evolution (3D)',
                            scene=dict(xaxis_title='Maturity (Years)', yaxis_title='Date', zaxis_title='Yield (%)',
                                       xaxis=dict(type='log')),
                            height=600, template='plotly_white'
                        )
                        st.plotly_chart(fig_3d, use_container_width=True)

            else:
                st.warning("⚠️ Could not fetch yield curve data. Check connection.")

    # ===================== HEALTH SCORECARD (after spreads available) =====================

    if show_health and (data_dict or not spreads.empty):
        st.header("🏥 Economic Health Scorecard")
        health_scores = compute_health_scorecard(data_dict, spreads)
        if health_scores:
            col_h1, col_h2 = st.columns([1, 1])
            with col_h1:
                fig_radar = plot_health_scorecard(health_scores)
                st.plotly_chart(fig_radar, use_container_width=True)
            with col_h2:
                st.subheader("Dimension Scores")
                for dim, score in health_scores.items():
                    if dim == 'Overall':
                        continue
                    color = '#2ecc71' if score >= 65 else '#f39c12' if score >= 40 else '#e74c3c'
                    bar_pct = int(score)
                    st.markdown(f"""
                    <div style="margin-bottom:8px;">
                      <div style="display:flex; justify-content:space-between; font-size:0.85rem; margin-bottom:2px;">
                        <span>{dim}</span><span style="color:{color};font-weight:600;">{score:.0f}</span>
                      </div>
                      <div style="background:#eee; border-radius:4px; height:8px;">
                        <div style="background:{color}; width:{bar_pct}%; height:8px; border-radius:4px;"></div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

                overall = health_scores.get('Overall', 50)
                label = 'Healthy' if overall >= 65 else 'Mixed' if overall >= 40 else 'Stressed'
                color = '#2ecc71' if overall >= 65 else '#f39c12' if overall >= 40 else '#e74c3c'
                st.markdown(f"""
                <div style="margin-top:16px; padding:14px; background:{color}22; border-left:4px solid {color}; border-radius:6px;">
                  <strong>Overall Economic Health: {overall:.0f}/100 — {label}</strong>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Load more indicators to compute the health scorecard.")

    # ===================== CYCLE DETECTION =====================

    phases = pd.Series()
    cycle_analysis = {}
    current_phase = 'Unknown'

    if show_cycle and data_dict:
        st.header("🔄 Economic Cycle Analysis")

        years = (end_date - start_date).days / 365
        if years < MIN_YEARS_FOR_CYCLE:
            st.warning(
                f"⚠️ You've selected {years:.1f} years of data. "
                f"Cycle detection works best with **{MIN_YEARS_FOR_CYCLE}+ years** — consider extending your start date."
            )

        with st.spinner("Detecting economic cycle..."):
            phases, cycle_analysis = detect_economic_cycle(data_dict)

            if phases.empty or 'error' in cycle_analysis:
                error_msg = cycle_analysis.get('message', 'Insufficient data for cycle detection.')
                available = cycle_analysis.get('available_indicators', [])

                st.error(f"🔄 Cycle Detection: {error_msg}")

                if available:
                    st.info(f"**Indicators loaded:** {', '.join(available)}")

                col_fix1, col_fix2 = st.columns(2)
                with col_fix1:
                    st.markdown("**💡 Quick Fixes:**")
                    st.markdown(f"- Extend date range to **{MIN_YEARS_FOR_CYCLE}+ years** (try 15 years)")
                    st.markdown("- Enable **Industrial Production** in the sidebar")
                    st.markdown("- Enable **Unemployment** in the sidebar")
                    st.markdown("- Enable **Leading Economic Index** in the sidebar")
                with col_fix2:
                    st.markdown("**📊 Recommended Indicators:**")
                    st.markdown("- GDP Growth *(quarterly → auto-resampled)*")
                    st.markdown("- Industrial Production *(monthly)*")
                    st.markdown("- Unemployment Rate *(monthly)*")
                    st.markdown("- Consumer Sentiment *(monthly)*")
                    st.markdown("- Leading Economic Index *(monthly)*")

            else:
                current_phase = cycle_analysis['current_phase']
                phase_duration = cycle_analysis['phase_duration_months']
                n_points = cycle_analysis.get('n_data_points', '?')
                avail = cycle_analysis.get('available_indicators', [])

                phase_emojis = {
                    'Early Expansion': '🟢', 'Late Expansion': '🟡',
                    'Early Contraction': '🟠', 'Late Contraction': '🔴',
                    'Expansion': '🟢', 'Contraction': '🔴'
                }
                phase_emoji = phase_emojis.get(current_phase, '⚪')

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Current Phase", f"{phase_emoji} {current_phase}")
                with c2:
                    st.metric("Duration in Phase", f"{phase_duration} months")
                with c3:
                    st.metric("Data Points Used", f"{n_points} months")

                st.caption(f"**Indicators used:** {', '.join(avail)}")

                cycle_fig = plot_economic_cycle(phases, cycle_analysis)
                st.plotly_chart(cycle_fig, use_container_width=True)

                with st.expander("📊 Contributing Indicators (Standardised)"):
                    indicators_scaled = cycle_analysis['indicators_scaled']
                    fig_ind = go.Figure()
                    for col in indicators_scaled.columns:
                        fig_ind.add_trace(go.Scatter(
                            x=indicators_scaled.index, y=indicators_scaled[col],
                            name=col, mode='lines', line=dict(width=2)
                        ))
                    fig_ind.add_hline(y=0, line_dash="dash", line_color="gray")
                    fig_ind.update_layout(
                        title='Standardized Economic Indicators', xaxis_title='Date',
                        yaxis_title='Standard Deviations from Mean',
                        template='plotly_white', hovermode='x unified', height=450
                    )
                    st.plotly_chart(fig_ind, use_container_width=True)

    # ===================== RECESSION INDICATORS =====================

    recession_indicators = pd.DataFrame()

    if show_recession and (data_dict or not spreads.empty):
        st.header("⚠️ Recession Indicators")

        recession_indicators = get_recession_indicators(data_dict, spreads)

        if not recession_indicators.empty:
            indicator_cols = [c for c in recession_indicators.columns if c != 'Recession_Probability']
            active_signals = sum(recession_indicators[col].iloc[-1] == 1 for col in indicator_cols)
            current_prob = recession_indicators['Recession_Probability'].iloc[-1]

            if current_prob < 20:
                risk_level, risk_color = "🟢 LOW", "green"
            elif current_prob < 40:
                risk_level, risk_color = "🟡 MODERATE", "orange"
            elif current_prob < 60:
                risk_level, risk_color = "🟠 ELEVATED", "darkorange"
            else:
                risk_level, risk_color = "🔴 HIGH", "red"

            st.subheader(f"Recession Risk: {risk_level}")
            st.caption(f"Active Signals: {active_signals}/{len(indicator_cols)}")

            c1, c2 = st.columns([1, 2])
            with c1:
                delta = None
                if len(recession_indicators) > 30:
                    past_prob = recession_indicators['Recession_Probability'].iloc[-30]
                    delta = f"{current_prob - past_prob:+.1f}%" if pd.notna(past_prob) and pd.notna(current_prob) else None
                st.metric("Recession Probability", f"{current_prob:.1f}%", delta=delta)

            with c2:
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number", value=current_prob,
                    title={'text': "Recession Probability"},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': risk_color},
                        'steps': [
                            {'range': [0, 20], 'color': "lightgreen"},
                            {'range': [20, 40], 'color': "lightyellow"},
                            {'range': [40, 60], 'color': "orange"},
                            {'range': [60, 100], 'color': "lightcoral"}
                        ],
                        'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 50}
                    }
                ))
                fig_gauge.update_layout(height=280)
                st.plotly_chart(fig_gauge, use_container_width=True)

            st.subheader("Individual Indicators")
            ind_cols = st.columns(min(len(indicator_cols), 4))
            for idx, col_name in enumerate(indicator_cols):
                with ind_cols[idx % 4]:
                    is_sig = recession_indicators[col_name].iloc[-1] == 1
                    st.metric(
                        label=col_name.replace('_', ' '),
                        value="SIGNAL" if is_sig else "Clear",
                        delta="⚠️" if is_sig else "✅"
                    )

            st.subheader("Historical Recession Probability")
            fig_prob = go.Figure()
            fig_prob.add_trace(go.Scatter(
                x=recession_indicators.index, y=recession_indicators['Recession_Probability'],
                fill='tozeroy', name='Recession Probability',
                line=dict(color='darkred', width=2), fillcolor='rgba(220, 20, 60, 0.3)'
            ))
            fig_prob.add_hline(y=50, line_dash="dash", line_color="red", annotation_text="High Risk Threshold")
            fig_prob.update_layout(
                title='Recession Probability Over Time',
                xaxis_title='Date', yaxis_title='Probability (%)',
                template='plotly_white', hovermode='x unified', height=400,
                yaxis=dict(range=[0, 100])
            )
            st.plotly_chart(fig_prob, use_container_width=True)

        else:
            st.warning("⚠️ No recession indicators could be calculated with current data.")

    # ===================== INTERPRETATION =====================

    if show_interpretation and (data_dict or not spreads.empty):
        interp_phase = current_phase if current_phase != 'Unknown' else 'Expansion'
        render_interpretation_section(
            data_dict=data_dict,
            current_phase=interp_phase,
            cycle_analysis=cycle_analysis,
            recession_indicators=recession_indicators,
            spreads=spreads
        )

    # ===================== CORRELATIONS =====================

    if data_dict and len(data_dict) > 1:
        with st.expander("📊 Correlation Matrix"):
            corr_matrix = calculate_correlations(data_dict)
            if not corr_matrix.empty:
                fig_corr = px.imshow(
                    corr_matrix, text_auto='.2f',
                    color_continuous_scale='RdBu_r', aspect='auto',
                    title='Indicator Correlations', labels=dict(color="Correlation"),
                    zmin=-1, zmax=1
                )
                fig_corr.update_layout(height=600)
                st.plotly_chart(fig_corr, use_container_width=True)

    # ===================== EXPORT =====================

    st.header("💾 Export Data")
    tab1, tab2, tab3, tab4 = st.tabs(["Indicators", "Yield Curve", "Cycle Analysis", "Recession Indicators"])

    with tab1:
        if data_dict:
            for metric, df in data_dict.items():
                if not df.empty:
                    st.download_button(
                        f"📥 {metric.replace('_', ' ')}",
                        df.to_csv(),
                        f"{metric}_{start_date}_{end_date}.csv",
                        "text/csv"
                    )

    with tab2:
        if not yield_curves.empty:
            st.download_button("📥 Yield Curve Data", yield_curves.to_csv(),
                               f"yield_curves_{start_date}_{end_date}.csv", "text/csv")
        if not spreads.empty:
            st.download_button("📥 Yield Spreads", spreads.to_csv(),
                               f"yield_spreads_{start_date}_{end_date}.csv", "text/csv")

    with tab3:
        if not phases.empty:
            cycle_export = pd.DataFrame({
                'Phase': phases,
                'Composite_Index': cycle_analysis['composite_index']
            })
            st.download_button("📥 Cycle Analysis", cycle_export.to_csv(),
                               f"economic_cycle_{start_date}_{end_date}.csv", "text/csv")

    with tab4:
        if not recession_indicators.empty:
            st.download_button("📥 Recession Indicators", recession_indicators.to_csv(),
                               f"recession_indicators_{start_date}_{end_date}.csv", "text/csv")


if __name__ == "__main__":
    economics_module()
