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
        "category": "output"
    },
    "GDP_Growth": {
        "fred_code": "A191RL1Q225SBEA",
        "unit": "Percent",
        "description": "Real GDP Growth Rate",
        "inverse": False,
        "category": "output"
    },
    "Inflation_CPI": {
        "fred_code": "CPIAUCSL",
        "unit": "Index (1982-84=100)",
        "description": "Consumer Price Index",
        "inverse": True,
        "category": "prices"
    },
    "Core_Inflation": {
        "fred_code": "CPILFESL",
        "unit": "Index (1982-84=100)",
        "description": "Core CPI (excl. Food & Energy)",
        "inverse": True,
        "category": "prices"
    },
    "PCE_Inflation": {
        "fred_code": "PCEPI",
        "unit": "Index (2017=100)",
        "description": "PCE Price Index",
        "inverse": True,
        "category": "prices"
    },
    "Unemployment": {
        "fred_code": "UNRATE",
        "unit": "Percentage",
        "description": "Unemployment Rate",
        "inverse": True,
        "category": "labor"
    },
    "Initial_Claims": {
        "fred_code": "ICSA",
        "unit": "Thousands",
        "description": "Initial Jobless Claims",
        "inverse": True,
        "category": "labor"
    },
    "Payrolls": {
        "fred_code": "PAYEMS",
        "unit": "Thousands",
        "description": "Total Nonfarm Payrolls",
        "inverse": False,
        "category": "labor"
    },
    "Fed_Funds_Rate": {
        "fred_code": "FEDFUNDS",
        "unit": "Percentage",
        "description": "Federal Funds Effective Rate",
        "inverse": False,
        "category": "rates"
    },
    "10Y_Treasury": {
        "fred_code": "DGS10",
        "unit": "Percentage",
        "description": "10-Year Treasury Rate",
        "inverse": False,
        "category": "rates"
    },
    "2Y_Treasury": {
        "fred_code": "DGS2",
        "unit": "Percentage",
        "description": "2-Year Treasury Rate",
        "inverse": False,
        "category": "rates"
    },
    "3M_Treasury": {
        "fred_code": "DGS3MO",
        "unit": "Percentage",
        "description": "3-Month Treasury Rate",
        "inverse": False,
        "category": "rates"
    },
    "5Y_Treasury": {
        "fred_code": "DGS5",
        "unit": "Percentage",
        "description": "5-Year Treasury Rate",
        "inverse": False,
        "category": "rates"
    },
    "30Y_Treasury": {
        "fred_code": "DGS30",
        "unit": "Percentage",
        "description": "30-Year Treasury Rate",
        "inverse": False,
        "category": "rates"
    },
    "Consumer_Sentiment": {
        "fred_code": "UMCSENT",
        "unit": "Index (1966:Q1=100)",
        "description": "University of Michigan Consumer Sentiment",
        "inverse": False,
        "category": "confidence"
    },
    "Industrial_Production": {
        "fred_code": "INDPRO",
        "unit": "Index (2017=100)",
        "description": "Industrial Production Index",
        "inverse": False,
        "category": "output"
    },
    "Capacity_Utilization": {
        "fred_code": "TCU",
        "unit": "Percentage",
        "description": "Capacity Utilization",
        "inverse": False,
        "category": "output"
    },
    "Housing_Starts": {
        "fred_code": "HOUST",
        "unit": "Thousands of Units",
        "description": "Housing Starts",
        "inverse": False,
        "category": "housing"
    },
    "Retail_Sales": {
        "fred_code": "RSXFS",
        "unit": "Millions of Dollars",
        "description": "Advance Retail Sales",
        "inverse": False,
        "category": "consumption"
    },
    "Personal_Spending": {
        "fred_code": "PCE",
        "unit": "Billions of Dollars",
        "description": "Personal Consumption Expenditures",
        "inverse": False,
        "category": "consumption"
    },
    "Leading_Index": {
        "fred_code": "USSLIND",
        "unit": "Index (2016=100)",
        "description": "Leading Economic Index",
        "inverse": False,
        "category": "composite"
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
    # Fallback for when only broad phases are detected
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

    except Exception as e:
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

        # Filter by date range
        df = df.loc[start_date:end_date]

        # Convert to numeric and drop NaN
        df[series_id] = pd.to_numeric(df[series_id], errors='coerce')
        return df.dropna()

    except Exception as e:
        return pd.DataFrame()


def calculate_growth_rate(data: pd.Series, periods: int = 4) -> pd.Series:
    """Calculate year-over-year growth rate."""
    return ((data / data.shift(periods)) - 1) * 100


def calculate_statistics(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate key statistics for the data."""
    if data.empty:
        return pd.DataFrame()

    stats = pd.DataFrame({
        'Current': data.iloc[-1] if len(data) > 0 else np.nan,
        'Mean': data.mean(),
        'Median': data.median(),
        'Std Dev': data.std(),
        'Min': data.min(),
        'Max': data.max(),
        '1Y Change (%)': ((data.iloc[-1] - data.iloc[-252]) / data.iloc[-252] * 100) if len(data) >= 252 else np.nan
    })
    return stats.T


def calculate_correlations(data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Calculate correlations between different metrics."""
    if len(data_dict) < 2:
        return pd.DataFrame()

    combined = pd.DataFrame()
    for metric, df in data_dict.items():
        if not df.empty and len(df.columns) > 0:
            combined[metric] = df.iloc[:, 0]

    if not combined.empty and len(combined.columns) > 1:
        combined = combined.dropna()
        if len(combined) > 10:
            return combined.corr()

    return pd.DataFrame()


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

    x_values = [maturity_map[m] for m in yields.index]
    is_inverted = any(yields.diff() < 0)
    color = '#d62728' if is_inverted else '#2ca02c'

    fig.add_trace(go.Scatter(
        x=x_values,
        y=yields.values,
        mode='lines+markers',
        name='Yield Curve',
        line=dict(color=color, width=3),
        marker=dict(size=8),
        hovertemplate='%{text}: %{y:.2f}%<extra></extra>',
        text=yields.index
    ))

    fig.update_layout(
        title='Current Treasury Yield Curve' + (' - INVERTED ⚠️' if is_inverted else ''),
        xaxis_title='Maturity (Years)',
        yaxis_title='Yield (%)',
        template='plotly_white',
        hovermode='x unified',
        height=500,
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
        height=500,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    return fig


# ==================== ECONOMIC CYCLE DETECTION ====================

def detect_economic_cycle(data_dict: Dict[str, pd.DataFrame]) -> Tuple[pd.Series, Dict]:
    """Detect economic cycle phases using multiple indicators."""

    indicators = pd.DataFrame()
    required_indicators = []

    if 'GDP_Growth' in data_dict and not data_dict['GDP_Growth'].empty:
        indicators['GDP_Growth'] = data_dict['GDP_Growth'].iloc[:, 0]
        required_indicators.append('GDP_Growth')
    elif 'GDP' in data_dict and not data_dict['GDP'].empty:
        gdp_growth = calculate_growth_rate(data_dict['GDP'].iloc[:, 0], periods=4)
        if len(gdp_growth.dropna()) > 0:
            indicators['GDP_Growth'] = gdp_growth
            required_indicators.append('GDP_Growth')

    if 'Industrial_Production' in data_dict and not data_dict['Industrial_Production'].empty:
        ip_growth = calculate_growth_rate(data_dict['Industrial_Production'].iloc[:, 0], periods=12)
        if len(ip_growth.dropna()) > 0:
            indicators['IP_Growth'] = ip_growth
            required_indicators.append('IP_Growth')

    if 'Unemployment' in data_dict and not data_dict['Unemployment'].empty:
        indicators['Unemployment_Inverted'] = -data_dict['Unemployment'].iloc[:, 0]
        required_indicators.append('Unemployment_Inverted')

    if 'Consumer_Sentiment' in data_dict and not data_dict['Consumer_Sentiment'].empty:
        indicators['Sentiment'] = data_dict['Consumer_Sentiment'].iloc[:, 0]
        required_indicators.append('Sentiment')

    if 'Leading_Index' in data_dict and not data_dict['Leading_Index'].empty:
        lei_growth = calculate_growth_rate(data_dict['Leading_Index'].iloc[:, 0], periods=12)
        if len(lei_growth.dropna()) > 0:
            indicators['LEI_Growth'] = lei_growth
            required_indicators.append('LEI_Growth')

    if 'Payrolls' in data_dict and not data_dict['Payrolls'].empty:
        payrolls_growth = calculate_growth_rate(data_dict['Payrolls'].iloc[:, 0], periods=12)
        if len(payrolls_growth.dropna()) > 0:
            indicators['Payrolls_Growth'] = payrolls_growth
            required_indicators.append('Payrolls_Growth')

    if 'Retail_Sales' in data_dict and not data_dict['Retail_Sales'].empty:
        retail_growth = calculate_growth_rate(data_dict['Retail_Sales'].iloc[:, 0], periods=12)
        if len(retail_growth.dropna()) > 0:
            indicators['Retail_Growth'] = retail_growth
            required_indicators.append('Retail_Growth')

    if 'Capacity_Utilization' in data_dict and not data_dict['Capacity_Utilization'].empty:
        indicators['Capacity_Util'] = data_dict['Capacity_Utilization'].iloc[:, 0]
        required_indicators.append('Capacity_Util')

    indicators = indicators.dropna()

    if indicators.empty:
        return pd.Series(), {'error': 'insufficient_data', 'available_indicators': required_indicators, 'min_required': 2}

    if len(indicators.columns) < 2:
        return pd.Series(), {'error': 'insufficient_indicators', 'available_indicators': required_indicators, 'min_required': 2}

    if len(indicators) < 30:
        return pd.Series(), {'error': 'insufficient_length', 'available_indicators': required_indicators, 'data_points': len(indicators), 'min_required': 30}

    scaler = StandardScaler()
    indicators_scaled = pd.DataFrame(
        scaler.fit_transform(indicators),
        index=indicators.index,
        columns=indicators.columns
    )

    composite = indicators_scaled.mean(axis=1)
    window_size = min(6, max(3, len(composite) // 10))
    composite_smooth = composite.rolling(window=window_size, center=True).mean()

    composite_clean = composite_smooth.dropna()

    if len(composite_clean) < 30:
        return pd.Series(), {
            'error': 'insufficient_length_after_alignment',
            'available_indicators': required_indicators,
            'data_points': len(composite_clean),
            'min_required': 30,
            'message': 'After aligning indicators and calculating growth rates, insufficient overlapping data remains.'
        }

    min_distance = max(6, len(composite_clean) // 20)
    min_prominence = max(0.3, composite_clean.std() * 0.5)

    peaks, _ = signal.find_peaks(composite_clean, distance=min_distance, prominence=min_prominence)
    troughs, _ = signal.find_peaks(-composite_clean, distance=min_distance, prominence=min_prominence)

    phases = pd.Series(index=composite.index, dtype=str)
    phases[:] = 'Unknown'

    peak_dates = composite_clean.index[peaks]
    trough_dates = composite_clean.index[troughs]

    all_turns = sorted(list(peak_dates) + list(trough_dates))

    for i in range(len(all_turns) - 1):
        start_date = all_turns[i]
        end_date = all_turns[i + 1]

        if start_date in peak_dates:
            phases.loc[start_date:end_date] = 'Contraction'
        elif start_date in trough_dates:
            phases.loc[start_date:end_date] = 'Expansion'

    if len(all_turns) > 0:
        if all_turns[0] in peak_dates:
            phases.loc[:all_turns[0]] = 'Expansion'
        else:
            phases.loc[:all_turns[0]] = 'Contraction'

        if all_turns[-1] in peak_dates:
            phases.loc[all_turns[-1]:] = 'Contraction'
        else:
            phases.loc[all_turns[-1]:] = 'Expansion'

    current_phase = phases.iloc[-1] if len(phases) > 0 else 'Unknown'
    current_composite = composite.iloc[-1] if len(composite) > 0 else 0

    refined_phases = phases.copy()

    for i in range(len(refined_phases)):
        if refined_phases.iloc[i] == 'Expansion':
            if composite.iloc[i] < composite.median():
                refined_phases.iloc[i] = 'Early Expansion'
            else:
                refined_phases.iloc[i] = 'Late Expansion'
        elif refined_phases.iloc[i] == 'Contraction':
            if composite.iloc[i] > -abs(composite.median()):
                refined_phases.iloc[i] = 'Early Contraction'
            else:
                refined_phases.iloc[i] = 'Late Contraction'

    analysis = {
        'current_phase': refined_phases.iloc[-1] if len(refined_phases) > 0 else 'Unknown',
        'composite_index': composite,
        'composite_smooth': composite_smooth,
        'peaks': peak_dates,
        'troughs': trough_dates,
        'indicators': indicators,
        'indicators_scaled': indicators_scaled,
        'phase_duration_months': len(refined_phases[refined_phases == refined_phases.iloc[-1]]) if len(refined_phases) > 0 else 0
    }

    return refined_phases, analysis


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
        peak_values = composite.loc[analysis['peaks']]
        fig.add_trace(
            go.Scatter(
                x=analysis['peaks'],
                y=peak_values,
                mode='markers',
                name='Peaks',
                marker=dict(color='red', size=10, symbol='triangle-down'),
                hovertemplate='Peak: %{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )

    if len(analysis['troughs']) > 0:
        trough_values = composite.loc[analysis['troughs']]
        fig.add_trace(
            go.Scatter(
                x=analysis['troughs'],
                y=trough_values,
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

    phase_numeric = phases.map({
        'Early Expansion': 2,
        'Late Expansion': 2,
        'Expansion': 2,
        'Early Contraction': -2,
        'Late Contraction': -2,
        'Contraction': -2,
        'Unknown': 0
    })

    current_phase = None
    segment_start = None

    for i, (date, phase) in enumerate(phases.items()):
        if phase != current_phase:
            if current_phase is not None and segment_start is not None:
                segment_data = phase_numeric.loc[segment_start:date]
                fig.add_trace(
                    go.Scatter(
                        x=segment_data.index,
                        y=segment_data.values,
                        fill='tozeroy',
                        fillcolor=phase_colors.get(current_phase, '#D3D3D3'),
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo='skip'
                    ),
                    row=2, col=1
                )
            current_phase = phase
            segment_start = date

    if current_phase is not None and segment_start is not None:
        segment_data = phase_numeric.loc[segment_start:]
        fig.add_trace(
            go.Scatter(
                x=segment_data.index,
                y=segment_data.values,
                fill='tozeroy',
                fillcolor=phase_colors.get(current_phase, '#D3D3D3'),
                line=dict(width=0),
                name=current_phase,
                hovertemplate=f'{current_phase}<extra></extra>'
            ),
            row=2, col=1
        )

    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Composite Index (Standardized)", row=1, col=1)
    fig.update_yaxes(title_text="Phase", showticklabels=False, row=2, col=1)

    fig.update_layout(
        height=700,
        template='plotly_white',
        hovermode='x unified',
        title_text='Economic Cycle Analysis'
    )

    return fig


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
        lei_6m_change = lei.pct_change(periods=6) * 100
        indicators['LEI_Declining'] = (lei_6m_change < -2).astype(int)
        lei_3m_change = lei.pct_change(periods=3) * 100
        indicators['LEI_3M_Declining'] = (lei_3m_change < -1).astype(int)

    if 'Consumer_Sentiment' in data_dict and not data_dict['Consumer_Sentiment'].empty:
        sent = data_dict['Consumer_Sentiment'].iloc[:, 0]
        sent_change = sent.pct_change(periods=12) * 100
        indicators['Sentiment_Collapse'] = (sent_change < -15).astype(int)
        sent_mean = sent.rolling(window=60).mean()
        indicators['Sentiment_Low'] = (sent < sent_mean * 0.85).astype(int)

    if 'Industrial_Production' in data_dict and not data_dict['Industrial_Production'].empty:
        ip = data_dict['Industrial_Production'].iloc[:, 0]
        ip_6m_change = ip.pct_change(periods=6) * 100
        indicators['IP_Declining'] = (ip_6m_change < -2).astype(int)
        ip_3m_change = ip.pct_change(periods=3) * 100
        indicators['IP_3M_Declining'] = (ip_3m_change < -1).astype(int)

    if 'Unemployment' in data_dict and not data_dict['Unemployment'].empty:
        unemp = data_dict['Unemployment'].iloc[:, 0]
        unemp_6m_change = unemp.diff(periods=6)
        indicators['Unemployment_Rising'] = (unemp_6m_change > 0.5).astype(int)

    if 'Initial_Claims' in data_dict and not data_dict['Initial_Claims'].empty:
        claims = data_dict['Initial_Claims'].iloc[:, 0]
        claims_ma = claims.rolling(window=4).mean()
        claims_threshold = claims.rolling(window=52).quantile(0.75)
        indicators['Claims_Elevated'] = (claims_ma > claims_threshold).astype(int)

    if 'Payrolls' in data_dict and not data_dict['Payrolls'].empty:
        payrolls = data_dict['Payrolls'].iloc[:, 0]
        payrolls_6m_change = payrolls.pct_change(periods=6) * 100
        indicators['Payrolls_Declining'] = (payrolls_6m_change < -1).astype(int)

    if 'GDP' in data_dict and not data_dict['GDP'].empty:
        gdp = data_dict['GDP'].iloc[:, 0]
        gdp_change = gdp.pct_change(periods=2) * 100
        indicators['GDP_Declining'] = (gdp_change < 0).astype(int)

    if 'GDP_Growth' in data_dict and not data_dict['GDP_Growth'].empty:
        gdp_growth = data_dict['GDP_Growth'].iloc[:, 0]
        indicators['GDP_Growth_Negative'] = (gdp_growth < 0).astype(int)

    if 'Retail_Sales' in data_dict and not data_dict['Retail_Sales'].empty:
        retail = data_dict['Retail_Sales'].iloc[:, 0]
        retail_6m_change = retail.pct_change(periods=6) * 100
        indicators['Retail_Declining'] = (retail_6m_change < -2).astype(int)

    if 'Housing_Starts' in data_dict and not data_dict['Housing_Starts'].empty:
        housing = data_dict['Housing_Starts'].iloc[:, 0]
        housing_6m_change = housing.pct_change(periods=6) * 100
        indicators['Housing_Declining'] = (housing_6m_change < -10).astype(int)

    if not indicators.empty:
        weights = {}
        high_weight = ['Sahm_Rule', 'Yield_Curve_Inverted', 'GDP_Declining', 'GDP_Growth_Negative']
        medium_weight = ['LEI_Declining', 'IP_Declining', 'Unemployment_Rising', 'Payrolls_Declining']
        low_weight = ['Sentiment_Collapse', 'Retail_Declining', 'Housing_Declining', 'Claims_Elevated',
                      'LEI_3M_Declining', 'IP_3M_Declining', 'Sentiment_Low', 'Yield_10Y3M_Inverted']

        for col in indicators.columns:
            if col in high_weight:
                weights[col] = 3.0
            elif col in medium_weight:
                weights[col] = 2.0
            elif col in low_weight:
                weights[col] = 1.0
            else:
                weights[col] = 1.0

        weighted_sum = pd.Series(0, index=indicators.index)
        total_weight = 0

        for col in indicators.columns:
            weighted_sum += indicators[col] * weights.get(col, 1.0)
            total_weight += weights.get(col, 1.0)

        if total_weight > 0:
            indicators['Recession_Probability'] = (weighted_sum / total_weight) * 100
        else:
            indicators['Recession_Probability'] = 0

    return indicators


# ==================== AI INTERPRETATION ENGINE ====================

def generate_interpretation(
    data_dict: Dict[str, pd.DataFrame],
    current_phase: str,
    cycle_analysis: Dict,
    recession_indicators: pd.DataFrame,
    spreads: pd.DataFrame
) -> Dict:
    """
    Generate comprehensive human-readable interpretation of the current economic environment.
    Returns structured interpretation with findings, outlook, and investment recommendations.
    """

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

    # --- Recession probability analysis ---
    recession_prob = 0
    if not recession_indicators.empty and 'Recession_Probability' in recession_indicators.columns:
        recession_prob = recession_indicators['Recession_Probability'].iloc[-1]
        indicator_cols = [c for c in recession_indicators.columns if c != 'Recession_Probability']
        active_signals = sum(recession_indicators[col].iloc[-1] == 1 for col in indicator_cols)

        if recession_prob >= 60:
            interpretation['warnings'].append(
                f"🔴 **High Recession Risk ({recession_prob:.0f}%)**: {active_signals} of {len(indicator_cols)} indicators are flashing warning signs."
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

        # Specific signal callouts
        if not recession_indicators.empty:
            if 'Sahm_Rule' in recession_indicators.columns and recession_indicators['Sahm_Rule'].iloc[-1] == 1:
                interpretation['warnings'].append("⚡ **Sahm Rule Triggered**: Unemployment has risen 0.5%+ above its 3-month low — historically a recession signal.")
            if 'Yield_Curve_Inverted' in recession_indicators.columns and recession_indicators['Yield_Curve_Inverted'].iloc[-1] == 1:
                interpretation['warnings'].append("📉 **Yield Curve Inverted (10Y-2Y)**: The curve is inverted, which has preceded every US recession since 1955 — typically with a 6–24 month lag.")
            if 'LEI_Declining' in recession_indicators.columns and recession_indicators['LEI_Declining'].iloc[-1] == 1:
                interpretation['warnings'].append("🔻 **Leading Economic Index Declining**: LEI has fallen >2% over 6 months, indicating forward economic weakness.")
            if 'GDP_Growth_Negative' in recession_indicators.columns and recession_indicators['GDP_Growth_Negative'].iloc[-1] == 1:
                interpretation['warnings'].append("📊 **Negative GDP Growth**: Real GDP growth has turned negative — technical recession territory.")

    # --- GDP / Growth analysis ---
    if 'GDP_Growth' in data_dict and not data_dict['GDP_Growth'].empty:
        gdp_g = data_dict['GDP_Growth'].iloc[:, 0]
        latest_gdp = gdp_g.iloc[-1]
        interpretation['data_sources_used'].append('GDP Growth')
        if latest_gdp > 3:
            interpretation['positives'].append(f"💪 **Strong GDP Growth** at {latest_gdp:.1f}% — economy running well above potential.")
        elif latest_gdp > 1.5:
            interpretation['positives'].append(f"✅ **Solid GDP Growth** at {latest_gdp:.1f}% — healthy expansion.")
        elif latest_gdp > 0:
            interpretation['key_findings'].append(f"📉 **Slowing GDP Growth** at {latest_gdp:.1f}% — growth is positive but losing momentum.")
        else:
            interpretation['warnings'].append(f"🔴 **Negative GDP Growth** at {latest_gdp:.1f}% — output contracting.")

    # --- Inflation analysis ---
    for inf_metric in ['Core_Inflation', 'Inflation_CPI', 'PCE_Inflation']:
        if inf_metric in data_dict and not data_dict[inf_metric].empty:
            inf_data = data_dict[inf_metric].iloc[:, 0]
            inf_yoy = calculate_growth_rate(inf_data, 12).iloc[-1]
            interpretation['data_sources_used'].append(inf_metric.replace('_', ' '))
            if inf_yoy > 5:
                interpretation['warnings'].append(f"🔥 **High Inflation** ({inf_metric.replace('_', ' ')}): {inf_yoy:.1f}% YoY — well above Fed target, pressuring margins and purchasing power.")
            elif inf_yoy > 3:
                interpretation['key_findings'].append(f"⚠️ **Above-Target Inflation** ({inf_metric.replace('_', ' ')}): {inf_yoy:.1f}% YoY — Fed likely to remain restrictive.")
            elif inf_yoy > 1.5:
                interpretation['positives'].append(f"✅ **Inflation Near Target** ({inf_metric.replace('_', ' ')}): {inf_yoy:.1f}% YoY — price stability improving.")
            else:
                interpretation['key_findings'].append(f"❄️ **Low Inflation** ({inf_metric.replace('_', ' ')}): {inf_yoy:.1f}% YoY — possible deflationary pressure or weak demand.")
            break  # Only report one inflation metric to avoid redundancy

    # --- Unemployment analysis ---
    if 'Unemployment' in data_dict and not data_dict['Unemployment'].empty:
        unemp = data_dict['Unemployment'].iloc[:, 0]
        current_unemp = unemp.iloc[-1]
        prev_unemp = unemp.iloc[-13] if len(unemp) > 13 else unemp.iloc[0]
        unemp_change = current_unemp - prev_unemp
        interpretation['data_sources_used'].append('Unemployment')
        if current_unemp < 4.5 and unemp_change <= 0:
            interpretation['positives'].append(f"💼 **Tight Labor Market**: Unemployment at {current_unemp:.1f}% and stable/improving — strong consumer backdrop.")
        elif current_unemp < 4.5 and unemp_change > 0.3:
            interpretation['key_findings'].append(f"💼 **Labor Market Cooling**: Unemployment at {current_unemp:.1f}%, up {unemp_change:.1f}pp YoY — watch closely.")
        elif current_unemp >= 5 and unemp_change > 0.5:
            interpretation['warnings'].append(f"📊 **Rising Unemployment**: {current_unemp:.1f}% (+{unemp_change:.1f}pp YoY) — labor market deteriorating.")

    # --- Yield curve analysis ---
    if not spreads.empty and '10Y-2Y' in spreads.columns:
        current_spread = spreads['10Y-2Y'].dropna().iloc[-1]
        interpretation['data_sources_used'].append('Yield Curve')
        if current_spread < -0.5:
            interpretation['warnings'].append(f"📉 **Deeply Inverted Yield Curve**: 10Y-2Y at {current_spread:.2f}% — deeply negative, strong recession signal.")
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
            interpretation['positives'].append(f"😊 **High Consumer Confidence**: Sentiment at {current_sent:.0f} — well above historical average of {hist_mean:.0f}.")
        elif current_sent > hist_mean * 0.9:
            interpretation['key_findings'].append(f"😐 **Average Consumer Confidence**: Sentiment at {current_sent:.0f} vs. historical mean of {hist_mean:.0f}.")
        else:
            interpretation['warnings'].append(f"😟 **Low Consumer Confidence**: Sentiment at {current_sent:.0f} — below historical average of {hist_mean:.0f}. Consumer spending risk.")

    # --- Fed Funds Rate context ---
    if 'Fed_Funds_Rate' in data_dict and not data_dict['Fed_Funds_Rate'].empty:
        ffr = data_dict['Fed_Funds_Rate'].iloc[:, 0]
        current_ffr = ffr.iloc[-1]
        prev_ffr = ffr.iloc[-13] if len(ffr) > 13 else ffr.iloc[0]
        ffr_change = current_ffr - prev_ffr
        interpretation['data_sources_used'].append('Fed Funds Rate')
        if ffr_change > 1:
            interpretation['key_findings'].append(f"🏦 **Fed Tightening Cycle**: Fed Funds at {current_ffr:.2f}%, up {ffr_change:.2f}pp YoY — restrictive monetary policy weighing on credit and valuations.")
        elif ffr_change < -0.5:
            interpretation['positives'].append(f"🏦 **Fed Easing Cycle**: Fed Funds at {current_ffr:.2f}%, down {abs(ffr_change):.2f}pp YoY — monetary tailwind for equities and credit.")
        else:
            interpretation['key_findings'].append(f"🏦 **Fed on Hold**: Fed Funds at {current_ffr:.2f}% — stable monetary policy.")

    # --- Leading Index ---
    if 'Leading_Index' in data_dict and not data_dict['Leading_Index'].empty:
        lei = data_dict['Leading_Index'].iloc[:, 0]
        lei_6m = lei.pct_change(6).iloc[-1] * 100
        interpretation['data_sources_used'].append('Leading Economic Index')
        if lei_6m > 2:
            interpretation['positives'].append(f"📈 **LEI Improving**: Leading Index up {lei_6m:.1f}% over 6 months — signals continued expansion ahead.")
        elif lei_6m > 0:
            interpretation['key_findings'].append(f"📊 **LEI Flat to Slightly Positive**: Leading Index up {lei_6m:.1f}% — modest forward momentum.")
        elif lei_6m > -2:
            interpretation['warnings'].append(f"⚠️ **LEI Weakening**: Leading Index down {abs(lei_6m):.1f}% over 6 months — forward economic momentum fading.")
        else:
            interpretation['warnings'].append(f"🔻 **LEI Declining Sharply**: Leading Index down {abs(lei_6m):.1f}% over 6 months — strong warning of economic slowdown ahead.")

    # --- Composite score for confidence ---
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

    # --- Generate headline and summary ---
    phase_emoji = CYCLE_PLAYBOOK.get(current_phase, {}).get('emoji', '📊')
    interpretation['headline'] = f"{phase_emoji} Currently in **{current_phase}** — {CYCLE_PLAYBOOK.get(current_phase, {}).get('description', '')}"

    n_warn = len(interpretation['warnings'])
    n_pos = len(interpretation['positives'])
    if n_warn == 0 and n_pos > 0:
        tone = "The data paints an broadly constructive picture."
    elif n_warn > n_pos:
        tone = "Warning signals outnumber positives — a defensive posture is prudent."
    elif n_pos > n_warn:
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

    # --- Phase banner ---
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
            🎯 Interpretation confidence: {interpretation['confidence']}
        </p>
    </div>
    """, unsafe_allow_html=True)

    # --- Summary ---
    st.markdown(interpretation['summary'])

    # --- Key findings tabs ---
    findings_tab, next_tab, assets_tab, signals_tab = st.tabs([
        "📋 Key Findings", "🔮 What to Watch Next", "💰 Asset Allocation", "📡 Leading Signals"
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

        # Macro backdrop
        st.markdown(f"**📖 Macro Backdrop:** {playbook.get('macro_backdrop', '')}")

    with next_tab:
        st.subheader(f"🔮 Transition to {interpretation['next_phase']}: What to Watch")

        st.markdown(
            f"The economy is currently in **{current_phase}**. "
            f"Based on historical patterns, the next transition will likely be toward **{interpretation['next_phase']}**. "
            f"Watch for these signals:"
        )

        signals = interpretation['watch_signals']
        if signals:
            for i, signal in enumerate(signals, 1):
                st.markdown(f"**{i}.** {signal}")
        else:
            st.info("Transition signals not available for current phase.")

        # Phase cycle visual
        phase_order = ['Early Expansion', 'Late Expansion', 'Early Contraction', 'Late Contraction']
        phase_colors_list = ['#2ecc71', '#f39c12', '#e67e22', '#c0392b']

        fig_cycle = go.Figure()

        for i, (phase, color) in enumerate(zip(phase_order, phase_colors_list)):
            is_current = (phase == current_phase or
                          (current_phase in ['Expansion'] and phase == 'Early Expansion') or
                          (current_phase in ['Contraction'] and phase == 'Early Contraction'))
            fig_cycle.add_trace(go.Scatterpolar(
                r=[1],
                theta=[i * 90],
                mode='markers+text',
                marker=dict(
                    size=40 if is_current else 25,
                    color=color,
                    opacity=1.0 if is_current else 0.4,
                    line=dict(width=3 if is_current else 1, color='white')
                ),
                text=[f"{'► ' if is_current else ''}{phase.replace(' ', '<br>')}"],
                textposition='middle center',
                name=phase,
                textfont=dict(size=9 if not is_current else 10, color='white')
            ))

        fig_cycle.update_layout(
            polar=dict(
                radialaxis=dict(visible=False, range=[0, 2]),
                angularaxis=dict(visible=False)
            ),
            showlegend=True,
            height=380,
            title="Business Cycle Position",
            template='plotly_white'
        )

        st.plotly_chart(fig_cycle, use_container_width=True)

    with assets_tab:
        st.subheader(f"💰 Investment Playbook for {current_phase}")
        st.markdown(f"*Based on historical performance during the **{current_phase}** phase of the business cycle.*")

        asset_recs = playbook.get('asset_recommendations', {})

        for asset_class, details in asset_recs.items():
            rating = details.get('rating', 'N/A')
            rationale = details.get('rationale', '')
            sectors = details.get('sectors', [])

            # Color-code by stars
            stars = rating.count('⭐')
            if stars >= 4:
                box_color = '#d4edda'
                border_color = '#28a745'
            elif stars == 3:
                box_color = '#fff3cd'
                border_color = '#ffc107'
            else:
                box_color = '#f8d7da'
                border_color = '#dc3545'

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

        # Avoid list
        avoid_items = playbook.get('avoid', [])
        if avoid_items:
            st.markdown("---")
            st.markdown(f"**🚫 Avoid / Underweight:** {' &nbsp;·&nbsp; '.join(avoid_items)}", unsafe_allow_html=True)

        st.caption(
            "⚠️ *This is a framework based on historical business cycle patterns, not financial advice. "
            "Actual conditions may differ. Consult a qualified financial advisor before making investment decisions.*"
        )

    with signals_tab:
        st.subheader("📡 Current Signal Dashboard")

        # Build a signal summary table
        signal_data = []

        if not recession_indicators.empty:
            indicator_cols = [c for c in recession_indicators.columns if c != 'Recession_Probability']
            for col in indicator_cols:
                is_on = recession_indicators[col].iloc[-1] == 1

                high_weight = ['Sahm_Rule', 'Yield_Curve_Inverted', 'GDP_Declining', 'GDP_Growth_Negative']
                medium_weight = ['LEI_Declining', 'IP_Declining', 'Unemployment_Rising', 'Payrolls_Declining']

                if col in high_weight:
                    importance = "🔴 High"
                elif col in medium_weight:
                    importance = "🟠 Medium"
                else:
                    importance = "🟡 Supporting"

                signal_data.append({
                    'Indicator': col.replace('_', ' '),
                    'Status': '🔴 SIGNALING' if is_on else '🟢 Clear',
                    'Importance': importance,
                    'Trend': 'Warning' if is_on else 'OK'
                })

        if signal_data:
            sig_df = pd.DataFrame(signal_data)
            st.dataframe(sig_df[['Indicator', 'Status', 'Importance']], use_container_width=True, hide_index=True)

            total = len(signal_data)
            active = sum(1 for s in signal_data if '🔴' in s['Status'])

            # Signal bar
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

        # Data sources used
        st.markdown("---")
        st.caption(f"**Data sources used in this analysis:** {', '.join(interpretation['data_sources_used']) if interpretation['data_sources_used'] else 'None loaded yet'}")


# ==================== MAIN DASHBOARD ====================

def create_economic_dashboard(data_dict: Dict[str, pd.DataFrame]) -> go.Figure:
    """Create a comprehensive dashboard view."""
    n_metrics = len([d for d in data_dict.values() if not d.empty])

    if n_metrics == 0:
        return None

    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols
    metric_names = [m for m in data_dict.keys() if not data_dict[m].empty]

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[METRIC_MAPPING.get(m, {}).get('description', m.replace('_', ' ')) for m in metric_names],
        vertical_spacing=0.1,
        horizontal_spacing=0.08
    )

    colors = px.colors.qualitative.Set3
    row, col = 1, 1

    for idx, (metric, df) in enumerate(data_dict.items()):
        if df.empty:
            continue

        series = df.iloc[:, 0]

        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series.values,
                name=metric,
                line=dict(color=colors[idx % len(colors)], width=2),
                mode='lines',
                showlegend=False,
                hovertemplate='%{y:.2f}<extra></extra>'
            ),
            row=row,
            col=col
        )

        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray', row=row, col=col)
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray', row=row, col=col)

        col += 1
        if col > n_cols:
            col = 1
            row += 1

    fig.update_layout(
        height=350 * n_rows,
        title_text="Economic Indicators Dashboard",
        template='plotly_white',
        hovermode='x unified'
    )

    return fig


def economics_module(analysis_context: Optional[Dict] = None):
    """Main enhanced economics module."""
    st.title("📊 Advanced Macro/Economic Context Analysis")

    st.markdown("""
    Comprehensive macroeconomic analysis featuring:
    - **Real-time Economic Indicators** from Federal Reserve Economic Data (FRED)
    - **Yield Curve Analysis** with inversion detection
    - **Economic Cycle Detection** using composite indicators
    - **Recession Probability** based on multiple signals
    - **AI Interpretation & Investment Playbook** — automatic cycle analysis with asset recommendations
    """)

    # Quick start guide
    with st.expander("🚀 Quick Start Guide", expanded=False):
        st.markdown("""
        ### Getting Started

        **1. Optional: Get a FREE FRED API Key** (Recommended)
        - Visit [FRED API Key Registration](https://fred.stlouisfed.org/docs/api/api_key.html)
        - Enter your email to receive an API key instantly
        - Paste it in the sidebar for faster data fetching

        **2. Select Your Analysis Modules** (Sidebar)

        **3. Choose Your Indicators** (Sidebar)
        Default selections are optimized for all features.

        **4. Set Date Range** — at least 5–10 years for cycle detection.

        **5. Explore the Results!**
        The **Interpretation & Playbook** section at the bottom automatically reads all loaded data
        and tells you where we are in the cycle, what's next, and how to position portfolios.
        """)

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        use_api = st.checkbox("Use FRED API (faster, requires free API key)", value=False)
        api_key = None

        if use_api:
            api_key = st.text_input(
                "FRED API Key",
                type="password",
                help="Get a free API key at https://fred.stlouisfed.org/docs/api/api_key.html"
            )
            if api_key:
                st.success("✅ API key configured")
        else:
            st.info("💡 Using CSV download (no API key needed, but slower)")

        st.subheader("📊 Analysis Modules")
        show_indicators = st.checkbox("Economic Indicators", value=True)
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
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start",
                value=datetime.now() - timedelta(days=365 * 10),
                max_value=datetime.now()
            )
        with col2:
            end_date = st.date_input(
                "End",
                value=datetime.now(),
                max_value=datetime.now()
            )

        st.subheader("📊 Display Options")
        show_growth = st.checkbox("Show growth rates", value=False)
        normalize_data = st.checkbox("Normalize to 100", value=False)

    # Validation
    if not any([show_indicators, show_yield_curve, show_cycle, show_recession]):
        st.warning("⚠️ Please select at least one analysis module.")
        return

    if start_date >= end_date:
        st.error("❌ Start date must be before end date.")
        return

    # Fetch data
    data_dict = {}
    failed_series = []

    if show_indicators and selected_metrics:
        with st.spinner("Fetching economic data from FRED..."):
            progress_bar = st.progress(0)

            for idx, metric in enumerate(selected_metrics):
                metric_info = METRIC_MAPPING[metric]
                fred_code = metric_info['fred_code']

                if use_api and api_key:
                    df = get_fred_data_api(
                        fred_code,
                        start_date.strftime("%Y-%m-%d"),
                        end_date.strftime("%Y-%m-%d"),
                        api_key
                    )
                else:
                    df = get_fred_data_csv(
                        fred_code,
                        start_date.strftime("%Y-%m-%d"),
                        end_date.strftime("%Y-%m-%d")
                    )

                if not df.empty:
                    data_dict[metric] = df
                else:
                    failed_series.append(metric)

                progress_bar.progress((idx + 1) / len(selected_metrics))

            progress_bar.empty()

    if failed_series:
        st.warning(f"⚠️ Could not fetch data for: {', '.join(failed_series)}")

    # ==================== ECONOMIC INDICATORS ====================

    if show_indicators and data_dict:
        st.header("📊 Economic Indicators")

        st.subheader("Current Economic Snapshot")
        cols = st.columns(min(len(data_dict), 4))

        for idx, (metric, df) in enumerate(data_dict.items()):
            with cols[idx % 4]:
                if not df.empty and len(df) > 0:
                    current_val = df.iloc[-1, 0]
                    prev_val = df.iloc[-2, 0] if len(df) > 1 else current_val
                    change = current_val - prev_val
                    change_pct = (change / prev_val * 100) if prev_val != 0 else 0

                    metric_info = METRIC_MAPPING[metric]

                    st.metric(
                        label=metric.replace('_', ' '),
                        value=f"{current_val:.2f}",
                        delta=f"{change_pct:+.2f}%",
                        delta_color="inverse" if metric_info.get('inverse', False) else "normal"
                    )

        st.subheader("Indicators Dashboard")
        dashboard_fig = create_economic_dashboard(data_dict)
        if dashboard_fig:
            st.plotly_chart(dashboard_fig, use_container_width=True)

        with st.expander("📈 Detailed Charts & Data"):
            for metric, df in data_dict.items():
                if df.empty:
                    continue

                metric_info = METRIC_MAPPING[metric]
                st.markdown(f"### {metric_info['description']}")
                series = df.iloc[:, 0]

                if normalize_data:
                    series = (series / series.iloc[0]) * 100

                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=series.index,
                    y=series.values,
                    name=metric_info['description'],
                    line=dict(color='#1f77b4', width=2),
                    mode='lines',
                    hovertemplate='%{y:.2f}<extra></extra>'
                ))

                if show_growth and len(series) > 4:
                    growth = calculate_growth_rate(series, periods=12)
                    fig.add_trace(go.Scatter(
                        x=growth.index,
                        y=growth.values,
                        name='YoY Growth %',
                        line=dict(color='#ff7f0e', width=2, dash='dash'),
                        yaxis='y2',
                        hovertemplate='%{y:.2f}%<extra></extra>'
                    ))

                layout_config = {
                    'title': f"{metric_info['description']} ({metric_info['unit']})",
                    'xaxis_title': "Date",
                    'yaxis_title': metric_info['unit'],
                    'template': 'plotly_white',
                    'hovermode': 'x unified',
                    'height': 400
                }

                if show_growth and len(series) > 4:
                    layout_config['yaxis2'] = dict(
                        title='YoY Growth %',
                        overlaying='y',
                        side='right'
                    )

                fig.update_layout(**layout_config)
                st.plotly_chart(fig, use_container_width=True)

                stats = calculate_statistics(df)
                st.dataframe(
                    stats.style.format("{:.2f}").background_gradient(cmap='Blues', axis=1),
                    use_container_width=True
                )

    # ==================== YIELD CURVE ANALYSIS ====================

    yield_curves = pd.DataFrame()
    spreads = pd.DataFrame()

    if show_yield_curve:
        st.header("📈 Yield Curve Analysis")

        with st.spinner("Fetching yield curve data..."):
            yield_curves = get_yield_curve_historical(
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d"),
                use_api,
                api_key
            )

            if not yield_curves.empty:
                spreads = calculate_yield_spreads(yield_curves)
                inversions = detect_inversions(spreads)

                st.subheader("Current Yield Curve")
                current_yields = yield_curves.iloc[-1].dropna()

                if not current_yields.empty:
                    current_fig = plot_yield_curve_current(current_yields)
                    st.plotly_chart(current_fig, use_container_width=True)

                    st.subheader("Current Spreads")
                    spread_cols = st.columns(min(len(spreads.columns), 4))

                    for idx, spread_name in enumerate(spreads.columns):
                        with spread_cols[idx % 4]:
                            current_spread = spreads[spread_name].iloc[-1]
                            prev_spread = spreads[spread_name].iloc[-30] if len(spreads) > 30 else current_spread
                            is_inverted = current_spread < 0

                            st.metric(
                                label=spread_name,
                                value=f"{current_spread:.2f}%",
                                delta=f"{current_spread - prev_spread:+.2f}%",
                                delta_color="inverse" if is_inverted else "normal"
                            )

                            if is_inverted:
                                st.error("⚠️ INVERTED")

                st.subheader("Historical Yield Spreads")
                spreads_fig = plot_yield_spreads(spreads, inversions)
                st.plotly_chart(spreads_fig, use_container_width=True)

                if inversions:
                    st.subheader("Inversion History")
                    for spread_name, inv_df in inversions.items():
                        if not inv_df.empty:
                            with st.expander(f"{spread_name} Inversions ({len(inv_df)} periods)"):
                                display_inv = inv_df.copy()
                                display_inv['Start'] = pd.to_datetime(display_inv['Start']).dt.strftime('%Y-%m-%d')
                                display_inv['End'] = pd.to_datetime(display_inv['End']).dt.strftime('%Y-%m-%d')
                                st.dataframe(
                                    display_inv.style.format({
                                        'Duration_Days': '{:.0f}',
                                        'Max_Inversion': '{:.2f}%'
                                    }).background_gradient(subset=['Max_Inversion'], cmap='Reds'),
                                    use_container_width=True
                                )

                # 3D yield curve — FIXED: 'M' → 'ME'
                with st.expander("🔍 3D Yield Curve Evolution"):
                    sample_dates = yield_curves.resample('ME').last().dropna(how='all').tail(36)

                    if not sample_dates.empty:
                        maturity_map = {'1M': 0.083, '3M': 0.25, '6M': 0.5, '1Y': 1, '2Y': 2,
                                        '3Y': 3, '5Y': 5, '7Y': 7, '10Y': 10, '20Y': 20, '30Y': 30}

                        fig_3d = go.Figure()

                        for idx, (date, yields_row) in enumerate(sample_dates.iterrows()):
                            yields_clean = yields_row.dropna()
                            x_values = [maturity_map[m] for m in yields_clean.index if m in maturity_map]
                            y_values = [yields_clean[m] for m in yields_clean.index if m in maturity_map]

                            fig_3d.add_trace(go.Scatter3d(
                                x=x_values,
                                y=[date] * len(x_values),
                                z=y_values,
                                mode='lines+markers',
                                name=date.strftime('%Y-%m'),
                                line=dict(width=2),
                                marker=dict(size=3),
                                showlegend=False
                            ))

                        fig_3d.update_layout(
                            title='Yield Curve Evolution (3D)',
                            scene=dict(
                                xaxis_title='Maturity (Years)',
                                yaxis_title='Date',
                                zaxis_title='Yield (%)',
                                xaxis=dict(type='log')
                            ),
                            height=600,
                            template='plotly_white'
                        )

                        st.plotly_chart(fig_3d, use_container_width=True)

            else:
                st.warning("⚠️ Could not fetch yield curve data.")

    # ==================== ECONOMIC CYCLE DETECTION ====================

    phases = pd.Series()
    cycle_analysis = {}
    current_phase = 'Unknown'

    if show_cycle and data_dict:
        st.header("🔄 Economic Cycle Analysis")

        with st.spinner("Detecting economic cycle..."):
            phases, cycle_analysis = detect_economic_cycle(data_dict)

            if phases.empty or 'error' in cycle_analysis:
                error_type = cycle_analysis.get('error', 'unknown')
                available = cycle_analysis.get('available_indicators', [])
                min_required = cycle_analysis.get('min_required', 2)

                if error_type == 'insufficient_indicators':
                    st.warning(f"⚠️ Need at least {min_required} different indicators for cycle detection.")
                elif error_type == 'insufficient_length':
                    data_points = cycle_analysis.get('data_points', 0)
                    st.warning(f"⚠️ Insufficient historical data ({data_points} points, need {min_required}). Extend date range.")
                elif error_type == 'insufficient_length_after_alignment':
                    data_points = cycle_analysis.get('data_points', 0)
                    st.warning(f"⚠️ Insufficient overlapping data after alignment ({data_points} points). Extend date range to 5–10 years.")
                else:
                    st.warning("⚠️ Insufficient data for cycle detection.")

            elif not phases.empty and cycle_analysis:
                current_phase = cycle_analysis['current_phase']
                phase_duration = cycle_analysis['phase_duration_months']

                phase_colors_map = {
                    'Early Expansion': '🟢', 'Late Expansion': '🟡',
                    'Early Contraction': '🟠', 'Late Contraction': '🔴',
                    'Expansion': '🟢', 'Contraction': '🔴'
                }

                phase_emoji = phase_colors_map.get(current_phase, '⚪')

                st.subheader(f"Current Phase: {phase_emoji} {current_phase}")
                st.info(f"Duration: {phase_duration} months")

                cycle_fig = plot_economic_cycle(phases, cycle_analysis)
                st.plotly_chart(cycle_fig, use_container_width=True)

                with st.expander("📊 Contributing Indicators"):
                    indicators_scaled = cycle_analysis['indicators_scaled']

                    fig = go.Figure()
                    for col in indicators_scaled.columns:
                        fig.add_trace(go.Scatter(
                            x=indicators_scaled.index,
                            y=indicators_scaled[col],
                            name=col,
                            mode='lines',
                            line=dict(width=2)
                        ))

                    fig.update_layout(
                        title='Standardized Economic Indicators',
                        xaxis_title='Date',
                        yaxis_title='Standard Deviations from Mean',
                        template='plotly_white',
                        hovermode='x unified',
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)

    # ==================== RECESSION INDICATORS ====================

    recession_indicators = pd.DataFrame()

    if show_recession and (data_dict or not yield_curves.empty):
        st.header("⚠️ Recession Indicators")

        spreads_for_recession = spreads if not spreads.empty else pd.DataFrame()
        recession_indicators = get_recession_indicators(data_dict, spreads_for_recession)

        if not recession_indicators.empty:
            indicator_cols = [c for c in recession_indicators.columns if c != 'Recession_Probability']
            active_signals = sum(recession_indicators[col].iloc[-1] == 1 for col in indicator_cols)
            total_indicators = len(indicator_cols)
            current_prob = recession_indicators['Recession_Probability'].iloc[-1]

            if current_prob < 20:
                risk_level = "🟢 LOW"
                risk_color = "green"
            elif current_prob < 40:
                risk_level = "🟡 MODERATE"
                risk_color = "orange"
            elif current_prob < 60:
                risk_level = "🟠 ELEVATED"
                risk_color = "darkorange"
            else:
                risk_level = "🔴 HIGH"
                risk_color = "red"

            st.subheader(f"Recession Risk: {risk_level}")
            st.caption(f"Active Signals: {active_signals}/{total_indicators} indicators signaling recession risk")

            col1, col2 = st.columns([1, 2])

            with col1:
                st.metric(
                    label="Recession Probability",
                    value=f"{current_prob:.1f}%",
                    delta=f"{current_prob - recession_indicators['Recession_Probability'].iloc[-30]:.1f}%" if len(recession_indicators) > 30 else None
                )

            with col2:
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=current_prob,
                    domain={'x': [0, 1], 'y': [0, 1]},
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
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                fig_gauge.update_layout(height=300)
                st.plotly_chart(fig_gauge, use_container_width=True)

            st.subheader("Individual Indicators")
            indicator_cols_display = st.columns(min(len(indicator_cols), 4))

            for idx, col_name in enumerate(indicator_cols):
                with indicator_cols_display[idx % 4]:
                    is_signaling = recession_indicators[col_name].iloc[-1] == 1
                    st.metric(
                        label=col_name.replace('_', ' '),
                        value="SIGNAL" if is_signaling else "Clear",
                        delta="⚠️" if is_signaling else "✅"
                    )

            st.subheader("Historical Recession Probability")
            fig_prob = go.Figure()
            fig_prob.add_trace(go.Scatter(
                x=recession_indicators.index,
                y=recession_indicators['Recession_Probability'],
                fill='tozeroy',
                name='Recession Probability',
                line=dict(color='darkred', width=2),
                fillcolor='rgba(220, 20, 60, 0.3)'
            ))
            fig_prob.add_hline(y=50, line_dash="dash", line_color="red", annotation_text="High Risk Threshold")
            fig_prob.update_layout(
                title='Recession Probability Over Time',
                xaxis_title='Date',
                yaxis_title='Probability (%)',
                template='plotly_white',
                hovermode='x unified',
                height=400,
                yaxis=dict(range=[0, 100])
            )
            st.plotly_chart(fig_prob, use_container_width=True)

        else:
            st.warning("⚠️ No recession indicators could be calculated with current data selection.")

    # ==================== 🧠 INTERPRETATION & INVESTMENT PLAYBOOK ====================

    if show_interpretation and (data_dict or not spreads.empty):
        # Determine best available phase
        interp_phase = current_phase if current_phase != 'Unknown' else 'Expansion'

        render_interpretation_section(
            data_dict=data_dict,
            current_phase=interp_phase,
            cycle_analysis=cycle_analysis,
            recession_indicators=recession_indicators,
            spreads=spreads
        )

    # ==================== CORRELATIONS ====================

    if data_dict and len(data_dict) > 1:
        with st.expander("📊 Statistical Analysis — Correlation Matrix"):
            corr_matrix = calculate_correlations(data_dict)

            if not corr_matrix.empty:
                fig_corr = px.imshow(
                    corr_matrix,
                    text_auto='.2f',
                    color_continuous_scale='RdBu_r',
                    aspect='auto',
                    title='Indicator Correlations',
                    labels=dict(color="Correlation"),
                    zmin=-1,
                    zmax=1
                )
                fig_corr.update_layout(height=600)
                st.plotly_chart(fig_corr, use_container_width=True)

    # ==================== EXPORT DATA ====================

    st.header("💾 Export Data")
    export_tabs = st.tabs(["Indicators", "Yield Curve", "Cycle Analysis", "Recession Indicators"])

    with export_tabs[0]:
        if data_dict:
            for metric, df in data_dict.items():
                if not df.empty:
                    csv = df.to_csv()
                    st.download_button(
                        label=f"📥 Download {metric.replace('_', ' ')}",
                        data=csv,
                        file_name=f"{metric}_{start_date}_{end_date}.csv",
                        mime="text/csv"
                    )

    with export_tabs[1]:
        if not yield_curves.empty:
            st.download_button(
                label="📥 Download Yield Curve Data",
                data=yield_curves.to_csv(),
                file_name=f"yield_curves_{start_date}_{end_date}.csv",
                mime="text/csv"
            )
            if not spreads.empty:
                st.download_button(
                    label="📥 Download Yield Spreads",
                    data=spreads.to_csv(),
                    file_name=f"yield_spreads_{start_date}_{end_date}.csv",
                    mime="text/csv"
                )

    with export_tabs[2]:
        if not phases.empty:
            cycle_export = pd.DataFrame({
                'Phase': phases,
                'Composite_Index': cycle_analysis['composite_index']
            })
            st.download_button(
                label="📥 Download Cycle Analysis",
                data=cycle_export.to_csv(),
                file_name=f"economic_cycle_{start_date}_{end_date}.csv",
                mime="text/csv"
            )

    with export_tabs[3]:
        if not recession_indicators.empty:
            st.download_button(
                label="📥 Download Recession Indicators",
                data=recession_indicators.to_csv(),
                file_name=f"recession_indicators_{start_date}_{end_date}.csv",
                mime="text/csv"
            )


# Run standalone
if __name__ == "__main__":
    st.set_page_config(
        page_title="Advanced Economic Analysis",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    economics_module()
