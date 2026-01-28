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
    
    # 10Y-2Y spread (most watched inversion indicator)
    if '10Y' in yield_curve_df.columns and '2Y' in yield_curve_df.columns:
        spreads['10Y-2Y'] = yield_curve_df['10Y'] - yield_curve_df['2Y']
    
    # 10Y-3M spread (also important)
    if '10Y' in yield_curve_df.columns and '3M' in yield_curve_df.columns:
        spreads['10Y-3M'] = yield_curve_df['10Y'] - yield_curve_df['3M']
    
    # 2Y-3M spread
    if '2Y' in yield_curve_df.columns and '3M' in yield_curve_df.columns:
        spreads['2Y-3M'] = yield_curve_df['2Y'] - yield_curve_df['3M']
    
    # 30Y-5Y spread (long-term steepness)
    if '30Y' in yield_curve_df.columns and '5Y' in yield_curve_df.columns:
        spreads['30Y-5Y'] = yield_curve_df['30Y'] - yield_curve_df['5Y']
    
    return spreads

def detect_inversions(spreads: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Detect yield curve inversions and their durations."""
    inversions = {}
    
    for spread_name in spreads.columns:
        spread_data = spreads[spread_name].dropna()
        
        # Find inversion periods (spread < 0)
        is_inverted = spread_data < 0
        
        # Find start and end dates of inversions
        inversion_changes = is_inverted.astype(int).diff()
        starts = spread_data.index[inversion_changes == 1]
        ends = spread_data.index[inversion_changes == -1]
        
        # Handle edge cases
        if len(starts) > 0 or len(ends) > 0:
            if len(starts) > 0 and (len(ends) == 0 or starts[0] < ends[0]):
                if is_inverted.iloc[0]:
                    starts = spread_data.index[[0]].append(starts)
            
            if len(ends) > 0 and (len(starts) == 0 or ends[-1] > starts[-1]):
                if is_inverted.iloc[-1]:
                    ends = ends.append(spread_data.index[[-1]])
            
            # Create DataFrame of inversion periods
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
    
    # Convert maturity labels to numeric for plotting
    maturity_map = {'1M': 0.083, '3M': 0.25, '6M': 0.5, '1Y': 1, '2Y': 2, 
                   '3Y': 3, '5Y': 5, '7Y': 7, '10Y': 10, '20Y': 20, '30Y': 30}
    
    x_values = [maturity_map[m] for m in yields.index]
    
    # Determine if inverted
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
        
        # Add inversion periods as shaded regions
        if spread_name in inversions and not inversions[spread_name].empty:
            for _, inv in inversions[spread_name].iterrows():
                fig.add_vrect(
                    x0=inv['Start'], x1=inv['End'],
                    fillcolor=colors[idx % len(colors)], opacity=0.1,
                    layer="below", line_width=0,
                )
    
    # Add zero line
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
    """
    Detect economic cycle phases using multiple indicators.
    Returns: (cycle_phase_series, analysis_dict)
    """
    
    # Combine key indicators
    indicators = pd.DataFrame()
    required_indicators = []
    
    # Growth indicators (positive = expansion)
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
    
    # Labor indicators (low unemployment = expansion, but inverted)
    if 'Unemployment' in data_dict and not data_dict['Unemployment'].empty:
        indicators['Unemployment_Inverted'] = -data_dict['Unemployment'].iloc[:, 0]  # Invert
        required_indicators.append('Unemployment_Inverted')
    
    # Confidence indicators
    if 'Consumer_Sentiment' in data_dict and not data_dict['Consumer_Sentiment'].empty:
        indicators['Sentiment'] = data_dict['Consumer_Sentiment'].iloc[:, 0]
        required_indicators.append('Sentiment')
    
    if 'Leading_Index' in data_dict and not data_dict['Leading_Index'].empty:
        lei_growth = calculate_growth_rate(data_dict['Leading_Index'].iloc[:, 0], periods=12)
        if len(lei_growth.dropna()) > 0:
            indicators['LEI_Growth'] = lei_growth
            required_indicators.append('LEI_Growth')
    
    # Additional useful indicators
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
    
    # Drop NaN and standardize
    indicators = indicators.dropna()
    
    if indicators.empty or len(indicators) < 50:
        return pd.Series(), {'error': 'insufficient_data', 'available_indicators': required_indicators, 'min_required': 2}
    
    # Standardize indicators
    scaler = StandardScaler()
    indicators_scaled = pd.DataFrame(
        scaler.fit_transform(indicators),
        index=indicators.index,
        columns=indicators.columns
    )
    
    # Create composite index (average of standardized indicators)
    composite = indicators_scaled.mean(axis=1)
    
    # Smooth the composite index
    composite_smooth = composite.rolling(window=6, center=True).mean()
    
    # Detect peaks and troughs using scipy
    peaks, _ = signal.find_peaks(composite_smooth.dropna(), distance=12, prominence=0.5)
    troughs, _ = signal.find_peaks(-composite_smooth.dropna(), distance=12, prominence=0.5)
    
    # Classify phases
    phases = pd.Series(index=composite.index, dtype=str)
    phases[:] = 'Unknown'
    
    composite_values = composite_smooth.dropna()
    peak_dates = composite_values.index[peaks]
    trough_dates = composite_values.index[troughs]
    
    # Combine and sort turning points
    all_turns = sorted(list(peak_dates) + list(trough_dates))
    
    # Assign phases between turning points
    for i in range(len(all_turns) - 1):
        start_date = all_turns[i]
        end_date = all_turns[i + 1]
        
        if start_date in peak_dates:
            phases.loc[start_date:end_date] = 'Contraction'
        elif start_date in trough_dates:
            phases.loc[start_date:end_date] = 'Expansion'
    
    # Handle edges
    if len(all_turns) > 0:
        if all_turns[0] in peak_dates:
            phases.loc[:all_turns[0]] = 'Expansion'
        else:
            phases.loc[:all_turns[0]] = 'Contraction'
        
        if all_turns[-1] in peak_dates:
            phases.loc[all_turns[-1]:] = 'Contraction'
        else:
            phases.loc[all_turns[-1]:] = 'Expansion'
    
    # Refine based on composite value
    current_phase = phases.iloc[-1] if len(phases) > 0 else 'Unknown'
    current_composite = composite.iloc[-1] if len(composite) > 0 else 0
    
    # Further classify expansion/contraction into early/late phases
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
    
    # Analysis dictionary
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
    
    # Plot composite index
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
    
    # Add peaks and troughs
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
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
    
    # Plot phases
    phase_colors = {
        'Early Expansion': '#90EE90',
        'Late Expansion': '#228B22',
        'Early Contraction': '#FFB6C1',
        'Late Contraction': '#DC143C',
        'Expansion': '#32CD32',
        'Contraction': '#FF6347',
        'Unknown': '#D3D3D3'
    }
    
    # Create phase visualization
    phase_numeric = phases.map({
        'Early Expansion': 2,
        'Late Expansion': 2,
        'Expansion': 2,
        'Early Contraction': -2,
        'Late Contraction': -2,
        'Contraction': -2,
        'Unknown': 0
    })
    
    # Color each phase segment
    current_phase = None
    segment_start = None
    
    for i, (date, phase) in enumerate(phases.items()):
        if phase != current_phase:
            if current_phase is not None and segment_start is not None:
                # Draw previous segment
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
    
    # Draw final segment
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
    
    # Update layout
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
    
    # Sahm Rule: Unemployment rate rises 0.5pp above 3-month low
    if 'Unemployment' in data_dict and not data_dict['Unemployment'].empty:
        unemp = data_dict['Unemployment'].iloc[:, 0]
        rolling_min = unemp.rolling(window=3).min()
        sahm = unemp - rolling_min
        indicators['Sahm_Rule'] = (sahm >= 0.5).astype(int)
    
    # Yield curve inversion (10Y-2Y)
    if '10Y-2Y' in spreads.columns:
        indicators['Yield_Curve_Inverted'] = (spreads['10Y-2Y'] < 0).astype(int)
    
    # Leading Economic Index decline
    if 'Leading_Index' in data_dict and not data_dict['Leading_Index'].empty:
        lei = data_dict['Leading_Index'].iloc[:, 0]
        lei_6m_change = lei.pct_change(periods=6) * 100
        indicators['LEI_Declining'] = (lei_6m_change < -2).astype(int)
    
    # Consumer sentiment collapse
    if 'Consumer_Sentiment' in data_dict and not data_dict['Consumer_Sentiment'].empty:
        sent = data_dict['Consumer_Sentiment'].iloc[:, 0]
        sent_change = sent.pct_change(periods=12) * 100
        indicators['Sentiment_Collapse'] = (sent_change < -15).astype(int)
    
    # Industrial production decline
    if 'Industrial_Production' in data_dict and not data_dict['Industrial_Production'].empty:
        ip = data_dict['Industrial_Production'].iloc[:, 0]
        ip_6m_change = ip.pct_change(periods=6) * 100
        indicators['IP_Declining'] = (ip_6m_change < -2).astype(int)
    
    # Calculate composite recession probability
    if not indicators.empty:
        indicators['Recession_Probability'] = indicators.mean(axis=1) * 100
    
    return indicators

# ==================== MAIN MODULE ====================

def create_economic_dashboard(data_dict: Dict[str, pd.DataFrame]) -> go.Figure:
    """Create a comprehensive dashboard view."""
    n_metrics = len([d for d in data_dict.values() if not d.empty])
    
    if n_metrics == 0:
        return None
    
    # Calculate grid layout
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    # Get metric names for titles
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
        
        # Update axes
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
    """)
    
    # Quick start guide
    with st.expander("🚀 Quick Start Guide", expanded=False):
        st.markdown("""
        ### Getting Started
        
        **1. Optional: Get a FREE FRED API Key** (Recommended)
        - Visit [FRED API Key Registration](https://fred.stlouisfed.org/docs/api/api_key.html)
        - Enter your email to receive an API key instantly
        - Paste it in the sidebar for faster data fetching
        - Without API key, the app still works but uses slower CSV downloads
        
        **2. Select Your Analysis Modules** (Sidebar)
        - ✅ **Economic Indicators**: Core metrics like GDP, unemployment, inflation
        - ✅ **Yield Curve Analysis**: Treasury rates and recession signals
        - ✅ **Economic Cycle Detection**: Expansion/contraction phase identification
        - ✅ **Recession Indicators**: Multi-factor probability assessment
        
        **3. Choose Your Indicators** (Sidebar)
        The app has smart defaults pre-selected, but you can customize:
        - **For Cycle Detection**: Need GDP/Industrial Production + Unemployment + Sentiment
        - **For Yield Curve**: Automatically fetches all Treasury rates
        - **For Recession Signals**: Uses unemployment, leading index, and yield curve
        
        **4. Set Date Range** (Sidebar)
        - Default: 10 years (recommended for cycle detection)
        - Minimum: 2-3 years for meaningful analysis
        - Maximum: All available data (varies by indicator)
        
        **5. Explore the Results!**
        - Current economic snapshot with metrics
        - Interactive charts and dashboards
        - Export data to CSV for further analysis
        """)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key input
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
        
        # Analysis modules
        st.subheader("📊 Analysis Modules")
        show_indicators = st.checkbox("Economic Indicators", value=True)
        show_yield_curve = st.checkbox("Yield Curve Analysis", value=True)
        show_cycle = st.checkbox("Economic Cycle Detection", value=True)
        show_recession = st.checkbox("Recession Indicators", value=True)
        
        # Metric selection by category
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
        
        # Default selections optimized for all features
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
        
        # Date range
        st.subheader("📅 Date Range")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start",
                value=datetime.now() - timedelta(days=365*10),
                max_value=datetime.now()
            )
        with col2:
            end_date = st.date_input(
                "End",
                value=datetime.now(),
                max_value=datetime.now()
            )
        
        # Display options
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
                
                # Try API first if configured
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
    
    # Show warnings for failed series
    if failed_series:
        st.warning(f"⚠️ Could not fetch data for: {', '.join(failed_series)}")
    
    # ==================== ECONOMIC INDICATORS ====================
    
    if show_indicators and data_dict:
        st.header("📊 Economic Indicators")
        
        # Display current values
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
        
        # Main visualization
        st.subheader("Indicators Dashboard")
        dashboard_fig = create_economic_dashboard(data_dict)
        if dashboard_fig:
            st.plotly_chart(dashboard_fig, use_container_width=True)
        
        # Detailed views
        with st.expander("📈 Detailed Charts & Data"):
            for metric, df in data_dict.items():
                if df.empty:
                    continue
                
                metric_info = METRIC_MAPPING[metric]
                
                st.markdown(f"### {metric_info['description']}")
                
                series = df.iloc[:, 0]
                
                # Apply transformations
                if normalize_data:
                    series = (series / series.iloc[0]) * 100
                
                fig = go.Figure()
                
                # Main series
                fig.add_trace(go.Scatter(
                    x=series.index,
                    y=series.values,
                    name=metric_info['description'],
                    line=dict(color='#1f77b4', width=2),
                    mode='lines',
                    hovertemplate='%{y:.2f}<extra></extra>'
                ))
                
                # Add growth rate if requested
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
                
                # Layout
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
                
                # Statistics
                stats = calculate_statistics(df)
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.dataframe(
                        stats.style.format("{:.2f}").background_gradient(cmap='Blues', axis=1),
                        use_container_width=True
                    )
    
    # ==================== YIELD CURVE ANALYSIS ====================
    
    if show_yield_curve:
        st.header("📈 Yield Curve Analysis")
        
        with st.spinner("Fetching yield curve data..."):
            # Get historical yield curve
            yield_curves = get_yield_curve_historical(
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d"),
                use_api,
                api_key
            )
            
            if not yield_curves.empty:
                # Calculate spreads
                spreads = calculate_yield_spreads(yield_curves)
                
                # Detect inversions
                inversions = detect_inversions(spreads)
                
                # Current yield curve
                st.subheader("Current Yield Curve")
                current_yields = yield_curves.iloc[-1].dropna()
                
                if not current_yields.empty:
                    current_fig = plot_yield_curve_current(current_yields)
                    st.plotly_chart(current_fig, use_container_width=True)
                    
                    # Display current spreads
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
                
                # Historical spreads
                st.subheader("Historical Yield Spreads")
                spreads_fig = plot_yield_spreads(spreads, inversions)
                st.plotly_chart(spreads_fig, use_container_width=True)
                
                # Inversion summary
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
                
                # 3D yield curve evolution
                with st.expander("🔍 3D Yield Curve Evolution"):
                    # Sample dates for 3D plot (monthly)
                    sample_dates = yield_curves.resample('M').last().dropna(how='all').tail(36)
                    
                    if not sample_dates.empty:
                        maturity_map = {'1M': 0.083, '3M': 0.25, '6M': 0.5, '1Y': 1, '2Y': 2,
                                       '3Y': 3, '5Y': 5, '7Y': 7, '10Y': 10, '20Y': 20, '30Y': 30}
                        
                        fig_3d = go.Figure()
                        
                        for idx, (date, yields) in enumerate(sample_dates.iterrows()):
                            yields_clean = yields.dropna()
                            x_values = [maturity_map[m] for m in yields_clean.index]
                            
                            fig_3d.add_trace(go.Scatter3d(
                                x=x_values,
                                y=[date] * len(x_values),
                                z=yields_clean.values,
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
    
    if show_cycle and data_dict:
        st.header("🔄 Economic Cycle Analysis")
        
        with st.spinner("Detecting economic cycle..."):
            phases, cycle_analysis = detect_economic_cycle(data_dict)
            
            if phases.empty or 'error' in cycle_analysis:
                # Show helpful guidance
                st.warning("⚠️ Insufficient data for cycle detection.")
                
                available = cycle_analysis.get('available_indicators', [])
                min_required = cycle_analysis.get('min_required', 2)
                
                st.info(f"""
                **Cycle detection requires at least {min_required} indicators with sufficient historical data.**
                
                Currently available: {len(available)} indicator(s)
                - {', '.join(available) if available else 'None'}
                
                **Recommended indicators for best results:**
                """)
                
                # Create recommendation table
                recommendations = pd.DataFrame({
                    'Category': ['Output & Production', 'Output & Production', 'Labor Market', 'Confidence', 'Composite'],
                    'Indicator': ['GDP or GDP Growth', 'Industrial Production', 'Unemployment', 'Consumer Sentiment', 'Leading Economic Index'],
                    'Why Important': [
                        'Core measure of economic activity',
                        'Leading indicator for manufacturing sector',
                        'Key labor market health indicator',
                        'Forward-looking consumer behavior',
                        'Aggregates multiple leading indicators'
                    ],
                    'Currently Selected': [
                        '✅' if any('GDP' in i for i in available) else '❌',
                        '✅' if 'IP_Growth' in available else '❌',
                        '✅' if 'Unemployment_Inverted' in available else '❌',
                        '✅' if 'Sentiment' in available else '❌',
                        '✅' if 'LEI_Growth' in available else '❌'
                    ]
                })
                
                st.dataframe(recommendations, use_container_width=True, hide_index=True)
                
                st.markdown("""
                **Quick Start:** Select these indicators from the sidebar:
                1. **Output & Production** → GDP or Industrial Production
                2. **Labor Market** → Unemployment
                3. **Confidence** → Consumer Sentiment
                4. **Composite Indices** → Leading Economic Index (highly recommended)
                
                Then expand the date range to at least 5 years for better cycle detection.
                """)
                
            elif not phases.empty and cycle_analysis:
                # Current phase
                current_phase = cycle_analysis['current_phase']
                phase_duration = cycle_analysis['phase_duration_months']
                
                # Phase indicators
                phase_colors = {
                    'Early Expansion': '🟢',
                    'Late Expansion': '🟡',
                    'Early Contraction': '🟠',
                    'Late Contraction': '🔴',
                    'Expansion': '🟢',
                    'Contraction': '🔴'
                }
                
                phase_emoji = phase_colors.get(current_phase, '⚪')
                
                st.subheader(f"Current Phase: {phase_emoji} {current_phase}")
                st.info(f"Duration: {phase_duration} months")
                
                # Cycle visualization
                cycle_fig = plot_economic_cycle(phases, cycle_analysis)
                st.plotly_chart(cycle_fig, use_container_width=True)
                
                # Phase characteristics
                with st.expander("📖 Phase Characteristics"):
                    st.markdown("""
                    **Early Expansion** 🟢
                    - GDP growth accelerating
                    - Unemployment declining
                    - Rising consumer confidence
                    - Credit expanding
                    
                    **Late Expansion** 🟡
                    - GDP growth near peak
                    - Tight labor markets
                    - Inflation pressures building
                    - Interest rates rising
                    
                    **Early Contraction** 🟠
                    - GDP growth slowing
                    - Unemployment rising
                    - Weakening consumer confidence
                    - Credit tightening
                    
                    **Late Contraction** 🔴
                    - GDP declining
                    - High unemployment
                    - Low consumer confidence
                    - Interest rates falling
                    """)
                
                # Contributing indicators
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
    
    if show_recession and (data_dict or (show_yield_curve and not yield_curves.empty)):
        st.header("⚠️ Recession Indicators")
        
        # Get spreads if available
        spreads_for_recession = pd.DataFrame()
        if show_yield_curve and 'yield_curves' in locals() and not yield_curves.empty:
            spreads_for_recession = calculate_yield_spreads(yield_curves)
        
        recession_indicators = get_recession_indicators(data_dict, spreads_for_recession)
        
        if not recession_indicators.empty:
            # Current probability
            current_prob = recession_indicators['Recession_Probability'].iloc[-1]
            
            # Risk level
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
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.metric(
                    label="Recession Probability",
                    value=f"{current_prob:.1f}%",
                    delta=f"{current_prob - recession_indicators['Recession_Probability'].iloc[-30]:.1f}%" if len(recession_indicators) > 30 else None
                )
            
            with col2:
                # Gauge chart
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
            
            # Individual indicators
            st.subheader("Individual Indicators")
            
            indicator_cols = st.columns(min(len(recession_indicators.columns) - 1, 4))
            
            for idx, col_name in enumerate([c for c in recession_indicators.columns if c != 'Recession_Probability']):
                with indicator_cols[idx % 4]:
                    is_signaling = recession_indicators[col_name].iloc[-1] == 1
                    
                    st.metric(
                        label=col_name.replace('_', ' '),
                        value="SIGNAL" if is_signaling else "Clear",
                        delta="⚠️" if is_signaling else "✅"
                    )
            
            # Historical probability
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
            
            # Add threshold lines
            fig_prob.add_hline(y=50, line_dash="dash", line_color="red", 
                             annotation_text="High Risk Threshold")
            
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
            
            # Indicator details
            with st.expander("📊 Indicator Details"):
                st.markdown("""
                **Sahm Rule**: Recession signal when unemployment rises 0.5pp above its 3-month low
                
                **Yield Curve Inversion**: 10Y-2Y spread below zero (historically preceded recessions by 6-24 months)
                
                **LEI Declining**: Leading Economic Index falls more than 2% over 6 months
                
                **Sentiment Collapse**: Consumer sentiment drops more than 15% year-over-year
                
                **IP Declining**: Industrial production falls more than 2% over 6 months
                """)
        else:
            st.info("💡 Add more indicators to calculate recession probability (Unemployment, Leading Index, Consumer Sentiment, etc.)")
    
    # ==================== CORRELATIONS & STATISTICS ====================
    
    if data_dict and len(data_dict) > 1:
        with st.expander("📊 Statistical Analysis"):
            st.subheader("Correlation Matrix")
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
                
                st.info("""
                **Interpretation Guide:**
                - **+0.7 to +1.0**: Strong positive correlation
                - **+0.3 to +0.7**: Moderate positive correlation
                - **-0.3 to +0.3**: Weak or no correlation
                - **-0.7 to -0.3**: Moderate negative correlation
                - **-1.0 to -0.7**: Strong negative correlation
                """)
    
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
        if show_yield_curve and 'yield_curves' in locals() and not yield_curves.empty:
            csv_curves = yield_curves.to_csv()
            st.download_button(
                label="📥 Download Yield Curve Data",
                data=csv_curves,
                file_name=f"yield_curves_{start_date}_{end_date}.csv",
                mime="text/csv"
            )
            
            csv_spreads = spreads.to_csv()
            st.download_button(
                label="📥 Download Yield Spreads",
                data=csv_spreads,
                file_name=f"yield_spreads_{start_date}_{end_date}.csv",
                mime="text/csv"
            )
    
    with export_tabs[2]:
        if show_cycle and 'phases' in locals() and not phases.empty:
            cycle_export = pd.DataFrame({
                'Phase': phases,
                'Composite_Index': cycle_analysis['composite_index']
            })
            csv_cycle = cycle_export.to_csv()
            st.download_button(
                label="📥 Download Cycle Analysis",
                data=csv_cycle,
                file_name=f"economic_cycle_{start_date}_{end_date}.csv",
                mime="text/csv"
            )
    
    with export_tabs[3]:
        if show_recession and 'recession_indicators' in locals() and not recession_indicators.empty:
            csv_recession = recession_indicators.to_csv()
            st.download_button(
                label="📥 Download Recession Indicators",
                data=csv_recession,
                file_name=f"recession_indicators_{start_date}_{end_date}.csv",
                mime="text/csv"
            )
    
    # ==================== INSIGHTS ====================
    
    if analysis_context and 'ticker' in analysis_context:
        st.header("💡 Economic Context Insights")
        
        insights = []
        
        # Phase-based insights
        if show_cycle and 'current_phase' in locals():
            phase_insights = {
                'Early Expansion': "Favorable for cyclical stocks and small caps. Rising tide lifts all boats.",
                'Late Expansion': "Focus on quality and pricing power. Watch for inflation pressures.",
                'Early Contraction': "Defensive positioning recommended. Flight to quality begins.",
                'Late Contraction': "Opportunities emerging. Position for recovery."
            }
            insights.append(f"**Current Phase**: {current_phase} - {phase_insights.get(current_phase, '')}")
        
        # Yield curve insights
        if show_yield_curve and 'current_yields' in locals():
            if '10Y-2Y' in spreads.columns:
                current_10y2y = spreads['10Y-2Y'].iloc[-1]
                if current_10y2y < 0:
                    insights.append(f"**⚠️ Yield Curve Inverted**: 10Y-2Y spread at {current_10y2y:.2f}%. Historically precedes recessions by 6-24 months.")
                elif current_10y2y < 0.5:
                    insights.append(f"**Flattening Yield Curve**: 10Y-2Y spread at {current_10y2y:.2f}%. Monitor for inversion.")
        
        # Recession risk insights
        if show_recession and 'current_prob' in locals():
            if current_prob > 60:
                insights.append(f"**🔴 High Recession Risk**: {current_prob:.1f}% probability. Consider defensive positioning.")
            elif current_prob > 40:
                insights.append(f"**🟠 Elevated Recession Risk**: {current_prob:.1f}% probability. Increase quality bias.")
        
        if insights:
            for insight in insights:
                st.info(insight)
        
        # General economic relationships
        with st.expander("📚 Key Economic Relationships"):
            st.markdown(f"""
            **Analyzing**: {analysis_context.get('ticker', 'N/A')}
            
            **Key Economic Relationships**:
            
            **Growth & Earnings**
            - GDP growth typically correlates with corporate earnings growth
            - Industrial production is a leading indicator for manufacturing firms
            - Retail sales drive consumer discretionary sectors
            
            **Inflation & Margins**
            - Rising inflation can compress profit margins for firms with limited pricing power
            - Core inflation (ex-food & energy) is watched closely by the Fed
            - PCE is the Fed's preferred inflation measure
            
            **Interest Rates & Valuations**
            - Higher rates increase discount rates, pressuring equity valuations
            - Fed Funds Rate affects borrowing costs across the economy
            - 10Y Treasury is the benchmark for long-term rates
            
            **Labor Market & Consumption**
            - Low unemployment supports consumer spending (70% of GDP)
            - Wage growth affects both consumer purchasing power and corporate costs
            - Initial jobless claims are a high-frequency labor market indicator
            
            **Credit & Liquidity**
            - Yield curve shape reflects market expectations for growth and rates
            - Inverted curve has preceded every recession since 1955
            - Steep curve suggests economic acceleration ahead
            
            **Sentiment & Forward Indicators**
            - Consumer sentiment drives spending decisions
            - Leading Economic Index aggregates multiple forward-looking indicators
            - Confidence indicators can be self-fulfilling
            """)

# Run standalone
if __name__ == "__main__":
    st.set_page_config(
        page_title="Advanced Economic Analysis",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    economics_module()
