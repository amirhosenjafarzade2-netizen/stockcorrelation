# economics.py - Fixed Macro/Economic Context Module
# Uses reliable FRED API and Yahoo Finance for economic data

import streamlit as st
import pandas as pd
import requests
from io import StringIO
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, Optional, List
import numpy as np

# FRED API Configuration
FRED_API_KEY = "your_fred_api_key_here"  # Get free key from https://fred.stlouisfed.org/docs/api/api_key.html
FRED_API_URL = "https://api.stlouisfed.org/fred/series/observations"

# Metric definitions with working FRED series
METRIC_MAPPING = {
    "GDP": {
        "fred_code": "GDP",
        "unit": "Billions USD",
        "description": "Gross Domestic Product",
        "inverse": False
    },
    "Inflation_CPI": {
        "fred_code": "CPIAUCSL",
        "unit": "Index (1982-84=100)",
        "description": "Consumer Price Index",
        "inverse": True
    },
    "Unemployment": {
        "fred_code": "UNRATE",
        "unit": "Percentage",
        "description": "Unemployment Rate",
        "inverse": True
    },
    "Fed_Funds_Rate": {
        "fred_code": "FEDFUNDS",
        "unit": "Percentage",
        "description": "Federal Funds Effective Rate",
        "inverse": False
    },
    "10Y_Treasury": {
        "fred_code": "DGS10",
        "unit": "Percentage",
        "description": "10-Year Treasury Constant Maturity Rate",
        "inverse": False
    },
    "Consumer_Sentiment": {
        "fred_code": "UMCSENT",
        "unit": "Index (1966:Q1=100)",
        "description": "University of Michigan Consumer Sentiment",
        "inverse": False
    },
    "Industrial_Production": {
        "fred_code": "INDPRO",
        "unit": "Index (2017=100)",
        "description": "Industrial Production Index",
        "inverse": False
    },
    "Housing_Starts": {
        "fred_code": "HOUST",
        "unit": "Thousands of Units",
        "description": "Housing Starts",
        "inverse": False
    },
    "Retail_Sales": {
        "fred_code": "RSXFS",
        "unit": "Millions of Dollars",
        "description": "Advance Retail Sales",
        "inverse": False
    }
}

@st.cache_data(ttl=3600)
def get_fred_data_api(series_id: str, start_date: str, end_date: str, api_key: Optional[str] = None) -> pd.DataFrame:
    """
    Fetch data from FRED API (requires API key).
    Get your free API key at: https://fred.stlouisfed.org/docs/api/api_key.html
    """
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
        st.warning(f"FRED API error for {series_id}: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_fred_data_csv(series_id: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch data from FRED CSV download (no API key required).
    This is more reliable but has rate limits.
    """
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
        st.warning(f"Error fetching {series_id}: {str(e)}")
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
    
    # Combine all metrics
    combined = pd.DataFrame()
    for metric, df in data_dict.items():
        if not df.empty and len(df.columns) > 0:
            combined[metric] = df.iloc[:, 0]
    
    if not combined.empty and len(combined.columns) > 1:
        # Align data by dropping NaN rows
        combined = combined.dropna()
        if len(combined) > 10:  # Need enough data points
            return combined.corr()
    
    return pd.DataFrame()

def create_economic_dashboard(data_dict: Dict[str, pd.DataFrame]) -> go.Figure:
    """Create a comprehensive dashboard view."""
    n_metrics = len([d for d in data_dict.values() if not d.empty])
    
    if n_metrics == 0:
        return None
    
    # Calculate grid layout
    n_cols = 2
    n_rows = (n_metrics + 1) // 2
    
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[m.replace('_', ' ') for m in data_dict.keys() if not data_dict[m].empty],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    colors = px.colors.qualitative.Set2
    row, col = 1, 1
    
    for metric, df in data_dict.items():
        if df.empty:
            continue
        
        series = df.iloc[:, 0]
        
        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series.values,
                name=metric,
                line=dict(color=colors[(row-1)*n_cols + (col-1) % len(colors)], width=2),
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
        height=300 * n_rows,
        title_text="Economic Indicators Dashboard",
        template='plotly_white',
        hovermode='x unified'
    )
    
    return fig

def economics_module(analysis_context: Optional[Dict] = None):
    """Main economics module."""
    st.title("📊 Macro/Economic Context Analysis")
    
    st.markdown("""
    This module provides macroeconomic indicators from the Federal Reserve Economic Data (FRED).
    These indicators help contextualize stock performance within broader economic trends.
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
        
        # Metric selection
        st.subheader("📈 Select Indicators")
        
        selected_metrics = st.multiselect(
            "Economic Indicators",
            options=list(METRIC_MAPPING.keys()),
            default=["GDP", "Inflation_CPI", "Unemployment", "Fed_Funds_Rate"],
            format_func=lambda x: METRIC_MAPPING[x]['description']
        )
        
        # Date range
        st.subheader("📅 Date Range")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start",
                value=datetime.now() - timedelta(days=365*5),
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
    if not selected_metrics:
        st.warning("⚠️ Please select at least one economic indicator.")
        return
    
    if start_date >= end_date:
        st.error("❌ Start date must be before end date.")
        return
    
    # Fetch data
    data_dict = {}
    failed_series = []
    
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
    
    if not data_dict:
        st.error("❌ No data could be fetched. Please check your selections or try again later.")
        st.info("💡 Tip: If using CSV method, FRED might be rate-limiting. Try using the API key option.")
        return
    
    # Display current values
    st.header("📊 Current Economic Snapshot")
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
    st.header("📈 Economic Indicators Visualization")
    
    # Dashboard view
    dashboard_fig = create_economic_dashboard(data_dict)
    if dashboard_fig:
        st.plotly_chart(dashboard_fig, use_container_width=True)
    
    # Tabs for detailed views
    tab1, tab2, tab3 = st.tabs(["📊 Detailed Charts", "📋 Data Tables", "📈 Statistics & Correlations"])
    
    with tab1:
        for metric, df in data_dict.items():
            if df.empty:
                continue
            
            metric_info = METRIC_MAPPING[metric]
            
            with st.expander(f"📊 {metric_info['description']}", expanded=False):
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
                    growth = calculate_growth_rate(series, periods=4)
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
    
    with tab2:
        for metric, df in data_dict.items():
            if not df.empty:
                st.subheader(f"{METRIC_MAPPING[metric]['description']}")
                
                # Show recent data
                display_df = df.tail(50).sort_index(ascending=False)
                display_df.columns = [METRIC_MAPPING[metric]['description']]
                
                st.dataframe(
                    display_df.style.format("{:.2f}").background_gradient(cmap='RdYlGn', axis=0),
                    use_container_width=True,
                    height=400
                )
    
    with tab3:
        # Statistics
        st.subheader("📊 Statistical Summary")
        
        for metric, df in data_dict.items():
            if not df.empty:
                with st.expander(f"{METRIC_MAPPING[metric]['description']} Statistics"):
                    stats = calculate_statistics(df)
                    st.dataframe(
                        stats.style.format("{:.2f}").background_gradient(cmap='Blues', axis=1),
                        use_container_width=True
                    )
        
        # Correlations
        if len(data_dict) > 1:
            st.subheader("🔗 Correlation Analysis")
            corr_matrix = calculate_correlations(data_dict)
            
            if not corr_matrix.empty:
                fig = px.imshow(
                    corr_matrix,
                    text_auto='.2f',
                    color_continuous_scale='RdBu_r',
                    aspect='auto',
                    title='Correlation Matrix',
                    labels=dict(color="Correlation")
                )
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
                
                st.info("""
                **Interpretation Guide:**
                - Values close to +1: Strong positive correlation
                - Values close to -1: Strong negative correlation
                - Values close to 0: Little to no correlation
                """)
    
    # Export data
    st.header("💾 Export Data")
    
    export_cols = st.columns(len(data_dict))
    for idx, (metric, df) in enumerate(data_dict.items()):
        if not df.empty:
            with export_cols[idx]:
                csv = df.to_csv()
                st.download_button(
                    label=f"📥 {metric.replace('_', ' ')}",
                    data=csv,
                    file_name=f"{metric}_{start_date}_{end_date}.csv",
                    mime="text/csv"
                )
    
    # Insights
    if analysis_context and 'ticker' in analysis_context:
        st.header("💡 Economic Context")
        st.info(f"""
        **Analyzing**: {analysis_context.get('ticker', 'N/A')}
        
        **Key Economic Relationships**:
        - **GDP & Earnings**: GDP growth typically correlates with corporate earnings growth
        - **Inflation & Margins**: Rising inflation can compress profit margins
        - **Interest Rates & Valuations**: Higher rates increase discount rates, pressuring valuations
        - **Unemployment & Spending**: Low unemployment supports consumer spending
        """)

# Run standalone
if __name__ == "__main__":
    economics_module()
