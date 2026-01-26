# economics.py - Enhanced Macro/Economic Context Module
# Integrates OECD/FRED data for GDP, inflation, rates, unemployment with improved features
# Add to your app's directory, then import/load in app.py like other modules

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

# Configuration
OECD_BASE_URL = "https://sdmx.oecd.org/public/rest/data/OECD.SDD.NAD,DSD_NAMAIN1@DF_QNA_EXPENDITURE_USD,1.0"
FRED_BASE_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"

# Mapping of metrics to OECD/FRED series
METRIC_MAPPING = {
    "GDP": {
        "oecd_code": "GAPAGDP",
        "fred_code": "GDP",
        "unit": "Billions USD",
        "description": "Gross Domestic Product"
    },
    "Inflation": {
        "oecd_code": "CPALTT01",
        "fred_code": "CPIAUCSL",
        "unit": "Index (2015=100)",
        "description": "Consumer Price Index"
    },
    "Unemployment": {
        "oecd_code": "LRHUTTTT",
        "fred_code": "UNRATE",
        "unit": "Percentage",
        "description": "Unemployment Rate"
    },
    "Interest_Rates": {
        "fred_code": "FEDFUNDS",
        "unit": "Percentage",
        "description": "Federal Funds Rate"
    },
    "10Y_Treasury": {
        "fred_code": "DGS10",
        "unit": "Percentage",
        "description": "10-Year Treasury Yield"
    },
    "Consumer_Confidence": {
        "fred_code": "UMCSENT",
        "unit": "Index",
        "description": "University of Michigan Consumer Sentiment"
    }
}

COUNTRY_CODES = {
    "USA": "United States",
    "DEU": "Germany",
    "JPN": "Japan",
    "TUR": "Turkey",
    "GBR": "United Kingdom",
    "FRA": "France",
    "CHN": "China",
    "IND": "India",
    "BRA": "Brazil",
    "CAN": "Canada"
}

@st.cache_data(ttl=3600)  # Cache for 1 hour
def collect_oecd_data(indicator: str, countries: List[str], frequency: str = "Q") -> pd.DataFrame:
    """
    Fetch OECD data with improved error handling and retry logic.
    
    Args:
        indicator: OECD indicator code
        countries: List of country codes
        frequency: Data frequency (Q=Quarterly, M=Monthly, A=Annual)
    
    Returns:
        DataFrame with dates as index and countries as columns
    """
    try:
        # Construct country filter
        country_filter = "+".join(countries)
        
        # Alternative OECD API endpoint that's more reliable
        url = f"https://stats.oecd.org/sdmx-json/data/DP_LIVE/.{indicator}.../OECD?contentType=csv&detail=code&separator=comma&csv-lang=en"
        
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        if response.status_code == 200 and len(response.text) > 0:
            data = pd.read_csv(StringIO(response.text))
            
            # Filter for selected countries
            if 'LOCATION' in data.columns:
                data = data[data['LOCATION'].isin(countries)]
            
            # Parse time column
            if 'TIME' in data.columns:
                data['TIME'] = pd.to_datetime(data['TIME'])
                data = data.pivot(index='TIME', columns='LOCATION', values='Value')
                data.index.name = 'Date'
                return data.sort_index()
        
        return pd.DataFrame()
    
    except requests.exceptions.RequestException as e:
        st.warning(f"OECD API error for {indicator}: {str(e)}")
        return pd.DataFrame()
    except Exception as e:
        st.warning(f"Error processing OECD data for {indicator}: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_fred_data(series_id: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
    """
    Fetch FRED data with improved error handling.
    
    Args:
        series_id: FRED series ID
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
    
    Returns:
        DataFrame with date index and series values
    """
    try:
        url = f"{FRED_BASE_URL}?id={series_id}"
        
        if start_date:
            url += f"&cosd={start_date}"
        if end_date:
            url += f"&coed={end_date}"
        
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        if response.status_code == 200:
            data = pd.read_csv(StringIO(response.text), index_col="DATE", parse_dates=True)
            data.columns = [series_id]
            return data
        
        return pd.DataFrame()
    
    except requests.exceptions.RequestException as e:
        st.warning(f"FRED API error for {series_id}: {str(e)}")
        return pd.DataFrame()
    except Exception as e:
        st.warning(f"Error processing FRED data for {series_id}: {str(e)}")
        return pd.DataFrame()

def calculate_statistics(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate key statistics for the data."""
    stats = pd.DataFrame({
        'Mean': data.mean(),
        'Median': data.median(),
        'Std Dev': data.std(),
        'Min': data.min(),
        'Max': data.max(),
        'Current': data.iloc[-1] if len(data) > 0 else np.nan,
        'Change (%)': ((data.iloc[-1] - data.iloc[0]) / data.iloc[0] * 100) if len(data) > 0 else np.nan
    })
    return stats.T

def calculate_correlations(data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Calculate correlations between different metrics."""
    if len(data_dict) < 2:
        return pd.DataFrame()
    
    # Combine all metrics into one dataframe
    combined = pd.DataFrame()
    for metric, df in data_dict.items():
        if not df.empty:
            # Use first column if multiple countries
            col_name = f"{metric}_{df.columns[0]}" if len(df.columns) > 0 else metric
            combined[col_name] = df.iloc[:, 0] if len(df.columns) > 0 else pd.Series()
    
    if not combined.empty and len(combined.columns) > 1:
        return combined.corr()
    
    return pd.DataFrame()

def create_comparison_chart(data_dict: Dict[str, pd.DataFrame], normalize: bool = False) -> go.Figure:
    """Create an interactive multi-metric comparison chart."""
    fig = make_subplots(
        rows=len(data_dict),
        cols=1,
        subplot_titles=list(data_dict.keys()),
        vertical_spacing=0.1
    )
    
    colors = px.colors.qualitative.Set2
    
    for idx, (metric, df) in enumerate(data_dict.items(), 1):
        if df.empty:
            continue
        
        for col_idx, col in enumerate(df.columns):
            plot_data = df[col].copy()
            
            if normalize:
                # Normalize to 100 at start
                plot_data = (plot_data / plot_data.iloc[0]) * 100
            
            fig.add_trace(
                go.Scatter(
                    x=plot_data.index,
                    y=plot_data.values,
                    name=f"{metric} - {col}",
                    line=dict(color=colors[col_idx % len(colors)]),
                    mode='lines',
                    showlegend=(idx == 1)  # Only show legend for first subplot
                ),
                row=idx,
                col=1
            )
    
    fig.update_layout(
        height=300 * len(data_dict),
        title_text="Economic Indicators Over Time",
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig

def economics_module(analysis_context: Optional[Dict] = None):
    """
    Main economics module with enhanced features.
    
    Args:
        analysis_context: Optional context from main app (ticker, dates, etc.)
    """
    st.title("📊 Macro/Economic Context Analysis")
    st.markdown("""
    Analyze macroeconomic indicators using data from OECD and FRED (Federal Reserve Economic Data).
    This helps contextualize stock performance within broader economic trends.
    """)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Country selection
        selected_countries = st.multiselect(
            "Select Countries",
            options=list(COUNTRY_CODES.keys()),
            default=["USA"],
            format_func=lambda x: f"{x} - {COUNTRY_CODES[x]}"
        )
        
        # Metric selection
        available_metrics = list(METRIC_MAPPING.keys())
        selected_metrics = st.multiselect(
            "Select Economic Indicators",
            options=available_metrics,
            default=["GDP", "Inflation", "Unemployment"],
            help="Choose which economic indicators to analyze"
        )
        
        # Date range
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=pd.to_datetime("2020-01-01"),
                max_value=datetime.now()
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=datetime.now(),
                max_value=datetime.now()
            )
        
        # Additional options
        normalize_data = st.checkbox("Normalize data (base 100)", value=False)
        show_statistics = st.checkbox("Show statistics", value=True)
        show_correlations = st.checkbox("Show correlations", value=True)
    
    # Validation
    if not selected_countries:
        st.warning("⚠️ Please select at least one country.")
        return
    
    if not selected_metrics:
        st.warning("⚠️ Please select at least one metric.")
        return
    
    if start_date >= end_date:
        st.error("❌ Start date must be before end date.")
        return
    
    # Data collection
    data_dict = {}
    
    with st.spinner("Fetching economic data..."):
        progress_bar = st.progress(0)
        
        for idx, metric in enumerate(selected_metrics):
            metric_info = METRIC_MAPPING[metric]
            
            # Try FRED first (more reliable for US data)
            if "fred_code" in metric_info:
                fred_data = get_fred_data(
                    metric_info["fred_code"],
                    start_date.strftime("%Y-%m-%d"),
                    end_date.strftime("%Y-%m-%d")
                )
                
                if not fred_data.empty:
                    # Filter by date range
                    fred_data = fred_data.loc[start_date:end_date]
                    data_dict[metric] = fred_data.rename(columns={metric_info["fred_code"]: "USA"})
            
            # Try OECD for multi-country data
            elif "oecd_code" in metric_info and len(selected_countries) > 0:
                oecd_data = collect_oecd_data(
                    metric_info["oecd_code"],
                    selected_countries
                )
                
                if not oecd_data.empty:
                    oecd_data = oecd_data.loc[start_date:end_date]
                    data_dict[metric] = oecd_data
            
            progress_bar.progress((idx + 1) / len(selected_metrics))
        
        progress_bar.empty()
    
    # Display results
    if not data_dict:
        st.error("❌ No data could be fetched. Please try different selections or check your internet connection.")
        return
    
    # Summary metrics
    st.header("📈 Current Values & Summary")
    cols = st.columns(len(data_dict))
    
    for idx, (metric, df) in enumerate(data_dict.items()):
        with cols[idx]:
            if not df.empty and len(df) > 0:
                current_val = df.iloc[-1].mean()
                prev_val = df.iloc[-2].mean() if len(df) > 1 else current_val
                change = current_val - prev_val
                change_pct = (change / prev_val * 100) if prev_val != 0 else 0
                
                st.metric(
                    label=metric.replace("_", " "),
                    value=f"{current_val:.2f}",
                    delta=f"{change_pct:.2f}%",
                    delta_color="normal" if metric not in ["Inflation", "Unemployment"] else "inverse"
                )
    
    # Main visualization
    st.header("📊 Economic Indicators Visualization")
    
    tab1, tab2, tab3 = st.tabs(["📈 Time Series", "📋 Data Tables", "📊 Statistics"])
    
    with tab1:
        # Create comparison chart
        fig = create_comparison_chart(data_dict, normalize=normalize_data)
        st.plotly_chart(fig, use_container_width=True)
        
        # Individual charts for each metric
        for metric, df in data_dict.items():
            if df.empty:
                continue
            
            with st.expander(f"📊 {metric.replace('_', ' ')} - Detailed View"):
                fig = go.Figure()
                
                for col in df.columns:
                    fig.add_trace(go.Scatter(
                        x=df.index,
                        y=df[col],
                        name=col,
                        mode='lines+markers',
                        marker=dict(size=4)
                    ))
                
                fig.update_layout(
                    title=f"{METRIC_MAPPING[metric]['description']} ({METRIC_MAPPING[metric]['unit']})",
                    xaxis_title="Date",
                    yaxis_title=METRIC_MAPPING[metric]['unit'],
                    hovermode='x unified',
                    template='plotly_white',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        for metric, df in data_dict.items():
            if not df.empty:
                st.subheader(f"{metric.replace('_', ' ')}")
                st.dataframe(
                    df.style.format("{:.2f}").background_gradient(cmap='RdYlGn', axis=0),
                    use_container_width=True
                )
    
    with tab3:
        if show_statistics:
            st.subheader("📊 Statistical Summary")
            for metric, df in data_dict.items():
                if not df.empty:
                    with st.expander(f"{metric.replace('_', ' ')} Statistics"):
                        stats = calculate_statistics(df)
                        st.dataframe(
                            stats.style.format("{:.2f}").background_gradient(cmap='Blues', axis=1),
                            use_container_width=True
                        )
        
        if show_correlations and len(data_dict) > 1:
            st.subheader("🔗 Correlation Matrix")
            corr_matrix = calculate_correlations(data_dict)
            
            if not corr_matrix.empty:
                fig = px.imshow(
                    corr_matrix,
                    text_auto='.2f',
                    color_continuous_scale='RdBu_r',
                    aspect='auto',
                    title='Correlation Between Economic Indicators'
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Download data
    st.header("💾 Export Data")
    
    for metric, df in data_dict.items():
        if not df.empty:
            csv = df.to_csv()
            st.download_button(
                label=f"Download {metric} Data (CSV)",
                data=csv,
                file_name=f"{metric}_{start_date}_{end_date}.csv",
                mime="text/csv"
            )
    
    # Insights section
    if analysis_context and 'ticker' in analysis_context:
        st.header("💡 Economic Context Insights")
        st.info(f"""
        **Analysis Context**: {analysis_context.get('ticker', 'N/A')}
        
        Understanding these macroeconomic indicators can help contextualize stock performance:
        - **GDP Growth**: Strong GDP growth often correlates with better corporate earnings
        - **Inflation**: High inflation can pressure profit margins and lead to rate hikes
        - **Unemployment**: Low unemployment suggests a healthy economy but can lead to wage inflation
        - **Interest Rates**: Higher rates increase borrowing costs and can compress valuations
        """)

# Standalone execution for testing
if __name__ == "__main__":
    economics_module(None)
