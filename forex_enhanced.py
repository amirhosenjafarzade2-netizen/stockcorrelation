# forex_enhanced.py - Professional Forex Analysis Module
# Advanced currency pair analysis with economic indicators, technical analysis, correlations, and ML predictions
# Enhanced with comprehensive currency coverage, better data handling, and professional features

import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

FRED_API_KEY = "your_fred_api_key_here"  # Get free from https://fred.stlouisfed.org/docs/api/api_key.html
FRED_API_URL = "https://api.stlouisfed.org/fred/series/observations"

# Expanded Currency Coverage - Yahoo Finance supports 150+ currency pairs
MAJOR_CURRENCIES = ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD"]
MINOR_CURRENCIES = ["SEK", "NOK", "DKK", "SGD", "HKD", "KRW", "MXN", "ZAR", "TRY", "BRL", "INR", "CNY"]
EXOTIC_CURRENCIES = ["PLN", "THB", "IDR", "MYR", "PHP", "CZK", "HUF", "ILS", "CLP", "ARS"]

ALL_CURRENCIES = MAJOR_CURRENCIES + MINOR_CURRENCIES + EXOTIC_CURRENCIES

# Currency to Country/Region Mapping (expanded)
CURRENCY_COUNTRIES = {
    # Major Currencies
    "USD": "US", "EUR": "EU", "GBP": "UK", "JPY": "JP", "AUD": "AU", "CAD": "CA", "CHF": "CH", "NZD": "NZ",
    # Minor Currencies
    "SEK": "SE", "NOK": "NO", "DKK": "DK", "SGD": "SG", "HKD": "HK", "KRW": "KR", "MXN": "MX", 
    "ZAR": "ZA", "TRY": "TR", "BRL": "BR", "INR": "IN", "CNY": "CN",
    # Exotic Currencies
    "PLN": "PL", "THB": "TH", "IDR": "ID", "MYR": "MY", "PHP": "PH", "CZK": "CZ", "HUF": "HU",
    "ILS": "IL", "CLP": "CL", "ARS": "AR"
}

# Economic Indicators - Extended Coverage
ECON_INDICATORS = {
    "Inflation": {
        "US": "CPIAUCSL", "EU": "CPHPTT01EZM659N", "UK": "CPALTT01GBM659N", "JP": "CPALTT01JPM659N",
        "AU": "CPALTT01AUM659N", "CA": "CPALTT01CAM659N", "CH": "CPALTT01CHM659N", "CN": "CPALTT01CNM659N",
        "NZ": "CPALTT01NZM659N", "SE": "CPALTT01SEM659N", "NO": "CPALTT01NOM659N", "MX": "CPALTT01MXM659N",
        "IN": "CPALTT01INM659N", "BR": "CPALTT01BRM659N", "ZA": "CPALTT01ZAM659N", "KR": "CPALTT01KRM659N"
    },
    "Interest Rate": {
        "US": "FEDFUNDS", "EU": "ECBDFR", "UK": "BOERUKQ", "JP": "IRSTCI01JPM156N",
        "AU": "IRLTLT01AUM156N", "CA": "IRSTCI01CAM156N", "CH": "IRLTLT01CHM156N", "CN": "IRLTLT01CNM156N",
        "NZ": "IRLTLT01NZM156N", "SE": "IRLTLT01SEM156N", "NO": "IRLTLT01NOM156N", "MX": "IRLTLT01MXM156N",
        "IN": "IRLTLT01INM156N", "BR": "IRLTLT01BRM156N", "ZA": "IRLTLT01ZAM156N", "KR": "IRLTLT01KRM156N"
    },
    "GDP": {
        "US": "GDP", "EU": "CLVMNACSCAB1GQEA19", "UK": "GBRRGDPQDSNAQ", "JP": "JPNRGDPEXP",
        "AU": "AUSGDPEXP", "CA": "NAEXKP01CAQ189S", "CH": "CHEGDP", "CN": "CHNGDPEXP",
        "NZ": "NZLRGDPEXP", "SE": "SWERGDP", "NO": "NORGDP", "MX": "MXNRGDPEXP",
        "IN": "INDNGDP", "BR": "BRAREXP", "ZA": "ZAFRGDPEXP", "KR": "KORRGDPEXP"
    },
    "Trade Balance": {
        "US": "NETEXP", "EU": "XTNTVA01EZM664S", "UK": "XTNTVA01GBM664S", "JP": "XTNTVA01JPM664S",
        "AU": "XTNTVA01AUM664S", "CA": "XTNTVA01CAM664S", "CH": "XTNTVA01CHM664S", "CN": "XTNTVA01CNM664S",
        "NZ": "XTNTVA01NZM664S", "SE": "XTNTVA01SEM664S", "MX": "XTNTVA01MXM664S", "IN": "XTNTVA01INM664S"
    },
    "Unemployment": {
        "US": "UNRATE", "EU": "LRHUTTTTEZM156S", "UK": "LMUNRRTTGBM156S", "JP": "LMUNRRTTJPM156S",
        "AU": "LMUNRRTTAUM156S", "CA": "LMUNRRTTCAM156S", "CH": "LMUNRRTTCHM156S", "NZ": "LMUNRRTTNZM156S",
        "SE": "LMUNRRTTSEM156S", "MX": "LMUNRRTTMXM156S", "KR": "LMUNRRTTKRM156S", "ZA": "LMUNRRTTZAM156S"
    }
}

# Popular Currency Pairs by Category
POPULAR_PAIRS = {
    "Major Pairs": ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"],
    "Minor Pairs": ["EURGBP", "EURJPY", "GBPJPY", "EURCHF", "EURAUD", "EURCAD", "AUDNZD", "AUDJPY"],
    "Exotic Pairs": ["USDTRY", "USDZAR", "USDMXN", "USDBRL", "USDINR", "USDSGD", "USDHKD", "USDKRW"],
    "Commodity Pairs": ["AUDUSD", "NZDUSD", "USDCAD"],  # Commodity-linked currencies
    "Safe Haven": ["USDJPY", "USDCHF", "XAUUSD"]  # Safe haven currencies
}

# ============================================================================
# DATA FETCHING FUNCTIONS
# ============================================================================

@st.cache_data(ttl=3600)
def fetch_fred_data(series_id: str, start_date: str, end_date: str, api_key: str) -> pd.DataFrame:
    """
    Fetch economic data from FRED API with improved error handling.
    """
    if not api_key or api_key == "your_fred_api_key_here":
        return pd.DataFrame()

    params = {
        'series_id': series_id,
        'api_key': api_key,
        'file_type': 'json',
        'observation_start': start_date,
        'observation_end': end_date,
        'sort_order': 'desc'
    }

    try:
        response = requests.get(FRED_API_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if 'observations' in data and data['observations']:
            df = pd.DataFrame(data['observations'])
            df['date'] = pd.to_datetime(df['date'])
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            df = df.dropna(subset=['value']).sort_values('date').set_index('date')
            return df[['value']]
        return pd.DataFrame()
    except requests.exceptions.RequestException as e:
        return pd.DataFrame()
    except Exception as e:
        return pd.DataFrame()

@st.cache_data(ttl=1800)
def fetch_fx_data(pair: str, start_date, end_date, interval='1d') -> pd.DataFrame:
    """
    Fetch FX rate data from yfinance with improved error handling and validation.
    """
    try:
        ticker = f"{pair}=X"
        data = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False)
        
        if data.empty:
            return pd.DataFrame()
        
        # Get all available columns
        df = pd.DataFrame()
        df['Rate'] = data['Close'] if 'Close' in data.columns else data['Adj Close']
        
        # Add additional price data if available
        if 'Open' in data.columns:
            df['Open'] = data['Open']
        if 'High' in data.columns:
            df['High'] = data['High']
        if 'Low' in data.columns:
            df['Low'] = data['Low']
        if 'Volume' in data.columns:
            df['Volume'] = data['Volume']
            
        return df.dropna()
    except Exception as e:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def fetch_multiple_fx_pairs(pairs: List[str], start_date, end_date) -> Dict[str, pd.DataFrame]:
    """
    Batch fetch multiple FX pairs efficiently.
    """
    result = {}
    for pair in pairs:
        df = fetch_fx_data(pair, start_date, end_date)
        if not df.empty:
            result[pair] = df
    return result

# ============================================================================
# TECHNICAL ANALYSIS FUNCTIONS
# ============================================================================

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate comprehensive technical indicators.
    """
    if df.empty or 'Rate' not in df.columns:
        return df
    
    result = df.copy()
    
    # Moving Averages
    result['SMA_20'] = result['Rate'].rolling(window=20).mean()
    result['SMA_50'] = result['Rate'].rolling(window=50).mean()
    result['SMA_200'] = result['Rate'].rolling(window=200).mean()
    
    # Exponential Moving Averages
    result['EMA_12'] = result['Rate'].ewm(span=12, adjust=False).mean()
    result['EMA_26'] = result['Rate'].ewm(span=26, adjust=False).mean()
    
    # MACD
    result['MACD'] = result['EMA_12'] - result['EMA_26']
    result['Signal_Line'] = result['MACD'].ewm(span=9, adjust=False).mean()
    result['MACD_Histogram'] = result['MACD'] - result['Signal_Line']
    
    # RSI (Relative Strength Index)
    delta = result['Rate'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    result['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    result['BB_Middle'] = result['Rate'].rolling(window=20).mean()
    bb_std = result['Rate'].rolling(window=20).std()
    result['BB_Upper'] = result['BB_Middle'] + (bb_std * 2)
    result['BB_Lower'] = result['BB_Middle'] - (bb_std * 2)
    result['BB_Width'] = (result['BB_Upper'] - result['BB_Lower']) / result['BB_Middle']
    
    # Volatility
    result['Returns'] = result['Rate'].pct_change()
    result['Volatility_20'] = result['Returns'].rolling(window=20).std() * np.sqrt(252)
    
    # Average True Range (ATR) - if OHLC data available
    if all(col in result.columns for col in ['High', 'Low', 'Open']):
        high_low = result['High'] - result['Low']
        high_close = np.abs(result['High'] - result['Rate'].shift())
        low_close = np.abs(result['Low'] - result['Rate'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        result['ATR'] = true_range.rolling(window=14).mean()
    
    return result

def calculate_momentum_indicators(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate momentum and trend indicators for current market conditions.
    """
    if df.empty or len(df) < 50:
        return {}
    
    indicators = {}
    
    # Current vs Moving Averages
    current_price = df['Rate'].iloc[-1]
    sma_20 = df['SMA_20'].iloc[-1] if 'SMA_20' in df.columns else None
    sma_50 = df['SMA_50'].iloc[-1] if 'SMA_50' in df.columns else None
    
    if sma_20:
        indicators['Price vs SMA20'] = ((current_price / sma_20) - 1) * 100
    if sma_50:
        indicators['Price vs SMA50'] = ((current_price / sma_50) - 1) * 100
    
    # RSI
    if 'RSI' in df.columns:
        indicators['RSI'] = df['RSI'].iloc[-1]
    
    # MACD Signal
    if 'MACD' in df.columns and 'Signal_Line' in df.columns:
        indicators['MACD'] = df['MACD'].iloc[-1]
        indicators['MACD Signal'] = df['Signal_Line'].iloc[-1]
    
    # Volatility
    if 'Volatility_20' in df.columns:
        indicators['Volatility (20d)'] = df['Volatility_20'].iloc[-1] * 100
    
    # Recent Performance
    if len(df) >= 5:
        indicators['5-Day Change %'] = ((df['Rate'].iloc[-1] / df['Rate'].iloc[-5]) - 1) * 100
    if len(df) >= 20:
        indicators['20-Day Change %'] = ((df['Rate'].iloc[-1] / df['Rate'].iloc[-20]) - 1) * 100
    
    return indicators

# ============================================================================
# STATISTICAL & CORRELATION ANALYSIS
# ============================================================================

def calculate_correlation_matrix(fx_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Calculate correlation matrix for multiple currency pairs.
    """
    if not fx_data:
        return pd.DataFrame()
    
    # Align all data to common dates and calculate returns
    returns_dict = {}
    for pair, df in fx_data.items():
        if not df.empty and 'Rate' in df.columns:
            returns = df['Rate'].pct_change().dropna()
            returns_dict[pair] = returns
    
    if not returns_dict:
        return pd.DataFrame()
    
    # Create aligned dataframe
    returns_df = pd.DataFrame(returns_dict)
    correlation_matrix = returns_df.corr()
    
    return correlation_matrix

def calculate_pair_statistics(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate comprehensive statistics for a currency pair.
    """
    if df.empty or 'Rate' not in df.columns:
        return {}
    
    stats_dict = {}
    
    # Price Statistics
    stats_dict['Current Price'] = df['Rate'].iloc[-1]
    stats_dict['Mean'] = df['Rate'].mean()
    stats_dict['Std Dev'] = df['Rate'].std()
    stats_dict['Min'] = df['Rate'].min()
    stats_dict['Max'] = df['Rate'].max()
    
    # Returns Statistics
    if 'Returns' in df.columns:
        returns = df['Returns'].dropna()
        stats_dict['Avg Daily Return %'] = returns.mean() * 100
        stats_dict['Return Volatility %'] = returns.std() * 100
        stats_dict['Sharpe Ratio (annualized)'] = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
        
        # Skewness and Kurtosis
        stats_dict['Skewness'] = returns.skew()
        stats_dict['Kurtosis'] = returns.kurtosis()
    
    # Drawdown
    cumulative = (1 + df['Returns']).cumprod() if 'Returns' in df.columns else df['Rate'] / df['Rate'].iloc[0]
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    stats_dict['Max Drawdown %'] = drawdown.min() * 100
    
    return stats_dict

# ============================================================================
# FORECASTING & PREDICTIONS
# ============================================================================

def advanced_forecast(series: pd.Series, periods: int = 30, method: str = 'ensemble') -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Advanced forecasting with confidence intervals using ensemble methods.
    Returns: (forecast, lower_bound, upper_bound)
    """
    if len(series) < 30:
        return pd.Series(), pd.Series(), pd.Series()
    
    # Prepare data
    x = np.arange(len(series)).reshape(-1, 1)
    y = series.values
    
    # Remove NaN values
    valid_idx = ~np.isnan(y)
    x = x[valid_idx]
    y = y[valid_idx]
    
    if len(y) < 10:
        return pd.Series(), pd.Series(), pd.Series()
    
    try:
        if method == 'linear':
            model = LinearRegression()
        elif method == 'rf':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:  # ensemble
            # Use weighted average of linear and RF
            lr_model = LinearRegression()
            rf_model = RandomForestRegressor(n_estimators=50, random_state=42)
            lr_model.fit(x, y)
            rf_model.fit(x, y)
            
            future_x = np.arange(len(series), len(series) + periods).reshape(-1, 1)
            lr_pred = lr_model.predict(future_x)
            rf_pred = rf_model.predict(future_x)
            forecast_values = 0.5 * lr_pred + 0.5 * rf_pred
            
            # Calculate confidence intervals based on historical volatility
            residuals = y - 0.5 * lr_model.predict(x) - 0.5 * rf_model.predict(x)
            std_error = np.std(residuals)
            
            last_date = series.index[-1]
            future_dates = [last_date + timedelta(days=i) for i in range(1, periods + 1)]
            
            forecast = pd.Series(forecast_values, index=future_dates)
            lower = pd.Series(forecast_values - 1.96 * std_error, index=future_dates)
            upper = pd.Series(forecast_values + 1.96 * std_error, index=future_dates)
            
            return forecast, lower, upper
    except Exception as e:
        return pd.Series(), pd.Series(), pd.Series()
    
    # Default return for non-ensemble methods
    model.fit(x, y)
    future_x = np.arange(len(series), len(series) + periods).reshape(-1, 1)
    forecast_values = model.predict(future_x)
    
    residuals = y - model.predict(x)
    std_error = np.std(residuals)
    
    last_date = series.index[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, periods + 1)]
    
    forecast = pd.Series(forecast_values, index=future_dates)
    lower = pd.Series(forecast_values - 1.96 * std_error, index=future_dates)
    upper = pd.Series(forecast_values + 1.96 * std_error, index=future_dates)
    
    return forecast, lower, upper

# ============================================================================
# ECONOMIC ANALYSIS FUNCTIONS
# ============================================================================

def calculate_real_rate(nominal_df: pd.DataFrame, inflation_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate real interest rate: Nominal - Inflation."""
    if nominal_df.empty or inflation_df.empty:
        return pd.DataFrame()
    
    # Align data by index
    combined = pd.concat([
        nominal_df.rename(columns={'value': 'Nominal'}), 
        inflation_df.rename(columns={'value': 'Inflation'})
    ], axis=1).dropna()
    
    if combined.empty:
        return pd.DataFrame()
    
    combined['Real Rate'] = combined['Nominal'] - combined['Inflation']
    return combined[['Real Rate']]

def calculate_interest_rate_differential(data_dict: Dict, pair: str) -> Optional[float]:
    """Calculate interest rate differential between two currencies."""
    base_curr, quote_curr = pair[:3], pair[3:]
    
    rate_base_key = f"Interest Rate ({base_curr})"
    rate_quote_key = f"Interest Rate ({quote_curr})"
    
    if rate_base_key in data_dict and rate_quote_key in data_dict:
        base_df = data_dict[rate_base_key]
        quote_df = data_dict[quote_quote_key]
        
        if not base_df.empty and not quote_df.empty:
            return base_df.iloc[-1, 0] - quote_df.iloc[-1, 0]
    
    return None

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_candlestick_chart(df: pd.DataFrame, pair: str) -> go.Figure:
    """Create candlestick chart with volume if available."""
    if df.empty:
        return go.Figure()
    
    has_ohlc = all(col in df.columns for col in ['Open', 'High', 'Low', 'Rate'])
    
    if has_ohlc and len(df) < 500:  # Only show candlesticks for reasonable data sizes
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3]
        )
        
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Rate'],
                name=pair
            ),
            row=1, col=1
        )
        
        if 'Volume' in df.columns:
            fig.add_trace(
                go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color='lightblue'),
                row=2, col=1
            )
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Rate'], mode='lines', name=pair))
    
    fig.update_layout(
        title=f"{pair} Exchange Rate",
        xaxis_title="Date",
        yaxis_title="Rate",
        height=600,
        template='plotly_white',
        hovermode='x unified'
    )
    
    return fig

def create_technical_chart(df: pd.DataFrame, pair: str) -> go.Figure:
    """Create comprehensive technical analysis chart."""
    if df.empty:
        return go.Figure()
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=(f'{pair} Price & Moving Averages', 'RSI', 'MACD')
    )
    
    # Price and Moving Averages
    fig.add_trace(go.Scatter(x=df.index, y=df['Rate'], name='Price', line=dict(color='black')), row=1, col=1)
    
    if 'SMA_20' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20', line=dict(dash='dash')), row=1, col=1)
    if 'SMA_50' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50', line=dict(dash='dash')), row=1, col=1)
    
    # Bollinger Bands
    if all(col in df.columns for col in ['BB_Upper', 'BB_Lower', 'BB_Middle']):
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], name='BB Upper', line=dict(color='gray', dash='dot')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], name='BB Lower', line=dict(color='gray', dash='dot'), fill='tonexty', fillcolor='rgba(128,128,128,0.1)'), row=1, col=1)
    
    # RSI
    if 'RSI' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple')), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    # MACD
    if 'MACD' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='blue')), row=3, col=1)
    if 'Signal_Line' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['Signal_Line'], name='Signal', line=dict(color='red')), row=3, col=1)
    if 'MACD_Histogram' in df.columns:
        colors = ['green' if val >= 0 else 'red' for val in df['MACD_Histogram']]
        fig.add_trace(go.Bar(x=df.index, y=df['MACD_Histogram'], name='Histogram', marker_color=colors), row=3, col=1)
    
    fig.update_layout(height=900, showlegend=True, template='plotly_white')
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    
    return fig

def create_correlation_heatmap(corr_matrix: pd.DataFrame) -> go.Figure:
    """Create correlation heatmap for currency pairs."""
    if corr_matrix.empty:
        return go.Figure()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(
        title="Currency Pair Correlation Matrix (Daily Returns)",
        height=600,
        template='plotly_white'
    )
    
    return fig

def create_indicator_comparison(df1: pd.DataFrame, df2: pd.DataFrame, label1: str, label2: str, title: str) -> go.Figure:
    """Create dual-axis comparison chart for economic indicators."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    if not df1.empty:
        fig.add_trace(
            go.Scatter(x=df1.index, y=df1.iloc[:, 0], name=label1, line=dict(color='blue')),
            secondary_y=False
        )
    
    if not df2.empty:
        fig.add_trace(
            go.Scatter(x=df2.index, y=df2.iloc[:, 0], name=label2, line=dict(color='red')),
            secondary_y=True
        )
    
    fig.update_layout(
        title=title,
        template='plotly_white',
        height=400,
        hovermode='x unified'
    )
    
    fig.update_yaxes(title_text=label1, secondary_y=False)
    fig.update_yaxes(title_text=label2, secondary_y=True)
    
    return fig

def create_forecast_chart(historical: pd.Series, forecast: pd.Series, lower: pd.Series, upper: pd.Series, title: str) -> go.Figure:
    """Create forecast chart with confidence intervals."""
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=historical.index,
        y=historical.values,
        name='Historical',
        line=dict(color='blue')
    ))
    
    # Forecast
    if not forecast.empty:
        fig.add_trace(go.Scatter(
            x=forecast.index,
            y=forecast.values,
            name='Forecast',
            line=dict(color='red', dash='dash')
        ))
        
        # Confidence interval
        if not lower.empty and not upper.empty:
            fig.add_trace(go.Scatter(
                x=list(upper.index) + list(upper.index[::-1]),
                y=list(upper.values) + list(lower.values[::-1]),
                fill='toself',
                fillcolor='rgba(255,0,0,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='95% Confidence',
                showlegend=True
            ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Value",
        height=500,
        template='plotly_white',
        hovermode='x unified'
    )
    
    return fig

# ============================================================================
# MAIN STREAMLIT APPLICATION
# ============================================================================

def forex_module(analysis_context=None):
    """
    Main Forex Analysis Module - Professional Edition
    """
    
    # Page Configuration
    st.set_page_config(page_title="Forex Analysis Pro", page_icon="📈", layout="wide")
    
    # Header
    st.markdown("# 📈 Professional Forex Analysis Platform")
    st.markdown("### Advanced Currency Analysis with Economic Indicators, Technical Analysis & Predictions")
    st.markdown("---")
    
    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key
        api_key = st.text_input(
            "FRED API Key",
            value=FRED_API_KEY,
            type="password",
            help="Get your free API key from https://fred.stlouisfed.org/docs/api/api_key.html"
        )
        
        st.markdown("---")
        
        # Currency Selection Mode
        selection_mode = st.radio(
            "Selection Mode",
            ["Quick Select (Popular Pairs)", "Custom Pairs", "Build Your Own"],
            help="Choose how to select currency pairs"
        )
        
        selected_pairs = []
        
        if selection_mode == "Quick Select (Popular Pairs)":
            category = st.selectbox("Category", list(POPULAR_PAIRS.keys()))
            selected_pairs = st.multiselect(
                "Select Pairs",
                POPULAR_PAIRS[category],
                default=POPULAR_PAIRS[category][:3]
            )
        
        elif selection_mode == "Custom Pairs":
            # Pre-formatted pair selection
            selected_pairs = st.multiselect(
                "Select Currency Pairs",
                [pair for category in POPULAR_PAIRS.values() for pair in category],
                default=["EURUSD", "GBPUSD"]
            )
        
        else:  # Build Your Own
            st.markdown("##### Build Custom Pairs")
            num_pairs = st.number_input("Number of pairs", 1, 10, 2)
            
            for i in range(num_pairs):
                col1, col2 = st.columns(2)
                with col1:
                    base = st.selectbox(f"Base {i+1}", ALL_CURRENCIES, key=f"base_{i}")
                with col2:
                    quote = st.selectbox(f"Quote {i+1}", ALL_CURRENCIES, key=f"quote_{i}", index=1)
                
                if base != quote:
                    selected_pairs.append(f"{base}{quote}")
        
        st.markdown("---")
        
        # Date Range
        st.subheader("📅 Date Range")
        period_preset = st.selectbox(
            "Quick Period",
            ["Custom", "1 Month", "3 Months", "6 Months", "1 Year", "2 Years", "5 Years"],
            index=4
        )
        
        end_date = datetime.today().date()
        
        if period_preset == "Custom":
            start_date = st.date_input("Start Date", value=end_date - timedelta(days=365))
            end_date = st.date_input("End Date", value=end_date)
        else:
            period_map = {
                "1 Month": 30, "3 Months": 90, "6 Months": 180,
                "1 Year": 365, "2 Years": 730, "5 Years": 1825
            }
            start_date = end_date - timedelta(days=period_map[period_preset])
        
        st.markdown("---")
        
        # Analysis Options
        st.subheader("🔧 Analysis Options")
        show_technical = st.checkbox("Technical Analysis", value=True)
        show_economic = st.checkbox("Economic Indicators", value=True)
        show_correlation = st.checkbox("Correlation Analysis", value=True)
        show_forecast = st.checkbox("Forecasting", value=True)
        
        forecast_days = 30
        if show_forecast:
            forecast_days = st.slider("Forecast Days", 7, 90, 30)
    
    # Main Content Area
    if not selected_pairs:
        st.info("👈 Please select at least one currency pair from the sidebar to begin analysis")
        st.markdown("### 💡 Quick Start Guide")
        st.markdown("""
        1. **Select Mode**: Choose how you want to pick currency pairs
        2. **Choose Pairs**: Select one or more currency pairs to analyze
        3. **Set Date Range**: Pick your analysis timeframe
        4. **Configure Analysis**: Enable the analyses you need
        5. **Click Analyze**: Start the comprehensive analysis
        """)
        
        # Show available currencies
        with st.expander("📚 Available Currencies"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**Major Currencies**")
                for curr in MAJOR_CURRENCIES:
                    st.markdown(f"- {curr}")
            with col2:
                st.markdown("**Minor Currencies**")
                for curr in MINOR_CURRENCIES:
                    st.markdown(f"- {curr}")
            with col3:
                st.markdown("**Exotic Currencies**")
                for curr in EXOTIC_CURRENCIES:
                    st.markdown(f"- {curr}")
        
        return
    
    # Analysis Button
    st.markdown("---")
    analyze_button = st.button("🚀 Run Comprehensive Analysis", type="primary", use_container_width=True)
    
    if not analyze_button:
        st.info("👆 Click the button above to start the analysis")
        return
    
    # ========================================================================
    # DATA LOADING & PROCESSING
    # ========================================================================
    
    st.markdown("---")
    st.header("📊 Loading Data...")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Fetch FX Data
    status_text.text("Fetching currency pair data...")
    fx_data = fetch_multiple_fx_pairs(selected_pairs, start_date, end_date)
    progress_bar.progress(0.3)
    
    if not fx_data:
        st.error("❌ Failed to fetch FX data. Please check your currency pairs and try again.")
        return
    
    # Calculate Technical Indicators
    if show_technical:
        status_text.text("Calculating technical indicators...")
        for pair in fx_data:
            fx_data[pair] = calculate_technical_indicators(fx_data[pair])
        progress_bar.progress(0.5)
    
    # Fetch Economic Data
    econ_data = {}
    if show_economic and api_key and api_key != "your_fred_api_key_here":
        status_text.text("Fetching economic indicators...")
        
        for i, pair in enumerate(selected_pairs):
            base_curr, quote_curr = pair[:3], pair[3:]
            base_country = CURRENCY_COUNTRIES.get(base_curr)
            quote_country = CURRENCY_COUNTRIES.get(quote_curr)
            
            if not base_country or not quote_country:
                continue
            
            econ_data[pair] = {}
            
            for ind_name, codes in ECON_INDICATORS.items():
                base_code = codes.get(base_country)
                quote_code = codes.get(quote_country)
                
                if base_code:
                    df = fetch_fred_data(base_code, start_date.strftime('%Y-%m-%d'), 
                                        end_date.strftime('%Y-%m-%d'), api_key)
                    if not df.empty:
                        econ_data[pair][f"{ind_name} ({base_curr})"] = df
                
                if quote_code:
                    df = fetch_fred_data(quote_code, start_date.strftime('%Y-%m-%d'), 
                                        end_date.strftime('%Y-%m-%d'), api_key)
                    if not df.empty:
                        econ_data[pair][f"{ind_name} ({quote_curr})"] = df
            
            # Calculate Real Interest Rates
            if f"Interest Rate ({base_curr})" in econ_data[pair] and f"Inflation ({base_curr})" in econ_data[pair]:
                real_rate = calculate_real_rate(
                    econ_data[pair][f"Interest Rate ({base_curr})"],
                    econ_data[pair][f"Inflation ({base_curr})"]
                )
                if not real_rate.empty:
                    econ_data[pair][f"Real Interest Rate ({base_curr})"] = real_rate
            
            if f"Interest Rate ({quote_curr})" in econ_data[pair] and f"Inflation ({quote_curr})" in econ_data[pair]:
                real_rate = calculate_real_rate(
                    econ_data[pair][f"Interest Rate ({quote_curr})"],
                    econ_data[pair][f"Inflation ({quote_curr})"]
                )
                if not real_rate.empty:
                    econ_data[pair][f"Real Interest Rate ({quote_curr})"] = real_rate
            
            progress_bar.progress(0.5 + (0.3 * (i + 1) / len(selected_pairs)))
    
    progress_bar.progress(1.0)
    status_text.text("✅ Data loading complete!")
    
    # ========================================================================
    # ANALYSIS TABS
    # ========================================================================
    
    st.markdown("---")
    
    tabs = ["📊 Overview", "📈 Technical Analysis", "🌍 Economic Indicators", 
            "🔮 Forecasting", "📊 Statistics", "🔗 Correlations", "💾 Export"]
    
    tab_objects = st.tabs(tabs)
    
    # ========================================================================
    # TAB 1: OVERVIEW
    # ========================================================================
    
    with tab_objects[0]:
        st.header("📊 Market Overview")
        
        # Summary Cards
        cols = st.columns(min(len(selected_pairs), 4))
        for idx, pair in enumerate(selected_pairs):
            if pair not in fx_data:
                continue
            
            df = fx_data[pair]
            current_price = df['Rate'].iloc[-1]
            prev_price = df['Rate'].iloc[0]
            change_pct = ((current_price / prev_price) - 1) * 100
            
            with cols[idx % 4]:
                delta_color = "normal" if change_pct >= 0 else "inverse"
                st.metric(
                    label=pair,
                    value=f"{current_price:.4f}",
                    delta=f"{change_pct:+.2f}%"
                )
        
        st.markdown("---")
        
        # Price Charts
        st.subheader("Exchange Rate Charts")
        
        for pair in selected_pairs:
            if pair in fx_data:
                with st.expander(f"📈 {pair}", expanded=True):
                    fig = create_candlestick_chart(fx_data[pair], pair)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Quick Stats
                    col1, col2, col3, col4 = st.columns(4)
                    df = fx_data[pair]
                    
                    with col1:
                        st.metric("Current", f"{df['Rate'].iloc[-1]:.4f}")
                    with col2:
                        change = df['Rate'].iloc[-1] - df['Rate'].iloc[-20] if len(df) >= 20 else 0
                        st.metric("20D Change", f"{change:.4f}")
                    with col3:
                        vol = df['Volatility_20'].iloc[-1] * 100 if 'Volatility_20' in df.columns else 0
                        st.metric("Volatility", f"{vol:.2f}%")
                    with col4:
                        rsi = df['RSI'].iloc[-1] if 'RSI' in df.columns else 0
                        st.metric("RSI", f"{rsi:.1f}")
    
    # ========================================================================
    # TAB 2: TECHNICAL ANALYSIS
    # ========================================================================
    
    with tab_objects[1]:
        if show_technical:
            st.header("📈 Technical Analysis")
            
            for pair in selected_pairs:
                if pair not in fx_data:
                    continue
                
                st.subheader(f"{pair} Technical Indicators")
                
                df = fx_data[pair]
                
                # Technical Chart
                fig = create_technical_chart(df, pair)
                st.plotly_chart(fig, use_container_width=True)
                
                # Momentum Indicators
                st.markdown("#### Current Momentum & Trend Indicators")
                momentum = calculate_momentum_indicators(df)
                
                if momentum:
                    col1, col2, col3 = st.columns(3)
                    items = list(momentum.items())
                    third = len(items) // 3 + 1
                    
                    with col1:
                        for key, value in items[:third]:
                            st.metric(key, f"{value:.2f}")
                    with col2:
                        for key, value in items[third:2*third]:
                            st.metric(key, f"{value:.2f}")
                    with col3:
                        for key, value in items[2*third:]:
                            st.metric(key, f"{value:.2f}")
                
                # Trading Signals
                st.markdown("#### Trading Signals")
                signals = []
                
                if 'RSI' in df.columns:
                    rsi = df['RSI'].iloc[-1]
                    if rsi > 70:
                        signals.append("🔴 RSI Overbought (>70) - Potential reversal signal")
                    elif rsi < 30:
                        signals.append("🟢 RSI Oversold (<30) - Potential buying opportunity")
                    else:
                        signals.append("🟡 RSI Neutral (30-70) - No strong signal")
                
                if all(col in df.columns for col in ['MACD', 'Signal_Line']):
                    if df['MACD'].iloc[-1] > df['Signal_Line'].iloc[-1] and df['MACD'].iloc[-2] <= df['Signal_Line'].iloc[-2]:
                        signals.append("🟢 MACD Bullish Crossover - Buy signal")
                    elif df['MACD'].iloc[-1] < df['Signal_Line'].iloc[-1] and df['MACD'].iloc[-2] >= df['Signal_Line'].iloc[-2]:
                        signals.append("🔴 MACD Bearish Crossover - Sell signal")
                
                if signals:
                    for signal in signals:
                        st.markdown(signal)
                else:
                    st.info("No strong trading signals at this time")
                
                st.markdown("---")
        else:
            st.info("Technical analysis is disabled. Enable it in the sidebar to view charts and indicators.")
    
    # ========================================================================
    # TAB 3: ECONOMIC INDICATORS
    # ========================================================================
    
    with tab_objects[2]:
        if show_economic and econ_data:
            st.header("🌍 Economic Indicators Analysis")
            
            for pair in selected_pairs:
                if pair not in econ_data or not econ_data[pair]:
                    continue
                
                st.subheader(f"{pair} Economic Comparison")
                
                base_curr, quote_curr = pair[:3], pair[3:]
                
                # Create comparison charts for each indicator
                for indicator in ["Inflation", "Interest Rate", "Real Interest Rate", "GDP", "Trade Balance", "Unemployment"]:
                    base_key = f"{indicator} ({base_curr})"
                    quote_key = f"{indicator} ({quote_curr})"
                    
                    if base_key in econ_data[pair] and quote_key in econ_data[pair]:
                        with st.expander(f"📊 {indicator} Comparison", expanded=False):
                            fig = create_indicator_comparison(
                                econ_data[pair][base_key],
                                econ_data[pair][quote_key],
                                base_curr,
                                quote_curr,
                                f"{indicator}: {base_curr} vs {quote_curr}"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Latest values
                            col1, col2 = st.columns(2)
                            with col1:
                                latest = econ_data[pair][base_key].iloc[-1, 0]
                                st.metric(f"{base_curr} Latest", f"{latest:.2f}")
                            with col2:
                                latest = econ_data[pair][quote_key].iloc[-1, 0]
                                st.metric(f"{quote_curr} Latest", f"{latest:.2f}")
                
                # Key Economic Insights
                st.markdown("#### 💡 Key Economic Insights")
                
                insights = []
                
                # Interest Rate Differential
                if f"Interest Rate ({base_curr})" in econ_data[pair] and f"Interest Rate ({quote_curr})" in econ_data[pair]:
                    base_rate = econ_data[pair][f"Interest Rate ({base_curr})"].iloc[-1, 0]
                    quote_rate = econ_data[pair][f"Interest Rate ({quote_curr})"].iloc[-1, 0]
                    diff = base_rate - quote_rate
                    insights.append(f"**Interest Rate Differential**: {diff:+.2f}% ({base_curr}: {base_rate:.2f}% | {quote_curr}: {quote_rate:.2f}%)")
                    
                    if abs(diff) > 1:
                        if diff > 0:
                            insights.append(f"→ Higher rates in {base_curr} may support currency strength")
                        else:
                            insights.append(f"→ Higher rates in {quote_curr} may support currency strength")
                
                # Real Rate Differential
                if f"Real Interest Rate ({base_curr})" in econ_data[pair] and f"Real Interest Rate ({quote_curr})" in econ_data[pair]:
                    base_real = econ_data[pair][f"Real Interest Rate ({base_curr})"].iloc[-1, 0]
                    quote_real = econ_data[pair][f"Real Interest Rate ({quote_curr})"].iloc[-1, 0]
                    real_diff = base_real - quote_real
                    insights.append(f"**Real Interest Rate Differential**: {real_diff:+.2f}% (Inflation-adjusted)")
                
                # Inflation Comparison
                if f"Inflation ({base_curr})" in econ_data[pair] and f"Inflation ({quote_curr})" in econ_data[pair]:
                    base_inf = econ_data[pair][f"Inflation ({base_curr})"].iloc[-1, 0]
                    quote_inf = econ_data[pair][f"Inflation ({quote_curr})"].iloc[-1, 0]
                    insights.append(f"**Inflation**: {base_curr} {base_inf:.2f}% | {quote_curr} {quote_inf:.2f}%")
                
                for insight in insights:
                    st.markdown(insight)
                
                st.markdown("---")
        else:
            if not show_economic:
                st.info("Economic indicators analysis is disabled. Enable it in the sidebar.")
            else:
                st.warning("⚠️ Economic data requires a valid FRED API key. Please enter your API key in the sidebar.")
                st.markdown("Get your free API key at: https://fred.stlouisfed.org/docs/api/api_key.html")
    
    # ========================================================================
    # TAB 4: FORECASTING
    # ========================================================================
    
    with tab_objects[3]:
        if show_forecast:
            st.header("🔮 Price Forecasting & Predictions")
            
            forecast_method = st.selectbox(
                "Forecasting Method",
                ["ensemble", "linear", "rf"],
                format_func=lambda x: {"ensemble": "Ensemble (Linear + Random Forest)", 
                                      "linear": "Linear Regression", 
                                      "rf": "Random Forest"}[x]
            )
            
            for pair in selected_pairs:
                if pair not in fx_data:
                    continue
                
                st.subheader(f"{pair} Forecast")
                
                df = fx_data[pair]
                
                # Generate forecast
                with st.spinner(f"Generating {forecast_days}-day forecast for {pair}..."):
                    forecast, lower, upper = advanced_forecast(
                        df['Rate'], 
                        periods=forecast_days, 
                        method=forecast_method
                    )
                
                if not forecast.empty:
                    # Create forecast chart
                    fig = create_forecast_chart(df['Rate'], forecast, lower, upper, f"{pair} Price Forecast")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Forecast summary
                    col1, col2, col3, col4 = st.columns(4)
                    
                    current = df['Rate'].iloc[-1]
                    forecast_end = forecast.iloc[-1]
                    change = ((forecast_end / current) - 1) * 100
                    
                    with col1:
                        st.metric("Current Price", f"{current:.4f}")
                    with col2:
                        st.metric(f"{forecast_days}D Forecast", f"{forecast_end:.4f}")
                    with col3:
                        st.metric("Expected Change", f"{change:+.2f}%")
                    with col4:
                        trend = "Bullish 📈" if change > 0 else "Bearish 📉" if change < 0 else "Neutral →"
                        st.metric("Trend", trend)
                    
                    # Confidence interval info
                    st.info(f"📊 95% Confidence Interval: [{lower.iloc[-1]:.4f}, {upper.iloc[-1]:.4f}]")
                else:
                    st.warning(f"Unable to generate forecast for {pair} - insufficient data")
                
                st.markdown("---")
            
            st.markdown("""
            **Disclaimer**: Forecasts are based on historical data and statistical models. 
            They should not be used as the sole basis for trading decisions. 
            Past performance does not guarantee future results.
            """)
        else:
            st.info("Forecasting is disabled. Enable it in the sidebar to view predictions.")
    
    # ========================================================================
    # TAB 5: STATISTICS
    # ========================================================================
    
    with tab_objects[4]:
        st.header("📊 Statistical Analysis")
        
        for pair in selected_pairs:
            if pair not in fx_data:
                continue
            
            st.subheader(f"{pair} Statistics")
            
            df = fx_data[pair]
            stats = calculate_pair_statistics(df)
            
            if stats:
                # Display statistics in columns
                col1, col2, col3 = st.columns(3)
                
                stats_items = list(stats.items())
                third = len(stats_items) // 3 + 1
                
                with col1:
                    st.markdown("##### Price Statistics")
                    for key, value in [item for item in stats_items if any(x in key for x in ['Price', 'Mean', 'Std', 'Min', 'Max'])]:
                        st.metric(key, f"{value:.4f}")
                
                with col2:
                    st.markdown("##### Return Statistics")
                    for key, value in [item for item in stats_items if any(x in key for x in ['Return', 'Volatility', 'Sharpe'])]:
                        st.metric(key, f"{value:.4f}")
                
                with col3:
                    st.markdown("##### Risk Metrics")
                    for key, value in [item for item in stats_items if any(x in key for x in ['Drawdown', 'Skewness', 'Kurtosis'])]:
                        st.metric(key, f"{value:.4f}")
                
                # Distribution chart
                if 'Returns' in df.columns:
                    st.markdown("##### Return Distribution")
                    returns = df['Returns'].dropna() * 100
                    
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(x=returns, nbinsx=50, name='Returns'))
                    fig.update_layout(
                        title=f"{pair} Daily Return Distribution (%)",
                        xaxis_title="Daily Return (%)",
                        yaxis_title="Frequency",
                        template='plotly_white',
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
    
    # ========================================================================
    # TAB 6: CORRELATIONS
    # ========================================================================
    
    with tab_objects[5]:
        if show_correlation and len(selected_pairs) > 1:
            st.header("🔗 Correlation Analysis")
            
            st.markdown("""
            Correlation analysis shows how currency pairs move relative to each other.
            - **High positive correlation (>0.7)**: Pairs tend to move together
            - **High negative correlation (<-0.7)**: Pairs tend to move in opposite directions
            - **Low correlation (around 0)**: Pairs move independently
            """)
            
            # Calculate correlation matrix
            corr_matrix = calculate_correlation_matrix(fx_data)
            
            if not corr_matrix.empty:
                # Heatmap
                st.subheader("Correlation Heatmap")
                fig = create_correlation_heatmap(corr_matrix)
                st.plotly_chart(fig, use_container_width=True)
                
                # Top correlations
                st.subheader("Notable Correlations")
                
                # Get upper triangle of correlation matrix
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
                corr_pairs = []
                
                for i in range(len(corr_matrix)):
                    for j in range(i+1, len(corr_matrix)):
                        corr_pairs.append((
                            corr_matrix.index[i],
                            corr_matrix.columns[j],
                            corr_matrix.iloc[i, j]
                        ))
                
                # Sort by absolute correlation
                corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Strongest Positive Correlations**")
                    for pair1, pair2, corr in corr_pairs[:5]:
                        if corr > 0:
                            st.markdown(f"- {pair1} ↔ {pair2}: **{corr:.3f}**")
                
                with col2:
                    st.markdown("**Strongest Negative Correlations**")
                    negative_corrs = [x for x in corr_pairs if x[2] < 0]
                    for pair1, pair2, corr in negative_corrs[:5]:
                        st.markdown(f"- {pair1} ↔ {pair2}: **{corr:.3f}**")
                
                # Correlation over time
                st.subheader("Rolling Correlation (30-Day)")
                
                if len(selected_pairs) >= 2:
                    pair1, pair2 = selected_pairs[0], selected_pairs[1]
                    
                    if pair1 in fx_data and pair2 in fx_data:
                        df1 = fx_data[pair1]['Rate'].pct_change()
                        df2 = fx_data[pair2]['Rate'].pct_change()
                        
                        # Align data
                        combined = pd.concat([df1, df2], axis=1).dropna()
                        rolling_corr = combined.iloc[:, 0].rolling(window=30).corr(combined.iloc[:, 1])
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=rolling_corr.index, y=rolling_corr, name='Rolling Correlation'))
                        fig.add_hline(y=0, line_dash="dash", line_color="gray")
                        fig.update_layout(
                            title=f"30-Day Rolling Correlation: {pair1} vs {pair2}",
                            xaxis_title="Date",
                            yaxis_title="Correlation",
                            template='plotly_white',
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Unable to calculate correlations - insufficient data")
        else:
            if not show_correlation:
                st.info("Correlation analysis is disabled. Enable it in the sidebar.")
            else:
                st.info("Please select at least 2 currency pairs to perform correlation analysis.")
    
    # ========================================================================
    # TAB 7: EXPORT
    # ========================================================================
    
    with tab_objects[6]:
        st.header("💾 Export Data")
        
        st.markdown("""
        Download the analysis data in CSV format for further analysis in Excel, Python, or other tools.
        """)
        
        # Export FX Data
        st.subheader("📈 Currency Pair Data")
        for pair in selected_pairs:
            if pair in fx_data:
                df = fx_data[pair]
                csv = df.to_csv()
                st.download_button(
                    label=f"📥 Download {pair} Data",
                    data=csv,
                    file_name=f"{pair}_fx_data_{start_date}_{end_date}.csv",
                    mime="text/csv"
                )
        
        # Export Economic Data
        if econ_data:
            st.subheader("🌍 Economic Indicators")
            for pair, indicators in econ_data.items():
                if indicators:
                    with st.expander(f"{pair} Economic Data"):
                        for ind_name, df in indicators.items():
                            if not df.empty:
                                csv = df.to_csv()
                                safe_name = ind_name.replace(" ", "_").replace("(", "").replace(")", "")
                                st.download_button(
                                    label=f"📥 {ind_name}",
                                    data=csv,
                                    file_name=f"{pair}_{safe_name}_{start_date}_{end_date}.csv",
                                    mime="text/csv",
                                    key=f"{pair}_{safe_name}"
                                )
        
        # Export Correlation Matrix
        if show_correlation and len(selected_pairs) > 1:
            st.subheader("🔗 Correlation Matrix")
            corr_matrix = calculate_correlation_matrix(fx_data)
            if not corr_matrix.empty:
                csv = corr_matrix.to_csv()
                st.download_button(
                    label="📥 Download Correlation Matrix",
                    data=csv,
                    file_name=f"correlation_matrix_{start_date}_{end_date}.csv",
                    mime="text/csv"
                )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray; padding: 20px;'>
        <p>📈 Professional Forex Analysis Platform | Data sources: Yahoo Finance, FRED</p>
        <p style='font-size: 0.8em;'>Disclaimer: This tool is for informational purposes only. 
        Always conduct your own research and consult with financial professionals before making trading decisions.</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    forex_module()
