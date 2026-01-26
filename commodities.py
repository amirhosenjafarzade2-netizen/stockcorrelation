# commodities.py - Advanced Commodities Analysis Module (FIXED)
# Integrates FRED prices, EIA inventory/production, yfinance fallback, with advanced analytics
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
from scipy import stats
from sklearn.linear_model import LinearRegression
import yfinance as yf

# API Configuration
FRED_API_KEY = "your_fred_api_key_here"
EIA_API_KEY = "your_eia_api_key_here"
FRED_API_URL = "https://api.stlouisfed.org/fred/series/observations"
EIA_API_URL = "https://api.eia.gov/v2"

# Commodity Data Sources (FIXED with yfinance tickers)
COMMODITIES = {
    "Energy": {
        "WTI_Crude": {
            "fred_code": "DCOILWTICO",
            "yf_ticker": "CL=F",
            "unit": "$/Barrel",
            "description": "WTI Crude Oil Spot Price",
            "inventory_link": "Crude_Oil_Stock",
            "production_cost": 45
        },
        "Brent_Crude": {
            "fred_code": "DCOILBRENTEU",
            "yf_ticker": "BZ=F",
            "unit": "$/Barrel",
            "description": "Brent Crude Oil Spot Price",
            "production_cost": 50
        },
        "Natural_Gas": {
            "fred_code": "DHHNGSP",
            "yf_ticker": "NG=F",
            "unit": "$/MMBtu",
            "description": "Henry Hub Natural Gas Spot Price",
            "inventory_link": "Natural_Gas_Storage",
            "production_cost": 2.5
        },
        "Gasoline": {
            "fred_code": "GASREGW",
            "yf_ticker": "RB=F",
            "unit": "$/Gallon",
            "description": "US Regular Gasoline Price",
            "inventory_link": "Gasoline_Stock"
        },
        "Heating_Oil": {
            "fred_code": "DHOILNYH",
            "yf_ticker": "HO=F",
            "unit": "$/Gallon",
            "description": "NY Harbor Heating Oil Spot Price"
        }
    },
    "Metals": {
        "Gold": {
            "fred_code": "GOLDAMGBD228NLBM",
            "yf_ticker": "GC=F",
            "unit": "$/Troy Oz",
            "description": "Gold Fixing Price (London)",
            "production_cost": 1200
        },
        "Silver": {
            "fred_code": "SLVPRUSD",
            "yf_ticker": "SI=F",
            "unit": "$/Troy Oz",
            "description": "Silver Price (London)",
            "production_cost": 18
        },
        "Copper": {
            "fred_code": "PCOPPUSDM",
            "yf_ticker": "HG=F",
            "unit": "$/Pound",
            "description": "Global Copper Price",
            "production_cost": 3.0
        },
        "Platinum": {
            "fred_code": "PLATINUMLBMA",
            "yf_ticker": "PL=F",
            "unit": "$/Troy Oz",
            "description": "Platinum Price (London)"
        },
        "Palladium": {
            "fred_code": "PALLADIUMLBMA",
            "yf_ticker": "PA=F",
            "unit": "$/Troy Oz",
            "description": "Palladium Price (London)"
        }
    },
    "Agriculture": {
        "Wheat": {
            "fred_code": "PWHEAMTUSDM",
            "yf_ticker": "ZW=F",
            "unit": "$/Bushel",
            "description": "Global Wheat Price"
        },
        "Corn": {
            "fred_code": "PMAIZMTUSDM",
            "yf_ticker": "ZC=F",
            "unit": "$/Bushel",
            "description": "Global Corn Price"
        },
        "Soybeans": {
            "fred_code": "PSOYBUSDQ",
            "yf_ticker": "ZS=F",
            "unit": "$/Bushel",
            "description": "Global Soybean Price"
        },
        "Coffee": {
            "fred_code": "PCOFFOTMUSDM",
            "yf_ticker": "KC=F",
            "unit": "cents/lb",
            "description": "Global Coffee Price"
        },
        "Cotton": {
            "fred_code": "PCOTTINDUSDM",
            "yf_ticker": "CT=F",
            "unit": "cents/lb",
            "description": "Global Cotton Price"
        }
    }
}

EIA_INVENTORY_SERIES = {
    "Crude_Oil_Stock": {
        "series": "petroleum/stoc/wstk",
        "filter": {"product": "EPC0", "duoarea": "NUS"},
        "unit": "Thousand Barrels",
        "description": "US Crude Oil Stocks (Excluding SPR)"
    },
    "Gasoline_Stock": {
        "series": "petroleum/stoc/wstk",
        "filter": {"product": "EPM0", "duoarea": "NUS"},
        "unit": "Thousand Barrels",
        "description": "US Total Gasoline Stocks"
    },
    "Distillate_Stock": {
        "series": "petroleum/stoc/wstk",
        "filter": {"product": "EPD0", "duoarea": "NUS"},
        "unit": "Thousand Barrels",
        "description": "US Distillate Fuel Oil Stocks"
    },
    "Natural_Gas_Storage": {
        "series": "natural-gas/stor/wkly",
        "unit": "Billion Cubic Feet",
        "description": "Natural Gas Underground Storage"
    },
    "Crude_Production": {
        "series": "petroleum/crd/crpdn",
        "filter": {"duoarea": "NUS"},
        "unit": "Thousand Barrels/Day",
        "description": "US Crude Oil Production"
    },
    "Refinery_Utilization": {
        "series": "petroleum/pnp/unc",
        "unit": "Percent",
        "description": "US Refinery Utilization Rate"
    }
}

# Data fetching functions
@st.cache_data(ttl=3600)
def get_fred_data_csv(series_id: str, start_date: str, end_date: str) -> pd.DataFrame:
    try:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        df = pd.read_csv(StringIO(response.text), index_col=0, parse_dates=True)
        
        if len(df.columns) == 0:
            return pd.DataFrame()
        
        df.columns = [series_id]
        
        try:
            df = df.loc[start_date:end_date]
        except:
            pass
        
        df[series_id] = pd.to_numeric(df[series_id], errors='coerce')
        df = df.dropna()
        
        if len(df) == 0:
            return pd.DataFrame()
            
        return df
    except Exception as e:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_fred_data_api(series_id: str, start_date: str, end_date: str, api_key: str) -> pd.DataFrame:
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
def get_yfinance_price(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch commodity prices from yfinance (fallback for FRED)."""
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
        if not data.empty and "Close" in data.columns:
            df = data[["Close"]].copy()
            df.columns = [ticker]
            return df.dropna()
        return pd.DataFrame()
    except Exception as e:
        return pd.DataFrame()

@st.cache_data(ttl=1800)
def get_eia_data(series_path: str, api_key: str, filters: Optional[Dict] = None,
                 start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
    if not api_key or api_key == "your_eia_api_key_here":
        return pd.DataFrame()
    try:
        url = f"{EIA_API_URL}/{series_path}/data/"
        params = {
            'api_key': api_key,
            'frequency': 'weekly',
            'data[]': 'value',
            'sort[0][column]': 'period',
            'sort[0][direction]': 'asc'
        }
        if filters:
            for key, value in filters.items():
                params[f'facets[{key}][]'] = value
        if start_date:
            params['start'] = start_date
        if end_date:
            params['end'] = end_date
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        if 'response' in data and 'data' in data['response']:
            df = pd.DataFrame(data['response']['data'])
            if 'period' in df.columns and 'value' in df.columns:
                df['period'] = pd.to_datetime(df['period'])
                df = df.set_index('period')[['value']]
                df['value'] = pd.to_numeric(df['value'], errors='coerce')
                return df.dropna()
        return pd.DataFrame()
    except Exception as e:
        return pd.DataFrame()

# ============= ADVANCED ANALYTICS FUNCTIONS =============
def calculate_crack_spread(crude_price: pd.DataFrame, gasoline_price: pd.DataFrame,
                          heating_oil_price: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Calculate 3-2-1 crack spread (refinery margin indicator)."""
    if crude_price.empty or gasoline_price.empty:
        return pd.DataFrame()
    
    combined = pd.concat([crude_price, gasoline_price], axis=1, join='inner')
    
    if heating_oil_price is not None and not heating_oil_price.empty:
        combined = pd.concat([combined, heating_oil_price], axis=1, join='inner')
        spread = ((2 * combined.iloc[:, 1] * 42) + (1 * combined.iloc[:, 2] * 42)) / 3 - combined.iloc[:, 0]
    else:
        spread = (2 * combined.iloc[:, 1] * 42) - combined.iloc[:, 0]
    
    return pd.DataFrame({'Crack_Spread': spread})

def calculate_inventory_coverage(inventory: pd.DataFrame, production: pd.DataFrame) -> pd.DataFrame:
    """Calculate days of supply coverage."""
    if inventory.empty or production.empty:
        return pd.DataFrame()
    
    combined = pd.concat([inventory, production], axis=1, join='inner')
    if len(combined) == 0:
        return pd.DataFrame()
    
    coverage = combined.iloc[:, 0] / combined.iloc[:, 1]
    return pd.DataFrame({'Days_Coverage': coverage})

def calculate_price_to_cost_ratio(price: pd.DataFrame, production_cost: float) -> pd.DataFrame:
    """Calculate price to production cost ratio."""
    if price.empty or production_cost is None:
        return pd.DataFrame()
    
    ratio = price.iloc[:, 0] / production_cost
    return pd.DataFrame({'Price_to_Cost': ratio})

def calculate_contango_backwardation(price: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    """Estimate market structure using price momentum."""
    if price.empty or len(price) < window:
        return pd.DataFrame()
    
    rolling_slope = price.iloc[:, 0].rolling(window=window).apply(
        lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) == window else np.nan
    )
    
    return pd.DataFrame({'Market_Structure': rolling_slope})

def calculate_seasonal_pattern(data: pd.DataFrame, period: int = 52) -> pd.DataFrame:
    """Extract seasonal pattern using moving average decomposition."""
    if data.empty or len(data) < period * 2:
        return pd.DataFrame()
    
    trend = data.iloc[:, 0].rolling(window=period, center=True).mean()
    detrended = data.iloc[:, 0] - trend
    seasonal = detrended.groupby(detrended.index.isocalendar().week).transform('mean')
    
    return pd.DataFrame({
        'Original': data.iloc[:, 0],
        'Trend': trend,
        'Seasonal': seasonal,
        'Residual': data.iloc[:, 0] - trend - seasonal
    })

def calculate_volatility_metrics(price: pd.DataFrame) -> Dict:
    """Calculate various volatility metrics."""
    if price.empty or len(price) < 30:
        return {}
    
    returns = price.iloc[:, 0].pct_change().dropna()
    
    return {
        'Current_Vol_30d': returns.tail(30).std() * np.sqrt(252) * 100,
        'Current_Vol_90d': returns.tail(90).std() * np.sqrt(252) * 100,
        'Avg_Vol_1Y': returns.std() * np.sqrt(252) * 100,
        'Max_Drawdown': ((price.iloc[:, 0] / price.iloc[:, 0].cummax()) - 1).min() * 100,
        'Sharpe_Ratio': (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
    }

def calculate_z_score(data: pd.DataFrame, window: int = 90) -> pd.DataFrame:
    """Calculate rolling z-score."""
    if data.empty or len(data) < window:
        return pd.DataFrame()
    
    rolling_mean = data.iloc[:, 0].rolling(window=window).mean()
    rolling_std = data.iloc[:, 0].rolling(window=window).std()
    
    z_score = (data.iloc[:, 0] - rolling_mean) / rolling_std
    
    return pd.DataFrame({'Z_Score': z_score})

def calculate_supply_demand_balance(price: pd.DataFrame, inventory: pd.DataFrame) -> pd.DataFrame:
    """Advanced supply/demand indicator."""
    if price.empty or inventory.empty:
        return pd.DataFrame()
    
    combined = pd.concat([price, inventory], axis=1, join='inner')
    if len(combined) < 50:
        return pd.DataFrame()
    
    inv_change = combined.iloc[:, 1].diff()
    price_change = combined.iloc[:, 0].pct_change() * 100
    
    sd_indicator = pd.DataFrame({
        'Price_Change_%': price_change,
        'Inventory_Change': inv_change,
        'Demand_Signal': -inv_change / inv_change.rolling(20).std()
    })
    
    return sd_indicator

def perform_regression_analysis(y: pd.DataFrame, x: pd.DataFrame, name: str = "Factor") -> Dict:
    """Perform linear regression and return statistics."""
    if y.empty or x.empty:
        return {}
    
    combined = pd.concat([y, x], axis=1, join='inner').dropna()
    if len(combined) < 30:
        return {}
    
    Y = combined.iloc[:, 0].values.reshape(-1, 1)
    X = combined.iloc[:, 1].values.reshape(-1, 1)
    
    model = LinearRegression()
    model.fit(X, Y)
    
    predictions = model.predict(X)
    r_squared = model.score(X, Y)
    correlation = np.corrcoef(X.flatten(), Y.flatten())[0, 1]
    
    return {
        'Factor': name,
        'Coefficient': model.coef_[0][0],
        'Intercept': model.intercept_[0],
        'R_Squared': r_squared,
        'Correlation': correlation
    }

def create_advanced_dashboard(price_data: Dict, inventory_data: Dict,
                             analytics: Dict, commodity_info: Dict) -> go.Figure:
    """Create comprehensive multi-panel dashboard."""
    panels = []
    if price_data:
        panels.append("Prices")
    if inventory_data:
        panels.append("Inventory")
    if 'crack_spread' in analytics:
        panels.append("Crack Spread")
    if 'z_score' in analytics and not analytics['z_score'].empty:
        panels.append("Volatility")
    
    n_panels = len(panels)
    if n_panels == 0:
        return None
    
    fig = make_subplots(
        rows=n_panels,
        cols=1,
        subplot_titles=panels,
        vertical_spacing=0.12,
        row_heights=[1/n_panels] * n_panels
    )
    
    colors = px.colors.qualitative.Set2
    row = 1
    
    # Panel 1: Prices
    if "Prices" in panels:
        for idx, (name, df) in enumerate(price_data.items()):
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df.iloc[:, 0],
                    name=name.replace('_', ' '),
                    line=dict(color=colors[idx % len(colors)], width=2),
                    showlegend=True
                ),
                row=row,
                col=1
            )
            
            if commodity_info.get('production_cost'):
                fig.add_hline(
                    y=commodity_info['production_cost'],
                    line_dash="dash",
                    line_color="red",
                    annotation_text="Production Cost",
                    row=row,
                    col=1
                )
        
        fig.update_yaxes(title_text="Price", row=row, col=1)
        row += 1
    
    # Panel 2: Inventory
    if "Inventory" in panels and inventory_data:
        for idx, (name, df) in enumerate(inventory_data.items()):
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df.iloc[:, 0],
                    name=name.replace('_', ' '),
                    fill='tozeroy',
                    line=dict(color=colors[(idx+2) % len(colors)], width=2),
                    showlegend=True
                ),
                row=row,
                col=1
            )
        
        fig.update_yaxes(title_text="Inventory", row=row, col=1)
        row += 1
    
    # Panel 3: Crack Spread
    if "Crack Spread" in panels and 'crack_spread' in analytics:
        spread = analytics['crack_spread']
        fig.add_trace(
            go.Scatter(
                x=spread.index,
                y=spread.iloc[:, 0],
                name="Crack Spread",
                line=dict(color='green', width=2),
                showlegend=True
            ),
            row=row,
            col=1
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=row, col=1)
        fig.update_yaxes(title_text="$/Barrel", row=row, col=1)
        row += 1
    
    # Panel 4: Z-Score
    if "Volatility" in panels and 'z_score' in analytics:
        z_score = analytics['z_score']
        fig.add_trace(
            go.Scatter(
                x=z_score.index,
                y=z_score.iloc[:, 0],
                name="Z-Score",
                line=dict(color='purple', width=2),
                showlegend=True
            ),
            row=row,
            col=1
        )
        fig.add_hline(y=2, line_dash="dash", line_color="red", annotation_text="Overbought", row=row, col=1)
        fig.add_hline(y=-2, line_dash="dash", line_color="green", annotation_text="Oversold", row=row, col=1)
        fig.update_yaxes(title_text="Std Devs", row=row, col=1)
    
    fig.update_layout(
        height=300 * n_panels,
        title_text="Advanced Commodities Dashboard",
        template='plotly_white',
        hovermode='x unified',
        showlegend=True
    )
    
    return fig

def commodities_module(analysis_context: Optional[Dict] = None):
    """Main commodities analysis module with advanced features."""
    st.title("📦 Advanced Commodities Market Analysis")
    
    st.markdown("""
    Comprehensive commodity analysis with **advanced analytics**: crack spreads, inventory coverage,
    supply/demand indicators, volatility metrics, seasonal patterns, and more.
    """)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Keys
        st.subheader("🔑 API Keys")
        use_fred_api = st.checkbox("Use FRED API", value=False)
        fred_key = st.text_input("FRED API Key", type="password") if use_fred_api else None
        
        use_eia_api = st.checkbox("Use EIA API", value=False)
        eia_key = st.text_input("EIA API Key", type="password") if use_eia_api else None
        
        # Commodity selection
        st.subheader("📊 Select Commodity")
        
        selected_category = st.selectbox("Category", options=list(COMMODITIES.keys()))
        selected_commodity = st.selectbox(
            "Commodity",
            options=list(COMMODITIES[selected_category].keys()),
            format_func=lambda x: COMMODITIES[selected_category][x]['description']
        )
        
        commodity_info = COMMODITIES[selected_category][selected_commodity]
        
        # Advanced analytics selection
        st.subheader("🔬 Advanced Analytics")
        
        show_volatility = st.checkbox("Volatility Analysis", value=True)
        show_z_score = st.checkbox("Z-Score (Mean Reversion)", value=True)
        show_seasonal = st.checkbox("Seasonal Decomposition", value=True)
        show_price_cost = st.checkbox("Price vs Production Cost", value='production_cost' in commodity_info)
        
        # Energy-specific analytics
        show_crack_spread = False
        show_inventory_analysis = False
        show_supply_demand = False
        
        if selected_category == "Energy":
            st.subheader("⚡ Energy-Specific Analytics")
            show_crack_spread = st.checkbox("Crack Spread (Refinery Margins)",
                                          value=selected_commodity in ["WTI_Crude", "Brent_Crude"])
            show_inventory_analysis = st.checkbox("Inventory Coverage Analysis", value=True)
            show_supply_demand = st.checkbox("Supply/Demand Balance", value=True)
        
        # Date range
        st.subheader("📅 Date Range")
        lookback = st.selectbox("Lookback Period",
                               options=['1Y', '2Y', '3Y', '5Y', '10Y'],
                               index=2)
        
        lookback_days = {'1Y': 365, '2Y': 730, '3Y': 1095, '5Y': 1825, '10Y': 3650}
        start_date = datetime.now() - timedelta(days=lookback_days[lookback])
        end_date = datetime.now()
    
    # Fetch primary commodity data (WITH YFINANCE FALLBACK)
    with st.spinner(f"Fetching {commodity_info['description']} data..."):
        price_data = pd.DataFrame()
        data_source = "FRED"
        
        # Try FRED first
        if use_fred_api and fred_key:
            price_data = get_fred_data_api(
                commodity_info['fred_code'],
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d"),
                fred_key
            )
        
        if price_data.empty:
            price_data = get_fred_data_csv(
                commodity_info['fred_code'],
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d")
            )
        
        # Fallback to yfinance if FRED fails
        if price_data.empty and 'yf_ticker' in commodity_info:
            price_data = get_yfinance_price(
                commodity_info['yf_ticker'],
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d")
            )
            if not price_data.empty:
                data_source = "Yahoo Finance"
                st.info(f"✅ Using {data_source} ({commodity_info['yf_ticker']}) for daily prices")
    
    if price_data.empty:
        st.error(f"❌ Could not fetch commodity price data for {selected_commodity}.")
        st.info(f"""
        **Troubleshooting:**
        - FRED Series Code: `{commodity_info['fred_code']}`
        - yfinance Ticker: `{commodity_info.get('yf_ticker', 'N/A')}`
        - Try enabling 'Use FRED API' and adding your API key
        - Check if the series is available on [FRED](https://fred.stlouisfed.org/series/{commodity_info['fred_code']})
        """)
        return
    
    # Ensure minimum data points
    if len(price_data) < 30:
        st.warning(f"⚠️ Only {len(price_data)} data points available. Some analytics may be limited.")
    
    # Fetch related data for energy commodities
    inventory_data = {}
    production_data = {}
    related_prices = {}
    
    if selected_category == "Energy" and use_eia_api and eia_key:
        # Fetch inventory data
        if 'inventory_link' in commodity_info:
            inv_key = commodity_info['inventory_link']
            if inv_key in EIA_INVENTORY_SERIES:
                with st.spinner(f"Fetching inventory data..."):
                    inv_series = EIA_INVENTORY_SERIES[inv_key]
                    inventory_data[inv_key] = get_eia_data(
                        inv_series['series'],
                        eia_key,
                        inv_series.get('filter'),
                        start_date.strftime("%Y-%m-%d"),
                        end_date.strftime("%Y-%m-%d")
                    )
        
        # Fetch production data
        if show_inventory_analysis:
            with st.spinner("Fetching production data..."):
                prod_series = EIA_INVENTORY_SERIES['Crude_Production']
                production_data['production'] = get_eia_data(
                    prod_series['series'],
                    eia_key,
                    prod_series.get('filter'),
                    start_date.strftime("%Y-%m-%d"),
                    end_date.strftime("%Y-%m-%d")
                )
        
        # Fetch related prices for crack spread
        if show_crack_spread and selected_commodity in ["WTI_Crude", "Brent_Crude"]:
            with st.spinner("Fetching gasoline and heating oil prices..."):
                gasoline_info = COMMODITIES["Energy"]["Gasoline"]
                heating_info = COMMODITIES["Energy"]["Heating_Oil"]
                
                if use_fred_api and fred_key:
                    related_prices['gasoline'] = get_fred_data_api(
                        gasoline_info['fred_code'],
                        start_date.strftime("%Y-%m-%d"),
                        end_date.strftime("%Y-%m-%d"),
                        fred_key
                    )
                    related_prices['heating_oil'] = get_fred_data_api(
                        heating_info['fred_code'],
                        start_date.strftime("%Y-%m-%d"),
                        end_date.strftime("%Y-%m-%d"),
                        fred_key
                    )
                else:
                    related_prices['gasoline'] = get_fred_data_csv(
                        gasoline_info['fred_code'],
                        start_date.strftime("%Y-%m-%d"),
                        end_date.strftime("%Y-%m-%d")
                    )
                    related_prices['heating_oil'] = get_fred_data_csv(
                        heating_info['fred_code'],
                        start_date.strftime("%Y-%m-%d"),
                        end_date.strftime("%Y-%m-%d")
                    )
                
                # Fallback to yfinance
                if related_prices.get('gasoline', pd.DataFrame()).empty:
                    related_prices['gasoline'] = get_yfinance_price(
                        gasoline_info['yf_ticker'],
                        start_date.strftime("%Y-%m-%d"),
                        end_date.strftime("%Y-%m-%d")
                    )
                
                if related_prices.get('heating_oil', pd.DataFrame()).empty:
                    related_prices['heating_oil'] = get_yfinance_price(
                        heating_info['yf_ticker'],
                        start_date.strftime("%Y-%m-%d"),
                        end_date.strftime("%Y-%m-%d")
                    )
    
    # ============= CALCULATE ADVANCED ANALYTICS =============
    analytics = {}
    
    # Volatility metrics
    if show_volatility and len(price_data) >= 30:
        analytics['volatility'] = calculate_volatility_metrics(price_data)
    
    # Z-Score
    if show_z_score and len(price_data) >= 90:
        analytics['z_score'] = calculate_z_score(price_data, window=min(90, len(price_data) // 2))
    
    # Seasonal decomposition
    if show_seasonal and len(price_data) >= 104:
        analytics['seasonal'] = calculate_seasonal_pattern(price_data, period=52)
    
    # Price to cost ratio
    if show_price_cost and 'production_cost' in commodity_info:
        analytics['price_to_cost'] = calculate_price_to_cost_ratio(
            price_data,
            commodity_info['production_cost']
        )
    
    # Market structure
    if len(price_data) >= 30:
        analytics['market_structure'] = calculate_contango_backwardation(price_data, window=min(30, len(price_data) // 3))
    
    # Energy-specific analytics
    if selected_category == "Energy":
        if show_crack_spread and 'gasoline' in related_prices and not related_prices['gasoline'].empty:
            analytics['crack_spread'] = calculate_crack_spread(
                price_data,
                related_prices['gasoline'],
                related_prices.get('heating_oil')
            )
        
        if show_inventory_analysis and inventory_data and production_data:
            inv_key = list(inventory_data.keys())[0]
            if not inventory_data[inv_key].empty and not production_data['production'].empty:
                analytics['inventory_coverage'] = calculate_inventory_coverage(
                    inventory_data[inv_key],
                    production_data['production']
                )
        
        if show_supply_demand and inventory_data:
            inv_key = list(inventory_data.keys())[0]
            if not inventory_data[inv_key].empty:
                analytics['supply_demand'] = calculate_supply_demand_balance(
                    price_data,
                    inventory_data[inv_key]
                )
    
    # ============= DISPLAY RESULTS =============
    
    # Key Metrics Summary
    st.header("📊 Key Metrics")
    cols = st.columns(4)
    
    current_price = price_data.iloc[-1, 0]
    
    # Safe price change calculation
    if len(price_data) >= 20:
        price_change_1m = ((current_price - price_data.iloc[-20, 0]) / price_data.iloc[-20, 0] * 100)
    elif len(price_data) > 1:
        price_change_1m = ((current_price - price_data.iloc[0, 0]) / price_data.iloc[0, 0] * 100)
    else:
        price_change_1m = 0
    
    with cols[0]:
        st.metric(
            "Current Price",
            f"{current_price:.2f}",
            f"{price_change_1m:+.2f}% (1M)"
        )
    
    with cols[1]:
        if 'volatility' in analytics:
            st.metric(
                "30D Volatility",
                f"{analytics['volatility']['Current_Vol_30d']:.1f}%",
                f"{analytics['volatility']['Current_Vol_30d'] - analytics['volatility']['Avg_Vol_1Y']:.1f}% vs avg",
                delta_color="inverse"
            )
    
    with cols[2]:
        if 'z_score' in analytics and not analytics['z_score'].empty and len(analytics['z_score']) > 0:
            try:
                z_val = analytics['z_score'].iloc[-1, 0]
                z_status = "Overbought" if z_val > 2 else "Oversold" if z_val < -2 else "Normal"
                st.metric(
                    "Z-Score Signal",
                    f"{z_val:.2f}",
                    z_status
                )
            except (IndexError, KeyError):
                pass
    
    with cols[3]:
        if 'price_to_cost' in analytics and not analytics['price_to_cost'].empty and len(analytics['price_to_cost']) > 0:
            try:
                ratio = analytics['price_to_cost'].iloc[-1, 0]
                profitability = "Highly Profitable" if ratio > 1.5 else "Profitable" if ratio > 1.2 else "Marginal"
                st.metric(
                    "Price/Cost Ratio",
                    f"{ratio:.2f}x",
                    profitability
                )
            except (IndexError, KeyError):
                pass
    
    # Main Dashboard
    st.header("📈 Interactive Dashboard")
    
    dashboard_data = {selected_commodity: price_data}
    fig = create_advanced_dashboard(dashboard_data, inventory_data, analytics, commodity_info)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    
    # Tabs for detailed analysis
    tabs_list = ["📊 Price Analysis", "🔬 Advanced Analytics"]
    if inventory_data:
        tabs_list.append("📦 Inventory Analysis")
    if 'crack_spread' in analytics and not analytics['crack_spread'].empty:
        tabs_list.append("⚙️ Refinery Margins")
    if 'seasonal' in analytics and not analytics['seasonal'].empty:
        tabs_list.append("📅 Seasonality")
    
    tabs = st.tabs(tabs_list)
    tab_idx = 0
    
    # Tab 1: Price Analysis
    with tabs[tab_idx]:
        tab_idx += 1
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=price_data.index,
                y=price_data.iloc[:, 0],
                name="Price",
                line=dict(color='#1f77b4', width=2)
            ))
            
            # Add moving averages if enough data
            if len(price_data) >= 50:
                ma_20 = price_data.iloc[:, 0].rolling(20).mean()
                ma_50 = price_data.iloc[:, 0].rolling(50).mean()
                
                fig.add_trace(go.Scatter(
                    x=ma_20.index,
                    y=ma_20,
                    name="20-Day MA",
                    line=dict(color='orange', width=1, dash='dash')
                ))
                
                fig.add_trace(go.Scatter(
                    x=ma_50.index,
                    y=ma_50,
                    name="50-Day MA",
                    line=dict(color='red', width=1, dash='dash')
                ))
            
            if 'production_cost' in commodity_info:
                fig.add_hline(
                    y=commodity_info['production_cost'],
                    line_dash="dot",
                    line_color="green",
                    annotation_text="Avg Production Cost"
                )
            
            fig.update_layout(
                title=f"{commodity_info['description']} - Price Chart ({data_source})",
                xaxis_title="Date",
                yaxis_title=commodity_info['unit'],
                template='plotly_white',
                height=500,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📈 Statistics")
            
            try:
                if len(price_data) >= 252:
                    high_52w = price_data.iloc[-252:, 0].max()
                    low_52w = price_data.iloc[-252:, 0].min()
                else:
                    high_52w = price_data.iloc[:, 0].max()
                    low_52w = price_data.iloc[:, 0].min()
                
                stats_data = {
                    'Current': current_price,
                    '52W High': high_52w,
                    '52W Low': low_52w,
                    'Mean': price_data.iloc[:, 0].mean(),
                    'Median': price_data.iloc[:, 0].median(),
                    'Std Dev': price_data.iloc[:, 0].std()
                }
                
                if 'volatility' in analytics:
                    stats_data.update({
                        '30D Vol (%)': analytics['volatility']['Current_Vol_30d'],
                        'Max Drawdown (%)': analytics['volatility']['Max_Drawdown'],
                        'Sharpe Ratio': analytics['volatility']['Sharpe_Ratio']
                    })
                
                stats_df = pd.DataFrame(stats_data.items(), columns=['Metric', 'Value'])
                st.dataframe(
                    stats_df.style.format({'Value': '{:.2f}'}),
                    hide_index=True,
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Error calculating statistics: {e}")
    
    # Tab 2: Advanced Analytics
    with tabs[tab_idx]:
        tab_idx += 1
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'z_score' in analytics and not analytics['z_score'].empty:
                st.subheader("📊 Z-Score Analysis")
                
                fig = go.Figure()
                z_data = analytics['z_score']
                
                fig.add_trace(go.Scatter(
                    x=z_data.index,
                    y=z_data.iloc[:, 0],
                    name="Z-Score",
                    line=dict(color='purple', width=2),
                    fill='tozeroy'
                ))
                
                fig.add_hline(y=2, line_dash="dash", line_color="red", annotation_text="Overbought (+2σ)")
                fig.add_hline(y=-2, line_dash="dash", line_color="green", annotation_text="Oversold (-2σ)")
                fig.add_hline(y=0, line_dash="dot", line_color="gray")
                
                fig.update_layout(
                    title="Mean Reversion Indicator",
                    xaxis_title="Date",
                    yaxis_title="Standard Deviations from Mean",
                    template='plotly_white',
                    height=400,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.info("""
                **Interpretation:**
                - **Z > +2**: Overbought (consider selling)
                - **Z < -2**: Oversold (consider buying)
                - **Z near 0**: Trading at mean (neutral)
                """)
        
        with col2:
            if 'market_structure' in analytics and not analytics['market_structure'].empty:
                st.subheader("📉 Market Structure")
                
                fig = go.Figure()
                ms_data = analytics['market_structure']
                
                fig.add_trace(go.Scatter(
                    x=ms_data.index,
                    y=ms_data.iloc[:, 0],
                    name="30D Trend Slope",
                    line=dict(color='teal', width=2),
                    fill='tozeroy'
                ))
                
                fig.add_hline(y=0, line_dash="dot", line_color="gray")
                
                fig.update_layout(
                    title="Contango/Backwardation Indicator",
                    xaxis_title="Date",
                    yaxis_title="Price Slope",
                    template='plotly_white',
                    height=400,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.info("""
                **Interpretation:**
                - **Positive Slope**: Contango (futures > spot, weak demand)
                - **Negative Slope**: Backwardation (spot > futures, strong demand)
                """)
        
        if 'price_to_cost' in analytics and not analytics['price_to_cost'].empty:
            st.subheader("💵 Price vs Production Cost")
            
            fig = go.Figure()
            ptc_data = analytics['price_to_cost']
            
            fig.add_trace(go.Scatter(
                x=ptc_data.index,
                y=ptc_data.iloc[:, 0],
                name="Price/Cost Ratio",
                line=dict(color='green', width=2),
                fill='tozeroy'
            ))
            
            fig.add_hline(y=1.5, line_dash="dash", line_color="darkgreen", annotation_text="Highly Profitable")
            fig.add_hline(y=1.2, line_dash="dash", line_color="orange", annotation_text="Profitable")
            fig.add_hline(y=1.0, line_dash="dot", line_color="red", annotation_text="Breakeven")
            
            fig.update_layout(
                title=f"Profitability Analysis (Breakeven: ${commodity_info['production_cost']})",
                xaxis_title="Date",
                yaxis_title="Price/Cost Ratio",
                template='plotly_white',
                height=400,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Tab 3: Inventory Analysis
    if inventory_data and tab_idx < len(tabs):
        with tabs[tab_idx]:
            tab_idx += 1
            
            inv_key = list(inventory_data.keys())[0]
            inv_data = inventory_data[inv_key]
            
            if not inv_data.empty:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=inv_data.index,
                        y=inv_data.iloc[:, 0],
                        name="Inventory Level",
                        line=dict(color='#1f77b4', width=2),
                        fill='tozeroy'
                    ))
                    
                    if len(inv_data) >= 260:
                        avg_5y = inv_data.iloc[:, 0].rolling(260).mean()
                        fig.add_trace(go.Scatter(
                            x=avg_5y.index,
                            y=avg_5y,
                            name="5Y Average",
                            line=dict(color='red', width=1, dash='dash')
                        ))
                    
                    fig.update_layout(
                        title=f"{EIA_INVENTORY_SERIES[inv_key]['description']}",
                        xaxis_title="Date",
                        yaxis_title=EIA_INVENTORY_SERIES[inv_key]['unit'],
                        template='plotly_white',
                        height=400,
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.subheader("📊 Inventory Stats")
                    
                    try:
                        current_inv = inv_data.iloc[-1, 0]
                        avg_inv = inv_data.iloc[:, 0].mean()
                        pct_vs_avg = ((current_inv - avg_inv) / avg_inv * 100) if avg_inv != 0 else 0
                        
                        st.metric(
                            "Current Level",
                            f"{current_inv:,.0f}",
                            f"{pct_vs_avg:+.1f}% vs avg"
                        )
                        
                        if len(inv_data) >= 2:
                            weekly_change = inv_data.iloc[-1, 0] - inv_data.iloc[-2, 0]
                            st.metric(
                                "Weekly Change",
                                f"{weekly_change:+,.0f}",
                                "Build" if weekly_change > 0 else "Draw"
                            )
                    except Exception as e:
                        st.error(f"Error: {e}")
                
                if 'inventory_coverage' in analytics and not analytics['inventory_coverage'].empty:
                    st.subheader("📅 Days of Supply Coverage")
                    
                    cov_data = analytics['inventory_coverage']
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=cov_data.index,
                        y=cov_data.iloc[:, 0],
                        name="Days Coverage",
                        line=dict(color='orange', width=2)
                    ))
                    
                    fig.update_layout(
                        title="Inventory / Daily Production",
                        xaxis_title="Date",
                        yaxis_title="Days",
                        template='plotly_white',
                        height=300,
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                if 'supply_demand' in analytics and not analytics['supply_demand'].empty:
                    st.subheader("⚖️ Supply/Demand Balance")
                    
                    sd_data = analytics['supply_demand']
                    
                    fig = make_subplots(
                        rows=2, cols=1,
                        subplot_titles=("Price Change", "Demand Signal"),
                        vertical_spacing=0.15
                    )
                    
                    fig.add_trace(
                        go.Scatter(
                            x=sd_data.index,
                            y=sd_data['Price_Change_%'],
                            name="Price Change %",
                            line=dict(color='blue', width=2)
                        ),
                        row=1, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(
                            x=sd_data.index,
                            y=sd_data['Demand_Signal'],
                            name="Demand Signal",
                            line=dict(color='green', width=2),
                            fill='tozeroy'
                        ),
                        row=2, col=1
                    )
                    
                    fig.add_hline(y=0, line_dash="dot", line_color="gray", row=2, col=1)
                    
                    fig.update_layout(
                        height=600,
                        template='plotly_white',
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
    
    # Tab 4: Crack Spread
    if 'crack_spread' in analytics and not analytics['crack_spread'].empty and tab_idx < len(tabs):
        with tabs[tab_idx]:
            tab_idx += 1
            
            st.subheader("⚙️ 3-2-1 Crack Spread (Refinery Margins)")
            
            spread_data = analytics['crack_spread']
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=spread_data.index,
                    y=spread_data.iloc[:, 0],
                    name="Crack Spread",
                    line=dict(color='green', width=2),
                    fill='tozeroy'
                ))
                
                fig.add_hline(y=0, line_dash="dot", line_color="gray")
                fig.add_hline(y=15, line_dash="dash", line_color="darkgreen", annotation_text="Profitable")
                
                fig.update_layout(
                    title="Refinery Profitability Indicator",
                    xaxis_title="Date",
                    yaxis_title="$/Barrel",
                    template='plotly_white',
                    height=400,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                try:
                    current_spread = spread_data.iloc[-1, 0]
                    avg_spread = spread_data.iloc[:, 0].mean()
                    max_spread = spread_data.iloc[:, 0].max()
                    
                    st.metric("Current Spread", f"${current_spread:.2f}", f"${current_spread - avg_spread:+.2f} vs avg")
                    st.metric("Average Spread", f"${avg_spread:.2f}")
                    st.metric("Max Spread", f"${max_spread:.2f}")
                except:
                    pass
            
            st.info("""
            **3-2-1 Crack Spread**: Simulates refinery margin from processing 3 barrels of crude 
            into 2 barrels of gasoline and 1 barrel of heating oil.
            """)
    
    # Tab 5: Seasonality
    if 'seasonal' in analytics and not analytics['seasonal'].empty and tab_idx < len(tabs):
        with tabs[tab_idx]:
            tab_idx += 1
            
            st.subheader("📅 Seasonal Decomposition")
            
            seasonal_data = analytics['seasonal']
            
            fig = make_subplots(
                rows=4, cols=1,
                subplot_titles=("Original", "Trend", "Seasonal", "Residual"),
                vertical_spacing=0.1
            )
            
            components = ['Original', 'Trend', 'Seasonal', 'Residual']
            colors = ['blue', 'red', 'green', 'gray']
            
            for idx, (comp, color) in enumerate(zip(components, colors), 1):
                fig.add_trace(
                    go.Scatter(
                        x=seasonal_data.index,
                        y=seasonal_data[comp],
                        name=comp,
                        line=dict(color=color, width=2),
                        showlegend=False
                    ),
                    row=idx, col=1
                )
            
            fig.update_layout(
                height=800,
                title_text="Time Series Decomposition",
                template='plotly_white',
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Export Section
    st.header("💾 Export Data")
    
    export_options = {'Price Data': price_data}
    
    if inventory_data:
        for key, val in inventory_data.items():
            if not val.empty:
                export_options[f'Inventory ({key})'] = val
    
    if 'crack_spread' in analytics and not analytics['crack_spread'].empty:
        export_options['Crack Spread'] = analytics['crack_spread']
    
    cols = st.columns(min(len(export_options), 4))
    for idx, (name, df) in enumerate(export_options.items()):
        with cols[idx % len(cols)]:
            csv = df.to_csv()
            st.download_button(
                label=f"📥 {name}",
                data=csv,
                file_name=f"{selected_commodity}_{name.replace(' ', '_')}_{start_date.strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    # Insights & Recommendations
    st.header("💡 Market Insights")
    
    insights = []
    
    if 'price_to_cost' in analytics and not analytics['price_to_cost'].empty and len(analytics['price_to_cost']) > 0:
        try:
            ratio = analytics['price_to_cost'].iloc[-1, 0]
            if ratio > 1.5:
                insights.append(f"✅ **Highly Profitable**: Price is {ratio:.1f}x production cost")
            elif ratio < 1.2:
                insights.append(f"⚠️ **Marginal Economics**: Price only {ratio:.1f}x cost")
        except:
            pass
    
    if 'z_score' in analytics and not analytics['z_score'].empty and len(analytics['z_score']) > 0:
        try:
            z_val = analytics['z_score'].iloc[-1, 0]
            if z_val > 2:
                insights.append(f"📊 **Overbought Signal**: Z-score at +{z_val:.1f}")
            elif z_val < -2:
                insights.append(f"📊 **Oversold Signal**: Z-score at {z_val:.1f}")
        except:
            pass
    
    if inventory_data:
        try:
            inv_key = list(inventory_data.keys())[0]
            inv_data = inventory_data[inv_key]
            if not inv_data.empty and len(inv_data) > 0:
                current_inv = inv_data.iloc[-1, 0]
                avg_inv = inv_data.iloc[:, 0].mean()
                if avg_inv > 0:
                    pct_diff = ((current_inv/avg_inv-1)*100)
                    if current_inv < avg_inv * 0.9:
                        insights.append(f"📦 **Low Inventory**: Stocks {pct_diff:.1f}% below average")
                    elif current_inv > avg_inv * 1.1:
                        insights.append(f"📦 **High Inventory**: Stocks {pct_diff:.1f}% above average")
        except:
            pass
    
    if 'crack_spread' in analytics and not analytics['crack_spread'].empty and len(analytics['crack_spread']) > 0:
        try:
            spread = analytics['crack_spread'].iloc[-1, 0]
            if spread > 20:
                insights.append(f"⚙️ **Strong Refinery Margins**: Crack spread at ${spread:.2f}")
            elif spread < 10:
                insights.append(f"⚙️ **Weak Refinery Margins**: Crack spread only ${spread:.2f}")
        except:
            pass
    
    if 'volatility' in analytics:
        current_vol = analytics['volatility']['Current_Vol_30d']
        avg_vol = analytics['volatility']['Avg_Vol_1Y']
        if current_vol > avg_vol * 1.3:
            insights.append(f"⚡ **Elevated Volatility**: 30D vol at {current_vol:.1f}% vs {avg_vol:.1f}% avg")
    
    if insights:
        for insight in insights:
            st.markdown(insight)
    else:
        st.info("Analysis shows balanced market conditions.")

# Run standalone
if __name__ == "__main__":
    commodities_module()
