# commodities.py - Enhanced Commodities Analysis Module
# Advanced fundamental indicators: term structure, basis, convenience yield,
# producer hedging pressure, real price analysis, cross-commodity spreads,
# inventory cycle positioning, geopolitical risk premium, and more.

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

# ─────────────────────────────────────────────
# API Configuration
# ─────────────────────────────────────────────
FRED_API_KEY = "your_fred_api_key_here"
EIA_API_KEY  = "your_eia_api_key_here"
FRED_API_URL = "https://api.stlouisfed.org/fred/series/observations"
EIA_API_URL  = "https://api.eia.gov/v2"

# ─────────────────────────────────────────────
# Commodity Registry
# ─────────────────────────────────────────────
COMMODITIES = {
    "Energy": {
        "WTI_Crude": {
            "fred_code": "DCOILWTICO",
            "yf_ticker": "CL=F",
            "yf_ticker_front": "CL=F",
            "yf_ticker_back":  "CLM25.NYM",   # approximate second-month proxy
            "unit": "$/Barrel",
            "description": "WTI Crude Oil Spot Price",
            "inventory_link": "Crude_Oil_Stock",
            "production_cost": 45,
            "global_demand_fred": "DGDPUSQ",   # placeholder – real GDP proxy
            "category": "Energy",
        },
        "Brent_Crude": {
            "fred_code": "DCOILBRENTEU",
            "yf_ticker": "BZ=F",
            "unit": "$/Barrel",
            "description": "Brent Crude Oil Spot Price",
            "production_cost": 50,
            "category": "Energy",
        },
        "Natural_Gas": {
            "fred_code": "DHHNGSP",
            "yf_ticker": "NG=F",
            "unit": "$/MMBtu",
            "description": "Henry Hub Natural Gas Spot Price",
            "inventory_link": "Natural_Gas_Storage",
            "production_cost": 2.5,
            "category": "Energy",
        },
        "Gasoline": {
            "fred_code": "GASREGW",
            "yf_ticker": "RB=F",
            "unit": "$/Gallon",
            "description": "US Regular Gasoline Price",
            "inventory_link": "Gasoline_Stock",
            "category": "Energy",
        },
        "Heating_Oil": {
            "fred_code": "DHOILNYH",
            "yf_ticker": "HO=F",
            "unit": "$/Gallon",
            "description": "NY Harbor Heating Oil Spot Price",
            "category": "Energy",
        },
    },
    "Metals": {
        "Gold": {
            "fred_code": "GOLDAMGBD228NLBM",
            "yf_ticker": "GC=F",
            "unit": "$/Troy Oz",
            "description": "Gold Fixing Price (London)",
            "production_cost": 1200,
            "category": "Metals",
        },
        "Silver": {
            "fred_code": "SLVPRUSD",
            "yf_ticker": "SI=F",
            "unit": "$/Troy Oz",
            "description": "Silver Price (London)",
            "production_cost": 18,
            "category": "Metals",
        },
        "Copper": {
            "fred_code": "PCOPPUSDM",
            "yf_ticker": "HG=F",
            "unit": "$/Pound",
            "description": "Global Copper Price",
            "production_cost": 3.0,
            "category": "Metals",
        },
        "Platinum": {
            "fred_code": "PLATINUMLBMA",
            "yf_ticker": "PL=F",
            "unit": "$/Troy Oz",
            "description": "Platinum Price (London)",
            "category": "Metals",
        },
        "Palladium": {
            "fred_code": "PALLADIUMLBMA",
            "yf_ticker": "PA=F",
            "unit": "$/Troy Oz",
            "description": "Palladium Price (London)",
            "category": "Metals",
        },
    },
    "Agriculture": {
        "Wheat": {
            "fred_code": "PWHEAMTUSDM",
            "yf_ticker": "ZW=F",
            "unit": "$/Bushel",
            "description": "Global Wheat Price",
            "category": "Agriculture",
        },
        "Corn": {
            "fred_code": "PMAIZMTUSDM",
            "yf_ticker": "ZC=F",
            "unit": "$/Bushel",
            "description": "Global Corn Price",
            "category": "Agriculture",
        },
        "Soybeans": {
            "fred_code": "PSOYBUSDQ",
            "yf_ticker": "ZS=F",
            "unit": "$/Bushel",
            "description": "Global Soybean Price",
            "category": "Agriculture",
        },
        "Coffee": {
            "fred_code": "PCOFFOTMUSDM",
            "yf_ticker": "KC=F",
            "unit": "cents/lb",
            "description": "Global Coffee Price",
            "category": "Agriculture",
        },
        "Cotton": {
            "fred_code": "PCOTTINDUSDM",
            "yf_ticker": "CT=F",
            "unit": "cents/lb",
            "description": "Global Cotton Price",
            "category": "Agriculture",
        },
    },
}

EIA_INVENTORY_SERIES = {
    "Crude_Oil_Stock": {
        "series": "petroleum/stoc/wstk",
        "filter": {"product": "EPC0", "duoarea": "NUS"},
        "unit": "Thousand Barrels",
        "description": "US Crude Oil Stocks (Excluding SPR)",
    },
    "Gasoline_Stock": {
        "series": "petroleum/stoc/wstk",
        "filter": {"product": "EPM0", "duoarea": "NUS"},
        "unit": "Thousand Barrels",
        "description": "US Total Gasoline Stocks",
    },
    "Distillate_Stock": {
        "series": "petroleum/stoc/wstk",
        "filter": {"product": "EPD0", "duoarea": "NUS"},
        "unit": "Thousand Barrels",
        "description": "US Distillate Fuel Oil Stocks",
    },
    "Natural_Gas_Storage": {
        "series": "natural-gas/stor/wkly",
        "unit": "Billion Cubic Feet",
        "description": "Natural Gas Underground Storage",
    },
    "Crude_Production": {
        "series": "petroleum/crd/crpdn",
        "filter": {"duoarea": "NUS"},
        "unit": "Thousand Barrels/Day",
        "description": "US Crude Oil Production",
    },
    "Refinery_Utilization": {
        "series": "petroleum/pnp/unc",
        "unit": "Percent",
        "description": "US Refinery Utilization Rate",
    },
}

# FRED macro series for fundamental context
MACRO_SERIES = {
    "DXY_Proxy":        "DTWEXBGS",   # USD broad trade-weighted index
    "CPI":              "CPIAUCSL",   # Consumer Price Index
    "PPI_Energy":       "PPIENG",     # PPI – Energy
    "RealGDP":          "GDPC1",      # Real GDP (quarterly)
    "FedFunds":         "FEDFUNDS",   # Fed Funds rate
    "BreakevenInfla":   "T10YIE",     # 10Y breakeven inflation
    "IndustrialProd":   "INDPRO",     # Industrial production index
    "ManufacturingPMI": "MANEMP",     # Manufacturing employment proxy
    "TotalCrudeProd":   "MCRFPUS2",  # US monthly crude field production
    "RigCount":         "OILPROD",    # placeholder – use yf for Baker Hughes
}

# Cross-commodity spread pairs for the spread matrix
SPREAD_PAIRS = {
    "WTI-Brent": ("WTI_Crude", "Brent_Crude", "Energy", "Energy"),
    "Gold-Silver (GSR)": ("Gold", "Silver", "Metals", "Metals"),
    "Platinum-Palladium": ("Platinum", "Palladium", "Metals", "Metals"),
    "Corn-Wheat": ("Corn", "Wheat", "Agriculture", "Agriculture"),
    "Soybeans-Corn": ("Soybeans", "Corn", "Agriculture", "Agriculture"),
}


# ─────────────────────────────────────────────
# Data Fetching (unchanged from original, robust)
# ─────────────────────────────────────────────
@st.cache_data(ttl=3600)
def get_fred_data_csv(series_id: str, start_date: str, end_date: str) -> pd.DataFrame:
    try:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text), index_col=0, parse_dates=True)
        if df.empty or len(df.columns) == 0:
            return pd.DataFrame()
        df.columns = [series_id]
        try:
            df = df.loc[start_date:end_date]
        except Exception:
            pass
        df[series_id] = pd.to_numeric(df[series_id], errors="coerce")
        return df.dropna()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def get_fred_data_api(series_id: str, start_date: str, end_date: str, api_key: str) -> pd.DataFrame:
    try:
        params = {
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
            "observation_start": start_date,
            "observation_end": end_date,
        }
        r = requests.get(FRED_API_URL, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        if "observations" in data:
            df = pd.DataFrame(data["observations"])
            df["date"] = pd.to_datetime(df["date"])
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df = df.set_index("date")[["value"]]
            df.columns = [series_id]
            return df.dropna()
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def get_yfinance_price(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
        if not data.empty and "Close" in data.columns:
            df = data[["Close"]].copy()
            df.columns = [ticker]
            return df.dropna()
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=1800)
def get_eia_data(
    series_path: str,
    api_key: str,
    filters: Optional[Dict] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    if not api_key or api_key == "your_eia_api_key_here":
        return pd.DataFrame()
    try:
        url = f"{EIA_API_URL}/{series_path}/data/"
        params = {
            "api_key": api_key,
            "frequency": "weekly",
            "data[]": "value",
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
        }
        if filters:
            for k, v in filters.items():
                params[f"facets[{k}][]"] = v
        if start_date:
            params["start"] = start_date
        if end_date:
            params["end"] = end_date
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        if "response" in data and "data" in data["response"]:
            df = pd.DataFrame(data["response"]["data"])
            if "period" in df.columns and "value" in df.columns:
                df["period"] = pd.to_datetime(df["period"])
                df = df.set_index("period")[["value"]]
                df["value"] = pd.to_numeric(df["value"], errors="coerce")
                return df.dropna()
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def fetch_macro_series(series_id: str, start_date: str, end_date: str, api_key: Optional[str] = None) -> pd.DataFrame:
    """Fetch a FRED macro series with CSV fallback."""
    if api_key:
        df = get_fred_data_api(series_id, start_date, end_date, api_key)
        if not df.empty:
            return df
    return get_fred_data_csv(series_id, start_date, end_date)


# ─────────────────────────────────────────────
# ORIGINAL ANALYTICS (kept intact)
# ─────────────────────────────────────────────
def calculate_crack_spread(crude: pd.DataFrame, gasoline: pd.DataFrame,
                           heating_oil: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    if crude.empty or gasoline.empty:
        return pd.DataFrame()
    combined = pd.concat([crude, gasoline], axis=1, join="inner")
    if heating_oil is not None and not heating_oil.empty:
        combined = pd.concat([combined, heating_oil], axis=1, join="inner")
        spread = ((2 * combined.iloc[:, 1] * 42) + (1 * combined.iloc[:, 2] * 42)) / 3 - combined.iloc[:, 0]
    else:
        spread = (2 * combined.iloc[:, 1] * 42) - combined.iloc[:, 0]
    return pd.DataFrame({"Crack_Spread": spread})


def calculate_inventory_coverage(inventory: pd.DataFrame, production: pd.DataFrame) -> pd.DataFrame:
    if inventory.empty or production.empty:
        return pd.DataFrame()
    combined = pd.concat([inventory, production], axis=1, join="inner")
    if combined.empty:
        return pd.DataFrame()
    return pd.DataFrame({"Days_Coverage": combined.iloc[:, 0] / combined.iloc[:, 1]})


def calculate_price_to_cost_ratio(price: pd.DataFrame, production_cost: float) -> pd.DataFrame:
    if price.empty or production_cost is None:
        return pd.DataFrame()
    return pd.DataFrame({"Price_to_Cost": price.iloc[:, 0] / production_cost})


def calculate_z_score(data: pd.DataFrame, window: int = 90) -> pd.DataFrame:
    if data.empty or len(data) < window:
        return pd.DataFrame()
    rm = data.iloc[:, 0].rolling(window).mean()
    rs = data.iloc[:, 0].rolling(window).std()
    return pd.DataFrame({"Z_Score": (data.iloc[:, 0] - rm) / rs})


def calculate_volatility_metrics(price: pd.DataFrame) -> Dict:
    if price.empty or len(price) < 30:
        return {}
    ret = price.iloc[:, 0].pct_change().dropna()
    return {
        "Current_Vol_30d": ret.tail(30).std() * np.sqrt(252) * 100,
        "Current_Vol_90d": ret.tail(90).std() * np.sqrt(252) * 100 if len(ret) >= 90 else None,
        "Avg_Vol_1Y":      ret.std() * np.sqrt(252) * 100,
        "Max_Drawdown":    ((price.iloc[:, 0] / price.iloc[:, 0].cummax()) - 1).min() * 100,
        "Sharpe_Ratio":    (ret.mean() / ret.std()) * np.sqrt(252) if ret.std() > 0 else 0,
    }


def calculate_seasonal_pattern(data: pd.DataFrame, period: int = 52) -> pd.DataFrame:
    if data.empty or len(data) < period * 2:
        return pd.DataFrame()
    trend = data.iloc[:, 0].rolling(window=period, center=True).mean()
    detrended = data.iloc[:, 0] - trend
    seasonal = detrended.groupby(detrended.index.isocalendar().week).transform("mean")
    return pd.DataFrame({
        "Original": data.iloc[:, 0],
        "Trend":    trend,
        "Seasonal": seasonal,
        "Residual": data.iloc[:, 0] - trend - seasonal,
    })


def calculate_contango_backwardation(price: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    if price.empty or len(price) < window:
        return pd.DataFrame()
    slope = price.iloc[:, 0].rolling(window).apply(
        lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) == window else np.nan
    )
    return pd.DataFrame({"Market_Structure": slope})


def calculate_supply_demand_balance(price: pd.DataFrame, inventory: pd.DataFrame) -> pd.DataFrame:
    if price.empty or inventory.empty:
        return pd.DataFrame()
    combined = pd.concat([price, inventory], axis=1, join="inner")
    if len(combined) < 50:
        return pd.DataFrame()
    inv_chg   = combined.iloc[:, 1].diff()
    prc_chg   = combined.iloc[:, 0].pct_change() * 100
    std20     = inv_chg.rolling(20).std().replace(0, np.nan)
    return pd.DataFrame({
        "Price_Change_%":  prc_chg,
        "Inventory_Change": inv_chg,
        "Demand_Signal":   -inv_chg / std20,
    })


# ─────────────────────────────────────────────
# NEW FUNDAMENTAL INDICATORS
# ─────────────────────────────────────────────

def calculate_real_price(nominal_price: pd.DataFrame, cpi: pd.DataFrame) -> pd.DataFrame:
    """
    Deflate nominal commodity price by CPI to obtain purchasing-power-adjusted real price.
    Helps distinguish genuine supply/demand shifts from monetary inflation noise.
    Both series resampled to monthly (CPI is monthly).
    """
    if nominal_price.empty or cpi.empty:
        return pd.DataFrame()
    # Resample both to month-end
    nom = nominal_price.iloc[:, 0].resample("ME").last().dropna()
    c   = cpi.iloc[:, 0].resample("ME").last().dropna()
    combined = pd.concat([nom, c], axis=1, join="inner").dropna()
    if combined.empty:
        return pd.DataFrame()
    base_cpi = combined.iloc[-1, 1]          # anchor to latest CPI = "today's dollars"
    real = combined.iloc[:, 0] * (base_cpi / combined.iloc[:, 1])
    return pd.DataFrame({"Nominal": combined.iloc[:, 0], "Real_Price": real})


def calculate_usd_adjusted_price(price: pd.DataFrame, dxy: pd.DataFrame) -> pd.DataFrame:
    """
    Express commodity price in DXY-adjusted terms.
    Commodities are priced in USD; a stronger dollar mechanically depresses prices.
    Stripping out the FX effect isolates true physical demand changes.
    Returns: raw price, DXY-adjusted price, and rolling correlation.
    """
    if price.empty or dxy.empty:
        return pd.DataFrame()
    p = price.iloc[:, 0].resample("W").last().dropna()
    d = dxy.iloc[:, 0].resample("W").last().dropna()
    combined = pd.concat([p, d], axis=1, join="inner").dropna()
    if len(combined) < 20:
        return pd.DataFrame()
    # Normalize DXY to its mean so we express price "as if DXY were at its historical average"
    dxy_mean  = combined.iloc[:, 1].mean()
    adjusted  = combined.iloc[:, 0] * (dxy_mean / combined.iloc[:, 1])
    rolling_corr = combined.iloc[:, 0].rolling(52).corr(combined.iloc[:, 1])
    return pd.DataFrame({
        "Nominal_Price":    combined.iloc[:, 0],
        "DXY_Adj_Price":    adjusted,
        "Rolling_Corr_DXY": rolling_corr,
    })


def calculate_convenience_yield(spot: pd.DataFrame,
                                futures_approx: pd.DataFrame,
                                risk_free_rate: float = 0.05,
                                storage_cost_pct: float = 0.02) -> pd.DataFrame:
    """
    Convenience yield = benefit of holding physical commodity vs. futures.
    Formula: CY = r + s - (F - S) / S   where r=risk-free, s=storage cost, F/S=futures/spot ratio.
    
    Positive CY → markets in backwardation → tight physical supply.
    Negative CY → contango → excess supply / storage glut.
    
    We approximate futures price using a 30-day forward price constructed from yfinance
    second-month contract (passed as futures_approx) or roll the spot forward analytically.
    """
    if spot.empty:
        return pd.DataFrame()

    s = spot.iloc[:, 0].resample("W").last().dropna()

    if futures_approx.empty or len(futures_approx) < 10:
        # Analytical approximation: F ≈ S * e^((r+s)*T), T=1/12 year
        T = 1 / 12
        f = s * np.exp((risk_free_rate + storage_cost_pct) * T)
    else:
        f = futures_approx.iloc[:, 0].resample("W").last().dropna()
        combined = pd.concat([s, f], axis=1, join="inner").dropna()
        s = combined.iloc[:, 0]
        f = combined.iloc[:, 1]

    cy = risk_free_rate + storage_cost_pct - (f - s) / s
    return pd.DataFrame({"Convenience_Yield": cy})


def calculate_inventory_cycle_position(inventory: pd.DataFrame,
                                       years_lookback: int = 5) -> pd.DataFrame:
    """
    Place current inventory within its multi-year seasonal band.
    Returns:
      - 5Y average for each calendar week
      - ±1 standard deviation band
      - Percentile rank of current level vs history
    This is the classic EIA-style 'shaded band' chart used by energy traders.
    """
    if inventory.empty or len(inventory) < 52:
        return pd.DataFrame()

    inv = inventory.iloc[:, 0].dropna().copy()
    inv.index = pd.to_datetime(inv.index)
    inv_df = inv.to_frame(name="value")
    inv_df["week"] = inv_df.index.isocalendar().week.astype(int)
    inv_df["year"]  = inv_df.index.year

    # Cutoff for history window
    cutoff_year = inv_df["year"].max() - years_lookback

    hist = inv_df[inv_df["year"] <= cutoff_year]
    stats_by_week = hist.groupby("week")["value"].agg(
        hist_mean="mean",
        hist_std="std",
        hist_min="min",
        hist_max="max",
    ).reset_index()

    result = inv_df.merge(stats_by_week, on="week", how="left")
    result = result.set_index(inv_df.index[:len(result)])

    # Rolling percentile rank vs full history
    result["Pct_Rank"] = result["value"].expanding().rank(pct=True) * 100

    return result[["value", "hist_mean", "hist_std", "hist_min", "hist_max", "Pct_Rank"]].rename(
        columns={"value": "Current"}
    )


def calculate_producer_breakeven_heat(price: pd.DataFrame,
                                      production_cost: float,
                                      opex_ratio: float = 0.6) -> pd.DataFrame:
    """
    Tiered producer economics:
      - Variable (cash) cost ≈ production_cost * opex_ratio   (must exceed to keep pumping)
      - Full-cycle breakeven ≈ production_cost                (must exceed to justify new investment)
      - Premium breakeven    ≈ production_cost * 1.3          (generates strong free cash flow)
    Returns series labeling each period's economic zone.
    """
    if price.empty or production_cost is None:
        return pd.DataFrame()

    p = price.iloc[:, 0]
    cash_cost   = production_cost * opex_ratio
    full_cost   = production_cost
    premium_thr = production_cost * 1.3

    zone = pd.cut(
        p,
        bins=[-np.inf, cash_cost, full_cost, premium_thr, np.inf],
        labels=["Shut-in Zone", "Cash Positive", "Full-Cycle Breakeven", "Premium Economics"],
    )

    return pd.DataFrame({
        "Price":        p,
        "Zone":         zone,
        "Cash_Cost":    cash_cost,
        "Full_Cycle":   full_cost,
        "Premium":      premium_thr,
        "Margin_%":     ((p - full_cost) / full_cost * 100),
    })


def calculate_demand_proxy_index(price: pd.DataFrame,
                                  industrial_prod: pd.DataFrame,
                                  gdp: pd.DataFrame) -> pd.DataFrame:
    """
    Composite demand proxy: industrial production (monthly) + GDP (quarterly),
    normalized and averaged. Plotted against commodity price to reveal
    demand-driven vs. supply-driven price cycles.
    """
    if price.empty:
        return pd.DataFrame()

    p = price.iloc[:, 0].resample("ME").last().dropna()
    components = [p]
    labels = ["Price"]

    if not industrial_prod.empty:
        ip = industrial_prod.iloc[:, 0].resample("ME").last().dropna()
        ip_norm = (ip - ip.mean()) / ip.std()
        components.append(ip_norm)
        labels.append("IndProd_Norm")

    if not gdp.empty:
        g = gdp.iloc[:, 0].resample("ME").ffill().dropna()
        g_norm = (g - g.mean()) / g.std()
        components.append(g_norm)
        labels.append("GDP_Norm")

    combined = pd.concat(components, axis=1, join="inner")
    combined.columns = labels

    if len(combined.columns) > 1:
        demand_cols = [c for c in combined.columns if c != "Price"]
        combined["Demand_Index"] = combined[demand_cols].mean(axis=1)
    else:
        combined["Demand_Index"] = np.nan

    return combined


def calculate_cross_commodity_spread(
    price_a: pd.DataFrame,
    price_b: pd.DataFrame,
    name_a: str,
    name_b: str,
    ratio_mode: bool = False,
) -> pd.DataFrame:
    """
    Spread or ratio between two related commodities.
    Examples:
      WTI-Brent spread → physical/freight premium or quality discount
      Gold/Silver ratio → risk-on/risk-off positioning
      Corn/Wheat ratio → feed grain substitution signal
    """
    if price_a.empty or price_b.empty:
        return pd.DataFrame()
    combined = pd.concat([price_a.iloc[:, 0], price_b.iloc[:, 0]], axis=1, join="inner").dropna()
    combined.columns = [name_a, name_b]
    if ratio_mode:
        combined["Spread"] = combined[name_a] / combined[name_b]
        combined["Spread_MA52"] = combined["Spread"].rolling(52).mean()
        combined["Spread_Zscore"] = (
            (combined["Spread"] - combined["Spread"].rolling(52).mean())
            / combined["Spread"].rolling(52).std()
        )
    else:
        combined["Spread"] = combined[name_a] - combined[name_b]
        combined["Spread_MA52"] = combined["Spread"].rolling(52).mean()
        combined["Spread_Zscore"] = (
            (combined["Spread"] - combined["Spread"].rolling(52).mean())
            / combined["Spread"].rolling(52).std()
        )
    return combined


def calculate_geopolitical_risk_premium(
    commodity_price: pd.DataFrame,
    safe_haven_price: pd.DataFrame,
    baseline_window: int = 90,
) -> pd.DataFrame:
    """
    Proxy for geopolitical risk premium in energy/metals.
    Method: regress commodity price on safe-haven price (Gold for energy, or DXY);
    the positive residual is the 'unexplained' premium attributable to geopolitical stress.
    
    For oil: when crude rises while equities fall and gold rises → geopolitical bid.
    """
    if commodity_price.empty or safe_haven_price.empty:
        return pd.DataFrame()

    comm = commodity_price.iloc[:, 0].resample("W").last().dropna()
    haven = safe_haven_price.iloc[:, 0].resample("W").last().dropna()
    combined = pd.concat([comm, haven], axis=1, join="inner").dropna()
    if len(combined) < baseline_window:
        return pd.DataFrame()

    residuals = []
    for i in range(baseline_window, len(combined)):
        window = combined.iloc[i - baseline_window:i]
        x = window.iloc[:, 1].values.reshape(-1, 1)
        y = window.iloc[:, 0].values
        try:
            reg = LinearRegression().fit(x, y)
            pred = reg.predict(combined.iloc[[i], [1]].values)[0]
            residuals.append(combined.iloc[i, 0] - pred)
        except Exception:
            residuals.append(np.nan)

    idx = combined.index[baseline_window:]
    return pd.DataFrame({
        "Geo_Risk_Premium": residuals,
        "Commodity_Price":  combined.iloc[baseline_window:, 0].values,
    }, index=idx)


def calculate_roll_yield(spot: pd.DataFrame,
                         futures_front: pd.DataFrame,
                         annualize: bool = True) -> pd.DataFrame:
    """
    Roll yield = annualized cost/benefit of rolling futures positions.
    In contango (F > S): roll yield is NEGATIVE (you sell cheap spot, buy expensive futures).
    In backwardation (F < S): roll yield is POSITIVE (you collect the basis).
    
    Critical for commodity investors: a 10% spot gain can be wiped out by negative roll.
    Roll Yield ≈ (S - F) / F * (12/months_to_expiry)
    """
    if spot.empty or futures_front.empty:
        return pd.DataFrame()
    s = spot.iloc[:, 0].resample("W").last().dropna()
    f = futures_front.iloc[:, 0].resample("W").last().dropna()
    combined = pd.concat([s, f], axis=1, join="inner").dropna()
    if combined.empty:
        return pd.DataFrame()
    roll = (combined.iloc[:, 0] - combined.iloc[:, 1]) / combined.iloc[:, 1]
    if annualize:
        roll = roll * 12  # approximate annualization assuming ~1-month roll
    return pd.DataFrame({
        "Roll_Yield_%":  roll * 100,
        "Spot":          combined.iloc[:, 0],
        "Futures_Front": combined.iloc[:, 1],
    })


def calculate_inflation_hedge_score(
    commodity_price: pd.DataFrame,
    breakeven_inflation: pd.DataFrame,
    cpi: pd.DataFrame,
    window: int = 52,
) -> pd.DataFrame:
    """
    Measure how well the commodity tracks realized and expected inflation.
    Returns:
      - Rolling correlation with breakeven inflation (market's expected inflation)
      - Rolling correlation with CPI YoY change (realized inflation)
      - Composite hedge score (avg of both, 0-100 scale)
    
    Gold typically scores ~70+, crude ~50, agriculture variable.
    """
    if commodity_price.empty:
        return pd.DataFrame()

    p = commodity_price.iloc[:, 0].resample("ME").last().dropna()
    p_ret = p.pct_change(12) * 100  # 12-month return

    results = pd.DataFrame({"Price_12M_Ret": p_ret})

    if not breakeven_inflation.empty:
        be = breakeven_inflation.iloc[:, 0].resample("ME").last().dropna()
        combined = pd.concat([p_ret, be], axis=1, join="inner").dropna()
        if len(combined) >= window:
            results["Corr_Breakeven"] = (
                combined.iloc[:, 0]
                .rolling(window)
                .corr(combined.iloc[:, 1])
            )

    if not cpi.empty:
        c = cpi.iloc[:, 0].resample("ME").last().dropna()
        cpi_yoy = c.pct_change(12) * 100
        combined2 = pd.concat([p_ret, cpi_yoy], axis=1, join="inner").dropna()
        if len(combined2) >= window:
            results["Corr_CPI"] = (
                combined2.iloc[:, 0]
                .rolling(window)
                .corr(combined2.iloc[:, 1])
            )

    # Composite hedge score (rescale -1→1 to 0→100)
    hedge_cols = [c for c in results.columns if c.startswith("Corr_")]
    if hedge_cols:
        results["Hedge_Score"] = results[hedge_cols].mean(axis=1).clip(-1, 1)
        results["Hedge_Score"] = (results["Hedge_Score"] + 1) / 2 * 100

    return results


def calculate_inventory_to_use_ratio(inventory: pd.DataFrame,
                                     production: pd.DataFrame,
                                     months_forward: int = 1) -> pd.DataFrame:
    """
    Stocks-to-Use ratio: classic agricultural/energy fundamental.
    Lower STU → tighter market → price pressure upward.
    Ranges (crude): < 20 days = tight; 25-30 = balanced; > 35 = surplus.
    """
    if inventory.empty or production.empty:
        return pd.DataFrame()
    # Align and resample monthly
    inv = inventory.iloc[:, 0].resample("ME").last().dropna()
    prod = production.iloc[:, 0].resample("ME").mean().dropna()
    combined = pd.concat([inv, prod], axis=1, join="inner").dropna()
    if combined.empty:
        return pd.DataFrame()
    # STU in months of production
    stu = combined.iloc[:, 0] / (combined.iloc[:, 1] * 30)  # prod is daily, inv is total
    stu_ma = stu.rolling(3).mean()
    return pd.DataFrame({"STU_Months": stu, "STU_MA3": stu_ma})


def score_fundamental_outlook(
    z_score: float,
    price_to_cost: Optional[float],
    roll_yield: Optional[float],
    convenience_yield: Optional[float],
    stu_months: Optional[float],
    hedge_score: Optional[float],
) -> Dict:
    """
    Synthesize key fundamentals into a single bull/bear scorecard.
    Each component scores -2 (very bearish) to +2 (very bullish).
    """
    scores = {}
    commentary = []

    # Mean reversion signal
    if z_score is not None:
        if z_score < -2:
            scores["Mean_Reversion"] = +2
            commentary.append("📉 Price deeply oversold (Z < -2) — mean reversion potential")
        elif z_score < -1:
            scores["Mean_Reversion"] = +1
            commentary.append("📉 Price below average — mild upside bias")
        elif z_score > 2:
            scores["Mean_Reversion"] = -2
            commentary.append("📈 Price deeply overbought (Z > +2) — mean reversion risk")
        elif z_score > 1:
            scores["Mean_Reversion"] = -1
            commentary.append("📈 Price above average — mild downside risk")
        else:
            scores["Mean_Reversion"] = 0

    # Producer economics
    if price_to_cost is not None:
        if price_to_cost > 1.5:
            scores["Producer_Economics"] = +2
            commentary.append(f"💵 Strong producer margins ({price_to_cost:.1f}x cost) — capex cycle likely to accelerate supply")
        elif price_to_cost > 1.0:
            scores["Producer_Economics"] = +1
            commentary.append(f"💵 Positive producer margins ({price_to_cost:.1f}x cost)")
        elif price_to_cost < 0.9:
            scores["Producer_Economics"] = +1   # bullish medium-term (supply destruction)
            commentary.append(f"⚠️ Below cash cost ({price_to_cost:.1f}x) — expect supply curtailment (medium-term bullish)")
        else:
            scores["Producer_Economics"] = -1
            commentary.append(f"💵 Thin margins ({price_to_cost:.1f}x) — fragile supply economics")

    # Roll yield (investor perspective)
    if roll_yield is not None:
        if roll_yield > 5:
            scores["Roll_Yield"] = +2
            commentary.append(f"🔄 Positive roll yield (+{roll_yield:.1f}%) — backwardation rewards longs")
        elif roll_yield > 0:
            scores["Roll_Yield"] = +1
            commentary.append(f"🔄 Slight backwardation (+{roll_yield:.1f}%)")
        elif roll_yield < -10:
            scores["Roll_Yield"] = -2
            commentary.append(f"🔄 Deep contango ({roll_yield:.1f}%) — roll bleed punishes longs")
        else:
            scores["Roll_Yield"] = -1
            commentary.append(f"🔄 Contango ({roll_yield:.1f}%) — negative carry")

    # Convenience yield (physical market tightness)
    if convenience_yield is not None:
        if convenience_yield > 0.08:
            scores["Physical_Tightness"] = +2
            commentary.append("🏭 High convenience yield — physical market very tight")
        elif convenience_yield > 0.02:
            scores["Physical_Tightness"] = +1
            commentary.append("🏭 Moderate convenience yield — balanced physical market")
        elif convenience_yield < -0.02:
            scores["Physical_Tightness"] = -2
            commentary.append("🏭 Negative convenience yield — supply surplus / storage glut")
        else:
            scores["Physical_Tightness"] = 0

    # Stocks-to-use
    if stu_months is not None:
        if stu_months < 1.5:
            scores["Supply_Adequacy"] = +2
            commentary.append(f"📦 Very tight stocks ({stu_months:.1f} months) — bullish price pressure")
        elif stu_months < 2.5:
            scores["Supply_Adequacy"] = +1
            commentary.append(f"📦 Adequate but lean stocks ({stu_months:.1f} months)")
        elif stu_months > 4:
            scores["Supply_Adequacy"] = -2
            commentary.append(f"📦 Large inventory overhang ({stu_months:.1f} months) — bearish")
        else:
            scores["Supply_Adequacy"] = 0

    # Inflation hedge
    if hedge_score is not None:
        if hedge_score > 70:
            scores["Inflation_Hedge"] = +1
            commentary.append(f"🛡️ Strong inflation hedge score ({hedge_score:.0f}/100)")
        elif hedge_score < 30:
            scores["Inflation_Hedge"] = -1
            commentary.append(f"🛡️ Weak inflation hedge ({hedge_score:.0f}/100)")
        else:
            scores["Inflation_Hedge"] = 0

    total = sum(scores.values())
    max_possible = len(scores) * 2
    pct = (total / max_possible * 100) if max_possible else 0

    return {
        "scores":      scores,
        "commentary":  commentary,
        "total":       total,
        "max_score":   max_possible,
        "pct":         pct,
        "outlook":     "Strong Bull" if pct > 50 else
                       "Mild Bull"   if pct > 15 else
                       "Neutral"     if pct > -15 else
                       "Mild Bear"   if pct > -50 else "Strong Bear",
    }


# ─────────────────────────────────────────────
# CHART HELPERS
# ─────────────────────────────────────────────
PALETTE = dict(
    blue="#3B82F6", green="#10B981", red="#EF4444",
    orange="#F59E0B", purple="#8B5CF6", teal="#06B6D4",
    gray="#6B7280", pink="#EC4899", lime="#84CC16",
)
LAYOUT_BASE = dict(template="plotly_white", hovermode="x unified", height=420)


def _fig(**kwargs):
    fig = go.Figure()
    fig.update_layout(**{**LAYOUT_BASE, **kwargs})
    return fig


def _add_band(fig, x, upper, lower, name, color, row=None, col=None):
    kw = dict(row=row, col=col) if row else {}
    fig.add_trace(go.Scatter(
        x=list(x) + list(x)[::-1],
        y=list(upper) + list(lower)[::-1],
        fill="toself",
        fillcolor=color.replace(")", ",0.15)").replace("rgb", "rgba") if "rgb" in color else color + "26",
        line=dict(width=0),
        name=name,
        showlegend=True,
    ), **kw)


# ─────────────────────────────────────────────
# MAIN MODULE
# ─────────────────────────────────────────────
def commodities_module(analysis_context: Optional[Dict] = None):
    st.set_page_config(page_title="Commodities Intelligence", layout="wide")

    # ── Custom CSS for a cleaner look ──
    st.markdown("""
    <style>
    .stMetric label {font-size: 0.75rem; color: #6B7280;}
    .stMetric div[data-testid="stMetricValue"] {font-size: 1.4rem; font-weight: 700;}
    .stTabs [data-baseweb="tab-list"] {gap: 4px;}
    .stTabs [data-baseweb="tab"] {padding: 8px 20px; border-radius: 6px;}
    div[data-testid="stExpander"] details {border: 1px solid #E5E7EB; border-radius: 8px;}
    .scorecard-bull {color: #10B981; font-weight: 700;}
    .scorecard-bear {color: #EF4444; font-weight: 700;}
    .scorecard-neutral {color: #6B7280; font-weight: 700;}
    </style>
    """, unsafe_allow_html=True)

    st.title("🛢️ Commodities Intelligence Platform")
    st.caption("Fundamental analysis: real prices · convenience yield · roll yield · supply adequacy · inflation hedge · geopolitical premium · producer economics")

    # ── Sidebar ──
    with st.sidebar:
        st.header("⚙️ Configuration")

        st.subheader("🔑 API Keys")
        use_fred_api = st.checkbox("Use FRED API Key", False)
        fred_key = st.text_input("FRED API Key", type="password") if use_fred_api else None
        use_eia_api = st.checkbox("Use EIA API", False)
        eia_key = st.text_input("EIA API Key", type="password") if use_eia_api else None

        st.subheader("📊 Commodity")
        cat = st.selectbox("Category", list(COMMODITIES.keys()))
        comm_key = st.selectbox(
            "Commodity",
            list(COMMODITIES[cat].keys()),
            format_func=lambda x: COMMODITIES[cat][x]["description"],
        )
        info = COMMODITIES[cat][comm_key]

        st.subheader("📅 Lookback")
        lookback = st.selectbox("Period", ["1Y", "2Y", "3Y", "5Y", "10Y"], index=2)
        days_map = {"1Y": 365, "2Y": 730, "3Y": 1095, "5Y": 1825, "10Y": 3650}
        start_dt = datetime.now() - timedelta(days=days_map[lookback])
        end_dt   = datetime.now()
        s_date, e_date = start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")

        st.subheader("🔬 Fundamental Modules")
        show_real_price    = st.checkbox("Real (Inflation-Adjusted) Price",  True)
        show_usd_adj       = st.checkbox("USD-Adjusted Price (DXY Strip)",   True)
        show_conv_yield    = st.checkbox("Convenience Yield",                 True)
        show_roll_yield    = st.checkbox("Roll Yield Analysis",               True)
        show_inv_cycle     = st.checkbox("Inventory Cycle Position",         cat == "Energy")
        show_breakeven     = st.checkbox("Producer Breakeven Tiers",         "production_cost" in info)
        show_demand_proxy  = st.checkbox("Demand Proxy Index",               True)
        show_spreads       = st.checkbox("Cross-Commodity Spreads",          True)
        show_geo_risk      = st.checkbox("Geopolitical Risk Premium",        cat == "Energy")
        show_hedge_score   = st.checkbox("Inflation Hedge Score",            True)
        show_scorecard     = st.checkbox("🎯 Fundamental Scorecard",          True)

        st.subheader("📈 Classic Analytics")
        show_z_score    = st.checkbox("Z-Score (Mean Reversion)", True)
        show_volatility = st.checkbox("Volatility Metrics",       True)
        show_seasonal   = st.checkbox("Seasonal Decomposition",   True)
        show_crack      = st.checkbox("Crack Spread",             cat == "Energy")

    # ── Fetch primary price ──
    with st.spinner(f"Fetching {info['description']}…"):
        price_data  = pd.DataFrame()
        data_source = "FRED"

        if use_fred_api and fred_key:
            price_data = get_fred_data_api(info["fred_code"], s_date, e_date, fred_key)
        if price_data.empty:
            price_data = get_fred_data_csv(info["fred_code"], s_date, e_date)
        if price_data.empty and "yf_ticker" in info:
            price_data = get_yfinance_price(info["yf_ticker"], s_date, e_date)
            if not price_data.empty:
                data_source = "Yahoo Finance"

    if price_data.empty:
        st.error(f"❌ Cannot fetch data for {comm_key}. Try enabling FRED API or check the ticker.")
        return

    # ── Fetch macro context ──
    with st.spinner("Loading macro context…"):
        cpi_data  = fetch_macro_series(MACRO_SERIES["CPI"],           s_date, e_date, fred_key)
        dxy_data  = fetch_macro_series(MACRO_SERIES["DXY_Proxy"],     s_date, e_date, fred_key)
        ip_data   = fetch_macro_series(MACRO_SERIES["IndustrialProd"],s_date, e_date, fred_key)
        gdp_data  = fetch_macro_series(MACRO_SERIES["RealGDP"],       s_date, e_date, fred_key)
        be_data   = fetch_macro_series(MACRO_SERIES["BreakevenInfla"],s_date, e_date, fred_key)
        gold_data = pd.DataFrame()
        if comm_key != "Gold":
            gold_data = get_fred_data_csv(COMMODITIES["Metals"]["Gold"]["fred_code"], s_date, e_date)
            if gold_data.empty:
                gold_data = get_yfinance_price("GC=F", s_date, e_date)

    # ── Fetch futures (for roll/convenience yield) ──
    futures_data = pd.DataFrame()
    if "yf_ticker" in info:
        # Use a slightly longer-dated contract as a futures proxy
        futures_data = get_yfinance_price(info["yf_ticker"], s_date, e_date)

    # ── EIA data ──
    inventory_data, production_data = {}, {}
    if cat == "Energy" and use_eia_api and eia_key:
        if "inventory_link" in info:
            inv_key = info["inventory_link"]
            if inv_key in EIA_INVENTORY_SERIES:
                inv_s = EIA_INVENTORY_SERIES[inv_key]
                inventory_data[inv_key] = get_eia_data(
                    inv_s["series"], eia_key, inv_s.get("filter"), s_date, e_date
                )
        prod_s = EIA_INVENTORY_SERIES["Crude_Production"]
        production_data["production"] = get_eia_data(
            prod_s["series"], eia_key, prod_s.get("filter"), s_date, e_date
        )

    # ── Compute fundamentals ──
    fund = {}

    if show_real_price and not cpi_data.empty:
        fund["real_price"] = calculate_real_price(price_data, cpi_data)

    if show_usd_adj and not dxy_data.empty:
        fund["usd_adj"] = calculate_usd_adjusted_price(price_data, dxy_data)

    if show_conv_yield:
        fund["conv_yield"] = calculate_convenience_yield(price_data, futures_data)

    if show_roll_yield:
        fund["roll_yield"] = calculate_roll_yield(price_data, futures_data)

    if show_breakeven and "production_cost" in info:
        fund["breakeven"] = calculate_producer_breakeven_heat(price_data, info["production_cost"])

    if show_demand_proxy:
        fund["demand_proxy"] = calculate_demand_proxy_index(price_data, ip_data, gdp_data)

    if show_geo_risk and not gold_data.empty and cat == "Energy":
        fund["geo_risk"] = calculate_geopolitical_risk_premium(price_data, gold_data)

    if show_hedge_score:
        fund["hedge_score"] = calculate_inflation_hedge_score(price_data, be_data, cpi_data)

    if show_inv_cycle and inventory_data:
        inv_key = list(inventory_data.keys())[0]
        if not inventory_data[inv_key].empty:
            fund["inv_cycle"] = calculate_inventory_cycle_position(inventory_data[inv_key])

    # ── Classic analytics ──
    classic = {}
    if show_z_score and len(price_data) >= 90:
        classic["z_score"] = calculate_z_score(price_data, window=min(90, len(price_data) // 2))
    if show_volatility and len(price_data) >= 30:
        classic["volatility"] = calculate_volatility_metrics(price_data)
    if show_seasonal and len(price_data) >= 104:
        classic["seasonal"] = calculate_seasonal_pattern(price_data)
    if show_crack and cat == "Energy":
        gasoline_info = COMMODITIES["Energy"]["Gasoline"]
        gasoline_data = get_fred_data_csv(gasoline_info["fred_code"], s_date, e_date)
        heating_data  = get_fred_data_csv(COMMODITIES["Energy"]["Heating_Oil"]["fred_code"], s_date, e_date)
        if gasoline_data.empty:
            gasoline_data = get_yfinance_price(gasoline_info["yf_ticker"], s_date, e_date)
        if heating_data.empty:
            heating_data = get_yfinance_price(COMMODITIES["Energy"]["Heating_Oil"]["yf_ticker"], s_date, e_date)
        classic["crack_spread"] = calculate_crack_spread(price_data, gasoline_data, heating_data)

    # ── STU ──
    stu_data = pd.DataFrame()
    if inventory_data and production_data:
        inv_key = list(inventory_data.keys())[0]
        if not inventory_data[inv_key].empty and not production_data.get("production", pd.DataFrame()).empty:
            stu_data = calculate_inventory_to_use_ratio(
                inventory_data[inv_key], production_data["production"]
            )

    # ─────────────────────────────────────────────
    # TOP METRICS ROW
    # ─────────────────────────────────────────────
    st.markdown("---")
    current_price = price_data.iloc[-1, 0]
    n = len(price_data)

    chg_1m = ((current_price / price_data.iloc[max(0, n - 21), 0]) - 1) * 100 if n > 1 else 0
    chg_1y = ((current_price / price_data.iloc[max(0, n - 252), 0]) - 1) * 100 if n > 252 else None

    c1, c2, c3, c4, c5, c6 = st.columns(6)

    with c1:
        st.metric("Price", f"{current_price:.2f} {info['unit']}", f"{chg_1m:+.1f}% (1M)")

    with c2:
        if "volatility" in classic:
            vol30 = classic["volatility"]["Current_Vol_30d"]
            vol_avg = classic["volatility"]["Avg_Vol_1Y"]
            st.metric("30D Volatility", f"{vol30:.1f}%", f"{vol30 - vol_avg:+.1f}% vs 1Y avg", delta_color="inverse")

    with c3:
        if "z_score" in classic and not classic["z_score"].empty:
            z_val = classic["z_score"].iloc[-1, 0]
            st.metric("Z-Score", f"{z_val:+.2f}",
                      "Overbought" if z_val > 2 else ("Oversold" if z_val < -2 else "Neutral"))

    with c4:
        if "roll_yield" in fund and not fund["roll_yield"].empty:
            ry = fund["roll_yield"]["Roll_Yield_%"].dropna()
            if len(ry) > 0:
                rv = ry.iloc[-1]
                st.metric("Roll Yield (Ann.)", f"{rv:+.1f}%",
                          "Backwardation ✓" if rv > 0 else "Contango ✗", delta_color="normal" if rv > 0 else "inverse")

    with c5:
        if "conv_yield" in fund and not fund["conv_yield"].empty:
            cy = fund["conv_yield"]["Convenience_Yield"].dropna()
            if len(cy) > 0:
                cyv = cy.iloc[-1] * 100
                st.metric("Convenience Yield", f"{cyv:.1f}%",
                          "Tight Market" if cyv > 2 else "Loose Market", delta_color="normal" if cyv > 2 else "inverse")

    with c6:
        if "hedge_score" in fund and not fund["hedge_score"].empty and "Hedge_Score" in fund["hedge_score"].columns:
            hs = fund["hedge_score"]["Hedge_Score"].dropna()
            if len(hs) > 0:
                hsv = hs.iloc[-1]
                st.metric("Inflation Hedge Score", f"{hsv:.0f}/100",
                          "Strong" if hsv > 65 else ("Moderate" if hsv > 40 else "Weak"))

    # ─────────────────────────────────────────────
    # SCORECARD
    # ─────────────────────────────────────────────
    if show_scorecard:
        with st.expander("🎯 Fundamental Scorecard — AI-synthesized outlook", expanded=True):
            z_val_sc    = classic["z_score"].iloc[-1, 0] if "z_score" in classic and not classic["z_score"].empty else None
            ptc_sc      = (current_price / info["production_cost"]) if "production_cost" in info else None
            ry_sc_val   = fund["roll_yield"]["Roll_Yield_%"].dropna().iloc[-1] if "roll_yield" in fund and not fund["roll_yield"].empty and len(fund["roll_yield"]["Roll_Yield_%"].dropna()) > 0 else None
            cy_sc       = fund["conv_yield"]["Convenience_Yield"].dropna().iloc[-1] if "conv_yield" in fund and not fund["conv_yield"].empty and len(fund["conv_yield"]["Convenience_Yield"].dropna()) > 0 else None
            stu_sc      = stu_data["STU_Months"].dropna().iloc[-1] if not stu_data.empty and len(stu_data["STU_Months"].dropna()) > 0 else None
            hs_sc       = fund["hedge_score"]["Hedge_Score"].dropna().iloc[-1] if "hedge_score" in fund and not fund["hedge_score"].empty and "Hedge_Score" in fund["hedge_score"].columns and len(fund["hedge_score"]["Hedge_Score"].dropna()) > 0 else None

            scorecard = score_fundamental_outlook(z_val_sc, ptc_sc, ry_sc_val, cy_sc, stu_sc, hs_sc)

            sc1, sc2 = st.columns([1, 2])
            with sc1:
                outlook = scorecard["outlook"]
                color_cls = (
                    "scorecard-bull" if "Bull" in outlook
                    else "scorecard-bear" if "Bear" in outlook
                    else "scorecard-neutral"
                )
                st.markdown(f"### Fundamental Outlook")
                st.markdown(f"<div class='{color_cls}' style='font-size:2rem;'>{outlook}</div>", unsafe_allow_html=True)
                st.progress(max(0.0, min(1.0, (scorecard["pct"] + 100) / 200)))
                st.caption(f"Score: {scorecard['total']:+d} / {scorecard['max_score']} ({scorecard['pct']:+.0f}%)")

                score_df = pd.DataFrame([
                    {"Factor": k.replace("_", " "), "Score": v}
                    for k, v in scorecard["scores"].items()
                ])
                if not score_df.empty:
                    fig_sc = go.Figure(go.Bar(
                        x=score_df["Score"],
                        y=score_df["Factor"],
                        orientation="h",
                        marker_color=[PALETTE["green"] if v >= 0 else PALETTE["red"] for v in score_df["Score"]],
                    ))
                    fig_sc.update_layout(
                        height=220, margin=dict(l=0, r=0, t=10, b=0),
                        template="plotly_white",
                        xaxis=dict(range=[-2.5, 2.5], tickvals=[-2, -1, 0, 1, 2]),
                        showlegend=False,
                    )
                    st.plotly_chart(fig_sc, use_container_width=True)

            with sc2:
                st.markdown("### Key Signals")
                for insight in scorecard["commentary"]:
                    st.markdown(f"- {insight}")
                if not scorecard["commentary"]:
                    st.info("Enable more fundamental modules to populate the scorecard.")

    # ─────────────────────────────────────────────
    # TABS
    # ─────────────────────────────────────────────
    tab_labels = [
        "📈 Price & Technicals",
        "💰 Real & FX-Adjusted Price",
        "🔄 Roll Yield & Term Structure",
        "🏭 Producer Economics",
        "📦 Supply Adequacy",
        "🌡️ Demand Proxy",
        "🛡️ Inflation Hedge",
        "🌐 Geo Risk Premium",
        "📊 Cross-Commodity Spreads",
        "📅 Seasonality",
    ]
    tabs = st.tabs(tab_labels)

    # ── Tab 0: Price & Technicals ──
    with tabs[0]:
        col_a, col_b = st.columns([3, 1])

        with col_a:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=price_data.index, y=price_data.iloc[:, 0],
                name="Price", line=dict(color=PALETTE["blue"], width=2),
            ))
            if len(price_data) >= 50:
                for w, color, label in [(20, PALETTE["orange"], "MA20"), (50, PALETTE["red"], "MA50")]:
                    ma = price_data.iloc[:, 0].rolling(w).mean()
                    fig.add_trace(go.Scatter(x=ma.index, y=ma, name=label,
                                            line=dict(color=color, width=1.5, dash="dash")))
            if "production_cost" in info:
                fig.add_hline(y=info["production_cost"], line_dash="dot", line_color=PALETTE["green"],
                              annotation_text="Avg Production Cost")
            fig.update_layout(title=f"{info['description']} ({data_source})",
                              yaxis_title=info["unit"], **LAYOUT_BASE)
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            st.subheader("Statistics")
            st.metric("Current", f"{current_price:.2f}")
            high_52 = price_data.iloc[-min(252, n):, 0].max()
            low_52  = price_data.iloc[-min(252, n):, 0].min()
            st.metric("52W High", f"{high_52:.2f}")
            st.metric("52W Low",  f"{low_52:.2f}")
            st.metric("Mean",     f"{price_data.iloc[:, 0].mean():.2f}")
            if chg_1y is not None:
                st.metric("1Y Return", f"{chg_1y:+.1f}%")
            if "volatility" in classic:
                st.metric("Max Drawdown", f"{classic['volatility']['Max_Drawdown']:.1f}%")
                st.metric("Sharpe Ratio",  f"{classic['volatility']['Sharpe_Ratio']:.2f}")

        if "z_score" in classic and not classic["z_score"].empty:
            fig_z = go.Figure()
            zd = classic["z_score"]
            fig_z.add_trace(go.Scatter(x=zd.index, y=zd.iloc[:, 0], name="Z-Score",
                                       line=dict(color=PALETTE["purple"], width=2), fill="tozeroy"))
            for y_val, color, label in [(2, "red", "Overbought +2σ"), (-2, "green", "Oversold -2σ")]:
                fig_z.add_hline(y=y_val, line_dash="dash", line_color=color, annotation_text=label)
            fig_z.add_hline(y=0, line_dash="dot", line_color=PALETTE["gray"])
            fig_z.update_layout(title="Z-Score (Mean Reversion Indicator)", yaxis_title="Std Devs", **LAYOUT_BASE)
            st.plotly_chart(fig_z, use_container_width=True)

        if "crack_spread" in classic and not classic["crack_spread"].empty:
            cs = classic["crack_spread"]
            fig_cs = go.Figure()
            fig_cs.add_trace(go.Scatter(x=cs.index, y=cs.iloc[:, 0], name="3-2-1 Crack Spread",
                                        line=dict(color=PALETTE["green"], width=2), fill="tozeroy"))
            fig_cs.add_hline(y=0, line_dash="dot", line_color=PALETTE["gray"])
            fig_cs.add_hline(y=15, line_dash="dash", line_color="darkgreen", annotation_text="Profitable threshold")
            fig_cs.update_layout(title="3-2-1 Crack Spread (Refinery Margin)", yaxis_title="$/Barrel", **LAYOUT_BASE)
            st.plotly_chart(fig_cs, use_container_width=True)

    # ── Tab 1: Real & FX-Adjusted Price ──
    with tabs[1]:
        st.subheader("💰 Real (Inflation-Adjusted) Price")
        st.caption("Removes CPI-driven price changes to reveal genuine supply/demand dynamics in constant dollars.")

        if "real_price" in fund and not fund["real_price"].empty:
            rp = fund["real_price"]
            fig_rp = go.Figure()
            fig_rp.add_trace(go.Scatter(x=rp.index, y=rp["Nominal"], name="Nominal Price",
                                        line=dict(color=PALETTE["blue"], width=2)))
            fig_rp.add_trace(go.Scatter(x=rp.index, y=rp["Real_Price"], name="Real Price (Today's $)",
                                        line=dict(color=PALETTE["orange"], width=2, dash="dash")))
            fig_rp.update_layout(title=f"{info['description']} — Nominal vs. Real Price",
                                 yaxis_title=info["unit"], **LAYOUT_BASE)
            st.plotly_chart(fig_rp, use_container_width=True)

            # Real price z-score
            rp_z = calculate_z_score(rp[["Real_Price"]], window=min(60, len(rp) // 2))
            if not rp_z.empty:
                fig_rpz = go.Figure()
                fig_rpz.add_trace(go.Scatter(x=rp_z.index, y=rp_z.iloc[:, 0], name="Real Price Z-Score",
                                             line=dict(color=PALETTE["teal"], width=2), fill="tozeroy"))
                fig_rpz.add_hline(y=2, line_dash="dash", line_color="red")
                fig_rpz.add_hline(y=-2, line_dash="dash", line_color="green")
                fig_rpz.update_layout(title="Real Price Z-Score (Removes Inflation Noise)",
                                      yaxis_title="Std Devs", **LAYOUT_BASE)
                st.plotly_chart(fig_rpz, use_container_width=True)
        else:
            st.info("CPI data unavailable. Enable FRED API for real price analysis.")

        st.subheader("💱 USD-Adjusted Price (DXY Stripped)")
        st.caption("Removes dollar strength/weakness effect. A commodity 'cheap in USD' may be fully valued in real trade-weighted terms.")

        if "usd_adj" in fund and not fund["usd_adj"].empty:
            ua = fund["usd_adj"]
            fig_ua = go.Figure()
            fig_ua.add_trace(go.Scatter(x=ua.index, y=ua["Nominal_Price"], name="USD Price",
                                        line=dict(color=PALETTE["blue"], width=2)))
            fig_ua.add_trace(go.Scatter(x=ua.index, y=ua["DXY_Adj_Price"], name="DXY-Adjusted Price",
                                        line=dict(color=PALETTE["pink"], width=2, dash="dash")))
            fig_ua.update_layout(title="Price Stripped of USD Effect", yaxis_title=info["unit"], **LAYOUT_BASE)
            st.plotly_chart(fig_ua, use_container_width=True)

            fig_corr = go.Figure()
            fig_corr.add_trace(go.Scatter(x=ua.index, y=ua["Rolling_Corr_DXY"], name="52W Rolling Corr (Price vs DXY)",
                                          line=dict(color=PALETTE["purple"], width=2), fill="tozeroy"))
            fig_corr.add_hline(y=0, line_dash="dot")
            fig_corr.update_layout(title="Rolling Correlation: Commodity Price vs. USD Index",
                                   yaxis_title="Correlation", yaxis_range=[-1, 1], **LAYOUT_BASE)
            st.plotly_chart(fig_corr, use_container_width=True)
            st.info("📌 Most commodities have negative DXY correlation (−0.4 to −0.8). When correlation weakens, demand factors dominate.")
        else:
            st.info("DXY data unavailable. Enable FRED API.")

    # ── Tab 2: Roll Yield & Term Structure ──
    with tabs[2]:
        st.subheader("🔄 Roll Yield & Term Structure")
        st.caption("""
        **Roll yield** is the profit or loss from rolling expiring futures contracts forward.
        In **backwardation** (spot > futures), longs earn a positive roll.
        In **contango** (futures > spot), longs suffer negative roll — a hidden cost that erodes commodity ETF returns.
        """)

        if "roll_yield" in fund and not fund["roll_yield"].empty:
            ry = fund["roll_yield"]
            fig_ry = make_subplots(rows=2, cols=1,
                                   subplot_titles=("Roll Yield (Annualized %)", "Spot vs Futures Proxy"),
                                   vertical_spacing=0.15)
            ryv = ry["Roll_Yield_%"]
            fig_ry.add_trace(go.Scatter(x=ryv.index, y=ryv, name="Roll Yield",
                                        line=dict(color=PALETTE["teal"], width=2), fill="tozeroy"), row=1, col=1)
            fig_ry.add_hline(y=0, line_dash="dot", line_color=PALETTE["gray"], row=1, col=1)
            fig_ry.add_trace(go.Scatter(x=ry.index, y=ry["Spot"], name="Spot",
                                        line=dict(color=PALETTE["blue"], width=2)), row=2, col=1)
            fig_ry.add_trace(go.Scatter(x=ry.index, y=ry["Futures_Front"], name="Futures (Front)",
                                        line=dict(color=PALETTE["orange"], width=2, dash="dash")), row=2, col=1)
            fig_ry.update_layout(height=600, template="plotly_white", hovermode="x unified",
                                 title_text="Roll Yield Analysis")
            fig_ry.update_yaxes(title_text="Roll Yield %", row=1, col=1)
            fig_ry.update_yaxes(title_text=info["unit"], row=2, col=1)
            st.plotly_chart(fig_ry, use_container_width=True)
        else:
            st.info("Roll yield requires futures data (Yahoo Finance fallback).")

        if "conv_yield" in fund and not fund["conv_yield"].empty:
            st.subheader("⚙️ Implied Convenience Yield")
            st.caption("Convenience yield is the implicit benefit of holding physical commodity inventory vs. a futures position. High CY = physical scarcity.")
            cy = fund["conv_yield"]
            fig_cy = go.Figure()
            cyp = cy["Convenience_Yield"] * 100
            fig_cy.add_trace(go.Scatter(x=cyp.index, y=cyp, name="Convenience Yield (%)",
                                        line=dict(color=PALETTE["lime"], width=2), fill="tozeroy"))
            fig_cy.add_hline(y=2,  line_dash="dash", line_color=PALETTE["green"],  annotation_text="Tight supply")
            fig_cy.add_hline(y=-2, line_dash="dash", line_color=PALETTE["red"],    annotation_text="Surplus / glut")
            fig_cy.add_hline(y=0,  line_dash="dot",  line_color=PALETTE["gray"])
            fig_cy.update_layout(title="Implied Convenience Yield (Physical Market Tightness)",
                                 yaxis_title="%", **LAYOUT_BASE)
            st.plotly_chart(fig_cy, use_container_width=True)

    # ── Tab 3: Producer Economics ──
    with tabs[3]:
        st.subheader("🏭 Producer Breakeven Tiers")
        st.caption("Three-tier framework: cash (variable) cost, full-cycle breakeven, premium economics threshold.")

        if "breakeven" in fund and not fund["breakeven"].empty:
            be = fund["breakeven"]
            fig_be = go.Figure()

            # Colored background zones
            fig_be.add_hrect(y0=-1e6,                  y1=be["Cash_Cost"].iloc[0],   fillcolor="#EF444420", line_width=0, annotation_text="Shut-in Zone", annotation_position="top left")
            fig_be.add_hrect(y0=be["Cash_Cost"].iloc[0], y1=be["Full_Cycle"].iloc[0],  fillcolor="#F59E0B20", line_width=0, annotation_text="Cash Positive")
            fig_be.add_hrect(y0=be["Full_Cycle"].iloc[0], y1=be["Premium"].iloc[0],    fillcolor="#10B98120", line_width=0, annotation_text="Full-Cycle OK")
            fig_be.add_hrect(y0=be["Premium"].iloc[0],   y1=1e6,                       fillcolor="#3B82F620", line_width=0, annotation_text="Premium Economics")

            fig_be.add_trace(go.Scatter(x=be.index, y=be["Price"], name="Price",
                                        line=dict(color=PALETTE["blue"], width=2.5)))
            for col, color, label in [
                ("Cash_Cost",  PALETTE["red"],    "Cash Cost"),
                ("Full_Cycle", PALETTE["orange"], "Full-Cycle Breakeven"),
                ("Premium",    PALETTE["green"],  "Premium Threshold"),
            ]:
                fig_be.add_hline(y=be[col].iloc[0], line_dash="dash", line_color=color,
                                 annotation_text=f"{label}: ${be[col].iloc[0]:.0f}")

            fig_be.update_layout(title="Producer Economics — Breakeven Tier Analysis",
                                 yaxis_title=info["unit"], **LAYOUT_BASE)
            st.plotly_chart(fig_be, use_container_width=True)

            # Margin % over time
            fig_mg = go.Figure()
            fig_mg.add_trace(go.Scatter(x=be.index, y=be["Margin_%"], name="Full-Cycle Margin %",
                                        fill="tozeroy", line=dict(color=PALETTE["teal"], width=2)))
            fig_mg.add_hline(y=0, line_dash="dot", line_color=PALETTE["gray"])
            fig_mg.add_hline(y=30, line_dash="dash", line_color=PALETTE["green"], annotation_text="Premium (+30%)")
            fig_mg.update_layout(title="Full-Cycle Margin Over Time (%)",
                                 yaxis_title="Margin vs Full-Cycle Cost (%)", **LAYOUT_BASE)
            st.plotly_chart(fig_mg, use_container_width=True)

            # Current zone callout
            current_zone = be["Zone"].dropna().iloc[-1] if not be["Zone"].dropna().empty else "N/A"
            zone_color = {
                "Shut-in Zone": "🔴",
                "Cash Positive": "🟡",
                "Full-Cycle Breakeven": "🟢",
                "Premium Economics": "💎",
            }.get(str(current_zone), "⚪")
            st.metric("Current Producer Zone", f"{zone_color} {current_zone}")
        else:
            st.info("No production cost defined for this commodity. Energy and metals support this feature.")

    # ── Tab 4: Supply Adequacy ──
    with tabs[4]:
        st.subheader("📦 Supply Adequacy")

        if "inv_cycle" in fund and not fund["inv_cycle"].empty:
            ic = fund["inv_cycle"]
            st.subheader("Inventory Seasonal Position (vs 5Y History)")
            st.caption("The shaded band shows ±1σ of historical inventory for each calendar week. Current level outside the band signals an unusual supply situation.")

            upper = ic["hist_mean"] + ic["hist_std"]
            lower = ic["hist_mean"] - ic["hist_std"]
            fig_ic = go.Figure()
            fig_ic.add_trace(go.Scatter(
                x=list(ic.index) + list(ic.index[::-1]),
                y=list(upper) + list(lower[::-1]),
                fill="toself", fillcolor=PALETTE["blue"] + "22",
                line=dict(width=0), name="±1σ Band",
            ))
            fig_ic.add_trace(go.Scatter(x=ic.index, y=ic["hist_mean"], name="5Y Avg",
                                        line=dict(color=PALETTE["gray"], width=1, dash="dash")))
            fig_ic.add_trace(go.Scatter(x=ic.index, y=ic["Current"], name="Current",
                                        line=dict(color=PALETTE["blue"], width=2.5)))
            fig_ic.update_layout(title="Inventory vs Seasonal Historical Band", **LAYOUT_BASE)
            st.plotly_chart(fig_ic, use_container_width=True)

            fig_pct = go.Figure()
            fig_pct.add_trace(go.Scatter(x=ic.index, y=ic["Pct_Rank"], name="Percentile Rank",
                                         fill="tozeroy", line=dict(color=PALETTE["purple"], width=2)))
            fig_pct.add_hline(y=80, line_dash="dash", line_color="red",   annotation_text="Surplus territory (>80th pct)")
            fig_pct.add_hline(y=20, line_dash="dash", line_color="green", annotation_text="Deficit territory (<20th pct)")
            fig_pct.update_layout(title="Inventory Percentile Rank (Historical)",
                                  yaxis_title="Percentile", yaxis_range=[0, 100], **LAYOUT_BASE)
            st.plotly_chart(fig_pct, use_container_width=True)
        else:
            st.info("Inventory cycle analysis requires EIA API access and Energy category selection.")

        if not stu_data.empty:
            st.subheader("Stocks-to-Use Ratio")
            st.caption("Months of production covered by existing inventory. Below 1.5M = tight; above 4M = oversupplied.")
            fig_stu = go.Figure()
            fig_stu.add_trace(go.Scatter(x=stu_data.index, y=stu_data["STU_Months"], name="STU (Months)",
                                         line=dict(color=PALETTE["orange"], width=2)))
            fig_stu.add_trace(go.Scatter(x=stu_data.index, y=stu_data["STU_MA3"], name="3M MA",
                                         line=dict(color=PALETTE["red"], width=1.5, dash="dash")))
            fig_stu.add_hline(y=1.5, line_dash="dash", line_color="green",  annotation_text="Tight (<1.5M)")
            fig_stu.add_hline(y=4.0, line_dash="dash", line_color="red",    annotation_text="Surplus (>4M)")
            fig_stu.update_layout(title="Stocks-to-Use Ratio (Months of Supply)", yaxis_title="Months", **LAYOUT_BASE)
            st.plotly_chart(fig_stu, use_container_width=True)
        else:
            st.info("Stocks-to-use requires EIA API access for inventory and production data.")

    # ── Tab 5: Demand Proxy ──
    with tabs[5]:
        st.subheader("🌡️ Demand Proxy Index")
        st.caption("Industrial production and GDP growth, normalized and overlaid with price. Reveals whether price moves are demand-driven or supply-driven.")

        if "demand_proxy" in fund and not fund["demand_proxy"].empty:
            dp = fund["demand_proxy"]

            fig_dp = make_subplots(rows=2, cols=1,
                                   subplot_titles=("Commodity Price (Monthly)", "Demand Proxy Index"),
                                   vertical_spacing=0.15)
            fig_dp.add_trace(go.Scatter(x=dp.index, y=dp["Price"], name="Price",
                                        line=dict(color=PALETTE["blue"], width=2)), row=1, col=1)

            if "Demand_Index" in dp.columns:
                fig_dp.add_trace(go.Scatter(x=dp.index, y=dp["Demand_Index"], name="Demand Index",
                                            line=dict(color=PALETTE["green"], width=2), fill="tozeroy"), row=2, col=1)
            if "IndProd_Norm" in dp.columns:
                fig_dp.add_trace(go.Scatter(x=dp.index, y=dp["IndProd_Norm"], name="Industrial Prod (norm)",
                                            line=dict(color=PALETTE["orange"], width=1.5, dash="dot")), row=2, col=1)
            if "GDP_Norm" in dp.columns:
                fig_dp.add_trace(go.Scatter(x=dp.index, y=dp["GDP_Norm"], name="GDP (norm)",
                                            line=dict(color=PALETTE["purple"], width=1.5, dash="dot")), row=2, col=1)

            fig_dp.add_hline(y=0, line_dash="dot", line_color=PALETTE["gray"], row=2, col=1)
            fig_dp.update_layout(height=650, template="plotly_white", hovermode="x unified")
            fig_dp.update_yaxes(title_text=info["unit"], row=1, col=1)
            fig_dp.update_yaxes(title_text="Normalized", row=2, col=1)
            st.plotly_chart(fig_dp, use_container_width=True)

            # Rolling correlation between price and demand index
            if "Demand_Index" in dp.columns:
                dp_clean = dp[["Price", "Demand_Index"]].dropna()
                if len(dp_clean) >= 24:
                    roll_corr = dp_clean["Price"].rolling(24).corr(dp_clean["Demand_Index"])
                    fig_rc = go.Figure()
                    fig_rc.add_trace(go.Scatter(x=roll_corr.index, y=roll_corr,
                                                name="24M Rolling Corr (Price vs Demand Index)",
                                                fill="tozeroy", line=dict(color=PALETTE["teal"], width=2)))
                    fig_rc.add_hline(y=0, line_dash="dot")
                    fig_rc.update_layout(title="Price—Demand Correlation (24M Rolling)",
                                         yaxis_title="Correlation", yaxis_range=[-1, 1], **LAYOUT_BASE)
                    st.plotly_chart(fig_rc, use_container_width=True)
                    st.info("📌 High positive correlation → demand-driven market. Low/negative correlation → supply shock or speculative episode.")
        else:
            st.info("Demand proxy requires Industrial Production data from FRED.")

    # ── Tab 6: Inflation Hedge ──
    with tabs[6]:
        st.subheader("🛡️ Inflation Hedge Score")
        st.caption("""
        Measures how reliably this commodity tracks inflation — both expected (breakeven) and realized (CPI).
        **Score 70–100**: Strong hedge.  **40–70**: Moderate.  **0–40**: Poor hedge (may even be pro-cyclical).
        """)

        if "hedge_score" in fund and not fund["hedge_score"].empty:
            hs = fund["hedge_score"]

            if "Hedge_Score" in hs.columns:
                fig_hs = go.Figure()
                fig_hs.add_trace(go.Scatter(x=hs.index, y=hs["Hedge_Score"], name="Hedge Score",
                                            fill="tozeroy", line=dict(color=PALETTE["lime"], width=2)))
                fig_hs.add_hline(y=70, line_dash="dash", line_color=PALETTE["green"], annotation_text="Strong Hedge")
                fig_hs.add_hline(y=40, line_dash="dash", line_color=PALETTE["orange"], annotation_text="Moderate Hedge")
                fig_hs.update_layout(title="Composite Inflation Hedge Score (0–100)",
                                     yaxis_title="Score", yaxis_range=[0, 100], **LAYOUT_BASE)
                st.plotly_chart(fig_hs, use_container_width=True)

            corr_cols = [c for c in hs.columns if c.startswith("Corr_")]
            if corr_cols:
                fig_corrs = go.Figure()
                colors_corr = [PALETTE["blue"], PALETTE["orange"]]
                for i, col in enumerate(corr_cols):
                    fig_corrs.add_trace(go.Scatter(x=hs.index, y=hs[col], name=col.replace("_", " "),
                                                   line=dict(color=colors_corr[i % 2], width=2)))
                fig_corrs.add_hline(y=0, line_dash="dot")
                fig_corrs.update_layout(title="Rolling Correlation: Price Return vs Inflation Measures",
                                        yaxis_title="Correlation", yaxis_range=[-1, 1], **LAYOUT_BASE)
                st.plotly_chart(fig_corrs, use_container_width=True)
        else:
            st.info("Inflation hedge score requires CPI/breakeven data from FRED.")

    # ── Tab 7: Geopolitical Risk Premium ──
    with tabs[7]:
        st.subheader("🌐 Geopolitical Risk Premium")
        st.caption("""
        Residual from regressing commodity price on a safe-haven proxy (Gold).
        A positive residual when gold is also rising = geopolitical *supply disruption* bid.
        A positive residual when gold is flat = demand surprise, not geopolitics.
        """)

        if "geo_risk" in fund and not fund["geo_risk"].empty:
            gr = fund["geo_risk"]
            fig_gr = make_subplots(rows=2, cols=1,
                                   subplot_titles=("Commodity Price", "Geopolitical Risk Premium (Regression Residual)"),
                                   vertical_spacing=0.15)
            fig_gr.add_trace(go.Scatter(x=gr.index, y=gr["Commodity_Price"], name="Price",
                                        line=dict(color=PALETTE["blue"], width=2)), row=1, col=1)
            fig_gr.add_trace(go.Scatter(x=gr.index, y=gr["Geo_Risk_Premium"], name="Geo Risk Premium",
                                        fill="tozeroy", line=dict(color=PALETTE["red"], width=2)), row=2, col=1)
            fig_gr.add_hline(y=0, line_dash="dot", line_color=PALETTE["gray"], row=2, col=1)
            fig_gr.update_layout(height=600, template="plotly_white", hovermode="x unified",
                                 title_text="Geopolitical Risk Premium — Gold-Adjusted")
            fig_gr.update_yaxes(title_text=info["unit"], row=1, col=1)
            fig_gr.update_yaxes(title_text=f"Premium ({info['unit']})", row=2, col=1)
            st.plotly_chart(fig_gr, use_container_width=True)
            st.info("📌 Large sustained premiums historically coincide with OPEC cuts, Middle East tensions, or major supply disruptions.")
        else:
            if cat != "Energy":
                st.info("Geopolitical risk premium is most meaningful for Energy commodities.")
            else:
                st.info("Enable the Geo Risk module in the sidebar and ensure Gold data is available.")

    # ── Tab 8: Cross-Commodity Spreads ──
    with tabs[8]:
        st.subheader("📊 Cross-Commodity Spread Analysis")
        st.caption("Spreads and ratios between related commodities reveal relative value, substitution effects, and market structure.")

        if show_spreads:
            spread_tabs = st.tabs(list(SPREAD_PAIRS.keys()))
            for i, (spread_name, (a_key, b_key, a_cat, b_cat)) in enumerate(SPREAD_PAIRS.items()):
                with spread_tabs[i]:
                    a_info = COMMODITIES[a_cat][a_key]
                    b_info = COMMODITIES[b_cat][b_key]
                    is_ratio = "GSR" in spread_name or "Ratio" in spread_name

                    with st.spinner(f"Loading {spread_name}…"):
                        pa = get_fred_data_csv(a_info["fred_code"], s_date, e_date)
                        if pa.empty and "yf_ticker" in a_info:
                            pa = get_yfinance_price(a_info["yf_ticker"], s_date, e_date)
                        pb = get_fred_data_csv(b_info["fred_code"], s_date, e_date)
                        if pb.empty and "yf_ticker" in b_info:
                            pb = get_yfinance_price(b_info["yf_ticker"], s_date, e_date)

                    sp = calculate_cross_commodity_spread(pa, pb, a_key, b_key, ratio_mode=is_ratio)
                    if not sp.empty:
                        label = f"{a_key}/{b_key} Ratio" if is_ratio else f"{a_key}−{b_key} Spread"
                        fig_sp = make_subplots(rows=2, cols=1,
                                               subplot_titles=(label, "Z-Score vs 52W Avg"),
                                               vertical_spacing=0.15)
                        fig_sp.add_trace(go.Scatter(x=sp.index, y=sp["Spread"], name=label,
                                                    line=dict(color=PALETTE["blue"], width=2)), row=1, col=1)
                        fig_sp.add_trace(go.Scatter(x=sp.index, y=sp["Spread_MA52"], name="52W MA",
                                                    line=dict(color=PALETTE["orange"], width=1.5, dash="dash")), row=1, col=1)
                        fig_sp.add_trace(go.Scatter(x=sp.index, y=sp["Spread_Zscore"], name="Z-Score",
                                                    fill="tozeroy", line=dict(color=PALETTE["purple"], width=2)), row=2, col=1)
                        fig_sp.add_hline(y=2, line_dash="dash", line_color="red",   row=2, col=1)
                        fig_sp.add_hline(y=-2, line_dash="dash", line_color="green", row=2, col=1)
                        fig_sp.add_hline(y=0, line_dash="dot",  line_color=PALETTE["gray"], row=2, col=1)
                        fig_sp.update_layout(height=600, template="plotly_white", hovermode="x unified")
                        st.plotly_chart(fig_sp, use_container_width=True)

                        current_z = sp["Spread_Zscore"].dropna().iloc[-1] if len(sp["Spread_Zscore"].dropna()) > 0 else None
                        if current_z is not None:
                            if abs(current_z) > 2:
                                direction = "extended" if current_z > 2 else "compressed"
                                st.warning(f"⚠️ {spread_name} is statistically {direction} (Z = {current_z:.1f}) — potential mean-reversion trade")
                            else:
                                st.success(f"✅ {spread_name} trading near historical norms (Z = {current_z:.1f})")
                    else:
                        st.info(f"Could not load both legs of {spread_name}.")
        else:
            st.info("Enable Cross-Commodity Spreads in the sidebar.")

    # ── Tab 9: Seasonality ──
    with tabs[9]:
        st.subheader("📅 Seasonal Decomposition")
        if "seasonal" in classic and not classic["seasonal"].empty:
            sd = classic["seasonal"]
            fig_sd = make_subplots(rows=4, cols=1,
                                   subplot_titles=["Original", "Trend", "Seasonal Component", "Residual"],
                                   vertical_spacing=0.08)
            for row_idx, (col, color) in enumerate(
                [("Original", PALETTE["blue"]), ("Trend", PALETTE["red"]),
                 ("Seasonal", PALETTE["green"]), ("Residual", PALETTE["gray"])], start=1
            ):
                fig_sd.add_trace(go.Scatter(x=sd.index, y=sd[col], name=col,
                                            line=dict(color=color, width=1.5), showlegend=False),
                                 row=row_idx, col=1)
            fig_sd.update_layout(height=900, template="plotly_white", hovermode="x unified",
                                 title_text="Time Series Decomposition (Moving Average)")
            st.plotly_chart(fig_sd, use_container_width=True)

            # Average weekly seasonal pattern
            st.subheader("Average Weekly Seasonal Pattern")
            seasonal_avg = sd["Seasonal"].groupby(sd.index.isocalendar().week.astype(int)).mean()
            fig_wav = go.Figure()
            fig_wav.add_trace(go.Bar(x=seasonal_avg.index, y=seasonal_avg.values,
                                     marker_color=[PALETTE["green"] if v >= 0 else PALETTE["red"] for v in seasonal_avg.values],
                                     name="Avg Seasonal Component"))
            fig_wav.update_layout(title="Average Seasonal Component by Week of Year",
                                  xaxis_title="Week", yaxis_title=info["unit"], **LAYOUT_BASE)
            st.plotly_chart(fig_wav, use_container_width=True)
        else:
            st.info("Seasonal decomposition requires at least 2 years of weekly data.")

    # ── Export ──
    st.markdown("---")
    st.header("💾 Export")
    export = {"Price": price_data}
    if "real_price" in fund and not fund["real_price"].empty:
        export["Real_Price"] = fund["real_price"]
    if "roll_yield" in fund and not fund["roll_yield"].empty:
        export["Roll_Yield"] = fund["roll_yield"]
    if "conv_yield" in fund and not fund["conv_yield"].empty:
        export["Convenience_Yield"] = fund["conv_yield"]
    if "breakeven" in fund and not fund["breakeven"].empty:
        export["Producer_Breakeven"] = fund["breakeven"][["Price", "Margin_%"]]
    if "hedge_score" in fund and not fund["hedge_score"].empty:
        export["Inflation_Hedge"] = fund["hedge_score"]

    cols_ex = st.columns(min(len(export), 5))
    for idx, (name, df) in enumerate(export.items()):
        with cols_ex[idx % len(cols_ex)]:
            st.download_button(
                f"📥 {name}",
                df.to_csv(),
                file_name=f"{comm_key}_{name}_{s_date}.csv",
                mime="text/csv",
            )


if __name__ == "__main__":
    commodities_module()
