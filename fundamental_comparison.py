import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Tuple
from datetime import date, datetime
import warnings
warnings.filterwarnings('ignore')


def render_fundamental_comparison(tickers: List[str] = None) -> None:
    """
    Enhanced fundamental comparison with comprehensive analysis and ranking system.
    
    Args:
        tickers: List of ticker symbols to compare (optional, uses UI input if None)
    """
    st.markdown("# 📊 Advanced Fundamental Analysis & Comparison")
    st.markdown("Compare financial metrics, analyze trends, and rank stocks by fundamental strength")
    
    # ══════════════════════════════════════════════════════════════════════════
    # INPUT SECTION
    # ══════════════════════════════════════════════════════════════════════════
    with st.expander("⚙️ Configuration", expanded=True):
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            default_tickers = ", ".join(tickers) if tickers else "AAPL, MSFT, GOOGL, AMZN, META"
            ticker_input = st.text_input(
                "Stock Tickers (comma-separated)",
                value=default_tickers,
                help="Enter 2-15 ticker symbols separated by commas"
            )
            tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
        
        with col2:
            frequency = st.selectbox(
                "Data Frequency",
                ["Annual", "Quarterly"],
                index=0,
                help="Choose annual or quarterly financial statements"
            )
        
        with col3:
            compare_btn = st.button(
                "🚀 Analyze",
                type="primary",
                use_container_width=True
            )
    
    # Validation
    if len(tickers) < 2:
        st.info("📌 Enter at least 2 tickers to compare fundamental metrics")
        return
    
    if len(tickers) > 15:
        st.warning("⚠️ Maximum 15 tickers supported. Using first 15.")
        tickers = tickers[:15]
    
    if not compare_btn:
        st.info("👆 Click 'Analyze' to load and compare fundamental data")
        return
    
    # ══════════════════════════════════════════════════════════════════════════
    # DATA FETCHING
    # ══════════════════════════════════════════════════════════════════════════
    with st.spinner(f"📥 Fetching fundamental data for {len(tickers)} stocks..."):
        try:
            is_annual = (frequency == "Annual")
            all_data = {}
            failed_tickers = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, ticker in enumerate(tickers):
                status_text.text(f"Loading {ticker}... ({idx+1}/{len(tickers)})")
                progress_bar.progress((idx + 1) / len(tickers))
                
                try:
                    ticker_obj = yf.Ticker(ticker)
                    
                    # Fetch financial statements
                    if is_annual:
                        income = ticker_obj.income_stmt
                        balance = ticker_obj.balance_sheet
                        cashflow = ticker_obj.cashflow
                    else:
                        income = ticker_obj.quarterly_income_stmt
                        balance = ticker_obj.quarterly_balance_sheet
                        cashflow = ticker_obj.quarterly_cashflow
                    
                    # Check for empty data
                    if income.empty or balance.empty or cashflow.empty:
                        failed_tickers.append(ticker)
                        continue
                    
                    # Get info with fallback
                    try:
                        info = ticker_obj.info
                        if not info or not isinstance(info, dict):
                            info = {}
                    except Exception:
                        info = {}
                    
                    all_data[ticker] = {
                        "income": income,
                        "balance": balance,
                        "cashflow": cashflow,
                        "info": info
                    }
                    
                except Exception as e:
                    failed_tickers.append(ticker)
                    st.warning(f"⚠️ Failed to fetch {ticker}: {str(e)}")
            
            progress_bar.empty()
            status_text.empty()
            
            if not all_data:
                st.error("❌ No data retrieved. Please check ticker symbols and try again.")
                return
            
            tickers = list(all_data.keys())
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.success(f"✅ Successfully loaded data for **{len(tickers)} stocks**: {', '.join(tickers)}")
            with col2:
                if failed_tickers:
                    st.error(f"❌ Failed: {', '.join(failed_tickers)}")
            
            # ══════════════════════════════════════════════════════════════════
            # HELPER FUNCTIONS
            # ══════════════════════════════════════════════════════════════════
            def safe_get(df, key, default=None):
                """Get row from DataFrame with fallback for alternative names"""
                if df.empty:
                    return pd.Series(dtype=float)
                
                if key in df.index:
                    return df.loc[key]
                
                # Alternative field names mapping
                alternatives = {
                    "Total Revenue": ["TotalRevenue", "Total Revenues", "Revenue"],
                    "Gross Profit": ["GrossProfit", "Gross Income"],
                    "Operating Income": ["OperatingIncome", "EBIT", "Operating Income Loss"],
                    "Net Income": ["NetIncome", "Net Income Common Stockholders", "Net Income Loss"],
                    "EBITDA": ["EBITDA", "Normalized EBITDA"],
                    "Operating Cash Flow": ["OperatingCashFlow", "Total Cash From Operating Activities", "Cash Flow From Continuing Operating Activities"],
                    "Capital Expenditure": ["CapitalExpenditure", "Capital Expenditures"],
                    "Free Cash Flow": ["FreeCashFlow", "Free Cash Flow"],
                    "Total Assets": ["TotalAssets", "Total Assets"],
                    "Total Debt": ["TotalDebt", "Long Term Debt", "Total Debt And Capital Lease Obligation"],
                    "Cash And Cash Equivalents": ["CashAndCashEquivalents", "Cash", "Cash Cash Equivalents And Short Term Investments"],
                    "Total Stockholder Equity": ["TotalStockholderEquity", "Stockholders Equity", "Total Equity Gross Minority Interest"],
                    "Diluted Average Shares": ["DilutedAverageShares", "Average Shares Diluted"],
                    "Basic Average Shares": ["BasicAverageShares", "Ordinary Shares Number"],
                    "Current Assets": ["CurrentAssets", "Total Current Assets"],
                    "Current Liabilities": ["CurrentLiabilities", "Total Current Liabilities"],
                    "Total Liabilities": ["TotalLiabilities", "Total Liabilities Net Minority Interest"]
                }
                
                for alt in alternatives.get(key, []):
                    if alt in df.index:
                        return df.loc[alt]
                
                if default is not None:
                    return pd.Series(default, index=df.columns if not df.empty else [])
                return pd.Series(dtype=float)
            
            def get_latest_value(series):
                """Get most recent non-null value"""
                if series.empty:
                    return np.nan
                valid = series.dropna()
                return valid.iloc[-1] if len(valid) > 0 else np.nan
            
            def safe_divide(numerator, denominator):
                """Safe division with NaN handling"""
                if pd.isna(numerator) or pd.isna(denominator):
                    return np.nan
                if denominator == 0:
                    return np.nan
                return numerator / denominator
            
            def calculate_growth_rate(series, periods=None):
                """Calculate average growth rate from time series"""
                if series.empty:
                    return np.nan
                
                valid = series.dropna()
                if len(valid) < 2:
                    return np.nan
                
                if periods:
                    valid = valid.tail(periods)
                
                return valid.pct_change().mean() * 100
            
            def calculate_cagr(series):
                """Calculate Compound Annual Growth Rate"""
                if series.empty:
                    return np.nan
                
                valid = series.dropna()
                if len(valid) < 2:
                    return np.nan
                
                start_val = valid.iloc[0]
                end_val = valid.iloc[-1]
                periods = len(valid) - 1
                
                if start_val <= 0:
                    return np.nan
                
                return ((end_val / start_val) ** (1 / periods) - 1) * 100
            
            # ══════════════════════════════════════════════════════════════════
            # METRIC CALCULATION
            # ══════════════════════════════════════════════════════════════════
            def calculate_metrics(ticker_data: Dict) -> Dict:
                """Calculate comprehensive metrics for a ticker"""
                income = ticker_data["income"]
                balance = ticker_data["balance"]
                cashflow = ticker_data["cashflow"]
                info = ticker_data["info"]
                
                # Extract core financial data
                revenue = safe_get(income, "Total Revenue")
                gross_profit = safe_get(income, "Gross Profit")
                operating_income = safe_get(income, "Operating Income")
                net_income = safe_get(income, "Net Income")
                ebitda = safe_get(income, "EBITDA")
                
                ocf = safe_get(cashflow, "Operating Cash Flow")
                capex = safe_get(cashflow, "Capital Expenditure", 0)
                fcf_calc = ocf + capex  # capex is usually negative
                
                total_assets = safe_get(balance, "Total Assets")
                current_assets = safe_get(balance, "Current Assets")
                current_liabilities = safe_get(balance, "Current Liabilities")
                total_liabilities = safe_get(balance, "Total Liabilities")
                total_debt = safe_get(balance, "Total Debt", 0)
                cash = safe_get(balance, "Cash And Cash Equivalents", 0)
                equity = safe_get(balance, "Total Stockholder Equity")
                
                shares = safe_get(income, "Diluted Average Shares")
                if shares.empty:
                    shares = safe_get(income, "Basic Average Shares")
                
                # Get latest values
                rev_latest = get_latest_value(revenue)
                gp_latest = get_latest_value(gross_profit)
                oi_latest = get_latest_value(operating_income)
                ni_latest = get_latest_value(net_income)
                ebitda_latest = get_latest_value(ebitda)
                ocf_latest = get_latest_value(ocf)
                fcf_latest = get_latest_value(fcf_calc)
                assets_latest = get_latest_value(total_assets)
                curr_assets_latest = get_latest_value(current_assets)
                curr_liab_latest = get_latest_value(current_liabilities)
                total_liab_latest = get_latest_value(total_liabilities)
                debt_latest = get_latest_value(total_debt)
                cash_latest = get_latest_value(cash)
                equity_latest = get_latest_value(equity)
                shares_latest = get_latest_value(shares)
                
                # Market data
                market_cap = info.get("marketCap", np.nan)
                enterprise_value = info.get("enterpriseValue", np.nan)
                current_price = info.get("currentPrice", info.get("regularMarketPrice", info.get("previousClose", np.nan)))
                
                # Debug: Print what we're getting
                # st.write(f"Debug {ticker}: Price={current_price}, MktCap={market_cap}, EPS={eps}")
                
                # Profitability Metrics
                gross_margin = safe_divide(gp_latest, rev_latest) * 100
                operating_margin = safe_divide(oi_latest, rev_latest) * 100
                net_margin = safe_divide(ni_latest, rev_latest) * 100
                ebitda_margin = safe_divide(ebitda_latest, rev_latest) * 100
                
                roe = safe_divide(ni_latest, equity_latest) * 100
                roa = safe_divide(ni_latest, assets_latest) * 100
                roic = safe_divide(oi_latest * 0.65, equity_latest + debt_latest) * 100  # Approximate ROIC
                
                # Liquidity & Solvency
                current_ratio = safe_divide(curr_assets_latest, curr_liab_latest)
                quick_ratio = safe_divide(curr_assets_latest - (curr_assets_latest * 0.3), curr_liab_latest)  # Approximate
                debt_to_equity = safe_divide(debt_latest, equity_latest)
                debt_to_assets = safe_divide(debt_latest, assets_latest)
                equity_ratio = safe_divide(equity_latest, assets_latest)
                
                # Per Share Metrics
                eps = safe_divide(ni_latest, shares_latest)
                revenue_per_share = safe_divide(rev_latest, shares_latest)
                fcf_per_share = safe_divide(fcf_latest, shares_latest)
                book_value_per_share = safe_divide(equity_latest, shares_latest)
                
                # Valuation Metrics - Use Finviz as fallback
                # Try multiple sources for P/E: yfinance -> Finviz -> calculated
                pe_ratio = info.get("trailingPE", info.get("forwardPE", np.nan))
                if pd.isna(pe_ratio) or pe_ratio <= 0 or pe_ratio > 1000:
                    # Try Finviz
                    if False and pd.notna(info.get('pe')):
                        pe_ratio = info['pe']
                    else:
                        # Calculate from price and EPS if available
                        if not pd.isna(current_price) and not pd.isna(eps) and eps > 0:
                            pe_ratio = current_price / eps
                
                # Try multiple sources for P/B: yfinance -> Finviz -> calculated
                pb_ratio = info.get("priceToBook", np.nan)
                if pd.isna(pb_ratio) or pb_ratio <= 0 or pb_ratio > 100:
                    # Try Finviz
                    if False and pd.notna(info.get('pb')):
                        pb_ratio = info['pb']
                    else:
                        # Calculate if we have market cap and equity
                        if not pd.isna(market_cap) and not pd.isna(equity_latest) and equity_latest > 0:
                            pb_ratio = market_cap / equity_latest
                
                # Try multiple sources for P/S: yfinance -> Finviz -> calculated
                ps_ratio = info.get("priceToSalesTrailing12Months", np.nan)
                if pd.isna(ps_ratio) or ps_ratio <= 0 or ps_ratio > 100:
                    # Try Finviz
                    if False and pd.notna(info.get('ps')):
                        ps_ratio = info['ps']
                    else:
                        if not pd.isna(market_cap) and not pd.isna(rev_latest) and rev_latest > 0:
                            ps_ratio = market_cap / rev_latest
                
                # Calculate Price/FCF
                price_to_fcf = np.nan
                if not pd.isna(market_cap) and not pd.isna(fcf_latest) and fcf_latest > 0:
                    price_to_fcf = market_cap / fcf_latest
                
                # Calculate EV metrics
                if pd.isna(enterprise_value) or enterprise_value <= 0:
                    if not pd.isna(market_cap) and not pd.isna(debt_latest) and not pd.isna(cash_latest):
                        enterprise_value = market_cap + debt_latest - cash_latest
                
                ev_to_revenue = np.nan
                if not pd.isna(enterprise_value) and not pd.isna(rev_latest) and rev_latest > 0:
                    ev_to_revenue = enterprise_value / rev_latest
                
                # Try to get EV/EBITDA from info first, then calculate
                ev_to_ebitda = info.get("enterpriseToEbitda", np.nan)
                if pd.isna(ev_to_ebitda) or ev_to_ebitda <= 0 or ev_to_ebitda > 100:
                    if not pd.isna(enterprise_value) and not pd.isna(ebitda_latest) and ebitda_latest > 0:
                        ev_to_ebitda = enterprise_value / ebitda_latest
                
                # Efficiency Metrics
                asset_turnover = safe_divide(rev_latest, assets_latest)
                
                # Cash Flow Quality
                fcf_to_ni = safe_divide(fcf_latest, ni_latest) if ni_latest and ni_latest > 0 else np.nan
                ocf_to_ni = safe_divide(ocf_latest, ni_latest) if ni_latest and ni_latest > 0 else np.nan
                fcf_margin = safe_divide(fcf_latest, rev_latest) * 100
                
                # Growth Metrics (MUST be calculated before PEG)
                revenue_growth = calculate_growth_rate(revenue)
                revenue_cagr = calculate_cagr(revenue)
                ni_growth = calculate_growth_rate(net_income)
                ni_cagr = calculate_cagr(net_income)
                fcf_growth = calculate_growth_rate(fcf_calc)
                eps_growth = calculate_growth_rate(net_income / shares) if not shares.empty else np.nan
                
                # PEG - Calculate AFTER growth metrics are defined
                peg_ratio = np.nan
                if False and pd.notna(info.get('peg')):
                    peg_ratio = info['peg']
                else:
                    # Calculate PEG Ratio (now that revenue_growth is defined)
                    if not pd.isna(pe_ratio) and not pd.isna(revenue_growth) and revenue_growth > 0:
                        peg_ratio = pe_ratio / revenue_growth
                
                # Time series for trends
                gross_margin_ts = (gross_profit / revenue * 100) if not revenue.empty else pd.Series()
                operating_margin_ts = (operating_income / revenue * 100) if not revenue.empty else pd.Series()
                net_margin_ts = (net_income / revenue * 100) if not revenue.empty else pd.Series()
                roe_ts = (net_income / equity * 100) if not equity.empty else pd.Series()
                
                return {
                    # Core Financials
                    "Revenue": rev_latest,
                    "Gross Profit": gp_latest,
                    "Operating Income": oi_latest,
                    "Net Income": ni_latest,
                    "EBITDA": ebitda_latest,
                    "Operating CF": ocf_latest,
                    "Free Cash Flow": fcf_latest,
                    "Total Assets": assets_latest,
                    "Total Debt": debt_latest,
                    "Cash": cash_latest,
                    "Equity": equity_latest,
                    "Market Cap": market_cap,
                    
                    # Profitability
                    "Gross Margin %": gross_margin,
                    "Operating Margin %": operating_margin,
                    "Net Margin %": net_margin,
                    "EBITDA Margin %": ebitda_margin,
                    "ROE %": roe,
                    "ROA %": roa,
                    "ROIC %": roic,
                    
                    # Liquidity & Solvency
                    "Current Ratio": current_ratio,
                    "Quick Ratio": quick_ratio,
                    "Debt/Equity": debt_to_equity,
                    "Debt/Assets": debt_to_assets,
                    "Equity Ratio": equity_ratio,
                    
                    # Per Share
                    "EPS": eps,
                    "Revenue/Share": revenue_per_share,
                    "FCF/Share": fcf_per_share,
                    "Book Value/Share": book_value_per_share,
                    
                    # Valuation
                    "P/E": pe_ratio,
                    "P/B": pb_ratio,
                    "P/S": ps_ratio,
                    "Price/FCF": price_to_fcf,
                    "EV/Revenue": ev_to_revenue,
                    "EV/EBITDA": ev_to_ebitda,
                    "PEG": peg_ratio,
                    
                    # Efficiency
                    "Asset Turnover": asset_turnover,
                    
                    # Cash Flow Quality
                    "FCF/NI": fcf_to_ni,
                    "OCF/NI": ocf_to_ni,
                    "FCF Margin %": fcf_margin,
                    
                    # Growth
                    "Revenue Growth %": revenue_growth,
                    "Revenue CAGR %": revenue_cagr,
                    "NI Growth %": ni_growth,
                    "NI CAGR %": ni_cagr,
                    "FCF Growth %": fcf_growth,
                    "EPS Growth %": eps_growth,
                    
                    # Time Series (for charts)
                    "_revenue_ts": revenue,
                    "_net_income_ts": net_income,
                    "_ocf_ts": ocf,
                    "_fcf_ts": fcf_calc,
                    "_gross_margin_ts": gross_margin_ts,
                    "_operating_margin_ts": operating_margin_ts,
                    "_net_margin_ts": net_margin_ts,
                    "_roe_ts": roe_ts,
                }
            
            # Calculate metrics for all tickers
            with st.spinner("📊 Calculating financial metrics..."):
                metrics_dict = {}
                for ticker in tickers:
                    metrics_dict[ticker] = calculate_metrics(all_data[ticker])
                
                comparison_df = pd.DataFrame(metrics_dict).T
            
            # ══════════════════════════════════════════════════════════════════
            # RANKING SECTION (NEW)
            # ══════════════════════════════════════════════════════════════════
            def calculate_fundamental_score(df: pd.DataFrame) -> pd.DataFrame:
                """
                Calculate comprehensive fundamental score and ranking.
                Higher score = Better fundamentals
                """
                scores = pd.DataFrame(index=df.index)
                
                # Define scoring criteria with weights
                criteria = {
                    # Profitability (40%)
                    "ROE %": {"weight": 0.10, "higher_better": True, "min_threshold": 0},
                    "Net Margin %": {"weight": 0.10, "higher_better": True, "min_threshold": 0},
                    "Operating Margin %": {"weight": 0.10, "higher_better": True, "min_threshold": 0},
                    "ROIC %": {"weight": 0.10, "higher_better": True, "min_threshold": 0},
                    
                    # Growth (25%)
                    "Revenue Growth %": {"weight": 0.10, "higher_better": True, "min_threshold": -10},
                    "NI Growth %": {"weight": 0.08, "higher_better": True, "min_threshold": -20},
                    "EPS Growth %": {"weight": 0.07, "higher_better": True, "min_threshold": -20},
                    
                    # Financial Health (20%)
                    "Current Ratio": {"weight": 0.05, "higher_better": True, "min_threshold": 0.5},
                    "Debt/Equity": {"weight": 0.08, "higher_better": False, "max_threshold": 5},
                    "FCF/NI": {"weight": 0.07, "higher_better": True, "min_threshold": 0},
                    
                    # Valuation (15%) - Made optional since they might be missing
                    "P/E": {"weight": 0.05, "higher_better": False, "max_threshold": 100, "optional": True},
                    "PEG": {"weight": 0.05, "higher_better": False, "max_threshold": 5, "optional": True},
                    "EV/EBITDA": {"weight": 0.05, "higher_better": False, "max_threshold": 50, "optional": True},
                }
                
                # Check which valuation metrics are available
                valuation_available = {}
                for metric in ["P/E", "PEG", "EV/EBITDA"]:
                    if metric in df.columns:
                        valid_count = df[metric].notna().sum()
                        valuation_available[metric] = valid_count
                
                # If no valuation data, redistribute weights
                if sum(valuation_available.values()) == 0:
                    # Remove valuation criteria and redistribute weights
                    for key in ["P/E", "PEG", "EV/EBITDA"]:
                        if key in criteria:
                            del criteria[key]
                    
                    # Recalculate weights (distribute evenly)
                    total_remaining_weight = sum(c["weight"] for c in criteria.values())
                    for key in criteria:
                        criteria[key]["weight"] = criteria[key]["weight"] / total_remaining_weight
                
                # Normalize and score each metric
                total_weight = sum(c["weight"] for c in criteria.values())
                
                for metric, params in criteria.items():
                    if metric not in df.columns:
                        continue
                    
                    values = df[metric].copy()
                    
                    # Filter out invalid values
                    if params["higher_better"]:
                        if "min_threshold" in params:
                            values = values.where(values >= params["min_threshold"], np.nan)
                    else:
                        if "max_threshold" in params:
                            values = values.where(values <= params["max_threshold"], np.nan)
                    
                    # Skip if no valid values
                    if values.isna().all():
                        scores[metric] = 0
                        continue
                    
                    # Normalize to 0-100 scale
                    min_val = values.min()
                    max_val = values.max()
                    
                    if min_val == max_val:
                        normalized = pd.Series(50, index=values.index)
                    else:
                        if params["higher_better"]:
                            normalized = (values - min_val) / (max_val - min_val) * 100
                        else:
                            normalized = (max_val - values) / (max_val - min_val) * 100
                    
                    # Apply weight
                    scores[metric] = normalized * params["weight"] / total_weight * 100
                
                # Calculate total score
                scores["Total Score"] = scores.sum(axis=1)
                
                # Create grade
                def assign_grade(score):
                    if pd.isna(score):
                        return "N/A"
                    elif score >= 80:
                        return "A+"
                    elif score >= 75:
                        return "A"
                    elif score >= 70:
                        return "A-"
                    elif score >= 65:
                        return "B+"
                    elif score >= 60:
                        return "B"
                    elif score >= 55:
                        return "B-"
                    elif score >= 50:
                        return "C+"
                    elif score >= 45:
                        return "C"
                    elif score >= 40:
                        return "C-"
                    elif score >= 35:
                        return "D+"
                    elif score >= 30:
                        return "D"
                    else:
                        return "F"
                
                scores["Grade"] = scores["Total Score"].apply(assign_grade)
                scores["Rank"] = scores["Total Score"].rank(ascending=False, method='min').astype(int)
                
                return scores
            
            # ══════════════════════════════════════════════════════════════════
            # TABS LAYOUT
            # ══════════════════════════════════════════════════════════════════
            tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
                "🏆 Rankings",
                "📊 Overview",
                "💰 Profitability",
                "💵 Cash Flow",
                "🏦 Financial Health",
                "📈 Valuation",
                "📉 Trends",
                "📥 Export"
            ])
            
            # ══════════════════════════════════════════════════════════════════
            # TAB 1: RANKINGS (NEW)
            # ══════════════════════════════════════════════════════════════════
            with tab1:
                st.markdown("## 🏆 Fundamental Strength Rankings")
                st.markdown("Comprehensive scoring based on profitability, growth, financial health, and valuation")
                
                # Calculate scores
                with st.spinner("📊 Calculating fundamental scores..."):
                    scores_df = calculate_fundamental_score(comparison_df)
                
                # Combine with key metrics
                ranking_display = pd.DataFrame({
                    "Rank": scores_df["Rank"],
                    "Score": scores_df["Total Score"],
                    "Grade": scores_df["Grade"],
                    "ROE %": comparison_df["ROE %"],
                    "Net Margin %": comparison_df["Net Margin %"],
                    "Revenue Growth %": comparison_df["Revenue Growth %"],
                    "Debt/Equity": comparison_df["Debt/Equity"],
                    "P/E": comparison_df["P/E"],
                    "Market Cap": comparison_df["Market Cap"]
                }).sort_values("Rank")
                
                # Display ranking table
                st.markdown("### 📋 Overall Rankings")
                
                def highlight_rank(row):
                    if row["Rank"] == 1:
                        return ['background-color: #FFD700'] * len(row)  # Gold
                    elif row["Rank"] == 2:
                        return ['background-color: #C0C0C0'] * len(row)  # Silver
                    elif row["Rank"] == 3:
                        return ['background-color: #CD7F32'] * len(row)  # Bronze
                    else:
                        return [''] * len(row)
                
                st.dataframe(
                    ranking_display.style
                    .format({
                        "Score": "{:.1f}",
                        "ROE %": "{:.2f}",
                        "Net Margin %": "{:.2f}",
                        "Revenue Growth %": "{:.2f}",
                        "Debt/Equity": "{:.2f}",
                        "P/E": "{:.2f}",
                        "Market Cap": "${:,.0f}"
                    }, na_rep="-")
                    .background_gradient(cmap="RdYlGn", subset=["Score"], vmin=0, vmax=100)
                    .apply(highlight_rank, axis=1),
                    use_container_width=True,
                    height=min(600, (len(ranking_display) + 1) * 35 + 3)
                )
                
                # Visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 📊 Score Distribution")
                    fig_scores = go.Figure()
                    
                    colors = ['gold' if r == 1 else 'silver' if r == 2 else 'brown' if r == 3 else 'lightblue' 
                             for r in ranking_display["Rank"]]
                    
                    fig_scores.add_trace(go.Bar(
                        x=ranking_display.index,
                        y=ranking_display["Score"],
                        marker_color=colors,
                        text=ranking_display["Grade"],
                        textposition='outside',
                        hovertemplate='<b>%{x}</b><br>Score: %{y:.1f}<br>Grade: %{text}<extra></extra>'
                    ))
                    
                    fig_scores.update_layout(
                        yaxis_title="Fundamental Score",
                        yaxis_range=[0, 100],
                        height=400,
                        showlegend=False
                    )
                    st.plotly_chart(fig_scores, use_container_width=True)
                
                with col2:
                    st.markdown("### 🎯 Grade Distribution")
                    grade_counts = ranking_display["Grade"].value_counts()
                    
                    fig_grades = go.Figure(data=[go.Pie(
                        labels=grade_counts.index,
                        values=grade_counts.values,
                        hole=0.4,
                        marker_colors=px.colors.diverging.RdYlGn[::-1]
                    )])
                    
                    fig_grades.update_layout(height=400)
                    st.plotly_chart(fig_grades, use_container_width=True)
                
                # Category breakdown
                st.markdown("---")
                st.markdown("### 📊 Category Breakdown")
                
                category_scores = pd.DataFrame({
                    "Profitability": comparison_df[["ROE %", "Net Margin %", "Operating Margin %", "ROIC %"]].mean(axis=1),
                    "Growth": comparison_df[["Revenue Growth %", "NI Growth %", "EPS Growth %"]].mean(axis=1),
                    "Financial Health": comparison_df[["Current Ratio"]].mean(axis=1) * 10 - comparison_df[["Debt/Equity"]].mean(axis=1) * 5,
                    "Valuation": 50 - comparison_df[["P/E", "PEG", "EV/EBITDA"]].mean(axis=1)
                })
                
                # Normalize categories
                for col in category_scores.columns:
                    min_val = category_scores[col].min()
                    max_val = category_scores[col].max()
                    if min_val != max_val:
                        category_scores[col] = (category_scores[col] - min_val) / (max_val - min_val) * 100
                
                fig_radar = go.Figure()
                
                for ticker in ranking_display.head(5).index:
                    if ticker in category_scores.index:
                        values = category_scores.loc[ticker].tolist()
                        values.append(values[0])  # Close the radar
                        
                        fig_radar.add_trace(go.Scatterpolar(
                            r=values,
                            theta=list(category_scores.columns) + [category_scores.columns[0]],
                            name=ticker,
                            fill='toself'
                        ))
                
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                    showlegend=True,
                    height=500,
                    title="Top 5 Stocks - Category Comparison"
                )
                st.plotly_chart(fig_radar, use_container_width=True)
                
                # Best/Worst in each category
                st.markdown("---")
                st.markdown("### 🥇 Category Leaders")
                
                col1, col2, col3, col4 = st.columns(4)
                
                categories_best = {
                    "💰 Profitability": "ROE %",
                    "📈 Growth": "Revenue Growth %",
                    "🏦 Financial Health": "Current Ratio",
                    "💎 Valuation": "PEG"
                }
                
                for idx, (col, (cat_name, metric)) in enumerate(zip([col1, col2, col3, col4], categories_best.items())):
                    with col:
                        if metric == "PEG":
                            valid = comparison_df[metric].replace([np.inf, -np.inf], np.nan).dropna()
                            if not valid.empty:
                                best = valid.idxmin()
                                best_val = valid.min()
                        else:
                            valid = comparison_df[metric].dropna()
                            if not valid.empty:
                                best = valid.idxmax()
                                best_val = valid.max()
                        
                        if not valid.empty:
                            st.metric(
                                cat_name,
                                best,
                                f"{best_val:.2f}" if metric != "Market Cap" else f"${best_val:,.0f}"
                            )
                
                # Detailed scoring breakdown
                st.markdown("---")
                st.markdown("### 🔍 Detailed Score Components")
                
                with st.expander("View Individual Category Scores"):
                    # Show scores for each component
                    component_cols = [col for col in scores_df.columns if col not in ["Total Score", "Grade", "Rank"]]
                    component_display = scores_df[component_cols + ["Total Score"]].sort_values("Total Score", ascending=False)
                    
                    st.dataframe(
                        component_display.style
                        .format("{:.2f}")
                        .background_gradient(cmap="RdYlGn", vmin=0, vmax=10),
                        use_container_width=True
                    )
                
                # Investment recommendations
                st.markdown("---")
                st.markdown("### 💡 Analysis Summary")
                
                top_3 = ranking_display.head(3)
                bottom_3 = ranking_display.tail(3)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**✅ Strongest Fundamentals**")
                    for idx, (ticker, row) in enumerate(top_3.iterrows(), 1):
                        st.markdown(f"{idx}. **{ticker}** - Grade {row['Grade']} (Score: {row['Score']:.1f})")
                        strengths = []
                        if row["ROE %"] > comparison_df["ROE %"].median():
                            strengths.append("Strong ROE")
                        if row["Revenue Growth %"] > comparison_df["Revenue Growth %"].median():
                            strengths.append("High Growth")
                        if row["Debt/Equity"] < comparison_df["Debt/Equity"].median():
                            strengths.append("Low Debt")
                        if strengths:
                            st.caption(f"   Key strengths: {', '.join(strengths)}")
                
                with col2:
                    st.markdown("**⚠️ Weaker Fundamentals**")
                    for idx, (ticker, row) in enumerate(bottom_3.iterrows(), 1):
                        st.markdown(f"{len(ranking_display) - len(bottom_3) + idx}. **{ticker}** - Grade {row['Grade']} (Score: {row['Score']:.1f})")
                        concerns = []
                        if row["ROE %"] < comparison_df["ROE %"].median():
                            concerns.append("Low ROE")
                        if row["Revenue Growth %"] < 0:
                            concerns.append("Negative Growth")
                        if row["Debt/Equity"] > comparison_df["Debt/Equity"].median():
                            concerns.append("High Debt")
                        if concerns:
                            st.caption(f"   Concerns: {', '.join(concerns)}")
            
            # ══════════════════════════════════════════════════════════════════
            # TAB 2: OVERVIEW
            # ══════════════════════════════════════════════════════════════════
            with tab2:
                st.markdown("### 📊 Key Metrics Comparison")
                
                overview_cols = [
                    "Market Cap", "Revenue", "Net Income", "Free Cash Flow",
                    "EPS", "P/E", "P/B", "ROE %", "Net Margin %", "Revenue Growth %"
                ]
                overview_df = comparison_df[overview_cols].copy()
                
                st.dataframe(
                    overview_df.style.format({
                        "Market Cap": "${:,.0f}",
                        "Revenue": "${:,.0f}",
                        "Net Income": "${:,.0f}",
                        "Free Cash Flow": "${:,.0f}",
                        "EPS": "${:.2f}",
                        "P/E": "{:.2f}",
                        "P/B": "{:.2f}",
                        "ROE %": "{:.2f}",
                        "Net Margin %": "{:.2f}",
                        "Revenue Growth %": "{:.2f}"
                    }, na_rep="-")
                    .background_gradient(cmap="RdYlGn", subset=["ROE %", "Net Margin %", "Revenue Growth %"]),
                    use_container_width=True
                )
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Market Capitalization**")
                    valid_mcap = overview_df["Market Cap"].dropna()
                    if not valid_mcap.empty:
                        fig = go.Figure(data=[go.Bar(
                            x=valid_mcap.index,
                            y=valid_mcap.values,
                            marker_color='lightblue',
                            text=valid_mcap.values,
                            texttemplate='$%{text:,.0s}',
                            textposition='outside'
                        )])
                        fig.update_layout(yaxis_title="Market Cap ($)", height=350)
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("**Revenue Distribution**")
                    valid_rev = overview_df["Revenue"].dropna()
                    if not valid_rev.empty:
                        fig = px.pie(
                            values=valid_rev.values,
                            names=valid_rev.index,
                            title="Revenue Share"
                        )
                        fig.update_layout(height=350)
                        st.plotly_chart(fig, use_container_width=True)
            
            # ══════════════════════════════════════════════════════════════════
            # TAB 3: PROFITABILITY
            # ══════════════════════════════════════════════════════════════════
            with tab3:
                st.markdown("### 💰 Profitability Metrics")
                
                profit_cols = [
                    "Gross Margin %", "Operating Margin %", "Net Margin %", "EBITDA Margin %",
                    "ROE %", "ROA %", "ROIC %", "Revenue/Share", "EPS"
                ]
                profit_df = comparison_df[profit_cols].copy()
                
                st.dataframe(
                    profit_df.style.format({
                        "Gross Margin %": "{:.2f}",
                        "Operating Margin %": "{:.2f}",
                        "Net Margin %": "{:.2f}",
                        "EBITDA Margin %": "{:.2f}",
                        "ROE %": "{:.2f}",
                        "ROA %": "{:.2f}",
                        "ROIC %": "{:.2f}",
                        "Revenue/Share": "${:.2f}",
                        "EPS": "${:.2f}"
                    }, na_rep="-").background_gradient(cmap="RdYlGn"),
                    use_container_width=True
                )
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Margin Comparison**")
                    margin_data = profit_df[["Gross Margin %", "Operating Margin %", "Net Margin %"]].dropna(how='all')
                    
                    if not margin_data.empty:
                        fig = go.Figure()
                        for col in margin_data.columns:
                            valid_data = margin_data[col].dropna()
                            if not valid_data.empty:
                                fig.add_trace(go.Bar(name=col, x=valid_data.index, y=valid_data.values))
                        fig.update_layout(barmode='group', yaxis_title="Margin %", height=400)
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("**Return Metrics**")
                    return_data = profit_df[["ROE %", "ROA %", "ROIC %"]].dropna(how='all')
                    
                    if not return_data.empty:
                        fig = go.Figure()
                        for col in return_data.columns:
                            valid_data = return_data[col].dropna()
                            if not valid_data.empty:
                                fig.add_trace(go.Bar(name=col, x=valid_data.index, y=valid_data.values))
                        fig.update_layout(barmode='group', yaxis_title="Return %", height=400)
                        st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("**ROE vs Net Margin**")
                scatter_data = profit_df[["Net Margin %", "ROE %"]].dropna()
                if len(scatter_data) > 0:
                    fig = px.scatter(
                        scatter_data,
                        x="Net Margin %",
                        y="ROE %",
                        text=scatter_data.index,
                        size=[100]*len(scatter_data),
                        height=400,
                        title="Profitability Positioning"
                    )
                    fig.update_traces(textposition='top center')
                    fig.add_hline(y=scatter_data["ROE %"].median(), line_dash="dash", line_color="gray")
                    fig.add_vline(x=scatter_data["Net Margin %"].median(), line_dash="dash", line_color="gray")
                    st.plotly_chart(fig, use_container_width=True)
            
            # ══════════════════════════════════════════════════════════════════
            # TAB 4: CASH FLOW
            # ══════════════════════════════════════════════════════════════════
            with tab4:
                st.markdown("### 💵 Cash Flow Analysis")
                
                cf_cols = [
                    "Operating CF", "Free Cash Flow", "FCF/Share", "FCF Margin %",
                    "FCF/NI", "OCF/NI", "Net Income", "Revenue"
                ]
                cf_df = comparison_df[cf_cols].copy()
                
                st.dataframe(
                    cf_df.style.format({
                        "Operating CF": "${:,.0f}",
                        "Free Cash Flow": "${:,.0f}",
                        "FCF/Share": "${:.2f}",
                        "FCF Margin %": "{:.2f}",
                        "FCF/NI": "{:.2f}",
                        "OCF/NI": "{:.2f}",
                        "Net Income": "${:,.0f}",
                        "Revenue": "${:,.0f}"
                    }, na_rep="-")
                    .background_gradient(cmap="RdYlGn", subset=["FCF Margin %", "FCF/NI", "OCF/NI"]),
                    use_container_width=True
                )
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Free Cash Flow vs Net Income**")
                    fcf_ni_data = cf_df[["Net Income", "Free Cash Flow"]].dropna(how='all')
                    if not fcf_ni_data.empty:
                        fig = go.Figure()
                        ni_valid = fcf_ni_data["Net Income"].dropna()
                        if not ni_valid.empty:
                            fig.add_trace(go.Bar(name="Net Income", x=ni_valid.index, y=ni_valid.values, marker_color='lightcoral'))
                        fcf_valid = fcf_ni_data["Free Cash Flow"].dropna()
                        if not fcf_valid.empty:
                            fig.add_trace(go.Bar(name="Free Cash Flow", x=fcf_valid.index, y=fcf_valid.values, marker_color='lightgreen'))
                        fig.update_layout(barmode='group', yaxis_title="Amount ($)", height=400)
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("**FCF Margin %**")
                    fcf_margin = cf_df["FCF Margin %"].dropna()
                    if not fcf_margin.empty:
                        fig = go.Figure(data=[go.Bar(
                            x=fcf_margin.index,
                            y=fcf_margin.values,
                            marker_color='steelblue',
                            text=fcf_margin.values,
                            texttemplate='%{text:.1f}%',
                            textposition='outside'
                        )])
                        fig.update_layout(yaxis_title="FCF Margin %", height=400)
                        st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("**Cash Flow Quality**")
                quality_data = cf_df[["FCF/NI", "OCF/NI"]].dropna(how='all')
                if not quality_data.empty:
                    fig = go.Figure()
                    for col in quality_data.columns:
                        valid_data = quality_data[col].dropna()
                        if not valid_data.empty:
                            fig.add_trace(go.Bar(name=col, x=valid_data.index, y=valid_data.values))
                    fig.update_layout(
                        barmode='group',
                        yaxis_title="Ratio",
                        height=350,
                        title="Cash Flow Quality (>1.0 = Good)"
                    )
                    fig.add_hline(y=1.0, line_dash="dash", line_color="green", annotation_text="Benchmark: 1.0")
                    st.plotly_chart(fig, use_container_width=True)
            
            # ══════════════════════════════════════════════════════════════════
            # TAB 5: FINANCIAL HEALTH
            # ══════════════════════════════════════════════════════════════════
            with tab5:
                st.markdown("### 🏦 Financial Health & Solvency")
                
                health_cols = [
                    "Total Assets", "Total Debt", "Cash", "Equity",
                    "Current Ratio", "Quick Ratio", "Debt/Equity", "Debt/Assets", "Equity Ratio"
                ]
                health_df = comparison_df[health_cols].copy()
                
                st.dataframe(
                    health_df.style.format({
                        "Total Assets": "${:,.0f}",
                        "Total Debt": "${:,.0f}",
                        "Cash": "${:,.0f}",
                        "Equity": "${:,.0f}",
                        "Current Ratio": "{:.2f}",
                        "Quick Ratio": "{:.2f}",
                        "Debt/Equity": "{:.2f}",
                        "Debt/Assets": "{:.2f}",
                        "Equity Ratio": "{:.2f}"
                    }, na_rep="-")
                    .background_gradient(cmap="RdYlGn", subset=["Current Ratio", "Quick Ratio", "Equity Ratio"])
                    .background_gradient(cmap="RdYlGn_r", subset=["Debt/Equity", "Debt/Assets"]),
                    use_container_width=True
                )
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Debt vs Cash Position**")
                    debt_cash_data = health_df[["Cash", "Total Debt"]].dropna(how='all')
                    if not debt_cash_data.empty:
                        fig = go.Figure()
                        cash_valid = debt_cash_data["Cash"].dropna()
                        if not cash_valid.empty:
                            fig.add_trace(go.Bar(name="Cash", x=cash_valid.index, y=cash_valid.values, marker_color='lightgreen'))
                        debt_valid = debt_cash_data["Total Debt"].dropna()
                        if not debt_valid.empty:
                            fig.add_trace(go.Bar(name="Total Debt", x=debt_valid.index, y=debt_valid.values, marker_color='lightcoral'))
                        fig.update_layout(barmode='group', yaxis_title="Amount ($)", height=400)
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("**Liquidity Ratios**")
                    liquidity_data = health_df[["Current Ratio", "Quick Ratio"]].dropna(how='all')
                    if not liquidity_data.empty:
                        fig = go.Figure()
                        for col in liquidity_data.columns:
                            valid_data = liquidity_data[col].dropna()
                            if not valid_data.empty:
                                fig.add_trace(go.Bar(name=col, x=valid_data.index, y=valid_data.values))
                        fig.update_layout(barmode='group', yaxis_title="Ratio", height=400)
                        fig.add_hline(y=1.0, line_dash="dash", line_color="gray", annotation_text="Minimum: 1.0")
                        st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("**Debt/Equity Ratio**")
                de_ratio = health_df["Debt/Equity"].dropna()
                if not de_ratio.empty:
                    fig = go.Figure(data=[go.Bar(
                        x=de_ratio.index,
                        y=de_ratio.values,
                        marker_color='indianred',
                        text=de_ratio.values,
                        texttemplate='%{text:.2f}',
                        textposition='outside'
                    )])
                    fig.update_layout(
                        yaxis_title="Debt/Equity",
                        height=350,
                        title="Lower is Better"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # ══════════════════════════════════════════════════════════════════
            # TAB 6: VALUATION
            # ══════════════════════════════════════════════════════════════════
            with tab6:
                st.markdown("### 📈 Valuation Metrics")
                
                val_cols = [
                    "P/E", "P/B", "P/S", "Price/FCF", "EV/Revenue", "EV/EBITDA",
                    "PEG", "Market Cap", "Revenue Growth %"
                ]
                val_df = comparison_df[val_cols].copy()
                
                # Check if we have any valid valuation data
                valuation_metrics = ["P/E", "P/B", "P/S", "EV/EBITDA"]
                valid_count = val_df[valuation_metrics].notna().sum().sum()
                total_possible = len(val_df) * len(valuation_metrics)
                
                if valid_count == 0:
                    st.warning("⚠️ No valuation data available. This can happen if:")
                    st.markdown("""
                    - Stocks don't have market data (not publicly traded)
                    - Tickers are incorrect
                    - Companies have negative earnings (P/E not applicable)
                    - Data is temporarily unavailable from yfinance
                    """)
                    
                    # Show what data we do have
                    st.markdown("**Available Data:**")
                    available_data = pd.DataFrame({
                        "Market Cap": comparison_df["Market Cap"],
                        "Revenue": comparison_df["Revenue"],
                        "Net Income": comparison_df["Net Income"],
                        "EPS": comparison_df["EPS"]
                    })
                    st.dataframe(
                        available_data.style.format({
                            "Market Cap": "${:,.0f}",
                            "Revenue": "${:,.0f}",
                            "Net Income": "${:,.0f}",
                            "EPS": "${:.2f}"
                        }, na_rep="-"),
                        use_container_width=True
                    )
                else:
                    st.caption(f"📊 {valid_count}/{total_possible} valuation metrics available")
                
                st.dataframe(
                    val_df.style.format({
                        "P/E": "{:.2f}",
                        "P/B": "{:.2f}",
                        "P/S": "{:.2f}",
                        "Price/FCF": "{:.2f}",
                        "EV/Revenue": "{:.2f}",
                        "EV/EBITDA": "{:.2f}",
                        "PEG": "{:.2f}",
                        "Market Cap": "${:,.0f}",
                        "Revenue Growth %": "{:.2f}"
                    }, na_rep="-"),
                    use_container_width=True
                )
                
                st.caption("💡 Lower multiples generally indicate cheaper valuation (except for high-growth stocks)")
                
                # Add valuation interpretation
                val_clean = val_df[["P/E", "P/B", "P/S", "EV/EBITDA", "PEG"]].copy()
                
                # Show table with conditional formatting
                def color_valuation(val):
                    """Color cells based on valuation levels"""
                    if pd.isna(val):
                        return ''
                    try:
                        val = float(val)
                        if val < 0:
                            return 'background-color: #ffcccc'  # Light red for negative
                        elif val > 100:
                            return 'background-color: #ffcccc'  # Light red for extreme
                        elif val < 15:
                            return 'background-color: #ccffcc'  # Light green for low
                        elif val < 25:
                            return 'background-color: #ffffcc'  # Light yellow for medium
                        else:
                            return 'background-color: #ffddcc'  # Light orange for high
                    except:
                        return ''
                
                st.dataframe(
                    val_df.style.format({
                        "P/E": "{:.2f}",
                        "P/B": "{:.2f}",
                        "P/S": "{:.2f}",
                        "Price/FCF": "{:.2f}",
                        "EV/Revenue": "{:.2f}",
                        "EV/EBITDA": "{:.2f}",
                        "PEG": "{:.2f}",
                        "Market Cap": "${:,.0f}",
                        "Revenue Growth %": "{:.2f}"
                    }, na_rep="-")
                    .applymap(color_valuation, subset=["P/E", "P/B", "P/S", "EV/EBITDA", "PEG"]),
                    use_container_width=True
                )
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Valuation Multiples**")
                    fig = go.Figure()
                    for col in ["P/E", "P/B", "P/S", "EV/EBITDA"]:
                        if col in val_df.columns:
                            valid_data = val_df[col].replace([np.inf, -np.inf], np.nan).dropna()
                            if not valid_data.empty:
                                fig.add_trace(go.Bar(name=col, x=valid_data.index, y=valid_data.values))
                    fig.update_layout(barmode='group', yaxis_title="Multiple", height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("**PEG Ratio Comparison**")
                    peg_data = val_df["PEG"].replace([np.inf, -np.inf], np.nan).dropna()
                    if not peg_data.empty:
                        fig = go.Figure(data=[go.Bar(
                            x=peg_data.index,
                            y=peg_data.values,
                            marker_color='purple',
                            text=peg_data.values,
                            texttemplate='%{text:.2f}',
                            textposition='outside'
                        )])
                        fig.update_layout(
                            yaxis_title="PEG Ratio",
                            height=400,
                            title="PEG < 1.0 = Potentially Undervalued"
                        )
                        fig.add_hline(y=1.0, line_dash="dash", line_color="green")
                        st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("**P/E vs Revenue Growth**")
                peg_scatter = val_df[["P/E", "Revenue Growth %"]].replace([np.inf, -np.inf], np.nan).dropna()
                if len(peg_scatter) > 1:
                    fig = px.scatter(
                        peg_scatter,
                        x="Revenue Growth %",
                        y="P/E",
                        text=peg_scatter.index,
                        size=[100]*len(peg_scatter),
                        height=400,
                        title="Growth vs Valuation"
                    )
                    fig.update_traces(textposition='top center')
                    fig.add_hline(y=peg_scatter["P/E"].median(), line_dash="dash", line_color="gray")
                    fig.add_vline(x=peg_scatter["Revenue Growth %"].median(), line_dash="dash", line_color="gray")
                    st.plotly_chart(fig, use_container_width=True)
            
            # ══════════════════════════════════════════════════════════════════
            # TAB 7: TRENDS
            # ══════════════════════════════════════════════════════════════════
            with tab7:
                st.markdown("### 📉 Historical Trends")
                
                def create_trend_chart(title, data_dict, ylabel):
                    """Helper function to create trend charts"""
                    fig = go.Figure()
                    for ticker, data in data_dict.items():
                        valid_data = data.dropna()
                        if not valid_data.empty:
                            fig.add_trace(go.Scatter(
                                x=valid_data.index,
                                y=valid_data.values,
                                name=ticker,
                                mode='lines+markers',
                                line=dict(width=2),
                                marker=dict(size=6)
                            ))
                    fig.update_layout(
                        title=title,
                        yaxis_title=ylabel,
                        height=400,
                        hovermode='x unified',
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    return fig
                
                # Revenue Trend
                st.markdown("**Revenue Trend**")
                revenue_trends = {}
                for ticker in tickers:
                    ts = comparison_df.loc[ticker, "_revenue_ts"]
                    if not ts.empty and len(ts.dropna()) > 0:
                        revenue_trends[ticker] = ts
                
                if revenue_trends:
                    fig = create_trend_chart("Revenue Over Time", revenue_trends, "Revenue ($)")
                    st.plotly_chart(fig, use_container_width=True)
                
                # Net Income Trend
                st.markdown("**Net Income Trend**")
                ni_trends = {}
                for ticker in tickers:
                    ts = comparison_df.loc[ticker, "_net_income_ts"]
                    if not ts.empty and len(ts.dropna()) > 0:
                        ni_trends[ticker] = ts
                
                if ni_trends:
                    fig = create_trend_chart("Net Income Over Time", ni_trends, "Net Income ($)")
                    st.plotly_chart(fig, use_container_width=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Net Margin Trend
                    st.markdown("**Net Margin % Trend**")
                    margin_trends = {}
                    for ticker in tickers:
                        ts = comparison_df.loc[ticker, "_net_margin_ts"]
                        if not ts.empty and len(ts.dropna()) > 0:
                            margin_trends[ticker] = ts
                    
                    if margin_trends:
                        fig = create_trend_chart("Net Margin % Over Time", margin_trends, "Net Margin %")
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # ROE Trend
                    st.markdown("**ROE % Trend**")
                    roe_trends = {}
                    for ticker in tickers:
                        ts = comparison_df.loc[ticker, "_roe_ts"]
                        if not ts.empty and len(ts.dropna()) > 0:
                            roe_trends[ticker] = ts
                    
                    if roe_trends:
                        fig = create_trend_chart("ROE % Over Time", roe_trends, "ROE %")
                        st.plotly_chart(fig, use_container_width=True)
                
                # Free Cash Flow Trend
                st.markdown("**Free Cash Flow Trend**")
                fcf_trends = {}
                for ticker in tickers:
                    ts = comparison_df.loc[ticker, "_fcf_ts"]
                    if not ts.empty and len(ts.dropna()) > 0:
                        fcf_trends[ticker] = ts
                
                if fcf_trends:
                    fig = create_trend_chart("Free Cash Flow Over Time", fcf_trends, "Free Cash Flow ($)")
                    st.plotly_chart(fig, use_container_width=True)
                
                # Growth comparison
                st.markdown("---")
                st.markdown("**Growth Comparison**")
                growth_comparison = comparison_df[[
                    "Revenue Growth %", "NI Growth %", "FCF Growth %", "EPS Growth %"
                ]].dropna(how='all')
                
                if not growth_comparison.empty:
                    fig = go.Figure()
                    for col in growth_comparison.columns:
                        valid_data = growth_comparison[col].dropna()
                        if not valid_data.empty:
                            fig.add_trace(go.Bar(name=col, x=valid_data.index, y=valid_data.values))
                    fig.update_layout(
                        barmode='group',
                        yaxis_title="Growth %",
                        height=400,
                        title="Average Growth Rates"
                    )
                    fig.add_hline(y=0, line_color="black", line_width=1)
                    st.plotly_chart(fig, use_container_width=True)
            
            # ══════════════════════════════════════════════════════════════════
            # TAB 8: EXPORT
            # ══════════════════════════════════════════════════════════════════
            with tab8:
                st.markdown("### 📥 Export Data")
                
                # Prepare export dataframes
                export_df = comparison_df[[col for col in comparison_df.columns if not col.startswith("_")]].copy()
                
                # Combine with scores
                scores_export = scores_df[["Rank", "Total Score", "Grade"]].copy()
                full_export = pd.concat([scores_export, export_df], axis=1)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**📊 Complete Analysis**")
                    csv_full = full_export.to_csv()
                    st.download_button(
                        label="Download Full Dataset (CSV)",
                        data=csv_full,
                        file_name=f"fundamental_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                with col2:
                    st.markdown("**🏆 Rankings Only**")
                    csv_rankings = ranking_display.to_csv()
                    st.download_button(
                        label="Download Rankings (CSV)",
                        data=csv_rankings,
                        file_name=f"stock_rankings_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                with col3:
                    st.markdown("**📈 Key Metrics**")
                    key_metrics_export = export_df[[
                        "Market Cap", "Revenue", "Net Income", "ROE %", "P/E",
                        "Revenue Growth %", "Debt/Equity"
                    ]].copy()
                    csv_key = key_metrics_export.to_csv()
                    st.download_button(
                        label="Download Key Metrics (CSV)",
                        data=csv_key,
                        file_name=f"key_metrics_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                # Data preview
                st.markdown("---")
                st.markdown("### 👁️ Data Preview")
                
                preview_option = st.selectbox(
                    "Select dataset to preview",
                    ["Complete Analysis", "Rankings", "Key Metrics"]
                )
                
                if preview_option == "Complete Analysis":
                    st.dataframe(full_export, use_container_width=True, height=400)
                elif preview_option == "Rankings":
                    st.dataframe(ranking_display, use_container_width=True, height=400)
                else:
                    st.dataframe(key_metrics_export, use_container_width=True, height=400)
        
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            with st.expander("🐛 Error Details"):
                st.exception(e)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    st.set_page_config(
        page_title="Advanced Fundamental Analysis",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    render_fundamental_comparison()
