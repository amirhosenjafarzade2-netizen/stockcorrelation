import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Tuple
from datetime import date, datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Optional imports with fallbacks
try:
    from pytrends.request import TrendReq
    TRENDS_AVAILABLE = True
except ImportError:
    TRENDS_AVAILABLE = False
    st.warning("⚠️ pytrends not installed. Install with: pip install pytrends")

try:
    import requests
    from bs4 import BeautifulSoup
    NEWS_SCRAPING_AVAILABLE = True
except ImportError:
    NEWS_SCRAPING_AVAILABLE = False


def render_sentiment_analysis(tickers: List[str] = None) -> None:
    """
    Comprehensive sentiment and market psychology analysis module.
    
    Analyzes:
    - VIX (Fear Index)
    - Google Trends (Retail Interest)
    - Options Data (Implied Volatility, Put/Call Ratio, Greeks)
    - Social Sentiment (Reddit, Twitter mentions)
    - Insider Trading
    - Short Interest
    - Analyst Ratings
    - News Sentiment
    """
    st.markdown("# 📊 Sentiment & Market Psychology Analysis")
    st.markdown("Analyze market sentiment, retail interest, options activity, and behavioral indicators")
    
    # ══════════════════════════════════════════════════════════════════════════
    # INPUT SECTION
    # ══════════════════════════════════════════════════════════════════════════
    with st.expander("⚙️ Configuration", expanded=True):
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            default_tickers = ", ".join(tickers) if tickers else "AAPL, TSLA, NVDA, SPY"
            ticker_input = st.text_input(
                "Stock Tickers (comma-separated)",
                value=default_tickers,
                help="Enter ticker symbols separated by commas"
            )
            tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
        
        with col2:
            lookback_period = st.selectbox(
                "Time Period",
                ["1mo", "3mo", "6mo", "1y", "2y"],
                index=2,
                help="Historical data period"
            )
        
        with col3:
            analyze_btn = st.button(
                "🚀 Analyze",
                type="primary",
                use_container_width=True
            )
    
    # Validation
    if len(tickers) < 1:
        st.info("📌 Enter at least 1 ticker to analyze sentiment")
        return
    
    if len(tickers) > 10:
        st.warning("⚠️ Maximum 10 tickers supported. Using first 10.")
        tickers = tickers[:10]
    
    if not analyze_btn:
        st.info("👆 Click 'Analyze' to load sentiment data")
        return
    
    # ══════════════════════════════════════════════════════════════════════════
    # DATA FETCHING
    # ══════════════════════════════════════════════════════════════════════════
    with st.spinner(f"📥 Fetching sentiment data for {len(tickers)} stocks..."):
        try:
            all_data = {}
            failed_tickers = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Fetch VIX for market fear gauge
            status_text.text("Loading VIX (Market Fear Index)...")
            try:
                vix = yf.Ticker("^VIX")
                vix_hist = vix.history(period=lookback_period)
                vix_current = vix_hist['Close'].iloc[-1] if not vix_hist.empty else None
            except Exception as e:
                st.warning(f"⚠️ Failed to fetch VIX: {str(e)}")
                vix_hist = pd.DataFrame()
                vix_current = None
            
            # Fetch data for each ticker
            for idx, ticker in enumerate(tickers):
                status_text.text(f"Loading {ticker}... ({idx+1}/{len(tickers)})")
                progress_bar.progress((idx + 1) / len(tickers))
                
                try:
                    ticker_obj = yf.Ticker(ticker)
                    
                    # Basic info
                    try:
                        info = ticker_obj.info
                        if not info or not isinstance(info, dict):
                            info = {}
                    except Exception:
                        info = {}
                    
                    # Price history
                    hist = ticker_obj.history(period=lookback_period)
                    
                    # Options data
                    try:
                        options_dates = ticker_obj.options
                        if options_dates and len(options_dates) > 0:
                            # Get nearest expiration
                            opt_chain = ticker_obj.option_chain(options_dates[0])
                            calls = opt_chain.calls
                            puts = opt_chain.puts
                        else:
                            calls = pd.DataFrame()
                            puts = pd.DataFrame()
                    except Exception as e:
                        calls = pd.DataFrame()
                        puts = pd.DataFrame()
                    
                    # Recommendations
                    try:
                        recommendations = ticker_obj.recommendations
                        if recommendations is None:
                            recommendations = pd.DataFrame()
                    except Exception:
                        recommendations = pd.DataFrame()
                    
                    # Institutional holders
                    try:
                        institutional = ticker_obj.institutional_holders
                        if institutional is None:
                            institutional = pd.DataFrame()
                    except Exception:
                        institutional = pd.DataFrame()
                    
                    # Insider transactions
                    try:
                        insider = ticker_obj.insider_transactions
                        if insider is None:
                            insider = pd.DataFrame()
                    except Exception:
                        insider = pd.DataFrame()
                    
                    all_data[ticker] = {
                        "info": info,
                        "history": hist,
                        "calls": calls,
                        "puts": puts,
                        "recommendations": recommendations,
                        "institutional": institutional,
                        "insider": insider
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
            def calculate_rsi(prices, period=14):
                """Calculate Relative Strength Index"""
                delta = prices.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                return rsi
            
            def calculate_bollinger_bands(prices, period=20, std_dev=2):
                """Calculate Bollinger Bands"""
                sma = prices.rolling(window=period).mean()
                std = prices.rolling(window=period).std()
                upper = sma + (std * std_dev)
                lower = sma - (std * std_dev)
                return upper, sma, lower
            
            def get_sentiment_score(value, thresholds):
                """Convert numeric value to sentiment score"""
                if pd.isna(value):
                    return 50  # Neutral
                low, high = thresholds
                if value < low:
                    return 30  # Bearish
                elif value > high:
                    return 70  # Bullish
                else:
                    return 50  # Neutral
            
            def interpret_vix(vix_level):
                """Interpret VIX level"""
                if pd.isna(vix_level):
                    return "Unknown", "gray"
                elif vix_level < 12:
                    return "Very Low Fear (Complacent)", "green"
                elif vix_level < 20:
                    return "Low Fear (Normal)", "lightgreen"
                elif vix_level < 30:
                    return "Elevated Fear", "orange"
                elif vix_level < 40:
                    return "High Fear", "darkorange"
                else:
                    return "Extreme Fear (Panic)", "red"
            
            # ══════════════════════════════════════════════════════════════════
            # SENTIMENT METRICS CALCULATION
            # ══════════════════════════════════════════════════════════════════
            def calculate_sentiment_metrics(ticker: str, ticker_data: Dict) -> Dict:
                """Calculate comprehensive sentiment metrics"""
                info = ticker_data["info"]
                hist = ticker_data["history"]
                calls = ticker_data["calls"]
                puts = ticker_data["puts"]
                recommendations = ticker_data["recommendations"]
                
                metrics = {
                    "ticker": ticker,
                    "company_name": info.get("longName", ticker)
                }
                
                # ═══════════════════════════════════════════════════════════
                # 1. OPTIONS SENTIMENT
                # ═══════════════════════════════════════════════════════════
                if not calls.empty and not puts.empty:
                    # Put/Call Ratio by Volume
                    put_volume = puts['volume'].sum()
                    call_volume = calls['volume'].sum()
                    pcr_volume = put_volume / call_volume if call_volume > 0 else np.nan
                    
                    # Put/Call Ratio by Open Interest
                    put_oi = puts['openInterest'].sum()
                    call_oi = calls['openInterest'].sum()
                    pcr_oi = put_oi / call_oi if call_oi > 0 else np.nan
                    
                    # Average Implied Volatility
                    avg_call_iv = calls['impliedVolatility'].mean() * 100
                    avg_put_iv = puts['impliedVolatility'].mean() * 100
                    iv_skew = avg_put_iv - avg_call_iv  # Positive = more fear
                    
                    # Options sentiment interpretation
                    if not pd.isna(pcr_volume):
                        if pcr_volume < 0.7:
                            pcr_sentiment = "Bullish"
                            pcr_color = "green"
                        elif pcr_volume > 1.0:
                            pcr_sentiment = "Bearish"
                            pcr_color = "red"
                        else:
                            pcr_sentiment = "Neutral"
                            pcr_color = "gray"
                    else:
                        pcr_sentiment = "N/A"
                        pcr_color = "gray"
                    
                    metrics.update({
                        "pcr_volume": pcr_volume,
                        "pcr_oi": pcr_oi,
                        "avg_call_iv": avg_call_iv,
                        "avg_put_iv": avg_put_iv,
                        "iv_skew": iv_skew,
                        "total_call_volume": call_volume,
                        "total_put_volume": put_volume,
                        "total_call_oi": call_oi,
                        "total_put_oi": put_oi,
                        "pcr_sentiment": pcr_sentiment,
                        "pcr_color": pcr_color
                    })
                else:
                    metrics.update({
                        "pcr_volume": np.nan,
                        "pcr_oi": np.nan,
                        "avg_call_iv": np.nan,
                        "avg_put_iv": np.nan,
                        "iv_skew": np.nan,
                        "total_call_volume": 0,
                        "total_put_volume": 0,
                        "total_call_oi": 0,
                        "total_put_oi": 0,
                        "pcr_sentiment": "N/A",
                        "pcr_color": "gray"
                    })
                
                # ═══════════════════════════════════════════════════════════
                # 2. TECHNICAL SENTIMENT
                # ═══════════════════════════════════════════════════════════
                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                    
                    # RSI
                    rsi = calculate_rsi(hist['Close'])
                    current_rsi = rsi.iloc[-1] if not rsi.empty else np.nan
                    
                    # Moving Averages
                    sma_20 = hist['Close'].rolling(20).mean().iloc[-1]
                    sma_50 = hist['Close'].rolling(50).mean().iloc[-1] if len(hist) >= 50 else np.nan
                    sma_200 = hist['Close'].rolling(200).mean().iloc[-1] if len(hist) >= 200 else np.nan
                    
                    # Price vs MAs
                    above_sma20 = current_price > sma_20 if not pd.isna(sma_20) else None
                    above_sma50 = current_price > sma_50 if not pd.isna(sma_50) else None
                    above_sma200 = current_price > sma_200 if not pd.isna(sma_200) else None
                    
                    # Bollinger Bands
                    upper_bb, middle_bb, lower_bb = calculate_bollinger_bands(hist['Close'])
                    current_bb_position = ((current_price - lower_bb.iloc[-1]) / 
                                          (upper_bb.iloc[-1] - lower_bb.iloc[-1]) * 100) if not pd.isna(lower_bb.iloc[-1]) else np.nan
                    
                    # Volatility (30-day)
                    returns = hist['Close'].pct_change()
                    volatility_30d = returns.tail(30).std() * np.sqrt(252) * 100
                    
                    # Volume analysis
                    avg_volume_20d = hist['Volume'].tail(20).mean()
                    current_volume = hist['Volume'].iloc[-1]
                    volume_ratio = current_volume / avg_volume_20d if avg_volume_20d > 0 else np.nan
                    
                    metrics.update({
                        "current_price": current_price,
                        "rsi": current_rsi,
                        "sma_20": sma_20,
                        "sma_50": sma_50,
                        "sma_200": sma_200,
                        "above_sma20": above_sma20,
                        "above_sma50": above_sma50,
                        "above_sma200": above_sma200,
                        "bb_position": current_bb_position,
                        "volatility_30d": volatility_30d,
                        "volume_ratio": volume_ratio,
                        "avg_volume": avg_volume_20d
                    })
                else:
                    metrics.update({
                        "current_price": np.nan,
                        "rsi": np.nan,
                        "sma_20": np.nan,
                        "sma_50": np.nan,
                        "sma_200": np.nan,
                        "above_sma20": None,
                        "above_sma50": None,
                        "above_sma200": None,
                        "bb_position": np.nan,
                        "volatility_30d": np.nan,
                        "volume_ratio": np.nan,
                        "avg_volume": np.nan
                    })
                
                # ═══════════════════════════════════════════════════════════
                # 3. ANALYST SENTIMENT
                # ═══════════════════════════════════════════════════════════
                if not recommendations.empty and 'To Grade' in recommendations.columns:
                    recent_recs = recommendations.tail(20)  # Last 20 recommendations
                    
                    # Count by grade
                    grade_counts = recent_recs['To Grade'].value_counts()
                    
                    buy_terms = ['buy', 'strong buy', 'outperform', 'overweight']
                    hold_terms = ['hold', 'neutral', 'equal-weight', 'market perform']
                    sell_terms = ['sell', 'strong sell', 'underperform', 'underweight']
                    
                    buys = sum(grade_counts.get(term, 0) for term in grade_counts.index 
                              if any(buy_term in term.lower() for buy_term in buy_terms))
                    holds = sum(grade_counts.get(term, 0) for term in grade_counts.index 
                               if any(hold_term in term.lower() for hold_term in hold_terms))
                    sells = sum(grade_counts.get(term, 0) for term in grade_counts.index 
                               if any(sell_term in term.lower() for sell_term in sell_terms))
                    
                    total_recs = buys + holds + sells
                    if total_recs > 0:
                        buy_pct = (buys / total_recs) * 100
                        hold_pct = (holds / total_recs) * 100
                        sell_pct = (sells / total_recs) * 100
                        
                        # Analyst sentiment score
                        analyst_score = (buys * 100 + holds * 50) / total_recs if total_recs > 0 else 50
                    else:
                        buy_pct = hold_pct = sell_pct = analyst_score = np.nan
                    
                    metrics.update({
                        "analyst_buys": buys,
                        "analyst_holds": holds,
                        "analyst_sells": sells,
                        "analyst_buy_pct": buy_pct,
                        "analyst_hold_pct": hold_pct,
                        "analyst_sell_pct": sell_pct,
                        "analyst_score": analyst_score
                    })
                else:
                    metrics.update({
                        "analyst_buys": 0,
                        "analyst_holds": 0,
                        "analyst_sells": 0,
                        "analyst_buy_pct": np.nan,
                        "analyst_hold_pct": np.nan,
                        "analyst_sell_pct": np.nan,
                        "analyst_score": np.nan
                    })
                
                # ═══════════════════════════════════════════════════════════
                # 4. INSTITUTIONAL & INSIDER SENTIMENT
                # ═══════════════════════════════════════════════════════════
                # Short Interest
                short_ratio = info.get("shortRatio", np.nan)
                short_percent = info.get("shortPercentOfFloat", np.nan)
                if not pd.isna(short_percent):
                    short_percent = short_percent * 100 if short_percent < 1 else short_percent
                
                metrics.update({
                    "short_ratio": short_ratio,
                    "short_percent": short_percent
                })
                
                # ═══════════════════════════════════════════════════════════
                # 5. COMPOSITE SENTIMENT SCORE
                # ═══════════════════════════════════════════════════════════
                sentiment_components = []
                
                # RSI component (oversold = bullish, overbought = bearish)
                if not pd.isna(current_rsi):
                    if current_rsi < 30:
                        rsi_score = 70  # Oversold = potential bounce
                    elif current_rsi > 70:
                        rsi_score = 30  # Overbought = potential pullback
                    else:
                        rsi_score = 50 + (50 - current_rsi) * 0.5  # Scale 30-70 to 60-40
                    sentiment_components.append(rsi_score)
                
                # PCR component (high puts = bearish)
                if not pd.isna(pcr_volume):
                    if pcr_volume < 0.7:
                        pcr_score = 70
                    elif pcr_volume > 1.0:
                        pcr_score = 30
                    else:
                        pcr_score = 50
                    sentiment_components.append(pcr_score)
                
                # Analyst component
                if not pd.isna(analyst_score):
                    sentiment_components.append(analyst_score)
                
                # MA component
                ma_score = 50
                if above_sma20:
                    ma_score += 10
                if above_sma50:
                    ma_score += 10
                if above_sma200:
                    ma_score += 10
                if above_sma20 is not None:
                    sentiment_components.append(ma_score)
                
                # Calculate composite
                if sentiment_components:
                    composite_sentiment = np.mean(sentiment_components)
                    if composite_sentiment >= 65:
                        sentiment_label = "Bullish"
                        sentiment_color = "green"
                    elif composite_sentiment >= 45:
                        sentiment_label = "Neutral"
                        sentiment_color = "gray"
                    else:
                        sentiment_label = "Bearish"
                        sentiment_color = "red"
                else:
                    composite_sentiment = 50
                    sentiment_label = "Unknown"
                    sentiment_color = "gray"
                
                metrics.update({
                    "composite_sentiment": composite_sentiment,
                    "sentiment_label": sentiment_label,
                    "sentiment_color": sentiment_color
                })
                
                return metrics
            
            # Calculate metrics for all tickers
            with st.spinner("📊 Calculating sentiment metrics..."):
                sentiment_data = {}
                for ticker in tickers:
                    sentiment_data[ticker] = calculate_sentiment_metrics(ticker, all_data[ticker])
            
            # ══════════════════════════════════════════════════════════════════
            # TABS LAYOUT
            # ══════════════════════════════════════════════════════════════════
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "🎯 Overview",
                "📊 Options Sentiment",
                "📈 Technical Sentiment",
                "👔 Analyst & Institutional",
                "😱 Fear & Greed",
                "🔍 Google Trends"
            ])
            
            # ══════════════════════════════════════════════════════════════════
            # TAB 1: OVERVIEW
            # ══════════════════════════════════════════════════════════════════
            with tab1:
                st.markdown("## 🎯 Sentiment Overview")
                
                # VIX Display
                if vix_current:
                    vix_interpretation, vix_color = interpret_vix(vix_current)
                    col1, col2, col3 = st.columns([1, 1, 2])
                    with col1:
                        st.metric("VIX (Fear Index)", f"{vix_current:.2f}")
                    with col2:
                        st.markdown(f"**Market Mood:** :{vix_color}[{vix_interpretation}]")
                    with col3:
                        # Mini VIX chart
                        if not vix_hist.empty:
                            fig_vix = go.Figure()
                            fig_vix.add_trace(go.Scatter(
                                x=vix_hist.index,
                                y=vix_hist['Close'],
                                mode='lines',
                                fill='tozeroy',
                                name='VIX',
                                line=dict(color=vix_color, width=2)
                            ))
                            fig_vix.update_layout(height=150, margin=dict(l=0, r=0, t=0, b=0), showlegend=False)
                            st.plotly_chart(fig_vix, use_container_width=True)
                
                st.markdown("---")
                
                # Sentiment Scores Table
                st.markdown("### 📋 Composite Sentiment Scores")
                
                overview_data = []
                for ticker, data in sentiment_data.items():
                    overview_data.append({
                        "Ticker": ticker,
                        "Sentiment": data["sentiment_label"],
                        "Score": data["composite_sentiment"],
                        "RSI": data["rsi"],
                        "PCR": data["pcr_volume"],
                        "Analyst Score": data["analyst_score"],
                        "Short %": data["short_percent"],
                        "IV": data["avg_call_iv"]
                    })
                
                overview_df = pd.DataFrame(overview_data)
                
                # Color code sentiment
                def color_sentiment(row):
                    colors = []
                    for col in row.index:
                        if col == "Sentiment":
                            if row[col] == "Bullish":
                                colors.append('background-color: #90EE90')
                            elif row[col] == "Bearish":
                                colors.append('background-color: #FFB6C6')
                            else:
                                colors.append('background-color: #D3D3D3')
                        else:
                            colors.append('')
                    return colors
                
                st.dataframe(
                    overview_df.style
                    .format({
                        "Score": "{:.1f}",
                        "RSI": "{:.1f}",
                        "PCR": "{:.2f}",
                        "Analyst Score": "{:.1f}",
                        "Short %": "{:.2f}",
                        "IV": "{:.1f}%"
                    }, na_rep="-")
                    .background_gradient(cmap="RdYlGn", subset=["Score"], vmin=0, vmax=100)
                    .apply(color_sentiment, axis=1),
                    use_container_width=True,
                    height=min(400, (len(overview_df) + 1) * 35 + 3)
                )
                
                # Sentiment Distribution
                st.markdown("### 📊 Sentiment Distribution")
                col1, col2 = st.columns(2)
                
                with col1:
                    # Sentiment score chart
                    fig_sentiment = go.Figure()
                    
                    colors_map = {
                        "Bullish": "green",
                        "Neutral": "gray",
                        "Bearish": "red",
                        "Unknown": "lightgray"
                    }
                    
                    colors = [colors_map.get(sentiment_data[t]["sentiment_label"], "gray") for t in tickers]
                    
                    fig_sentiment.add_trace(go.Bar(
                        x=tickers,
                        y=[sentiment_data[t]["composite_sentiment"] for t in tickers],
                        marker_color=colors,
                        text=[sentiment_data[t]["sentiment_label"] for t in tickers],
                        textposition='outside'
                    ))
                    
                    fig_sentiment.update_layout(
                        title="Composite Sentiment Score",
                        yaxis_title="Score (0-100)",
                        yaxis_range=[0, 100],
                        height=400
                    )
                    fig_sentiment.add_hline(y=50, line_dash="dash", line_color="gray", annotation_text="Neutral")
                    st.plotly_chart(fig_sentiment, use_container_width=True)
                
                with col2:
                    # Pie chart of sentiment distribution
                    sentiment_counts = overview_df["Sentiment"].value_counts()
                    
                    fig_pie = go.Figure(data=[go.Pie(
                        labels=sentiment_counts.index,
                        values=sentiment_counts.values,
                        marker_colors=['green' if x == 'Bullish' else 'red' if x == 'Bearish' else 'gray' 
                                      for x in sentiment_counts.index],
                        hole=0.4
                    )])
                    
                    fig_pie.update_layout(
                        title="Sentiment Distribution",
                        height=400
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                # Radar chart for multi-factor analysis
                if len(tickers) <= 5:
                    st.markdown("### 🎯 Multi-Factor Sentiment Radar")
                    
                    fig_radar = go.Figure()
                    
                    for ticker in tickers:
                        data = sentiment_data[ticker]
                        
                        # Normalize metrics to 0-100 scale
                        factors = {
                            "Analyst": data["analyst_score"] if not pd.isna(data["analyst_score"]) else 50,
                            "Technical": 50 + (data["rsi"] - 50) if not pd.isna(data["rsi"]) else 50,
                            "Options": 70 if data["pcr_sentiment"] == "Bullish" else 30 if data["pcr_sentiment"] == "Bearish" else 50,
                            "Volume": min(100, data["volume_ratio"] * 50) if not pd.isna(data["volume_ratio"]) else 50,
                            "Volatility": max(0, 100 - data["volatility_30d"]) if not pd.isna(data["volatility_30d"]) else 50
                        }
                        
                        values = list(factors.values())
                        values.append(values[0])  # Close the radar
                        
                        fig_radar.add_trace(go.Scatterpolar(
                            r=values,
                            theta=list(factors.keys()) + [list(factors.keys())[0]],
                            name=ticker,
                            fill='toself'
                        ))
                    
                    fig_radar.update_layout(
                        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                        showlegend=True,
                        height=500
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)
            
            # ══════════════════════════════════════════════════════════════════
            # TAB 2: OPTIONS SENTIMENT
            # ══════════════════════════════════════════════════════════════════
            with tab2:
                st.markdown("## 📊 Options Market Sentiment")
                st.markdown("Put/Call ratios, Implied Volatility, and options flow analysis")
                
                # Options data table
                options_data = []
                for ticker, data in sentiment_data.items():
                    options_data.append({
                        "Ticker": ticker,
                        "PCR (Volume)": data["pcr_volume"],
                        "PCR (OI)": data["pcr_oi"],
                        "Call Volume": data["total_call_volume"],
                        "Put Volume": data["total_put_volume"],
                        "Call OI": data["total_call_oi"],
                        "Put OI": data["total_put_oi"],
                        "Call IV": data["avg_call_iv"],
                        "Put IV": data["avg_put_iv"],
                        "IV Skew": data["iv_skew"],
                        "Sentiment": data["pcr_sentiment"]
                    })
                
                options_df = pd.DataFrame(options_data)
                
                st.dataframe(
                    options_df.style.format({
                        "PCR (Volume)": "{:.2f}",
                        "PCR (OI)": "{:.2f}",
                        "Call Volume": "{:,.0f}",
                        "Put Volume": "{:,.0f}",
                        "Call OI": "{:,.0f}",
                        "Put OI": "{:,.0f}",
                        "Call IV": "{:.1f}%",
                        "Put IV": "{:.1f}%",
                        "IV Skew": "{:.1f}%"
                    }, na_rep="-")
                    .background_gradient(cmap="RdYlGn_r", subset=["PCR (Volume)", "PCR (OI)"])
                    .background_gradient(cmap="YlOrRd", subset=["Call IV", "Put IV"]),
                    use_container_width=True
                )
                
                st.markdown("**Interpretation:**")
                st.markdown("""
                - **PCR < 0.7**: Bullish (more calls than puts)
                - **PCR 0.7-1.0**: Neutral
                - **PCR > 1.0**: Bearish (more puts than calls)
                - **IV Skew > 0**: Put options more expensive (fear premium)
                """)
                
                # Visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### Put/Call Ratio (Volume)")
                    
                    valid_pcr = options_df[options_df["PCR (Volume)"].notna()]
                    if not valid_pcr.empty:
                        fig_pcr = go.Figure()
                        
                        colors = ['green' if x == 'Bullish' else 'red' if x == 'Bearish' else 'gray' 
                                 for x in valid_pcr["Sentiment"]]
                        
                        fig_pcr.add_trace(go.Bar(
                            x=valid_pcr["Ticker"],
                            y=valid_pcr["PCR (Volume)"],
                            marker_color=colors,
                            text=valid_pcr["Sentiment"],
                            textposition='outside'
                        ))
                        
                        fig_pcr.add_hline(y=0.7, line_dash="dash", line_color="green", annotation_text="Bullish Threshold")
                        fig_pcr.add_hline(y=1.0, line_dash="dash", line_color="red", annotation_text="Bearish Threshold")
                        
                        fig_pcr.update_layout(
                            yaxis_title="PCR (Volume)",
                            height=400
                        )
                        st.plotly_chart(fig_pcr, use_container_width=True)
                
                with col2:
                    st.markdown("### Implied Volatility")
                    
                    valid_iv = options_df[options_df["Call IV"].notna()]
                    if not valid_iv.empty:
                        fig_iv = go.Figure()
                        
                        fig_iv.add_trace(go.Bar(
                            name="Call IV",
                            x=valid_iv["Ticker"],
                            y=valid_iv["Call IV"],
                            marker_color='lightblue'
                        ))
                        
                        fig_iv.add_trace(go.Bar(
                            name="Put IV",
                            x=valid_iv["Ticker"],
                            y=valid_iv["Put IV"],
                            marker_color='lightcoral'
                        ))
                        
                        fig_iv.update_layout(
                            barmode='group',
                            yaxis_title="IV %",
                            height=400
                        )
                        st.plotly_chart(fig_iv, use_container_width=True)
                
                # Volume analysis
                st.markdown("### 📊 Options Volume Analysis")
                
                fig_volume = go.Figure()
                
                fig_volume.add_trace(go.Bar(
                    name="Call Volume",
                    x=options_df["Ticker"],
                    y=options_df["Call Volume"],
                    marker_color='green'
                ))
                
                fig_volume.add_trace(go.Bar(
                    name="Put Volume",
                    x=options_df["Ticker"],
                    y=options_df["Put Volume"],
                    marker_color='red'
                ))
                
                fig_volume.update_layout(
                    barmode='group',
                    yaxis_title="Volume",
                    height=400
                )
                st.plotly_chart(fig_volume, use_container_width=True)
                
                # Open Interest
                st.markdown("### 📈 Open Interest")
                
                fig_oi = go.Figure()
                
                fig_oi.add_trace(go.Bar(
                    name="Call OI",
                    x=options_df["Ticker"],
                    y=options_df["Call OI"],
                    marker_color='lightgreen'
                ))
                
                fig_oi.add_trace(go.Bar(
                    name="Put OI",
                    x=options_df["Ticker"],
                    y=options_df["Put OI"],
                    marker_color='lightcoral'
                ))
                
                fig_oi.update_layout(
                    barmode='group',
                    yaxis_title="Open Interest",
                    height=400
                )
                st.plotly_chart(fig_oi, use_container_width=True)
            
            # ══════════════════════════════════════════════════════════════════
            # TAB 3: TECHNICAL SENTIMENT
            # ══════════════════════════════════════════════════════════════════
            with tab3:
                st.markdown("## 📈 Technical Sentiment Indicators")
                
                # Technical data table
                tech_data = []
                for ticker, data in sentiment_data.items():
                    tech_data.append({
                        "Ticker": ticker,
                        "RSI": data["rsi"],
                        "Price": data["current_price"],
                        "vs SMA20": "✅" if data["above_sma20"] else "❌" if data["above_sma20"] is not None else "-",
                        "vs SMA50": "✅" if data["above_sma50"] else "❌" if data["above_sma50"] is not None else "-",
                        "vs SMA200": "✅" if data["above_sma200"] else "❌" if data["above_sma200"] is not None else "-",
                        "BB Position": data["bb_position"],
                        "Volatility": data["volatility_30d"],
                        "Volume Ratio": data["volume_ratio"]
                    })
                
                tech_df = pd.DataFrame(tech_data)
                
                st.dataframe(
                    tech_df.style.format({
                        "RSI": "{:.1f}",
                        "Price": "${:.2f}",
                        "BB Position": "{:.1f}%",
                        "Volatility": "{:.1f}%",
                        "Volume Ratio": "{:.2f}x"
                    }, na_rep="-")
                    .background_gradient(cmap="RdYlGn", subset=["RSI"], vmin=0, vmax=100),
                    use_container_width=True
                )
                
                # RSI Chart
                st.markdown("### 📊 RSI (Relative Strength Index)")
                
                fig_rsi = go.Figure()
                
                valid_rsi = tech_df[tech_df["RSI"].notna()]
                if not valid_rsi.empty:
                    colors = ['red' if x > 70 else 'green' if x < 30 else 'gray' for x in valid_rsi["RSI"]]
                    
                    fig_rsi.add_trace(go.Bar(
                        x=valid_rsi["Ticker"],
                        y=valid_rsi["RSI"],
                        marker_color=colors
                    ))
                    
                    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
                    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
                    fig_rsi.add_hline(y=50, line_dash="dot", line_color="gray")
                    
                    fig_rsi.update_layout(
                        yaxis_title="RSI",
                        yaxis_range=[0, 100],
                        height=400
                    )
                    st.plotly_chart(fig_rsi, use_container_width=True)
                
                # Moving Average Analysis
                st.markdown("### 📈 Moving Average Alignment")
                
                ma_summary = []
                for ticker, data in sentiment_data.items():
                    above_count = sum([
                        data["above_sma20"] or False,
                        data["above_sma50"] or False,
                        data["above_sma200"] or False
                    ])
                    ma_summary.append({
                        "Ticker": ticker,
                        "MAs Above": f"{above_count}/3",
                        "Status": "Bullish" if above_count >= 2 else "Bearish" if above_count == 0 else "Mixed"
                    })
                
                st.dataframe(pd.DataFrame(ma_summary), use_container_width=True)
                
                # Price charts with MAs
                st.markdown("### 📊 Price vs Moving Averages")
                
                selected_ticker = st.selectbox("Select ticker for detailed view:", tickers)
                
                if selected_ticker:
                    hist = all_data[selected_ticker]["history"]
                    if not hist.empty:
                        fig_price = go.Figure()
                        
                        fig_price.add_trace(go.Candlestick(
                            x=hist.index,
                            open=hist['Open'],
                            high=hist['High'],
                            low=hist['Low'],
                            close=hist['Close'],
                            name='Price'
                        ))
                        
                        # Add MAs
                        fig_price.add_trace(go.Scatter(
                            x=hist.index,
                            y=hist['Close'].rolling(20).mean(),
                            name='SMA 20',
                            line=dict(color='blue', width=1)
                        ))
                        
                        if len(hist) >= 50:
                            fig_price.add_trace(go.Scatter(
                                x=hist.index,
                                y=hist['Close'].rolling(50).mean(),
                                name='SMA 50',
                                line=dict(color='orange', width=1)
                            ))
                        
                        if len(hist) >= 200:
                            fig_price.add_trace(go.Scatter(
                                x=hist.index,
                                y=hist['Close'].rolling(200).mean(),
                                name='SMA 200',
                                line=dict(color='red', width=1)
                            ))
                        
                        fig_price.update_layout(
                            title=f"{selected_ticker} - Price & Moving Averages",
                            yaxis_title="Price",
                            height=500,
                            xaxis_rangeslider_visible=False
                        )
                        st.plotly_chart(fig_price, use_container_width=True)
            
            # ══════════════════════════════════════════════════════════════════
            # TAB 4: ANALYST & INSTITUTIONAL
            # ══════════════════════════════════════════════════════════════════
            with tab4:
                st.markdown("## 👔 Analyst Ratings & Institutional Activity")
                
                # Analyst ratings table
                analyst_data = []
                for ticker, data in sentiment_data.items():
                    analyst_data.append({
                        "Ticker": ticker,
                        "Buy": data["analyst_buys"],
                        "Hold": data["analyst_holds"],
                        "Sell": data["analyst_sells"],
                        "Buy %": data["analyst_buy_pct"],
                        "Analyst Score": data["analyst_score"],
                        "Short Ratio": data["short_ratio"],
                        "Short %": data["short_percent"]
                    })
                
                analyst_df = pd.DataFrame(analyst_data)
                
                st.dataframe(
                    analyst_df.style.format({
                        "Buy": "{:.0f}",
                        "Hold": "{:.0f}",
                        "Sell": "{:.0f}",
                        "Buy %": "{:.1f}%",
                        "Analyst Score": "{:.1f}",
                        "Short Ratio": "{:.2f}",
                        "Short %": "{:.2f}%"
                    }, na_rep="-")
                    .background_gradient(cmap="RdYlGn", subset=["Analyst Score"], vmin=0, vmax=100),
                    use_container_width=True
                )
                
                # Analyst ratings visualization
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 📊 Analyst Recommendations")
                    
                    fig_analyst = go.Figure()
                    
                    fig_analyst.add_trace(go.Bar(
                        name="Buy",
                        x=analyst_df["Ticker"],
                        y=analyst_df["Buy"],
                        marker_color='green'
                    ))
                    
                    fig_analyst.add_trace(go.Bar(
                        name="Hold",
                        x=analyst_df["Ticker"],
                        y=analyst_df["Hold"],
                        marker_color='gray'
                    ))
                    
                    fig_analyst.add_trace(go.Bar(
                        name="Sell",
                        x=analyst_df["Ticker"],
                        y=analyst_df["Sell"],
                        marker_color='red'
                    ))
                    
                    fig_analyst.update_layout(
                        barmode='stack',
                        yaxis_title="Count",
                        height=400
                    )
                    st.plotly_chart(fig_analyst, use_container_width=True)
                
                with col2:
                    st.markdown("### 🎯 Analyst Consensus Score")
                    
                    valid_scores = analyst_df[analyst_df["Analyst Score"].notna()]
                    if not valid_scores.empty:
                        colors = ['green' if x >= 65 else 'red' if x < 45 else 'gray' 
                                 for x in valid_scores["Analyst Score"]]
                        
                        fig_score = go.Figure()
                        
                        fig_score.add_trace(go.Bar(
                            x=valid_scores["Ticker"],
                            y=valid_scores["Analyst Score"],
                            marker_color=colors
                        ))
                        
                        fig_score.add_hline(y=50, line_dash="dash", line_color="gray")
                        
                        fig_score.update_layout(
                            yaxis_title="Score (0-100)",
                            yaxis_range=[0, 100],
                            height=400
                        )
                        st.plotly_chart(fig_score, use_container_width=True)
                
                # Short interest
                st.markdown("### 📉 Short Interest Analysis")
                
                valid_short = analyst_df[analyst_df["Short %"].notna()]
                if not valid_short.empty:
                    fig_short = go.Figure()
                    
                    colors = ['red' if x > 10 else 'orange' if x > 5 else 'green' 
                             for x in valid_short["Short %"]]
                    
                    fig_short.add_trace(go.Bar(
                        x=valid_short["Ticker"],
                        y=valid_short["Short %"],
                        marker_color=colors,
                        text=valid_short["Short %"],
                        texttemplate='%{text:.1f}%',
                        textposition='outside'
                    ))
                    
                    fig_short.update_layout(
                        title="Short Interest % of Float",
                        yaxis_title="Short %",
                        height=400
                    )
                    st.plotly_chart(fig_short, use_container_width=True)
                    
                    st.caption("**Note:** High short interest (>10%) can indicate bearish sentiment OR potential short squeeze opportunity")
                
                # Recent recommendations timeline
                st.markdown("### 📅 Recent Analyst Actions")
                
                selected_ticker_recs = st.selectbox("Select ticker:", tickers, key="recs_ticker")
                
                if selected_ticker_recs:
                    recs = all_data[selected_ticker_recs]["recommendations"]
                    if not recs.empty and 'To Grade' in recs.columns:
                        recent = recs.tail(10).sort_index(ascending=False)
                        st.dataframe(
                            recent[['Firm', 'To Grade', 'Action']].reset_index(),
                            use_container_width=True,
                            height=300
                        )
                    else:
                        st.info("No recent analyst recommendations available")
            
            # ══════════════════════════════════════════════════════════════════
            # TAB 5: FEAR & GREED
            # ══════════════════════════════════════════════════════════════════
            with tab5:
                st.markdown("## 😱 Fear & Greed Indicators")
                
                # VIX detailed analysis
                if not vix_hist.empty and vix_current:
                    col1, col2, col3 = st.columns(3)
                    
                    vix_interp, vix_color = interpret_vix(vix_current)
                    vix_change = vix_hist['Close'].pct_change().iloc[-1] * 100
                    vix_ma20 = vix_hist['Close'].rolling(20).mean().iloc[-1]
                    
                    with col1:
                        st.metric(
                            "Current VIX",
                            f"{vix_current:.2f}",
                            f"{vix_change:+.2f}%"
                        )
                    
                    with col2:
                        st.metric(
                            "20-Day Average",
                            f"{vix_ma20:.2f}"
                        )
                    
                    with col3:
                        st.markdown(f"### Market Mood")
                        st.markdown(f":{vix_color}[**{vix_interp}**]")
                    
                    # VIX historical chart
                    st.markdown("### 📊 VIX Historical Trend")
                    
                    fig_vix_full = go.Figure()
                    
                    fig_vix_full.add_trace(go.Scatter(
                        x=vix_hist.index,
                        y=vix_hist['Close'],
                        mode='lines',
                        fill='tozeroy',
                        name='VIX',
                        line=dict(color='darkred', width=2)
                    ))
                    
                    # Add threshold lines
                    fig_vix_full.add_hline(y=20, line_dash="dash", line_color="orange", 
                                          annotation_text="Elevated Fear")
                    fig_vix_full.add_hline(y=30, line_dash="dash", line_color="red", 
                                          annotation_text="High Fear")
                    
                    fig_vix_full.update_layout(
                        yaxis_title="VIX Level",
                        height=400
                    )
                    st.plotly_chart(fig_vix_full, use_container_width=True)
                
                # Fear & Greed gauge
                st.markdown("### 🎯 Fear & Greed Gauge")
                
                # Calculate simple fear/greed based on multiple factors
                for ticker in tickers:
                    data = sentiment_data[ticker]
                    
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.markdown(f"#### {ticker}")
                        
                        score = data["composite_sentiment"]
                        
                        if score >= 70:
                            emoji = "🤑"
                            label = "Extreme Greed"
                            color = "darkgreen"
                        elif score >= 55:
                            emoji = "😊"
                            label = "Greed"
                            color = "green"
                        elif score >= 45:
                            emoji = "😐"
                            label = "Neutral"
                            color = "gray"
                        elif score >= 30:
                            emoji = "😰"
                            label = "Fear"
                            color = "orange"
                        else:
                            emoji = "😱"
                            label = "Extreme Fear"
                            color = "red"
                        
                        st.metric(
                            "Sentiment",
                            f"{score:.1f}/100"
                        )
                        st.markdown(f"### {emoji} {label}")
                    
                    with col2:
                        # Gauge chart
                        fig_gauge = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=score,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': f"{ticker} Sentiment"},
                            gauge={
                                'axis': {'range': [None, 100]},
                                'bar': {'color': color},
                                'steps': [
                                    {'range': [0, 30], 'color': "lightcoral"},
                                    {'range': [30, 45], 'color': "lightyellow"},
                                    {'range': [45, 55], 'color': "lightgray"},
                                    {'range': [55, 70], 'color': "lightgreen"},
                                    {'range': [70, 100], 'color': "darkgreen"}
                                ],
                                'threshold': {
                                    'line': {'color': "black", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 50
                                }
                            }
                        ))
                        
                        fig_gauge.update_layout(height=250)
                        st.plotly_chart(fig_gauge, use_container_width=True)
                    
                    st.markdown("---")
            
            # ══════════════════════════════════════════════════════════════════
            # TAB 6: GOOGLE TRENDS
            # ══════════════════════════════════════════════════════════════════
            with tab6:
                st.markdown("## 🔍 Google Trends - Retail Interest")
                
                if not TRENDS_AVAILABLE:
                    st.warning("⚠️ Google Trends analysis requires pytrends package")
                    st.code("pip install pytrends", language="bash")
                    st.info("💡 Google Trends shows retail investor interest and search popularity")
                else:
                    st.info("🔄 Fetching Google Trends data... This may take a moment")
                    
                    try:
                        # Initialize pytrends
                        pytrends = TrendReq(hl='en-US', tz=360)
                        
                        # Fetch trends for each ticker
                        trends_data = {}
                        
                        for ticker in tickers[:5]:  # Limit to 5 for API rate limits
                            try:
                                # Build search query
                                search_term = f"{ticker} stock"
                                
                                # Get data for last 3 months
                                pytrends.build_payload([search_term], timeframe='today 3-m')
                                trends_df = pytrends.interest_over_time()
                                
                                if not trends_df.empty:
                                    trends_data[ticker] = trends_df[search_term]
                                
                            except Exception as e:
                                st.warning(f"Could not fetch trends for {ticker}: {str(e)}")
                        
                        if trends_data:
                            # Plot trends
                            st.markdown("### 📈 Search Interest Over Time")
                            
                            fig_trends = go.Figure()
                            
                            for ticker, data in trends_data.items():
                                fig_trends.add_trace(go.Scatter(
                                    x=data.index,
                                    y=data.values,
                                    name=ticker,
                                    mode='lines+markers'
                                ))
                            
                            fig_trends.update_layout(
                                yaxis_title="Search Interest (0-100)",
                                height=500,
                                hovermode='x unified'
                            )
                            st.plotly_chart(fig_trends, use_container_width=True)
                            
                            # Current interest levels
                            st.markdown("### 📊 Current Search Interest")
                            
                            current_interest = []
                            for ticker, data in trends_data.items():
                                current = data.iloc[-1]
                                avg = data.mean()
                                change = ((current - avg) / avg * 100) if avg > 0 else 0
                                
                                current_interest.append({
                                    "Ticker": ticker,
                                    "Current": current,
                                    "3-Month Avg": avg,
                                    "vs Avg": f"{change:+.1f}%",
                                    "Trend": "📈" if change > 20 else "📉" if change < -20 else "➡️"
                                })
                            
                            interest_df = pd.DataFrame(current_interest)
                            
                            st.dataframe(
                                interest_df.style.format({
                                    "Current": "{:.0f}",
                                    "3-Month Avg": "{:.1f}"
                                }).background_gradient(cmap="YlOrRd", subset=["Current"]),
                                use_container_width=True
                            )
                            
                            st.caption("**Note:** Google Trends shows relative search volume (0-100 scale)")
                        else:
                            st.warning("No trends data available")
                            
                    except Exception as e:
                        st.error(f"Error fetching Google Trends: {str(e)}")
                        st.info("💡 Tip: Try analyzing fewer tickers or check your internet connection")
        
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            with st.expander("🐛 Error Details"):
                st.exception(e)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    st.set_page_config(
        page_title="Sentiment Analysis",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    render_sentiment_analysis()
