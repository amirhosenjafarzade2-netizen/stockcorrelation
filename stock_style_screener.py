# stock_style_screener.py
# Production-ready multi-style stock screener with robust error handling

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date, timedelta
import time
from typing import List

# Try to import from screener.py
try:
    from screener import get_sp500_tickers, get_finviz_tickers
    HAS_SCREENER = True
except ImportError:
    HAS_SCREENER = False

# Known Dividend Aristocrats (S&P 500 companies with 25+ years of dividend increases)
DIVIDEND_ARISTOCRATS = {
    'MMM', 'ABBV', 'ABT', 'ADP', 'AFL', 'ALB', 'APD', 'ATR', 'BDX', 'BF.B',
    'CAT', 'CBSH', 'CHD', 'CHRW', 'CINF', 'CL', 'CLX', 'CTAS', 'CVX', 'DOV',
    'ECL', 'ED', 'EMR', 'ESS', 'EXPD', 'FRT', 'GD', 'GPC', 'GWW', 'HRL',
    'IBM', 'ITW', 'JNJ', 'KMB', 'KO', 'KVUE', 'LEG', 'LIN', 'LOW', 'MCD',
    'MDT', 'MKC', 'NDSN', 'NEE', 'NUE', 'O', 'PEP', 'PG', 'PNR', 'PPG',
    'QCOM', 'ROP', 'RSG', 'SHW', 'SPGI', 'SWK', 'SYY', 'TROW', 'TGT', 'UNP',
    'V', 'WMT', 'WST'
}

def fallback_sp500_tickers() -> List[str]:
    return ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "JPM", "V",
            "UNH", "MA", "XOM", "PG", "JNJ", "HD", "CVX", "MRK", "ABBV", "KO", "PEP"]

# ── Helper Functions ───────────────────────────────────────────────
def calculate_rsi(prices: pd.Series, period: int = 14) -> float:
    try:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if len(rsi) > 0 and not pd.isna(rsi.iloc[-1]) else np.nan
    except:
        return np.nan

def calculate_max_drawdown(prices: pd.Series) -> float:
    try:
        cummax = prices.cummax()
        drawdown = (prices - cummax) / cummax
        return drawdown.min() * 100
    except:
        return np.nan

def safe_divide(num, denom, default=0):
    try:
        if denom and denom > 0 and not pd.isna(num) and not pd.isna(denom):
            return num / denom
        return default
    except:
        return default

def safe_get(value, multiplier=1, default=np.nan):
    if value is not None and not pd.isna(value):
        return value * multiplier
    return default

# ── Volume Spike Helper ────────────────────────────────────────────
@st.cache_data(ttl=1800, show_spinner=False)
def get_high_volume_stocks(tickers, days_back=1, min_multiplier=2.0, avg_period=20, min_avg_volume=400_000):
    if not tickers:
        return pd.DataFrame()
    
    end = date.today()
    start = end - timedelta(days=days_back + avg_period + 30)
    
    try:
        data = yf.download(tickers, start=start, end=end, progress=False, threads=True)
    except:
        return pd.DataFrame()
    
    if 'Volume' not in data.columns.levels[0]:
        return pd.DataFrame()
    
    vol = data['Volume']
    close = data['Close'] if 'Close' in data else data['Adj Close']
    results = []
    
    for ticker in vol.columns:
        v = vol[ticker].dropna()
        if len(v) < avg_period + days_back:
            continue
        
        hist_vol = v.iloc[:-days_back] if days_back > 0 else v[:-1]
        avg_vol = hist_vol.tail(avg_period).mean()
        
        if pd.isna(avg_vol) or avg_vol < min_avg_volume:
            continue
        
        recent = v.tail(days_back) if days_back > 0 else v.tail(1)
        ratios = recent / avg_vol
        max_ratio = ratios.max()
        if max_ratio < min_multiplier:
            continue
        
        spike_idx = ratios.idxmax()
        spike_vol = recent.loc[spike_idx]
        spike_date = spike_idx.date()
        
        if spike_idx in close.index:
            prior_idx = close.index[close.index < spike_idx]
            prev_close = close.loc[prior_idx[-1]] if len(prior_idx) > 0 else np.nan
            price_chg = (close.loc[spike_idx] - prev_close) / prev_close * 100 if not pd.isna(prev_close) else np.nan
        else:
            price_chg = np.nan
        
        results.append({
            'Ticker': ticker,
            'Spike Date': spike_date,
            'Max Volume': int(spike_vol),
            'Avg Vol': int(avg_vol),
            'Volume ×': round(max_ratio, 2),
            'Price Δ %': round(price_chg, 1) if not pd.isna(price_chg) else np.nan,
            'Last Close': round(close.iloc[-1], 2) if not close.empty else np.nan
        })
    
    if not results:
        return pd.DataFrame()
    
    return pd.DataFrame(results).sort_values('Volume ×', ascending=False)

# ── Main Screener ──────────────────────────────────────────────────
def render_stock_style_screener():
    st.title("🎯 Production Stock Screener")
    st.caption("Robust filtering with rate limiting and error handling")
    
    col_u, col_s, col_p = st.columns([2, 2, 1])
    
    with col_u:
        universe = st.selectbox("Universe", ["S&P 500", "Custom list"], index=0)
    
    with col_s:
        style = st.selectbox("Stock Style", [
            "Consistent Growth", "Classic Value", "High Momentum", "High Quality / ROE",
            "Dividend Aristocrats", "Defensive / Low Volatility", "High Recent Volume",
            "GARP (Growth at Reasonable Price)", "High Free Cash Flow", "Low Debt Champions"
        ])
    
    with col_p:
        max_screen = st.slider("Max stocks", 50, 600, 150, step=50)
    
    with st.expander("⚙️ Advanced Settings"):
        min_liquidity = st.number_input("Min avg daily volume", 50_000, 5_000_000, 150_000, step=50_000)
        rate_limit_sleep = st.slider("API sleep (sec)", 0.0, 1.0, 0.35, 0.05)
    
    params = {}
    
    if style == "Consistent Growth":
        c1, c2, c3, c4 = st.columns(4)
        with c1: params['min_return'] = st.slider("Min return %", 8.0, 25.0, 12.0)
        with c2: params['max_vol'] = st.slider("Max vol %", 12.0, 35.0, 22.0)
        with c3: params['min_sharpe'] = st.slider("Min Sharpe", 0.4, 2.0, 0.8, 0.1)
        with c4: params['max_drawdown'] = st.slider("Max DD %", -50, -10, -25)
    
    elif style == "Classic Value":
        c1, c2, c3, c4 = st.columns(4)
        with c1: params['max_pe'] = st.slider("Max P/E", 10.0, 30.0, 18.0)
        with c2: params['max_pb'] = st.slider("Max P/B", 1.0, 5.0, 2.5)
        with c3: params['min_yield'] = st.slider("Min yield %", 0.0, 6.0, 2.0)
        with c4: params['max_debt_equity'] = st.slider("Max D/E", 0.5, 3.0, 1.5)
    
    elif style == "High Momentum":
        c1, c2, c3 = st.columns(3)
        with c1: params['min_6m_ret'] = st.slider("Min 6m %", 10.0, 60.0, 25.0)
        with c2: params['min_3m_ret'] = st.slider("Min 3m %", 5.0, 40.0, 15.0)
        with c3: params['max_rsi'] = st.slider("Max RSI", 50, 85, 70)
    
    elif style == "High Quality / ROE":
        c1, c2, c3 = st.columns(3)
        with c1: params['min_roe'] = st.slider("Min ROE %", 10.0, 40.0, 20.0)
        with c2: params['min_margin'] = st.slider("Min margin %", 5.0, 35.0, 12.0)
        with c3: params['min_roa'] = st.slider("Min ROA %", 5.0, 25.0, 10.0)
    
    elif style == "Dividend Aristocrats":
        c1, c2 = st.columns(2)
        with c1: params['min_yield'] = st.slider("Min yield %", 2.0, 8.0, 2.5)
        with c2: params['max_payout'] = st.slider("Max payout %", 30, 90, 70)
    
    elif style == "Defensive / Low Volatility":
        c1, c2 = st.columns(2)
        with c1: params['max_beta'] = st.slider("Max Beta", 0.4, 1.2, 0.85)
        with c2: params['max_vol'] = st.slider("Max vol %", 10.0, 25.0, 18.0)
    
    elif style == "High Recent Volume":
        c1, c2, c3 = st.columns(3)
        with c1: params['days_back'] = st.slider("Lookback days", 0, 10, 2)
        with c2: params['min_vol_mult'] = st.slider("Min multiplier", 1.5, 6.0, 2.5, 0.5)
        with c3: params['min_avg_vol'] = st.number_input("Min avg vol", 100_000, 2_000_000, 400_000, 100_000)
    
    elif style == "GARP (Growth at Reasonable Price)":
        c1, c2, c3 = st.columns(3)
        with c1: params['min_eps_growth'] = st.slider("Min EPS growth %", 10.0, 40.0, 20.0)
        with c2: params['max_peg'] = st.slider("Max PEG", 0.5, 2.5, 1.5)
        with c3: params['max_pe'] = st.slider("Max P/E", 15.0, 40.0, 25.0)
    
    elif style == "High Free Cash Flow":
        c1, c2 = st.columns(2)
        with c1: params['min_fcf_yield'] = st.slider("Min FCF yield %", 4.0, 15.0, 6.0)
        with c2: params['min_margin'] = st.slider("Min margin %", 8.0, 30.0, 12.0)
    
    elif style == "Low Debt Champions":
        c1, c2 = st.columns(2)
        with c1: params['max_debt_equity'] = st.slider("Max D/E", 0.1, 1.0, 0.5)
        with c2: params['min_current'] = st.slider("Min current ratio", 1.0, 3.0, 1.5)
    
    if st.button("🚀 Run Screener", type="primary", use_container_width=True):
        with st.spinner("Loading..."):
            if universe == "S&P 500":
                tickers = get_sp500_tickers() if HAS_SCREENER else fallback_sp500_tickers()
            else:
                custom = st.text_input("Comma separated tickers", "AAPL,MSFT,NVDA")
                tickers = [t.strip().upper() for t in custom.split(",") if t.strip()]
            
            tickers = tickers[:max_screen]
            
            if not tickers:
                st.error("No tickers to screen")
                return
            
            if style == "High Recent Volume":
                df = get_high_volume_stocks(tickers, params['days_back'], params['min_vol_mult'], 20, params['min_avg_vol'])
                if df.empty:
                    st.warning("No volume spikes found")
                else:
                    st.success(f"**{len(df)}** stocks with volume spikes")
                    st.dataframe(df.style.background_gradient(subset=['Volume ×'], cmap='Oranges'), use_container_width=True)
                    st.download_button("📥 CSV", df.to_csv(index=False), f"volume_{date.today()}.csv")
                return
            
            if style == "Dividend Aristocrats":
                tickers = [t for t in tickers if t in DIVIDEND_ARISTOCRATS]
                if not tickers:
                    st.warning("No Dividend Aristocrats in universe")
                    return
            
            try:
                start = date.today() - timedelta(days=1900)
                prices = yf.download(tickers, start=start, progress=False)['Adj Close']
            except:
                st.error("Download failed")
                return
            
            results = []
            progress = st.progress(0)
            skipped = 0
            
            for idx, ticker in enumerate(prices.columns):
                progress.progress((idx + 1) / len(prices.columns))
                
                try:
                    s = yf.Ticker(ticker)
                    info = s.info or {}
                    time.sleep(rate_limit_sleep)
                    
                    if style != "Dividend Aristocrats":
                        avg_vol = info.get('averageVolume', 0) or 0
                        if avg_vol < min_liquidity:
                            skipped += 1
                            continue
                    
                    p = prices[ticker].dropna()
                    if len(p) < 200:
                        continue
                    
                    ret_daily = p.pct_change().dropna()
                    ann_ret = (p.iloc[-1]/p.iloc[0]) ** (252/len(p)) - 1
                    ann_vol = ret_daily.std() * np.sqrt(252)
                    sharpe = safe_divide(ann_ret, ann_vol)
                    max_dd = calculate_max_drawdown(p)
                    
                    pe = safe_get(info.get('trailingPE'))
                    pb = safe_get(info.get('priceToBook'))
                    peg = safe_get(info.get('pegRatio'))
                    roe = safe_get(info.get('returnOnEquity'), 100)
                    roa = safe_get(info.get('returnOnAssets'), 100)
                    margin = safe_get(info.get('profitMargins'), 100)
                    yield_ = safe_get(info.get('dividendYield'), 100)
                    beta = safe_get(info.get('beta'))
                    payout = safe_get(info.get('payoutRatio'), 100)
                    debt_eq = safe_get(info.get('debtToEquity'), 0.01)
                    current = safe_get(info.get('currentRatio'))
                    earnings_gr = safe_get(info.get('earningsGrowth'), 100)
                    sector = info.get('sector', '—')
                    
                    fcf = info.get('freeCashflow')
                    mkt_cap = info.get('marketCap')
                    fcf_yield = (fcf / mkt_cap * 100) if (fcf and mkt_cap and mkt_cap > 0) else np.nan
                    
                    ret_3m = (p.iloc[-1] / p.iloc[-63]) - 1 if len(p) >= 63 else np.nan
                    ret_6m = (p.iloc[-1] / p.iloc[-126]) - 1 if len(p) >= 126 else np.nan
                    rsi = calculate_rsi(p)
                    
                    match = False
                    score = 0
                    
                    if style == "Consistent Growth":
                        match = (ann_ret*100 >= params['min_return'] and
                                ann_vol*100 <= params['max_vol'] and
                                sharpe >= params['min_sharpe'] and
                                max_dd >= params['max_drawdown'])
                        score = sharpe if not pd.isna(sharpe) else 0
                    
                    elif style == "Classic Value":
                        match = ((pe and pe > 0 and pe <= params['max_pe']) and
                                (pb and pb > 0 and pb <= params['max_pb']) and
                                (yield_ and yield_ >= params['min_yield']) and
                                (debt_eq and debt_eq <= params['max_debt_equity']))
                        score = (1/pe if pe and pe > 0 else 0) + (yield_/10 if yield_ else 0)
                    
                    elif style == "High Momentum":
                        match = ((not pd.isna(ret_6m) and ret_6m*100 >= params['min_6m_ret']) and
                                (not pd.isna(ret_3m) and ret_3m*100 >= params['min_3m_ret']) and
                                (pd.isna(rsi) or rsi <= params['max_rsi']))
                        score = ret_6m if not pd.isna(ret_6m) else 0
                    
                    elif style == "High Quality / ROE":
                        match = ((roe and roe >= params['min_roe']) and
                                (margin and margin >= params['min_margin']) and
                                (roa and roa >= params['min_roa']))
                        score = roe if roe else 0
                    
                    elif style == "Dividend Aristocrats":
                        match = ((yield_ and yield_ >= params['min_yield']) and
                                (not payout or payout <= params['max_payout']))
                        score = yield_ if yield_ else 0
                    
                    elif style == "Defensive / Low Volatility":
                        match = ((beta and beta <= params['max_beta']) and
                                ann_vol*100 <= params['max_vol'])
                        score = safe_divide(1, ann_vol + 1e-6)
                    
                    elif style == "GARP (Growth at Reasonable Price)":
                        match = ((earnings_gr and earnings_gr >= params['min_eps_growth']) and
                                (peg and peg > 0 and peg <= params['max_peg']) and
                                (pe and pe > 0 and pe <= params['max_pe']))
                        score = earnings_gr / peg if (peg and peg > 0 and earnings_gr) else 0
                    
                    elif style == "High Free Cash Flow":
                        match = ((not pd.isna(fcf_yield) and fcf_yield >= params['min_fcf_yield']) and
                                (margin and margin >= params['min_margin']))
                        score = fcf_yield if not pd.isna(fcf_yield) else 0
                    
                    elif style == "Low Debt Champions":
                        match = ((debt_eq and debt_eq <= params['max_debt_equity']) and
                                (current and current >= params['min_current']))
                        score = safe_divide(1, debt_eq + 0.01)
                    
                    if match:
                        results.append({
                            'Ticker': ticker,
                            'Sector': sector,
                            'Price': round(p.iloc[-1], 2),
                            'Ret %': round(ann_ret*100, 1),
                            'Vol %': round(ann_vol*100, 1),
                            'Sharpe': round(sharpe, 2) if not pd.isna(sharpe) else '—',
                            'DD %': round(max_dd, 1) if not pd.isna(max_dd) else '—',
                            'P/E': round(pe, 1) if pe else '—',
                            'P/B': round(pb, 1) if pb else '—',
                            'PEG': round(peg, 2) if peg else '—',
                            'ROE %': round(roe, 1) if roe else '—',
                            'Yield %': round(yield_, 1) if yield_ else '—',
                            'Beta': round(beta, 2) if beta else '—',
                            'D/E': round(debt_eq, 2) if debt_eq else '—',
                            'Score': round(score, 3)
                        })
                
                except Exception as e:
                    continue
            
            progress.empty()
            
            if not results:
                st.warning(f"No matches. ({skipped} skipped for low liquidity)")
                return
            
            df = pd.DataFrame(results).sort_values('Score', ascending=False)
            
            st.success(f"✅ **{len(df)}** matches ({skipped} skipped for liquidity)")
            
            c1, c2, c3, c4 = st.columns(4)
            with c1: st.metric("Found", len(df))
            with c2: st.metric("Avg Score", f"{df['Score'].mean():.2f}")
            with c3:
                if 'Ret %' in df.columns:
                    st.metric("Avg Return", f"{df['Ret %'].mean():.1f}%")
            with c4:
                if 'Sharpe' in df.columns:
                    sharpe_vals = [x for x in df['Sharpe'] if x != '—']
                    if sharpe_vals:
                        st.metric("Avg Sharpe", f"{np.mean(sharpe_vals):.2f}")
            
            st.dataframe(
                df.style.background_gradient(subset=['Score'], cmap='YlGn'),
                use_container_width=True,
                height=(min(20, len(df)) + 1) * 35
            )
            
            st.download_button(
                "📥 Download CSV",
                df.to_csv(index=False),
                f"{style.lower().replace(' ','_').replace('/','_')}_{date.today().strftime('%Y%m%d')}.csv",
                "text/csv"
            )

if __name__ == "__main__":
    st.set_page_config(page_title="Production Stock Screener", layout="wide")
    render_stock_style_screener()
