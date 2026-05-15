# insider_institutional.py
# 🏦 Insider & Institutional Activity Module
# Surfaces insider transactions, institutional ownership, short interest,
# and ownership structure — the "smart money" behavioral signals.

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════════════════
# DATA LAYER
# ══════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_all_holder_data(ticker: str) -> Optional[Dict]:
    """
    Fetch all ownership / insider data for a single ticker.
    Returns a dict of DataFrames + scalar info fields, or None on failure.
    """
    try:
        stock = yf.Ticker(ticker)
        info  = stock.info or {}

        # ── DataFrames ────────────────────────────────────────────
        institutional   = _safe_df(stock.institutional_holders)
        mutualfund      = _safe_df(stock.mutualfund_holders)
        insider_txns    = _safe_df(stock.insider_transactions)
        insider_buys    = _safe_df(stock.insider_purchases)
        insider_roster  = _safe_df(stock.insider_roster_holders)
        major           = _safe_df(stock.major_holders)

        # ── Scalar ownership fields from info ─────────────────────
        ownership_info = {
            # float / shares
            'shares_outstanding':         info.get('sharesOutstanding'),
            'float_shares':               info.get('floatShares'),
            'shares_short':               info.get('sharesShort'),
            'shares_short_prior':         info.get('sharesShortPriorMonth'),
            'short_ratio':                info.get('shortRatio'),           # days-to-cover
            'short_pct_of_float':         info.get('shortPercentOfFloat'),
            # ownership %
            'pct_held_by_institutions':   info.get('heldPercentInstitutions'),
            'pct_held_by_insiders':       info.get('heldPercentInsiders'),
            # price / market data
            'current_price':              info.get('currentPrice') or info.get('regularMarketPrice'),
            'market_cap':                 info.get('marketCap'),
            'company_name':               info.get('longName', ticker),
            'sector':                     info.get('sector', 'N/A'),
            'industry':                   info.get('industry', 'N/A'),
            # 52-week range
            '52w_high':                   info.get('fiftyTwoWeekHigh'),
            '52w_low':                    info.get('fiftyTwoWeekLow'),
        }

        return {
            'ticker':           ticker,
            'info':             ownership_info,
            'institutional':    institutional,
            'mutualfund':       mutualfund,
            'insider_txns':     insider_txns,
            'insider_buys':     insider_buys,
            'insider_roster':   insider_roster,
            'major':            major,
        }

    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return None


def _safe_df(obj) -> pd.DataFrame:
    """Return obj if it's a non-empty DataFrame, else empty DataFrame."""
    if isinstance(obj, pd.DataFrame) and not obj.empty:
        return obj.copy()
    return pd.DataFrame()


# ══════════════════════════════════════════════════════════════
# FORMATTING HELPERS
# ══════════════════════════════════════════════════════════════

def _fmt_shares(n) -> str:
    if n is None or (isinstance(n, float) and np.isnan(n)):
        return "N/A"
    n = float(n)
    if abs(n) >= 1e9:  return f"{n/1e9:.2f}B"
    if abs(n) >= 1e6:  return f"{n/1e6:.2f}M"
    if abs(n) >= 1e3:  return f"{n/1e3:.1f}K"
    return f"{n:,.0f}"


def _fmt_money(n) -> str:
    if n is None or (isinstance(n, float) and np.isnan(n)):
        return "N/A"
    n = float(n)
    if abs(n) >= 1e12: return f"${n/1e12:.2f}T"
    if abs(n) >= 1e9:  return f"${n/1e9:.2f}B"
    if abs(n) >= 1e6:  return f"${n/1e6:.2f}M"
    if abs(n) >= 1e3:  return f"${n/1e3:.1f}K"
    return f"${n:,.0f}"


def _fmt_pct(n, decimals=1) -> str:
    if n is None or (isinstance(n, float) and np.isnan(n)):
        return "N/A"
    return f"{float(n)*100:.{decimals}f}%"


def _signal_color(label: str) -> str:
    """Map a bullish/bearish/neutral label to hex color."""
    if label == "Bullish":   return "#27ae60"
    if label == "Bearish":   return "#e74c3c"
    return "#f39c12"


def _render_summary_card(title: str, value: str, subtitle: str,
                          color: str = "#1f77b4"):
    bg = {"#27ae60": "#d4edda", "#e74c3c": "#f8d7da",
          "#f39c12": "#fff3cd"}.get(color, "#e8f4fd")
    st.markdown(f"""
    <div style="background:{bg}; border-left:5px solid {color};
                padding:12px 16px; border-radius:8px; margin:4px 0;">
        <div style="font-size:0.78em; color:#555; font-weight:600;
                    text-transform:uppercase; letter-spacing:.04em;">{title}</div>
        <div style="font-size:1.35em; font-weight:800; color:{color};
                    margin:2px 0;">{value}</div>
        <div style="font-size:0.82em; color:#444;">{subtitle}</div>
    </div>""", unsafe_allow_html=True)


def _score_badge(label: str, value: str, color: str):
    css = {"green": "background:#d4edda;color:#155724",
           "orange": "background:#fff3cd;color:#856404",
           "red": "background:#f8d7da;color:#721c24"}.get(color, "background:#e8f4fd;color:#004085")
    st.markdown(
        f'<span style="{css};padding:4px 12px;border-radius:20px;'
        f'font-weight:700;font-size:1em;display:inline-block">'
        f'{label}: {value}</span>',
        unsafe_allow_html=True
    )


# ══════════════════════════════════════════════════════════════
# SIGNAL ENGINE
# ══════════════════════════════════════════════════════════════

def _compute_insider_signal(insider_txns: pd.DataFrame,
                             insider_buys: pd.DataFrame) -> Dict:
    """
    Derive a composite insider signal from recent transaction data.
    Returns dict with signal label, score (0-10), and explanation.
    """
    if insider_txns.empty and insider_buys.empty:
        return {"label": "No Data", "score": 5,
                "explanation": "No insider transaction data available.",
                "buy_count": 0, "sell_count": 0,
                "buy_shares": 0, "sell_shares": 0}

    buy_count = sell_count = buy_shares = sell_shares = 0

    # ── Parse individual transactions ────────────────────────────
    if not insider_txns.empty:
        txns = insider_txns.copy()

        # Normalise column names defensively
        txns.columns = [c.strip() for c in txns.columns]

        # Identify purchase vs sale rows
        if 'Transaction' in txns.columns:
            buys  = txns[txns['Transaction'].str.contains('Purchase|Buy|Acquisition',
                                                          case=False, na=False)]
            sells = txns[txns['Transaction'].str.contains('Sale|Sell|Disposition',
                                                          case=False, na=False)]
        elif 'Text' in txns.columns:
            buys  = txns[txns['Text'].str.contains('Purchase|Buy',
                                                   case=False, na=False)]
            sells = txns[txns['Text'].str.contains('Sale|Sell',
                                                   case=False, na=False)]
        else:
            buys  = pd.DataFrame()
            sells = pd.DataFrame()

        shares_col = 'Shares' if 'Shares' in txns.columns else None

        buy_count   = len(buys)
        sell_count  = len(sells)
        if shares_col:
            buy_shares  = pd.to_numeric(buys[shares_col],  errors='coerce').sum()
            sell_shares = pd.to_numeric(sells[shares_col], errors='coerce').sum()

    # ── Parse summary purchase activity ──────────────────────────
    if not insider_buys.empty:
        ib = insider_buys.copy()
        # Row labels are in first column; values in 'Shares'
        label_col = ib.columns[0]
        if 'Shares' in ib.columns:
            def _row(keyword):
                mask = ib[label_col].astype(str).str.contains(keyword, case=False, na=False)
                rows = ib[mask]['Shares']
                return pd.to_numeric(rows, errors='coerce').iloc[0] if len(rows) else 0

            agg_buys  = _row('Purchase')
            agg_sells = _row('Sale')
            if agg_buys  and not np.isnan(float(agg_buys)):
                buy_shares  = max(buy_shares,  float(agg_buys))
            if agg_sells and not np.isnan(float(agg_sells)):
                sell_shares = max(sell_shares, float(agg_sells))

    # ── Score ─────────────────────────────────────────────────────
    total_shares = buy_shares + sell_shares
    if total_shares > 0:
        buy_ratio = buy_shares / total_shares
    elif buy_count + sell_count > 0:
        buy_ratio = buy_count / (buy_count + sell_count)
    else:
        buy_ratio = 0.5

    # Score 0-10: 10 = all buys, 0 = all sells, 5 = neutral
    score = round(buy_ratio * 10, 1)

    if score >= 7:
        label = "Bullish"
        explanation = (f"Insiders made {buy_count} purchase transaction(s) "
                       f"({_fmt_shares(buy_shares)} shares) vs "
                       f"{sell_count} sale(s) ({_fmt_shares(sell_shares)} shares). "
                       f"Cluster buying is a strong positive signal.")
    elif score <= 3:
        label = "Bearish"
        explanation = (f"Insiders sold predominantly: {sell_count} sale(s) "
                       f"({_fmt_shares(sell_shares)} shares) vs "
                       f"{buy_count} purchase(s) ({_fmt_shares(buy_shares)} shares). "
                       f"Elevated insider selling warrants caution.")
    else:
        label = "Neutral"
        explanation = (f"Mixed insider activity: {buy_count} purchase(s) / "
                       f"{sell_count} sale(s). No clear directional signal.")

    return {
        "label": label, "score": score, "explanation": explanation,
        "buy_count": buy_count, "sell_count": sell_count,
        "buy_shares": buy_shares, "sell_shares": sell_shares,
    }


def _compute_short_signal(info: Dict) -> Dict:
    """Derive short-interest signal from info scalars."""
    pct = info.get('short_pct_of_float')
    days = info.get('short_ratio')

    if pct is None:
        return {"label": "No Data", "score": 5,
                "pct": None, "days": None,
                "explanation": "Short interest data not available."}

    pct_float = float(pct)

    # High short interest = bearish (could also mean squeeze potential — noted in UI)
    if pct_float >= 0.15:
        label = "Elevated"
        score = 3
        explanation = (f"{pct_float*100:.1f}% of float sold short "
                       f"({days:.1f} days-to-cover). "
                       f"High short interest reflects negative market consensus — "
                       f"but also creates potential for a short squeeze.")
    elif pct_float >= 0.05:
        label = "Moderate"
        score = 5
        explanation = (f"{pct_float*100:.1f}% of float sold short "
                       f"({f'{days:.1f}' if days else 'N/A'} days-to-cover). "
                       f"Moderate short interest within normal range.")
    else:
        label = "Low"
        score = 7
        explanation = (f"Only {pct_float*100:.1f}% of float sold short. "
                       f"Low short interest suggests limited bearish conviction.")

    return {"label": label, "score": score,
            "pct": pct_float, "days": days,
            "explanation": explanation}


def _compute_institutional_signal(info: Dict,
                                   institutional: pd.DataFrame) -> Dict:
    """Signal from institutional ownership level and concentration."""
    pct_inst  = info.get('pct_held_by_institutions')
    pct_ins   = info.get('pct_held_by_insiders')

    n_holders = len(institutional) if not institutional.empty else 0

    if pct_inst is None:
        return {"label": "No Data", "score": 5,
                "explanation": "Institutional ownership data not available.",
                "pct_inst": None, "pct_ins": None, "n_holders": n_holders}

    pct = float(pct_inst)

    # 40-80% institutional ownership generally considered healthy
    if 0.40 <= pct <= 0.80:
        label = "Healthy"
        score = 7
        explanation = (f"{pct*100:.1f}% institutional ownership — "
                       f"strong smart-money validation without over-concentration risk.")
    elif pct > 0.80:
        label = "High Concentration"
        score = 5
        explanation = (f"{pct*100:.1f}% institutional ownership. "
                       f"High concentration means large coordinated selling could be disruptive.")
    elif pct > 0.15:
        label = "Moderate"
        score = 5
        explanation = f"{pct*100:.1f}% institutional ownership — below typical large-cap levels."
    else:
        label = "Low"
        score = 4
        explanation = (f"Only {pct*100:.1f}% held by institutions. "
                       f"Limited institutional validation; may be too small or too risky for funds.")

    return {"label": label, "score": score,
            "explanation": explanation,
            "pct_inst": pct, "pct_ins": pct_ins, "n_holders": n_holders}


def _composite_signal(insider_sig: Dict, short_sig: Dict,
                       inst_sig: Dict) -> Dict:
    """Weighted composite of the three signals."""
    # weights: insider 50%, institutional 30%, short 20%
    weights = {"insider": 0.50, "inst": 0.30, "short": 0.20}
    scores  = {
        "insider": insider_sig.get("score", 5),
        "inst":    inst_sig.get("score", 5),
        "short":   short_sig.get("score", 5),
    }
    composite = sum(weights[k] * scores[k] for k in weights)

    if composite >= 6.5:
        label = "Bullish"
    elif composite <= 4.0:
        label = "Bearish"
    else:
        label = "Neutral"

    return {"label": label, "score": round(composite, 1),
            "breakdown": scores, "weights": weights}


# ══════════════════════════════════════════════════════════════
# CHART BUILDERS
# ══════════════════════════════════════════════════════════════

def _chart_ownership_pie(pct_inst: float, pct_ins: float) -> go.Figure:
    """Donut chart: institutional / insider / public breakdown."""
    pct_public = max(0.0, 1.0 - pct_inst - pct_ins)
    labels  = ["Institutional", "Insider", "Public / Retail"]
    values  = [pct_inst * 100, pct_ins * 100, pct_public * 100]
    colors  = ["#3498db", "#e67e22", "#95a5a6"]

    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.48,
        marker=dict(colors=colors, line=dict(color='white', width=2)),
        textinfo='label+percent',
        textfont_size=13,
    ))
    fig.update_layout(
        title="Ownership Structure",
        height=340,
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=-0.15),
        margin=dict(t=50, b=60, l=10, r=10),
        template='plotly_white',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    return fig


def _chart_short_interest_gauge(pct: float, days: Optional[float]) -> go.Figure:
    """Gauge for short % of float."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct * 100,
        number={'suffix': '%', 'font': {'size': 28}},
        title={'text': 'Short % of Float', 'font': {'size': 15}},
        gauge={
            'axis': {'range': [0, 30], 'tickwidth': 1},
            'bar':  {'color': '#2c3e50', 'thickness': 0.25},
            'steps': [
                {'range': [0,  5],  'color': '#d4edda'},
                {'range': [5,  15], 'color': '#fff3cd'},
                {'range': [15, 30], 'color': '#f8d7da'},
            ],
            'threshold': {
                'line': {'color': '#e74c3c', 'width': 3},
                'thickness': 0.75,
                'value': 15,
            },
            'borderwidth': 1, 'bordercolor': '#ccc',
        }
    ))
    subtitle = f"Days-to-cover: {days:.1f}" if days else ""
    if subtitle:
        fig.add_annotation(
            text=subtitle, x=0.5, y=0.18,
            xref='paper', yref='paper',
            showarrow=False, font=dict(size=13, color='#555'),
        )
    fig.update_layout(
        height=280, margin=dict(t=60, b=20, l=20, r=20),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    )
    return fig


def _chart_insider_timeline(insider_txns: pd.DataFrame,
                             ticker: str) -> Optional[go.Figure]:
    """
    Scatter plot of insider transactions over time.
    Buys = green triangles up, Sells = red triangles down.
    """
    if insider_txns.empty:
        return None

    txns = insider_txns.copy()
    txns.columns = [c.strip() for c in txns.columns]

    # Date column
    date_col = next((c for c in txns.columns
                     if 'date' in c.lower() or 'start' in c.lower()), None)
    if date_col is None:
        return None
    txns[date_col] = pd.to_datetime(txns[date_col], errors='coerce')
    txns = txns.dropna(subset=[date_col])

    # Transaction type
    type_col = next((c for c in ['Transaction', 'Text'] if c in txns.columns), None)
    shares_col = 'Shares' if 'Shares' in txns.columns else None
    insider_col = next((c for c in ['Insider', 'filerName'] if c in txns.columns), None)
    position_col = next((c for c in ['Position', 'filerRelation'] if c in txns.columns), None)

    if type_col:
        txns['_is_buy'] = txns[type_col].str.contains(
            'Purchase|Buy|Acquisition', case=False, na=False)
    else:
        txns['_is_buy'] = False

    buys  = txns[txns['_is_buy']]
    sells = txns[~txns['_is_buy']]

    def _hover(df):
        parts = []
        if insider_col  and insider_col  in df.columns: parts.append(df[insider_col].astype(str))
        if position_col and position_col in df.columns: parts.append(df[position_col].astype(str))
        if shares_col   and shares_col   in df.columns:
            parts.append(df[shares_col].apply(lambda x: f"{_fmt_shares(x)} shares"))
        if not parts:
            return [""] * len(df)
        return pd.Series(["<br>".join(filter(None, row))
                          for row in zip(*parts)], index=df.index)

    fig = go.Figure()
    if not buys.empty:
        y_val = (pd.to_numeric(buys[shares_col], errors='coerce')
                 if shares_col else pd.Series([1]*len(buys), index=buys.index))
        fig.add_trace(go.Scatter(
            x=buys[date_col], y=y_val,
            mode='markers',
            name='Purchase',
            marker=dict(symbol='triangle-up', size=12,
                        color='#27ae60', line=dict(color='white', width=1)),
            hovertemplate='<b>PURCHASE</b><br>%{text}<br>Date: %{x}<extra></extra>',
            text=_hover(buys),
        ))
    if not sells.empty:
        y_val = (pd.to_numeric(sells[shares_col], errors='coerce')
                 if shares_col else pd.Series([1]*len(sells), index=sells.index))
        fig.add_trace(go.Scatter(
            x=sells[date_col], y=y_val,
            mode='markers',
            name='Sale',
            marker=dict(symbol='triangle-down', size=12,
                        color='#e74c3c', line=dict(color='white', width=1)),
            hovertemplate='<b>SALE</b><br>%{text}<br>Date: %{x}<extra></extra>',
            text=_hover(sells),
        ))

    fig.update_layout(
        title=f"Insider Transaction Timeline — {ticker}",
        xaxis_title="Date",
        yaxis_title="Shares Traded",
        template='plotly_white',
        height=380,
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        hovermode='closest',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    return fig


def _chart_buy_sell_bar(sig: Dict) -> go.Figure:
    """Horizontal bar: insider buy vs sell share volume."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=['Shares'],
        x=[sig['buy_shares']],
        name='Purchases',
        orientation='h',
        marker_color='#27ae60',
        text=[_fmt_shares(sig['buy_shares'])],
        textposition='auto',
    ))
    fig.add_trace(go.Bar(
        y=['Shares'],
        x=[sig['sell_shares']],
        name='Sales',
        orientation='h',
        marker_color='#e74c3c',
        text=[_fmt_shares(sig['sell_shares'])],
        textposition='auto',
    ))
    fig.update_layout(
        barmode='group',
        title='Insider Buy vs Sell Volume',
        template='plotly_white',
        height=220,
        margin=dict(t=45, b=20, l=10, r=10),
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        paper_bgcolor='rgba(0,0,0,0)',
    )
    return fig


def _chart_top_institutions(institutional: pd.DataFrame,
                             mutualfund: pd.DataFrame) -> Optional[go.Figure]:
    """Horizontal bar of top 10 holders by shares, colour-coded by type."""
    frames = []
    if not institutional.empty:
        df = institutional.copy()
        df['_type'] = 'Institution'
        frames.append(df)
    if not mutualfund.empty:
        df = mutualfund.copy()
        df['_type'] = 'Mutual Fund'
        frames.append(df)

    if not frames:
        return None

    combined = pd.concat(frames, ignore_index=True)
    shares_col = 'Shares' if 'Shares' in combined.columns else None
    holder_col = 'Holder' if 'Holder' in combined.columns else None

    if shares_col is None or holder_col is None:
        return None

    combined[shares_col] = pd.to_numeric(combined[shares_col], errors='coerce')
    combined = combined.dropna(subset=[shares_col])
    combined = combined.nlargest(12, shares_col)
    combined = combined.sort_values(shares_col)

    color_map = {'Institution': '#3498db', 'Mutual Fund': '#9b59b6'}
    colors = combined['_type'].map(color_map).fillna('#95a5a6')

    fig = go.Figure(go.Bar(
        x=combined[shares_col],
        y=combined[holder_col],
        orientation='h',
        marker_color=colors.tolist(),
        text=combined[shares_col].apply(_fmt_shares),
        textposition='outside',
        customdata=combined['_type'],
        hovertemplate='<b>%{y}</b><br>Shares: %{text}<br>Type: %{customdata}<extra></extra>',
    ))

    # Legend patches
    for label, color in color_map.items():
        fig.add_trace(go.Bar(
            x=[None], y=[None], name=label,
            marker_color=color, showlegend=True,
        ))

    fig.update_layout(
        title='Top Institutional & Fund Holders',
        xaxis_title='Shares Held',
        template='plotly_white',
        height=max(380, len(combined) * 32 + 100),
        margin=dict(t=50, b=30, l=10, r=80),
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        paper_bgcolor='rgba(0,0,0,0)',
        barmode='overlay',
    )
    return fig


def _chart_composite_gauge(score: float, label: str) -> go.Figure:
    """Gauge showing composite signal score 0-10."""
    color = _signal_color(label)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={'font': {'size': 34, 'color': color}},
        title={'text': 'Composite Signal Score (0–10)', 'font': {'size': 14}},
        gauge={
            'axis': {'range': [0, 10], 'tickwidth': 1},
            'bar':  {'color': color, 'thickness': 0.3},
            'steps': [
                {'range': [0,   4],  'color': '#f8d7da'},
                {'range': [4,   6.5],'color': '#fff3cd'},
                {'range': [6.5, 10], 'color': '#d4edda'},
            ],
            'borderwidth': 1, 'bordercolor': '#ccc',
        }
    ))
    fig.update_layout(
        height=260,
        margin=dict(t=55, b=15, l=20, r=20),
        paper_bgcolor='rgba(0,0,0,0)',
    )
    return fig


def _chart_signal_waterfall(comp: Dict) -> go.Figure:
    """Waterfall-style contribution chart for composite signal."""
    labels = ['Insider\n(50%)', 'Institutional\n(30%)', 'Short Interest\n(20%)', 'Composite']
    scores = list(comp['breakdown'].values())
    weights = list(comp['weights'].values())
    contribs = [s * w for s, w in zip(scores, weights)]

    fig = go.Figure(go.Waterfall(
        orientation='v',
        measure=['relative', 'relative', 'relative', 'total'],
        x=labels,
        y=contribs + [0],
        text=[f"{c:.2f}" for c in contribs] + [f"{comp['score']:.1f}"],
        textposition='outside',
        connector={'line': {'color': '#636363'}},
        increasing={'marker': {'color': '#27ae60'}},
        decreasing={'marker': {'color': '#e74c3c'}},
        totals={'marker': {'color': '#2c3e50'}},
    ))
    fig.add_hline(y=5, line_dash='dash', line_color='#aaa',
                  annotation_text='Neutral (5.0)', annotation_position='top right')
    fig.update_layout(
        title='Composite Score Build-Up',
        yaxis=dict(range=[0, 10.5], title='Weighted Score'),
        template='plotly_white',
        height=340,
        paper_bgcolor='rgba(0,0,0,0)',
    )
    return fig


# ══════════════════════════════════════════════════════════════
# SECTION RENDERERS
# ══════════════════════════════════════════════════════════════

def _render_header(data: Dict):
    info = data['info']
    name  = info.get('company_name', data['ticker'])
    price = info.get('current_price')
    mcap  = info.get('market_cap')
    sector = info.get('sector', 'N/A')
    industry = info.get('industry', 'N/A')

    st.markdown(f"### {name} `{data['ticker']}`")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Price",       f"${price:.2f}" if price else "N/A")
    c2.metric("Market Cap",  _fmt_money(mcap))
    c3.metric("Sector",      sector)
    c4.metric("Industry",    industry)
    st.divider()


def _render_composite_signal(comp: Dict, insider_sig: Dict,
                               short_sig: Dict, inst_sig: Dict):
    color = _signal_color(comp['label'])
    bg    = {"Bullish": "#d4edda", "Bearish": "#f8d7da"}.get(comp['label'], "#fff3cd")

    st.markdown(f"""
    <div style="background:{bg}; border-left:7px solid {color};
                padding:18px 22px; border-radius:10px; margin:10px 0 18px;">
        <div style="font-size:1.7em; font-weight:900; color:{color};">
            Composite Signal: {comp['label']}
        </div>
        <div style="font-size:1.05em; color:#333; margin-top:6px;">
            Score <b>{comp['score']:.1f} / 10</b> &nbsp;·&nbsp;
            Insider {insider_sig['label']} &nbsp;·&nbsp;
            Institutional {inst_sig['label']} &nbsp;·&nbsp;
            Short Interest {short_sig['label']}
        </div>
        <div style="color:#555; font-size:0.88em; margin-top:8px;">
            Composite = 50% insider signal + 30% institutional ownership signal + 20% short-interest signal.
            Score ≥6.5 = Bullish · ≤4.0 = Bearish · between = Neutral.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.plotly_chart(_chart_composite_gauge(comp['score'], comp['label']),
                        use_container_width=True)
    with col2:
        st.plotly_chart(_chart_signal_waterfall(comp), use_container_width=True)


def _render_insider_section(data: Dict, sig: Dict):
    st.subheader("🔑 Insider Transactions")

    color = _signal_color(sig['label'])
    c1, c2, c3, c4 = st.columns(4)
    with c1: _render_summary_card("Insider Signal",   sig['label'],
                                   f"Score {sig['score']:.1f}/10", color)
    with c2: _render_summary_card("Purchase Txns",    str(sig['buy_count']),
                                   f"{_fmt_shares(sig['buy_shares'])} shares", "#27ae60")
    with c3: _render_summary_card("Sale Txns",        str(sig['sell_count']),
                                   f"{_fmt_shares(sig['sell_shares'])} shares", "#e74c3c")
    with c4:
        net = sig['buy_shares'] - sig['sell_shares']
        net_color = "#27ae60" if net >= 0 else "#e74c3c"
        _render_summary_card("Net Insider Flow",
                              _fmt_shares(abs(net)),
                              "net purchases" if net >= 0 else "net sales",
                              net_color)

    st.caption(f"💡 {sig['explanation']}")

    # Timeline chart
    fig_timeline = _chart_insider_timeline(data['insider_txns'], data['ticker'])
    fig_bar      = _chart_buy_sell_bar(sig)

    if fig_timeline:
        col1, col2 = st.columns([3, 2])
        with col1:
            st.plotly_chart(fig_timeline, use_container_width=True)
        with col2:
            st.plotly_chart(fig_bar, use_container_width=True)
    else:
        if sig['buy_shares'] > 0 or sig['sell_shares'] > 0:
            st.plotly_chart(fig_bar, use_container_width=True)

    # Insider roster
    if not data['insider_roster'].empty:
        with st.expander("📋 Insider Roster (Current Holdings)", expanded=False):
            roster = data['insider_roster'].copy()

            # Drop URL columns
            url_cols = [c for c in roster.columns if 'url' in c.lower() or 'URL' in c]
            roster = roster.drop(columns=url_cols, errors='ignore')

            # Format share columns
            share_cols = [c for c in roster.columns
                          if 'shares' in c.lower() or 'position' in c.lower()]
            for col in share_cols:
                roster[col] = pd.to_numeric(roster[col], errors='coerce').apply(
                    lambda x: _fmt_shares(x) if pd.notna(x) else "—"
                )

            st.dataframe(roster, use_container_width=True, hide_index=True)

    # Raw transaction table
    if not data['insider_txns'].empty:
        with st.expander("📄 Raw Insider Transactions", expanded=False):
            txns = data['insider_txns'].copy()
            url_cols = [c for c in txns.columns if 'url' in c.lower() or c == 'URL']
            txns = txns.drop(columns=url_cols, errors='ignore')

            share_cols = [c for c in txns.columns if 'shares' in c.lower()]
            val_cols   = [c for c in txns.columns if 'value'  in c.lower()]
            for col in share_cols:
                txns[col] = pd.to_numeric(txns[col], errors='coerce').apply(
                    lambda x: _fmt_shares(x) if pd.notna(x) else "—")
            for col in val_cols:
                txns[col] = pd.to_numeric(txns[col], errors='coerce').apply(
                    lambda x: _fmt_money(x) if pd.notna(x) else "—")

            st.dataframe(txns, use_container_width=True, hide_index=True)

    # Purchases summary table (net share purchase activity)
    if not data['insider_buys'].empty:
        with st.expander("📊 Insider Purchase Activity Summary", expanded=False):
            st.dataframe(data['insider_buys'], use_container_width=True, hide_index=True)


def _render_institutional_section(data: Dict, sig: Dict):
    st.subheader("🏛️ Institutional Ownership")

    info = data['info']
    pct_inst = info.get('pct_held_by_institutions')
    pct_ins  = info.get('pct_held_by_insiders')
    shares_out = info.get('shares_outstanding')
    float_sh   = info.get('float_shares')

    color = {"Healthy": "#27ae60", "Low": "#e74c3c",
             "High Concentration": "#f39c12"}.get(sig['label'], "#1f77b4")

    c1, c2, c3, c4 = st.columns(4)
    with c1: _render_summary_card("Institutional Held",
                                   _fmt_pct(pct_inst), f"{sig['n_holders']} holders", color)
    with c2: _render_summary_card("Insider Held",
                                   _fmt_pct(pct_ins), "of shares outstanding", "#e67e22")
    with c3: _render_summary_card("Shares Outstanding",
                                   _fmt_shares(shares_out), "total issued", "#1f77b4")
    with c4: _render_summary_card("Float",
                                   _fmt_shares(float_sh), "freely tradeable shares", "#8e44ad")

    st.caption(f"💡 {sig['explanation']}")

    # Ownership pie + top holders chart
    col1, col2 = st.columns([1, 1])

    with col1:
        if pct_inst is not None and pct_ins is not None:
            st.plotly_chart(_chart_ownership_pie(float(pct_inst), float(pct_ins)),
                            use_container_width=True)
        else:
            st.info("Ownership breakdown data not available.")

    with col2:
        fig_top = _chart_top_institutions(data['institutional'], data['mutualfund'])
        if fig_top:
            st.plotly_chart(fig_top, use_container_width=True)
        else:
            st.info("Holder detail data not available.")

    # Institutional detail table
    if not data['institutional'].empty:
        with st.expander("📋 Top Institutional Holders", expanded=False):
            ih = data['institutional'].copy()
            if 'Shares' in ih.columns:
                ih['Shares'] = pd.to_numeric(ih['Shares'], errors='coerce').apply(
                    lambda x: _fmt_shares(x) if pd.notna(x) else "—")
            if 'Value' in ih.columns:
                ih['Value'] = pd.to_numeric(ih['Value'], errors='coerce').apply(
                    lambda x: _fmt_money(x) if pd.notna(x) else "—")
            if 'pctHeld' in ih.columns:
                ih['pctHeld'] = pd.to_numeric(ih['pctHeld'], errors='coerce').apply(
                    lambda x: f"{x*100:.2f}%" if pd.notna(x) else "—")
            st.dataframe(ih, use_container_width=True, hide_index=True)

    if not data['mutualfund'].empty:
        with st.expander("📋 Top Mutual Fund Holders", expanded=False):
            mf = data['mutualfund'].copy()
            if 'Shares' in mf.columns:
                mf['Shares'] = pd.to_numeric(mf['Shares'], errors='coerce').apply(
                    lambda x: _fmt_shares(x) if pd.notna(x) else "—")
            if 'Value' in mf.columns:
                mf['Value'] = pd.to_numeric(mf['Value'], errors='coerce').apply(
                    lambda x: _fmt_money(x) if pd.notna(x) else "—")
            if 'pctHeld' in mf.columns:
                mf['pctHeld'] = pd.to_numeric(mf['pctHeld'], errors='coerce').apply(
                    lambda x: f"{x*100:.2f}%" if pd.notna(x) else "—")
            st.dataframe(mf, use_container_width=True, hide_index=True)


def _render_short_interest_section(data: Dict, sig: Dict):
    st.subheader("📉 Short Interest")

    info = data['info']
    shares_short = info.get('shares_short')
    shares_short_prior = info.get('shares_short_prior')
    short_pct    = info.get('short_pct_of_float')
    short_ratio  = info.get('short_ratio')

    label_color = {"Low": "#27ae60", "Moderate": "#f39c12",
                   "Elevated": "#e74c3c"}.get(sig['label'], "#1f77b4")

    c1, c2, c3, c4 = st.columns(4)
    with c1: _render_summary_card("Short % of Float",
                                   _fmt_pct(short_pct),
                                   sig['label'], label_color)
    with c2: _render_summary_card("Shares Short",
                                   _fmt_shares(shares_short),
                                   "currently sold short", "#e74c3c")
    with c3: _render_summary_card("Prior Month Short",
                                   _fmt_shares(shares_short_prior),
                                   "one month ago", "#95a5a6")
    with c4:
        if shares_short and shares_short_prior and float(shares_short_prior) > 0:
            chg = (float(shares_short) - float(shares_short_prior)) / float(shares_short_prior)
            chg_color = "#e74c3c" if chg > 0 else "#27ae60"
            _render_summary_card("Short Interest Change",
                                  f"{chg:+.1%}",
                                  "vs prior month", chg_color)
        else:
            _render_summary_card("Days-to-Cover",
                                  f"{short_ratio:.1f}" if short_ratio else "N/A",
                                  "short ratio", "#8e44ad")

    st.caption(f"💡 {sig['explanation']}")

    col1, col2 = st.columns([1, 1])
    with col1:
        if short_pct is not None:
            st.plotly_chart(
                _chart_short_interest_gauge(float(short_pct), short_ratio),
                use_container_width=True
            )
        else:
            st.info("Short interest data not available.")

    with col2:
        # Month-over-month bar if both values present
        if shares_short and shares_short_prior:
            fig_si = go.Figure()
            fig_si.add_trace(go.Bar(
                x=['Prior Month', 'Current'],
                y=[float(shares_short_prior), float(shares_short)],
                marker_color=['#95a5a6', '#e74c3c'],
                text=[_fmt_shares(shares_short_prior), _fmt_shares(shares_short)],
                textposition='outside',
            ))
            fig_si.update_layout(
                title='Short Interest: Month-over-Month',
                yaxis_title='Shares Short',
                template='plotly_white', height=280,
                margin=dict(t=45, b=20, l=10, r=10),
                paper_bgcolor='rgba(0,0,0,0)',
            )
            st.plotly_chart(fig_si, use_container_width=True)
        else:
            st.info("Month-over-month comparison unavailable.")

    # Short squeeze potential assessment
    if short_pct is not None and float(short_pct) >= 0.10:
        with st.expander("⚡ Short Squeeze Potential", expanded=True):
            squeeze_score = min(10, float(short_pct) * 50 + (
                (10 - float(short_ratio)) if short_ratio else 0) / 2)
            st.markdown(f"""
**Short squeeze conditions present:**

| Factor | Value | Assessment |
|---|---|---|
| Short % of Float | {float(short_pct)*100:.1f}% | {'🔥 High' if float(short_pct) >= 0.20 else '⚠️ Elevated'} |
| Days-to-Cover | {f'{float(short_ratio):.1f}' if short_ratio else 'N/A'} | {'🔥 Difficult to cover' if short_ratio and float(short_ratio) >= 5 else '⚠️ Moderate'} |

*A squeeze occurs when rising prices force short sellers to cover, creating a buying cascade.
High short interest + positive catalyst = elevated squeeze potential.*
            """)


def _render_combined_view(data: Dict, insider_sig: Dict,
                           inst_sig: Dict, short_sig: Dict):
    """
    Compact summary across all three signals in a single score card grid.
    Useful for multi-ticker comparison.
    """
    st.subheader("📊 Signal Summary Dashboard")

    col1, col2, col3 = st.columns(3)

    with col1:
        color = _signal_color(insider_sig['label'])
        _render_summary_card(
            "🔑 Insider Signal",
            f"{insider_sig['label']}",
            f"Score {insider_sig['score']:.1f}/10 · "
            f"{insider_sig['buy_count']} buys / {insider_sig['sell_count']} sells",
            color
        )

    with col2:
        color = {"Healthy": "#27ae60", "Low": "#e74c3c",
                 "High Concentration": "#f39c12"}.get(inst_sig['label'], "#1f77b4")
        _render_summary_card(
            "🏛️ Institutional Ownership",
            f"{inst_sig['label']}",
            f"Score {inst_sig['score']:.1f}/10 · "
            f"{_fmt_pct(inst_sig.get('pct_inst'))} held · {inst_sig['n_holders']} holders",
            color
        )

    with col3:
        color = {"Low": "#27ae60", "Moderate": "#f39c12",
                 "Elevated": "#e74c3c"}.get(short_sig['label'], "#1f77b4")
        _render_summary_card(
            "📉 Short Interest",
            f"{short_sig['label']}",
            f"Score {short_sig['score']:.1f}/10 · "
            f"{_fmt_pct(short_sig.get('pct'))} of float short",
            color
        )


def _render_education():
    with st.expander("📚 Interpretation Guide", expanded=False):
        st.markdown("""
| Signal | What it measures | Bullish threshold | Bearish threshold |
|---|---|---|---|
| **Insider Signal** | Net buying vs selling by officers, directors, 10%+ holders | Cluster buys, >70% buy ratio | Persistent selling across multiple insiders |
| **Institutional Ownership** | % held by funds and institutions | 40–80% (smart-money validation) | <15% (too risky/small for funds) or >85% (crowding risk) |
| **Short Interest** | % of float sold short | <5% (low bearish conviction) | >15% (high bearish consensus — but also squeeze risk) |
| **Composite Score** | Weighted average of the three signals | ≥6.5 / 10 | ≤4.0 / 10 |

**Key principles:**
- **Cluster buying** (multiple insiders buying within weeks of each other) is the strongest insider signal — individual sales are often for personal liquidity reasons, not bearish views.
- **Institutional ownership** between 40–80% is the sweet spot: enough smart-money validation to signal quality, not so concentrated that a single large redemption destabilises the stock.
- **Short interest > 15%** can be bearish *or* a setup for a squeeze — cross-check with insider buys. High short interest + insider buying = classic contrarian combination.
- This data lags reality by 30–45 days (13F filings are quarterly; Form 4 must be filed within 2 business days of the transaction).

**Limitations:** yfinance sourced from Yahoo Finance. Institutional data is from the most recent 13F filing period and may not reflect current positions.
        """)


# ══════════════════════════════════════════════════════════════
# MULTI-TICKER COMPARISON
# ══════════════════════════════════════════════════════════════

def _render_multi_comparison(tickers: List[str]):
    st.subheader("🔀 Multi-Ticker Comparison")
    st.caption("Compare insider, institutional, and short-interest signals across tickers.")

    rows = []
    progress = st.progress(0, text="Fetching data…")

    for i, ticker in enumerate(tickers):
        progress.progress((i + 1) / len(tickers), text=f"Loading {ticker}…")
        d = fetch_all_holder_data(ticker)
        if d is None:
            continue
        insider_sig = _compute_insider_signal(d['insider_txns'], d['insider_buys'])
        inst_sig    = _compute_institutional_signal(d['info'], d['institutional'])
        short_sig   = _compute_short_signal(d['info'])
        comp        = _composite_signal(insider_sig, short_sig, inst_sig)

        info = d['info']
        rows.append({
            'Ticker':           ticker,
            'Company':          info.get('company_name', ticker)[:28],
            'Composite Signal': comp['label'],
            'Score':            comp['score'],
            'Insider':          insider_sig['label'],
            'Insider Score':    insider_sig['score'],
            '# Inst Holders':   inst_sig['n_holders'],
            'Inst %':           _fmt_pct(inst_sig.get('pct_inst')),
            'Short % Float':    _fmt_pct(short_sig.get('pct')),
            'Days-to-Cover':    f"{short_sig['days']:.1f}" if short_sig.get('days') else 'N/A',
            'Insider Signal':   inst_sig['label'],
        })

    progress.empty()

    if not rows:
        st.warning("No data could be fetched for any ticker.")
        return

    df = pd.DataFrame(rows)

    def _color_signal(val):
        if val == 'Bullish':  return 'background-color: #d4edda; color: #155724'
        if val == 'Bearish':  return 'background-color: #f8d7da; color: #721c24'
        if val == 'Elevated': return 'background-color: #f8d7da; color: #721c24'
        if val == 'Neutral':  return 'background-color: #fff3cd; color: #856404'
        return ''

    styled = (
        df.style
          .applymap(_color_signal, subset=['Composite Signal', 'Insider'])
          .background_gradient(subset=['Score', 'Insider Score'], cmap='RdYlGn',
                               vmin=0, vmax=10)
          .format({'Score': '{:.1f}', 'Insider Score': '{:.1f}'})
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # Composite score bar chart
    df_sorted = df.sort_values('Score', ascending=True)
    colors = [_signal_color(lbl) for lbl in df_sorted['Composite Signal']]

    fig = go.Figure(go.Bar(
        y=df_sorted['Ticker'],
        x=df_sorted['Score'],
        orientation='h',
        marker_color=colors,
        text=df_sorted['Score'].apply(lambda x: f"{x:.1f}"),
        textposition='outside',
        customdata=df_sorted['Composite Signal'],
        hovertemplate='<b>%{y}</b>  Score: %{x:.1f}  (%{customdata})<extra></extra>',
    ))
    fig.add_vline(x=5, line_dash='dash', line_color='#aaa',
                  annotation_text='Neutral', annotation_position='top')
    fig.update_layout(
        title='Composite Signal Score — Cross-Ticker',
        xaxis=dict(range=[0, 11], title='Score (0–10)'),
        template='plotly_white',
        height=max(300, len(df_sorted) * 40 + 100),
        paper_bgcolor='rgba(0,0,0,0)',
    )
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════

def render_insider_institutional():
    st.title("🏦 Insider & Institutional Activity")
    st.markdown(
        "Insider transactions · institutional ownership · short interest · "
        "composite smart-money signal"
    )

    # ── Sidebar controls ─────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Configuration")

        view_mode = st.radio(
            "View Mode",
            ["Single Ticker Deep-Dive", "Multi-Ticker Comparison"],
            index=0,
        )

        ticker_input = st.text_input(
            "Ticker(s)" if view_mode == "Multi-Ticker Comparison"
            else "Ticker",
            value="AAPL",
            placeholder="AAPL  or  AAPL,MSFT,TSLA",
            help="For multi-ticker mode, comma-separate up to 10 tickers."
        ).upper().strip()

        run_btn = st.button("🚀 Run Analysis", type="primary",
                            use_container_width=True)

        st.divider()
        st.subheader("📖 Signal Weights")
        st.caption("Composite = 50% insider + 30% institutional + 20% short interest")
        st.caption("These reflect the relative predictive value of each signal in academic literature.")

    if not run_btn:
        st.info("👈 Enter a ticker and press **Run Analysis**.")
        _render_education()
        return

    # ── Parse tickers ─────────────────────────────────────────────
    raw_tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
    raw_tickers = list(dict.fromkeys(raw_tickers))[:10]  # dedupe, max 10

    if not raw_tickers:
        st.error("Please enter at least one ticker.")
        return

    # ── Multi-ticker comparison ───────────────────────────────────
    if view_mode == "Multi-Ticker Comparison":
        _render_multi_comparison(raw_tickers)
        _render_education()
        return

    # ── Single ticker deep-dive ───────────────────────────────────
    ticker = raw_tickers[0]

    with st.spinner(f"Fetching data for **{ticker}**…"):
        data = fetch_all_holder_data(ticker)

    if data is None:
        st.error(f"❌ Could not fetch data for **{ticker}**.")
        return

    # Compute signals
    insider_sig = _compute_insider_signal(data['insider_txns'], data['insider_buys'])
    inst_sig    = _compute_institutional_signal(data['info'], data['institutional'])
    short_sig   = _compute_short_signal(data['info'])
    comp        = _composite_signal(insider_sig, short_sig, inst_sig)

    # ── Render sections ──────────────────────────────────────────
    _render_header(data)
    _render_composite_signal(comp, insider_sig, short_sig, inst_sig)
    _render_combined_view(data, insider_sig, inst_sig, short_sig)

    st.divider()

    tab1, tab2, tab3 = st.tabs([
        "🔑 Insider Transactions",
        "🏛️ Institutional Ownership",
        "📉 Short Interest",
    ])

    with tab1:
        _render_insider_section(data, insider_sig)

    with tab2:
        _render_institutional_section(data, inst_sig)

    with tab3:
        _render_short_interest_section(data, short_sig)

    st.divider()
    _render_education()

    st.caption(
        "🏦 Insider & Institutional Activity — data sourced from Yahoo Finance via yfinance. "
        "13F filings are quarterly; Form 4 must be filed within 2 business days. "
        "Not financial advice."
    )


# ── App.py registration hook ─────────────────────────────────
# Add to get_available_modules() in app.py:
#
#   try:
#       from insider_institutional import render_insider_institutional
#       modules["🏦 Insider & Institutional"] = {
#           "func": render_insider_institutional,
#           "desc": "Insider transactions, institutional ownership, short interest & composite smart-money signal",
#           "uses_context": False
#       }
#   except ImportError:
#       pass


if __name__ == "__main__":
    render_insider_institutional()
