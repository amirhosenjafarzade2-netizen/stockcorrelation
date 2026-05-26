import streamlit as stimport yfinance as yfimport numpy as npimport pandas as pdimport plotly.express as pximport plotly.graph_objects as gofrom datetime import datetimeimport requestsfrom bs4 import BeautifulSoupimport time

DEFAULT_MOS = 25.0

─────────────────────────────────────────────────────────────

Finviz

─────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)def get_finviz_data(ticker: str) -> dict:data = {}

headers = {
    'User-Agent': 'Mozilla/5.0'
}

url = f"https://finviz.com/quote.ashx?t={ticker}"

try:
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table', {'class': 'snapshot-table2'})

    if table:
        rows = table.find_all('tr')

        for row in rows:
            cells = row.find_all('td')

            for i in range(0, len(cells), 2):

                if i + 1 < len(cells):

                    key = cells[i].text.strip()
                    value = cells[i + 1].text.strip()

                    try:
                        clean = (
                            value
                            .replace('%', '')
                            .replace(',', '')
                            .replace('$', '')
                        )

                        data[key] = float(clean)

                    except:
                        data[key] = value

    return data

except:
    return {}

─────────────────────────────────────────────────────────────

Data Fetch

─────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)def fetch_stock_data(ticker: str):

try:
    ticker_clean = ticker.replace('.', '-')

    stock = yf.Ticker(ticker_clean)

    info = stock.info or {}

    data = {
        'ticker': ticker_clean,

        'name': info.get('longName', ticker_clean),

        'sector': info.get('sector', 'N/A'),

        'current_price': info.get(
            'currentPrice',
            info.get('regularMarketPrice', 0)
        ),

        'current_eps': info.get('trailingEps', 0),

        'forward_eps': info.get(
            'forwardEps',
            info.get('trailingEps', 0)
        ),

        'dividend_per_share': info.get(
            'dividendRate',
            0
        ),

        'book_value': info.get(
            'bookValue',
            0
        ),

        'historical_pe': info.get(
            'trailingPE',
            15
        ),

        'beta': info.get(
            'beta',
            1
        ),

        'roe': (
            info.get('returnOnEquity', 0.15) * 100
            if info.get('returnOnEquity')
            else 15
        ),

        'market_cap': info.get(
            'marketCap',
            0
        ),

        'shares_outstanding': info.get(
            'sharesOutstanding',
            1
        ),

        'fcf': info.get(
            'freeCashflow',
            0
        ),

        'net_debt': info.get(
            'totalDebt',
            0
        ) - info.get(
            'totalCash',
            0
        ),

        'debt': info.get(
            'totalDebt',
            0
        ),

        'analyst_growth': (
            info.get('earningsGrowth', 0.10) * 100
            if info.get('earningsGrowth')
            else 10
        ),

        'wacc': 9.0,

        'stable_growth': 3.0,

        'desired_return': 10.0,

        'years_high_growth': 5,
    }

    # safer clamps
    data['analyst_growth'] = max(
        -5,
        min(data['analyst_growth'], 20)
    )

    data['roe'] = max(
        -50,
        min(data['roe'], 50)
    )

    data['historical_pe'] = max(
        1,
        min(data['historical_pe'], 40)
    )

    finviz = get_finviz_data(ticker_clean)

    if finviz:

        if isinstance(finviz.get('P/E'), (float, int)):
            data['historical_pe'] = min(
                max(finviz.get('P/E'), 1),
                40
            )

    return data

except:
    return None

─────────────────────────────────────────────────────────────

Models

─────────────────────────────────────────────────────────────

def core_valuation(inputs):

eps = inputs['forward_eps']
pe = inputs['historical_pe']

growth = inputs['analyst_growth'] / 100

years = inputs['years_high_growth']

desired = inputs['desired_return'] / 100

future_eps = eps * ((1 + growth) ** years)

future_price = future_eps * pe

intrinsic = future_price / ((1 + desired) ** years)

return {
    'intrinsic_value': max(intrinsic, 0)
}

def graham_intrinsic_value(inputs):

eps = inputs['current_eps']

growth = min(inputs['analyst_growth'], 15)

if eps <= 0:
    return {'intrinsic_value': 0}

intrinsic = eps * (8.5 + 2 * growth)

return {
    'intrinsic_value': max(intrinsic, 0)
}

def comparable_company_analysis(inputs):

eps = inputs['current_eps']

pe = inputs['historical_pe']

intrinsic = eps * pe

return {
    'intrinsic_value': max(intrinsic, 0)
}

def residual_income(inputs):

bv = inputs['book_value']

roe = inputs['roe'] / 100

r = inputs['desired_return'] / 100

growth = min(inputs['analyst_growth'] / 100, 0.15)

years = 10

if r <= growth:
    growth = r - 0.02

residual = bv * (roe - r)

pv = 0

for t in range(1, years + 1):

    ri_t = residual * ((1 + growth) ** t)

    pv += ri_t / ((1 + r) ** t)

intrinsic = bv + pv

return {
    'intrinsic_value': max(intrinsic, 0)
}

def two_stage_dcf(inputs):

fcf = inputs['fcf']

if fcf <= 0:
    return {'intrinsic_value': 0}

growth = min(inputs['analyst_growth'] / 100, 0.15)

stable_growth = min(
    inputs['stable_growth'] / 100,
    0.04
)

wacc = inputs['wacc'] / 100

years = inputs['years_high_growth']

shares = inputs['shares_outstanding']

if stable_growth >= wacc:
    stable_growth = wacc - 0.02

pv_fcf = 0

for t in range(1, years + 1):

    fcf_t = fcf * ((1 + growth) ** t)

    pv_fcf += fcf_t / ((1 + wacc) ** t)

terminal_fcf = (
    fcf
    * ((1 + growth) ** years)
    * (1 + stable_growth)
)

terminal_value = (
    terminal_fcf
    / (wacc - stable_growth)
)

pv_terminal = (
    terminal_value
    / ((1 + wacc) ** years)
)

enterprise_value = pv_fcf + pv_terminal

equity_value = (
    enterprise_value
    - inputs.get('net_debt', 0)
)

intrinsic = equity_value / shares

return {
    'intrinsic_value': max(intrinsic, 0)
}

def owner_earnings(inputs):

oe = inputs['fcf']

if oe <= 0:
    return {'intrinsic_value': 0}

g = min(inputs['analyst_growth'] / 100, 0.12)

r = inputs['desired_return'] / 100

shares = inputs['shares_outstanding']

if g >= r:
    g = r - 0.02

pv = 0

for t in range(1, 11):

    pv += (
        oe
        * ((1 + g) ** t)
        / ((1 + r) ** t)
    )

terminal = (
    oe
    * ((1 + g) ** 11)
    / (r - g)
) / ((1 + r) ** 10)

equity_value = pv + terminal

intrinsic = equity_value / shares

return {
    'intrinsic_value': max(intrinsic, 0)
}

─────────────────────────────────────────────────────────────

Dispatcher

─────────────────────────────────────────────────────────────

def calculate_valuation(inputs):

model = inputs['model']

model_map = {
    "Core Valuation (Excel)": core_valuation,
    "Graham Intrinsic Value": graham_intrinsic_value,
    "Comparable Company Analysis": comparable_company_analysis,
    "Residual Income (RI)": residual_income,
    "Two-Stage DCF": two_stage_dcf,
    "Owner Earnings": owner_earnings,
}

if model not in model_map:
    return {
        'intrinsic_value': 0
    }

base = model_map[model](inputs)

intrinsic = base['intrinsic_value']

mos = inputs.get('core_mos', DEFAULT_MOS)

safe_buy_price = intrinsic * (1 - mos / 100)

current = inputs['current_price']

undervaluation = (
    (
        intrinsic - current
    ) / current
) * 100 if current > 0 else 0

verdict = (
    "Strong Buy"
    if undervaluation > 30
    else "Buy"
    if undervaluation > 10
    else "Hold"
    if undervaluation > -10
    else "Sell"
)

return {
    'intrinsic_value': intrinsic,
    'safe_buy_price': safe_buy_price,
    'undervaluation': undervaluation,
    'verdict': verdict
}

─────────────────────────────────────────────────────────────

Validation

─────────────────────────────────────────────────────────────

def validate_inputs(inputs):

if inputs['current_price'] <= 0:
    return False

if inputs['shares_outstanding'] <= 0:
    return False

return True

─────────────────────────────────────────────────────────────

S&P 500

─────────────────────────────────────────────────────────────

@st.cache_data(ttl=86400)def get_sp500_tickers():

url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

try:
    df = pd.read_html(url)[0]

    df = df[['Symbol', 'Security', 'GICS Sector']]

    df['Symbol'] = df['Symbol'].str.replace(
        '.',
        '-',
        regex=False
    )

    return df

except:

    return pd.DataFrame({
        'Symbol': ['AAPL', 'MSFT', 'GOOGL']
    })

─────────────────────────────────────────────────────────────

Screener

─────────────────────────────────────────────────────────────

def run_sp500_screener(model,min_undervaluation=10,max_stocks=100):

sp500 = get_sp500_tickers().head(max_stocks)

results = []

progress = st.progress(0)

total = len(sp500)

for i, row in sp500.iterrows():

    ticker = row['Symbol']

    try:

        inputs = fetch_stock_data(ticker)

        if inputs and validate_inputs(inputs):

            inputs['model'] = model

            val = calculate_valuation(inputs)

            if (
                val['undervaluation']
                >= min_undervaluation
            ):

                results.append({

                    'Ticker': ticker,

                    'Name': inputs['name'],

                    'Sector': inputs['sector'],

                    'Price': inputs['current_price'],

                    'Intrinsic': val['intrinsic_value'],

                    'Safe Buy': val['safe_buy_price'],

                    'Undervaluation %': val['undervaluation'],

                    'Verdict': val['verdict'],

                    'P/E': inputs['historical_pe'],

                    'Growth %': inputs['analyst_growth']
                })

    except:
        pass

    progress.progress((i + 1) / total)

progress.empty()

if not results:
    return pd.DataFrame()

df = pd.DataFrame(results)

return df.sort_values(
    'Undervaluation %',
    ascending=False
)

─────────────────────────────────────────────────────────────

UI

─────────────────────────────────────────────────────────────

def render_advanced_valuation():

st.title("💹 Fixed Advanced Valuation")

ticker = st.text_input(
    "Ticker",
    "AAPL"
).upper()

model = st.selectbox(
    "Model",
    [
        "Core Valuation (Excel)",
        "Graham Intrinsic Value",
        "Comparable Company Analysis",
        "Residual Income (RI)",
        "Two-Stage DCF",
        "Owner Earnings"
    ]
)

mos = st.slider(
    "Margin of Safety %",
    0,
    50,
    25
)

if st.button("Analyze"):

    with st.spinner("Analyzing..."):

        inputs = fetch_stock_data(ticker)

        if not inputs:

            st.error("Could not fetch data.")

            return

        inputs['model'] = model
        inputs['core_mos'] = mos

        results = calculate_valuation(inputs)

        st.subheader(f"{inputs['name']} ({ticker})")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Current Price",
            f"${inputs['current_price']:.2f}"
        )

        c2.metric(
            "Intrinsic Value",
            f"${results['intrinsic_value']:.2f}"
        )

        c3.metric(
            "Safe Buy Price",
            f"${results['safe_buy_price']:.2f}"
        )

        c4.metric(
            "Undervaluation",
            f"{results['undervaluation']:.1f}%"
        )

        st.markdown(
            f"## {results['verdict']}"
        )

        st.dataframe(pd.DataFrame({

            'Metric': [
                'EPS',
                'Forward EPS',
                'P/E',
                'ROE',
                'Growth',
                'Market Cap'
            ],

            'Value': [
                inputs['current_eps'],
                inputs['forward_eps'],
                inputs['historical_pe'],
                inputs['roe'],
                inputs['analyst_growth'],
                inputs['market_cap']
            ]
        }))

st.divider()

st.subheader("🏛️ S&P 500 Screener")

screener_model = st.selectbox(
    "Screener Model",
    [
        "Core Valuation (Excel)",
        "Comparable Company Analysis",
        "Residual Income (RI)",
        "Two-Stage DCF",
        "Owner Earnings"
    ],
    key="screen_model"
)

min_underv = st.slider(
    "Minimum Undervaluation %",
    0,
    100,
    15
)

max_stocks = st.slider(
    "Max Stocks",
    10,
    500,
    100
)

if st.button("Run Screener"):

    with st.spinner("Screening..."):

        df = run_sp500_screener(
            screener_model,
            min_underv,
            max_stocks
        )

        if df.empty:

            st.warning("No stocks found.")

        else:

            st.success(
                f"Found {len(df)} stocks."
            )

            st.dataframe(
                df.style.format({

                    'Price': '${:.2f}',

                    'Intrinsic': '${:.2f}',

                    'Safe Buy': '${:.2f}',

                    'Undervaluation %': '{:.1f}%',

                    'P/E': '{:.2f}',

                    'Growth %': '{:.1f}%'
                }),
                use_container_width=True
            )

            fig = px.bar(
                df.head(10),
                x='Ticker',
                y='Undervaluation %',
                hover_data=['Price', 'Intrinsic']
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

─────────────────────────────────────────────────────────────

Main

─────────────────────────────────────────────────────────────

if name == "main":

st.set_page_config(
    page_title="Fixed Valuation",
    layout="wide"
)

render_advanced_valuation()
