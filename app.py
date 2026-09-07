import streamlit as st
import ccxt
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import datetime
import ta

st.set_page_config(page_title="Ultimate Algo-Backtesting Studio", layout="wide")
st.title("🛡️ Institutional Algo-Backtester (Zero Bias Engine)")

# ---------------------------------------------------------
# 1. ASSET SUGGESTIONS DATABASE
# ---------------------------------------------------------
CRYPTO_PAIRS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT", 
    "ADA/USDT", "DOGE/USDT", "AVAX/USDT", "LINK/USDT", "SUI/USDT"
]

STOCK_INDICES_COMMODITIES = {
    "--- भारतीय शेअर्स (NSE) ---": "RELIANCE.NS",
    "Reliance Industries": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "Infosys": "INFY.NS",
    "Tata Motors": "TATAMOTORS.NS",
    "State Bank of India": "SBIN.NS",
    "--- जागतिक शेअर्स (US) ---": "AAPL",
    "Apple": "AAPL",
    "NVIDIA": "NVDA",
    "Tesla": "TSLA",
    "Microsoft": "MSFT",
    "--- मुख्य इंडेक्स (Indices) ---": "^NSEI",
    "Nifty 50": "^NSEI",
    "Bank Nifty": "^NSEBANK",
    "S&P 500 (US)": "^GSPC",
    "Nasdaq 100": "^IXIC",
    "--- कमोडिटीज (Commodities) ---": "GC=F",
    "Gold (सोने)": "GC=F",
    "Silver (चांदी)": "SI=F",
    "Crude Oil (कच्चे तेल)": "CL=F",
    "Natural Gas": "NG=F"
}

# ---------------------------------------------------------
# 2. SIDEBAR: SELECTION
# ---------------------------------------------------------
st.sidebar.header("१. मार्केट & ॲसेट निवडा")
market_category = st.sidebar.radio("मार्केट कॅटेगरी", ["Crypto (Bybit / Kraken / OKX)", "Stocks, Indices & Commodities"])

if market_category == "Crypto (Bybit / Kraken / OKX)":
    exchange_name = st.sidebar.selectbox("Exchange", ['bybit', 'kraken', 'okx', 'gateio', 'coinbase'])
    selected_preset = st.sidebar.selectbox("लोकप्रिय कॉईन्स (Suggestions)", ["इतर (Custom Symbol)"] + CRYPTO_PAIRS)
    if selected_preset == "इतर (Custom Symbol)":
        symbol = st.sidebar.text_input("Pair Symbol लिहा", "BTC/USDT")
    else:
        symbol = selected_preset
    timeframe = st.sidebar.selectbox("Timeframe", ["1m", "5m", "15m", "1h", "4h", "1d"], index=2)
else:
    exchange_name = "Global Market"
    preset_label = st.sidebar.selectbox("लोकप्रिय टिकर्स (Suggestions)", list(STOCK_INDICES_COMMODITIES.keys()))
    default_val = STOCK_INDICES_COMMODITIES[preset_label]
    symbol = st.sidebar.text_input("किंवा कोणताही टिकर टाका (Ticker Symbol)", value=default_val)
    timeframe = st.sidebar.selectbox("Timeframe", ["1m", "5m", "15m", "1h", "1d"], index=2)

st.sidebar.header("२. कालावधी (Date Range)")
today = datetime.date.today()
default_start = today - datetime.timedelta(days=60)
start_date = st.sidebar.date_input("Start Date", value=default_start)
end_date = st.sidebar.date_input("End Date", value=today)

st.sidebar.header("३. कॅपिटल, रिस्क & फ्रिक्शन")
initial_capital = st.sidebar.number_input("भांडवल ($/₹)", value=10000.0, step=1000.0)
risk_per_trade_pct = st.sidebar.number_input("Risk Per Trade (%)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
risk_to_reward = st.sidebar.number_input("Risk to Reward (1 : X)", min_value=0.5, max_value=10.0, value=2.0, step=0.5)
fee_pct = st.sidebar.number_input("Exchange Fee (%)", value=0.075, step=0.01)
slippage_pct = st.sidebar.number_input("Slippage Buffer (%)", value=0.05, step=0.01)

# ---------------------------------------------------------
# 3. DATA FETCHING (TZ-CLEAN & ROBUST)
# ---------------------------------------------------------
@st.cache_data(ttl=60)
def fetch_clean_data(cat, ex_name, sym, tf, s_date, e_date):
    try:
        s_dt = datetime.datetime.combine(s_date, datetime.time.min)
        e_dt = datetime.datetime.combine(e_date, datetime.time.max)

        if "Crypto" in cat:
            exchange = getattr(ccxt, ex_name)({'enableRateLimit': True})
            since = int(s_dt.timestamp() * 1000)
            end_ts = int(e_dt.timestamp() * 1000)
            
            all_bars = []
            for _ in range(5):  # जास्तीत जास्त 5000 कँडल्स
                bars = exchange.fetch_ohlcv(sym, timeframe=tf, since=since, limit=1000)
                if not bars:
                    break
                all_bars.extend(bars)
                since = bars[-1][0] + 1
                if bars[-1][0] >= end_ts or len(bars) < 1000:
                    break

            df = pd.DataFrame(all_bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df = df.loc[s_dt:e_dt]
        else:
            tf_map = {"1m": "1m", "5m": "5m", "15m": "15m", "1h": "60m", "1d": "1d"}
            ticker = yf.Ticker(sym)
            df = ticker.history(start=s_date, end=e_date + datetime.timedelta(days=1), interval=tf_map[tf])
            if df.empty:
                return None
            df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
            # Safe timezone cleanup
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df = df.loc[s_dt:e_dt]

        if not df.empty and len(df) > 1:
            df = df.iloc[:-1]  # चालू अर्धवट कँडल काढून टाकणे
        return df
    except Exception as e:
        st.error(f"डेटा फेचिंग एरर: {e}")
        return None

# ---------------------------------------------------------
# 4. STRATEGY SELECTION MODE
# ---------------------------------------------------------
st.subheader("⚙️ स्ट्रॅटेजी पद्धत निवडा (Strategy Engine Mode)")
strategy_mode = st.radio("मोड निवडा:", ["१. No-Code Indicators (रेडिमेड टेक्निकल इंडिकेटर्स)", "२. Pure Python Code (स्वतःचा कोड)"], horizontal=True)

df_signal_logic = None

if strategy_mode == "१. No-Code Indicators (रेडिमेड टेक्निकल इंडिकेटर्स)":
    c1, c2 = st.columns(2)
    with c1:
        indicator_choice = st.selectbox(
            "इंडिकेटर स्ट्रॅटेजी निवडा", 
            ["EMA Crossover (Dual EMAs)", "RSI Overbought/Oversold", "MACD Crossover", "Bollinger Bands Breakout"]
        )
    
    with c2:
        if indicator_choice == "EMA Crossover (Dual EMAs)":
            fast_ema_val = st.number_input("Fast EMA Length (1 ते 1000)", min_value=1, max_value=1000, value=9)
            slow_ema_val = st.number_input("Slow EMA Length (1 ते 1000)", min_value=1, max_value=1000, value=21)
        elif indicator_choice == "RSI Overbought/Oversold":
            rsi_period = st.number_input("RSI Period", min_value=2, max_value=100, value=14)
            rsi_buy = st.number_input("RSI Oversold (Buy Level)", value=30)
            rsi_sell = st.number_input("RSI Overbought (Exit Level)", value=70)
        elif indicator_choice == "MACD Crossover":
            macd_fast = st.number_input("MACD Fast Period", value=12)
            macd_slow = st.number_input("MACD Slow Period", value=26)
            macd_signal = st.number_input("MACD Signal Period", value=9)
        elif indicator_choice == "Bollinger Bands Breakout":
            bb_window = st.number_input("BB Period", value=20)
            bb_dev = st.number_input("BB Std Dev", value=2.0)

else:
    default_code = """# 'df' मध्ये open, high, low, close, volume उपलब्ध आहेत.
# स्ट्रॅटेजीच्या शेवटी 'Signal' कॉलम सेट करा (1 = BUY, 0 = EXIT)

df['EMA_20'] = ta.trend.ema_indicator(df['close'], window=20)
df['RSI'] = ta.momentum.rsi(df['close'], window=14)

df['Signal'] = 0
buy_cond = (df['close'] > df['EMA_20']) & (df['RSI'] > 50)
exit_cond = (df['close'] < df['EMA_20'])

df.loc[buy_cond, 'Signal'] = 1
df.loc[exit_cond, 'Signal'] = 0
"""
    user_code = st.text_area("Python Script Editor", value=default_code, height=200)

# ---------------------------------------------------------
# 5. EXECUTION & BACKTEST ENGINE
# ---------------------------------------------------------
if st.button("🚀 Run Backtest Now"):
    with st.spinner("ऑथेंटिक डेटा आणत आहे व बॅकटेस्ट चालवत आहे..."):
        df = fetch_clean_data(market_category, exchange_name, symbol, timeframe, start_date, end_date)

    if df is None or df.empty:
        st.warning("दिलेल्या सिम्बॉल किंवा तारखेसाठी डेटा मिळाला नाही. तारीख किंवा सिम्बॉल तपासा.")
        st.stop()

    # Apply Indicators / Strategy
    if strategy_mode == "१. No-Code Indicators (रेडिमेड टेक्निकल इंडिकेटर्स)":
        df['Signal'] = 0
        if indicator_choice == "EMA Crossover (Dual EMAs)":
            df['Fast_EMA'] = ta.trend.ema_indicator(df['close'], window=fast_ema_val)
            df['Slow_EMA'] = ta.trend.ema_indicator(df['close'], window=slow_ema_val)
            df.loc[df['Fast_EMA'] > df['Slow_EMA'], 'Signal'] = 1
            df.loc[df['Fast_EMA'] <= df['Slow_EMA'], 'Signal'] = 0
        elif indicator_choice == "RSI Overbought/Oversold":
            df['RSI'] = ta.momentum.rsi(df['close'], window=rsi_period)
            df.loc[df['RSI'] < rsi_buy, 'Signal'] = 1
            df.loc[df['RSI'] > rsi_sell, 'Signal'] = 0
        elif indicator_choice == "MACD Crossover":
            macd = ta.trend.MACD(df['close'], window_fast=macd_fast, window_slow=macd_slow, window_sign=macd_signal)
            df['MACD'] = macd.macd()
            df['MACD_Sig'] = macd.macd_signal()
            df.loc[df['MACD'] > df['MACD_Sig'], 'Signal'] = 1
            df.loc[df['MACD'] <= df['MACD_Sig'], 'Signal'] = 0
        elif indicator_choice == "Bollinger Bands Breakout":
            bb = ta.volatility.BollingerBands(df['close'], window=int(bb_window), window_dev=bb_dev)
            df['BB_High'] = bb.bollinger_hband()
            df['BB_Low'] = bb.bollinger_lband()
            df.loc[df['close'] > df['BB_High'], 'Signal'] = 1
            df.loc[df['close'] < df['BB_Low'], 'Signal'] = 0
    else:
        custom_env = {'df': df.copy(), 'pd': pd, 'np': np, 'ta': ta}
        try:
            exec(user_code, custom_env)
            df = custom_env['df']
            if 'Signal' not in df.columns:
                st.error("एरर: कोडमध्ये `df['Signal']` कॉलम तयार केलेला नाही!")
                st.stop()
        except Exception as e:
            st.error(f"पायथन कोड एरर: {e}")
            st.stop()

    # Zero-Flaw Simulation Loop
    capital = initial_capital
    portfolio_curve = [capital]
    trades = []
    in_pos = False
    entry_price = 0.0
    stop_loss = 0.0
    take_profit = 0.0
    qty = 0.0
    total_friction_pct = (fee_pct + slippage_pct) / 100.0

    for i in range(1, len(df)):
        bar_open = df['open'].iloc[i]
        bar_high = df['high'].iloc[i]
        bar_low = df['low'].iloc[i]
        bar_time = df.index[i]
        # Next-bar Open execution to eliminate look-ahead bias
        sig_prev = df['Signal'].iloc[i-1]

        if in_pos:
            hit_sl = bar_low <= stop_loss
            hit_tp = bar_high >= take_profit

            if hit_sl:
                exit_price = stop_loss * (1 - total_friction_pct)
                pnl = qty * (exit_price - entry_price)
                capital += pnl
                trades.append({'Time': bar_time, 'Type': 'SL HIT', 'Price': exit_price, 'PnL': pnl, 'Capital': capital})
                in_pos = False
            elif hit_tp:
                exit_price = take_profit * (1 - total_friction_pct)
                pnl = qty * (exit_price - entry_price)
                capital += pnl
                trades.append({'Time': bar_time, 'Type': 'TP HIT', 'Price': exit_price, 'PnL': pnl, 'Capital': capital})
                in_pos = False
            elif sig_prev == 0:
                exit_price = bar_open * (1 - total_friction_pct)
                pnl = qty * (exit_price - entry_price)
                capital += pnl
                trades.append({'Time': bar_time, 'Type': 'SIGNAL EXIT', 'Price': exit_price, 'PnL': pnl, 'Capital': capital})
                in_pos = False

        if not in_pos and sig_prev == 1:
            entry_price = bar_open * (1 + total_friction_pct)
            risk_amount = capital * (risk_per_trade_pct / 100.0)
            sl_distance = entry_price * 0.015
            stop_loss = entry_price - sl_distance
            take_profit = entry_price + (sl_distance * risk_to_reward)
            qty = risk_amount / sl_distance
            in_pos = True
            trades.append({'Time': bar_time, 'Type': 'BUY ENTRY', 'Price': entry_price, 'PnL': 0.0, 'Capital': capital})

        portfolio_curve.append(capital)

    # Metrics
    df['Portfolio_Value'] = portfolio_curve
    net_profit = capital - initial_capital
    net_return_pct = (net_profit / initial_capital) * 100
    closed_trades = [t for t in trades if 'HIT' in t['Type'] or 'EXIT' in t['Type']]
    wins = [t for t in closed_trades if t['PnL'] > 0]
    win_rate = (len(wins) / len(closed_trades) * 100) if closed_trades else 0.0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("अंतिम भांडवल", f"${capital:,.2f}")
    col2.metric("Net Return (%)", f"{net_return_pct:.2f}%", delta=f"{net_profit:,.2f}")
    col3.metric("पूर्ण झालेले ट्रेड्स", f"{len(closed_trades)}")
    col4.metric("Win Rate", f"{win_rate:.1f}%")

    # Equity Curve Chart
    fig_equity = go.Figure()
    fig_equity.add_trace(go.Scatter(x=df.index, y=df['Portfolio_Value'], line=dict(color='#00ffaa', width=2), name="Equity Curve"))
    fig_equity.update_layout(title="कॅपिटल ग्रोथ चार्ट (Equity Curve)", xaxis_title="Date", yaxis_title="Capital")
    st.plotly_chart(fig_equity, use_container_width=True)

    with st.expander("📝 ट्रेड-बाय-ट्रेड हिस्ट्री (Trade Log)"):
        if trades:
            st.dataframe(pd.DataFrame(trades))
        else:
            st.write("या कालावधीत एकही ट्रेड ट्रिगर झाला नाही.")
