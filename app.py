import streamlit as st
import ccxt
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import datetime
import ta

st.set_page_config(page_title="Institutional Algo-Backtester", layout="wide")
st.title("🛡️ Custom Algorithmic Backtesting Engine")
st.caption("Custom Date Range | Zero Look-Ahead | Zero Intrabar Bias | Realistic Friction")

# ---------------------------------------------------------
# 1. SIDEBAR: ASSET, EXCHANGE & TIMEFRAME
# ---------------------------------------------------------
st.sidebar.header("१. मार्केट & डेटा सोर्स")
market_category = st.sidebar.radio("मार्केट प्रकार", ["Crypto (CCXT)", "Stocks & Commodities (Yahoo Global)"])

if market_category == "Crypto (CCXT)":
    exchange_name = st.sidebar.selectbox("Exchange", ['binance', 'bybit', 'okx', 'kraken', 'coinbase'])
    symbol = st.sidebar.text_input("Pair Symbol", "BTC/USDT")
    timeframe = st.sidebar.selectbox("Timeframe", ["1m", "5m", "15m", "1h", "4h", "1d"], index=2)
else:
    exchange_name = "Yahoo Finance"
    symbol = st.sidebar.text_input("Ticker Symbol", "RELIANCE.NS")
    timeframe = st.sidebar.selectbox("Timeframe", ["1m", "5m", "15m", "1h", "1d"], index=2)

# ---------------------------------------------------------
# 2. SIDEBAR: START DATE & END DATE SELECTION
# ---------------------------------------------------------
st.sidebar.header("२. बॅकटेस्ट कालावधी (Date Range)")
today = datetime.date.today()
default_start = today - datetime.timedelta(days=90)

start_date = st.sidebar.date_input("Start Date", value=default_start)
end_date = st.sidebar.date_input("End Date", value=today)

if start_date >= end_date:
    st.sidebar.error("Start Date ही End Date पेक्षा आधीची असावी!")

st.sidebar.header("३. कॅपिटल, रिस्क & फ्रिक्शन")
initial_capital = st.sidebar.number_input("प्रारंभिक भांडवल ($/₹)", value=10000.0, step=1000.0)
risk_per_trade_pct = st.sidebar.number_input("Risk Per Trade (%)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
risk_to_reward = st.sidebar.number_input("Risk to Reward Ratio (1 : X)", min_value=1.0, max_value=10.0, value=2.0, step=0.5)

fee_pct = st.sidebar.number_input("Exchange Fee Per Trade (%)", value=0.1, step=0.01)
slippage_pct = st.sidebar.number_input("Slippage Buffer (%)", value=0.05, step=0.01)

# ---------------------------------------------------------
# DATA FETCHING ENGINE (DATE RANGE FILTERED)
# ---------------------------------------------------------
@st.cache_data(ttl=60)
def fetch_data_by_date(cat, ex_name, sym, tf, s_date, e_date):
    try:
        s_dt = datetime.datetime.combine(s_date, datetime.time.min)
        e_dt = datetime.datetime.combine(e_date, datetime.time.max)

        if cat == "Crypto (CCXT)":
            exchange = getattr(ccxt, ex_name)()
            since = int(s_dt.timestamp() * 1000)
            
            all_bars = []
            while True:
                bars = exchange.fetch_ohlcv(sym, timeframe=tf, since=since, limit=1000)
                if not bars:
                    break
                all_bars.extend(bars)
                since = bars[-1][0] + 1
                if bars[-1][0] >= int(e_dt.timestamp() * 1000):
                    break

            df = pd.DataFrame(all_bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df = df.loc[s_dt:e_dt]
        else:
            tf_map = {"1m": "1m", "5m": "5m", "15m": "15m", "1h": "60m", "1d": "1d"}
            ticker = yf.Ticker(sym)
            df = ticker.history(start=s_date, end=e_date + datetime.timedelta(days=1), interval=tf_map[tf])
            df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
            df.index = df.index.tz_localize(None) if df.index.tz is not None else df.index
            df = df.loc[s_dt:e_dt]

        if not df.empty:
            df = df.iloc[:-1]  # चालू असलेली अपूर्ण कँडल वगळणे
        return df

    except Exception as e:
        st.error(f"डेटा फेच करताना एरर आला: {e}")
        return None

# ---------------------------------------------------------
# STRATEGY CODE INPUT
# ---------------------------------------------------------
st.subheader("💻 तुमची पायथन स्ट्रॅटेजी (Strategy Code)")
default_code = """# 'df' मध्ये open, high, low, close, volume उपलब्ध आहेत.
# स्ट्रॅटेजीच्या शेवटी 'Signal' कॉलम सेट करा (1 = BUY, 0 = EXIT)

df['RSI'] = ta.momentum.rsi(df['close'], window=14)
df['EMA_20'] = ta.trend.ema_indicator(df['close'], window=20)

df['Signal'] = 0
buy_cond = (df['close'] > df['EMA_20']) & (df['RSI'] > 55)
exit_cond = (df['close'] < df['EMA_20']) | (df['RSI'] < 40)

df.loc[buy_cond, 'Signal'] = 1
df.loc[exit_cond, 'Signal'] = 0
"""
user_code = st.text_area("Python Script", value=default_code, height=200)

# ---------------------------------------------------------
# BACKTEST EXECUTION
# ---------------------------------------------------------
if st.button("🚀 Run Backtest"):
    with st.spinner(f"{start_date} ते {end_date} दरम्यानचा डेटा लोड होत आहे..."):
        df = fetch_data_by_date(market_category, exchange_name, symbol, timeframe, start_date, end_date)

    if df is None or df.empty:
        st.warning("दिलेल्या तारखांमध्ये कोणताही डेटा मिळाला नाही. कृपया तारीख किंवा सिम्बॉल तपासा.")
        st.stop()

    custom_env = {'df': df.copy(), 'pd': pd, 'np': np, 'ta': ta}
    try:
        exec(user_code, custom_env)
        df = custom_env['df']
        if 'Signal' not in df.columns:
            st.error("एरर: कोडमध्ये `df['Signal']` कॉलम तयार केलेला नाही!")
            st.stop()
    except Exception as e:
        st.error(f"पायथन सिंटॅक्स एरर: {e}")
        st.stop()

    # Simulation Logic
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
        signal_from_prev_candle = df['Signal'].iloc[i-1]

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
            elif signal_from_prev_candle == 0:
                exit_price = bar_open * (1 - total_friction_pct)
                pnl = qty * (exit_price - entry_price)
                capital += pnl
                trades.append({'Time': bar_time, 'Type': 'SIGNAL EXIT', 'Price': exit_price, 'PnL': pnl, 'Capital': capital})
                in_pos = False

        if not in_pos and signal_from_prev_candle == 1:
            entry_price = bar_open * (1 + total_friction_pct)
            risk_amount = capital * (risk_per_trade_pct / 100.0)
            sl_distance = entry_price * 0.015
            stop_loss = entry_price - sl_distance
            take_profit = entry_price + (sl_distance * risk_to_reward)
            qty = risk_amount / sl_distance
            in_pos = True
            trades.append({'Time': bar_time, 'Type': 'BUY ENTRY', 'Price': entry_price, 'PnL': 0.0, 'Capital': capital})

        portfolio_curve.append(capital)

    df['Portfolio_Value'] = portfolio_curve
    net_profit = capital - initial_capital
    net_return_pct = (net_profit / initial_capital) * 100

    closed_trades = [t for t in trades if 'HIT' in t['Type'] or 'EXIT' in t['Type']]
    wins = [t for t in closed_trades if t['PnL'] > 0]
    win_rate = (len(wins) / len(closed_trades) * 100) if closed_trades else 0.0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("अंतिम भांडवल", f"${capital:,.2f}")
    col2.metric("Net Return (%)", f"{net_return_pct:.2f}%", delta=f"{net_profit:,.2f}")
    col3.metric("एकूण ट्रेड्स", f"{len(closed_trades)}")
    col4.metric("Win Rate", f"{win_rate:.1f}%")

    fig_equity = go.Figure()
    fig_equity.add_trace(go.Scatter(x=df.index, y=df['Portfolio_Value'], line=dict(color='#00ffaa', width=2), name="Equity"))
    fig_equity.update_layout(title=f"इक्विटी कर्व ({start_date} ते {end_date})", xaxis_title="Date", yaxis_title="Capital")
    st.plotly_chart(fig_equity, use_container_width=True)

    with st.expander("🔍 ट्रेड हिस्ट्री (Trade-by-Trade Log)"):
        if trades:
            st.dataframe(pd.DataFrame(trades))
        else:
            st.write("या कालावधीत एकही ट्रेड ट्रिगर झाला नाही.")
