import streamlit as st
import pandas as pd
from trading_strategy import get_market_data, generate_signals, simulate_signals
from streamlit_autorefresh import st_autorefresh

# تنظیمات اولیه
st.set_page_config(page_title="سیگنال‌های معاملاتی XAU/USD", layout="wide")
st.title("سیگنال‌های معاملاتی XAU/USD (اوتت مارکت)")
st.write("این برنامه سیگنال‌های خرید و فروش برای XAU/USD با لوریج 1:200 تولید می‌کند.")

# تنظیم تازه‌سازی خودکار (هر 5 دقیقه)
st_autorefresh(interval=300000, key="datarefresh")

# دریافت داده‌ها و تولید سیگنال‌ها
data = get_market_data(days=500)
signals = generate_signals(data, initial_capital=1000,
                           risk_per_trade=0.01, leverage=200)
simulated_signals = simulate_signals(signals, initial_capital=1000)

# نمایش سیگنال‌های اخیر
st.subheader("سیگنال‌های اخیر")
st.dataframe(simulated_signals.tail(7))

# نمایش نمودارها
st.subheader("نمودار قیمت و اندیکاتورها")
st.line_chart(signals[['price', 'ema50', 'ema200', 'vwap']].tail(100))
st.line_chart(signals[['rsi', 'stochastic_k', 'stochastic_d']].tail(100))
st.line_chart(signals[['adx']].tail(100))

# اطلاعات اضافی
st.subheader("اطلاعات استراتژی")
st.write(f"دقت XGBoost: {signals['xgb_signal'].notnull().mean() * 100:.2f}%")
st.write(f"تعداد سیگنال‌ها: {len(simulated_signals)}")
st.write(
    f"وین‌ریت: {(simulated_signals['type'].notnull().sum() / len(simulated_signals) * 100) if len(simulated_signals) > 0 else 0:.2f}%")
st.write(f"میانگین سیگنال روزانه: {len(simulated_signals) / 500:.2f}")
