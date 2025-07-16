import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import warnings
warnings.filterwarnings('ignore')

# دریافت داده‌ها برای چند تایم‌فریم


def get_market_data(days, intervals=['5m', '15m', '1h']):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    data = {}
    for interval in intervals:
        xauusd = yf.download('GC=F', start=start_date,
                             end=end_date, interval=interval)
        dxy = yf.download('DX-Y.NYB', start=start_date,
                          end=end_date, interval=interval)
        xauusd['Volume'] = xauusd['Volume'].replace(
            0, np.nan).fillna(method='ffill')
        dxy['Volume'] = dxy['Volume'].replace(0, np.nan).fillna(method='ffill')
        data[interval] = {'xauusd': xauusd, 'dxy': dxy}
    return data

# شبیه‌سازی تحلیل احساسات


def get_sentiment_data(data_index):
    return pd.Series(np.random.choice([1, -1, 0], size=len(data_index), p=[0.4, 0.4, 0.2]), index=data_index)

# شبیه‌سازی داده‌های فاندامنتال


def get_fundamental_data(data_index, dxy):
    interest_rate = dxy['Close'].pct_change().rolling(window=50).mean() * 100
    inflation = dxy['Close'].pct_change().rolling(window=20).std() * 100
    volatility = dxy['Close'].rolling(window=20).std() * 100
    return pd.DataFrame({
        'interest_rate': interest_rate,
        'inflation': inflation,
        'volatility': volatility
    }, index=data_index).fillna(method='ffill')

# محاسبه EMA


def calculate_ema(data, period):
    return data['Close'].ewm(span=period, adjust=False).mean()

# محاسبه RSI و واگرایی


def calculate_rsi_divergence(data, periods=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    bullish_div = (rsi.diff() > 0) & (data['Close'].diff() < 0)
    bearish_div = (rsi.diff() < 0) & (data['Close'].diff() > 0)
    return rsi, bullish_div, bearish_div

# محاسبه Stochastic Oscillator


def calculate_stochastic(data, k_period=14, d_period=3):
    low_min = data['Low'].rolling(window=k_period).min()
    high_max = data['High'].rolling(window=k_period).max()
    k = 100 * (data['Close'] - low_min) / (high_max - low_min)
    d = k.rolling(window=d_period).mean()
    return k, d

# محاسبه Volume Profile


def calculate_volume_profile(data, price_bins=50):
    price_range = data['Close'].max() - data['Close'].min()
    bins = np.linspace(data['Close'].min(), data['Close'].max(), price_bins)
    volume_hist, bin_edges = np.histogram(
        data['Close'], bins=bins, weights=data['Volume'])
    value_area_volume = volume_hist.sum() * 0.7
    sorted_indices = np.argsort(volume_hist)[::-1]
    cumulative_volume = 0
    vah, val = None, None
    for idx in sorted_indices:
        cumulative_volume += volume_hist[idx]
        if cumulative_volume >= value_area_volume:
            vah = bin_edges[idx + 1]
            val = bin_edges[idx]
            break
    return vah, val

# محاسبه Bollinger Bands و Squeeze


def calculate_bollinger_bands(data, window=20, num_std=2):
    rolling_mean = data['Close'].rolling(window=window).mean()
    rolling_std = data['Close'].rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    bandwidth = (upper_band - lower_band) / rolling_mean
    squeeze = bandwidth < bandwidth.rolling(window=50).mean() * 0.8
    return rolling_mean, upper_band, lower_band, squeeze

# محاسبه MACD


def calculate_macd(data, fast=12, slow=26, signal=9):
    exp1 = data['Close'].ewm(span=fast, adjust=False).mean()
    exp2 = data['Close'].ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

# محاسبه Ichimoku Cloud


def calculate_ichimoku(data):
    high_9 = data['High'].rolling(window=9).max()
    low_9 = data['Low'].rolling(window=9).min()
    high_26 = data['High'].rolling(window=26).max()
    low_26 = data['Low'].rolling(window=26).min()
    tenkan_sen = (high_9 + low_9) / 2
    kijun_sen = (high_26 + low_26) / 2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
    senkou_span_b = ((data['High'].rolling(window=52).max(
    ) + data['Low'].rolling(window=52).min()) / 2).shift(26)
    return senkou_span_a, senkou_span_b

# محاسبه VWAP


def calculate_vwap(data):
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    vwap = (typical_price * data['Volume']).cumsum() / data['Volume'].cumsum()
    return vwap

# محاسبه ATR


def calculate_atr(data, periods=14):
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    atr = true_range.rolling(window=periods).mean()
    return atr

# محاسبه ADX


def calculate_adx(data, periods=14):
    high = data['High']
    low = data['Low']
    close = data['Close']
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where(plus_dm > minus_dm, 0).where(plus_dm > 0, 0)
    minus_dm = minus_dm.where(minus_dm > plus_dm, 0).where(minus_dm > 0, 0)
    tr = calculate_atr(data, periods)
    plus_di = 100 * plus_dm.ewm(span=periods, adjust=False).mean() / tr
    minus_di = 100 * minus_dm.ewm(span=periods, adjust=False).mean() / tr
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.ewm(span=periods, adjust=False).mean()
    return adx

# محاسبه Cumulative Delta


def calculate_cumulative_delta(data):
    delta = np.where(data['Close'] > data['Open'],
                     data['Volume'], -data['Volume'])
    cumulative_delta = delta.cumsum()
    return pd.Series(cumulative_delta, index=data.index)

# شناسایی Order Blocks، FVG و BOS


def identify_smc(data, window=10):
    demand_zones = []
    supply_zones = []
    fvg = []
    bos = []
    for i in range(window, len(data) - window):
        if (data['Low'].iloc[i] == data['Low'].iloc[i-window:i+window].min()) and \
           (data['Volume'].iloc[i] > data['Volume'].rolling(window=20).mean().iloc[i] * 2.0):
            demand_zones.append(
                {'time': data.index[i], 'price': data['Low'].iloc[i]})
        if (data['High'].iloc[i] == data['High'].iloc[i-window:i+window].max()) and \
           (data['Volume'].iloc[i] > data['Volume'].rolling(window=20).mean().iloc[i] * 2.0):
            supply_zones.append(
                {'time': data.index[i], 'price': data['High'].iloc[i]})
        if (data['High'].iloc[i-1] < data['Low'].iloc[i+1]):
            fvg.append({'time': data.index[i], 'type': 'bullish',
                       'top': data['Low'].iloc[i+1], 'bottom': data['High'].iloc[i-1]})
        elif (data['Low'].iloc[i-1] > data['High'].iloc[i+1]):
            fvg.append({'time': data.index[i], 'type': 'bearish',
                       'top': data['Low'].iloc[i-1], 'bottom': data['High'].iloc[i+1]})
        if (data['High'].iloc[i] > data['High'].iloc[i-window:i].max()) and \
           (data['Close'].iloc[i] > data['High'].iloc[i-window:i].max()) and \
           (data['Volume'].iloc[i] > data['Volume'].rolling(window=20).mean().iloc[i] * 1.2):
            bos.append(
                {'time': data.index[i], 'type': 'bullish', 'price': data['High'].iloc[i]})
        elif (data['Low'].iloc[i] < data['Low'].iloc[i-window:i].min()) and \
             (data['Close'].iloc[i] < data['Low'].iloc[i-window:i].min()) and \
             (data['Volume'].iloc[i] > data['Volume'].rolling(window=20).mean().iloc[i] * 1.2):
            bos.append(
                {'time': data.index[i], 'type': 'bearish', 'price': data['Low'].iloc[i]})
    return demand_zones, supply_zones, fvg, bos

# شناسایی الگوهای کندلی


def identify_price_action(data):
    bullish_engulfing = (data['Close'].shift(1) < data['Open'].shift(1)) & \
                        (data['Close'] > data['Open'].shift(1)) & \
                        (data['Close'] - data['Open'] >
                         data['Open'].shift(1) - data['Close'].shift(1))
    bearish_engulfing = (data['Close'].shift(1) > data['Open'].shift(1)) & \
                        (data['Close'] < data['Open'].shift(1)) & \
                        (data['Open'] - data['Close'] >
                         data['Close'].shift(1) - data['Open'].shift(1))
    pin_bar_bullish = (data['Low'] < data['Low'].shift(1)) & \
                      (data['Close'] > data['Open']) & \
                      ((data['High'] - data['Close']) /
                       (data['Close'] - data['Low']) < 0.3)
    pin_bar_bearish = (data['High'] > data['High'].shift(1)) & \
                      (data['Close'] < data['Open']) & \
                      ((data['Close'] - data['Low']) /
                       (data['High'] - data['Close']) < 0.3)
    return bullish_engulfing, bearish_engulfing, pin_bar_bullish, pin_bar_bearish

# شناسایی الگوهای هارمونیک


def identify_harmonic_patterns(data, window=50):
    bullish_patterns = []
    bearish_patterns = []

    for i in range(window, len(data) - window):
        high = data['High'].iloc[i-window:i+window]
        low = data['Low'].iloc[i-window:i+window]
        close = data['Close'].iloc[i]

        X = low.min()
        A = high.max()
        B = low[high.idxmax():].min()
        C = high[low[high.idxmax():].idxmin():].max()
        D = close

        if X < A and A > B and B < C and C > D:
            AB = A - B
            BC = C - B
            CD = C - D
            XA = A - X

            AB_XA = AB / XA
            BC_AB = BC / AB
            CD_BC = CD / BC

            if 0.618 - 0.05 <= AB_XA <= 0.618 + 0.05 and \
               0.382 - 0.05 <= BC_AB <= 0.886 + 0.05 and \
               1.272 - 0.05 <= CD_BC <= 1.618 + 0.05:
                bullish_patterns.append(
                    {'time': data.index[i], 'price': D, 'type': 'Gartley'})

            if 0.382 - 0.05 <= AB_XA <= 0.50 + 0.05 and \
               0.382 - 0.05 <= BC_AB <= 0.886 + 0.05 and \
               1.618 - 0.05 <= CD_BC <= 2.618 + 0.05:
                bullish_patterns.append(
                    {'time': data.index[i], 'price': D, 'type': 'Bat'})

        elif X > A and A < B and B > C and C < D:
            AB = B - A
            BC = B - C
            CD = D - C
            XA = X - A

            AB_XA = AB / XA
            BC_AB = BC / AB
            CD_BC = CD / BC

            if 0.618 - 0.05 <= AB_XA <= 0.618 + 0.05 and \
               0.382 - 0.05 <= BC_AB <= 0.886 + 0.05 and \
               1.272 - 0.05 <= CD_BC <= 1.618 + 0.05:
                bearish_patterns.append(
                    {'time': data.index[i], 'price': D, 'type': 'Gartley'})

            if 0.382 - 0.05 <= AB_XA <= 0.50 + 0.05 and \
               0.382 - 0.05 <= BC_AB <= 0.886 + 0.05 and \
               1.618 - 0.05 <= CD_BC <= 2.618 + 0.05:
                bearish_patterns.append(
                    {'time': data.index[i], 'price': D, 'type': 'Bat'})

    return bullish_patterns, bearish_patterns

# شناسایی الگوهای RTM پیشرفته


def identify_advanced_rtm_patterns(data, window=20):
    rtm_bullish = []
    rtm_bearish = []

    for i in range(window, len(data) - window):
        liquidity_high = data['High'].iloc[i-window:i].max()
        liquidity_low = data['Low'].iloc[i-window:i].min()
        current_price = data['Close'].iloc[i]
        volume_spike = data['Volume'].iloc[i] > data['Volume'].rolling(
            window=20).mean().iloc[i] * 2.0
        atr = calculate_atr(data, periods=14).iloc[i]

        if (data['Low'].iloc[i] <= liquidity_low) and \
           (data['Close'].iloc[i] > data['Open'].iloc[i]) and \
           (data['Close'].iloc[i] > data['Close'].iloc[i-1]) and \
           volume_spike and \
           (data['Close'].iloc[i] > data['Close'].iloc[i-2:i].min() + 0.2 * atr):
            rtm_bullish.append(
                {'time': data.index[i], 'price': current_price, 'type': 'Advanced RTM Bullish'})

        if (data['High'].iloc[i] >= liquidity_high) and \
           (data['Close'].iloc[i] < data['Open'].iloc[i]) and \
           (data['Close'].iloc[i] < data['Close'].iloc[i-1]) and \
           volume_spike and \
           (data['Close'].iloc[i] < data['Close'].iloc[i-2:i].max() - 0.2 * atr):
            rtm_bearish.append(
                {'time': data.index[i], 'price': current_price, 'type': 'Advanced RTM Bearish'})

    return rtm_bullish, rtm_bearish

# شناسایی خرید و فروش بانک‌ها (Bank Order Flow) با Cumulative Delta


def identify_bank_order_flow(data, window=20):
    bank_bullish = []
    bank_bearish = []
    cumulative_delta = calculate_cumulative_delta(data)
    delta_ma = cumulative_delta.diff().rolling(window=20).mean()

    for i in range(window, len(data) - window):
        volume_ma = data['Volume'].rolling(window=20).mean().iloc[i]
        atr = calculate_atr(data, periods=14).iloc[i]
        delta = cumulative_delta.iloc[i] - cumulative_delta.iloc[i-1]

        if (data['Volume'].iloc[i] > volume_ma * 2) and \
           (data['Low'].iloc[i] <= data['Low'].iloc[i-window:i].min()) and \
           (data['Close'].iloc[i] > data['Open'].iloc[i]) and \
           (data['Close'].iloc[i] > data['Close'].iloc[i-1]) and \
           (delta > 2 * delta_ma.iloc[i]):
            bank_bullish.append(
                {'time': data.index[i], 'price': data['Close'].iloc[i], 'type': 'Bank Bullish'})

        if (data['Volume'].iloc[i] > volume_ma * 2) and \
           (data['High'].iloc[i] >= data['High'].iloc[i-window:i].max()) and \
           (data['Close'].iloc[i] < data['Open'].iloc[i]) and \
           (data['Close'].iloc[i] < data['Close'].iloc[i-1]) and \
           (delta < -2 * delta_ma.iloc[i]):
            bank_bearish.append(
                {'time': data.index[i], 'price': data['Close'].iloc[i], 'type': 'Bank Bearish'})

    return bank_bullish, bank_bearish

# محاسبه فیبوناچی


def calculate_fibonacci_levels(data, window=20):
    high = data['High'].rolling(window=window).max()
    low = data['Low'].rolling(window=window).min()
    fib_618 = low + (high - low) * 0.618
    fib_1618 = low + (high - low) * 1.618
    return fib_618, fib_1618

# مدل XGBoost


def train_xgboost(data, signals):
    features = pd.DataFrame(index=data.index)
    features['rsi'] = signals['rsi']
    features['macd'] = signals['macd']
    features['atr'] = signals['atr']
    features['adx'] = signals['adx']
    features['stochastic_k'] = signals['stochastic_k']
    features['stochastic_d'] = signals['stochastic_d']
    features['volume'] = signals['volume'] / signals['volume_ma']
    features['dxy_trend'] = signals['dxy_trend']
    features['price_to_fib'] = (
        data['Close'] - signals['fib_618']) / signals['atr']
    features['price_to_val'] = (
        data['Close'] - signals['val']) / signals['atr']
    features['ichimoku_trend'] = signals['price'] > signals['senkou_span_a']
    features['vwap_diff'] = (data['Close'] - signals['vwap']) / signals['atr']
    features['bullish_engulfing'] = signals['bullish_engulfing']
    features['bearish_engulfing'] = signals['bearish_engulfing']
    features['pin_bar_bullish'] = signals['pin_bar_bullish']
    features['pin_bar_bearish'] = signals['pin_bar_bearish']
    features['trend'] = signals['ema50'] > signals['ema200']
    features['harmonic_bullish'] = signals['harmonic_bullish']
    features['harmonic_bearish'] = signals['harmonic_bearish']
    features['rtm_bullish'] = signals['rtm_bullish']
    features['rtm_bearish'] = signals['rtm_bearish']
    features['bank_bullish'] = signals['bank_bullish']
    features['bank_bearish'] = signals['bank_bearish']
    features['interest_rate'] = signals['interest_rate']
    features['inflation'] = signals['inflation']
    features['volatility'] = signals['volatility']

    target = signals['positions'].shift(-1).fillna(0)
    features = features.dropna()
    target = target.loc[features.index]

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train,
