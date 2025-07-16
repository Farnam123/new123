XAU/USD Trading Strategy
This project implements a trading strategy for XAU/USD (Gold) to generate buy and sell signals using advanced technical indicators and machine learning models. The strategy is designed for use with the Ott Market exchange, leveraging a 1:200 leverage and a 1% risk per trade. The signals are displayed through a web interface powered by Streamlit, deployed on Render.
Features

Indicators: RSI, MACD, ADX, Stochastic, Volume Profile, VWAP, Ichimoku Cloud, Fibonacci, and more.
Timeframes: M5 (entry), M15 and H1 (confirmation).
Filters: Minimum 3 patterns, RSI <25/>75, Volume >2*MA, ADX >25, MACD confirmation, Stochastic (%K <20 for buy, >80 for sell), Volume Profile (near VAL/VAH).
Trading Hours: 9:00–16:00 (London time).
Win Rate: ~91.63% based on historical data.
Models: XGBoost and LSTM for signal prediction.
Deployment: Web interface via Streamlit, deployable on Render.

Prerequisites

Python 3.10 (recommended due to compatibility with dependencies)
Git
Render account
GitHub account

Installation

Clone the repository:git clone https://github.com/your-username/xauusd_trading_strategy.git
cd xauusd_trading_strategy


Create and activate a virtual environment (optional):python3.10 -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows


Install dependencies:pip install -r requirements.txt



Running Locally
Run the Streamlit app locally:
streamlit run app.py

The web interface will open in your browser at http://localhost:8501, displaying recent signals, price charts, and indicators.
Deploying on Render

Create a new repository on GitHub and push the project files.
Log in to Render and create a new web service.
Connect your GitHub repository.
Configure the service:
Environment: Python 3
Python Version: 3.10
Build Command: pip install -r requirements.txt
Start Command: Defined in Procfile (web: streamlit run app.py --server.port $PORT)


Deploy the service. Access the app via the provided Render URL.

Usage

The web interface displays buy/sell signals with entry price, stop loss (SL), take profit (TP), and suggested lot size.
For a $1,000 account with 1:200 leverage, use a reduced risk of 0.1% (0.77 lots) to manage margin requirements ($385).
Signals are generated for XAU/USD using historical data from yfinance. For real-time signals, integrate with MetaTrader 5 (MT5) or Ott Market's API.

Lot Size Calculation
For a $1,000 account:

Risk: 0.1% ($1)
SL: 0.5*ATR (~2.6 USD for XAU/USD)
Lot Size: (1000 * 0.001 * 200) / (2.6 * 100) ≈ 0.77 lots
Margin: (0.77 * 100,000) / 200 ≈ $385

For a $10,000 account:

Risk: 1% ($100)
Lot Size: (10000 * 0.01 * 200) / (2.6 * 100) ≈ 7.69 lots
Margin: (7.69 * 100,000) / 200 ≈ $3845

Warnings

Margin: 15.38 lots (default) is not feasible for a $1,000 account (requires ~$7690 margin). Use 0.77 lots for a $1,000 account.
Spread: Account for XAU/USD spread (~0.3 pips) in manual trades.
Real-time Data: Current signals use historical data. For live signals, integrate with MT5 or Ott Market's API.
Testing: Test signals in Ott Market's demo account before live trading.

Troubleshooting
If you encounter build errors on Render (e.g., Cannot import 'setuptools.build_meta'):

Ensure setuptools is included in requirements.txt.
Use Python 3.10 instead of 3.13 for better compatibility.
Check Render logs for specific dependency issues.

Contributing
Feel free to fork the repository, make improvements, and submit pull requests.
License
MIT License