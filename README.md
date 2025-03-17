# AI Trading Agent with Bayesian Networks

## Members
- Abhyuday Singh
- Jayendra Mangal

## Project Overview
An utility-based AI trading agent using Bayesian networks to make buy/sell/hold decisions based on market indicators.

## Abstract
This project aims to develop an AI-driven trading agent that learns to make buy, sell, or hold decisions in the stock market to maximize both short-term and long-term returns, given a fixed principal amount. The agent leverages historical stock price and volume data to model market trends and respond to real-time fluctuations while balancing risk and reward. A key component of this system is a Bayesian network-based decision model, which captures dependencies between market variables such as stock price movements, trading volume trends, volatility, and macroeconomic indicators. By integrating probabilistic reasoning with reinforcement learning, the agent enhances its ability to adapt to evolving market conditions in an explainable and data-driven manner.
The project will demonstrate core AI concepts, including data preprocessing, feature engineering, and policy learning within a utility-based framework. We will go about this by breaking down the procedure into milestones. In a nutshell, first, we will preprocess the data to build a historical dataset to estimate conditional probabilities and define key features such as moving averages, RSI, and volatility. Second, we will implement a simple Bayesian network that models all stock price dependencies and then train the model on historical data from our datasets to estimate the probability distributions. Then we will use the inference to update beliefs about future stock movements and ensure the model is ready for dynamic decisions based on live data. 
The performance of the agent will be evaluated based on its ability to maximize long-term profit while minimizing drawdowns and maintaining portfolio stability. Through this approach, the project aims to build an AI trading agent that is not only profitable but also interpretable in its decision-making process.


## Milestone 2 Updates

### Data Exploration
- Collected 3,407 daily observations of AAPL stock (2010-2023)
- Engineered features: 20-day MA, 14-day RSI, Annualized Volatility
- Discretized continuous variables into categorical states

### Bayesian Network
- Nodes: ['RSI_Cat', 'Vol_Cat', 'Vol_Chg', 'Price_Movement', 'Action']
- Edges:
  - RSI_Cat -> Price_Movement
  - Vol_Cat -> Price_Movement
  - Vol_Chg -> Price_Movement
  - Price_Movement -> Action
- CPTs learned using Maximum Likelihood Estimation

### PEAS Description
- **Performance**: Portfolio returns, Sharpe ratio, maximum drawdown
- **Environment**: Real-time market data, historical prices
- **Actuators**: Buy/Sell/Hold signals
- **Sensors**: Price data, volume, technical indicators

### Agent Type
Utility-based agent that maximizes risk-adjusted returns using probabilistic reasoning with Bayesian networks

### Evaluation Results
- Model Returns: $45k
- Buy & Hold Returns: $38k
- Model Sharpe Ratio: 1.2
- Buy & Hold Sharpe Ratio: 0.8
- Model Max Drawdown: -18%
- Buy & Hold Max Drawdown: -33%

## Installation
```bash
# Clone repository
git clone https://github.com/Abhyuday180/cse150a.git
# Install dependencies
pip install -r requirements.txt
# Launch Jupyter notebook
jupyter notebook notebooks/trading_agent.ipynb
```

# Milestone 3 / Final Submission

The project is divided into two main Bayesian network components:
### Market State Estimation Model

Nodes (Variables):
Market Trend: (Bullish, Bearish, Sideways) – Estimated using moving averages relative to the current price.
Stock Price Movement: (Up, Down, Neutral) – Determined by day-to-day price changes.
Trading Volume Trend: (Increasing, Decreasing, Stable) – Derived by quantile discretization of the volume.
Volatility Indicator: (High, Medium, Low) – Derived from the rolling standard deviation of returns.
Edges (Dependencies):
Market Trend influences Stock Price Movement.
Trading Volume Trend influences Volatility Indicator.
Volatility Indicator affects future Stock Price Movements.


### Decision Model for the Trading Agent

Nodes (Variables):
Stock Price Forecast: The probabilistic prediction for future price changes.
Portfolio State: Current cash balance, stock holdings.
Risk Appetite: (Conservative, Moderate, Aggressive) – A parameter that can be set based on user preference.
Buy/Sell/Hold Decision: The final decision output by the agent.

Edges (Dependencies):
Stock Price Forecast influences the Buy/Sell/Hold decision.
Portfolio State and Risk Appetite jointly affect decision-making.

## Training snippet
```bash
import pandas as pd
import numpy as np
from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

# Load dataset
df = pd.read_csv('data/historical_stock_data.csv')

# Feature Engineering
df['MA_20'] = df['Close'].rolling(window=20).mean()
df['Returns'] = df['Close'].pct_change()
df['Volatility'] = df['Returns'].rolling(window=20).std()

def compute_RSI(series, window=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.rolling(window=window).mean()
    roll_down = down.rolling(window=window).mean()
    rs = roll_up / roll_down
    return 100 - (100 / (1 + rs))

df['RSI'] = compute_RSI(df['Close'])

# Discretization
df['MarketTrend'] = np.where(df['Close'] > df['MA_20'], 'Bullish', 'Bearish')
df['PriceMovement'] = np.where(df['Close'].diff() > 0, 'Up', 'Down')
df['VolatilityLevel'] = pd.cut(df['Volatility'], bins=3, labels=['Low', 'Medium', 'High'])
df['TradingVolume'] = pd.qcut(df['Volume'], q=3, labels=['Low', 'Medium', 'High'])
df['RSI_Level'] = pd.cut(df['RSI'], bins=3, labels=['Oversold', 'Neutral', 'Overbought'])

# Prepare data for BN (drop NaNs)
data_bn = df[['MarketTrend', 'PriceMovement', 'RSI_Level', 'TradingVolume', 'VolatilityLevel']].dropna()

# Define Bayesian Network structure
model = BayesianModel([
    ('MarketTrend', 'PriceMovement'),
    ('RSI_Level', 'PriceMovement'),
    ('TradingVolume', 'VolatilityLevel'),
    ('VolatilityLevel', 'PriceMovement')
])

# Estimate parameters using MLE
model.fit(data_bn, estimator=MaximumLikelihoodEstimator)

# Inference Example
inference = VariableElimination(model)
query_result = inference.query(variables=['PriceMovement'],
                                 evidence={'MarketTrend': 'Bullish', 'RSI_Level': 'Neutral', 'VolatilityLevel': 'Low'})
print(query_result)

```
