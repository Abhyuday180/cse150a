# AI Trading Agent with Bayesian Networks

# Members
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
