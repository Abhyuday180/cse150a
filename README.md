# AI Trading Agent with Bayesian Networks

## Project Overview
An utility-based AI trading agent using Bayesian networks to make buy/sell/hold decisions based on market indicators.

## Milestone 2 Updates

### Data Exploration
- Collected 3,407 daily observations of AAPL stock (2010-2023)
- Engineered features: 20-day MA, 14-day RSI, Annualized Volatility
- Discretized continuous variables into categorical states

### Bayesian Network
![Network Structure](images/network.png)
- Nodes: RSI_Cat, Vol_Cat, Vol_Chg → Price_Movement → Action
- CPTs learned using Maximum Likelihood Estimation

### PEAS Description
- **Performance**: Portfolio returns, Sharpe ratio, maximum drawdown
- **Environment**: Real-time market data, historical prices
- **Actuators**: Buy/Sell/Hold signals
- **Sensors**: Price data, volume, technical indicators

### Agent Type
Utility-based agent that maximizes risk-adjusted returns using probabilistic reasoning with Bayesian networks

### Evaluation Results
- Model Returns: $45,320.00
- Buy & Hold Returns: $38,450.00

## Installation
```bash
git clone https://github.com/yourusername/ai-trading-agent.git
cd ai-trading-agent
pip install -r requirements.txt
