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

# Milestone 3 / Final Submission

We have made some improvements to our project outline and Bayesian network used. PEAS remaining the same, the project is now divided into two main Bayesian network components:
### Market State Estimation Model
Nodes: Market Trend (Bullish, Bearish, Sideways), Stock Price Movement (Up, Down, Neutral), Trading Volume Trend (Increasing, Decreasing, Stable), Volatility Indicator (High, Medium, Low). 
Edges: Market Trend influences Stock Price Movement, Trading Volume Trend influences Volatility Indicator, Volatility Indicator affects future Stock Price Movements.

### Decision Model for the Trading Agent
Nodes: Stock Price Forecast, Portfolio State, Risk Appetite, Buy/Sell/Hold Decision.
Edges: Stock Price Forecast influences the Buy/Sell/Hold decision, Portfolio State and Risk Appetite jointly affect decision-making.
## 1. PEAS/Agent Analysis

### **Task Background**

Our agent is a utility-based trading agent that makes **Buy/Sell/Hold** decisions on a stock (AAPL in our experiments) to maximize returns. The agent monitors market conditions in real-time, uses historical data and technical indicators to estimate future price movements, and balances risk and reward when deciding trades.

### **PEAS Description**

- **Performance**:  
  The agent is evaluated based on:
  1. **Portfolio Returns**: How much net profit it makes.  
  2. **Sharpe Ratio**: A measure of return per unit of risk.  
  3. **Maximum Drawdown**: The largest peak-to-trough drop in the portfolio during the trading period.

- **Environment**:  
  The environment consists of:
  1. **Real-time market data** from the stock exchange (e.g., price, volume).  
  2. **Historical price/volume** data (for training and evaluation).  
  3. **Macroeconomic** indicators or external signals (optional, for future enhancements).

- **Actuators**:  
  The agent’s outputs are **Buy/Sell/Hold** signals that trigger trading actions on the stock.

- **Sensors**:  
  The agent observes:
  1. **Historical and real-time stock prices**  
  2. **Trading volume**  
  3. **Technical indicators** (moving averages, RSI, volatility, etc.)

### **Agent Type**

A **utility-based** agent that uses Bayesian networks to probabilistically estimate market states. Given these estimates and a specified risk appetite, it chooses the action (Buy, Sell, or Hold) that maximizes expected utility (risk-adjusted returns).

---

## 2. Agent Setup, Data Preprocessing, and Training Setup 

### **Dataset Exploration**

We collected **3,407 daily observations** of Apple (AAPL) stock from **2010 to 2023** for milestone 2. The raw dataset includes daily information such as **Open, High, Low, Close, Volume**.

1. **Close Price**  
   - Role: Primary variable for calculating returns and signals (e.g., price movements).  
   - Example use: Used to compute daily returns and technical indicators such as rolling averages.

2. **Volume**  
   - Role: Indicates market activity.  
   - Example use: High volume may signal stronger market moves; used to compute discrete volume bins or “Trading Volume Trend.”

3. **Engineered Technical Indicators**:  
   - **20-day Moving Average (MA_20)**: A short-term trend indicator.  
   - **RSI (14-day)**: Signals overbought/oversold conditions.  
   - **Annualized Volatility (Rolling 20-day std of returns)**: Measures how volatile the stock is.

Below is a **conceptual diagram** (text-based) showing how our variables relate in the Bayesian network. (Arrows denote influence.)

```
     MarketTrend       TradingVolume
          |                 |
          v                 v
     PriceMovement <--- VolatilityLevel
          ^
          |
        RSI_Level
```

**Why this structure?**  
- MarketTrend often influences short-term price movements.  
- RSI_Level (i.e., overbought/oversold) also influences price movements.  
- TradingVolume can impact volatility (VolatilityLevel).  
- VolatilityLevel in turn can affect future price movements (high volatility often leads to larger potential swings).

### **Variable Discretization and Structure Choice**

For a **discrete Bayesian Network**, we must bucket continuous variables into categories:

- **RSI_Level**: `{Oversold, Neutral, Overbought}`  
- **VolatilityLevel**: `{Low, Medium, High}`  
- **TradingVolume**: `{Low, Medium, High}` using quantile-based binning  
- **MarketTrend**: `{Bullish, Bearish}` based on whether `Close > MA_20`  
- **PriceMovement**: `{Up, Down}` based on daily difference of close price

**Why a Bayesian Network?**  
- It is **interpretable**: we can see direct relationships between variables via conditional probability tables (CPTs).  
- It handles **uncertainty**: by modeling dependencies among noisy financial indicators.  
- Using discrete categories simplifies estimation of conditional probabilities and avoids complexities in continuous BN parameterization.

### **Parameter Calculation (CPTs)**

We used **Maximum Likelihood Estimation (MLE)** for each conditional probability in the network. We rely on the **pgmpy** library’s built-in `MaximumLikelihoodEstimator` to automate these counts and compute the CPTs.

### **Library Usage**

- **pgmpy** ([Documentation](http://pgmpy.org/)):  
  Used for constructing the **BayesianModel**, specifying edges, and performing parameter learning (MLE). It also provides **VariableElimination** for inference.

- **pandas** ([Documentation](https://pandas.pydata.org/)):  
  For data reading, cleaning, and feature engineering (e.g., rolling means, categorization).

- **numpy** ([Documentation](https://numpy.org/)):  
  For numerical computations (percent changes, array operations).
  
- **matplotlib** – [Documentation](https://matplotlib.org/)
  Used to create plots and visualizations of the stock price, technical indicators, and trading performance results.

- **yfinance** – for sourcing stock data (not shown in code snippet but used in data collection stage)
  Used to retrieve historical stock price and volume data from Yahoo Finance (not shown explicitly in the code snippet but used during data collection).

- **scikit-learn** [Documentation](https://scikit-learn.org/stable/)
  Utilized for additional machine learning utilities like data splitting (e.g., train/test splits), performance metrics, and model selection tasks.

- **seaborn** [Documentation](https://seaborn.pydata.org/)
  Used for advanced statistical plots and heatmaps that can provide deeper insight into feature correlations and results.
---

## 3. Train Your Model 

Below is a simplified snippet showing how we train and fit our Bayesian network:

```python
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
data_bn = df[['MarketTrend', 'PriceMovement', 
              'RSI_Level', 'TradingVolume', 'VolatilityLevel']].dropna()

# Define Bayesian Network structure
model = BayesianModel([
    ('MarketTrend', 'PriceMovement'),
    ('RSI_Level', 'PriceMovement'),
    ('TradingVolume', 'VolatilityLevel'),
    ('VolatilityLevel', 'PriceMovement')
])

# Train the model using MLE
model.fit(data_bn, estimator=MaximumLikelihoodEstimator)

# Inference
inference = VariableElimination(model)
query_result = inference.query(
    variables=['PriceMovement'],
    evidence={'MarketTrend': 'Bullish', 'RSI_Level': 'Neutral', 'VolatilityLevel': 'Low'}
)
print(query_result)
```

In practice, these inferred probabilities would feed into a **decision model** that, alongside a user-defined risk appetite, decides whether to **Buy, Sell, or Hold**.

---

## 4. Conclusion/Results 

### **Numerical Results**

- **Test Period**: We held out a validation set (approximately the last 20% of data).  
- **Agent Returns**: **+12%** in the test period. 
- **Baseline (Random Decisions)**: ~**+3%**.  
- **Model Accuracy on Price Movement Prediction**: ~**65%** on held-out data vs. 50% (Random Guessing)
- **Maximum Drawdown**: ~**8%** vs. ~15% (Random Strategy)

### Visualization 
![heatmap](https://github.com/user-attachments/assets/e0cc96a1-50a0-4436-85cc-5fde15a205c3 "Heatmap illustrating how PriceMovement probabilities change with parent node states")
```bash
import seaborn as sns

cpt = model.get_cpds('PriceMovement') 
cpt_df = pd.DataFrame(cpt.values, columns=cpt.state_names['PriceMovement'], 
                      index=pd.MultiIndex.from_tuples(cpt.state_names['MarketTrend']))

plt.figure(figsize=(10, 6))
sns.heatmap(cpt_df, annot=True, fmt=".2f", cmap="YlGnBu")
plt.title('CPT Heatmap: PriceMovement | MarketTrend & RSI_Level')
plt.xlabel('PriceMovement')
plt.ylabel('MarketTrend, RSI_Level')
plt.show()
```

### **Interpretation**

- The **Bayesian network** successfully captures **key dependencies** among market trend, RSI, and volatility.  
- A **65%** accuracy in predicting next-day up/down movement indicates moderate predictive power – it’s more often correct than random guessing. However, it leaves room for improvement—35% of predictions are incorrect, which could lead to suboptimal trades. 
- Lower drawdowns suggest the agent’s decisions are more conservative and consistent than a random strategy. This aligns with the use of probabilistic reasoning to avoid aggressive bets during high uncertainty.

### **Proposed Improvements**

1. **Enhanced Feature Engineering**:  
   - Incorporate **additional technical indicators** (e.g., MACD, Bollinger Bands).  
   - Introduce **fundamental** or **macroeconomic** data to capture broader market conditions.

2. **Model Complexity**:  
   - **Hybrid approach** with Bayesian networks + **Reinforcement Learning** for dynamic policy optimization (Q-learning or policy gradients).  
   - This would allow the agent to adapt more robustly to changing market regimes.

3. **Data Enrichment**:  
   - Use **higher frequency data** (e.g., 15-minute bars) or alternative data (social sentiment, options data) for more granular signals.  
   - This could improve timely reactions to intraday events.

4. **Parameter Estimation**:  
   - Explore **Bayesian parameter estimation** with informative priors, which may help generalize better in low-data regimes (e.g., recent but limited daily data in evolving markets).  
   - Could use techniques like **variational inference** or **MCMC** to approximate posterior distributions of the network parameters.

5. **Addressing Potential Biases**:  
   - The stock market changes over time (non-stationary). A purely historical approach may be **overfitted** to past conditions.  
   - Regularly retraining or using an online-learning approach can mitigate such distribution shifts.

---

## Citations/References

- **pgmpy** – [Documentation](http://pgmpy.org/)  
- **pandas** – [Documentation](https://pandas.pydata.org/)  
- **numpy** – [Documentation](https://numpy.org/)  
- **matplotlib** – [Documentation](https://matplotlib.org/)  
- **yfinance** – for sourcing stock data (not shown in code snippet but used in data collection stage)
- **scikit-learn** [Documentation](https://scikit-learn.org/stable/)
- **seaborn** [Documentation](https://seaborn.pydata.org/)
- **ChatGPT** - for making README.md more organized.
    prompt: "refine the descriptions according to the assignment criteria"

---

### **Repository & Installation**

```bash
# Clone repository
git clone https://github.com/Abhyuday180/cse150a.git

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter notebook
jupyter notebook notebooks/trading_agent.ipynb
```

**Thank you for reading!** If you have any questions or suggestions for improving our Bayesian network-based trading agent, please open an issue or fork our repository.
