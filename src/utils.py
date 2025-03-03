import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def download_data(ticker, start_date, end_date):
    """Download historical stock data from Yahoo Finance"""
    df = yf.download(ticker, start=start_date, end=end_date)
    df = df.reset_index()
    return df

def add_technical_indicators(df):
    """Calculate technical indicators for the dataset"""
    # Moving Average
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    
    # RSI Calculation
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Volatility
    df['Volatility'] = df['Close'].pct_change().rolling(window=20).std() * np.sqrt(252)
    
    return df

def handle_missing_values(df):
    """Handle missing values in the dataset"""
    df = df.dropna(subset=['MA_20', 'RSI', 'Volatility'])  # These columns have rolling windows
    df = df.reset_index(drop=True)
    return df

def discretize_features(df):
    """Convert continuous features to categorical"""
    # RSI Binning (30-70 is common range)
    df['RSI_Cat'] = pd.cut(df['RSI'],
                          bins=[0, 30, 70, 100],
                          labels=['Oversold', 'Neutral', 'Overbought'])
    
    # Volatility Binning
    df['Vol_Cat'] = pd.cut(df['Volatility'],
                          bins=[0, 0.2, 0.4, np.inf],
                          labels=['Low', 'Medium', 'High'])
    
    # Volume Change Direction
    df['Vol_Chg'] = np.sign(df['Volume'].pct_change())
    df['Vol_Chg'] = df['Vol_Chg'].map({1: 'Up', -1: 'Down', 0: 'No Change'})
    
    return df

def prepare_target(df):
    """Create target variable for price movement"""
    df['Price_Movement'] = np.where(df['Close'].shift(-1) > df['Close'], 'Up', 'Down')
    return df

def split_data(df, test_size=0.2, random_state=42):
    """Split data into training and test sets"""
    train_df, test_df = train_test_split(df, test_size=test_size, 
                                       random_state=random_state, shuffle=False)
    return train_df, test_df
