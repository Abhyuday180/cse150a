from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator

def build_network():
    model = BayesianModel([
        ('MarketTrend', 'PriceMovement'),
        ('RSI_Level', 'PriceMovement'),
        ('TradingVolume', 'VolatilityLevel'),
        ('VolatilityLevel', 'PriceMovement')
    ])
    
    # Fit the model using Maximum Likelihood Estimator from pgmpy
    model.fit(data, estimator=MaximumLikelihoodEstimator)
    
    # Print out the learned CPDs (Conditional Probability Distributions)
    print("Learned CPDs:")
    for cpd in model.get_cpds():
        print(cpd)
    
    return model
