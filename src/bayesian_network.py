from pgmpy.models import BayesianNetwork

def build_network():
    model = BayesianNetwork([
        ('RSI_Cat', 'Price_Movement'),
        ('Vol_Cat', 'Price_Movement'),
        ('Vol_Chg', 'Price_Movement'),
        ('Price_Movement', 'Action')
    ])
    return model
