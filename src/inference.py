from pgmpy.inference import VariableElimination

def perform_inference(model, evidence):
    # Initialize the inference object with the trained model
    infer = VariableElimination(model)
    
    # Query the 'PriceMovement' node given the provided evidence
    query_result = infer.query(variables=['PriceMovement'], evidence=evidence)
    
    # Print the inference result to check the output probabilities
    print("Inference result for PriceMovement given evidence {}: \n{}".format(evidence, query_result))
    
    return query_result
