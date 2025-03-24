import numpy as np

def safe_softmax(x):
    """
    Computes the softmax function with numerical stability.
    
    This function subtracts the maximum value from the input to prevent overflow, 
    ensuring stability in exponential calculations. If the resulting probabilities 
    contain NaN or infinite values, it falls back to a uniform distribution.
    
    Args:
        x (numpy.ndarray): Input array.
    
    Returns:
        numpy.ndarray: Softmax probabilities.
    """
    x = x - np.max(x)
    probs = np.exp(x) / np.sum(np.exp(x))
    if np.any(np.isnan(probs)) or np.any(np.isinf(probs)):
        probs = np.ones_like(x) / len(x)
    return probs
