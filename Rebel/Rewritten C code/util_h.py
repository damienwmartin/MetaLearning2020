def normalize_probabilities_safe(x, epsilon=1e-200):
    
    total = sum(x) + len(x)*epsilon
    return [(x_i + epsilon)/total for x_i in x]

def resize(x, size):
    """
    Python implementation of the C++ standard .resize() operation
    """
    N = len(x)
    if size > N:
        y = x[:]
        y.extend([0]*(size - N))
        return y
    else:
        return x[:size]
