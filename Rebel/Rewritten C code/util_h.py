def normalize_probabilities_safe(x, epsilon=1e-200):
    
    total = sum(x) + len(x)*epsilon
    return [(x_i + epsilon)/total for x_i in x]