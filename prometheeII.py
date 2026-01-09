# promethee_module.py
import numpy as np

def preference_function_type_v(d, q, p):
    if d <= q:
        return 0
    elif d > p:
        return 1
    else:
        return (d - q) / (p - q)

def compute_preference_matrix(data, weights, q_vals, p_vals):
    n, m = data.shape
    preference_matrix = np.zeros((n, n))
    for i in range(n):
        for k in range(n):
            if i == k:
                continue
            score = sum(
                weights[j] * preference_function_type_v(data[i, j] - data[k, j], q_vals[j], p_vals[j])
                for j in range(m)
            )
            preference_matrix[i, k] = score
    return preference_matrix

def compute_flows(matrix):
    n = matrix.shape[0]
    phi_plus = np.sum(matrix, axis=1) / (n - 1)
    phi_minus = np.sum(matrix, axis=0) / (n - 1)
    phi = phi_plus - phi_minus
    return phi

def promethee_matrix(data, pesos, q_vals, p_vals):
    matrix = compute_preference_matrix(data, pesos, q_vals, p_vals)
    return compute_flows(matrix)

