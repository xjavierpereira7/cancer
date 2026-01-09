import numpy as np

def mavt_matrix(data, weights):
    """
    Calcula las utilidades MAVT para cada alternativa.

    :param data: ndarray (alternativas x criterios), ya normalizado o crudo
    :param weights: Lista o array de pesos de los criterios
    :return: Array con valores de utilidad para cada alternativa
    """
    data = np.array(data)
    weights = np.array(weights)

    # Normalización lineal min-max por columna (criterio)
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    denominators = max_vals - min_vals + 1e-8  # para evitar división por cero

    normalized_data = (data - min_vals) / denominators

    # Cálculo de utilidad agregada por alternativa
    utilities = normalized_data @ weights

    return utilities
