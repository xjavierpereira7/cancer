import numpy as np


def todim_matrix(data, weights, alpha=1.0, beta=1.0, lambd=1.0):
    """
    Calcula los puntajes de TODIM para cada alternativa.

    :param data: ndarray (alternativas x criterios)
    :param weights: Lista o array de pesos de los criterios
    :param alpha: Exponente para ganancias (por defecto 1.0)
    :param beta: Exponente para pérdidas (por defecto 1.0)
    :param lambd: Parámetro de pérdida (por defecto 1.0)
    :return: Array con valores de riesgo (phi) para cada alternativa
    """
    n, m = data.shape
    phi = np.zeros(n)

    for i in range(n):
        total_value = 0
        for k in range(n):
            if i == k:
                continue
            dominances = []
            for j in range(m):
                diff = data[i, j] - data[k, j]
                w_j = weights[j]
                if diff >= 0:
                    value = w_j * (diff ** alpha)
                else:
                    value = -lambd * w_j * ((-diff) ** beta)
                dominances.append(value)
            total_value += sum(dominances)
        phi[i] = total_value / (n - 1)

    return phi
