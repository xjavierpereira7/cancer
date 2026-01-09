def color_por_riesgo(valor):
    if valor < 0.3:
        return 'red'
    elif valor < 0.7:
        return 'orange'
    else:
        return 'green'