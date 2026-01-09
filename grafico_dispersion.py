import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def mostrar_grafico_dispersion(df_scatter, fontsize_title, fontsize_labels, height_disp=500, width_disp=500):
    """
    Muestra una grilla cuadrada (n x n) de gráficos de dispersión para comparar métodos.

    Parámetros:
    - df_scatter: DataFrame con columnas = métodos, índice = comunas.
    - fontsize_labels: tamaño de fuente para etiquetas.
    - height_disp, width_disp: dimensiones base para cada subplot.
    """

    st.markdown("## 🌫️ Dispersión de rankings entre métodos", unsafe_allow_html=True)

    # Detectar métodos disponibles en el DataFrame
    metodos_posibles = ['PROMETHEE', 'TODIM', 'MAVT', 'Red Neuronal']
    metodos = [m for m in metodos_posibles if m in df_scatter.columns]

    n = len(metodos)
    if n < 2:
        st.warning("Se necesitan al menos dos métodos para mostrar el gráfico de dispersión.")
        return

    # Crear grilla cuadrada n x n
    fig = make_subplots(rows=n, cols=n, shared_xaxes=True, shared_yaxes=True,
                        horizontal_spacing=0.02, vertical_spacing=0.02)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue  # Saltar la diagonal

            x = df_scatter[metodos[j]]
            y = df_scatter[metodos[i]]

            fig.add_trace(go.Histogram2dContour(
                x=x,
                y=y,
                colorscale='Blues',
                contours=dict(coloring='heatmap'),
                showscale=False,
                ncontours=15,
                opacity=0.6
            ), row=i + 1, col=j + 1)

            fig.add_trace(go.Scatter(
                x=x,
                y=y,
                mode='markers',
                marker=dict(size=4, color='black', opacity=0.5),
                showlegend=False
            ), row=i + 1, col=j + 1)

            min_val = min(x.min(), y.min())
            max_val = max(x.max(), y.max())

            fig.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                line=dict(color='gray', dash='dot'),
                showlegend=False
            ), row=i + 1, col=j + 1)

            if i == n - 1:
                fig.update_xaxes(title_text=metodos[j], row=i + 1, col=j + 1)
            if j == 0:
                fig.update_yaxes(title_text=metodos[i], row=i + 1, col=j + 1)

    # Tamaño total depende del número de filas/columnas (n x n)
    fig.update_layout(
        height=height_disp * n,
        width=width_disp * n,
        font=dict(size=fontsize_labels),
        margin=dict(t=50, l=50, r=20, b=20)
    )

    st.plotly_chart(fig, use_container_width=False, key="grafico_dispersion_cuadrado")
