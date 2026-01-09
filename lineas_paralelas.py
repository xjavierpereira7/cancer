import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np

def mostrar_lineas_comparativas(df_rankings, height):
    st.markdown("## 🌊 Comparación de métodos por comuna ", unsafe_allow_html=True)

    columnas_metodos = df_rankings.select_dtypes(include=['number']).columns.tolist()

    if len(columnas_metodos) < 2:
        st.warning("Se necesitan al menos dos métodos con valores numéricos para comparar.")
        return

    # Calcular promedio
    df_rankings['RankingPromedio'] = df_rankings[columnas_metodos].mean(axis=1)

    # Escala azul a amarillo usando YlGnBu invertido o personalizada
    cmap = cm.get_cmap('viridis')  # Invertido: azules bajos → amarillo alto
    norm = mcolors.Normalize(vmin=df_rankings['RankingPromedio'].min(), vmax=df_rankings['RankingPromedio'].max())

    fig = go.Figure()

    for _, row in df_rankings.iterrows():
        metodo_vals = row[columnas_metodos].values
        promedio = row['RankingPromedio']
        color_rgb = cmap(norm(promedio))
        color_hex = mcolors.to_hex(color_rgb)

        fig.add_trace(go.Scatter(
            x=columnas_metodos,
            y=metodo_vals,
            mode='lines+markers',
            name=row['comuna'],
            line=dict(color=color_hex, width=2),
            marker=dict(size=6),
            hovertemplate=f"<b>{row['comuna']}</b><br>" +
                          "<br>".join([f"{m}: {v:.2f}" for m, v in zip(columnas_metodos, metodo_vals)]) +
                          f"<br>Promedio: {promedio:.2f}<extra></extra>"
        ))

    # Escala para el colorbar
    color_values = np.linspace(df_rankings['RankingPromedio'].min(), df_rankings['RankingPromedio'].max(), 100)

    fig.add_trace(go.Scatter(
        x=[None]*100,
        y=[None]*100,
        mode='markers',
        marker=dict(
            colorscale='YlGnBu_r',  # azul -> verde-amarillo
            cmin=df_rankings['RankingPromedio'].min(),
            cmax=df_rankings['RankingPromedio'].max(),
            color=color_values,
            colorbar=dict(
                title=dict(
                    text="",
                    side="right"  # Esto reemplaza 'titleside'
                ),
                outlinewidth=0
            ),
            showscale=True,
            size=0.1  # invisible
        ),
        hoverinfo='none',
        showlegend=False
    ))

    fig.update_layout(
        title='',
        xaxis_title='Método',
        yaxis_title='Ranking',
        height=height,
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)
