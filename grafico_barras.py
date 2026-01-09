# modulo_grafico_barras.py

import plotly.express as px
import streamlit as st


def mostrar_grafico_comparativo(comparacion_df, fontsize_title, fontsize_ticks, fontsize_labels, height_comp, width_comp):
    """
    Muestra un gráfico de barras comparativo para rankings de comunas.
    """
    st.markdown("## Comparación de rankings entre métodos", unsafe_allow_html=True)

    # Detectar métodos disponibles en el DataFrame
    columnas_disponibles = list(comparacion_df.columns)
    metodos_presentes = [col for col in columnas_disponibles if col != 'comuna']

    if len(metodos_presentes) < 2:
        st.warning("Se requieren al menos dos métodos con datos válidos para generar el gráfico comparativo.")
        return

    fig_comparacion = px.bar(
        comparacion_df,
        x='comuna',
        y=metodos_presentes,
        title="",
        labels={"value": "Ranking Normalizado", "variable": "Método"},
        barmode='group'
    )
    fig_comparacion.update_layout(
        xaxis_title="comuna",
        yaxis_title="Ranking Normalizado",
        xaxis_tickangle=-45,
        xaxis=dict(
            tickfont=dict(size=fontsize_ticks),
            title_font=dict(size=fontsize_labels)
        ),
        yaxis=dict(
            tickfont=dict(size=fontsize_ticks),
            title_font=dict(size=fontsize_labels)
        ),
        legend=dict(font=dict(size=fontsize_labels)),
        height=height_comp,
        width=width_comp,
        margin=dict(t=50, b=150, l=50, r=50),
        template="plotly_white"
    )
    st.plotly_chart(fig_comparacion, use_container_width=True, key="comparacion_general")
