import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import streamlit as st
import io

def mostrar_mapa_calor(df_heatmap, figsize=(12,8), fontsize_labels=12):
    st.markdown("## 🔥 Mapa de calor de rankings", unsafe_allow_html=True)
    df_heatmap = df_heatmap.set_index('comuna')

    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    heatmap = sns.heatmap(df_heatmap.T, cmap='coolwarm', cbar=True, ax=ax)

    ax.tick_params(axis='x', labelsize=fontsize_labels, rotation=45)
    ax.tick_params(axis='y', labelsize=fontsize_labels)
    ax.set_xlabel("")

    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=fontsize_labels)

    fig.tight_layout()

    # Guardar figura en buffer y mostrar como imagen
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300)
    buf.seek(0)
    st.image(buf)
