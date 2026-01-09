import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

def mostrar_mapa_calor_jerarquico(df, figsize_heatmap_j=(10, 8), fontsize_labels_map_j=10):
    """
    Muestra un mapa de calor jerárquico (clustermap) para criterios numéricos aplicados a comunas.
    Funciona automáticamente con cualquier conjunto de columnas numéricas.

    Parámetros:
    - df: DataFrame que incluye al menos una columna con nombres de comunas y otras numéricas.
    - figsize_heatmap_j: tamaño del gráfico.
    - fontsize_labels_map_j: tamaño de fuente para etiquetas.
    """

    st.markdown("## 📊 Mapa de calor jerárquico", unsafe_allow_html=True)

    # Verificar que exista la columna con el nombre del comuna
    if 'comuna' not in df.columns:
        st.error("El DataFrame debe contener una columna llamada 'comuna'.")
        return

    # Seleccionar columnas numéricas (excluyendo la de nombre)
    columnas_numericas = df.drop(columns=['comuna'], errors='ignore').select_dtypes(include=['number']).columns.tolist()

    if not columnas_numericas:
        st.warning("No hay columnas numéricas disponibles para generar el mapa de calor.")
        return

    if len(columnas_numericas) == 1:
        st.info("Solo hay un criterio numérico. El mapa de calor puede no ser representativo.")

    # Construcción del DataFrame para el clustermap
    df_criterios = df[['comuna'] + columnas_numericas].set_index('comuna')

    # Normalizar los datos
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df_criterios),
                             columns=df_criterios.columns,
                             index=df_criterios.index)

    # Crear el clustermap
    g = sns.clustermap(df_scaled,
                       cmap="viridis",
                       metric="euclidean",
                       method="ward",
                       figsize=figsize_heatmap_j,
                       xticklabels=True,
                       yticklabels=True,
                       cbar_pos=(0.9, 0.2, 0.01, 0.4)
                       )

    # Ajustes visuales
    plt.setp(g.ax_heatmap.get_xticklabels(), fontsize=fontsize_labels_map_j, rotation=45)
    plt.setp(g.ax_heatmap.get_yticklabels(), fontsize=fontsize_labels_map_j)
    g.cax.tick_params(labelsize=fontsize_labels_map_j)
    g.ax_heatmap.set_ylabel("")

    # Mostrar en Streamlit
    st.pyplot(g.fig)
