import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


def mostrar_comparacion_resultados(gdf, metodos_resultados):
    """
    Muestra una tabla comparativa de resultados de ranking de comunas
    usando diferentes métodos y un gráfico de barras para visualización.

    Parámetros:
    - gdf: GeoDataFrame con la columna 'nombre' de las comunas.
    - metodos_resultados: dict con los nombres de métodos como claves y
      listas o Series de valores phi como valores.
    """
    # Combinar los resultados en un DataFrame
    resultados_df = pd.DataFrame(metodos_resultados)
    resultados_df["comuna"] = gdf["nombre"].values
    resultados_df.set_index("comuna", inplace=True)

    # Mostrar tabla
    st.subheader("Comparación de Rankings entre Métodos")
    st.dataframe(resultados_df.style.format(precision=3))

    # Mostrar gráfico de barras
    st.subheader("Visualización de Rankings por Método")
    st.markdown("Ranking relativo de cada glaciar según el método de evaluación.")

    fig, ax = plt.subplots(figsize=(10, 6))
    resultados_df.plot(kind="bar", ax=ax)
    ax.set_ylabel("Ranking / Valor Phi")
    ax.set_title("Comparación de Métodos de Evaluación")
    ax.legend(title="Método")
    st.pyplot(fig)

    # Botón de descarga de tabla CSV
    csv = resultados_df.reset_index().to_csv(index=False).encode("utf-8")
    st.download_button("Descargar tabla CSV", csv, "comparacion_resultados.csv", "text/csv")
