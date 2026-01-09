import folium
import streamlit as st
from streamlit_folium import st_folium
from utilidad import color_por_riesgo  # Asegúrate de tener esta función en un módulo común


def mostrar_mapas_pequenos(gdf, metodos_disponibles, rankings, map_center, width, height):

    st.markdown("---")
    st.markdown("## 🔍 Comparación Visual por Método", unsafe_allow_html=True)

    columnas = st.columns(2)  # Crear 2 columnas por fila
    col_index = 0

    for metodo_nombre, ranking_actual in metodos_disponibles.items():
        if ranking_actual is not None:
            gdf_temp = gdf.copy()
            gdf_temp['ranking'] = ranking_actual

            # Crear mapa pequeño con color consistente (sin normalizar)
            mapa = folium.Map(location=map_center, zoom_start=8, tiles="OpenStreetMap")
            for _, row in gdf_temp.iterrows():
                color = color_por_riesgo(row['ranking'])
                folium.CircleMarker(
                    location=[row['lat'], row['lon']],
                    radius=2,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.8,
                    popup=f"{row['comuna']}: {row['ranking']:.2f}"
                ).add_to(mapa)

            # Mostrar el mapa en una columna
            with columnas[col_index % 2]:
                st.markdown(f"#### {metodo_nombre}")
                st_folium(mapa, width=width, height=height)

            col_index += 1
