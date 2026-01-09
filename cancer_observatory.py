import joblib
import streamlit as st
import pandas as pd
# import geopandas as gpd
import numpy as np
import folium
import yaml
from streamlit_folium import st_folium
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from prometheeII import promethee_matrix
from todim import todim_matrix
from mavt import mavt_matrix

from config_visual import config
from evaluacion_mcda import evaluar_metodo_mcda
from mapas_pequenos import mostrar_mapas_pequenos
from mapa_calor import mostrar_mapa_calor
from grafico_barras import mostrar_grafico_comparativo
from grafico_dispersion import mostrar_grafico_dispersion
from mapa_calor_jerarquico import mostrar_mapa_calor_jerarquico
from lineas_paralelas import mostrar_lineas_comparativas

from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import statsmodels.api as sm
from red_neuronal import guardar_evaluacion_para_entrenamiento, reentrenar_red_neuronal

st.set_page_config(layout="wide", page_title="Evaluación de localidades")

# ========= Estilo CSS con background atractivo =========
st.markdown("""
<style>
body {
    background-image: url('https://images.unsplash.com/photo-1587049352854-9d18ddce3c1a');
    background-size: cover;
    background-attachment: fixed;
}

.main {
    background-color: rgba(255, 255, 255, 0.85);
    padding: 2rem;
    border-radius: 1rem;
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
}

.navbar {
    background-color: #2c3e50;
    padding: 12px;
    color: white;
    text-align: center;
    font-size: 24px;
    font-weight: bold;
    border-bottom: 4px solid #2980b9;
    margin-bottom: 20px;
    border-radius: 0 0 10px 10px;
}

.stButton > button {
    background-color: #2980b9;
    color: white;
    border-radius: 8px;
    padding: 8px 16px;
    border: none;
}

.stButton > button:hover {
    background-color: #3498db;
}

.taxonomia-panel {
    position: fixed;
    top: 100px;
    right: 20px;
    width: 400px;
    max-height: 80%;
    overflow-y: auto;
    background-color: white;
    border: 2px solid #888;
    padding: 16px;
    z-index: 9999;
    font-size: 13px;
    box-shadow: 2px 2px 8px rgba(0,0,0,0.3);
    border-radius: 6px;
}
.taxonomia-panel h2 {
    font-size: 18px;
    margin-bottom: 10px;
    color: #2c3e50;
}
.taxonomia-panel ul {
    padding-left: 20px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="navbar">🧊 Cance-R : Evaluación de Riesgo de Cancer</div>', unsafe_allow_html=True)


# ========= Función para cargar y renderizar taxonomía desde YAML =========
def cargar_taxonomia_desde_dict(data):
    def recorrer_nodos(nodo):
        html = "<ul>"
        if isinstance(nodo, dict):
            for clave, valor in nodo.items():
                html += "<li>"
                if isinstance(valor, dict):
                    nombre = valor.get("nombre", "")
                    descripcion = valor.get("descripcion", "")
                    if nombre or descripcion:
                        html += f"<b>{clave}:</b> {nombre}"
                        if descripcion:
                            html += f"<br><i>{descripcion}</i>"
                        # Recursión para subnodos
                        subnodos = {k: v for k, v in valor.items() if isinstance(v, dict) or isinstance(v, list)}
                        if subnodos:
                            html += recorrer_nodos(subnodos)
                    else:
                        html += f"<b>{clave}</b>"
                        html += recorrer_nodos(valor)
                elif isinstance(valor, list):
                    html += f"<b>{clave}</b><ul>"
                    for item in valor:
                        if isinstance(item, dict):
                            nombre = item.get("nombre", "")
                            descripcion = item.get("descripcion", "")
                            html += "<li>"
                            if nombre:
                                html += f"<b>{nombre}</b>"
                            if descripcion:
                                html += f"<br><i>{descripcion}</i>"
                            html += "</li>"
                        else:
                            html += f"<li>{item}</li>"
                    html += "</ul>"
                else:
                    html += f"<b>{clave}:</b> {valor}"
                html += "</li>"
        html += "</ul>"
        return html

    return f"""<div class="taxonomia-panel"><h2>📖 Taxonomía de Evaluación</h2>{recorrer_nodos(data)}</div>"""


def cargar_taxonomia_desde_yaml(path="cancer_taxonomy.yaml"):
    with open(path, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)
    return cargar_taxonomia_desde_dict(data)

# Normalización de los rankings (min-max)
def normalizar_rankings(ranking):
    min_val = min(ranking)
    max_val = max(ranking)
    return [(x - min_val) / (max_val - min_val) for x in ranking]

def reset_session_state_por_cambio_archivo():
    """Reinicia los valores del sidebar y rankings si el archivo CSV cambia."""
    st.session_state.ranking_promethee = None
    st.session_state.ranking_todim = None
    st.session_state.ranking_mavt = None
    st.session_state.ranking_red_neuronal = None

    st.session_state["metodo"] = "PROMETHEE"
    st.session_state["grafico"] = "Selecciona gráfico"
    st.session_state["riesgo_filtro"] = "Todos"
    st.session_state["opcion_taxonomia"] = "Ocultar"

    st.session_state["q"] = 0.05
    st.session_state["p"] = 0.2
    st.session_state["alpha"] = 1.0
    st.session_state["beta"] = 1.0
    st.session_state["lambda"] = 1.0


# ========= Estado =========
if 'mostrar_taxonomia' not in st.session_state:
    st.session_state.mostrar_taxonomia = False
if 'mostrar_tabla' not in st.session_state:
    st.session_state.mostrar_tabla = False
if 'mostrar_comparacion' not in st.session_state:
    st.session_state.mostrar_comparacion = False

# Mostrar opciones del sidebar
st.sidebar.title("Menú de Configuración")

# ========= Sidebar =========
archivo = st.sidebar.file_uploader("📁 Subir archivo CSV de localidades", type="csv")

# Detectar si el archivo cambió
if 'archivo_anterior' not in st.session_state:
    st.session_state.archivo_anterior = None

if archivo is not None and archivo != st.session_state.archivo_anterior:
    st.session_state.archivo_anterior = archivo
    reset_session_state_por_cambio_archivo()


metodo = st.sidebar.selectbox(
    "🧮 Selecciona el método de evaluación",
    options=["PROMETHEE", "TODIM", "MAVT", "Red Neuronal"],
    key="metodo"
)

opciones_riesgo = ['Todos', 'red', 'orange', 'green']
riesgo_filtro = st.sidebar.selectbox(
    "🎯 Filtrar por nivel de riesgo",
    options=opciones_riesgo,
    index=0,
    key="riesgo_filtro"
)

opcion_taxonomia = st.sidebar.selectbox(
    "🔍 Mostrar u ocultar la taxonomía",
    options=["Ocultar", "Mostrar"],
    index=0,
    key="opcion_taxonomia"
)
# Control de visibilidad según la opción
st.session_state.mostrar_taxonomia = (opcion_taxonomia == "Mostrar")

grafico = st.sidebar.selectbox(
    "🧮 Selecciona el gráfico comparativo",
    options=["Selecciona gráfico", "Mapas", "Mapa de calor", "Gráfico de rankings",
             "Dispersión bidireccional", "Mapa de calor jerárquico", "Coordenadas paralelas"],
    key="grafico"
)

# Parametros PROMETHEE
if metodo == "PROMETHEE":
    st.sidebar.markdown("### Parámetros de PROMETHEE")
    q_valor = st.sidebar.slider("Valor q (indiferencia)", 0.0, 3.0, step=0.01, key="q")
    p_valor = st.sidebar.slider("Valor p (preferencia)", 0.0, 3.0, step=0.01, key="p")

# Parametros TODIM
if metodo == "TODIM":
    st.sidebar.markdown("### Parámetros de TODIM")
    alpha = st.sidebar.slider("α (parámetro TODIM)", 0.1, 3.0, step=0.1, key="alpha")
    beta = st.sidebar.slider("β (ponderación MAVT)", 0.1, 3.0, step=0.1, key="beta")
    lambd = st.sidebar.slider("λ (aversión a la pérdida)", 0.1, 5.0, step=0.1, key="lambda")

# Inicializa variables persistentes en sesión
if 'ranking_promethee' not in st.session_state:
    st.session_state.ranking_promethee = None

if 'ranking_todim' not in st.session_state:
    st.session_state.ranking_todim = None

if 'ranking_mavt' not in st.session_state:
    st.session_state.ranking_mavt = None

if 'ranking_red_neuronal' not in st.session_state:
    st.session_state.ranking_red_neuronal = None

# Permite al usuario activar/desactivar la regeneración de rede neuronal
actualizar_red = st.sidebar.checkbox("🔄 Actualizar red neuronal con evaluaciones actuales")


# ========= Mapa base =========
map_center = [-33.6, -70.2]
m = folium.Map(location=map_center, zoom_start=8, tiles="OpenStreetMap")
col1, col2 = st.columns([2, 1])  # Ajusta la proporción mapa : gráfico
height_main_map=400
width_main_map=500


# ========= colores de puntos en el Mapa base =========
def color_por_riesgo(valor):
    if valor < 0.3:
        return 'red'
    elif valor < 0.7:
        return 'orange'
    else:
        return 'green'


# ========= CARGAR  Y PROCESAR DATOS =========
if archivo:
    df = pd.read_csv(archivo, sep=';', encoding='ISO-8859-1')
    df.columns = df.columns.str.strip().str.lower()
    # Crear lista de rankings disponibles para análisis comparativo
    ranking_disponibles = {}

    for col in df.columns:
        if col not in ['lat', 'lon', 'comuna']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    criterios = [col for col in df.columns if col not in ['lat', 'lon', 'comuna']]
    # Pesos de los criterios
    pesos = np.full(len(criterios), 1 / len(criterios))

    # gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['lon'], df['lat']), crs="EPSG:4326")
    data = df[criterios].to_numpy()

    # Evaluación
    (ranking,
     st.session_state.ranking_promethee,
     st.session_state.ranking_todim,
     st.session_state.ranking_mavt,
     st.session_state.ranking_red_neuronal) = evaluar_metodo_mcda(
        metodo=metodo,
        data=data,
        criterios=criterios,
        pesos=pesos,
        q_vals=np.full(len(criterios), q_valor) if metodo == "PROMETHEE" else None,
        p_vals=np.full(len(criterios), p_valor) if metodo == "PROMETHEE" else None,
        alpha=alpha if metodo == "TODIM" else None,
        beta=beta if metodo == "TODIM" else None,
        lambd=lambd if metodo == "TODIM" else None
    )

    ranking_normalizado_por_metodo = {}
    # Normalizar los rankings solo después de presionar el botón
    if st.session_state.ranking_promethee is not None:
        ranking_promethee_normalizado = normalizar_rankings(st.session_state.ranking_promethee)
        ranking_disponibles["PROMETHEE"] = st.session_state.ranking_promethee
        ranking_normalizado_por_metodo["PROMETHEE"] = ranking_promethee_normalizado
        guardar_evaluacion_para_entrenamiento(df, ranking_promethee_normalizado, 'ranking_entrenamiento_cancer.csv')
    if st.session_state.ranking_todim is not None:
        ranking_todim_normalizado = normalizar_rankings(st.session_state.ranking_todim)
        ranking_disponibles["TODIM"] = st.session_state.ranking_todim
        ranking_normalizado_por_metodo["TODIM"] = ranking_todim_normalizado
        guardar_evaluacion_para_entrenamiento(df, ranking_todim_normalizado, 'ranking_entrenamiento_cancer.csv')
    if st.session_state.ranking_mavt is not None:
        ranking_mavt_normalizado = normalizar_rankings(st.session_state.ranking_mavt)
        ranking_disponibles["MAVT"] = st.session_state.ranking_mavt
        ranking_normalizado_por_metodo["MAVT"] = ranking_mavt_normalizado
        guardar_evaluacion_para_entrenamiento(df, ranking_mavt_normalizado, 'ranking_entrenamiento_cancer.csv')
    if st.session_state.ranking_red_neuronal is not None:
        ranking_red_neuronal_normalizado = normalizar_rankings(st.session_state.ranking_red_neuronal)
        ranking_disponibles["Red Neuronal"] = st.session_state.ranking_red_neuronal
        ranking_normalizado_por_metodo["Red Neuronal"] = ranking_red_neuronal_normalizado
        # df['phi'] = ranking_red_neuronal_normalizado

    # Reentrena la red neuronal con los resultados generados, si el usuario decide reentrenar
    if actualizar_red:
        metodo_entrenamiento = st.selectbox("Selecciona método para generar etiquetas", ["PROMETHEE", "TODIM", "MAVT"])
        # Asumimos que ya se generó el ranking correspondiente en session_state
        if metodo_entrenamiento == "PROMETHEE" and st.session_state.ranking_promethee is not None:
            df['phi'] = ranking_promethee_normalizado
            reentrenar_red_neuronal(df, 'modelo_cancer_pulmon.pkl')
        elif metodo_entrenamiento == "TODIM" and st.session_state.ranking_todim is not None:
            df['phi'] = ranking_todim_normalizado
            reentrenar_red_neuronal(df, 'modelo_cancer_pulmon.pkl')
        elif metodo_entrenamiento == "MAVT" and st.session_state.ranking_mavt is not None:
            df['phi'] = ranking_mavt_normalizado
            reentrenar_red_neuronal(df, 'modelo_cancer_pulmon.pkl')
        else:
            st.warning("No hay ranking disponible para el método seleccionado.")

    ranking = normalizar_rankings(ranking)
    # gdf['riesgo'] = ranking

    df['riesgo'] = ranking

    # Visualiza los localidades en el mapa, con codigo de colores
    if riesgo_filtro == 'Todos':
        riesgo_filtro = ['red', 'orange', 'green']
    else:
        riesgo_filtro = [riesgo_filtro]

    gdf_filtrado = df[df['riesgo'].apply(color_por_riesgo).isin(riesgo_filtro)]

    for _, fila in gdf_filtrado.iterrows():
        lat, lon = fila['lat'], fila['lon']
        color = color_por_riesgo(fila['riesgo'])
        popup_text = f"<b>{fila['comuna']}</b><br>ranking: {fila['riesgo']:.3f}"
        folium.Circle(
            location=[lat, lon],
            radius=1500,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.9,
            popup=folium.Popup(popup_text, max_width=300)
        ).add_to(m)
    with col2:
        st.markdown("### 📊 Ranking de Comunas")

        if 'ranking' in locals():
            ranking_df = pd.DataFrame({
                'comuna': df['comuna'],
                'Ranking': ranking
            }).sort_values(by='Ranking', ascending=False)

            fig_bar = px.bar(
                ranking_df,
                x='Ranking',
                y='comuna',
                orientation='h',
                height=height_main_map,
                color='Ranking',
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig_bar, use_container_width=True)
else:
    # Crear lista de rankings disponibles para análisis comparativo
    ranking_disponibles = {}
    # Limpiar rankings si no hay archivo cargado
    st.session_state.ranking_promethee = None
    st.session_state.ranking_todim = None
    st.session_state.ranking_mavt = None
    st.session_state.ranking_red_neuronal = None
    grafico = []



# ========= Mostrar mapa =========
with col1:
    st.markdown("### 🌍 Mapa de Comunas")
    st_folium(m, width=width_main_map, height=height_main_map)

# ========= Mostrar taxonomía =========
if st.session_state.mostrar_taxonomia:
    st.markdown(cargar_taxonomia_desde_yaml(), unsafe_allow_html=True)


# Calcular centro del mapa si no está definido
if 'map_center' not in globals():
    map_center = [df['lat'].mean(), df['lon'].mean()]


#=== ESCALAS DE LOS GRAFICOS
# General
esc_g = config["general"]["escala"]
figsize = config["general"]["figsize"](esc_g)

# Barras
esc_b = config["barras"]["escala"]
fs_b = config["barras"]["font_scale"]
figsize_barras = config["barras"]["figsize"](esc_b)
fontsize_title_barras = config["barras"]["fontsize_title"](esc_b, fs_b)
fontsize_labels_barras = config["barras"]["fontsize_labels"](esc_b, fs_b)
fontsize_ticks_barras = config["barras"]["fontsize_ticks"](esc_b, fs_b)
height_comp_barras = config["barras"]["height"](esc_b, fs_b)
width_comp_barras = config["barras"]["width"](esc_b, fs_b)

# Heatmap
esc_h = config["heatmap"]["escala_heatmap"]
fs_h = config["heatmap"]["font_scale_heatmap"]
figsize_heatmap = config["heatmap"]["figsize_heatmap"](esc_h)
fontsize_title_heatmap = config["heatmap"]["fontsize_title_heatmap"](esc_h, fs_h)
fontsize_labels_heatmap = config["heatmap"]["fontsize_labels_heatmap"](esc_h, fs_h)
fontsize_ticks_heatmap = config["heatmap"]["fontsize_ticks_heatmap"](esc_h, fs_h)
height_comp_heatmap = config["heatmap"]["height_heatmap"](esc_h, fs_h)
width_comp_heatmap = config["heatmap"]["width_heatmap"](esc_h, fs_h)

# Mapa
esc_m = config["mapa"]["escala"]
fs_m = config["mapa"]["font_scale"]
fontsize_title_map = config["mapa"]["fontsize_title"](esc_m, fs_m)
fontsize_labels_map = config["mapa"]["fontsize_labels"](esc_m, fs_m)
fontsize_ticks_map = config["mapa"]["fontsize_ticks"](esc_m, fs_m)
height_mapa = config["mapa"]["height"](esc_m)
width_mapa = config["mapa"]["width"](esc_m)
escala = config["mapa"]["escala"]


# Dispersión
esc_d = config["dispersion"]["escala_d"]
fs_d = config["dispersion"]["font_scale_d"]
fontsize_title_d = config["dispersion"]["fontsize_title_d"](esc_m, fs_d)
fontsize_labels_d = config["dispersion"]["fontsize_labels_d"](esc_m, fs_d)
height_disp = config["dispersion"]["height"](esc_d)
width_disp = config["dispersion"]["width"](esc_d)

# Mapa J
esc_j = config["mapa_j"]["escala_j"]
fs_j = config["mapa_j"]["font_scale_j"]
figsize_heatmap_j = config["mapa_j"]["figsize_j"](esc_j)
fontsize_title_map_j = config["mapa_j"]["fontsize_title_j"](esc_j, fs_j)
fontsize_labels_map_j = config["mapa_j"]["fontsize_labels_j"](esc_j, fs_j)
fontsize_ticks_map_j = config["mapa_j"]["fontsize_ticks_j"](esc_j, fs_j)
height_mapa_j = config["mapa_j"]["height_j"](esc_j)
width_mapa_j = config["mapa_j"]["width_j"](esc_j)

# Líneas
esc_l = config["lineas"]["escala_lineas"]
fs_l = config["lineas"]["font_scale_lineas"]
figsize_lineas = config["lineas"]["figsize_lineas"](esc_l)
fontsize_title_lineas = config["lineas"]["fontsize_title_lineas"](esc_l, fs_l)
fontsize_labels_lineas = config["lineas"]["fontsize_labels_lineas"](esc_l, fs_l)
fontsize_ticks_lineas = config["lineas"]["fontsize_ticks_lineas"](esc_l, fs_l)
height_lineas = config["lineas"]["height_lineas"](esc_l)
width_lineas = config["lineas"]["width_lineas"](esc_l)

#========= Mostrar gráficos comparativos  =========
#if st.session_state.mostrar_comparacion:
# Activar automáticamente la comparación si se elige un gráfico válido
graficos_comparativos = [
    "Mapas",
    "Mapa de calor",
    "Gráfico de rankings",
    "Dispersión bidireccional",
    "Mapa de calor jerárquico",
    "Coordenadas paralelas"
]

if grafico in graficos_comparativos:
    st.session_state["mostrar_comparacion"] = True
else:
    st.session_state["mostrar_comparacion"] = False

# 3. Mostrar gráficos solo si hay al menos 2 métodos ejecutados
if st.session_state.get("mostrar_comparacion", False):

    if len(ranking_disponibles) >= 2:
        nombres_localidades = df["comuna"].values
        comparacion_data = {"comuna": nombres_localidades}
        metodos_validos = []

        for metodo, ranking in ranking_disponibles.items():
            if ranking is not None and len(ranking) == len(nombres_localidades):
                comparacion_data[metodo] = ranking_normalizado_por_metodo[metodo]
                metodos_validos.append(metodo)
            else:
                st.warning(f"⚠️ Método '{metodo}' descartado: ranking inválido "
                           f"(esperado={len(nombres_localidades)}, recibido={len(ranking) if ranking is not None else 'None'})")

        if len(metodos_validos) >= 2:
            comparacion_df = pd.DataFrame(comparacion_data)
            comparacion_df = comparacion_df.sort_values(
                by=metodos_validos[0], ascending=False).reset_index(drop=True)

            # Mostrar gráfico
            if grafico == "Mapas":
                mostrar_mapas_pequenos(df, ranking_disponibles, comparacion_df, map_center, width_mapa, height_mapa)
            elif grafico == "Mapa de calor":
                mostrar_mapa_calor(comparacion_df, figsize_heatmap, fontsize_labels_heatmap)
            elif grafico == "Gráfico de rankings":
                mostrar_grafico_comparativo(
                    comparacion_df,
                    fontsize_title_barras,
                    fontsize_ticks_barras,
                    fontsize_labels_barras,
                    height_comp_barras,
                    width_comp_barras
                )
            elif grafico == "Dispersión bidireccional":
                mostrar_grafico_dispersion(comparacion_df, fontsize_title_d, fontsize_labels_d, height_disp, width_disp)
            elif grafico == "Mapa de calor jerárquico":
                mostrar_mapa_calor_jerarquico(comparacion_df, figsize_heatmap_j, fontsize_labels_map_j)
            elif grafico == "Coordenadas paralelas":
                 mostrar_lineas_comparativas(comparacion_df, height_lineas)
        else:
            st.warning(
                "No hay suficientes métodos válidos para comparar rankings. Verifica que todos tengan la misma cantidad de localidades.")
    else:
        st.warning("Debes ejecutar al menos dos métodos para comparar rankings.")


