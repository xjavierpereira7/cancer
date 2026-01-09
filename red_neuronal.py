import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import joblib
# librería para IA
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import streamlit as st


# ======= MODULOS PARA IA =========================
def guardar_evaluacion_para_entrenamiento(df_comunas, ranking, archivo):
    """
    Guarda evaluaciones para entrenar red neuronal
    ADAPTADO PARA DATOS DE CÁNCER DE PULMÓN

    Args:
        df_comunas: DataFrame con datos de comunas (no glaciares)
        ranking: Array con rankings normalizados (0-1)
        archivo: Nombre del archivo CSV para guardar
    """
    # Hacer copia para no modificar original
    df_train = df_comunas.copy()

    # AGREGAR: Excluir columnas no numéricas específicas de comunas
    # Mantener columnas originales pero excluir geográficas y de nombre
    columnas_a_excluir = []

    # Buscar columnas comunes de datos de comunas
    posibles_columnas_no_numericas = [
        'comuna', 'nombre', 'nombre_comuna', 'municipio', 'ciudad',
        'lat', 'lon', 'latitude', 'longitude', 'geometry',
        'region', 'provincia', 'codigo', 'id'
    ]

    for col in posibles_columnas_no_numericas:
        if col in df_train.columns:
            columnas_a_excluir.append(col)

    # Si existe 'nombre_glaciar' (del código original), también excluir
    if 'nombre_glaciar' in df_train.columns:
        columnas_a_excluir.append('nombre_glaciar')

    # Si vamos a excluir columnas, crear nuevo DataFrame solo con columnas numéricas
    if columnas_a_excluir:
        # Obtener todas las columnas
        todas_columnas = df_train.columns.tolist()

        # Filtrar solo las que no están en exclusiones
        columnas_numericas = [col for col in todas_columnas if col not in columnas_a_excluir]

        # Crear nuevo DataFrame solo con columnas numéricas
        df_numerico = df_train[columnas_numericas].copy()

        # AGREGAR: Convertir columnas a numérico por si acaso
        for col in df_numerico.columns:
            df_numerico[col] = pd.to_numeric(df_numerico[col], errors='coerce')

        # Reemplazar DataFrame
        df_train = df_numerico

    # AGREGAR: Verificar que tenemos el mismo número de filas
    if len(df_train) != len(ranking):
        st.warning(f"⚠️ Advertencia: Número de filas no coincide. Datos: {len(df_train)}, Ranking: {len(ranking)}")
        # Ajustar al mínimo común
        min_len = min(len(df_train), len(ranking))
        df_train = df_train.iloc[:min_len]
        ranking = ranking[:min_len]

    # AGREGAR: Verificar que no haya NaN en los datos numéricos
    if df_train.isnull().any().any():
        st.warning("⚠️ Advertencia: Hay valores NaN en los datos. Se llenarán con la media.")
        df_train = df_train.fillna(df_train.mean())

    # Agregar el ranking como columna 'phi'
    df_train["phi"] = ranking

    # Guardar o combinar con archivo existente
    try:
        # AGREGAR: Especificar separador para CSV de comunas
        df_existente = pd.read_csv(archivo, sep=';')
        df_combinado = pd.concat([df_existente, df_train], ignore_index=True)
        #st.info(f"✅ Se añadieron {len(df_train)} registros al archivo existente")
    except FileNotFoundError:
        df_combinado = df_train
        #st.info(f"✅ Archivo nuevo creado con {len(df_train)} registros")
    except pd.errors.EmptyDataError:
        df_combinado = df_train
        #st.info(f"✅ Archivo nuevo creado con {len(df_train)} registros")

    # AGREGAR: Guardar con punto y coma para consistencia
    df_combinado.to_csv(archivo, index=False, sep=';')

    return df_combinado


def reentrenar_red_neuronal(df, modelo_guardado):
    """
    Reentrena la red neuronal con los datos actuales
    ADAPTADO PARA DATOS DE CÁNCER DE PULMÓN

    Args:
        df: DataFrame que contiene datos y columna 'phi' (ranking)
        modelo_guardado: Ruta donde guardar el modelo entrenado
    """

    # AGREGAR: Verificar que existe columna 'phi'
    if 'phi' not in df.columns:
        st.error("❌ Error: No se encuentra la columna 'phi' en los datos")
        return None

    # AGREGAR: Identificar columnas a excluir para entrenamiento
    # Excluir columnas no numéricas o de identificación
    columnas_a_excluir = ['phi']  # Siempre excluir la variable objetivo

    # Posibles columnas de identificación (ajustar según tus datos)
    columnas_identificacion = [
        'comuna', 'nombre', 'nombre_comuna',
        'lat', 'lon', 'latitude', 'longitude',
        'nombre_glaciar', 'geometry'  # Por si viene del código original
    ]

    for col in columnas_identificacion:
        if col in df.columns:
            columnas_a_excluir.append(col)

    # AGREGAR: Preparar características (X) y variable objetivo (y)
    # Solo usar columnas numéricas que no están excluidas
    columnas_X = [col for col in df.columns if col not in columnas_a_excluir]

    if not columnas_X:
        st.error("❌ Error: No hay columnas disponibles para entrenamiento")
        return None

    X = df[columnas_X]
    y = df['phi']

    # AGREGAR: Manejar valores NaN
    if X.isnull().any().any() or y.isnull().any():
        st.warning("⚠️ Advertencia: Se encontraron valores NaN. Limpiando datos...")
        # Crear máscara de filas sin NaN
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]

    # AGREGAR: Verificar que tenemos suficientes datos
    if len(X) < 10:
        st.error(f"❌ Error: Insuficientes datos para entrenar. Solo {len(X)} muestras válidas.")
        return None

    # Dividir datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    # Escalar datos
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # AGREGAR: Configurar modelo según número de características
    n_caracteristicas = X_train.shape[1]

    # Ajustar arquitectura de red según complejidad de datos
    if n_caracteristicas <= 5:
        capas_ocultas = (10, 5)  # Arquitectura simple
    elif n_caracteristicas <= 10:
        capas_ocultas = (20, 10)  # Arquitectura media
    else:
        capas_ocultas = (30, 15, 8)  # Arquitectura compleja

    # AGREGAR: Configurar modelo con parámetros para datos de salud
    model = MLPRegressor(
        hidden_layer_sizes=capas_ocultas,
        max_iter=2000,  # Aumentar iteraciones para convergencia
        random_state=42,
        early_stopping=True,  # Detener si no mejora
        validation_fraction=0.1,  # 10% para validación
        n_iter_no_change=50,  # Detener después de 50 épocas sin mejora
        alpha=0.001,  # Regularización L2
        learning_rate_init=0.001,
        verbose=False  # No mostrar progreso en consola
    )

    # Entrenar modelo
    with st.spinner(f"🔧 Entrenando red neuronal con {len(X_train)} muestras..."):
        model.fit(X_train_scaled, y_train)

    # AGREGAR: Evaluar modelo
    score_entrenamiento = model.score(X_train_scaled, y_train)
    score_prueba = model.score(X_test_scaled, y_test)

    # Guardar modelo, escalador y metadatos
    datos_guardar = {
        'model': model,
        'scaler': scaler,
        'features': columnas_X,  # Guardar nombres de características
        'train_score': score_entrenamiento,
        'test_score': score_prueba,
        'n_samples': len(X)
    }

    # joblib.dump(datos_guardar, modelo_guardado)
    joblib.dump((model, scaler), modelo_guardado)

    # AGREGAR: Mostrar resultados del entrenamiento
    st.success(f"""
    ✅ Red neuronal reentrenada y guardada correctamente.

    **Estadísticas del modelo:**
    - Muestras de entrenamiento: {len(X_train)}
    - Muestras de prueba: {len(X_test)}
    - R² en entrenamiento: {score_entrenamiento:.4f}
    - R² en prueba: {score_prueba:.4f}
    - Características usadas: {len(columnas_X)}
    - Modelo guardado en: `{modelo_guardado}`
    """)

    # Mostrar características usadas (máximo 10)
    if len(columnas_X) <= 10:
        st.write("**Características utilizadas:**", ", ".join(columnas_X))
    else:
        st.write(f"**Primeras 10 características:** {', '.join(columnas_X[:10])}...")

    return model, scaler, columnas_X


def predecir_con_red_neuronal(data, criterios, modelo_path):
    if not os.path.exists(modelo_path):
        st.error("❌ Modelo no encontrado. Entrena la red neuronal primero.")
        return None

    try:
        # CARGAR modelo (compatible con ambos formatos)
        modelo_data = joblib.load(modelo_path)

        # DETERMINAR formato
        # if isinstance(modelo_data, dict):
        #     # Nuevo formato: diccionario
        #     model = modelo_data['model']
        #     scaler = modelo_data['scaler']
        # elif isinstance(modelo_data, tuple) and len(modelo_data) == 2:
        #     # Viejo formato: tupla (model, scaler)
        #     model, scaler = modelo_data
        # else:
        #     st.error(f"❌ Formato de modelo desconocido: {type(modelo_data)}")
        #     return None

        model, scaler = joblib.load(modelo_path)

        # CONVERTIR datos
        data_df = pd.DataFrame(data, columns=criterios)

        # ESCALAR
        data_scaled = scaler.transform(data_df)

        # PREDECIR
        ranking = model.predict(data_scaled)

        return ranking

    except Exception as e:
        st.error(f"❌ Error en predicción: {str(e)}")
        return None
# ======== FUNCIÓN ADICIONAL ÚTIL =========
def verificar_datos_entrenamiento(archivo_csv="ranking_entrenamiento_cancer.csv"):
    """
    Verifica los datos de entrenamiento guardados
    Útil para diagnóstico
    """
    try:
        df = pd.read_csv(archivo_csv, sep=';')
        st.write(f"📊 **Datos de entrenamiento guardados:**")
        st.write(f"- Registros: {len(df)}")
        st.write(f"- Columnas: {len(df.columns)}")
        st.write(f"- Columnas disponibles: {', '.join(df.columns.tolist())}")

        if 'phi' in df.columns:
            st.write(f"- Rango de 'phi': {df['phi'].min():.3f} a {df['phi'].max():.3f}")

        return df
    except FileNotFoundError:
        st.info("ℹ️ No hay datos de entrenamiento guardados aún")
        return None
    except Exception as e:
        st.error(f"❌ Error al leer datos: {str(e)}")
        return None

# ======== FIN MODULOS IA =======================