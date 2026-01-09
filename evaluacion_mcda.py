# archivo: evaluacion_mcda.py

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import os

from prometheeII import promethee_matrix
from todim import todim_matrix
from mavt import mavt_matrix  # Asumo que estas funciones ya están implementadas
from red_neuronal import predecir_con_red_neuronal

def evaluar_metodo_mcda(metodo, data, criterios, pesos, q_vals=None, p_vals=None,
                         alpha=None, beta=None, lambd=None):
    """
    Evalúa los datos usando el método multicriterio seleccionado.

    Parámetros:
    - metodo: nombre del método ('PROMETHEE', 'TODIM', 'MAVT', 'Red Neuronal')
    - data: matriz de datos (numpy)
    - criterios: lista de criterios (columnas)
    - pesos: vector de pesos
    - q_vals, p_vals: arrays de umbrales para PROMETHEE
    - alpha, beta, lambd: parámetros para TODIM

    Devuelve:
    - ranking: vector de resultados
    """

    if metodo == "PROMETHEE":
        ranking = promethee_matrix(data, pesos, q_vals, p_vals)
        st.session_state.ranking_promethee = ranking

    elif metodo == "TODIM":
        ranking = todim_matrix(data, pesos, alpha=alpha, beta=beta, lambd=lambd)
        st.session_state.ranking_todim = ranking

    elif metodo == "MAVT":
        ranking = mavt_matrix(data, pesos)
        st.session_state.ranking_mavt = ranking

    elif metodo == "Red Neuronal":
        try:
            carpeta_actual = os.path.dirname(os.path.abspath(__file__))
            modelo_path = os.path.join(carpeta_actual, "modelo_cancer_pulmon.pkl")
            data_df = pd.DataFrame(data, columns=criterios)
            model, scaler = joblib.load(modelo_path)
            data_scaled = scaler.transform(data_df)
            ranking = model.predict(data_scaled)
            ranking = predecir_con_red_neuronal(data, criterios,'modelo_cancer_pulmon.pkl')
            st.session_state.ranking_red_neuronal = ranking

        except Exception as e:
            st.error(f"Error al cargar el modelo de red neuronal: {e}")
            st.stop()

    else:
        st.error("Método de evaluación no soportado.")
        st.stop()

    return (ranking,
            st.session_state.ranking_promethee,
            st.session_state.ranking_todim,
            st.session_state.ranking_mavt,
            st.session_state.ranking_red_neuronal)

