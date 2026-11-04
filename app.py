import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Configuración de la página
st.set_page_config(page_title="Big Five - Clasificador de Personalidad", layout="wide")

# Título de la aplicación
st.title("🧠 Clasificador de Personalidad: Big Five & K-Means")
st.write("Ingresa tus puntajes para conocer a qué cluster perteneces, qué famosos se parecen a ti y visualizar tu perfil comparativo.")

# Cargar el modelo, el escalador y el dataset original
@st.cache_resource
def cargar_recursos():
    model = joblib.load('kmeans_bigFive_model.joblib')
    scaler = joblib.load('scaler_bigFive.joblib')
    # Cargamos el dataset con la columna de clusters ya calculada para buscar famosos
    url_data = "https://raw.githubusercontent.com/gerrymay/fcaData/refs/heads/master/oceanbig5.csv"
    df_original = pd.read_csv(url_data)

    # Entrenar de nuevo la asignación de clusters para tener la misma correspondencia en la app
    # usando las features del dataset original escaladas
    features = df_original[['op', 'co', 'ex', 'ag', 'ne']]
    features_scaled = scaler.transform(features)
    df_original['cluster'] = model.predict(features_scaled)
    return model, scaler, df_original

try:
    kmeans_cargado, scaler_cargado, df_famosos = cargar_recursos()
except Exception as e:
    st.error(f"Error al cargar los modelos o datos: {e}")
    st.stop()

# Layout de dos columnas: Inputs y Resultados
col1, col2 = st.columns([1, 2])

with col1:
    st.header("Tus Resultados")
    op = st.slider("Apertura (OP)", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
    co = st.slider("Responsabilidad (CO)", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
    ex = st.slider("Extroversión (EX)", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
    ag = st.slider("Amabilidad (AG)", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
    ne = st.slider("Neuroticismo (NE)", min_value=0.0, max_value=100.0, value=50.0, step=0.1)

# Procesar la predicción al presionar el botón
if st.button("Clasificarme"):
    # Preparar el dataframe de entrada
    mis_datos = pd.DataFrame([{
        'op': op,
        'co': co,
        'ex': ex,
        'ag': ag,
        'ne': ne
    }])

    # Escalar datos e identificar cluster
    mis_datos_escalados = scaler_cargado.transform(mis_datos)
    mi_cluster = kmeans_cargado.predict(mis_datos_escalados)[0]

    with col2:
        st.success(f"### 🎉 ¡Has sido asignado al Cluster: {mi_cluster}!")

        # Mostrar el top 5 de famosos del mismo cluster
        st.subheader("👥 Famosos en tu mismo cluster:")
        famosos_cluster = df_famosos[df_famosos['cluster'] == mi_cluster]
        st.dataframe(famosos_cluster[['usuario', 'op', 'co', 'ex', 'ag', 'ne']].head(5), use_container_width=True)

        # Generación del gráfico de radar comparativo
        st.subheader("📊 Comparativa de Perfil de Personalidad")

        categorias = ['op', 'co', 'ex', 'ag', 'ne']
        num_vars = len(categorias)

        # Promedio del cluster asignado
        valores_cluster = famosos_cluster[categorias].mean().tolist()
        valores_cluster += valores_cluster[:1]

        # Valores del usuario
        valores_usuario = [op, co, ex, ag, ne]
        valores_usuario += valores_usuario[:1]

        # Calcular ángulos
        angulos = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angulos += angulos[:1]

        # Dibujar gráfico de radar
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        plt.xticks(angulos[:-1], categorias, color='grey', size=12)

        # Polígono de la media del cluster
        ax.plot(angulos, valores_cluster, color='#1aafa5', linewidth=2, linestyle='solid', label=f'Promedio Cluster {mi_cluster}')
        ax.fill(angulos, valores_cluster, color='#1aafa5', alpha=0.25)

        # Polígono del usuario
        ax.plot(angulos, valores_usuario, color='#e74c3c', linewidth=2, linestyle='solid', label='Mis Resultados')
        ax.fill(angulos, valores_usuario, color='#e74c3c', alpha=0.25)

        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        st.pyplot(fig)
