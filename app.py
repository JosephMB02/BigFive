import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ==========================================
# 1. CONFIGURACIÓN DE LA APLICACIÓN
# ==========================================

st.set_page_config(
    page_title="Big Five - Test de Personalidad",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Test de Personalidad Big Five")

st.markdown("""
Bienvenido a la aplicación Big Five.

Contesta las siguientes 5 preguntas para conocer tus
cinco características de personalidad y descubrir a
qué grupo perteneces mediante un modelo de aprendizaje
no supervisado K-Means.
""")

st.divider()


# ==========================================
# 2. CARGAR MODELO, SCALER Y DATASET
# ==========================================

@st.cache_resource
def cargar_recursos():

    modelo = joblib.load("kmeans_bigFive_model.joblib")

    scaler = joblib.load("scaler_bigFive.joblib")

    url = "https://raw.githubusercontent.com/gerrymay/fcaData/refs/heads/master/oceanbig5.csv"

    df = pd.read_csv(url)

    columnas = ["op", "co", "ex", "ag", "ne"]

    if not all(col in df.columns for col in columnas):
        raise ValueError("El dataset no contiene las columnas necesarias.")

    df = df.dropna(subset=columnas).copy()

    datos_escalados = scaler.transform(df[columnas])

    df["cluster"] = modelo.predict(datos_escalados)

    return modelo, scaler, df


try:

    kmeans, scaler, df_original = cargar_recursos()

except Exception as e:

    st.error(f"Error al cargar los recursos: {e}")

    st.stop()


# ==========================================
# 3. TEST BIG FIVE: SOLAMENTE 5 PREGUNTAS
# ==========================================

st.header("📝 Cuestionario de personalidad")

st.write("""
Responde las siguientes preguntas seleccionando
la opción que mejor represente tu personalidad.
""")

preguntas = {

    "op": {
        "nombre": "Apertura (OP)",
        "pregunta": "¿Qué tanto te gusta aprender cosas nuevas y experimentar nuevas experiencias?"
    },

    "co": {
        "nombre": "Responsabilidad (CO)",
        "pregunta": "¿Qué tan organizado y responsable eres al realizar tus actividades?"
    },

    "ex": {
        "nombre": "Extroversión (EX)",
        "pregunta": "¿Qué tanto disfrutas convivir y conversar con otras personas?"
    },

    "ag": {
        "nombre": "Amabilidad (AG)",
        "pregunta": "¿Qué tanto te preocupas por ayudar y comprender a los demás?"
    },

    "ne": {
        "nombre": "Neuroticismo (NE)",
        "pregunta": "¿Qué tan fácilmente te estresas o preocupas ante situaciones difíciles?"
    }

}


# ==========================================
# 4. OPCIONES DE RESPUESTA
# ==========================================

opciones = {

    "1 - Muy poco": 1,
    "2 - Poco": 2,
    "3 - Regular": 3,
    "4 - Mucho": 4,
    "5 - Demasiado": 5

}


# ==========================================
# 5. FORMULARIO DE LAS 5 PREGUNTAS
# ==========================================

respuestas = {}

with st.form("formulario_bigfive"):

    for clave, datos in preguntas.items():

        st.subheader(datos["nombre"])

        respuestas[clave] = st.radio(
            datos["pregunta"],
            options=list(opciones.keys()),
            index=None,
            key=clave
        )

    enviar = st.form_submit_button(
        "Finalizar test y conocer mi personalidad",
        use_container_width=True
    )


# ==========================================
# 6. CALCULAR LOS CINCO RESULTADOS
# ==========================================

if enviar:

    if any(valor is None for valor in respuestas.values()):

        st.warning("Debes contestar las 5 preguntas.")

        st.stop()

    puntuaciones = {}

    for clave, respuesta in respuestas.items():

        valor = opciones[respuesta]

        # Convertir de escala 1-5 a escala 0-100
        puntuaciones[clave] = ((valor - 1) / 4) * 100


    # ======================================
    # 7. MOSTRAR RESULTADOS
    # ======================================

    st.divider()

    st.header("📊 Tus resultados Big Five")

    col1, col2, col3, col4, col5 = st.columns(5)

    columnas = [col1, col2, col3, col4, col5]

    nombres = [
        ("op", "Apertura"),
        ("co", "Responsabilidad"),
        ("ex", "Extroversión"),
        ("ag", "Amabilidad"),
        ("ne", "Neuroticismo")
    ]

    for columna, (clave, nombre) in zip(columnas, nombres):

        with columna:

            st.metric(
                nombre,
                f"{puntuaciones[clave]:.1f}/100"
            )


    # ======================================
    # 8. ASIGNAR CLUSTER CON K-MEANS
    # ======================================

    mis_datos = pd.DataFrame([{

        "op": puntuaciones["op"],
        "co": puntuaciones["co"],
        "ex": puntuaciones["ex"],
        "ag": puntuaciones["ag"],
        "ne": puntuaciones["ne"]

    }])

    datos_escalados = scaler.transform(mis_datos)

    mi_cluster = int(kmeans.predict(datos_escalados)[0])

    st.divider()

    st.success(
        f"🎉 ¡Has sido asignado al Cluster {mi_cluster}!"
    )


    # ======================================
    # 9. CALCULAR PROMEDIOS DEL CLUSTER
    # ======================================

    caracteristicas = ["op", "co", "ex", "ag", "ne"]

    famosos_cluster = df_original[
        df_original["cluster"] == mi_cluster
    ]

    if famosos_cluster.empty:

        st.warning("No se encontraron registros para este cluster.")

        st.stop()

    promedios = famosos_cluster[caracteristicas].mean()


    # ======================================
    # 10. DESCRIPCIÓN DEL CLUSTER
    # ======================================

    descripciones = {

        "op": "Creatividad y curiosidad: personas interesadas en aprender, explorar nuevas ideas y experimentar.",

        "co": "Organización y disciplina: personas responsables, ordenadas y comprometidas con sus objetivos.",

        "ex": "Sociabilidad y energía: personas comunicativas, expresivas y que disfrutan convivir con otros.",

        "ag": "Amabilidad y cooperación: personas empáticas, comprensivas y dispuestas a colaborar.",

        "ne": "Sensibilidad emocional: personas que pueden experimentar emociones intensas y mayor preocupación ante situaciones difíciles."

    }

    nombres_rasgos = {

        "op": "Apertura",
        "co": "Responsabilidad",
        "ex": "Extroversión",
        "ag": "Amabilidad",
        "ne": "Neuroticismo"

    }

    rasgos_ordenados = promedios.sort_values(ascending=False)

    rasgo_principal = rasgos_ordenados.index[0]

    segundo_rasgo = rasgos_ordenados.index[1]

    st.header("🧠 Descripción de tu personalidad")

    st.subheader(
        f"Cluster {mi_cluster}: {nombres_rasgos[rasgo_principal]} predominante"
    )

    st.write(descripciones[rasgo_principal])

    st.write(
        f"Tu segundo rasgo predominante es "
        f"{nombres_rasgos[segundo_rasgo]}."
    )

    st.info("""
    Esta descripción se genera utilizando los promedios
    de las cinco características de las personas
    pertenecientes al cluster asignado por K-means.
    """)


    # ======================================
    # 11. GRÁFICA DE BARRAS
    # ======================================

    st.divider()

    st.header("📊 Características predominantes de tu cluster")

    fig1, ax1 = plt.subplots(figsize=(9, 5))

    valores = promedios.values

    etiquetas = [
        nombres_rasgos[c] for c in caracteristicas
    ]

    barras = ax1.bar(etiquetas, valores)

    ax1.set_title(
        f"Promedio de personalidad - Cluster {mi_cluster}"
    )

    ax1.set_ylabel("Puntuación promedio")

    ax1.set_xlabel("Características Big Five")

    ax1.set_ylim(0, 100)

    ax1.bar_label(
        barras,
        fmt="%.1f",
        padding=3
    )

    ax1.grid(axis="y", alpha=0.3)

    st.pyplot(fig1)


    # ======================================
    # 12. TOP 5 FAMOSOS RELACIONADOS
    # ======================================

    st.divider()

    st.header("🌟 Top 5 famosos relacionados con tu perfil")

    famosos = {

        "op": [
            "Albert Einstein",
            "Leonardo da Vinci",
            "Frida Kahlo",
            "Salvador Dalí",
            "Marie Curie"
        ],

        "co": [
            "Marie Curie",
            "Benjamin Franklin",
            "Serena Williams",
            "Angela Merkel",
            "Warren Buffett"
        ],

        "ex": [
            "Dwayne Johnson",
            "Oprah Winfrey",
            "Will Smith",
            "Robin Williams",
            "Ellen DeGeneres"
        ],

        "ag": [
            "Mahatma Gandhi",
            "Nelson Mandela",
            "Madre Teresa",
            "Malala Yousafzai",
            "Fred Rogers"
        ],

        "ne": [
            "Vincent van Gogh",
            "Edgar Allan Poe",
            "Frida Kahlo",
            "Sylvia Plath",
            "Virginia Woolf"
        ]

    }

    lista_famosos = famosos[rasgo_principal]

    for i, famoso in enumerate(lista_famosos, start=1):

        st.write(f"**{i}. {famoso}**")

    st.caption("""
    Personalidades ilustrativas relacionadas con el rasgo
    predominante. No se afirma que hayan realizado el test
    ni que pertenezcan científicamente al mismo cluster.
    """)


    # ======================================
    # 13. GRÁFICA DE RADAR
    # ======================================

    st.divider()

    st.header("🎯 Comparación de tu perfil de personalidad")

    categorias = ["OP", "CO", "EX", "AG", "NE"]

    valores_usuario = [

        puntuaciones["op"],
        puntuaciones["co"],
        puntuaciones["ex"],
        puntuaciones["ag"],
        puntuaciones["ne"]

    ]

    valores_cluster = promedios[caracteristicas].values.tolist()

    valores_usuario += valores_usuario[:1]

    valores_cluster += valores_cluster[:1]

    angulos = np.linspace(
        0,
        2 * np.pi,
        len(categorias),
        endpoint=False
    ).tolist()

    angulos += angulos[:1]

    fig2, ax2 = plt.subplots(
        figsize=(7, 7),
        subplot_kw=dict(polar=True)
    )

    ax2.plot(
        angulos,
        valores_usuario,
        linewidth=2,
        label="Mis resultados"
    )

    ax2.fill(
        angulos,
        valores_usuario,
        alpha=0.20
    )

    ax2.plot(
        angulos,
        valores_cluster,
        linewidth=2,
        label=f"Promedio Cluster {mi_cluster}"
    )

    ax2.fill(
        angulos,
        valores_cluster,
        alpha=0.20
    )

    ax2.set_xticks(angulos[:-1])

    ax2.set_xticklabels(categorias)

    ax2.set_ylim(0, 100)

    ax2.set_title("Comparación de características Big Five")

    ax2.legend(
        loc="upper right",
        bbox_to_anchor=(1.35, 1.15)
    )

    st.pyplot(fig2)


    # ======================================
    # 14. CONCLUSIÓN
    # ======================================

    st.divider()

    st.header("📌 Conclusión")

    st.write(f"""
    De acuerdo con las respuestas proporcionadas en
    el test Big Five, se calcularon cinco características
    de personalidad: Apertura, Responsabilidad,
    Extroversión, Amabilidad y Neuroticismo.

    Los resultados fueron procesados mediante el modelo
    de aprendizaje no supervisado K-means.

    El modelo te asignó al Cluster {mi_cluster}.

    La característica predominante de este grupo es
    {nombres_rasgos[rasgo_principal]}.

    Las gráficas permiten observar las características
    predominantes del cluster y comparar tu perfil
    personal con el promedio de sus integrantes.

    Esto demuestra cómo el aprendizaje no supervisado
    permite agrupar datos según sus características
    sin utilizar categorías previamente etiquetadas.
    """)

    st.success("¡Gracias por completar el test Big Five!")
