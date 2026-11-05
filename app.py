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

Contesta las 50 preguntas para conocer tus cinco
rasgos de personalidad y descubrir a qué grupo
perteneces mediante un modelo de aprendizaje
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

    # Comprobar que existan las características necesarias
    if not all(col in df.columns for col in columnas):
        raise ValueError("El dataset no contiene las columnas Big Five necesarias.")

    # Eliminar registros incompletos
    df = df.dropna(subset=columnas).copy()

    # Escalar datos
    datos_escalados = scaler.transform(df[columnas])

    # Asignar clusters utilizando el modelo entrenado
    df["cluster"] = modelo.predict(datos_escalados)

    return modelo, scaler, df


try:

    kmeans, scaler, df_original = cargar_recursos()

except Exception as e:

    st.error(f"Error al cargar los recursos: {e}")

    st.stop()


# ==========================================
# 3. PREGUNTAS DEL TEST BIG FIVE
# ==========================================

# Formato:
# (pregunta, característica, puntuación_inversa)

preguntas = [

    # APERTURA (OP)

    ("Tengo un vocabulario amplio.", "op", False),
    ("Me cuesta comprender ideas abstractas.", "op", True),
    ("Tengo una imaginación muy activa.", "op", False),
    ("No me interesan las ideas abstractas.", "op", True),
    ("Tengo buenas ideas.", "op", False),
    ("No tengo una buena imaginación.", "op", True),
    ("Comprendo rápidamente las cosas.", "op", False),
    ("Utilizo palabras difíciles.", "op", False),
    ("Me gusta reflexionar sobre las cosas.", "op", False),
    ("Tengo muchas ideas.", "op", False),

    # RESPONSABILIDAD (CO)

    ("Siempre estoy preparado para mis actividades.", "co", False),
    ("Dejo mis pertenencias desordenadas.", "co", True),
    ("Presto atención a los detalles.", "co", False),
    ("Suelo hacer desorden.", "co", True),
    ("Realizo mis tareas inmediatamente.", "co", False),
    ("Olvido guardar las cosas en su lugar.", "co", True),
    ("Me gusta mantener todo en orden.", "co", False),
    ("Evito cumplir con mis responsabilidades.", "co", True),
    ("Sigo un horario para realizar mis actividades.", "co", False),
    ("Soy exigente con la calidad de mi trabajo.", "co", False),

    # EXTROVERSIÓN (EX)

    ("Soy el alma de las reuniones.", "ex", False),
    ("No hablo demasiado.", "ex", True),
    ("Me siento cómodo estando con otras personas.", "ex", False),
    ("Prefiero mantenerme en segundo plano.", "ex", True),
    ("Inicio conversaciones fácilmente.", "ex", False),
    ("Tengo poco que decir.", "ex", True),
    ("Hablo con muchas personas en las reuniones.", "ex", False),
    ("No me gusta llamar la atención.", "ex", True),
    ("No me importa ser el centro de atención.", "ex", False),
    ("Soy reservado cuando estoy con desconocidos.", "ex", True),

    # AMABILIDAD (AG)

    ("Me preocupo poco por los demás.", "ag", True),
    ("Me intereso por las personas.", "ag", False),
    ("Insulto a las personas.", "ag", True),
    ("Comprendo los sentimientos de los demás.", "ag", False),
    ("No me interesan los problemas de otras personas.", "ag", True),
    ("Tengo un corazón sensible.", "ag", False),
    ("Dedico tiempo para ayudar a los demás.", "ag", False),
    ("Soy indiferente ante las emociones de otros.", "ag", True),
    ("Me preocupo por cómo se sienten los demás.", "ag", False),
    ("Hago que las personas se sientan cómodas.", "ag", False),

    # NEUROTICISMO (NE)

    ("Generalmente estoy relajado.", "ne", True),
    ("Me estreso fácilmente.", "ne", False),
    ("Me preocupo por muchas cosas.", "ne", False),
    ("Rara vez me siento triste.", "ne", True),
    ("Me altero fácilmente.", "ne", False),
    ("Me molesto con facilidad.", "ne", False),
    ("Mi estado de ánimo cambia frecuentemente.", "ne", False),
    ("Tengo cambios frecuentes de humor.", "ne", False),
    ("Me irrito fácilmente.", "ne", False),
    ("Frecuentemente me siento triste.", "ne", False)

]


# ==========================================
# 4. OPCIONES DE RESPUESTA
# ==========================================

opciones = {
    "1 - Totalmente en desacuerdo": 1,
    "2 - En desacuerdo": 2,
    "3 - Ni de acuerdo ni en desacuerdo": 3,
    "4 - De acuerdo": 4,
    "5 - Totalmente de acuerdo": 5
}


# ==========================================
# 5. FORMULARIO DE LAS 50 PREGUNTAS
# ==========================================

st.header("📝 Cuestionario de personalidad")

st.write("""
Lee cada afirmación y selecciona la respuesta que
mejor describa tu forma habitual de ser.
""")

respuestas = {}

with st.form("formulario_bigfive"):

    for i, (pregunta, rasgo, inversa) in enumerate(preguntas):

        respuestas[i] = st.radio(
            f"{i + 1}. {pregunta}",
            options=list(opciones.keys()),
            index=None,
            key=f"pregunta_{i}"
        )

    enviar = st.form_submit_button(
        "Finalizar test y conocer mi personalidad",
        use_container_width=True
    )


# ==========================================
# 6. CALCULAR LOS CINCO RASGOS
# ==========================================

if enviar:

    # Verificar que se contestaron las 50 preguntas

    if any(respuesta is None for respuesta in respuestas.values()):

        st.warning("Debes contestar las 50 preguntas antes de continuar.")

        st.stop()

    resultados = {
        "op": [],
        "co": [],
        "ex": [],
        "ag": [],
        "ne": []
    }

    # Calcular puntuaciones

    for i, (pregunta, rasgo, inversa) in enumerate(preguntas):

        valor = opciones[respuestas[i]]

        # Invertir puntuación cuando corresponda
        if inversa:
            valor = 6 - valor

        resultados[rasgo].append(valor)

    # Promedio de cada rasgo convertido a escala 0-100
    # para mantener la escala de entrada utilizada
    # por la aplicación original.

    puntuaciones = {}

    for rasgo in resultados:

        promedio = np.mean(resultados[rasgo])

        puntuaciones[rasgo] = ((promedio - 1) / 4) * 100

    # ======================================
    # 7. MOSTRAR RESULTADOS
    # ======================================

    st.divider()

    st.header("📊 Tus resultados Big Five")

    col1, col2, col3, col4, col5 = st.columns(5)

    columnas = [col1, col2, col3, col4, col5]

    nombres = [
        ("OP", "Apertura"),
        ("CO", "Responsabilidad"),
        ("EX", "Extroversión"),
        ("AG", "Amabilidad"),
        ("NE", "Neuroticismo")
    ]

    for columna, (clave, nombre) in zip(columnas, nombres):

        with columna:

            st.metric(
                nombre,
                f"{puntuaciones[clave.lower()]:.1f}/100"
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
        f"🎉 ¡Tu personalidad pertenece al Cluster {mi_cluster}!"
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
    La descripción se genera utilizando los promedios
    de los cinco rasgos de las personas pertenecientes
    al cluster asignado por K-means.
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

    barras = ax1.bar(
        etiquetas,
        valores
    )

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

    # Referencias ilustrativas según el rasgo predominante.
    # No representan evaluaciones psicológicas reales
    # de estas personas.

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
    Estas personalidades son ejemplos ilustrativos
    asociados con el rasgo predominante. No se afirma
    que hayan realizado el test ni que pertenezcan
    científicamente al mismo cluster.
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

    valores_cluster = promedios.values.tolist()

    # Cerrar los polígonos

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
    De acuerdo con las respuestas proporcionadas en el
    test de personalidad Big Five, tus cinco características
    fueron procesadas y utilizadas como entrada para el
    modelo de aprendizaje no supervisado K-means.

    El modelo te asignó al Cluster {mi_cluster}.

    La característica predominante de este grupo es
    {nombres_rasgos[rasgo_principal]}.

    Mediante las gráficas de barras y radar puedes
    observar las características que predominan en tu
    grupo y comparar tu perfil personal con el promedio
    de los integrantes del cluster.

    Esto permite comprender cómo el aprendizaje no
    supervisado identifica grupos de datos con
    características similares sin utilizar categorías
    previamente etiquetadas.
    """)

    st.success("¡Gracias por completar el test Big Five!")
