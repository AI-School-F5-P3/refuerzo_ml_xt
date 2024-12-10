import streamlit as st
import joblib
import pandas as pd
from db import add_data

def load_model_ml(path):
    with open(path, 'rb') as file:
        model = joblib.load(file)
    return model

st.title("Proyecto de Refuerzo")

option = st.sidebar.selectbox(
    "Selecciona un modelo",
    ("Inicio", "Modelo ML", "Modelo DL")
)

def project_description():
    st.title("🛰️ Segmentación de Clientes para Telecomunicaciones")
    
    st.markdown("""
    ### Objetivo del Proyecto
    Desarrollar un sistema de clasificación avanzado para personalizar ofertas 
    en el sector de telecomunicaciones utilizando técnicas de machine learning.

    ### Modelos Utilizados
    - **Máquina de Vectores de Soporte (SVC)**
    - **Red Neuronal**

    ### Características Clave
    - Análisis predictivo de comportamiento de clientes
    - Personalización de ofertas basada en segmentación
    - Mejora de estrategias de retención y marketing
    """)

    # Visualización conceptual
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Proceso de Clasificación")
        st.image("https://via.placeholder.com/400x300.png?text=Diagrama+de+Flujo+de+Clasificación")
    
    with col2:
        st.subheader("Impacto Esperado")
        st.metric(label="Precisión Estimada", value="44%")
        # st.metric(label="Reducción de Churn", value="20%")

if option == 'Inicio':
    project_description()
elif option == "Modelo ML":
    marital_map = {
        "Soltero": 0,
        "Casado": 1
    }

    st.header("Modelo ML - SVM")
    col1, col2 = st.columns(2)
    marital = col1.selectbox("Estado civil:", tuple(marital_map.keys()))
    ed = col1.selectbox("Nivel de educación:", [1, 2, 3, 4, 5])
    reside = col1. slider("Habitantes por casa:", 1, 10)
    tenure = col2.number_input("Tiempo trabajando (meses):", 0, 120)
    income = col2.number_input("Ingresos:", 0, 2000)
    address = col2.slider("Dirección:", 0, 55)
    employ = col2.slider("Tipo de trabajo:", 0, 47)
    
    data = {
            'ed': [ed],
            'tenure': [tenure],
            'employ': [employ],
            'reside': [reside],
            'income': [income],
            'marital': [marital_map[marital]],
            'address': [address]
        }
    if st.button("Predecir"):
        input_data = pd.DataFrame(data)
        model = load_model_ml('models/model_svc_7f.joblib')
        pred = model.predict(input_data)[0]
        st.success(f"{pred}")
        query = add_data(data, int(pred))
        if query == 1:
            st.success("Datos guardados correctamente.")
        else:
            st.text(query)
