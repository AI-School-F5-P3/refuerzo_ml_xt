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

if option == "Modelo ML":
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
        model = load_model_ml('../models/model_svc_7f.joblib')
        pred = model.predict(input_data)[0]
        st.success(f"{pred}")
        query = add_data(data, int(pred))
        if query == 1:
            st.success("Datos guardados correctamente.")
        else:
            st.text(query)
