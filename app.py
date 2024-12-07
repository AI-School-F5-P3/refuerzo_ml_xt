import streamlit as st

st.title("Proyecto de Refuerzo")

option = st.sidebar.selectbox(
    "Selecciona un modelo",
    ("Inicio", "Modelo ML", "Modelo DL")
)

if option == "Modelo ML":
    st.header("Modelo ML - SVM")
    col1, col2 = st.columns(2)
    
    age = col1.number_input("Edad:", 0, 120)
    ed = col1.selectbox("Nivel de educación:", [1, 2, 3, 4, 5])
    region = col1. selectbox("Región:", [1, 2, 3])
    tenure = col2.number_input("Tiempo trabajando (meses):", 0, 120)
    income = col2.number_input("Ingresos:", 0, 2000)
    address = col2.slider("Dirección:", 0, 55)
    employ = col2.slider("Tipo de trabajo:", 0, 47)

    btn = st.button("Predecir")

