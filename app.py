import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

# Cargar modelo
@st.cache_resource
def load_model():
    return joblib.load("modelo_regresion_precio_final.joblib")

modelo = load_model()

# Título principal
st.title("Dashboard de Predicción de Precio por Noche - Airbnb")
st.markdown("Estima el precio de una propiedad según sus características y visualiza estadísticas generales.")

# Gráfico con Plotly
df = pd.DataFrame({
    "Ciudad": ["Madrid", "París", "Londres", "Berlín"],
    "Precio Promedio (USD)": [120, 150, 180, 110]
})

fig = px.bar(df, x="Ciudad", y="Precio Promedio (USD)", title="Precio promedio de Airbnb por ciudad")
st.plotly_chart(fig)

# Formulario de predicción
st.subheader("Predicción personalizada")
pais = st.selectbox("País", ['España', 'México', 'Colombia', 'Estados Unidos'])
habitaciones = st.slider("Habitaciones", 1, 10, 2)
baños = st.slider("Baños", 1, 5, 1)
camas = st.slider("Camas", 1, 10, 2)
wifi = st.selectbox("¿Tiene Wifi?", ['Sí', 'No'])

wifi_bin = 1 if wifi == 'Sí' else 0

input_df = pd.DataFrame([{
    'pais': pais,
    'habitaciones': habitaciones,
    'baños': baños,
    'camas': camas,
    'wifi': wifi_bin,
}])

if st.button("Predecir"):
    try:
        pred = modelo.predict(input_df)[0]
        st.success(f"Precio estimado por noche: ${pred:,.2f}")
    except Exception as e:
        st.error("Error al predecir")
        st.exception(e)
