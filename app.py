import dash
from dash import html, dcc
import plotly.express as px
import pandas as pd
import joblib
import pandas as pd
import streamlit as st

@st.cache_resource
def load_model():
    return joblib.load("modelo_regresion_precio_final.joblib")

modelo = load_model()

# Datos de ejemplo
df = pd.DataFrame({
    "Ciudad": ["Madrid", "París", "Londres", "Berlín"],
    "Precio Promedio (USD)": [120, 150, 180, 110]
})

fig = px.bar(df, x="Ciudad", y="Precio Promedio (USD)", title="Precio promedio de Airbnb por ciudad")

st.title("Dashboard de Predicción de Precio por Noche - Airbnb")

st.markdown("Estima el precio de una propiedad según sus características.")

# Inputs del usuario
pais = st.selectbox("País", ['España', 'México', 'Colombia', 'Estados Unidos'])
habitaciones = st.slider("Habitaciones", 1, 10, 2)
baños = st.slider("Baños", 1, 5, 1)
camas = st.slider("Camas", 1, 10, 2)
wifi = st.selectbox("¿Tiene Wifi?", ['Sí', 'No'])

# Convertir a formato correcto para el modelo
wifi_bin = 1 if wifi == 'Sí' else 0

input_df = pd.DataFrame([{
    'pais': pais,
    'habitaciones': habitaciones,
    'baños': baños,
    'camas': camas,
    'wifi': wifi_bin,
}])

# Mostrar predicción
if st.button("Predecir"):
    try:
        pred = modelo.predict(input_df)[0]
        st.success(f"Precio estimado por noche: ${pred:,.2f}")
    except Exception as e:
        st.error("Ocurrió un error al predecir.")
        st.exception(e)

# App Dash
app = dash.Dash(__name__)
server = app.server  # <- Importante para que Render lo ejecute

app.layout = html.Div([
    html.H1("Dashboard de Ejemplo"),
    dcc.Graph(figure=fig)
])

if __name__ == "__main__":
    app.run_server(debug=True)
