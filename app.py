import dash
from dash import html, dcc
import plotly.express as px
import pandas as pd

# Datos de ejemplo
df = pd.DataFrame({
    "Ciudad": ["Madrid", "París", "Londres", "Berlín"],
    "Precio Promedio (USD)": [120, 150, 180, 110]
})

fig = px.bar(df, x="Ciudad", y="Precio Promedio (USD)", title="Precio promedio de Airbnb por ciudad")

# App Dash
app = dash.Dash(__name__)
server = app.server  # <- Importante para que Render lo ejecute

app.layout = html.Div([
    html.H1("Dashboard de Ejemplo"),
    dcc.Graph(figure=fig)
])

if __name__ == "__main__":
    app.run_server(debug=True)
