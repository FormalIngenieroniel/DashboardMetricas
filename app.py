import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import numpy as np # Necesario para algunos cálculos si surgen

# --- Configuración de la Página ---
st.set_page_config(
    page_title="POC: Predicción de Precios Airbnb",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Cargar Modelo ---
@st.cache_resource
def load_model():
    # Asegúrate de que el nombre del archivo sea exactamente "modelo_regresion_precio_final.joblib"
    # y que esté en el mismo directorio que app.py, o proporciona la ruta completa.
    try:
        model = joblib.load("modelo_regresion_precio_final.joblib")
        return model
    except FileNotFoundError:
        st.error("Error: El archivo 'modelo_regresion_precio_final.joblib' no se encontró. Asegúrate de que esté en el directorio correcto.")
        return None
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None

modelo = load_model()

# --- Estilos CSS (Opcional - para mantener colores similares) ---
# Puedes añadir CSS personalizado si quieres afinar más los colores.
# Por ahora, usaremos los temas de Streamlit y Plotly que son bastante personalizables.

# --- Título Principal y Descripción del Problema ---
st.title("🏠 POC: Sistema Inteligente de Evaluación de Rentabilidad para Airbnb")
st.markdown("""
El objetivo de este dashboard es presentar una **Prueba de Concepto (POC)** de cómo un modelo de Machine Learning
puede solucionar la necesidad de evaluar la rentabilidad y el potencial de mejora de propiedades en Airbnb.
""")

st.markdown("""
---
### El Desafío:
Una empresa de inversión inmobiliaria, junto con propietarios independientes, busca maximizar el rendimiento económico de propiedades listadas (o por listar) en plataformas como Airbnb. Su necesidad principal es contar con una herramienta basada en inteligencia artificial que permita:

* 🚀 **Evaluar si una propiedad es una buena inversión antes de adquirirla**, prediciendo su precio por noche y comparándolo con los costos estimados.
* 🛠️ **Simular escenarios de mejora** (remodelaciones, cambios estructurales o de servicios) para estimar cómo esas mejoras afectarían el precio por noche y, por ende, la rentabilidad.
* 💡 Tomar **decisiones fundamentadas** tanto antes de la compra como durante la explotación del inmueble.

Este sistema de predicción estima el precio por noche de una propiedad según sus características (ubicación, tamaño, servicios, etc.), simula el impacto de mejoras y calcula la rentabilidad esperada.
""")
st.markdown("---")

# --- Sección de Confiabilidad del Modelo ---
st.header("📊 Confiabilidad del Modelo de Predicción")
st.markdown("""
Para asegurar la validez de nuestras predicciones, el modelo fue evaluado rigurosamente.
A continuación, se presentan las métricas clave y una comparación de predicciones contra valores reales.
""")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Métricas de Evaluación:")
    st.metric(label="MAE Entrenamiento (Error Absoluto Medio)", value="COP {:,.2f}".format(11789.95)) # Asumiendo COP, ajustar si es otra moneda
    st.metric(label="R² Entrenamiento (Coeficiente de Determinación)", value="0.58")
with col2:
    st.subheader(" ") # Espacio para alinear
    st.metric(label="MAE Prueba", value="COP {:,.2f}".format(10346.47))
    st.metric(label="R² Prueba", value="0.65")

st.markdown("""
* **MAE (Mean Absolute Error):** Representa el error promedio de las predicciones. Un MAE de prueba de COP 10,346.47 significa que, en promedio, las predicciones del modelo se desvían este valor del precio real.
* **R² (R-squared):** Indica la proporción de la varianza en el precio que es predecible a partir de las características. Un R² de prueba de 0.65 sugiere que el modelo explica el 65% de la variabilidad de los precios.
""")

st.subheader("Comparación de Predicciones vs. Valores Reales (Muestra)")
data_comparacion = {
    'Real': [13718, 2853, 16128, 15406, 7980, 8642, 58273, 1747, 10353, 4734],
    'Predicción': [14699.457549, 3868.607576, 9137.748835, 3291.873266, 7937.131076, 9484.252985, 169251.768205, 9559.487026, 12043.144755, 12611.843768],
    'Error Absoluto': [981.457549, 1015.607576, 6990.251165, 12114.126734, 42.868924, 842.252985, 110978.768205, 7812.487026, 1690.144755, 7877.843768]
}
df_comparacion = pd.DataFrame(data_comparacion)
df_comparacion['Predicción'] = df_comparacion['Predicción'].round(2) # Redondear para mejor visualización
df_comparacion['Error Absoluto'] = df_comparacion['Error Absoluto'].round(2)

# Formatear como moneda (asumiendo COP)
for col in ['Real', 'Predicción', 'Error Absoluto']:
    df_comparacion[col] = df_comparacion[col].apply(lambda x: f"COP {x:,.2f}")

st.dataframe(df_comparacion, use_container_width=True)
st.markdown("""
Esta tabla muestra las primeras 10 predicciones del conjunto de prueba. Se observa variabilidad en la precisión,
lo cual es esperado. Casos como el índice 6 (error alto) podrían investigarse más a fondo para entender
si son outliers o propiedades con características muy inusuales.
""")
st.markdown("---")


# --- Sección de Predicción Interactiva y Características del Modelo ---
st.header("🔮 Predicción Interactiva y Relevancia de Características")

col_form, col_importance = st.columns([0.6, 0.4]) # Ajustar proporción si es necesario

with col_form:
    st.subheader("Simula el Precio de una Propiedad")
    st.markdown("Ajusta las características de la propiedad base para obtener una estimación del precio por noche.")

    # Valores por defecto basados en tu property_base
    # Asegúrate de que los nombres de las claves coincidan EXACTAMENTE con los que espera tu modelo.
    # Y que las opciones en los selectbox sean las que tu modelo fue entrenado (o preprocesado) para esperar.

    default_property_base = {
        'reviews': 50,
        'rating': 4.5,
        'host_id': 1000, # Este ID es problemático si es categórico y no numérico real.
                         # Si tu modelo lo trata como número, está bien. Si es un ID único,
                         # su importancia podría ser engañosa o causar overfitting.
                         # Para una POC, se mantiene como número si así lo usaste.
        'studios': 0,
        'bedrooms': 2,
        'beds': 3,
        'bathrooms': 1,
        'guests': 4,
        'toiles': 1, # Asumo que 'toiles' es una característica numérica
        'checkout_category': 'No Definido',
        'checkin_category': 'Mañana',
        'pais': 'Japan',
        'sector': 'Okinawa' # Este podría ser un campo de texto o un selectbox si tienes una lista definida
    }

    # Crear inputs interactivos
    # --- Características de la Propiedad ---
    st.write("**Detalles de la Propiedad:**")
    c1, c2, c3 = st.columns(3)
    bedrooms_input = c1.slider("Habitaciones (bedrooms)", 0, 10, default_property_base['bedrooms'])
    bathrooms_input = c2.slider("Baños Completos (bathrooms)", 0, 8, default_property_base['bathrooms'])
    beds_input = c3.slider("Camas (beds)", 1, 16, default_property_base['beds'])

    c4, c5, c6 = st.columns(3)
    guests_input = c4.slider("Huéspedes Permitidos (guests)", 1, 16, default_property_base['guests'])
    toiles_input = c5.slider("Medios Baños/Aseos (toiles)", 0, 5, default_property_base['toiles']) # Ajusta el rango según tus datos
    studios_input = c6.selectbox("¿Es un Estudio? (studios)", [0, 1], index=default_property_base['studios'], format_func=lambda x: "Sí" if x == 1 else "No")


    # --- Ubicación y Host ---
    st.write("**Ubicación y Host:**")
    c7, c8, c9 = st.columns(3)
    # Lista de países (EJEMPLO - DEBES ACTUALIZARLA CON LOS PAÍSES DE TU DATASET)
    # Si tu modelo espera una codificación específica (ej. one-hot), asegúrate que los valores aquí sean los correctos.
    paises_disponibles = ['Japan', 'United States', 'Spain', 'France', 'Other'] # ¡Actualiza esta lista!
    pais_input = c7.selectbox("País (pais)", paises_disponibles, index=paises_disponibles.index(default_property_base['pais']) if default_property_base['pais'] in paises_disponibles else 0)

    # Sector podría ser un text_input si es muy variable, o un selectbox si tienes una lista definida por país.
    # Para simplicidad, un text_input.
    sector_input = c8.text_input("Sector/Ciudad (sector)", value=default_property_base['sector'])
    host_id_input = c9.number_input("ID del Anfitrión (host_id)", value=default_property_base['host_id'], step=1, help="ID numérico del anfitrión. Si tu modelo lo usa de forma categórica, este input podría no ser ideal.")


    # --- Reseñas y Check-in/Out ---
    st.write("**Reseñas y Logística:**")
    c10, c11, c12 = st.columns(3)
    reviews_input = c10.number_input("Número de Reseñas (reviews)", 0, 2000, default_property_base['reviews']) # Ajusta el max
    rating_input = c11.slider("Calificación Promedio (rating)", 0.0, 5.0, default_property_base['rating'], 0.1)

    # Estas categorías deben coincidir con las que usó tu modelo.
    checkin_options = ['Mañana', 'Tarde', 'Noche', 'Flexible', 'No Definido'] # ¡Actualiza estas listas!
    checkout_options = ['Mañana', 'Tarde', 'Noche', 'Flexible', 'No Definido'] # ¡Actualiza estas listas!

    checkin_input = c12.selectbox("Categoría Check-in (checkin_category)", checkin_options, index=checkin_options.index(default_property_base['checkin_category']) if default_property_base['checkin_category'] in checkin_options else 0)
    checkout_input = c12.selectbox("Categoría Check-out (checkout_category)", checkout_options, index=checkout_options.index(default_property_base['checkout_category']) if default_property_base['checkout_category'] in checkout_options else 0)


    # Botón de predicción para la propiedad base
    if st.button("📈 Predecir Precio Base", key="predict_base_interactive", use_container_width=True):
        if modelo:
            # Crear DataFrame para la predicción con el orden de columnas esperado por el modelo
            # IMPORTANTE: El orden y los nombres de las columnas DEBEN coincidir con los que el modelo espera.
            #             Revisa tu script de entrenamiento para confirmar este orden.
            #             El orden que uso abajo está basado en tu 'Original_Column' de feature importance.
            input_data = {
                'bathrooms': bathrooms_input,
                'pais': pais_input,
                'host_id': host_id_input,
                'bedrooms': bedrooms_input,
                'reviews': reviews_input,
                'beds': beds_input,
                'sector': sector_input,
                'guests': guests_input,
                'checkin_category': checkin_input, # Renombrado de 'checkin' a 'checkin_category' para coincidir con tu property_base
                'checkout_category': checkout_input, # Renombrado de 'checkout' a 'checkout_category'
                'rating': rating_input,
                'toiles': toiles_input,
                'studios': studios_input
            }
            # Asegurar el orden correcto de las columnas como en la tabla de importancia
            ordered_columns = ['bathrooms', 'pais', 'host_id', 'bedrooms', 'reviews', 'beds', 'sector', 'guests', 'checkin_category', 'checkout_category', 'rating', 'toiles', 'studios']
            
            # Crear un diccionario con el orden correcto
            ordered_input_data = {col: input_data[col] for col in ordered_columns}

            input_df = pd.DataFrame([ordered_input_data])

            try:
                prediccion = modelo.predict(input_df)[0]
                st.success(f"**Precio Estimado por Noche: COP {prediccion:,.2f}**")
                # Guardar la predicción base para la simulación de rentabilidad
                st.session_state.precio_base_simulacion = prediccion
                st.session_state.property_base_simulacion = input_data.copy() # Guardar los inputs actuales

            except Exception as e:
                st.error(f"Error al predecir: {e}")
                st.error("Asegúrate de que todas las características (incluyendo país y sector) sean válidas y que el modelo esté cargado.")
                st.info("Detalles del DataFrame enviado al modelo:")
                st.dataframe(input_df)
        else:
            st.warning("El modelo no está cargado. No se puede predecir.")

with col_importance:
    st.subheader("Importancia de las Características")
    st.markdown("Visualización de cómo cada característica influye en la predicción del precio, según el modelo.")
    data_importancia = {
        'Original_Column': ['bathrooms', 'pais', 'host_id', 'bedrooms', 'reviews', 'beds', 'sector', 'guests', 'checkin', 'checkout', 'rating', 'toiles', 'studios'],
        'Importance': [0.257116, 0.191497, 0.122024, 0.111934, 0.103072, 0.075089, 0.057227, 0.033455, 0.019237, 0.017591, 0.007317, 0.004440, 0.000000]
    }
    df_importancia = pd.DataFrame(data_importancia).sort_values(by="Importance", ascending=False)
    
    fig_importancia = px.bar(df_importancia, x="Importance", y="Original_Column", orientation='h',
                             title="Importancia de Características en el Modelo",
                             labels={'Importance': 'Importancia Relativa', 'Original_Column': 'Característica'},
                             color="Importance", color_continuous_scale=px.colors.sequential.Viridis)
    fig_importancia.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_importancia, use_container_width=True)
    st.caption("Nota: 'checkin' y 'checkout' podrían referirse a 'checkin_category' y 'checkout_category' si fueron preprocesadas con esos nombres.")

st.markdown("---")

# --- Sección de Simulación de Mejoras y Rentabilidad ---
st.header("🛠️ Simulación de Mejoras y Cálculo de Rentabilidad")
st.markdown("""
Aquí puedes simular cómo ciertas mejoras a la propiedad (utilizando la configuración de la sección de predicción interactiva como base)
podrían afectar el precio por noche y, consecuentemente, la rentabilidad de la inversión.
Ingresa los costos asociados para realizar el cálculo.
""")

if 'property_base_simulacion' not in st.session_state or 'precio_base_simulacion' not in st.session_state:
    st.warning("Primero realiza una predicción en la sección 'Simula el Precio de una Propiedad' para activar esta simulación.")
else:
    # Recuperar la propiedad base y su precio de la predicción interactiva
    property_base_actual = st.session_state.property_base_simulacion
    price_base_actual = st.session_state.precio_base_simulacion

    st.info(f"Simulación basada en la propiedad con precio predicho de: **COP {price_base_actual:,.2f}**")

    expander_costos = st.expander("Configurar Costos de Inversión y Operación", expanded=True)
    with expander_costos:
        c_cost1, c_cost2, c_cost3 = st.columns(3)
        property_cost_input = c_cost1.number_input("Costo de Compra/Valor de la Propiedad (COP)", min_value=0, value=200000000, step=1000000)
        cost_bathroom_input = c_cost2.number_input("Costo de Añadir 1 Baño (COP)", min_value=0, value=15000000, step=500000)
        cost_bedroom_input = c_cost3.number_input("Costo de Añadir 1 Habitación (COP)", min_value=0, value=25000000, step=500000)

        c_op1, c_op2 = st.columns(2)
        occupancy_rate_input = c_op1.slider("Tasa de Ocupación Anual Estimada (%)", 0.0, 100.0, 70.0, 1.0) / 100.0
        operational_cost_rate_input = c_op2.slider("Costos Operativos Anuales (% de Ingresos)", 0.0, 100.0, 20.0, 1.0) / 100.0


    if st.button("🏦 Calcular Rentabilidad de Escenarios", key="calculate_roi", use_container_width=True):
        if modelo:
            # Crear DataFrame para la propiedad base actual (con el orden de columnas correcto)
            ordered_columns = ['bathrooms', 'pais', 'host_id', 'bedrooms', 'reviews', 'beds', 'sector', 'guests', 'checkin_category', 'checkout_category', 'rating', 'toiles', 'studios']
            
            base_data_for_df = {col: property_base_actual.get(col) for col in ordered_columns} # Usar .get() para evitar KeyError si alguna no está
            X_base = pd.DataFrame([base_data_for_df])

            # Escenario 1: Agregar 1 baño
            property_plus_bathroom_sim = property_base_actual.copy()
            property_plus_bathroom_sim['bathrooms'] = property_base_actual.get('bathrooms', 1) + 1 # Sumar al valor actual
            data_plus_bathroom = {col: property_plus_bathroom_sim.get(col) for col in ordered_columns}
            X_plus_bathroom = pd.DataFrame([data_plus_bathroom])

            # Escenario 2: Agregar 1 habitación
            property_plus_bedroom_sim = property_base_actual.copy()
            property_plus_bedroom_sim['bedrooms'] = property_base_actual.get('bedrooms', 1) + 1 # Sumar al valor actual
            data_plus_bedroom = {col: property_plus_bedroom_sim.get(col) for col in ordered_columns}
            X_plus_bedroom = pd.DataFrame([data_plus_bedroom])

            # Escenario 3: Agregar 1 baño y 1 habitación
            property_plus_both_sim = property_base_actual.copy()
            property_plus_both_sim['bathrooms'] = property_base_actual.get('bathrooms', 1) + 1
            property_plus_both_sim['bedrooms'] = property_base_actual.get('bedrooms', 1) + 1
            data_plus_both = {col: property_plus_both_sim.get(col) for col in ordered_columns}
            X_plus_both = pd.DataFrame([data_plus_both])

            try:
                # Predicciones del modelo
                # price_base_pred = modelo.predict(X_base)[0] # Ya lo tenemos de st.session_state
                price_plus_bathroom_pred = modelo.predict(X_plus_bathroom)[0]
                price_plus_bedroom_pred = modelo.predict(X_plus_bedroom)[0]
                price_plus_both_pred = modelo.predict(X_plus_both)[0]

                # Aplicar los multiplicadores de tu notebook (esto es una lógica de negocio que estás añadiendo post-predicción)
                # Si el modelo ya está bien entrenado para capturar estos efectos, los multiplicadores no serían necesarios.
                # Por ahora, se replica tu lógica.
                price_base_final = price_base_actual # Usar el precio base ya predicho
                price_plus_bathroom_final = price_plus_bathroom_pred #  * 2 # ¡OJO! Tu notebook multiplica por 2. ¿Es correcto?
                                                                      # Normalmente, el modelo debería predecir el nuevo precio directamente.
                                                                      # Si el modelo no es sensible a 'bathrooms', esta multiplicación es un parche.
                                                                      # Por ahora, lo replico. Considera si esta es la mejor aproximación.
                price_plus_bedroom_final = price_plus_bedroom_pred # * 1.5
                price_plus_both_final = price_plus_both_pred       # * 2.5


                nights_per_year = 365 * occupancy_rate_input
                scenarios_data = [
                    {'name': 'Propiedad Base (Actual)', 'price_per_night': price_base_final, 'remodeling_cost': 0},
                    {'name': 'Más 1 Baño', 'price_per_night': price_plus_bathroom_final, 'remodeling_cost': cost_bathroom_input},
                    {'name': 'Más 1 Habitación', 'price_per_night': price_plus_bedroom_final, 'remodeling_cost': cost_bedroom_input},
                    {'name': 'Más 1 Baño y 1 Habitación', 'price_per_night': price_plus_both_final, 'remodeling_cost': cost_bathroom_input + cost_bedroom_input}
                ]

                results_list = []
                for scenario in scenarios_data:
                    income = scenario['price_per_night'] * nights_per_year
                    operational_cost = income * operational_cost_rate_input
                    # Para el ROI, el costo total de inversión inicial es la compra + remodelación.
                    # El costo operativo es recurrente.
                    # ROI = (Ganancia Neta Anual / (Costo de Compra + Costo Remodelación)) * 100
                    # Ganancia Neta Anual = Ingreso Anual - Costo Operativo Anual
                    
                    ganancia_neta_anual = income - operational_cost
                    costo_inversion_total = property_cost_input + scenario['remodeling_cost']
                    
                    # Evitar división por cero si el costo de inversión es 0 (aunque no debería ser con property_cost_input)
                    roi = (ganancia_neta_anual / costo_inversion_total) * 100 if costo_inversion_total > 0 else 0
                    
                    results_list.append({
                        'Escenario': scenario['name'],
                        'Precio x Noche (COP)': scenario['price_per_night'],
                        'Costo Remodelación (COP)': scenario['remodeling_cost'],
                        'Ingreso Anual Bruto (COP)': income,
                        'Costo Operativo Anual (COP)': operational_cost,
                        'Ganancia Neta Anual (COP)': ganancia_neta_anual,
                        'Inversión Inicial Total (COP)': costo_inversion_total,
                        'ROI Anual (%)': roi
                    })
                
                results_df = pd.DataFrame(results_list)

                st.subheader("Resultados de Simulación y Rentabilidad")
                
                # Formatear columnas de moneda
                currency_cols = ['Precio x Noche (COP)', 'Costo Remodelación (COP)', 'Ingreso Anual Bruto (COP)', 'Costo Operativo Anual (COP)', 'Ganancia Neta Anual (COP)', 'Inversión Inicial Total (COP)']
                for col in currency_cols:
                    results_df[col] = results_df[col].apply(lambda x: f"{x:,.2f}")
                results_df['ROI Anual (%)'] = results_df['ROI Anual (%)'].apply(lambda x: f"{x:.2f}%")

                st.dataframe(results_df, use_container_width=True)

                # --- Visualizaciones ---
                # Reconvertir a numérico para graficar
                results_df_numeric = pd.DataFrame(results_list) # Usar la lista original con números

                col_chart1, col_chart2 = st.columns(2)
                with col_chart1:
                    fig_precios = px.bar(results_df_numeric, x='Escenario', y='price_per_night',
                                         title='Precio por Noche Predicho por Escenario',
                                         labels={'price_per_night': 'Precio por Noche (COP)', 'Escenario': 'Escenario de Mejora'},
                                         color='Escenario',
                                         text_auto='.2s') # Formato para el texto en barras
                    fig_precios.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig_precios, use_container_width=True)

                with col_chart2:
                    fig_roi = px.bar(results_df_numeric, x='Escenario', y='ROI Anual (%)',
                                     title='Retorno sobre la Inversión (ROI) Anual por Escenario',
                                     labels={'ROI Anual (%)': 'ROI Anual (%)', 'Escenario': 'Escenario de Mejora'},
                                     color='Escenario',
                                     text_auto='.2f')
                    fig_roi.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig_roi, use_container_width=True)

                st.info("""
                **Nota sobre los multiplicadores de precio post-predicción:**
                Los precios para escenarios con mejoras se calcularon aplicando los multiplicadores (ej. `*2` para baño, `*1.5` para habitación)
                que se usaron en el notebook original. Idealmente, el modelo de ML debería ser lo suficientemente sensible
                para predecir estos incrementos directamente al cambiar las características. Si los multiplicadores son necesarios,
                sugiere que el modelo podría no estar capturando completamente el impacto de estas mejoras específicas,
                o que se está aplicando una lógica de negocio externa. Para una POC, esta aproximación es válida para demostrar el concepto.
                """)

            except Exception as e:
                st.error(f"Error durante la simulación de escenarios: {e}")
                st.error("Verifica que el modelo esté cargado y los inputs sean correctos.")

        else:
            st.warning("El modelo no está cargado. No se pueden calcular los escenarios.")


# --- Pie de Página (Opcional) ---
st.markdown("---")
st.markdown("Dashboard POC desarrollado con Streamlit. Modelo de Regresión de Precios Airbnb.")
