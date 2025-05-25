import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import numpy as np
import json # Importar json

# --- Configuración de la Página ---
st.set_page_config(
    page_title="POC: Predicción de Precios Airbnb",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Cargar Modelo y Extraer Categorías ---
@st.cache_resource
def load_model_and_categories():
    model = None
    dynamic_categories = {} # To store lists of unique categories for each feature
    try:
        model = joblib.load("modelo_regresion_precio_final.joblib")
    except FileNotFoundError:
        st.error("Error: El archivo 'modelo_regresion_precio_final.joblib' no se encontró.")
        return None, None
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None, None

    if model:
        try:
            preprocessor = model.named_steps.get('preprocessor')
            if not preprocessor:
                st.warning("Paso 'preprocessor' no encontrado en el pipeline del modelo.")
                return model, dynamic_categories # Devolver dynamic_categories aunque esté vacío

            cat_transformer_tuple = None
            for t_name, t_obj, t_cols in preprocessor.transformers_:
                if t_name == 'cat':
                    cat_transformer_tuple = (t_obj, t_cols)
                    break
            
            if cat_transformer_tuple:
                cat_pipeline_or_encoder, original_cat_cols = cat_transformer_tuple
                ohe = None
                if hasattr(cat_pipeline_or_encoder, 'named_steps'): # Es un Pipeline
                    for step_name in ['onehotencoder', 'one_hot_encoder', 'ohe', 'onehot']: # Añadido 'onehot'
                        if step_name in cat_pipeline_or_encoder.named_steps:
                            ohe = cat_pipeline_or_encoder.named_steps[step_name]
                            break
                    if not ohe and hasattr(cat_pipeline_or_encoder.steps[-1][1], 'categories_'):
                        ohe = cat_pipeline_or_encoder.steps[-1][1]
                elif hasattr(cat_pipeline_or_encoder, 'categories_'): # Es el OneHotEncoder directamente
                    ohe = cat_pipeline_or_encoder

                if ohe and hasattr(ohe, 'categories_'):
                    if len(original_cat_cols) == len(ohe.categories_):
                        for i, col_name in enumerate(original_cat_cols):
                            categories = ohe.categories_[i]
                            dynamic_categories[col_name] = sorted([str(cat) for cat in categories if pd.notna(cat)])
                    else:
                        st.warning(f"Discrepancia en el número de columnas categóricas ({len(original_cat_cols)}) y las categorías del OHE ({len(ohe.categories_)}). No se pudieron extraer todas las categorías dinámicas.")

                else:
                    st.warning("No se pudo encontrar el OneHotEncoder o sus categorías dentro del pipeline 'cat'. Las listas desplegables podrían usar valores predeterminados.")
            else:
                st.warning("Transformador categórico ('cat') no encontrado en el preprocesador. Las listas desplegables usarán valores predeterminados.")

        except Exception as e:
            st.warning(f"Error al extraer categorías dinámicas del modelo: {e}. Se usarán valores predeterminados.")
    
    return model, dynamic_categories

modelo, dynamic_categories = load_model_and_categories()

# --- Valores por defecto para Selectores ---
# Estos se usarán si dynamic_categories no se puede poblar o el mapeo falla.
pais_sector_mapping_default = {
    'Japan': ['Okinawa', 'Tokyo', 'Kyoto', 'Hokkaido'],
    'United States': ['New York', 'California', 'Florida'],
    'Spain': ['Madrid', 'Barcelona', 'Seville'],
    'France': ['Paris', 'Nice', 'Lyon'],
    'Colombia': ['Bogotá D.C.', 'Medellín', 'Cartagena'], # Ejemplo, actualiza con tus datos
    'Other': ['OtherSector']
}
default_checkin_options = ['Mañana', 'Tarde', 'Noche', 'Flexible', 'No Definido']
default_checkout_options = ['Mañana', 'Tarde', 'Noche', 'Flexible', 'No Definido']
default_paises = ['Japan', 'United States', 'Spain', 'France', 'Colombia', 'Other']


# --- Función para cargar el mapeo país-sector desde JSON ---
@st.cache_resource # Cache para no recargar en cada interacción
def load_pais_sector_mapping_from_file(filepath="pais_sector_mapping.json"):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
        st.success(f"Mapeo país-sector cargado exitosamente desde '{filepath}'.")
        return mapping
    except FileNotFoundError:
        st.error(f"Error: El archivo '{filepath}' no se encontró. "
                 "Asegúrate de generar este archivo desde tu notebook de entrenamiento "
                 "y colocarlo en el mismo directorio que la app. "
                 "Se usarán los valores por defecto para el mapeo país-sector.")
        return pais_sector_mapping_default # Usar el default si el archivo no existe
    except json.JSONDecodeError:
        st.error(f"Error: El archivo '{filepath}' no es un JSON válido. Se usarán los valores por defecto.")
        return pais_sector_mapping_default
    except Exception as e:
        st.error(f"Error desconocido al cargar '{filepath}': {e}. Se usarán los valores por defecto.")
        return pais_sector_mapping_default

# --- Cargar el mapeo país-sector ---
# Esta variable contendrá el mapeo real (del archivo) o el por defecto si falla la carga.
# Se guarda en st.session_state para persistir entre reruns si es necesario,
# aunque @st.cache_resource ya ayuda con la carga eficiente.
if 'pais_sector_mapping' not in st.session_state:
    st.session_state.pais_sector_mapping = load_pais_sector_mapping_from_file()


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
# ... (resto de la sección sin cambios) ...
st.markdown("""
Para asegurar la validez de nuestras predicciones, el modelo fue evaluado rigurosamente.
A continuación, se presentan las métricas clave y una comparación de predicciones contra valores reales.
""")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Métricas de Evaluación:")
    st.metric(label="MAE Entrenamiento (Error Absoluto Medio)", value="COP {:,.2f}".format(11789.95))
    st.metric(label="R² Entrenamiento (Coeficiente de Determinación)", value="0.58")
with col2:
    st.subheader(" ")
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
df_comparacion['Predicción'] = df_comparacion['Predicción'].round(2)
df_comparacion['Error Absoluto'] = df_comparacion['Error Absoluto'].round(2)
for col_comp in ['Real', 'Predicción', 'Error Absoluto']:
    df_comparacion[col_comp] = df_comparacion[col_comp].apply(lambda x: f"COP {x:,.2f}")
st.dataframe(df_comparacion, use_container_width=True)
st.markdown("""
Esta tabla muestra las primeras 10 predicciones del conjunto de prueba. Se observa variabilidad en la precisión,
lo cual es esperado. Casos como el índice 6 (error alto) podrían investigarse más a fondo para entender
si son outliers o propiedades con características muy inusuales.
""")
st.markdown("---")


# --- Sección de Predicción Interactiva y Características del Modelo ---
st.header("🔮 Predicción Interactiva y Relevancia de Características")

col_form, col_importance = st.columns([0.6, 0.4])

with col_form:
    st.subheader("Simula el Precio de una Propiedad")
    st.markdown("Ajusta las características de la propiedad base para obtener una estimación del precio por noche.")

    default_property_base = {
        'reviews': 50, 'rating': 4.5, 'host_id': 1000, 'studios': 0,
        'bedrooms': 2, 'beds': 3, 'bathrooms': 1, 'guests': 4, 'toiles': 1,
        'checkout_category': 'No Definido', 'checkin_category': 'Mañana',
        'pais': 'Colombia', # País por defecto
        'sector': 'Bogotá D.C.' # Sector por defecto para Colombia
    }

    # --- Usar categorías dinámicas o por defecto ---
    # Para paises_options, priorizar las llaves del mapeo cargado
    if st.session_state.pais_sector_mapping and st.session_state.pais_sector_mapping != pais_sector_mapping_default:
        paises_options = sorted(list(st.session_state.pais_sector_mapping.keys()))
    else: # Fallback si el mapeo es el default o está vacío
        paises_options = dynamic_categories.get('pais', default_paises) if dynamic_categories else default_paises
        if not paises_options: # Si sigue vacío, usar el default_paises directamente
             paises_options = default_paises

    # Asegurar que el país por defecto esté en las opciones, si no, tomar el primero
    if default_property_base['pais'] not in paises_options and paises_options:
        default_property_base['pais'] = paises_options[0]
    elif not paises_options: # Si no hay opciones de país (muy improbable)
        st.warning("No hay países disponibles para seleccionar.")
        paises_options = [default_property_base['pais']] # Evitar error en selectbox

    checkin_options = dynamic_categories.get('checkin_category', default_checkin_options) if dynamic_categories else default_checkin_options
    checkout_options = dynamic_categories.get('checkout_category', default_checkout_options) if dynamic_categories else default_checkout_options


    st.write("**Detalles de la Propiedad:**")
    c1, c2, c3 = st.columns(3)
    bedrooms_input = c1.slider("Habitaciones (bedrooms)", 0, 10, default_property_base['bedrooms'])
    bathrooms_input = c2.slider("Baños Completos (bathrooms)", 0, 8, default_property_base['bathrooms'])
    beds_input = c3.slider("Camas (beds)", 1, 16, default_property_base['beds'])

    c4, c5, c6 = st.columns(3)
    guests_input = c4.slider("Huéspedes Permitidos (guests)", 1, 16, default_property_base['guests'])
    toiles_input = c5.slider("Medios Baños/Aseos (toiles)", 0, 5, default_property_base['toiles'])
    studios_input = c6.selectbox("¿Es un Estudio? (studios)", [0, 1], index=default_property_base['studios'], format_func=lambda x: "Sí" if x == 1 else "No")


    st.write("**Ubicación y Host:**")
    c7, c8, c9 = st.columns(3)
    
    default_pais_index = paises_options.index(default_property_base['pais']) if default_property_base['pais'] in paises_options else 0
    pais_input = c7.selectbox("País (pais)", paises_options, index=default_pais_index, key="pais_selector")

    # --- LÓGICA REVISADA PARA SECTORES DEPENDIENTES ---
    sectores_para_pais_seleccionado = []
    if pais_input:
        # Usar el mapeo cargado en st.session_state
        sectores_para_pais_seleccionado = st.session_state.pais_sector_mapping.get(pais_input, [])

    default_sector_value = None
    if pais_input == default_property_base['pais'] and default_property_base['sector'] in sectores_para_pais_seleccionado:
        default_sector_value = default_property_base['sector']
    elif sectores_para_pais_seleccionado: # Si hay sectores para el país, tomar el primero como default si el original no aplica
        default_sector_value = sectores_para_pais_seleccionado[0]
    
    sector_input = None # Inicializar sector_input
    if sectores_para_pais_seleccionado:
        # Asegurar que el default_sector_value es válido antes de buscar su índice
        idx_sector = 0
        if default_sector_value and default_sector_value in sectores_para_pais_seleccionado:
            idx_sector = sectores_para_pais_seleccionado.index(default_sector_value)
        
        sector_input = c8.selectbox(
            "Sector/Ciudad (sector)",
            options=sectores_para_pais_seleccionado,
            index=idx_sector
        )
    else:
        c8.info(f"No hay sectores definidos para '{pais_input}' en el mapeo cargado.")
        # sector_input permanece None

    host_id_input = c9.number_input("ID del Anfitrión (host_id)", value=default_property_base['host_id'], step=1, min_value=0)


    st.write("**Reseñas y Logística:**")
    c10, c11, c12_a, c12_b = st.columns(4) # Ajustado para dos selectbox en la misma línea visual
    reviews_input = c10.number_input("Número de Reseñas (reviews)", 0, 2000, default_property_base['reviews'])
    rating_input = c11.slider("Calificación Promedio (rating)", 0.0, 5.0, default_property_base['rating'], 0.1)

    default_checkin_index = checkin_options.index(default_property_base['checkin_category']) if default_property_base['checkin_category'] in checkin_options else 0
    checkin_input = c12_a.selectbox("Categoría Check-in", checkin_options, index=default_checkin_index)
    
    default_checkout_index = checkout_options.index(default_property_base['checkout_category']) if default_property_base['checkout_category'] in checkout_options else 0
    checkout_input = c12_b.selectbox("Categoría Check-out", checkout_options, index=default_checkout_index, key="checkout_cat_key")


    if st.button("📈 Predecir Precio Base", key="predict_base_interactive", use_container_width=True):
        if modelo:
            if not pais_input:
                st.warning("Por favor, seleccione un país.")
            elif not sector_input and sectores_para_pais_seleccionado: # Hay opciones de sector pero ninguna seleccionada (no debería pasar con selectbox)
                st.warning(f"Por favor, seleccione un sector para '{pais_input}'.")
            elif not sector_input and not sectores_para_pais_seleccionado: # No hay sectores para el país
                 st.error(f"No se puede predecir: El país '{pais_input}' no tiene sectores configurados en el mapeo o el mapeo no se cargó correctamente.")
            else: # Tenemos pais y sector
                input_data = {
                    'bathrooms': bathrooms_input, 'pais': pais_input, 'host_id': host_id_input,
                    'bedrooms': bedrooms_input, 'reviews': reviews_input, 'beds': beds_input,
                    'sector': sector_input, 'guests': guests_input,
                    'checkin_category': checkin_input, 'checkout_category': checkout_input,
                    'rating': rating_input, 'toiles': toiles_input, 'studios': studios_input
                }
                
                input_df = pd.DataFrame([input_data])
                # Asegurar que el orden de las columnas coincide con el entrenamiento del modelo
                # Esto es crucial. Si el preprocesador espera un orden específico, hay que respetarlo.
                # O, mejor, si el preprocesador se ajustó con DataFrames, manejará los nombres de columna.
                try:
                    prediccion = modelo.predict(input_df)[0]
                    st.success(f"**Precio Estimado por Noche: COP {prediccion:,.2f}**")
                    st.session_state.precio_base_simulacion = prediccion
                    st.session_state.property_base_simulacion = input_data.copy()
                except Exception as e:
                    st.error(f"Error al predecir: {e}")
                    st.error("Asegúrate de que todas las características necesarias por el modelo estén presentes y con los tipos de datos correctos.")
                    st.dataframe(input_df)
                    # st.write("Columnas del DataFrame enviado:", input_df.columns.tolist())
                    # if hasattr(modelo, 'feature_names_in_'):
                    #     st.write("Características esperadas por el modelo:", modelo.feature_names_in_)
                    # elif hasattr(modelo.named_steps.get('preprocessor'), 'get_feature_names_out'):
                    #    st.write("Características después del preprocesamiento:", modelo.named_steps.get('preprocessor').get_feature_names_out())


        else:
            st.warning("El modelo no está cargado. No se puede predecir.")

with col_importance:
    st.subheader("Importancia de las Características")
    # ... (resto de la sección sin cambios) ...
    st.markdown("Visualización de cómo cada característica influye en la predicción del precio, según el modelo.")
    # Esta data de importancia viene de tu notebook, ya procesada para mostrar columnas originales
    data_importancia_agrupada = {
        'Original_Column': ['bathrooms', 'pais', 'host_id', 'bedrooms', 'reviews', 'beds', 'sector', 'guests', 'checkin_category', 'checkout_category', 'rating', 'toiles', 'studios'], # Asegúrate que los nombres coincidan con las columnas originales
        'Importance': [0.257116, 0.191497, 0.122024, 0.111934, 0.103072, 0.075089, 0.057227, 0.033455, 0.019237, 0.017591, 0.007317, 0.004440, 0.000000]
    }
    # Renombrar 'checkin' a 'checkin_category' y 'checkout' a 'checkout_category' si es necesario para coincidir
    for i, col_name in enumerate(data_importancia_agrupada['Original_Column']):
        if col_name == 'checkin':
            data_importancia_agrupada['Original_Column'][i] = 'checkin_category'
        if col_name == 'checkout':
            data_importancia_agrupada['Original_Column'][i] = 'checkout_category'

    df_importancia = pd.DataFrame(data_importancia_agrupada).sort_values(by="Importance", ascending=False)
    
    fig_importancia = px.bar(df_importancia, x="Importance", y="Original_Column", orientation='h',
                                title="Importancia de Características en el Modelo",
                                labels={'Importance': 'Importancia Relativa', 'Original_Column': 'Característica'},
                                color="Importance", color_continuous_scale=px.colors.sequential.Viridis)
    fig_importancia.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_importancia, use_container_width=True)

st.markdown("---")

# --- Sección de Simulación de Mejoras y Rentabilidad ---
st.header("🛠️ Simulación de Mejoras y Cálculo de Rentabilidad")
# ... (resto de la sección sin cambios, asumiendo que `property_base_actual` y las predicciones de escenarios
#      se manejarán correctamente con `pais` y `sector` válidos de la simulación interactiva) ...
st.markdown("""
Aquí puedes simular cómo ciertas mejoras a la propiedad (utilizando la configuración de la sección de predicción interactiva como base)
podrían afectar el precio por noche y, consecuentemente, la rentabilidad de la inversión.
Ingresa los costos asociados para realizar el cálculo.
""")

if 'property_base_simulacion' not in st.session_state or 'precio_base_simulacion' not in st.session_state:
    st.warning("Primero realiza una predicción en la sección 'Simula el Precio de una Propiedad' para activar esta simulación.")
else:
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
            base_df_sim = pd.DataFrame([property_base_actual]) 

            # Escenarios de mejora
            property_plus_bathroom_sim = property_base_actual.copy()
            property_plus_bathroom_sim['bathrooms'] = property_base_actual.get('bathrooms', 1) + 1
            X_plus_bathroom = pd.DataFrame([property_plus_bathroom_sim])

            property_plus_bedroom_sim = property_base_actual.copy()
            property_plus_bedroom_sim['bedrooms'] = property_base_actual.get('bedrooms', 1) + 1
            X_plus_bedroom = pd.DataFrame([property_plus_bedroom_sim])

            property_plus_both_sim = property_base_actual.copy()
            property_plus_both_sim['bathrooms'] = property_base_actual.get('bathrooms', 1) + 1
            property_plus_both_sim['bedrooms'] = property_base_actual.get('bedrooms', 1) + 1
            X_plus_both = pd.DataFrame([property_plus_both_sim])
            
            try:
                price_plus_bathroom_pred = modelo.predict(X_plus_bathroom)[0]
                price_plus_bedroom_pred = modelo.predict(X_plus_bedroom)[0]
                price_plus_both_pred = modelo.predict(X_plus_both)[0]

                price_base_final = price_base_actual
                price_plus_bathroom_final = price_plus_bathroom_pred 
                price_plus_bedroom_final = price_plus_bedroom_pred  
                price_plus_both_final = price_plus_both_pred     

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
                    ganancia_neta_anual = income - operational_cost
                    costo_inversion_total = property_cost_input + scenario['remodeling_cost']
                    roi = (ganancia_neta_anual / costo_inversion_total) * 100 if costo_inversion_total > 0 else 0
                    results_list.append({
                        'Escenario': scenario['name'], 'Precio x Noche (COP)': scenario['price_per_night'],
                        'Costo Remodelación (COP)': scenario['remodeling_cost'], 'Ingreso Anual Bruto (COP)': income,
                        'Costo Operativo Anual (COP)': operational_cost, 'Ganancia Neta Anual (COP)': ganancia_neta_anual,
                        'Inversión Inicial Total (COP)': costo_inversion_total, 'ROI Anual (%)': roi
                    })
                results_df = pd.DataFrame(results_list)
                st.subheader("Resultados de Simulación y Rentabilidad")
                currency_cols = ['Precio x Noche (COP)', 'Costo Remodelación (COP)', 'Ingreso Anual Bruto (COP)', 'Costo Operativo Anual (COP)', 'Ganancia Neta Anual (COP)', 'Inversión Inicial Total (COP)']
                for col_curr in currency_cols: results_df[col_curr] = results_df[col_curr].apply(lambda x: f"{x:,.2f}")
                results_df['ROI Anual (%)'] = results_df['ROI Anual (%)'].apply(lambda x: f"{x:.2f}%")
                st.dataframe(results_df, use_container_width=True)

                results_df_numeric = pd.DataFrame(results_list) # Usar la lista original con números
                col_chart1, col_chart2 = st.columns(2)
                with col_chart1:
                    fig_precios = px.bar(results_df_numeric, x='Escenario', y='price_per_night', title='Precio por Noche Predicho por Escenario', labels={'price_per_night': 'Precio por Noche (COP)', 'Escenario': 'Escenario de Mejora'}, color='Escenario', text_auto='.2s')
                    fig_precios.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig_precios, use_container_width=True)
                with col_chart2:
                    fig_roi = px.bar(results_df_numeric, x='Escenario', y='ROI Anual (%)', title='Retorno sobre la Inversión (ROI) Anual por Escenario', labels={'ROI Anual (%)': 'ROI Anual (%)', 'Escenario': 'Escenario de Mejora'}, color='Escenario', text_auto='.2f')
                    fig_roi.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig_roi, use_container_width=True)
                st.info("""
                **Nota sobre los multiplicadores de precio post-predicción:**
                Si tu modelo ya está bien entrenado para capturar el impacto de añadir baños/habitaciones, los multiplicadores artificiales podrían no ser necesarios o incluso distorsionar la predicción "pura" del modelo para esos escenarios. Evalúa si son apropiados para tu POC.
                """)
            except Exception as e:
                st.error(f"Error durante la simulación de escenarios: {e}")
        else:
            st.warning("El modelo no está cargado.")

# --- Pie de Página (Opcional) ---
st.markdown("---")
st.markdown("Dashboard POC desarrollado con Streamlit. Modelo de Regresión de Precios Airbnb.")
