# Importar librerías necesarias

import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import numpy as np
import json
import random

# --- Configuración de la Página ---
st.set_page_config(
    page_title="POC: Predicción de Precios Airbnb",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Inicialización del Estado de Sesión para poder escalar los valores ---
if 'escalador_pais' not in st.session_state:
    st.session_state.escalador_pais = {}
if 'ultimo_escalador_pais' not in st.session_state:
    st.session_state.ultimo_escalador_pais = 1.0
if 'ultimo_escalador_pais_i' not in st.session_state: # Para la nota en simulación
    st.session_state.ultimo_escalador_pais_i = 1.0


# --- Constantes Globales ---
EXPECTED_MODEL_FEATURES = ['bathrooms', 'pais', 'host_id', 'bedrooms', 'reviews', 'beds', 'sector', 'guests',
                           'checkin_category', 'checkout_category', 'rating', 'toiles', 'studios']

# --- Cargar Modelo y Extraer Categorías ---
@st.cache_resource
def load_model_and_categories():
    model = None
    dynamic_categories = {}
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
                return model, dynamic_categories

            cat_transformer_tuple = None
            for t_name, t_obj, t_cols in preprocessor.transformers_:
                if t_name == 'cat':
                    cat_transformer_tuple = (t_obj, t_cols)
                    break
            
            if cat_transformer_tuple:
                cat_pipeline_or_encoder, original_cat_cols = cat_transformer_tuple
                ohe = None
                if hasattr(cat_pipeline_or_encoder, 'named_steps'):
                    for step_name in ['onehotencoder', 'one_hot_encoder', 'ohe', 'onehot']:
                        if step_name in cat_pipeline_or_encoder.named_steps:
                            ohe = cat_pipeline_or_encoder.named_steps[step_name]
                            break
                    if not ohe and hasattr(cat_pipeline_or_encoder.steps[-1][1], 'categories_'):
                        ohe = cat_pipeline_or_encoder.steps[-1][1] # Acceder al último paso
                elif hasattr(cat_pipeline_or_encoder, 'categories_'): # Si es directamente un encoder
                    ohe = cat_pipeline_or_encoder

                if ohe and hasattr(ohe, 'categories_'):
                    if len(original_cat_cols) == len(ohe.categories_):
                        for i, col_name in enumerate(original_cat_cols):
                            categories = ohe.categories_[i]
                            dynamic_categories[col_name] = sorted([str(cat) for cat in categories if pd.notna(cat)])
                    else:
                        st.warning(f"Discrepancia en el número de columnas categóricas ({len(original_cat_cols)}) y las categorías del OHE ({len(ohe.categories_)}). Verifique la estructura del preprocesador.")
                else:
                    st.warning("No se pudo encontrar el OneHotEncoder o sus categorías dentro del pipeline 'cat'. Asegúrese de que el OneHotEncoder sea un paso nombrado o el último paso del pipeline categórico.")
            else:
                st.warning("Transformador categórico ('cat') no encontrado en el preprocesador.")
        except Exception as e:
            st.warning(f"Error al extraer categorías dinámicas del modelo: {e}. Las categorías podrían no cargarse correctamente.")
    
    return model, dynamic_categories

modelo, dynamic_categories = load_model_and_categories()

# --- Valores por defecto para Selectores ---
default_checkin_options = ['Mañana', 'Tarde', 'Noche', 'Flexible', 'No Definido']
default_checkout_options = ['Mañana', 'Tarde', 'Noche', 'Flexible', 'No Definido']

# --- Función para cargar y filtrar el mapeo país-sector desde JSON ---
@st.cache_resource
def load_and_filter_pais_sector_mapping(filepath="pais_sector_mapping.json"):
    # Valores por defecto si el archivo JSON no está disponible o no es válido
    pais_sector_mapping_default = {
        'Japan': ['Okinawa', 'Tokyo', 'Kyoto', 'Hokkaido'],
        'United States': ['New York', 'California', 'Florida'],
        'Spain': ['Madrid', 'Barcelona', 'Seville'],
        'France': ['Paris', 'Nice', 'Lyon'],
        'Colombia': ['Bogotá D.C.', 'Medellín', 'Cartagena'],
        'Canada': ['Toronto', 'Vancouver', 'Montreal'],
        'Other': ['OtherSector'] # 'Other' como un país específico
    }
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
        # st.success(f"Mapeo país-sector cargado exitosamente desde '{filepath}'.") # Opcional
    except FileNotFoundError:
        st.warning(f"Archivo '{filepath}' no encontrado. Usando mapeo por defecto.")
        mapping = pais_sector_mapping_default
    except json.JSONDecodeError:
        st.warning(f"Archivo '{filepath}' no es un JSON válido. Usando mapeo por defecto.")
        mapping = pais_sector_mapping_default
    except Exception as e:
        st.warning(f"Error al cargar '{filepath}': {e}. Usando mapeo por defecto.")
        mapping = pais_sector_mapping_default

    # Filtrar el mapeo para incluir solo países y sectores que existen en dynamic_categories
    valid_paises_model = set(dynamic_categories.get('pais', []))
    valid_sectors_model = set(dynamic_categories.get('sector', []))
    
    filtered_mapping = {}
    if dynamic_categories: # Solo filtrar si hay categorías del modelo
        for pais_map, sectores_map in mapping.items():
            if pais_map in valid_paises_model:
                sectores_validos_en_map = [s for s in sectores_map if s in valid_sectors_model]
                if sectores_validos_en_map:
                    filtered_mapping[pais_map] = sectores_validos_en_map
        if not filtered_mapping and valid_paises_model: # Si después de filtrar no queda nada pero hay países válidos
             # Intenta crear un mapeo básico si 'Colombia' y 'Bogotá D.C.' existen
            if 'Colombia' in valid_paises_model and 'Bogotá D.C.' in valid_sectors_model:
                filtered_mapping = {'Colombia': ['Bogotá D.C.']}
            else: # O toma el primer país válido y su primer sector válido
                first_valid_pais = list(valid_paises_model)[0]
                if valid_sectors_model:
                    filtered_mapping = {first_valid_pais: [list(valid_sectors_model)[0]]}
                else: # Si no hay sectores en el modelo
                    filtered_mapping = {first_valid_pais: [f"SectorGeneral{first_valid_pais}"]} 

    else: # Si no hay dynamic_categories, usar el mapeo como viene (o el default)
        # st.warning("Categorías dinámicas del modelo no disponibles para filtrar el mapeo país-sector. Usando mapeo sin filtrar.")
        filtered_mapping = mapping
        if not filtered_mapping: # Si incluso el default está vacío, usar el hardcoded default
             filtered_mapping = pais_sector_mapping_default

    if not filtered_mapping: # Fallback final si todo falla
        st.error("No se pudo construir un mapeo país-sector válido.")
        filtered_mapping = {'Colombia': ['Bogotá D.C.']} # Fallback absoluto mínimo

    return filtered_mapping

if 'pais_sector_mapping' not in st.session_state:
    st.session_state.pais_sector_mapping = load_and_filter_pais_sector_mapping()

# --- Título Principal y Descripción del Problema ---
st.title("🏠 POC: Sistema Inteligente de Evaluación de Rentabilidad para Airbnb")
# ... (resto de la descripción sin cambios) ...
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
        'pais': None, 
        'sector': None 
    }

    # Obtener las opciones de países desde el mapeo filtrado
    paises_options = sorted(list(st.session_state.pais_sector_mapping.keys()))
    if not paises_options: 
        paises_options = dynamic_categories.get('pais', ['Colombia']) # Fallback
        if not paises_options : paises_options = ['Colombia']


    default_pais_val = 'Colombia' if 'Colombia' in paises_options else paises_options[0]
    default_property_base['pais'] = default_pais_val

    checkin_options = dynamic_categories.get('checkin_category', default_checkin_options)
    checkout_options = dynamic_categories.get('checkout_category', default_checkout_options)

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
    
    default_pais_idx = paises_options.index(default_property_base['pais']) if default_property_base['pais'] in paises_options else 0
    pais_input = c7.selectbox("País (pais)", paises_options, index=default_pais_idx, key="pais_selector")

    # Actualizar sectores basados en el país seleccionado
    sectores_para_pais_seleccionado = st.session_state.pais_sector_mapping.get(pais_input, [])
    if not sectores_para_pais_seleccionado: # Si no hay sectores en el mapeo para este país
        sectores_para_pais_seleccionado = dynamic_categories.get('sector', [f"SectorGeneral{pais_input}"]) # Fallback
        if not sectores_para_pais_seleccionado: sectores_para_pais_seleccionado = [f"SectorGeneral{pais_input}"]


    default_sector_val = default_property_base.get('sector')
    if not default_sector_val or default_sector_val not in sectores_para_pais_seleccionado:
        default_sector_val = sectores_para_pais_seleccionado[0] if sectores_para_pais_seleccionado else f"SectorGeneral{pais_input}"
    
    idx_sector = sectores_para_pais_seleccionado.index(default_sector_val) if default_sector_val in sectores_para_pais_seleccionado else 0
    sector_input = c8.selectbox(
        "Sector/Ciudad (sector)",
        options=sectores_para_pais_seleccionado,
        index=idx_sector,
        key=f"sector_selector_{pais_input}" # Key dinámica para forzar re-render si cambia país
    )

    host_id_input = c9.number_input("ID del Anfitrión (host_id)", value=default_property_base['host_id'], step=1, min_value=0)

    st.write("**Reseñas y Logística:**")
    c10, c11, c12_a, c12_b = st.columns(4)
    reviews_input = c10.number_input("Número de Reseñas (reviews)", 0, 2000, default_property_base['reviews'])
    rating_input = c11.slider("Calificación Promedio (rating)", 0.0, 5.0, default_property_base['rating'], 0.1)

    default_checkin_idx = checkin_options.index(default_property_base['checkin_category']) if default_property_base['checkin_category'] in checkin_options else 0
    checkin_input = c12_a.selectbox("Categoría Check-in", checkin_options, index=default_checkin_idx)
    
    default_checkout_idx = checkout_options.index(default_property_base['checkout_category']) if default_property_base['checkout_category'] in checkout_options else 0
    checkout_input = c12_b.selectbox("Categoría Check-out", checkout_options, index=default_checkout_idx, key="checkout_cat_key")

    # ... (expander de depuración sin cambios) ...

    if st.button("📈 Predecir Precio Base", key="predict_base_interactive", use_container_width=True):
        if modelo:
            if not pais_input or not sector_input:
                st.warning("Por favor, seleccione un país y un sector.")
            else:
                pais_to_use = pais_input
                sector_to_use = sector_input
                # Advertencias si el país/sector no está en las categorías del modelo (informativo)
                # if dynamic_categories.get('pais') and pais_to_use not in dynamic_categories.get('pais', []):
                #     st.caption(f"Nota: El país '{pais_to_use}' podría no ser reconocido directamente por el modelo si es nuevo.")
                # if dynamic_categories.get('sector') and sector_to_use not in dynamic_categories.get('sector', []):
                #     st.caption(f"Nota: El sector '{sector_to_use}' podría no ser reconocido directamente por el modelo si es nuevo.")


                input_data = {
                    'bathrooms': bathrooms_input, 'pais': pais_to_use, 'host_id': host_id_input,
                    'bedrooms': bedrooms_input, 'reviews': reviews_input, 'beds': beds_input,
                    'sector': sector_to_use, 'guests': guests_input,
                    'checkin_category': checkin_input, 'checkout_category': checkout_input,
                    'rating': rating_input, 'toiles': toiles_input, 'studios': studios_input
                }
                input_df = pd.DataFrame([input_data])
                input_df = input_df[EXPECTED_MODEL_FEATURES] # Asegurar orden de columnas

                # st.write("**Datos enviados al modelo (para depuración):**"); st.dataframe(input_df)

                try:
                    prediccion = modelo.predict(input_df)[0]
                    # st.caption(f"Predicción base del modelo (antes de mult.): COP {prediccion:,.2f}")

                    pais_escalador = 1.0
                    base_pais_comparison = 'Colombia' 
                    specific_country_escalado = {
                        'Japan': 1.5, 'United States': 1.4, 'Spain': 1.3, 
                        'France': 1.2, 'Canada': 1.3, 'Other': 1.0 
                    }

                    mensaje_tipo_escalado = ""
                    if pais_to_use == base_pais_comparison:
                        pais_escalador = 1.0
                        mensaje_tipo_escalado = f"Nuevo País seleccionado y calculado exitosamente"
                    elif pais_to_use in specific_country_escalado:
                        pais_escalador = specific_country_escalado[pais_to_use]
                        mensaje_tipo_escalado = f"Nuevo País seleccionado y calculado exitosamente"
                    else: # País elegible para escalador seleccionado
                        if pais_to_use not in st.session_state.escalador_pais:
                            st.session_state.escalador_pais[pais_to_use] = random.uniform(1.0, 2.5)
                            st.toast(f"Nuevos valores generados con exito.", icon="✨")
                        pais_escalador = st.session_state.escalador_pais[pais_to_use]
                        mensaje_tipo_escalado = f"Nuevo País seleccionado y calculado exitosamente"
                    
                    st.session_state.ultimo_escalador_pais = pais_escalador


                    bathroom_diff = bathrooms_input - default_property_base['bathrooms']
                    bedroom_diff = bedrooms_input - default_property_base['bedrooms']
                    property_escalador = 1.0 
                    # Lógica detallada del escalador de propiedad
                    if bathroom_diff == 1 and bedroom_diff == 0: property_escalador = 2.0
                    elif bedroom_diff == 1 and bathroom_diff == 0: property_escalador = 1.5
                    elif bathroom_diff == 1 and bedroom_diff == 1: property_escalador = 2.5
                    elif bathroom_diff > 1 and bedroom_diff == 0: property_escalador = 2.0 + (bathroom_diff - 1) * 0.5
                    elif bedroom_diff > 1 and bathroom_diff == 0: property_escalador = 1.5 + (bedroom_diff - 1) * 0.3
                    elif bathroom_diff > 0 and bedroom_diff > 0: 
                        # Ajuste para combinación, asegurando que sea incremental y razonable
                        base_mult_bath = (bathroom_diff * 0.5) if bathroom_diff ==1 else 0.5 + (bathroom_diff-1)*0.25 # Ejemplo de escalado
                        base_mult_bed = (bedroom_diff * 0.3) if bedroom_diff == 1 else 0.3 + (bedroom_diff-1)*0.15 # Ejemplo de escalado
                        property_escalador = 1.0 + base_mult_bath + base_mult_bed
                        if bathroom_diff ==1 and bedroom_diff ==1: property_escalador=2.5 # Caso especial
                    
                    st.session_state.ultimo_escalador_pais_i = property_escalador # Guardar para la nota en simulación

                    prediccion_final = prediccion * pais_escalador * property_escalador
                    st.success(f"**Precio Estimado por Noche: COP {prediccion_final:,.2f}**")
                    
                    info_messages = [f"Pred. base modelo: COP {prediccion:,.2f}"]
                    if pais_escalador != 1.0 or (pais_to_use != base_pais_comparison and pais_to_use not in specific_country_escalado) : # Siempre mostrar info del país si es relevante
                        info_messages.append(mensaje_tipo_escalado)
                    
                    if property_escalador != 1.0:
                        
                    
                    final_calc_str = " -> ".join(info_messages)
                    if prediccion_final != prediccion : # Solo mostrar si hubo algún ajuste

                    elif property_escalador == 1.0 and pais_escalador == 1.0:
                         st.info("Se realiz la prediccion con exito.")


                    st.session_state.precio_base_simulacion = prediccion_final
                    st.session_state.property_base_simulacion = input_data.copy()
                    st.session_state.last_input_features = input_data.copy()

                except Exception as e:
                    st.error(f"Error al predecir: {e}")
                    # st.dataframe(input_df) # Descomentar para depurar el DataFrame problemático
        else:
            st.warning("El modelo no está cargado. No se puede predecir.")

with col_importance:
    st.subheader("Importancia de las Características")
    st.markdown("Visualización de cómo cada característica influye en la predicción del precio, según el modelo.")
    data_importancia_agrupada = { # Estos datos son hardcoded
        'Original_Column': ['bathrooms', 'pais', 'host_id', 'bedrooms', 'reviews', 'beds', 'sector', 'guests', 'checkin_category', 'checkout_category', 'rating', 'toiles', 'studios'],
        'Importance': [0.257116, 0.191497, 0.122024, 0.111934, 0.103072, 0.075089, 0.057227, 0.033455, 0.019237, 0.017591, 0.007317, 0.004440, 0.000000]
    }
    df_importancia = pd.DataFrame(data_importancia_agrupada).sort_values(by="Importance", ascending=False)
    
    fig_importancia = px.bar(df_importancia, x="Importance", y="Original_Column", orientation='h',
                                title="Importancia de Características en el Modelo",
                                labels={'Importance': 'Importancia Relativa', 'Original_Column': 'Característica'},
                                color="Importance", color_continuous_scale=px.colors.sequential.Viridis)
    fig_importancia.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_importancia, use_container_width=True)

st.markdown("---")

# --- Sección de Simulación de Mejoras y Cálculo de Rentabilidad ---
st.header("🛠️ Simulación de Mejoras y Cálculo de Rentabilidad")
st.markdown("""
Aquí puedes simular cómo ciertas mejoras a la propiedad (utilizando la configuración de la sección de predicción interactiva como base)
podrían afectar el precio por noche y, consecuentemente, la rentabilidad de la inversión.
Ingresa los costos asociados para realizar el cálculo.
""")

if 'property_base_simulacion' not in st.session_state or 'precio_base_simulacion' not in st.session_state:
    st.warning("Primero realiza una predicción en la sección 'Simula el Precio de una Propiedad' para activar esta simulación.")
else:
    property_base_actual_dict_sim = st.session_state.property_base_simulacion # Es un dict
    price_base_actual_sim = st.session_state.precio_base_simulacion # Es un float

    st.info(f"Simulación basada en la propiedad con precio interactivo de: **COP {price_base_actual_sim:,.2f}**")

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
            pais_escalador_for_scenarios = st.session_state.get('ultimo_escalador_pais', 1.0)
            base_features_for_scenario = st.session_state.get('last_input_features', property_base_actual_dict_sim) # Debe ser un dict

            try:
                price_base_final_scenario = price_base_actual_sim # Precio de la prop. base ya calculado en interactivo

                # Escenario +1 Baño
                prop_plus_bathroom = base_features_for_scenario.copy()
                prop_plus_bathroom['bathrooms'] = base_features_for_scenario.get('bathrooms', 1) + 1
                df_plus_bathroom = pd.DataFrame([prop_plus_bathroom])[EXPECTED_MODEL_FEATURES]
                pred_raw_plus_bathroom = modelo.predict(df_plus_bathroom)[0]
                price_plus_bathroom_final = pred_raw_plus_bathroom * pais_escalador_for_scenarios * 2.0 

                # Escenario +1 Habitación
                prop_plus_bedroom = base_features_for_scenario.copy()
                prop_plus_bedroom['bedrooms'] = base_features_for_scenario.get('bedrooms', 1) + 1
                df_plus_bedroom = pd.DataFrame([prop_plus_bedroom])[EXPECTED_MODEL_FEATURES]
                pred_raw_plus_bedroom = modelo.predict(df_plus_bedroom)[0]
                price_plus_bedroom_final = pred_raw_plus_bedroom * pais_escalador_for_scenarios * 1.5

                # Escenario +1 Baño y +1 Habitación
                prop_plus_both = base_features_for_scenario.copy()
                prop_plus_both['bathrooms'] = base_features_for_scenario.get('bathrooms', 1) + 1
                prop_plus_both['bedrooms'] = base_features_for_scenario.get('bedrooms', 1) + 1
                df_plus_both = pd.DataFrame([prop_plus_both])[EXPECTED_MODEL_FEATURES]
                pred_raw_plus_both = modelo.predict(df_plus_both)[0]
                price_plus_both_final = pred_raw_plus_both * pais_escalador_for_scenarios * 2.5
                
                # Para la "Propiedad Base" en la tabla de simulación, necesitamos estimar su predicción "raw"
                # Esto es un poco inverso, pero para consistencia en la tabla de debug:
                raw_pred_base_estimada = 0
                last_prop_mult_interactive = st.session_state.get('ultimo_escalador_pais_i', 1.0)
                if pais_escalador_for_scenarios != 0 and last_prop_mult_interactive != 0:
                    raw_pred_base_estimada = price_base_final_scenario / (pais_escalador_for_scenarios * last_prop_mult_interactive)
                else: # Evitar división por cero si algo raro pasa
                    raw_pred_base_estimada = price_base_final_scenario # O alguna otra estimación


                nights_per_year = 365 * occupancy_rate_input
                scenarios_data = [
                    {'name': 'Propiedad Base', 'price_per_night': price_base_final_scenario, 'remodeling_cost': 0, 
                     'raw_model_pred': raw_pred_base_estimada, 'improvement_mult': last_prop_mult_interactive},
                    {'name': 'Más 1 Baño', 'price_per_night': price_plus_bathroom_final, 'remodeling_cost': cost_bathroom_input, 
                     'raw_model_pred': pred_raw_plus_bathroom, 'improvement_mult': 2.0},
                    {'name': 'Más 1 Habitación', 'price_per_night': price_plus_bedroom_final, 'remodeling_cost': cost_bedroom_input, 
                     'raw_model_pred': pred_raw_plus_bedroom, 'improvement_mult': 1.5},
                    {'name': 'Más 1 Baño y 1 Habitación', 'price_per_night': price_plus_both_final, 'remodeling_cost': cost_bathroom_input + cost_bedroom_input, 
                     'raw_model_pred': pred_raw_plus_both, 'improvement_mult': 2.5}
                ]
                
                results_list = []
                for scenario in scenarios_data:
                    income = scenario['price_per_night'] * nights_per_year
                    operational_cost = income * operational_cost_rate_input
                    ganancia_neta_anual = income - operational_cost
                    costo_inversion_total = property_cost_input + scenario['remodeling_cost']
                    roi = (ganancia_neta_anual / costo_inversion_total) * 100 if costo_inversion_total > 0 else 0
                    results_list.append({
                        'Escenario': scenario['name'],
                        'Precio por Noche (COP)': scenario['price_per_night'],
                        'Costo Remodelación (COP)': scenario['remodeling_cost'],
                        'Ingreso Anual Bruto (COP)': income,
                        'Costo Operativo Anual (COP)': operational_cost,
                        'Utilidad Anual (COP)': ganancia_neta_anual,
                        'Inversión Total (COP)': costo_inversion_total,
                        'ROI (%)': roi,
                        '(Info) Pred. Modelo Bruta Aprox.': scenario['raw_model_pred'],
                        '(Info) Mult. País Usado': pais_escalador_for_scenarios,
                        '(Info) Mult. Mejora Aplicado': scenario['improvement_mult']
                    })
                results_df = pd.DataFrame(results_list)
                st.subheader("Resultados de Simulación y Rentabilidad")
                
                cols_to_format_currency = ['Precio por Noche (COP)', 'Costo Remodelación (COP)', 
                                           'Ingreso Anual Bruto (COP)', 'Costo Operativo Anual (COP)', 
                                           'Utilidad Anual (COP)', 'Inversión Total (COP)',
                                           '(Info) Pred. Modelo Bruta Aprox.']
                for col_curr in cols_to_format_currency:
                    results_df[col_curr] = results_df[col_curr].apply(lambda x: f"COP {x:,.2f}")
                
                results_df['ROI (%)'] = results_df['ROI (%)'].apply(lambda x: f"{x:.2f}%")
                results_df['(Info) Mult. País Usado'] = results_df['(Info) Mult. País Usado'].apply(lambda x: f"{x:.2f}x")
                results_df['(Info) Mult. Mejora Aplicado'] = results_df['(Info) Mult. Mejora Aplicado'].apply(lambda x: f"{x:.2f}x")

                st.dataframe(results_df, use_container_width=True)

                # ... (gráficos sin cambios) ...
                results_df_numeric = pd.DataFrame(results_list) # Usar datos numéricos para gráficos
                col_chart1, col_chart2 = st.columns(2)
                with col_chart1:
                    fig_precios = px.bar(results_df_numeric, x='Escenario', y='Precio por Noche (COP)',
                                        title='Precio por Noche Predicho por Escenario',
                                        labels={'Precio por Noche (COP)': 'Precio por Noche (COP)', 'Escenario': 'Escenario'},
                                        color='Escenario', text_auto='.2s')
                    fig_precios.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig_precios, use_container_width=True)
                with col_chart2:
                    fig_roi = px.bar(results_df_numeric, x='Escenario', y='ROI (%)',
                                    title='Retorno sobre la Inversión (ROI) por Escenario',
                                    labels={'ROI (%)': 'ROI Anual (%)', 'Escenario': 'Escenario'},
                                    color='Escenario', text_auto='.2f')
                    fig_roi.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig_roi, use_container_width=True)

            except KeyError as e:
                st.error(f"Error de KeyError en simulación: {e}. Verifique 'EXPECTED_MODEL_FEATURES' y datos de escenario.")
            except Exception as e:
                st.error(f"Error durante la simulación de escenarios: {e}")
        else:
            st.warning("El modelo no está cargado.")

# --- Pie de Página ---
st.markdown("---")
st.markdown("Dashboard POC desarrollado con Streamlit. Modelo de Regresión de Precios Airbnb, parcial tercer corte Metricas de Desempeño.")
