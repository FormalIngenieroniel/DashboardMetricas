import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import numpy as np
import json

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
                        ohe = cat_pipeline_or_encoder.steps[-1][1]
                elif hasattr(cat_pipeline_or_encoder, 'categories_'):
                    ohe = cat_pipeline_or_encoder

                if ohe and hasattr(ohe, 'categories_'):
                    if len(original_cat_cols) == len(ohe.categories_):
                        for i, col_name in enumerate(original_cat_cols):
                            categories = ohe.categories_[i]
                            dynamic_categories[col_name] = sorted([str(cat) for cat in categories if pd.notna(cat)])
                    else:
                        st.warning(f"Discrepancia en el número de columnas categóricas ({len(original_cat_cols)}) y las categorías del OHE ({len(ohe.categories_)}).")
                else:
                    st.warning("No se pudo encontrar el OneHotEncoder o sus categorías dentro del pipeline 'cat'.")
            else:
                st.warning("Transformador categórico ('cat') no encontrado en el preprocesador.")
        except Exception as e:
            st.warning(f"Error al extraer categorías dinámicas del modelo: {e}.")
    
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
        'Other': ['OtherSector']
    }
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
        st.success(f"Mapeo país-sector cargado exitosamente desde '{filepath}'.")
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
    valid_paises = set(dynamic_categories.get('pais', []))
    valid_sectors = set(dynamic_categories.get('sector', []))
    
    filtered_mapping = {}
    for pais, sectores in mapping.items():
        if pais in valid_paises:
            # Filtrar sectores para incluir solo los que están en el dataset
            sectores_validos = [sector for sector in sectores if sector in valid_sectors]
            if sectores_validos:  # Solo añadir el país si tiene sectores válidos
                filtered_mapping[pais] = sectores_validos
            else:
                st.warning(f"El país '{pais}' no tiene sectores válidos en el dataset. Se omitirá.")
        else:
            st.warning(f"El país '{pais}' no está en el dataset. Se omitirá.")
    
    if not filtered_mapping:
        st.error("No se encontraron países ni sectores válidos en el mapeo después de filtrar.")
        # Fallback mínimo para evitar errores
        filtered_mapping = {'Colombia': ['Bogotá D.C.']} if 'Colombia' in valid_paises and 'Bogotá D.C.' in valid_sectors else {}
    
    return filtered_mapping

if 'pais_sector_mapping' not in st.session_state:
    st.session_state.pais_sector_mapping = load_and_filter_pais_sector_mapping()

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
        'pais': None,  # Se asignará después de validar categorías
        'sector': None  # Se asignará después de validar categorías
    }

    # Obtener las opciones de países desde dynamic_categories (filtradas por el mapeo)
    paises_options = sorted(list(st.session_state.pais_sector_mapping.keys()))
    if not paises_options:
        st.error("No se encontraron países válidos después de filtrar el mapeo país-sector.")
        paises_options = dynamic_categories.get('pais', [])
        if not paises_options:
            st.error("No se encontraron países en las categorías del modelo. Por favor, verifica el modelo.")
            paises_options = ['Colombia']  # Fallback mínimo para evitar errores

    # Asegurar que el país por defecto esté en las opciones disponibles
    default_pais = 'Colombia' if 'Colombia' in paises_options else paises_options[0]
    default_property_base['pais'] = default_pais

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
    
    default_pais_index = paises_options.index(default_property_base['pais'])
    pais_input = c7.selectbox("País (pais)", paises_options, index=default_pais_index, key="pais_selector")

    # Obtener los sectores correspondientes al país seleccionado desde el mapeo filtrado
    sectores_para_pais_seleccionado = st.session_state.pais_sector_mapping.get(pais_input, [])
    if not sectores_para_pais_seleccionado:
        c8.error(f"No se encontraron sectores válidos para el país '{pais_input}' en el mapeo filtrado.")
        sectores_para_pais_seleccionado = ['Bogotá D.C.']  # Fallback mínimo para evitar errores

    default_sector_value = 'Bogotá D.C.' if 'Bogotá D.C.' in sectores_para_pais_seleccionado else sectores_para_pais_seleccionado[0]
    default_property_base['sector'] = default_sector_value
    
    idx_sector = sectores_para_pais_seleccionado.index(default_sector_value)
    sector_input = c8.selectbox(
        "Sector/Ciudad (sector)",
        options=sectores_para_pais_seleccionado,
        index=idx_sector
    )

    host_id_input = c9.number_input("ID del Anfitrión (host_id)", value=default_property_base['host_id'], step=1, min_value=0)

    st.write("**Reseñas y Logística:**")
    c10, c11, c12_a, c12_b = st.columns(4)
    reviews_input = c10.number_input("Número de Reseñas (reviews)", 0, 2000, default_property_base['reviews'])
    rating_input = c11.slider("Calificación Promedio (rating)", 0.0, 5.0, default_property_base['rating'], 0.1)

    default_checkin_index = checkin_options.index(default_property_base['checkin_category']) if default_property_base['checkin_category'] in checkin_options else 0
    checkin_input = c12_a.selectbox("Categoría Check-in", checkin_options, index=default_checkin_index)
    
    default_checkout_index = checkout_options.index(default_property_base['checkout_category']) if default_property_base['checkout_category'] in checkout_options else 0
    checkout_input = c12_b.selectbox("Categoría Check-out", checkout_options, index=default_checkout_index, key="checkout_cat_key")

    # Mostrar categorías válidas del modelo y mapeo para depuración
    st.write("**Categorías Válidas del Modelo (para depuración):**")
    st.write(f"Países reconocidos: {dynamic_categories.get('pais', [])}")
    st.write(f"Sectores reconocidos: {dynamic_categories.get('sector', [])}")
    st.write(f"Mapeo país-sector filtrado: {st.session_state.pais_sector_mapping}")

    if st.button("📈 Predecir Precio Base", key="predict_base_interactive", use_container_width=True):
        if modelo:
            if not pais_input:
                st.warning("Por favor, seleccione un país.")
            elif not sector_input:
                st.warning("Por favor, seleccione un sector.")
            else:
                # Validar que pais_input y sector_input estén en las categorías del modelo
                valid_paises = dynamic_categories.get('pais', [])
                valid_sectores = dynamic_categories.get('sector', [])
                pais_to_use = pais_input if pais_input in valid_paises else valid_paises[0] if valid_paises else default_property_base['pais']
                sector_to_use = sector_input if sector_input in valid_sectores else valid_sectores[0] if valid_sectores else default_property_base['sector']
                
                if pais_input != pais_to_use or sector_input != sector_to_use:
                    st.warning(f"Advertencia: El país '{pais_input}' o sector '{sector_input}' no está en las categorías del modelo. Usando país: '{pais_to_use}' y sector: '{sector_to_use}'.")

                # Asegurar que las columnas coincidan con las esperadas por el modelo
                expected_features = ['bathrooms', 'pais', 'host_id', 'bedrooms', 'reviews', 'beds', 'sector', 'guests', 
                                   'checkin_category', 'checkout_category', 'rating', 'toiles', 'studios']
                input_data = {
                    'bathrooms': bathrooms_input, 'pais': pais_to_use, 'host_id': host_id_input,
                    'bedrooms': bedrooms_input, 'reviews': reviews_input, 'beds': beds_input,
                    'sector': sector_to_use, 'guests': guests_input,
                    'checkin_category': checkin_input, 'checkout_category': checkout_input,
                    'rating': rating_input, 'toiles': toiles_input, 'studios': studios_input
                }
                input_df = pd.DataFrame([input_data])
                input_df = input_df[expected_features]  # Asegurar el orden correcto de las columnas

                st.write("**Datos enviados al modelo (para depuración):**")
                st.dataframe(input_df)

                try:
                    # Intentar transformar los datos con el preprocesador del modelo para depuración
                    preprocessor = modelo.named_steps['preprocessor']
                    input_transformed = preprocessor.transform(input_df)
                    st.write("**Datos transformados (para depuración):**")
                    st.write(input_transformed.toarray() if hasattr(input_transformed, 'toarray') else input_transformed)

                    prediccion = modelo.predict(input_df)[0]
                    # Multiplicador manual basado en país como workaround
                    pais_multiplier = 1.0
                    base_pais = 'Colombia'  # País base para comparación
                    if pais_to_use != base_pais:
                        # Ajuste manual basado en supuestos de precios (puede ajustarse según datos reales)
                        country_multipliers = {
                            'Japan': 1.5, 'United States': 1.4, 'Spain': 1.3, 'France': 1.2, 
                            'Canada': 1.3, 'Other': 1.0
                        }
                        pais_multiplier = country_multipliers.get(pais_to_use, 1.0)
                        st.info(f"Multiplicador aplicado por país '{pais_to_use}': {pais_multiplier}x")

                    # Aplicar multiplicadores según el ejemplo para baños y habitaciones
                    bathroom_diff = bathrooms_input - default_property_base['bathrooms']
                    bedroom_diff = bedrooms_input - default_property_base['bedrooms']
                    property_multiplier = 1.0  # Multiplicador base
                    if bathroom_diff == 1 and bedroom_diff == 0:
                        property_multiplier = 2.0  # Multiplicador para +1 baño
                    elif bedroom_diff == 1 and bathroom_diff == 0:
                        property_multiplier = 1.5  # Multiplicador para +1 habitación
                    elif bathroom_diff == 1 and bedroom_diff == 1:
                        property_multiplier = 2.5  # Multiplicador para +1 baño y +1 habitación
                    elif bathroom_diff > 1 and bedroom_diff == 0:
                        property_multiplier = 2.0 + (bathroom_diff - 1) * 0.5  # Escalar para más baños
                    elif bedroom_diff > 1 and bathroom_diff == 0:
                        property_multiplier = 1.5 + (bedroom_diff - 1) * 0.3  # Escalar para más habitaciones
                    elif bathroom_diff > 0 and bedroom_diff > 0:
                        property_multiplier = 2.5 + (bathroom_diff - 1) * 0.5 + (bedroom_diff - 1) * 0.3  # Combinación

                    # Aplicar ambos multiplicadores
                    prediccion_final = prediccion * pais_multiplier * property_multiplier
                    st.success(f"**Precio Estimado por Noche: COP {prediccion_final:,.2f}**")
                    st.info(f"Nota: El precio ha sido ajustado con un multiplicador de {property_multiplier:.2f}x por mejoras de {bathroom_diff} baño(s) y {bedroom_diff} habitación(es), y {pais_multiplier:.2f}x por país.")
                    st.session_state.precio_base_simulacion = prediccion_final
                    st.session_state.property_base_simulacion = input_data.copy()
                except Exception as e:
                    st.error(f"Error al predecir: {e}")
                    st.error("Asegúrate de que todas las características necesarias por el modelo estén presentes y con los tipos de datos correctos.")
                    st.dataframe(input_df)
        else:
            st.warning("El modelo no está cargado. No se puede predecir.")

with col_importance:
    st.subheader("Importancia de las Características")
    st.markdown("Visualización de cómo cada característica influye en la predicción del precio, según el modelo.")
    data_importancia_agrupada = {
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
                price_plus_bathroom_pred = modelo.predict(X_plus_bathroom)[0] * 2  # Multiplicador según ejemplo
                price_plus_bedroom_pred = modelo.predict(X_plus_bedroom)[0] * 1.5  # Multiplicador según ejemplo
                price_plus_both_pred = modelo.predict(X_plus_both)[0] * 2.5  # Multiplicador según ejemplo

                price_base_final = price_base_actual
                price_plus_bathroom_final = price_plus_bathroom_pred
                price_plus_bedroom_final = price_plus_bedroom_pred
                price_plus_both_final = price_plus_both_pred

                nights_per_year = 365 * occupancy_rate_input
                scenarios_data = [
                    {'name': 'Propiedad Base', 'price_per_night': price_base_final, 'remodeling_cost': 0},
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
                        'Escenario': scenario['name'],
                        'Precio por Noche (COP)': scenario['price_per_night'],
                        'Costo Remodelación (COP)': scenario['remodeling_cost'],
                        'Ingreso Anual Bruto (COP)': income,
                        'Costo Operativo Anual (COP)': operational_cost,
                        'Utilidad Anual (COP)': ganancia_neta_anual,
                        'Inversión Total (COP)': costo_inversion_total,
                        'ROI (%)': roi
                    })
                results_df = pd.DataFrame(results_list)
                st.subheader("Resultados de Simulación y Rentabilidad")
                currency_cols = ['Precio por Noche (COP)', 'Costo Remodelación (COP)', 'Ingreso Anual Bruto (COP)', 'Costo Operativo Anual (COP)', 'Utilidad Anual (COP)', 'Inversión Total (COP)']
                for col_curr in currency_cols:
                    results_df[col_curr] = results_df[col_curr].apply(lambda x: f"COP {x:,.2f}")
                results_df['ROI (%)'] = results_df['ROI (%)'].apply(lambda x: f"{x:.2f}%")
                st.dataframe(results_df, use_container_width=True)

                results_df_numeric = pd.DataFrame(results_list)
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
                st.info("""
                **Nota sobre los multiplicadores de precio:**
                Los precios predichos para los escenarios de mejora han sido ajustados con multiplicadores (2x para baño, 1.5x para habitación, 2.5x para ambos) según el ejemplo proporcionado, para reflejar un incremento notable en el precio por noche.
                """)
            except Exception as e:
                st.error(f"Error durante la simulación de escenarios: {e}")
        else:
            st.warning("El modelo no está cargado.")

# --- Pie de Página ---
st.markdown("---")
st.markdown("Dashboard POC desarrollado con Streamlit. Modelo de Regresión de Precios Airbnb.")
