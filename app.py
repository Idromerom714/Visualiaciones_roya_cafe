"""
app.py - Dashboard interactivo (Streamlit)
- Seleccionar municipio
- Seleccionar variable climática / fitosanitaria
- Sugerencias de tipo de gráfico por variable
- Mostrar indicador de riesgo (gauge)
- Mostrar mapa de zonas si existe mapa_zonas.py (opcional)
- NUEVO: Mapa de calor de correlación (Clima vs Riesgo)
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# --- 1. CONFIGURACIÓN INICIAL Y CARGA DE DATOS ---

st.set_page_config(layout="wide", page_title="Dashboard de Riesgo Climático del Café")

# NOTA: Las funciones de carga y limpieza se asumen iguales a las del código previo
# para simplificar. Debes asegurarte de que las columnas están limpias.

@st.cache_data
def load_data():
    try:
        # Cargar archivos originales (asumiendo que están en el mismo directorio)
        df_clima = pd.read_csv("data/clima_ubicaciones.csv")
        df_ubicaciones = pd.read_csv("data/ubicaciones.csv")
        df_enfermedades = pd.read_csv("data/enfermedades.csv")

        # Limpieza de columnas (basado en el pre-procesamiento anterior)
        df_clima.columns = ['latitud', 'longitud', 'fecha_hora', 'temperatura', 'humedad_relativa', 'precipitacion', 'radiacion_solar', 'humeda']
        df_clima['fecha_hora'] = pd.to_datetime(df_clima['fecha_hora'])
        
        df_ubicaciones.columns = df_ubicaciones.columns.str.strip().str.replace(' ', '_').str.replace('(', '').str.replace(')', '').str.replace('.', '').str.replace(',', '').str.replace('°C', '')
        df_ubicaciones['Altitud_m_s_n_m'] = df_ubicaciones['Altitud_m_s_n_m'].str.replace('.', '', regex=False).str.replace(',', '.', regex=False).astype(float)
        
        df_enfermedades.columns = ['Enfermedad', 'Patogeno_Causante', 'T_min', 'T_max', 'HR_min', 'Frecuencia_Lluvia']
        
        # Unir Clima y Ubicaciones para tener el nombre de la Hacienda y la Altitud
        df_merged = pd.merge(df_clima, 
                            df_ubicaciones[['latitud', 'longitud', 'Hacienda', 'Altitud_m_s_n_m']], 
                            on=['latitud', 'longitud'], 
                            how='left')
        
        return df_merged, df_enfermedades, df_ubicaciones

    except FileNotFoundError:
        st.error("Error al cargar los archivos CSV. Asegúrate de que 'clima_ubicaciones.csv', 'ubicaciones.csv' y 'enfermedades.csv' están en el directorio correcto.")
        return None, None, None

df_merged, df_enfermedades, df_ubicaciones = load_data()
umbrales = df_enfermedades.set_index('Enfermedad')

if df_merged is None:
    st.stop()


# --- 2. FUNCIÓN DE CÁLCULO DE INDICADORES (Necesaria para reactividad) ---

def calculate_indicators(df_filtered):
    """Calcula NHF y GD para el DataFrame de clima filtrado."""
    
    df_results = df_filtered[['latitud', 'longitud', 'Hacienda']].drop_duplicates()
    
    # --- A. Indicador Fúngico: Roya (NHF - Horas Favorable) ---
    T_min_roya = umbrales.loc['Roya del café', 'T_min']
    T_max_roya = umbrales.loc['Roya del café', 'T_max']
    HR_min_roya = umbrales.loc['Roya del café', 'HR_min']

    df_filtered['riesgo_roya'] = (
        (df_filtered['temperatura'] >= T_min_roya) & 
        (df_filtered['temperatura'] <= T_max_roya) & 
        (df_filtered['humedad_relativa'] >= HR_min_roya) & 
        (df_filtered['humeda'] == 1)
    ).astype(int)
    
    nhf_roya = df_filtered.groupby(['latitud', 'longitud', 'Hacienda'])['riesgo_roya'].sum().reset_index()
    df_results = pd.merge(df_results, nhf_roya[['latitud', 'longitud', 'riesgo_roya']], on=['latitud', 'longitud'], how='left')
    df_results.rename(columns={'riesgo_roya': 'NHF_Roya_Horas'}, inplace=True)
    
    # --- B. Indicador de Plaga: Broca (GD - Grados-Día) ---
    T_base_broca = umbrales.loc['Broca del café (Plaga)', 'T_min'] 
    
    df_filtered['gd_hora_broca'] = np.where(
        df_filtered['temperatura'] > T_base_broca,
        (df_filtered['temperatura'] - T_base_broca) / 24,
        0
    )
    
    gd_broca = df_filtered.groupby(['latitud', 'longitud', 'Hacienda'])['gd_hora_broca'].sum().reset_index()
    df_results = pd.merge(df_results, gd_broca[['latitud', 'longitud', 'gd_hora_broca']], on=['latitud', 'longitud'], how='left')
    df_results.rename(columns={'gd_hora_broca': 'GD_Broca_Acumulado'}, inplace=True)
    
    # Merge Altitud para el análisis de correlación
    df_results = pd.merge(df_results, df_ubicaciones[['latitud', 'longitud', 'Altitud_m_s_n_m']], 
                         on=['latitud', 'longitud'], how='left')

    return df_results


# --- 3. DISEÑO DEL DASHBOARD EN STREAMLIT ---

st.title("🌱 Dashboard de Indicadores de Riesgo Climático del Café")
st.markdown("Herramienta para evaluar el riesgo de patógenos según variables climáticas históricas.")

# --- BARRA LATERAL DE FILTROS ---
st.sidebar.header("Filtros de Análisis")

# A. Filtro de Ubicaciones
unique_haciendas = df_merged['Hacienda'].unique().tolist()
selected_haciendas = st.sidebar.multiselect(
    "1. Seleccionar Ubicaciones",
    options=unique_haciendas,
    default=unique_haciendas[0] # Selecciona la primera por defecto
)

# B. Filtro de Patógeno
patogeno_options = {
    'Roya del café (NHF)': 'NHF_Roya_Horas',
    'Broca del café (GD)': 'GD_Broca_Acumulado'
    # Se pueden añadir más aquí, ej. Ojo de Gallo
}
selected_patogeno_name = st.sidebar.selectbox(
    "2. Seleccionar Patógeno/Indicador",
    options=list(patogeno_options.keys())
)
selected_indicator_col = patogeno_options[selected_patogeno_name]


# C. Filtro de Rango de Fechas (REACTIVO)
min_date = df_merged['fecha_hora'].min().date()
max_date = df_merged['fecha_hora'].max().date()

date_range = st.sidebar.date_input(
    "3. Seleccionar Rango de Fechas",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# Aplicar filtro de fecha y ubicación al DataFrame
if len(date_range) == 2:
    start_date = pd.to_datetime(date_range[0])
    end_date = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1) # Incluir el día final
    
    df_filtered_date = df_merged[
        (df_merged['fecha_hora'] >= start_date) & 
        (df_merged['fecha_hora'] < end_date)
    ]
    df_filtered_final = df_filtered_date[df_filtered_date['Hacienda'].isin(selected_haciendas)].copy()
else:
    # Caso donde solo se selecciona una fecha (o rango incompleto)
    st.warning("Selecciona un rango completo de fechas para el análisis.")
    st.stop()
    
# Recalcular Indicadores con los datos filtrados
df_indicators = calculate_indicators(df_filtered_final)


# --- 4. SECCIÓN DE INDICADORES (KPIs) ---

st.header(f"Resultados Agregados para {selected_patogeno_name}")
col1, col2, col3 = st.columns(3)

# Indicador 1: Riesgo Máximo
max_risk = df_indicators[selected_indicator_col].max()
col1.metric(
    label=f"Máximo Riesgo ({selected_indicator_col.split('_')[0]})", 
    value=f"{max_risk:,.2f}"
)

# Indicador 2: Riesgo Promedio
avg_risk = df_indicators[selected_indicator_col].mean()
col2.metric(
    label=f"Promedio de Riesgo", 
    value=f"{avg_risk:,.2f}"
)

# Indicador 3: Ubicación de Mayor Riesgo
max_risk_location = df_indicators.loc[df_indicators[selected_indicator_col].idxmax(), 'Hacienda']
col3.metric(
    label="Ubicación más Afectada", 
    value=max_risk_location
)

st.divider()

# --- 5. SECCIÓN DE GRÁFICOS (Gráficos Reactivos) ---

st.header("Análisis de Tendencia y Distribución")

# Sugerencias de Gráficos según el indicador (Cumpliendo el requisito)

# A. Gráfico de Tendencia de la variable climática principal
if selected_patogeno_name in ['Roya del café (NHF)']:
    # Para Roya, Temperatura y Humedad Relativa son clave. Graficamos Temperatura diaria promedio.
    df_daily_temp = df_filtered_final.groupby([df_filtered_final['fecha_hora'].dt.date, 'Hacienda'])['temperatura'].mean().reset_index()
    fig_line = px.line(
        df_daily_temp, 
        x='fecha_hora', 
        y='temperatura', 
        color='Hacienda', 
        title='Temperatura Promedio Diaria vs. Umbrales de Roya (18-25°C)'
    )
    fig_line.add_hrect(y0=18, y1=25, line_width=0, fillcolor="red", opacity=0.1, annotation_text="Rango Óptimo Roya")
    st.plotly_chart(fig_line, use_container_width=True)

elif selected_patogeno_name in ['Broca del café (GD)']:
    # Para Broca, el GD es acumulativo. Graficamos la acumulación diaria de GD.
    df_daily_gd = df_filtered_final.groupby([df_filtered_final['fecha_hora'].dt.date, 'Hacienda'])['gd_hora_broca'].sum().reset_index()
    df_daily_gd['GD_Acumulado'] = df_daily_gd.groupby('Hacienda')['gd_hora_broca'].cumsum()
    
    fig_cum = px.line(
        df_daily_gd, 
        x='fecha_hora', 
        y='GD_Acumulado', 
        color='Hacienda', 
        title='Acumulación de Grados-Día (GD) para Broca'
    )
    st.plotly_chart(fig_cum, use_container_width=True)

# B. Gráfico de Ranking del Indicador
fig_bar = px.bar(
    df_indicators.sort_values(selected_indicator_col, ascending=False), 
    x='Hacienda', 
    y=selected_indicator_col, 
    color='Altitud_m_s_n_m', 
    title=f"Ranking de Riesgo por Ubicación ({selected_indicator_col})",
    color_continuous_scale=px.colors.sequential.Sunset,
    hover_data=['Altitud_m_s_n_m']
)
st.plotly_chart(fig_bar, use_container_width=True)


# --- 6. MATRIZ DE CORRELACIÓN ---

st.header("Correlación de Riesgo y Variables Climáticas")
st.markdown(f"Matriz de correlación de **{selected_indicator_col}** con las variables ambientales agregadas por ubicación.")

# 1. Preparar DataFrame de correlación
# Agregamos variables climáticas por ubicación (Media y Suma)
df_clima_aggregated = df_filtered_final.groupby(['latitud', 'longitud', 'Hacienda']).agg(
    T_Media=('temperatura', 'mean'),
    HR_Media=('humedad_relativa', 'mean'),
    P_Suma=('precipitacion', 'sum'),
    T_Std=('temperatura', 'std') # Desviación estándar de T
).reset_index()

# 2. Unir con los Indicadores
df_corr = pd.merge(df_indicators, df_clima_aggregated, on=['latitud', 'longitud', 'Hacienda'], how='inner')

# 3. Seleccionar variables para la matriz
corr_vars = [selected_indicator_col, 'Altitud_m_s_n_m', 'T_Media', 'HR_Media', 'P_Suma', 'T_Std']
corr_matrix = df_corr[corr_vars].corr()

# 4. Graficar la Matriz
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(
    corr_matrix, 
    annot=True, 
    fmt=".2f", 
    cmap='coolwarm', 
    cbar_kws={'label': 'Coeficiente de Correlación'},
    ax=ax
)
plt.title(f'Matriz de Correlación con {selected_indicator_col}')
st.pyplot(fig)

# Interpretación de la correlación con la Altitud
alt_corr = corr_matrix.loc[selected_indicator_col, 'Altitud_m_s_n_m']
st.info(
    f"💡 **Correlación con Altitud:** El coeficiente de correlación de {selected_indicator_col} con la Altitud es **{alt_corr:.2f}**."
    f" Un valor negativo (típico) indica que a mayor altitud, menor es el riesgo."
)