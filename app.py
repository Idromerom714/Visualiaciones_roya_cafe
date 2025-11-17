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
import plotly.graph_objects as go
import importlib.util
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Logit, NegativeBinomial # Se añade NegativeBinomial

# --- 1. CONFIGURACIÓN INICIAL Y CARGA DE DATOS ---

st.set_page_config(layout="wide", page_title="Dashboard de Riesgo Climático del Café")

@st.cache_data
def load_data():
    """Carga y pre-procesa los datos."""
    try:
        df_clima = pd.read_csv("data/clima_ubicaciones.csv")
        df_ubicaciones = pd.read_csv("ubicaciones.csv")
        df_enfermedades = pd.read_csv("data/enfermedades.csv")

        # Limpieza de columnas de clima
        df_clima.columns = ['latitud', 'longitud', 'fecha_hora', 'temperatura', 
                            'humedad_relativa', 'precipitacion', 'radiacion_solar', 'humeda']
        df_clima['fecha_hora'] = pd.to_datetime(df_clima['fecha_hora'])
        
        # Limpieza y tipado en df_ubicaciones
        df_ubicaciones.columns = df_ubicaciones.columns.str.strip().str.replace(' ', '_').str.replace('(', '').str.replace(')', '').str.replace('.', '', regex=False).str.replace(',', '.', regex=False).str.replace('°C', '')
        # Se asume que el nombre de la columna de altitud es Altitud_m_s_n_m después de la limpieza
        df_ubicaciones.rename(columns={'Altitud_m_s._n._m': 'Altitud_m_s_n_m'}, inplace=True)
        df_ubicaciones['Altitud_m_s_n_m'] = pd.to_numeric(df_ubicaciones['Altitud_m_s_n_m'], errors='coerce')
        
        df_enfermedades.columns = ['Enfermedad', 'Patogeno_Causante', 'T_min', 'T_max', 'HR_min', 'Frecuencia_Lluvia']
        
        df_merged = pd.merge(df_clima, 
                            df_ubicaciones[['latitud', 'longitud', 'Hacienda', 'Altitud_m_s_n_m']], 
                            on=['latitud', 'longitud'], 
                            how='left')
        
        return df_merged, df_enfermedades, df_ubicaciones

    except FileNotFoundError:
        st.error("Error al cargar los archivos CSV. Asegúrate de que los archivos están en el directorio de ejecución.")
        return None, None, None

df_merged, df_enfermedades, df_ubicaciones = load_data()
umbrales = df_enfermedades.set_index('Enfermedad')

if df_merged is None:
    st.stop()



# --- 2. FUNCIONES DE CÁLCULO Y SIMULACIÓN ---

@st.cache_data
def calculate_indicators(df_filtered, umbrales):
    """Calcula NHF y GD para el DataFrame de clima filtrado."""
    
    df_results = df_filtered[['latitud', 'longitud', 'Hacienda']].drop_duplicates()
    
    # --- A. Indicador Fúngico: Roya (NHF) ---
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
    
    # --- B. Indicador de Plaga: Broca (GD) ---
    T_base_broca = umbrales.loc['Broca del café (Plaga)', 'T_min'] 
    
    df_filtered['gd_hora_broca'] = np.where(
        df_filtered['temperatura'] > T_base_broca,
        (df_filtered['temperatura'] - T_base_broca) / 24,
        0
    )
    
    gd_broca = df_filtered.groupby(['latitud', 'longitud', 'Hacienda'])['gd_hora_broca'].sum().reset_index()
    df_results = pd.merge(df_results, gd_broca[['latitud', 'longitud', 'gd_hora_broca']], on=['latitud', 'longitud'], how='left')
    df_results.rename(columns={'gd_hora_broca': 'GD_Broca_Acumulado'}, inplace=True)
    
    # Merge Altitud para el análisis de correlación y modelo
    df_results = pd.merge(df_results, df_ubicaciones[['latitud', 'longitud', 'Altitud_m_s_n_m']], 
                         on=['latitud', 'longitud'], how='left')

    return df_results


def simulate_incidence(df_indicators):
    """Simula datos binarios de incidencia de Roya basados en NHF y Altitud."""
    np.random.seed(42) 
    # Asegurar que no hay NaNs y crear Altitud en km
    df_indicators['Altitud_km'] = df_indicators['Altitud_m_s_n_m'].fillna(0) / 1000
    df_indicators['NHF_Roya_Horas'] = df_indicators['NHF_Roya_Horas'].fillna(0)

    # Modelo: logit(P) = -3.0 + 0.05*NHF - 1.5*Altitud_km
    linear_predictor = -3.0 + (0.05 * df_indicators['NHF_Roya_Horas']) - (1.5 * df_indicators['Altitud_km'])
    probability = 1 / (1 + np.exp(-linear_predictor))

    # Clip para garantizar valores válidos en [0,1]
    probability = np.clip(probability, 0.0, 1.0)

    # np.random.binomial acepta p como array; pasar solo p (sin size) cuando p es vector
    try:
        simulated = np.random.binomial(n=1, p=probability)
    except Exception:
        # Fallback: usar probabilidad escalar promedio si algo falla
        p_scalar = float(np.nan_to_num(probability.mean(), nan=0.0))
        simulated = np.random.binomial(n=1, p=p_scalar, size=len(df_indicators))

    df_indicators['Incidencia_Roya_Simulada'] = simulated

    return df_indicators

def simulate_count(df_indicators):
    """Simula datos de conteo (Número de Brocas) basados en GD y Altitud."""
    np.random.seed(43) 
    # Asegurar que no hay NaNs y crear Altitud en km
    df_indicators['Altitud_km'] = df_indicators['Altitud_m_s_n_m'].fillna(0) / 1000
    df_indicators['GD_Broca_Acumulado'] = df_indicators['GD_Broca_Acumulado'].fillna(0)

    # Modelo: log(lambda) = 1.0 + 0.1*GD - 0.8*Altitud_km
    linear_predictor_log = 1.0 + (0.1 * df_indicators['GD_Broca_Acumulado']) - (0.8 * df_indicators['Altitud_km'])
    lambda_mean = np.exp(linear_predictor_log)

    # Garantizar valores válidos para lambda (no negativos, no NaN)
    lambda_mean = np.nan_to_num(lambda_mean, nan=0.0)
    lambda_mean = np.clip(lambda_mean, 0.0, None)

    # Convertir a numpy array y generar conteos por elemento
    lam_array = np.asarray(lambda_mean)
    try:
        simulated_counts = np.random.poisson(lam=lam_array)
    except Exception:
        # Fallback: usar media escalar si la generación por elemento falla
        lam_scalar = float(np.mean(lam_array)) if len(lam_array) > 0 else 0.0
        simulated_counts = np.random.poisson(lam=lam_scalar, size=len(df_indicators))

    df_indicators['Conteo_Broca_Simulada'] = simulated_counts

    return df_indicators

def run_logistic_regression(df_model):
    """Ejecuta un modelo de Regresión Logística (Logit) y devuelve el resumen."""
    df_model = df_model.dropna(subset=['NHF_Roya_Horas', 'Altitud_km', 'Incidencia_Roya_Simulada'])
    if df_model.empty: return "Error: Datos insuficientes después de la limpieza."
    
    Y = df_model['Incidencia_Roya_Simulada']
    X = df_model[['NHF_Roya_Horas', 'Altitud_km']]
    X = sm.add_constant(X, prepend=False)
    
    try:
        model = Logit(Y, X)
        result = model.fit(disp=False)
        return result
    except Exception as e:
        return f"Error al ejecutar el modelo Logístico: {e}"

def run_negative_binomial(df_model):
    """Ejecuta un modelo de Regresión Binomial Negativa."""
    df_model = df_model.dropna(subset=['GD_Broca_Acumulado', 'Altitud_km', 'Conteo_Broca_Simulada'])
    if df_model.empty: return "Error: Datos insuficientes después de la limpieza."

    Y = df_model['Conteo_Broca_Simulada']
    X = df_model[['GD_Broca_Acumulado', 'Altitud_km']]
    X = sm.add_constant(X, prepend=False)
    
    try:
        model = NegativeBinomial(Y, X)
        result = model.fit(disp=False)
        return result
    except Exception as e:
        return f"Error al ejecutar el modelo Binomial Negativo: {e}"


# --- 3. DISEÑO DEL DASHBOARD ---

st.title("🌱 Dashboard de Indicadores de Riesgo Climático del Café")
st.markdown("Herramienta para evaluar el riesgo de patógenos según variables climáticas históricas.")

# --- BARRA LATERAL DE FILTROS ---
st.sidebar.header("Filtros de Análisis")

# A. Filtro de Ubicaciones
unique_haciendas = df_merged['Hacienda'].unique().tolist()
selected_haciendas = st.sidebar.multiselect(
    "1. Seleccionar Ubicaciones",
    options=unique_haciendas,
    default=unique_haciendas[0] 
)

# B. Filtro de Patógeno
patogeno_options = {
    'Roya del café (NHF)': 'NHF_Roya_Horas',
    'Broca del café (GD)': 'GD_Broca_Acumulado'
}
selected_patogeno_name = st.sidebar.selectbox(
    "2. Seleccionar Patógeno/Indicador",
    options=list(patogeno_options.keys())
)
selected_indicator_col = patogeno_options[selected_patogeno_name]

# C. Filtro de Rango de Fechas
min_date = df_merged['fecha_hora'].min().date()
max_date = df_merged['fecha_hora'].max().date()

date_range = st.sidebar.date_input(
    "3. Seleccionar Rango de Fechas",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# Aplicar filtro de fecha y ubicación
if len(date_range) == 2:
    start_date = pd.to_datetime(date_range[0])
    end_date = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1)
    
    df_filtered_date = df_merged[
        (df_merged['fecha_hora'] >= start_date) & 
        (df_merged['fecha_hora'] < end_date)
    ]
    df_filtered_final = df_filtered_date[df_filtered_date['Hacienda'].isin(selected_haciendas)].copy()
else:
    st.warning("Selecciona un rango completo de fechas para el análisis.")
    st.stop()
    
# Recalcular Indicadores con los datos filtrados
df_indicators = calculate_indicators(df_filtered_final, umbrales)

# Simular la incidencia/conteo basado en la selección del usuario
if selected_patogeno_name == 'Roya del café (NHF)':
    df_indicators = simulate_incidence(df_indicators) 
elif selected_patogeno_name == 'Broca del café (GD)':
    df_indicators = simulate_count(df_indicators) 


# --- SECCIÓN DE INDICADORES (KPIs) ---

st.header(f"Resultados Agregados para {selected_patogeno_name}")
col1, col2, col3 = st.columns(3)

max_risk = df_indicators[selected_indicator_col].max()
col1.metric(
    label=f"Máximo Riesgo ({selected_indicator_col.split('_')[0]})", 
    value=f"{max_risk:,.2f}"
)

avg_risk = df_indicators[selected_indicator_col].mean()
col2.metric(
    label=f"Promedio de Riesgo", 
    value=f"{avg_risk:,.2f}"
)

max_risk_location = df_indicators.loc[df_indicators[selected_indicator_col].idxmax(), 'Hacienda']
col3.metric(
    label="Ubicación más Afectada", 
    value=max_risk_location
)

st.divider()

# --- SECCIÓN DE GRÁFICOS ---

st.header("Análisis de Tendencia y Distribución")

# Gráfico A: Tendencia de la variable climática principal
if selected_patogeno_name == 'Roya del café (NHF)':
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

elif selected_patogeno_name == 'Broca del café (GD)':
    # Asegurarse de que la columna 'gd_hora_broca' existe (si no, calcularla localmente)
    df_temp = df_filtered_final.copy()
    if 'gd_hora_broca' not in df_temp.columns:
        try:
            T_base_broca = umbrales.loc['Broca del café (Plaga)', 'T_min']
        except Exception:
            T_base_broca = 20  # valor por defecto si no se encuentra en umbrales

        df_temp['gd_hora_broca'] = np.where(
            df_temp['temperatura'] > T_base_broca,
            (df_temp['temperatura'] - T_base_broca) / 24,
            0
        )

    df_daily_gd = df_temp.groupby([df_temp['fecha_hora'].dt.date, 'Hacienda'])['gd_hora_broca'].sum().reset_index()
    df_daily_gd.rename(columns={'fecha_hora': 'date', 'gd_hora_broca': 'gd_hora_broca'}, inplace=True)
    df_daily_gd['GD_Acumulado'] = df_daily_gd.groupby('Hacienda')['gd_hora_broca'].cumsum()

    fig_cum = px.line(
        df_daily_gd, 
        x='date', 
        y='GD_Acumulado', 
        color='Hacienda', 
        title='Acumulación de Grados-Día (GD) para Broca'
    )
    st.plotly_chart(fig_cum, use_container_width=True)

# Gráfico B: Ranking del Indicador
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


# --- 4. MATRIZ DE CORRELACIÓN ---

st.header("Correlación de Riesgo y Variables Climáticas")
st.markdown(f"Matriz de correlación de **{selected_indicator_col}** con las variables ambientales agregadas por ubicación.")

# 1. Preparar DataFrame de correlación
df_clima_aggregated = df_filtered_final.groupby(['latitud', 'longitud', 'Hacienda']).agg(
    T_Media=('temperatura', 'mean'),
    HR_Media=('humedad_relativa', 'mean'),
    P_Suma=('precipitacion', 'sum'),
    T_Std=('temperatura', 'std')
).reset_index()
df_corr = pd.merge(df_indicators, df_clima_aggregated, on=['latitud', 'longitud', 'Hacienda'], how='inner')

# 2. Verificación Lógica (Requiere al menos dos puntos para variabilidad)
if len(selected_haciendas) < 2:
    st.warning("⚠️ **Advertencia de Correlación:** Selecciona **al menos dos ubicaciones** para calcular la Matriz de Correlación. La correlación requiere variabilidad espacial entre fincas.")
else:
    corr_vars = [selected_indicator_col, 'Altitud_m_s_n_m', 'T_Media', 'HR_Media', 'P_Suma', 'T_Std']
    corr_matrix = df_corr[corr_vars].corr()

    # 3. Graficar la Matriz
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

    # 4. Interpretación de la correlación con la Altitud
    alt_corr = corr_matrix.loc[selected_indicator_col, 'Altitud_m_s_n_m']
    st.info(
        f"💡 **Correlación con Altitud:** El coeficiente de correlación de **{selected_indicator_col}** con la Altitud es **{alt_corr:.2f}**."
        f" Esto es clave para determinar las zonas agroecológicas de mayor o menor riesgo."
    )


# --- 5. MODELADO ESTADÍSTICO (Validación del Indicador) ---

st.divider()
st.header("🔬 Validación del Indicador de Riesgo (Modelo Estadístico)")

if len(df_indicators) < 5:
    st.warning("Se necesitan **más de 5 ubicaciones** seleccionadas para que el análisis de regresión sea estadísticamente estable.")
else:
    if selected_patogeno_name == 'Roya del café (NHF)':
        st.markdown("Se utiliza **Regresión Logística** para modelar la probabilidad de **Incidencia de Roya (Simulada)**.")
        result = run_logistic_regression(df_indicators)

        if isinstance(result, str):
            st.error(f"Error al ejecutar el modelo: {result}")
        else:
            col_m1, col_m2 = st.columns([1, 2])
            
            # Columna 1: Resumen del Modelo
            col_m1.subheader("Resumen del Modelo Logístico")
            col_m1.dataframe(pd.DataFrame({
                'Métrica': ['Observaciones', 'Pseudo R-cuadrado', 'Log-Verosimilitud'],
                'Valor': [result.nobs, f"{result.prsquared:.3f}", f"{result.llf:.2f}"]
            }).set_index('Métrica'))
            
            # Columna 2: Coeficientes e Interpretación
            coef_df = result.summary2().tables[1]
            coef_df = coef_df[['Coef.', 'Std.Err.', 'P>|z|']]
            coef_df.columns = ['Coeficiente', 'Error Estándar', 'P-Valor']
            
            col_m2.subheader("Coeficientes del Modelo")
            col_m2.dataframe(coef_df, use_container_width=True)
            
            st.markdown("#### Interpretación Clave (Logística)")
            p_nhf = coef_df.loc['NHF_Roya_Horas', 'P-Valor']
            coef_nhf = coef_df.loc['NHF_Roya_Horas', 'Coeficiente']
            st.info(f"**Riesgo Climático (NHF_Roya_Horas):** Coeficiente: **{coef_nhf:.3f}** | P-Valor: **{p_nhf:.3f}**.")
            st.info(f"**Altitud (Altitud_km):** Coeficiente: **{coef_df.loc['Altitud_km', 'Coeficiente']:.3f}** | P-Valor: **{coef_df.loc['Altitud_km', 'P-Valor']:.3f}**.")


    elif selected_patogeno_name == 'Broca del café (GD)':
        st.markdown("Se utiliza **Regresión Binomial Negativa** para modelar el **Conteo de Brocas (Simulado)**.")
        result = run_negative_binomial(df_indicators)
        
        if isinstance(result, str):
            st.error(f"Error al ejecutar el modelo: {result}")
        else:
            col_m1, col_m2 = st.columns([1, 2])
            
            # Columna 1: Resumen del Modelo
            col_m1.subheader("Resumen del Modelo Binomial Negativo")
            col_m1.dataframe(pd.DataFrame({
                'Métrica': ['Observaciones', 'Log-Verosimilitud', 'Alpha (Sobredispersión)'],
                'Valor': [result.nobs, f"{result.llf:.2f}", f"{result.params.get('alpha', 0.0):.3f}"]
            }).set_index('Métrica'))
            
            # Columna 2: Coeficientes e Interpretación
            coef_df = result.summary2().tables[1]
            coef_df = coef_df[['Coef.', 'Std.Err.', 'P>|z|']]
            coef_df.columns = ['Coeficiente', 'Error Estándar', 'P-Valor']
            
            col_m2.subheader("Coeficientes del Modelo")
            col_m2.dataframe(coef_df, use_container_width=True)
            
            st.markdown("#### Interpretación Clave (Binomial Negativa)")
            
            # Interpretación de GD
            p_gd = coef_df.loc['GD_Broca_Acumulado', 'P-Valor']
            coef_gd = coef_df.loc['GD_Broca_Acumulado', 'Coeficiente']
            
            st.info(
                f"**Riesgo Climático (GD_Broca_Acumulado):** Coeficiente: **{coef_gd:.3f}** | P-Valor: **{p_gd:.3f}**.\n"
                f"Un coeficiente positivo indica que más Grados-Día se asocian con un mayor conteo de la plaga."
            )
            
            p_alt = coef_df.loc['Altitud_km', 'P-Valor']
            coef_alt = coef_df.loc['Altitud_km', 'Coeficiente']
            st.info(
                f"**Altitud (Altitud_km):** Coeficiente: **{coef_alt:.3f}** | P-Valor: **{p_alt:.3f}**.\n"
                f"Un coeficiente negativo (típico) significa que la Altitud reduce el conteo de la plaga."
            )
