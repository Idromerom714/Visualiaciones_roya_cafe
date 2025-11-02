"""
app.py - Dashboard interactivo (Streamlit)
- Seleccionar municipio
- Seleccionar variable climática / fitosanitaria
- Sugerencias de tipo de gráfico por variable
- Mostrar indicador de riesgo (gauge)
- Mostrar mapa de zonas si existe mapa_zonas.py (opcional)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import importlib.util
import os

st.set_page_config(page_title="Dashboard Roya del Café", layout="wide")

# ----------------------------
# RUTAS A DATOS (ajusta si es necesario)
# ----------------------------
DATA_DIR = "data"
CLIMA_CIR = os.path.join(DATA_DIR, "clima_Circasia.csv")
CLIMA_MAN = os.path.join(DATA_DIR, "clima_Manizales.csv")
RIESGO_CIR = os.path.join(DATA_DIR, "indicador_riesgo_Circasia.csv")
RIESGO_MAN = os.path.join(DATA_DIR, "indicador_riesgo_Manizales.csv")
ZONAS_CSV = os.path.join(DATA_DIR, "muestreo.csv")   # tu CSV de zonas (zonas_muestreo_cafeteras)

# ----------------------------
# CARGA DE DATOS (tolerante si faltan archivos)
# ----------------------------
@st.cache_data
def load_safe(path):
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            return df
        except Exception:
            return None
    return None

clima_cir = load_safe(CLIMA_CIR)
clima_man = load_safe(CLIMA_MAN)
riesgo_cir = load_safe(RIESGO_CIR)
riesgo_man = load_safe(RIESGO_MAN)
zonas = load_safe(ZONAS_CSV)

# Crear lista de municipios disponibles a partir de datos cargados
municipios_disponibles = []
if clima_cir is not None or riesgo_cir is not None:
    municipios_disponibles.append("Circasia")
if clima_man is not None or riesgo_man is not None:
    municipios_disponibles.append("Manizales")
if len(municipios_disponibles) == 0:
    municipios_disponibles = ["Circasia", "Manizales"]  # defaults

# ----------------------------
# Mapa (intenta importar mapa_zonas.py si existe)
# ----------------------------
mapa_disponible = False
map_obj = None
if os.path.exists("mapa_zonas.py"):
    try:
        spec = importlib.util.spec_from_file_location("mapa_zonas", "mapa_zonas.py")
        mapa_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mapa_mod)
        # asumimos que el módulo define un objeto 'm' (folium.Map)
        if hasattr(mapa_mod, "m"):
            map_obj = mapa_mod.m
            mapa_disponible = True
    except Exception:
        mapa_disponible = False

# ----------------------------
# UI - Sidebar filtros
# ----------------------------
st.sidebar.title("Filtros")
municipio = st.sidebar.selectbox("Selecciona municipio", municipios_disponibles)

# Determinar dataset de clima y riesgo según municipio
if municipio == "Circasia":
    clima_df = clima_cir
    riesgo_df = riesgo_cir
    default_latlon = (4.6165, -75.6521)
else:
    clima_df = clima_man
    riesgo_df = riesgo_man
    default_latlon = (5.0427, -75.5707)

# Construir lista de variables disponibles (clima + calculadas)
variables_clima = []
if clima_df is not None:
    # normalizar nombres de columnas para mostrar amigable
    numeric_cols = clima_df.select_dtypes(include=[np.number]).columns.tolist()
    variables_clima = numeric_cols

# Añadir variables calculadas si el dataset de riesgo existe
variables_extra = []
if riesgo_df is not None:
    # asumimos columna 'riesgo_enfermedad' o similar
    if "riesgo_enfermedad" in riesgo_df.columns:
        variables_extra.append("riesgo_enfermedad")
    if "max_horas_mojadas" in riesgo_df.columns:
        variables_extra.append("max_horas_mojadas")

all_variables = variables_clima + variables_extra
if len(all_variables) == 0:
    # fallback / opciones genéricas
    all_variables = ["temp", "humidity", "precip", "solar_radiation", "max_horas_mojadas", "riesgo_enfermedad"]

variable = st.sidebar.selectbox("Selecciona variable a graficar", all_variables)

# Sugerencia automática de tipo de gráfico según variable
def sugerir_graficos(var):
    var_low = var.lower()
    if any(k in var_low for k in ["temp", "temperature", "tmax", "tmin"]):
        return ["Serie temporal", "Histograma", "Caja (boxplot)"]
    if any(k in var_low for k in ["humid", "rh", "humidity"]):
        return ["Serie temporal", "Area / Línea", "Caja (boxplot)"]
    if any(k in var_low for k in ["precip", "rain", "lluv"]):
        return ["Barras (acumulado)", "Serie temporal", "Mapa de calor"]
    if any(k in var_low for k in ["radiat", "solar"]):
        return ["Serie temporal", "Dispersión vs Riesgo"]
    if any(k in var_low for k in ["hoja", "wet", "horas"]):
        return ["Caja (boxplot)", "Serie temporal"]
    if "riesgo" in var_low or "risk" in var_low:
        return ["Serie temporal", "Gauge", "Mapa temático"]
    # default
    return ["Serie temporal", "Histograma"]

tipos_recomendados = sugerir_graficos(variable)
tipo_grafico = st.sidebar.selectbox("Tipo de gráfico", tipos_recomendados)

# Rango de fechas (si existe columna datetime/fecha)
fecha_min, fecha_max = None, None
if clima_df is not None:
    # intentar detectar columna datetime
    time_cols = [c for c in clima_df.columns if "date" in c.lower() or "time" in c.lower() or "datetime" in c.lower()]
    if len(time_cols) > 0:
        time_col = time_cols[0]
        try:
            clima_df[time_col] = pd.to_datetime(clima_df[time_col])
            fecha_min = clima_df[time_col].min().date()
            fecha_max = clima_df[time_col].max().date()
            fecha_sel = st.sidebar.date_input("Rango de fechas (inicio)", value=fecha_min)
            fecha_sel2 = st.sidebar.date_input("Rango de fechas (fin)", value=fecha_max)
        except Exception:
            fecha_sel = None
            fecha_sel2 = None
    else:
        fecha_sel = None
        fecha_sel2 = None
else:
    fecha_sel = None
    fecha_sel2 = None

st.sidebar.markdown("---")
st.sidebar.write("Mostrar indicador de riesgo")
mostrar_gauge = st.sidebar.checkbox("Mostrar indicador de riesgo", value=True)

# ----------------------------
# LAYOUT PRINCIPAL
# ----------------------------
st.title("Monitoreo climático y riesgo de roya del café")
st.write("Explora variables climáticas y el indicador de riesgo por municipio.")

# PANEL SUPERIOR: tarjetas con estadísticas rápidas
col1, col2, col3, col4 = st.columns([1,1,1,1])
# calcular valores clave (con tolerancia a que no haya datos)
def safe_agg(df, col, func="mean"):
    try:
        if df is None or col not in df.columns:
            return None
        if func == "mean":
            return float(df[col].dropna().mean())
        if func == "median":
            return float(df[col].dropna().median())
        if func == "last":
            # intentar por columna datetime
            tcols = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
            if tcols:
                df_sorted = df.sort_values(by=tcols[0])
                return float(df_sorted[col].dropna().iloc[-1])
            return float(df[col].dropna().iloc[-1])
    except Exception:
        return None

col1.metric("Temperatura (media °C)", f"{safe_agg(clima_df, 'temp'):.2f}" if safe_agg(clima_df, 'temp') is not None else "N/A")
col2.metric("Humedad (%)", f"{safe_agg(clima_df, 'rh'):.1f}" if safe_agg(clima_df, 'rh') is not None else "N/A")
col3.metric("Precipitación (mm/h)", f"{safe_agg(clima_df, 'precip'):.2f}" if safe_agg(clima_df, 'precip') is not None else "N/A")
col4.metric("Horas hoja mojada (max)", f"{safe_agg(riesgo_df, 'max_horas_mojadas'):.1f}" if safe_agg(riesgo_df, 'max_horas_mojadas') is not None else "N/A")

st.markdown("---")

# ----------------------------
# AREA PRINCIPAL: gráfico dinámico
# ----------------------------
left_col, right_col = st.columns((2,1))

with left_col:
    st.subheader(f"Gráfico: {variable} en {municipio}")

    # Preparar dataframe filtrado por fecha si aplica
    df_plot = None
    if clima_df is not None and variable in clima_df.columns:
        df_plot = clima_df.copy()
        # si hay columna de tiempo, convertir a datetime y filtrar
        time_cols = [c for c in df_plot.columns if "date" in c.lower() or "time" in c.lower() or "datetime" in c.lower()]
        if time_cols:
            df_plot[time_cols[0]] = pd.to_datetime(df_plot[time_cols[0]])
            if fecha_sel is not None and fecha_sel2 is not None:
                start = pd.to_datetime(fecha_sel)
                end = pd.to_datetime(fecha_sel2)
                df_plot = df_plot[(df_plot[time_cols[0]] >= start) & (df_plot[time_cols[0]] <= end)]
        x_axis = time_cols[0] if time_cols else None

    elif riesgo_df is not None and variable in riesgo_df.columns:
        df_plot = riesgo_df.copy()
        # convertir fecha si es necesario
        date_cols = [c for c in df_plot.columns if "date" in c.lower()]
        if date_cols:
            df_plot[date_cols[0]] = pd.to_datetime(df_plot[date_cols[0]])
            x_axis = date_cols[0]
        else:
            x_axis = None
    else:
        # si la variable no está en tablas, intentar mostrar una de las columnas numéricas de clima
        if clima_df is not None:
            numeric_cols = clima_df.select_dtypes(include=[np.number]).columns.tolist()
            sel = variable if variable in numeric_cols else numeric_cols[0] if numeric_cols else None
            if sel:
                df_plot = clima_df.copy()
                x_axis = [c for c in df_plot.columns if "date" in c.lower() or "time" in c.lower() or "datetime" in c.lower()]
                x_axis = x_axis[0] if x_axis else None
                variable = sel

    # Renderizar según tipo seleccionado
    if df_plot is None:
        st.warning("No hay datos cargados para la variable seleccionada.")
    else:
        if tipo_grafico == "Serie temporal":
            if x_axis:
                fig = px.line(df_plot, x=x_axis, y=variable, title=f"{variable} en {municipio}")
            else:
                fig = px.line(df_plot, y=variable, title=f"{variable} - serie")
            st.plotly_chart(fig, use_container_width=True)

        elif tipo_grafico == "Histograma":
            fig = px.histogram(df_plot, x=variable, nbins=30, title=f"Histograma de {variable}")
            st.plotly_chart(fig, use_container_width=True)

        elif "Caja" in tipo_grafico:
            fig = px.box(df_plot, y=variable, title=f"Boxplot de {variable}")
            st.plotly_chart(fig, use_container_width=True)

        elif "Barras" in tipo_grafico:
            # para precip acumulada por día si hay datetime
            if x_axis:
                df_plot['date_only'] = pd.to_datetime(df_plot[x_axis]).dt.date
                agg = df_plot.groupby('date_only')[variable].sum().reset_index()
                fig = px.bar(agg, x='date_only', y=variable, title=f"Acumulado diario de {variable}")
            else:
                fig = px.bar(df_plot, x=variable, y=df_plot.index, title=f"Barras {variable}")
            st.plotly_chart(fig, use_container_width=True)

        elif "Dispersión" in tipo_grafico or "Dispersión vs Riesgo" in tipo_grafico:
            # scatter contra riesgo si disponible
            if riesgo_df is not None and "riesgo_enfermedad" in riesgo_df.columns:
                # intentar merge por fecha si existe
                merged = None
                # join by date if both datasets have date-like column
                date_clima = [c for c in clima_df.columns if "date" in c.lower() or "time" in c.lower() or "datetime" in c.lower()]
                date_ries = [c for c in riesgo_df.columns if "date" in c.lower()]
                if date_clima and date_ries:
                    c = clima_df.copy(); r = riesgo_df.copy()
                    c[date_clima[0]] = pd.to_datetime(c[date_clima[0]]); r[date_ries[0]] = pd.to_datetime(r[date_ries[0]])
                    merged = pd.merge(c, r, left_on=date_clima[0], right_on=date_ries[0], how="inner")
                if merged is not None and variable in merged.columns:
                    fig = px.scatter(merged, x=variable, y="riesgo_enfermedad", trendline="ols",
                                     title=f"{variable} vs riesgo_enfermedad")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No hay series que permitan correlacionar variable con riesgo por fecha.")
            else:
                st.warning("Datos de riesgo no disponibles para dispersión.")

with right_col:
    st.subheader("Indicador de riesgo")
    if mostrar_gauge and (riesgo_df is not None):
        # calcular riesgo actual promedio (por ejemplo últimos 7 días)
        risk_val = None
        date_col = [c for c in riesgo_df.columns if "date" in c.lower()]

        # Si existe columna de fecha y riesgo
        if date_col and "riesgo_roya" in riesgo_df.columns:
            temp = riesgo_df.copy()
            temp[date_col[0]] = pd.to_datetime(temp[date_col[0]])
            last_week = temp.sort_values(by=date_col[0]).tail(7)
            try:
                risk_val = last_week["riesgo_roya"].mode()[0]  # valor más frecuente en la última semana
            except Exception:
                risk_val = temp["riesgo_roya"].mode()[0]
        elif "riesgo_roya" in riesgo_df.columns:
            risk_val = riesgo_df["riesgo_roya"].mode()[0]

        # --- Convertir categoría de riesgo a valor numérico ---
        if isinstance(risk_val, str):
            riesgo_map = {
                "bajo": 0.15,
                "moderado": 0.45,
                "alto": 0.75,
                "crítico": 0.95
            }
            risk_val_num = riesgo_map.get(risk_val.lower(), 0)
        else:
            risk_val_num = risk_val

        # --- Gauge con Plotly ---
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_val_num,
            title={"text": f"Riesgo de enfermedad ({str(risk_val).upper()})"},
            gauge={
                "axis": {"range": [0, 1]},
                "bar": {"color": "black"},
                "steps": [
                    {"range": [0, 0.3], "color": "green"},
                    {"range": [0.3, 0.6], "color": "yellow"},
                    {"range": [0.6, 1.0], "color": "red"}
                ]
            }
        ))
        st.plotly_chart(fig_gauge, config={"displayModeBar": True, "responsive": True}, use_container_width=True)
    else:
        st.info("Indicador de riesgo oculto o datos no disponibles.")

# ----------------------------
# MAPA DE ZONAS (sección inferior)
# ----------------------------
st.markdown("---")
st.subheader("Mapa de zonas de muestreo (centros y polígonos)")

if mapa_disponible and map_obj is not None:
    # insertar mapa folium desde mapa_zonas.py
    from streamlit_folium import st_folium
    st_folium(map_obj, width=900, height=600)
else:
    # si no existe módulo, intentar pintar puntos desde CSV 'zonas' si existe
    if zonas is not None:
        # mostrar mapa simple con centros (usando plotly)
        fig_map = px.scatter_mapbox(zonas, lat="latitud", lon="longitud", hover_name="zona",
                                    color="hacienda", zoom=13, height=600)
        fig_map.update_layout(mapbox_style="open-street-map")
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.info("Mapa no disponible. Crea 'mapa_zonas.py' o añade 'data/muestreo.csv' con las zonas.")

st.markdown("---")
st.write("Dashboard desarrollado para el monitoreo climático y riesgo de roya del café.")