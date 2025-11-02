import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium

# ------------------------
# CONFIGURACIÓN BÁSICA
# ------------------------
st.set_page_config(page_title="Dashboard Cafetero", layout="wide")

st.title("🌱 Dashboard de Monitoreo Cafetero")
st.markdown("Visualización de clima, riesgo y zonas de muestreo para Manizales y Circasia.")

# ------------------------
# CARGA DE DATOS
# ------------------------
@st.cache_data
def load_data():
    clima_circ = pd.read_csv("data/clima_Circasia.csv")
    clima_mani = pd.read_csv("data/clima_Manizales.csv")
    riesgo_circ = pd.read_csv("data/indicador_riesgo_Circasia.csv")
    riesgo_mani = pd.read_csv("data/indicador_riesgo_Manizales.csv")
    ubic = pd.read_csv("data/ubicaciones.csv")
    zonas = pd.read_csv("data/muestreo.csv")
    return clima_circ, clima_mani, riesgo_circ, riesgo_mani, ubic, zonas

clima_circ, clima_mani, riesgo_circ, riesgo_mani, ubic, zonas = load_data()

# ------------------------
# FILTROS LATERALES
# ------------------------
st.sidebar.header("Filtros")
municipio = st.sidebar.selectbox("Seleccionar ubicación", ["Circasia", "Manizales"])
tipo_dato = st.sidebar.selectbox("Tipo de dato", ["Clima", "Riesgo", "Zonas de muestreo"])

# ------------------------
# MAPA INTERACTIVO
# ------------------------
if municipio == "Circasia":
    lat, lon = 4.6165, -75.6521
    clima = clima_circ
    riesgo = riesgo_circ
else:
    lat, lon = 5.0427, -75.5707
    clima = clima_mani
    riesgo = riesgo_mani

m = folium.Map(location=[lat, lon], zoom_start=13, tiles="CartoDB positron")

# Añadir zonas de muestreo
if tipo_dato == "Zonas de muestreo":
    for _, z in zonas.iterrows():
        if z["municipio"] == municipio:
            folium.Marker(
                [z["latitud"], z["longitud"]],
                popup=f"Zona {z['zona_id']}",
                icon=folium.Icon(color="green", icon="leaf")
            ).add_to(m)

# Mostrar mapa
st.subheader(f"📍 Mapa interactivo - {municipio}")
st_folium(m, width=900, height=500)

# ------------------------
# GRÁFICOS BÁSICOS
# ------------------------
st.subheader("📈 Datos recientes")

if tipo_dato == "Clima":
    st.dataframe(clima.tail(10))
elif tipo_dato == "Riesgo":
    st.dataframe(riesgo.tail(10))
else:
    st.dataframe(zonas[zonas["municipio"] == municipio])
