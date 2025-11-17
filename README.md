# Visualizaciones de patologías del café

Proyecto: dashboard interactivo para visualizar indicadores climáticos y de riesgo (roya, broca) en cafetales.

Estado actual (resumen)
- El frontend está implementado en `Streamlit` (`app.py`).
- Las funciones de cálculo y modelado (simulaciones, regresión logística y binomial negativa) están incluidas.
- Datos: hay CSV de ejemplo en `data/` y se copiaron a la raíz (`clima_ubicaciones.csv`, `ubicaciones.csv`, `enfermedades.csv`) para compatibilidad con `app.py`.
- Se resolvieron varios errores durante el arranque: dependencias faltantes como `statsmodels` y `streamlit` fueron instaladas; funciones de simulación fueron parcheadas para manejar NaNs.


Requisitos
- Python 3.10+ (el repo se probó con Python 3.12 en el contenedor de desarrollo).
- Dependencias (lista parcial, ver `requirements.txt`):
	- streamlit
	- pandas
	- numpy
	- plotly
	- seaborn
	- matplotlib
	- statsmodels
	- folium (opcional, para mapas)
	- streamlit-folium (opcional, para renderizar `folium` en Streamlit)

Archivos de datos esperados
- `clima_ubicaciones.csv`: series horarias de clima con columnas (latitud, longitud, fecha_hora, temperatura, humedad_relativa, precipitacion, radiacion_solar, humeda).
- `ubicaciones.csv`: tabla de ubicaciones con columnas que incluyan `latitud`, `longitud`, `Hacienda`, `Altitud_m_s_n_m`.
- `enfermedades.csv`: tabla con umbrales por enfermedad (Enfermedad, Patogeno_Causante, T_min, T_max, HR_min, Frecuencia_Lluvia).

Cómo ejecutar (pasos reproducibles)

1) (opcional, recomendado) Crear y activar un entorno virtual:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2) Instalar dependencias:

```bash
pip install -r requirements.txt
# Si falta algo, instalar manualmente: pip install statsmodels streamlit-folium
```

3) Asegurar que los CSV están disponibles en el directorio de ejecución (raíz) o modificar las rutas en `app.py`:

```bash
cp data/clima_ubicaciones.csv .
cp data/ubicaciones.csv .
cp data/enfermedades.csv .
```

4) Ejecutar la app (ver logs en primer plano):

```bash
python3 -m streamlit run app.py --server.port 8501
```

O, si el ejecutable `streamlit` está disponible directamente:

```bash
/home/codespace/.python/current/bin/streamlit run app.py --server.port 8501
```

5) Abrir en el navegador: `http://localhost:8501` (o la URL local que muestre Streamlit).

Problemas comunes y soluciones rápidas
- ModuleNotFoundError: instala la dependencia faltante con `pip install <paquete>` usando el mismo intérprete que ejecuta Streamlit.
- Port 8501 in use: elegir otro puerto `--server.port 8502` o matar el proceso que ocupa el puerto: `pkill -f streamlit`.
- Errores de columna (KeyError): revisar los encabezados de los CSV con `head -n 5 data/<archivo>` y adaptar `app.py` a los nombres reales.

Estado conocido y tareas pendientes
- Mapas: si quieres renderizar el mapa de zonas con polígonos (folium), instala `streamlit-folium` y asegúrate de que `mapa_zonas.py` exporte el objeto `m` (folium.Map). Si no, la app dibuja puntos desde `data/muestreo.csv` si existe.
- Validación estadística: el código usa datos simulados para demostración; para análisis reales, sustituir las simulaciones por datos observados.
- Tests/CI: no existen pruebas automatizadas en el repo actualmente.
