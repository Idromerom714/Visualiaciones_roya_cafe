import folium
import numpy as np
from shapely.geometry import Polygon

# --- Coordenadas base ---
locations = {
    "Hacienda 1 (Circasia)": [4.6165, -75.6521],
    "Hacienda 2 (Manizales)": [5.0427, -75.5707]
}

# --- Parámetros geométricos ---
hectarea_x = 200  # m
hectarea_y = 100  # m
parcelas = 2
zonas_x = 5   # 5 columnas
zonas_y = 2   # 2 filas
zona_w = 20   # ancho 20 m
zona_h = 50   # alto 50 m

# Conversión metros → grados
def meters_to_degrees(lat, dx, dy):
    dlat = dy / 111320
    dlon = dx / (111320 * np.cos(np.radians(lat)))
    return dlat, dlon

# Crear mapa centrado
m = folium.Map(location=[4.845, -75.58], zoom_start=10, tiles="CartoDB positron")

# --- Función para crear haciendas ---
def crear_hacienda(nombre, lat, lon, color):
    folium.Marker(
        [lat, lon],
        popup=f"<b>{nombre}</b>",
        icon=folium.Icon(color=color)
    ).add_to(m)
    
    # Hacienda: 200x100 m → 2 parcelas de 100x100
    for p in range(parcelas):
        offset_x = (p - 0.5) * 100
        dlat_p, dlon_p = meters_to_degrees(lat, offset_x, 0)
        parcela_lon = lon + dlon_p
        parcela_lat = lat

        # Dibujar cada zona dentro de la parcela
        for i in range(zonas_y):
            for j in range(zonas_x):
                cx = (j - (zonas_x/2 - 0.5)) * zona_w
                cy = (i - 0.5) * zona_h - 25
                dlat_c, dlon_c = meters_to_degrees(lat, cx, cy)
                z_lat = parcela_lat + dlat_c
                z_lon = parcela_lon + dlon_c

                # Calcular vértices del rectángulo
                half_w, half_h = zona_w/2, zona_h/2
                corners = [
                    meters_to_degrees(lat, cx - half_w, cy - half_h),
                    meters_to_degrees(lat, cx + half_w, cy - half_h),
                    meters_to_degrees(lat, cx + half_w, cy + half_h),
                    meters_to_degrees(lat, cx - half_w, cy + half_h)
                ]
                poly_coords = [[parcela_lat + c[0], parcela_lon + c[1]] for c in corners]

                zona_id = f"{nombre.split()[1][0]}-P{p+1}-Z{i*zonas_x+j+1}"

                # Dibujar zona como polígono
                folium.Polygon(
                    locations=poly_coords,
                    color=color,
                    weight=1,
                    fill=True,
                    fill_opacity=0.25,
                    popup=zona_id
                ).add_to(m)

                # Punto central
                folium.CircleMarker(
                    [z_lat, z_lon],
                    radius=3,
                    color=color,
                    fill=True,
                    fill_opacity=0.9,
                    popup=f"{zona_id} (Centro)"
                ).add_to(m)

# Crear ambas haciendas
crear_hacienda("Hacienda 1 (Circasia)", 4.6165, -75.6521, "green")
crear_hacienda("Hacienda 2 (Manizales)", 5.0427, -75.5707, "blue")

# --- Leyenda HTML ---
legend_html = """
<div style="
     position: fixed;
     bottom: 50px; left: 50px; width: 220px; height: 110px;
     background-color: white; z-index:9999; font-size:14px;
     border:2px solid grey; border-radius:8px; padding:10px;">
<b>Leyenda:</b><br>
<span style="color:green;">&#9679;</span> Hacienda 1 (Circasia)<br>
<span style="color:blue;">&#9679;</span> Hacienda 2 (Manizales)<br>
&#9632; Zonas de muestreo (20×50 m)<br>
● Punto central de cada zona
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))

# --- Guardar mapa ---
mapa_html = m._repr_html_()

print("✅ Mapa generado y guardado como: 'Mapa_zonas_muestreo_cafeteras.html'")
m
