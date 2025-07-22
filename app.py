import os
import json
import ee
from ee import ServiceAccountCredentials
import streamlit as st
import numpy as np
import pandas as pd
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt

# -----------------------------------------
# Page configuration
# -----------------------------------------
st.set_page_config(
    page_title="Kodaikanal Landslide Dashboard",
    layout="wide"
)
st.title("📍 Kodaikanal Landslide Prediction Dashboard")

# -----------------------------------------
# Earth Engine Authentication (Streamlit Secrets)
# -----------------------------------------
sa_info = st.secrets.get("EE_CREDENTIALS_JSON", None)
if sa_info is None:
    st.error("❌ Earth Engine credentials not found. Please set EE_CREDENTIALS_JSON in Streamlit secrets.")
    st.stop()

key_path = "/tmp/ee_key.json"
with open(key_path, "w") as f:
    json.dump(dict(sa_info), f)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path
credentials = ServiceAccountCredentials(sa_info["client_email"], key_path)
ee.Initialize(credentials, project=sa_info["project_id"])

# -----------------------------------------
# Define AOI (Kodaikanal bounds)
# -----------------------------------------
region_coords = [77.3, 10.1, 77.7, 10.4]
region = ee.Geometry.Rectangle(region_coords)

# -----------------------------------------
# Cached EE computations
# -----------------------------------------
@st.cache_data(show_spinner=False)
def get_ndvi():
    col = (
        ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
          .filterBounds(region)
          .filterDate('2023-06-01', '2023-11-30')
    )
    return col.map(lambda img: img.normalizedDifference(['SR_B5','SR_B4']).rename('NDVI')).median()

@st.cache_data(show_spinner=False)
def get_slope():
    return ee.Terrain.slope(ee.Image('USGS/SRTMGL1_003'))

@st.cache_data(show_spinner=False)
def get_mask():
    ndvi  = get_ndvi()
    slope = get_slope()
    mask  = ndvi.lt(0.2).And(slope.gt(15))
    return mask.updateMask(mask)

@st.cache_data(show_spinner=False)
def get_landslide_scars():
    return (
        ee.ImageCollection('COPERNICUS/S2')
          .filterBounds(region)
          .filterDate('2023-01-01','2023-12-31')
          .median()
          .normalizedDifference(['B8','B4'])
          .rename('LandslideScars')
    )

@st.cache_data(show_spinner=False)
def get_points():
    pts_fc = (
        get_mask()
        .addBands(ee.Image.pixelLonLat())
        .sample(region=region, scale=500, dropNulls=True)
        .select(['latitude','longitude'])
        .limit(500)
    )
    feats = pts_fc.getInfo()['features']
    coords = [(f['properties']['latitude'], f['properties']['longitude'])
              for f in feats]
    return pd.DataFrame(coords, columns=['Latitude','Longitude'])

@st.cache_data(show_spinner=False)
def get_ndvi_histogram():
    feats = get_ndvi().sample(region=region, scale=500, numPixels=1000).getInfo()['features']
    return [f['properties']['NDVI'] for f in feats if 'NDVI' in f['properties']]

@st.cache_data(show_spinner=False)
def get_hillshade():
    return ee.Terrain.hillshade(ee.Image('USGS/SRTMGL1_003'))

# -----------------------------------------
# Visualization parameters
# -----------------------------------------
vis_params = {
    'NDVI':      {'min':0,'max':1,'palette':['white','green']},
    'Slope':     {'min':0,'max':60},
    'Mask':      {'palette':['#ff000080']},    # red @50% opacity
    'Scars':     {'min':0,'max':1,'palette':['brown']},
    'Hillshade': {'min':0,'max':255}           # grayscale hillshade
}

# -----------------------------------------
# Sidebar controls
# -----------------------------------------
st.sidebar.header("Map Layers & Options")
show_ndvi      = st.sidebar.checkbox("NDVI", value=False)
show_slope     = st.sidebar.checkbox("Slope", value=False)
show_mask      = st.sidebar.checkbox("Landslide Mask", value=True)
show_scars     = st.sidebar.checkbox("Landslide Scars (Satellite)", value=True)
show_hillshade = st.sidebar.checkbox("Hillshade (Relief)", value=False)
show_points    = st.sidebar.checkbox("Prediction Points", value=False)
show_hist      = st.sidebar.checkbox("NDVI Histogram", value=True)

if st.sidebar.button("🔄 Refresh Data"):
    for fn in (get_ndvi, get_slope, get_mask, get_landslide_scars,
               get_hillshade, get_points, get_ndvi_histogram):
        fn.clear()
    st.experimental_rerun()

# -----------------------------------------
# Helper to add EE layers
# -----------------------------------------
def add_ee_layer(m, ee_img, vis, name, opacity=1.0):
    map_id = ee_img.getMapId(vis)
    folium.raster_layers.TileLayer(
        tiles=map_id['tile_fetcher'].url_format,
        attr='Google Earth Engine',
        name=name,
        overlay=True,
        control=True,
        opacity=opacity
    ).add_to(m)

# -----------------------------------------
# Build Folium map
# -----------------------------------------
m = folium.Map(location=[10.27,77.49], zoom_start=12, tiles=None)

# Base layers
folium.TileLayer("OpenStreetMap", name="OSM").add_to(m)
folium.TileLayer("Stamen Terrain", name="Terrain").add_to(m)
folium.TileLayer("Esri.WorldImagery", name="Satellite", control=True).add_to(m)

# EE overlays (hillshade first so other layers sit on top)
if show_hillshade:
    add_ee_layer(m, get_hillshade(),    vis_params['Hillshade'], 'Hillshade',    opacity=0.5)
if show_ndvi:
    add_ee_layer(m, get_ndvi(),         vis_params['NDVI'],      'NDVI')
if show_slope:
    add_ee_layer(m, get_slope(),        vis_params['Slope'],     'Slope')
if show_mask:
    add_ee_layer(m, get_mask(),         vis_params['Mask'],      'Landslide Mask')
if show_scars:
    add_ee_layer(m, get_landslide_scars(), vis_params['Scars'],  'Landslide Scars')

# Optional point markers
points_df = None
if show_points:
    points_df = get_points()
    for _, row in points_df.iterrows():
        folium.CircleMarker(
            location=[row.Latitude, row.Longitude],
            radius=4, color='blue', fill=True
        ).add_to(m)

folium.LayerControl().add_to(m)

# -----------------------------------------
# Layout: map + side panel
# -----------------------------------------
col1, col2 = st.columns((3,1))
with col1:
    st_folium(m, width=700, height=600)

with col2:
    if show_points and points_df is not None and not points_df.empty:
        st.subheader("Predicted Landslide Coordinates")
        st.dataframe(points_df)
    elif show_hist:
        st.subheader("NDVI Value Distribution")
        vals = get_ndvi_histogram()
        fig, ax = plt.subplots()
        ax.hist(vals, bins=30)
        ax.set_xlabel("NDVI")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)
    else:
        st.write("Toggle 'Prediction Points' or 'NDVI Histogram' to view data.")
