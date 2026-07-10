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
# 1) Page config & Title
# -----------------------------------------
st.set_page_config(page_title="Kodaikanal Landslide Dashboard", layout="wide")
st.title("Kodaikanal Landslide Prediction Dashboard")

# -----------------------------------------
# 2) Earth Engine auth
# -----------------------------------------
sa = st.secrets.get("EE_CREDENTIALS_JSON", None)
if sa is None:
    st.error("Set EE_CREDENTIALS_JSON in Streamlit secrets.")
    st.stop()

key_path = "/tmp/ee_key.json"
with open(key_path, "w") as f:
    json.dump(dict(sa), f)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path
creds = ServiceAccountCredentials(sa["client_email"], key_path)
ee.Initialize(creds, project=sa["project_id"])

# -----------------------------------------
# 3) Define AOI
# -----------------------------------------
region = ee.Geometry.Rectangle([77.3, 10.1, 77.7, 10.4])

# -----------------------------------------
# 4) EE computations
# cache_resource is used (not cache_data) because ee.Image objects are not picklable
# -----------------------------------------
@st.cache_resource
def get_ndvi():
    col = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").filterBounds(region).filterDate("2023-06-01", "2023-11-30")
    return col.map(lambda i: i.normalizedDifference(["SR_B5", "SR_B4"]).rename("NDVI")).median()

@st.cache_resource
def get_slope():
    return ee.Terrain.slope(ee.Image("USGS/SRTMGL1_003"))

@st.cache_resource
def get_mask():
    ndvi = get_ndvi()
    slope = get_slope()
    mask = ndvi.lt(0.2).And(slope.gt(15))
    return mask.updateMask(mask)

@st.cache_resource
def get_scars():
    col = ee.ImageCollection("COPERNICUS/S2").filterBounds(region).filterDate("2023-01-01", "2023-12-31")
    return col.median().normalizedDifference(["B8", "B4"]).rename("LandslideScars")

@st.cache_data
def get_points():
    fc = get_mask().addBands(ee.Image.pixelLonLat()).sample(region=region, scale=500, dropNulls=True).select(["latitude", "longitude"]).limit(500)
    feats = fc.getInfo()["features"]
    rows = [(f["properties"]["latitude"], f["properties"]["longitude"]) for f in feats]
    return pd.DataFrame(rows, columns=["Latitude", "Longitude"])

@st.cache_data
def get_hist():
    feats = get_ndvi().sample(region=region, scale=500, numPixels=1000).getInfo()["features"]
    return [f["properties"]["NDVI"] for f in feats if "NDVI" in f["properties"]]

@st.cache_resource
def get_hillshade():
    return ee.Terrain.hillshade(ee.Image("USGS/SRTMGL1_003"))

# -----------------------------------------
# 5) Visualization parameters
# -----------------------------------------
vis = {
    "NDVI": {"min": 0, "max": 1, "palette": ["white", "green"]},
    "Slope": {"min": 0, "max": 60},
    "Mask": {"palette": ["#ff000080"]},
    "Scars": {"min": 0, "max": 1, "palette": ["brown"]},
    "Hillshade": {"min": 0, "max": 255}
}

# -----------------------------------------
# 6) Sidebar controls
# -----------------------------------------
st.sidebar.header("Map Layers & Options")
show_ndvi = st.sidebar.checkbox("NDVI", False)
show_slope = st.sidebar.checkbox("Slope", False)
show_mask = st.sidebar.checkbox("Landslide Mask", True)
show_scars = st.sidebar.checkbox("Landslide Scars (Satellite)", True)
show_hillshade = st.sidebar.checkbox("Hillshade (Relief)", False)
show_points = st.sidebar.checkbox("Prediction Points", False)
show_hist = st.sidebar.checkbox("NDVI Histogram", True)

if st.sidebar.button("Refresh Data"):
    for fn in (get_ndvi, get_slope, get_mask, get_scars, get_points, get_hist, get_hillshade):
        fn.clear()
    st.experimental_rerun()

# -----------------------------------------
# 7) Folium Map
# -----------------------------------------
m = folium.Map(location=[10.27, 77.49], zoom_start=12, tiles=None)

folium.TileLayer("OpenStreetMap", name="OSM", attr="OpenStreetMap contributors").add_to(m)
folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri, USGS, NOAA",
    name="Satellite"
).add_to(m)

def add_ee(m_, img, prm, name, opacity=1.0):
    mid = img.getMapId(prm)
    folium.raster_layers.TileLayer(
        tiles=mid["tile_fetcher"].url_format,
        attr="Google Earth Engine",
        name=name,
        overlay=True,
        control=True,
        opacity=opacity
    ).add_to(m_)

if show_hillshade:
    add_ee(m, get_hillshade(), vis["Hillshade"], "Hillshade", opacity=0.5)

if show_ndvi:
    add_ee(m, get_ndvi(), vis["NDVI"], "NDVI")

if show_slope:
    add_ee(m, get_slope(), vis["Slope"], "Slope")

if show_mask:
    add_ee(m, get_mask(), vis["Mask"], "Landslide Mask")

if show_scars:
    add_ee(m, get_scars(), vis["Scars"], "Landslide Scars")

if show_points:
    df = get_points()
    for _, r in df.iterrows():
        folium.CircleMarker(
            location=[r.Latitude, r.Longitude],
            radius=4, color="blue", fill=True
        ).add_to(m)

folium.LayerControl().add_to(m)
st_folium(m, width=700, height=600)

# -----------------------------------------
# 8) Bottom panel: table or histogram
# -----------------------------------------
if show_points and not get_points().empty:
    st.subheader("Predicted Landslide Coordinates")
    st.dataframe(get_points())
elif show_hist:
    st.subheader("NDVI Value Distribution")
    vals = get_hist()
    fig, ax = plt.subplots()
    ax.hist(vals, bins=30)
    ax.set_xlabel("NDVI")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)
else:
    st.write("Toggle Prediction Points or NDVI Histogram to view data.")
