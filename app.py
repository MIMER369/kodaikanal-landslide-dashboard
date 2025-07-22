import os, json
import ee
from ee import ServiceAccountCredentials
import streamlit as st
import numpy as np
import folium
from streamlit_folium import st_folium
import rasterio
from urllib import request

# Configure page layout
st.set_page_config(layout="wide")
st.title("📍 Kodaikanal Landslide Detection")

# — Earth Engine Authentication via Secrets —
# Secrets are stored in Streamlit Cloud or local .streamlit/secrets.toml
sa_info = st.secrets["EE_CREDENTIALS_JSON"]
key_path = "/tmp/ee_key.json"

# Write the service account JSON to a temporary file
with open(key_path, "w") as f:
    json.dump(dict(sa_info), f)

# Point Google libraries to the service account key
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path

# Initialize Earth Engine with service account credentials
credentials = ServiceAccountCredentials(sa_info["client_email"], key_path)
ee.Initialize(credentials, project=sa_info["project_id"])

# Define your area of interest as an ee.Geometry
region = ee.Geometry.Rectangle([77.45, 10.22, 77.55, 10.32])

st.write("Click to fetch and display the latest precomputed landslide mask:")

if st.button("Fetch & Display Mask"):
    st.info("Generating download URL from Earth Engine…")

    # Load your precomputed mask asset (replace with your real EE asset path)
    mask_ee = ee.Image("users/your_username/mask_kodaikanal")

    # Serialize the ee.Geometry to client-side JSON
    region_info = region.getInfo()

    # Safely request a download URL
    try:
        url = mask_ee.getDownloadURL({
            "region":     region_info,
            "scale":      30,
            "crs":        "EPSG:4326",
            "fileFormat": "GeoTIFF"
        })
    except Exception as e:
        st.error(f"❌ Failed to get download URL: {e}")
        st.stop()

    # Download the GeoTIFF to a temporary file
    st.info("Downloading mask…")
    resp = request.urlopen(url)
    with open("/tmp/mask.tif", "wb") as f:
        f.write(resp.read())

    # Read the mask into a numpy array
    with rasterio.open("/tmp/mask.tif") as src:
        mask_arr = src.read(1)

    # Display on a Folium map
    m = folium.Map(location=[10.27, 77.49], zoom_start=12)
    folium.TileLayer("Stamen Terrain").add_to(m)
    folium.raster_layers.ImageOverlay(
        image=mask_arr.astype(np.uint8) * 255,
        bounds=[[10.22, 77.45], [10.32, 77.55]],
        opacity=0.6,
        name="Landslide Mask"
    ).add_to(m)
    st_folium(m, width=700, height=500)
