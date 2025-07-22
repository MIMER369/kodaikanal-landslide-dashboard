import os, json, io, zipfile
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
sa_info = st.secrets["EE_CREDENTIALS_JSON"]
key_path = "/tmp/ee_key.json"
with open(key_path, "w") as f:
    json.dump(dict(sa_info), f)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path

# Initialize Earth Engine
credentials = ServiceAccountCredentials(sa_info["client_email"], key_path)
ee.Initialize(credentials, project=sa_info["project_id"])

# Define AOI
region = ee.Geometry.Rectangle([77.45, 10.22, 77.55, 10.32])

st.write("Click to fetch and display the latest precomputed landslide mask:")

if st.button("Fetch & Display Mask"):
    st.info("Generating download URL from Earth Engine…")

    # Load the published mask asset
    mask_ee = ee.Image("projects/landslide-demo-466508/assets/mask_kodaikanal")

    # Serialize region to JSON
    region_json = region.getInfo()

    # Request download URL
    try:
        url = mask_ee.getDownloadURL({
            "region": region_json,
            "scale": 30,
            "crs": "EPSG:4326",
            "fileFormat": "GeoTIFF"
        })
    except Exception as e:
        st.error(f"❌ Failed to get download URL: {e}")
        st.stop()

    # Download the content (likely a ZIP archive)
    st.info("Downloading mask (may be zipped)…")
    try:
        resp = request.urlopen(url)
        data = resp.read()
        if not data:
            st.error("❌ Download returned empty data.")
            st.stop()
    except Exception as e:
        st.error(f"❌ HTTP error downloading mask: {e}")
        st.stop()

    # Extract GeoTIFF from ZIP (if zipped), else write directly
    tmp_path = "/tmp/mask.tif"
    try:
        # Try to open as a zip
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            # Find the first TIFF in the archive
            tif_name = next((n for n in zf.namelist() if n.endswith('.tif')), None)
            if tif_name is None:
                raise ValueError("No .tif found in ZIP archive.")
            tif_data = zf.read(tif_name)
            with open(tmp_path, "wb") as f:
                f.write(tif_data)
    except zipfile.BadZipFile:
        # Not a zip, assume raw TIF
        with open(tmp_path, "wb") as f:
            f.write(data)
    except Exception as e:
        st.error(f"❌ Error extracting GeoTIFF: {e}")
        st.stop()

    # Verify file size
    file_size = os.path.getsize(tmp_path)
    if file_size < 100:
        st.error(f"❌ Downloaded file too small ({file_size} bytes). Check your asset or region.")
        st.stop()

    # Read the mask into numpy
    try:
        with rasterio.open(tmp_path) as src:
            mask_arr = src.read(1)
    except Exception as e:
        st.error(f"❌ Failed to open GeoTIFF: {e}")
        st.stop()

    # Display with Folium
    m = folium.Map(location=[10.27, 77.49], zoom_start=12)
    folium.TileLayer("Stamen Terrain").add_to(m)
    folium.raster_layers.ImageOverlay(
        image=mask_arr.astype(np.uint8) * 255,
        bounds=[[10.22, 77.45], [10.32, 77.55]],
        opacity=0.6,
        name="Landslide Mask"
    ).add_to(m)
    st_folium(m, width=700, height=500)
