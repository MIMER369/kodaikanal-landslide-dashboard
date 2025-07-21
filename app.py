import os, json
import ee
from ee import ServiceAccountCredentials
import streamlit as st

# … st.set_page_config, st.title …

# — Earth Engine Authentication via Secrets —
sa_info = st.secrets["EE_CREDENTIALS_JSON"]
with open("/tmp/ee_key.json", "w") as f:
    json.dump(dict(sa_info), f)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/content/landslide-demo-466508-922dd630cf91.json"

# Service account credentials & Initialize
credentials = ServiceAccountCredentials(sa_info["client_email"], "/content/landslide-demo-466508-922dd630cf91.json")
ee.Initialize(credentials, project=sa_info["landslide-demo-466508"])

# Define your region
region = ee.Geometry.Rectangle([77.45, 10.22, 77.55, 10.32])

st.write("Click to fetch and display the latest precomputed landslide mask:")

if st.button("Fetch & Display Mask"):
    st.info("Generating download URL from Earth Engine…")
    # Use your actual Earth Engine asset ID here
    mask_ee = ee.Image("users/your_username/mask_kodaikanal")
    url = mask_ee.getDownloadURL({
        "region":    region,
        "scale":     30,
        "crs":       "EPSG:4326",
        "fileFormat":"GeoTIFF"
    })

    st.info("Downloading mask…")
    resp = request.urlopen(url)
    with open("/tmp/mask.tif", "wb") as f:
        f.write(resp.read())

    # Read the GeoTIFF into a NumPy array
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

