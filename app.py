import os
import ee
import streamlit as st
import numpy as np
import tensorflow as tf
from urllib import request
from streamlit_folium import st_folium
import folium
import pandas as pd
import io

st.set_page_config(layout="wide")
st.title("📍 Kodaikanal Landslide Detection")

# ————— Earth Engine Authentication —————
# 1️⃣ If GOOGLE_APPLICATION_CREDENTIALS is set (in Colab), use it directly:
key_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if key_path:
    try:
        ee.Initialize()
        st.success("Earth Engine initialized using GOOGLE_APPLICATION_CREDENTIALS.")
    except Exception as e:
         st.error(f"Earth Engine initialization failed with GOOGLE_APPLICATION_CREDENTIALS: {e}. Ensure the path is correct and credentials are valid.")
else:
    # 2️⃣ Otherwise (in Streamlit Cloud), read from st.secrets
    try:
        import json
        creds = json.loads(st.secrets["EE_CREDENTIALS_JSON"])
        with open("/tmp/ee_key.json", "w") as f:
            json.dump(creds, f)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/tmp/ee_key.json"
        ee.Initialize()
        st.success("Earth Engine initialized using st.secrets.")
    except Exception as e:
        st.error(f"Earth Engine initialization failed with st.secrets: {e}. Ensure EE_CREDENTIALS_JSON secret is set correctly.")


# Load model once using Streamlit's caching
@st.cache_resource
def load_unet_model():
    # Assuming the model is saved in the Colab environment's current directory
    # or accessible via Google Drive mount if needed
    model_path = 'unet_best.h5' # Or 'unet_kodai.h5' based on your training cell
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}. Please train and save the model first.")
        return None
    try:
        # Explicitly load with custom_objects if any custom layers were used
        # from tensorflow.keras.layers import ... # Import any custom layers if needed
        # custom_objects = {'CustomLayerName': CustomLayerClass}
        # model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        model = tf.keras.models.load_model(model_path)
        st.success("Model loaded successfully.")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_unet_model()

# Define region (ensure this matches your analysis region)
# Assuming region is defined elsewhere in the notebook or define it here
# You might need to pass region coordinates or a GeoJSON if running standalone
region = ee.Geometry.Rectangle([77.45,10.22,77.55,10.32]) # Example region, adjust as needed

st.write("Click the button below to fetch the latest NDVI data for the region and predict landslides.")

if st.button("Fetch NDVI & Predict Landslides"):
    if model is not None: # Proceed only if model was loaded successfully
        st.info("Fetching latest NDVI data from Earth Engine...")
        try:
            # Adjust date range for the latest data as needed
            latest_ndvi_img = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') # Or 'LANDSAT/LC09/C02/T1_L2'
                               .filterBounds(region)
                               .filterDate('2024-01-01','2024-12-31') # Example: filter for current year
                               .map(lambda i: i.normalizedDifference(['SR_B5','SR_B4']).rename('NDVI')) # Bands for NDVI (adjust if using different sensor)
                               .median().clip(region))

            # Get download URL for NDVI
            # Adjust dimensions to match your model's expected input or tiling strategy
            # Assuming your model takes 128x128 patches and you need to tile a 256x256 area
            download_dims = [256, 256] # Example dimensions for area to predict on
            url = latest_ndvi_img.getDownloadURL({
                'region':      region,
                'format':      'NPY',
                'dimensions':  download_dims,
                'filePerBand': False
            })

            st.info("Downloading NDVI data...")
            # Download the content and load from an in-memory buffer
            response = request.urlopen(url)
            buffer = io.BytesIO(response.read())
            ndvi_array = np.load(buffer, allow_pickle=False)

            # Access the numerical data by field name 'NDVI' if it's a structured array
            try:
                # Use extract_float helper function if available and needed, or simple access
                # Assuming a simple array or known field name 'NDVI'
                if isinstance(ndvi_array.dtype, np.dtype) and ndvi_array.dtype.names is not None and 'NDVI' in ndvi_array.dtype.names:
                    ndvi_data = ndvi_array['NDVI'].astype(np.float32)
                else:
                    ndvi_data = ndvi_array.astype(np.float32)

            except Exception as e:
                st.error(f"Error processing downloaded NDVI data: {e}")
                ndvi_data = None # Set to None if processing fails

            if ndvi_data is not None:
                # Normalize NDVI for model input (if your model expects normalized input)
                ndvi_min = ndvi_data.min()
                ndvi_max = ndvi_data.max()
                if ndvi_max - ndvi_min > 0:
                     ndvi_normalized = (ndvi_data - ndvi_min) / (ndvi_max - ndvi_min + 1e-8)
                else:
                     ndvi_normalized = np.zeros_like(ndvi_data) # Handle case of constant values

                # Ensure ndvi has shape [H, W, 1] for model prediction
                if ndvi_normalized.ndim == 2:
                    ndvi_input = ndvi_normalized[..., np.newaxis]
                else:
                    ndvi_input = ndvi_normalized


                st.info("Making landslide prediction using UNet model...")
                # --- Tiling and Prediction Logic (Adapt from cell JShX9mAOC9Dz or similar) ---
                # Assuming model input size is 128x128
                PATCH_SIZE = 128
                h_ndvi, w_ndvi, _ = ndvi_input.shape
                full_mask = np.zeros((h_ndvi, w_ndvi), dtype=bool)
                tiles = []
                coords = []

                # Ensure tiling loop bounds are correct
                for i in range(0, h_ndvi, PATCH_SIZE):
                    for j in range(0, w_ndvi, PATCH_SIZE):
                        i_end = min(i + PATCH_SIZE, h_ndvi)
                        j_end = min(j + PATCH_SIZE, w_ndvi)
                        tile = ndvi_input[i:i_end, j:j_end, :]

                        # Pad tile if smaller than PATCH_SIZE (important for consistent model input)
                        if tile.shape[0] < PATCH_SIZE or tile.shape[1] < PATCH_SIZE:
                             padded_tile = np.pad(tile, ((0, PATCH_SIZE - tile.shape[0]), (0, PATCH_SIZE - tile.shape[1]), (0,0)), mode='constant')
                        else:
                             padded_tile = tile

                        if padded_tile.shape[:2] == (PATCH_SIZE, PATCH_SIZE): # Ensure padded tile is correct size
                            tiles.append(padded_tile)
                            coords.append((i,j, i_end-i, j_end-j)) # Store original coordinates and dimensions
                        else:
                            st.warning(f"Skipping tile at ({i},{j}) due to incorrect padded shape: {padded_tile.shape}")


                if tiles: # Proceed only if tiles were created
                    tiles_np = np.stack(tiles) # Shape (n_tiles, PATCH_SIZE, PATCH_SIZE, 1)
                    preds_tiles = model.predict(tiles_np) > 0.5 # Predict on batched tiles (Shape n_tiles, PATCH_SIZE, PATCH_SIZE, 1)

                    # Reconstruct full mask from predicted tiles
                    for idx, (i, j, tile_h, tile_w) in enumerate(coords):
                        # Extract the relevant part of the prediction tile (remove padding if any)
                        pred_tile = preds_tiles[idx, :tile_h, :tile_w, 0]
                        full_mask[i:i+tile_h, j:j+tile_w] = pred_tile

                    predicted_mask = full_mask # Use the reconstructed full mask for visualization
                    st.success("Prediction complete.")

                    # --- End Tiling and Prediction Logic ---

                    # Visualize the prediction on a map
                    st.info("Displaying prediction on map...")
                    m = folium.Map(location=[10.27,77.49], zoom_start=12) # Center map on your region
                    folium.TileLayer('Stamen Terrain').add_to(m)

                    # Ensure image data is correctly formatted for ImageOverlay (0-1 or 0-255)
                    # Convert boolean mask to uint8 (0 or 1), then scale to 0-255 if colormap expects it
                    # Or use a colormap that handles 0-1 directly
                    # Let's use a colormap that maps 1 to red and 0 to transparent
                    folium.raster_layers.ImageOverlay(
                        image=predicted_mask.astype(np.uint8), # Convert boolean mask to uint8 (0 or 1)
                        bounds=[[region.bounds().getInfo()['coordinates'][0][0][1], region.bounds().getInfo()['coordinates'][0][0][0]],
                                [region.bounds().getInfo()['coordinates'][0][2][1], region.bounds().getInfo()['coordinates'][0][2][0]]], # Bounds [minLat, minLon], [maxLat, maxLon]
                        opacity=0.6,
                        colormap=lambda x: (1, 0, 0, x), # Red overlay where mask is 1, transparent where 0
                        attr='Earth Engine Landslide Prediction' # Added dummy attribution
                    ).add_to(m)
                    st_folium(m, width=800, height=600)

                else:
                    st.warning("No valid tiles were created for prediction.")

        except Exception as e:
            st.error(f"An error occurred during data fetching or prediction: {e}")

# Add download buttons for generated files if they exist
# Assuming these files are generated elsewhere in the notebook or app logic
# For a standalone app, you would generate/save these files within the app logic
# Example: Check if files exist before offering download
st.sidebar.header("Downloads")
if os.path.exists('kodai_landslide_mask.tif'):
    with open('kodai_landslide_mask.tif','rb') as f:
        st.sidebar.download_button("Download GeoTIFF Mask", f, "kodai_landslide_mask.tif", "image/tiff")

if os.path.exists('overlay_landslide.png'):
    with open('overlay_landslide.png','rb') as f:
        st.sidebar.download_button("Download Overlay PNG", f, "overlay_landslide.png", "image/png")


