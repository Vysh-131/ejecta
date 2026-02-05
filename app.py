import streamlit as st
import pandas as pd
import os
import shutil
import cv2
import numpy as np
from PIL import Image
import tempfile

# module import
try:
    from ejecta_local import EjectaDetector, Config as EngineConfig
    from scrape import LunarScraper, ScraperConfig
except ImportError as e:
    st.error(f"Could not import your scripts. {e}")
    st.stop()

# config
st.set_page_config(
    page_title="Lunar Ejecta Finder",
    layout="wide",
    initial_sidebar_state="expanded"
)

# css
st.markdown("""
<style>
    .stProgress > div > div > div > div { background-color: #00ff00; }
    .reportview-container { background: #0e1117; }
    div[data-testid="stMetricValue"] { font-size: 24px; color: #00ff00; }
</style>
""", unsafe_allow_html=True)

if 'detector' not in st.session_state:
    with st.spinner("Loading normal and inverted models"):
        try:
            # We initialize the engine once so we don't reload it every click
            st.session_state.detector = EjectaDetector()
            st.session_state.scraper = LunarScraper()
            # Inject the loaded detector into the scraper
            st.session_state.scraper.detector = st.session_state.detector
            st.success("Initialized")
        except Exception as e:
            st.error(f"Failed to load models: {e}")

# sidebar settings
st.sidebar.title("Control Panel")

st.sidebar.subheader("Detection Sensitivity")
conf_thres = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
iou_thres = st.sidebar.slider("Overlap Threshold (IoU)", 0.0, 1.0, 0.4, 0.05)
use_tta = st.sidebar.checkbox("Use TTA (Slower, More Accurate)", value=True)

# Update the global config of your engine dynamically
EngineConfig.CONF_THRES = conf_thres
EngineConfig.IOU_THRES = iou_thres
EngineConfig.USE_TTA = use_tta

st.sidebar.markdown("---")
st.sidebar.info(
    "**Hardware Status:**\n"
    f"- Device: `{st.session_state.detector.device}`\n"
    f"- Models: `n.pt`, `inv.pt`"
)

# --- MAIN TABS ---
tab1, tab2 = st.tabs(["Live Moon Scanner", "Local File Analyzer"])

# ==============================================================================
# TAB 1: LIVE MOON SCANNER (The Scraper)
# ==============================================================================
with tab1:
    st.header("Ejecta Scanner")
    st.markdown("Scan specific coordinates on the Moon using the USGS LROC WAC Mosaic.")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1: lat_min = st.number_input("Min Lat", value=9.0, format="%.4f")
    with col2: lat_max = st.number_input("Max Lat", value=10.0, format="%.4f")
    with col3: lon_min = st.number_input("Min Lon", value=-21.0, format="%.4f")
    with col4: lon_max = st.number_input("Max Lon", value=-20.0, format="%.4f")

    # Scan Button
    if st.button("Starting scan", type="primary"):
        results_container = st.container()
        status_text = st.empty()
        progress_bar = st.progress(0)
        
        # Create a specific job ID for this run
        job_id = f"scan_{lat_min}_{lon_min}"
        
        # Run the Generator
        scanner = st.session_state.scraper.run_scan_job(lat_min, lat_max, lon_min, lon_max, job_id=job_id)
        
        scan_data = [] # To store results for display
        
        try:
            for update in scanner:
                if update['status'] == 'scanning':
                    # Update Progress
                    pct = int((update['current'] / update['total']) * 100)
                    progress_bar.progress(pct)
                    status_text.text(f" Scanning Tile {update['current']}/{update['total']} @ {update['lat']:.2f}, {update['lon']:.2f}...")
                
                elif update['status'] == 'completed':
                    progress_bar.progress(100)
                    status_text.success(f"Complete! Found {update['hits']} ejecta sites.")
                    
                    if update['hits'] > 0:
                        # Load results for display
                        csv_path = update['report']
                        df = pd.read_csv(csv_path)
                        scan_data = df.to_dict('records')
        except Exception as e:
            st.error(f"Scan failed: {e}")

        # --- DISPLAY RESULTS ---
        if scan_data:
            st.markdown("### Detection Gallery")
            
            # Show Metrics
            m1, m2 = st.columns(2)
            m1.metric("Total Hits", len(scan_data))
            avg_conf = np.mean([x['Confidence'] for x in scan_data])
            m2.metric("Avg Confidence", f"{avg_conf:.2f}")

            # Show Interactive Table
            st.dataframe(
                pd.DataFrame(scan_data)[['Latitude', 'Longitude', 'Confidence', 'Model', 'LROC_Link']],
                column_config={
                    "LROC_Link": st.column_config.LinkColumn("View on LROC Map")
                },
                use_container_width=True
            )
            
            # Show Image Gallery (Chips)
            st.markdown("#### Visual Confirmation")
            # We display images in a grid
            cols = st.columns(4)
            for idx, hit in enumerate(scan_data[:12]): # Show top 12
                with cols[idx % 4]:
                    img_path = os.path.join("static", "scans", job_id, "detections", hit['Image_Chip'])
                    if os.path.exists(img_path):
                        st.image(img_path, caption=f"Lat: {hit['Latitude']}\nConf: {hit['Confidence']:.2f}", use_column_width=True)
                    else:
                        st.warning("Image missing")

# ==============================================================================
# TAB 2: LOCAL FOLDER BATCH PROCESSOR
# ==============================================================================
with tab2:
    st.header("📂 Local Batch Processor")
    st.markdown("Process an entire folder of images on your computer.")

    # 1. Inputs for Folder Paths
    col1, col2 = st.columns(2)
    with col1:
        input_dir = st.text_input("Input Folder Path", value="test_images_folder", help="The full path to the folder containing your raw images.")
    with col2:
        output_dir = st.text_input("Output Folder Path", value="robust_output", help="Where to save the detections.")

    # 2. The "Run" Button
    if st.button("⚡ Run Batch Analysis", type="primary"):
        # Validation checks
        if not os.path.exists(input_dir):
            st.error(f"Error: The input folder `{input_dir}` does not exist.")
        else:
            # Create output dir if missing
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                st.success(f"Created output folder: `{output_dir}`")

            # Get list of images
            valid_exts = ('.jpg', '.jpeg', '.png', '.tif', '.bmp')
            files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_exts)]
            
            if not files:
                st.warning("No valid images found in the input folder.")
            else:
                # --- START PROCESSING LOOP ---
                progress_bar = st.progress(0)
                status_text = st.empty()
                results_placeholder = st.empty()
                
                total_files = len(files)
                hits_count = 0
                
                for i, filename in enumerate(files):
                    # Update status
                    status_text.text(f"Processing {i+1}/{total_files}: {filename}...")
                    progress_bar.progress(int(((i+1) / total_files) * 100))
                    
                    try:
                        # Read Image
                        img_path = os.path.join(input_dir, filename)
                        image = cv2.imread(img_path)
                        
                        if image is None:
                            continue

                        # CALL THE ENGINE
                        # We use the session state engine loaded at startup
                        result_img, json_hits = st.session_state.detector.analyze_single_image(image)
                        
                        # Save Result
                        save_name = f"detected_{filename}"
                        save_path = os.path.join(output_dir, save_name)
                        cv2.imwrite(save_path, result_img)
                        
                        # Live Feedback for Hits
                        if json_hits:
                            hits_count += len(json_hits)
                            # Show the latest hit immediately in the UI
                            result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                            results_placeholder.image(result_rgb, caption=f"Hit found in {filename}", use_column_width=True)
                            
                    except Exception as e:
                        st.error(f"Failed to process {filename}: {e}")

                # --- FINISH ---
                progress_bar.progress(100)
                status_text.success(f" Batch Complete! Processed {total_files} images.")
                st.info(f" Total Ejecta Found: {hits_count}")
                st.markdown(f"**Results saved to:** `{os.path.abspath(output_dir)}`")