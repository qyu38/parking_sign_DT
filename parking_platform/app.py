"""
Main Streamlit application for Parking Sign Digital Platform MVP.
"""
import streamlit as st
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime
import time
import pydeck as pdk

# Add current directory to path
current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir))

# Import modules
from processing.detection_wrapper import load_detection_model
from processing.pipeline import process_video
from data.gps_handler import parse_gps_file
from data.storage import Storage, generate_video_id

# Page configuration
st.set_page_config(
    page_title="Parking Sign Digital Platform",
    page_icon="🅿️",
    layout="wide"
)

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'processor' not in st.session_state:
    st.session_state.processor = None

# Storage setup
STORAGE_FILE = current_dir / "data" / "results.json"
storage = Storage(STORAGE_FILE)

# Directories
UPLOADS_DIR = current_dir / "uploads"
PROCESSED_DIR = current_dir / "processed"
UPLOADS_DIR.mkdir(exist_ok=True)
PROCESSED_DIR.mkdir(exist_ok=True)


@st.cache_resource
def load_models():
    """Load the Qwen VLM model (cached)."""
    with st.spinner("Loading Qwen VLM model... This may take a minute."):
        model, processor = load_detection_model()
        return model, processor


def main():
    """Main application."""
    st.title("🅿️ Parking Sign Digital Platform")
    st.markdown("Upload inspection videos to automatically detect and extract parking sign information.")
    
    # Load model if not already loaded
    if not st.session_state.model_loaded:
        st.session_state.model, st.session_state.processor = load_models()
        st.session_state.model_loaded = True
        st.success("Model loaded successfully!")
    
    # Sidebar navigation
    page = st.sidebar.selectbox("Navigation", ["Upload & Process", "Map Visualization"])
    
    if page == "Upload & Process":
        upload_and_process_page()
    elif page == "Map Visualization":
        map_visualization_page()


def upload_and_process_page():
    """Page for uploading videos and processing them."""
    st.header("Upload & Process Video")
    
    # Video upload
    st.subheader("1. Upload Inspection Video")
    uploaded_video = st.file_uploader(
        "Choose a video file",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Upload your inspection video file"
    )
    
    # GPS data upload (optional)
    st.subheader("2. Upload GPS Data (Optional)")
    st.markdown("""
    Upload GPS data file in CSV or JSON format.
    
    **CSV Format:**
    ```
    timestamp,latitude,longitude
    12.5,-37.8136,144.9631
    13.0,-37.8137,144.9632
    ```
    
    **JSON Format:**
    ```json
    [
        {"timestamp": 12.5, "latitude": -37.8136, "longitude": 144.9631},
        {"timestamp": 13.0, "latitude": -37.8137, "longitude": 144.9632}
    ]
    ```
    """)
    uploaded_gps = st.file_uploader(
        "Choose a GPS data file",
        type=['csv', 'json'],
        help="Upload GPS data file (optional)"
    )
    
    # Processing parameters
    st.subheader("3. Processing Parameters")
    interval_seconds = st.slider(
        "Frame extraction interval (seconds)",
        min_value=0.1,
        max_value=1.0,
        value=0.2,
        step=0.1,
        help="Interval between extracted frames. Lower values = more frames but slower processing."
    )
    
    # Process button
    if st.button("Process Video", type="primary", disabled=uploaded_video is None):
        if uploaded_video is None:
            st.error("Please upload a video file first.")
            return
        
        # Save uploaded files
        video_id = generate_video_id()
        video_path = UPLOADS_DIR / f"{video_id}_{uploaded_video.name}"
        
        with open(video_path, "wb") as f:
            f.write(uploaded_video.getbuffer())
        
        st.success(f"Video saved: {uploaded_video.name}")
        
        # Parse GPS data if provided
        gps_data = None
        if uploaded_gps is not None:
            gps_file_path = UPLOADS_DIR / f"{video_id}_gps{Path(uploaded_gps.name).suffix}"
            with open(gps_file_path, "wb") as f:
                f.write(uploaded_gps.getbuffer())
            
            try:
                gps_data = parse_gps_file(gps_file_path)
                st.success(f"GPS data loaded: {len(gps_data)} points")
            except Exception as e:
                st.warning(f"Failed to parse GPS data: {e}. Processing without GPS.")
        
        # Process video
        st.subheader("Processing Status")
        progress_bar = st.progress(0)
        status_text = st.empty()
        log_container = st.empty()
        
        # Store progress in session state for callback
        if 'processing_log' not in st.session_state:
            st.session_state.processing_log = []
        
        def update_progress(progress, message):
            """Callback function to update progress bar and status."""
            progress_bar.progress(progress)
            status_text.text(message)
            st.session_state.processing_log.append(f"[{progress*100:.1f}%] {message}")
            # Show last 5 log entries
            log_text = "\n".join(st.session_state.processing_log[-5:])
            log_container.text(log_text)
        
        try:
            # Verify model is loaded
            if st.session_state.model is None or st.session_state.processor is None:
                st.error("Model not loaded! Please refresh the page.")
                return
            
            st.session_state.processing_log = []
            status_text.text("Starting video processing...")
            progress_bar.progress(0.05)
            
            print(f"Starting video processing: {video_path}")
            print(f"Video file exists: {video_path.exists()}")
            print(f"Video file size: {video_path.stat().st_size / (1024*1024):.2f} MB")
            
            # Process video with progress callback
            st.info("⏳ **Processing may take 5-10 minutes per frame** - VLM inference is computationally intensive. Please be patient and check the terminal for progress updates.")
            
            with st.spinner("Processing video... This may take several minutes."):
                results = process_video(
                    video_path=video_path,
                    gps_data=gps_data,
                    interval_seconds=interval_seconds,
                    model=st.session_state.model,
                    processor=st.session_state.processor,
                    output_base_dir=PROCESSED_DIR,
                    video_id=video_id,
                    progress_callback=update_progress
                )
            
            print(f"Processing complete. Found {len(results)} signs.")
            
            progress_bar.progress(0.95)
            status_text.text("Saving results...")
            
            # Save results
            storage.add_video_results(video_id, str(video_path), results)
            
            progress_bar.progress(1.0)
            status_text.text("Processing complete!")
            
            st.success(f"✅ Processing complete! Detected {len(results)} parking signs.")
            
            # Show summary
            if results:
                st.subheader("Processing Summary")
                summary_df = pd.DataFrame([
                    {
                        "Sign Index": r.get('sign_index'),
                        "Text": r.get('extracted_text', 'N/A'),
                        "Arrow": r.get('arrow_direction', 'none'),
                        "Frame Time": f"{r.get('frame_timestamp', 0):.1f}s",
                        "GPS": f"({r.get('latitude', 'N/A')}, {r.get('longitude', 'N/A')})" if r.get('latitude') else "N/A"
                    }
                    for r in results
                ])
                st.dataframe(summary_df, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error during processing: {e}")
            import traceback
            st.code(traceback.format_exc())


def map_visualization_page():
    """Page for visualizing detected signs on a map."""
    st.header("Map Visualization")
    
    # Load all results
    all_results = storage.load_results()
    
    if not all_results:
        st.info("No processed videos found. Please upload and process a video first.")
        return
    
    # Get all signs
    all_signs = storage.get_all_signs()
    
    if not all_signs:
        st.info("No parking signs detected yet.")
        return
    
    st.success(f"Found {len(all_signs)} parking signs from {len(all_results)} video(s)")
    
    # Filter signs with GPS coordinates
    signs_with_gps = [s for s in all_signs if s.get('latitude') and s.get('longitude')]
    
    if not signs_with_gps:
        st.warning("No signs with GPS coordinates found. GPS data may not have been provided during processing.")
        
        # Show signs without GPS
        st.subheader("Signs Without GPS Coordinates")
        signs_df = pd.DataFrame([
            {
                "Sign Index": s.get('sign_index'),
                "Text": s.get('extracted_text', 'N/A'),
                "Arrow": s.get('arrow_direction', 'none'),
                "Frame Time": f"{s.get('frame_timestamp', 0):.1f}s",
                "Video": Path(s.get('video_path', '')).name
            }
            for s in all_signs
        ])
        st.dataframe(signs_df, use_container_width=True)
        return
    
    # Create map data
    map_data = pd.DataFrame([
        {
            "lat": s['latitude'],
            "lon": s['longitude'],
            "text": s.get('extracted_text', 'N/A'),
            "arrow": s.get('arrow_direction', 'none'),
            "sign_index": s.get('sign_index'),
            "frame_time": f"{s.get('frame_timestamp', 0):.1f}s"
        }
        for s in signs_with_gps
    ])
    
    # Display map with zoom-responsive markers
    st.subheader("Parking Signs Map")
    
    # Calculate center point for initial view
    center_lat = map_data['lat'].mean()
    center_lon = map_data['lon'].mean()
    
    # Create pydeck layer with zoom-responsive markers
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_data,
        get_position=["lon", "lat"],
        get_color=[255, 0, 0, 160],  # Red color with transparency
        get_radius=50,  # Base radius in meters - will scale with zoom
        radius_min_pixels=3,  # Minimum size in pixels (when zoomed out)
        radius_max_pixels=15,  # Maximum size in pixels (when zoomed in)
        pickable=True,  # Enable tooltips
        auto_highlight=True,
    )
    
    # Create tooltip
    tooltip = {
        "html": "<b>Sign {sign_index}</b><br/>"
               "Text: {text}<br/>"
               "Arrow: {arrow}<br/>"
               "Time: {frame_time}",
        "style": {
            "backgroundColor": "steelblue",
            "color": "white"
        }
    }
    
    # Create view state
    view_state = pdk.ViewState(
        latitude=center_lat,
        longitude=center_lon,
        zoom=13,
        pitch=0,
        bearing=0
    )
    
    # Create deck with OpenStreetMap (no API key required)
    deck = pdk.Deck(
        map_style=None,  # Uses OpenStreetMap by default
        initial_view_state=view_state,
        layers=[layer],
        tooltip=tooltip
    )
    
    st.pydeck_chart(deck, use_container_width=True)
    
    # Show detailed table
    st.subheader("Detected Signs Details")
    
    # Add filter options
    col1, col2 = st.columns(2)
    with col1:
        filter_text = st.text_input("Filter by text", "")
    with col2:
        filter_arrow = st.selectbox("Filter by arrow direction", ["All", "left", "right", "none"])
    
    # Filter signs
    filtered_signs = signs_with_gps
    if filter_text:
        filtered_signs = [s for s in filtered_signs if filter_text.lower() in s.get('extracted_text', '').lower()]
    if filter_arrow != "All":
        filtered_signs = [s for s in filtered_signs if s.get('arrow_direction') == filter_arrow]
    
    # Display filtered results
    if filtered_signs:
        signs_df = pd.DataFrame([
            {
                "Sign Index": s.get('sign_index'),
                "Text": s.get('extracted_text', 'N/A'),
                "Arrow": s.get('arrow_direction', 'none'),
                "Frame Time": f"{s.get('frame_timestamp', 0):.1f}s",
                "Latitude": f"{s.get('latitude', 0):.6f}",
                "Longitude": f"{s.get('longitude', 0):.6f}",
                "Video": Path(s.get('video_path', '')).name
            }
            for s in filtered_signs
        ])
        st.dataframe(signs_df, use_container_width=True)
        
        # Show cropped sign images if available
        st.subheader("Sign Images")
        cols = st.columns(min(3, len(filtered_signs)))
        
        for idx, sign in enumerate(filtered_signs[:9]):  # Show max 9 images
            col = cols[idx % 3]
            with col:
                cropped_path = sign.get('cropped_image_path')
                if cropped_path and Path(cropped_path).exists():
                    st.image(str(cropped_path), caption=f"Sign {sign.get('sign_index')}: {sign.get('extracted_text', 'N/A')}")
                else:
                    st.info(f"Sign {sign.get('sign_index')}: Image not available")
    else:
        st.info("No signs match the filter criteria.")


if __name__ == "__main__":
    main()

