# Parking Sign Digital Platform MVP

A Streamlit-based MVP platform for processing inspection videos, detecting parking signs, extracting text via OCR, and visualizing results on an interactive map.

## Features

- **Video Upload**: Upload inspection videos for processing
- **GPS Data Integration**: Associate GPS coordinates with video frames (optional)
- **Automated Processing**: 
  - Frame extraction from videos
  - Parking sign detection using Qwen VLM
  - OCR text extraction from detected signs
  - Arrow direction detection
- **Map Visualization**: Interactive map showing all detected parking signs with GPS coordinates
- **Sign Details**: View cropped sign images and extracted text

## Installation

1. **Navigate to the platform directory:**
   ```bash
   cd parking_platform
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure existing scripts are accessible:**
   The platform imports functions from the following scripts in the parent directory:
   - `parking_sign_detection.py`
   - `parking_sign_ocr.py`
   - `extract_frames.py`

   Make sure these scripts are in the parent directory (`D:\Parking\`).

## Usage

1. **Start the Streamlit application:**
   ```bash
   streamlit run app.py
   ```

2. **Upload & Process:**
   - Navigate to the "Upload & Process" page
   - Upload an inspection video file (MP4, AVI, MOV, MKV)
   - (Optional) Upload GPS data file (CSV or JSON format)
   - Set frame extraction interval (default: 1 second)
   - Click "Process Video"
   - Wait for processing to complete

3. **View Results:**
   - Navigate to the "Map Visualization" page
   - View all detected parking signs on an interactive map
   - Filter signs by text or arrow direction
   - View cropped sign images and details

## GPS Data Format

### CSV Format
```csv
timestamp,latitude,longitude
12.5,-37.8136,144.9631
13.0,-37.8137,144.9632
```

### JSON Format
```json
[
    {"timestamp": 12.5, "latitude": -37.8136, "longitude": 144.9631},
    {"timestamp": 13.0, "latitude": -37.8137, "longitude": 144.9632}
]
```

**Notes:**
- Timestamp should be in seconds (relative to video start)
- GPS data is interpolated for frames between GPS points
- If GPS data is not provided, signs will be detected but won't appear on the map

## Project Structure

```
parking_platform/
├── app.py                      # Main Streamlit application
├── processing/
│   ├── video_processor.py      # Video frame extraction
│   ├── detection_wrapper.py    # Sign detection wrapper
│   ├── ocr_wrapper.py          # OCR extraction wrapper
│   └── pipeline.py             # Main processing pipeline
├── data/
│   ├── gps_handler.py         # GPS data parsing and association
│   ├── storage.py              # JSON-based data storage
│   └── results.json            # Stored processing results (created at runtime)
├── utils/
│   └── image_utils.py          # Image processing utilities
├── uploads/                    # Uploaded videos (created at runtime)
├── processed/                  # Processed frames and results (created at runtime)
├── requirements.txt
└── README.md
```

## Processing Pipeline

1. **Frame Extraction**: Extract frames from video at specified intervals
2. **Sign Detection**: Detect parking signs in each frame using Qwen VLM
3. **OCR Extraction**: Extract text and arrow direction from detected signs
4. **GPS Association**: Associate GPS coordinates with frames based on timestamps
5. **Image Cropping**: Crop detected sign regions and save for visualization
6. **Data Storage**: Save all results to JSON file

## Model Information

The platform uses the **Qwen3-VL-4B-Instruct** model from ModelScope for:
- Parking sign detection (bounding box coordinates)
- OCR text extraction
- Arrow direction detection

The model is loaded once at application startup and cached for subsequent processing.

## Limitations (MVP)

- No user authentication
- Simple JSON file storage (no database)
- Single-user interface
- Processing happens synchronously (may take time for long videos)
- Basic error handling

## Troubleshooting

**Model loading fails:**
- Ensure you have sufficient GPU memory (or use CPU mode)
- Check that ModelScope is properly installed and configured

**GPS data not working:**
- Verify GPS file format matches the expected CSV or JSON format
- Check that timestamps are in seconds relative to video start
- Ensure GPS data covers the video duration

**No signs detected:**
- Check that the video contains visible parking signs
- Try adjusting the frame extraction interval
- Verify the model loaded successfully

## Future Enhancements

- Real-time processing with progress updates
- Batch video processing
- Export results to CSV/Excel
- Advanced filtering and search
- Sign classification and categorization
- Database integration for production use

