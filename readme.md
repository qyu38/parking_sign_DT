# Parking Sign Detection Platform

Automated parking sign detection and OCR system using Qwen3-VL-4B-Instruct vision-language model. Processes inspection videos to detect parking signs, extract text, and geotag with GPS coordinates.

## Architecture

**Core Components:**
- **Standalone Scripts** (root): Original detection/OCR implementations
- **parking_platform/**: Streamlit MVP application with modular processing pipeline

## Directory Structure

### Root Level Scripts

- `parking_sign_detection.py` - Standalone parking sign detection using Qwen VLM. Detects bounding boxes in images, outputs JSON with normalized coordinates.
- `parking_sign_ocr.py` - Standalone OCR extraction from detected sign regions. Extracts text and arrow directions.
- `extract_frames.py` - Video frame extraction utility. Extracts frames at configurable intervals using OpenCV.
- `visualize_parking_detections.py` - Visualization tool for detection results.

### parking_platform/ - Streamlit Application

**Entry Point:**
- `app.py` - Main Streamlit app with upload/process and map visualization pages.

**Processing Pipeline:**
- `processing/pipeline.py` - Orchestrates video processing workflow (frame extraction → detection → OCR → GPS association).
- `processing/video_processor.py` - Frame extraction from videos at configurable intervals.
- `processing/detection_wrapper.py` - Wrapper for Qwen VLM sign detection. Handles device mapping, bbox normalization, coordinate conversion.
- `processing/ocr_wrapper.py` - OCR extraction from cropped sign regions. Parses JSON output, extracts text and arrow directions.

**Data Management:**
- `data/storage.py` - JSON-based storage for processing results (MVP).
- `data/gps_handler.py` - GPS data parsing (CSV/JSON) and timestamp-based frame association with interpolation.
- `data/results.json` - Runtime storage file (generated).

**Utilities:**
- `utils/image_utils.py` - Image cropping and processing utilities.

**Runtime Directories:**
- `uploads/` - User-uploaded videos (excluded from git).
- `processed/frames/` - Extracted video frames.
- `processed/cropped_signs/` - Cropped sign images (excluded from git).

**Test Files:**
- `test_*.py` - Unit tests for detection, OCR, and pipeline components.

## Model

Uses **Qwen3-VL-4B-Instruct** from ModelScope:
- Detection: Returns bounding boxes in normalized [0, 1000] coordinates
- OCR: Extracts text and arrow direction (left/right/none) from cropped regions
- Device: Auto device mapping with CUDA support

## Usage

**Standalone Scripts:**
```bash
python parking_sign_detection.py  # Process images
python parking_sign_ocr.py        # Extract OCR from detections
python extract_frames.py video.mp4 1.0  # Extract frames
```

**Streamlit App:**
```bash
cd parking_platform
pip install -r requirements.txt
streamlit run app.py
```

## Data Flow

1. Video → Frame extraction (configurable interval)
2. Frame → VLM detection (bounding boxes)
3. Bbox → Crop → VLM OCR (text + arrow direction)
4. Frame timestamp → GPS interpolation → Geotagging
5. Results → JSON storage + Map visualization

## Dependencies

Core: `torch`, `modelscope`, `opencv-python`, `pillow`, `streamlit`, `pydeck`

See `parking_platform/requirements.txt` for full list.
