"""
Main processing pipeline that orchestrates video processing, sign detection, and OCR.
"""
from pathlib import Path
from typing import List, Dict, Any, Optional
import sys

# Import processing modules
from .video_processor import extract_frames_from_video
from .detection_wrapper import detect_signs_in_image
from .ocr_wrapper import extract_text_from_signs

# Import utilities
parent_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_dir))
from utils.image_utils import crop_sign_region, save_cropped_sign
from data.gps_handler import associate_gps_with_frame


def process_video(
    video_path: Path,
    gps_data: Optional[List[Dict]],
    interval_seconds: float,
    model,
    processor,
    output_base_dir: Path,
    video_id: str,
    progress_callback=None
) -> List[Dict[str, Any]]:
    """
    Process a video through the full pipeline: frame extraction, detection, OCR, GPS association.
    
    Args:
        video_path: Path to the video file
        gps_data: List of GPS data points (from gps_handler.parse_gps_file)
        interval_seconds: Interval between extracted frames in seconds
        model: Loaded Qwen-VL model
        processor: Model processor
        output_base_dir: Base directory for saving processed frames and cropped signs
        video_id: Unique identifier for this video
    
    Returns:
        List of sign detection results with all metadata
    """
    # Create output directories
    frames_dir = output_base_dir / "frames" / video_id
    cropped_dir = output_base_dir / "cropped_signs" / video_id
    
    # Step 1: Extract frames from video
    print(f"Extracting frames from video: {video_path}")
    if progress_callback:
        progress_callback(0.1, "Extracting frames from video...")
    
    try:
        frames_with_timestamps = extract_frames_from_video(
            video_path, interval_seconds, frames_dir
        )
    except Exception as e:
        print(f"Error extracting frames: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    total_frames = len(frames_with_timestamps)
    if total_frames == 0:
        print("Warning: No frames extracted from video")
        return []
    
    if progress_callback:
        progress_callback(0.2, f"Extracted {total_frames} frames. Starting detection...")
    
    all_signs = []
    sign_counter = 1
    
    # Step 2: Process each frame
    for frame_idx, (frame_path, frame_timestamp) in enumerate(frames_with_timestamps):
        if progress_callback:
            progress = 0.2 + (frame_idx / total_frames) * 0.7
            progress_callback(progress, f"Processing frame {frame_idx + 1}/{total_frames}...")
        
        try:
            # Step 2a: Detect parking signs
            print(f"\n{'='*60}")
            print(f"Processing frame {frame_idx + 1}/{total_frames}: {frame_path.name}")
            print(f"{'='*60}")
            print("  [1/2] Running sign detection (this may take 10-30 seconds)...")
            import time
            start_time = time.time()
            detections = detect_signs_in_image(frame_path, model, processor)
            detection_time = time.time() - start_time
            print(f"  ✓ Detection complete in {detection_time:.1f}s - Found {len(detections)} detections")
            
            if not detections:
                print("  → No signs detected, skipping OCR for this frame")
                continue
            
            # Step 2b: Extract OCR text from detected signs
            print(f"  [2/2] Running OCR extraction for {len(detections)} sign(s)...")
            ocr_start_time = time.time()
            signs = extract_text_from_signs(frame_path, detections, model, processor)
            ocr_time = time.time() - ocr_start_time
            print(f"  ✓ OCR complete in {ocr_time:.1f}s - Extracted text from {len(signs)} signs")
        
            # Step 2c: Associate GPS coordinates
            gps_coords = None
            if gps_data:
                gps_coords = associate_gps_with_frame(frame_timestamp, gps_data)
            
            # Step 2d: Process each detected sign
            for sign in signs:
                # Find the corresponding detection for this sign
                sign_index = sign.get('sign_index', sign_counter)
                detection_idx = min(sign_index - 1, len(detections) - 1)
                detection = detections[detection_idx] if detections else {}
                bbox = detection.get('bbox', [])
                
                # Crop and save the sign image
                cropped_image = None
                cropped_image_path = None
                if bbox and len(bbox) == 4:
                    try:
                        cropped_image = crop_sign_region(frame_path, bbox)
                        cropped_filename = f"sign_{sign_counter:06d}.jpg"
                        cropped_image_path = cropped_dir / cropped_filename
                        save_cropped_sign(cropped_image, cropped_image_path)
                    except Exception as e:
                        print(f"Warning: Failed to crop sign from {frame_path}: {e}")
                
                # Create sign result entry
                sign_result = {
                    "sign_index": sign_counter,
                    "frame_path": str(frame_path),
                    "frame_timestamp": frame_timestamp,
                    "bbox": bbox,
                    "extracted_text": sign.get('extracted_text', ''),
                    "arrow_direction": sign.get('arrow_direction', 'none'),
                    "cropped_image_path": str(cropped_image_path) if cropped_image_path else None
                }
                
                # Add GPS coordinates if available
                if gps_coords:
                    sign_result['latitude'] = gps_coords['latitude']
                    sign_result['longitude'] = gps_coords['longitude']
                
                all_signs.append(sign_result)
                sign_counter += 1
        
        except Exception as e:
            print(f"Error processing frame {frame_path}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if progress_callback:
        progress_callback(0.95, f"Processing complete! Found {len(all_signs)} signs.")
    
    return all_signs

