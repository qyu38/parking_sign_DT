"""
Video processing module for extracting frames from videos.
Adapted from extract_frames.py to work as a module function.
"""
import math
from pathlib import Path
from typing import List, Tuple
import cv2


def extract_frames_from_video(video_path: Path, interval_seconds: float, output_dir: Path) -> List[Tuple[Path, float]]:
    """
    Extract frames from a video at specified intervals.
    
    Args:
        video_path: Path to the video file
        interval_seconds: Interval between extracted frames in seconds
        output_dir: Directory to save extracted frames
    
    Returns:
        List of tuples: (frame_path, timestamp_in_seconds)
    """
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be greater than zero.")
    
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    
    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        capture.release()
        raise RuntimeError("Unable to determine FPS from video stream.")
    
    frame_interval = max(1, math.floor(fps * interval_seconds))
    frame_idx = 0
    saved_count = 0
    frames_with_timestamps = []
    
    success, frame = capture.read()
    while success:
        if frame_idx % frame_interval == 0:
            frame_name = f"frame_{saved_count:06d}.jpg"
            frame_path = output_dir / frame_name
            cv2.imwrite(str(frame_path), frame)
            
            # Calculate timestamp in seconds
            timestamp = frame_idx / fps
            frames_with_timestamps.append((frame_path, timestamp))
            saved_count += 1
        
        frame_idx += 1
        success, frame = capture.read()
    
    capture.release()
    return frames_with_timestamps

