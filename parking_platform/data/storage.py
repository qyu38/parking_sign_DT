"""
Data storage module for saving and loading processing results.
Uses JSON file storage for MVP.
"""
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid


class Storage:
    """Simple JSON-based storage for processing results."""
    
    def __init__(self, storage_file: Path):
        """
        Initialize storage.
        
        Args:
            storage_file: Path to JSON file for storing results
        """
        self.storage_file = storage_file
        self.storage_file.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_storage_file()
    
    def _ensure_storage_file(self):
        """Ensure storage file exists with proper structure."""
        if not self.storage_file.exists():
            self._data = {"videos": []}
            self._save()
        else:
            self._load()
    
    def _load(self):
        """Load data from storage file."""
        try:
            with open(self.storage_file, 'r', encoding='utf-8') as f:
                self._data = json.load(f)
                if 'videos' not in self._data:
                    self._data['videos'] = []
        except (json.JSONDecodeError, FileNotFoundError):
            self._data = {"videos": []}
            self._save()
    
    def _save(self):
        """Save data to storage file."""
        with open(self.storage_file, 'w', encoding='utf-8') as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)
    
    def add_video_results(self, video_id: str, video_path: str, results: List[Dict]) -> None:
        """
        Add processing results for a video.
        
        Args:
            video_id: Unique identifier for the video
            video_path: Path to the original video file
            results: List of sign detection results
        """
        video_entry = {
            "video_id": video_id,
            "video_path": str(video_path),
            "upload_timestamp": datetime.now().isoformat(),
            "signs": results
        }
        
        self._data['videos'].append(video_entry)
        self._save()
    
    def load_results(self) -> List[Dict]:
        """
        Load all processing results.
        
        Returns:
            List of video entries with their detected signs
        """
        return self._data.get('videos', [])
    
    def get_all_signs(self) -> List[Dict]:
        """
        Get all detected signs from all videos.
        
        Returns:
            List of all sign detections with video metadata
        """
        all_signs = []
        for video in self._data.get('videos', []):
            video_id = video.get('video_id')
            video_path = video.get('video_path')
            for sign in video.get('signs', []):
                sign_with_metadata = sign.copy()
                sign_with_metadata['video_id'] = video_id
                sign_with_metadata['video_path'] = video_path
                all_signs.append(sign_with_metadata)
        return all_signs


def generate_video_id() -> str:
    """Generate a unique video ID."""
    return str(uuid.uuid4())

