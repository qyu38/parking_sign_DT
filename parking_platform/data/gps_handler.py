"""
GPS data handling module for parsing and associating GPS coordinates with video frames.
"""
import json
import csv
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime


def parse_gps_file(gps_file_path: Path) -> List[Dict]:
    """
    Parse GPS data from CSV or JSON file.
    
    Expected CSV format:
        timestamp,latitude,longitude
        12.5,-37.8136,144.9631
        13.0,-37.8137,144.9632
    
    Expected JSON format:
        [
            {"timestamp": 12.5, "latitude": -37.8136, "longitude": 144.9631},
            {"timestamp": 13.0, "latitude": -37.8137, "longitude": 144.9632}
        ]
    
    Args:
        gps_file_path: Path to GPS data file (CSV or JSON)
    
    Returns:
        List of dictionaries with 'timestamp', 'latitude', 'longitude' keys
    """
    gps_data = []
    
    if not gps_file_path.exists():
        raise FileNotFoundError(f"GPS file not found: {gps_file_path}")
    
    suffix = gps_file_path.suffix.lower()
    
    if suffix == '.json':
        with open(gps_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                gps_data = data
            else:
                raise ValueError("JSON file must contain a list of GPS points")
    
    elif suffix == '.csv':
        with open(gps_file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    timestamp = float(row.get('timestamp', 0))
                    latitude = float(row.get('latitude', 0))
                    longitude = float(row.get('longitude', 0))
                    gps_data.append({
                        'timestamp': timestamp,
                        'latitude': latitude,
                        'longitude': longitude
                    })
                except (ValueError, KeyError) as e:
                    print(f"Warning: Skipping invalid GPS row: {row}, Error: {e}")
                    continue
    
    else:
        raise ValueError(f"Unsupported GPS file format: {suffix}. Use .csv or .json")
    
    # Sort by timestamp
    gps_data.sort(key=lambda x: x['timestamp'])
    
    return gps_data


def associate_gps_with_frame(frame_timestamp: float, gps_data: List[Dict]) -> Optional[Dict]:
    """
    Associate GPS coordinates with a frame based on timestamp.
    Uses linear interpolation if frame timestamp is between GPS points.
    
    Args:
        frame_timestamp: Timestamp of the frame in seconds
        gps_data: List of GPS data points sorted by timestamp
    
    Returns:
        Dictionary with 'latitude' and 'longitude', or None if no GPS data available
    """
    if not gps_data:
        return None
    
    # If frame timestamp is before first GPS point, use first point
    if frame_timestamp <= gps_data[0]['timestamp']:
        return {
            'latitude': gps_data[0]['latitude'],
            'longitude': gps_data[0]['longitude']
        }
    
    # If frame timestamp is after last GPS point, use last point
    if frame_timestamp >= gps_data[-1]['timestamp']:
        return {
            'latitude': gps_data[-1]['latitude'],
            'longitude': gps_data[-1]['longitude']
        }
    
    # Find the two GPS points that bracket the frame timestamp
    for i in range(len(gps_data) - 1):
        t1 = gps_data[i]['timestamp']
        t2 = gps_data[i + 1]['timestamp']
        
        if t1 <= frame_timestamp <= t2:
            # Linear interpolation
            ratio = (frame_timestamp - t1) / (t2 - t1) if t2 != t1 else 0
            
            lat1 = gps_data[i]['latitude']
            lon1 = gps_data[i]['longitude']
            lat2 = gps_data[i + 1]['latitude']
            lon2 = gps_data[i + 1]['longitude']
            
            interpolated_lat = lat1 + (lat2 - lat1) * ratio
            interpolated_lon = lon1 + (lon2 - lon1) * ratio
            
            return {
                'latitude': interpolated_lat,
                'longitude': interpolated_lon
            }
    
    # Fallback: return last GPS point
    return {
        'latitude': gps_data[-1]['latitude'],
        'longitude': gps_data[-1]['longitude']
    }

