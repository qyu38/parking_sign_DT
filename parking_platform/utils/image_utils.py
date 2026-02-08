"""
Image processing utilities for cropping and saving sign regions.
"""
from pathlib import Path
from PIL import Image
from typing import List


def crop_sign_region(image_path: Path, bbox: List[int]) -> Image.Image:
    """
    Crop a sign region from an image based on bounding box.
    
    Args:
        image_path: Path to the full image
        bbox: Bounding box [x1, y1, x2, y2] in pixel coordinates
    
    Returns:
        Cropped PIL Image
    """
    image = Image.open(image_path).convert('RGB')
    x1, y1, x2, y2 = bbox
    
    # Ensure coordinates are within image bounds
    width, height = image.size
    x1 = max(0, min(x1, width))
    y1 = max(0, min(y1, height))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))
    
    # Ensure x1 < x2 and y1 < y2
    x_min = min(x1, x2)
    x_max = max(x1, x2)
    y_min = min(y1, y2)
    y_max = max(y1, y2)
    
    # Crop the image
    cropped = image.crop((x_min, y_min, x_max, y_max))
    return cropped


def save_cropped_sign(image: Image.Image, output_path: Path) -> Path:
    """
    Save a cropped sign image to disk.
    
    Args:
        image: PIL Image to save
        output_path: Path where to save the image
    
    Returns:
        Path to saved image
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return output_path

