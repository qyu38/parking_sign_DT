"""
Visualize bounding boxes from parking_detections.json on images
Displays all detected parking signs with their bounding boxes drawn on the images
"""
# run: python visualize_parking_detections.py parking_detections.json

import json
import os
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from collections import defaultdict


def load_detections(json_path: str) -> list:
    """
    Load detections from parking_detections.json file
    Returns a list of detection entries with image_path and detections
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        detections = json.load(f)
    
    return detections


def draw_bounding_boxes(image_path: str, detections: list, show_labels: bool = True):
    """
    Draw bounding boxes on an image and display it
    
    Args:
        image_path: Path to the image file
        detections: List of detection dictionaries with bounding boxes
        show_labels: Whether to show labels on bounding boxes
    """
    # Load image
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return
    
    # Create a drawing context
    draw = ImageDraw.Draw(img)
    img_width, img_height = img.size
    
    # Try to load a font (fallback to default if not available)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
        font_small = ImageFont.truetype("arial.ttf", 12)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        except:
            font = ImageFont.load_default()
            font_small = ImageFont.load_default()
    
    # Colors for different detections (cycling through colors)
    colors = [
        (255, 0, 0),      # Red
        (0, 255, 0),      # Green
        (0, 0, 255),      # Blue
        (255, 255, 0),    # Yellow
        (255, 0, 255),    # Magenta
        (0, 255, 255),    # Cyan
        (255, 128, 0),    # Orange
        (128, 0, 255),    # Purple
    ]
    
    # Draw each bounding box
    for idx, detection in enumerate(detections):
        bbox = detection.get('bbox', [])
        
        if not bbox or len(bbox) < 4:
            print(f"  Warning: Invalid bounding box, skipping...")
            continue
        
        # Extract coordinates (already in pixel format: [x1, y1, x2, y2])
        x1 = float(bbox[0])
        y1 = float(bbox[1])
        x2 = float(bbox[2])
        y2 = float(bbox[3])
        
        # Ensure coordinates are within image bounds
        x1 = max(0, min(x1, img_width))
        y1 = max(0, min(y1, img_height))
        x2 = max(0, min(x2, img_width))
        y2 = max(0, min(y2, img_height))
        
        # Ensure coordinates are in correct order
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        # Skip if bbox is invalid (zero area)
        if x1 == x2 or y1 == y2:
            continue
        
        # Get color for this detection
        color = colors[idx % len(colors)]
        
        # Draw bounding box rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # Draw label background and text
        if show_labels:
            # Prepare label text
            label = "Parking Sign"
            
            # Calculate text size
            try:
                bbox_text = draw.textbbox((0, 0), label, font=font_small)
                text_width = bbox_text[2] - bbox_text[0]
                text_height = bbox_text[3] - bbox_text[1]
            except:
                # Fallback if textbbox not available
                text_width = len(label) * 6
                text_height = 16
            
            # Draw label background
            label_y = max(0, y1 - text_height - 4)
            draw.rectangle(
                [x1, label_y, x1 + text_width + 8, label_y + text_height + 4],
                fill=color,
                outline=color
            )
            
            # Draw label text
            draw.text(
                (x1 + 4, label_y + 2),
                label,
                fill=(255, 255, 255),
                font=font_small
            )
    
    # Display the image
    plt.figure(figsize=(12, 8))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"{Path(image_path).name} - {len(detections)} detection(s)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def visualize_detections(json_path: str):
    """
    Main function to visualize all detections from parking_detections.json on their respective images
    
    Args:
        json_path: Path to parking_detections.json file
    """
    # Load detections
    print(f"Loading detections from {json_path}...")
    all_detections = load_detections(json_path)
    
    if not all_detections:
        print("No detections found in parking_detections.json")
        return
    
    print(f"Found detections for {len(all_detections)} image(s)")
    
    # Process each image
    for entry in all_detections:
        image_path = entry.get('image_path', '')
        detections = entry.get('detections', [])
        
        if not image_path:
            continue
        
        # Check if image exists
        image_path_obj = Path(image_path)
        if not image_path_obj.exists():
            print(f"\nWarning: Image not found: {image_path}")
            continue
        
        # Skip if no detections
        if not detections:
            print(f"\nSkipping {image_path_obj.name}: No detections")
            continue
        
        print(f"\nProcessing: {image_path_obj.name}")
        print(f"  Found {len(detections)} detection(s):")
        for idx, det in enumerate(detections):
            bbox = det.get('bbox', [])
            print(f"    - Detection {idx + 1}: bbox {bbox}")
        
        # Draw and display
        draw_bounding_boxes(str(image_path_obj), detections)
    
    print("\nVisualization complete!")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize_parking_detections.py <parking_detections.json>")
        print("\nExample:")
        print("  python visualize_parking_detections.py parking_detections.json")
        sys.exit(1)
    
    json_path = sys.argv[1]
    
    if not os.path.exists(json_path):
        print(f"Error: JSON file not found: {json_path}")
        sys.exit(1)
    
    visualize_detections(json_path)

