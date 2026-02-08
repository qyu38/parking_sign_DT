import torch
from modelscope import Qwen3VLForConditionalGeneration, AutoProcessor
from PIL import Image
from pathlib import Path
import json
import re
from typing import List, Dict, Any, Optional
from tqdm import tqdm

# Set device and load model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load Qwen3-VL model (matching the notebook)
model_name = "Qwen/Qwen3-VL-4B-Instruct"
print(f"Loading model: {model_name}")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_name,
    dtype=torch.float16,
    device_map="auto"
).to(device)

processor = AutoProcessor.from_pretrained(model_name)
print("Model loaded successfully!")


def normalize_bounding_box(bbox: Any) -> List[float]:
    """
    Normalize bounding box to flattened format [x1, y1, x2, y2]
    Handles various input formats:
    - Flattened: [x1, y1, x2, y2]
    - Nested: [[x1, y1], [x2, y2]]
    - Triple nested: [[[x1, y1], [x2, y2]]]
    """
    if not bbox:
        return []
    
    # If already flattened and has 4 elements
    if isinstance(bbox, list) and len(bbox) == 4 and all(isinstance(x, (int, float)) for x in bbox):
        return [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
    
    # If nested format [[x1, y1], [x2, y2]]
    if isinstance(bbox, list) and len(bbox) == 2:
        if isinstance(bbox[0], list) and isinstance(bbox[1], list):
            if len(bbox[0]) == 2 and len(bbox[1]) == 2:
                return [float(bbox[0][0]), float(bbox[0][1]), float(bbox[1][0]), float(bbox[1][1])]
    
    # If triple nested [[[x1, y1], [x2, y2]]]
    if isinstance(bbox, list) and len(bbox) == 1 and isinstance(bbox[0], list):
        return normalize_bounding_box(bbox[0])
    
    # If it's a flat list with more than 4 elements, take first 4
    if isinstance(bbox, list) and len(bbox) >= 4:
        return [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
    
    # Return empty if can't normalize
    return []


def extract_json_from_text(text: str) -> Optional[List[Dict]]:
    """Extract JSON from model output, handling markdown code blocks"""
    # Remove markdown code blocks if present
    text = text.strip()
    
    # Try to find JSON in code blocks
    json_match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', text, re.DOTALL)
    if json_match:
        text = json_match.group(1)
    else:
        # Try to find JSON array directly - find the first [ and last ]
        start_idx = text.find('[')
        end_idx = text.rfind(']')
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            text = text[start_idx:end_idx + 1]
    
    # Clean up the text
    text = text.strip()
    
    # Try to fix common JSON issues
    # Remove trailing commas before } or ]
    text = re.sub(r',\s*}', '}', text)
    text = re.sub(r',\s*]', ']', text)
    # Fix single quotes to double quotes (basic)
    text = re.sub(r"'(\w+)':", r'"\1":', text)
    text = re.sub(r":\s*'([^']*)'", r': "\1"', text)
    
    try:
        parsed = json.loads(text)
        # Validate it's a list
        if isinstance(parsed, list):
            return parsed
        return None
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        # Try one more time with more aggressive cleaning
        try:
            # Remove any text before first [ and after last ]
            start = text.find('[')
            end = text.rfind(']')
            if start != -1 and end != -1:
                text = text[start:end+1]
                parsed = json.loads(text)
                return parsed if isinstance(parsed, list) else None
        except:
            pass
        return None


def convert_normalized_to_pixel(bbox_normalized: List[float], width: int, height: int) -> List[int]:
    """
    Convert normalized coordinates [0, 1000] to pixel coordinates.
    Handles both normalized [0, 1000] and already pixel coordinates.
    
    Args:
        bbox_normalized: Bounding box in normalized or pixel format [x1, y1, x2, y2]
        width: Image width in pixels
        height: Image height in pixels
    
    Returns:
        Bounding box in pixel coordinates [x1, y1, x2, y2]
    """
    if len(bbox_normalized) != 4:
        return []
    
    x1, y1, x2, y2 = bbox_normalized
    
    # Check if coordinates are already in pixel range (if max coord > 1000, assume pixels)
    if max(x1, y1, x2, y2) > 1000:
        # Already in pixel coordinates, just ensure they're integers
        return [int(x1), int(y1), int(x2), int(y2)]
    
    # Convert from normalized [0, 1000] to pixel coordinates
    x1_pixel = int((x1 * width) / 1000.0)
    y1_pixel = int((y1 * height) / 1000.0)
    x2_pixel = int((x2 * width) / 1000.0)
    y2_pixel = int((y2 * height) / 1000.0)
    
    # Ensure x1 < x2 and y1 < y2
    x_min = min(x1_pixel, x2_pixel)
    x_max = max(x1_pixel, x2_pixel)
    y_min = min(y1_pixel, y2_pixel)
    y_max = max(y1_pixel, y2_pixel)
    
    return [x_min, y_min, x_max, y_max]


def expand_bbox(bbox: List[int], image_width: int, image_height: int, expansion_factor: float = 0.1) -> List[int]:
    """
    Expand bounding box by a percentage while keeping the same center point.
    
    Args:
        bbox: Bounding box in pixel coordinates [x1, y1, x2, y2]
        image_width: Image width in pixels (for clamping)
        image_height: Image height in pixels (for clamping)
        expansion_factor: Expansion factor (0.1 = 10% expansion)
    
    Returns:
        Expanded bounding box [x1, y1, x2, y2] clamped to image boundaries
    """
    if len(bbox) != 4:
        return bbox
    
    x1, y1, x2, y2 = bbox
    
    # Calculate center point
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    
    # Calculate current width and height
    width = x2 - x1
    height = y2 - y1
    
    # Expand width and height by expansion_factor (10%)
    new_width = width * (1.0 + expansion_factor*2)
    new_height = height * (1.0 + expansion_factor)
    
    # Calculate new coordinates centered on the same center point
    new_x1 = center_x - new_width / 2.0
    new_x2 = center_x + new_width / 2.0
    new_y1 = center_y - new_height / 2.0
    new_y2 = center_y + new_height / 2.0
    
    # Clamp to image boundaries
    new_x1 = max(0, min(new_x1, image_width))
    new_x2 = max(0, min(new_x2, image_width))
    new_y1 = max(0, min(new_y1, image_height))
    new_y2 = max(0, min(new_y2, image_height))
    
    return [int(new_x1), int(new_y1), int(new_x2), int(new_y2)]


def detect_parking_signs(image_path: Path, model, processor) -> List[Dict[str, Any]]:
    """
    Detect Australian parking signs in a single image.
    Returns detections with bounding boxes only (no OCR).
    
    Args:
        image_path: Path to the image file
        model: Loaded Qwen-VL model
        processor: Model processor
    
    Returns:
        List of detection dictionaries with bounding boxes
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    image_width, image_height = image.size
    
    # Create simplified prompt for parking sign detection (no OCR)
    prompt = """You are analyzing an image for Australian parking signs. Detect all parking signs in the image and provide their bounding box coordinates.

CRITICAL: You MUST output ONLY valid JSON. No markdown, no explanations, no code blocks, no text before or after.

JSON SCHEMA (STRICT):
[
  {
    "bbox": [x1, y1, x2, y2]
  }
]

REQUIREMENTS:
1. Output format: A JSON array of objects
2. Each object MUST have:
   - "bbox": array with exactly 4 numbers [x1, y1, x2, y2] in normalized coordinates [0, 1000]
     where (x1,y1) is top-left corner and (x2,y2) is bottom-right corner
     IMPORTANT: Use normalized format where image width/height map to [0, 1000] range
3. If no parking signs found: return empty array []
4. Coordinates must be integers in normalized range [0, 1000]

INSTRUCTIONS:
- Examine the entire image carefully
- Identify all parking signs (any sign related to parking regulations)
- For each sign, provide bounding box coordinates in normalized format [0, 1000]
- Only detect parking signs, ignore other types of signs
- If multiple signs are present, include each one separately

OUTPUT ONLY THE JSON ARRAY (no other text):"""

    # Prepare message
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }
    ]
    
    # Prepare inputs
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)
    
    # Generate output with constrained parameters for JSON output
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,  # Use greedy decoding for deterministic output
            temperature=0.0,  # Zero temperature for maximum determinism
            pad_token_id=processor.tokenizer.eos_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            repetition_penalty=1.1  # Slight penalty to avoid repetition
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
    
    # Extract JSON from output
    json_data = extract_json_from_text(output_text)
    
    if json_data is None or not isinstance(json_data, list):
        return []
    
    # Process each detection
    detections = []
    for item in json_data:
        bbox_raw = item.get('bbox', [])
        
        # Normalize bounding box to flattened format
        bbox_normalized = normalize_bounding_box(bbox_raw)
        
        # Skip if bounding box couldn't be normalized
        if not bbox_normalized or len(bbox_normalized) != 4:
            continue
        
        # Convert normalized coordinates to pixel coordinates
        bbox_pixel = convert_normalized_to_pixel(bbox_normalized, image_width, image_height)
        
        # Skip if conversion failed
        if not bbox_pixel or len(bbox_pixel) != 4:
            continue
        
        # Validate bbox is not all zeros and has valid dimensions
        if bbox_pixel[0] == bbox_pixel[2] or bbox_pixel[1] == bbox_pixel[3]:
            continue
        
        # Expand bounding box by 10% while keeping the same center point
        bbox_expanded = expand_bbox(bbox_pixel, image_width, image_height, expansion_factor=0.1)
        
        detections.append({
            "bbox": bbox_expanded
        })
    
    return detections


def main():
    """Main function to process all images and save results."""
    # Input directory
    input_dir = Path(r"D:\Datasets\Google\split_180")
    
    if not input_dir.exists():
        print(f"Error: Directory {input_dir} does not exist!")
        return
    
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    # Get all image files
    image_files = [f for f in input_dir.iterdir() if f.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"No image files found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} image files to process")
    
    # Process all images
    results = []
    
    for image_path in tqdm(image_files, desc="Processing images"):
        try:
            detections = detect_parking_signs(image_path, model, processor)
            
            results.append({
                "image_path": str(image_path),
                "detections": detections
            })
        except Exception as e:
            print(f"\nError processing {image_path}: {e}")
            results.append({
                "image_path": str(image_path),
                "detections": [],
                "error": str(e)
            })
    
    # Save results to JSON
    output_file = Path("parking_detections.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nProcessing complete! Results saved to {output_file}")
    print(f"Total images processed: {len(results)}")
    print(f"Images with detections: {sum(1 for r in results if r.get('detections'))}")
    total_detections = sum(len(r.get('detections', [])) for r in results)
    print(f"Total parking signs detected: {total_detections}")


if __name__ == "__main__":
    main()
