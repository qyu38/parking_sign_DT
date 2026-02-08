"""
Wrapper module for parking sign detection.
Self-contained to avoid importing from parking_sign_detection.py which loads model at import time.
"""
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from PIL import Image
import torch
import json
import re


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


def load_detection_model():
    """
    Load the Qwen VLM model for detection.
    This function should be called once at application startup.
    
    Returns:
        Tuple of (model, processor)
    """
    from modelscope import Qwen3VLForConditionalGeneration, AutoProcessor
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "Qwen/Qwen3-VL-4B-Instruct"
    
    # When using device_map="auto", don't call .to(device) as it conflicts with automatic device placement
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="auto"
    )
    
    processor = AutoProcessor.from_pretrained(model_name)
    
    return model, processor


def detect_signs_in_image(image_path: Path, model, processor) -> List[Dict[str, Any]]:
    """
    Detect parking signs in an image.
    Wrapper that fixes device handling for models with device_map="auto".
    
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
    prompt = """Analyze this image and detect all parking signs. Parking signs include signs with text like "2HR", "1P", "PERMIT ZONE", "NO PARKING", "TICKET", time restrictions (e.g., "8AM-6PM"), or any sign related to parking regulations.

CRITICAL: Output ONLY valid JSON array. No markdown, no explanations, no code blocks, no text before or after.

JSON FORMAT:
[
  {
    "bbox": [x1, y1, x2, y2]
  }
]

REQUIREMENTS:
- Output a JSON array of objects
- Each object must have "bbox" with exactly 4 numbers [x1, y1, x2, y2]
- Coordinates must be in normalized format [0, 1000] where:
  - x1, y1 = top-left corner
  - x2, y2 = bottom-right corner
  - Image width maps to x range [0, 1000]
  - Image height maps to y range [0, 1000]
- If no parking signs found, return: []
- Coordinates must be integers

INSTRUCTIONS:
1. Scan the entire image carefully for parking signs
2. Look for rectangular signs with parking-related text or symbols
3. For each parking sign found, provide bounding box coordinates
4. Include all visible parking signs, even if partially visible
5. Use normalized coordinates [0, 1000] based on image dimensions

OUTPUT ONLY THE JSON ARRAY:"""

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
    
    # Fix device handling for device_map="auto"
    try:
        device = next(model.parameters()).device
        inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}
    except (StopIteration, AttributeError):
        # Fallback: try model.device if available
        if hasattr(model, 'device'):
            inputs = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        else:
            # Last resort: use CPU
            inputs = {k: v.to('cpu') if hasattr(v, 'to') else v for k, v in inputs.items()}
    
    # Generate output
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            repetition_penalty=1.1
        )
        # Handle both dict and object access for input_ids
        input_ids = inputs['input_ids'] if isinstance(inputs, dict) else inputs.input_ids
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
    
    # Extract JSON from output
    json_data = extract_json_from_text(output_text)
    
    if json_data is None or not isinstance(json_data, list):
        # Debug only if extraction fails
        print(f"WARNING: JSON extraction failed. Raw output: {output_text[:200]}")
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
