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

# Load Qwen3-VL model (matching parking_sign_detection.py)
model_name = "Qwen/Qwen3-VL-4B-Instruct"
print(f"Loading model: {model_name}")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_name,
    dtype=torch.float16,
    device_map="auto"
).to(device)

processor = AutoProcessor.from_pretrained(model_name)
print("Model loaded successfully!")


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


def crop_image(image: Image.Image, bbox: List[int]) -> Image.Image:
    """
    Crop image to bounding box region.
    
    Args:
        image: PIL Image object
        bbox: Bounding box [x1, y1, x2, y2] in pixel coordinates
    
    Returns:
        Cropped PIL Image
    """
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


def extract_text_from_sign(cropped_image: Image.Image, model, processor) -> List[Dict[str, Any]]:
    """
    Extract text and arrow direction from a cropped parking sign image.
    Handles multiple signs within a single bounding box.
    
    Args:
        cropped_image: PIL Image of the cropped sign region
        model: Loaded Qwen-VL model
        processor: Model processor
    
    Returns:
        List of sign dictionaries with sign_index, extracted_text, and arrow_direction
    """
    prompt = """You are analyzing a cropped image of an Australian parking sign. Extract all text content and detect arrow directions.

CRITICAL: You MUST output ONLY valid JSON. No markdown, no explanations, no code blocks, no text before or after.

JSON SCHEMA (STRICT):
[
  {
    "sign_index": 1,
    "extracted_text": "2HR 8AM-6PM",
    "arrow_direction": "left"
  }
]

REQUIREMENTS:
1. Output format: A JSON array of objects
2. Each object MUST have:
   - "sign_index": sequential integer starting from 1
   - "extracted_text": string containing ALL visible text, numbers, and symbols on the sign
   - "arrow_direction": one of "left", "right", or "none"
3. If multiple signs are visible in this cropped region, create separate objects for each
4. Extract ALL alphanumeric characters, numbers, and symbols (e.g., "2HR", "8AM-6PM", "PERMIT ZONE 5", etc.)
5. Arrow detection:
   - "left" if there is a left-pointing arrow (← or pointing left)
   - "right" if there is a right-pointing arrow (→ or pointing right)
   - "none" if no arrow or arrow points in other directions
   - "left and right" if the arrows point to both directions
6. If no signs found: return empty array []

INSTRUCTIONS:
- Examine the entire cropped image carefully
- Identify all parking signs visible in this region
- For each sign, extract ALL visible text, numbers, and symbols
- Detect arrow direction for each sign (left, right, or none, or left and right)
- Create separate entries for each distinct sign if multiple are present
- Preserve the exact text as it appears (including spaces, hyphens, etc.)

OUTPUT ONLY THE JSON ARRAY (no other text):"""

    # Prepare message
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": cropped_image
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
    
    # Generate output
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,  # Use greedy decoding for deterministic output
            temperature=0.0,  # Zero temperature for maximum determinism
            pad_token_id=processor.tokenizer.eos_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            repetition_penalty=1.1
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
    
    # Validate and clean up the results
    signs = []
    for item in json_data:
        if not isinstance(item, dict):
            continue
        
        sign_index = item.get('sign_index')
        extracted_text = item.get('extracted_text', '').strip()
        arrow_direction = item.get('arrow_direction', 'none').lower()
        
        # Validate arrow_direction
        if arrow_direction not in ['left', 'right', 'none']:
            arrow_direction = 'none'
        
        # Only add if we have text or a valid arrow
        if extracted_text or arrow_direction != 'none':
            signs.append({
                "sign_index": sign_index if isinstance(sign_index, int) else len(signs) + 1,
                "extracted_text": extracted_text,
                "arrow_direction": arrow_direction
            })
    
    return signs


def process_image_ocr(image_path: Path, detections: List[Dict], model, processor) -> Dict[str, Any]:
    """
    Process a single image to extract text from all detected parking signs.
    
    Args:
        image_path: Path to the image file
        detections: List of detection dictionaries with 'bbox' keys
        model: Loaded Qwen-VL model
        processor: Model processor
    
    Returns:
        Dictionary with image_path and signs array
    """
    # Load the full image
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return {
            "image_path": str(image_path),
            "signs": []
        }
    
    all_signs = []
    sign_counter = 1
    
    # Process each bounding box
    for detection in detections:
        bbox = detection.get('bbox', [])
        
        if len(bbox) != 4:
            continue
        
        # Crop the bounding box region
        cropped_image = crop_image(image, bbox)
        
        # Extract text from the cropped region
        signs = extract_text_from_sign(cropped_image, model, processor)
        
        # Update sign_index to be sequential across all bounding boxes
        for sign in signs:
            sign['sign_index'] = sign_counter
            sign_counter += 1
            all_signs.append(sign)
    
    return {
        "image_path": str(image_path),
        "signs": all_signs
    }


def main():
    """Main function to process all images and extract OCR text from parking signs."""
    # Input JSON file with bounding box detections
    input_json = Path("parking_detections.json")
    
    if not input_json.exists():
        print(f"Error: File {input_json} does not exist!")
        print("Please run parking_sign_detection.py first to generate bounding boxes.")
        return
    
    # Load detections
    print(f"Loading detections from {input_json}")
    with open(input_json, 'r', encoding='utf-8') as f:
        detection_data = json.load(f)
    
    if not detection_data:
        print("No detection data found in JSON file.")
        return
    
    print(f"Found {len(detection_data)} images to process")
    
    # Process all images
    results = []
    
    for item in tqdm(detection_data, desc="Processing images for OCR"):
        image_path = Path(item.get('image_path', ''))
        detections = item.get('detections', [])
        
        if not image_path.exists():
            print(f"\nWarning: Image not found: {image_path}")
            results.append({
                "image_path": str(image_path),
                "signs": []
            })
            continue
        
        if not detections:
            # No detections for this image
            results.append({
                "image_path": str(image_path),
                "signs": []
            })
            continue
        
        try:
            result = process_image_ocr(image_path, detections, model, processor)
            results.append(result)
        except Exception as e:
            print(f"\nError processing {image_path}: {e}")
            results.append({
                "image_path": str(image_path),
                "signs": []
            })
    
    # Save results to JSON
    output_file = Path("parking_sign_ocr_results.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nOCR processing complete! Results saved to {output_file}")
    print(f"Total images processed: {len(results)}")
    print(f"Images with extracted text: {sum(1 for r in results if r.get('signs'))}")
    total_signs = sum(len(r.get('signs', [])) for r in results)
    print(f"Total signs with extracted text: {total_signs}")


if __name__ == "__main__":
    main()

