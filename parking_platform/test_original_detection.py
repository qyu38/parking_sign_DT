"""
Test the original parking_sign_detection.py to see if it works on the same image.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, '..')

# Find test image
processed_dir = Path("processed/frames")
test_image = None

if processed_dir.exists():
    for video_dir in processed_dir.iterdir():
        if video_dir.is_dir():
            frames = list(video_dir.glob("*.jpg")) + list(video_dir.glob("*.png"))
            if frames:
                test_image = frames[0]
                break

if not test_image:
    print("No test image found!")
    sys.exit(1)

print(f"Testing original detection script on: {test_image}")
print("=" * 60)

# Import and test original script
try:
    # We need to modify the original script to not load model at import
    # Instead, let's manually test the detection function
    from parking_sign_detection import detect_parking_signs
    
    print("Loading model (this may take a moment)...")
    import torch
    from modelscope import Qwen3VLForConditionalGeneration, AutoProcessor
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "Qwen/Qwen3-VL-4B-Instruct"
    
    # Load model without .to(device) to avoid error
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_name)
    
    print("Model loaded. Running detection...")
    detections = detect_parking_signs(test_image, model, processor)
    
    print(f"\nResults: {len(detections)} detections found")
    for i, det in enumerate(detections):
        print(f"  Detection {i+1}: {det}")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

