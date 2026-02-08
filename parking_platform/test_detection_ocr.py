"""
Test script to verify detection and OCR wrappers work correctly.
Tests on actual frame images from processed videos.
"""
import sys
from pathlib import Path
from PIL import Image
import torch

print("=" * 60)
print("Testing Detection and OCR Wrappers")
print("=" * 60)

# Load model
print("\n1. Loading model...")
from processing.detection_wrapper import load_detection_model, detect_signs_in_image
from processing.ocr_wrapper import extract_text_from_signs

model, processor = load_detection_model()
print("   ✓ Model loaded!")

# Find test images
print("\n2. Looking for test images...")
processed_dir = Path("processed/frames")
test_images = []

if processed_dir.exists():
    # Find any frame images
    for video_dir in processed_dir.iterdir():
        if video_dir.is_dir():
            frames = list(video_dir.glob("*.jpg")) + list(video_dir.glob("*.png"))
            if frames:
                test_images.extend(frames[:3])  # Take first 3 frames
                break

if not test_images:
    print("   ⚠ No processed frames found. Please process a video first.")
    print("   Looking for any images in uploads or test directories...")
    # Check uploads or parent directory
    uploads_dir = Path("uploads")
    if uploads_dir.exists():
        test_images = list(uploads_dir.glob("*.jpg")) + list(uploads_dir.glob("*.png"))
    
    # Check parent directory for any test images
    parent_dir = Path("..")
    if not test_images:
        test_images = list(parent_dir.glob("*.jpg")) + list(parent_dir.glob("*.png"))[:1]

if not test_images:
    print("   ✗ No test images found!")
    print("   Please provide a test image path or process a video first.")
    sys.exit(1)

print(f"   ✓ Found {len(test_images)} test image(s)")

# Test detection on first image
test_image = test_images[0]
print(f"\n3. Testing detection on: {test_image.name}")
print(f"   Image path: {test_image}")

try:
    # Verify image exists and can be opened
    img = Image.open(test_image)
    print(f"   Image size: {img.size}")
    print(f"   Image mode: {img.mode}")
    
    print("\n   Running detection (this may take 10-30 seconds)...")
    detections = detect_signs_in_image(test_image, model, processor)
    
    print(f"\n   Detection Results:")
    print(f"   - Number of detections: {len(detections)}")
    
    if detections:
        for i, det in enumerate(detections):
            bbox = det.get('bbox', [])
            print(f"   - Detection {i+1}: bbox = {bbox}")
        
        # Test OCR on first detection
        print(f"\n4. Testing OCR extraction...")
        print(f"   Using {len(detections)} detection(s) from the image...")
        
        signs = extract_text_from_signs(test_image, detections, model, processor)
        
        print(f"\n   OCR Results:")
        print(f"   - Number of signs with text: {len(signs)}")
        
        for i, sign in enumerate(signs):
            text = sign.get('extracted_text', '')
            arrow = sign.get('arrow_direction', 'none')
            print(f"   - Sign {i+1}:")
            print(f"     Text: '{text}'")
            print(f"     Arrow: {arrow}")
    else:
        print("\n   ⚠ No detections found!")
        print("   This could indicate:")
        print("   - The image doesn't contain parking signs")
        print("   - The detection prompt needs adjustment")
        print("   - The model output format is incorrect")
        
        # Try to see what the model is actually outputting
        print("\n   Debugging: Checking model output...")
        # We'll add debug output to see raw model response
        
except Exception as e:
    print(f"\n   ✗ Error during testing: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("Testing complete!")
print("=" * 60)

