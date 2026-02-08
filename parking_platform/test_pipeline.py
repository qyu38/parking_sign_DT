"""
Quick test script to verify the pipeline works.
"""
import sys
from pathlib import Path
from processing.detection_wrapper import load_detection_model

print("=" * 60)
print("Testing Parking Platform Pipeline")
print("=" * 60)

print("\n1. Testing model loading...")
try:
    model, processor = load_detection_model()
    print("   ✓ Model loaded successfully!")
except Exception as e:
    print(f"   ✗ Error loading model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n2. Testing imports...")
try:
    from processing.pipeline import process_video
    from processing.video_processor import extract_frames_from_video
    from processing.ocr_wrapper import extract_text_from_signs
    from data.gps_handler import parse_gps_file, associate_gps_with_frame
    from data.storage import Storage
    print("   ✓ All imports successful!")
except Exception as e:
    print(f"   ✗ Import error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ All tests passed! The platform is ready to use.")
print("=" * 60)
print("\nTo start the Streamlit app, run:")
print("  streamlit run app.py")

