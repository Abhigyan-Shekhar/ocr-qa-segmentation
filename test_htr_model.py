#!/usr/bin/env python3
"""
Quick test script for HTR model inference
Tests if the model can be loaded and run predictions
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scripts.inference_htr import HTRModel

def main():
    print("=" * 60)
    print("HTR Model Test")
    print("=" * 60)
    
    # Check if model exists
    model_path = "models/htr_model.keras"
    config_path = "models/config.json"
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("Please download the model from Kaggle first!")
        return 1
    
    print(f"\n✅ Model found: {model_path}")
    print(f"   Size: {os.path.getsize(model_path) / (1024**2):.1f} MB")
    
    # Load model
    try:
        htr = HTRModel(model_path, config_path)
        print("\n✅ Model loaded successfully!")
        print(f"   Image dimensions: {htr.img_height}x{htr.img_width}")
        print(f"   Character set size: {len(htr.characters)}")
    except Exception as e:
        print(f"\n❌ Error loading model: {e}")
        return 1
    
    print("\n" + "=" * 60)
    print("Model is ready for inference!")
    print("=" * 60)
    print("\nUsage examples:")
    print("  # Single image:")
    print("  python scripts/inference_htr.py --image test.png")
    print("\n  # Directory of images:")
    print("  python scripts/inference_htr.py --images-dir test_images/")
    print("\n  # Save results to file:")
    print("  python scripts/inference_htr.py --image test.png --output results.txt")
    print("=" * 60)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
