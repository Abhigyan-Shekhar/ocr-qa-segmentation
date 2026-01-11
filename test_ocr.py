#!/usr/bin/env python3
"""
Quick test script to verify OCR engine and PaddleOCR/Tesseract fallback.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from ocr_engine import OCREngine
import numpy as np

def main():
    print("=" * 60)
    print("OCR Engine Test - PaddleOCR vs Tesseract")
    print("=" * 60)
    print()
    
    # Test 1: Try PaddleOCR
    print("Test 1: Attempting PaddleOCR initialization...")
    try:
        ocr_paddle = OCREngine(engine='paddleocr')
        print("✅ SUCCESS: PaddleOCR loaded successfully!")
        print("   You have superior handwriting recognition!")
        paddle_available = True
    except ImportError as e:
        print(f"⚠️  PaddleOCR not available: {e}")
        print("   This is expected on Python 3.14+")
        paddle_available = False
    except Exception as e:
        print(f"❌ Error loading PaddleOCR: {e}")
        paddle_available = False
    
    print()
    
    # Test 2: Try Tesseract
    print("Test 2: Testing Tesseract fallback...")
    try:
        ocr_tess = OCREngine(engine='tesseract')
        print("✅ SUCCESS: Tesseract loaded successfully!")
        print("   Fallback mechanism is working!")
        tesseract_available = True
    except ImportError as e:
        print(f"❌ Tesseract not available: {e}")
        print("   Install with: brew install tesseract")
        tesseract_available = False
    except Exception as e:
        print(f"❌ Error loading Tesseract: {e}")
        tesseract_available = False
    
    print()
    print("=" * 60)
    print("Summary:")
    print("=" * 60)
    
    if paddle_available:
        print("🎉 PaddleOCR: AVAILABLE (best for handwriting)")
        print("   Your system is ready for superior OCR!")
    else:
        print("⚠️  PaddleOCR: NOT AVAILABLE")
        print("   Reason: Likely Python 3.14+ (requires <3.13)")
        print("   Solution: Install Python 3.12 or use Docker")
    
    print()
    
    if tesseract_available:
        print("✅ Tesseract: AVAILABLE (good for typed text)")
        print("   Fallback mechanism is working!")
    else:
        print("❌ Tesseract: NOT AVAILABLE")
        print("   Install: brew install tesseract")
    
    print()
    
    if paddle_available or tesseract_available:
        print("✅ System is functional!")
        if paddle_available:
            print("   Recommended: Use PaddleOCR for best results")
        else:
            print("   Currently using: Tesseract (acceptable quality)")
    else:
        print("❌ No OCR engine available - please install dependencies")
    
    print("=" * 60)
    
    return 0 if (paddle_available or tesseract_available) else 1

if __name__ == "__main__":
    sys.exit(main())
