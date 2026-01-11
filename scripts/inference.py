#!/usr/bin/env python3
"""
Inference script for OCR and QA segmentation.

Usage:
    python inference.py --images exam1.jpg exam2.jpg --model models/crf_model.pkl --output results.json
"""

import argparse
import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from preprocessing import ImagePreprocessor
from ocr_engine import OCREngine
from feature_extraction import FeatureExtractor
from crf_model import CRFModel
from postprocessing import QAPairExtractor
from utils import visualize_predictions


def process_document(image_paths, model_path, ocr_engine='paddleocr', 
                     visualize=False, output_vis=None):
    """
    Process a document and extract QA pairs.
    
    Args:
        image_paths: List of image file paths
        model_path: Path to trained CRF model
        ocr_engine: OCR engine to use
        visualize: Whether to visualize predictions
        output_vis: Output path for visualization
        
    Returns:
        List of QA pair dictionaries
    """
    print("="*70)
    print("OCR & QUESTION-ANSWER SEGMENTATION")
    print("="*70)
    
    # Step 1: Preprocessing
    print("\n[1/5] Preprocessing images...")
    preprocessor = ImagePreprocessor(target_width=1200, enable_deskew=True)
    processed_image = preprocessor.process(image_paths)
    print(f"  ✓ Stitched {len(image_paths)} image(s)")
    
    # Step 2: OCR
    print("\n[2/5] Running OCR...")
    ocr = OCREngine(engine=ocr_engine, lang='en')
    lines = ocr.extract_lines(processed_image)
    print(f"  ✓ Extracted {len(lines)} text lines")
    
    if len(lines) == 0:
        print("\n⚠ No text detected in image(s)")
        return []
    
    # Step 3: Feature extraction
    print("\n[3/5] Extracting features...")
    img_width, img_height = ocr.get_image_dimensions(processed_image)
    feature_extractor = FeatureExtractor(img_width, img_height)
    features = feature_extractor.extract_features(lines)
    crf_features = feature_extractor.features_to_crf_format(features)
    print(f"  ✓ Extracted {len(features[0])} features per line")
    
    # Step 4: CRF prediction
    print("\n[4/5] Running CRF model...")
    model = CRFModel()
    model.load(model_path)
    tags = model.predict_single(crf_features)
    print(f"  ✓ Predicted sequence tags")
    
    # Print tag distribution
    tag_counts = {}
    for tag in tags:
        tag_counts[tag] = tag_counts.get(tag, 0) + 1
    print(f"  Tag distribution: {tag_counts}")
    
    # Step 5: Extract QA pairs
    print("\n[5/5] Extracting QA pairs...")
    extractor = QAPairExtractor(min_confidence=0.3)
    pairs = extractor.extract_pairs(lines, tags, include_orphans=False)
    print(f"  ✓ Extracted {len(pairs)} question-answer pairs")
    
    # Visualize if requested
    if visualize and len(image_paths) == 1:
        print("\n[Visualization]")
        visualize_predictions(image_paths[0], lines, tags, output_vis)
    
    return pairs


def main():
    parser = argparse.ArgumentParser(
        description='Process exam images and extract question-answer pairs'
    )
    parser.add_argument('--images', nargs='+', required=True,
                       help='Exam image file path(s)')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained CRF model')
    parser.add_argument('--output', type=str,
                       help='Output JSON file path')
    parser.add_argument('--ocr-engine', choices=['paddleocr', 'tesseract'],
                       default='paddleocr',
                       help='OCR engine to use (default: paddleocr)')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize predictions (single image only)')
    parser.add_argument('--output-vis', type=str,
                       help='Output path for visualization image')
    parser.add_argument('--print-text', action='store_true',
                       help='Print formatted QA pairs to console')
    
    args = parser.parse_args()
    
    # Verify image files exist
    for img_path in args.images:
        if not Path(img_path).exists():
            print(f"Error: Image not found: {img_path}")
            return 1
    
    # Verify model exists
    if not Path(args.model).exists():
        print(f"Error: Model not found: {args.model}")
        return 1
    
    # Process document
    try:
        pairs = process_document(
            args.images,
            args.model,
            ocr_engine=args.ocr_engine,
            visualize=args.visualize,
            output_vis=args.output_vis
        )
        
        # Convert to dictionaries
        extractor = QAPairExtractor()
        pairs_dict = extractor.pairs_to_dict(pairs)
        
        # Print if requested
        if args.print_text:
            print("\n" + extractor.pairs_to_formatted_text(pairs))
        
        # Save to JSON if output specified
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(pairs_dict, f, indent=2, ensure_ascii=False)
            
            print(f"\n✓ Results saved to: {output_path}")
        else:
            # Print JSON to console
            print("\n" + "="*70)
            print("RESULTS (JSON)")
            print("="*70)
            print(json.dumps(pairs_dict, indent=2, ensure_ascii=False))
        
        print("\n✓ Processing complete!")
        
    except Exception as e:
        print(f"\n✗ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
