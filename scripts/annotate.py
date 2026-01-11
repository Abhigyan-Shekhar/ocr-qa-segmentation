#!/usr/bin/env python3
"""
Simple CLI tool for annotating exam images to create training data.

Usage:
    python annotate.py --image exam.jpg --output annotated_data.json
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


VALID_TAGS = ['B-Q', 'I-Q', 'B-A', 'I-A', 'O']


def print_instructions():
    """Print annotation instructions."""
    print("\n" + "="*70)
    print("ANNOTATION INSTRUCTIONS")
    print("="*70)
    print("\nTag each line with one of:")
    print("  B-Q  : Begin Question")
    print("  I-Q  : Inside Question (continuation)")
    print("  B-A  : Begin Answer")
    print("  I-A  : Inside Answer (continuation)")
    print("  O    : Other (margins, headers, etc.)")
    print("\nCommands:")
    print("  <tag> : Assign tag to current line")
    print("  s     : Skip this line (tag as O)")
    print("  u     : Undo last annotation")
    print("  q     : Quit and save")
    print("  h     : Show help")
    print("="*70)


def annotate_lines(lines, image_width, image_height):
    """
    Interactively annotate OCR lines.
    
    Args:
        lines: List of OCR lines
        image_width: Image width
        image_height: Image height
        
    Returns:
        (features, labels) tuple
    """
    print_instructions()
    
    # Extract features
    feature_extractor = FeatureExtractor(image_width, image_height)
    
    labels = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Show line info
        print(f"\n{'─'*70}")
        print(f"Line {i+1}/{len(lines)}")
        print(f"{'─'*70}")
        print(f"Text: {line.text}")
        print(f"BBox: {line.bbox}")
        print(f"Conf: {line.confidence:.2%}")
        
        # Get user input
        while True:
            tag = input("\nTag (B-Q/I-Q/B-A/I-A/O/s/u/q/h): ").strip().upper()
            
            if tag == 'H':
                print_instructions()
                continue
            elif tag == 'Q':
                print("Saving and quitting...")
                # Pad remaining lines with O
                while len(labels) < len(lines):
                    labels.append('O')
                return labels
            elif tag == 'U':
                if len(labels) > 0:
                    removed = labels.pop()
                    i -= 1
                    print(f"Undid: {removed}")
                    break
                else:
                    print("Nothing to undo")
                    continue
            elif tag == 'S':
                tag = 'O'
            
            if tag in VALID_TAGS:
                labels.append(tag)
                print(f"✓ Tagged as: {tag}")
                i += 1
                break
            else:
                print(f"Invalid tag: {tag}. Use one of: {', '.join(VALID_TAGS)}")
    
    return labels


def main():
    parser = argparse.ArgumentParser(
        description='Annotate exam images to create training data'
    )
    parser.add_argument('--image', type=str, required=True,
                       help='Exam image file path')
    parser.add_argument('--output', type=str, required=True,
                       help='Output JSON file path')
    parser.add_argument('--append', action='store_true',
                       help='Append to existing annotation file')
    parser.add_argument('--ocr-engine', choices=['paddleocr', 'tesseract'],
                       default='paddleocr',
                       help='OCR engine to use')
    
    args = parser.parse_args()
    
    # Check image exists
    if not Path(args.image).exists():
        print(f"Error: Image not found: {args.image}")
        return 1
    
    # Load existing annotations if appending
    existing_data = []
    if args.append and Path(args.output).exists():
        with open(args.output, 'r') as f:
            existing_data = json.load(f)
        print(f"Loaded {len(existing_data)} existing annotations")
    
    # Preprocess image
    print("\nPreprocessing image...")
    preprocessor = ImagePreprocessor(target_width=1200, enable_deskew=False)
    processed = preprocessor.process([args.image])
    
    # Run OCR
    print("Running OCR...")
    ocr = OCREngine(engine=args.ocr_engine)
    lines = ocr.extract_lines(processed)
    
    if len(lines) == 0:
        print("No text detected in image")
        return 1
    
    print(f"Detected {len(lines)} text lines")
    
    # Get image dimensions
    img_width, img_height = ocr.get_image_dimensions(processed)
    
    # Extract features
    feature_extractor = FeatureExtractor(img_width, img_height)
    
    # Annotate
    print("\nStarting annotation...")
    labels = annotate_lines(lines, img_width, img_height)
    
    if len(labels) != len(lines):
        print(f"\nWarning: Only annotated {len(labels)}/{len(lines)} lines")
        # Pad with O tags
        while len(labels) < len(lines):
            labels.append('O')
    
    # Extract features with the annotated labels
    features = feature_extractor.extract_features(lines, prev_tags=labels)
    crf_features = feature_extractor.features_to_crf_format(features)
    
    # Create annotation entry
    annotation = {
        'image_path': str(args.image),
        'features': crf_features,
        'labels': labels,
        'num_lines': len(lines)
    }
    
    # Add to existing data
    existing_data.append(annotation)
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(existing_data, f, indent=2)
    
    print(f"\n✓ Annotation saved to: {output_path}")
    print(f"Total annotations: {len(existing_data)}")
    
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nAnnotation cancelled")
        sys.exit(1)
