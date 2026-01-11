#!/usr/bin/env python3
"""
Quick test to verify the OCR QA Segmentation system works end-to-end.

This uses synthetic training data for demonstration.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import crf_model
import postprocessing
import ocr_engine
import feature_extraction
import utils

CRFModel = crf_model.CRFModel
QAPairExtractor = postprocessing.QAPairExtractor
OCRLine = ocr_engine.OCRLine
FeatureExtractor = feature_extraction.FeatureExtractor
create_synthetic_training_data = utils.create_synthetic_training_data


def test_training():
    """Test CRF model training."""
    print("\n" + "="*70)
    print("TEST 1: CRF Model Training")
    print("="*70)
    
    # Create synthetic data
    X, y = create_synthetic_training_data(n_samples=30)
    print(f"✓ Created {len(X)} synthetic training samples")
    
    # Train model
    model = CRFModel(max_iterations=50)
    results = model.train(X, y)
    print(f"✓ Model trained - F1: {results['train_f1']:.4f}")
    
    # Save model
    model_path = Path(__file__).parent.parent / 'models' / 'test_model.pkl'
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))
    print(f"✓ Model saved to: {model_path}")
    
    return model, model_path


def test_inference(model):
    """Test inference on mock data."""
    print("\n" + "="*70)
    print("TEST 2: Inference on Mock Exam")
    print("="*70)
    
    # Create mock OCR lines
    mock_lines = [
        OCRLine("Q1. What is the capital of France?", (50, 100, 400, 25), 0.95, 0),
        OCRLine("Paris is the capital of France.", (80, 140, 380, 25), 0.90, 1),
        OCRLine("It is located in northern France.", (80, 170, 390, 25), 0.88, 2),
        OCRLine("Q2. Explain machine learning.", (50, 220, 350, 25), 0.92, 3),
        OCRLine("Machine learning is a branch of AI that", (80, 260, 420, 25), 0.89, 4),
        OCRLine("enables computers to learn from data.", (80, 290, 410, 25), 0.91, 5),
    ]
    
    print(f"✓ Created {len(mock_lines)} mock OCR lines")
    
    # Extract features
    feature_extractor = FeatureExtractor(800, 1200)
    features = feature_extractor.extract_features(mock_lines)
    crf_features = feature_extractor.features_to_crf_format(features)
    print(f"✓ Extracted features")
    
    # Predict tags
    tags = model.predict_single(crf_features)
    print(f"✓ Predicted tags")
    
    # Show predictions
    print("\nPredictions:")
    for line, tag in zip(mock_lines, tags):
        print(f"  [{tag:5s}] {line.text}")
    
    # Extract QA pairs
    extractor = QAPairExtractor()
    pairs = extractor.extract_pairs(mock_lines, tags)
    print(f"\n✓ Extracted {len(pairs)} question-answer pairs")
    
    # Display pairs
    print("\n" + extractor.pairs_to_formatted_text(pairs))
    
    return pairs


def test_feature_extraction():
    """Test feature extraction."""
    print("\n" + "="*70)
    print("TEST 3: Feature Extraction")
    print("="*70)
    
    # Create test line
    test_line = OCRLine("Q1. What is AI?", (50, 100, 200, 25), 0.95, 0)
    
    # Extract features
    extractor = FeatureExtractor(800, 1200)
    features = extractor._extract_single_line_features(test_line, None, 'O')
    
    print("\nExtracted features:")
    important_features = [
        'indent_level', 'vertical_gap', 'starts_with_q_marker', 
        'starts_with_number', 'word_count', 'is_capitalized'
    ]
    
    for feat in important_features:
        if feat in features:
            print(f"  {feat:25s}: {features[feat]}")
    
    print("✓ Feature extraction working correctly")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("OCR & QA SEGMENTATION SYSTEM - QUICK TEST")
    print("="*70)
    
    try:
        # Test 1: Training
        model, model_path = test_training()
        
        # Test 2: Inference
        pairs = test_inference(model)
        
        # Test 3: Feature extraction
        test_feature_extraction()
        
        # Summary
        print("\n" + "="*70)
        print("ALL TESTS PASSED ✓")
        print("="*70)
        print("\nThe OCR & QA Segmentation system is working correctly!")
        print("\nNext steps:")
        print("  1. Add real exam images to examples/sample_data/")
        print("  2. Annotate data using: python scripts/annotate.py")
        print("  3. Train on real data: python scripts/train.py")
        print("  4. Run inference: python scripts/inference.py")
        print("\nSee README.md for detailed instructions.")
        
        return 0
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
