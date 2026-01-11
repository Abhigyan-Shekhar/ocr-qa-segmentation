#!/usr/bin/env python3
"""
Training script for CRF model.

Usage:
    python train.py --data data/training_data.json --output models/crf_model.pkl
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from crf_model import CRFModel
from utils import load_training_data, create_synthetic_training_data
import json


def split_data(X, y, val_split=0.2):
    """Split data into train and validation sets."""
    n_samples = len(X)
    n_val = int(n_samples * val_split)
    
    indices = list(range(n_samples))
    import random
    random.shuffle(indices)
    
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]
    
    X_train = [X[i] for i in train_indices]
    y_train = [y[i] for i in train_indices]
    X_val = [X[i] for i in val_indices]
    y_val = [y[i] for i in val_indices]
    
    return X_train, y_train, X_val, y_val


def main():
    parser = argparse.ArgumentParser(description='Train CRF model for QA segmentation')
    parser.add_argument('--data', type=str, help='Path to training data JSON')
    parser.add_argument('--output', type=str, required=True,
                       help='Output path for trained model')
    parser.add_argument('--val-split', type=float, default=0.2,
                       help='Validation split ratio (default: 0.2)')
    parser.add_argument('--c1', type=float, default=0.1,
                       help='L1 regularization coefficient')
    parser.add_argument('--c2', type=float, default=0.1,
                       help='L2 regularization coefficient')
    parser.add_argument('--max-iter', type=int, default=100,
                       help='Maximum training iterations')
    parser.add_argument('--use-synthetic', action='store_true',
                       help='Use synthetic data for testing')
    
    args = parser.parse_args()
    
    # Load data
    if args.use_synthetic:
        print("Creating synthetic training data...")
        X, y = create_synthetic_training_data(n_samples=50)
    elif args.data:
        print(f"Loading training data from: {args.data}")
        X, y = load_training_data(args.data)
    else:
        print("Error: Must specify --data or --use-synthetic")
        return 1
    
    print(f"Loaded {len(X)} training samples")
    
    # Split into train/val
    if args.val_split > 0:
        X_train, y_train, X_val, y_val = split_data(X, y, args.val_split)
        print(f"Train: {len(X_train)}, Val: {len(X_val)}")
    else:
        X_train, y_train = X, y
        X_val, y_val = None, None
        print(f"Train: {len(X_train)} (no validation split)")
    
    # Initialize model
    print("\nInitializing CRF model...")
    model = CRFModel(
        c1=args.c1,
        c2=args.c2,
        max_iterations=args.max_iter
    )
    
    # Train
    print("\nTraining...")
    results = model.train(X_train, y_train, X_val, y_val)
    
    # Print feature weights
    print("\nAnalyzing feature importance...")
    model.print_feature_weights(top_n=10)
    
    # Save model
    print(f"\nSaving model to: {args.output}")
    model.save(args.output)
    
    # Save training metadata
    metadata_path = Path(args.output).with_suffix('.json')
    metadata = {
        'train_samples': results.get('train_samples'),
        'val_samples': results.get('val_samples'),
        'train_f1': results.get('train_f1'),
        'val_f1': results.get('val_f1'),
        'c1': args.c1,
        'c2': args.c2,
        'max_iterations': args.max_iter
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Training metadata saved to: {metadata_path}")
    print("\nTraining complete!")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
