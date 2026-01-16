#!/usr/bin/env python3
"""
HTR (Handwritten Text Recognition) Inference Script

This script uses the trained CRNN+CTC model to perform handwritten text recognition
on input images. The preprocessing matches the training pipeline exactly.
"""

import os
import sys
import json
import argparse
import numpy as np
import cv2
from pathlib import Path

import tensorflow as tf
from tensorflow import keras


class HTRModel:
    """Wrapper for HTR model inference"""
    
    def __init__(self, model_path, config_path=None):
        """
        Initialize the HTR model.
        
        Args:
            model_path: Path to the .keras model file
            config_path: Path to config.json (optional, will use defaults if not provided)
        """
        # Load model
        print(f"Loading model from {model_path}...")
        self.model = keras.models.load_model(model_path)
        print("✅ Model loaded successfully")
        
        # Load or use default config
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            self.img_height = config['image_dimensions']['height']
            self.img_width = config['image_dimensions']['width']
            self.characters = config['character_set']
        else:
            # Default config (matching training)
            self.img_height = 32
            self.img_width = 128
            self.characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?'-():;\"/ "
        
        # Create character mappings
        self.char_to_num = {char: idx + 1 for idx, char in enumerate(self.characters)}
        self.num_to_char = {idx + 1: char for idx, char in enumerate(self.characters)}
        self.num_to_char[0] = ''  # CTC blank
        
    def preprocess_image(self, image_path):
        """
        Preprocess image to match training preprocessing.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Preprocessed image as numpy array
        """
        # Read image in grayscale
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Resize maintaining aspect ratio
        h, w = img.shape
        new_width = min(int(w * (self.img_height / h)), self.img_width)
        img = cv2.resize(img, (new_width, self.img_height))
        
        # Right-pad with white if needed
        if new_width < self.img_width:
            img = np.pad(img, ((0, 0), (0, self.img_width - new_width)), 
                        constant_values=255)
        
        # Normalize: invert and scale to [0, 1]
        img = 1.0 - (img.astype(np.float32) / 255.0)
        
        # Add channel dimension
        img = np.expand_dims(img, axis=-1)
        
        return img
    
    def decode_prediction(self, pred):
        """
        Decode CTC prediction using greedy decoding.
        
        Args:
            pred: Model prediction output
            
        Returns:
            Decoded text string
        """
        # Get most likely character at each time step
        indices = np.argmax(pred, axis=1)
        
        # Remove duplicates and blanks (CTC decoding)
        chars = []
        prev_idx = -1
        for idx in indices:
            if idx != 0 and idx != prev_idx:  # Not blank and not duplicate
                if idx in self.num_to_char:
                    chars.append(self.num_to_char[idx])
            prev_idx = idx
        
        return ''.join(chars)
    
    def predict(self, image_path, verbose=True):
        """
        Predict text from image.
        
        Args:
            image_path: Path to input image
            verbose: Print processing info
            
        Returns:
            Predicted text string
        """
        if verbose:
            print(f"Processing: {image_path}")
        
        # Preprocess
        img = self.preprocess_image(image_path)
        
        # Add batch dimension
        img_batch = np.expand_dims(img, axis=0)
        
        # Predict
        pred = self.model.predict(img_batch, verbose=0)
        
        # Decode
        text = self.decode_prediction(pred[0])
        
        if verbose:
            print(f"Prediction: '{text}'")
        
        return text
    
    def predict_batch(self, image_paths, verbose=True):
        """
        Predict text from multiple images.
        
        Args:
            image_paths: List of image paths
            verbose: Print processing info
            
        Returns:
            List of predicted text strings
        """
        results = []
        
        if verbose:
            print(f"Processing {len(image_paths)} images...")
        
        for img_path in image_paths:
            text = self.predict(img_path, verbose=False)
            results.append(text)
            if verbose:
                print(f"  {os.path.basename(img_path)}: '{text}'")
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description='HTR Inference - Handwritten Text Recognition'
    )
    parser.add_argument(
        '--image', '-i',
        type=str,
        help='Path to input image'
    )
    parser.add_argument(
        '--images-dir', '-d',
        type=str,
        help='Directory containing images to process'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='models/htr_model.keras',
        help='Path to model file (default: models/htr_model.keras)'
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='models/config.json',
        help='Path to config file (default: models/config.json)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output file to save predictions (optional)'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.image and not args.images_dir:
        parser.error("Must provide either --image or --images-dir")
    
    if not os.path.exists(args.model):
        print(f"❌ Model not found: {args.model}")
        sys.exit(1)
    
    # Load model
    htr = HTRModel(args.model, args.config)
    
    # Process image(s)
    predictions = []
    
    if args.image:
        # Single image
        if not os.path.exists(args.image):
            print(f"❌ Image not found: {args.image}")
            sys.exit(1)
        
        text = htr.predict(args.image)
        predictions.append((args.image, text))
        
    elif args.images_dir:
        # Directory of images
        if not os.path.isdir(args.images_dir):
            print(f"❌ Directory not found: {args.images_dir}")
            sys.exit(1)
        
        # Find all image files
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(Path(args.images_dir).glob(f'*{ext}'))
            image_paths.extend(Path(args.images_dir).glob(f'*{ext.upper()}'))
        
        if not image_paths:
            print(f"❌ No images found in {args.images_dir}")
            sys.exit(1)
        
        print(f"Found {len(image_paths)} images")
        
        # Process all images
        texts = htr.predict_batch([str(p) for p in image_paths])
        predictions = [(str(p), t) for p, t in zip(image_paths, texts)]
    
    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            for img_path, text in predictions:
                f.write(f"{img_path}\t{text}\n")
        print(f"\n✅ Predictions saved to {args.output}")
    
    print("\n✅ Done!")


if __name__ == '__main__':
    main()
