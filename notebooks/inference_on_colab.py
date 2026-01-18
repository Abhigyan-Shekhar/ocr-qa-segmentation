#!/usr/bin/env python3
"""
TrOCR Inference Script for Google Colab
Loads fine-tuned model and runs inference on new images
"""

import os
import torch
from PIL import Image
from typing import Union, List
from pathlib import Path

# ============================================================================
# SETUP
# ============================================================================

def install_dependencies():
    """Install required packages"""
    import subprocess
    subprocess.run([
        "pip", "install", "-q",
        "transformers>=4.30.0",
        "torch>=2.0.0",
        "Pillow>=9.0.0"
    ], check=True)
    print("✓ Dependencies installed")

# Uncomment if not already installed
# install_dependencies()

from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# ============================================================================
# MODEL LOADING
# ============================================================================

class TrOCRInference:
    """
    TrOCR Inference Engine
    Loads fine-tuned model and provides prediction methods
    """
    
    def __init__(
        self,
        model_path: str = "/content/tr_ocr_finetuned",
        device: str = None
    ):
        """
        Initialize the inference engine
        
        Args:
            model_path: Path to the fine-tuned model directory
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.model_path = model_path
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading model from: {model_path}")
        print(f"Device: {self.device}")
        
        # Load processor and model
        self.processor = TrOCRProcessor.from_pretrained(model_path)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        print("✓ Model loaded successfully")
    
    def preprocess_image(
        self,
        image: Union[str, Path, Image.Image],
        target_size: tuple = (384, 384)
    ) -> Image.Image:
        """
        Preprocess image for inference
        
        Args:
            image: Path to image or PIL Image
            target_size: Target size for resizing
            
        Returns:
            Preprocessed PIL Image
        """
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        
        # Convert to RGB
        image = image.convert('RGB')
        
        # Resize
        image = image.resize(target_size, Image.Resampling.LANCZOS)
        
        return image
    
    def predict(
        self,
        image: Union[str, Path, Image.Image],
        num_beams: int = 4,
        max_length: int = 128
    ) -> str:
        """
        Predict text from a single image
        
        Args:
            image: Path to image or PIL Image
            num_beams: Number of beams for beam search
            max_length: Maximum output length
            
        Returns:
            Predicted text string
        """
        # Preprocess
        image = self.preprocess_image(image)
        
        # Encode
        pixel_values = self.processor(
            images=image,
            return_tensors="pt"
        ).pixel_values.to(self.device)
        
        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                pixel_values,
                num_beams=num_beams,
                max_length=max_length,
                early_stopping=True
            )
        
        # Decode
        predicted_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]
        
        return predicted_text
    
    def predict_batch(
        self,
        images: List[Union[str, Path, Image.Image]],
        batch_size: int = 8,
        num_beams: int = 4,
        max_length: int = 128
    ) -> List[str]:
        """
        Predict text from multiple images
        
        Args:
            images: List of image paths or PIL Images
            batch_size: Batch size for inference
            num_beams: Number of beams for beam search
            max_length: Maximum output length
            
        Returns:
            List of predicted text strings
        """
        results = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            
            # Preprocess batch
            processed_images = [self.preprocess_image(img) for img in batch]
            
            # Encode
            pixel_values = self.processor(
                images=processed_images,
                return_tensors="pt"
            ).pixel_values.to(self.device)
            
            # Generate
            with torch.no_grad():
                generated_ids = self.model.generate(
                    pixel_values,
                    num_beams=num_beams,
                    max_length=max_length,
                    early_stopping=True
                )
            
            # Decode
            batch_texts = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )
            
            results.extend(batch_texts)
        
        return results


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

# Global model instance
_model = None

def load_model(model_path: str = "/content/tr_ocr_finetuned") -> TrOCRInference:
    """
    Load the model (caches globally)
    
    Args:
        model_path: Path to the fine-tuned model
        
    Returns:
        TrOCRInference instance
    """
    global _model
    if _model is None:
        _model = TrOCRInference(model_path)
    return _model


def predict_image(image_path: str, model_path: str = "/content/tr_ocr_finetuned") -> str:
    """
    Main function to predict text from an image
    
    Args:
        image_path: Path to the image file
        model_path: Path to the fine-tuned model directory
        
    Returns:
        Recognized text/latex string
    """
    model = load_model(model_path)
    result = model.predict(image_path)
    
    print(f"\n{'='*50}")
    print(f"Image: {image_path}")
    print(f"{'='*50}")
    print(f"Recognized Text: {result}")
    print(f"{'='*50}\n")
    
    return result


# ============================================================================
# COLAB-SPECIFIC UTILITIES
# ============================================================================

def upload_and_predict():
    """
    Interactive function for Colab: upload image and predict
    """
    from google.colab import files
    
    print("Upload an image to recognize:")
    uploaded = files.upload()
    
    for filename in uploaded.keys():
        print(f"\nProcessing: {filename}")
        result = predict_image(filename)
        return result


def predict_from_url(image_url: str) -> str:
    """
    Predict text from an image URL
    
    Args:
        image_url: URL of the image
        
    Returns:
        Recognized text
    """
    import urllib.request
    from io import BytesIO
    
    # Download image
    with urllib.request.urlopen(image_url) as response:
        image_data = response.read()
    
    image = Image.open(BytesIO(image_data))
    
    model = load_model()
    result = model.predict(image)
    
    print(f"\n{'='*50}")
    print(f"URL: {image_url[:50]}...")
    print(f"{'='*50}")
    print(f"Recognized Text: {result}")
    print(f"{'='*50}\n")
    
    return result


def segment_and_predict(image_path: str) -> List[dict]:
    """
    Segment an image into lines and predict each line
    Uses horizontal projection for line detection
    
    Args:
        image_path: Path to the full page image
        
    Returns:
        List of dicts with line info and predictions
    """
    import numpy as np
    from scipy.ndimage import gaussian_filter1d
    
    # Load image
    image = Image.open(image_path).convert('L')  # Grayscale
    img_array = np.array(image)
    
    # Binarize
    threshold = np.median(img_array)
    binary = (img_array < threshold).astype(np.uint8) * 255
    
    # Horizontal projection
    h_projection = np.sum(binary, axis=1)
    h_projection = gaussian_filter1d(h_projection, sigma=3)
    
    # Find line boundaries
    threshold = np.max(h_projection) * 0.1
    is_text = h_projection > threshold
    
    lines = []
    in_line = False
    start = 0
    
    for i, has_text in enumerate(is_text):
        if has_text and not in_line:
            in_line = True
            start = i
        elif not has_text and in_line:
            in_line = False
            if i - start > 10:  # Minimum height
                lines.append((start, i))
    
    if in_line:
        lines.append((start, len(is_text)))
    
    # Predict each line
    model = load_model()
    image_rgb = Image.open(image_path).convert('RGB')
    width = image_rgb.width
    
    results = []
    for idx, (y1, y2) in enumerate(lines):
        # Crop line with some padding
        padding = 5
        y1 = max(0, y1 - padding)
        y2 = min(image_rgb.height, y2 + padding)
        
        line_crop = image_rgb.crop((0, y1, width, y2))
        
        # Predict
        text = model.predict(line_crop)
        
        results.append({
            "line_number": idx + 1,
            "bbox": (0, y1, width, y2),
            "text": text
        })
        
        print(f"Line {idx + 1}: {text}")
    
    return results


# ============================================================================
# MAIN / DEMO
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TrOCR Inference Demo")
    print("=" * 60)
    
    # Check if model exists
    model_path = "/content/tr_ocr_finetuned"
    
    if not os.path.exists(model_path):
        print(f"\n⚠️  Model not found at: {model_path}")
        print("Please upload your fine-tuned model to this location.")
        print("\nExpected structure:")
        print("  /content/tr_ocr_finetuned/")
        print("  ├── config.json")
        print("  ├── pytorch_model.bin (or model.safetensors)")
        print("  ├── tokenizer_config.json")
        print("  ├── vocab.json")
        print("  └── ...")
    else:
        print("\n✓ Model found!")
        print("\nUsage examples:")
        print("  # Single image prediction")
        print('  result = predict_image("/path/to/image.png")')
        print("")
        print("  # Upload and predict (Colab)")
        print("  result = upload_and_predict()")
        print("")
        print("  # Predict from URL")
        print('  result = predict_from_url("https://example.com/image.png")')
        print("")
        print("  # Segment page and predict lines")
        print('  results = segment_and_predict("/path/to/page.png")')
        
        # Quick test if model exists
        print("\n" + "-" * 40)
        print("Running quick model test...")
        
        try:
            model = load_model(model_path)
            
            # Create a test image
            test_img = Image.new('RGB', (384, 64), 'white')
            from PIL import ImageDraw
            draw = ImageDraw.Draw(test_img)
            draw.text((10, 20), "Hello World", fill='black')
            test_img.save("/tmp/test_ocr.png")
            
            result = model.predict("/tmp/test_ocr.png")
            print(f"Test prediction: '{result}'")
            print("✓ Model is working!")
            
        except Exception as e:
            print(f"⚠️  Test failed: {e}")
