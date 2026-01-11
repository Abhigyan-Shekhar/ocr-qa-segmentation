"""
Image preprocessing module for OCR and QA segmentation.

Handles multi-page stitching, deskewing, binarization, and noise reduction.
"""

import cv2
import numpy as np
from typing import List, Union, Tuple
from pathlib import Path


class ImagePreprocessor:
    """Preprocessor for handwritten exam images."""
    
    def __init__(self, target_width: int = 1200, enable_deskew: bool = True):
        """
        Initialize the preprocessor.
        
        Args:
            target_width: Target width for resizing images
            enable_deskew: Whether to apply deskewing
        """
        self.target_width = target_width
        self.enable_deskew = enable_deskew
    
    def load_images(self, image_paths: List[Union[str, Path]]) -> List[np.ndarray]:
        """
        Load images from file paths.
        
        Args:
            image_paths: List of paths to image files
            
        Returns:
            List of loaded images as numpy arrays
        """
        images = []
        for path in image_paths:
            img = cv2.imread(str(path))
            if img is None:
                raise ValueError(f"Failed to load image: {path}")
            images.append(img)
        return images
    
    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Resize image to target width while maintaining aspect ratio.
        
        Args:
            image: Input image
            
        Returns:
            Resized image
        """
        h, w = image.shape[:2]
        if w != self.target_width:
            ratio = self.target_width / w
            new_h = int(h * ratio)
            image = cv2.resize(image, (self.target_width, new_h), 
                             interpolation=cv2.INTER_CUBIC)
        return image
    
    def deskew(self, image: np.ndarray) -> np.ndarray:
        """
        Deskew image using projection profile method.
        
        Args:
            image: Input grayscale image
            
        Returns:
            Deskewed image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply binary threshold
        _, binary = cv2.threshold(gray, 0, 255, 
                                 cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find coordinates of all white pixels
        coords = np.column_stack(np.where(binary > 0))
        
        if len(coords) == 0:
            return image
        
        # Calculate minimum area bounding box
        angle = cv2.minAreaRect(coords)[-1]
        
        # Adjust angle
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        
        # Rotate image to deskew
        if abs(angle) > 0.5:  # Only rotate if angle is significant
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, M, (w, h),
                                  flags=cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_REPLICATE)
        
        return image
    
    def enhance_handwriting(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image specifically for handwriting OCR.
        
        Uses CLAHE for contrast + bilateral filtering to preserve edges.
        
        Args:
            image: Input image
            
        Returns:
            Enhanced image optimized for handwriting
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Bilateral filtering - removes noise while preserving edges
        filtered = cv2.bilateralFilter(enhanced, d=9, sigmaColor=75, sigmaSpace=75)
        
        # Sharpen slightly to make strokes clearer
        kernel = np.array([[-1, -1, -1], 
                          [-1,  9, -1],
                          [-1, -1, -1]])
        sharpened = cv2.filter2D(filtered, -1, kernel)
        
        return sharpened
    
    def binarize(self, image: np.ndarray, for_handwriting: bool = True) -> np.ndarray:
        """
        Apply adaptive thresholding optimized for handwriting.
        
        Args:
            image: Input image
            for_handwriting: Use handwriting-optimized parameters
            
        Returns:
            Binarized image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        if for_handwriting:
            # Larger block size for handwriting (captures more context)
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 
                blockSize=15,  # Increased from 11 for better handwriting context
                C=10           # Increased from 2 for better contrast
            )
        else:
            # Original parameters for typed text
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
        
        return binary
    
    def denoise(self, image: np.ndarray, preserve_detail: bool = True) -> np.ndarray:
        """
        Remove noise while optionally preserving handwriting details.
        
        Args:
            image: Input image
            preserve_detail: If True, use gentle denoising for handwriting
            
        Returns:
            Denoised image
        """
        if preserve_detail:
            # Gentle denoising for handwriting - preserves fine strokes
            denoised = cv2.fastNlMeansDenoising(image, None, h=10, 
                                               templateWindowSize=7, 
                                               searchWindowSize=21)
            return denoised
        else:
            # Original aggressive denoising for typed text
            denoised = cv2.medianBlur(image, 3)
            kernel = np.ones((2, 2), np.uint8)
            denoised = cv2.morphologyEx(denoised, cv2.MORPH_OPEN, kernel)
            return denoised
    
    def stitch_images(self, images: List[np.ndarray]) -> np.ndarray:
        """
        Vertically stitch multiple images into one continuous scroll.
        
        Args:
            images: List of images to stitch
            
        Returns:
            Stitched image
        """
        if len(images) == 0:
            raise ValueError("No images to stitch")
        
        if len(images) == 1:
            return images[0]
        
        # Ensure all images have the same width
        target_width = images[0].shape[1]
        resized_images = []
        
        for img in images:
            if img.shape[1] != target_width:
                h, w = img.shape[:2]
                ratio = target_width / w
                new_h = int(h * ratio)
                img = cv2.resize(img, (target_width, new_h))
            resized_images.append(img)
        
        # Vertically concatenate
        stitched = np.vstack(resized_images)
        
        return stitched
    
    def process(self, image_paths: List[Union[str, Path]], 
                handwriting_mode: bool = True,
                return_intermediate: bool = False) -> Union[np.ndarray, Tuple]:
        """
        Complete preprocessing pipeline.
        
        Args:
            image_paths: List of image file paths
            handwriting_mode: Optimize for handwriting recognition (default True)
            return_intermediate: Whether to return intermediate results
            
        Returns:
            Preprocessed image, or tuple of (final, intermediate_dict) if return_intermediate=True
        """
        # Load images
        images = self.load_images(image_paths)
        
        intermediate = {} if return_intermediate else None
        
        # Resize each image
        resized = [self.resize_image(img) for img in images]
        if return_intermediate:
            intermediate['resized'] = resized.copy()
        
        # Deskew if enabled
        if self.enable_deskew:
            deskewed = [self.deskew(img) for img in resized]
            if return_intermediate:
                intermediate['deskewed'] = deskewed.copy()
        else:
            deskewed = resized
        
        # Stitch images
        stitched = self.stitch_images(deskewed)
        if return_intermediate:
            intermediate['stitched'] = stitched.copy()
        
        # Convert to grayscale for further processing
        if len(stitched.shape) == 3:
            gray = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
        else:
            gray = stitched
        
        # Apply handwriting enhancement if enabled
        if handwriting_mode:
            enhanced = self.enhance_handwriting(gray)
            if return_intermediate:
                intermediate['enhanced'] = enhanced.copy()
            
            # Gentle denoising to preserve handwriting details
            denoised = self.denoise(enhanced, preserve_detail=True)
        else:
            # Standard aggressive denoising for typed text
            denoised = self.denoise(gray, preserve_detail=False)
        
        if return_intermediate:
            intermediate['denoised'] = denoised.copy()
        
        # Return enhanced grayscale for better OCR compatibility
        # (OCR engines handle binarization internally with better algorithms)
        final = denoised
        
        if return_intermediate:
            return final, intermediate
        return final
