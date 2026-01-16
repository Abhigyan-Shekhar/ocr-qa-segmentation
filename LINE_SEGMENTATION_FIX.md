# Line Segmentation Fix - Quick Patch for Colab

If you're already in Colab, copy-paste this improved function to replace the `segment_lines` function in Cell 4:

```python
from scipy.ndimage import gaussian_filter1d

def segment_lines(img: np.ndarray, remove_lines: bool = True) -> List[Tuple[np.ndarray, Tuple]]:
    """
    Segment image into individual text lines (IMPROVED VERSION)
    Returns: [(line_image, (x, y, w, h)), ...]
    """
    # Preprocess
    if remove_lines:
        img = remove_ruled_lines(img)
    
    # Denoise
    denoised = cv2.fastNlMeansDenoising(img, None, 10, 7, 21)
    
    # Binarize with adaptive threshold for better handwriting
    binary = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 15, 10)
    
    # Dilate slightly to connect broken characters
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=1)
    
    # Horizontal projection
    h_projection = np.sum(binary, axis=1)
    
    # IMPROVED: Use median instead of max, lower threshold
    threshold = np.median(h_projection[h_projection > 0]) * 0.3
    
    # Smooth projection to reduce noise
    h_projection_smooth = gaussian_filter1d(h_projection, sigma=2)
    
    # Find line boundaries with minimum gap
    line_regions = []
    in_line = False
    start_y = 0
    min_gap = 15  # Minimum pixels between lines
    last_end = 0
    
    for y, val in enumerate(h_projection_smooth):
        if val > threshold and not in_line:
            # Start new line only if minimum gap from last line
            if y - last_end > min_gap:
                start_y = y
                in_line = True
        elif val <= threshold and in_line:
            # End line if significant height
            if y - start_y > 15:  # Minimum line height
                line_regions.append((start_y, y))
                last_end = y
            in_line = False
    
    # If still in line at end
    if in_line and len(h_projection_smooth) - start_y > 15:
        line_regions.append((start_y, len(h_projection_smooth)))
    
    # Extract line images with bounding boxes
    lines = []
    for start_y, end_y in line_regions:
        # Add margin
        start_y = max(0, start_y - 5)
        end_y = min(img.shape[0], end_y + 5)
        
        # Vertical projection to find x bounds
        line_strip = binary[start_y:end_y, :]
        v_projection = np.sum(line_strip, axis=0)
        
        # Find first and last non-zero columns
        non_zero = np.where(v_projection > 0)[0]
        if len(non_zero) > 0:
            start_x = max(0, non_zero[0] - 10)
            end_x = min(img.shape[1], non_zero[-1] + 10)
            
            # Extract line (from denoised grayscale)
            line_img = denoised[start_y:end_y, start_x:end_x]
            bbox = (start_x, start_y, end_x - start_x, end_y - start_y)
            lines.append((line_img, bbox))
    
    return lines
```

**Key improvements:**
1. Adaptive thresholding (better for handwriting)
2. Median-based threshold (more robust)
3. Gaussian smoothing of projection
4. Minimum gap enforcement (prevents merging)
5. Better margin handling

**Then re-run from Cell 9 onwards!**
