# TrOCR Refinement Options for Ruled Paper

## Current Status ✅

**Working:** TrOCR produces good results on **blank paper** with handwriting  
**Problem:** Ruled lines interfere with text recognition

---

## Option 1: Line Removal Preprocessing ⭐ (Recommended)

### Approach
Remove horizontal ruled lines before feeding to TrOCR using computer vision.

### Algorithm

```python
import cv2
import numpy as np

def remove_ruled_lines(image, line_thickness_range=(1, 3)):
    """
    Remove horizontal ruled lines from image
    
    Args:
        image: Grayscale image
        line_thickness_range: Expected thickness of ruled lines in pixels
    
    Returns:
        Image with ruled lines removed
    """
    # 1. Detect horizontal lines using morphological operations
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    detect_horizontal = cv2.morphologyEx(image, cv2.MORPH_OPEN, 
                                         horizontal_kernel, iterations=2)
    
    # 2. Create line mask
    cnts = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, 
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        # Only remove thin horizontal lines (ruled lines)
        if line_thickness_range[0] <= h <= line_thickness_range[1] and w > 100:
            cv2.drawContours(image, [c], -1, (255, 255, 255), -1)
    
    # 3. Optional: Inpainting to fill removed areas
    # mask = cv2.threshold(detect_horizontal, 0, 255, cv2.THRESH_BINARY)[1]
    # image = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
    
    return image
```

### Pros
- ✅ Simple and effective
- ✅ No retraining needed
- ✅ Works with existing TrOCR model
- ✅ Fast (< 100ms per image)

### Cons
- ⚠️ May remove parts of letters that touch lines (like 'g', 'y', 'p')
- ⚠️ Requires tuning for different line thicknesses

### Implementation Steps
1. Add to preprocessing pipeline before line segmentation
2. Test on sample ruled paper images
3. Tune `line_thickness_range` parameter
4. Optionally add inpainting to restore removed pixels

---

## Option 2: Hough Line Transform (More Robust)

### Approach
Use Hough transform to detect precise line positions and mask them.

### Algorithm

```python
def remove_lines_hough(image):
    """
    Remove ruled lines using Hough line detection
    """
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    
    # Detect lines (only horizontal: theta near 0 or π)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100,
                           minLineLength=100, maxLineGap=10)
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Only horizontal lines
            if abs(y2 - y1) < 3:  # Near horizontal
                cv2.line(image, (x1, y1), (x2, y2), (255, 255, 255), 3)
    
    return image
```

### Pros
- ✅ More precise line detection
- ✅ Can handle slanted ruled lines
- ✅ Better at avoiding text

### Cons
- ⚠️ Slower than morphological approach
- ⚠️ May miss faint lines

---

## Option 3: Fine-tune TrOCR on Ruled Paper Dataset

### Approach
Fine-tune TrOCR model on images with ruled lines.

### Steps

```python
from transformers import VisionEncoderDecoderModel, TrOCRProcessor, Seq2SeqTrainer

# Load base model
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')

# Prepare dataset (ruled paper handwriting)
# - Collect 500-1000 ruled paper images
# - Annotate ground truth text

# Fine-tune
trainer = Seq2SeqTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    # ... trainer config
)

trainer.train()
```

### Pros
- ✅ Model learns to ignore ruled lines
- ✅ Best accuracy potential
- ✅ No preprocessing needed after training

### Cons
- ❌ Requires large dataset (500-1000 images)
- ❌ Requires GPU for training (hours)
- ❌ More complex implementation

---

## Option 4: Background Subtraction (Adaptive)

### Approach
Estimate ruled line pattern and subtract from image.

### Algorithm

```python
def adaptive_line_removal(image):
    """
    Detect ruled line pattern and subtract
    """
    # 1. Estimate line spacing using autocorrelation
    h_projection = np.sum(image, axis=1)
    autocorr = np.correlate(h_projection, h_projection, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    
    # Find peaks (line spacing)
    peaks = find_peaks(autocorr, distance=20)[0]
    line_spacing = np.median(np.diff(peaks)) if len(peaks) > 1 else None
    
    if line_spacing:
        # 2. Create synthetic ruled line template
        template = create_line_template(image.shape, line_spacing)
        
        # 3. Subtract template
        result = cv2.subtract(image, template)
        return result
    
    return image
```

### Pros
- ✅ Adaptive to different line spacings
- ✅ Preserves text better

### Cons
- ⚠️ Complex implementation
- ⚠️ May fail on irregular spacing

---

## Recommended Action Plan

### Short-term (This Week) ⭐

**Use Option 1: Line Removal Preprocessing**

1. **Implement** basic morphological line removal
2. **Test** on 5-10 ruled paper samples
3. **Tune** parameters (line thickness, kernel size)
4. **Integrate** into existing pipeline

**Expected time:** 2-3 hours  
**Expected improvement:** 60-80% accuracy on ruled paper

### Code Integration

Update your Colab notebook line segmentation:

```python
def segment_lines(image_path, remove_lines=True):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # NEW: Remove ruled lines
    if remove_lines:
        gray = remove_ruled_lines(gray)
    
    # Rest of existing segmentation code...
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    # ...
```

### Medium-term (Next Week)

**Test Option 2: Hough Transform** for comparison

- Compare accuracy vs. morphological approach
- Test on various paper types (wide-ruled, college-ruled, graph paper)

### Long-term (Optional)

**Option 3: Fine-tuning** if you need production-level accuracy
- Collect ruled paper dataset
- Fine-tune on Kaggle/Colab GPU

---

## Testing Strategy

### Create Test Set

1. **Write test text** on different paper types:
   - Blank paper (baseline - already working)
   - Wide-ruled
   - College-ruled
   - Graph paper

2. **Measure accuracy:**
   ```python
   from difflib import SequenceMatcher
   
   def text_similarity(predicted, ground_truth):
       return SequenceMatcher(None, predicted, ground_truth).ratio()
   ```

3. **Compare:**
   - Baseline (no preprocessing)
   - With line removal
   - Different line removal parameters

---

## Expected Results

| Paper Type | Current | After Line Removal | After Fine-tuning |
|------------|---------|-------------------|-------------------|
| Blank | ✅ Good | ✅ Good | ✅ Excellent |
| Wide-ruled | ❌ Poor | ⚠️ 60-80% | ✅ 90%+ |
| College-ruled | ❌ Poor | ⚠️ 50-70% | ✅ 85%+ |
| Graph paper | ❌ Poor | ⚠️ 40-60% | ✅ 80%+ |

---

## Quick Win: Start Here ⚡

Add this to your existing Colab notebook **right now**:

```python
# Add after imports
def quick_line_removal(img):
    """Quick ruled line removal - start with this"""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
    lines = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=2)
    # Remove detected lines
    img[lines > 0] = 255
    return img

# Use in segment_lines function
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = quick_line_removal(gray)  # Add this line
```

Test and let me know results!
