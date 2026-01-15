# HTR Notebooks

This directory contains Jupyter/Colab notebooks for handwriting recognition experiments.

## Notebooks

### 1. Training Notebook (Kaggle)

**File:** `train_htr_tensorflow.ipynb`

**Platform:** Kaggle (requires GPU)

**Purpose:** Train CRNN+CTC model on IAM Handwriting Database

**Quick Start:**
1. Upload to Kaggle
2. Attach IAM dataset from Kaggle Datasets
3. Enable GPU accelerator
4. Run all cells

**Output:** `htr_model.keras` (31.4 MB)

---

### 2. TrOCR Inference (Google Colab)

**File:** `htr_trocr_colab.ipynb`

**Platform:** Google Colab

**Purpose:** Line-by-line handwriting recognition using Microsoft TrOCR

**Features:**
- Automatic line segmentation (horizontal projection)
- TrOCR model (`microsoft/trocr-base-handwritten`)
- Works best on typed text and clear handwriting

**Quick Start:**
1. Open in Google Colab
2. Upload your handwritten page
3. Run all cells
4. Download recognized text

---

## Model Files

**Note:** Large model files (`.keras`, `.h5`) are excluded from git.

**To use pre-trained model:**
- Download from Kaggle output after training
- Place in `/models/` directory
- Use `scripts/inference_htr.py` for local inference

---

## Requirements

**For Kaggle:** Built-in TensorFlow 2.x

**For Colab:**
```bash
pip install transformers pillow opencv-python-headless matplotlib torch
```

---

## Results

See [EXPLORATION.md](../EXPLORATION.md) for complete documentation of all approaches tested.
