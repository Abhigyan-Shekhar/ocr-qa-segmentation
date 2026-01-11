# Quick Start Guide

## Installation

1. **Navigate to project directory:**
   ```bash
   cd /Users/abhigyanshekhar/Desktop/intern/ocr_qa_segmentation
   ```

2. **Activate virtual environment:**
   ```bash
   source venv/bin/activate
   ```

3. **Verify installation:**
   ```bash
   python scripts/quick_test.py
   ```

   You should see:
   ```
   ============================================================
   ALL TESTS PASSED ✓
   ============================================================
   ```

## Usage

### Option 1: Test with Synthetic Data

Train a quick model to test the system:

```bash
source venv/bin/activate
python scripts/train.py --use-synthetic --output models/demo_model.pkl
```

### Option 2: Process Real Exams

1. **Annotate training data:**
   ```bash
   python scripts/annotate.py \
       --image path/to/exam.jpg \
       --output data/training_data.json \
       --append
   ```

2. **Train model:**
   ```bash
   python scripts/train.py \
       --data data/training_data.json \
       --output models/crf_model.pkl
   ```

3. **Run inference:**
   ```bash
   python scripts/inference.py \
       --images exam_page1.jpg exam_page2.jpg \
       --model models/crf_model.pkl \
       --output results.json \
       --print-text
   ```

## Notes

- **OCR Engine:** Using Tesseract (PaddleOCR requires Python <3.13)
- **Virtual Environment:** Always activate with `source venv/bin/activate`
- **Tesseract Installation:** If Tesseract is not installed, install via: `brew install tesseract`

## See More

- [README.md](README.md) - Full documentation
- [examples/demo.ipynb](examples/demo.ipynb) - Interactive demo
