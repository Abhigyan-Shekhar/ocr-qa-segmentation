# Testing the Complete Pipeline on Google Colab

## Quick Start Guide

### Step 1: Upload Notebook to Colab

1. Go to https://colab.research.google.com/
2. Click **File** → **Upload notebook**
3. Upload `complete_htr_qa_pipeline.ipynb`

---

### Step 2: Prepare Files

You'll need to upload these 2 files during execution:

#### ✅ File 1: CRF Model
- **File:** `models/qa_segmentation_crf_squad.pkl` (41 KB)
- **When:** Cell 3 (Upload CRF Model)
- **Location:** Already in your repo

#### ✅ File 2: Test Image
- **When:** Cell 8 (Upload & Process Image)
- **Recommendation:** Use a handwritten Q&A exam page
- **Best results:** Blank paper (no ruled lines) or use line removal

---

### Step 3: Run the Notebook

**Option A: Run All (Fastest)**
1. Click **Runtime** → **Run all**
2. Upload files when prompted
3. Wait ~5-10 minutes
4. Download JSON output

**Option B: Step-by-Step**
1. Run cells 1-7 (setup & model loading)
2. Upload CRF model when prompted (cell 3)
3. Upload test image (cell 8)
4. Run cells 9-13 (processing & results)

---

## What to Expect

### Cell 3 Output:
```
✅ CRF model loaded
   Validation F1: 1.0000
   Labels: ['B-Q', 'I-Q', 'B-A', 'I-A', 'O']
```

### Cell 9 Output:
```
✅ Found 8 text lines
[Visualization showing detected lines]
```

### Cell 10 Output:
```
Processing line 1/8... ✓ Text: Q1. What is machine learning?
Processing line 2/8... ✓ Text: Machine learning is a branch...
...
```

### Cell 11 Output:
```
[B-Q  ] Q1. What is machine learning?
[B-A  ] Machine learning is a branch...
[I-A  ] of artificial intelligence...
```

### Cell 12 Output:
```
Q1: What is machine learning?
A: Machine learning is a branch of artificial intelligence...
```

---

## Test Images

### Where to get test images:

**Option 1: Use Previous Test Images**
- Check if you have any handwritten exam samples from earlier testing

**Option 2: Create New Test**
1. Write a simple Q&A on blank paper:
   ```
   Q1. What is AI?
   A: AI is artificial intelligence
   
   Q2. What is ML?
   A: ML is machine learning
   ```
2. Take photo with phone
3. Upload to Colab

**Option 3: Use TrOCR Test Image**
- If you have the ruled paper image from TrOCR testing, use that
- The pipeline will remove ruled lines automatically

---

## Troubleshooting

### Issue: TrOCR model loading slow
**Solution:** First run takes ~2-3 minutes to download model. Subsequent runs are instant.

### Issue: No lines detected
**Cause:** Image quality too low or no text
**Solution:** Ensure clear handwriting, good lighting

### Issue: Poor OCR accuracy
**Cause:** Heavy cursive or complex handwriting
**Solution:** Use clearer handwriting or typed text for demo

### Issue: Wrong Q&A segmentation
**Cause:** Unusual format or no markers
**Solution:** Ensure questions start with "Q1.", "Q2." etc.

---

## Expected Results

**For well-formatted exam (typed/clear handwriting):**
- ✅ Line detection: 95-100% accurate
- ✅ OCR accuracy: 70-90% (varies by handwriting)
- ✅ Q&A segmentation: 100% (with CRF)

**Output JSON structure:**
```json
{
  "input_image": "exam_page.jpg",
  "num_lines": 8,
  "num_qa_pairs": 2,
  "qa_pairs": [
    {
      "question_number": 1,
      "question": "What is machine learning?",
      "answer": "Machine learning is..."
    },
    {
      "question_number": 2,
      "question": "What is deep learning?",
      "answer": "Deep learning is..."
    }
  ],
  "raw_text": "Q1. What is machine learning? Machine learning is...",
  "tagged_lines": [...]
}
```

---

## After Testing

Once you confirm it works:
1. Download the output JSON
2. Share results or screenshots
3. We'll push to GitHub with documentation

---

## Quick Checklist

Before running:
- [ ] Uploaded notebook to Colab
- [ ] Have `qa_segmentation_crf_squad.pkl` ready
- [ ] Have test image ready
- [ ] Connected to Colab runtime

During execution:
- [ ] Cell 1: Dependencies installed
- [ ] Cell 2: TrOCR loaded
- [ ] Cell 3: CRF uploaded & loaded
- [ ] Cell 8: Image uploaded
- [ ] Cells 9-12: Processing successful
- [ ] Cell 13: JSON downloaded

---

## Need Help?

If anything doesn't work:
1. Check error messages in Colab
2. Verify file uploads completed
3. Ensure image is readable
4. Try simpler test case first

Ready to test! 🚀
