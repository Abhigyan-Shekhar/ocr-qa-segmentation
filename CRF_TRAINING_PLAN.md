# CRF Training Plan - Q&A Segmentation

## Recommended Kaggle Datasets

### Option 1: SQuAD (Stanford Question Answering) ⭐ BEST
**URL:** https://www.kaggle.com/datasets/stanfordu/stanford-question-answering-dataset

**Format:**
```json
{
  "data": [{
    "paragraphs": [{
      "context": "full text...",
      "qas": [{
        "question": "What is...?",
        "answers": [{"text": "...", "answer_start": 123}]
      }]
    }]
  }]
}
```

**Pros:**
- ✅ 100,000+ question-answer pairs
- ✅ Clean, structured data
- ✅ Can simulate document layout
- ✅ Well-maintained

**Training approach:**
1. Convert Q&A pairs to synthetic "exam page" format
2. Add simulated line breaks
3. Generate BIO tags
4. Train CRF

---

### Option 2: Natural Questions (Google)
**URL:** https://www.kaggle.com/datasets/google/natural-questions

**Pros:**
- Large scale (300k+ examples)
- Real questions
- Multiple answer types

**Cons:**
- More complex format
- Needs more preprocessing

---

### Option 3: MS MARCO Q&A
**URL:** https://www.kaggle.com/datasets/microsoft/ms-marco

**Pros:**
- Large dataset
- Passage-based Q&A
- Good for context learning

---

### Option 4: Create Synthetic Exam Data ⚡ FASTEST

**Approach:** Generate realistic exam pages programmatically

**Template:**
```
Q1. [Question text from dataset]
    (possibly multi-line)

A: [Answer text from dataset]
   (possibly multi-line)

Q2. [Next question]
...
```

**Advantage:**
- Full control over format
- Can add variations (indentation, spacing, markers)
- Simulates real exam structure

---

## Recommended Approach: SQuAD + Synthetic Generation

### Step 1: Download SQuAD
```python
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()
api.dataset_download_files('stanfordu/stanford-question-answering-dataset', 
                          path='./data', unzip=True)
```

### Step 2: Convert to Exam Format

```python
import json
import random

def squad_to_exam_format(squad_file, num_pages=100):
    """
    Convert SQuAD Q&A to synthetic exam pages
    """
    with open(squad_file) as f:
        squad = json.load(f)
    
    exam_pages = []
    
    for article in squad['data'][:num_pages]:
        page_lines = []
        
        for para in article['paragraphs']:
            for qa in para['qas'][:2]:  # 2 Q&A per paragraph
                # Question
                q_text = qa['question']
                q_lines = split_into_lines(q_text, prefix=f"Q{len(page_lines)+1}. ")
                page_lines.extend(q_lines)
                
                # Answer
                a_text = qa['answers'][0]['text'] if qa['answers'] else "Not found"
                a_lines = split_into_lines(a_text, prefix="A: ", indent=True)
                page_lines.extend(a_lines)
                
                # Spacing
                page_lines.append("")
        
        exam_pages.append(page_lines)
    
    return exam_pages

def split_into_lines(text, prefix="", indent=False, max_len=60):
    """Split text into realistic line lengths"""
    words = text.split()
    lines = []
    current_line = prefix
    indent_str = "    " if indent else ""
    
    for word in words:
        if len(current_line + word) > max_len and current_line:
            lines.append(current_line)
            current_line = indent_str + word + " "
        else:
            current_line += word + " "
    
    if current_line.strip():
        lines.append(current_line.strip())
    
    return lines
```

### Step 3: Generate BIO Labels

```python
def generate_bio_labels(exam_lines):
    """
    Annotate lines with BIO tags
    """
    labels = []
    in_question = False
    in_answer = False
    
    for line in exam_lines:
        line = line.strip()
        
        if not line:
            labels.append('O')
            in_question = False
            in_answer = False
        elif line.startswith('Q') and '. ' in line[:5]:
            labels.append('B-Q')
            in_question = True
            in_answer = False
        elif in_question and not line.startswith('A'):
            labels.append('I-Q')
        elif line.startswith('A:') or line.startswith('A. '):
            labels.append('B-A')
            in_question = False
            in_answer = True
        elif in_answer:
            labels.append('I-A')
        else:
            labels.append('O')
    
    return labels
```

### Step 4: Extract Features

```python
from src.feature_extraction import FeatureExtractor
from src.ocr_engine import OCRLine

def lines_to_features(exam_lines, page_width=800, page_height=1200):
    """
    Convert text lines to CRF features
    """
    # Simulate OCR output
    ocr_lines = []
    y_pos = 100
    
    for idx, line in enumerate(exam_lines):
        # Calculate indent
        indent = len(line) - len(line.lstrip())
        x_pos = 50 + (indent * 10)
        
        # Create OCR line
        ocr_line = OCRLine(
            text=line.strip(),
            bbox=(x_pos, y_pos, 400, 25),
            confidence=0.95,
            line_number=idx
        )
        ocr_lines.append(ocr_line)
        
        y_pos += 35  # Line height + gap
    
    # Extract features
    extractor = FeatureExtractor(page_width, page_height)
    features = extractor.extract_features(ocr_lines)
    crf_features = extractor.features_to_crf_format(features)
    
    return crf_features
```

### Step 5: Train CRF

```python
from src.crf_model import CRFModel

# Load and process data
exam_pages = squad_to_exam_format('data/train-v2.0.json', num_pages=200)

X_train = []
y_train = []

for page_lines in exam_pages:
    labels = generate_bio_labels(page_lines)
    features = lines_to_features(page_lines)
    
    X_train.append(features)
    y_train.append(labels)

# Split train/val
split = int(0.8 * len(X_train))
X_val = X_train[split:]
y_val = y_train[split:]
X_train = X_train[:split]
y_train = y_train[:split]

# Train
model = CRFModel(max_iterations=100)
results = model.train(X_train, y_train, X_val, y_val)

# Save
model.save('models/qa_segmentation_model.pkl')
```

---

## Quick Start Script

I'll create a complete training script that:
1. Downloads SQuAD data
2. Generates synthetic exam pages
3. Extracts features
4. Trains CRF
5. Evaluates performance

**Estimated time:** 15-30 minutes

Want me to create this script now?
