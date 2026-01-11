# 🚀 Deploying OCR Gallery to GitHub Pages

This guide will help you deploy your OCR Q&A Segmentation Gallery to GitHub Pages for **FREE hosting**.

## 📋 Prerequisites

- GitHub account
- Git installed on your computer
- Your OCR project repository

---

## ⚡ Quick Deployment (5 Minutes)

### Step 1: Prepare Your Repository

The gallery is already in the `docs/` folder. GitHub Pages can serve directly from this folder!

```bash
# Navigate to your project
cd /Users/abhigyanshekhar/Desktop/intern/ocr_qa_segmentation

# Check if you have a git repository
git status

# If not initialized, create one:
# git init
# git add .
# git commit -m "Initial commit with OCR gallery"
```

### Step 2: Push to GitHub

```bash
# Create a new repository on GitHub (via web interface)
# Then link it to your local repo:

git remote add origin https://github.com/YOUR_USERNAME/ocr-qa-segmentation.git
git branch -M main
git push -u origin main
```

### Step 3: Enable GitHub Pages

1. Go to your repository on GitHub
2. Click **Settings** (top right)
3. Scroll down to **Pages** (left sidebar)
4. Under **Source**, select:
   - Branch: `main`
   - Folder: `/docs`
5. Click **Save**

### Step 4: Access Your Site

After 2-3 minutes, your site will be live at:

```
https://YOUR_USERNAME.github.io/ocr-qa-segmentation/
```

---

## 🎨 Customization

### Add Your Own Data

Replace the demo data in `docs/app.js`:

1. **Option A: Inline Data** - Edit the `demoData` array in `app.js`

2. **Option B: External JSON** - Create `docs/data/ocr_results.json`:

```json
[
  {
    "id": 1,
    "image": "images/exam1.jpg",
    "title": "Exam Document 1",
    "confidence": 0.95,
    "pages": 2,
    "engine": "Tesseract LSTM",
    "processingTime": "1.2s",
    "qaItems": [
      {
        "question": "Q1. Your question here?",
        "answer": "Your answer here."
      }
    ]
  }
]
```

Then uncomment the `loadRealData()` function in `app.js` (bottom of file).

### Add Your Images

1. Create folder: `docs/images/`
2. Add your exam images (JPG/PNG)
3. Update image paths in your data

```bash
mkdir docs/images
cp examples/sample_data/test_document.png docs/images/
```

### Update GitHub Link

In `docs/index.html`, find line with:
```html
<a href="https://github.com/Abhigyan-Shekhar/ocr-qa-segmentation" target="_blank">
```

Replace with your actual GitHub repo URL.

---

## 🔧 Advanced: Auto-Generate Gallery from OCR Results

Create a Python script to convert your OCR results to gallery data:

```python
# scripts/export_to_gallery.py
import json
import os
from pathlib import Path

def export_results_to_gallery(results_dir, output_file):
    """Convert OCR results to gallery format"""
    gallery_data = []
    
    # Load all result JSON files
    for result_file in Path(results_dir).glob("*.json"):
        with open(result_file) as f:
            data = json.load(f)
        
        gallery_item = {
            "id": len(gallery_data) + 1,
            "image": f"images/{result_file.stem}.jpg",
            "title": f"Document {len(gallery_data) + 1}",
            "confidence": data.get("avg_confidence", 0.0),
            "pages": len(data.get("pages", [1])),
            "engine": data.get("ocr_engine", "Tesseract"),
            "processingTime": data.get("processing_time", "N/A"),
            "qaItems": data.get("qa_pairs", [])
        }
        gallery_data.append(gallery_item)
    
    # Export to docs/data/
    os.makedirs("docs/data", exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(gallery_data, f, indent=2)
    
    print(f"✓ Exported {len(gallery_data)} items to {output_file}")

if __name__ == "__main__":
    export_results_to_gallery("outputs/", "docs/data/ocr_results.json")
```

Run it after processing exams:
```bash
python scripts/export_to_gallery.py
```

---

## 🌐 Alternative Hosting Options

### Option 1: Netlify (Easiest)

1. Go to [netlify.com](https://netlify.com)
2. Sign in with GitHub
3. Click "New site from Git"
4. Select your repository
5. Set publish directory: `docs`
6. Deploy!

**Advantages:**
- Custom domains (free)
- Automatic HTTPS
- Instant preview URLs
- Form handling (if needed)

### Option 2: Vercel

```bash
npm install -g vercel
cd docs
vercel --prod
```

### Option 3: Cloudflare Pages

1. Go to [pages.cloudflare.com](https://pages.cloudflare.com)
2. Connect GitHub repository
3. Build settings:
   - Build command: (leave empty)
   - Build output: `docs`
4. Deploy!

---

## 📱 Local Testing

Before deploying, test locally:

```bash
# Option 1: Python simple server
cd docs
python3 -m http.server 8000

# Then open: http://localhost:8000
```

```bash
# Option 2: VS Code Live Server extension
# Right-click index.html → Open with Live Server
```

---

## 🎯 SEO & Performance Tips

### 1. Add Meta Tags (already included in index.html)

```html
<meta name="description" content="OCR Q&A Segmentation Gallery - View digitized exam results">
<meta property="og:title" content="OCR Q&A Gallery">
<meta property="og:image" content="preview.png">
```

### 2. Optimize Images

```bash
# Install imagemagick
brew install imagemagick  # macOS

# Compress images
for img in docs/images/*.jpg; do
    convert "$img" -quality 85 -resize 1200x1200\> "$img"
done
```

### 3. Add Google Analytics (Optional)

In `index.html` before `</head>`:

```html
<script async src="https://www.googletagmanager.com/gtag/js?id=YOUR-GA-ID"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'YOUR-GA-ID');
</script>
```

---

## ✅ Deployment Checklist

- [ ] Gallery works locally (`python3 -m http.server`)
- [ ] Images are in `docs/images/`
- [ ] Data is updated in `app.js` or `data/ocr_results.json`
- [ ] GitHub link updated in footer
- [ ] Repository pushed to GitHub
- [ ] GitHub Pages enabled in Settings
- [ ] Site accessible at `https://YOUR_USERNAME.github.io/REPO_NAME/`

---

## 🐛 Troubleshooting

### Issue: 404 Page Not Found

**Solution:** Make sure you selected `/docs` folder in GitHub Pages settings, not root.

### Issue: CSS/JS Not Loading

**Solution:** Check file paths. GitHub Pages is case-sensitive!

```html
<!-- ❌ Wrong -->
<link rel="stylesheet" href="Styles.css">

<!-- ✓ Correct -->
<link rel="stylesheet" href="styles.css">
```

### Issue: Images Not Showing

**Solution:** Use relative paths, not absolute:

```javascript
// ❌ Wrong
image: "/images/exam1.jpg"

// ✓ Correct
image: "images/exam1.jpg"
```

### Issue: Custom Domain Not Working

**Solution:** Add a `CNAME` file in `docs/`:

```bash
echo "your-domain.com" > docs/CNAME
git add docs/CNAME
git commit -m "Add custom domain"
git push
```

---

## 🎉 You're Done!

Your OCR gallery is now live and accessible to anyone with the URL. Share it in your portfolio, resume, or with collaborators!

**Next Steps:**
- Add more documents to showcase
- Integrate with your OCR processing pipeline
- Add authentication for private documents
- Export functionality to PDF/Word

---

**Need help?** Open an issue on GitHub or check the [GitHub Pages docs](https://docs.github.com/en/pages).
