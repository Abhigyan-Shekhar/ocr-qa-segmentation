# 📸 OCR Q&A Segmentation Gallery

A beautiful, Google Images-style gallery for viewing OCR results from handwritten exam papers.

![Gallery Preview](https://img.shields.io/badge/Status-Ready_to_Deploy-success)

## ✨ Features

- 🎨 **Modern Dark Theme** - Premium glassmorphism design with smooth animations
- 🔍 **Smart Search** - Search through questions, answers, and metadata
- 🏷️ **Confidence Filters** - Filter by High/Medium/Low confidence scores
- 🖼️ **Lightbox Modal** - Click to view full details with zoom controls
- ⌨️ **Keyboard Navigation** - Arrow keys to navigate, ESC to close
- 📱 **Fully Responsive** - Works on desktop, tablet, and mobile
- ⚡ **Zero Dependencies** - Pure HTML/CSS/JavaScript
- 🚀 **GitHub Pages Ready** - Deploy in 5 minutes

## 🎯 Quick Start

### View Locally

```bash
cd docs
python3 -m http.server 8000
```

Then open: **http://localhost:8000**

### Deploy to GitHub Pages

See [DEPLOYMENT.md](DEPLOYMENT.md) for full instructions.

**Quick steps:**
1. Push your repo to GitHub
2. Go to Settings → Pages
3. Select branch `main` and folder `/docs`
4. Done! Site will be live at `https://YOUR_USERNAME.github.io/REPO_NAME/`

## 📁 File Structure

```
docs/
├── index.html          # Main gallery page
├── styles.css          # Premium dark theme styling
├── app.js              # Gallery functionality + demo data
├── data/
│   └── sample_data.json    # Sample OCR results
├── images/             # (Create this) Your exam images
└── DEPLOYMENT.md       # Deployment guide
```

## 🔧 Customization

### Add Your OCR Results

**Option 1: Edit demo data in `app.js`**

Find the `demoData` array and modify:

```javascript
const demoData = [
  {
    id: 1,
    image: "images/your_exam.jpg",
    title: "Your Document",
    confidence: 0.95,
    // ... rest of data
  }
];
```

**Option 2: Use external JSON file**

1. Create your data in `data/ocr_results.json`
2. Uncomment the `loadRealData()` section at bottom of `app.js`

### Add Your Images

```bash
mkdir docs/images
cp your-exam-images/*.jpg docs/images/
```

Update image paths in your data to match.

## 🎨 Color Customization

Edit CSS variables in `styles.css`:

```css
:root {
  --accent-primary: #6366f1;    /* Main accent color */
  --accent-secondary: #8b5cf6;  /* Secondary accent */
  --bg-primary: #0a0e1a;        /* Background */
  /* ... */
}
```

## ⌨️ Keyboard Shortcuts

When lightbox is open:
- `←` Previous image
- `→` Next image
- `ESC` Close lightbox

## 📊 Data Format

Expected JSON structure:

```json
{
  "id": 1,
  "image": "images/exam.jpg",
  "title": "Document Title",
  "confidence": 0.95,
  "pages": 2,
  "engine": "Tesseract LSTM",
  "processingTime": "1.2s",
  "qaItems": [
    {
      "question": "Q1. Question text?",
      "answer": "Answer text here."
    }
  ]
}
```

## 🚀 Deployment Options

- **GitHub Pages** (Free) - See [DEPLOYMENT.md](DEPLOYMENT.md)
- **Netlify** (Free) - Drag & drop the `docs/` folder
- **Vercel** (Free) - Connect GitHub repo
- **Cloudflare Pages** (Free) - Connect GitHub repo

## 🌟 Features Showcase

### Gallery Grid
- Auto-adjusting grid layout
- Hover animations
- Confidence badges
- Document previews

### Search & Filter
- Real-time search across all content
- Filter by confidence level
- Result count display

### Lightbox
- Full-screen document viewer
- Zoom controls (in/out/reset)
- Q&A pair display
- Metadata panel
- Download JSON
- Copy text to clipboard

## 🎓 Perfect For

- Academic projects
- Portfolio showcasing
- Client demos
- Research presentations
- Educational tools

## 📝 License

See main project LICENSE

## 🤝 Contributing

This gallery is part of the OCR Q&A Segmentation project. See main README for contribution guidelines.

---

**Built with ❤️ using vanilla HTML/CSS/JavaScript**
