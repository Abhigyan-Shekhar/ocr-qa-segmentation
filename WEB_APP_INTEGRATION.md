# Web App Integration - Handwriting Notice

## Step 1: Add to HTML (docs/index.html)

Add this section **after the upload-card div closes (around line 77)**:

```html
<!-- Handwriting Info Box -->
<div class="info-box handwriting-notice">
    <div class="info-icon">📝</div>
    <div class="info-content">
        <h3>For Handwritten Text Recognition</h3>
        <p>This web app works best for <strong>typed text and screenshots</strong> (uses Tesseract.js).</p>
        <p>For <strong>handwritten text on blank paper</strong>, use our advanced TrOCR notebook with line removal preprocessing:</p>
        <a href="https://colab.research.google.com/drive/YOUR_NOTEBOOK_ID" class="btn btn-colab" target="_blank" rel="noopener">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                <path d="M12 2L2 7v10c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V7l-10-5z"></path>
            </svg>
            Open TrOCR Handwriting Notebook (Google Colab)
        </a>
        <p class="info-note">⚡ Works on blank paper | ⚠️ Ruled lines may affect accuracy</p>
    </div>
</div>
```

**Where to place it:**
After this closing tag:
```html
                </div>  <!-- closes upload-card -->
            </upload-section>
```

Add the info box, then close the section.

---

## Step 2: Add to CSS (docs/styles.css)

Add this at the **bottom of the file (around line 747)**:

```css
/* ===== INFO BOX ===== */
.info-box {
    max-width: 900px;
    margin: var(--spacing-lg) auto 0;
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%);
    border: 2px solid rgba(99, 102, 241, 0.3);
    border-radius: 16px;
    padding: var(--spacing-lg);
    display: flex;
    gap: var(--spacing-md);
    align-items: flex-start;
}

.info-icon {
    font-size: 3rem;
    flex-shrink: 0;
    filter: drop-shadow(0 0 10px rgba(99, 102, 241, 0.5));
}

.info-content h3 {
    font-size: 1.4rem;
    margin-bottom: var(--spacing-sm);
    background: var(--accent-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.info-content p {
    color: var(--text-secondary);
    line-height: 1.6;
    margin-bottom: var(--spacing-sm);
}

.btn-colab {
    background: linear-gradient(135deg, #F9AB00 0%, #F57C00 100%);
    color: white;
    box-shadow: 0 4px 16px rgba(245, 158, 11, 0.3);
    margin: var(--spacing-md) 0;
    padding: var(--spacing-sm) var(--spacing-md);
    border-radius: 12px;
    text-decoration: none;
    display: inline-flex;
    align-items: center;
    gap: var(--spacing-xs);
    font-weight: 600;
    transition: all var(--transition-fast);
}

.btn-colab:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 24px rgba(245, 158, 11, 0.4);
}

.btn-colab svg {
    stroke: white;
    flex-shrink: 0;
}

.info-note {
    font-size: 0.9rem;
    color: var(--text-muted);
    font-style: italic;
    margin-top: var(--spacing-xs);
}
```

---

## Step 3: Get Colab Shareable Link

1. **Upload your notebook** to Google Drive
   - Upload `notebooks/htr_trocr_colab.ipynb`
   
2. **Open in Colab:**
   - Right-click → "Open with" → "Google Colaboratory"

3. **Get shareable link:**
   - Click "Share" button (top right)
   - Change to "Anyone with the link can view"
   - Copy the link
   - Replace `YOUR_NOTEBOOK_ID` in the HTML above

---

## Step 4: Test Locally

```bash
cd /Users/abhigyanshekhar/Desktop/intern/ocr_qa_segmentation/docs
python3 -m http.server 8000
```

Open: http://localhost:8000

---

## Complete Integration Checklist

- [ ] Add HTML code to `docs/index.html` (after upload-card)
- [ ] Add CSS code to `docs/styles.css` (at bottom)
- [ ] Upload notebook to Google Drive
- [ ] Get Colab shareable link
- [ ] Replace `YOUR_NOTEBOOK_ID` with real link
- [ ] Test locally
- [ ] Commit and push to GitHub
- [ ] Verify on GitHub Pages

---

## Expected Result

You'll see a prominent orange "Open TrOCR Handwriting Notebook" button with:
- Beautiful gradient styling
- Clear usage guidance
- Direct link to Colab notebook
- Note about paper types

This keeps your web app lightweight (client-side only) while providing access to advanced handwriting recognition! 🎯
