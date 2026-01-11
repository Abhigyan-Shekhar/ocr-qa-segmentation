// ===== OCR Q&A Extraction App =====
// Client-side OCR using Tesseract.js

let ocrWorker = null;
let currentImage = null;
let extractedData = null;
let startTime = null;

// ===== INITIALIZATION =====
document.addEventListener('DOMContentLoaded', () => {
    initializeEventListeners();
    initializeTesseract();
});

function initializeEventListeners() {
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('fileInput');

    // Click to browse
    dropzone.addEventListener('click', () => fileInput.click());

    // File input change
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFiles(e.target.files);
        }
    });

    // Drag and drop
    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('dragover');
    });

    dropzone.addEventListener('dragleave', () => {
        dropzone.classList.remove('dragover');
    });

    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            handleFiles(e.dataTransfer.files);
        }
    });

    // Upload another button
    document.getElementById('uploadAnotherBtn').addEventListener('click', resetToUpload);

    // Download button
    document.getElementById('downloadBtn').addEventListener('click', downloadResults);

    // Copy text button
    document.getElementById('copyTextBtn').addEventListener('click', copyToClipboard);

    // View tabs
    document.querySelectorAll('.view-tab').forEach(tab => {
        tab.addEventListener('click', () => switchView(tab.dataset.view));
    });
}

async function initializeTesseract() {
    try {
        ocrWorker = await Tesseract.createWorker('eng', 1, {
            logger: m => updateProgress(m)
        });
        console.log('Tesseract worker initialized');
    } catch (error) {
        console.error('Failed to initialize Tesseract:', error);
    }
}

// ===== FILE HANDLING =====
function handleFiles(files) {
    if (files.length === 0) return;

    const file = files[0]; // For now, handle single file

    // Validate file type
    if (!file.type.startsWith('image/')) {
        alert('Please upload an image file (JPG, PNG, etc.)');
        return;
    }

    // Validate file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
        alert('File size must be less than 10MB');
        return;
    }

    // Read and process the file
    const reader = new FileReader();
    reader.onload = (e) => {
        currentImage = {
            data: e.target.result,
            name: file.name,
            size: file.size
        };
        processImage();
    };
    reader.readAsDataURL(file);
}

// ===== OCR PROCESSING =====
async function processImage() {
    startTime = Date.now();

    // Show processing section
    document.getElementById('uploadSection').style.display = 'none';
    document.getElementById('processingSection').style.display = 'block';
    document.getElementById('resultsSection').style.display = 'none';

    try {
        // Perform OCR
        updateProcessingStatus('Extracting text from image...', 20);

        const result = await ocrWorker.recognize(currentImage.data);

        updateProcessingStatus('Analyzing Q&A patterns...', 70);

        // Extract Q&A pairs from OCR text
        const qaData = extractQAPairs(result.data.text, result.data.confidence);

        updateProcessingStatus('Finalizing results...', 90);

        // Store results
        extractedData = {
            rawText: result.data.text,
            confidence: result.data.confidence,
            qaPairs: qaData.pairs,
            processingTime: ((Date.now() - startTime) / 1000).toFixed(2) + 's'
        };

        // Show results
        setTimeout(() => {
            displayResults();
        }, 500);

    } catch (error) {
        console.error('OCR Error:', error);
        alert('Failed to process image. Please try again.');
        resetToUpload();
    }
}

function updateProgress(m) {
    if (m.status === 'recognizing text') {
        const percent = Math.round(m.progress * 100);
        document.getElementById('progressBar').style.width = percent + '%';
    }
}

function updateProcessingStatus(text, progress) {
    document.getElementById('processingText').textContent = text;
    document.getElementById('progressBar').style.width = progress + '%';
}

// ===== Q&A EXTRACTION =====
function extractQAPairs(text, confidence) {
    const lines = text.split('\n').filter(line => line.trim().length > 0);
    const pairs = [];
    let currentQuestion = null;
    let currentAnswer = [];
    let questionNumber = 1;

    // Regex patterns for question detection
    const questionPatterns = [
        /^(Q|q|Question|question)\s*[\d.]+/,  // Q1, Question 1, etc.
        /^\d+[\.\)]/,                          // 1., 1), etc.
        /\?$/                                   // Ends with ?
    ];

    for (let i = 0; i < lines.length; i++) {
        const line = lines[i].trim();

        // Check if this line looks like a question
        const isQuestion = questionPatterns.some(pattern => pattern.test(line));

        if (isQuestion || line.match(/\?$/)) {
            // Save previous Q&A pair if exists
            if (currentQuestion && currentAnswer.length > 0) {
                pairs.push({
                    number: questionNumber++,
                    question: currentQuestion,
                    answer: currentAnswer.join(' ').trim()
                });
            }

            // Start new question
            currentQuestion = line;
            currentAnswer = [];
        } else if (currentQuestion) {
            // This is part of the answer
            currentAnswer.push(line);
        } else {
            // Text before first question - might be header/title
            if (pairs.length === 0) {
                // Skip preamble
                continue;
            }
        }
    }

    // Add the last Q&A pair
    if (currentQuestion && currentAnswer.length > 0) {
        pairs.push({
            number: questionNumber,
            question: currentQuestion,
            answer: currentAnswer.join(' ').trim()
        });
    }

    // If no Q&A pairs detected, create a single entry with all text
    if (pairs.length === 0) {
        pairs.push({
            number: 1,
            question: 'Extracted Text',
            answer: text.trim()
        });
    }

    return { pairs, confidence };
}

// ===== DISPLAY RESULTS =====
function displayResults() {
    // Hide processing, show results
    document.getElementById('processingSection').style.display = 'none';
    document.getElementById('resultsSection').style.display = 'block';

    // Display image preview
    document.getElementById('previewImage').src = currentImage.data;
    document.getElementById('imageSize').textContent = formatFileSize(currentImage.size);
    document.getElementById('processingTime').textContent = extractedData.processingTime;

    // Display confidence
    const confidencePercent = Math.round(extractedData.confidence);
    const confidenceBadge = document.getElementById('confidenceBadge');
    confidenceBadge.textContent = confidencePercent + '% confidence';
    confidenceBadge.className = 'confidence-badge ' + getConfidenceClass(extractedData.confidence / 100);

    // Display Q&A pairs in sidebar
    const qaResults = document.getElementById('qaResults');
    qaResults.innerHTML = extractedData.qaPairs.length > 0
        ? `<div class="qa-summary">
             <div class="summary-item">
                 <span class="summary-number">${extractedData.qaPairs.length}</span>
                 <span class="summary-label">Q&A Pairs</span>
             </div>
             <div class="summary-item">
                 <span class="summary-number">${extractedData.rawText.split(/\s+/).length}</span>
                 <span class="summary-label">Words</span>
             </div>
           </div>`
        : '<p class="no-results">No Q&A pairs detected</p>';

    // Display Q&A list
    const qaList = document.getElementById('qaList');
    qaList.innerHTML = extractedData.qaPairs.map(qa => `
        <div class="qa-card">
            <div class="qa-header">
                <span class="qa-number">Q${qa.number}</span>
            </div>
            <div class="qa-question">${escapeHtml(qa.question)}</div>
            <div class="qa-answer">${escapeHtml(qa.answer)}</div>
        </div>
    `).join('');

    // Display raw text
    document.getElementById('rawText').textContent = extractedData.rawText;

    // Display JSON
    const jsonData = {
        image: currentImage.name,
        processingTime: extractedData.processingTime,
        confidence: extractedData.confidence,
        qaPairs: extractedData.qaPairs,
        rawText: extractedData.rawText
    };
    document.getElementById('jsonOutput').textContent = JSON.stringify(jsonData, null, 2);
}

// ===== UTILITY FUNCTIONS =====
function getConfidenceClass(confidence) {
    if (confidence >= 0.85) return 'confidence-high';
    if (confidence >= 0.70) return 'confidence-medium';
    return 'confidence-low';
}

function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function switchView(view) {
    // Update tabs
    document.querySelectorAll('.view-tab').forEach(tab => {
        tab.classList.toggle('active', tab.dataset.view === view);
    });

    // Update panels
    document.querySelectorAll('.view-panel').forEach(panel => {
        panel.classList.remove('active');
    });
    document.getElementById(view + 'View').classList.add('active');
}

function resetToUpload() {
    document.getElementById('uploadSection').style.display = 'block';
    document.getElementById('processingSection').style.display = 'none';
    document.getElementById('resultsSection').style.display = 'none';
    document.getElementById('fileInput').value = '';
    currentImage = null;
    extractedData = null;
}

function downloadResults() {
    if (!extractedData) return;

    const jsonData = {
        image: currentImage.name,
        processingTime: extractedData.processingTime,
        confidence: extractedData.confidence,
        qaPairs: extractedData.qaPairs,
        rawText: extractedData.rawText
    };

    const dataStr = JSON.stringify(jsonData, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `ocr_result_${Date.now()}.json`;
    link.click();
    URL.revokeObjectURL(url);
}

function copyToClipboard() {
    if (!extractedData) return;

    const text = extractedData.qaPairs.map(qa =>
        `Q${qa.number}. ${qa.question}\n${qa.answer}\n`
    ).join('\n');

    navigator.clipboard.writeText(text).then(() => {
        const btn = document.getElementById('copyTextBtn');
        btn.innerHTML = `
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                <polyline points="20 6 9 17 4 12"></polyline>
            </svg>
        `;
        setTimeout(() => {
            btn.innerHTML = `
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                    <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                    <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                </svg>
            `;
        }, 2000);
    });
}

// Cleanup on page unload
window.addEventListener('beforeunload', async () => {
    if (ocrWorker) {
        await ocrWorker.terminate();
    }
});
