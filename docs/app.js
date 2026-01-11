// ===== DEMO DATA =====
// In production, this would be loaded from a JSON file or API
const demoData = [
    {
        id: 1,
        image: "https://via.placeholder.com/600x800/1a2035/6366f1?text=Exam+Page+1",
        title: "Exam Document 1",
        confidence: 0.95,
        pages: 2,
        engine: "Tesseract LSTM",
        processingTime: "1.2s",
        qaItems: [
            {
                question: "Q1. What is the capital of France?",
                answer: "Paris is the capital of France. It is located in the north-central part of the country and is known for its cultural heritage, architecture, and historical significance."
            },
            {
                question: "Q2. Explain the process of photosynthesis.",
                answer: "Photosynthesis is the process by which green plants use sunlight to synthesize nutrients from carbon dioxide and water. It involves the green pigment chlorophyll and generates oxygen as a by-product."
            }
        ]
    },
    {
        id: 2,
        image: "https://via.placeholder.com/600x800/1a2035/8b5cf6?text=Exam+Page+2",
        title: "Exam Document 2",
        confidence: 0.88,
        pages: 1,
        engine: "PaddleOCR",
        processingTime: "0.9s",
        qaItems: [
            {
                question: "Q1. Define machine learning in your own words.",
                answer: "Machine learning is a subset of artificial intelligence that enables computers to learn from data and improve their performance on tasks without being explicitly programmed. It uses algorithms to identify patterns in data."
            },
            {
                question: "Q2. What are the three basic types of machine learning?",
                answer: "The three basic types of machine learning are: 1) Supervised Learning - learning from labeled data, 2) Unsupervised Learning - finding patterns in unlabeled data, and 3) Reinforcement Learning - learning through trial and error with rewards."
            }
        ]
    },
    {
        id: 3,
        image: "https://via.placeholder.com/600x800/1a2035/10b981?text=Exam+Page+3",
        title: "Exam Document 3",
        confidence: 0.92,
        pages: 3,
        engine: "Tesseract LSTM",
        processingTime: "2.1s",
        qaItems: [
            {
                question: "Q1. Describe the water cycle.",
                answer: "The water cycle is the continuous movement of water on, above, and below the surface of the Earth. It involves evaporation, condensation, precipitation, and collection, creating a循环 that sustains life on our planet."
            }
        ]
    },
    {
        id: 4,
        image: "https://via.placeholder.com/600x800/1a2035/f59e0b?text=Exam+Page+4",
        title: "Exam Document 4",
        confidence: 0.78,
        pages: 1,
        engine: "Tesseract LSTM",
        processingTime: "1.0s",
        qaItems: [
            {
                question: "Q1. What is Newton's first law of motion?",
                answer: "Newton's first law of motion states that an object at rest stays at rest and an object in motion stays in motion with the same speed and direction unless acted upon by an unbalanced force. This is also known as the law of inertia."
            },
            {
                question: "Q2. Calculate the area of a circle with radius 5cm.",
                answer: "Area = πr² = 3.14159 × 5² = 3.14159 × 25 = 78.54 cm²"
            }
        ]
    },
    {
        id: 5,
        image: "https://via.placeholder.com/600x800/1a2035/6366f1?text=Exam+Page+5",
        title: "Exam Document 5",
        confidence: 0.91,
        pages: 2,
        engine: "PaddleOCR",
        processingTime: "1.5s",
        qaItems: [
            {
                question: "Q1. What is the theory of relativity?",
                answer: "The theory of relativity, developed by Albert Einstein, consists of special relativity and general relativity. Special relativity describes the physics of moving bodies in the absence of gravitational effects, while general relativity explains gravity as a curvature of spacetime."
            }
        ]
    },
    {
        id: 6,
        image: "https://via.placeholder.com/600x800/1a2035/ef4444?text=Exam+Page+6",
        title: "Exam Document 6",
        confidence: 0.68,
        pages: 1,
        engine: "Tesseract LSTM",
        processingTime: "1.3s",
        qaItems: [
            {
                question: "Q1. Explain DNA replication.",
                answer: "DNA replication is the biological process of producing two identical replicas from one original DNA molecule. This process occurs in all living organisms and is the basis for biological inheritance. The double helix unwinds and each strand serves as a template for a new complementary strand."
            }
        ]
    }
];

// ===== STATE MANAGEMENT =====
let currentFilter = 'all';
let currentSearchQuery = '';
let currentLightboxIndex = 0;
let filteredData = [...demoData];
let zoomLevel = 1;

// ===== UTILITY FUNCTIONS =====
function getConfidenceClass(confidence) {
    if (confidence >= 0.85) return 'confidence-high';
    if (confidence >= 0.70) return 'confidence-medium';
    return 'confidence-low';
}

function getConfidenceLabel(confidence) {
    if (confidence >= 0.85) return 'High';
    if (confidence >= 0.70) return 'Medium';
    return 'Needs Review';
}

function filterData() {
    filteredData = demoData.filter(item => {
        // Apply confidence filter
        const confidenceMatch =
            currentFilter === 'all' ||
            (currentFilter === 'high' && item.confidence >= 0.85) ||
            (currentFilter === 'medium' && item.confidence >= 0.70 && item.confidence < 0.85) ||
            (currentFilter === 'low' && item.confidence < 0.70);

        // Apply search filter
        const searchMatch = currentSearchQuery === '' ||
            item.title.toLowerCase().includes(currentSearchQuery.toLowerCase()) ||
            item.qaItems.some(qa =>
                qa.question.toLowerCase().includes(currentSearchQuery.toLowerCase()) ||
                qa.answer.toLowerCase().includes(currentSearchQuery.toLowerCase())
            );

        return confidenceMatch && searchMatch;
    });

    renderGallery();
}

// ===== GALLERY RENDERING =====
function renderGallery() {
    const gallery = document.getElementById('gallery');
    const resultCount = document.getElementById('resultCount');

    resultCount.textContent = `${filteredData.length} result${filteredData.length !== 1 ? 's' : ''}`;

    if (filteredData.length === 0) {
        gallery.innerHTML = `
            <div style="grid-column: 1 / -1; text-align: center; padding: 4rem; color: var(--text-muted);">
                <svg width="80" height="80" viewBox="0 0 24 24" fill="none" stroke="currentColor" style="margin: 0 auto 1rem;">
                    <circle cx="11" cy="11" r="8"></circle>
                    <path d="m21 21-4.35-4.35"></path>
                </svg>
                <h3>No results found</h3>
                <p>Try adjusting your filters or search query</p>
            </div>
        `;
        return;
    }

    gallery.innerHTML = filteredData.map((item, index) => {
        const confidenceClass = getConfidenceClass(item.confidence);
        const confidencePercent = Math.round(item.confidence * 100);
        const previewText = item.qaItems[0]?.question || 'No questions detected';

        return `
            <div class="gallery-item" data-index="${index}" style="animation-delay: ${index * 0.05}s">
                <img src="${item.image}" alt="${item.title}" class="gallery-item-image" loading="lazy">
                <div class="gallery-item-content">
                    <div class="gallery-item-header">
                        <h3 class="gallery-item-title">${item.title}</h3>
                        <span class="confidence-badge ${confidenceClass}">${confidencePercent}%</span>
                    </div>
                    <p class="gallery-item-preview">${previewText}</p>
                    <div class="gallery-item-meta">
                        <span>
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                                <polyline points="14 2 14 8 20 8"></polyline>
                            </svg>
                            ${item.pages} page${item.pages !== 1 ? 's' : ''}
                        </span>
                        <span>
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                                <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path>
                                <circle cx="12" cy="7" r="4"></circle>
                            </svg>
                            ${item.qaItems.length} Q&A
                        </span>
                    </div>
                </div>
            </div>
        `;
    }).join('');

    // Add click listeners
    document.querySelectorAll('.gallery-item').forEach(item => {
        item.addEventListener('click', () => {
            const index = parseInt(item.dataset.index);
            openLightbox(index);
        });
    });
}

// ===== LIGHTBOX =====
function openLightbox(index) {
    currentLightboxIndex = index;
    const item = filteredData[index];
    const lightbox = document.getElementById('lightbox');

    // Populate lightbox content
    document.getElementById('lightboxImage').src = item.image;
    document.getElementById('lightboxTitle').textContent = item.title;

    const confidenceClass = getConfidenceClass(item.confidence);
    const confidencePercent = Math.round(item.confidence * 100);
    const confidenceBadge = document.getElementById('lightboxConfidence');
    confidenceBadge.textContent = `${confidencePercent}% - ${getConfidenceLabel(item.confidence)}`;
    confidenceBadge.className = `confidence-badge ${confidenceClass}`;

    // Populate Q&A items
    const qaContainer = document.getElementById('qaContainer');
    qaContainer.innerHTML = item.qaItems.map(qa => `
        <div class="qa-item">
            <div class="qa-question">${qa.question}</div>
            <div class="qa-answer">${qa.answer}</div>
        </div>
    `).join('');

    // Populate metadata
    document.getElementById('metaDocId').textContent = item.id;
    document.getElementById('metaPages').textContent = item.pages;
    document.getElementById('metaEngine').textContent = item.engine;
    document.getElementById('metaTime').textContent = item.processingTime;

    // Show lightbox
    lightbox.classList.add('active');
    document.body.style.overflow = 'hidden';

    // Reset zoom
    zoomLevel = 1;
    updateZoom();
}

function closeLightbox() {
    const lightbox = document.getElementById('lightbox');
    lightbox.classList.remove('active');
    document.body.style.overflow = '';
}

function navigateLightbox(direction) {
    currentLightboxIndex += direction;

    if (currentLightboxIndex < 0) {
        currentLightboxIndex = filteredData.length - 1;
    } else if (currentLightboxIndex >= filteredData.length) {
        currentLightboxIndex = 0;
    }

    openLightbox(currentLightboxIndex);
}

function updateZoom() {
    const image = document.getElementById('lightboxImage');
    image.style.transform = `scale(${zoomLevel})`;
}

function downloadJSON() {
    const item = filteredData[currentLightboxIndex];
    const dataStr = JSON.stringify(item, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `ocr_result_${item.id}.json`;
    link.click();
    URL.revokeObjectURL(url);
}

function copyText() {
    const item = filteredData[currentLightboxIndex];
    const text = item.qaItems.map(qa => `${qa.question}\n${qa.answer}`).join('\n\n');
    navigator.clipboard.writeText(text).then(() => {
        const btn = document.getElementById('copyText');
        const originalText = btn.innerHTML;
        btn.innerHTML = `
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                <polyline points="20 6 9 17 4 12"></polyline>
            </svg>
            Copied!
        `;
        setTimeout(() => {
            btn.innerHTML = originalText;
        }, 2000);
    });
}

// ===== EVENT LISTENERS =====
document.addEventListener('DOMContentLoaded', () => {
    // Initial render
    renderGallery();

    // Search functionality
    const searchInput = document.getElementById('searchInput');
    searchInput.addEventListener('input', (e) => {
        currentSearchQuery = e.target.value;
        filterData();
    });

    // Filter tabs
    document.querySelectorAll('.tab').forEach(tab => {
        tab.addEventListener('click', () => {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            currentFilter = tab.dataset.filter;
            filterData();
        });
    });

    // Lightbox controls
    document.getElementById('prevImage').addEventListener('click', () => navigateLightbox(-1));
    document.getElementById('nextImage').addEventListener('click', () => navigateLightbox(1));

    document.querySelector('.lightbox-close').addEventListener('click', closeLightbox);
    document.querySelector('.lightbox-overlay').addEventListener('click', closeLightbox);

    // Zoom controls
    document.getElementById('zoomIn').addEventListener('click', () => {
        zoomLevel = Math.min(zoomLevel + 0.25, 3);
        updateZoom();
    });

    document.getElementById('zoomOut').addEventListener('click', () => {
        zoomLevel = Math.max(zoomLevel - 0.25, 0.5);
        updateZoom();
    });

    document.getElementById('resetZoom').addEventListener('click', () => {
        zoomLevel = 1;
        updateZoom();
    });

    // Action buttons
    document.getElementById('downloadJson').addEventListener('click', downloadJSON);
    document.getElementById('copyText').addEventListener('click', copyText);

    // Keyboard navigation
    document.addEventListener('keydown', (e) => {
        if (!document.getElementById('lightbox').classList.contains('active')) return;

        if (e.key === 'Escape') closeLightbox();
        if (e.key === 'ArrowLeft') navigateLightbox(-1);
        if (e.key === 'ArrowRight') navigateLightbox(1);
    });

    // View toggle (placeholder for future expansion)
    document.getElementById('toggleView').addEventListener('click', function () {
        const gallery = document.getElementById('gallery');
        if (gallery.style.gridTemplateColumns === '1fr') {
            gallery.style.gridTemplateColumns = 'repeat(auto-fill, minmax(300px, 1fr))';
            this.innerHTML = `
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                    <rect x="3" y="3" width="7" height="7"></rect>
                    <rect x="14" y="3" width="7" height="7"></rect>
                    <rect x="14" y="14" width="7" height="7"></rect>
                    <rect x="3" y="14" width="7" height="7"></rect>
                </svg>
                Grid View
            `;
        } else {
            gallery.style.gridTemplateColumns = '1fr';
            this.innerHTML = `
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                    <line x1="8" y1="6" x2="21" y2="6"></line>
                    <line x1="8" y1="12" x2="21" y2="12"></line>
                    <line x1="8" y1="18" x2="21" y2="18"></line>
                    <line x1="3" y1="6" x2="3.01" y2="6"></line>
                    <line x1="3" y1="12" x2="3.01" y2="12"></line>
                    <line x1="3" y1="18" x2="3.01" y2="18"></line>
                </svg>
                List View
            `;
        }
    });
});

// ===== LOADING REAL DATA (Optional) =====
// Uncomment this section to load data from a JSON file instead of using demo data
/*
async function loadRealData() {
    try {
        const response = await fetch('data/ocr_results.json');
        const data = await response.json();
        demoData.length = 0;
        demoData.push(...data);
        filterData();
    } catch (error) {
        console.error('Failed to load OCR data:', error);
        // Fallback to demo data
        renderGallery();
    }
}

// Call this instead of renderGallery() in DOMContentLoaded
// loadRealData();
*/
