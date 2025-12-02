/**
 * AI Brain Hemorrhage Detection System - Frontend Logic
 * Handles file upload, inference requests, and results visualization
 */

// State management
let currentFile = null;
let analysisResults = null;

// DOM Elements
const uploadZone = document.getElementById('uploadZone');
const fileInput = document.getElementById('fileInput');
const imagePreview = document.getElementById('imagePreview');
const previewImage = document.getElementById('previewImage');
const analyzeBtn = document.getElementById('analyzeBtn');
const loadingSection = document.getElementById('loading');
const resultsSection = document.getElementById('results');

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    initializeUploadZone();
    loadModelInfo();
});

/**
 * Initialize drag-and-drop upload zone
 */
function initializeUploadZone() {
    // Click to upload
    uploadZone.addEventListener('click', () => {
        fileInput.click();
    });
    
    // File input change
    fileInput.addEventListener('change', (e) => {
        handleFileSelect(e.target.files[0]);
    });
    
    // Drag and drop
    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('dragover');
    });
    
    uploadZone.addEventListener('dragleave', () => {
        uploadZone.classList.remove('dragover');
    });
    
    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('dragover');
        handleFileSelect(e.dataTransfer.files[0]);
    });
    
    // Remove image button
    document.getElementById('removeImage').addEventListener('click', (e) => {
        e.stopPropagation();
        resetUpload();
    });
    
    // Analyze button
    analyzeBtn.addEventListener('click', runAnalysis);
}

/**
 * Load model information from API
 */
async function loadModelInfo() {
    try {
        const response = await fetch('/api/model-info');
        const data = await response.json();
        
        if (data.status === 'ready') {
            console.log('Models loaded:', data);
        } else {
            showAlert('error', 'Models not loaded. Please refresh the page.');
        }
    } catch (error) {
        console.error('Error loading model info:', error);
    }
}

/**
 * Handle file selection
 */
function handleFileSelect(file) {
    if (!file) return;
    
    // Validate file type
    const validTypes = ['image/png', 'image/jpeg', 'image/jpg'];
    if (!validTypes.includes(file.type)) {
        showAlert('error', 'Please upload a PNG or JPEG image.');
        return;
    }
    
    // Validate file size (16MB max)
    if (file.size > 16 * 1024 * 1024) {
        showAlert('error', 'File size must be less than 16MB.');
        return;
    }
    
    currentFile = file;
    
    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        imagePreview.style.display = 'block';
        analyzeBtn.disabled = false;
        uploadZone.style.display = 'none';
    };
    reader.readAsDataURL(file);
    
    // Reset results
    resultsSection.style.display = 'none';
}

/**
 * Reset upload state
 */
function resetUpload() {
    currentFile = null;
    analysisResults = null;
    fileInput.value = '';
    imagePreview.style.display = 'none';
    uploadZone.style.display = 'block';
    analyzeBtn.disabled = true;
    resultsSection.style.display = 'none';
}

/**
 * Run analysis on uploaded image
 */
async function runAnalysis() {
    if (!currentFile) return;
    
    try {
        // Show loading
        loadingSection.style.display = 'block';
        resultsSection.style.display = 'none';
        analyzeBtn.disabled = true;
        
        // Upload file
        const formData = new FormData();
        formData.append('file', currentFile);
        
        const uploadResponse = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });
        
        if (!uploadResponse.ok) {
            const error = await uploadResponse.json();
            throw new Error(error.error || 'Upload failed');
        }
        
        const uploadData = await uploadResponse.json();
        
        // Run analysis
        const analysisResponse = await fetch('/api/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                filename: uploadData.filename
            })
        });
        
        if (!analysisResponse.ok) {
            const error = await analysisResponse.json();
            throw new Error(error.error || 'Analysis failed');
        }
        
        const analysisData = await analysisResponse.json();
        analysisResults = analysisData;
        
        // Display results
        displayResults(analysisData);
        
    } catch (error) {
        console.error('Analysis error:', error);
        showAlert('error', error.message);
    } finally {
        loadingSection.style.display = 'none';
        analyzeBtn.disabled = false;
    }
}

/**
 * Display analysis results
 */
function displayResults(data) {
    const results = data.results;
    
    // Stage 1 Results
    document.getElementById('hemorrhageStatus').textContent = 
        results.hemorrhage_detected ? 'Hemorrhage Detected' : 'No Hemorrhage';
    
    document.getElementById('hemorrhageStatus').className = 
        'result-value ' + (results.hemorrhage_detected ? 'positive' : 'negative');
    
    document.getElementById('detectionProbability').textContent = 
        results.detection_probability;
    
    const confidenceBadge = document.getElementById('confidenceBadge');
    confidenceBadge.textContent = results.detection_confidence;
    confidenceBadge.className = 'confidence-badge confidence-' + 
        results.detection_confidence.toLowerCase();
    
    // Stage 2 Results
    const subtypesContainer = document.getElementById('subtypesContainer');
    const subtypesList = document.getElementById('subtypesList');
    
    if (results.subtypes && results.subtypes.length > 0) {
        subtypesContainer.style.display = 'block';
        subtypesList.innerHTML = '';
        
        results.subtypes.forEach(subtype => {
            const item = document.createElement('div');
            item.className = 'subtype-item';
            item.innerHTML = `
                <span class="subtype-name">📍 ${subtype.name}</span>
                <span class="subtype-probability">${subtype.probability}</span>
            `;
            subtypesList.appendChild(item);
        });
    } else {
        subtypesContainer.style.display = 'none';
    }
    
    // Grad-CAM Visualizations
    const gradcamContainer = document.getElementById('gradcamContainer');
    const gradcamGrid = document.getElementById('gradcamGrid');
    
    if (data.gradcam && data.gradcam.length > 0) {
        gradcamContainer.style.display = 'block';
        gradcamGrid.innerHTML = '';
        
        data.gradcam.forEach(item => {
            const card = document.createElement('div');
            card.className = 'gradcam-card';
            card.innerHTML = `
                <img src="${item.image_url}" alt="${item.subtype} Grad-CAM" 
                     class="gradcam-image" onclick="openImageModal('${item.image_url}')">
                <div class="gradcam-label">
                    ${item.subtype} (${(item.probability * 100).toFixed(2)}%)
                </div>
            `;
            gradcamGrid.appendChild(card);
        });
    } else {
        gradcamContainer.style.display = 'none';
    }
    
    // Show results section
    resultsSection.style.display = 'block';
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

/**
 * Open image in modal (for zooming)
 */
function openImageModal(imageUrl) {
    // Create modal
    const modal = document.createElement('div');
    modal.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.9);
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 9999;
        cursor: zoom-out;
    `;
    
    const img = document.createElement('img');
    img.src = imageUrl;
    img.style.cssText = `
        max-width: 90%;
        max-height: 90%;
        border-radius: 10px;
        box-shadow: 0 0 50px rgba(0, 212, 255, 0.5);
    `;
    
    modal.appendChild(img);
    document.body.appendChild(modal);
    
    modal.addEventListener('click', () => {
        document.body.removeChild(modal);
    });
}

/**
 * Show alert message
 */
function showAlert(type, message) {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type}`;
    alertDiv.innerHTML = `
        <span>${type === 'error' ? '❌' : type === 'success' ? '✓' : 'ℹ️'}</span>
        <span>${message}</span>
    `;
    
    const container = document.querySelector('.container');
    container.insertBefore(alertDiv, container.firstChild);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        alertDiv.remove();
    }, 5000);
}

/**
 * Format bytes to human-readable size
 */
function formatBytes(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

/**
 * Download results as JSON
 */
function downloadResults() {
    if (!analysisResults) return;
    
    const dataStr = JSON.stringify(analysisResults, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    
    const link = document.createElement('a');
    link.href = url;
    link.download = `hemorrhage_analysis_${Date.now()}.json`;
    link.click();
    
    URL.revokeObjectURL(url);
}
