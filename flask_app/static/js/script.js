const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const resultsSection = document.getElementById('resultsSection');
const loadingSection = document.getElementById('loadingSection');
const uploadSection = document.querySelector('.upload-section');

// Drag and drop handlers
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('drag-over');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('drag-over');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('drag-over');

    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
});

// File input change handler
fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFile(e.target.files[0]);
    }
});

// Click to upload
uploadArea.addEventListener('click', () => {
    fileInput.click();
});

function handleFile(file) {
    // Validate file type
    const validTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/gif', 'image/bmp'];
    if (!validTypes.includes(file.type)) {
        alert('Please upload a valid image file (PNG, JPG, JPEG, GIF, or BMP)');
        return;
    }

    // Validate file size (16MB max)
    if (file.size > 16 * 1024 * 1024) {
        alert('File size must be less than 16MB');
        return;
    }

    uploadImage(file);
}

function uploadImage(file) {
    // Show loading, hide upload and results
    uploadSection.style.display = 'none';
    resultsSection.style.display = 'none';
    loadingSection.style.display = 'block';

    const formData = new FormData();
    formData.append('file', file);

    fetch('/detect', {
        method: 'POST',
        body: formData
    })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                displayResults(data);
            } else {
                alert('Error: ' + (data.error || 'Failed to process image'));
                resetUpload();
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred while processing the image');
            resetUpload();
        });
}

function displayResults(data) {
    // Hide loading, show results
    loadingSection.style.display = 'none';
    resultsSection.style.display = 'block';

    // Display original image
    const originalImage = document.getElementById('originalImage');
    originalImage.src = data.original_image;

    // Display result image
    const resultImage = document.getElementById('resultImage');
    resultImage.src = data.result_image;

    // Display detections
    const detectionsList = document.getElementById('detectionsList');
    detectionsList.innerHTML = '';

    if (data.detections && data.detections.length > 0) {
        data.detections.forEach((detection, index) => {
            const detectionItem = document.createElement('div');
            detectionItem.className = 'detection-item';
            detectionItem.style.animationDelay = `${index * 0.1}s`;

            const confidence = (detection.confidence * 100).toFixed(2);

            detectionItem.innerHTML = `
                <div>
                    <strong>${detection.class}</strong> - Confidence: ${confidence}%
                </div>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: ${confidence}%"></div>
                </div>
            `;

            detectionsList.appendChild(detectionItem);
        });
    } else {
        detectionsList.innerHTML = '<p style="color: var(--text-secondary);">No hemorrhages detected in this scan.</p>';
    }
}

function resetUpload() {
    // Reset file input
    fileInput.value = '';

    // Show upload section, hide others
    uploadSection.style.display = 'block';
    resultsSection.style.display = 'none';
    loadingSection.style.display = 'none';
}
