document.addEventListener('DOMContentLoaded', () => {
    // Elements
    const fileInput = document.getElementById('fileInput');
    const dropZone = document.getElementById('dropZone');

    // View Elements
    const imageViewer = document.getElementById('imageViewer');
    const displayedImage = document.getElementById('displayedImage');
    const placeholderContent = imageViewer.querySelector('.placeholder-content');
    const imageFileName = document.getElementById('imageFileName');
    const gradcamOverlay = document.getElementById('gradcamOverlay');

    // Result Elements
    const uploadSection = document.getElementById('uploadSection');
    const resultsList = document.getElementById('resultsList');
    const statusCard = document.getElementById('statusCard');
    const statusText = document.getElementById('statusText');
    const statusScore = document.getElementById('statusScore');
    const subtypesList = document.getElementById('subtypesList');

    // --- Event Listeners ---

    // Click to upload
    dropZone.addEventListener('click', () => fileInput.click());

    // Drag & Drop
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = 'var(--accent-blue)';
        dropZone.style.backgroundColor = '#EFF6FF';
    });
    dropZone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = '#CBD5E1';
        dropZone.style.backgroundColor = '#F8FAFC';
    });
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = '#CBD5E1';
        dropZone.style.backgroundColor = '#F8FAFC';

        if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
    });

    // File Input
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) handleFile(e.target.files[0]);
    });


    // --- Core Logic ---

    async function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please upload an image file (PNG, JPG).');
            return;
        }

        // 1. Show Preview immediately
        showImagePreview(file);
        imageFileName.textContent = file.name;

        // 2. Show loading state in Results
        setLoadingState(true);

        const formData = new FormData();
        formData.append('file', file);

        try {
            // Upload
            const upRes = await fetch('/api/upload', { method: 'POST', body: formData });
            if (!upRes.ok) throw new Error('Upload failed');
            const upData = await upRes.json();

            // Analyze
            const anRes = await fetch('/api/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ filename: upData.filename })
            });

            if (!anRes.ok) throw new Error('Analysis failed');
            const data = await anRes.json();

            displayResults(data);

        } catch (error) {
            console.error(error);
            alert(`Error: ${error.message}`);
            setLoadingState(false);
        }
    }

    function showImagePreview(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            displayedImage.src = e.target.result;
            displayedImage.style.display = 'block';
            placeholderContent.style.display = 'none';
        }
        reader.readAsDataURL(file);
    }

    function setLoadingState(isLoading) {
        if (isLoading) {
            uploadSection.style.display = 'none';
            resultsList.style.display = 'flex'; // Use flex for column layout

            // Temporary loading HTML
            resultsList.innerHTML = `
                <div style="display: flex; flex-direction: column; align-items: center; padding: 2rem;">
                    <div class="spinner" style="border: 3px solid #f3f3f3; border-top: 3px solid var(--accent-blue); border-radius: 50%; width: 40px; height: 40px; animation: spin 1s linear infinite; margin-bottom: 1rem;"></div>
                    <p>Analyzing Scan...</p>
                </div>
                <style>@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }</style>
            `;
        }
    }

    function displayResults(data) {
        // Restore Results List HTML structure (since we overwrote it for loading)
        // We will rebuild it cleanly
        resultsList.innerHTML = '';
        resultsList.style.display = 'flex';

        const isHemorrhage = data.results.hemorrhage_detected;
        const confidence = data.results.detection_probability;

        // 1. Status Card
        const statusEl = document.createElement('div');
        statusEl.className = `result-card ${isHemorrhage ? 'danger' : 'success'}`;
        statusEl.innerHTML = `
            <div class="card-content">
                <div class="card-label">Hemorrhage Detection</div>
                <div class="card-value-row">
                    <span class="status-indicator"></span>
                    <span class="status-text">${isHemorrhage ? 'Hemorrhage Detected' : 'No Hemorrhage'}</span>
                </div>
            </div>
            <div class="card-score">
                <span class="score-label">Confidence</span>
                <span class="score-value">${confidence}</span>
            </div>
        `;
        resultsList.appendChild(statusEl);


        // 2. Subtypes (if any)
        if (data.results.subtypes && data.results.subtypes.length > 0) {

            data.results.subtypes.forEach(sub => {
                const subCard = document.createElement('div');
                subCard.className = 'result-card warning'; // Use yellow/warning for subtypes
                // Or use specific colored cards based on severity if needed

                subCard.innerHTML = `
                    <div class="card-content">
                        <div class="card-label">Subtype Identified</div>
                        <div class="card-value-row">
                            <span class="status-indicator" style="color: #D97706;"></span>
                            <span class="status-text" style="color: #92400E;">${sub.name}</span>
                        </div>
                    </div>
                    <div class="card-score">
                        <span class="score-label">Probability</span>
                        <span class="score-value" style="color: #92400E;">${sub.probability}</span>
                    </div>
                `;
                resultsList.appendChild(subCard);
            });
        } else if (isHemorrhage) {
            const subCard = document.createElement('div');
            subCard.className = 'result-card warning';
            subCard.innerHTML = `
                    <div class="card-content">
                         <div class="card-label">Subtype Analysis</div>
                         <div class="card-value-row">
                             <span class="status-text" style="color: #92400E; font-size: 0.9rem;">No specific subtype confirmed with high confidence.</span>
                         </div>
                    </div>
                `;
            resultsList.appendChild(subCard);
        }

        // 3. Grad-CAM Update (Optional: overlay or replace image)
        if (data.gradcam && data.gradcam.length > 0) {
            // Update the main image viewer with the Grad-CAM version
            displayedImage.src = data.gradcam[0].image_url;
        }

        // Add "Analyze Another" button
        const btnContainer = document.createElement('div');
        btnContainer.style.marginTop = '2rem';
        btnContainer.style.textAlign = 'center';
        btnContainer.innerHTML = `
            <button onclick="location.reload()" style="padding: 0.75rem 1.5rem; background: #fff; border: 1px solid #ddd; border-radius: 8px; cursor: pointer; font-weight: 500;">
                Analyze Another Scan
            </button>
        `;
        resultsList.appendChild(btnContainer);
    }
});
