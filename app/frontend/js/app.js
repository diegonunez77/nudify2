// Global variables to store the current images
let currentOriginalUrl = '';
let currentResultUrl = '';
let processingHistory = [];
let isProcessing = false;

// Initialize the page
document.addEventListener('DOMContentLoaded', function() {
    // Set up image upload preview
    document.getElementById('imageUpload').addEventListener('change', handleImageUpload);
    
    // Initialize with the default URL preview
    previewImageFromUrl(document.getElementById('imageUrl').value.trim());
    
    // Load history from localStorage if available
    loadHistory();
    
    // Set up drag and drop functionality
    setupDragAndDrop();
    
    // Set up process image button
    document.getElementById('processButton').addEventListener('click', function() {
        if (window.auth && !window.auth.isAuthenticated()) {
            showAlert('Please sign in to process images', 'danger');
            return;
        }
        processImage();
    });
    
    // Watch for URL input changes to update preview
    document.getElementById('imageUrl').addEventListener('input', debounce(function() {
        const url = this.value.trim();
        if (url) {
            previewImageFromUrl(url);
        }
    }, 500));

    // Add keyboard shortcut for processing (Ctrl+Enter)
    document.addEventListener('keydown', function(e) {
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter' && !isProcessing) {
            processImage();
        }
    });

    // Check backend health on load
    checkBackendHealth();
    
    // Watch for URL input changes to update preview
    document.getElementById('imageUrl').addEventListener('blur', function() {
        const url = this.value.trim();
        if (url) {
            previewImageFromUrl(url);
        }
    });
});

// Setup drag and drop functionality
function setupDragAndDrop() {
    const dropArea = document.querySelector('.upload-area');
    
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false);
    });
    
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, highlight, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, unhighlight, false);
    });
    
    function highlight() {
        dropArea.style.borderColor = 'var(--primary)';
        dropArea.style.backgroundColor = 'rgba(255, 43, 115, 0.05)';
    }
    
    function unhighlight() {
        dropArea.style.borderColor = 'rgba(255,255,255,0.1)';
        dropArea.style.backgroundColor = 'rgba(61,61,61,0.3)';
    }
    
    dropArea.addEventListener('drop', handleDrop, false);
    
    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length) {
            document.getElementById('imageUpload').files = files;
            handleImageUpload({target: {files: files}});
        }
    }
}

// Debounce function to limit how often a function can be called
function debounce(func, wait) {
    let timeout;
    return function() {
        const context = this;
        const args = arguments;
        clearTimeout(timeout);
        timeout = setTimeout(() => {
            func.apply(context, args);
        }, wait);
    };
}

// Handle image upload and preview
function handleImageUpload(event) {
    const file = event.target.files[0];
    if (file) {
        // Validate file type
        const validTypes = ['image/jpeg', 'image/png', 'image/webp', 'image/gif'];
        if (!validTypes.includes(file.type)) {
            showAlert('Please upload a valid image file (JPEG, PNG, WebP or GIF)', 'danger');
            return;
        }
        
        // Validate file size (max 10MB)
        const maxSize = 10 * 1024 * 1024; // 10MB in bytes
        if (file.size > maxSize) {
            showAlert('Image file is too large. Maximum size is 10MB', 'danger');
            return;
        }
        
        const reader = new FileReader();
        reader.onload = function(e) {
            const imageUrl = e.target.result;
            document.getElementById('imagePreview').src = imageUrl;
            document.getElementById('imagePreviewContainer').classList.remove('hidden');
            // Clear the URL input since we're using an uploaded file
            document.getElementById('imageUrl').value = '';
            
            // Update image metadata
            const img = new Image();
            img.onload = function() {
                const width = this.width;
                const height = this.height;
                const aspectRatio = (width / height).toFixed(2);
                const fileSize = (file.size / 1024 / 1024).toFixed(2);
                
                document.getElementById('imageMetadata').innerHTML = `
                    <span class="badge bg-dark me-2">${width}×${height}px</span>
                    <span class="badge bg-dark me-2">${file.type.split('/')[1].toUpperCase()}</span>
                    <span class="badge bg-dark me-2">${fileSize}MB</span>
                    <span class="badge bg-dark">Ratio ${aspectRatio}</span>
                `;
            };
            img.src = imageUrl;
        };
        reader.readAsDataURL(file);
    }
}

// Preview image from URL
function previewImageFromUrl(url) {
    if (url) {
        const preview = document.getElementById('imagePreview');
        preview.src = url;
        document.getElementById('imagePreviewContainer').classList.remove('hidden');
        // Clear any file upload
        document.getElementById('imageUpload').value = '';
        
        // Update image metadata
        updateImageMetadata(url);
    }
}

// Update image metadata
function updateImageMetadata(url) {
    const metadataElement = document.getElementById('imageMetadata');
    metadataElement.innerHTML = '<span class="spinner-border spinner-border-sm" role="status"></span> Loading image info...';
    
    // Create a new image to get dimensions
    const img = new Image();
    img.onload = function() {
        const width = this.width;
        const height = this.height;
        const aspectRatio = (width / height).toFixed(2);
        const fileType = url.split('.').pop().toUpperCase();
        
        metadataElement.innerHTML = `
            <span class="badge bg-dark me-2">${width}×${height}px</span>
            <span class="badge bg-dark me-2">${fileType}</span>
            <span class="badge bg-dark">Ratio ${aspectRatio}</span>
        `;
    };
    
    img.onerror = function() {
        metadataElement.innerHTML = '<span class="badge bg-danger">Invalid image or URL</span>';
    };
    
    img.src = url;
}

// Set preset prompt
function setPresetPrompt(type) {
    let promptText = '';
    switch(type) {
        case 'bikini':
            promptText = 'female breasts, bikini, bikini top, bikini bottom';
            break;
        case 'swimsuit':
            promptText = 'female body, one-piece swimsuit, revealing swimwear';
            break;
        case 'lingerie':
            promptText = 'female body, lingerie, bra, underwear, lace';
            break;
        case 'nude':
            promptText = 'female nude, naked body, exposed breasts, no clothing';
            break;
        case 'sheer':
            promptText = 'female body, sheer clothing, see-through fabric, transparent dress';
            break;
        case 'fantasy':
            promptText = 'fantasy outfit, revealing magical costume, enchanted lingerie';
            break;
    }
    document.getElementById('prompt').value = promptText;
    
    // Highlight the selected preset button
    document.querySelectorAll('.preset-btn').forEach(btn => {
        btn.classList.remove('active');
        if (btn.innerText.toLowerCase().includes(type.toLowerCase())) {
            btn.classList.add('active');
        }
    });
}

// Check backend health
async function checkBackendHealth() {
    try {
        const apiUrl = window.location.hostname === 'localhost' ? 'http://localhost:5000/health' : '/health';
        const response = await fetch(apiUrl);
        
        if (response.ok) {
            console.log('Backend is healthy');
        } else {
            showAlert('Warning: Backend service may not be available. Some features might not work properly.', 'warning');
        }
    } catch (error) {
        console.error('Backend health check failed:', error);
        showAlert('Warning: Unable to connect to the backend service. Please try again later.', 'warning');
    }
}

// Process the image
async function processImage() {
    // Check if user is authenticated
    if (window.auth && !window.auth.isAuthenticated()) {
        showAlert('Please sign in to process images', 'danger');
        return;
    }
    
    // Get input values
    const imageUrl = document.getElementById('imageUrl').value.trim();
    const imageFile = document.getElementById('imageUpload').files[0];
    const prompt = document.getElementById('prompt').value.trim();
    
    // Validate inputs
    if (!imageUrl && !imageFile) {
        showAlert('Please provide an image URL or upload an image', 'danger');
        return;
    }
    
    if (!prompt) {
        showAlert('Please enter a transformation prompt', 'danger');
        return;
    }
    
    // Show loading state
    setProcessingState(true);
    isProcessing = true;
    
    try {
        let requestData;
        let apiUrl;
        
        if (imageUrl) {
            // Using URL
            requestData = { image_url: imageUrl, prompt: prompt };
            currentOriginalUrl = imageUrl;
            apiUrl = window.location.hostname === 'localhost' ? 'http://localhost:5000/process-image' : '/process-image';
        } else if (imageFile) {
            // Using file upload
            const formData = new FormData();
            formData.append('file', imageFile);
            formData.append('prompt', prompt);
            
            // Create a temporary URL for the uploaded file
            currentOriginalUrl = URL.createObjectURL(imageFile);
            
            // Use the upload endpoint
            apiUrl = window.location.hostname === 'localhost' ? 'http://localhost:5000/upload' : '/upload';
            
            // Make the request
            try {
                const uploadResponse = await fetch(apiUrl, {
                    method: 'POST',
                    body: formData
                });
                
                if (uploadResponse.ok) {
                    const responseData = await uploadResponse.json();
                    if (responseData.error) {
                        showAlert(`Error: ${responseData.error}`, 'danger');
                        setProcessingState(false);
                        isProcessing = false;
                        return;
                    }
                    
                    // If we get here, file upload is implemented
                    const processResponse = await uploadResponse.blob();
                    handleProcessingResult(processResponse, prompt);
                    return;
                } else {
                    // If upload endpoint returns 501 Not Implemented
                    if (uploadResponse.status === 501) {
                        // Fall back to URL-based API with a warning
                        showAlert('File upload is not yet fully implemented. Using image URL instead.', 'warning');
                        // Continue with URL-based processing below
                    } else {
                        throw new Error(`Upload failed with status ${uploadResponse.status}`);
                    }
                }
            } catch (uploadError) {
                console.error('Upload error:', uploadError);
                showAlert('File upload failed. Please try using an image URL instead.', 'warning');
                setProcessingState(false);
                isProcessing = false;
                return;
            }
        }
        
        // Use relative URL to ensure it works in any environment
        const response = await fetch(apiUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
        });

        if (response.ok) {
            const blob = await response.blob();
            handleProcessingResult(blob, prompt);
        } else {
            // Handle error response
            try {
                const errorData = await response.json();
                showAlert(`Error: ${errorData.error || errorData.message || 'Failed to transform image'}`, 'danger');
            } catch (e) {
                showAlert(`Error: Failed to transform image (Status ${response.status})`, 'danger');
            }
        }
    } catch (error) {
        console.error('Processing error:', error);
        showAlert(`Network error: ${error.message}`, 'danger');
    } finally {
        setProcessingState(false);
        isProcessing = false;
    }
}

// Handle the result of image processing
function handleProcessingResult(blob, prompt) {
    const resultUrl = URL.createObjectURL(blob);
    
    // Display the result
    document.getElementById('originalImage').src = currentOriginalUrl;
    document.getElementById('resultImage').src = resultUrl;
    document.getElementById('resultContainer').classList.remove('hidden');
    currentResultUrl = resultUrl;
    
    // Scroll to results
    document.getElementById('resultContainer').scrollIntoView({ behavior: 'smooth' });
    
    // Add to history
    addToHistory({
        originalUrl: currentOriginalUrl,
        resultUrl: resultUrl,
        prompt: prompt,
        timestamp: new Date().toISOString()
    });
    
    showAlert('Image transformed successfully!', 'success');
    
    // Update credit display if authenticated
    if (window.auth && window.auth.isAuthenticated()) {
        window.auth.updateCreditDisplay();
    }
}

// Set processing state
function setProcessingState(isProcessing) {
    const processBtn = document.getElementById('processBtn');
    const processBtnText = document.getElementById('processBtnText');
    const processBtnSpinner = document.getElementById('processBtnSpinner');
    
    processBtn.disabled = isProcessing;
    
    if (isProcessing) {
        processBtnText.textContent = 'Processing...';
        processBtnSpinner.classList.remove('hidden');
    } else {
        processBtnText.textContent = 'Transform Image';
        processBtnSpinner.classList.add('hidden');
    }
}

// Show alert message
function showAlert(message, type) {
    const alertContainer = document.getElementById('alertContainer');
    const alert = document.createElement('div');
    alert.className = `alert alert-${type} alert-dismissible fade show`;
    alert.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;
    
    alertContainer.innerHTML = '';
    alertContainer.appendChild(alert);
    
    // Auto dismiss after 5 seconds
    setTimeout(() => {
        alert.classList.remove('show');
        setTimeout(() => alertContainer.removeChild(alert), 150);
    }, 5000);
}

// Add to history
function addToHistory(item) {
    processingHistory.unshift(item); // Add to the beginning
    if (processingHistory.length > 10) {
        processingHistory.pop(); // Keep only the last 10 items
    }
    
    // Save to localStorage
    localStorage.setItem('nudify2History', JSON.stringify(processingHistory));
    
    // Update the UI
    updateHistoryUI();
}

// Load history from localStorage
function loadHistory() {
    const savedHistory = localStorage.getItem('nudify2History');
    if (savedHistory) {
        processingHistory = JSON.parse(savedHistory);
        updateHistoryUI();
    }
}

// Update history UI
function updateHistoryUI() {
    const historyList = document.getElementById('historyList');
    historyList.innerHTML = '';
    
    if (processingHistory.length === 0) {
        const emptyItem = document.createElement('li');
        emptyItem.className = 'list-group-item text-center text-muted';
        emptyItem.textContent = 'No history yet';
        historyList.appendChild(emptyItem);
        return;
    }
    
    processingHistory.forEach((item, index) => {
        const historyItem = document.createElement('li');
        historyItem.className = 'list-group-item history-item';
        
        // Format the date
        const date = new Date(item.timestamp);
        const formattedDate = date.toLocaleString();
        
        historyItem.innerHTML = `
            <div class="d-flex justify-content-between align-items-center">
                <div>
                    <small class="text-muted">${formattedDate}</small>
                    <div class="text-truncate" style="max-width: 200px;">${item.prompt}</div>
                </div>
                <button class="btn btn-sm btn-outline-primary">View</button>
            </div>
        `;
        
        historyItem.querySelector('button').addEventListener('click', () => {
            loadHistoryItem(index);
        });
        
        historyList.appendChild(historyItem);
    });
}

// Load history item
function loadHistoryItem(index) {
    const item = processingHistory[index];
    if (item) {
        // Set the form values
        document.getElementById('imageUrl').value = item.originalUrl;
        document.getElementById('prompt').value = item.prompt;
        
        // Preview the original image
        previewImageFromUrl(item.originalUrl);
        
        // Display the result
        document.getElementById('originalImage').src = item.originalUrl;
        document.getElementById('resultImage').src = item.resultUrl;
        document.getElementById('resultContainer').classList.remove('hidden');
        
        // Update current URLs
        currentOriginalUrl = item.originalUrl;
        currentResultUrl = item.resultUrl;
    }
}

// Download the result image
function downloadImage() {
    if (currentResultUrl) {
        const link = document.createElement('a');
        link.href = currentResultUrl;
        link.download = 'nudify2_result.jpg';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }
}

// Share the result (placeholder for future implementation)
function shareResult() {
    if (currentResultUrl) {
        // Check if Web Share API is available
        if (navigator.share) {
            // Create a blob from the image URL
            fetch(currentResultUrl)
                .then(res => res.blob())
                .then(blob => {
                    const file = new File([blob], 'nudify2-result.jpg', { type: 'image/jpeg' });
                    navigator.share({
                        title: 'My Nudify2 Transformation',
                        text: 'Check out this AI-transformed image from Nudify2!',
                        files: [file]
                    }).then(() => {
                        console.log('Share successful');
                    }).catch((error) => {
                        console.error('Error sharing:', error);
                        showFallbackShare();
                    });
                });
        } else {
            showFallbackShare();
        }
    } else {
        showAlert('No result to share. Transform an image first.', 'warning');
    }
}

// Fallback share method
function showFallbackShare() {
    showAlert('Direct sharing is not supported in your browser. You can download the image and share it manually.', 'info');
}
