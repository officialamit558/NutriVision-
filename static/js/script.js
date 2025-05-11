document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const dropArea = document.getElementById('drop-area');
    const fileInput = document.getElementById('file-input');
    const previewContainer = document.getElementById('preview-container');
    const previewImage = document.getElementById('preview-image');
    const uploadButton = document.getElementById('upload-button');
    const removeButton = document.getElementById('remove-image');
    const resultsContainer = document.getElementById('results-container');
    const loader = document.getElementById('loader');
    const newUploadButton = document.getElementById('new-upload-button');
    const resultImage = document.getElementById('result-image');
    const foodName = document.getElementById('food-name');
    const confidenceLevel = document.getElementById('confidence-level');
    const confidenceValue = document.getElementById('confidence-value');
    
    // Camera elements
    const cameraButton = document.getElementById('camera-button');
    const cameraModal = document.getElementById('camera-modal');
    const closeCamera = document.getElementById('close-camera');
    const cameraView = document.getElementById('camera-view');
    const takePhotoButton = document.getElementById('take-photo');
    const switchCameraButton = document.getElementById('switch-camera');
    const cameraCanvas = document.getElementById('camera-canvas');
    
    // Nutrition elements
    const weight = document.getElementById('weight');
    const calories = document.getElementById('calories');
    const protein = document.getElementById('protein');
    const carbohydrates = document.getElementById('carbohydrates');
    const fats = document.getElementById('fats');
    const fiber = document.getElementById('fiber');
    const sugars = document.getElementById('sugars');
    const sodium = document.getElementById('sodium');
    
    // Weight selector element
    const weightSelector = document.getElementById('weight-selector');
    const servingSizeSelect = document.getElementById('serving-size');
    
    // Camera stream variables
    let stream = null;
    let facingMode = 'environment'; // Start with back camera
    
    // Event listeners for drag and drop
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
        document.querySelector('.upload-prompt').classList.add('highlight');
    }
    
    function unhighlight() {
        document.querySelector('.upload-prompt').classList.remove('highlight');
    }
    
    // Handle dropped files
    dropArea.addEventListener('drop', handleDrop, false);
    
    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        
        if (files.length > 0) {
            fileInput.files = files;
            handleFiles(files);
        }
    }
    
    // Handle file selection via input
    fileInput.addEventListener('change', function() {
        handleFiles(this.files);
    });
    
    // File handling
    function handleFiles(files) {
        if (files.length > 0) {
            const file = files[0];
            if (!file.type.match('image.*')) {
                alert('Please select an image file.');
                return;
            }
            
            const reader = new FileReader();
            reader.onload = function(e) {
                previewImage.src = e.target.result;
                previewContainer.hidden = false;
                document.querySelector('.upload-prompt').hidden = true;
                uploadButton.disabled = false;
            };
            reader.readAsDataURL(file);
        }
    }
    
    // Remove image button
    removeButton.addEventListener('click', function() {
        previewContainer.hidden = true;
        document.querySelector('.upload-prompt').hidden = false;
        uploadButton.disabled = true;
        fileInput.value = null;
    });
    
    // Camera button
    cameraButton.addEventListener('click', function() {
        openCamera();
    });
    
    // Close camera button
    closeCamera.addEventListener('click', function() {
        closeAndStopCamera();
    });
    
    // Take photo button
    takePhotoButton.addEventListener('click', function() {
        capturePhoto();
    });
    
    // Switch camera button
    switchCameraButton.addEventListener('click', function() {
        switchCamera();
    });
    
    // Open and initialize camera
    function openCamera() {
        cameraModal.style.display = 'block';
        startCamera();
    }
    
    // Start the camera with current facingMode
    function startCamera() {
        // Stop any existing stream first
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
        }
        
        // Setup camera options
        const constraints = {
            video: {
                facingMode: facingMode,
                width: { ideal: 1280 },
                height: { ideal: 720 }
            },
            audio: false
        };
        
        // Get access to the camera
        navigator.mediaDevices.getUserMedia(constraints)
            .then(function(mediaStream) {
                stream = mediaStream;
                cameraView.srcObject = mediaStream;
                cameraView.play();
            })
            .catch(function(err) {
                console.error("Camera error: ", err);
                alert("Error accessing the camera: " + err.message);
            });
    }
    
    // Switch between front and back cameras
    function switchCamera() {
        facingMode = facingMode === 'environment' ? 'user' : 'environment';
        startCamera();
    }
    
    // Capture a photo from the video stream
    function capturePhoto() {
        if (!stream) return;
        
        // Configure the canvas to match the video dimensions
        const width = cameraView.videoWidth;
        const height = cameraView.videoHeight;
        cameraCanvas.width = width;
        cameraCanvas.height = height;
        
        // Draw the current video frame to the canvas
        const context = cameraCanvas.getContext('2d');
        context.drawImage(cameraView, 0, 0, width, height);
        
        // Convert the canvas to a data URL and set it as the preview image
        const imageData = cameraCanvas.toDataURL('image/png');
        previewImage.src = imageData;
        
        // Show the preview and enable the upload button
        previewContainer.hidden = false;
        document.querySelector('.upload-prompt').hidden = true;
        uploadButton.disabled = false;
        
        // Close the camera
        closeAndStopCamera();
        
        // Convert data URL to Blob for form submission
        dataURLtoBlob(imageData).then(blob => {
            // Create a File object from the Blob
            const file = new File([blob], "camera-photo.png", { type: "image/png" });
            
            // Create a FileList-like object to mimic the file input
            const dataTransfer = new DataTransfer();
            dataTransfer.items.add(file);
            fileInput.files = dataTransfer.files;
        });
    }
    
    // Convert a data URL to a Blob
    function dataURLtoBlob(dataURL) {
        return new Promise((resolve) => {
            const binary = atob(dataURL.split(',')[1]);
            const array = [];
            for (let i = 0; i < binary.length; i++) {
                array.push(binary.charCodeAt(i));
            }
            resolve(new Blob([new Uint8Array(array)], { type: 'image/png' }));
        });
    }
    
    // Close and stop the camera
    function closeAndStopCamera() {
        cameraModal.style.display = 'none';
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            stream = null;
        }
    }
    
    // Upload and analyze button
    uploadButton.addEventListener('click', function() {
        if (!fileInput.files.length) return;
        
        const formData = new FormData();
        formData.append('file', fileInput.files[0]);
        
        // Show loader, hide other containers
        loader.hidden = false;
        resultsContainer.hidden = true;
        dropArea.hidden = true;
        
        // Send the image to server for analysis
        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Server error: ' + response.status);
            }
            return response.json();
        })
        .then(data => {
            // Hide loader
            loader.hidden = true;
            
            if (data.error) {
                // Handle the non-food error or other errors
                if (data.is_food === false) {
                    // This is a non-food item, show special error
                    showNonFoodError(data);
                } else {
                    // This is a different type of error
                    throw new Error(data.error);
                }
            } else {
                // Display results for valid food item
                displayResults(data);
                // Show results container
                resultsContainer.hidden = false;
            }
        })
        .catch(error => {
            alert('Error: ' + error.message);
            loader.hidden = true;
            dropArea.hidden = false;
        });
    });
    
    // Function to show non-food error with additional information
    function showNonFoodError(data) {
        // Create or get error message container
        let errorContainer = document.getElementById('error-container');
        if (!errorContainer) {
            errorContainer = document.createElement('div');
            errorContainer.id = 'error-container';
            errorContainer.className = 'error-container';
            document.querySelector('main').appendChild(errorContainer);
        }
        
        // Set the content of the error message
        errorContainer.innerHTML = `
            <div class="error-header">
                <h3>Non-Food Item Detected</h3>
                <button id="close-error"><i class="fas fa-times"></i></button>
            </div>
            <div class="error-content">
                <img src="${data.image_path}" alt="Uploaded Image" class="error-image">
                <div class="error-message">
                    <p><strong>Detected:</strong> ${data.class}</p>
                    <p><strong>Confidence:</strong> ${data.confidence.toFixed(1)}%</p>
                    <p class="error-text">${data.error}</p>
                    <button id="try-again-button">Try Again with Different Image</button>
                </div>
            </div>
        `;
        
        // Show the error container
        errorContainer.style.display = 'block';
        
        // Add event listener to the close button
        document.getElementById('close-error').addEventListener('click', function() {
            errorContainer.style.display = 'none';
            dropArea.hidden = false;
        });
        
        // Add event listener to the try again button
        document.getElementById('try-again-button').addEventListener('click', function() {
            errorContainer.style.display = 'none';
            dropArea.hidden = false;
            previewContainer.hidden = true;
            document.querySelector('.upload-prompt').hidden = false;
            uploadButton.disabled = true;
            fileInput.value = null;
        });
    }
    
    // New upload button
    newUploadButton.addEventListener('click', function() {
        resultsContainer.hidden = true;
        dropArea.hidden = false;
        previewContainer.hidden = true;
        document.querySelector('.upload-prompt').hidden = false;
        uploadButton.disabled = true;
        fileInput.value = null;
        
        // Hide error container if it exists
        const errorContainer = document.getElementById('error-container');
        if (errorContainer) {
            errorContainer.style.display = 'none';
        }
    });
    
    // Display results function
    function displayResults(data) {
        // Set the result image
        resultImage.src = data.image_path;
        
        // Set the food name
        foodName.textContent = data.class;
        
        // Set the confidence level
        confidenceLevel.style.width = data.confidence + '%';
        confidenceValue.textContent = data.confidence.toFixed(1);
        
        // Set the color based on confidence
        if (data.confidence > 80) {
            confidenceLevel.style.backgroundColor = '#4caf50';
        } else if (data.confidence > 60) {
            confidenceLevel.style.backgroundColor = '#ff9800';
        } else {
            confidenceLevel.style.backgroundColor = '#f44336';
        }
        
        // Clear any existing options
        servingSizeSelect.innerHTML = '';
        
        // Check if we have nutrition options
        if (data.nutrition_options && data.nutrition_options.length > 0) {
            // Make the weight selector visible
            weightSelector.style.display = 'flex';
            
            // Add options for each serving size
            data.nutrition_options.forEach((nutrition, index) => {
                const option = document.createElement('option');
                option.value = index;
                option.textContent = `${nutrition.weight}g`;
                servingSizeSelect.appendChild(option);
            });
            
            // Show nutrition for the first option by default
            updateNutritionInfo(data.nutrition_options[0]);
            
            // Add event listener for changing serving size
            servingSizeSelect.onchange = function() {
                const selectedIndex = parseInt(this.value);
                updateNutritionInfo(data.nutrition_options[selectedIndex]);
            };
        } else {
            // Hide the weight selector if no nutrition data
            weightSelector.style.display = 'none';
            
            // Set all nutrition values to N/A
            weight.textContent = 'N/A';
            calories.textContent = 'N/A';
            protein.textContent = 'N/A';
            carbohydrates.textContent = 'N/A';
            fats.textContent = 'N/A';
            fiber.textContent = 'N/A';
            sugars.textContent = 'N/A';
            sodium.textContent = 'N/A';
        }
    }
    
    // Function to update nutrition information based on selected option
    function updateNutritionInfo(nutrition) {
        weight.textContent = nutrition.weight + 'g';
        calories.textContent = nutrition.calories + ' kcal';
        protein.textContent = nutrition.protein + 'g';
        carbohydrates.textContent = nutrition.carbohydrates + 'g';
        fats.textContent = nutrition.fats + 'g';
        fiber.textContent = nutrition.fiber + 'g';
        sugars.textContent = nutrition.sugars + 'g';
        sodium.textContent = nutrition.sodium + 'mg';
    }
    
    // Close camera when the modal is clicked outside the content
    window.onclick = function(event) {
        if (event.target === cameraModal) {
            closeAndStopCamera();
        }
        
        // Close error container if clicked outside
        const errorContainer = document.getElementById('error-container');
        if (errorContainer && event.target === errorContainer) {
            errorContainer.style.display = 'none';
            dropArea.hidden = false;
        }
    };
    
    // Handle escape key to close the camera
    document.addEventListener('keydown', function(event) {
        if (event.key === 'Escape') {
            if (cameraModal.style.display === 'block') {
                closeAndStopCamera();
            }
            
            // Close error container if open
            const errorContainer = document.getElementById('error-container');
            if (errorContainer && errorContainer.style.display === 'block') {
                errorContainer.style.display = 'none';
                dropArea.hidden = false;
            }
        }
    });
}); 