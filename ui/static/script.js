
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const loader = document.getElementById('loader');
const results = document.getElementById('results');

// Drag & Drop
dropZone.addEventListener('click', () => fileInput.click());

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length) {
        handleFile(e.dataTransfer.files[0]);
    }
});

fileInput.addEventListener('change', () => {
    if (fileInput.files.length) {
        handleFile(fileInput.files[0]);
    }
});

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        alert("Please upload a valid image file.");
        return;
    }

    // UI State
    dropZone.classList.add('hidden');
    loader.classList.remove('hidden');

    const formData = new FormData();
    formData.append('file', file);

    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert(data.error);
            location.reload();
            return;
        }
        displayResults(data);
    })
    .catch(err => {
        console.error(err);
        alert("An error occurred during analysis.");
        location.reload();
    })
    .finally(() => {
        loader.classList.add('hidden');
    });
}

function displayResults(data) {
    results.classList.remove('hidden');
    
    // Set Images
    document.getElementById('img-original').src = data.visualizations.original;
    document.getElementById('img-green').src = data.visualizations.green;
    document.getElementById('img-denoised').src = data.visualizations.denoised;
    document.getElementById('img-clahe').src = data.visualizations.clahe;

    // Set Text
    const statusParams = document.getElementById('diagnosis-text');
    statusParams.textContent = data.prediction;
    statusParams.className = `status ${data.prediction}`;
    
    document.getElementById('confidence-score').textContent = data.confidence;
}
