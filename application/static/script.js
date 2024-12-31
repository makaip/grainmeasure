let uploadedFile = null; // Keep track of the uploaded file

// Tab switching functionality
document.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', () => {
        document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));

        tab.classList.add('active');
        document.getElementById(tab.dataset.target).classList.add('active');

        // Show saturation slider only on Color Algorithm tab
        const showSaturation = tab.dataset.target === 'coloralg-tab';
        document.getElementById('saturation-controls').style.display = showSaturation ? 'flex' : 'none';
    });
});

// Update saturation factor display
const saturationSlider = document.getElementById('saturation-slider');
const saturationValue = document.getElementById('saturation-value');
saturationSlider.addEventListener('input', () => {
    saturationValue.textContent = (saturationSlider.value / 100).toFixed(1);
});

// Handle file upload
document.getElementById('image').addEventListener('change', event => {
    uploadedFile = event.target.files[0];
});

// Form submission for processing
document.getElementById('upload-form').addEventListener('submit', async function (event) {
    event.preventDefault();

    const activeTab = document.querySelector('.tab.active').dataset.target;
    const algorithm = activeTab === 'coloralg-tab' ? 'coloralg' : 'contouralg';

    if (!uploadedFile) {
        alert('Please upload a file first.');
        return;
    }

    showLoader(); // Show loader modal when processing starts

    const formData = new FormData();
    formData.append('image', uploadedFile);

    if (algorithm === 'coloralg') {
        const saturationFactor = document.getElementById('saturation-slider').value / 100;
        formData.append('saturation', saturationFactor);
    }

    try {
        const response = await fetch(`/process?algorithm=${algorithm}`, {
            method: 'POST',
            body: formData,
        });

        if (response.ok) {
            const data = await response.json();
            const targetId = algorithm === 'coloralg' ? 'results-coloralg' : 'results-contouralg';
            displayResults(data, targetId);
        } else {
            alert('Failed to process the image. Please try again.');
        }
    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred. Please try again.');
    } finally {
        hideLoader(); // Hide loader modal after processing finishes
    }
});

// Display results
function displayResults(data, targetId) {
    const resultsDiv = document.getElementById(targetId);
    resultsDiv.innerHTML = `
        <div class="results-row">
            <p>Average Length: <span>${data.average_length}</span></p>
            <p>Grain Count: <span>${data.grain_count}</span></p>
        </div>
    `;

    const imageGrid = document.createElement('div');
    imageGrid.className = 'image-grid';

    ['binary_image', 'contours_image', 'ellipses_image', 'histogram_image'].forEach(type => {
        const img = document.createElement('img');
        img.src = `data:image/png;base64,${data[type]}`;
        img.addEventListener('click', () => openModal(img.src));
        imageGrid.appendChild(img);
    });

    resultsDiv.appendChild(imageGrid);
}

// Modal functionality
const modal = document.getElementById('modal');
const modalImage = document.getElementById('modal-image');

function openModal(src) {
    modalImage.src = src;
    modal.style.display = 'flex';
}

modal.addEventListener('click', () => {
    modal.style.display = 'none';
});

// Show the loader modal
function showLoader() {
    const loaderModal = document.getElementById('loader-modal');
    const loader = loaderModal.querySelector('.loader');
    loaderModal.style.display = 'flex';
    loader.style.display = 'block'; // Make sure the spinner is visible
}

// Hide the loader modal
function hideLoader() {
    const loaderModal = document.getElementById('loader-modal');
    loaderModal.style.display = 'none'; // Hide the entire modal
}
