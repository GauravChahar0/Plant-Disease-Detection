/* ═══════════════════════════════════════════════════
   PlantGuard AI — Frontend JavaScript
   ═══════════════════════════════════════════════════ */

document.addEventListener('DOMContentLoaded', () => {

    // ── DOM Elements ──────────────────────────────
    const navbar       = document.getElementById('navbar');
    const navToggle    = document.getElementById('navToggle');
    const navLinks     = document.querySelector('.nav-links');

    const uploadZone   = document.getElementById('uploadZone');
    const fileInput    = document.getElementById('fileInput');
    const placeholder  = document.getElementById('uploadPlaceholder');
    const preview      = document.getElementById('uploadPreview');
    const previewImg   = document.getElementById('previewImage');
    const removeBtn    = document.getElementById('removeBtn');
    const analyzeBtn   = document.getElementById('analyzeBtn');

    const resultsEmpty   = document.getElementById('resultsEmpty');
    const resultsLoading = document.getElementById('resultsLoading');
    const resultsContent = document.getElementById('resultsContent');
    const resultsError   = document.getElementById('resultsError');
    const resultsCard    = document.getElementById('resultsCard');
    const retryBtn       = document.getElementById('retryBtn');

    let selectedFile = null;


    // ═══════════════════════════════════════════════
    // NAVBAR
    // ═══════════════════════════════════════════════
    window.addEventListener('scroll', () => {
        navbar.classList.toggle('scrolled', window.scrollY > 50);
    });

    navToggle.addEventListener('click', () => {
        navLinks.classList.toggle('active');
    });

    // Close mobile nav on link click
    navLinks.querySelectorAll('a').forEach(link => {
        link.addEventListener('click', () => {
            navLinks.classList.remove('active');
        });
    });


    // ═══════════════════════════════════════════════
    // FILE UPLOAD — DRAG & DROP
    // ═══════════════════════════════════════════════
    uploadZone.addEventListener('click', (e) => {
        if (e.target !== removeBtn && !removeBtn.contains(e.target)) {
            fileInput.click();
        }
    });

    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('drag-over');
    });

    uploadZone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('drag-over');
    });

    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('drag-over');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    removeBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        clearFile();
    });

    retryBtn.addEventListener('click', () => {
        if (selectedFile) {
            analyzeImage();
        } else {
            showState('empty');
        }
    });


    // ═══════════════════════════════════════════════
    // FILE HANDLING
    // ═══════════════════════════════════════════════
    function handleFile(file) {
        // Validate file type
        const validTypes = ['image/jpeg', 'image/png', 'image/bmp', 'image/jpg', 'image/webp'];
        if (!validTypes.includes(file.type)) {
            showError('Invalid file type. Please upload a JPG, PNG, or BMP image.');
            return;
        }

        // Validate file size (10 MB max)
        if (file.size > 10 * 1024 * 1024) {
            showError('File is too large. Maximum size is 10 MB.');
            return;
        }

        selectedFile = file;

        // Show preview
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImg.src = e.target.result;
            placeholder.style.display = 'none';
            preview.style.display = 'flex';
            analyzeBtn.disabled = false;
        };
        reader.readAsDataURL(file);

        // Reset results
        showState('empty');
    }

    function clearFile() {
        selectedFile = null;
        fileInput.value = '';
        previewImg.src = '';
        placeholder.style.display = 'flex';
        preview.style.display = 'none';
        analyzeBtn.disabled = true;
        showState('empty');
    }


    // ═══════════════════════════════════════════════
    // ANALYZE — API CALL
    // ═══════════════════════════════════════════════
    analyzeBtn.addEventListener('click', () => {
        if (selectedFile) {
            analyzeImage();
        }
    });

    
async function analyzeImage() {
    showState('loading');
    analyzeBtn.disabled = true;

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
        // Determine backend predict URL (works on Live Server and when served directly by FastAPI)

        /* for running in local machine */

        // const backendHost = window.location.hostname === 'localhost' ? 'localhost' : '127.0.0.1';
        // const predictUrl = (window.location.protocol.startsWith('http') && window.location.port === '8000')
        //     ? '/api/predict'
        //     : `http://${backendHost}:8000/api/predict`;

        const predictUrl = 'https://plant-disease-detection-1-44wp.onrender.com/api/predict';

        const response = await fetch(
            predictUrl,
            {
                method: 'POST',
                body: formData
            }
        );

        // Read response as text first to avoid JSON parsing errors
        const rawText = await response.text();

        console.log('Backend status:', response.status);
        console.log('Backend response:', rawText);

        let data = {};

        if (rawText.trim()) {
            try {
                data = JSON.parse(rawText);
            } catch {
                throw new Error(
                    `Backend returned invalid JSON. HTTP ${response.status}`
                );
            }
        }

        if (!response.ok) {
            throw new Error(
                data.detail || `Server error: ${response.status}`
            );
        }

        if (!rawText.trim()) {
            throw new Error('Backend returned an empty response.');
        }

        if (
            data.success &&
            data.predictions &&
            data.predictions.length > 0
        ) {
            renderResults(data);
        } else {
            throw new Error(
                'No predictions returned from the model.'
            );
        }

    } catch (error) {
        console.error('Prediction error:', error);
        showError(error.message);
    } finally {
        analyzeBtn.disabled = false;
    }
}



    // ═══════════════════════════════════════════════
    // RENDER RESULTS
    // ═══════════════════════════════════════════════
    function renderResults(data) {
        const top = data.top_prediction;

        // Plant & Disease names
        document.getElementById('resultPlant').textContent = top.plant;
        document.getElementById('resultDisease').textContent = top.disease;

        // Severity badge
        const severityEl = document.getElementById('resultSeverity');
        severityEl.textContent = getSeverityLabel(top.severity);
        severityEl.className = 'result-severity severity-' + top.severity;

        // Update result-top background based on severity
        const resultTop = document.getElementById('resultTop');
        resultTop.style.background = getSeverityGradient(top.severity);
        resultTop.style.borderColor = getSeverityBorder(top.severity);

        // Confidence ring
        const confidence = top.confidence;
        const ringFill = document.getElementById('ringFill');
        const ringText = document.getElementById('ringText');
        const circumference = 2 * Math.PI * 42; // ~264
        const offset = circumference - (confidence / 100) * circumference;

        // Set stroke color based on severity
        ringFill.style.stroke = getSeverityColor(top.severity);
        ringText.textContent = '0%';

        // Animate after a small delay
        requestAnimationFrame(() => {
            setTimeout(() => {
                ringFill.style.strokeDashoffset = offset;
                animateCounter(ringText, 0, confidence, 1200);
            }, 100);
        });

        // Disease info
        document.getElementById('resultDescription').textContent = top.description;
        document.getElementById('resultTreatment').textContent = top.treatment;

        // Other predictions (skip first / top)
        const barsContainer = document.getElementById('predictionBars');
        barsContainer.innerHTML = '';

        const others = data.predictions.slice(1);
        others.forEach((pred, i) => {
            const bar = document.createElement('div');
            bar.className = 'prediction-bar';
            bar.innerHTML = `
                <span class="prediction-label" title="${pred.plant} — ${pred.disease}">
                    ${pred.plant} — ${pred.disease}
                </span>
                <div class="prediction-track">
                    <div class="prediction-fill" style="width: 0%"></div>
                </div>
                <span class="prediction-value">${pred.confidence.toFixed(1)}%</span>
            `;
            barsContainer.appendChild(bar);

            // Animate bar fill
            setTimeout(() => {
                bar.querySelector('.prediction-fill').style.width = pred.confidence + '%';
            }, 200 + i * 150);
        });

        showState('content');

        // Scroll results into view on mobile
        if (window.innerWidth <= 900) {
            setTimeout(() => {
                resultsCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }, 300);
        }
    }


    // ═══════════════════════════════════════════════
    // UI STATE MANAGEMENT
    // ═══════════════════════════════════════════════
    function showState(state) {
        resultsEmpty.style.display   = state === 'empty'   ? 'block' : 'none';
        resultsLoading.style.display = state === 'loading' ? 'block' : 'none';
        resultsContent.style.display = state === 'content' ? 'block' : 'none';
        resultsError.style.display   = state === 'error'   ? 'block' : 'none';

        // Reset ring when not showing content
        if (state !== 'content') {
            const ringFill = document.getElementById('ringFill');
            ringFill.style.strokeDashoffset = 264;
        }
    }

    function showError(message) {
        document.getElementById('errorMessage').textContent = message;
        showState('error');
    }


    // ═══════════════════════════════════════════════
    // HELPERS
    // ═══════════════════════════════════════════════
    function getSeverityLabel(severity) {
        const labels = {
            'none':     '✅ Healthy',
            'moderate': '⚠️ Moderate',
            'high':     '🔶 High Risk',
            'critical': '🔴 Critical',
        };
        return labels[severity] || '❓ Unknown';
    }

    function getSeverityColor(severity) {
        const colors = {
            'none':     '#16a34a',
            'moderate': '#f59e0b',
            'high':     '#f97316',
            'critical': '#ef4444',
        };
        return colors[severity] || '#64748b';
    }

    function getSeverityGradient(severity) {
        const gradients = {
            'none':     'linear-gradient(135deg, #f0fdf4 0%, #ecfdf5 100%)',
            'moderate': 'linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%)',
            'high':     'linear-gradient(135deg, #fff7ed 0%, #ffedd5 100%)',
            'critical': 'linear-gradient(135deg, #fef2f2 0%, #fecaca 100%)',
        };
        return gradients[severity] || 'linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%)';
    }

    function getSeverityBorder(severity) {
        const borders = {
            'none':     '#bbf7d0',
            'moderate': '#fde68a',
            'high':     '#fdba74',
            'critical': '#fca5a5',
        };
        return borders[severity] || '#e2e8f0';
    }

    function animateCounter(element, from, to, duration) {
        const start = performance.now();
        function update(timestamp) {
            const elapsed = timestamp - start;
            const progress = Math.min(elapsed / duration, 1);
            // Ease-out cubic
            const eased = 1 - Math.pow(1 - progress, 3);
            const current = from + (to - from) * eased;
            element.textContent = current.toFixed(1) + '%';
            if (progress < 1) {
                requestAnimationFrame(update);
            }
        }
        requestAnimationFrame(update);
    }


    // ═══════════════════════════════════════════════
    // SCROLL REVEAL ANIMATIONS
    // ═══════════════════════════════════════════════
    const revealElements = document.querySelectorAll(
        '.step-card, .plant-card, .upload-card, .results-card'
    );

    revealElements.forEach(el => el.classList.add('reveal'));

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('revealed');
                observer.unobserve(entry.target);
            }
        });
    }, {
        threshold: 0.15,
        rootMargin: '0px 0px -40px 0px'
    });

    revealElements.forEach(el => observer.observe(el));

});
