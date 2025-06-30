// static/js/main.js

document.addEventListener('DOMContentLoaded', () => {
    // --- DOM Element References (same as before) ---
    const textForm = document.getElementById('text-recommendation-form');
    const imageForm = document.getElementById('image-recommendation-form');
    const textQueryInput = document.getElementById('text-query-input');
    const imageUploader = document.getElementById('image-uploader');
    const imageFilenameDisplay = document.getElementById('image-filename');
    const resultsContainer = document.getElementById('results-container');
    const statusContainer = document.getElementById('status-container');
    const colsSlider = document.getElementById('cols-slider');
    const sliderValue = document.getElementById('slider-value');
    const inspireBtn = document.getElementById('inspire-btn');

    // --- State Management ---
    let lastRequest = { endpoint: null, body: null };
    let isTyping = false;

    // --- Typewriter Logic ---
    async function typeWriter(element, text, speed = 40) {
        isTyping = true;
        element.placeholder = '';
        for (let i = 0; i < text.length; i++) {
            await new Promise(resolve => setTimeout(resolve, speed));
            element.placeholder += text.charAt(i);
        }
        isTyping = false;
    }

    async function fetchAndAnimateSuggestion() {
        if (isTyping) return;
        try {
            const response = await fetch('/api/suggestion');
            const data = await response.json();
            if (data.suggestion) await typeWriter(textQueryInput, data.suggestion);
        } catch (error) {
            console.error("Failed to fetch suggestion:", error);
            textQueryInput.placeholder = "e.g., A mind-bending sci-fi thriller...";
        }
    }

    // --- UI Interaction ---
    const tabs = document.querySelectorAll('.tab-link');
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            document.querySelector('.tab-link.active').classList.remove('active');
            tab.classList.add('active');
            const targetId = tab.dataset.tab;
            document.querySelector('.tab-content.active').classList.remove('active');
            document.getElementById(targetId).classList.add('active');
            clearAllAndShowEmptyState();
        });
    });

    colsSlider.addEventListener('input', (e) => {
        sliderValue.textContent = e.target.value;
        resultsContainer.style.gridTemplateColumns = `repeat(${e.target.value}, 1fr)`;
    });

    imageUploader.addEventListener('change', () => {
        const file = imageUploader.files[0];
        imageFilenameDisplay.textContent = file ? file.name : '';
    });

    inspireBtn.addEventListener('click', fetchAndAnimateSuggestion);

    // --- Form Submission ---
    textForm.addEventListener('submit', (e) => {
        e.preventDefault();
        const query = textQueryInput.value.trim();
        if (!query) { showError("Please enter a movie description."); return; }
        const body = { query, num_recs: parseInt(colsSlider.value) };
        fetchRecommendations('/api/recommend/text', body, e.submitter);
    });

    imageForm.addEventListener('submit', (e) => {
        e.preventDefault();
        const file = imageUploader.files[0];
        if (!file) { showError("Please upload a poster image."); return; }
        const formData = new FormData();
        formData.append('poster', file);
        formData.append('num_recs', parseInt(colsSlider.value));
        fetchRecommendations('/api/recommend/image', formData, e.submitter);
    });

    // --- Core Fetch & Display Logic ---
    async function fetchRecommendations(endpoint, body, submitBtn) {
        lastRequest = { endpoint, body }; // Store for "Try Again"
        setButtonLoadingState(submitBtn, true);
        clearAllAndShowSkeletons();
        
        try {
            const options = { method: 'POST' };
            if (body instanceof FormData) options.body = body;
            else { options.headers = { 'Content-Type': 'application/json' }; options.body = JSON.stringify(body); }

            const response = await fetch(endpoint, options);
            resultsContainer.innerHTML = '';
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || `Server responded with status ${response.status}`);
            }
            const recommendations = await response.json();
            displayRecommendations(recommendations);
        } catch (error) {
            console.error('Fetch error:', error);
            showError(error.message, true); // Show error with a "Try Again" button
        } finally {
            setButtonLoadingState(submitBtn, false);
        }
    }
    
    // --- Professional UI Enhancements ---
    function setButtonLoadingState(button, isLoading) {
        if (!button) return;
        if (isLoading) {
            button.disabled = true;
            button.innerHTML = `<span class="spinner"></span> Finding...`;
        } else {
            button.disabled = false;
            button.innerHTML = button.form.id.includes('text') ? 'Find Movies' : 'Find Similar Movies';
        }
    }

    async function sendFeedback(tmdb_id, feedback, btn) {
        // Visually update button immediately for better UX
        const parent = btn.parentElement;
        parent.querySelectorAll('.feedback-btn').forEach(b => b.classList.remove('liked', 'disliked'));
        btn.classList.add(feedback === 'like' ? 'liked' : 'disliked');

        try {
            await fetch('/api/feedback', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ tmdb_id, feedback })
            });
        } catch (error) {
            console.error("Failed to send feedback:", error);
            // Optionally revert the button style on failure
            btn.classList.remove('liked', 'disliked');
        }
    }

    function displayRecommendations(recs) {
        if (!recs || recs.length === 0) {
            showError("CineBot couldn't find any matches. Please try a different query!");
            return;
        }
        recs.forEach((rec, index) => {
            const card = document.createElement('div');
            card.className = 'movie-card';
            card.style.animationDelay = `${index * 100}ms`; // Staggered animation
            
            const posterUrl = rec.poster_url || 'https://via.placeholder.com/400x600.png?text=No+Poster';
            const trailerLink = rec.trailer_url ? `<a href="${rec.trailer_url}" target="_blank">Trailer</a>` : '';
            const detailsLink = rec.tmdb_id ? `<a href="https://www.themoviedb.org/movie/${rec.tmdb_id}" target="_blank">Details</a>` : '';

            card.innerHTML = `
                <div class="card-feedback">
                    <button class="feedback-btn" data-feedback="like" data-id="${rec.tmdb_id}" title="Good rec!"><svg viewBox="0 0 24 24"><path fill="currentColor" d="M23,10C23,8.89 22.1,8 21,8H14.68L15.64,3.43C15.66,3.33 15.67,3.22 15.67,3.11C15.67,2.7 15.5,2.32 15.23,2.05L14.17,1L7.59,7.58C7.22,7.95 7,8.45 7,9V19A2,2 0 0,0 9,21H18C18.83,21 19.54,20.5 19.84,19.78L22.86,12.73C22.95,12.5 23,12.26 23,12V10Z"></path></svg></button>
                    <button class="feedback-btn" data-feedback="dislike" data-id="${rec.tmdb_id}" title="Not what I wanted"><svg viewBox="0 0 24 24"><path fill="currentColor" d="M19,15H21A2,2 0 0,1 23,17V19A2,2 0 0,1 21,21H12.2C11.66,21 11.14,20.76 10.84,20.34L7.82,13.29C7.73,13.06 7.67,12.81 7.67,12.55V10.5A1.5,1.5 0 0,1 9.17,9L10.23,2.95C10.5,2.68 10.86,2.5 11.28,2.5C11.67,2.5 12.04,2.66 12.31,2.92L13.37,4L12.41,8.57C12.38,8.67 12.37,8.78 12.37,8.89V15H19M1,15H5V3H1V15Z"></path></svg></button>
                </div>
                <img src="${posterUrl}" alt="Poster for ${rec.title}" class="poster-img">
                <div class="card-content">
                    <h4>${rec.title || 'Recommendation'}</h4>
                    <div class="card-explanation"><p><i>CineBot says:</i> ${rec.explanation || '...'}</p></div>
                    <div class="card-actions">${trailerLink}${detailsLink}</div>
                </div>`;

            card.querySelectorAll('.feedback-btn').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    e.stopPropagation(); // Prevent card hover effects
                    sendFeedback(btn.dataset.id, btn.dataset.feedback, btn);
                });
            });
            resultsContainer.appendChild(card);
        });
    }

    // --- UI State Management ---
    function clearAllAndShowSkeletons() {
        statusContainer.innerHTML = '';
        resultsContainer.innerHTML = '';
        const count = parseInt(colsSlider.value);
        for (let i = 0; i < count; i++) {
            const skeleton = document.createElement('div');
            skeleton.className = 'skeleton-card';
            resultsContainer.appendChild(skeleton);
        }
    }

    function clearAllAndShowEmptyState() {
        statusContainer.innerHTML = '';
        resultsContainer.innerHTML = `<div class="empty-state">...</div>`; // simplified
    }

    function showError(message, showTryAgain = false) {
        resultsContainer.innerHTML = '';
        let tryAgainBtnHTML = '';
        if (showTryAgain && lastRequest.endpoint) {
            tryAgainBtnHTML = `<button class="try-again-btn">Try Again</button>`;
        }
        statusContainer.innerHTML = `<div class="error-message"><strong>Oops!</strong> ${message}${tryAgainBtnHTML}</div>`;

        if (showTryAgain && lastRequest.endpoint) {
            document.querySelector('.try-again-btn').addEventListener('click', () => {
                const formId = lastRequest.endpoint.includes('text') ? 'text-recommendation-form' : 'image-recommendation-form';
                const submitBtn = document.querySelector(`#${formId} .submit-btn`);
                fetchRecommendations(lastRequest.endpoint, lastRequest.body, submitBtn);
            });
        }
    }

    // --- Initial Setup ---
    function initialize() {
        resultsContainer.style.gridTemplateColumns = `repeat(${colsSlider.value}, 1fr)`;
        clearAllAndShowEmptyState();
        fetchAndAnimateSuggestion();
    }

    initialize();
});