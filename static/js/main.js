// static/js/main.js (Refactored to fix poster uploads)

document.addEventListener('DOMContentLoaded', () => {
    // --- DOM Element References ---
    const textForm = document.getElementById('text-recommendation-form');
    const imageForm = document.getElementById('image-recommendation-form');
    const textQueryInput = document.getElementById('text-query-input');
    const imageUploader = document.getElementById('image-uploader');
    const imageFilenameDisplay = document.getElementById('image-filename');
    const chatContainer = document.getElementById('chat-container');
    const imageResultsContainer = document.getElementById('image-results-container');
    const inspireBtn = document.getElementById('inspire-btn');

    let isTyping = false;
    const buttonOriginalContent = {};

    // --- Typewriter & UI Logic (Unchanged) ---
    async function typeWriter(element, text, speed = 40) {
        isTyping = true;
        element.placeholder = '';
        for (let i = 0; i < text.length; i++) {
            await new Promise(resolve => setTimeout(resolve, speed));
            if (!isTyping) break;
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
            textQueryInput.placeholder = "Ask for a movie or a follow-up question...";
        }
    }

    document.querySelectorAll('.tab-link').forEach(tab => {
        tab.addEventListener('click', (e) => {
            document.querySelector('.tab-link.active').classList.remove('active');
            e.currentTarget.classList.add('active');
            const targetId = e.currentTarget.dataset.tab;
            document.querySelector('.tab-content.active').classList.remove('active');
            document.getElementById(targetId).classList.add('active');
        });
    });

    imageUploader.addEventListener('change', () => {
        const file = imageUploader.files[0];
        imageFilenameDisplay.textContent = file ? file.name : 'No file chosen';
    });

    inspireBtn.addEventListener('click', fetchAndAnimateSuggestion);

    textQueryInput.addEventListener('keydown', (e) => {
        if (isTyping) isTyping = false;
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            textForm.requestSubmit();
        }
    });

    textQueryInput.addEventListener('input', () => {
        textQueryInput.style.height = 'auto';
        textQueryInput.style.height = (textQueryInput.scrollHeight) + 'px';
    });

    // --- FORM SUBMISSION EVENT LISTENERS ---
    // These now call specific handlers.

    textForm.addEventListener('submit', (e) => {
        e.preventDefault();
        handleTextSubmit(e.currentTarget);
    });

    imageForm.addEventListener('submit', (e) => {
        e.preventDefault();
        handleImageSubmit(e.currentTarget);
    });

    // --- NEW: DEDICATED HANDLER FOR TEXT/CHAT QUERIES ---
    async function handleTextSubmit(form) {
        const query = textQueryInput.value.trim();
        if (!query || isTyping) return;
        const submitBtn = form.querySelector('button[type="submit"]');

        addMessageToChat('user', query);

        const formData = new FormData();
        formData.append('query', query);

        textQueryInput.value = '';
        textQueryInput.style.height = 'auto';
        textQueryInput.placeholder = '';
        
        setButtonLoadingState(submitBtn, true);
        addMessageToChat('assistant', '', true); // Show typing indicator

        try {
            const response = await fetch('/api/chat', { method: 'POST', body: formData });
            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || `Server responded with status ${response.status}`);
            }
            
            updateLastAssistantMessage(data.llm_response_text, data.html_cards);

        } catch (error) {
            updateLastAssistantMessage(`Oops! An error occurred: ${error.message}`, null, true);
        } finally {
            setButtonLoadingState(submitBtn, false);
            textQueryInput.focus();
        }
    }

    // --- NEW: DEDICATED HANDLER FOR IMAGE/POSTER QUERIES ---
    async function handleImageSubmit(form) {
        const file = imageUploader.files[0];
        if (!file) {
            showErrorInImageTab("Please upload a poster image first.");
            return;
        }
        const submitBtn = form.querySelector('button[type="submit"]');
        const formData = new FormData();
        formData.append('poster', file);

        setButtonLoadingState(submitBtn, true);
        showSkeletonsInImageTab(3); // Show 3 skeletons by default

        try {
            const response = await fetch('/api/chat', { method: 'POST', body: formData });
            const data = await response.json();
            
            imageResultsContainer.innerHTML = ''; // Clear skeletons
            if (!response.ok) { 
                throw new Error(data.error || 'Failed to get recommendations.'); 
            }

            // The result is placed directly in the image results container.
            imageResultsContainer.innerHTML = data.html_cards;

        } catch (error) {
            showErrorInImageTab(error.message);
        } finally {
            setButtonLoadingState(submitBtn, false);
        }
    }
    
    // --- Chat UI Display Functions (Unchanged) ---
    function addMessageToChat(role, text, isLoading = false) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `chat-message ${role}`;
        
        if (isLoading) {
            messageDiv.innerHTML = `<div class="typing-indicator"><span></span><span></span><span></span></div>`;
        } else {
            const contentDiv = document.createElement('div');
            contentDiv.className = 'chat-text-content';
            contentDiv.textContent = text;
            messageDiv.appendChild(contentDiv);
        }
        
        chatContainer.appendChild(messageDiv);
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    function updateLastAssistantMessage(text, htmlCards, isError = false) {
        const lastMessage = chatContainer.querySelector('.chat-message.assistant:last-child');
        if (!lastMessage) return;
        
        lastMessage.innerHTML = '';
            
        if (isError) {
            lastMessage.innerHTML = `<p class="error-message">${text}</p>`;
        } else {
            const conversationalText = text.replace(/MOVIE:.*?\nEXPLANATION:/gs, '').replace(/MOVIE:.*$/gs, '').trim();
            const textContentDiv = document.createElement('div');
            textContentDiv.className = 'chat-text-content';
            textContentDiv.textContent = conversationalText || text;
            lastMessage.appendChild(textContentDiv);

            if (htmlCards) {
                const cardsContainer = document.createElement('div');
                cardsContainer.className = 'chat-results-container';
                cardsContainer.innerHTML = htmlCards;
                lastMessage.appendChild(cardsContainer);
            }
        }
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    // --- Helper Functions (Unchanged) ---
    function setButtonLoadingState(button, isLoading) {
        if (!button) return;
        const buttonId = button.id || 'submit-btn-' + Math.random();

        if (isLoading) {
            buttonOriginalContent[buttonId] = button.innerHTML;
            button.disabled = true;
            button.innerHTML = `<span class="spinner"></span>`;
            if (button.form?.id === 'text-recommendation-form') {
                textQueryInput.disabled = true;
                textQueryInput.placeholder = "CineBot is thinking...";
            }
        } else {
            button.disabled = false;
            button.innerHTML = buttonOriginalContent[buttonId] || 'Submit';
            if (button.form?.id === 'text-recommendation-form') {
                textQueryInput.disabled = false;
                fetchAndAnimateSuggestion();
            }
        }
    }
    
    function showSkeletonsInImageTab(count) {
        imageResultsContainer.innerHTML = Array(count).fill('<div class="skeleton-card"></div>').join('');
    }

    function showErrorInImageTab(message) {
        imageResultsContainer.innerHTML = `<div class="error-message">${message}</div>`;
    }

    // --- Initial Setup ---
    function initialize() {
        addMessageToChat("assistant", "Hello! I'm CineBot. Ask me for a movie based on a description, a specific question (like 'who directed...?'), or upload a poster!");
        fetchAndAnimateSuggestion();
    }

    initialize();
});