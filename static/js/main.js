// static/js/main.js

document.addEventListener('DOMContentLoaded', () => {
    // --- DOM Element References ---
    const textForm = document.getElementById('text-recommendation-form');
    const imageForm = document.getElementById('image-recommendation-form');
    const textQueryInput = document.getElementById('text-query-input');
    const imageUploader = document.getElementById('image-uploader');
    const imageFilenameDisplay = document.getElementById('image-filename');
    const chatContainer = document.getElementById('chat-container');
    const imageResultsContainer = document.getElementById('image-results-container');
    const colsSlider = document.getElementById('cols-slider');
    const sliderValue = document.getElementById('slider-value');
    const inspireBtn = document.getElementById('inspire-btn');

    // --- State Management ---
    let chatHistory = [];
    let isTyping = false;
    const buttonOriginalContent = {};

    // --- Typewriter Logic ---
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
            if (!response.ok) throw new Error('Failed to fetch suggestion');
            const data = await response.json();
            if (data.suggestion) await typeWriter(textQueryInput, data.suggestion);
        } catch (error) {
            console.error("Failed to fetch suggestion:", error);
            textQueryInput.placeholder = "Ask for a movie or a follow-up question...";
        }
    }

    // --- UI Interaction ---
    document.querySelectorAll('.tab-link').forEach(tab => {
        tab.addEventListener('click', (e) => {
            document.querySelector('.tab-link.active').classList.remove('active');
            e.currentTarget.classList.add('active');
            const targetId = e.currentTarget.dataset.tab;
            document.querySelector('.tab-content.active').classList.remove('active');
            document.getElementById(targetId).classList.add('active');
        });
    });

    colsSlider.addEventListener('input', (e) => {
        sliderValue.textContent = e.target.value;
        // This relies on your CSS using a variable, e.g., grid-template-columns: repeat(var(--grid-cols, 3), 1fr);
        imageResultsContainer.style.setProperty('--grid-cols', e.target.value);
    });

    imageUploader.addEventListener('change', () => {
        const file = imageUploader.files[0];
        imageFilenameDisplay.textContent = file ? file.name : 'No file chosen';
    });

    inspireBtn.addEventListener('click', fetchAndAnimateSuggestion);

    // --- NEW: Textarea Enhancements ---
    textQueryInput.addEventListener('keydown', (e) => {
        if (isTyping) isTyping = false; // Stop typewriter if user starts typing
        
        // Submit on Enter, but not Shift+Enter
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault(); // Prevent new line
            textForm.requestSubmit(); // Programmatically submit the form
        }
    });

    textQueryInput.addEventListener('input', () => {
        // Auto-resize the textarea
        textQueryInput.style.height = 'auto';
        textQueryInput.style.height = (textQueryInput.scrollHeight) + 'px';
    });
    // --- End of New Code ---

    // --- Form Submission Logic ---
    textForm.addEventListener('submit', (e) => {
        e.preventDefault();
        const query = textQueryInput.value.trim();
        if (!query || isTyping) return;

        addMessageToChat('user', query);
        chatHistory.push({ role: "user", content: query });
        textQueryInput.value = '';
        textQueryInput.style.height = 'auto'; // Reset height after submit
        textQueryInput.placeholder = '';

        const body = { query, history: chatHistory, num_recs: 3 };
        fetchTextRecommendations('/api/recommend/text', body, e.currentTarget.querySelector('button[type="submit"]'));
    });

    imageForm.addEventListener('submit', (e) => {
        e.preventDefault();
        const file = imageUploader.files[0];
        if (!file) {
            showErrorInImageTab("Please upload a poster image.");
            return;
        }
        const formData = new FormData();
        formData.append('poster', file);
        formData.append('num_recs', parseInt(colsSlider.value));
        fetchImageRecommendations('/api/recommend/image', formData, e.currentTarget.querySelector('button[type="submit"]'));
    });

    // ... (The rest of your main.js file remains exactly the same as the one I provided before) ...
    // fetchTextRecommendations, fetchImageRecommendations, addMessageToChat, etc.

    // --- Fetch & Display Logic ---
    async function fetchTextRecommendations(endpoint, body, submitBtn) {
        setButtonLoadingState(submitBtn, true);
        addMessageToChat('assistant', '', true);

        try {
            const options = { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) };
            const response = await fetch(endpoint, options);
            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || `Server responded with status ${response.status}`);
            }
            
            updateLastAssistantMessage(data.llm_response, data.html);
            chatHistory.push({ role: "assistant", content: data.llm_response });

        } catch (error) {
            updateLastAssistantMessage(`Oops! An error occurred: ${error.message}`, null, true);
            chatHistory.push({ role: "assistant", content: `Error: ${error.message}` });
        } finally {
            setButtonLoadingState(submitBtn, false);
            textQueryInput.focus();
        }
    }

    async function fetchImageRecommendations(endpoint, body, submitBtn) {
        setButtonLoadingState(submitBtn, true);
        imageResultsContainer.innerHTML = '';
        showSkeletonsInImageTab(parseInt(colsSlider.value));

        try {
            const options = { method: 'POST', body };
            const response = await fetch(endpoint, options);
            const data = await response.json();
            
            imageResultsContainer.innerHTML = ''; // Clear skeletons
            if (!response.ok) { throw new Error(data.error); }

            imageResultsContainer.innerHTML = data.html;
            attachFeedbackListeners(imageResultsContainer);

        } catch (error) {
            showErrorInImageTab(error.message);
        } finally {
            setButtonLoadingState(submitBtn, false);
        }
    }

    // --- Chat UI Display Functions ---
    function addMessageToChat(role, text, isLoading = false) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `chat-message ${role}`;
        
        if (isLoading) {
            messageDiv.innerHTML = `<div class="typing-indicator"><span></span><span></span><span></span></div>`;
        } else {
            const contentDiv = document.createElement('div');
            contentDiv.className = 'chat-text-content';
            contentDiv.textContent = text; // Use textContent for security
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
            const parsedText = text.replace(/MOVIE:.*?\nEXPLANATION:/gs, '').replace(/MOVIE:.*$/gs, '').trim();
            const textContentDiv = document.createElement('div');
            textContentDiv.className = 'chat-text-content';
            textContentDiv.textContent = parsedText || text; // Fallback to raw text if parsing fails
            lastMessage.appendChild(textContentDiv);

            if (htmlCards) {
                const cardsContainer = document.createElement('div');
                cardsContainer.className = 'chat-results-container';
                cardsContainer.innerHTML = htmlCards;
                lastMessage.appendChild(cardsContainer);
                attachFeedbackListeners(cardsContainer);
            }
        }
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    // --- Helper Functions ---
    function attachFeedbackListeners(container) {
        container.querySelectorAll('.feedback-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.stopPropagation();
                const button = e.currentTarget;
                sendFeedback(button.dataset.id, button.dataset.feedback, button);
            });
        });
    }

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

    async function sendFeedback(tmdb_id, feedback, btn) {
        btn.classList.add('feedback-sent');
        btn.disabled = true;
        const sibling = btn.parentElement.querySelector(`.feedback-btn:not([data-feedback='${feedback}'])`);
        if(sibling) sibling.disabled = true;

        try {
            await fetch('/api/feedback', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ tmdb_id, feedback })
            });
            console.log(`Feedback sent: ${tmdb_id} - ${feedback}`);
        } catch (error) {
            console.error('Failed to send feedback:', error);
            btn.classList.remove('feedback-sent');
            btn.disabled = false;
            if(sibling) sibling.disabled = false;
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
        imageResultsContainer.style.setProperty('--grid-cols', colsSlider.value);
        addMessageToChat("assistant", "Hello! I'm CineBot. Ask me for a movie based on a description, theme, or even a similar poster!");
        fetchAndAnimateSuggestion();
    }

    initialize();
});