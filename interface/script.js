function showTab(tabId) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(tab => tab.style.display = 'none');
    // Show the selected tab
    document.getElementById(tabId).style.display = 'block';
}

function getTextRecommendations() {
    const textQuery = document.getElementById('textQuery').value;
    if (!textQuery) {
        alert("Please enter a description!");
        return;
    }

    // Simulate a backend request (Replace with your API call)
    const mockRecommendations = [
        {
            title: 'Mock Movie 1',
            explanation: 'This is a great mock movie because reasons.',
            posterUrl: 'https://picsum.photos/seed/1/200/300',
            trailerUrl: 'https://www.youtube.com/watch?v=dQw4w9WgXcQ',
        },
        {
            title: 'Mock Movie 2',
            explanation: 'An action-packed thrill ride!',
            posterUrl: 'https://picsum.photos/seed/2/200/300',
            trailerUrl: 'https://www.youtube.com/watch?v=dQw4w9WgXcQ',
        }
    ];

    displayRecommendations(mockRecommendations, 'textRecommendations');
}

function getImageRecommendations() {
    const imageUpload = document.getElementById('imageUpload').files[0];
    if (!imageUpload) {
        alert("Please upload an image!");
        return;
    }

    // Simulate a backend request (Replace with your API call)
    const mockRecommendations = [
        {
            title: 'Mock Movie 1',
            explanation: 'This is a great mock movie because reasons.',
            posterUrl: 'https://picsum.photos/seed/1/200/300',
            trailerUrl: 'https://www.youtube.com/watch?v=dQw4w9WgXcQ',
        },
        {
            title: 'Mock Movie 2',
            explanation: 'An action-packed thrill ride!',
            posterUrl: 'https://picsum.photos/seed/2/200/300',
            trailerUrl: 'https://www.youtube.com/watch?v=dQw4w9WgXcQ',
        }
    ];

    displayRecommendations(mockRecommendations, 'imageRecommendations');
}

function displayRecommendations(recommendations, containerId) {
    const container = document.getElementById(containerId);
    container.innerHTML = ''; // Clear previous recommendations

    recommendations.forEach(rec => {
        const card = document.createElement('div');
        card.classList.add('movie-card');
        card.innerHTML = `
            <h3 class="movie-title">${rec.title}</h3>
            <img src="${rec.posterUrl}" alt="${rec.title}">
            <p>${rec.explanation}</p>
            <a href="${rec.trailerUrl}" target="_blank">Watch Trailer</a>
        `;
        container.appendChild(card);
    });
}

// Initialize default tab view
showTab('textTab');
