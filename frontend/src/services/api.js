import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || '/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const chatAPI = {
  // Send text query with cancellation support
  sendTextQuery: async (query, chatHistory = [], signal = null) => {
    const formData = new FormData();
    formData.append('query', query);
    formData.append('chat_history', JSON.stringify(chatHistory));
    
    const response = await api.post('/chat', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      signal, // AbortController signal for cancellation
    });
    return response.data;
  },

  // Send poster image with chat history and cancellation support
  sendPosterImage: async (imageFile, chatHistory = [], signal = null) => {
    const formData = new FormData();
    formData.append('poster', imageFile);
    formData.append('chat_history', JSON.stringify(chatHistory));
    
    const response = await api.post('/chat', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      signal, // AbortController signal for cancellation
    });
    return response.data;
  },

  // Get random suggestion
  getSuggestion: async () => {
    const response = await api.get('/suggestion');
    return response.data;
  },
};

export default api;
