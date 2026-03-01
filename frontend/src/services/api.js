import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || '/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const chatAPI = {
  // Send text query
  sendTextQuery: async (query, chatHistory = []) => {
    const formData = new FormData();
    formData.append('query', query);
    formData.append('chat_history', JSON.stringify(chatHistory));
    
    const response = await api.post('/chat', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },

  // Send poster image
  sendPosterImage: async (imageFile) => {
    const formData = new FormData();
    formData.append('poster', imageFile);
    
    const response = await api.post('/chat', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
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
