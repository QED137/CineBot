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

export const articlesAPI = {
  // Get articles with optional filters
  getArticles: async (limit = 20, source = null, search = null) => {
    const params = new URLSearchParams();
    params.append('limit', limit.toString());
    if (source) params.append('source', source);
    if (search) params.append('search', search);
    
    const response = await api.get(`/articles?${params}`);
    return response.data;
  },

  // Get featured articles
  getFeaturedArticles: async (count = 5) => {
    const response = await api.get(`/articles/featured?count=${count}`);
    return response.data;
  },

  // Get available article sources
  getSources: async () => {
    const response = await api.get('/articles/sources');
    return response.data;
  },

  // Search articles by keyword
  searchArticles: async (query, limit = 10) => {
    const params = new URLSearchParams();
    params.append('search', query);
    params.append('limit', limit.toString());
    
    const response = await api.get(`/articles?${params}`);
    return response.data;
  },
};

export default api;
