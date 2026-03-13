import { useState, useEffect } from 'react';

export default function useWatchlist() {
  const [watchlist, setWatchlist] = useState([]);

  useEffect(() => {
    // Load watchlist from localStorage
    const savedWatchlist = [];
    for (let i = 0; i < localStorage.length; i++) {
      const key = localStorage.key(i);
      if (key.startsWith('fav_')) {
        const tmdbId = key.replace('fav_', '');
        const movieData = localStorage.getItem(`movie_${tmdbId}`);
        if (movieData) {
          try {
            savedWatchlist.push(JSON.parse(movieData));
          } catch (e) {
            console.error('Failed to parse movie data:', e);
          }
        }
      }
    }
    setWatchlist(savedWatchlist);
  }, []);

  const addToWatchlist = (movie) => {
    if (movie.tmdb_id) {
      localStorage.setItem(`fav_${movie.tmdb_id}`, 'true');
      localStorage.setItem(`movie_${movie.tmdb_id}`, JSON.stringify(movie));
      setWatchlist(prev => [...prev.filter(m => m.tmdb_id !== movie.tmdb_id), movie]);
    }
  };

  const removeFromWatchlist = (tmdbId) => {
    localStorage.removeItem(`fav_${tmdbId}`);
    localStorage.removeItem(`movie_${tmdbId}`);
    setWatchlist(prev => prev.filter(m => m.tmdb_id !== tmdbId));
  };

  const isInWatchlist = (tmdbId) => {
    return localStorage.getItem(`fav_${tmdbId}`) === 'true';
  };

  return {
    watchlist,
    addToWatchlist,
    removeFromWatchlist,
    isInWatchlist,
  };
}
