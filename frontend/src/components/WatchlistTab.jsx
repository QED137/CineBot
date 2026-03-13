import { motion } from 'framer-motion';
import { Heart, Film, Trash2, ExternalLink } from 'lucide-react';
import { useState, useEffect } from 'react';
import MovieCard from './MovieCard';
import EmptyState from './EmptyState';

export default function WatchlistTab() {
  const [favorites, setFavorites] = useState([]);

  const loadFavorites = () => {
    const favMovies = [];
    for (let i = 0; i < localStorage.length; i++) {
      const key = localStorage.key(i);
      if (key.startsWith('movie_')) {
        const tmdbId = key.replace('movie_', '');
        const isFavorite = localStorage.getItem(`fav_${tmdbId}`) === 'true';
        if (isFavorite) {
          try {
            const movieData = JSON.parse(localStorage.getItem(key));
            favMovies.push(movieData);
          } catch (e) {
            console.error('Failed to parse movie:', e);
          }
        }
      }
    }
    setFavorites(favMovies);
  };

  useEffect(() => {
    loadFavorites();
    
    // Reload when favorites change
    const interval = setInterval(loadFavorites, 1000);
    return () => clearInterval(interval);
  }, []);

  const clearAllFavorites = () => {
    if (window.confirm('Remove all movies from watchlist?')) {
      favorites.forEach(movie => {
        localStorage.removeItem(`fav_${movie.tmdb_id}`);
        localStorage.removeItem(`movie_${movie.tmdb_id}`);
      });
      setFavorites([]);
    }
  };

  if (favorites.length === 0) {
    return (
      <div className="h-full flex items-center justify-center p-8">
        <EmptyState
          icon={Heart}
          title="Your watchlist is empty"
          message="Click the heart icon on any movie to add it to your watchlist"
        />
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-2xl font-bold text-white flex items-center gap-2">
            <Heart className="w-6 h-6 fill-red-500 text-red-500" />
            My Watchlist
          </h2>
          <p className="text-sm text-slate-400 mt-1">
            {favorites.length} {favorites.length === 1 ? 'movie' : 'movies'} saved
          </p>
        </div>
        
        {favorites.length > 0 && (
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={clearAllFavorites}
            className="flex items-center gap-2 px-4 py-2 bg-red-500/10 hover:bg-red-500/20 border border-red-500/30 rounded-lg text-red-400 text-sm font-medium transition-colors"
          >
            <Trash2 className="w-4 h-4" />
            Clear All
          </motion.button>
        )}
      </div>

      {/* Movies Grid */}
      <div className="flex-1 overflow-y-auto">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
          {favorites.map((movie, index) => (
            <MovieCard key={movie.tmdb_id || index} movie={movie} index={index} />
          ))}
        </div>
      </div>
    </div>
  );
}
