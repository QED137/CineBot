import { motion } from 'framer-motion';
import { Star, Play, ExternalLink, Heart, Clock } from 'lucide-react';
import { useState } from 'react';

export default function MovieCard({ movie, index }) {
  const {
    title = 'Unknown Movie',
    explanation = '',
    poster_url,
    trailer_url,
    tagline,
    overview,
    tmdb_id,
    vote_average,
    vote_count,
    release_date,
  } = movie;

  const [isFavorite, setIsFavorite] = useState(
    localStorage.getItem(`fav_${tmdb_id}`) === 'true'
  );

  const posterImage = poster_url || 'https://via.placeholder.com/400x600.png?text=No+Poster';
  const rating = vote_average ? Number(vote_average).toFixed(1) : null;
  const year = release_date ? new Date(release_date).getFullYear() : null;

  const toggleFavorite = () => {
    const newState = !isFavorite;
    setIsFavorite(newState);
    if (tmdb_id) {
      localStorage.setItem(`fav_${tmdb_id}`, newState.toString());
      if (newState) {
        // Save full movie data for watchlist
        localStorage.setItem(`movie_${tmdb_id}`, JSON.stringify(movie));
      } else {
        // Remove movie data when unfavorited
        localStorage.removeItem(`movie_${tmdb_id}`);
      }
    }
  };
  
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay: index * 0.1 }}
      whileHover={{ y: -8, transition: { duration: 0.2 } }}
      className="group relative bg-gradient-to-br from-slate-800/80 to-slate-900/80 backdrop-blur-xl rounded-2xl overflow-hidden border border-slate-700/50 hover:border-indigo-500/50 transition-all duration-300 hover:shadow-2xl hover:shadow-indigo-500/20"
    >
      {/* Favorite Button */}
      <motion.button
        whileTap={{ scale: 0.9 }}
        onClick={toggleFavorite}
        className="absolute top-3 right-3 z-10 p-2 rounded-full bg-black/50 backdrop-blur-sm border border-white/10 hover:bg-black/70 transition-all"
      >
        <Heart
          className={`w-5 h-5 transition-all ${
            isFavorite ? 'fill-red-500 text-red-500' : 'text-white'
          }`}
        />
      </motion.button>

      <div className="relative">
        <img
          src={posterImage}
          alt={`${title} poster`}
          className="w-full h-72 object-cover group-hover:scale-105 transition-transform duration-500"
          onError={(e) => {
            e.target.src = 'https://via.placeholder.com/400x600.png?text=No+Poster';
          }}
        />
        
        {/* Rating Badge */}
        {rating && (
          <div className="absolute top-3 left-3 flex items-center gap-1 px-2.5 py-1 rounded-full bg-black/70 backdrop-blur-sm border border-yellow-500/30">
            <Star className="w-4 h-4 fill-yellow-400 text-yellow-400" />
            <span className="text-sm font-semibold text-white">{rating}</span>
            {vote_count && (
              <span className="text-xs text-slate-300">({vote_count})</span>
            )}
          </div>
        )}

        {/* Year Badge */}
        {year && (
          <div className="absolute top-3 left-3 mt-10 flex items-center gap-1 px-2.5 py-1 rounded-full bg-black/70 backdrop-blur-sm border border-slate-500/30">
            <Clock className="w-3 h-3 text-slate-300" />
            <span className="text-xs font-medium text-slate-300">{year}</span>
          </div>
        )}

        {/* Trailer Overlay */}
        {trailer_url && (
          <a
            href={trailer_url}
            target="_blank"
            rel="noopener noreferrer"
            className="absolute inset-0 bg-gradient-to-t from-black/80 via-black/40 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex items-center justify-center"
          >
            <motion.div
              whileHover={{ scale: 1.1 }}
              className="bg-indigo-600 rounded-full p-5 shadow-2xl shadow-indigo-500/50"
            >
              <Play className="w-8 h-8 text-white fill-white" />
            </motion.div>
          </a>
        )}
      </div>

      <div className="p-5 space-y-3">
        <div className="flex items-start justify-between gap-2">
          <h3 className="text-lg font-bold text-white line-clamp-2 flex-1 leading-tight">
            {title}
          </h3>
        </div>
        
        {tagline && (
          <p className="text-sm text-indigo-300 italic line-clamp-2">
            "{tagline}"
          </p>
        )}
        
        {explanation && (
          <div className="bg-gradient-to-r from-indigo-500/10 to-purple-500/10 border border-indigo-500/20 rounded-lg p-3">
            <p className="text-xs text-slate-300 leading-relaxed">
              <span className="text-indigo-400 font-semibold">🤖 CineBot says:</span> {explanation}
            </p>
          </div>
        )}
        
        {overview && (
          <p className="text-xs text-slate-400 line-clamp-3 leading-relaxed">{overview}</p>
        )}

        {tmdb_id && (
          <motion.a
            whileHover={{ x: 4 }}
            href={`https://www.themoviedb.org/movie/${tmdb_id}`}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1.5 mt-2 text-xs font-medium text-indigo-400 hover:text-indigo-300 transition-colors group/link"
          >
            <span>More Details</span>
            <ExternalLink className="w-3.5 h-3.5 group-hover/link:translate-x-0.5 transition-transform" />
          </motion.a>
        )}
      </div>
    </motion.div>
  );
}
