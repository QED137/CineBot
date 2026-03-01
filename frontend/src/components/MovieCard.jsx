export default function MovieCard({ movie, index }) {
  const {
    title = 'Unknown Movie',
    explanation = '',
    poster_url,
    trailer_url,
    tagline,
    overview,
    tmdb_id,
  } = movie;

  const posterImage = poster_url || 'https://via.placeholder.com/400x600.png?text=No+Poster';
  
  return (
    <div className="bg-slate-800/50 rounded-xl overflow-hidden border border-slate-700 hover:border-primary/50 transition-all duration-300 hover:shadow-xl hover:shadow-primary/20 fade-in">
      <div className="relative group">
        <img
          src={posterImage}
          alt={`${title} poster`}
          className="w-full h-64 object-cover group-hover:scale-105 transition-transform duration-300"
          onError={(e) => {
            e.target.src = 'https://via.placeholder.com/400x600.png?text=No+Poster';
          }}
        />
        {trailer_url && (
          <a
            href={trailer_url}
            target="_blank"
            rel="noopener noreferrer"
            className="absolute inset-0 bg-black/60 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center"
          >
            <div className="bg-primary rounded-full p-4 transform hover:scale-110 transition-transform">
              <svg className="w-8 h-8 text-white" fill="currentColor" viewBox="0 0 20 20">
                <path d="M6.3 2.841A1.5 1.5 0 004 4.11V15.89a1.5 1.5 0 002.3 1.269l9.344-5.89a1.5 1.5 0 000-2.538L6.3 2.84z" />
              </svg>
            </div>
          </a>
        )}
      </div>

      <div className="p-4">
        <h3 className="text-lg font-semibold text-white mb-2 line-clamp-1">{title}</h3>
        
        {tagline && (
          <p className="text-sm text-primary italic mb-2 line-clamp-1">"{tagline}"</p>
        )}
        
        {explanation && (
          <p className="text-sm text-slate-300 mb-3">{explanation}</p>
        )}
        
        {overview && (
          <p className="text-xs text-slate-400 line-clamp-3">{overview}</p>
        )}

        {tmdb_id && (
          <a
            href={`https://www.themoviedb.org/movie/${tmdb_id}`}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-block mt-3 text-xs text-secondary hover:text-primary transition-colors"
          >
            View on TMDB →
          </a>
        )}
      </div>
    </div>
  );
}
