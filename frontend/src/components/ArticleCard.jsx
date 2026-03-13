import React from 'react';
import { Clock, ExternalLink } from 'lucide-react';

/**
 * ArticleCard Component
 * Displays a single article/blog post with image, title, summary, and metadata
 */
const ArticleCard = ({ article, onClick }) => {
  const {
    title,
    summary,
    imageUrl,
    url,
    source,
    category,
    publishDate,
    readTime,
    author
  } = article;

  // Format date
  const formatDate = (dateStr) => {
    if (!dateStr) return '';
    const date = new Date(dateStr);
    return date.toLocaleDateString('en-US', { 
      month: 'short', 
      day: 'numeric',
      year: 'numeric' 
    });
  };

  const handleCardClick = (e) => {
    if (onClick) {
      onClick(article);
    } else {
      // Open article in new tab
      window.open(url, '_blank', 'noopener,noreferrer');
    }
  };

  const handleExternalClick = (e) => {
    e.stopPropagation();
    window.open(url, '_blank', 'noopener,noreferrer');
  };

  return (
    <div
      onClick={handleCardClick}
      className="group bg-gray-800/40 backdrop-blur-sm rounded-lg overflow-hidden 
                 hover:bg-gray-800/60 transition-all duration-200 cursor-pointer
                 border border-gray-700/30 hover:border-purple-500/40 
                 hover:shadow-lg hover:shadow-purple-500/10 flex flex-col h-full"
    >
      {/* Article Image */}
      {imageUrl ? (
        <div className="relative h-44 overflow-hidden bg-gray-900 flex-shrink-0">
          <img
            src={imageUrl}
            alt={title}
            loading="lazy"
            className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
            onError={(e) => {
              e.target.src = 'https://via.placeholder.com/400x300.png?text=Article+Image';
            }}
          />
          <div className="absolute inset-0 bg-gradient-to-t from-gray-900 to-transparent opacity-60"></div>
          
          {/* Category Badge */}
          {category && (
            <div className="absolute top-2 left-2">
              <span className="px-2 py-1 text-xs font-medium bg-purple-600/80 
                             text-white rounded-md backdrop-blur-sm">
                {category}
              </span>
            </div>
          )}
        </div>
      ) : (
        <div className="h-44 bg-gradient-to-br from-purple-900/20 to-blue-900/20 
                        flex items-center justify-center flex-shrink-0">
          <span className="text-gray-500 text-sm">{source}</span>
        </div>
      )}

      {/* Article Content */}
      <div className="p-4 flex flex-col flex-grow">
        {/* Title */}
        <h3 className="text-base font-semibold text-white mb-2 line-clamp-2 
                       group-hover:text-purple-300 transition-colors leading-snug">
          {title}
        </h3>

        {/* Summary */}
        <p className="text-gray-400 text-sm mb-3 line-clamp-2 leading-relaxed flex-grow">
          {summary}
        </p>

        {/* Metadata Footer */}
        <div className="flex items-center justify-between text-xs text-gray-500 
                        border-t border-gray-700/30 pt-2 mt-auto">
          <div className="flex items-center gap-3 truncate">
            {/* Source */}
            <span className="font-medium text-purple-400 truncate">{source}</span>
            
            {/* Read Time */}
            {readTime && (
              <div className="flex items-center gap-1 flex-shrink-0">
                <Clock size={12} />
                <span>{readTime} min</span>
              </div>
            )}
          </div>

          {/* External Link Icon */}
          <button
            onClick={handleExternalClick}
            className="p-1 hover:bg-purple-600/20 rounded transition-colors flex-shrink-0"
            aria-label="Open article"
          >
            <ExternalLink size={14} className="text-purple-400" />
          </button>
        </div>
      </div>
    </div>
  );
};

export default ArticleCard;
