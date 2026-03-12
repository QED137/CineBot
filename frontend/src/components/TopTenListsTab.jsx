import { motion } from 'framer-motion';
import { ListOrdered, TrendingUp, Clock, ExternalLink, Search, Sparkles } from 'lucide-react';
import { useState, useEffect } from 'react';
import { articlesAPI } from '../services/api';
import SkeletonLoader from './SkeletonLoader';

export default function TopTenListsTab() {
  const [articles, setArticles] = useState([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [filteredArticles, setFilteredArticles] = useState([]);

  useEffect(() => {
    fetchTasteOfCinemaArticles();
  }, []);

  useEffect(() => {
    if (searchQuery.trim()) {
      const filtered = articles.filter(article =>
        article.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
        (article.summary && article.summary.toLowerCase().includes(searchQuery.toLowerCase()))
      );
      setFilteredArticles(filtered);
    } else {
      setFilteredArticles(articles);
    }
  }, [searchQuery, articles]);

  const fetchTasteOfCinemaArticles = async () => {
    setLoading(true);
    try {
      // Fetch articles from only Taste of Cinema source
      // getArticles(limit, source, search)
      const data = await articlesAPI.getArticles(24, 'tasteofcinema', null);
      setArticles(data.articles || []);
      setFilteredArticles(data.articles || []);
    } catch (error) {
      console.error('Failed to fetch Taste of Cinema articles:', error);
      setArticles([]);
      setFilteredArticles([]);
    } finally {
      setLoading(false);
    }
  };

  const formatDate = (dateString) => {
    if (!dateString) return '';
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
  };

  if (loading) {
    return (
      <div className="h-full p-6">
        <div className="mb-8">
          <div className="h-8 w-64 bg-slate-700/50 rounded animate-pulse mb-2"></div>
          <div className="h-4 w-96 bg-slate-700/30 rounded animate-pulse"></div>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {[...Array(6)].map((_, i) => (
            <SkeletonLoader key={i} height="h-80" />
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col p-6 overflow-y-auto">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center gap-3 mb-3">
          <div className="p-2 bg-gradient-to-br from-purple-500/20 to-pink-500/20 rounded-lg border border-purple-500/30">
            <ListOrdered className="w-6 h-6 text-purple-400" />
          </div>
          <h2 className="text-3xl font-bold text-white">
            10 Best Lists
          </h2>
          <Sparkles className="w-5 h-5 text-yellow-400" />
        </div>
        <p className="text-slate-300 text-sm mb-4">
          Curated film lists from <span className="text-purple-400 font-semibold">Taste of Cinema</span> - 
          Discover "10 Most...", "Best of...", and essential film rankings
        </p>

        {/* Search Bar */}
        <div className="relative max-w-md">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400" />
          <input
            type="text"
            placeholder="Search lists... (e.g., 'crime', 'classic', '90s')"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-10 pr-4 py-2.5 bg-slate-800/50 border border-slate-700 
                     rounded-lg text-white placeholder-slate-400 focus:outline-none 
                     focus:border-purple-500 focus:ring-2 focus:ring-purple-500/20 transition-all"
          />
        </div>

        {searchQuery && (
          <p className="text-sm text-slate-400 mt-2">
            Found {filteredArticles.length} list{filteredArticles.length !== 1 ? 's' : ''}
          </p>
        )}
      </div>

      {/* Articles Grid */}
      {filteredArticles.length === 0 ? (
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center">
            <ListOrdered className="w-16 h-16 text-slate-600 mx-auto mb-4" />
            <h3 className="text-xl font-semibold text-white mb-2">
              {searchQuery ? 'No lists found' : 'No articles available'}
            </h3>
            <p className="text-slate-400">
              {searchQuery ? 'Try a different search term' : 'Check back later for new content'}
            </p>
          </div>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 pb-6">
          {filteredArticles.map((article, index) => (
            <ArticleListCard key={article.id || index} article={article} index={index} />
          ))}
        </div>
      )}
    </div>
  );
}

// Article Card Component (Movie card-like palette)
function ArticleListCard({ article, index }) {
  const { title, url, publishDate, source, imageUrl, summary } = article;

  return (
    <motion.a
      href={url}
      target="_blank"
      rel="noopener noreferrer"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay: index * 0.05 }}
      whileHover={{ y: -8, scale: 1.02, transition: { duration: 0.2 } }}
      className="group block bg-gradient-to-br from-slate-800/80 to-slate-900/80 backdrop-blur-xl 
                 rounded-2xl overflow-hidden border border-slate-700/50 hover:border-purple-500/50 
                 transition-all duration-300 hover:shadow-2xl hover:shadow-purple-500/20"
    >
      {/* Image */}
      <div className="relative h-48 overflow-hidden bg-slate-900">
        {imageUrl ? (
          <img
            src={imageUrl}
            alt={title}
            loading="lazy"
            className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-500"
            onError={(e) => {
              e.target.style.display = 'none';
              e.target.parentElement.classList.add('flex', 'items-center', 'justify-center');
              e.target.parentElement.innerHTML = '<div class="text-slate-600"><svg class="w-16 h-16" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"></path></svg></div>';
            }}
          />
        ) : (
          <div className="w-full h-full flex items-center justify-center bg-gradient-to-br from-purple-900/20 to-slate-900">
            <ListOrdered className="w-16 h-16 text-slate-700" />
          </div>
        )}

        {/* Source Badge */}
        <div className="absolute top-3 left-3 px-3 py-1 rounded-full bg-purple-500/90 backdrop-blur-sm 
                      border border-purple-400/30 text-xs font-semibold text-white">
          {source || 'Taste of Cinema'}
        </div>

        {/* External Link Icon */}
        <div className="absolute top-3 right-3 p-2 rounded-full bg-black/50 backdrop-blur-sm 
                      opacity-0 group-hover:opacity-100 transition-opacity">
          <ExternalLink className="w-4 h-4 text-white" />
        </div>
      </div>

      {/* Content */}
      <div className="p-5">
        {/* Title */}
        <h3 className="text-lg font-bold text-white mb-3 line-clamp-3 
                     group-hover:text-purple-300 transition-colors leading-snug">
          {title}
        </h3>

        {/* Summary */}
        {summary && (
          <p className="text-sm text-slate-400 mb-4 line-clamp-3 leading-relaxed">
            {summary}
          </p>
        )}

        {/* Footer */}
        <div className="flex items-center justify-between pt-3 border-t border-slate-700/50">
          <div className="flex items-center gap-2 text-xs text-slate-500">
            <Clock className="w-3.5 h-3.5" />
            <span>{formatDate(publishDate)}</span>
          </div>
          <div className="flex items-center gap-1 text-xs font-medium text-purple-400 
                        group-hover:text-purple-300 transition-colors">
            Read List
            <ExternalLink className="w-3 h-3" />
          </div>
        </div>
      </div>
    </motion.a>
  );

  function formatDate(dateString) {
    if (!dateString) return 'Recently';
    const date = new Date(dateString);
    const now = new Date();
    const diffTime = Math.abs(now - date);
    const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));

    if (diffDays === 0) return 'Today';
    if (diffDays === 1) return 'Yesterday';
    if (diffDays < 7) return `${diffDays} days ago`;
    if (diffDays < 30) return `${Math.floor(diffDays / 7)} week${Math.floor(diffDays / 7) > 1 ? 's' : ''} ago`;
    
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
  }
}
