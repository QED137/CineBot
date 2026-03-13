import React, { useState, useEffect, useRef } from 'react';
import { Newspaper, Search, RefreshCw, Filter, AlertCircle, ChevronUp } from 'lucide-react';
import ArticleCard from './ArticleCard';
import SkeletonLoader from './SkeletonLoader';

/**
 * ArticlesTab Component
 * Displays movie blog articles and news with filtering and search
 */
const ArticlesTab = () => {
  const [articles, setArticles] = useState([]);
  const [recommendedArticles, setRecommendedArticles] = useState([]);
  const [featuredArticles, setFeaturedArticles] = useState([]);
  const [sources, setSources] = useState([]);
  const [loading, setLoading] = useState(true);
  const [loadingRecommended, setLoadingRecommended] = useState(false);
  const [error, setError] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedSource, setSelectedSource] = useState(null);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [recommendedMovies, setRecommendedMovies] = useState([]);
  const [showBackToTop, setShowBackToTop] = useState(false);
  const scrollContainerRef = useRef(null);

  // Get recommended movies from localStorage
  const getRecommendedMovies = () => {
    try {
      const movies = localStorage.getItem('recommendedMovies');
      const timestamp = localStorage.getItem('recommendedMoviesTimestamp');
      
      if (!movies) return [];
      
      // Only use recommendations from last 30 minutes
      const thirtyMinutes = 30 * 60 * 1000;
      if (timestamp && (Date.now() - parseInt(timestamp)) < thirtyMinutes) {
        return JSON.parse(movies);
      }
      
      return [];
    } catch (e) {
      console.error('Error reading recommended movies:', e);
      return [];
    }
  };

  // Fetch articles on mount
  useEffect(() => {
    const movies = getRecommendedMovies();
    setRecommendedMovies(movies);
    
    fetchArticles();
    fetchFeaturedArticles();
    fetchSources();
    
    // Fetch articles about recommended movies
    if (movies.length > 0) {
      fetchRecommendedArticles(movies);
    }
  }, []);

  // Refetch when source filter changes
  useEffect(() => {
    if (!loading) {
      fetchArticles();
    }
  }, [selectedSource]);

  const fetchArticles = async () => {
    try {
      setLoading(true);
      setError(null);

      const params = new URLSearchParams();
      params.append('limit', '12'); // Reduced from 20 for better performance
      if (selectedSource) {
        params.append('source', selectedSource);
      }

      const response = await fetch(`/api/articles?${params}`);
      const data = await response.json();

      if (data.error) {
        throw new Error(data.error);
      }

      setArticles(data.articles || []);
    } catch (err) {
      console.error('Error fetching articles:', err);
      setError('Failed to load articles. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const fetchRecommendedArticles = async (movies) => {
    if (!movies || movies.length === 0) return;
    
    try {
      setLoadingRecommended(true);
      
      // Search for articles about each recommended movie
      const allResults = [];
      
      // Take top 5 movies to avoid too many API calls
      const moviesToSearch = movies.slice(0, 5);
      
      for (const movie of moviesToSearch) {
        if (!movie.title) continue;
        
        const params = new URLSearchParams();
        params.append('search', movie.title);
        params.append('limit', '2'); // 2 articles per movie
        
        const response = await fetch(`/api/articles?${params}`);
        const data = await response.json();
        
        if (data.articles && data.articles.length > 0) {
          // Add movie context to each article for display
          const articlesWithContext = data.articles.map(article => ({
            ...article,
            relatedMovie: movie.title
          }));
          allResults.push(...articlesWithContext);
        }
      }
      
      // Remove duplicates based on article id
      const uniqueArticles = Array.from(
        new Map(allResults.map(article => [article.id, article])).values()
      );
      
      setRecommendedArticles(uniqueArticles.slice(0, 6)); // Top 6 articles
    } catch (err) {
      console.error('Error fetching recommended articles:', err);
    } finally {
      setLoadingRecommended(false);
    }
  };

  const fetchFeaturedArticles = async () => {
    try {
      const response = await fetch('/api/articles/featured?count=3');
      const data = await response.json();
      setFeaturedArticles(data.articles || []);
    } catch (err) {
      console.error('Error fetching featured articles:', err);
    }
  };

  const fetchSources = async () => {
    try {
      const response = await fetch('/api/articles/sources');
      const data = await response.json();
      setSources(data.sources || []);
    } catch (err) {
      console.error('Error fetching sources:', err);
    }
  };

  const handleSearch = async (e) => {
    e.preventDefault();
    
    if (!searchQuery.trim()) {
      fetchArticles();
      return;
    }

    try {
      setLoading(true);
      setError(null);

      const params = new URLSearchParams();
      params.append('search', searchQuery);
      params.append('limit', '20');

      const response = await fetch(`/api/articles?${params}`);
      const data = await response.json();

      if (data.error) {
        throw new Error(data.error);
      }

      setArticles(data.articles || []);
    } catch (err) {
      console.error('Error searching articles:', err);
      setError('Search failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleRefresh = async () => {
    setIsRefreshing(true);
    
    // Re-get recommended movies
    const movies = getRecommendedMovies();
    setRecommendedMovies(movies);
    
    const promises = [
      fetchArticles(),
      fetchFeaturedArticles()
    ];
    
    if (movies.length > 0) {
      promises.push(fetchRecommendedArticles(movies));
    }
    
    await Promise.all(promises);
    setIsRefreshing(false);
  };

  const handleSourceFilter = (sourceKey) => {
    setSelectedSource(sourceKey === selectedSource ? null : sourceKey);
  };

  const clearSearch = () => {
    setSearchQuery('');
    fetchArticles();
  };

  const handleScroll = () => {
    if (!scrollContainerRef.current) return;
    setShowBackToTop(scrollContainerRef.current.scrollTop > 500);
  };

  const scrollToTop = () => {
    if (!scrollContainerRef.current) return;
    scrollContainerRef.current.scrollTo({ top: 0, behavior: 'smooth' });
  };

  return (
    <div className="relative h-full">
      <div
        ref={scrollContainerRef}
        onScroll={handleScroll}
        className="h-full overflow-y-auto bg-gradient-to-b from-slate-900 to-slate-950"
      >
      <div className="max-w-7xl mx-auto px-4 py-8">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div className="flex items-center gap-3">
          <Newspaper className="text-purple-400" size={32} />
          <div>
            <h1 className="text-3xl font-bold text-white">Film Articles & News</h1>
            <p className="text-gray-400 text-sm mt-1">
              Latest stories from top movie blogs and publications
            </p>
          </div>
        </div>

        <button
          onClick={handleRefresh}
          disabled={isRefreshing}
          className="flex items-center gap-2 px-4 py-2 bg-purple-600 hover:bg-purple-700 
                     text-white rounded-lg transition-colors disabled:opacity-50"
        >
          <RefreshCw size={16} className={isRefreshing ? 'animate-spin' : ''} />
          Refresh
        </button>
      </div>

      {/* Search Bar */}
      <form onSubmit={handleSearch} className="mb-6">
        <div className="relative">
          <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 
                           text-gray-400" size={20} />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search articles by keyword..."
            className="w-full pl-12 pr-24 py-3 bg-gray-800/50 border border-gray-700 
                       rounded-lg text-white placeholder-gray-500 
                       focus:outline-none focus:border-purple-500"
          />
          {searchQuery && (
            <button
              type="button"
              onClick={clearSearch}
              className="absolute right-2 top-1/2 transform -translate-y-1/2 
                         px-3 py-1.5 text-sm text-gray-400 hover:text-white"
            >
              Clear
            </button>
          )}
        </div>
      </form>

      {/* Source Filters */}
      {sources.length > 0 && (
        <div className="mb-8">
          <div className="flex items-center gap-2 mb-3">
            <Filter size={16} className="text-gray-400" />
            <span className="text-sm text-gray-400">Filter by source:</span>
          </div>
          <div className="flex flex-wrap gap-2">
            {sources.map((source) => (
              <button
                key={source.key}
                onClick={() => handleSourceFilter(source.key)}
                className={`px-3 py-1.5 text-sm rounded-full transition-all ${
                  selectedSource === source.key
                    ? 'bg-purple-600 text-white'
                    : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
                }`}
              >
                {source.name}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Articles About Recommended Movies (Primary Feature) */}
      {!searchQuery && !selectedSource && recommendedArticles.length > 0 && (
        <div className="mb-12">
          <h2 className="text-xl font-bold text-white mb-2 flex items-center gap-2">
            <span className="text-purple-400">🎬</span> Featured: About Your Recommendations
          </h2>
          <p className="text-sm text-gray-400 mb-4">
            Articles and discussions about the movies CineBot recommended to you
          </p>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {recommendedArticles.map((article, index) => (
              <ArticleCard key={article.id || index} article={article} />
            ))}
          </div>
        </div>
      )}

      {/* No Recommendations Yet - Guide Users */}
      {!searchQuery && !selectedSource && recommendedArticles.length === 0 && recommendedMovies.length === 0 && (
        <div className="mb-12 p-6 bg-gradient-to-r from-purple-900/20 to-blue-900/20 
                        border border-purple-500/30 rounded-xl">
          <h3 className="text-lg font-semibold text-white mb-2 flex items-center gap-2">
            <span className="text-purple-400">💡</span> Get Personalized Articles
          </h3>
          <p className="text-gray-300 text-sm">
            Ask CineBot for movie recommendations in the <span className="text-purple-400 font-semibold">Movie Finder</span> tab, 
            and we'll show you articles and discussions about those movies here!
          </p>
        </div>
      )}

      {/* Industry Featured Stories */}
      {!searchQuery && !selectedSource && featuredArticles.length > 0 && (
        <div className="mb-12">
          <h2 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
            <span className="text-yellow-400">⭐</span> Industry News
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {featuredArticles.map((article, index) => (
              <ArticleCard key={article.id || index} article={article} />
            ))}
          </div>
        </div>
      )}

      {/* Error State */}
      {error && (
        <div className="flex items-center gap-3 p-4 bg-red-900/20 border border-red-500/50 
                        rounded-lg text-red-400 mb-6">
          <AlertCircle size={20} />
          <span>{error}</span>
        </div>
      )}

      {/* Loading State */}
      {loading && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {[...Array(6)].map((_, i) => (
            <SkeletonLoader key={i} height="h-96" />
          ))}
        </div>
      )}

      {/* Articles Grid */}
      {!loading && articles.length > 0 && (
        <div>
          <h2 className="text-xl font-bold text-white mb-4">
            {searchQuery ? `Search Results (${articles.length})` : 
             selectedSource ? `${sources.find(s => s.key === selectedSource)?.name} Articles` : 
             'Latest Articles'}
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {articles.map((article, index) => (
              <ArticleCard key={article.id || index} article={article} />
            ))}
          </div>
        </div>
      )}

      {/* Empty State */}
      {!loading && articles.length === 0 && !error && (
        <div className="text-center py-16">
          <Newspaper size={64} className="mx-auto text-gray-600 mb-4" />
          <h3 className="text-xl font-semibold text-gray-400 mb-2">
            No articles found
          </h3>
          <p className="text-gray-500">
            {searchQuery 
              ? 'Try a different search term' 
              : 'Check back later for new content'}
          </p>
        </div>
      )}
      </div>
      </div>

      {showBackToTop && (
        <button
          type="button"
          onClick={scrollToTop}
          className="sm:hidden fixed bottom-24 right-4 z-30 rounded-full bg-purple-600 text-white p-3 shadow-lg border border-purple-400/30 hover:bg-purple-500 transition-colors"
          aria-label="Back to top"
        >
          <ChevronUp size={20} />
        </button>
      )}
    </div>
  );
};

export default ArticlesTab;
