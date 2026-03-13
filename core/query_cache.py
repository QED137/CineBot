"""
Query result caching for production performance optimization.
Uses LRU cache to store frequently accessed query results.
"""

import logging
import hashlib
import json
from functools import lru_cache
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class QueryCache:
    """Simple in-memory cache for query results with TTL."""
    
    def __init__(self, max_size: int = 1000, ttl_minutes: int = 30):
        self.cache = {}
        self.max_size = max_size
        self.ttl = timedelta(minutes=ttl_minutes)
        logger.info(f"QueryCache initialized: max_size={max_size}, ttl={ttl_minutes}min")
    
    def _make_key(self, query: str, params: Optional[Dict] = None) -> str:
        """Create a cache key from query and parameters."""
        cache_string = query + json.dumps(params or {}, sort_keys=True)
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def get(self, query: str, params: Optional[Dict] = None) -> Optional[Any]:
        """Get cached result if available and not expired."""
        key = self._make_key(query, params)
        
        if key in self.cache:
            result, timestamp = self.cache[key]
            
            # Check if expired
            if datetime.now() - timestamp < self.ttl:
                logger.debug(f"Cache HIT for query key: {key[:8]}...")
                return result
            else:
                # Remove expired entry
                logger.debug(f"Cache EXPIRED for query key: {key[:8]}...")
                del self.cache[key]
        
        logger.debug(f"Cache MISS for query key: {key[:8]}...")
        return None
    
    def set(self, query: str, result: Any, params: Optional[Dict] = None):
        """Store result in cache."""
        key = self._make_key(query, params)
        
        # Simple LRU: if cache is full, remove oldest entry
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]
            logger.debug(f"Cache full, evicted oldest entry")
        
        self.cache[key] = (result, datetime.now())
        logger.debug(f"Cache SET for query key: {key[:8]}...")
    
    def clear(self):
        """Clear all cached entries."""
        self.cache.clear()
        logger.info("Cache cleared")
    
    def size(self) -> int:
        """Return current cache size."""
        return len(self.cache)


# Global cache instance
_genre_cache = QueryCache(max_size=100, ttl_minutes=60)  # Genre queries cached for 1 hour
_vector_cache = QueryCache(max_size=500, ttl_minutes=30)  # Vector queries cached for 30 min
_graph_cache = QueryCache(max_size=200, ttl_minutes=45)   # Graph queries cached for 45 min


def get_genre_cache() -> QueryCache:
    """Get genre query cache instance."""
    return _genre_cache


def get_vector_cache() -> QueryCache:
    """Get vector search cache instance."""
    return _vector_cache


def get_graph_cache() -> QueryCache:
    """Get graph query cache instance."""
    return _graph_cache


def clear_all_caches():
    """Clear all query caches."""
    _genre_cache.clear()
    _vector_cache.clear()
    _graph_cache.clear()
    logger.info("All caches cleared")


# LRU cache for movie embedding lookups (in-process cache)
@lru_cache(maxsize=1000)
def cached_movie_similarity(query_embedding_hash: str, top_k: int) -> str:
    """
    Placeholder for caching movie similarity results.
    The actual implementation would be in core_rag.py
    Returns a hash that can be used to look up cached results.
    """
    return f"{query_embedding_hash}_{top_k}"
