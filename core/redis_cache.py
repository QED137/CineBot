"""
Redis-based persistent caching with automatic fallback to in-memory cache.
"""

import logging
import json
import hashlib
from typing import Optional, Any, Dict
from datetime import timedelta
from config import settings

logger = logging.getLogger(__name__)

# Try to import Redis, but don't fail if not available
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("Redis not installed. Install with: pip install redis")


class RedisCache:
    """
    Redis-based cache with automatic fallback to in-memory if Redis unavailable.
    """
    
    def __init__(
        self, 
        host: str = 'localhost',
        port: int = 6379,
        db: int = 0,
        ttl_seconds: int = 3600,
        prefix: str = 'cinebot'
    ):
        self.ttl_seconds = ttl_seconds
        self.prefix = prefix
        self.redis_client = None
        self.fallback_cache = {}  # In-memory fallback
        
        # Try to connect to Redis
        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.Redis(
                    host=host,
                    port=port,
                    db=db,
                    decode_responses=True,
                    socket_connect_timeout=2,
                    socket_timeout=2
                )
                # Test connection
                self.redis_client.ping()
                logger.info(f"Redis connected at {host}:{port} (db={db})")
            except (redis.ConnectionError, redis.TimeoutError) as e:
                logger.warning(f"Redis connection failed: {e}. Using in-memory fallback.")
                self.redis_client = None
            except Exception as e:
                logger.error(f"Redis error: {e}. Using in-memory fallback.")
                self.redis_client = None
        else:
            logger.warning("Redis not installed. Using in-memory cache (not persistent).")
    
    def _make_key(self, key: str, params: Optional[Dict] = None) -> str:
        """Create a cache key with prefix and hash of parameters."""
        if params:
            param_str = json.dumps(params, sort_keys=True)
            param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
            return f"{self.prefix}:{key}:{param_hash}"
        return f"{self.prefix}:{key}"
    
    def get(self, key: str, params: Optional[Dict] = None) -> Optional[Any]:
        """Get value from cache (Redis or fallback)."""
        cache_key = self._make_key(key, params)
        
        # Try Redis first
        if self.redis_client:
            try:
                value = self.redis_client.get(cache_key)
                if value:
                    logger.debug(f"🟢 Redis HIT: {cache_key}")
                    return json.loads(value)
                logger.debug(f"⚪ Redis MISS: {cache_key}")
                return None
            except Exception as e:
                logger.error(f"Redis get error: {e}. Using fallback.")
                self.redis_client = None  # Disable Redis on error
        
        # Fallback to in-memory
        if cache_key in self.fallback_cache:
            logger.debug(f"🔵 Memory HIT: {cache_key}")
            return self.fallback_cache[cache_key]
        
        logger.debug(f"⚪ Memory MISS: {cache_key}")
        return None
    
    def set(self, key: str, value: Any, params: Optional[Dict] = None, ttl: Optional[int] = None):
        """Set value in cache (Redis or fallback)."""
        cache_key = self._make_key(key, params)
        ttl_to_use = ttl if ttl is not None else self.ttl_seconds
        
        # Try Redis first
        if self.redis_client:
            try:
                self.redis_client.setex(
                    cache_key,
                    ttl_to_use,
                    json.dumps(value)
                )
                logger.debug(f"Redis SET: {cache_key} (TTL: {ttl_to_use}s)")
                return
            except Exception as e:
                logger.error(f"Redis set error: {e}. Using fallback.")
                self.redis_client = None  # Disable Redis on error
        
        # Fallback to in-memory (no TTL support in fallback)
        self.fallback_cache[cache_key] = value
        logger.debug(f"Memory SET: {cache_key}")
        
        # Limit fallback cache size
        if len(self.fallback_cache) > 1000:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self.fallback_cache))
            del self.fallback_cache[oldest_key]
    
    def delete(self, key: str, params: Optional[Dict] = None):
        """Delete key from cache."""
        cache_key = self._make_key(key, params)
        
        if self.redis_client:
            try:
                self.redis_client.delete(cache_key)
                logger.debug(f"Redis DELETE: {cache_key}")
            except Exception as e:
                logger.error(f"Redis delete error: {e}")
        
        if cache_key in self.fallback_cache:
            del self.fallback_cache[cache_key]
            logger.debug(f"Memory DELETE: {cache_key}")
    
    def clear(self):
        """Clear all cache entries with our prefix."""
        if self.redis_client:
            try:
                # Delete all keys with our prefix
                pattern = f"{self.prefix}:*"
                keys = self.redis_client.keys(pattern)
                if keys:
                    self.redis_client.delete(*keys)
                    logger.info(f"Redis cleared {len(keys)} keys")
            except Exception as e:
                logger.error(f"Redis clear error: {e}")
        
        self.fallback_cache.clear()
        logger.info("Memory cache cleared")
    
    def size(self) -> int:
        """Return current cache size."""
        if self.redis_client:
            try:
                pattern = f"{self.prefix}:*"
                return len(self.redis_client.keys(pattern))
            except Exception as e:
                logger.error(f"Redis size error: {e}")
        
        return len(self.fallback_cache)
    
    def health_check(self) -> Dict[str, Any]:
        """Check cache health and return status."""
        status = {
            'redis_available': REDIS_AVAILABLE,
            'redis_connected': self.redis_client is not None,
            'using_fallback': self.redis_client is None,
            'cache_size': self.size()
        }
        
        if self.redis_client:
            try:
                self.redis_client.ping()
                status['redis_ping'] = True
            except Exception as e:
                status['redis_ping'] = False
                status['redis_error'] = str(e)
        
        return status


# Global cache instances with Redis backend
_genre_cache = RedisCache(
    host=settings.REDIS_HOST,
    port=settings.REDIS_PORT,
    db=settings.REDIS_DB,
    ttl_seconds=3600,
    prefix='cinebot:genre'
)  # 1 hour
_vector_cache = RedisCache(
    host=settings.REDIS_HOST,
    port=settings.REDIS_PORT,
    db=settings.REDIS_DB,
    ttl_seconds=1800,
    prefix='cinebot:vector'
)  # 30 min
_graph_cache = RedisCache(
    host=settings.REDIS_HOST,
    port=settings.REDIS_PORT,
    db=settings.REDIS_DB,
    ttl_seconds=2700,
    prefix='cinebot:graph'
)  # 45 min


def get_genre_cache() -> RedisCache:
    """Get genre query cache instance."""
    return _genre_cache


def get_vector_cache() -> RedisCache:
    """Get vector search cache instance."""
    return _vector_cache


def get_graph_cache() -> RedisCache:
    """Get graph query cache instance."""
    return _graph_cache


def clear_all_caches():
    """Clear all query caches."""
    _genre_cache.clear()
    _vector_cache.clear()
    _graph_cache.clear()
    logger.info("All caches cleared")


def get_cache_health() -> Dict[str, Any]:
    """Get health status of all caches."""
    return {
        'genre_cache': _genre_cache.health_check(),
        'vector_cache': _vector_cache.health_check(),
        'graph_cache': _graph_cache.health_check()
    }
