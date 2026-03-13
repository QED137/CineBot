# Redis Cache Setup Guide

## What is Redis?

Redis is an in-memory data store that provides **persistent, fast caching** across server restarts. Unlike in-memory caches that are lost when the server restarts, Redis stores data on disk while keeping it in memory for speed.

## Current Implementation

CineBot now uses a **smart hybrid cache** that:
- [OK] Automatically uses Redis if available (production)
- [OK] Falls back to in-memory cache if Redis unavailable (development)
- [OK] No code changes needed - works in both scenarios

## Setup Options

### Option 1: Docker (Recommended)

**Quick start with Docker:**
```bash
# Start Redis container
docker run -d --name cinebot-redis -p 6379:6379 redis:alpine

# Verify it's running
docker ps | grep redis

# Restart your FastAPI backend
pkill -f uvicorn
uvicorn app_fastapi:app --host 0.0.0.0 --port 8000 --reload
```

**Check cache health:**
```bash
curl http://localhost:8000/api/health
```

### Option 2: Install Redis Locally

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install redis-server
sudo systemctl start redis-server
sudo systemctl enable redis-server
```

**macOS:**
```bash
brew install redis
brew services start redis
```

**Windows:**
Download from: https://github.com/tporadowski/redis/releases

### Option 3: Use Managed Redis (Production)

**Cloud providers:**
- **AWS**: ElastiCache for Redis
- **Azure**: Azure Cache for Redis
- **Google Cloud**: Memorystore for Redis
- **DigitalOcean**: Managed Redis
- **Redis Cloud**: Free tier available

**Update your `.env`:**
```bash
REDIS_HOST=your-redis-host.cloud.com
REDIS_PORT=6379
REDIS_PASSWORD=your-password
```

## Monitoring Cache Performance

### 1. Health Check
```bash
curl http://localhost:8000/api/health
```

**Response:**
```json
{
  "status": "healthy",
  "cache": {
    "genre_cache": {
      "redis_available": true,
      "redis_connected": true,
      "using_fallback": false,
      "cache_size": 12,
      "redis_ping": true
    },
    "vector_cache": {...},
    "graph_cache": {...}
  },
  "backend": "FastAPI",
  "version": "2.0.0"
}
```

### 2. Cache Statistics
```bash
curl http://localhost:8000/api/cache/stats
```

**Shows:**
- Number of cached entries
- Redis connection status
- Cache hit/miss ratio (in logs)

### 3. Clear Cache
```bash
curl -X POST http://localhost:8000/api/cache/clear
```

## Cache Configuration

**Cache TTL (Time To Live):**
- Genre queries: 3600 seconds (1 hour)
- Vector searches: 1800 seconds (30 minutes)
- Graph queries: 2700 seconds (45 minutes)

**To customize**, edit `/workspaces/CineBot/core/redis_cache.py`:
```python
_genre_cache = RedisCache(ttl_seconds=7200, prefix='cinebot:genre')  # 2 hours
```

## Performance Comparison

### Without Redis (In-Memory Cache)
- [OK] Fast for single server
- [FAIL] Lost on server restart
- [FAIL] Not shared across multiple servers
- [FAIL] Limited by server RAM

### With Redis
- [OK] Persists across restarts
- [OK] Shared across multiple servers (horizontal scaling)
- [OK] Dedicated memory management
- [OK] Can handle millions of entries
- [OK] Built-in TTL and eviction policies

## How It Works

### First Query (Cold Start)
```
User: "romantic movies"
   ↓
1. Check Redis → MISS (not cached)
2. Execute Neo4j query (5-15 seconds)
3. Store result in Redis with 1-hour TTL
4. Return to user
```

### Subsequent Queries (Cache Hit)
```
User: "romantic movies"
   ↓
1. Check Redis → HIT! [TARGET]
2. Return cached result (~5ms)
3. Skip Neo4j entirely
```

### After 1 Hour (TTL Expired)
```
User: "romantic movies"
   ↓
1. Check Redis → EXPIRED (auto-deleted)
2. Execute fresh Neo4j query
3. Update Redis cache
4. Return to user
```

## Fallback Behavior

**If Redis is unavailable:**
```python
[WARNING] Redis connection failed: Connection refused. Using in-memory fallback.
```

CineBot will automatically:
1. Log a warning
2. Switch to in-memory cache
3. Continue working normally
4. Try Redis again on next restart

**No downtime!** The app works with or without Redis.

## Production Checklist

- [ ] Redis installed and running
- [ ] Redis accessible from app server
- [ ] Firewall allows port 6379
- [ ] Redis password configured (if production)
- [ ] Monitoring alerts for Redis downtime
- [ ] Backup strategy for Redis data (optional)
- [ ] Resource limits configured (maxmemory)

## Troubleshooting

### [FAIL] Connection refused
```bash
# Check if Redis is running
redis-cli ping
# Should return: PONG

# If not running, start it
docker start cinebot-redis
# OR
sudo systemctl start redis-server
```

### [FAIL] Using fallback cache
**Check logs:**
```
[WARNING] Redis not installed. Using in-memory cache (not persistent).
```

**Solution:** Install Redis (see Setup Options above)

### [FAIL] Cache not clearing
```bash
# Manual Redis clear
redis-cli FLUSHALL

# Or via API
curl -X POST http://localhost:8000/api/cache/clear
```

### [SEARCH] Debug Cache Behavior

**View cache logs:**
```bash
# In app logs, you'll see:
 Redis HIT: cinebot:genre:abc123
 Redis MISS: cinebot:genre:xyz789
[BACKUP] Redis SET: cinebot:genre:abc123 (TTL: 3600s)
```

**Legend:**
-  = Cache hit (fast!)
-  = Cache miss (slow query)
- [BACKUP] = Data cached for future
-  = Fallback cache used

## Redis Commands (Useful)

```bash
# Connect to Redis CLI
redis-cli

# View all keys
KEYS cinebot:*

# Get specific key
GET "cinebot:genre:abc123"

# Check TTL (time remaining)
TTL "cinebot:genre:abc123"

# Delete specific key
DEL "cinebot:genre:abc123"

# Clear all CineBot keys
KEYS cinebot:* | xargs redis-cli DEL

# Monitor Redis activity (live)
MONITOR

# Get Redis info
INFO
```

## Environment Variables

**Optional Redis configuration in `.env`:**
```bash
# Redis connection (defaults to localhost:6379)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=  # Leave empty if no password

# Cache TTL (optional, in seconds)
GENRE_CACHE_TTL=3600
VECTOR_CACHE_TTL=1800
GRAPH_CACHE_TTL=2700
```

## Summary

Your CineBot now has **production-grade caching** with:

[OK] **Automatic Redis detection** - uses Redis if available
[OK] **Graceful fallback** - works without Redis too
[OK] **Monitoring endpoints** - check health and stats
[OK] **Performance boost** - 1000x faster on cache hits

**Next steps:**
1. Install Redis (Docker/local/cloud)
2. Check `/api/health` to verify connection
3. Test performance with repeated queries
4. Monitor cache stats at `/api/cache/stats`

Your queries will now be **blazing fast** and **persistent** across restarts! [DEPLOY]
