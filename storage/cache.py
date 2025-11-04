"""
Redis caching layer for deduplication and performance optimization
"""
from redis import asyncio as aioredis
from typing import Optional, Any, List
import json
import hashlib
import logging
from datetime import timedelta

from config.settings import settings

logger = logging.getLogger(__name__)


# ============================================
# REDIS CONNECTION
# ============================================

class RedisCache:
    """Async Redis cache manager"""

    def __init__(self):
        self.redis: Optional[aioredis.Redis] = None
        self._pool: Optional[aioredis.ConnectionPool] = None

    async def connect(self):
        """Initialize Redis connection pool"""
        try:
            self._pool = aioredis.ConnectionPool.from_url(
                settings.REDIS_URL,
                max_connections=settings.REDIS_MAX_CONNECTIONS,
                decode_responses=True
            )
            self.redis = aioredis.Redis(connection_pool=self._pool)

            # Test connection
            await self.redis.ping()
            logger.info("Redis connection established")
        except Exception as e:
            logger.error(f"Error connecting to Redis: {e}")
            raise

    async def close(self):
        """Close Redis connections"""
        if self.redis:
            await self.redis.close()
            await self._pool.disconnect()
            logger.info("Redis connections closed")

    async def health_check(self) -> bool:
        """Check Redis health"""
        try:
            return await self.redis.ping()
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return False


# Global cache instance
cache = RedisCache()


# ============================================
# URL DEDUPLICATION
# ============================================

def url_hash(url: str) -> str:
    """Generate SHA256 hash for URL"""
    return hashlib.sha256(url.strip().lower().encode()).hexdigest()


async def is_url_scraped(url: str) -> bool:
    """
    Check if URL has been scraped recently

    Returns:
        True if URL is in cache (recently scraped)
    """
    try:
        key = f"scraped:url:{url_hash(url)}"
        exists = await cache.redis.exists(key)
        return bool(exists)
    except Exception as e:
        logger.error(f"Error checking URL cache: {e}")
        return False


async def mark_url_scraped(url: str, ttl: int = None):
    """
    Mark URL as scraped in cache

    Args:
        url: The URL
        ttl: Time to live in seconds (default: REDIS_CACHE_TTL)
    """
    try:
        key = f"scraped:url:{url_hash(url)}"
        ttl = ttl or settings.REDIS_CACHE_TTL
        await cache.redis.setex(key, ttl, url)
        logger.debug(f"URL marked as scraped: {url}")
    except Exception as e:
        logger.error(f"Error marking URL as scraped: {e}")


async def remove_url_from_cache(url: str):
    """Remove URL from scraped cache"""
    try:
        key = f"scraped:url:{url_hash(url)}"
        await cache.redis.delete(key)
    except Exception as e:
        logger.error(f"Error removing URL from cache: {e}")


# ============================================
# CONTENT DEDUPLICATION
# ============================================

def content_hash(content: str) -> str:
    """Generate SHA256 hash for content"""
    return hashlib.sha256(content.encode()).hexdigest()


async def is_content_duplicate(content: str) -> bool:
    """Check if content hash exists in cache"""
    try:
        key = f"content:hash:{content_hash(content)}"
        exists = await cache.redis.exists(key)
        return bool(exists)
    except Exception as e:
        logger.error(f"Error checking content cache: {e}")
        return False


async def cache_content_hash(content: str, ttl: int = None):
    """Store content hash in cache"""
    try:
        key = f"content:hash:{content_hash(content)}"
        ttl = ttl or settings.REDIS_CACHE_TTL
        await cache.redis.setex(key, ttl, "1")
    except Exception as e:
        logger.error(f"Error caching content hash: {e}")


# ============================================
# GENERIC KEY-VALUE CACHE
# ============================================

async def get_cached(key: str) -> Optional[Any]:
    """Get value from cache"""
    try:
        value = await cache.redis.get(key)
        if value:
            return json.loads(value)
        return None
    except Exception as e:
        logger.error(f"Error getting cached value for {key}: {e}")
        return None


async def set_cached(key: str, value: Any, ttl: int = None):
    """Set value in cache with TTL"""
    try:
        ttl = ttl or settings.REDIS_CACHE_TTL
        serialized = json.dumps(value)
        await cache.redis.setex(key, ttl, serialized)
    except Exception as e:
        logger.error(f"Error setting cached value for {key}: {e}")


async def delete_cached(key: str):
    """Delete value from cache"""
    try:
        await cache.redis.delete(key)
    except Exception as e:
        logger.error(f"Error deleting cached key {key}: {e}")


# ============================================
# RATE LIMITING
# ============================================

async def is_rate_limited(domain: str, max_requests: int = 10, window: int = 60) -> bool:
    """
    Check if domain is rate limited

    Args:
        domain: The domain name
        max_requests: Max requests per window
        window: Time window in seconds

    Returns:
        True if rate limited
    """
    try:
        key = f"ratelimit:{domain}"
        count = await cache.redis.incr(key)

        if count == 1:
            # First request, set expiration
            await cache.redis.expire(key, window)

        return count > max_requests
    except Exception as e:
        logger.error(f"Error checking rate limit: {e}")
        return False


async def get_rate_limit_remaining(domain: str) -> int:
    """Get remaining requests for domain"""
    try:
        key = f"ratelimit:{domain}"
        count = await cache.redis.get(key)
        return int(count) if count else 0
    except Exception as e:
        logger.error(f"Error getting rate limit: {e}")
        return 0


# ============================================
# JOB QUEUE HELPERS
# ============================================

async def enqueue_url(url: str, priority: int = 0):
    """
    Add URL to scraping queue (sorted set by priority)

    Higher priority = processed first
    """
    try:
        await cache.redis.zadd("scrape:queue", {url: priority})
        logger.debug(f"URL enqueued with priority {priority}: {url}")
    except Exception as e:
        logger.error(f"Error enqueuing URL: {e}")


async def dequeue_url() -> Optional[str]:
    """Get next URL from queue (highest priority)"""
    try:
        # ZPOPMAX returns [(url, score)]
        result = await cache.redis.zpopmax("scrape:queue", count=1)
        if result:
            return result[0][0]  # Return URL
        return None
    except Exception as e:
        logger.error(f"Error dequeuing URL: {e}")
        return None


async def get_queue_size() -> int:
    """Get number of URLs in queue"""
    try:
        return await cache.redis.zcard("scrape:queue")
    except Exception as e:
        logger.error(f"Error getting queue size: {e}")
        return 0


async def clear_queue():
    """Clear scraping queue"""
    try:
        await cache.redis.delete("scrape:queue")
        logger.info("Scraping queue cleared")
    except Exception as e:
        logger.error(f"Error clearing queue: {e}")


# ============================================
# SESSION MANAGEMENT
# ============================================

async def store_session(session_id: str, data: dict, ttl: int = 3600):
    """Store session data"""
    try:
        key = f"session:{session_id}"
        await cache.redis.hset(key, mapping=data)
        await cache.redis.expire(key, ttl)
    except Exception as e:
        logger.error(f"Error storing session: {e}")


async def get_session(session_id: str) -> Optional[dict]:
    """Get session data"""
    try:
        key = f"session:{session_id}"
        data = await cache.redis.hgetall(key)
        return dict(data) if data else None
    except Exception as e:
        logger.error(f"Error getting session: {e}")
        return None


async def delete_session(session_id: str):
    """Delete session"""
    try:
        key = f"session:{session_id}"
        await cache.redis.delete(key)
    except Exception as e:
        logger.error(f"Error deleting session: {e}")


# ============================================
# STATISTICS
# ============================================

async def increment_stat(stat_name: str, amount: int = 1):
    """Increment a statistic counter"""
    try:
        key = f"stats:{stat_name}"
        await cache.redis.incrby(key, amount)
    except Exception as e:
        logger.error(f"Error incrementing stat {stat_name}: {e}")


async def get_stat(stat_name: str) -> int:
    """Get statistic value"""
    try:
        key = f"stats:{stat_name}"
        value = await cache.redis.get(key)
        return int(value) if value else 0
    except Exception as e:
        logger.error(f"Error getting stat {stat_name}: {e}")
        return 0


async def get_all_stats() -> dict:
    """Get all statistics"""
    try:
        keys = await cache.redis.keys("stats:*")
        stats = {}
        for key in keys:
            stat_name = key.replace("stats:", "")
            value = await cache.redis.get(key)
            stats[stat_name] = int(value) if value else 0
        return stats
    except Exception as e:
        logger.error(f"Error getting all stats: {e}")
        return {}


# ============================================
# CACHE UTILITIES
# ============================================

async def flush_cache():
    """Flush entire cache (use with caution!)"""
    try:
        await cache.redis.flushdb()
        logger.warning("Redis cache flushed")
    except Exception as e:
        logger.error(f"Error flushing cache: {e}")


async def get_cache_info() -> dict:
    """Get Redis server info"""
    try:
        info = await cache.redis.info()
        return {
            "connected_clients": info.get("connected_clients"),
            "used_memory_human": info.get("used_memory_human"),
            "total_commands_processed": info.get("total_commands_processed"),
            "keyspace": await cache.redis.dbsize(),
        }
    except Exception as e:
        logger.error(f"Error getting cache info: {e}")
        return {}
