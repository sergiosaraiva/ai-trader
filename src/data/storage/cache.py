"""Cache manager for fast data access."""

import hashlib
import json
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Any

import pandas as pd


class CacheManager:
    """Manage data caching for performance optimization."""

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        redis_url: Optional[str] = None,
        default_ttl: int = 3600,
    ):
        """
        Initialize cache manager.

        Args:
            cache_dir: Directory for file-based cache
            redis_url: Redis connection URL for distributed cache
            default_ttl: Default time-to-live in seconds
        """
        self.cache_dir = cache_dir or Path("cache")
        self.redis_url = redis_url
        self.default_ttl = default_ttl
        self._redis = None

        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_redis(self):
        """Get Redis connection."""
        if self._redis is None and self.redis_url:
            try:
                import redis

                self._redis = redis.from_url(self.redis_url)
            except ImportError:
                pass
        return self._redis

    def _generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        key_data = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True, default=str)
        return hashlib.md5(key_data.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        # Try Redis first
        redis_client = self._get_redis()
        if redis_client:
            try:
                data = redis_client.get(key)
                if data:
                    return pickle.loads(data)
            except Exception:
                pass

        # Fall back to file cache
        cache_file = self.cache_dir / f"{key}.pkl"
        meta_file = self.cache_dir / f"{key}.meta"

        if cache_file.exists() and meta_file.exists():
            # Check TTL
            with open(meta_file, "r") as f:
                meta = json.load(f)
            expires_at = datetime.fromisoformat(meta["expires_at"])

            if datetime.now() < expires_at:
                with open(cache_file, "rb") as f:
                    return pickle.load(f)
            else:
                # Expired, remove files
                cache_file.unlink(missing_ok=True)
                meta_file.unlink(missing_ok=True)

        return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds

        Returns:
            True if successful
        """
        ttl = ttl or self.default_ttl

        # Try Redis first
        redis_client = self._get_redis()
        if redis_client:
            try:
                redis_client.setex(key, ttl, pickle.dumps(value))
                return True
            except Exception:
                pass

        # Fall back to file cache
        cache_file = self.cache_dir / f"{key}.pkl"
        meta_file = self.cache_dir / f"{key}.meta"

        with open(cache_file, "wb") as f:
            pickle.dump(value, f)

        meta = {
            "created_at": datetime.now().isoformat(),
            "expires_at": (datetime.now() + timedelta(seconds=ttl)).isoformat(),
            "ttl": ttl,
        }
        with open(meta_file, "w") as f:
            json.dump(meta, f)

        return True

    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        # Try Redis
        redis_client = self._get_redis()
        if redis_client:
            try:
                redis_client.delete(key)
            except Exception:
                pass

        # Delete file cache
        cache_file = self.cache_dir / f"{key}.pkl"
        meta_file = self.cache_dir / f"{key}.meta"
        cache_file.unlink(missing_ok=True)
        meta_file.unlink(missing_ok=True)

        return True

    def clear(self) -> int:
        """Clear all cached data."""
        count = 0

        # Clear Redis
        redis_client = self._get_redis()
        if redis_client:
            try:
                redis_client.flushdb()
            except Exception:
                pass

        # Clear file cache
        for f in self.cache_dir.glob("*.pkl"):
            f.unlink()
            count += 1
        for f in self.cache_dir.glob("*.meta"):
            f.unlink()

        return count

    def cache_dataframe(
        self,
        key: str,
        df: pd.DataFrame,
        ttl: Optional[int] = None,
    ) -> bool:
        """Cache a pandas DataFrame."""
        return self.set(key, df, ttl)

    def get_dataframe(self, key: str) -> Optional[pd.DataFrame]:
        """Get cached DataFrame."""
        return self.get(key)

    def cached(self, ttl: Optional[int] = None):
        """
        Decorator for caching function results.

        Usage:
            @cache_manager.cached(ttl=3600)
            def expensive_function(x, y):
                return x + y
        """

        def decorator(func):
            def wrapper(*args, **kwargs):
                key = f"{func.__name__}_{self._generate_key(*args, **kwargs)}"
                result = self.get(key)
                if result is not None:
                    return result
                result = func(*args, **kwargs)
                self.set(key, result, ttl)
                return result

            return wrapper

        return decorator
