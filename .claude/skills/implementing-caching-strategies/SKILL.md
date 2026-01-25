---
name: implementing-caching-strategies
description: This skill should be used when the user asks to "add caching", "implement cache invalidation", "cache expensive operations", "add TTL cache", "hash-based caching". Implements caching with hash-based invalidation, TTL expiration, and value-change detection for services and feature calculations.
version: 1.0.0
---

# Implementing Caching Strategies

## Quick Reference

- Use hash-based invalidation for config-dependent caches
- Use TTL for time-sensitive data with `timedelta`
- Use value-change detection for smart regeneration
- Clean up expired entries to prevent memory leaks
- Include cache metadata in responses (cached, generated_at)

## When to Use

- Caching expensive calculations (feature selection, model predictions)
- Caching LLM responses (explanations with value-change invalidation)
- Caching file-based data (config hashes)
- Services with expensive initialization

## When NOT to Use

- Frequently changing data (use direct computation)
- Small/fast operations (overhead not worth it)
- User-specific data without proper isolation

## Implementation Guide

```
What type of cache invalidation?
├─ Config-based → Use hash of config parameters
│   └─ Include hash in cache filename/key
│   └─ Auto-invalidate when config changes
├─ Time-based → Use TTL with timedelta
│   └─ Store generated_at timestamp
│   └─ Clean up expired entries periodically
└─ Value-based → Compare current vs cached values
    └─ Define threshold for significant change
    └─ Regenerate when change exceeds threshold

Cache storage strategy?
├─ File-based → Use JSON files with hash in filename
│   └─ Validate cached data structure on load
├─ In-memory → Use dict with thread lock
│   └─ Add cleanup method for expired entries
└─ Both → File for persistence, memory for speed
```

## Examples

**Example 1: Hash-Based Cache Invalidation**

```python
# From: backend/src/models/feature_selection/manager.py:24-80
import hashlib
import json
from pathlib import Path
from typing import Dict, Optional

CACHE_HASH_LENGTH = 8

class FeatureSelectionManager:
    """Manager with config-hash based caching."""

    def __init__(self, config: RFECVConfig, cache_dir: Path):
        self.config = config
        self.cache_dir = cache_dir
        self._config_hash = self._compute_config_hash()

    def _compute_config_hash(self) -> str:
        """Compute hash of config for cache invalidation."""
        config_dict = {
            "step": self.config.step,
            "min_features": self.config.min_features_to_select,
            "cv": self.config.cv,
            "scoring": self.config.scoring,
        }
        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:CACHE_HASH_LENGTH]

    def _get_cache_path(self, timeframe: str) -> Path:
        """Get cache path including config hash."""
        return self.cache_dir / f"{timeframe}_rfecv_{self._config_hash}.json"
```

**Explanation**: Hash computed from relevant config parameters. `sort_keys=True` ensures deterministic hashing. Hash included in filename for automatic versioning.

**Example 2: Cache Loading with Validation**

```python
# From: backend/src/models/feature_selection/manager.py:82-120
def _load_from_cache(
    self,
    timeframe: str,
    n_features: int,
) -> Optional[Dict]:
    """Load cached selection if valid."""
    if not self.config.cache_enabled:
        return None

    cache_path = self._get_cache_path(timeframe)
    if not cache_path.exists():
        return None

    try:
        with open(cache_path, "r") as f:
            cached = json.load(f)

        # CRITICAL: Validate feature count matches
        if cached.get("n_original_features") != n_features:
            logger.warning(
                f"Cache invalidated for {timeframe}: "
                f"feature count changed ({cached.get('n_original_features')} → {n_features})"
            )
            cache_path.unlink()
            return None

        logger.info(f"Loaded cached selection for {timeframe}")
        return cached

    except Exception as e:
        logger.warning(f"Could not load cache: {e}")
        return None
```

**Explanation**: Validate cached data matches current state. Remove invalid cache files. Log cache hits and invalidations.

**Example 3: TTL-Based Caching with Value Change Detection**

```python
# From: backend/src/api/services/explanation_service.py:19-90
from datetime import datetime, timedelta
from threading import Lock
from typing import Dict, Any, Optional

class ExplanationService:
    """Service with TTL and value-change based caching."""

    CACHE_TTL = timedelta(hours=1)
    CONFIDENCE_THRESHOLD = 0.05  # 5% change triggers regeneration
    VIX_THRESHOLD = 2.0

    def __init__(self):
        self._lock = Lock()
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._last_values: Dict[str, Any] = {}

    def _should_regenerate(self, current_values: Dict[str, Any]) -> bool:
        """Check if we should regenerate based on value changes."""
        if not self._last_values:
            return True

        # Direction change always triggers regeneration
        if current_values.get("direction") != self._last_values.get("direction"):
            return True

        # Significant confidence change triggers regeneration
        conf_diff = abs(
            current_values.get("confidence", 0) -
            self._last_values.get("confidence", 0)
        )
        if conf_diff >= self.CONFIDENCE_THRESHOLD:
            return True

        # Significant VIX change triggers regeneration
        vix_diff = abs(
            current_values.get("vix", 0) -
            self._last_values.get("vix", 0)
        )
        if vix_diff >= self.VIX_THRESHOLD:
            return True

        return False
```

**Explanation**: Multiple invalidation conditions. Thresholds for "significant" changes. Always regenerate on direction change.

**Example 4: Cache Cleanup to Prevent Memory Leaks**

```python
# From: backend/src/api/services/explanation_service.py:92-110
def _cleanup_expired_cache(self) -> None:
    """Remove expired cache entries to prevent memory leak."""
    now = datetime.now()
    expired_keys = [
        k for k, v in self._cache.items()
        if now - v["generated_at"] > self.CACHE_TTL
    ]
    for key in expired_keys:
        del self._cache[key]

    if expired_keys:
        logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")

def get_explanation(self, prediction: Dict) -> Dict[str, Any]:
    """Get explanation with caching."""
    with self._lock:
        # Cleanup before checking cache
        self._cleanup_expired_cache()

        cache_key = self._get_cache_key(prediction)

        if cache_key in self._cache:
            cached = self._cache[cache_key]
            if not self._should_regenerate(prediction):
                return {**cached["explanation"], "cached": True}

        # Generate new explanation
        explanation = self._generate_explanation(prediction)

        # Update cache
        self._cache[cache_key] = {
            "explanation": explanation,
            "generated_at": datetime.now(),
        }
        self._last_values = prediction.copy()

        return {**explanation, "cached": False}
```

**Explanation**: Cleanup before cache check. Include `cached` flag in response. Update `_last_values` for change detection.

**Example 5: Cache Metadata in Response**

```python
# Pattern from: backend/src/api/services/explanation_service.py
def get_cached_explanation(self, prediction_id: str) -> Dict[str, Any]:
    """Get explanation with metadata."""
    with self._lock:
        if prediction_id in self._cache:
            entry = self._cache[prediction_id]
            return {
                "explanation": entry["explanation"],
                "cached": True,
                "generated_at": entry["generated_at"].isoformat(),
                "age_seconds": (datetime.now() - entry["generated_at"]).total_seconds(),
            }

        return {
            "explanation": None,
            "cached": False,
            "reason": "Not in cache",
        }
```

**Explanation**: Return cache metadata (cached flag, age). Useful for debugging and UI display.

## Quality Checklist

- [ ] Hash computed with `sort_keys=True` for determinism
- [ ] Pattern matches `backend/src/models/feature_selection/manager.py`
- [ ] Cache validation checks data consistency
- [ ] TTL implemented with `timedelta`
- [ ] Cleanup method prevents memory leaks
- [ ] Thread-safe with `Lock` for shared caches
- [ ] Cached flag in responses

## Common Mistakes

- **Mutable default in hash**: Non-deterministic hashing
  - Wrong: Hash dict without sorted keys
  - Correct: `json.dumps(config, sort_keys=True)`

- **No cache validation**: Stale data causes errors
  - Wrong: Load cache without checking structure
  - Correct: Validate and invalidate if mismatch

- **Memory leak**: Cache grows unbounded
  - Wrong: Never clean up expired entries
  - Correct: Call `_cleanup_expired_cache()` before cache access

- **Race conditions**: Concurrent access corrupts cache
  - Wrong: Access cache without lock
  - Correct: Use `threading.Lock` for all cache operations

## Validation

- [ ] Pattern confirmed in `backend/src/models/feature_selection/manager.py`
- [ ] Pattern confirmed in `backend/src/api/services/explanation_service.py`
- [ ] Cache invalidation tested with config changes
- [ ] Memory cleanup verified

## Related Skills

- `creating-python-services` - Services that use caching
- `creating-ml-features` - Feature calculations worth caching

---

<!-- Skill Metadata
Version: 1.0.0
Created: 2026-01-23
Last Verified: 2026-01-23
Last Modified: 2026-01-23
Patterns From: .claude/discovery/codebase-patterns.md v3.0 (Pattern 4.11, 5.8)
Lines: 220
-->
