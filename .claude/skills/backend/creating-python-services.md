---
name: creating-python-services
description: This skill should be used when the user asks to "create a service", "add a singleton", "implement business logic", "wrap an ML model". Creates thread-safe singleton service classes with lazy initialization, caching, and status tracking for expensive resources.
version: 1.1.0
---

# Creating Python Services

## Quick Reference

- Use singleton pattern with module-level instance: `service = ServiceClass()`
- Thread-safe operations with `threading.Lock()`
- Lazy initialization with `initialize()` method
- Status properties: `is_loaded`, `is_initialized`
- Cache with TTL using `timedelta`

## When to Use

- Managing expensive resources (ML models, connections)
- Shared state across API requests
- Implementing caching layers
- Wrapping external APIs or data sources
- Business logic that needs initialization

## When NOT to Use

- Stateless utility functions (use plain functions)
- Request-scoped operations (use dependencies)
- Simple data transformations (use Pydantic)

## Implementation Guide

```
Does service manage expensive resources?
├─ Yes → Use lazy initialization pattern
│   └─ Add initialize() method called at startup
│   └─ Track status with _initialized flag
└─ No → Consider simpler pattern

Does service need thread safety?
├─ Yes → Add self._lock = Lock()
│   └─ Use with self._lock: for shared state
└─ No → Skip locking (single-threaded only)

Does service need caching?
├─ Yes → Add _cache dict with TTL
│   └─ Check cache freshness before returning
└─ No → Return fresh results each call
```

## Examples

**Example 1: Service Class Structure**

```python
# From: src/api/services/model_service.py:28-66
class ModelService:
    """Service for MTF Ensemble model loading and prediction.

    Uses singleton pattern - model is loaded once and shared across requests.
    Provides thread-safe prediction with caching.
    """

    # Default model directory
    DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "mtf_ensemble"

    # Cache TTL
    PREDICTION_CACHE_TTL = timedelta(minutes=1)

    def __init__(self, model_dir: Optional[Path] = None):
        self._lock = Lock()
        self._model_dir = Path(model_dir) if model_dir else self.DEFAULT_MODEL_DIR

        # Model instance (lazy loaded)
        self._ensemble = None
        self._config = None

        # Prediction cache
        self._cache: Dict[str, Dict] = {}
        self._cache_timestamp: Optional[datetime] = None

        # Status
        self._initialized = False
        self._initialization_error: Optional[str] = None

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._ensemble is not None and self._ensemble.is_trained

    @property
    def is_initialized(self) -> bool:
        """Check if service is initialized."""
        return self._initialized
```

**Explanation**: Class constants for configuration, Lock for thread safety, private attributes with underscore prefix, status properties for external checks.

**Example 2: Initialize Method Pattern**

```python
# From: src/api/services/model_service.py:67-94
def initialize(self, warm_up: bool = True) -> bool:
    """Initialize model service by loading the ensemble.

    Args:
        warm_up: Whether to run a warm-up prediction

    Returns:
        True if successful, False otherwise
    """
    if self._initialized:
        return True

    logger.info("Initializing ModelService...")

    try:
        self._load_model()

        if warm_up:
            self._warm_up()

        self._initialized = True
        logger.info("ModelService initialized successfully")
        return True

    except Exception as e:
        self._initialization_error = str(e)
        logger.error(f"Failed to initialize ModelService: {e}")
        return False
```

**Explanation**: Idempotent (returns True if already initialized), stores error for debugging, logs progress, returns success status.

**Example 3: Thread-Safe Method with Caching**

```python
# From: src/api/services/model_service.py:163-234
def predict(
    self,
    df_5min: pd.DataFrame,
    use_cache: bool = True,
) -> Dict[str, Any]:
    """Make a prediction using the MTF Ensemble.

    Args:
        df_5min: 5-minute OHLCV DataFrame with sufficient history
        use_cache: Whether to use cached predictions

    Returns:
        Dict with prediction details
    """
    if not self.is_loaded:
        raise RuntimeError("Model not loaded. Call initialize() first.")

    # Generate cache key from latest timestamp
    if df_5min is not None and len(df_5min) > 0:
        cache_key = str(df_5min.index[-1])
    else:
        cache_key = "unknown"

    # Check cache
    with self._lock:
        if use_cache and cache_key in self._cache:
            cached = self._cache[cache_key]
            cache_age = datetime.now() - cached["cached_at"]
            if cache_age < self.PREDICTION_CACHE_TTL:
                logger.debug(f"Returning cached prediction ({cache_age.seconds}s old)")
                return cached["prediction"]

    # Make prediction
    try:
        with self._lock:
            prediction = self._ensemble.predict(df_5min)

        # Convert to dict
        result = {
            "direction": "long" if prediction.direction == 1 else "short",
            "confidence": float(prediction.confidence),
            # ... additional fields
        }

        # Cache result
        with self._lock:
            self._cache[cache_key] = {
                "prediction": result,
                "cached_at": datetime.now(),
            }

        return result

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise
```

**Explanation**: Status check first, cache key generation, lock for cache check, lock for prediction, lock for cache update. TTL-based cache invalidation.

**Example 4: Singleton Instantiation**

```python
# From: src/api/services/model_service.py:389-391
# Singleton instance
model_service = ModelService()
```

**Explanation**: Module-level instance created at import time. Other modules import this instance directly.

**Example 5: Usage in API Startup**

```python
# From: src/api/main.py:51-55
logger.info("Initializing model service...")
try:
    model_service.initialize(warm_up=True)
except Exception as e:
    logger.warning(f"Model service initialization failed: {e}")
```

**Explanation**: Initialize in lifespan manager, catch and log errors (don't crash on init failure).

## Quality Checklist

- [ ] Thread-safe with `Lock()` for shared state
- [ ] Lazy initialization with `initialize()` method
- [ ] Status properties `is_loaded` and `is_initialized`
- [ ] Pattern matches `src/api/services/model_service.py:28-94`
- [ ] Cache with TTL if needed
- [ ] Error stored in `_initialization_error`
- [ ] Singleton instance at module level
- [ ] Called in `src/api/main.py` lifespan

## Common Mistakes

- **Missing status check**: Always check `is_loaded` before operations
  - Wrong: `def predict(self, df): return self._model.predict(df)`
  - Correct: Check `self.is_loaded` first, raise RuntimeError if not

- **Lock not used**: Protect shared state access
  - Wrong: `self._cache[key] = result`
  - Correct: `with self._lock: self._cache[key] = result`

- **Cache without TTL**: Old cache never invalidates
  - Wrong: `if key in self._cache: return self._cache[key]`
  - Correct: Check `cache_age < self.PREDICTION_CACHE_TTL`

## Validation

- [ ] Pattern confirmed in `src/api/services/model_service.py:28-94`
- [ ] Singleton used in `src/api/main.py:13`
- [ ] Initialize called in `src/api/main.py:51-55`

## Related Skills

- `creating-fastapi-endpoints` - Use services in API routes
- `creating-pydantic-schemas` - Define service method return types
- `writing-pytest-tests` - Mock services in tests

---

*Version 1.0.0 | Last verified: 2026-01-16 | Source: src/api/services/*
