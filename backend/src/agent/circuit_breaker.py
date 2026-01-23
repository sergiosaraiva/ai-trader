"""Generic circuit breaker pattern for external service calls.

Implements the circuit breaker pattern to prevent cascading failures when
external services (broker, database, APIs) become unavailable.
"""

import asyncio
import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Callable, Optional, Any, TypeVar, Generic

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation, requests pass through
    OPEN = "open"  # Failures exceeded threshold, requests blocked
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for a circuit breaker."""

    name: str
    failure_threshold: int = 5  # Failures to open circuit
    success_threshold: int = 2  # Successes in half-open to close
    timeout_seconds: float = 30.0  # Time before half-open
    half_open_max_calls: int = 3  # Max calls in half-open state


class CircuitBreaker(Generic[T]):
    """Generic circuit breaker for external service calls.

    Usage:
        breaker = CircuitBreaker(config)

        # Sync usage
        result = breaker.call(some_sync_function, arg1, arg2)

        # Async usage
        result = await breaker.call_async(some_async_function, arg1, arg2)

    The circuit breaker tracks failures and opens the circuit when the
    failure threshold is exceeded. After a timeout period, the circuit
    enters half-open state where a limited number of calls are allowed
    to test if the service has recovered.
    """

    def __init__(self, config: CircuitBreakerConfig):
        """Initialize circuit breaker.

        Args:
            config: Circuit breaker configuration
        """
        self.config = config
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._half_open_calls = 0
        self._lock = threading.Lock()

        logger.info(
            f"CircuitBreaker '{config.name}' initialized: "
            f"failure_threshold={config.failure_threshold}, "
            f"timeout={config.timeout_seconds}s"
        )

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        with self._lock:
            return self._state

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self._check_state() == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (blocking requests)."""
        return self._check_state() == CircuitState.OPEN

    def _check_state(self) -> CircuitState:
        """Check and potentially update circuit state.

        Handles automatic transition from OPEN to HALF_OPEN after timeout.
        """
        with self._lock:
            if self._state == CircuitState.OPEN and self._last_failure_time:
                elapsed = (datetime.now() - self._last_failure_time).total_seconds()
                if elapsed >= self.config.timeout_seconds:
                    logger.info(
                        f"CircuitBreaker '{self.config.name}': "
                        f"Transitioning from OPEN to HALF_OPEN after {elapsed:.1f}s"
                    )
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_calls = 0
                    self._success_count = 0

            return self._state

    def _record_success(self) -> None:
        """Record a successful call."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    logger.info(
                        f"CircuitBreaker '{self.config.name}': "
                        f"Closing circuit after {self._success_count} successes"
                    )
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._success_count = 0
            else:
                # Reset failure count on success in closed state
                self._failure_count = 0

    def _record_failure(self, error: Exception) -> None:
        """Record a failed call."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = datetime.now()

            if self._state == CircuitState.HALF_OPEN:
                # Any failure in half-open reopens the circuit
                logger.warning(
                    f"CircuitBreaker '{self.config.name}': "
                    f"Reopening circuit after failure in HALF_OPEN: {error}"
                )
                self._state = CircuitState.OPEN
                self._success_count = 0

            elif self._failure_count >= self.config.failure_threshold:
                logger.warning(
                    f"CircuitBreaker '{self.config.name}': "
                    f"Opening circuit after {self._failure_count} failures"
                )
                self._state = CircuitState.OPEN

    def _can_execute(self) -> bool:
        """Check if a call can be executed."""
        state = self._check_state()

        if state == CircuitState.CLOSED:
            return True

        if state == CircuitState.HALF_OPEN:
            with self._lock:
                if self._half_open_calls < self.config.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False

        # Circuit is OPEN
        return False

    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute a synchronous function through the circuit breaker.

        Args:
            func: Function to call
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            CircuitOpenError: If circuit is open
            Exception: Re-raises any exception from the function
        """
        if not self._can_execute():
            raise CircuitOpenError(
                f"Circuit '{self.config.name}' is open, blocking call"
            )

        try:
            result = func(*args, **kwargs)
            self._record_success()
            return result
        except Exception as e:
            self._record_failure(e)
            raise

    async def call_async(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute an async function through the circuit breaker.

        Args:
            func: Async function to call
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            CircuitOpenError: If circuit is open
            Exception: Re-raises any exception from the function
        """
        if not self._can_execute():
            raise CircuitOpenError(
                f"Circuit '{self.config.name}' is open, blocking call"
            )

        try:
            result = await func(*args, **kwargs)
            self._record_success()
            return result
        except Exception as e:
            self._record_failure(e)
            raise

    def reset(self) -> None:
        """Manually reset the circuit breaker to closed state."""
        with self._lock:
            logger.info(f"CircuitBreaker '{self.config.name}': Manual reset")
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._last_failure_time = None
            self._half_open_calls = 0

    def get_status(self) -> dict:
        """Get current circuit breaker status.

        Returns:
            Dictionary with status information
        """
        with self._lock:
            return {
                "name": self.config.name,
                "state": self._state.value,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "last_failure": (
                    self._last_failure_time.isoformat()
                    if self._last_failure_time
                    else None
                ),
                "config": {
                    "failure_threshold": self.config.failure_threshold,
                    "success_threshold": self.config.success_threshold,
                    "timeout_seconds": self.config.timeout_seconds,
                },
            }


class CircuitOpenError(Exception):
    """Raised when attempting to call through an open circuit."""

    pass


# Pre-configured circuit breakers for common services
def create_broker_circuit_breaker() -> CircuitBreaker:
    """Create a circuit breaker for broker connections."""
    return CircuitBreaker(
        CircuitBreakerConfig(
            name="broker",
            failure_threshold=3,
            success_threshold=2,
            timeout_seconds=60.0,
            half_open_max_calls=2,
        )
    )


def create_database_circuit_breaker() -> CircuitBreaker:
    """Create a circuit breaker for database operations."""
    return CircuitBreaker(
        CircuitBreakerConfig(
            name="database",
            failure_threshold=5,
            success_threshold=3,
            timeout_seconds=30.0,
            half_open_max_calls=3,
        )
    )


def create_api_circuit_breaker(name: str = "external_api") -> CircuitBreaker:
    """Create a circuit breaker for external API calls."""
    return CircuitBreaker(
        CircuitBreakerConfig(
            name=name,
            failure_threshold=5,
            success_threshold=2,
            timeout_seconds=45.0,
            half_open_max_calls=2,
        )
    )
