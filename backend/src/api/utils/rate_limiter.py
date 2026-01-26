"""Rate limiting utilities for API endpoints.

Provides configurable rate limiting to prevent abuse and ensure fair usage.
"""

import os
import logging
from typing import Callable

from fastapi import Request, Response
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

logger = logging.getLogger(__name__)

# Configuration from environment
RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"
RATE_LIMIT_DEFAULT = os.getenv("RATE_LIMIT_DEFAULT", "100/minute")
RATE_LIMIT_AGENT_COMMANDS = os.getenv("RATE_LIMIT_AGENT_COMMANDS", "10/minute")
RATE_LIMIT_TRADING = os.getenv("RATE_LIMIT_TRADING", "30/minute")
RATE_LIMIT_PREDICTIONS = os.getenv("RATE_LIMIT_PREDICTIONS", "60/minute")


def get_client_ip(request: Request) -> str:
    """Get client IP address from request.

    Handles X-Forwarded-For header for requests behind proxies.

    Args:
        request: FastAPI request

    Returns:
        Client IP address string
    """
    # Check for proxy headers
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        # Get first IP in chain (original client)
        return forwarded.split(",")[0].strip()

    # Check X-Real-IP header
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip

    # Fall back to direct client IP
    return get_remote_address(request)


# Create limiter instance
limiter = Limiter(
    key_func=get_client_ip,
    enabled=RATE_LIMIT_ENABLED,
    default_limits=[RATE_LIMIT_DEFAULT],
)


def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> Response:
    """Custom handler for rate limit exceeded.

    Returns a JSON response with details about the rate limit.

    Args:
        request: The request that exceeded the rate limit
        exc: The rate limit exception

    Returns:
        JSON response with 429 status code
    """
    from fastapi.responses import JSONResponse

    client_ip = get_client_ip(request)
    logger.warning(
        f"Rate limit exceeded: client={client_ip}, "
        f"path={request.url.path}, "
        f"limit={exc.detail}"
    )

    return JSONResponse(
        status_code=429,
        content={
            "error": "rate_limit_exceeded",
            "detail": f"Rate limit exceeded: {exc.detail}",
            "retry_after": getattr(exc, "retry_after", 60),
        },
        headers={
            "Retry-After": str(getattr(exc, "retry_after", 60)),
            "X-RateLimit-Limit": str(exc.detail),
        },
    )


def setup_rate_limiting(app) -> None:
    """Setup rate limiting middleware on FastAPI app.

    Args:
        app: FastAPI application instance
    """
    if not RATE_LIMIT_ENABLED:
        logger.info("Rate limiting disabled (RATE_LIMIT_ENABLED=false)")
        return

    # Add state for limiter
    app.state.limiter = limiter

    # Add exception handler
    app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)

    # Add middleware
    app.add_middleware(SlowAPIMiddleware)

    logger.info(
        f"Rate limiting enabled: default={RATE_LIMIT_DEFAULT}, "
        f"agent={RATE_LIMIT_AGENT_COMMANDS}, "
        f"trading={RATE_LIMIT_TRADING}"
    )


# Decorators for different rate limit tiers
def agent_command_limit() -> Callable:
    """Rate limit for agent command endpoints (stricter)."""
    return limiter.limit(RATE_LIMIT_AGENT_COMMANDS)


def trading_limit() -> Callable:
    """Rate limit for trading endpoints."""
    return limiter.limit(RATE_LIMIT_TRADING)


def predictions_limit() -> Callable:
    """Rate limit for prediction endpoints."""
    return limiter.limit(RATE_LIMIT_PREDICTIONS)


def default_limit() -> Callable:
    """Default rate limit."""
    return limiter.limit(RATE_LIMIT_DEFAULT)
