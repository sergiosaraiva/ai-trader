"""FastAPI dependencies for authentication and authorization.

Provides authentication for admin endpoints like configuration management.
"""

import os
import logging
from typing import Optional

from fastapi import Header, HTTPException, status

logger = logging.getLogger(__name__)

# Optional admin API key for securing admin endpoints
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY") or None  # Treat empty string as None


def verify_admin_auth(x_admin_key: Optional[str] = Header(None)) -> bool:
    """Verify admin API key if configured.

    This dependency should be used on all administrative endpoints that
    modify system configuration, trigger operations, or access sensitive data.

    Args:
        x_admin_key: Admin API key from X-Admin-Key header

    Returns:
        True if authentication succeeds

    Raises:
        HTTPException: 401 if auth fails or key is missing when required

    Example:
        @router.put("/config", dependencies=[Depends(verify_admin_auth)])
        async def update_config(...):
            ...
    """
    if ADMIN_API_KEY is None:
        # No auth required if key not set (for development/testing)
        logger.debug("Admin auth not configured, allowing access")
        return True

    if x_admin_key != ADMIN_API_KEY:
        logger.warning("Admin authentication failed - invalid or missing key")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-Admin-Key header. Set ADMIN_API_KEY environment variable to enable admin authentication.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    logger.debug("Admin authentication successful")
    return True


def is_admin_auth_enabled() -> bool:
    """Check if admin authentication is enabled.

    Returns:
        True if ADMIN_API_KEY is configured, False otherwise
    """
    return ADMIN_API_KEY is not None
