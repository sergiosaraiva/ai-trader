"""Configuration management API endpoints.

Provides REST API for:
- Viewing current configuration
- Updating configuration parameters
- Hot reloading configuration
- Viewing configuration history
- Resetting to defaults
"""

import logging
from typing import Dict, Any, Optional, List

from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, Field, validator
from sqlalchemy.orm import Session

from ..database.session import get_session
from ..database.models import ConfigurationSetting, ConfigurationHistory
from ...config import trading_config
from ..utils.logging import log_exception
from ..dependencies import verify_admin_auth, is_admin_auth_enabled

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/config", tags=["Configuration"])


# Pydantic schemas
class ConfigUpdateRequest(BaseModel):
    """Request schema for configuration updates."""

    category: str = Field(..., description="Configuration category (trading, model, risk, system)")
    updates: Dict[str, Any] = Field(..., description="Dictionary of parameter updates")
    updated_by: Optional[str] = Field(None, description="User/service making the change")
    reason: Optional[str] = Field(None, description="Reason for the change")

    @validator("category")
    def validate_category(cls, v):
        """Validate category name."""
        valid_categories = ["trading", "model", "risk", "system"]
        if v not in valid_categories:
            raise ValueError(f"Invalid category. Must be one of: {valid_categories}")
        return v


class ConfigResponse(BaseModel):
    """Response schema for configuration data."""

    status: str
    data: Dict[str, Any]
    timestamp: Optional[str] = None


class SettingResponse(BaseModel):
    """Response schema for individual configuration setting."""

    id: int
    category: str
    key: str
    value: Any
    value_type: str
    description: Optional[str]
    version: int
    updated_by: Optional[str]
    updated_at: str
    created_at: str


class HistoryResponse(BaseModel):
    """Response schema for configuration history."""

    id: int
    category: str
    key: str
    old_value: Optional[Any]
    new_value: Any
    version: int
    changed_by: Optional[str]
    changed_at: str
    reason: Optional[str]


@router.get("")
async def get_all_config() -> ConfigResponse:
    """Get all configuration parameters.

    Returns:
        Complete configuration including all categories
    """
    try:
        config_data = trading_config.get_all()

        return ConfigResponse(
            status="success",
            data=config_data,
        )

    except Exception as e:
        log_exception(logger, "Failed to retrieve configuration", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve configuration: {str(e)}",
        )


@router.get("/category/{category}")
async def get_category_config(category: str) -> ConfigResponse:
    """Get configuration for a specific category.

    Args:
        category: Configuration category (trading, model, risk, system)

    Returns:
        Configuration parameters for the specified category
    """
    try:
        config_data = trading_config.get_category(category)

        return ConfigResponse(
            status="success",
            data={category: config_data},
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        log_exception(logger, f"Failed to retrieve {category} configuration", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve configuration: {str(e)}",
        )


@router.put("", dependencies=[Depends(verify_admin_auth)])
async def update_config(
    request: ConfigUpdateRequest,
    db: Session = Depends(get_session),
) -> ConfigResponse:
    """Update configuration parameters.

    Args:
        request: Configuration update request
        db: Database session

    Returns:
        Update result with timestamp
    """
    try:
        # Perform update
        result = trading_config.update(
            category=request.category,
            updates=request.updates,
            updated_by=request.updated_by,
            reason=request.reason,
            db_session=db,
        )

        return ConfigResponse(
            status="success",
            data=result,
            timestamp=result.get("timestamp"),
        )

    except ValueError as e:
        logger.warning(f"Configuration update validation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        log_exception(logger, "Configuration update failed", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Configuration update failed: {str(e)}",
        )


@router.post("/reload", dependencies=[Depends(verify_admin_auth)])
async def reload_config(
    db: Session = Depends(get_session),
) -> ConfigResponse:
    """Hot reload configuration from database.

    Reloads all configuration without restarting the service.
    Triggers callbacks to notify dependent services.

    Args:
        db: Database session

    Returns:
        Reload result with timestamp
    """
    try:
        result = trading_config.reload(db_session=db)

        if result["status"] == "error":
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.get("message", "Reload failed"),
            )

        return ConfigResponse(
            status="success",
            data=result,
            timestamp=result.get("timestamp"),
        )

    except HTTPException:
        raise
    except Exception as e:
        log_exception(logger, "Configuration reload failed", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Configuration reload failed: {str(e)}",
        )


@router.get("/settings")
async def list_all_settings(
    db: Session = Depends(get_session),
) -> Dict[str, List[SettingResponse]]:
    """List all configuration settings from database.

    Args:
        db: Database session

    Returns:
        List of all configuration settings with metadata
    """
    try:
        settings = db.query(ConfigurationSetting).order_by(
            ConfigurationSetting.category,
            ConfigurationSetting.key,
        ).all()

        result = []
        for setting in settings:
            result.append(SettingResponse(
                id=setting.id,
                category=setting.category,
                key=setting.key,
                value=setting.value,
                value_type=setting.value_type,
                description=setting.description,
                version=setting.version,
                updated_by=setting.updated_by,
                updated_at=setting.updated_at.isoformat(),
                created_at=setting.created_at.isoformat(),
            ))

        return {
            "status": "success",
            "count": len(result),
            "settings": result,
        }

    except Exception as e:
        log_exception(logger, "Failed to list configuration settings", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list settings: {str(e)}",
        )


@router.get("/history", dependencies=[Depends(verify_admin_auth)])
async def get_config_history(
    category: Optional[str] = None,
    key: Optional[str] = None,
    limit: int = 50,
    db: Session = Depends(get_session),
) -> Dict[str, Any]:
    """Get configuration change history.

    Args:
        category: Optional category filter
        key: Optional key filter (requires category)
        limit: Maximum number of records to return
        db: Database session

    Returns:
        List of configuration changes with audit information
    """
    try:
        query = db.query(ConfigurationHistory).order_by(
            ConfigurationHistory.changed_at.desc()
        )

        if category:
            query = query.filter(ConfigurationHistory.category == category)

        if key:
            if not category:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="category required when filtering by key",
                )
            query = query.filter(ConfigurationHistory.key == key)

        history = query.limit(limit).all()

        result = []
        for record in history:
            result.append(HistoryResponse(
                id=record.id,
                category=record.category,
                key=record.key,
                old_value=record.old_value,
                new_value=record.new_value,
                version=record.version,
                changed_by=record.changed_by,
                changed_at=record.changed_at.isoformat(),
                reason=record.reason,
            ))

        return {
            "status": "success",
            "count": len(result),
            "history": result,
        }

    except HTTPException:
        raise
    except Exception as e:
        log_exception(logger, "Failed to retrieve configuration history", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve history: {str(e)}",
        )


@router.post("/reset/{category}/{key}", dependencies=[Depends(verify_admin_auth)])
async def reset_config_key(
    category: str,
    key: str,
    db: Session = Depends(get_session),
) -> ConfigResponse:
    """Reset a specific configuration parameter to its default value.

    Args:
        category: Configuration category
        key: Configuration key
        db: Database session

    Returns:
        Reset result with default value
    """
    try:
        result = trading_config.reset_to_defaults(
            category=category,
            key=key,
            db_session=db,
        )

        return ConfigResponse(
            status="success",
            data=result,
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        log_exception(logger, f"Failed to reset {category}.{key}", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reset configuration: {str(e)}",
        )


@router.post("/reset/{category}", dependencies=[Depends(verify_admin_auth)])
async def reset_category(
    category: str,
    db: Session = Depends(get_session),
) -> ConfigResponse:
    """Reset an entire configuration category to defaults.

    Args:
        category: Configuration category
        db: Database session

    Returns:
        Reset result
    """
    try:
        result = trading_config.reset_to_defaults(
            category=category,
            db_session=db,
        )

        return ConfigResponse(
            status="success",
            data=result,
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        log_exception(logger, f"Failed to reset {category} category", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reset configuration: {str(e)}",
        )


@router.get("/validate")
async def validate_config() -> Dict[str, Any]:
    """Validate current configuration.

    Returns:
        Validation result with any errors found
    """
    try:
        errors = trading_config.validate()

        if errors:
            return {
                "status": "invalid",
                "valid": False,
                "errors": errors,
            }
        else:
            return {
                "status": "valid",
                "valid": True,
                "message": "Configuration is valid",
            }

    except Exception as e:
        log_exception(logger, "Configuration validation failed", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Validation failed: {str(e)}",
        )


@router.get("/auth-status")
async def get_auth_status() -> Dict[str, Any]:
    """Get authentication status for config endpoints.

    Returns:
        Information about whether admin authentication is enabled
    """
    return {
        "status": "success",
        "admin_auth_enabled": is_admin_auth_enabled(),
        "message": "Set ADMIN_API_KEY environment variable to enable authentication for config endpoints",
    }
