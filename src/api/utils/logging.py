"""Structured logging utilities for API services.

This module provides:
- LoggingMixin: A mixin class that adds structured logging context
- Helper functions for consistent error logging with stack traces
- Request context tracking
"""

import logging
import traceback
from datetime import datetime
from typing import Any, Dict, Optional
from contextvars import ContextVar

# Context variable for tracking request ID across async boundaries
request_id_ctx: ContextVar[Optional[str]] = ContextVar('request_id', default=None)


class LoggingMixin:
    """Mixin class that provides structured logging with context.

    Usage:
        class MyService(LoggingMixin):
            def __init__(self):
                super().__init__()
                self.setup_logger(__name__)

            def do_something(self):
                self.log_info("Doing something", extra_field="value")
                self.log_error("Error occurred", error=some_error)
    """

    def setup_logger(self, name: str) -> None:
        """Set up the logger for this class.

        Args:
            name: Logger name (typically __name__)
        """
        self.logger = logging.getLogger(name)

    def _add_context(self, **kwargs) -> Dict[str, Any]:
        """Add standard context to log messages.

        Args:
            **kwargs: Additional context fields

        Returns:
            Dictionary with timestamp, request_id, and custom fields
        """
        context = {
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": request_id_ctx.get(),
        }
        context.update(kwargs)
        return context

    def log_info(self, message: str, **kwargs) -> None:
        """Log info message with context.

        Args:
            message: Log message
            **kwargs: Additional context fields
        """
        context = self._add_context(**kwargs)
        self.logger.info(f"{message} | Context: {context}")

    def log_warning(self, message: str, **kwargs) -> None:
        """Log warning message with context.

        Args:
            message: Log message
            **kwargs: Additional context fields
        """
        context = self._add_context(**kwargs)
        self.logger.warning(f"{message} | Context: {context}")

    def log_error(
        self,
        message: str,
        error: Optional[Exception] = None,
        include_trace: bool = True,
        **kwargs,
    ) -> None:
        """Log error message with context and optional stack trace.

        Args:
            message: Log message
            error: Exception object if available
            include_trace: Whether to include full stack trace
            **kwargs: Additional context fields
        """
        context = self._add_context(**kwargs)

        if error:
            context["error_type"] = type(error).__name__
            context["error_message"] = str(error)

        error_msg = f"{message} | Context: {context}"

        if include_trace and error:
            trace = "".join(traceback.format_exception(type(error), error, error.__traceback__))
            error_msg += f"\nStack trace:\n{trace}"

        self.logger.error(error_msg)

    def log_debug(self, message: str, **kwargs) -> None:
        """Log debug message with context.

        Args:
            message: Log message
            **kwargs: Additional context fields
        """
        context = self._add_context(**kwargs)
        self.logger.debug(f"{message} | Context: {context}")


def log_exception(
    logger: logging.Logger,
    message: str,
    error: Exception,
    include_trace: bool = True,
    **context,
) -> None:
    """Log an exception with full context and stack trace.

    Args:
        logger: Logger instance to use
        message: Descriptive message about what failed
        error: The exception that occurred
        include_trace: Whether to include full stack trace
        **context: Additional context fields
    """
    context_dict = {
        "timestamp": datetime.utcnow().isoformat(),
        "request_id": request_id_ctx.get(),
        "error_type": type(error).__name__,
        "error_message": str(error),
        **context,
    }

    error_msg = f"{message} | Context: {context_dict}"

    if include_trace:
        trace = "".join(traceback.format_exception(type(error), error, error.__traceback__))
        error_msg += f"\nStack trace:\n{trace}"

    logger.error(error_msg)


def log_operation_start(
    logger: logging.Logger,
    operation: str,
    **context,
) -> None:
    """Log the start of an operation with context.

    Args:
        logger: Logger instance to use
        operation: Name of the operation
        **context: Additional context fields
    """
    context_dict = {
        "timestamp": datetime.utcnow().isoformat(),
        "request_id": request_id_ctx.get(),
        "operation": operation,
        **context,
    }
    logger.info(f"Starting: {operation} | Context: {context_dict}")


def log_operation_complete(
    logger: logging.Logger,
    operation: str,
    duration_seconds: Optional[float] = None,
    **context,
) -> None:
    """Log the completion of an operation with context.

    Args:
        logger: Logger instance to use
        operation: Name of the operation
        duration_seconds: Optional duration in seconds
        **context: Additional context fields
    """
    context_dict = {
        "timestamp": datetime.utcnow().isoformat(),
        "request_id": request_id_ctx.get(),
        "operation": operation,
        **context,
    }

    if duration_seconds is not None:
        context_dict["duration_seconds"] = duration_seconds

    logger.info(f"Completed: {operation} | Context: {context_dict}")


def set_request_id(request_id: str) -> None:
    """Set the request ID for the current context.

    Args:
        request_id: Unique identifier for the request
    """
    request_id_ctx.set(request_id)


def get_request_id() -> Optional[str]:
    """Get the request ID for the current context.

    Returns:
        Request ID if set, None otherwise
    """
    return request_id_ctx.get()
