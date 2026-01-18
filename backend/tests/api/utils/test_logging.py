"""Tests for logging utilities.

This module tests structured logging functionality including:
- LoggingMixin class for service logging
- Exception logging with stack traces
- Request context tracking
- Operation logging helpers
"""

import logging
import pytest
from unittest.mock import Mock, patch, call
from datetime import datetime

from src.api.utils.logging import (
    LoggingMixin,
    log_exception,
    log_operation_start,
    log_operation_complete,
    set_request_id,
    get_request_id,
    request_id_ctx,
)


class TestLogException:
    """Tests for the log_exception function."""

    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger for testing."""
        return Mock(spec=logging.Logger)

    def test_log_exception_with_trace(self, mock_logger):
        """Test that exception is logged with stack trace."""
        error = ValueError("Test error")
        log_exception(mock_logger, "Operation failed", error, include_trace=True)

        # Verify logger.error was called
        assert mock_logger.error.called
        call_args = mock_logger.error.call_args[0][0]

        # Check that message includes error details
        assert "Operation failed" in call_args
        assert "ValueError" in call_args
        assert "Test error" in call_args
        assert "Stack trace:" in call_args

    def test_log_exception_without_trace(self, mock_logger):
        """Test that exception can be logged without stack trace."""
        error = ValueError("Test error")
        log_exception(mock_logger, "Operation failed", error, include_trace=False)

        assert mock_logger.error.called
        call_args = mock_logger.error.call_args[0][0]

        # Check that message includes error details but no stack trace
        assert "Operation failed" in call_args
        assert "ValueError" in call_args
        assert "Test error" in call_args
        assert "Stack trace:" not in call_args

    def test_log_exception_with_context(self, mock_logger):
        """Test that additional context is included in log."""
        error = ValueError("Test error")
        log_exception(
            mock_logger,
            "Operation failed",
            error,
            include_trace=False,
            user_id=123,
            endpoint="/api/test",
        )

        call_args = mock_logger.error.call_args[0][0]
        assert "user_id" in call_args
        assert "123" in call_args
        assert "endpoint" in call_args
        assert "/api/test" in call_args

    @patch("src.api.utils.logging.request_id_ctx")
    def test_log_exception_includes_request_id(self, mock_ctx, mock_logger):
        """Test that request ID from context is included."""
        mock_ctx.get.return_value = "req-12345"
        error = ValueError("Test error")

        log_exception(mock_logger, "Operation failed", error, include_trace=False)

        call_args = mock_logger.error.call_args[0][0]
        assert "req-12345" in call_args

    def test_log_exception_with_different_exception_types(self, mock_logger):
        """Test logging different exception types."""
        exceptions = [
            ValueError("Value error"),
            TypeError("Type error"),
            RuntimeError("Runtime error"),
            KeyError("Key error"),
        ]

        for error in exceptions:
            log_exception(mock_logger, "Test", error, include_trace=False)
            call_args = mock_logger.error.call_args[0][0]
            assert type(error).__name__ in call_args
            assert str(error) in call_args

    def test_log_exception_timestamp(self, mock_logger):
        """Test that timestamp is included in log."""
        error = ValueError("Test error")

        with patch("src.api.utils.logging.datetime") as mock_datetime:
            mock_datetime.utcnow.return_value.isoformat.return_value = "2024-01-15T10:30:00"
            log_exception(mock_logger, "Operation failed", error, include_trace=False)

            call_args = mock_logger.error.call_args[0][0]
            assert "2024-01-15T10:30:00" in call_args


class TestRequestContext:
    """Tests for request context management."""

    def test_set_and_get_request_id(self):
        """Test setting and getting request ID."""
        request_id_ctx.set(None)  # Reset
        set_request_id("req-123")
        assert get_request_id() == "req-123"

    def test_get_request_id_when_not_set(self):
        """Test getting request ID when not set."""
        request_id_ctx.set(None)  # Reset
        assert get_request_id() is None

    def test_request_id_updates(self):
        """Test that request ID can be updated."""
        set_request_id("req-1")
        assert get_request_id() == "req-1"

        set_request_id("req-2")
        assert get_request_id() == "req-2"

    def test_request_id_isolation(self):
        """Test that request ID is properly isolated per context."""
        # This test verifies the basic functionality
        # In real async code, each task would have its own context
        request_id_ctx.set(None)  # Reset
        assert get_request_id() is None

        set_request_id("test-req")
        assert get_request_id() == "test-req"

        request_id_ctx.set(None)  # Reset
        assert get_request_id() is None


class TestLoggingMixin:
    """Tests for the LoggingMixin class."""

    class TestService(LoggingMixin):
        """Test service class using LoggingMixin."""

        def __init__(self):
            super().__init__()
            self.setup_logger(__name__)

    @pytest.fixture
    def service(self):
        """Create a test service instance."""
        return self.TestService()

    @pytest.fixture
    def mock_logger(self, service):
        """Mock the logger on the service."""
        service.logger = Mock(spec=logging.Logger)
        return service.logger

    def test_mixin_initializes_logger(self, service):
        """Test that mixin properly initializes logger."""
        assert hasattr(service, "logger")
        assert isinstance(service.logger, logging.Logger)

    def test_log_info(self, service, mock_logger):
        """Test log_info method."""
        service.log_info("Test message", user_id=123)

        assert mock_logger.info.called
        call_args = mock_logger.info.call_args[0][0]
        assert "Test message" in call_args
        assert "user_id" in call_args
        assert "123" in call_args

    def test_log_warning(self, service, mock_logger):
        """Test log_warning method."""
        service.log_warning("Warning message", component="api")

        assert mock_logger.warning.called
        call_args = mock_logger.warning.call_args[0][0]
        assert "Warning message" in call_args
        assert "component" in call_args
        assert "api" in call_args

    def test_log_error_without_exception(self, service, mock_logger):
        """Test log_error method without exception object."""
        service.log_error("Error occurred", endpoint="/test")

        assert mock_logger.error.called
        call_args = mock_logger.error.call_args[0][0]
        assert "Error occurred" in call_args
        assert "endpoint" in call_args

    def test_log_error_with_exception(self, service, mock_logger):
        """Test log_error method with exception object."""
        error = ValueError("Test error")
        service.log_error("Operation failed", error=error, include_trace=True)

        assert mock_logger.error.called
        call_args = mock_logger.error.call_args[0][0]
        assert "Operation failed" in call_args
        assert "ValueError" in call_args
        assert "Test error" in call_args
        assert "Stack trace:" in call_args

    def test_log_error_without_trace(self, service, mock_logger):
        """Test log_error method without stack trace."""
        error = ValueError("Test error")
        service.log_error("Operation failed", error=error, include_trace=False)

        assert mock_logger.error.called
        call_args = mock_logger.error.call_args[0][0]
        assert "Operation failed" in call_args
        assert "ValueError" in call_args
        assert "Stack trace:" not in call_args

    def test_log_debug(self, service, mock_logger):
        """Test log_debug method."""
        service.log_debug("Debug info", step="initialization")

        assert mock_logger.debug.called
        call_args = mock_logger.debug.call_args[0][0]
        assert "Debug info" in call_args
        assert "step" in call_args

    @patch("src.api.utils.logging.request_id_ctx")
    def test_log_methods_include_request_id(self, mock_ctx, service, mock_logger):
        """Test that all log methods include request ID from context."""
        mock_ctx.get.return_value = "req-456"

        service.log_info("Info message")
        info_call = mock_logger.info.call_args[0][0]
        assert "req-456" in info_call

        service.log_warning("Warning message")
        warning_call = mock_logger.warning.call_args[0][0]
        assert "req-456" in warning_call

        service.log_error("Error message")
        error_call = mock_logger.error.call_args[0][0]
        assert "req-456" in error_call

        service.log_debug("Debug message")
        debug_call = mock_logger.debug.call_args[0][0]
        assert "req-456" in debug_call

    def test_log_methods_include_timestamp(self, service, mock_logger):
        """Test that all log methods include timestamp."""
        with patch("src.api.utils.logging.datetime") as mock_datetime:
            mock_datetime.utcnow.return_value.isoformat.return_value = "2024-01-15T10:00:00"

            service.log_info("Test")
            call_args = mock_logger.info.call_args[0][0]
            assert "2024-01-15T10:00:00" in call_args

    def test_add_context_method(self, service):
        """Test _add_context helper method."""
        context = service._add_context(user_id=123, action="test")

        assert "timestamp" in context
        assert "request_id" in context
        assert context["user_id"] == 123
        assert context["action"] == "test"

    def test_multiple_kwargs(self, service, mock_logger):
        """Test logging with multiple keyword arguments."""
        service.log_info(
            "Complex operation",
            user_id=123,
            endpoint="/api/trade",
            duration=1.5,
            status="success",
        )

        call_args = mock_logger.info.call_args[0][0]
        assert "user_id" in call_args
        assert "endpoint" in call_args
        assert "duration" in call_args
        assert "status" in call_args


class TestLogOperationStart:
    """Tests for log_operation_start function."""

    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger for testing."""
        return Mock(spec=logging.Logger)

    def test_log_operation_start_basic(self, mock_logger):
        """Test basic operation start logging."""
        log_operation_start(mock_logger, "data_processing")

        assert mock_logger.info.called
        call_args = mock_logger.info.call_args[0][0]
        assert "Starting: data_processing" in call_args
        assert "operation" in call_args

    def test_log_operation_start_with_context(self, mock_logger):
        """Test operation start logging with additional context."""
        log_operation_start(
            mock_logger, "model_training", dataset="EURUSD", timeframe="1H"
        )

        call_args = mock_logger.info.call_args[0][0]
        assert "Starting: model_training" in call_args
        assert "dataset" in call_args
        assert "EURUSD" in call_args
        assert "timeframe" in call_args
        assert "1H" in call_args

    @patch("src.api.utils.logging.request_id_ctx")
    def test_log_operation_start_includes_request_id(self, mock_ctx, mock_logger):
        """Test that request ID is included in operation start log."""
        mock_ctx.get.return_value = "req-789"
        log_operation_start(mock_logger, "api_call")

        call_args = mock_logger.info.call_args[0][0]
        assert "req-789" in call_args

    def test_log_operation_start_timestamp(self, mock_logger):
        """Test that timestamp is included."""
        with patch("src.api.utils.logging.datetime") as mock_datetime:
            mock_datetime.utcnow.return_value.isoformat.return_value = "2024-01-15T12:00:00"
            log_operation_start(mock_logger, "test_operation")

            call_args = mock_logger.info.call_args[0][0]
            assert "2024-01-15T12:00:00" in call_args


class TestLogOperationComplete:
    """Tests for log_operation_complete function."""

    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger for testing."""
        return Mock(spec=logging.Logger)

    def test_log_operation_complete_basic(self, mock_logger):
        """Test basic operation completion logging."""
        log_operation_complete(mock_logger, "data_processing")

        assert mock_logger.info.called
        call_args = mock_logger.info.call_args[0][0]
        assert "Completed: data_processing" in call_args

    def test_log_operation_complete_with_duration(self, mock_logger):
        """Test operation completion logging with duration."""
        log_operation_complete(mock_logger, "model_training", duration_seconds=12.5)

        call_args = mock_logger.info.call_args[0][0]
        assert "Completed: model_training" in call_args
        assert "duration_seconds" in call_args
        assert "12.5" in call_args

    def test_log_operation_complete_with_context(self, mock_logger):
        """Test operation completion logging with additional context."""
        log_operation_complete(
            mock_logger,
            "backtest",
            duration_seconds=5.0,
            trades=100,
            profit=1234.56,
        )

        call_args = mock_logger.info.call_args[0][0]
        assert "Completed: backtest" in call_args
        assert "trades" in call_args
        assert "100" in call_args
        assert "profit" in call_args
        assert "1234.56" in call_args

    def test_log_operation_complete_without_duration(self, mock_logger):
        """Test operation completion without duration."""
        log_operation_complete(mock_logger, "quick_check", status="success")

        call_args = mock_logger.info.call_args[0][0]
        assert "Completed: quick_check" in call_args
        assert "status" in call_args
        # Duration should not be present
        assert "duration_seconds" not in call_args or "None" in call_args

    @patch("src.api.utils.logging.request_id_ctx")
    def test_log_operation_complete_includes_request_id(self, mock_ctx, mock_logger):
        """Test that request ID is included in operation complete log."""
        mock_ctx.get.return_value = "req-999"
        log_operation_complete(mock_logger, "api_call")

        call_args = mock_logger.info.call_args[0][0]
        assert "req-999" in call_args

    def test_log_operation_complete_timestamp(self, mock_logger):
        """Test that timestamp is included."""
        with patch("src.api.utils.logging.datetime") as mock_datetime:
            mock_datetime.utcnow.return_value.isoformat.return_value = "2024-01-15T12:30:00"
            log_operation_complete(mock_logger, "test_operation")

            call_args = mock_logger.info.call_args[0][0]
            assert "2024-01-15T12:30:00" in call_args


class TestIntegrationScenarios:
    """Integration tests for common logging scenarios."""

    class TestService(LoggingMixin):
        """Test service for integration scenarios."""

        def __init__(self):
            super().__init__()
            self.setup_logger(__name__)

    @pytest.fixture
    def service(self):
        """Create test service instance."""
        service = self.TestService()
        service.logger = Mock(spec=logging.Logger)
        return service

    def test_request_lifecycle_logging(self, service):
        """Test logging throughout a request lifecycle."""
        # Simulate a request with ID
        set_request_id("req-integration-1")

        # Start operation
        service.log_info("Request received", endpoint="/api/predict")

        # Log progress
        service.log_debug("Loading model")
        service.log_debug("Generating features")

        # Complete successfully
        service.log_info("Request completed", status="success")

        # Verify all logs were called
        assert service.logger.info.call_count == 2
        assert service.logger.debug.call_count == 2

        # Cleanup
        request_id_ctx.set(None)

    def test_error_handling_scenario(self, service):
        """Test logging during error scenarios."""
        set_request_id("req-error-1")

        try:
            # Simulate an operation that fails
            service.log_info("Starting risky operation")
            raise ValueError("Something went wrong")
        except ValueError as e:
            service.log_error("Operation failed", error=e, include_trace=True)

        assert service.logger.info.called
        assert service.logger.error.called

        error_call = service.logger.error.call_args[0][0]
        assert "Operation failed" in error_call
        assert "ValueError" in error_call
        assert "Stack trace:" in error_call

        # Cleanup
        request_id_ctx.set(None)

    def test_multiple_context_updates(self, service):
        """Test logging with context updates throughout execution."""
        set_request_id("req-multi-1")

        service.log_info("Step 1", progress=0)
        service.log_info("Step 2", progress=50)
        service.log_info("Step 3", progress=100)

        assert service.logger.info.call_count == 3

        # Verify each call had the request ID
        for call_obj in service.logger.info.call_args_list:
            call_args = call_obj[0][0]
            assert "req-multi-1" in call_args

        # Cleanup
        request_id_ctx.set(None)
