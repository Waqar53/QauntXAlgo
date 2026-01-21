from __future__ import annotations

"""
Structured logging configuration using structlog.

Provides JSON-formatted logs for production and human-readable logs for development.
"""

import logging
import sys
from typing import Any, Optional

import structlog
from structlog.types import Processor

from quantxalgo.config.settings import get_settings


def add_app_context(
    logger: logging.Logger, 
    method_name: str, 
    event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Add application context to all log entries."""
    event_dict["app"] = "quantxalgo"
    event_dict["version"] = "0.1.0"
    return event_dict


def configure_logging() -> None:
    """Configure structured logging for the application."""
    settings = get_settings()
    
    # Common processors for all environments
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
        add_app_context,
    ]
    
    if settings.is_development:
        # Development: colored, human-readable output
        processors: list[Processor] = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True)
        ]
    else:
        # Production: JSON output for log aggregation
        processors = shared_processors + [
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer()
        ]
    
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.log_level.upper()),
    )
    
    # Set third-party loggers to WARNING to reduce noise
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("yfinance").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def get_logger(name: Optional[str] = None) -> structlog.stdlib.BoundLogger:
    """Get a configured logger instance.
    
    Args:
        name: Logger name, typically __name__ of the calling module.
        
    Returns:
        Configured structlog logger.
        
    Example:
        >>> from quantxalgo.config.logging_config import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing order", order_id="123", symbol="AAPL")
    """
    return structlog.get_logger(name)


class LogContext:
    """Context manager for adding temporary context to logs.
    
    Example:
        >>> with LogContext(request_id="abc123", user_id="user1"):
        ...     logger.info("Processing request")  # Includes request_id and user_id
    """
    
    def __init__(self, **kwargs: Any) -> None:
        self.context = kwargs
        self._token: Any = None
    
    def __enter__(self) -> "LogContext":
        self._token = structlog.contextvars.bind_contextvars(**self.context)
        return self
    
    def __exit__(self, *args: Any) -> None:
        structlog.contextvars.unbind_contextvars(*self.context.keys())


# Convenience function for one-off context binding
def bind_context(**kwargs: Any) -> None:
    """Bind context variables for all subsequent log calls in this context.
    
    Example:
        >>> bind_context(backtest_id="bt_123", strategy="ma_crossover")
        >>> logger.info("Starting backtest")  # Includes backtest_id and strategy
    """
    structlog.contextvars.bind_contextvars(**kwargs)


def clear_context() -> None:
    """Clear all bound context variables."""
    structlog.contextvars.clear_contextvars()
