from __future__ import annotations

"""
Custom exceptions for QuantXalgo.

All custom exceptions inherit from QuantXalgoError for easy catching.
"""

from typing import Optional, Any


class QuantXalgoError(Exception):
    """Base exception for all QuantXalgo errors."""
    
    def __init__(self, message: str, details: Optional[dict] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}
    
    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


# =============================================================================
# DATA ERRORS
# =============================================================================

class DataError(QuantXalgoError):
    """Base exception for data-related errors."""
    pass


class DataFetchError(DataError):
    """Error fetching data from a source."""
    
    def __init__(self, source: str, symbol: str, message: str) -> None:
        super().__init__(
            f"Failed to fetch data from {source} for {symbol}: {message}",
            {"source": source, "symbol": symbol}
        )


class DataValidationError(DataError):
    """Data validation failed."""
    pass


class MissingDataError(DataError):
    """Required data is missing."""
    
    def __init__(self, symbol: str, start_date: str, end_date: str) -> None:
        super().__init__(
            f"Missing data for {symbol} from {start_date} to {end_date}",
            {"symbol": symbol, "start_date": start_date, "end_date": end_date}
        )


class InsufficientDataError(DataError):
    """Not enough data for the requested operation."""
    
    def __init__(self, required: int, available: int, symbol: str = "") -> None:
        super().__init__(
            f"Insufficient data: required {required}, available {available}",
            {"required": required, "available": available, "symbol": symbol}
        )


# =============================================================================
# STRATEGY ERRORS
# =============================================================================

class StrategyError(QuantXalgoError):
    """Base exception for strategy-related errors."""
    pass


class StrategyNotFoundError(StrategyError):
    """Strategy not found in registry."""
    
    def __init__(self, strategy_name: str) -> None:
        super().__init__(
            f"Strategy '{strategy_name}' not found in registry",
            {"strategy_name": strategy_name}
        )


class StrategyInitializationError(StrategyError):
    """Failed to initialize strategy."""
    pass


class StrategyExecutionError(StrategyError):
    """Error during strategy execution."""
    pass


class InvalidParameterError(StrategyError):
    """Invalid strategy parameter."""
    
    def __init__(self, param_name: str, value: any, reason: str) -> None:
        super().__init__(
            f"Invalid parameter '{param_name}' = {value}: {reason}",
            {"param_name": param_name, "value": value, "reason": reason}
        )


# =============================================================================
# EXECUTION ERRORS
# =============================================================================

class ExecutionError(QuantXalgoError):
    """Base exception for execution-related errors."""
    pass


class OrderRejectedError(ExecutionError):
    """Order was rejected."""
    
    def __init__(self, order_id: str, reason: str) -> None:
        super().__init__(
            f"Order {order_id} rejected: {reason}",
            {"order_id": order_id, "reason": reason}
        )


class InsufficientFundsError(ExecutionError):
    """Insufficient funds for order."""
    
    def __init__(self, required: float, available: float) -> None:
        super().__init__(
            f"Insufficient funds: required {required:.2f}, available {available:.2f}",
            {"required": required, "available": available}
        )


class InsufficientPositionError(ExecutionError):
    """Insufficient position for sell order."""
    
    def __init__(self, symbol: str, required: float, available: float) -> None:
        super().__init__(
            f"Insufficient position in {symbol}: required {required}, available {available}",
            {"symbol": symbol, "required": required, "available": available}
        )


class BrokerError(ExecutionError):
    """Error from broker/exchange."""
    pass


# =============================================================================
# RISK ERRORS
# =============================================================================

class RiskError(QuantXalgoError):
    """Base exception for risk-related errors."""
    pass


class RiskLimitExceededError(RiskError):
    """Risk limit exceeded."""
    
    def __init__(self, limit_type: str, current: float, limit: float) -> None:
        super().__init__(
            f"Risk limit exceeded: {limit_type} = {current:.2%}, limit = {limit:.2%}",
            {"limit_type": limit_type, "current": current, "limit": limit}
        )


class DrawdownLimitExceededError(RiskError):
    """Drawdown limit exceeded - kill switch triggered."""
    
    def __init__(self, current_drawdown: float, max_drawdown: float) -> None:
        super().__init__(
            f"KILL SWITCH: Drawdown {current_drawdown:.2%} exceeds limit {max_drawdown:.2%}",
            {"current_drawdown": current_drawdown, "max_drawdown": max_drawdown}
        )


class PositionLimitExceededError(RiskError):
    """Position size limit exceeded."""
    pass


class ExposureLimitExceededError(RiskError):
    """Exposure limit exceeded."""
    pass


# =============================================================================
# CONFIGURATION ERRORS
# =============================================================================

class ConfigurationError(QuantXalgoError):
    """Base exception for configuration errors."""
    pass


class MissingConfigurationError(ConfigurationError):
    """Required configuration is missing."""
    
    def __init__(self, config_key: str) -> None:
        super().__init__(
            f"Missing required configuration: {config_key}",
            {"config_key": config_key}
        )


class InvalidConfigurationError(ConfigurationError):
    """Configuration value is invalid."""
    pass


# =============================================================================
# BACKTEST ERRORS
# =============================================================================

class BacktestError(QuantXalgoError):
    """Base exception for backtest errors."""
    pass


class BacktestConfigurationError(BacktestError):
    """Invalid backtest configuration."""
    pass


class BacktestExecutionError(BacktestError):
    """Error during backtest execution."""
    pass
