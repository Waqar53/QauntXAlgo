from __future__ import annotations

"""
Abstract base class for all trading strategies.

All strategies must inherit from this class and implement the required methods.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, Dict, List

import pandas as pd

from quantxalgo.config.logging_config import get_logger
from quantxalgo.core.enums import Side
from quantxalgo.core.events import SignalEvent, FillEvent, MarketDataEvent
from quantxalgo.core.types import BarData, PositionData

logger = get_logger(__name__)


@dataclass
class StrategyContext:
    """Context provided to strategies during execution.
    
    Contains market data, portfolio state, and utilities needed for strategy logic.
    """
    
    current_time: datetime = field(default_factory=datetime.utcnow)
    
    # Historical data per symbol
    bars: Dict[str, pd.DataFrame] = field(default_factory=dict)
    
    # Current positions
    positions: Dict[str, PositionData] = field(default_factory=dict)
    
    # Portfolio state
    cash: float = 0.0
    equity: float = 0.0
    
    # Computed features per symbol
    features: Dict[str, pd.DataFrame] = field(default_factory=dict)
    
    def get_position(self, symbol: str) -> Optional[PositionData]:
        """Get current position for a symbol."""
        return self.positions.get(symbol)
    
    def has_position(self, symbol: str) -> bool:
        """Check if we have a position in a symbol."""
        pos = self.positions.get(symbol)
        return pos is not None and pos.get("quantity", 0) != 0
    
    def get_bars(self, symbol: str, count: int = 100) -> pd.DataFrame:
        """Get last N bars for a symbol."""
        if symbol not in self.bars:
            return pd.DataFrame()
        return self.bars[symbol].tail(count)
    
    def get_close(self, symbol: str) -> Optional[float]:
        """Get latest close price for a symbol."""
        if symbol not in self.bars or self.bars[symbol].empty:
            return None
        return self.bars[symbol]["close"].iloc[-1]


class Strategy(ABC):
    """Abstract base class for all trading strategies.
    
    Strategies receive market data events and generate trading signals.
    They should be stateless where possible, with state stored in StrategyContext.
    
    Example:
        >>> class MyStrategy(Strategy):
        ...     def __init__(self):
        ...         super().__init__(name="MyStrategy", params={"fast": 10})
        ...     
        ...     def generate_signals(self, context, event):
        ...         if should_buy:
        ...             return [SignalEvent(...)]
        ...         return []
    """
    
    def __init__(
        self,
        name: str,
        params: Optional[Dict[str, Any]] = None,
        symbols: Optional[List[str]] = None,
    ) -> None:
        """Initialize strategy.
        
        Args:
            name: Unique name for this strategy instance.
            params: Strategy parameters (periods, thresholds, etc.).
            symbols: Symbols this strategy trades.
        """
        self.name = name
        self.params = params or {}
        self.symbols = symbols or []
        
        self._is_initialized = False
        self._trade_count = 0
        
        logger.info(
            "Strategy created",
            name=self.name,
            params=self.params,
            symbols=self.symbols,
        )
    
    @abstractmethod
    def initialize(self, context: StrategyContext) -> None:
        """Initialize strategy with historical data.
        
        Called once before backtesting or live trading starts.
        Use this to pre-compute indicators or warm up state.
        
        Args:
            context: Strategy context with historical data.
        """
        pass
    
    @abstractmethod
    def generate_signals(
        self,
        context: StrategyContext,
        event: MarketDataEvent,
    ) -> list[SignalEvent]:
        """Generate trading signals based on new market data.
        
        This is the core strategy logic. Called on each new bar.
        
        Args:
            context: Current strategy context.
            event: New market data event.
            
        Returns:
            List of signal events (can be empty).
        """
        pass
    
    def on_fill(self, fill: FillEvent) -> None:
        """Handle fill notification.
        
        Called when an order from this strategy is filled.
        Override to track trades, adjust state, etc.
        
        Args:
            fill: Fill event with execution details.
        """
        self._trade_count += 1
        logger.debug(
            "Strategy fill received",
            strategy=self.name,
            symbol=fill.symbol,
            side=fill.side.value,
            quantity=fill.quantity,
            price=fill.price,
        )
    
    def on_order_rejected(self, order_id: str, reason: str) -> None:
        """Handle order rejection.
        
        Called when an order from this strategy is rejected.
        
        Args:
            order_id: ID of rejected order.
            reason: Rejection reason.
        """
        logger.warning(
            "Order rejected",
            strategy=self.name,
            order_id=order_id,
            reason=reason,
        )
    
    @property
    @abstractmethod
    def required_history(self) -> int:
        """Number of historical bars required for initialization.
        
        The backtester will ensure this many bars are available
        before calling generate_signals.
        
        Returns:
            Number of bars required.
        """
        pass
    
    def create_signal(
        self,
        symbol: str,
        side: Side,
        strength: float = 1.0,
        target_quantity: Optional[float] = None,
        target_price: Optional[float] = None,
        reason: str = "",
        metadata: Optional[dict] = None,
    ) -> SignalEvent:
        """Helper method to create a signal event.
        
        Args:
            symbol: Symbol to trade.
            side: BUY or SELL.
            strength: Signal strength (0-1).
            target_quantity: Optional target quantity.
            target_price: Optional target price.
            reason: Human-readable reason for signal.
            metadata: Additional metadata.
            
        Returns:
            SignalEvent instance.
        """
        return SignalEvent(
            timestamp=datetime.utcnow(),
            symbol=symbol,
            side=side,
            strength=strength,
            strategy=self.name,
            target_quantity=target_quantity,
            target_price=target_price,
            reason=reason,
            metadata=metadata or {},
        )
    
    def should_trade(self, symbol: str, context: StrategyContext) -> bool:
        """Check if we should trade this symbol.
        
        Override for custom filtering logic.
        
        Args:
            symbol: Symbol to check.
            context: Current context.
            
        Returns:
            True if trading is allowed.
        """
        return symbol in self.symbols or not self.symbols
    
    @property
    def trade_count(self) -> int:
        """Number of trades executed by this strategy."""
        return self._trade_count
    
    @property
    def is_initialized(self) -> bool:
        """Whether strategy has been initialized."""
        return self._is_initialized
    
    def mark_initialized(self) -> None:
        """Mark strategy as initialized."""
        self._is_initialized = True
        logger.info("Strategy initialized", name=self.name)
    
    def reset(self) -> None:
        """Reset strategy state.
        
        Called between backtests or when restarting.
        """
        self._is_initialized = False
        self._trade_count = 0
        logger.debug("Strategy reset", name=self.name)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', params={self.params})"
