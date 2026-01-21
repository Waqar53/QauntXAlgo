"""
Portfolio Manager for QuantXalgo.

Central manager for tracking positions, cash, equity, and portfolio state.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import pandas as pd

from quantxalgo.config.logging_config import get_logger
from quantxalgo.core.enums import Side
from quantxalgo.core.events import FillEvent
from quantxalgo.core.types import PositionData, PortfolioSnapshot
from quantxalgo.portfolio.position import Position

logger = get_logger(__name__)


class PortfolioManager:
    """Central portfolio management system.
    
    Tracks all positions, cash, and provides portfolio-level analytics.
    
    Example:
        >>> pm = PortfolioManager(initial_capital=1_000_000)
        >>> pm.on_fill(fill_event)
        >>> print(pm.total_equity)
    """
    
    def __init__(self, initial_capital: float) -> None:
        """Initialize portfolio manager.
        
        Args:
            initial_capital: Starting cash.
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: dict[str, Position] = {}
        
        # Trade history
        self.trades: list[dict] = []
        
        # Equity curve
        self.equity_curve: list[dict] = []
        
        # Peak equity for drawdown
        self._peak_equity = initial_capital
        
        logger.info(
            "Portfolio initialized",
            initial_capital=initial_capital,
        )
    
    def on_fill(self, fill: FillEvent) -> None:
        """Process a fill event.
        
        Updates positions and cash based on the fill.
        
        Args:
            fill: Fill event with execution details.
        """
        symbol = fill.symbol
        
        # Ensure position exists
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)
        
        pos = self.positions[symbol]
        
        # Calculate cash impact
        if fill.side == Side.BUY:
            cash_delta = -(fill.quantity * fill.price + fill.commission)
            pos.add(fill.quantity, fill.price, fill.timestamp)
        else:  # SELL
            cash_delta = fill.quantity * fill.price - fill.commission
            pos.add(-fill.quantity, fill.price, fill.timestamp)
        
        self.cash += cash_delta
        
        # Record trade
        self.trades.append({
            "timestamp": fill.timestamp,
            "symbol": symbol,
            "side": fill.side.value,
            "quantity": fill.quantity,
            "price": fill.price,
            "commission": fill.commission,
            "cash_delta": cash_delta,
        })
        
        # Remove flat positions
        if pos.is_flat:
            del self.positions[symbol]
        
        logger.debug(
            "Fill processed",
            symbol=symbol,
            side=fill.side.value,
            quantity=fill.quantity,
            price=fill.price,
            cash=self.cash,
        )
    
    def update_prices(self, prices: dict[str, float]) -> None:
        """Update current prices for all positions.
        
        Args:
            prices: Dictionary mapping symbol to current price.
        """
        for symbol, price in prices.items():
            if symbol in self.positions:
                self.positions[symbol].update_price(price)
    
    @property
    def total_equity(self) -> float:
        """Total portfolio value (cash + positions)."""
        return self.cash + self.positions_value
    
    @property
    def positions_value(self) -> float:
        """Total value of all positions."""
        return sum(pos.market_value for pos in self.positions.values())
    
    @property
    def total_exposure(self) -> float:
        """Total market exposure (absolute value of positions)."""
        return sum(pos.exposure for pos in self.positions.values())
    
    @property
    def net_exposure(self) -> float:
        """Net market exposure (long - short)."""
        return sum(pos.market_value for pos in self.positions.values())
    
    @property
    def gross_exposure(self) -> float:
        """Gross exposure (|long| + |short|)."""
        return self.total_exposure
    
    @property
    def leverage(self) -> float:
        """Current leverage ratio."""
        return self.total_exposure / self.total_equity if self.total_equity > 0 else 0
    
    @property
    def unrealized_pnl(self) -> float:
        """Total unrealized P&L across all positions."""
        return sum(pos.unrealized_pnl for pos in self.positions.values())
    
    @property
    def realized_pnl(self) -> float:
        """Total realized P&L."""
        return sum(pos.realized_pnl for pos in self.positions.values())
    
    @property
    def total_pnl(self) -> float:
        """Total P&L (realized + unrealized)."""
        return self.total_equity - self.initial_capital
    
    @property
    def total_return(self) -> float:
        """Total return as a decimal."""
        return self.total_pnl / self.initial_capital
    
    @property
    def current_drawdown(self) -> float:
        """Current drawdown from peak."""
        self._peak_equity = max(self._peak_equity, self.total_equity)
        return (self.total_equity - self._peak_equity) / self._peak_equity
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a symbol."""
        return self.positions.get(symbol)
    
    def has_position(self, symbol: str) -> bool:
        """Check if we have a position in a symbol."""
        pos = self.positions.get(symbol)
        return pos is not None and not pos.is_flat
    
    def get_position_quantity(self, symbol: str) -> float:
        """Get position quantity for a symbol."""
        pos = self.positions.get(symbol)
        return pos.quantity if pos else 0.0
    
    def record_equity(self, timestamp: datetime) -> None:
        """Record current equity for the equity curve.
        
        Args:
            timestamp: Current timestamp.
        """
        self.equity_curve.append({
            "timestamp": timestamp,
            "equity": self.total_equity,
            "cash": self.cash,
            "positions_value": self.positions_value,
            "drawdown_pct": self.current_drawdown,
        })
    
    def snapshot(self) -> PortfolioSnapshot:
        """Get current portfolio snapshot."""
        return {
            "timestamp": datetime.utcnow(),
            "total_equity": self.total_equity,
            "cash": self.cash,
            "positions_value": self.positions_value,
            "unrealized_pnl": self.unrealized_pnl,
            "realized_pnl": self.realized_pnl,
            "total_exposure": self.total_exposure,
            "net_exposure": self.net_exposure,
            "leverage": self.leverage,
            "positions": [pos.to_dict() for pos in self.positions.values()],
        }
    
    def get_equity_dataframe(self) -> pd.DataFrame:
        """Get equity curve as DataFrame."""
        if not self.equity_curve:
            return pd.DataFrame()
        return pd.DataFrame(self.equity_curve).set_index("timestamp")
    
    def reset(self) -> None:
        """Reset portfolio to initial state."""
        self.cash = self.initial_capital
        self.positions.clear()
        self.trades.clear()
        self.equity_curve.clear()
        self._peak_equity = self.initial_capital
        
        logger.info("Portfolio reset", initial_capital=self.initial_capital)
