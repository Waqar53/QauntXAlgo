"""
Position tracking and management.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np


@dataclass
class Position:
    """Represents a position in a single instrument.
    
    Tracks quantity, cost basis, and calculates P&L.
    """
    
    symbol: str
    quantity: float = 0.0
    avg_cost: float = 0.0
    current_price: float = 0.0
    entry_date: Optional[datetime] = None
    
    # Realized P&L from closed portions
    realized_pnl: float = 0.0
    
    def update_price(self, price: float) -> None:
        """Update current market price."""
        self.current_price = price
    
    def add(self, quantity: float, price: float, date: Optional[datetime] = None) -> None:
        """Add to position (buy for long, sell for short).
        
        Args:
            quantity: Quantity to add (positive for buys).
            price: Execution price.
            date: Trade date.
        """
        if self.quantity == 0:
            # New position
            self.avg_cost = price
            self.quantity = quantity
            self.entry_date = date or datetime.utcnow()
        elif np.sign(self.quantity) == np.sign(quantity):
            # Adding to existing position
            total_cost = (self.avg_cost * abs(self.quantity)) + (price * abs(quantity))
            self.quantity += quantity
            self.avg_cost = total_cost / abs(self.quantity)
        else:
            # Reducing or reversing position
            if abs(quantity) <= abs(self.quantity):
                # Partial close
                pnl = (price - self.avg_cost) * abs(quantity) * np.sign(self.quantity)
                self.realized_pnl += pnl
                self.quantity += quantity
            else:
                # Full close + reversal
                pnl = (price - self.avg_cost) * abs(self.quantity) * np.sign(self.quantity)
                self.realized_pnl += pnl
                remaining = quantity + self.quantity
                self.quantity = remaining
                self.avg_cost = price
                self.entry_date = date or datetime.utcnow()
        
        self.current_price = price
    
    def reduce(self, quantity: float, price: float) -> float:
        """Reduce position and return realized P&L.
        
        Args:
            quantity: Quantity to reduce (positive value).
            price: Execution price.
            
        Returns:
            Realized P&L from this reduction.
        """
        reduce_qty = min(abs(quantity), abs(self.quantity))
        
        if self.is_long:
            pnl = (price - self.avg_cost) * reduce_qty
            self.quantity -= reduce_qty
        else:
            pnl = (self.avg_cost - price) * reduce_qty
            self.quantity += reduce_qty
        
        self.realized_pnl += pnl
        self.current_price = price
        
        return pnl
    
    @property
    def is_long(self) -> bool:
        """Whether position is long."""
        return self.quantity > 0
    
    @property
    def is_short(self) -> bool:
        """Whether position is short."""
        return self.quantity < 0
    
    @property
    def is_flat(self) -> bool:
        """Whether position is flat (no exposure)."""
        return abs(self.quantity) < 1e-8
    
    @property
    def market_value(self) -> float:
        """Current market value of position."""
        return self.quantity * self.current_price
    
    @property
    def cost_basis(self) -> float:
        """Total cost basis."""
        return abs(self.quantity) * self.avg_cost
    
    @property
    def unrealized_pnl(self) -> float:
        """Unrealized P&L based on current price."""
        if self.is_long:
            return (self.current_price - self.avg_cost) * self.quantity
        else:
            return (self.avg_cost - self.current_price) * abs(self.quantity)
    
    @property
    def unrealized_pnl_pct(self) -> float:
        """Unrealized P&L as percentage."""
        if self.cost_basis == 0:
            return 0.0
        return self.unrealized_pnl / self.cost_basis * 100
    
    @property
    def total_pnl(self) -> float:
        """Total P&L (realized + unrealized)."""
        return self.realized_pnl + self.unrealized_pnl
    
    @property
    def exposure(self) -> float:
        """Absolute exposure (market value)."""
        return abs(self.market_value)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "avg_cost": self.avg_cost,
            "current_price": self.current_price,
            "market_value": self.market_value,
            "unrealized_pnl": self.unrealized_pnl,
            "unrealized_pnl_pct": self.unrealized_pnl_pct,
            "realized_pnl": self.realized_pnl,
            "entry_date": self.entry_date.isoformat() if self.entry_date else None,
        }
    
    def __repr__(self) -> str:
        direction = "LONG" if self.is_long else "SHORT" if self.is_short else "FLAT"
        return (
            f"Position({self.symbol}, {direction}, "
            f"qty={self.quantity:.2f}, avg={self.avg_cost:.2f}, "
            f"pnl={self.unrealized_pnl:.2f})"
        )
