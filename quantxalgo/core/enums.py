from __future__ import annotations

"""
Enumerations for QuantXalgo.

These enums provide type-safe constants used throughout the trading platform.
"""

from enum import Enum, auto
from typing import Optional


class Side(str, Enum):
    """Order/position side."""
    BUY = "BUY"
    SELL = "SELL"
    
    @property
    def sign(self) -> int:
        """Return +1 for BUY, -1 for SELL."""
        return 1 if self == Side.BUY else -1
    
    @property
    def opposite(self) -> "Side":
        """Return the opposite side."""
        return Side.SELL if self == Side.BUY else Side.BUY


class OrderType(str, Enum):
    """Order type definitions."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"
    TRAILING_STOP = "TRAILING_STOP"
    
    @property
    def requires_price(self) -> bool:
        """Whether this order type requires a limit price."""
        return self in (OrderType.LIMIT, OrderType.STOP_LIMIT)
    
    @property
    def requires_stop_price(self) -> bool:
        """Whether this order type requires a stop price."""
        return self in (OrderType.STOP, OrderType.STOP_LIMIT, OrderType.TRAILING_STOP)


class OrderStatus(str, Enum):
    """Order lifecycle status."""
    CREATED = "CREATED"           # Order created but not submitted
    PENDING = "PENDING"           # Awaiting risk approval
    SUBMITTED = "SUBMITTED"       # Sent to broker
    ACCEPTED = "ACCEPTED"         # Accepted by broker
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"            # Fully executed
    CANCELLED = "CANCELLED"      # Cancelled by user/system
    REJECTED = "REJECTED"        # Rejected by broker/risk
    EXPIRED = "EXPIRED"          # Time in force expired
    
    @property
    def is_terminal(self) -> bool:
        """Whether this is a terminal (final) state."""
        return self in (
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
        )
    
    @property
    def is_active(self) -> bool:
        """Whether the order is still active/working."""
        return self in (
            OrderStatus.PENDING,
            OrderStatus.SUBMITTED,
            OrderStatus.ACCEPTED,
            OrderStatus.PARTIALLY_FILLED,
        )


class TimeInForce(str, Enum):
    """Order time in force."""
    GTC = "GTC"   # Good Till Cancelled
    DAY = "DAY"   # Day order (expires at market close)
    IOC = "IOC"   # Immediate or Cancel
    FOK = "FOK"   # Fill or Kill
    GTD = "GTD"   # Good Till Date
    OPG = "OPG"   # At the Opening
    CLS = "CLS"   # At the Close


class AssetClass(str, Enum):
    """Asset class classification."""
    EQUITY = "EQUITY"
    ETF = "ETF"
    OPTION = "OPTION"
    FUTURE = "FUTURE"
    FX = "FX"
    CRYPTO = "CRYPTO"
    BOND = "BOND"
    COMMODITY = "COMMODITY"
    
    @property
    def is_derivative(self) -> bool:
        """Whether this is a derivative instrument."""
        return self in (AssetClass.OPTION, AssetClass.FUTURE)
    
    @property
    def typical_lot_size(self) -> int:
        """Typical minimum lot size for this asset class."""
        return {
            AssetClass.EQUITY: 1,
            AssetClass.ETF: 1,
            AssetClass.OPTION: 100,
            AssetClass.FUTURE: 1,
            AssetClass.FX: 1000,
            AssetClass.CRYPTO: 1,
            AssetClass.BOND: 1000,
            AssetClass.COMMODITY: 1,
        }.get(self, 1)


class SignalStrength(str, Enum):
    """Signal strength classification."""
    STRONG_SELL = "STRONG_SELL"
    SELL = "SELL"
    WEAK_SELL = "WEAK_SELL"
    NEUTRAL = "NEUTRAL"
    WEAK_BUY = "WEAK_BUY"
    BUY = "BUY"
    STRONG_BUY = "STRONG_BUY"
    
    @property
    def numeric_value(self) -> float:
        """Convert to numeric value in range [-1, 1]."""
        return {
            SignalStrength.STRONG_SELL: -1.0,
            SignalStrength.SELL: -0.66,
            SignalStrength.WEAK_SELL: -0.33,
            SignalStrength.NEUTRAL: 0.0,
            SignalStrength.WEAK_BUY: 0.33,
            SignalStrength.BUY: 0.66,
            SignalStrength.STRONG_BUY: 1.0,
        }[self]
    
    @property
    def side(self) -> Optional[Side]:
        """Get the implied trading side."""
        if self.numeric_value > 0:
            return Side.BUY
        elif self.numeric_value < 0:
            return Side.SELL
        return None


class HealthStatus(str, Enum):
    """Strategy health status."""
    HEALTHY = "HEALTHY"               # Performing well
    UNDERPERFORMING = "UNDERPERFORMING"  # Below expectations
    CRITICAL = "CRITICAL"             # Severe issues
    TERMINATED = "TERMINATED"         # Strategy stopped
    PAUSED = "PAUSED"                 # Temporarily paused
    
    @property
    def allows_trading(self) -> bool:
        """Whether trading is allowed in this state."""
        return self in (HealthStatus.HEALTHY, HealthStatus.UNDERPERFORMING)


class EventType(str, Enum):
    """Event type classification for the event-driven engine."""
    MARKET_DATA = "MARKET_DATA"
    SIGNAL = "SIGNAL"
    ORDER = "ORDER"
    FILL = "FILL"
    PORTFOLIO = "PORTFOLIO"
    RISK = "RISK"
    SYSTEM = "SYSTEM"
    
    @property
    def priority(self) -> int:
        """Event processing priority (lower = higher priority)."""
        return {
            EventType.RISK: 0,
            EventType.FILL: 1,
            EventType.ORDER: 2,
            EventType.SIGNAL: 3,
            EventType.MARKET_DATA: 4,
            EventType.PORTFOLIO: 5,
            EventType.SYSTEM: 6,
        }[self]


class Interval(str, Enum):
    """Data interval/timeframe."""
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAY_1 = "1d"
    WEEK_1 = "1w"
    MONTH_1 = "1M"
    
    @property
    def minutes(self) -> int:
        """Convert to minutes."""
        return {
            Interval.MINUTE_1: 1,
            Interval.MINUTE_5: 5,
            Interval.MINUTE_15: 15,
            Interval.MINUTE_30: 30,
            Interval.HOUR_1: 60,
            Interval.HOUR_4: 240,
            Interval.DAY_1: 1440,
            Interval.WEEK_1: 10080,
            Interval.MONTH_1: 43200,
        }[self]
    
    @property
    def is_intraday(self) -> bool:
        """Whether this is an intraday interval."""
        return self in (
            Interval.MINUTE_1,
            Interval.MINUTE_5,
            Interval.MINUTE_15,
            Interval.MINUTE_30,
            Interval.HOUR_1,
            Interval.HOUR_4,
        )


class RegimeType(str, Enum):
    """Market regime classification."""
    BULL_LOW_VOL = "BULL_LOW_VOL"
    BULL_HIGH_VOL = "BULL_HIGH_VOL"
    BEAR_LOW_VOL = "BEAR_LOW_VOL"
    BEAR_HIGH_VOL = "BEAR_HIGH_VOL"
    SIDEWAYS = "SIDEWAYS"
    CRISIS = "CRISIS"
