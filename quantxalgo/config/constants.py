"""
System-wide constants for QuantXalgo.

These are immutable values used throughout the platform.
"""

from decimal import Decimal
from typing import Final

# =============================================================================
# TIME CONSTANTS
# =============================================================================

TRADING_DAYS_PER_YEAR: Final[int] = 252
TRADING_HOURS_PER_DAY: Final[float] = 6.5  # US markets: 9:30 AM - 4:00 PM
MINUTES_PER_TRADING_DAY: Final[int] = 390

# Annualization factors for different frequencies
ANNUALIZATION_FACTORS: Final[dict[str, float]] = {
    "1m": 252 * 390,      # 1 minute bars
    "5m": 252 * 78,       # 5 minute bars
    "15m": 252 * 26,      # 15 minute bars
    "1h": 252 * 6.5,      # Hourly bars
    "1d": 252,            # Daily bars
    "1w": 52,             # Weekly bars
    "1M": 12,             # Monthly bars
}

# =============================================================================
# FINANCIAL CONSTANTS
# =============================================================================

DEFAULT_RISK_FREE_RATE: Final[float] = 0.05  # 5% annual
MIN_VOLATILITY: Final[float] = 0.0001  # Minimum volatility to avoid div by zero

# Basis points conversions
BPS_TO_DECIMAL: Final[float] = 0.0001
PERCENT_TO_DECIMAL: Final[float] = 0.01

# =============================================================================
# ASSET CLASSES
# =============================================================================

SUPPORTED_ASSET_CLASSES: Final[tuple[str, ...]] = (
    "EQUITY",
    "ETF",
    "OPTION",
    "FUTURE",
    "FX",
    "CRYPTO",
)

# =============================================================================
# ORDER TYPES
# =============================================================================

SUPPORTED_ORDER_TYPES: Final[tuple[str, ...]] = (
    "MARKET",
    "LIMIT",
    "STOP",
    "STOP_LIMIT",
    "TRAILING_STOP",
)

SUPPORTED_TIME_IN_FORCE: Final[tuple[str, ...]] = (
    "GTC",   # Good Till Cancelled
    "DAY",   # Day only
    "IOC",   # Immediate or Cancel
    "FOK",   # Fill or Kill
    "GTD",   # Good Till Date
)

# =============================================================================
# DATA INTERVALS
# =============================================================================

SUPPORTED_INTERVALS: Final[tuple[str, ...]] = (
    "1m", "5m", "15m", "30m",  # Intraday
    "1h", "4h",                 # Hourly
    "1d", "1w", "1M",           # Daily+
)

INTERVAL_TO_MINUTES: Final[dict[str, int]] = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "4h": 240,
    "1d": 1440,
    "1w": 10080,
    "1M": 43200,
}

# =============================================================================
# RISK THRESHOLDS
# =============================================================================

# Default risk limits (can be overridden in settings)
DEFAULT_MAX_POSITION_SIZE: Final[float] = 0.10      # 10% of portfolio
DEFAULT_MAX_SECTOR_EXPOSURE: Final[float] = 0.25    # 25% per sector
DEFAULT_MAX_LEVERAGE: Final[float] = 2.0            # 2x leverage
DEFAULT_MAX_DRAWDOWN: Final[float] = 0.15           # 15% max drawdown

# Kill switch thresholds
CRITICAL_DRAWDOWN_THRESHOLD: Final[float] = 0.20    # 20% triggers review
EMERGENCY_DRAWDOWN_THRESHOLD: Final[float] = 0.30   # 30% liquidation

# =============================================================================
# STRATEGY HEALTH
# =============================================================================

STRATEGY_EVALUATION_WINDOW: Final[int] = 63  # ~3 months
MIN_SHARPE_THRESHOLD: Final[float] = 0.0     # Minimum acceptable Sharpe
MIN_TRADES_FOR_EVALUATION: Final[int] = 30   # Minimum trades to evaluate

# =============================================================================
# CACHE KEYS
# =============================================================================

CACHE_KEY_PREFIX: Final[str] = "qx"
CACHE_KEY_OHLCV: Final[str] = f"{CACHE_KEY_PREFIX}:ohlcv"
CACHE_KEY_FEATURES: Final[str] = f"{CACHE_KEY_PREFIX}:features"
CACHE_KEY_PORTFOLIO: Final[str] = f"{CACHE_KEY_PREFIX}:portfolio"

# =============================================================================
# API LIMITS
# =============================================================================

MAX_SYMBOLS_PER_REQUEST: Final[int] = 100
MAX_BARS_PER_REQUEST: Final[int] = 10000
DEFAULT_PAGE_SIZE: Final[int] = 50

# =============================================================================
# DECIMAL PRECISION
# =============================================================================

PRICE_PRECISION: Final[int] = 4
QUANTITY_PRECISION: Final[int] = 8
PERCENTAGE_PRECISION: Final[int] = 6
