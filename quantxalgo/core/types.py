"""
Type definitions for QuantXalgo.

Uses TypedDict for structured data types with full type safety.
"""

from datetime import datetime
from typing import TypedDict, List, Dict, Any, Optional

from quantxalgo.core.enums import Side, OrderType, OrderStatus, TimeInForce, AssetClass


class BarData(TypedDict, total=False):
    """OHLCV bar data structure."""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: float
    trades: int
    interval: str


class TickData(TypedDict, total=False):
    """Tick-level market data."""
    symbol: str
    timestamp: datetime
    bid: float
    ask: float
    bid_size: float
    ask_size: float
    last: float
    last_size: float


class OrderData(TypedDict, total=False):
    """Order data structure."""
    order_id: str
    symbol: str
    side: Side
    order_type: OrderType
    quantity: float
    limit_price: float
    stop_price: float
    time_in_force: TimeInForce
    status: OrderStatus
    strategy: str
    created_at: datetime
    updated_at: datetime
    filled_quantity: float
    avg_fill_price: float


class FillData(TypedDict, total=False):
    """Order fill/execution data."""
    fill_id: str
    order_id: str
    symbol: str
    side: Side
    quantity: float
    price: float
    commission: float
    slippage: float
    timestamp: datetime
    is_partial: bool


class PositionData(TypedDict, total=False):
    """Position data structure."""
    symbol: str
    quantity: float
    avg_cost: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    realized_pnl: float
    entry_date: datetime
    asset_class: AssetClass
    strategy: str


class TradeData(TypedDict, total=False):
    """Completed trade (round-trip) data."""
    trade_id: str
    symbol: str
    side: Side
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_pct: float
    entry_time: datetime
    exit_time: datetime
    holding_period: int
    commission: float
    slippage: float
    strategy: str


class EquityPoint(TypedDict, total=False):
    """Equity curve data point."""
    timestamp: datetime
    equity: float
    cash: float
    positions_value: float
    drawdown: float
    drawdown_pct: float


class PortfolioSnapshot(TypedDict, total=False):
    """Portfolio state snapshot."""
    timestamp: datetime
    total_equity: float
    cash: float
    positions_value: float
    unrealized_pnl: float
    realized_pnl: float
    total_exposure: float
    net_exposure: float
    leverage: float
    positions: List[PositionData]


class RiskMetrics(TypedDict, total=False):
    """Risk metrics snapshot."""
    current_drawdown: float
    max_drawdown: float
    daily_var: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    beta: float
    correlation_to_benchmark: float


class PerformanceMetrics(TypedDict, total=False):
    """Performance metrics summary."""
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    avg_holding_period: float
    total_trades: int
    winning_trades: int
    losing_trades: int


class BacktestResult(TypedDict, total=False):
    """Backtest result summary."""
    strategy: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_equity: float
    performance: PerformanceMetrics
    equity_curve: List[EquityPoint]
    trades: List[TradeData]
    

class StrategyConfig(TypedDict, total=False):
    """Strategy configuration."""
    name: str
    enabled: bool
    symbols: List[str]
    params: Dict[str, Any]
    capital_allocation: float
    max_position_size: float

