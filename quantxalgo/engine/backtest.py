"""
Event-Driven Backtesting Engine.

The core simulation engine that drives bar-by-bar backtesting with
realistic execution modeling.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import pandas as pd
import numpy as np

from quantxalgo.config.logging_config import get_logger
from quantxalgo.config.settings import get_settings
from quantxalgo.core.enums import Side, OrderType, TimeInForce
from quantxalgo.core.events import (
    MarketDataEvent,
    SignalEvent,
    OrderEvent,
    FillEvent,
)
from quantxalgo.core.types import PerformanceMetrics, BacktestResult, EquityPoint
from quantxalgo.core.exceptions import BacktestError, InsufficientDataError
from quantxalgo.strategies.base import Strategy, StrategyContext

logger = get_logger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    
    # Time range
    start_date: datetime
    end_date: datetime
    
    # Capital
    initial_capital: float = 1_000_000.0
    
    # Execution modeling
    slippage_bps: float = 5.0          # Slippage in basis points
    commission_per_share: float = 0.005  # Commission per share
    commission_min: float = 1.0         # Minimum commission
    
    # Position sizing
    default_position_size_pct: float = 0.02  # 2% of equity per trade
    max_position_size_pct: float = 0.10      # 10% max per position
    
    # Risk limits
    max_positions: int = 20
    max_exposure_pct: float = 1.0
    
    # Data
    warmup_bars: int = 200   # Bars for indicator warmup


@dataclass
class BacktestState:
    """Current state during backtest execution."""
    
    cash: float
    equity: float
    positions: dict = field(default_factory=dict)
    pending_orders: list = field(default_factory=list)
    trades: list = field(default_factory=list)
    equity_curve: list = field(default_factory=list)
    current_bar: int = 0


class Backtester:
    """Event-driven backtesting engine.
    
    Simulates strategy execution on historical data with realistic
    execution modeling including slippage, commissions, and partial fills.
    
    Example:
        >>> config = BacktestConfig(
        ...     start_date=datetime(2020, 1, 1),
        ...     end_date=datetime(2023, 12, 31),
        ...     initial_capital=1_000_000
        ... )
        >>> backtester = Backtester(config)
        >>> result = await backtester.run(strategy, data)
    """
    
    def __init__(self, config: BacktestConfig) -> None:
        """Initialize backtester.
        
        Args:
            config: Backtest configuration.
        """
        self.config = config
        self._state: Optional[BacktestState] = None
        self._context: Optional[StrategyContext] = None
    
    async def run(
        self,
        strategy: Strategy,
        data: dict[str, pd.DataFrame],
    ) -> BacktestResult:
        """Run backtest simulation.
        
        Args:
            strategy: Strategy to test.
            data: Dictionary mapping symbol to OHLCV DataFrame.
            
        Returns:
            BacktestResult with performance metrics and trades.
        """
        logger.info(
            "Starting backtest",
            strategy=strategy.name,
            start=self.config.start_date.isoformat(),
            end=self.config.end_date.isoformat(),
            capital=self.config.initial_capital,
        )
        
        # Validate data
        self._validate_data(data)
        
        # Initialize state
        self._state = BacktestState(
            cash=self.config.initial_capital,
            equity=self.config.initial_capital,
        )
        
        # Build aligned data index
        aligned_data, date_index = self._align_data(data)
        
        # Filter to date range
        start_idx = date_index.searchsorted(self.config.start_date)
        end_idx = date_index.searchsorted(self.config.end_date)
        
        # Ensure warmup period
        warmup_start = max(0, start_idx - self.config.warmup_bars)
        
        # Initialize context
        self._context = StrategyContext(
            cash=self._state.cash,
            equity=self._state.equity,
            bars={sym: df.iloc[:warmup_start] for sym, df in aligned_data.items()},
        )
        
        # Initialize strategy
        strategy.initialize(self._context)
        
        # Main simulation loop
        for i in range(warmup_start, end_idx):
            current_date = date_index[i]
            self._state.current_bar = i
            
            # Update bars in context
            for symbol, df in aligned_data.items():
                self._context.bars[symbol] = df.iloc[:i+1]
            
            # Skip warmup period for trading
            if i < start_idx:
                continue
            
            # Process each symbol
            for symbol, df in aligned_data.items():
                if i >= len(df):
                    continue
                
                bar = df.iloc[i]
                
                # Create market data event
                event = MarketDataEvent(
                    timestamp=current_date,
                    symbol=symbol,
                    open=bar["open"],
                    high=bar["high"],
                    low=bar["low"],
                    close=bar["close"],
                    volume=bar.get("volume", 0),
                )
                
                # Update positions with current prices
                self._update_position_prices(symbol, bar["close"])
                
                # Generate signals
                signals = strategy.generate_signals(self._context, event)
                
                # Process signals
                for signal in signals:
                    await self._process_signal(signal, bar)
            
            # Update equity curve
            self._update_equity_curve(current_date)
            
            # Update context
            self._context.cash = self._state.cash
            self._context.equity = self._state.equity
            self._context.positions = self._state.positions.copy()
        
        # Close all positions at end
        await self._close_all_positions(aligned_data, date_index[end_idx - 1])
        
        # Calculate final results
        result = self._calculate_results(strategy)
        
        logger.info(
            "Backtest complete",
            strategy=strategy.name,
            total_return=f"{result['performance']['total_return']:.2%}",
            sharpe=f"{result['performance']['sharpe_ratio']:.2f}",
            max_dd=f"{result['performance']['max_drawdown']:.2%}",
            trades=result['performance']['total_trades'],
        )
        
        return result
    
    async def _process_signal(
        self,
        signal: SignalEvent,
        bar: pd.Series,
    ) -> None:
        """Process a trading signal and execute order."""
        symbol = signal.symbol
        side = signal.side
        current_price = bar["close"]
        
        # Calculate position size
        position_value = self._state.equity * self.config.default_position_size_pct
        position_value = min(
            position_value,
            self._state.equity * self.config.max_position_size_pct,
        )
        position_value *= signal.strength  # Scale by signal strength
        
        quantity = position_value / current_price
        
        # Check if we have existing position
        current_position = self._state.positions.get(symbol, {})
        current_qty = current_position.get("quantity", 0)
        
        # Determine actual trade
        if side == Side.BUY:
            if current_qty < 0:
                # Close short first
                quantity = abs(current_qty)
            elif current_qty > 0:
                # Already long, skip or add
                return
        else:  # SELL
            if current_qty > 0:
                # Close long
                quantity = current_qty
            elif current_qty < 0:
                # Already short, skip
                return
            else:
                # New short position
                pass
        
        # Apply slippage
        slippage = current_price * (self.config.slippage_bps / 10000)
        if side == Side.BUY:
            exec_price = current_price + slippage
        else:
            exec_price = current_price - slippage
        
        # Calculate commission
        commission = max(
            quantity * self.config.commission_per_share,
            self.config.commission_min,
        )
        
        # Check if we have enough cash (for buys)
        trade_cost = quantity * exec_price + commission
        if side == Side.BUY and trade_cost > self._state.cash:
            # Reduce quantity to fit
            available = self._state.cash - commission
            quantity = available / exec_price
            trade_cost = quantity * exec_price + commission
        
        if quantity <= 0:
            return
        
        # Execute trade
        fill = FillEvent(
            timestamp=bar.name,
            order_id=f"order_{self._state.current_bar}_{symbol}",
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=exec_price,
            commission=commission,
            slippage=slippage * quantity,
            strategy=signal.strategy,
        )
        
        # Update cash
        if side == Side.BUY:
            self._state.cash -= trade_cost
        else:
            self._state.cash += quantity * exec_price - commission
        
        # Update positions
        self._update_position(fill)
        
        # Record trade
        self._state.trades.append({
            "timestamp": bar.name,
            "symbol": symbol,
            "side": side.value,
            "quantity": quantity,
            "price": exec_price,
            "commission": commission,
            "slippage": slippage * quantity,
        })
        
        logger.debug(
            "Trade executed",
            symbol=symbol,
            side=side.value,
            quantity=quantity,
            price=exec_price,
            commission=commission,
        )
    
    def _update_position(self, fill: FillEvent) -> None:
        """Update position after a fill."""
        symbol = fill.symbol
        
        if symbol not in self._state.positions:
            self._state.positions[symbol] = {
                "quantity": 0,
                "avg_cost": 0,
                "current_price": fill.price,
                "realized_pnl": 0,
            }
        
        pos = self._state.positions[symbol]
        old_qty = pos["quantity"]
        
        if fill.side == Side.BUY:
            new_qty = old_qty + fill.quantity
            if old_qty >= 0:
                # Adding to long or opening long
                total_cost = (pos["avg_cost"] * old_qty) + (fill.price * fill.quantity)
                pos["avg_cost"] = total_cost / new_qty if new_qty > 0 else 0
            else:
                # Closing short
                pnl = (pos["avg_cost"] - fill.price) * fill.quantity
                pos["realized_pnl"] += pnl
        else:  # SELL
            new_qty = old_qty - fill.quantity
            if old_qty > 0:
                # Closing long
                pnl = (fill.price - pos["avg_cost"]) * fill.quantity
                pos["realized_pnl"] += pnl
            else:
                # Adding to short or opening short
                if old_qty == 0:
                    pos["avg_cost"] = fill.price
                else:
                    total_cost = (pos["avg_cost"] * abs(old_qty)) + (fill.price * fill.quantity)
                    pos["avg_cost"] = total_cost / abs(new_qty) if new_qty != 0 else 0
        
        pos["quantity"] = new_qty
        pos["current_price"] = fill.price
        
        # Remove position if flat
        if abs(new_qty) < 0.0001:
            del self._state.positions[symbol]
    
    def _update_position_prices(self, symbol: str, price: float) -> None:
        """Update current prices for positions."""
        if symbol in self._state.positions:
            self._state.positions[symbol]["current_price"] = price
    
    def _update_equity_curve(self, timestamp: datetime) -> None:
        """Update equity curve."""
        positions_value = sum(
            pos["quantity"] * pos["current_price"]
            for pos in self._state.positions.values()
        )
        
        self._state.equity = self._state.cash + positions_value
        
        # Calculate drawdown
        if self._state.equity_curve:
            peak = max(pt["equity"] for pt in self._state.equity_curve)
            peak = max(peak, self._state.equity)
        else:
            peak = self._state.equity
        
        drawdown = (self._state.equity - peak) / peak if peak > 0 else 0
        
        self._state.equity_curve.append({
            "timestamp": timestamp,
            "equity": self._state.equity,
            "cash": self._state.cash,
            "positions_value": positions_value,
            "drawdown": drawdown,
            "drawdown_pct": drawdown,
        })
    
    async def _close_all_positions(
        self,
        data: dict[str, pd.DataFrame],
        final_date: datetime,
    ) -> None:
        """Close all open positions at end of backtest."""
        for symbol, pos in list(self._state.positions.items()):
            if pos["quantity"] == 0:
                continue
            
            # Get final price
            if symbol in data and len(data[symbol]) > 0:
                final_price = data[symbol]["close"].iloc[-1]
            else:
                final_price = pos["current_price"]
            
            side = Side.SELL if pos["quantity"] > 0 else Side.BUY
            quantity = abs(pos["quantity"])
            
            # Execute closing trade
            fill = FillEvent(
                timestamp=final_date,
                order_id=f"close_{symbol}",
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=final_price,
                commission=quantity * self.config.commission_per_share,
                strategy="CLOSE",
            )
            
            self._update_position(fill)
            
            if side == Side.SELL:
                self._state.cash += quantity * final_price
            else:
                self._state.cash -= quantity * final_price
    
    def _validate_data(self, data: dict[str, pd.DataFrame]) -> None:
        """Validate input data."""
        if not data:
            raise BacktestError("No data provided")
        
        for symbol, df in data.items():
            required_cols = ["open", "high", "low", "close"]
            missing = set(required_cols) - set(df.columns)
            if missing:
                raise BacktestError(f"Missing columns for {symbol}: {missing}")
            
            if len(df) < self.config.warmup_bars:
                raise InsufficientDataError(
                    required=self.config.warmup_bars,
                    available=len(df),
                    symbol=symbol,
                )
    
    def _align_data(
        self,
        data: dict[str, pd.DataFrame],
    ) -> tuple[dict[str, pd.DataFrame], pd.DatetimeIndex]:
        """Align data to common date index."""
        # Get union of all dates
        all_dates = set()
        for df in data.values():
            all_dates.update(df.index)
        
        date_index = pd.DatetimeIndex(sorted(all_dates))
        
        # Reindex all data
        aligned = {}
        for symbol, df in data.items():
            aligned[symbol] = df.reindex(date_index).ffill()
        
        return aligned, date_index
    
    def _calculate_results(self, strategy: Strategy) -> BacktestResult:
        """Calculate final backtest results."""
        equity_df = pd.DataFrame(self._state.equity_curve)
        
        if equity_df.empty:
            raise BacktestError("No equity curve data")
        
        equity_df.set_index("timestamp", inplace=True)
        
        # Calculate returns
        returns = equity_df["equity"].pct_change().dropna()
        
        # Performance metrics
        total_return = (
            self._state.equity - self.config.initial_capital
        ) / self.config.initial_capital
        
        ann_return = returns.mean() * 252
        ann_vol = returns.std() * np.sqrt(252)
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0
        
        # Sortino ratio
        downside = returns[returns < 0]
        downside_std = downside.std() * np.sqrt(252) if len(downside) > 0 else ann_vol
        sortino = ann_return / downside_std if downside_std > 0 else 0
        
        # Max drawdown
        cummax = equity_df["equity"].cummax()
        drawdowns = (equity_df["equity"] - cummax) / cummax
        max_dd = drawdowns.min()
        
        # Calmar ratio
        calmar = ann_return / abs(max_dd) if max_dd != 0 else 0
        
        # Trade statistics
        trades_df = pd.DataFrame(self._state.trades)
        total_trades = len(trades_df)
        
        if total_trades > 0:
            # This is simplified - would need entry/exit matching for accurate stats
            win_rate = 0.5  # Placeholder
            profit_factor = 1.0  # Placeholder
            avg_win = 0
            avg_loss = 0
        else:
            win_rate = 0
            profit_factor = 0
            avg_win = 0
            avg_loss = 0
        
        performance: PerformanceMetrics = {
            "total_return": total_return,
            "annualized_return": ann_return,
            "volatility": ann_vol,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_dd,
            "calmar_ratio": calmar,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "avg_holding_period": 0,
            "total_trades": total_trades,
            "winning_trades": 0,
            "losing_trades": 0,
        }
        
        return {
            "strategy": strategy.name,
            "start_date": self.config.start_date,
            "end_date": self.config.end_date,
            "initial_capital": self.config.initial_capital,
            "final_equity": self._state.equity,
            "performance": performance,
            "equity_curve": self._state.equity_curve,
            "trades": self._state.trades,
        }
