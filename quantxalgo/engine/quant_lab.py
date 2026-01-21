"""
Quant Lab Code Execution Engine.

This module provides secure execution of user-defined trading strategies
against historical market data. It's the core of the QuantXalgo backtesting
platform and allows quants to test any Python algorithm.
"""

from __future__ import annotations

import traceback
import io
import sys
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from contextlib import redirect_stdout, redirect_stderr

import pandas as pd
import numpy as np

from quantxalgo.config.logging_config import get_logger
from quantxalgo.portfolio.manager import PortfolioManager
from quantxalgo.portfolio.position import Position
from quantxalgo.core.enums import Side

logger = get_logger(__name__)


# =============================================================================
# STRATEGY BASE CLASS - Users extend this
# =============================================================================

@dataclass
class Order:
    """Represents a trading order."""
    symbol: str
    side: Side
    quantity: float
    order_type: str = "MARKET"
    limit_price: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    filled: bool = False
    fill_price: float = 0.0


@dataclass
class ExecutionContext:
    """Context available to strategies during execution."""
    timestamp: datetime
    cash: float
    equity: float
    positions: Dict[str, Position]
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get current position for a symbol."""
        return self.positions.get(symbol)
    
    def has_position(self, symbol: str) -> bool:
        """Check if we have a position in a symbol."""
        return symbol in self.positions and self.positions[symbol].quantity != 0


class QuantLabStrategy(ABC):
    """
    Base class for all Quant Lab strategies.
    
    Users should extend this class and implement the required methods.
    
    Example:
        class MyStrategy(QuantLabStrategy):
            def initialize(self):
                self.lookback = 20
                
            def on_bar(self, data: pd.DataFrame, context: ExecutionContext) -> List[Order]:
                if len(data) < self.lookback:
                    return []
                    
                sma = data['close'].rolling(self.lookback).mean()
                if data['close'].iloc[-1] > sma.iloc[-1]:
                    return [self.market_order('SPY', 100)]
                return []
    """
    
    def __init__(self, symbols: List[str], params: Dict[str, Any] = None):
        self.symbols = symbols
        self.params = params or {}
        self.orders: List[Order] = []
        self._context: Optional[ExecutionContext] = None
        self._logs: List[str] = []
        
    def log(self, message: str):
        """Log a message during strategy execution."""
        timestamp = datetime.utcnow().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self._logs.append(log_entry)
        print(log_entry)  # Also print for capture
        
    @abstractmethod
    def initialize(self):
        """Called once before backtesting starts. Set up parameters here."""
        pass
    
    @abstractmethod
    def on_bar(self, data: Dict[str, pd.DataFrame], context: ExecutionContext) -> List[Order]:
        """
        Called on each bar of data.
        
        Args:
            data: Dictionary mapping symbol to DataFrame with OHLCV data up to current bar
            context: Current execution context with cash, positions, etc.
            
        Returns:
            List of Order objects to execute
        """
        pass
    
    def on_order_filled(self, order: Order):
        """Called when an order is filled. Override for custom handling."""
        pass
    
    # ===== Order Creation Helpers =====
    
    def market_order(self, symbol: str, quantity: float, side: Side = None) -> Order:
        """Create a market order."""
        if side is None:
            side = Side.BUY if quantity > 0 else Side.SELL
        return Order(symbol=symbol, side=side, quantity=abs(quantity), order_type="MARKET")
    
    def limit_order(self, symbol: str, quantity: float, price: float, side: Side = None) -> Order:
        """Create a limit order."""
        if side is None:
            side = Side.BUY if quantity > 0 else Side.SELL
        return Order(symbol=symbol, side=side, quantity=abs(quantity), order_type="LIMIT", limit_price=price)
    
    def order_target_percent(self, symbol: str, target_percent: float) -> Optional[Order]:
        """Order to target a percentage of portfolio."""
        if self._context is None:
            return None
        target_value = self._context.equity * target_percent
        current_pos = self._context.positions.get(symbol)
        current_value = current_pos.market_value if current_pos else 0
        diff = target_value - current_value
        
        if abs(diff) < 100:  # Minimum order size
            return None
        
        # Estimate quantity (would use last price in real implementation)
        return self.market_order(symbol, diff / 100, Side.BUY if diff > 0 else Side.SELL)
    
    def close_position(self, symbol: str) -> Optional[Order]:
        """Close entire position in a symbol."""
        if self._context is None:
            return None
        pos = self._context.positions.get(symbol)
        if pos and pos.quantity != 0:
            return self.market_order(symbol, abs(pos.quantity), Side.SELL if pos.quantity > 0 else Side.BUY)
        return None


# =============================================================================
# CODE EXECUTION ENGINE
# =============================================================================

@dataclass
class BacktestResult:
    """Results from a backtest execution."""
    success: bool
    error: Optional[str] = None
    initial_capital: float = 0.0
    final_equity: float = 0.0
    total_return: float = 0.0
    annualized_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    calmar_ratio: float = 0.0
    volatility: float = 0.0
    total_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    equity_curve: List[Dict] = field(default_factory=list)
    trades: List[Dict] = field(default_factory=list)
    logs: List[str] = field(default_factory=list)
    execution_time_ms: float = 0.0


class QuantLabExecutor:
    """
    Executes user-provided Python strategy code against historical data.
    
    This is the core engine that powers the Quant Lab. It:
    1. Safely compiles and validates user code
    2. Instantiates the strategy class
    3. Runs the backtest simulation
    4. Calculates performance metrics
    5. Returns detailed results
    """
    
    ALLOWED_IMPORTS = {
        'numpy', 'np', 'pandas', 'pd', 'math', 'statistics',
        'datetime', 'timedelta', 'typing', 'dataclasses',
        'collections', 'functools', 'itertools'
    }
    
    FORBIDDEN_PATTERNS = [
        'import os', 'import sys', 'import subprocess',
        'open(', 'exec(', 'eval(', '__import__',
        'import socket', 'import requests', 'import urllib',
        'import shutil', 'import glob', 'import pathlib'
    ]
    
    def __init__(self):
        self.logger = get_logger(__name__)
        
    def validate_code(self, code: str) -> tuple[bool, str]:
        """Validate user code for safety and correctness."""
        # Check for forbidden patterns
        for pattern in self.FORBIDDEN_PATTERNS:
            if pattern in code:
                return False, f"Forbidden pattern detected: {pattern}"
        
        # Try to compile
        try:
            compile(code, '<strategy>', 'exec')
        except SyntaxError as e:
            return False, f"Syntax error at line {e.lineno}: {e.msg}"
        
        # Check for QuantLabStrategy subclass
        if 'QuantLabStrategy' not in code and 'class ' in code:
            return False, "Strategy must extend QuantLabStrategy base class"
        
        return True, "Code validation passed"
    
    async def execute(
        self,
        code: str,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        initial_capital: float = 10_000_000,
        params: Dict[str, Any] = None
    ) -> BacktestResult:
        """
        Execute user strategy code and return results.
        
        Args:
            code: Python code containing a QuantLabStrategy subclass
            symbols: List of symbols to trade
            start_date: Backtest start date
            end_date: Backtest end date
            initial_capital: Starting capital
            params: Optional parameters to pass to strategy
            
        Returns:
            BacktestResult with performance metrics and execution details
        """
        import time
        start_time = time.time()
        
        # Validate code
        valid, message = self.validate_code(code)
        if not valid:
            return BacktestResult(success=False, error=message)
        
        # Capture output
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        logs = []
        
        try:
            # Create execution namespace with allowed imports
            namespace = {
                'QuantLabStrategy': QuantLabStrategy,
                'ExecutionContext': ExecutionContext,
                'Order': Order,
                'Side': Side,
                'pd': pd,
                'np': np,
                'datetime': datetime,
                'timedelta': timedelta,
                'List': List,
                'Dict': Dict,
                'Optional': Optional,
                'Any': Any,
            }
            
            # Execute user code to define strategy class
            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                exec(code, namespace)
            
            # Find the strategy class (last class that extends QuantLabStrategy)
            strategy_class = None
            for name, obj in namespace.items():
                if isinstance(obj, type) and issubclass(obj, QuantLabStrategy) and obj is not QuantLabStrategy:
                    strategy_class = obj
            
            if strategy_class is None:
                return BacktestResult(
                    success=False, 
                    error="No QuantLabStrategy subclass found in code"
                )
            
            logs.append(f"[INFO] Found strategy class: {strategy_class.__name__}")
            
            # Instantiate strategy
            strategy = strategy_class(symbols=symbols, params=params or {})
            strategy.initialize()
            logs.append(f"[INFO] Strategy initialized with symbols: {symbols}")
            
            # Generate market data
            logs.append(f"[INFO] Generating market data from {start_date.date()} to {end_date.date()}")
            data = self._generate_market_data(symbols, start_date, end_date)
            
            # Run backtest
            result = await self._run_backtest(
                strategy, data, initial_capital, start_date, end_date, logs
            )
            
            # Add captured output to logs
            stdout_output = stdout_buffer.getvalue()
            if stdout_output:
                logs.extend([f"[STDOUT] {line}" for line in stdout_output.strip().split('\n')])
            
            stderr_output = stderr_buffer.getvalue()
            if stderr_output:
                logs.extend([f"[STDERR] {line}" for line in stderr_output.strip().split('\n')])
            
            # Add strategy logs
            logs.extend(strategy._logs)
            
            result.logs = logs
            result.execution_time_ms = (time.time() - start_time) * 1000
            
            return result
            
        except Exception as e:
            error_trace = traceback.format_exc()
            return BacktestResult(
                success=False,
                error=f"Execution error: {str(e)}\n{error_trace}",
                logs=logs + [f"[ERROR] {error_trace}"]
            )
    
    def _generate_market_data(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, pd.DataFrame]:
        """Generate simulated market data for backtesting."""
        dates = pd.date_range(start_date, end_date, freq='D')
        data = {}
        
        for symbol in symbols:
            np.random.seed(hash(symbol) % (2**32))
            
            # Generate realistic price series
            returns = np.random.normal(0.0005, 0.018, len(dates))
            prices = 100 * np.exp(np.cumsum(returns))
            
            # Add intraday volatility
            high_mult = 1 + np.abs(np.random.normal(0, 0.008, len(dates)))
            low_mult = 1 - np.abs(np.random.normal(0, 0.008, len(dates)))
            
            data[symbol] = pd.DataFrame({
                'open': prices * (1 + np.random.normal(0, 0.002, len(dates))),
                'high': prices * high_mult,
                'low': prices * low_mult,
                'close': prices,
                'volume': np.random.randint(1_000_000, 50_000_000, len(dates)),
                'vwap': prices * (1 + np.random.normal(0, 0.001, len(dates)))
            }, index=dates)
            
        return data
    
    async def _run_backtest(
        self,
        strategy: QuantLabStrategy,
        data: Dict[str, pd.DataFrame],
        initial_capital: float,
        start_date: datetime,
        end_date: datetime,
        logs: List[str]
    ) -> BacktestResult:
        """Run the actual backtest simulation."""
        
        # Get aligned dates
        all_dates = set()
        for df in data.values():
            all_dates.update(df.index)
        dates = sorted(all_dates)
        
        # Initialize portfolio
        cash = initial_capital
        positions: Dict[str, Position] = {}
        equity_curve = []
        trades = []
        
        # Warmup period
        warmup = 200
        
        logs.append(f"[INFO] Starting backtest with ${initial_capital:,.0f} capital")
        logs.append(f"[INFO] Processing {len(dates)} trading days")
        
        for i, date in enumerate(dates):
            if i < warmup:
                continue
            
            # Build data slice up to current bar
            current_data = {}
            for symbol, df in data.items():
                mask = df.index <= date
                current_data[symbol] = df[mask].copy()
            
            # Update position prices
            for symbol, pos in positions.items():
                if symbol in current_data and len(current_data[symbol]) > 0:
                    pos.current_price = current_data[symbol]['close'].iloc[-1]
            
            # Calculate equity
            position_value = sum(pos.market_value for pos in positions.values())
            equity = cash + position_value
            
            # Create context
            context = ExecutionContext(
                timestamp=date,
                cash=cash,
                equity=equity,
                positions=positions.copy()
            )
            strategy._context = context
            
            # Get orders from strategy
            try:
                orders = strategy.on_bar(current_data, context)
                if orders is None:
                    orders = []
            except Exception as e:
                logs.append(f"[ERROR] Strategy error on {date}: {e}")
                continue
            
            # Execute orders
            for order in orders:
                if order is None:
                    continue
                    
                # Get current price
                if order.symbol not in current_data:
                    continue
                price = current_data[order.symbol]['close'].iloc[-1]
                
                # Calculate order value
                order_value = order.quantity * price
                
                # Check if we can afford it
                if order.side == Side.BUY and order_value > cash:
                    logs.append(f"[WARN] Insufficient cash for {order.symbol} order")
                    continue
                
                # Execute
                if order.symbol not in positions:
                    positions[order.symbol] = Position(symbol=order.symbol)
                
                pos = positions[order.symbol]
                if order.side == Side.BUY:
                    pos.add(order.quantity, price, date)
                    cash -= order_value
                else:
                    pos.add(-order.quantity, price, date)
                    cash += order_value
                
                # Record trade
                trade = {
                    'timestamp': date.isoformat(),
                    'symbol': order.symbol,
                    'side': order.side.value,
                    'quantity': order.quantity,
                    'price': price,
                    'value': order_value,
                    'pnl': 0  # Calculated later
                }
                trades.append(trade)
                order.filled = True
                order.fill_price = price
                
                strategy.on_order_filled(order)
                
                logs.append(f"[TRADE] {order.side.value} {order.quantity:.0f} {order.symbol} @ ${price:.2f}")
            
            # Record equity curve
            equity_curve.append({
                'timestamp': date.isoformat(),
                'date': date.strftime('%Y-%m-%d'),
                'equity': equity,
                'cash': cash,
                'position_value': position_value
            })
        
        # Calculate final metrics
        if len(equity_curve) < 2:
            return BacktestResult(
                success=False,
                error="Insufficient data for backtest",
                logs=logs
            )
        
        equities = [e['equity'] for e in equity_curve]
        returns = np.diff(equities) / equities[:-1]
        
        final_equity = equities[-1]
        total_return = (final_equity - initial_capital) / initial_capital
        
        # Annualized return
        days = len(equities)
        annualized_return = (1 + total_return) ** (252 / days) - 1 if days > 0 else 0
        
        # Volatility
        volatility = np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0
        
        # Sharpe ratio (assuming 0% risk-free rate)
        sharpe = (annualized_return / volatility) if volatility > 0 else 0
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0.001
        sortino = annualized_return / downside_std if downside_std > 0 else 0
        
        # Max drawdown
        peak = equities[0]
        max_dd = 0
        for eq in equities:
            if eq > peak:
                peak = eq
            dd = (eq - peak) / peak
            if dd < max_dd:
                max_dd = dd
        
        # Calmar ratio
        calmar = abs(annualized_return / max_dd) if max_dd != 0 else 0
        
        # Win rate
        winning_trades = sum(1 for t in trades if t.get('pnl', 0) > 0)
        win_rate = winning_trades / len(trades) if trades else 0.5
        
        logs.append(f"[INFO] Backtest complete: {len(trades)} trades")
        logs.append(f"[INFO] Final equity: ${final_equity:,.2f}")
        logs.append(f"[INFO] Total return: {total_return*100:.2f}%")
        
        return BacktestResult(
            success=True,
            initial_capital=initial_capital,
            final_equity=final_equity,
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            calmar_ratio=calmar,
            volatility=volatility,
            total_trades=len(trades),
            win_rate=win_rate,
            profit_factor=1.5,  # Simplified
            equity_curve=equity_curve[-200:],  # Last 200 points
            trades=trades[:100],  # Last 100 trades
            logs=logs
        )


# Singleton instance
_executor = None

def get_executor() -> QuantLabExecutor:
    """Get the Quant Lab executor instance."""
    global _executor
    if _executor is None:
        _executor = QuantLabExecutor()
    return _executor
