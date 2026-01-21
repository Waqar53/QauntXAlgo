"""
QuantXalgo Universal Python Execution Engine.

This is the core of the Quant Lab IDE - it can execute ANY Python trading code
against market data, with full support for:
- Custom imports (numpy, pandas, scipy, sklearn, etc.)
- Any trading logic
- Real-time logging and output capture
- Full portfolio simulation with order execution
"""

from __future__ import annotations

import traceback
import io
import sys
import ast
import time
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from contextlib import redirect_stdout, redirect_stderr
from abc import ABC, abstractmethod

import pandas as pd
import numpy as np

from quantxalgo.config.logging_config import get_logger

logger = get_logger(__name__)


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class Position:
    """Represents a trading position."""
    symbol: str
    quantity: float = 0
    avg_cost: float = 0
    current_price: float = 0
    unrealized_pnl: float = 0
    realized_pnl: float = 0
    
    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price


@dataclass  
class Order:
    """Represents a trading order."""
    symbol: str
    side: str  # 'BUY' or 'SELL'
    quantity: float
    order_type: str = "MARKET"
    limit_price: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    filled: bool = False
    fill_price: float = 0.0
    
    
@dataclass
class ExecutionContext:
    """Full execution context available to strategies."""
    timestamp: datetime
    cash: float
    equity: float
    positions: Dict[str, Position]
    
    def get_position(self, symbol: str) -> Optional[Position]:
        return self.positions.get(symbol)
    
    def has_position(self, symbol: str) -> bool:
        pos = self.positions.get(symbol)
        return pos is not None and pos.quantity != 0
    
    def get_position_value(self, symbol: str) -> float:
        pos = self.positions.get(symbol)
        return pos.market_value if pos else 0


@dataclass
class BacktestResult:
    """Complete backtest results."""
    success: bool
    error: Optional[str] = None
    initial_capital: float = 0
    final_equity: float = 0
    total_return: float = 0
    annualized_return: float = 0
    sharpe_ratio: float = 0
    sortino_ratio: float = 0
    max_drawdown: float = 0
    calmar_ratio: float = 0
    volatility: float = 0
    total_trades: int = 0
    win_rate: float = 0
    profit_factor: float = 0
    avg_win: float = 0
    avg_loss: float = 0
    best_trade: float = 0
    worst_trade: float = 0
    equity_curve: List[Dict] = field(default_factory=list)
    trades: List[Dict] = field(default_factory=list)
    logs: List[str] = field(default_factory=list)
    execution_time_ms: float = 0
    symbols_traded: List[str] = field(default_factory=list)
    daily_returns: List[float] = field(default_factory=list)


# =============================================================================
# STRATEGY BASE CLASS - For structured strategies
# =============================================================================

class QuantLabStrategy(ABC):
    """
    Base class for structured trading strategies.
    
    Users can extend this class for a guided approach to strategy development,
    or write completely custom code that doesn't use this class at all.
    """
    
    def __init__(self, symbols: List[str], params: Dict[str, Any] = None):
        self.symbols = symbols
        self.params = params or {}
        self._logs: List[str] = []
        self._context: Optional[ExecutionContext] = None
        
    def log(self, message: str):
        """Log a message during execution."""
        ts = datetime.utcnow().strftime("%H:%M:%S.%f")[:-3]
        entry = f"[{ts}] {message}"
        self._logs.append(entry)
        print(entry)
        
    @abstractmethod
    def initialize(self):
        """Called once before backtesting starts."""
        pass
    
    @abstractmethod
    def on_bar(self, data: Dict[str, pd.DataFrame], context: ExecutionContext) -> List[Order]:
        """Called on each bar of data. Return list of orders."""
        pass
    
    def on_order_filled(self, order: Order):
        """Called when an order is filled."""
        pass
    
    # Order creation helpers
    def market_order(self, symbol: str, quantity: float) -> Order:
        side = "BUY" if quantity > 0 else "SELL"
        return Order(symbol=symbol, side=side, quantity=abs(quantity), order_type="MARKET")
    
    def limit_order(self, symbol: str, quantity: float, price: float) -> Order:
        side = "BUY" if quantity > 0 else "SELL"
        return Order(symbol=symbol, side=side, quantity=abs(quantity), order_type="LIMIT", limit_price=price)
    
    def close_position(self, symbol: str) -> Optional[Order]:
        if self._context is None:
            return None
        pos = self._context.positions.get(symbol)
        if pos and pos.quantity != 0:
            side = "SELL" if pos.quantity > 0 else "BUY"
            return Order(symbol=symbol, side=side, quantity=abs(pos.quantity), order_type="MARKET")
        return None
    
    def order_target_percent(self, symbol: str, pct: float) -> Optional[Order]:
        if self._context is None:
            return None
        target_value = self._context.equity * pct
        current_value = self._context.get_position_value(symbol)
        diff = target_value - current_value
        if abs(diff) < 100:
            return None
        return self.market_order(symbol, diff / 100)


# =============================================================================
# UNIVERSAL PYTHON EXECUTOR - Can run ANY Python code
# =============================================================================

class UniversalPythonExecutor:
    """
    Universal Python code executor for the Quant Lab.
    
    This executor can run ANY Python trading code, not just QuantLabStrategy
    subclasses. It provides a complete trading environment with:
    
    - Full market data access
    - Order execution simulation
    - Portfolio tracking
    - Performance analytics
    """
    
    # Imports that are always available
    STANDARD_IMPORTS = """
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass

# Trading primitives
from quantxalgo.engine.universal_executor import (
    QuantLabStrategy, Order, Position, ExecutionContext
)
"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        
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
        Execute any Python trading code.
        
        The code can:
        1. Define a QuantLabStrategy subclass (structured approach)
        2. Define any function that returns orders
        3. Be completely custom code that manipulates the namespace
        """
        start_time = time.time()
        logs = []
        
        def log(msg: str):
            ts = datetime.utcnow().strftime("%H:%M:%S.%f")[:-3]
            logs.append(f"[{ts}] {msg}")
        
        log("╔══════════════════════════════════════════════════════════╗")
        log("║       QUANTXALGO UNIVERSAL EXECUTION ENGINE              ║")
        log("╚══════════════════════════════════════════════════════════╝")
        log(f"Symbols: {', '.join(symbols)}")
        log(f"Period: {start_date.date()} to {end_date.date()}")
        log(f"Capital: ${initial_capital:,.0f}")
        log("")
        
        # Capture stdout/stderr
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        
        try:
            # Create execution namespace with all available imports
            namespace = self._create_namespace()
            
            # Execute user code
            log("[COMPILE] Parsing user code...")
            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                exec(code, namespace)
            log("[COMPILE] Code compiled successfully")
            
            # Find strategy class or function
            strategy = self._find_strategy(namespace, symbols, params)
            
            if strategy is None:
                return BacktestResult(
                    success=False,
                    error="No strategy found. Define a class extending QuantLabStrategy or a function named 'run_strategy'",
                    logs=logs
                )
            
            log(f"[STRATEGY] Found: {strategy.__class__.__name__}")
            
            # Initialize strategy
            if hasattr(strategy, 'initialize'):
                strategy.initialize()
                if hasattr(strategy, '_logs'):
                    logs.extend(strategy._logs)
                    strategy._logs = []
            
            # Generate market data
            log("[DATA] Generating market data...")
            data = self._generate_market_data(symbols, start_date, end_date)
            log(f"[DATA] Generated {len(next(iter(data.values())))} bars per symbol")
            
            # Run backtest
            log("[BACKTEST] Starting simulation...")
            result = await self._run_backtest(
                strategy, data, initial_capital, start_date, end_date, logs
            )
            
            # Capture any printed output
            stdout = stdout_buffer.getvalue()
            if stdout:
                for line in stdout.strip().split('\n'):
                    if line:
                        logs.append(f"[OUTPUT] {line}")
            
            # Add strategy logs
            if hasattr(strategy, '_logs'):
                logs.extend(strategy._logs)
            
            result.logs = logs
            result.execution_time_ms = (time.time() - start_time) * 1000
            result.symbols_traded = symbols
            
            log("")
            log("╔══════════════════════════════════════════════════════════╗")
            log("║                    EXECUTION COMPLETE                    ║")
            log("╚══════════════════════════════════════════════════════════╝")
            log(f"Final Equity: ${result.final_equity:,.2f}")
            log(f"Total Return: {result.total_return*100:.2f}%")
            log(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
            log(f"Total Trades: {result.total_trades}")
            
            return result
            
        except SyntaxError as e:
            return BacktestResult(
                success=False,
                error=f"Syntax Error at line {e.lineno}: {e.msg}",
                logs=logs + [f"[ERROR] {traceback.format_exc()}"]
            )
        except Exception as e:
            return BacktestResult(
                success=False,
                error=f"Execution Error: {str(e)}",
                logs=logs + [f"[ERROR] {traceback.format_exc()}"]
            )
    
    def _create_namespace(self) -> Dict:
        """Create the execution namespace with all imports."""
        return {
            # Core
            'np': np,
            'pd': pd,
            'numpy': np,
            'pandas': pd,
            'datetime': datetime,
            'timedelta': timedelta,
            'math': __import__('math'),
            'statistics': __import__('statistics'),
            
            # Trading primitives
            'QuantLabStrategy': QuantLabStrategy,
            'Order': Order,
            'Position': Position,
            'ExecutionContext': ExecutionContext,
            
            # Typing
            'Dict': Dict,
            'List': List,
            'Optional': Optional,
            'Any': Any,
            'Tuple': __import__('typing').Tuple,
            
            # Collections
            'defaultdict': __import__('collections').defaultdict,
            'dataclass': __import__('dataclasses').dataclass,
            
            # Print function
            'print': print,
        }
    
    def _find_strategy(self, namespace: Dict, symbols: List[str], params: Dict) -> Optional[QuantLabStrategy]:
        """Find and instantiate the strategy from the namespace."""
        # Look for QuantLabStrategy subclass
        for name, obj in namespace.items():
            if isinstance(obj, type) and issubclass(obj, QuantLabStrategy) and obj is not QuantLabStrategy:
                return obj(symbols=symbols, params=params or {})
        
        # Look for a run_strategy function
        if 'run_strategy' in namespace and callable(namespace['run_strategy']):
            # Wrap function in a strategy class
            class FunctionStrategy(QuantLabStrategy):
                def __init__(self, symbols, params):
                    super().__init__(symbols, params)
                    self.func = namespace['run_strategy']
                    
                def initialize(self):
                    pass
                    
                def on_bar(self, data, context):
                    result = self.func(data, context)
                    if result is None:
                        return []
                    if isinstance(result, Order):
                        return [result]
                    return result
            
            return FunctionStrategy(symbols=symbols, params=params or {})
        
        return None
    
    def _generate_market_data(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, pd.DataFrame]:
        """Generate realistic market data for backtesting."""
        dates = pd.date_range(start_date, end_date, freq='D')
        data = {}
        
        # Symbol-specific characteristics for more realistic simulation
        symbol_params = {
            'AAPL': {'drift': 0.0004, 'vol': 0.018, 'base': 150},
            'GOOGL': {'drift': 0.0003, 'vol': 0.020, 'base': 140},
            'AMZN': {'drift': 0.0005, 'vol': 0.022, 'base': 170},
            'MSFT': {'drift': 0.0004, 'vol': 0.017, 'base': 350},
            'TSLA': {'drift': 0.0006, 'vol': 0.035, 'base': 250},
            'NVDA': {'drift': 0.0008, 'vol': 0.030, 'base': 450},
            'META': {'drift': 0.0005, 'vol': 0.025, 'base': 320},
            'NFLX': {'drift': 0.0003, 'vol': 0.028, 'base': 400},
            'AMD': {'drift': 0.0006, 'vol': 0.032, 'base': 120},
            'CRM': {'drift': 0.0003, 'vol': 0.022, 'base': 250},
            'JPM': {'drift': 0.0002, 'vol': 0.015, 'base': 150},
            'GS': {'drift': 0.0002, 'vol': 0.018, 'base': 350},
            'SPY': {'drift': 0.0003, 'vol': 0.012, 'base': 450},
            'QQQ': {'drift': 0.0004, 'vol': 0.015, 'base': 380},
        }
        
        for symbol in symbols:
            np.random.seed(hash(symbol) % (2**32))
            
            params = symbol_params.get(symbol, {'drift': 0.0003, 'vol': 0.020, 'base': 100})
            
            # Generate returns with regime switching
            n = len(dates)
            regime = np.random.choice([0, 1], size=n, p=[0.7, 0.3])
            
            # Normal regime: slight upward drift
            # Volatile regime: higher volatility
            returns = np.where(
                regime == 0,
                np.random.normal(params['drift'], params['vol'], n),
                np.random.normal(-0.001, params['vol'] * 1.5, n)
            )
            
            prices = params['base'] * np.exp(np.cumsum(returns))
            
            # Generate OHLCV
            high_mult = 1 + np.abs(np.random.normal(0, 0.008, n))
            low_mult = 1 - np.abs(np.random.normal(0, 0.008, n))
            
            data[symbol] = pd.DataFrame({
                'open': prices * (1 + np.random.normal(0, 0.002, n)),
                'high': prices * high_mult,
                'low': prices * low_mult,
                'close': prices,
                'volume': np.random.randint(5_000_000, 50_000_000, n),
                'vwap': prices * (1 + np.random.normal(0, 0.001, n))
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
        """Run the backtest simulation."""
        
        # Get aligned dates
        all_dates = set()
        for df in data.values():
            all_dates.update(df.index)
        dates = sorted(all_dates)
        
        # Initialize state
        cash = initial_capital
        positions: Dict[str, Position] = {}
        equity_curve = []
        trades = []
        trade_pnls = []
        
        warmup = min(60, len(dates) // 4)
        
        logs.append(f"[BACKTEST] Processing {len(dates) - warmup} trading days...")
        
        for i, date in enumerate(dates):
            if i < warmup:
                continue
            
            # Build data slice
            current_data = {}
            for symbol, df in data.items():
                current_data[symbol] = df[df.index <= date].copy()
            
            # Update position prices
            for symbol, pos in positions.items():
                if symbol in current_data and len(current_data[symbol]) > 0:
                    new_price = current_data[symbol]['close'].iloc[-1]
                    if pos.quantity != 0:
                        pos.unrealized_pnl = (new_price - pos.avg_cost) * pos.quantity
                    pos.current_price = new_price
            
            # Calculate equity
            position_value = sum(pos.market_value for pos in positions.values())
            equity = cash + position_value
            
            # Create context
            context = ExecutionContext(
                timestamp=date,
                cash=cash,
                equity=equity,
                positions={k: v for k, v in positions.items()}
            )
            strategy._context = context
            
            # Get orders
            try:
                orders = strategy.on_bar(current_data, context)
                if orders is None:
                    orders = []
                
                # Capture strategy logs
                if hasattr(strategy, '_logs') and strategy._logs:
                    logs.extend(strategy._logs)
                    strategy._logs = []
                    
            except Exception as e:
                logs.append(f"[ERROR] Strategy error on {date.date()}: {e}")
                continue
            
            # Execute orders
            for order in orders:
                if order is None:
                    continue
                
                if order.symbol not in current_data:
                    continue
                    
                price = current_data[order.symbol]['close'].iloc[-1]
                order_value = order.quantity * price
                
                # Validate order
                if order.side == "BUY" and order_value > cash * 0.95:
                    logs.append(f"[WARN] Insufficient cash for {order.symbol}")
                    continue
                
                # Initialize position if needed
                if order.symbol not in positions:
                    positions[order.symbol] = Position(symbol=order.symbol)
                
                pos = positions[order.symbol]
                
                # Execute
                if order.side == "BUY":
                    # Calculate new average cost
                    total_cost = pos.avg_cost * pos.quantity + price * order.quantity
                    pos.quantity += order.quantity
                    pos.avg_cost = total_cost / pos.quantity if pos.quantity > 0 else 0
                    cash -= order_value
                else:  # SELL
                    # Realize P&L
                    pnl = (price - pos.avg_cost) * min(order.quantity, pos.quantity)
                    pos.realized_pnl += pnl
                    pos.quantity -= order.quantity
                    cash += order_value
                    trade_pnls.append(pnl)
                
                pos.current_price = price
                
                # Record trade
                trades.append({
                    'timestamp': date.isoformat(),
                    'symbol': order.symbol,
                    'side': order.side,
                    'quantity': order.quantity,
                    'price': price,
                    'value': order_value,
                    'pnl': pos.realized_pnl
                })
                
                order.filled = True
                order.fill_price = price
                
                if hasattr(strategy, 'on_order_filled'):
                    strategy.on_order_filled(order)
                
                side_emoji = "🟢" if order.side == "BUY" else "🔴"
                logs.append(f"[TRADE] {side_emoji} {order.side} {order.quantity:.0f} {order.symbol} @ ${price:.2f}")
            
            # Record equity curve
            equity_curve.append({
                'timestamp': date.isoformat(),
                'date': date.strftime('%Y-%m-%d'),
                'equity': equity,
                'cash': cash,
                'position_value': position_value
            })
        
        # Calculate metrics
        return self._calculate_metrics(
            initial_capital, equity_curve, trades, trade_pnls, logs
        )
    
    def _calculate_metrics(
        self,
        initial_capital: float,
        equity_curve: List[Dict],
        trades: List[Dict],
        trade_pnls: List[float],
        logs: List[str]
    ) -> BacktestResult:
        """Calculate comprehensive performance metrics."""
        
        if len(equity_curve) < 2:
            return BacktestResult(
                success=False,
                error="Insufficient data for metrics calculation",
                logs=logs
            )
        
        equities = [e['equity'] for e in equity_curve]
        returns = np.diff(equities) / equities[:-1]
        
        final_equity = equities[-1]
        total_return = (final_equity - initial_capital) / initial_capital
        
        # Annualized metrics
        days = len(equities)
        years = days / 252
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        volatility = np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0
        sharpe = annualized_return / volatility if volatility > 0 else 0
        
        # Sortino (downside only)
        downside = returns[returns < 0]
        downside_std = np.std(downside) * np.sqrt(252) if len(downside) > 0 else 0.001
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
        
        calmar = abs(annualized_return / max_dd) if max_dd != 0 else 0
        
        # Trade metrics
        winning_trades = [p for p in trade_pnls if p > 0]
        losing_trades = [p for p in trade_pnls if p < 0]
        
        win_rate = len(winning_trades) / len(trade_pnls) if trade_pnls else 0
        avg_win = np.mean(winning_trades) if winning_trades else 0
        avg_loss = np.mean(losing_trades) if losing_trades else 0
        
        profit_factor = abs(sum(winning_trades) / sum(losing_trades)) if losing_trades and sum(losing_trades) != 0 else float('inf')
        
        best_trade = max(trade_pnls) if trade_pnls else 0
        worst_trade = min(trade_pnls) if trade_pnls else 0
        
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
            profit_factor=profit_factor if profit_factor != float('inf') else 99.99,
            avg_win=avg_win,
            avg_loss=avg_loss,
            best_trade=best_trade,
            worst_trade=worst_trade,
            equity_curve=equity_curve[-250:],  # Last year
            trades=trades[-100:],  # Last 100 trades
            logs=logs,
            daily_returns=returns.tolist()[-100:]
        )


# Singleton
_executor = None

def get_executor() -> UniversalPythonExecutor:
    global _executor
    if _executor is None:
        _executor = UniversalPythonExecutor()
    return _executor
