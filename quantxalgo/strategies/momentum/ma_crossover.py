from __future__ import annotations

"""Moving Average Crossover Strategy.

Classic momentum strategy that goes long when fast MA crosses above slow MA,
and short when fast MA crosses below slow MA.
"""

from typing import Optional

import numpy as np
import pandas as pd

from quantxalgo.config.logging_config import get_logger
from quantxalgo.core.enums import Side
from quantxalgo.core.events import SignalEvent, MarketDataEvent
from quantxalgo.strategies.base import Strategy, StrategyContext
from quantxalgo.strategies.registry import StrategyRegistry

logger = get_logger(__name__)


@StrategyRegistry.register("ma_crossover")
class MACrossoverStrategy(Strategy):
    """Moving Average Crossover Strategy.
    
    Implements the classic dual moving average crossover:
    - Long when fast MA crosses above slow MA (golden cross)
    - Exit long / go short when fast MA crosses below slow MA (death cross)
    
    This is a trend-following strategy that works well in trending markets
    but may whipsaw in ranging markets.
    
    Parameters:
        fast_period: Period for fast moving average (default: 10)
        slow_period: Period for slow moving average (default: 50)
        use_ema: Use EMA instead of SMA (default: True)
        position_size_pct: Position size as % of equity (default: 0.02)
        allow_short: Allow short positions (default: False)
    """
    
    def __init__(
        self,
        name: str = "ma_crossover",
        params: dict | None = None,
        symbols: list[str] | None = None,
    ) -> None:
        # Default parameters
        default_params = {
            "fast_period": 10,
            "slow_period": 50,
            "use_ema": True,
            "position_size_pct": 0.02,
            "allow_short": False,
        }
        
        # Merge with provided params
        merged_params = {**default_params, **(params or {})}
        
        super().__init__(name=name, params=merged_params, symbols=symbols)
        
        # Validate parameters
        if merged_params["fast_period"] >= merged_params["slow_period"]:
            raise ValueError("fast_period must be less than slow_period")
        
        # State tracking per symbol
        self._prev_fast: dict[str, float] = {}
        self._prev_slow: dict[str, float] = {}
    
    def initialize(self, context: StrategyContext) -> None:
        """Pre-compute moving averages on historical data."""
        for symbol in self.symbols or context.bars.keys():
            if symbol in context.bars:
                bars = context.bars[symbol]
                if len(bars) >= self.required_history:
                    fast, slow = self._compute_mas(bars["close"])
                    self._prev_fast[symbol] = fast.iloc[-2] if len(fast) > 1 else np.nan
                    self._prev_slow[symbol] = slow.iloc[-2] if len(slow) > 1 else np.nan
        
        self.mark_initialized()
    
    def generate_signals(
        self,
        context: StrategyContext,
        event: MarketDataEvent,
    ) -> list[SignalEvent]:
        """Generate crossover signals.
        
        Returns:
            List containing a signal if crossover detected, empty otherwise.
        """
        symbol = event.symbol
        
        if not self.should_trade(symbol, context):
            return []
        
        bars = context.get_bars(symbol, self.required_history + 10)
        if len(bars) < self.required_history:
            return []
        
        # Compute current MAs
        fast_ma, slow_ma = self._compute_mas(bars["close"])
        
        current_fast = fast_ma.iloc[-1]
        current_slow = slow_ma.iloc[-1]
        
        prev_fast = self._prev_fast.get(symbol, np.nan)
        prev_slow = self._prev_slow.get(symbol, np.nan)
        
        # Update state for next bar
        self._prev_fast[symbol] = current_fast
        self._prev_slow[symbol] = current_slow
        
        if np.isnan(prev_fast) or np.isnan(prev_slow):
            return []
        
        signals = []
        has_position = context.has_position(symbol)
        position = context.get_position(symbol)
        is_long = position and position.get("quantity", 0) > 0
        is_short = position and position.get("quantity", 0) < 0
        
        # Golden Cross: Fast crosses above slow
        if prev_fast <= prev_slow and current_fast > current_slow:
            if is_short:
                # Close short first
                signals.append(self.create_signal(
                    symbol=symbol,
                    side=Side.BUY,
                    strength=1.0,
                    reason="Close short on golden cross",
                ))
            
            if not is_long:
                signals.append(self.create_signal(
                    symbol=symbol,
                    side=Side.BUY,
                    strength=1.0,
                    reason=f"Golden cross: Fast MA ({current_fast:.2f}) > Slow MA ({current_slow:.2f})",
                    metadata={
                        "fast_ma": current_fast,
                        "slow_ma": current_slow,
                        "crossover_type": "golden",
                    },
                ))
                
                logger.info(
                    "Golden cross detected",
                    strategy=self.name,
                    symbol=symbol,
                    fast_ma=current_fast,
                    slow_ma=current_slow,
                )
        
        # Death Cross: Fast crosses below slow
        elif prev_fast >= prev_slow and current_fast < current_slow:
            if is_long:
                # Exit long position
                signals.append(self.create_signal(
                    symbol=symbol,
                    side=Side.SELL,
                    strength=1.0,
                    reason=f"Death cross: Fast MA ({current_fast:.2f}) < Slow MA ({current_slow:.2f})",
                    metadata={
                        "fast_ma": current_fast,
                        "slow_ma": current_slow,
                        "crossover_type": "death",
                    },
                ))
                
                logger.info(
                    "Death cross detected",
                    strategy=self.name,
                    symbol=symbol,
                    fast_ma=current_fast,
                    slow_ma=current_slow,
                )
            
            if self.params["allow_short"] and not is_short:
                signals.append(self.create_signal(
                    symbol=symbol,
                    side=Side.SELL,
                    strength=1.0,
                    reason="Short on death cross",
                ))
        
        return signals
    
    def _compute_mas(self, prices: pd.Series) -> tuple[pd.Series, pd.Series]:
        """Compute fast and slow moving averages."""
        fast_period = self.params["fast_period"]
        slow_period = self.params["slow_period"]
        use_ema = self.params["use_ema"]
        
        if use_ema:
            fast = prices.ewm(span=fast_period, adjust=False).mean()
            slow = prices.ewm(span=slow_period, adjust=False).mean()
        else:
            fast = prices.rolling(window=fast_period).mean()
            slow = prices.rolling(window=slow_period).mean()
        
        return fast, slow
    
    @property
    def required_history(self) -> int:
        """Require enough history for slow MA plus some buffer."""
        return self.params["slow_period"] + 10
    
    def reset(self) -> None:
        """Reset strategy state."""
        super().reset()
        self._prev_fast.clear()
        self._prev_slow.clear()
