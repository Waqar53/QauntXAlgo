from __future__ import annotations

"""Breakout Strategy.

Trend-following strategy that enters when price breaks above/below
a channel defined by recent highs/lows.
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


@StrategyRegistry.register("breakout")
class BreakoutStrategy(Strategy):
    """Donchian Channel Breakout Strategy.
    
    Classic trend-following strategy used by the Turtle Traders:
    - Long when price breaks above the highest high of last N bars
    - Short when price breaks below the lowest low of last N bars
    - Exit when price crosses the middle of the channel
    
    Parameters:
        entry_period: Period for entry channel (default: 20)
        exit_period: Period for exit channel (default: 10)
        atr_period: Period for ATR calculation (default: 20)
        atr_multiplier: ATR multiplier for position sizing (default: 2.0)
        allow_short: Allow short positions (default: True)
    """
    
    def __init__(
        self,
        name: str = "breakout",
        params: dict | None = None,
        symbols: list[str] | None = None,
    ) -> None:
        default_params = {
            "entry_period": 20,
            "exit_period": 10,
            "atr_period": 20,
            "atr_multiplier": 2.0,
            "allow_short": True,
        }
        
        merged_params = {**default_params, **(params or {})}
        super().__init__(name=name, params=merged_params, symbols=symbols)
        
        # Track entry prices for position management
        self._entry_prices: dict[str, float] = {}
    
    def initialize(self, context: StrategyContext) -> None:
        """Initialize strategy."""
        self.mark_initialized()
    
    def generate_signals(
        self,
        context: StrategyContext,
        event: MarketDataEvent,
    ) -> list[SignalEvent]:
        """Generate breakout signals."""
        symbol = event.symbol
        
        if not self.should_trade(symbol, context):
            return []
        
        bars = context.get_bars(symbol, self.required_history + 10)
        if len(bars) < self.required_history:
            return []
        
        # Calculate channels
        entry_period = self.params["entry_period"]
        exit_period = self.params["exit_period"]
        
        # Entry channel (20-day high/low by default)
        entry_high = bars["high"].rolling(window=entry_period).max()
        entry_low = bars["low"].rolling(window=entry_period).min()
        
        # Exit channel (10-day high/low by default)
        exit_high = bars["high"].rolling(window=exit_period).max()
        exit_low = bars["low"].rolling(window=exit_period).min()
        
        current_close = bars["close"].iloc[-1]
        prev_close = bars["close"].iloc[-2]
        
        # Previous bar's channel values (to detect breakout)
        prev_entry_high = entry_high.iloc[-2]
        prev_entry_low = entry_low.iloc[-2]
        prev_exit_high = exit_high.iloc[-2]
        prev_exit_low = exit_low.iloc[-2]
        
        signals = []
        position = context.get_position(symbol)
        has_position = context.has_position(symbol)
        
        qty = position.get("quantity", 0) if position else 0
        is_long = qty > 0
        is_short = qty < 0
        
        # LONG ENTRY: Price breaks above entry channel high
        if not has_position and current_close > prev_entry_high:
            signals.append(self.create_signal(
                symbol=symbol,
                side=Side.BUY,
                strength=1.0,
                reason=f"Breakout above {entry_period}-day high ({prev_entry_high:.2f})",
                metadata={
                    "breakout_level": prev_entry_high,
                    "breakout_type": "long_entry",
                },
            ))
            self._entry_prices[symbol] = current_close
            
            logger.info(
                "Long breakout signal",
                strategy=self.name,
                symbol=symbol,
                price=current_close,
                level=prev_entry_high,
            )
        
        # SHORT ENTRY: Price breaks below entry channel low
        elif not has_position and self.params["allow_short"] and current_close < prev_entry_low:
            signals.append(self.create_signal(
                symbol=symbol,
                side=Side.SELL,
                strength=1.0,
                reason=f"Breakout below {entry_period}-day low ({prev_entry_low:.2f})",
                metadata={
                    "breakout_level": prev_entry_low,
                    "breakout_type": "short_entry",
                },
            ))
            self._entry_prices[symbol] = current_close
            
            logger.info(
                "Short breakout signal",
                strategy=self.name,
                symbol=symbol,
                price=current_close,
                level=prev_entry_low,
            )
        
        # LONG EXIT: Price breaks below exit channel low
        elif is_long and current_close < prev_exit_low:
            signals.append(self.create_signal(
                symbol=symbol,
                side=Side.SELL,
                strength=1.0,
                reason=f"Exit long: below {exit_period}-day low ({prev_exit_low:.2f})",
                metadata={
                    "exit_level": prev_exit_low,
                    "breakout_type": "long_exit",
                    "entry_price": self._entry_prices.get(symbol),
                },
            ))
            self._entry_prices.pop(symbol, None)
            
            logger.info(
                "Long exit signal",
                strategy=self.name,
                symbol=symbol,
                price=current_close,
                level=prev_exit_low,
            )
        
        # SHORT EXIT: Price breaks above exit channel high  
        elif is_short and current_close > prev_exit_high:
            signals.append(self.create_signal(
                symbol=symbol,
                side=Side.BUY,
                strength=1.0,
                reason=f"Exit short: above {exit_period}-day high ({prev_exit_high:.2f})",
                metadata={
                    "exit_level": prev_exit_high,
                    "breakout_type": "short_exit",
                    "entry_price": self._entry_prices.get(symbol),
                },
            ))
            self._entry_prices.pop(symbol, None)
            
            logger.info(
                "Short exit signal",
                strategy=self.name,
                symbol=symbol,
                price=current_close,
                level=prev_exit_high,
            )
        
        return signals
    
    @property
    def required_history(self) -> int:
        """Require entry period plus buffer."""
        return self.params["entry_period"] + 5
    
    def reset(self) -> None:
        """Reset strategy state."""
        super().reset()
        self._entry_prices.clear()
