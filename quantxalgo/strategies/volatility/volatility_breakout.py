from __future__ import annotations

"""Volatility Breakout Strategy.

Trades breakouts when volatility expands after periods of compression.
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


@StrategyRegistry.register("volatility_breakout")
class VolatilityBreakoutStrategy(Strategy):
    """Volatility Breakout Strategy.
    
    Identifies periods of low volatility (compression) and trades
    the subsequent breakout when volatility expands.
    
    Parameters:
        atr_period: ATR period for volatility (default: 14)
        atr_lookback: Lookback for ATR comparison (default: 50)
        compression_threshold: ATR percentile for compression (default: 25)
        breakout_multiplier: ATR multiple for breakout (default: 1.5)
        trailing_stop_atr: ATR multiple for trailing stop (default: 2.0)
    """
    
    def __init__(
        self,
        name: str = "volatility_breakout",
        params: dict | None = None,
        symbols: list[str] | None = None,
    ) -> None:
        default_params = {
            "atr_period": 14,
            "atr_lookback": 50,
            "compression_threshold": 25,
            "breakout_multiplier": 1.5,
            "trailing_stop_atr": 2.0,
        }
        
        merged_params = {**default_params, **(params or {})}
        super().__init__(name=name, params=merged_params, symbols=symbols)
        
        self._in_compression: dict[str, bool] = {}
        self._entry_prices: dict[str, float] = {}
        self._trailing_stops: dict[str, float] = {}
    
    def initialize(self, context: StrategyContext) -> None:
        self.mark_initialized()
    
    def generate_signals(
        self,
        context: StrategyContext,
        event: MarketDataEvent,
    ) -> list[SignalEvent]:
        """Generate volatility breakout signals."""
        symbol = event.symbol
        
        if not self.should_trade(symbol, context):
            return []
        
        bars = context.get_bars(symbol, self.required_history)
        if len(bars) < self.required_history:
            return []
        
        # Calculate ATR
        atr = self._calculate_atr(bars, self.params["atr_period"])
        current_atr = atr.iloc[-1]
        
        # Get ATR percentile over lookback
        atr_lookback = atr.tail(self.params["atr_lookback"])
        atr_percentile = (atr_lookback < current_atr).sum() / len(atr_lookback) * 100
        
        current_close = bars["close"].iloc[-1]
        current_high = bars["high"].iloc[-1]
        current_low = bars["low"].iloc[-1]
        prev_close = bars["close"].iloc[-2]
        
        signals = []
        position = context.get_position(symbol)
        has_position = context.has_position(symbol)
        
        # Detect compression (low volatility)
        is_compressed = atr_percentile < self.params["compression_threshold"]
        was_compressed = self._in_compression.get(symbol, False)
        self._in_compression[symbol] = is_compressed
        
        if not has_position:
            # Look for breakout from compression
            if was_compressed and not is_compressed:
                breakout_dist = current_atr * self.params["breakout_multiplier"]
                
                # Bullish breakout
                if current_close > prev_close + breakout_dist:
                    signals.append(self.create_signal(
                        symbol=symbol,
                        side=Side.BUY,
                        strength=min(1.0, atr_percentile / 50),
                        reason=f"Volatility breakout (up): ATR percentile {atr_percentile:.0f}%",
                        metadata={
                            "atr": current_atr,
                            "atr_percentile": atr_percentile,
                            "breakout_type": "bullish",
                        },
                    ))
                    self._entry_prices[symbol] = current_close
                    self._trailing_stops[symbol] = current_close - current_atr * self.params["trailing_stop_atr"]
                
                # Bearish breakout
                elif current_close < prev_close - breakout_dist:
                    signals.append(self.create_signal(
                        symbol=symbol,
                        side=Side.SELL,
                        strength=min(1.0, atr_percentile / 50),
                        reason=f"Volatility breakout (down): ATR percentile {atr_percentile:.0f}%",
                        metadata={
                            "atr": current_atr,
                            "atr_percentile": atr_percentile,
                            "breakout_type": "bearish",
                        },
                    ))
                    self._entry_prices[symbol] = current_close
                    self._trailing_stops[symbol] = current_close + current_atr * self.params["trailing_stop_atr"]
        
        else:
            # Manage existing position with trailing stop
            qty = position.get("quantity", 0) if position else 0
            trailing_stop = self._trailing_stops.get(symbol, 0)
            
            if qty > 0:  # Long position
                # Update trailing stop
                new_stop = current_close - current_atr * self.params["trailing_stop_atr"]
                if new_stop > trailing_stop:
                    self._trailing_stops[symbol] = new_stop
                    trailing_stop = new_stop
                
                # Check stop
                if current_low < trailing_stop:
                    signals.append(self.create_signal(
                        symbol=symbol,
                        side=Side.SELL,
                        strength=1.0,
                        reason=f"Trailing stop hit at {trailing_stop:.2f}",
                    ))
                    self._cleanup_position(symbol)
            
            elif qty < 0:  # Short position
                # Update trailing stop
                new_stop = current_close + current_atr * self.params["trailing_stop_atr"]
                if new_stop < trailing_stop:
                    self._trailing_stops[symbol] = new_stop
                    trailing_stop = new_stop
                
                # Check stop
                if current_high > trailing_stop:
                    signals.append(self.create_signal(
                        symbol=symbol,
                        side=Side.BUY,
                        strength=1.0,
                        reason=f"Trailing stop hit at {trailing_stop:.2f}",
                    ))
                    self._cleanup_position(symbol)
        
        return signals
    
    def _calculate_atr(self, ohlc: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average True Range."""
        high = ohlc["high"]
        low = ohlc["low"]
        close = ohlc["close"]
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.ewm(span=period, adjust=False).mean()
    
    def _cleanup_position(self, symbol: str) -> None:
        """Clean up position tracking data."""
        self._entry_prices.pop(symbol, None)
        self._trailing_stops.pop(symbol, None)
    
    @property
    def required_history(self) -> int:
        return self.params["atr_lookback"] + self.params["atr_period"] + 10
    
    def reset(self) -> None:
        super().reset()
        self._in_compression.clear()
        self._entry_prices.clear()
        self._trailing_stops.clear()
