from __future__ import annotations

"""TTM Squeeze Strategy.

Trades based on Bollinger Band and Keltner Channel squeeze.
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


@StrategyRegistry.register("squeeze")
class SqueezeStrategy(Strategy):
    """TTM Squeeze Strategy.
    
    Identifies when Bollinger Bands are inside Keltner Channels (squeeze)
    and trades the direction of momentum when squeeze releases.
    
    Parameters:
        bb_period: Bollinger Band period (default: 20)
        bb_std: Bollinger Band standard deviations (default: 2.0)
        kc_period: Keltner Channel period (default: 20)
        kc_atr_mult: Keltner Channel ATR multiplier (default: 1.5)
        momentum_period: Momentum calculation period (default: 12)
    """
    
    def __init__(
        self,
        name: str = "squeeze",
        params: dict | None = None,
        symbols: list[str] | None = None,
    ) -> None:
        default_params = {
            "bb_period": 20,
            "bb_std": 2.0,
            "kc_period": 20,
            "kc_atr_mult": 1.5,
            "momentum_period": 12,
        }
        
        merged_params = {**default_params, **(params or {})}
        super().__init__(name=name, params=merged_params, symbols=symbols)
        
        self._prev_squeeze: dict[str, bool] = {}
        self._prev_momentum: dict[str, float] = {}
    
    def initialize(self, context: StrategyContext) -> None:
        self.mark_initialized()
    
    def generate_signals(
        self,
        context: StrategyContext,
        event: MarketDataEvent,
    ) -> list[SignalEvent]:
        """Generate squeeze signals."""
        symbol = event.symbol
        
        if not self.should_trade(symbol, context):
            return []
        
        bars = context.get_bars(symbol, self.required_history)
        if len(bars) < self.required_history:
            return []
        
        # Calculate Bollinger Bands
        close = bars["close"]
        bb_period = self.params["bb_period"]
        bb_std = self.params["bb_std"]
        
        bb_mid = close.rolling(bb_period).mean()
        bb_std_val = close.rolling(bb_period).std()
        bb_upper = bb_mid + bb_std * bb_std_val
        bb_lower = bb_mid - bb_std * bb_std_val
        
        # Calculate Keltner Channels
        kc_period = self.params["kc_period"]
        kc_mult = self.params["kc_atr_mult"]
        
        tr = self._true_range(bars)
        atr = tr.ewm(span=kc_period, adjust=False).mean()
        
        kc_mid = close.ewm(span=kc_period, adjust=False).mean()
        kc_upper = kc_mid + kc_mult * atr
        kc_lower = kc_mid - kc_mult * atr
        
        # Detect squeeze (BB inside KC)
        is_squeeze = (bb_lower.iloc[-1] > kc_lower.iloc[-1]) and (bb_upper.iloc[-1] < kc_upper.iloc[-1])
        was_squeeze = self._prev_squeeze.get(symbol, False)
        
        # Calculate momentum
        momentum = self._calculate_momentum(close, self.params["momentum_period"])
        current_momentum = momentum.iloc[-1]
        prev_momentum = self._prev_momentum.get(symbol, 0)
        
        # Update state
        self._prev_squeeze[symbol] = is_squeeze
        self._prev_momentum[symbol] = current_momentum
        
        signals = []
        position = context.get_position(symbol)
        has_position = context.has_position(symbol)
        
        # Entry: Squeeze releases with momentum direction
        if was_squeeze and not is_squeeze:
            if current_momentum > 0:
                signals.append(self.create_signal(
                    symbol=symbol,
                    side=Side.BUY,
                    strength=min(1.0, abs(current_momentum) / 10),
                    reason=f"Squeeze release (bullish): momentum={current_momentum:.2f}",
                    metadata={
                        "squeeze_released": True,
                        "momentum": current_momentum,
                    },
                ))
                logger.info(
                    "Bullish squeeze release",
                    strategy=self.name,
                    symbol=symbol,
                    momentum=current_momentum,
                )
            elif current_momentum < 0:
                signals.append(self.create_signal(
                    symbol=symbol,
                    side=Side.SELL,
                    strength=min(1.0, abs(current_momentum) / 10),
                    reason=f"Squeeze release (bearish): momentum={current_momentum:.2f}",
                    metadata={
                        "squeeze_released": True,
                        "momentum": current_momentum,
                    },
                ))
                logger.info(
                    "Bearish squeeze release",
                    strategy=self.name,
                    symbol=symbol,
                    momentum=current_momentum,
                )
        
        # Exit: Momentum reversal
        elif has_position:
            qty = position.get("quantity", 0) if position else 0
            
            if qty > 0 and current_momentum < 0 and prev_momentum > 0:
                signals.append(self.create_signal(
                    symbol=symbol,
                    side=Side.SELL,
                    strength=1.0,
                    reason="Momentum reversed to bearish",
                ))
            elif qty < 0 and current_momentum > 0 and prev_momentum < 0:
                signals.append(self.create_signal(
                    symbol=symbol,
                    side=Side.BUY,
                    strength=1.0,
                    reason="Momentum reversed to bullish",
                ))
        
        return signals
    
    def _true_range(self, ohlc: pd.DataFrame) -> pd.Series:
        """Calculate True Range."""
        high = ohlc["high"]
        low = ohlc["low"]
        close = ohlc["close"]
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    def _calculate_momentum(self, close: pd.Series, period: int) -> pd.Series:
        """Calculate momentum using linear regression."""
        # Simplified: use price change over period
        return close - close.shift(period)
    
    @property
    def required_history(self) -> int:
        return max(self.params["bb_period"], self.params["kc_period"]) + self.params["momentum_period"] + 10
    
    def reset(self) -> None:
        super().reset()
        self._prev_squeeze.clear()
        self._prev_momentum.clear()
