from __future__ import annotations

"""RSI Mean Reversion Strategy.

Trades based on RSI oversold/overbought conditions.
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


@StrategyRegistry.register("rsi_mean_reversion")
class RSIMeanReversionStrategy(Strategy):
    """RSI Mean Reversion Strategy.
    
    Simple mean reversion strategy based on RSI:
    - Long when RSI drops below oversold level
    - Exit when RSI returns to neutral zone
    - Short when RSI rises above overbought level
    
    Parameters:
        rsi_period: RSI calculation period (default: 14)
        oversold: Oversold threshold (default: 30)
        overbought: Overbought threshold (default: 70)
        neutral_low: Lower neutral zone (default: 45)
        neutral_high: Upper neutral zone (default: 55)
        allow_short: Allow short positions (default: False)
    """
    
    def __init__(
        self,
        name: str = "rsi_mean_reversion",
        params: dict | None = None,
        symbols: list[str] | None = None,
    ) -> None:
        default_params = {
            "rsi_period": 14,
            "oversold": 30,
            "overbought": 70,
            "neutral_low": 45,
            "neutral_high": 55,
            "allow_short": False,
        }
        
        merged_params = {**default_params, **(params or {})}
        super().__init__(name=name, params=merged_params, symbols=symbols)
        
        self._prev_rsi: dict[str, float] = {}
    
    def initialize(self, context: StrategyContext) -> None:
        """Initialize with historical RSI."""
        for symbol in self.symbols or context.bars.keys():
            if symbol in context.bars:
                bars = context.bars[symbol]
                if len(bars) >= self.required_history:
                    rsi = self._calculate_rsi(bars["close"])
                    if len(rsi) > 1:
                        self._prev_rsi[symbol] = rsi.iloc[-2]
        
        self.mark_initialized()
    
    def generate_signals(
        self,
        context: StrategyContext,
        event: MarketDataEvent,
    ) -> list[SignalEvent]:
        """Generate RSI-based mean reversion signals."""
        symbol = event.symbol
        
        if not self.should_trade(symbol, context):
            return []
        
        bars = context.get_bars(symbol, self.required_history + 10)
        if len(bars) < self.required_history:
            return []
        
        # Calculate RSI
        rsi = self._calculate_rsi(bars["close"])
        current_rsi = rsi.iloc[-1]
        prev_rsi = self._prev_rsi.get(symbol, current_rsi)
        
        # Update for next bar
        self._prev_rsi[symbol] = current_rsi
        
        oversold = self.params["oversold"]
        overbought = self.params["overbought"]
        neutral_low = self.params["neutral_low"]
        neutral_high = self.params["neutral_high"]
        
        signals = []
        position = context.get_position(symbol)
        qty = position.get("quantity", 0) if position else 0
        is_long = qty > 0
        is_short = qty < 0
        
        # LONG ENTRY: RSI crosses into oversold
        if not is_long and prev_rsi >= oversold and current_rsi < oversold:
            strength = 1 - (current_rsi / oversold)  # Lower RSI = stronger signal
            signals.append(self.create_signal(
                symbol=symbol,
                side=Side.BUY,
                strength=strength,
                reason=f"RSI oversold ({current_rsi:.1f} < {oversold})",
                metadata={
                    "rsi": current_rsi,
                    "threshold": oversold,
                    "signal_type": "long_entry",
                },
            ))
            
            logger.info(
                "RSI oversold signal",
                strategy=self.name,
                symbol=symbol,
                rsi=current_rsi,
            )
        
        # LONG EXIT: RSI returns to neutral zone
        elif is_long and current_rsi >= neutral_low:
            signals.append(self.create_signal(
                symbol=symbol,
                side=Side.SELL,
                strength=1.0,
                reason=f"RSI returned to neutral ({current_rsi:.1f})",
                metadata={
                    "rsi": current_rsi,
                    "signal_type": "long_exit",
                },
            ))
        
        # SHORT ENTRY: RSI crosses into overbought
        elif (
            not is_short 
            and self.params["allow_short"]
            and prev_rsi <= overbought 
            and current_rsi > overbought
        ):
            strength = (current_rsi - overbought) / (100 - overbought)
            signals.append(self.create_signal(
                symbol=symbol,
                side=Side.SELL,
                strength=strength,
                reason=f"RSI overbought ({current_rsi:.1f} > {overbought})",
                metadata={
                    "rsi": current_rsi,
                    "threshold": overbought,
                    "signal_type": "short_entry",
                },
            ))
        
        # SHORT EXIT: RSI returns to neutral zone
        elif is_short and current_rsi <= neutral_high:
            signals.append(self.create_signal(
                symbol=symbol,
                side=Side.BUY,
                strength=1.0,
                reason=f"RSI returned to neutral ({current_rsi:.1f})",
                metadata={
                    "rsi": current_rsi,
                    "signal_type": "short_exit",
                },
            ))
        
        return signals
    
    def _calculate_rsi(self, prices: pd.Series) -> pd.Series:
        """Calculate RSI."""
        period = self.params["rsi_period"]
        delta = prices.diff()
        
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        
        avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
        avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    @property
    def required_history(self) -> int:
        return self.params["rsi_period"] + 10
    
    def reset(self) -> None:
        super().reset()
        self._prev_rsi.clear()
