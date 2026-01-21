from __future__ import annotations

"""Bollinger Band Mean Reversion Strategy.

Mean reversion strategy that buys when price touches lower band
and sells when price reaches the middle or upper band.
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


@StrategyRegistry.register("bollinger_mean_reversion")
class BollingerBandStrategy(Strategy):
    """Bollinger Band Mean Reversion Strategy.
    
    Classic mean reversion strategy:
    - Long when price closes below lower band (oversold)
    - Exit long at middle band (mean reversion complete)
    - Short when price closes above upper band (overbought)
    - Exit short at middle band
    
    Works best in ranging/sideways markets. May suffer in strong trends.
    
    Parameters:
        period: Bollinger Band period (default: 20)
        std_dev: Standard deviation multiplier (default: 2.0)
        entry_threshold: How far beyond band to enter (default: 0.0)
        allow_short: Allow short positions (default: False)
        use_rsi_filter: Use RSI to confirm oversold/overbought (default: True)
        rsi_period: RSI period for filter (default: 14)
        rsi_oversold: RSI oversold level (default: 30)
        rsi_overbought: RSI overbought level (default: 70)
    """
    
    def __init__(
        self,
        name: str = "bollinger_mean_reversion",
        params: dict | None = None,
        symbols: list[str] | None = None,
    ) -> None:
        default_params = {
            "period": 20,
            "std_dev": 2.0,
            "entry_threshold": 0.0,
            "allow_short": False,
            "use_rsi_filter": True,
            "rsi_period": 14,
            "rsi_oversold": 30,
            "rsi_overbought": 70,
        }
        
        merged_params = {**default_params, **(params or {})}
        super().__init__(name=name, params=merged_params, symbols=symbols)
    
    def initialize(self, context: StrategyContext) -> None:
        """Initialize strategy."""
        self.mark_initialized()
    
    def generate_signals(
        self,
        context: StrategyContext,
        event: MarketDataEvent,
    ) -> list[SignalEvent]:
        """Generate mean reversion signals based on Bollinger Bands."""
        symbol = event.symbol
        
        if not self.should_trade(symbol, context):
            return []
        
        bars = context.get_bars(symbol, self.required_history + 10)
        if len(bars) < self.required_history:
            return []
        
        close = bars["close"]
        
        # Calculate Bollinger Bands
        period = self.params["period"]
        std_dev = self.params["std_dev"]
        
        sma = close.rolling(window=period).mean()
        std = close.rolling(window=period).std()
        
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        middle = sma
        
        current_close = close.iloc[-1]
        current_upper = upper.iloc[-1]
        current_lower = lower.iloc[-1]
        current_middle = middle.iloc[-1]
        
        # Calculate %B (position within bands)
        pct_b = (current_close - current_lower) / (current_upper - current_lower)
        
        # Optional RSI filter
        rsi_ok_long = True
        rsi_ok_short = True
        
        if self.params["use_rsi_filter"]:
            rsi = self._calculate_rsi(close, self.params["rsi_period"])
            current_rsi = rsi.iloc[-1]
            rsi_ok_long = current_rsi < self.params["rsi_oversold"]
            rsi_ok_short = current_rsi > self.params["rsi_overbought"]
        else:
            current_rsi = None
        
        signals = []
        position = context.get_position(symbol)
        qty = position.get("quantity", 0) if position else 0
        is_long = qty > 0
        is_short = qty < 0
        
        # LONG ENTRY: Price below lower band + RSI oversold
        if not is_long and current_close < current_lower and rsi_ok_long:
            signals.append(self.create_signal(
                symbol=symbol,
                side=Side.BUY,
                strength=min(1.0, (current_lower - current_close) / current_close * 100),
                reason=f"Price below lower BB ({current_lower:.2f}), %B={pct_b:.2f}",
                metadata={
                    "bb_lower": current_lower,
                    "bb_middle": current_middle,
                    "bb_upper": current_upper,
                    "pct_b": pct_b,
                    "rsi": current_rsi,
                    "signal_type": "long_entry",
                },
            ))
            
            logger.info(
                "Long entry signal (oversold)",
                strategy=self.name,
                symbol=symbol,
                price=current_close,
                lower_band=current_lower,
                pct_b=pct_b,
            )
        
        # LONG EXIT: Price reaches middle band
        elif is_long and current_close >= current_middle:
            signals.append(self.create_signal(
                symbol=symbol,
                side=Side.SELL,
                strength=1.0,
                reason=f"Mean reversion complete at middle band ({current_middle:.2f})",
                metadata={
                    "bb_middle": current_middle,
                    "pct_b": pct_b,
                    "signal_type": "long_exit",
                },
            ))
            
            logger.info(
                "Long exit signal (mean reversion)",
                strategy=self.name,
                symbol=symbol,
                price=current_close,
                middle_band=current_middle,
            )
        
        # SHORT ENTRY: Price above upper band + RSI overbought
        elif (
            not is_short 
            and self.params["allow_short"]
            and current_close > current_upper 
            and rsi_ok_short
        ):
            signals.append(self.create_signal(
                symbol=symbol,
                side=Side.SELL,
                strength=min(1.0, (current_close - current_upper) / current_close * 100),
                reason=f"Price above upper BB ({current_upper:.2f}), %B={pct_b:.2f}",
                metadata={
                    "bb_upper": current_upper,
                    "pct_b": pct_b,
                    "rsi": current_rsi,
                    "signal_type": "short_entry",
                },
            ))
        
        # SHORT EXIT: Price reaches middle band
        elif is_short and current_close <= current_middle:
            signals.append(self.create_signal(
                symbol=symbol,
                side=Side.BUY,
                strength=1.0,
                reason=f"Short mean reversion complete at middle band",
                metadata={
                    "bb_middle": current_middle,
                    "signal_type": "short_exit",
                },
            ))
        
        return signals
    
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        
        avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
        avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    @property
    def required_history(self) -> int:
        return max(self.params["period"], self.params["rsi_period"]) + 10
