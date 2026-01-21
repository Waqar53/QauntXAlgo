from __future__ import annotations

"""Pairs Trading Strategy.

Statistical arbitrage strategy that trades the spread between
two cointegrated assets.
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


@StrategyRegistry.register("pairs_trading")
class PairsTradingStrategy(Strategy):
    """Pairs Trading Strategy.
    
    Finds two correlated/cointegrated assets and trades the spread:
    - Long spread when z-score < -entry_threshold
    - Short spread when z-score > entry_threshold
    - Exit when z-score returns to 0
    
    Parameters:
        pair: Tuple of two symbols (leg1, leg2)
        lookback: Period for calculating spread statistics (default: 60)
        entry_threshold: Z-score threshold for entry (default: 2.0)
        exit_threshold: Z-score threshold for exit (default: 0.5)
        hedge_ratio_method: 'ols' or 'rolling' (default: 'rolling')
    """
    
    def __init__(
        self,
        name: str = "pairs_trading",
        params: dict | None = None,
        symbols: list[str] | None = None,
    ) -> None:
        default_params = {
            "lookback": 60,
            "entry_threshold": 2.0,
            "exit_threshold": 0.5,
            "hedge_ratio_method": "rolling",
        }
        
        merged_params = {**default_params, **(params or {})}
        super().__init__(name=name, params=merged_params, symbols=symbols)
        
        if len(self.symbols) != 2:
            raise ValueError("Pairs trading requires exactly 2 symbols")
        
        self.leg1, self.leg2 = self.symbols[0], self.symbols[1]
        self._position_state = "flat"  # flat, long_spread, short_spread
        self._hedge_ratio = 1.0
    
    def initialize(self, context: StrategyContext) -> None:
        """Calculate initial hedge ratio."""
        if self.leg1 in context.bars and self.leg2 in context.bars:
            bars1 = context.bars[self.leg1]
            bars2 = context.bars[self.leg2]
            
            if len(bars1) >= self.params["lookback"]:
                self._hedge_ratio = self._calculate_hedge_ratio(
                    bars1["close"], bars2["close"]
                )
        
        self.mark_initialized()
    
    def generate_signals(
        self,
        context: StrategyContext,
        event: MarketDataEvent,
    ) -> list[SignalEvent]:
        """Generate pairs trading signals."""
        symbol = event.symbol
        
        # Only process on leg1 to avoid duplicate signals
        if symbol != self.leg1:
            return []
        
        bars1 = context.get_bars(self.leg1, self.required_history)
        bars2 = context.get_bars(self.leg2, self.required_history)
        
        if len(bars1) < self.params["lookback"] or len(bars2) < self.params["lookback"]:
            return []
        
        # Calculate spread and z-score
        lookback = self.params["lookback"]
        self._hedge_ratio = self._calculate_hedge_ratio(
            bars1["close"].tail(lookback),
            bars2["close"].tail(lookback),
        )
        
        spread = bars1["close"] - self._hedge_ratio * bars2["close"]
        spread_mean = spread.tail(lookback).mean()
        spread_std = spread.tail(lookback).std()
        
        if spread_std == 0:
            return []
        
        current_zscore = (spread.iloc[-1] - spread_mean) / spread_std
        
        entry_thresh = self.params["entry_threshold"]
        exit_thresh = self.params["exit_threshold"]
        
        signals = []
        
        # Entry signals
        if self._position_state == "flat":
            if current_zscore < -entry_thresh:
                # Long spread: Buy leg1, sell leg2
                signals.extend([
                    self.create_signal(
                        symbol=self.leg1,
                        side=Side.BUY,
                        strength=abs(current_zscore) / 3,
                        reason=f"Long spread: z={current_zscore:.2f}",
                        metadata={"hedge_ratio": self._hedge_ratio, "z_score": current_zscore},
                    ),
                    self.create_signal(
                        symbol=self.leg2,
                        side=Side.SELL,
                        strength=abs(current_zscore) / 3,
                        reason=f"Short hedge: z={current_zscore:.2f}",
                        metadata={"hedge_ratio": self._hedge_ratio},
                    ),
                ])
                self._position_state = "long_spread"
                
            elif current_zscore > entry_thresh:
                # Short spread: Sell leg1, buy leg2
                signals.extend([
                    self.create_signal(
                        symbol=self.leg1,
                        side=Side.SELL,
                        strength=abs(current_zscore) / 3,
                        reason=f"Short spread: z={current_zscore:.2f}",
                        metadata={"hedge_ratio": self._hedge_ratio, "z_score": current_zscore},
                    ),
                    self.create_signal(
                        symbol=self.leg2,
                        side=Side.BUY,
                        strength=abs(current_zscore) / 3,
                        reason=f"Long hedge: z={current_zscore:.2f}",
                        metadata={"hedge_ratio": self._hedge_ratio},
                    ),
                ])
                self._position_state = "short_spread"
        
        # Exit signals
        elif self._position_state == "long_spread" and current_zscore > -exit_thresh:
            signals.extend([
                self.create_signal(
                    symbol=self.leg1, side=Side.SELL, strength=1.0,
                    reason="Exit long spread",
                ),
                self.create_signal(
                    symbol=self.leg2, side=Side.BUY, strength=1.0,
                    reason="Exit short hedge",
                ),
            ])
            self._position_state = "flat"
            
        elif self._position_state == "short_spread" and current_zscore < exit_thresh:
            signals.extend([
                self.create_signal(
                    symbol=self.leg1, side=Side.BUY, strength=1.0,
                    reason="Exit short spread",
                ),
                self.create_signal(
                    symbol=self.leg2, side=Side.SELL, strength=1.0,
                    reason="Exit long hedge",
                ),
            ])
            self._position_state = "flat"
        
        return signals
    
    def _calculate_hedge_ratio(self, y: pd.Series, x: pd.Series) -> float:
        """Calculate hedge ratio using OLS regression."""
        if len(y) != len(x) or len(y) < 10:
            return 1.0
        
        # Simple OLS: beta = cov(y,x) / var(x)
        covariance = np.cov(y, x)[0, 1]
        variance = np.var(x)
        
        if variance == 0:
            return 1.0
        
        return covariance / variance
    
    @property
    def required_history(self) -> int:
        return self.params["lookback"] + 20
    
    def reset(self) -> None:
        super().reset()
        self._position_state = "flat"
        self._hedge_ratio = 1.0
