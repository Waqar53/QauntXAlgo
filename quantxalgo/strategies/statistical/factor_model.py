from __future__ import annotations

"""Factor Model Strategy.

Multi-factor strategy based on classic quant factors:
- Momentum
- Value
- Quality
- Volatility
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


@StrategyRegistry.register("factor_model")
class FactorModelStrategy(Strategy):
    """Multi-Factor Model Strategy.
    
    Ranks stocks based on multiple factors and goes long top quintile,
    short bottom quintile.
    
    Factors:
    - Momentum: 12-month return minus last month
    - Value: Book-to-market proxy (inverse of recent returns)
    - Quality: Profitability proxy (stability of returns)
    - Low Volatility: Inverse of realized volatility
    
    Parameters:
        momentum_period: Momentum lookback (default: 252)
        skip_period: Skip most recent period for momentum (default: 21)
        volatility_period: Volatility calculation period (default: 60)
        rebalance_frequency: Days between rebalancing (default: 21)
        long_only: Whether to only go long (default: True)
        top_pct: Top percentile to go long (default: 20)
        factor_weights: Dict of factor weights (default: equal)
    """
    
    def __init__(
        self,
        name: str = "factor_model",
        params: dict | None = None,
        symbols: list[str] | None = None,
    ) -> None:
        default_params = {
            "momentum_period": 252,
            "skip_period": 21,
            "volatility_period": 60,
            "rebalance_frequency": 21,
            "long_only": True,
            "top_pct": 20,
            "factor_weights": {
                "momentum": 0.4,
                "low_vol": 0.3,
                "quality": 0.3,
            },
        }
        
        merged_params = {**default_params, **(params or {})}
        super().__init__(name=name, params=merged_params, symbols=symbols)
        
        self._last_rebalance = 0
        self._current_longs: set[str] = set()
        self._current_shorts: set[str] = set()
        self._bar_count = 0
    
    def initialize(self, context: StrategyContext) -> None:
        """Initialize factor calculations."""
        self.mark_initialized()
    
    def generate_signals(
        self,
        context: StrategyContext,
        event: MarketDataEvent,
    ) -> list[SignalEvent]:
        """Generate factor-based signals."""
        self._bar_count += 1
        
        # Only rebalance at specified frequency
        if self._bar_count - self._last_rebalance < self.params["rebalance_frequency"]:
            return []
        
        # Need multiple symbols to rank
        if len(self.symbols) < 5:
            return []
        
        # Calculate factor scores for all symbols
        factor_scores = {}
        
        for symbol in self.symbols:
            bars = context.get_bars(symbol, self.required_history)
            if len(bars) < self.required_history:
                continue
            
            scores = self._calculate_factor_scores(bars)
            if scores:
                factor_scores[symbol] = scores
        
        if len(factor_scores) < 5:
            return []
        
        # Calculate composite score
        composite_scores = {}
        weights = self.params["factor_weights"]
        
        for symbol, scores in factor_scores.items():
            composite = sum(
                scores.get(factor, 0) * weight
                for factor, weight in weights.items()
            )
            composite_scores[symbol] = composite
        
        # Rank symbols
        sorted_symbols = sorted(
            composite_scores.keys(),
            key=lambda s: composite_scores[s],
            reverse=True
        )
        
        top_pct = self.params["top_pct"] / 100
        n_long = max(1, int(len(sorted_symbols) * top_pct))
        
        new_longs = set(sorted_symbols[:n_long])
        new_shorts = set(sorted_symbols[-n_long:]) if not self.params["long_only"] else set()
        
        signals = []
        
        # Exit old positions
        for symbol in self._current_longs - new_longs:
            signals.append(self.create_signal(
                symbol=symbol, side=Side.SELL, strength=1.0,
                reason="Factor rank dropped",
            ))
        
        for symbol in self._current_shorts - new_shorts:
            signals.append(self.create_signal(
                symbol=symbol, side=Side.BUY, strength=1.0,
                reason="Factor rank improved",
            ))
        
        # Enter new positions
        for symbol in new_longs - self._current_longs:
            signals.append(self.create_signal(
                symbol=symbol, side=Side.BUY,
                strength=composite_scores[symbol],
                reason=f"Top factor rank: {composite_scores[symbol]:.2f}",
                metadata={"factor_scores": factor_scores.get(symbol, {})},
            ))
        
        for symbol in new_shorts - self._current_shorts:
            signals.append(self.create_signal(
                symbol=symbol, side=Side.SELL,
                strength=abs(composite_scores[symbol]),
                reason=f"Bottom factor rank: {composite_scores[symbol]:.2f}",
            ))
        
        # Update state
        self._current_longs = new_longs
        self._current_shorts = new_shorts
        self._last_rebalance = self._bar_count
        
        if signals:
            logger.info(
                "Factor rebalance",
                strategy=self.name,
                longs=list(new_longs),
                shorts=list(new_shorts),
                signals=len(signals),
            )
        
        return signals
    
    def _calculate_factor_scores(self, bars: pd.DataFrame) -> dict[str, float]:
        """Calculate factor scores for a single symbol."""
        close = bars["close"]
        returns = close.pct_change().dropna()
        
        if len(returns) < self.params["momentum_period"]:
            return {}
        
        scores = {}
        
        # Momentum: 12-1 month return
        mom_period = self.params["momentum_period"]
        skip = self.params["skip_period"]
        
        if len(close) >= mom_period:
            momentum_return = (close.iloc[-skip] / close.iloc[-mom_period]) - 1
            scores["momentum"] = self._normalize_score(momentum_return, -0.5, 1.0)
        
        # Low Volatility: Inverse of realized vol
        vol_period = self.params["volatility_period"]
        if len(returns) >= vol_period:
            volatility = returns.tail(vol_period).std() * np.sqrt(252)
            # Lower vol = higher score
            scores["low_vol"] = self._normalize_score(1 / (volatility + 0.01), 0, 20)
        
        # Quality: Consistency of returns (Sharpe-like)
        if len(returns) >= vol_period:
            mean_return = returns.tail(vol_period).mean()
            vol = returns.tail(vol_period).std()
            if vol > 0:
                scores["quality"] = self._normalize_score(mean_return / vol, -1, 1)
        
        return scores
    
    def _normalize_score(self, value: float, min_val: float, max_val: float) -> float:
        """Normalize score to [0, 1] range."""
        return max(0, min(1, (value - min_val) / (max_val - min_val)))
    
    @property
    def required_history(self) -> int:
        return self.params["momentum_period"] + 10
    
    def reset(self) -> None:
        super().reset()
        self._last_rebalance = 0
        self._current_longs.clear()
        self._current_shorts.clear()
        self._bar_count = 0
