"""
Market Regime Detection.

Detects market regimes (bull, bear, sideways, crisis) and adjusts
strategy behavior accordingly.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

from quantxalgo.config.logging_config import get_logger
from quantxalgo.core.enums import RegimeType

logger = get_logger(__name__)


@dataclass
class RegimeState:
    """Current regime state."""
    
    regime: RegimeType
    confidence: float  # 0-1
    volatility: float
    trend: float  # -1 (bearish) to +1 (bullish)
    detected_at: datetime
    duration_days: int = 0


class RegimeDetector:
    """Detect market regime from price data.
    
    Uses multiple indicators to classify market regime:
    - Trend: Moving average cross and momentum
    - Volatility: Realized vs historical volatility
    - Correlation: Cross-asset correlation (crisis indicator)
    
    Example:
        >>> detector = RegimeDetector()
        >>> regime = detector.detect(prices_df)
        >>> if regime.regime == RegimeType.CRISIS:
        ...     reduce_exposure()
    """
    
    def __init__(
        self,
        short_window: int = 20,
        long_window: int = 200,
        vol_window: int = 20,
        trend_threshold: float = 0.02,
        vol_threshold: float = 0.25,
    ) -> None:
        """Initialize detector.
        
        Args:
            short_window: Short-term trend window.
            long_window: Long-term trend window.
            vol_window: Volatility calculation window.
            trend_threshold: Threshold for trend classification.
            vol_threshold: High volatility threshold (annualized).
        """
        self.short_window = short_window
        self.long_window = long_window
        self.vol_window = vol_window
        self.trend_threshold = trend_threshold
        self.vol_threshold = vol_threshold
        
        self._current_regime: Optional[RegimeState] = None
        self._regime_history: list[RegimeState] = []
    
    def detect(self, prices: pd.Series) -> RegimeState:
        """Detect current market regime.
        
        Args:
            prices: Price series (typically close prices).
            
        Returns:
            Current regime state.
        """
        if len(prices) < self.long_window:
            return RegimeState(
                regime=RegimeType.UNKNOWN,
                confidence=0.0,
                volatility=0.0,
                trend=0.0,
                detected_at=datetime.utcnow(),
            )
        
        # Calculate trend indicators
        short_ma = prices.rolling(self.short_window).mean().iloc[-1]
        long_ma = prices.rolling(self.long_window).mean().iloc[-1]
        current_price = prices.iloc[-1]
        
        # Trend score: -1 to +1
        if long_ma > 0:
            price_vs_long = (current_price - long_ma) / long_ma
            ma_cross = (short_ma - long_ma) / long_ma
            trend_score = np.clip((price_vs_long + ma_cross) / 2, -1, 1)
        else:
            trend_score = 0.0
        
        # Calculate volatility
        returns = prices.pct_change().dropna()
        current_vol = returns.tail(self.vol_window).std() * np.sqrt(252)
        historical_vol = returns.std() * np.sqrt(252)
        
        vol_ratio = current_vol / historical_vol if historical_vol > 0 else 1.0
        
        # Determine regime
        regime, confidence = self._classify_regime(trend_score, current_vol, vol_ratio)
        
        # Check for regime change
        if self._current_regime and self._current_regime.regime != regime:
            logger.info(
                "Regime change detected",
                old_regime=self._current_regime.regime.value,
                new_regime=regime.value,
                confidence=f"{confidence:.0%}",
            )
            self._regime_history.append(self._current_regime)
        
        duration = 0
        if self._current_regime and self._current_regime.regime == regime:
            duration = self._current_regime.duration_days + 1
        
        self._current_regime = RegimeState(
            regime=regime,
            confidence=confidence,
            volatility=current_vol,
            trend=trend_score,
            detected_at=datetime.utcnow(),
            duration_days=duration,
        )
        
        return self._current_regime
    
    def _classify_regime(
        self,
        trend: float,
        volatility: float,
        vol_ratio: float,
    ) -> tuple[RegimeType, float]:
        """Classify regime based on indicators.
        
        Returns:
            Tuple of (regime, confidence).
        """
        # Crisis: High volatility + negative trend
        if volatility > self.vol_threshold * 1.5 and trend < -self.trend_threshold:
            confidence = min(1.0, vol_ratio * 0.5 + abs(trend))
            return RegimeType.CRISIS, confidence
        
        # High volatility regime
        if volatility > self.vol_threshold:
            confidence = min(1.0, vol_ratio * 0.7)
            return RegimeType.HIGH_VOLATILITY, confidence
        
        # Low volatility regime
        if volatility < self.vol_threshold * 0.5:
            confidence = min(1.0, (1 - vol_ratio) * 0.7 + 0.3)
            return RegimeType.LOW_VOLATILITY, confidence
        
        # Bullish: Positive trend
        if trend > self.trend_threshold:
            confidence = min(1.0, abs(trend) * 2 + 0.3)
            return RegimeType.BULL, confidence
        
        # Bearish: Negative trend
        if trend < -self.trend_threshold:
            confidence = min(1.0, abs(trend) * 2 + 0.3)
            return RegimeType.BEAR, confidence
        
        # Sideways: No clear trend
        confidence = 1 - abs(trend) * 2
        return RegimeType.SIDEWAYS, max(0.3, confidence)
    
    def get_scaling_factor(self, regime: Optional[RegimeType] = None) -> float:
        """Get position sizing scaling factor for current regime.
        
        Returns:
            Scaling factor (0.0 to 1.0).
        """
        regime = regime or (self._current_regime.regime if self._current_regime else RegimeType.UNKNOWN)
        
        scaling_factors = {
            RegimeType.BULL: 1.0,
            RegimeType.BEAR: 0.5,
            RegimeType.SIDEWAYS: 0.7,
            RegimeType.CRISIS: 0.1,
            RegimeType.HIGH_VOLATILITY: 0.4,
            RegimeType.LOW_VOLATILITY: 0.9,
            RegimeType.UNKNOWN: 0.5,
        }
        
        return scaling_factors.get(regime, 0.5)
    
    def get_strategy_recommendations(self) -> dict[str, float]:
        """Get recommended strategy weights based on regime.
        
        Returns:
            Dictionary of strategy type to weight.
        """
        if not self._current_regime:
            return {"momentum": 0.5, "mean_reversion": 0.5}
        
        regime = self._current_regime.regime
        
        recommendations = {
            RegimeType.BULL: {
                "momentum": 0.7,
                "mean_reversion": 0.15,
                "statistical": 0.15,
            },
            RegimeType.BEAR: {
                "momentum": 0.3,
                "mean_reversion": 0.3,
                "statistical": 0.4,
            },
            RegimeType.SIDEWAYS: {
                "momentum": 0.2,
                "mean_reversion": 0.5,
                "statistical": 0.3,
            },
            RegimeType.CRISIS: {
                "momentum": 0.0,
                "mean_reversion": 0.3,
                "statistical": 0.7,
            },
            RegimeType.HIGH_VOLATILITY: {
                "momentum": 0.4,
                "mean_reversion": 0.2,
                "statistical": 0.4,
            },
            RegimeType.LOW_VOLATILITY: {
                "momentum": 0.4,
                "mean_reversion": 0.4,
                "statistical": 0.2,
            },
        }
        
        return recommendations.get(regime, {"momentum": 0.5, "mean_reversion": 0.5})
    
    @property
    def current_regime(self) -> Optional[RegimeState]:
        """Get current regime state."""
        return self._current_regime
    
    @property
    def regime_history(self) -> list[RegimeState]:
        """Get regime history."""
        return self._regime_history.copy()
