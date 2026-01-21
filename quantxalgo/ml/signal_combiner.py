"""
Signal Combiner for multi-model ensemble.

Combines signals from multiple sources into a unified trading signal.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from quantxalgo.config.logging_config import get_logger
from quantxalgo.core.enums import Side

logger = get_logger(__name__)


@dataclass
class CombinedSignal:
    """Combined signal from multiple sources."""
    
    symbol: str
    direction: Side
    strength: float  # -1 to 1
    confidence: float  # 0 to 1
    sources: dict[str, float]  # Source name -> signal
    agreement: float  # 0 to 1
    
    @property
    def is_tradeable(self) -> bool:
        """Check if signal is strong enough to trade."""
        return abs(self.strength) > 0.3 and self.confidence > 0.5


class SignalCombiner:
    """Combine signals from multiple models/strategies.
    
    Supports multiple combination methods:
    - Equal weight
    - Performance weighted
    - Confidence weighted
    - Voting
    
    Example:
        >>> combiner = SignalCombiner()
        >>> combiner.add_signal("momentum", 0.8, confidence=0.7)
        >>> combiner.add_signal("ml_alpha", 0.5, confidence=0.9)
        >>> combined = combiner.combine()
    """
    
    def __init__(
        self,
        method: str = "confidence_weighted",
        min_agreement: float = 0.5,
    ) -> None:
        """Initialize signal combiner.
        
        Args:
            method: Combination method.
            min_agreement: Minimum source agreement to trade.
        """
        self.method = method
        self.min_agreement = min_agreement
        
        # Store signals by symbol
        self._signals: dict[str, dict[str, tuple[float, float]]] = {}
        
        # Source weights (for performance weighting)
        self._source_weights: dict[str, float] = {}
    
    def add_signal(
        self,
        symbol: str,
        source: str,
        signal: float,
        confidence: float = 1.0,
    ) -> None:
        """Add a signal from a source.
        
        Args:
            symbol: Trading symbol.
            source: Signal source name.
            signal: Signal value (-1 to 1).
            confidence: Confidence level (0 to 1).
        """
        if symbol not in self._signals:
            self._signals[symbol] = {}
        
        self._signals[symbol][source] = (signal, confidence)
    
    def set_source_weight(self, source: str, weight: float) -> None:
        """Set weight for a signal source.
        
        Args:
            source: Source name.
            weight: Weight (0 to 1).
        """
        self._source_weights[source] = weight
    
    def combine(self, symbol: str) -> CombinedSignal:
        """Combine signals for a symbol.
        
        Args:
            symbol: Trading symbol.
            
        Returns:
            Combined signal.
        """
        if symbol not in self._signals or not self._signals[symbol]:
            return CombinedSignal(
                symbol=symbol,
                direction=Side.BUY,  # Neutral
                strength=0.0,
                confidence=0.0,
                sources={},
                agreement=0.0,
            )
        
        signals = self._signals[symbol]
        
        if self.method == "equal":
            combined, confidence = self._combine_equal(signals)
        elif self.method == "confidence_weighted":
            combined, confidence = self._combine_confidence_weighted(signals)
        elif self.method == "performance_weighted":
            combined, confidence = self._combine_performance_weighted(signals)
        elif self.method == "voting":
            combined, confidence = self._combine_voting(signals)
        else:
            combined, confidence = self._combine_equal(signals)
        
        # Calculate agreement
        signal_directions = [np.sign(s[0]) for s in signals.values() if s[0] != 0]
        if signal_directions:
            dominant = max(set(signal_directions), key=signal_directions.count)
            agreement = signal_directions.count(dominant) / len(signal_directions)
        else:
            agreement = 0.0
        
        direction = Side.BUY if combined > 0 else Side.SELL
        
        return CombinedSignal(
            symbol=symbol,
            direction=direction,
            strength=combined,
            confidence=confidence,
            sources={src: sig[0] for src, sig in signals.items()},
            agreement=agreement,
        )
    
    def combine_all(self) -> dict[str, CombinedSignal]:
        """Combine signals for all symbols.
        
        Returns:
            Dictionary of symbol -> combined signal.
        """
        return {symbol: self.combine(symbol) for symbol in self._signals}
    
    def _combine_equal(
        self,
        signals: dict[str, tuple[float, float]],
    ) -> tuple[float, float]:
        """Equal weight combination."""
        signal_values = [s[0] for s in signals.values()]
        confidences = [s[1] for s in signals.values()]
        
        combined = np.mean(signal_values)
        confidence = np.mean(confidences)
        
        return combined, confidence
    
    def _combine_confidence_weighted(
        self,
        signals: dict[str, tuple[float, float]],
    ) -> tuple[float, float]:
        """Confidence-weighted combination."""
        weighted_signals = []
        total_weight = 0
        
        for signal, confidence in signals.values():
            weighted_signals.append(signal * confidence)
            total_weight += confidence
        
        if total_weight == 0:
            return 0.0, 0.0
        
        combined = sum(weighted_signals) / total_weight
        avg_confidence = total_weight / len(signals)
        
        return combined, avg_confidence
    
    def _combine_performance_weighted(
        self,
        signals: dict[str, tuple[float, float]],
    ) -> tuple[float, float]:
        """Performance-weighted combination."""
        weighted_signals = []
        total_weight = 0
        confidences = []
        
        for source, (signal, confidence) in signals.items():
            weight = self._source_weights.get(source, 1.0)
            weighted_signals.append(signal * weight)
            total_weight += weight
            confidences.append(confidence)
        
        if total_weight == 0:
            return 0.0, 0.0
        
        combined = sum(weighted_signals) / total_weight
        avg_confidence = np.mean(confidences)
        
        return combined, avg_confidence
    
    def _combine_voting(
        self,
        signals: dict[str, tuple[float, float]],
    ) -> tuple[float, float]:
        """Voting-based combination."""
        bullish = sum(1 for s, _ in signals.values() if s > 0)
        bearish = sum(1 for s, _ in signals.values() if s < 0)
        total = len(signals)
        
        if bullish > bearish:
            combined = bullish / total
        elif bearish > bullish:
            combined = -bearish / total
        else:
            combined = 0.0
        
        confidence = max(bullish, bearish) / total
        
        return combined, confidence
    
    def clear(self, symbol: Optional[str] = None) -> None:
        """Clear signals.
        
        Args:
            symbol: Symbol to clear (None = clear all).
        """
        if symbol:
            self._signals.pop(symbol, None)
        else:
            self._signals.clear()
    
    def get_signal_summary(self) -> pd.DataFrame:
        """Get summary of all signals."""
        data = []
        
        for symbol, signals in self._signals.items():
            combined = self.combine(symbol)
            data.append({
                "Symbol": symbol,
                "Direction": combined.direction.value,
                "Strength": f"{combined.strength:.2f}",
                "Confidence": f"{combined.confidence:.2f}",
                "Agreement": f"{combined.agreement:.0%}",
                "Sources": len(signals),
                "Tradeable": "✓" if combined.is_tradeable else "✗",
            })
        
        return pd.DataFrame(data)
