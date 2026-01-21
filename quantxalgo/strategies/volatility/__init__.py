"""
Volatility-based strategies.

Strategies that trade volatility or adjust based on volatility conditions.
"""

from quantxalgo.strategies.volatility.volatility_breakout import VolatilityBreakoutStrategy
from quantxalgo.strategies.volatility.squeeze import SqueezeStrategy

__all__ = ["VolatilityBreakoutStrategy", "SqueezeStrategy"]
