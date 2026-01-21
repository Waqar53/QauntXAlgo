"""Statistical Arbitrage strategies."""

from quantxalgo.strategies.statistical.pairs_trading import PairsTradingStrategy
from quantxalgo.strategies.statistical.factor_model import FactorModelStrategy

__all__ = ["PairsTradingStrategy", "FactorModelStrategy"]
