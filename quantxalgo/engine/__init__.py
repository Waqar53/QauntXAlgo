"""Event-Driven Backtesting Engine."""

from quantxalgo.engine.backtest import Backtester, BacktestConfig
from quantxalgo.engine.event_queue import EventQueue
from quantxalgo.engine.validation import WalkForwardValidator, ValidationResult
from quantxalgo.engine.competition import StrategyCompetition
from quantxalgo.engine.regime import RegimeDetector, RegimeState

__all__ = [
    "Backtester",
    "BacktestConfig",
    "EventQueue",
    "WalkForwardValidator",
    "ValidationResult",
    "StrategyCompetition",
    "RegimeDetector",
    "RegimeState",
]

