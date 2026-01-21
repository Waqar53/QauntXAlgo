"""
Strategy Competition System.

Runs multiple strategies in parallel, tracks performance,
and allocates capital dynamically based on results.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from quantxalgo.config.logging_config import get_logger
from quantxalgo.core.types import PerformanceMetrics
from quantxalgo.strategies.base import Strategy
from quantxalgo.metrics.risk_metrics import MetricsEngine

logger = get_logger(__name__)


@dataclass
class StrategyPerformance:
    """Tracked performance for a single strategy."""
    
    strategy_name: str
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    volatility: float = 0.0
    trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    
    # Rolling metrics
    returns_series: list = field(default_factory=list)
    
    # Competition score
    score: float = 0.0
    rank: int = 0
    
    # Status
    is_active: bool = True
    allocation: float = 0.0


class StrategyCompetition:
    """Multi-strategy competition and ranking system.
    
    Tracks performance of multiple strategies and provides:
    - Real-time performance ranking
    - Dynamic capital allocation based on performance
    - Automatic strategy disabling based on drawdown
    
    Example:
        >>> competition = StrategyCompetition(
        ...     strategies=[strategy1, strategy2, strategy3],
        ...     min_sharpe=0.5,
        ...     max_drawdown=0.15
        ... )
        >>> allocations = competition.get_allocations()
    """
    
    def __init__(
        self,
        strategies: list[Strategy],
        min_sharpe: float = 0.0,
        max_drawdown: float = 0.20,
        ranking_window: int = 60,
        reallocation_frequency: int = 21,
    ) -> None:
        """Initialize competition.
        
        Args:
            strategies: List of competing strategies.
            min_sharpe: Minimum Sharpe to remain active.
            max_drawdown: Maximum drawdown before disabling.
            ranking_window: Days for rolling performance calculation.
            reallocation_frequency: Days between reallocations.
        """
        self.strategies = {s.name: s for s in strategies}
        self.min_sharpe = min_sharpe
        self.max_drawdown = max_drawdown
        self.ranking_window = ranking_window
        self.reallocation_frequency = reallocation_frequency
        
        # Performance tracking
        self.performance: dict[str, StrategyPerformance] = {
            s.name: StrategyPerformance(strategy_name=s.name)
            for s in strategies
        }
        
        self.metrics_engine = MetricsEngine()
        self._bar_count = 0
        self._last_reallocation = 0
        
        logger.info(
            "Strategy competition initialized",
            num_strategies=len(strategies),
            min_sharpe=min_sharpe,
            max_drawdown=max_drawdown,
        )
    
    def update_performance(
        self,
        strategy_name: str,
        daily_return: float,
        trade_completed: bool = False,
        trade_pnl: float = 0.0,
    ) -> None:
        """Update strategy performance with new data.
        
        Args:
            strategy_name: Name of strategy.
            daily_return: Daily return percentage.
            trade_completed: Whether a trade was completed.
            trade_pnl: P&L if trade completed.
        """
        if strategy_name not in self.performance:
            return
        
        perf = self.performance[strategy_name]
        perf.returns_series.append(daily_return)
        
        # Keep only ranking window
        if len(perf.returns_series) > self.ranking_window:
            perf.returns_series = perf.returns_series[-self.ranking_window:]
        
        if trade_completed:
            perf.trades += 1
        
        # Recalculate metrics
        if len(perf.returns_series) >= 10:
            returns = pd.Series(perf.returns_series)
            
            perf.total_return = (1 + returns).prod() - 1
            perf.volatility = self.metrics_engine.volatility(returns)
            perf.sharpe_ratio = self.metrics_engine.sharpe_ratio(returns)
            
            # Calculate drawdown
            equity = (1 + returns).cumprod()
            perf.max_drawdown = (equity / equity.expanding().max() - 1).min()
        
        # Check if strategy should be disabled
        if perf.max_drawdown < -self.max_drawdown:
            perf.is_active = False
            logger.warning(
                "Strategy disabled due to drawdown",
                strategy=strategy_name,
                drawdown=f"{perf.max_drawdown:.2%}",
            )
    
    def rank_strategies(self) -> list[StrategyPerformance]:
        """Rank all strategies by composite score.
        
        Returns:
            Sorted list of strategy performances.
        """
        active_strategies = [
            p for p in self.performance.values()
            if p.is_active
        ]
        
        if not active_strategies:
            return list(self.performance.values())
        
        # Calculate composite score
        for perf in active_strategies:
            # Score = Sharpe * (1 - abs(DD)) * (1 + return)
            dd_penalty = 1 - min(abs(perf.max_drawdown), 1)
            return_bonus = 1 + max(perf.total_return, 0)
            
            perf.score = perf.sharpe_ratio * dd_penalty * return_bonus
        
        # Sort by score
        sorted_perfs = sorted(
            active_strategies,
            key=lambda p: p.score,
            reverse=True
        )
        
        # Assign ranks
        for i, perf in enumerate(sorted_perfs):
            perf.rank = i + 1
        
        return sorted_perfs
    
    def get_allocations(
        self,
        method: str = "sharpe_weighted",
    ) -> dict[str, float]:
        """Get capital allocations for each strategy.
        
        Args:
            method: Allocation method (equal, sharpe_weighted, rank_weighted)
            
        Returns:
            Dictionary mapping strategy name to allocation (0-1).
        """
        self._bar_count += 1
        
        # Only reallocate at specified frequency
        if self._bar_count - self._last_reallocation < self.reallocation_frequency:
            return {name: perf.allocation for name, perf in self.performance.items()}
        
        self._last_reallocation = self._bar_count
        
        rankings = self.rank_strategies()
        active = [p for p in rankings if p.is_active and p.sharpe_ratio >= self.min_sharpe]
        
        if not active:
            # Equal weight all strategies if none meet criteria
            weight = 1.0 / len(self.performance) if self.performance else 0
            allocations = {name: weight for name in self.performance}
        elif method == "equal":
            weight = 1.0 / len(active)
            allocations = {p.strategy_name: weight for p in active}
        elif method == "sharpe_weighted":
            total_sharpe = sum(max(p.sharpe_ratio, 0.01) for p in active)
            allocations = {
                p.strategy_name: max(p.sharpe_ratio, 0.01) / total_sharpe
                for p in active
            }
        elif method == "rank_weighted":
            # Higher rank = more weight
            weights = [(len(active) - p.rank + 1) for p in active]
            total_weight = sum(weights)
            allocations = {
                p.strategy_name: w / total_weight
                for p, w in zip(active, weights)
            }
        else:
            allocations = {}
        
        # Set inactive strategies to 0
        for name in self.performance:
            if name not in allocations:
                allocations[name] = 0.0
        
        # Store allocations
        for name, alloc in allocations.items():
            self.performance[name].allocation = alloc
        
        logger.info(
            "Strategy allocations updated",
            allocations={k: f"{v:.1%}" for k, v in allocations.items()},
            method=method,
        )
        
        return allocations
    
    def get_leaderboard(self) -> pd.DataFrame:
        """Get strategy leaderboard as DataFrame."""
        rankings = self.rank_strategies()
        
        data = []
        for perf in rankings:
            data.append({
                "Rank": perf.rank,
                "Strategy": perf.strategy_name,
                "Return": f"{perf.total_return:.2%}",
                "Sharpe": f"{perf.sharpe_ratio:.2f}",
                "Max DD": f"{perf.max_drawdown:.2%}",
                "Trades": perf.trades,
                "Score": f"{perf.score:.2f}",
                "Active": "✓" if perf.is_active else "✗",
                "Allocation": f"{perf.allocation:.1%}",
            })
        
        return pd.DataFrame(data)
    
    def enable_strategy(self, strategy_name: str) -> None:
        """Re-enable a disabled strategy."""
        if strategy_name in self.performance:
            self.performance[strategy_name].is_active = True
            logger.info("Strategy re-enabled", strategy=strategy_name)
    
    def disable_strategy(self, strategy_name: str) -> None:
        """Manually disable a strategy."""
        if strategy_name in self.performance:
            self.performance[strategy_name].is_active = False
            logger.info("Strategy disabled", strategy=strategy_name)
