"""
Capital Allocation Engine.

Allocates capital across multiple strategies based on various methods.
"""

from typing import Optional

import numpy as np
import pandas as pd

from quantxalgo.config.logging_config import get_logger
from quantxalgo.strategies.base import Strategy

logger = get_logger(__name__)


class CapitalAllocator:
    """Dynamic capital allocation across strategies.
    
    Supports multiple allocation methods:
    - Equal weight
    - Risk parity (volatility-based)
    - Performance-based (Sharpe-weighted)
    - Kelly criterion
    """
    
    def __init__(
        self,
        total_capital: float,
        min_allocation: float = 0.01,
        max_allocation: float = 0.40,
    ) -> None:
        """Initialize allocator.
        
        Args:
            total_capital: Total capital to allocate.
            min_allocation: Minimum allocation per strategy (0-1).
            max_allocation: Maximum allocation per strategy (0-1).
        """
        self.total_capital = total_capital
        self.min_allocation = min_allocation
        self.max_allocation = max_allocation
    
    def equal_weight(
        self,
        strategies: list[Strategy],
    ) -> dict[str, float]:
        """Equal allocation to all strategies.
        
        Args:
            strategies: List of strategies.
            
        Returns:
            Dictionary mapping strategy name to allocation (0-1).
        """
        if not strategies:
            return {}
        
        weight = 1.0 / len(strategies)
        weight = np.clip(weight, self.min_allocation, self.max_allocation)
        
        return {s.name: weight for s in strategies}
    
    def volatility_parity(
        self,
        strategies: list[Strategy],
        returns: dict[str, pd.Series],
        target_vol: float = 0.10,
        lookback: int = 60,
    ) -> dict[str, float]:
        """Allocate inversely proportional to volatility.
        
        Targets equal risk contribution from each strategy.
        
        Args:
            strategies: List of strategies.
            returns: Dictionary mapping strategy name to returns series.
            target_vol: Target portfolio volatility.
            lookback: Lookback period for volatility calculation.
            
        Returns:
            Dictionary mapping strategy name to allocation.
        """
        if not strategies or not returns:
            return {}
        
        # Calculate volatilities
        vols = {}
        for strategy in strategies:
            if strategy.name in returns:
                ret = returns[strategy.name].tail(lookback)
                vol = ret.std() * np.sqrt(252)
                vols[strategy.name] = max(vol, 0.01)  # Floor at 1%
        
        if not vols:
            return self.equal_weight(strategies)
        
        # Inverse volatility weights
        inv_vols = {name: 1 / vol for name, vol in vols.items()}
        total = sum(inv_vols.values())
        
        weights = {name: iv / total for name, iv in inv_vols.items()}
        
        # Apply constraints
        return self._apply_constraints(weights)
    
    def performance_weighted(
        self,
        strategies: list[Strategy],
        performance: dict[str, float],
        sharpe_floor: float = 0.0,
    ) -> dict[str, float]:
        """Allocate based on risk-adjusted performance (Sharpe ratio).
        
        Better performing strategies get more capital.
        
        Args:
            strategies: List of strategies.
            performance: Dictionary mapping strategy name to Sharpe ratio.
            sharpe_floor: Minimum Sharpe to receive allocation.
            
        Returns:
            Dictionary mapping strategy name to allocation.
        """
        if not strategies or not performance:
            return {}
        
        # Filter by floor and normalize
        eligible = {
            s.name: max(0, performance.get(s.name, 0) - sharpe_floor)
            for s in strategies
            if performance.get(s.name, 0) > sharpe_floor
        }
        
        if not eligible:
            # Fall back to equal weight for active strategies
            return self.equal_weight(strategies)
        
        total = sum(eligible.values()) or 1
        weights = {name: score / total for name, score in eligible.items()}
        
        # Give remaining weight equally to non-performers
        allocated = sum(weights.values())
        remaining = 1.0 - allocated
        
        if remaining > 0:
            non_allocated = [
                s.name for s in strategies
                if s.name not in weights
            ]
            if non_allocated:
                per_strat = (remaining * 0.5) / len(non_allocated)
                for name in non_allocated:
                    weights[name] = per_strat
        
        return self._apply_constraints(weights)
    
    def kelly_criterion(
        self,
        strategies: list[Strategy],
        win_rates: dict[str, float],
        avg_wins: dict[str, float],
        avg_losses: dict[str, float],
        kelly_fraction: float = 0.25,  # Use 25% Kelly for safety
    ) -> dict[str, float]:
        """Allocate using fractional Kelly criterion.
        
        Kelly provides optimal growth rate but with high volatility,
        so we use a fraction of Kelly for stability.
        
        Args:
            strategies: List of strategies.
            win_rates: Win rate per strategy (0-1).
            avg_wins: Average win size per strategy.
            avg_losses: Average loss size per strategy.
            kelly_fraction: Fraction of Kelly to use (0-1).
            
        Returns:
            Dictionary mapping strategy name to allocation.
        """
        if not strategies:
            return {}
        
        kelly_sizes = {}
        
        for strategy in strategies:
            name = strategy.name
            if name not in win_rates:
                continue
            
            p = win_rates[name]  # Probability of win
            w = avg_wins.get(name, 0)  # Average win
            l = abs(avg_losses.get(name, 0))  # Average loss
            
            if l == 0:
                continue
            
            # Kelly formula: f* = (p * w - (1-p) * l) / (w * l)
            # Simplified: f* = p/l - q/w where q = 1-p
            b = w / l  # Win/loss ratio
            q = 1 - p
            
            kelly = (p * b - q) / b if b > 0 else 0
            kelly = max(0, kelly)  # Can't be negative
            
            kelly_sizes[name] = kelly * kelly_fraction
        
        if not kelly_sizes:
            return self.equal_weight(strategies)
        
        # Normalize to sum to 1
        total = sum(kelly_sizes.values())
        if total > 1:
            kelly_sizes = {k: v / total for k, v in kelly_sizes.items()}
        
        return self._apply_constraints(kelly_sizes)
    
    def _apply_constraints(
        self,
        weights: dict[str, float],
    ) -> dict[str, float]:
        """Apply min/max constraints to weights.
        
        Args:
            weights: Raw weights.
            
        Returns:
            Constrained weights.
        """
        constrained = {}
        
        for name, weight in weights.items():
            constrained[name] = np.clip(
                weight,
                self.min_allocation,
                self.max_allocation,
            )
        
        # Renormalize if needed
        total = sum(constrained.values())
        if total > 1:
            constrained = {k: v / total for k, v in constrained.items()}
        
        return constrained
    
    def get_capital(
        self,
        allocations: dict[str, float],
    ) -> dict[str, float]:
        """Convert allocations to dollar amounts.
        
        Args:
            allocations: Dictionary of weight allocations.
            
        Returns:
            Dictionary mapping strategy name to capital amount.
        """
        return {
            name: weight * self.total_capital
            for name, weight in allocations.items()
        }
