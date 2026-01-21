"""
Advanced Risk Overlays for institutional risk management.

Provides portfolio-level risk controls beyond position-level checks.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from quantxalgo.config.logging_config import get_logger
from quantxalgo.portfolio.manager import PortfolioManager

logger = get_logger(__name__)


@dataclass
class RiskOverlayResult:
    """Result of risk overlay check."""
    
    name: str
    triggered: bool
    current_value: float
    threshold: float
    action: str = "NONE"  # NONE, REDUCE, HEDGE, HALT
    message: str = ""


class RiskOverlays:
    """Advanced portfolio risk overlays.
    
    Implements institutional-grade risk controls:
    - Correlation-based concentration risk
    - Tail risk monitoring (VaR, CVaR)
    - Sector/factor exposure limits
    - Liquidity risk monitoring
    
    Example:
        >>> overlays = RiskOverlays()
        >>> results = overlays.check_all(portfolio, market_data)
        >>> for r in results:
        ...     if r.triggered:
        ...         take_action(r.action)
    """
    
    def __init__(
        self,
        max_correlation: float = 0.7,
        max_var_95: float = 0.05,
        max_sector_exposure: float = 0.30,
        max_beta: float = 1.5,
        min_liquidity_days: float = 5.0,
    ) -> None:
        """Initialize risk overlays.
        
        Args:
            max_correlation: Maximum average portfolio correlation.
            max_var_95: Maximum 95% VaR.
            max_sector_exposure: Maximum single sector exposure.
            max_beta: Maximum portfolio beta.
            min_liquidity_days: Minimum days to liquidate.
        """
        self.max_correlation = max_correlation
        self.max_var_95 = max_var_95
        self.max_sector_exposure = max_sector_exposure
        self.max_beta = max_beta
        self.min_liquidity_days = min_liquidity_days
    
    def check_all(
        self,
        portfolio: PortfolioManager,
        returns: dict[str, pd.Series],
        volumes: Optional[dict[str, pd.Series]] = None,
        benchmark_returns: Optional[pd.Series] = None,
    ) -> list[RiskOverlayResult]:
        """Run all risk overlay checks.
        
        Args:
            portfolio: Current portfolio state.
            returns: Historical returns by symbol.
            volumes: Historical volumes by symbol.
            benchmark_returns: Benchmark return series.
            
        Returns:
            List of risk overlay results.
        """
        results = []
        
        results.append(self.check_correlation(portfolio, returns))
        results.append(self.check_var(portfolio, returns))
        results.append(self.check_concentration(portfolio))
        
        if benchmark_returns is not None:
            results.append(self.check_beta(portfolio, returns, benchmark_returns))
        
        if volumes:
            results.append(self.check_liquidity(portfolio, volumes))
        
        # Log triggered overlays
        triggered = [r for r in results if r.triggered]
        if triggered:
            logger.warning(
                "Risk overlays triggered",
                overlays=[r.name for r in triggered],
                actions=[r.action for r in triggered],
            )
        
        return results
    
    def check_correlation(
        self,
        portfolio: PortfolioManager,
        returns: dict[str, pd.Series],
    ) -> RiskOverlayResult:
        """Check portfolio correlation concentration."""
        symbols = list(portfolio.positions.keys())
        
        if len(symbols) < 2:
            return RiskOverlayResult(
                name="CORRELATION",
                triggered=False,
                current_value=0.0,
                threshold=self.max_correlation,
            )
        
        # Build returns matrix
        returns_df = pd.DataFrame({
            sym: returns.get(sym, pd.Series())
            for sym in symbols
            if sym in returns
        }).dropna()
        
        if len(returns_df.columns) < 2 or len(returns_df) < 20:
            return RiskOverlayResult(
                name="CORRELATION",
                triggered=False,
                current_value=0.0,
                threshold=self.max_correlation,
            )
        
        # Calculate correlation matrix
        corr_matrix = returns_df.corr()
        
        # Average pairwise correlation (excluding diagonal)
        n = len(corr_matrix)
        avg_corr = (corr_matrix.sum().sum() - n) / (n * (n - 1))
        
        triggered = avg_corr > self.max_correlation
        
        return RiskOverlayResult(
            name="CORRELATION",
            triggered=triggered,
            current_value=avg_corr,
            threshold=self.max_correlation,
            action="DIVERSIFY" if triggered else "NONE",
            message=f"Avg correlation: {avg_corr:.2%}" if triggered else "",
        )
    
    def check_var(
        self,
        portfolio: PortfolioManager,
        returns: dict[str, pd.Series],
        confidence: float = 0.95,
        lookback: int = 252,
    ) -> RiskOverlayResult:
        """Check portfolio Value at Risk."""
        if not portfolio.positions:
            return RiskOverlayResult(
                name="VAR",
                triggered=False,
                current_value=0.0,
                threshold=self.max_var_95,
            )
        
        # Calculate portfolio returns
        weights = {}
        total_value = portfolio.total_equity
        
        for sym, pos in portfolio.positions.items():
            weights[sym] = pos.market_value / total_value if total_value > 0 else 0
        
        # Weighted portfolio returns
        portfolio_returns = pd.Series(0.0, index=range(lookback))
        
        for sym, weight in weights.items():
            if sym in returns:
                ret = returns[sym].tail(lookback)
                if len(ret) == lookback:
                    portfolio_returns += weight * ret.values
        
        # Historical VaR
        var_95 = -np.percentile(portfolio_returns, (1 - confidence) * 100)
        
        triggered = var_95 > self.max_var_95
        
        return RiskOverlayResult(
            name="VAR",
            triggered=triggered,
            current_value=var_95,
            threshold=self.max_var_95,
            action="REDUCE" if triggered else "NONE",
            message=f"95% VaR: {var_95:.2%}" if triggered else "",
        )
    
    def check_concentration(
        self,
        portfolio: PortfolioManager,
    ) -> RiskOverlayResult:
        """Check position concentration risk."""
        if not portfolio.positions:
            return RiskOverlayResult(
                name="CONCENTRATION",
                triggered=False,
                current_value=0.0,
                threshold=self.max_sector_exposure,
            )
        
        total_value = portfolio.total_equity
        
        # Find largest position weight
        max_weight = 0.0
        max_symbol = ""
        
        for sym, pos in portfolio.positions.items():
            weight = abs(pos.market_value) / total_value if total_value > 0 else 0
            if weight > max_weight:
                max_weight = weight
                max_symbol = sym
        
        triggered = max_weight > self.max_sector_exposure
        
        return RiskOverlayResult(
            name="CONCENTRATION",
            triggered=triggered,
            current_value=max_weight,
            threshold=self.max_sector_exposure,
            action="REDUCE" if triggered else "NONE",
            message=f"Max position ({max_symbol}): {max_weight:.2%}" if triggered else "",
        )
    
    def check_beta(
        self,
        portfolio: PortfolioManager,
        returns: dict[str, pd.Series],
        benchmark_returns: pd.Series,
    ) -> RiskOverlayResult:
        """Check portfolio beta exposure."""
        if not portfolio.positions or len(benchmark_returns) < 60:
            return RiskOverlayResult(
                name="BETA",
                triggered=False,
                current_value=1.0,
                threshold=self.max_beta,
            )
        
        # Calculate weighted portfolio beta
        weights = {}
        total_value = portfolio.total_equity
        
        for sym, pos in portfolio.positions.items():
            weights[sym] = pos.market_value / total_value if total_value > 0 else 0
        
        portfolio_beta = 0.0
        benchmark_var = benchmark_returns.var()
        
        if benchmark_var == 0:
            return RiskOverlayResult(
                name="BETA",
                triggered=False,
                current_value=1.0,
                threshold=self.max_beta,
            )
        
        for sym, weight in weights.items():
            if sym in returns:
                ret = returns[sym]
                # Align indices
                common = ret.index.intersection(benchmark_returns.index)
                if len(common) >= 60:
                    cov = np.cov(ret.loc[common], benchmark_returns.loc[common])[0, 1]
                    beta = cov / benchmark_var
                    portfolio_beta += weight * beta
        
        triggered = abs(portfolio_beta) > self.max_beta
        
        return RiskOverlayResult(
            name="BETA",
            triggered=triggered,
            current_value=portfolio_beta,
            threshold=self.max_beta,
            action="HEDGE" if triggered else "NONE",
            message=f"Portfolio beta: {portfolio_beta:.2f}" if triggered else "",
        )
    
    def check_liquidity(
        self,
        portfolio: PortfolioManager,
        volumes: dict[str, pd.Series],
        avg_period: int = 20,
    ) -> RiskOverlayResult:
        """Check portfolio liquidity risk."""
        if not portfolio.positions:
            return RiskOverlayResult(
                name="LIQUIDITY",
                triggered=False,
                current_value=0.0,
                threshold=self.min_liquidity_days,
            )
        
        max_days_to_liquidate = 0.0
        illiquid_symbol = ""
        
        for sym, pos in portfolio.positions.items():
            if sym not in volumes:
                continue
            
            vol = volumes[sym].tail(avg_period).mean()
            
            if vol > 0:
                # Assume we can trade 10% of volume
                tradable_per_day = vol * 0.10
                days_to_liquidate = abs(pos.quantity) / tradable_per_day
                
                if days_to_liquidate > max_days_to_liquidate:
                    max_days_to_liquidate = days_to_liquidate
                    illiquid_symbol = sym
        
        triggered = max_days_to_liquidate > self.min_liquidity_days
        
        return RiskOverlayResult(
            name="LIQUIDITY",
            triggered=triggered,
            current_value=max_days_to_liquidate,
            threshold=self.min_liquidity_days,
            action="REDUCE" if triggered else "NONE",
            message=f"Days to liquidate {illiquid_symbol}: {max_days_to_liquidate:.1f}" if triggered else "",
        )
