"""
Metrics Engine for QuantXalgo.

Calculate all performance and risk metrics for strategies and portfolios.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from quantxalgo.config.constants import TRADING_DAYS_PER_YEAR, DEFAULT_RISK_FREE_RATE
from quantxalgo.config.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class PerformanceReport:
    """Comprehensive performance report."""
    
    # Returns
    total_return: float
    annualized_return: float
    
    # Risk
    volatility: float
    max_drawdown: float
    max_drawdown_duration: int  # In days
    
    # Risk-adjusted
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    
    # Trade stats
    total_trades: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    avg_trade_pnl: float
    avg_holding_period: float
    
    # Other
    skewness: float
    kurtosis: float
    tail_ratio: float


class MetricsEngine:
    """Calculate all performance and risk metrics.
    
    Provides comprehensive analysis of trading performance
    including risk-adjusted returns, drawdown analysis,
    and trade statistics.
    
    Example:
        >>> engine = MetricsEngine()
        >>> sharpe = engine.sharpe_ratio(returns)
        >>> report = engine.full_report(returns, trades)
    """
    
    def __init__(
        self,
        risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
        annualization_factor: int = TRADING_DAYS_PER_YEAR,
    ) -> None:
        """Initialize metrics engine.
        
        Args:
            risk_free_rate: Annual risk-free rate.
            annualization_factor: Days to annualize (252 for daily).
        """
        self.risk_free_rate = risk_free_rate
        self.annualization_factor = annualization_factor
    
    # =========================================================================
    # RETURN METRICS
    # =========================================================================
    
    def total_return(self, equity_curve: pd.Series) -> float:
        """Calculate total return."""
        if len(equity_curve) < 2:
            return 0.0
        return (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
    
    def annualized_return(self, returns: pd.Series) -> float:
        """Calculate annualized return."""
        total_days = len(returns)
        cumulative = (1 + returns).prod()
        return cumulative ** (self.annualization_factor / total_days) - 1
    
    def cagr(self, equity_curve: pd.Series) -> float:
        """Compound Annual Growth Rate."""
        if len(equity_curve) < 2:
            return 0.0
        
        total_days = len(equity_curve)
        years = total_days / self.annualization_factor
        
        if years <= 0:
            return 0.0
        
        ending = equity_curve.iloc[-1]
        beginning = equity_curve.iloc[0]
        
        return (ending / beginning) ** (1 / years) - 1
    
    # =========================================================================
    # RISK METRICS
    # =========================================================================
    
    def volatility(
        self,
        returns: pd.Series,
        annualize: bool = True,
    ) -> float:
        """Calculate return volatility (standard deviation)."""
        vol = returns.std()
        if annualize:
            vol *= np.sqrt(self.annualization_factor)
        return vol
    
    def downside_deviation(
        self,
        returns: pd.Series,
        mar: float = 0.0,
        annualize: bool = True,
    ) -> float:
        """Calculate downside deviation (semi-deviation).
        
        Args:
            returns: Return series.
            mar: Minimum acceptable return.
            annualize: Whether to annualize.
        """
        downside = returns[returns < mar]
        if len(downside) == 0:
            return 0.0
        
        dd = np.sqrt((downside ** 2).mean())
        if annualize:
            dd *= np.sqrt(self.annualization_factor)
        return dd
    
    def max_drawdown(
        self,
        equity_curve: pd.Series,
    ) -> tuple[float, datetime, datetime]:
        """Calculate maximum drawdown with peak and trough dates.
        
        Returns:
            Tuple of (max_drawdown, peak_date, trough_date).
        """
        rolling_max = equity_curve.expanding().max()
        drawdowns = equity_curve / rolling_max - 1
        
        max_dd = drawdowns.min()
        
        if max_dd == 0:
            return 0.0, None, None
        
        trough_idx = drawdowns.idxmin()
        peak_idx = equity_curve[:trough_idx].idxmax()
        
        return max_dd, peak_idx, trough_idx
    
    def drawdown_duration(self, equity_curve: pd.Series) -> int:
        """Calculate maximum drawdown duration in bars."""
        rolling_max = equity_curve.expanding().max()
        is_drawdown = equity_curve < rolling_max
        
        # Find longest consecutive drawdown period
        groups = (~is_drawdown).cumsum()
        drawdown_lengths = is_drawdown.groupby(groups).cumsum()
        
        return int(drawdown_lengths.max()) if len(drawdown_lengths) > 0 else 0
    
    def var(
        self,
        returns: pd.Series,
        confidence: float = 0.95,
    ) -> float:
        """Calculate Value at Risk (historical).
        
        Args:
            returns: Return series.
            confidence: Confidence level (0.95 = 95%).
            
        Returns:
            VaR as a positive number representing potential loss.
        """
        return -np.percentile(returns, (1 - confidence) * 100)
    
    def cvar(
        self,
        returns: pd.Series,
        confidence: float = 0.95,
    ) -> float:
        """Calculate Conditional VaR (Expected Shortfall).
        
        Mean of returns worse than VaR.
        """
        var = self.var(returns, confidence)
        return -returns[returns <= -var].mean()
    
    # =========================================================================
    # RISK-ADJUSTED METRICS
    # =========================================================================
    
    def sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(returns) < 2:
            return 0.0
        
        excess_returns = returns - self.risk_free_rate / self.annualization_factor
        
        if excess_returns.std() == 0:
            return 0.0
        
        return np.sqrt(self.annualization_factor) * excess_returns.mean() / excess_returns.std()
    
    def sortino_ratio(self, returns: pd.Series, mar: float = 0.0) -> float:
        """Calculate Sortino ratio using downside deviation."""
        if len(returns) < 2:
            return 0.0
        
        excess_return = returns.mean() - mar / self.annualization_factor
        downside = self.downside_deviation(returns, mar, annualize=False)
        
        if downside == 0:
            return 0.0 if excess_return <= 0 else np.inf
        
        return np.sqrt(self.annualization_factor) * excess_return / downside
    
    def calmar_ratio(self, returns: pd.Series, equity_curve: pd.Series) -> float:
        """Calculate Calmar ratio (CAGR / MaxDD)."""
        cagr = self.cagr(equity_curve)
        max_dd, _, _ = self.max_drawdown(equity_curve)
        
        if max_dd == 0:
            return np.inf if cagr > 0 else 0.0
        
        return cagr / abs(max_dd)
    
    def omega_ratio(self, returns: pd.Series, threshold: float = 0.0) -> float:
        """Calculate Omega ratio (probability-weighted gains / losses)."""
        above = returns[returns > threshold] - threshold
        below = threshold - returns[returns < threshold]
        
        if below.sum() == 0:
            return np.inf
        
        return above.sum() / below.sum()
    
    def information_ratio(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series,
    ) -> float:
        """Calculate Information Ratio vs benchmark."""
        active_return = returns - benchmark_returns
        tracking_error = active_return.std()
        
        if tracking_error == 0:
            return 0.0
        
        return np.sqrt(self.annualization_factor) * active_return.mean() / tracking_error
    
    # =========================================================================
    # TRADE STATISTICS
    # =========================================================================
    
    def win_rate(self, trades: list[dict]) -> float:
        """Calculate percentage of profitable trades."""
        if not trades:
            return 0.0
        
        profitable = sum(1 for t in trades if t.get("pnl", 0) > 0)
        return profitable / len(trades)
    
    def profit_factor(self, trades: list[dict]) -> float:
        """Calculate gross profit / gross loss."""
        gross_profit = sum(t["pnl"] for t in trades if t.get("pnl", 0) > 0)
        gross_loss = abs(sum(t["pnl"] for t in trades if t.get("pnl", 0) < 0))
        
        if gross_loss == 0:
            return np.inf if gross_profit > 0 else 0.0
        
        return gross_profit / gross_loss
    
    def avg_win_loss(self, trades: list[dict]) -> tuple[float, float]:
        """Calculate average win and average loss."""
        wins = [t["pnl"] for t in trades if t.get("pnl", 0) > 0]
        losses = [t["pnl"] for t in trades if t.get("pnl", 0) < 0]
        
        avg_win = np.mean(wins) if wins else 0.0
        avg_loss = np.mean(losses) if losses else 0.0
        
        return avg_win, avg_loss
    
    def expectancy(self, trades: list[dict]) -> float:
        """Calculate trade expectancy (expected value per trade)."""
        if not trades:
            return 0.0
        
        return sum(t.get("pnl", 0) for t in trades) / len(trades)
    
    def payoff_ratio(self, trades: list[dict]) -> float:
        """Calculate payoff ratio (avg win / avg loss)."""
        avg_win, avg_loss = self.avg_win_loss(trades)
        
        if avg_loss == 0:
            return np.inf if avg_win > 0 else 0.0
        
        return abs(avg_win / avg_loss)
    
    # =========================================================================
    # DISTRIBUTION METRICS
    # =========================================================================
    
    def skewness(self, returns: pd.Series) -> float:
        """Calculate return skewness."""
        return returns.skew()
    
    def kurtosis(self, returns: pd.Series) -> float:
        """Calculate return kurtosis (excess)."""
        return returns.kurtosis()
    
    def tail_ratio(self, returns: pd.Series) -> float:
        """Ratio of right tail (95th) to left tail (5th)."""
        p95 = np.percentile(returns, 95)
        p5 = np.percentile(returns, 5)
        
        if p5 == 0:
            return np.inf if p95 > 0 else 0.0
        
        return abs(p95 / p5)
    
    # =========================================================================
    # FULL REPORT
    # =========================================================================
    
    def full_report(
        self,
        equity_curve: pd.Series,
        trades: list[dict],
    ) -> PerformanceReport:
        """Generate comprehensive performance report."""
        returns = equity_curve.pct_change().dropna()
        
        max_dd, _, _ = self.max_drawdown(equity_curve)
        avg_win, avg_loss = self.avg_win_loss(trades)
        
        return PerformanceReport(
            total_return=self.total_return(equity_curve),
            annualized_return=self.annualized_return(returns),
            volatility=self.volatility(returns),
            max_drawdown=max_dd,
            max_drawdown_duration=self.drawdown_duration(equity_curve),
            sharpe_ratio=self.sharpe_ratio(returns),
            sortino_ratio=self.sortino_ratio(returns),
            calmar_ratio=self.calmar_ratio(returns, equity_curve),
            total_trades=len(trades),
            win_rate=self.win_rate(trades),
            profit_factor=self.profit_factor(trades),
            avg_win=avg_win,
            avg_loss=avg_loss,
            avg_trade_pnl=self.expectancy(trades),
            avg_holding_period=0,  # Would need trade details
            skewness=self.skewness(returns),
            kurtosis=self.kurtosis(returns),
            tail_ratio=self.tail_ratio(returns),
        )
