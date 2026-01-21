"""
Tearsheet Generator for QuantXalgo.

Generates professional one-page strategy tearsheets.
"""

from datetime import datetime
from typing import Optional

import pandas as pd

from quantxalgo.config.logging_config import get_logger
from quantxalgo.metrics.risk_metrics import MetricsEngine

logger = get_logger(__name__)


class TearsheetGenerator:
    """Generate professional strategy tearsheets.
    
    Creates a comprehensive one-page summary suitable for
    investor presentations.
    
    Example:
        >>> generator = TearsheetGenerator()
        >>> tearsheet = generator.generate(equity_curve, trades)
        >>> print(tearsheet.to_text())
    """
    
    def __init__(self) -> None:
        self.metrics = MetricsEngine()
    
    def generate(
        self,
        equity_curve: pd.Series,
        trades: list[dict],
        strategy_name: str = "Strategy",
        benchmark_equity: Optional[pd.Series] = None,
    ) -> dict:
        """Generate tearsheet data.
        
        Args:
            equity_curve: Strategy equity curve.
            trades: List of trades.
            strategy_name: Strategy name.
            benchmark_equity: Benchmark equity curve.
            
        Returns:
            Tearsheet data dictionary.
        """
        returns = equity_curve.pct_change().dropna()
        
        tearsheet = {
            "meta": {
                "strategy_name": strategy_name,
                "generated_at": datetime.utcnow().isoformat(),
                "period_start": equity_curve.index[0].strftime("%Y-%m-%d"),
                "period_end": equity_curve.index[-1].strftime("%Y-%m-%d"),
            },
            "performance": self._performance_section(equity_curve, returns),
            "risk": self._risk_section(equity_curve, returns),
            "trading": self._trading_section(trades),
            "returns": self._returns_section(returns),
        }
        
        if benchmark_equity is not None:
            tearsheet["benchmark"] = self._benchmark_section(
                returns, benchmark_equity.pct_change().dropna()
            )
        
        return tearsheet
    
    def _performance_section(
        self,
        equity_curve: pd.Series,
        returns: pd.Series,
    ) -> dict:
        """Generate performance section."""
        return {
            "total_return": self.metrics.total_return(equity_curve),
            "cagr": self.metrics.cagr(equity_curve),
            "best_year": returns.resample("YE").apply(
                lambda x: (1 + x).prod() - 1
            ).max(),
            "worst_year": returns.resample("YE").apply(
                lambda x: (1 + x).prod() - 1
            ).min(),
            "best_month": returns.resample("ME").apply(
                lambda x: (1 + x).prod() - 1
            ).max(),
            "worst_month": returns.resample("ME").apply(
                lambda x: (1 + x).prod() - 1
            ).min(),
            "positive_months": (
                returns.resample("ME").apply(lambda x: (1 + x).prod() - 1) > 0
            ).mean(),
        }
    
    def _risk_section(
        self,
        equity_curve: pd.Series,
        returns: pd.Series,
    ) -> dict:
        """Generate risk section."""
        max_dd, peak, trough = self.metrics.max_drawdown(equity_curve)
        
        return {
            "volatility": self.metrics.volatility(returns),
            "sharpe_ratio": self.metrics.sharpe_ratio(returns),
            "sortino_ratio": self.metrics.sortino_ratio(returns),
            "calmar_ratio": self.metrics.calmar_ratio(returns, equity_curve),
            "max_drawdown": max_dd,
            "var_95": self.metrics.var(returns, 0.95),
            "cvar_95": self.metrics.cvar(returns, 0.95),
            "omega_ratio": self.metrics.omega_ratio(returns),
            "skewness": self.metrics.skewness(returns),
            "kurtosis": self.metrics.kurtosis(returns),
        }
    
    def _trading_section(self, trades: list[dict]) -> dict:
        """Generate trading section."""
        if not trades:
            return {
                "total_trades": 0,
                "win_rate": 0,
                "profit_factor": 0,
                "avg_trade": 0,
            }
        
        return {
            "total_trades": len(trades),
            "win_rate": self.metrics.win_rate(trades),
            "profit_factor": self.metrics.profit_factor(trades),
            "avg_win": self.metrics.avg_win_loss(trades)[0],
            "avg_loss": self.metrics.avg_win_loss(trades)[1],
            "expectancy": self.metrics.expectancy(trades),
            "payoff_ratio": self.metrics.payoff_ratio(trades),
        }
    
    def _returns_section(self, returns: pd.Series) -> dict:
        """Generate returns distribution section."""
        return {
            "mean_daily": returns.mean(),
            "median_daily": returns.median(),
            "std_daily": returns.std(),
            "best_day": returns.max(),
            "worst_day": returns.min(),
            "positive_days": (returns > 0).mean(),
        }
    
    def _benchmark_section(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series,
    ) -> dict:
        """Generate benchmark comparison section."""
        common = returns.index.intersection(benchmark_returns.index)
        returns = returns.loc[common]
        benchmark = benchmark_returns.loc[common]
        
        covariance = returns.cov(benchmark)
        variance = benchmark.var()
        beta = covariance / variance if variance > 0 else 0
        
        return {
            "beta": beta,
            "alpha": returns.mean() * 252 - beta * benchmark.mean() * 252,
            "correlation": returns.corr(benchmark),
            "information_ratio": self.metrics.information_ratio(returns, benchmark),
            "up_capture": (
                returns[benchmark > 0].sum() / benchmark[benchmark > 0].sum()
                if benchmark[benchmark > 0].sum() != 0 else 0
            ),
            "down_capture": (
                returns[benchmark < 0].sum() / benchmark[benchmark < 0].sum()
                if benchmark[benchmark < 0].sum() != 0 else 0
            ),
        }
    
    def to_text(self, tearsheet: dict) -> str:
        """Convert tearsheet to formatted text."""
        lines = []
        meta = tearsheet["meta"]
        
        lines.append("=" * 60)
        lines.append(f" {meta['strategy_name']} - Strategy Tearsheet")
        lines.append("=" * 60)
        lines.append(f"Period: {meta['period_start']} to {meta['period_end']}")
        lines.append("")
        
        # Performance
        perf = tearsheet["performance"]
        lines.append("PERFORMANCE")
        lines.append("-" * 40)
        lines.append(f"  Total Return:    {perf['total_return']:>10.2%}")
        lines.append(f"  CAGR:            {perf['cagr']:>10.2%}")
        lines.append(f"  Best Year:       {perf['best_year']:>10.2%}")
        lines.append(f"  Worst Year:      {perf['worst_year']:>10.2%}")
        lines.append("")
        
        # Risk
        risk = tearsheet["risk"]
        lines.append("RISK METRICS")
        lines.append("-" * 40)
        lines.append(f"  Volatility:      {risk['volatility']:>10.2%}")
        lines.append(f"  Sharpe Ratio:    {risk['sharpe_ratio']:>10.2f}")
        lines.append(f"  Sortino Ratio:   {risk['sortino_ratio']:>10.2f}")
        lines.append(f"  Max Drawdown:    {risk['max_drawdown']:>10.2%}")
        lines.append(f"  VaR (95%):       {risk['var_95']:>10.2%}")
        lines.append("")
        
        # Trading
        trading = tearsheet["trading"]
        lines.append("TRADING STATISTICS")
        lines.append("-" * 40)
        lines.append(f"  Total Trades:    {trading['total_trades']:>10}")
        lines.append(f"  Win Rate:        {trading['win_rate']:>10.1%}")
        lines.append(f"  Profit Factor:   {trading['profit_factor']:>10.2f}")
        lines.append("")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)
