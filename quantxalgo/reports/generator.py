"""
Report Generator for QuantXalgo.

Generates comprehensive performance reports.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import pandas as pd

from quantxalgo.config.logging_config import get_logger
from quantxalgo.metrics.risk_metrics import MetricsEngine

logger = get_logger(__name__)


@dataclass
class ReportConfig:
    """Report configuration."""
    
    include_trades: bool = True
    include_monthly: bool = True
    include_drawdown: bool = True
    include_rolling: bool = True
    rolling_window: int = 252


class ReportGenerator:
    """Generate comprehensive performance reports.
    
    Creates detailed reports for backtests including:
    - Performance summary
    - Monthly returns table
    - Drawdown analysis
    - Trade statistics
    - Rolling metrics
    
    Example:
        >>> generator = ReportGenerator()
        >>> report = generator.generate(backtest_result)
        >>> generator.save_html(report, "report.html")
    """
    
    def __init__(
        self,
        config: Optional[ReportConfig] = None,
    ) -> None:
        self.config = config or ReportConfig()
        self.metrics = MetricsEngine()
    
    def generate(
        self,
        equity_curve: pd.Series,
        trades: list[dict],
        benchmark: Optional[pd.Series] = None,
    ) -> dict:
        """Generate comprehensive report.
        
        Args:
            equity_curve: Equity curve series.
            trades: List of trade dictionaries.
            benchmark: Optional benchmark equity curve.
            
        Returns:
            Report dictionary with all sections.
        """
        report = {}
        returns = equity_curve.pct_change().dropna()
        
        # Summary section
        report["summary"] = self._generate_summary(equity_curve, returns, trades)
        
        # Monthly returns
        if self.config.include_monthly:
            report["monthly_returns"] = self._generate_monthly_returns(returns)
        
        # Drawdown analysis
        if self.config.include_drawdown:
            report["drawdown"] = self._generate_drawdown_analysis(equity_curve)
        
        # Trade statistics
        if self.config.include_trades:
            report["trades"] = self._generate_trade_stats(trades)
        
        # Rolling metrics
        if self.config.include_rolling:
            report["rolling"] = self._generate_rolling_metrics(returns)
        
        # Benchmark comparison
        if benchmark is not None:
            report["benchmark"] = self._generate_benchmark_comparison(
                returns, benchmark.pct_change().dropna()
            )
        
        logger.info("Report generated", sections=list(report.keys()))
        return report
    
    def _generate_summary(
        self,
        equity_curve: pd.Series,
        returns: pd.Series,
        trades: list[dict],
    ) -> dict:
        """Generate performance summary."""
        max_dd, peak_date, trough_date = self.metrics.max_drawdown(equity_curve)
        
        return {
            "start_date": equity_curve.index[0].strftime("%Y-%m-%d"),
            "end_date": equity_curve.index[-1].strftime("%Y-%m-%d"),
            "total_days": len(equity_curve),
            "starting_equity": equity_curve.iloc[0],
            "ending_equity": equity_curve.iloc[-1],
            "total_return": self.metrics.total_return(equity_curve),
            "cagr": self.metrics.cagr(equity_curve),
            "volatility": self.metrics.volatility(returns),
            "sharpe_ratio": self.metrics.sharpe_ratio(returns),
            "sortino_ratio": self.metrics.sortino_ratio(returns),
            "calmar_ratio": self.metrics.calmar_ratio(returns, equity_curve),
            "max_drawdown": max_dd,
            "max_dd_peak": peak_date.strftime("%Y-%m-%d") if peak_date else None,
            "max_dd_trough": trough_date.strftime("%Y-%m-%d") if trough_date else None,
            "var_95": self.metrics.var(returns, 0.95),
            "cvar_95": self.metrics.cvar(returns, 0.95),
            "skewness": self.metrics.skewness(returns),
            "kurtosis": self.metrics.kurtosis(returns),
            "total_trades": len(trades),
        }
    
    def _generate_monthly_returns(self, returns: pd.Series) -> pd.DataFrame:
        """Generate monthly returns table."""
        monthly = returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)
        
        # Create pivot table: years as rows, months as columns
        monthly_df = monthly.to_frame("return")
        monthly_df["year"] = monthly_df.index.year
        monthly_df["month"] = monthly_df.index.month
        
        pivot = monthly_df.pivot(index="year", columns="month", values="return")
        pivot.columns = [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
        ][:len(pivot.columns)]
        
        # Add yearly total
        pivot["Year"] = (1 + pivot.fillna(0)).prod(axis=1) - 1
        
        return pivot
    
    def _generate_drawdown_analysis(self, equity_curve: pd.Series) -> dict:
        """Generate drawdown analysis."""
        rolling_max = equity_curve.expanding().max()
        drawdowns = equity_curve / rolling_max - 1
        
        # Find top 5 drawdowns
        dd_periods = []
        in_drawdown = False
        start_idx = None
        
        for i, (date, dd) in enumerate(drawdowns.items()):
            if dd < 0 and not in_drawdown:
                in_drawdown = True
                start_idx = i
            elif dd == 0 and in_drawdown:
                in_drawdown = False
                dd_periods.append((start_idx, i))
        
        if in_drawdown:
            dd_periods.append((start_idx, len(drawdowns) - 1))
        
        # Calculate stats for each period
        top_drawdowns = []
        for start, end in dd_periods:
            period_dd = drawdowns.iloc[start:end + 1].min()
            duration = end - start
            top_drawdowns.append({
                "start": drawdowns.index[start].strftime("%Y-%m-%d"),
                "end": drawdowns.index[end].strftime("%Y-%m-%d"),
                "drawdown": period_dd,
                "duration": duration,
            })
        
        # Sort by severity
        top_drawdowns.sort(key=lambda x: x["drawdown"])
        
        return {
            "current_drawdown": drawdowns.iloc[-1],
            "avg_drawdown": drawdowns[drawdowns < 0].mean(),
            "max_duration_days": self.metrics.drawdown_duration(equity_curve),
            "top_drawdowns": top_drawdowns[:5],
        }
    
    def _generate_trade_stats(self, trades: list[dict]) -> dict:
        """Generate trade statistics."""
        if not trades:
            return {"total_trades": 0}
        
        trades_df = pd.DataFrame(trades)
        
        # Calculate P&L if not present
        if "pnl" not in trades_df.columns:
            # Simplified: assume P&L is available or calculate from price changes
            trades_df["pnl"] = 0
        
        winning = trades_df[trades_df["pnl"] > 0]
        losing = trades_df[trades_df["pnl"] < 0]
        
        return {
            "total_trades": len(trades_df),
            "winning_trades": len(winning),
            "losing_trades": len(losing),
            "win_rate": len(winning) / len(trades_df) if len(trades_df) > 0 else 0,
            "avg_win": winning["pnl"].mean() if len(winning) > 0 else 0,
            "avg_loss": losing["pnl"].mean() if len(losing) > 0 else 0,
            "largest_win": winning["pnl"].max() if len(winning) > 0 else 0,
            "largest_loss": losing["pnl"].min() if len(losing) > 0 else 0,
            "avg_trade_pnl": trades_df["pnl"].mean(),
            "total_pnl": trades_df["pnl"].sum(),
        }
    
    def _generate_rolling_metrics(self, returns: pd.Series) -> pd.DataFrame:
        """Generate rolling metrics."""
        window = self.config.rolling_window
        
        rolling = pd.DataFrame(index=returns.index)
        
        # Rolling returns (annualized)
        rolling["return_252d"] = returns.rolling(window).apply(
            lambda x: (1 + x).prod() ** (252 / len(x)) - 1
        )
        
        # Rolling volatility
        rolling["volatility_252d"] = returns.rolling(window).std() * (252 ** 0.5)
        
        # Rolling Sharpe
        rolling["sharpe_252d"] = rolling["return_252d"] / rolling["volatility_252d"]
        
        # Rolling max drawdown
        equity = (1 + returns).cumprod()
        rolling["max_dd_252d"] = equity.rolling(window).apply(
            lambda x: (x / x.expanding().max() - 1).min()
        )
        
        return rolling.dropna()
    
    def _generate_benchmark_comparison(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series,
    ) -> dict:
        """Generate benchmark comparison."""
        # Align returns
        common = returns.index.intersection(benchmark_returns.index)
        returns = returns.loc[common]
        benchmark = benchmark_returns.loc[common]
        
        # Calculate alpha and beta
        covariance = returns.cov(benchmark)
        variance = benchmark.var()
        beta = covariance / variance if variance > 0 else 0
        alpha = returns.mean() * 252 - beta * benchmark.mean() * 252
        
        # Tracking error
        tracking_error = (returns - benchmark).std() * (252 ** 0.5)
        
        # Information ratio
        active_return = (returns.mean() - benchmark.mean()) * 252
        info_ratio = active_return / tracking_error if tracking_error > 0 else 0
        
        return {
            "alpha": alpha,
            "beta": beta,
            "correlation": returns.corr(benchmark),
            "tracking_error": tracking_error,
            "information_ratio": info_ratio,
            "strategy_return": (1 + returns).prod() - 1,
            "benchmark_return": (1 + benchmark).prod() - 1,
            "excess_return": (1 + returns).prod() - (1 + benchmark).prod(),
        }
    
    def to_dataframe(self, report: dict) -> pd.DataFrame:
        """Convert summary to DataFrame for display."""
        if "summary" in report:
            summary = report["summary"]
            data = [
                ("Total Return", f"{summary['total_return']:.2%}"),
                ("CAGR", f"{summary['cagr']:.2%}"),
                ("Volatility", f"{summary['volatility']:.2%}"),
                ("Sharpe Ratio", f"{summary['sharpe_ratio']:.2f}"),
                ("Sortino Ratio", f"{summary['sortino_ratio']:.2f}"),
                ("Max Drawdown", f"{summary['max_drawdown']:.2%}"),
                ("Calmar Ratio", f"{summary['calmar_ratio']:.2f}"),
                ("Total Trades", summary['total_trades']),
            ]
            return pd.DataFrame(data, columns=["Metric", "Value"])
        return pd.DataFrame()
