"""
Walk-Forward Validation Engine.

Implements walk-forward optimization and out-of-sample testing
to prevent overfitting.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Type, Optional

import numpy as np
import pandas as pd

from quantxalgo.config.logging_config import get_logger
from quantxalgo.core.types import PerformanceMetrics
from quantxalgo.engine.backtest import Backtester, BacktestConfig
from quantxalgo.strategies.base import Strategy

logger = get_logger(__name__)


@dataclass
class ValidationFold:
    """Single walk-forward fold results."""
    
    fold_number: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    best_params: dict
    train_sharpe: float
    test_sharpe: float
    test_return: float
    test_max_dd: float


@dataclass
class ValidationResult:
    """Complete walk-forward validation results."""
    
    strategy_name: str
    total_folds: int
    folds: list[ValidationFold]
    
    # Aggregated out-of-sample metrics
    oos_sharpe: float
    oos_return: float
    oos_max_dd: float
    oos_win_rate: float  # % of folds profitable
    
    # Robustness metrics
    sharpe_degradation: float  # In-sample vs out-of-sample Sharpe drop
    parameter_stability: float  # How stable are optimal params across folds
    
    @property
    def is_robust(self) -> bool:
        """Check if strategy passes robustness criteria."""
        return (
            self.oos_sharpe > 0.5 and
            self.oos_win_rate > 0.6 and
            self.sharpe_degradation < 0.5
        )


class WalkForwardValidator:
    """Walk-forward optimization and validation.
    
    Implements rolling-window validation to test strategy robustness:
    1. Train on window [0, train_end]
    2. Test on window [train_end, test_end]
    3. Roll forward and repeat
    
    Example:
        >>> validator = WalkForwardValidator(
        ...     train_window=252,
        ...     test_window=63,
        ...     step_size=21
        ... )
        >>> result = await validator.validate(
        ...     strategy_cls=MACrossover,
        ...     data=data,
        ...     param_grid={"fast_period": [5, 10, 20], "slow_period": [20, 50, 100]}
        ... )
    """
    
    def __init__(
        self,
        train_window: int = 252,
        test_window: int = 63,
        step_size: int = 21,
        optimization_metric: str = "sharpe_ratio",
    ) -> None:
        """Initialize validator.
        
        Args:
            train_window: Number of bars for training (default: 1 year)
            test_window: Number of bars for testing (default: 3 months)
            step_size: Number of bars to roll forward (default: 1 month)
            optimization_metric: Metric to optimize (default: sharpe_ratio)
        """
        self.train_window = train_window
        self.test_window = test_window
        self.step_size = step_size
        self.optimization_metric = optimization_metric
    
    def generate_folds(
        self,
        data: pd.DataFrame,
    ) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
        """Generate train/test folds for walk-forward.
        
        Args:
            data: Full historical data.
            
        Returns:
            List of (train_data, test_data) tuples.
        """
        folds = []
        start = 0
        
        while start + self.train_window + self.test_window <= len(data):
            train_end = start + self.train_window
            test_end = train_end + self.test_window
            
            train_data = data.iloc[start:train_end]
            test_data = data.iloc[train_end:test_end]
            
            folds.append((train_data, test_data))
            start += self.step_size
        
        logger.info(
            "Generated validation folds",
            total_folds=len(folds),
            train_window=self.train_window,
            test_window=self.test_window,
        )
        
        return folds
    
    async def validate(
        self,
        strategy_cls: Type[Strategy],
        data: dict[str, pd.DataFrame],
        param_grid: dict,
        symbols: list[str],
        initial_capital: float = 1_000_000,
    ) -> ValidationResult:
        """Run walk-forward validation.
        
        Args:
            strategy_cls: Strategy class to validate.
            data: Historical data by symbol.
            param_grid: Parameter grid to search.
            symbols: Symbols to trade.
            initial_capital: Starting capital.
            
        Returns:
            ValidationResult with all metrics.
        """
        # Use first symbol's data for fold generation
        reference_data = list(data.values())[0]
        folds_data = self.generate_folds(reference_data)
        
        fold_results = []
        all_best_params = []
        
        for fold_num, (train_ref, test_ref) in enumerate(folds_data):
            train_start = train_ref.index[0]
            train_end = train_ref.index[-1]
            test_start = test_ref.index[0]
            test_end = test_ref.index[-1]
            
            logger.info(
                f"Processing fold {fold_num + 1}/{len(folds_data)}",
                train_period=f"{train_start.date()} to {train_end.date()}",
                test_period=f"{test_start.date()} to {test_end.date()}",
            )
            
            # Slice data for each symbol
            train_data = {
                sym: df[(df.index >= train_start) & (df.index <= train_end)]
                for sym, df in data.items()
            }
            test_data = {
                sym: df[(df.index >= test_start) & (df.index <= test_end)]
                for sym, df in data.items()
            }
            
            # Optimize on training data
            best_params, train_sharpe = await self._optimize(
                strategy_cls, train_data, param_grid, symbols, initial_capital
            )
            all_best_params.append(best_params)
            
            # Test on out-of-sample
            test_result = await self._backtest_params(
                strategy_cls, test_data, best_params, symbols, initial_capital
            )
            
            fold_results.append(ValidationFold(
                fold_number=fold_num + 1,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                best_params=best_params,
                train_sharpe=train_sharpe,
                test_sharpe=test_result["performance"]["sharpe_ratio"],
                test_return=test_result["performance"]["total_return"],
                test_max_dd=test_result["performance"]["max_drawdown"],
            ))
        
        # Calculate aggregated metrics
        return self._aggregate_results(strategy_cls.__name__, fold_results, all_best_params)
    
    async def _optimize(
        self,
        strategy_cls: Type[Strategy],
        data: dict[str, pd.DataFrame],
        param_grid: dict,
        symbols: list[str],
        initial_capital: float,
    ) -> tuple[dict, float]:
        """Find optimal parameters on training data."""
        best_params = {}
        best_metric = -np.inf
        
        # Generate parameter combinations
        param_combinations = self._generate_param_combinations(param_grid)
        
        for params in param_combinations:
            result = await self._backtest_params(
                strategy_cls, data, params, symbols, initial_capital
            )
            
            metric = result["performance"].get(self.optimization_metric, 0)
            
            if metric > best_metric:
                best_metric = metric
                best_params = params.copy()
        
        return best_params, best_metric
    
    async def _backtest_params(
        self,
        strategy_cls: Type[Strategy],
        data: dict[str, pd.DataFrame],
        params: dict,
        symbols: list[str],
        initial_capital: float,
    ) -> dict:
        """Run backtest with specific parameters."""
        # Get date range from data
        all_dates = set()
        for df in data.values():
            all_dates.update(df.index)
        
        dates = sorted(all_dates)
        start_date = dates[0]
        end_date = dates[-1]
        
        strategy = strategy_cls(
            name=f"{strategy_cls.__name__}_opt",
            params=params,
            symbols=symbols,
        )
        
        config = BacktestConfig(
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
        )
        
        backtester = Backtester(config)
        return await backtester.run(strategy, data)
    
    def _generate_param_combinations(self, param_grid: dict) -> list[dict]:
        """Generate all parameter combinations from grid."""
        if not param_grid:
            return [{}]
        
        import itertools
        
        keys = param_grid.keys()
        values = param_grid.values()
        
        combinations = []
        for combo in itertools.product(*values):
            combinations.append(dict(zip(keys, combo)))
        
        return combinations
    
    def _aggregate_results(
        self,
        strategy_name: str,
        folds: list[ValidationFold],
        all_params: list[dict],
    ) -> ValidationResult:
        """Aggregate fold results into final validation result."""
        if not folds:
            return ValidationResult(
                strategy_name=strategy_name,
                total_folds=0,
                folds=[],
                oos_sharpe=0,
                oos_return=0,
                oos_max_dd=0,
                oos_win_rate=0,
                sharpe_degradation=0,
                parameter_stability=0,
            )
        
        # Out-of-sample metrics (average across folds)
        oos_sharpes = [f.test_sharpe for f in folds]
        oos_returns = [f.test_return for f in folds]
        oos_max_dds = [f.test_max_dd for f in folds]
        train_sharpes = [f.train_sharpe for f in folds]
        
        oos_sharpe = np.mean(oos_sharpes)
        oos_return = np.mean(oos_returns)
        oos_max_dd = np.min(oos_max_dds)  # Worst drawdown
        
        # Win rate: % of folds with positive return
        profitable_folds = sum(1 for r in oos_returns if r > 0)
        oos_win_rate = profitable_folds / len(folds)
        
        # Sharpe degradation
        avg_train_sharpe = np.mean(train_sharpes)
        sharpe_degradation = (avg_train_sharpe - oos_sharpe) / avg_train_sharpe if avg_train_sharpe > 0 else 0
        
        # Parameter stability (simplified: check if most common params repeat)
        if all_params:
            param_strs = [str(p) for p in all_params]
            unique_params = len(set(param_strs))
            parameter_stability = 1 - (unique_params / len(all_params))
        else:
            parameter_stability = 0
        
        result = ValidationResult(
            strategy_name=strategy_name,
            total_folds=len(folds),
            folds=folds,
            oos_sharpe=oos_sharpe,
            oos_return=oos_return,
            oos_max_dd=oos_max_dd,
            oos_win_rate=oos_win_rate,
            sharpe_degradation=sharpe_degradation,
            parameter_stability=parameter_stability,
        )
        
        logger.info(
            "Validation complete",
            strategy=strategy_name,
            oos_sharpe=f"{oos_sharpe:.2f}",
            oos_return=f"{oos_return:.2%}",
            oos_win_rate=f"{oos_win_rate:.1%}",
            is_robust=result.is_robust,
        )
        
        return result
