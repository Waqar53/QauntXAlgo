"""
Stress Testing Framework.

Simulates portfolio behavior under historical and hypothetical crisis scenarios.
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
class StressScenario:
    """Definition of a stress scenario."""
    
    name: str
    description: str
    shocks: dict[str, float]  # Asset class/symbol -> percentage shock
    volatility_mult: float = 2.0  # Volatility multiplier
    correlation_shock: float = 0.2  # Correlation increases


@dataclass
class StressTestResult:
    """Result of a stress test."""
    
    scenario_name: str
    portfolio_loss: float
    portfolio_loss_pct: float
    worst_position: str
    worst_position_loss: float
    var_breach: bool
    survives: bool


# Predefined historical crisis scenarios
HISTORICAL_SCENARIOS = {
    "2008_financial_crisis": StressScenario(
        name="2008 Financial Crisis",
        description="Lehman Brothers collapse, credit crisis",
        shocks={
            "SPY": -0.38,
            "QQQ": -0.42,
            "IWM": -0.40,
            "XLF": -0.55,  # Financials
            "GLD": 0.05,   # Gold up
            "TLT": 0.20,   # Treasuries up
        },
        volatility_mult=4.0,
        correlation_shock=0.3,
    ),
    "2020_covid_crash": StressScenario(
        name="COVID-19 Crash",
        description="March 2020 market crash",
        shocks={
            "SPY": -0.34,
            "QQQ": -0.28,
            "IWM": -0.41,
            "XLE": -0.50,  # Energy
            "XLF": -0.35,
            "GLD": -0.03,
            "TLT": 0.15,
        },
        volatility_mult=5.0,
        correlation_shock=0.4,
    ),
    "2000_dotcom_bust": StressScenario(
        name="Dot-com Bust",
        description="Tech bubble collapse 2000-2002",
        shocks={
            "SPY": -0.25,
            "QQQ": -0.75,  # Tech decimated
            "IWM": -0.30,
            "XLK": -0.70,  # Tech sector
            "GLD": 0.10,
        },
        volatility_mult=2.5,
        correlation_shock=0.2,
    ),
    "flash_crash": StressScenario(
        name="Flash Crash",
        description="May 2010 flash crash scenario",
        shocks={
            "SPY": -0.09,
            "QQQ": -0.10,
            "IWM": -0.10,
        },
        volatility_mult=10.0,
        correlation_shock=0.5,
    ),
    "rate_shock": StressScenario(
        name="Interest Rate Shock",
        description="Rapid rate increase scenario",
        shocks={
            "SPY": -0.15,
            "QQQ": -0.20,
            "TLT": -0.25,  # Bonds down
            "XLF": 0.05,   # Financials up
            "GLD": -0.10,
        },
        volatility_mult=2.0,
        correlation_shock=0.2,
    ),
}


class StressTester:
    """Stress testing framework for portfolios.
    
    Simulates crisis scenarios and measures portfolio resilience.
    
    Example:
        >>> tester = StressTester()
        >>> result = tester.run_scenario(portfolio, "2008_financial_crisis")
        >>> if not result.survives:
        ...     reduce_risk()
    """
    
    def __init__(
        self,
        max_acceptable_loss: float = 0.25,
        scenarios: Optional[dict[str, StressScenario]] = None,
    ) -> None:
        """Initialize stress tester.
        
        Args:
            max_acceptable_loss: Maximum acceptable portfolio loss.
            scenarios: Custom scenarios (uses historical if None).
        """
        self.max_acceptable_loss = max_acceptable_loss
        self.scenarios = scenarios or HISTORICAL_SCENARIOS
    
    def run_scenario(
        self,
        portfolio: PortfolioManager,
        scenario_name: str,
    ) -> StressTestResult:
        """Run a single stress scenario.
        
        Args:
            portfolio: Portfolio to test.
            scenario_name: Name of scenario to run.
            
        Returns:
            Stress test result.
        """
        if scenario_name not in self.scenarios:
            raise ValueError(f"Unknown scenario: {scenario_name}")
        
        scenario = self.scenarios[scenario_name]
        return self._apply_scenario(portfolio, scenario)
    
    def run_all(
        self,
        portfolio: PortfolioManager,
    ) -> list[StressTestResult]:
        """Run all stress scenarios.
        
        Args:
            portfolio: Portfolio to test.
            
        Returns:
            List of stress test results.
        """
        results = []
        
        for scenario_name in self.scenarios:
            result = self.run_scenario(portfolio, scenario_name)
            results.append(result)
        
        # Sort by severity
        results.sort(key=lambda r: r.portfolio_loss_pct)
        
        logger.info(
            "Stress tests complete",
            scenarios_run=len(results),
            passed=sum(1 for r in results if r.survives),
            failed=sum(1 for r in results if not r.survives),
            worst_case=results[0].scenario_name if results else None,
            worst_loss=f"{results[0].portfolio_loss_pct:.2%}" if results else "N/A",
        )
        
        return results
    
    def _apply_scenario(
        self,
        portfolio: PortfolioManager,
        scenario: StressScenario,
    ) -> StressTestResult:
        """Apply scenario shocks to portfolio."""
        if not portfolio.positions:
            return StressTestResult(
                scenario_name=scenario.name,
                portfolio_loss=0.0,
                portfolio_loss_pct=0.0,
                worst_position="",
                worst_position_loss=0.0,
                var_breach=False,
                survives=True,
            )
        
        initial_equity = portfolio.total_equity
        total_loss = 0.0
        worst_loss = 0.0
        worst_symbol = ""
        
        for symbol, position in portfolio.positions.items():
            # Get shock for this symbol (or use market shock)
            shock = scenario.shocks.get(
                symbol,
                scenario.shocks.get("SPY", -0.20)
            )
            
            # Apply shock
            position_value = position.market_value
            position_loss = position_value * shock
            
            if position.quantity < 0:  # Short position
                position_loss = -position_loss  # Shorts profit from declines
            
            total_loss += position_loss
            
            if position_loss < worst_loss:
                worst_loss = position_loss
                worst_symbol = symbol
        
        portfolio_loss_pct = total_loss / initial_equity if initial_equity > 0 else 0
        
        survives = abs(portfolio_loss_pct) < self.max_acceptable_loss
        var_breach = abs(portfolio_loss_pct) > 0.10  # 10% VaR breach
        
        return StressTestResult(
            scenario_name=scenario.name,
            portfolio_loss=total_loss,
            portfolio_loss_pct=portfolio_loss_pct,
            worst_position=worst_symbol,
            worst_position_loss=worst_loss,
            var_breach=var_breach,
            survives=survives,
        )
    
    def create_custom_scenario(
        self,
        name: str,
        description: str,
        shocks: dict[str, float],
        volatility_mult: float = 2.0,
    ) -> StressScenario:
        """Create a custom stress scenario.
        
        Args:
            name: Scenario name.
            description: Description.
            shocks: Asset shocks.
            volatility_mult: Volatility multiplier.
            
        Returns:
            StressScenario object.
        """
        scenario = StressScenario(
            name=name,
            description=description,
            shocks=shocks,
            volatility_mult=volatility_mult,
        )
        
        self.scenarios[name.lower().replace(" ", "_")] = scenario
        return scenario
    
    def get_summary_report(
        self,
        results: list[StressTestResult],
    ) -> pd.DataFrame:
        """Generate summary report from results."""
        data = []
        
        for r in results:
            data.append({
                "Scenario": r.scenario_name,
                "P&L": f"${r.portfolio_loss:,.0f}",
                "Loss %": f"{r.portfolio_loss_pct:.2%}",
                "Worst Position": r.worst_position,
                "VaR Breach": "Yes" if r.var_breach else "No",
                "Status": "✓ PASS" if r.survives else "✗ FAIL",
            })
        
        return pd.DataFrame(data)
