"""
Risk Manager for QuantXalgo.

Central risk management system that enforces limits and protects capital.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np

from quantxalgo.config.logging_config import get_logger
from quantxalgo.core.enums import Side
from quantxalgo.core.events import OrderEvent, RiskEvent
from quantxalgo.core.exceptions import (
    RiskLimitExceededError,
    PositionLimitExceededError,
    ExposureLimitExceededError,
)
from quantxalgo.portfolio.manager import PortfolioManager

logger = get_logger(__name__)


@dataclass
class RiskCheck:
    """Result of a risk check."""
    
    passed: bool
    check_type: str
    message: str
    current_value: float = 0.0
    limit: float = 0.0
    severity: str = "INFO"  # INFO, WARNING, CRITICAL


@dataclass
class RiskLimits:
    """Risk limit configuration."""
    
    max_position_size: float = 0.10       # 10% per position
    max_sector_exposure: float = 0.25     # 25% per sector
    max_total_exposure: float = 1.0       # 100% gross
    max_leverage: float = 2.0             # 2x leverage
    max_drawdown: float = 0.15            # 15% drawdown
    daily_loss_limit: float = 0.03        # 3% daily loss
    max_correlation: float = 0.70         # Max strategy correlation
    target_volatility: float = 0.10       # 10% annual target vol


class RiskManager:
    """Central risk management orchestrator.
    
    Performs pre-trade and portfolio-level risk checks,
    enforces limits, and triggers protective actions.
    
    Example:
        >>> rm = RiskManager(limits=RiskLimits())
        >>> check = rm.check_order(order, portfolio)
        >>> if check.passed:
        ...     execute_order(order)
    """
    
    def __init__(
        self,
        limits: Optional[RiskLimits] = None,
    ) -> None:
        """Initialize risk manager.
        
        Args:
            limits: Risk limit configuration.
        """
        self.limits = limits or RiskLimits()
        
        # Track daily P&L
        self._daily_start_equity: Optional[float] = None
        self._daily_date: Optional[datetime] = None
        
        # Risk events history
        self._risk_events: list[RiskEvent] = []
        
        logger.info(
            "Risk manager initialized",
            max_position=self.limits.max_position_size,
            max_exposure=self.limits.max_total_exposure,
            max_drawdown=self.limits.max_drawdown,
        )
    
    def check_order(
        self,
        order: OrderEvent,
        portfolio: PortfolioManager,
        current_price: float,
    ) -> RiskCheck:
        """Pre-trade risk check for an order.
        
        Args:
            order: Order to check.
            portfolio: Current portfolio state.
            current_price: Current price of the symbol.
            
        Returns:
            RiskCheck result.
        """
        checks = [
            self._check_position_size(order, portfolio, current_price),
            self._check_total_exposure(order, portfolio, current_price),
            self._check_available_capital(order, portfolio, current_price),
        ]
        
        # Find first failed check
        for check in checks:
            if not check.passed:
                logger.warning(
                    "Risk check failed",
                    order_id=order.order_id,
                    symbol=order.symbol,
                    check_type=check.check_type,
                    message=check.message,
                )
                return check
        
        return RiskCheck(
            passed=True,
            check_type="ALL",
            message="All risk checks passed",
        )
    
    def check_portfolio(
        self,
        portfolio: PortfolioManager,
    ) -> list[RiskCheck]:
        """Portfolio-level risk checks.
        
        Args:
            portfolio: Current portfolio state.
            
        Returns:
            List of risk check results.
        """
        checks = [
            self._check_drawdown(portfolio),
            self._check_daily_loss(portfolio),
            self._check_leverage(portfolio),
        ]
        
        return checks
    
    def _check_position_size(
        self,
        order: OrderEvent,
        portfolio: PortfolioManager,
        current_price: float,
    ) -> RiskCheck:
        """Check if order would exceed position size limit."""
        order_value = order.quantity * current_price
        new_position_value = order_value
        
        # Add existing position
        existing = portfolio.get_position(order.symbol)
        if existing:
            if order.side == Side.BUY:
                new_position_value = (existing.quantity + order.quantity) * current_price
            else:
                new_position_value = (existing.quantity - order.quantity) * current_price
        
        position_pct = abs(new_position_value) / portfolio.total_equity
        limit = self.limits.max_position_size
        
        if position_pct > limit:
            return RiskCheck(
                passed=False,
                check_type="POSITION_SIZE",
                message=f"Position would be {position_pct:.1%}, limit is {limit:.1%}",
                current_value=position_pct,
                limit=limit,
                severity="WARNING",
            )
        
        return RiskCheck(
            passed=True,
            check_type="POSITION_SIZE",
            message="Position size within limit",
            current_value=position_pct,
            limit=limit,
        )
    
    def _check_total_exposure(
        self,
        order: OrderEvent,
        portfolio: PortfolioManager,
        current_price: float,
    ) -> RiskCheck:
        """Check if order would exceed total exposure limit."""
        order_value = order.quantity * current_price
        
        if order.side == Side.BUY:
            new_exposure = portfolio.total_exposure + order_value
        else:
            # Selling might reduce exposure if closing
            existing = portfolio.get_position(order.symbol)
            if existing and existing.is_long:
                new_exposure = portfolio.total_exposure - order_value
            else:
                new_exposure = portfolio.total_exposure + order_value
        
        exposure_pct = new_exposure / portfolio.total_equity
        limit = self.limits.max_total_exposure
        
        if exposure_pct > limit:
            return RiskCheck(
                passed=False,
                check_type="TOTAL_EXPOSURE",
                message=f"Exposure would be {exposure_pct:.1%}, limit is {limit:.1%}",
                current_value=exposure_pct,
                limit=limit,
                severity="WARNING",
            )
        
        return RiskCheck(
            passed=True,
            check_type="TOTAL_EXPOSURE",
            message="Exposure within limit",
            current_value=exposure_pct,
            limit=limit,
        )
    
    def _check_available_capital(
        self,
        order: OrderEvent,
        portfolio: PortfolioManager,
        current_price: float,
    ) -> RiskCheck:
        """Check if we have enough cash for the order."""
        if order.side != Side.BUY:
            return RiskCheck(
                passed=True,
                check_type="CAPITAL",
                message="Sell order, no capital check needed",
            )
        
        order_cost = order.quantity * current_price
        
        if order_cost > portfolio.cash:
            return RiskCheck(
                passed=False,
                check_type="CAPITAL",
                message=f"Order cost {order_cost:.2f} exceeds cash {portfolio.cash:.2f}",
                current_value=order_cost,
                limit=portfolio.cash,
                severity="CRITICAL",
            )
        
        return RiskCheck(
            passed=True,
            check_type="CAPITAL",
            message="Sufficient capital available",
            current_value=order_cost,
            limit=portfolio.cash,
        )
    
    def _check_drawdown(
        self,
        portfolio: PortfolioManager,
    ) -> RiskCheck:
        """Check current drawdown against limit."""
        drawdown = abs(portfolio.current_drawdown)
        limit = self.limits.max_drawdown
        
        if drawdown >= limit:
            return RiskCheck(
                passed=False,
                check_type="DRAWDOWN",
                message=f"Drawdown {drawdown:.1%} exceeds limit {limit:.1%}",
                current_value=drawdown,
                limit=limit,
                severity="CRITICAL",
            )
        
        if drawdown >= limit * 0.8:
            return RiskCheck(
                passed=True,
                check_type="DRAWDOWN",
                message=f"Drawdown {drawdown:.1%} approaching limit {limit:.1%}",
                current_value=drawdown,
                limit=limit,
                severity="WARNING",
            )
        
        return RiskCheck(
            passed=True,
            check_type="DRAWDOWN",
            message="Drawdown within limit",
            current_value=drawdown,
            limit=limit,
        )
    
    def _check_daily_loss(
        self,
        portfolio: PortfolioManager,
    ) -> RiskCheck:
        """Check daily loss against limit."""
        today = datetime.utcnow().date()
        
        # Reset daily tracking on new day
        if self._daily_date != today:
            self._daily_date = today
            self._daily_start_equity = portfolio.total_equity
        
        if self._daily_start_equity is None:
            return RiskCheck(passed=True, check_type="DAILY_LOSS", message="First day")
        
        daily_return = (portfolio.total_equity - self._daily_start_equity) / self._daily_start_equity
        limit = self.limits.daily_loss_limit
        
        if daily_return <= -limit:
            return RiskCheck(
                passed=False,
                check_type="DAILY_LOSS",
                message=f"Daily loss {daily_return:.1%} exceeds limit -{limit:.1%}",
                current_value=abs(daily_return),
                limit=limit,
                severity="CRITICAL",
            )
        
        return RiskCheck(
            passed=True,
            check_type="DAILY_LOSS",
            message="Daily P&L within limit",
            current_value=daily_return,
            limit=limit,
        )
    
    def _check_leverage(
        self,
        portfolio: PortfolioManager,
    ) -> RiskCheck:
        """Check leverage against limit."""
        leverage = portfolio.leverage
        limit = self.limits.max_leverage
        
        if leverage > limit:
            return RiskCheck(
                passed=False,
                check_type="LEVERAGE",
                message=f"Leverage {leverage:.1f}x exceeds limit {limit:.1f}x",
                current_value=leverage,
                limit=limit,
                severity="CRITICAL",
            )
        
        return RiskCheck(
            passed=True,
            check_type="LEVERAGE",
            message="Leverage within limit",
            current_value=leverage,
            limit=limit,
        )
    
    def scale_for_volatility(
        self,
        base_size: float,
        current_volatility: float,
    ) -> float:
        """Scale position size based on volatility targeting.
        
        Args:
            base_size: Base position size.
            current_volatility: Current annualized volatility.
            
        Returns:
            Scaled position size.
        """
        if current_volatility <= 0:
            return base_size
        
        target_vol = self.limits.target_volatility
        scale = target_vol / current_volatility
        
        # Cap scaling to prevent extreme positions
        scale = np.clip(scale, 0.25, 2.0)
        
        return base_size * scale
