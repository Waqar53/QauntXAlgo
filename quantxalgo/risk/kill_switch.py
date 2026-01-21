"""
Kill Switch for QuantXalgo.

Emergency risk control that liquidates positions when limits are breached.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional

from quantxalgo.config.logging_config import get_logger
from quantxalgo.portfolio.manager import PortfolioManager

logger = get_logger(__name__)


class KillAction(str, Enum):
    """Action to take when kill switch triggers."""
    NONE = "NONE"
    REDUCE_50 = "REDUCE_50"         # Reduce positions by 50%
    REDUCE_75 = "REDUCE_75"         # Reduce positions by 75%
    LIQUIDATE_ALL = "LIQUIDATE_ALL"  # Full liquidation
    HALT_TRADING = "HALT_TRADING"    # Stop all new orders


@dataclass
class KillSwitchStatus:
    """Kill switch status."""
    
    triggered: bool
    reason: Optional[str] = None
    value: Optional[float] = None
    action: KillAction = KillAction.NONE
    triggered_at: Optional[datetime] = None


class KillSwitch:
    """Emergency risk control system.
    
    Monitors portfolio for extreme conditions and triggers
    protective actions to prevent catastrophic losses.
    
    Example:
        >>> ks = KillSwitch(max_drawdown=0.15, daily_loss_limit=0.03)
        >>> status = ks.check(portfolio)
        >>> if status.triggered:
        ...     liquidate_all()
    """
    
    def __init__(
        self,
        max_drawdown: float = 0.15,
        daily_loss_limit: float = 0.03,
        cooldown_period: timedelta = timedelta(hours=24),
    ) -> None:
        """Initialize kill switch.
        
        Args:
            max_drawdown: Max drawdown before triggering.
            daily_loss_limit: Max daily loss before triggering.
            cooldown_period: Time before allowing trading after trigger.
        """
        self.max_drawdown = max_drawdown
        self.daily_loss_limit = daily_loss_limit
        self.cooldown_period = cooldown_period
        
        self.is_triggered = False
        self.trigger_time: Optional[datetime] = None
        self.trigger_reason: Optional[str] = None
        
        # Daily tracking
        self._daily_start_equity = None
        self._daily_date = None
        
        logger.info(
            "Kill switch initialized",
            max_drawdown=max_drawdown,
            daily_loss_limit=daily_loss_limit,
        )
    
    def check(self, portfolio: PortfolioManager) -> KillSwitchStatus:
        """Check if kill switch should be triggered.
        
        Args:
            portfolio: Current portfolio state.
            
        Returns:
            Kill switch status.
        """
        # Check if in cooldown
        if self.is_triggered and self.trigger_time:
            if datetime.utcnow() - self.trigger_time < self.cooldown_period:
                return KillSwitchStatus(
                    triggered=True,
                    reason="COOLDOWN",
                    action=KillAction.HALT_TRADING,
                    triggered_at=self.trigger_time,
                )
        
        # Reset if cooldown expired
        if self.is_triggered:
            cooldown_expired = (
                self.trigger_time and
                datetime.utcnow() - self.trigger_time >= self.cooldown_period
            )
            if cooldown_expired:
                self.reset()
        
        # Check drawdown
        current_drawdown = abs(portfolio.current_drawdown)
        if current_drawdown >= self.max_drawdown:
            return self._trigger(
                reason="MAX_DRAWDOWN",
                value=current_drawdown,
                action=KillAction.LIQUIDATE_ALL,
            )
        
        # Check daily loss
        daily_loss = self._calculate_daily_loss(portfolio)
        if daily_loss >= self.daily_loss_limit:
            return self._trigger(
                reason="DAILY_LOSS",
                value=daily_loss,
                action=KillAction.REDUCE_75,
            )
        
        # Warning levels
        if current_drawdown >= self.max_drawdown * 0.8:
            return KillSwitchStatus(
                triggered=False,
                reason="DRAWDOWN_WARNING",
                value=current_drawdown,
                action=KillAction.REDUCE_50,
            )
        
        return KillSwitchStatus(triggered=False)
    
    def _trigger(
        self,
        reason: str,
        value: float,
        action: KillAction,
    ) -> KillSwitchStatus:
        """Trigger the kill switch.
        
        Args:
            reason: Reason for triggering.
            value: The value that caused the trigger.
            action: Action to take.
            
        Returns:
            Kill switch status.
        """
        self.is_triggered = True
        self.trigger_time = datetime.utcnow()
        self.trigger_reason = reason
        
        logger.critical(
            "KILL SWITCH TRIGGERED",
            reason=reason,
            value=f"{value:.2%}",
            action=action.value,
        )
        
        return KillSwitchStatus(
            triggered=True,
            reason=reason,
            value=value,
            action=action,
            triggered_at=self.trigger_time,
        )
    
    def _calculate_daily_loss(self, portfolio: PortfolioManager) -> float:
        """Calculate daily loss percentage."""
        today = datetime.utcnow().date()
        
        if self._daily_date != today:
            self._daily_date = today
            self._daily_start_equity = portfolio.total_equity
            return 0.0
        
        if self._daily_start_equity is None or self._daily_start_equity <= 0:
            return 0.0
        
        daily_return = (portfolio.total_equity - self._daily_start_equity) / self._daily_start_equity
        
        return abs(min(0, daily_return))
    
    def reset(self) -> None:
        """Reset the kill switch."""
        self.is_triggered = False
        self.trigger_time = None
        self.trigger_reason = None
        
        logger.info("Kill switch reset")
    
    def force_trigger(self, reason: str = "MANUAL") -> KillSwitchStatus:
        """Force trigger the kill switch manually.
        
        Args:
            reason: Reason for manual trigger.
            
        Returns:
            Kill switch status.
        """
        return self._trigger(
            reason=reason,
            value=0.0,
            action=KillAction.LIQUIDATE_ALL,
        )
