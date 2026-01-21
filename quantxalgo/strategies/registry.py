from __future__ import annotations

"""Strategy Registry for QuantXalgo.

Central registry for strategy discovery, registration, and instantiation.
"""

from typing import Callable, Type

from quantxalgo.config.logging_config import get_logger
from quantxalgo.core.exceptions import StrategyNotFoundError
from quantxalgo.strategies.base import Strategy

logger = get_logger(__name__)


class StrategyRegistry:
    """Central registry for strategy management.
    
    Provides a plugin-like system for registering and discovering strategies.
    
    Example:
        >>> @StrategyRegistry.register("ma_crossover")
        ... class MACrossoverStrategy(Strategy):
        ...     pass
        ...
        >>> strategy = StrategyRegistry.create("ma_crossover", params={"fast": 10})
    """
    
    _strategies: dict[str, Type[Strategy]] = {}
    
    @classmethod
    def register(cls, name: str) -> Callable[[Type[Strategy]], Type[Strategy]]:
        """Decorator to register a strategy class.
        
        Args:
            name: Unique name for the strategy.
            
        Returns:
            Decorator function.
            
        Example:
            >>> @StrategyRegistry.register("momentum")
            ... class MomentumStrategy(Strategy):
            ...     pass
        """
        def decorator(strategy_cls: Type[Strategy]) -> Type[Strategy]:
            if name in cls._strategies:
                logger.warning(
                    "Strategy already registered, overwriting",
                    name=name,
                    old=cls._strategies[name].__name__,
                    new=strategy_cls.__name__,
                )
            
            cls._strategies[name] = strategy_cls
            logger.debug("Strategy registered", name=name, cls=strategy_cls.__name__)
            return strategy_cls
        
        return decorator
    
    @classmethod
    def register_class(cls, name: str, strategy_cls: Type[Strategy]) -> None:
        """Register a strategy class directly (non-decorator).
        
        Args:
            name: Unique name for the strategy.
            strategy_cls: Strategy class to register.
        """
        cls._strategies[name] = strategy_cls
        logger.debug("Strategy registered", name=name, cls=strategy_cls.__name__)
    
    @classmethod
    def create(
        cls,
        name: str,
        params: dict | None = None,
        symbols: list[str] | None = None,
    ) -> Strategy:
        """Create a strategy instance by name.
        
        Args:
            name: Registered name of the strategy.
            params: Strategy parameters.
            symbols: Symbols to trade.
            
        Returns:
            Instantiated strategy.
            
        Raises:
            StrategyNotFoundError: If strategy is not registered.
        """
        if name not in cls._strategies:
            raise StrategyNotFoundError(name)
        
        strategy_cls = cls._strategies[name]
        
        # Instantiate with parameters
        strategy = strategy_cls(
            name=name,
            params=params or {},
            symbols=symbols or [],
        )
        
        logger.info(
            "Strategy created",
            name=name,
            params=params,
            symbols=symbols,
        )
        
        return strategy
    
    @classmethod
    def get(cls, name: str) -> Type[Strategy]:
        """Get strategy class by name.
        
        Args:
            name: Registered name of the strategy.
            
        Returns:
            Strategy class.
            
        Raises:
            StrategyNotFoundError: If strategy is not registered.
        """
        if name not in cls._strategies:
            raise StrategyNotFoundError(name)
        return cls._strategies[name]
    
    @classmethod
    def list_strategies(cls) -> list[str]:
        """List all registered strategy names.
        
        Returns:
            List of strategy names.
        """
        return sorted(cls._strategies.keys())
    
    @classmethod
    def get_info(cls, name: str) -> dict:
        """Get detailed info about a strategy.
        
        Args:
            name: Strategy name.
            
        Returns:
            Dictionary with strategy info.
        """
        if name not in cls._strategies:
            raise StrategyNotFoundError(name)
        
        strategy_cls = cls._strategies[name]
        
        return {
            "name": name,
            "class": strategy_cls.__name__,
            "module": strategy_cls.__module__,
            "doc": strategy_cls.__doc__ or "No documentation",
        }
    
    @classmethod
    def clear(cls) -> None:
        """Clear all registered strategies."""
        cls._strategies.clear()
        logger.debug("Strategy registry cleared")
    
    @classmethod
    def count(cls) -> int:
        """Get count of registered strategies."""
        return len(cls._strategies)
