"""
Abstract base class for all data sources.

All data source adapters must implement this interface.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

import pandas as pd

from quantxalgo.core.enums import AssetClass, Interval


class DataSource(ABC):
    """Abstract base class for data source adapters.
    
    All data sources (Yahoo, Alpha Vantage, Polygon, etc.) must implement
    this interface to ensure consistent data access patterns.
    """
    
    def __init__(self, name: str) -> None:
        """Initialize data source.
        
        Args:
            name: Human-readable name of the data source.
        """
        self.name = name
        self._is_connected = False
    
    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the data source."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to the data source."""
        pass
    
    @abstractmethod
    async def fetch_ohlcv(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: Interval = Interval.DAY_1,
    ) -> pd.DataFrame:
        """Fetch OHLCV data for a symbol.
        
        Args:
            symbol: Ticker symbol (e.g., "AAPL", "SPY").
            start: Start date for data.
            end: End date for data.
            interval: Data interval/timeframe.
            
        Returns:
            DataFrame with columns: open, high, low, close, volume.
            Index should be DatetimeIndex in UTC.
            
        Raises:
            DataFetchError: If data fetch fails.
            MissingDataError: If no data available for the period.
        """
        pass
    
    @abstractmethod
    async def fetch_multiple(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
        interval: Interval = Interval.DAY_1,
    ) -> dict[str, pd.DataFrame]:
        """Fetch OHLCV data for multiple symbols.
        
        Args:
            symbols: List of ticker symbols.
            start: Start date for data.
            end: End date for data.
            interval: Data interval/timeframe.
            
        Returns:
            Dictionary mapping symbol to DataFrame.
        """
        pass
    
    @abstractmethod
    async def get_symbols(
        self,
        asset_class: AssetClass = AssetClass.EQUITY
    ) -> list[str]:
        """Get available symbols for an asset class.
        
        Args:
            asset_class: Type of asset to get symbols for.
            
        Returns:
            List of available symbols.
        """
        pass
    
    @abstractmethod
    def validate_symbol(self, symbol: str) -> bool:
        """Validate if a symbol is available.
        
        Args:
            symbol: Symbol to validate.
            
        Returns:
            True if symbol is valid and available.
        """
        pass
    
    async def health_check(self) -> bool:
        """Check if data source is healthy and accessible.
        
        Returns:
            True if source is healthy.
        """
        return self._is_connected
    
    @property
    def is_connected(self) -> bool:
        """Check if source is connected."""
        return self._is_connected
    
    async def __aenter__(self) -> "DataSource":
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.disconnect()
