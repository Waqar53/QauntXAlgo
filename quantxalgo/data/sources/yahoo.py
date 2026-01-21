"""
Yahoo Finance data source adapter.

Uses yfinance library to fetch free historical data from Yahoo Finance.
"""

import asyncio
from datetime import datetime
from typing import Optional

import pandas as pd
import yfinance as yf

from quantxalgo.config.logging_config import get_logger
from quantxalgo.core.enums import AssetClass, Interval
from quantxalgo.core.exceptions import DataFetchError, MissingDataError
from quantxalgo.data.sources.base import DataSource

logger = get_logger(__name__)


class YahooDataSource(DataSource):
    """Yahoo Finance data source adapter.
    
    Provides free historical OHLCV data for equities, ETFs, and indices.
    
    Note: Yahoo Finance has rate limits and may occasionally return
    incomplete data. Use caching for production workloads.
    
    Example:
        >>> async with YahooDataSource() as yahoo:
        ...     df = await yahoo.fetch_ohlcv("AAPL", start, end)
    """
    
    # Mapping from our intervals to yfinance intervals
    INTERVAL_MAP = {
        Interval.MINUTE_1: "1m",
        Interval.MINUTE_5: "5m",
        Interval.MINUTE_15: "15m",
        Interval.MINUTE_30: "30m",
        Interval.HOUR_1: "1h",
        Interval.DAY_1: "1d",
        Interval.WEEK_1: "1wk",
        Interval.MONTH_1: "1mo",
    }
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0) -> None:
        """Initialize Yahoo Finance adapter.
        
        Args:
            max_retries: Maximum retry attempts for failed requests.
            retry_delay: Delay between retries in seconds.
        """
        super().__init__(name="Yahoo Finance")
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._session = None
    
    async def connect(self) -> None:
        """Establish connection (no-op for yfinance)."""
        self._is_connected = True
        logger.info("Yahoo Finance data source connected")
    
    async def disconnect(self) -> None:
        """Close connection (no-op for yfinance)."""
        self._is_connected = False
        logger.info("Yahoo Finance data source disconnected")
    
    async def fetch_ohlcv(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: Interval = Interval.DAY_1,
    ) -> pd.DataFrame:
        """Fetch OHLCV data from Yahoo Finance.
        
        Args:
            symbol: Ticker symbol (e.g., "AAPL", "SPY").
            start: Start date for data.
            end: End date for data.
            interval: Data interval/timeframe.
            
        Returns:
            DataFrame with columns: open, high, low, close, volume, adj_close.
            
        Raises:
            DataFetchError: If fetch fails after retries.
            MissingDataError: If no data returned for period.
        """
        yf_interval = self.INTERVAL_MAP.get(interval, "1d")
        
        for attempt in range(self.max_retries):
            try:
                # Run yfinance in thread pool (it's synchronous)
                df = await asyncio.get_event_loop().run_in_executor(
                    None,
                    self._fetch_sync,
                    symbol,
                    start,
                    end,
                    yf_interval,
                )
                
                if df.empty:
                    raise MissingDataError(
                        symbol=symbol,
                        start_date=start.strftime("%Y-%m-%d"),
                        end_date=end.strftime("%Y-%m-%d"),
                    )
                
                # Normalize column names
                df = self._normalize_columns(df)
                
                logger.debug(
                    "Fetched data from Yahoo",
                    symbol=symbol,
                    rows=len(df),
                    start=df.index[0].isoformat(),
                    end=df.index[-1].isoformat(),
                )
                
                return df
                
            except MissingDataError:
                raise
            except Exception as e:
                logger.warning(
                    "Yahoo fetch attempt failed",
                    symbol=symbol,
                    attempt=attempt + 1,
                    error=str(e),
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise DataFetchError(
                        source=self.name,
                        symbol=symbol,
                        message=str(e),
                    )
    
    def _fetch_sync(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str,
    ) -> pd.DataFrame:
        """Synchronous fetch (called in executor)."""
        ticker = yf.Ticker(symbol)
        df = ticker.history(
            start=start,
            end=end,
            interval=interval,
            auto_adjust=True,  # Adjust for splits/dividends
            actions=False,     # Don't include dividends/splits columns
        )
        return df
    
    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names to lowercase."""
        df.columns = df.columns.str.lower()
        
        # Ensure we have required columns
        required = ["open", "high", "low", "close", "volume"]
        for col in required:
            if col not in df.columns:
                raise DataFetchError(
                    source=self.name,
                    symbol="unknown",
                    message=f"Missing required column: {col}",
                )
        
        # Ensure UTC timezone
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        
        return df
    
    async def fetch_multiple(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
        interval: Interval = Interval.DAY_1,
    ) -> dict[str, pd.DataFrame]:
        """Fetch OHLCV data for multiple symbols concurrently.
        
        Args:
            symbols: List of ticker symbols.
            start: Start date for data.
            end: End date for data.
            interval: Data interval/timeframe.
            
        Returns:
            Dictionary mapping symbol to DataFrame.
        """
        tasks = [
            self.fetch_ohlcv(symbol, start, end, interval)
            for symbol in symbols
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        data = {}
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                logger.error(
                    "Failed to fetch symbol",
                    symbol=symbol,
                    error=str(result),
                )
            else:
                data[symbol] = result
        
        return data
    
    async def get_symbols(
        self,
        asset_class: AssetClass = AssetClass.EQUITY
    ) -> list[str]:
        """Get available symbols.
        
        Note: Yahoo Finance doesn't provide a symbols list API.
        This returns a curated list of popular symbols.
        """
        # Popular symbols by asset class
        symbols_by_class = {
            AssetClass.EQUITY: [
                "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
                "JPM", "V", "JNJ", "WMT", "PG", "UNH", "HD", "MA",  
            ],
            AssetClass.ETF: [
                "SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "VEA", "VWO",
                "EFA", "EEM", "GLD", "SLV", "TLT", "HYG", "XLF", "XLK",
            ],
            AssetClass.CRYPTO: [
                "BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD",
                "ADA-USD", "DOGE-USD", "DOT-USD", "AVAX-USD", "MATIC-USD",
            ],
        }
        
        return symbols_by_class.get(asset_class, [])
    
    def validate_symbol(self, symbol: str) -> bool:
        """Validate if a symbol exists on Yahoo Finance."""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return info.get("regularMarketPrice") is not None
        except Exception:
            return False
