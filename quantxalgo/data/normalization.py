"""
Data normalization utilities.

Handles cleaning, validation, and standardization of market data.
"""

from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from quantxalgo.config.logging_config import get_logger
from quantxalgo.core.exceptions import DataValidationError

logger = get_logger(__name__)


class DataNormalizer:
    """Normalize and clean market data.
    
    Ensures all data follows a consistent format:
    - UTC timezone
    - Lowercase column names
    - No missing critical values
    - Consistent data types
    """
    
    REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]
    
    def normalize(
        self,
        df: pd.DataFrame,
        symbol: str,
        fill_missing: bool = True,
        validate: bool = True,
    ) -> pd.DataFrame:
        """Normalize a DataFrame of OHLCV data.
        
        Args:
            df: Raw OHLCV DataFrame.
            symbol: Symbol for logging/error messages.
            fill_missing: Whether to fill missing values.
            validate: Whether to validate data after normalization.
            
        Returns:
            Normalized DataFrame.
            
        Raises:
            DataValidationError: If validation fails.
        """
        if df.empty:
            return df
        
        # Work on a copy
        df = df.copy()
        
        # Lowercase column names
        df.columns = df.columns.str.lower()
        
        # Ensure DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Ensure UTC timezone
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        
        # Sort by index
        df = df.sort_index()
        
        # Remove duplicates
        df = df[~df.index.duplicated(keep="last")]
        
        # Fill missing values if requested
        if fill_missing:
            df = self._fill_missing(df)
        
        # Handle outliers
        df = self._handle_outliers(df)
        
        # Validate
        if validate:
            self._validate(df, symbol)
        
        logger.debug(
            "Normalized data",
            symbol=symbol,
            rows=len(df),
            missing_pct=df.isnull().sum().sum() / df.size * 100,
        )
        
        return df
    
    def _fill_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values using appropriate methods."""
        # Forward fill for OHLC (prices carry forward)
        for col in ["open", "high", "low", "close"]:
            if col in df.columns:
                df[col] = df[col].ffill()
        
        # Zero fill for volume (no volume = 0)
        if "volume" in df.columns:
            df["volume"] = df["volume"].fillna(0)
        
        return df
    
    def _handle_outliers(
        self,
        df: pd.DataFrame,
        zscore_threshold: float = 10.0,
    ) -> pd.DataFrame:
        """Handle outliers in price data.
        
        Uses z-score to identify outliers and replaces with previous value.
        Only applied to OHLC columns (not volume).
        """
        price_cols = ["open", "high", "low", "close"]
        
        for col in price_cols:
            if col not in df.columns:
                continue
            
            series = df[col]
            
            # Calculate z-score
            mean = series.mean()
            std = series.std()
            
            if std > 0:
                zscore = np.abs((series - mean) / std)
                outliers = zscore > zscore_threshold
                
                if outliers.any():
                    logger.warning(
                        "Outliers detected",
                        column=col,
                        count=outliers.sum(),
                    )
                    # Replace with forward fill
                    df.loc[outliers, col] = np.nan
                    df[col] = df[col].ffill()
        
        return df
    
    def _validate(self, df: pd.DataFrame, symbol: str) -> None:
        """Validate normalized data."""
        # Check required columns
        missing_cols = set(self.REQUIRED_COLUMNS) - set(df.columns)
        if missing_cols:
            raise DataValidationError(
                f"Missing required columns for {symbol}: {missing_cols}"
            )
        
        # Check for remaining NaN in critical columns
        for col in ["close"]:
            if df[col].isnull().any():
                raise DataValidationError(
                    f"NaN values remain in {col} for {symbol}"
                )
        
        # Validate price relationships: high >= low
        invalid_hl = df["high"] < df["low"]
        if invalid_hl.any():
            logger.warning(
                "Invalid high/low relationship",
                symbol=symbol,
                count=invalid_hl.sum(),
            )
            # Fix by swapping
            df.loc[invalid_hl, ["high", "low"]] = df.loc[invalid_hl, ["low", "high"]].values
        
        # Check for negative prices
        for col in ["open", "high", "low", "close"]:
            if (df[col] < 0).any():
                raise DataValidationError(
                    f"Negative prices in {col} for {symbol}"
                )
        
        # Check for negative volume
        if (df["volume"] < 0).any():
            df["volume"] = df["volume"].abs()
    
    def calculate_returns(
        self,
        df: pd.DataFrame,
        periods: list[int] = [1, 5, 20],
        log_returns: bool = True,
    ) -> pd.DataFrame:
        """Calculate returns for various periods.
        
        Args:
            df: OHLCV DataFrame with 'close' column.
            periods: List of periods for return calculation.
            log_returns: If True, calculate log returns.
            
        Returns:
            DataFrame with return columns added.
        """
        df = df.copy()
        
        for period in periods:
            col_name = f"return_{period}d"
            
            if log_returns:
                df[col_name] = np.log(df["close"] / df["close"].shift(period))
            else:
                df[col_name] = df["close"].pct_change(period)
        
        return df
    
    def adjust_for_splits(
        self,
        df: pd.DataFrame,
        splits: pd.Series,
    ) -> pd.DataFrame:
        """Adjust OHLCV data for stock splits.
        
        Args:
            df: OHLCV DataFrame.
            splits: Series with split ratios indexed by date.
            
        Returns:
            Split-adjusted DataFrame.
        """
        df = df.copy()
        
        # Calculate cumulative adjustment factor
        adj_factor = pd.Series(1.0, index=df.index)
        
        for date, ratio in splits.items():
            if date in df.index:
                adj_factor.loc[:date] *= ratio
        
        # Apply adjustment
        for col in ["open", "high", "low", "close"]:
            if col in df.columns:
                df[col] = df[col] / adj_factor
        
        if "volume" in df.columns:
            df["volume"] = df["volume"] * adj_factor
        
        return df
