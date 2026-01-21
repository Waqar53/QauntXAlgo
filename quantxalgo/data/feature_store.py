"""
Feature Store for QuantXalgo.

Centralized feature engineering and storage for technical indicators,
statistical features, and derived data used by strategies.
"""

from typing import Callable, Optional

import numpy as np
import pandas as pd

from quantxalgo.config.logging_config import get_logger

logger = get_logger(__name__)


class FeatureStore:
    """Centralized feature engineering and caching.
    
    Computes and caches technical indicators and derived features
    used by trading strategies.
    
    Example:
        >>> fs = FeatureStore()
        >>> features = fs.compute_all(df)
        >>> rsi = fs.rsi(df["close"], period=14)
    """
    
    def __init__(self, cache_features: bool = True) -> None:
        """Initialize feature store.
        
        Args:
            cache_features: Whether to cache computed features.
        """
        self.cache_features = cache_features
        self._cache: dict[str, pd.DataFrame] = {}
    
    # =========================================================================
    # RETURNS & VOLATILITY
    # =========================================================================
    
    def returns(
        self,
        prices: pd.Series,
        periods: list[int] = [1, 5, 20],
        log: bool = True,
    ) -> pd.DataFrame:
        """Calculate returns for multiple periods.
        
        Args:
            prices: Price series.
            periods: List of periods.
            log: Use log returns if True.
            
        Returns:
            DataFrame with return columns.
        """
        result = pd.DataFrame(index=prices.index)
        
        for period in periods:
            if log:
                result[f"return_{period}"] = np.log(prices / prices.shift(period))
            else:
                result[f"return_{period}"] = prices.pct_change(period)
        
        return result
    
    def volatility(
        self,
        returns: pd.Series,
        window: int = 20,
        annualize: bool = True,
    ) -> pd.Series:
        """Calculate rolling volatility.
        
        Args:
            returns: Return series.
            window: Rolling window size.
            annualize: If True, annualize volatility (assuming daily data).
            
        Returns:
            Volatility series.
        """
        vol = returns.rolling(window=window).std()
        
        if annualize:
            vol = vol * np.sqrt(252)
        
        return vol
    
    def realized_volatility(
        self,
        ohlc: pd.DataFrame,
        window: int = 20,
    ) -> pd.Series:
        """Calculate realized volatility using Parkinson estimator.
        
        More efficient than close-to-close volatility.
        """
        hl_ratio = np.log(ohlc["high"] / ohlc["low"])
        factor = 1 / (4 * np.log(2))
        
        return np.sqrt(factor * (hl_ratio ** 2).rolling(window=window).mean()) * np.sqrt(252)
    
    # =========================================================================
    # MOVING AVERAGES
    # =========================================================================
    
    def sma(self, series: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average."""
        return series.rolling(window=period).mean()
    
    def ema(self, series: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average."""
        return series.ewm(span=period, adjust=False).mean()
    
    def wma(self, series: pd.Series, period: int) -> pd.Series:
        """Weighted Moving Average."""
        weights = np.arange(1, period + 1)
        return series.rolling(window=period).apply(
            lambda x: np.dot(x, weights) / weights.sum(),
            raw=True,
        )
    
    def dema(self, series: pd.Series, period: int) -> pd.Series:
        """Double Exponential Moving Average."""
        ema1 = self.ema(series, period)
        ema2 = self.ema(ema1, period)
        return 2 * ema1 - ema2
    
    # =========================================================================
    # MOMENTUM INDICATORS
    # =========================================================================
    
    def rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index.
        
        Args:
            prices: Price series.
            period: RSI period.
            
        Returns:
            RSI values (0-100).
        """
        delta = prices.diff()
        
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        
        avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
        avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def macd(
        self,
        prices: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> pd.DataFrame:
        """MACD (Moving Average Convergence Divergence).
        
        Returns:
            DataFrame with columns: macd, signal, histogram.
        """
        fast_ema = self.ema(prices, fast)
        slow_ema = self.ema(prices, slow)
        
        macd_line = fast_ema - slow_ema
        signal_line = self.ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return pd.DataFrame({
            "macd": macd_line,
            "macd_signal": signal_line,
            "macd_hist": histogram,
        })
    
    def roc(self, prices: pd.Series, period: int = 10) -> pd.Series:
        """Rate of Change."""
        return (prices - prices.shift(period)) / prices.shift(period) * 100
    
    def momentum(self, prices: pd.Series, period: int = 10) -> pd.Series:
        """Momentum (price difference)."""
        return prices - prices.shift(period)
    
    def stochastic(
        self,
        ohlc: pd.DataFrame,
        k_period: int = 14,
        d_period: int = 3,
    ) -> pd.DataFrame:
        """Stochastic Oscillator.
        
        Returns:
            DataFrame with %K and %D columns.
        """
        low_min = ohlc["low"].rolling(window=k_period).min()
        high_max = ohlc["high"].rolling(window=k_period).max()
        
        k = 100 * (ohlc["close"] - low_min) / (high_max - low_min)
        d = k.rolling(window=d_period).mean()
        
        return pd.DataFrame({
            "stoch_k": k,
            "stoch_d": d,
        })
    
    # =========================================================================
    # VOLATILITY INDICATORS
    # =========================================================================
    
    def bollinger_bands(
        self,
        prices: pd.Series,
        period: int = 20,
        std_dev: float = 2.0,
    ) -> pd.DataFrame:
        """Bollinger Bands.
        
        Returns:
            DataFrame with upper, middle, lower, bandwidth, %b columns.
        """
        middle = self.sma(prices, period)
        std = prices.rolling(window=period).std()
        
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        bandwidth = (upper - lower) / middle
        pct_b = (prices - lower) / (upper - lower)
        
        return pd.DataFrame({
            "bb_upper": upper,
            "bb_middle": middle,
            "bb_lower": lower,
            "bb_bandwidth": bandwidth,
            "bb_pct_b": pct_b,
        })
    
    def atr(self, ohlc: pd.DataFrame, period: int = 14) -> pd.Series:
        """Average True Range."""
        high = ohlc["high"]
        low = ohlc["low"]
        close = ohlc["close"]
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        return tr.ewm(span=period, adjust=False).mean()
    
    def keltner_channels(
        self,
        ohlc: pd.DataFrame,
        ema_period: int = 20,
        atr_period: int = 10,
        atr_mult: float = 2.0,
    ) -> pd.DataFrame:
        """Keltner Channels."""
        middle = self.ema(ohlc["close"], ema_period)
        atr_val = self.atr(ohlc, atr_period)
        
        upper = middle + (atr_mult * atr_val)
        lower = middle - (atr_mult * atr_val)
        
        return pd.DataFrame({
            "kc_upper": upper,
            "kc_middle": middle,
            "kc_lower": lower,
        })
    
    # =========================================================================
    # VOLUME INDICATORS
    # =========================================================================
    
    def vwap(self, ohlc: pd.DataFrame) -> pd.Series:
        """Volume Weighted Average Price (cumulative for the day)."""
        typical_price = (ohlc["high"] + ohlc["low"] + ohlc["close"]) / 3
        return (typical_price * ohlc["volume"]).cumsum() / ohlc["volume"].cumsum()
    
    def obv(self, ohlc: pd.DataFrame) -> pd.Series:
        """On-Balance Volume."""
        direction = np.sign(ohlc["close"].diff())
        return (direction * ohlc["volume"]).cumsum()
    
    def volume_ratio(
        self,
        volume: pd.Series,
        period: int = 20,
    ) -> pd.Series:
        """Volume ratio vs average."""
        avg_volume = volume.rolling(window=period).mean()
        return volume / avg_volume
    
    # =========================================================================
    # TREND INDICATORS
    # =========================================================================
    
    def adx(self, ohlc: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Average Directional Index.
        
        Returns:
            DataFrame with ADX, +DI, -DI columns.
        """
        high = ohlc["high"]
        low = ohlc["low"]
        close = ohlc["close"]
        
        # True Range
        tr = pd.concat([
            high - low,
            abs(high - close.shift()),
            abs(low - close.shift()),
        ], axis=1).max(axis=1)
        
        # Directional Movement
        up_move = high.diff()
        down_move = -low.diff()
        
        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)
        
        # Smoothed values
        atr = tr.ewm(span=period, adjust=False).mean()
        plus_di = 100 * plus_dm.ewm(span=period, adjust=False).mean() / atr
        minus_di = 100 * minus_dm.ewm(span=period, adjust=False).mean() / atr
        
        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.ewm(span=period, adjust=False).mean()
        
        return pd.DataFrame({
            "adx": adx,
            "plus_di": plus_di,
            "minus_di": minus_di,
        })
    
    # =========================================================================
    # COMPOSITE FEATURES
    # =========================================================================
    
    def compute_all(
        self,
        ohlcv: pd.DataFrame,
        include_returns: bool = True,
    ) -> pd.DataFrame:
        """Compute all standard features.
        
        Args:
            ohlcv: OHLCV DataFrame.
            include_returns: Whether to include return features.
            
        Returns:
            DataFrame with all features.
        """
        features = ohlcv.copy()
        
        # Returns
        if include_returns:
            returns_df = self.returns(ohlcv["close"], periods=[1, 5, 20, 60])
            features = pd.concat([features, returns_df], axis=1)
        
        # Volatility
        features["volatility_20"] = self.volatility(
            features.get("return_1", ohlcv["close"].pct_change()),
            window=20,
        )
        
        # Moving averages
        for period in [5, 10, 20, 50, 200]:
            features[f"sma_{period}"] = self.sma(ohlcv["close"], period)
            features[f"ema_{period}"] = self.ema(ohlcv["close"], period)
        
        # Momentum
        features["rsi_14"] = self.rsi(ohlcv["close"], 14)
        
        macd_df = self.macd(ohlcv["close"])
        features = pd.concat([features, macd_df], axis=1)
        
        stoch_df = self.stochastic(ohlcv)
        features = pd.concat([features, stoch_df], axis=1)
        
        # Volatility indicators
        bb_df = self.bollinger_bands(ohlcv["close"])
        features = pd.concat([features, bb_df], axis=1)
        
        features["atr_14"] = self.atr(ohlcv, 14)
        
        # Volume
        features["volume_ratio"] = self.volume_ratio(ohlcv["volume"])
        features["obv"] = self.obv(ohlcv)
        
        # Trend
        adx_df = self.adx(ohlcv)
        features = pd.concat([features, adx_df], axis=1)
        
        logger.debug(
            "Computed all features",
            feature_count=len(features.columns),
            rows=len(features),
        )
        
        return features
    
    def clear_cache(self) -> None:
        """Clear the feature cache."""
        self._cache.clear()
        logger.debug("Feature cache cleared")
