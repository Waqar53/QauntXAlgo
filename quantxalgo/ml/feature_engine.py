"""
ML Feature Engineering.

Automated feature generation and selection for ML models.
"""

from typing import Optional

import numpy as np
import pandas as pd

from quantxalgo.config.logging_config import get_logger
from quantxalgo.data.feature_store import FeatureStore

logger = get_logger(__name__)


class MLFeatureEngine:
    """Automated ML feature engineering.
    
    Generates features suitable for ML models:
    - Technical indicators (normalized)
    - Price pattern features
    - Volatility features
    - Cross-asset features
    - Lagged targets
    
    Example:
        >>> engine = MLFeatureEngine()
        >>> X, y = engine.prepare_training_data(ohlcv, target_horizon=5)
    """
    
    def __init__(
        self,
        feature_store: Optional[FeatureStore] = None,
    ) -> None:
        """Initialize feature engine.
        
        Args:
            feature_store: Feature store for technical indicators.
        """
        self.feature_store = feature_store or FeatureStore()
    
    def generate_features(
        self,
        ohlcv: pd.DataFrame,
        include_ta: bool = True,
        include_patterns: bool = True,
        include_cross_asset: bool = False,
        other_assets: Optional[dict[str, pd.DataFrame]] = None,
    ) -> pd.DataFrame:
        """Generate all features for ML.
        
        Args:
            ohlcv: OHLCV DataFrame.
            include_ta: Include technical analysis features.
            include_patterns: Include price pattern features.
            include_cross_asset: Include cross-asset features.
            other_assets: Other asset data for cross-asset features.
            
        Returns:
            DataFrame with all features.
        """
        features = pd.DataFrame(index=ohlcv.index)
        
        # Base features
        features = pd.concat([
            features,
            self._generate_price_features(ohlcv),
        ], axis=1)
        
        # Technical indicators
        if include_ta:
            ta_features = self.feature_store.compute_all(ohlcv, include_returns=False)
            # Drop raw OHLCV columns
            ta_features = ta_features.drop(
                columns=["open", "high", "low", "close", "volume"],
                errors="ignore"
            )
            features = pd.concat([features, ta_features], axis=1)
        
        # Pattern features
        if include_patterns:
            features = pd.concat([
                features,
                self._generate_pattern_features(ohlcv),
            ], axis=1)
        
        # Cross-asset features
        if include_cross_asset and other_assets:
            features = pd.concat([
                features,
                self._generate_cross_asset_features(ohlcv, other_assets),
            ], axis=1)
        
        # Normalize features
        features = self._normalize_features(features)
        
        logger.info(
            "Features generated",
            num_features=len(features.columns),
            num_rows=len(features),
        )
        
        return features
    
    def _generate_price_features(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """Generate price-based features."""
        close = ohlcv["close"]
        high = ohlcv["high"]
        low = ohlcv["low"]
        volume = ohlcv["volume"]
        
        features = pd.DataFrame(index=ohlcv.index)
        
        # Returns at various horizons
        for period in [1, 2, 3, 5, 10, 20]:
            features[f"return_{period}d"] = close.pct_change(period)
            features[f"log_return_{period}d"] = np.log(close / close.shift(period))
        
        # Volatility
        features["realized_vol_5"] = close.pct_change().rolling(5).std() * np.sqrt(252)
        features["realized_vol_20"] = close.pct_change().rolling(20).std() * np.sqrt(252)
        features["vol_ratio"] = features["realized_vol_5"] / features["realized_vol_20"]
        
        # Range features
        features["daily_range"] = (high - low) / close
        features["range_ratio"] = features["daily_range"] / features["daily_range"].rolling(20).mean()
        
        # Gap features
        features["gap"] = ohlcv["open"] / close.shift(1) - 1
        
        # Volume features
        features["volume_ma_ratio"] = volume / volume.rolling(20).mean()
        features["volume_change"] = volume.pct_change()
        
        # Price position
        features["close_position"] = (close - low) / (high - low + 1e-10)
        
        return features
    
    def _generate_pattern_features(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """Generate pattern-based features."""
        close = ohlcv["close"]
        high = ohlcv["high"]
        low = ohlcv["low"]
        open_ = ohlcv["open"]
        
        features = pd.DataFrame(index=ohlcv.index)
        
        # Candlestick patterns
        features["body_size"] = abs(close - open_) / (high - low + 1e-10)
        features["upper_shadow"] = (high - np.maximum(close, open_)) / (high - low + 1e-10)
        features["lower_shadow"] = (np.minimum(close, open_) - low) / (high - low + 1e-10)
        features["is_bullish"] = (close > open_).astype(int)
        
        # Consecutive patterns
        features["consecutive_up"] = (close > close.shift(1)).rolling(5).sum()
        features["consecutive_down"] = (close < close.shift(1)).rolling(5).sum()
        
        # Higher highs / lower lows
        features["higher_high"] = (high > high.shift(1)).astype(int)
        features["lower_low"] = (low < low.shift(1)).astype(int)
        
        # Distance from high/low
        rolling_high = high.rolling(20).max()
        rolling_low = low.rolling(20).min()
        features["dist_from_high"] = (rolling_high - close) / rolling_high
        features["dist_from_low"] = (close - rolling_low) / rolling_low
        
        return features
    
    def _generate_cross_asset_features(
        self,
        ohlcv: pd.DataFrame,
        other_assets: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """Generate cross-asset features."""
        close = ohlcv["close"]
        features = pd.DataFrame(index=ohlcv.index)
        
        for asset_name, asset_data in other_assets.items():
            if "close" not in asset_data.columns:
                continue
            
            other_close = asset_data["close"].reindex(ohlcv.index).ffill()
            other_returns = other_close.pct_change()
            
            # Correlation
            features[f"corr_{asset_name}_20"] = close.pct_change().rolling(20).corr(other_returns)
            
            # Beta
            features[f"beta_{asset_name}_20"] = (
                close.pct_change().rolling(20).cov(other_returns) /
                other_returns.rolling(20).var()
            )
            
            # Relative strength
            features[f"rel_strength_{asset_name}"] = (
                close.pct_change(20) - other_close.pct_change(20)
            )
        
        return features
    
    def _normalize_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Normalize features to [-1, 1] range using rolling z-score."""
        normalized = pd.DataFrame(index=features.index)
        
        for col in features.columns:
            series = features[col]
            
            # Rolling normalization (252-day window)
            rolling_mean = series.rolling(252, min_periods=50).mean()
            rolling_std = series.rolling(252, min_periods=50).std()
            
            # Z-score
            zscore = (series - rolling_mean) / (rolling_std + 1e-10)
            
            # Clip to [-3, 3] and scale to [-1, 1]
            normalized[col] = np.clip(zscore / 3, -1, 1)
        
        return normalized
    
    def prepare_training_data(
        self,
        ohlcv: pd.DataFrame,
        target_horizon: int = 5,
        target_type: str = "return",
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target for training.
        
        Args:
            ohlcv: OHLCV DataFrame.
            target_horizon: Forward return horizon.
            target_type: 'return' or 'direction'.
            
        Returns:
            Tuple of (X features, y target).
        """
        features = self.generate_features(ohlcv)
        
        # Generate target
        close = ohlcv["close"]
        forward_return = close.shift(-target_horizon) / close - 1
        
        if target_type == "direction":
            target = (forward_return > 0).astype(int)
        else:
            target = forward_return
        
        # Align features and target, drop NaN
        combined = pd.concat([features, target.rename("target")], axis=1).dropna()
        
        X = combined.drop(columns=["target"])
        y = combined["target"]
        
        logger.info(
            "Training data prepared",
            features=len(X.columns),
            samples=len(X),
            target_horizon=target_horizon,
        )
        
        return X, y
    
    def select_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        method: str = "importance",
        n_top: int = 30,
    ) -> list[str]:
        """Select top features based on importance.
        
        Args:
            X: Feature DataFrame.
            y: Target series.
            method: Selection method ('importance', 'correlation').
            n_top: Number of features to select.
            
        Returns:
            List of selected feature names.
        """
        if method == "correlation":
            # Simple correlation-based selection
            correlations = X.apply(lambda col: col.corr(y)).abs()
            selected = correlations.nlargest(n_top).index.tolist()
        else:
            # Use all features (placeholder for tree importance)
            selected = list(X.columns)[:n_top]
        
        logger.info(
            "Features selected",
            method=method,
            selected=len(selected),
        )
        
        return selected
