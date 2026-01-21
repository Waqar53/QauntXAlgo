"""
ML-based Alpha Model.

Generates alpha signals using machine learning.
"""

from typing import Optional

import numpy as np
import pandas as pd

from quantxalgo.config.logging_config import get_logger
from quantxalgo.ml.feature_engine import MLFeatureEngine

logger = get_logger(__name__)


class AlphaModel:
    """ML-based alpha signal generation.
    
    Trains a model to predict forward returns and generates
    alpha signals based on predictions.
    
    Supports multiple model types:
    - Linear (Ridge regression)
    - Tree-based (LightGBM, XGBoost)
    - Ensemble
    
    Example:
        >>> model = AlphaModel(model_type="lightgbm")
        >>> model.fit(X_train, y_train)
        >>> alphas = model.predict_alpha(X_test)
    """
    
    def __init__(
        self,
        model_type: str = "ridge",
        target_horizon: int = 5,
        quantile_threshold: float = 0.2,
    ) -> None:
        """Initialize alpha model.
        
        Args:
            model_type: Model type ('ridge', 'lightgbm', 'xgboost').
            target_horizon: Forward return horizon for prediction.
            quantile_threshold: Top/bottom quantile for signals.
        """
        self.model_type = model_type
        self.target_horizon = target_horizon
        self.quantile_threshold = quantile_threshold
        
        self.model = None
        self.feature_engine = MLFeatureEngine()
        self.feature_names: list[str] = []
        self._is_fitted = False
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_split: float = 0.2,
    ) -> dict:
        """Train the alpha model.
        
        Args:
            X: Feature DataFrame.
            y: Target series.
            validation_split: Fraction for validation.
            
        Returns:
            Training metrics dictionary.
        """
        self.feature_names = list(X.columns)
        
        # Split for validation
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Train model based on type
        if self.model_type == "ridge":
            metrics = self._fit_ridge(X_train, y_train, X_val, y_val)
        elif self.model_type == "lightgbm":
            metrics = self._fit_lightgbm(X_train, y_train, X_val, y_val)
        else:
            metrics = self._fit_ridge(X_train, y_train, X_val, y_val)
        
        self._is_fitted = True
        
        logger.info(
            "Alpha model trained",
            model_type=self.model_type,
            train_r2=f"{metrics.get('train_r2', 0):.4f}",
            val_r2=f"{metrics.get('val_r2', 0):.4f}",
            features=len(self.feature_names),
        )
        
        return metrics
    
    def _fit_ridge(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> dict:
        """Fit Ridge regression model."""
        from sklearn.linear_model import Ridge
        from sklearn.metrics import r2_score
        
        self.model = Ridge(alpha=1.0)
        self.model.fit(X_train.fillna(0), y_train)
        
        train_pred = self.model.predict(X_train.fillna(0))
        val_pred = self.model.predict(X_val.fillna(0))
        
        return {
            "train_r2": r2_score(y_train, train_pred),
            "val_r2": r2_score(y_val, val_pred),
            "train_ic": np.corrcoef(y_train, train_pred)[0, 1],
            "val_ic": np.corrcoef(y_val, val_pred)[0, 1],
        }
    
    def _fit_lightgbm(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> dict:
        """Fit LightGBM model."""
        try:
            import lightgbm as lgb
            from sklearn.metrics import r2_score
            
            train_data = lgb.Dataset(X_train.fillna(0), label=y_train)
            val_data = lgb.Dataset(X_val.fillna(0), label=y_val, reference=train_data)
            
            params = {
                "objective": "regression",
                "metric": "rmse",
                "learning_rate": 0.05,
                "num_leaves": 31,
                "max_depth": 6,
                "min_data_in_leaf": 100,
                "feature_fraction": 0.8,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                "verbose": -1,
            }
            
            self.model = lgb.train(
                params,
                train_data,
                num_boost_round=200,
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)],
            )
            
            train_pred = self.model.predict(X_train.fillna(0))
            val_pred = self.model.predict(X_val.fillna(0))
            
            return {
                "train_r2": r2_score(y_train, train_pred),
                "val_r2": r2_score(y_val, val_pred),
                "train_ic": np.corrcoef(y_train, train_pred)[0, 1],
                "val_ic": np.corrcoef(y_val, val_pred)[0, 1],
            }
        except ImportError:
            logger.warning("LightGBM not installed, falling back to Ridge")
            return self._fit_ridge(X_train, y_train, X_val, y_val)
    
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Predict raw alpha scores.
        
        Args:
            X: Feature DataFrame.
            
        Returns:
            Predicted alpha scores.
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted")
        
        # Ensure same features
        X = X[self.feature_names].fillna(0)
        
        if hasattr(self.model, "predict"):
            predictions = self.model.predict(X)
        else:
            predictions = np.zeros(len(X))
        
        return pd.Series(predictions, index=X.index)
    
    def predict_alpha(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate alpha signals with confidence.
        
        Args:
            X: Feature DataFrame.
            
        Returns:
            DataFrame with alpha and confidence columns.
        """
        raw_predictions = self.predict(X)
        
        # Convert to z-scores for ranking
        mean = raw_predictions.rolling(252, min_periods=50).mean()
        std = raw_predictions.rolling(252, min_periods=50).std()
        z_scores = (raw_predictions - mean) / (std + 1e-10)
        
        # Generate signals
        alpha = pd.DataFrame(index=X.index)
        alpha["raw_prediction"] = raw_predictions
        alpha["z_score"] = z_scores
        
        # Long signal: top quantile, Short signal: bottom quantile
        alpha["signal"] = 0
        alpha.loc[z_scores > z_scores.quantile(1 - self.quantile_threshold), "signal"] = 1
        alpha.loc[z_scores < z_scores.quantile(self.quantile_threshold), "signal"] = -1
        
        # Confidence based on z-score magnitude
        alpha["confidence"] = np.clip(np.abs(z_scores) / 3, 0, 1)
        
        return alpha
    
    def get_feature_importance(self) -> pd.Series:
        """Get feature importance from model."""
        if not self._is_fitted:
            return pd.Series()
        
        if hasattr(self.model, "feature_importance"):
            # LightGBM
            importance = self.model.feature_importance()
        elif hasattr(self.model, "coef_"):
            # Linear model
            importance = np.abs(self.model.coef_)
        else:
            return pd.Series()
        
        return pd.Series(importance, index=self.feature_names).sort_values(ascending=False)
    
    @property
    def is_fitted(self) -> bool:
        return self._is_fitted
