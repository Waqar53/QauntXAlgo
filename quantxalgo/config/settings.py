"""
Application settings using Pydantic v2.

Settings are loaded from environment variables with support for .env files.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal, Union

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    """Database configuration."""
    
    model_config = SettingsConfigDict(env_prefix="DATABASE_")
    
    url: str = "postgresql+asyncpg://quantxalgo:quantxalgo@localhost:5432/quantxalgo"
    pool_size: int = 10
    max_overflow: int = 20
    echo: bool = False


class RedisSettings(BaseSettings):
    """Redis cache configuration."""
    
    model_config = SettingsConfigDict(env_prefix="REDIS_")
    
    url: str = "redis://localhost:6379/0"
    cache_ttl: int = 3600  # 1 hour default


class DataProviderSettings(BaseSettings):
    """Data provider API configuration."""
    
    model_config = SettingsConfigDict(env_prefix="")
    
    # Yahoo Finance (free, no key required)
    yahoo_enabled: bool = True
    
    # Alpha Vantage
    alpha_vantage_api_key: str = ""
    alpha_vantage_enabled: bool = False
    
    # Polygon.io
    polygon_api_key: str = ""
    polygon_enabled: bool = False


class TradingSettings(BaseSettings):
    """Trading simulation configuration."""
    
    model_config = SettingsConfigDict(env_prefix="")
    
    initial_capital: float = 1_000_000.0
    base_currency: str = "USD"
    default_slippage_bps: float = 5.0
    default_commission_per_share: float = 0.005


class RiskSettings(BaseSettings):
    """Risk management configuration."""
    
    model_config = SettingsConfigDict(env_prefix="")
    
    max_position_size_pct: float = 10.0
    max_total_exposure_pct: float = 100.0
    max_drawdown_pct: float = 15.0
    target_volatility_pct: float = 10.0
    max_correlation: float = 0.70
    daily_loss_limit_pct: float = 3.0


class Settings(BaseSettings):
    """Main application settings."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    # Environment
    quantxalgo_env: Literal["development", "staging", "production"] = "development"
    debug: bool = True
    log_level: str = "INFO"
    
    # API Server
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4
    
    # Paths
    data_dir: Path = Path("./data")
    logs_dir: Path = Path("./logs")
    cache_dir: Path = Path("./data/cache")
    
    # Secrets
    secret_key: str = "change-this-in-production"
    jwt_secret: str = "change-this-in-production"
    
    # Nested settings
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    data_providers: DataProviderSettings = Field(default_factory=DataProviderSettings)
    trading: TradingSettings = Field(default_factory=TradingSettings)
    risk: RiskSettings = Field(default_factory=RiskSettings)
    
    @field_validator("data_dir", "logs_dir", "cache_dir", mode="before")
    @classmethod
    def ensure_path(cls, v: Union[str, Path]) -> Path:
        return Path(v)
    
    @property
    def is_production(self) -> bool:
        return self.quantxalgo_env == "production"
    
    @property
    def is_development(self) -> bool:
        return self.quantxalgo_env == "development"
    
    def ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "raw").mkdir(exist_ok=True)
        (self.data_dir / "processed").mkdir(exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
