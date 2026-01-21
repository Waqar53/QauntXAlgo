"""Data source adapters."""

from quantxalgo.data.sources.base import DataSource
from quantxalgo.data.sources.yahoo import YahooDataSource

__all__ = ["DataSource", "YahooDataSource"]
