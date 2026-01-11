"""Database manager for persistent storage."""

from datetime import datetime
from typing import Optional, List, Dict, Any
import pandas as pd


class DatabaseManager:
    """Manage database connections and operations."""

    def __init__(self, connection_string: Optional[str] = None):
        """
        Initialize database manager.

        Args:
            connection_string: Database connection string
        """
        self.connection_string = connection_string
        self._engine = None
        self._connected = False

    def connect(self) -> bool:
        """Establish database connection."""
        try:
            from sqlalchemy import create_engine

            if not self.connection_string:
                raise ValueError("Connection string required")

            self._engine = create_engine(self.connection_string)
            self._connected = True
            return True

        except ImportError:
            raise ImportError("sqlalchemy not installed. Install with: pip install sqlalchemy")

    def disconnect(self) -> None:
        """Close database connection."""
        if self._engine:
            self._engine.dispose()
            self._connected = False

    def save_ohlcv(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        table_name: str = "ohlcv_data",
    ) -> int:
        """
        Save OHLCV data to database.

        Args:
            df: OHLCV dataframe
            symbol: Trading symbol
            timeframe: Data timeframe
            table_name: Target table name

        Returns:
            Number of rows inserted
        """
        if not self._connected:
            raise ConnectionError("Not connected to database")

        df = df.copy()
        df["symbol"] = symbol
        df["timeframe"] = timeframe
        df["timestamp"] = df.index

        rows = df.to_sql(
            table_name,
            self._engine,
            if_exists="append",
            index=False,
        )

        return rows or len(df)

    def load_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        table_name: str = "ohlcv_data",
    ) -> pd.DataFrame:
        """
        Load OHLCV data from database.

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            start_date: Start date filter
            end_date: End date filter
            table_name: Source table name

        Returns:
            OHLCV dataframe
        """
        if not self._connected:
            raise ConnectionError("Not connected to database")

        query = f"""
            SELECT timestamp, open, high, low, close, volume
            FROM {table_name}
            WHERE symbol = :symbol AND timeframe = :timeframe
        """
        params: Dict[str, Any] = {"symbol": symbol, "timeframe": timeframe}

        if start_date:
            query += " AND timestamp >= :start_date"
            params["start_date"] = start_date
        if end_date:
            query += " AND timestamp <= :end_date"
            params["end_date"] = end_date

        query += " ORDER BY timestamp"

        df = pd.read_sql(query, self._engine, params=params)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)

        return df

    def save_predictions(
        self,
        predictions: List[Dict],
        table_name: str = "predictions",
    ) -> int:
        """Save model predictions to database."""
        if not self._connected:
            raise ConnectionError("Not connected to database")

        df = pd.DataFrame(predictions)
        rows = df.to_sql(table_name, self._engine, if_exists="append", index=False)
        return rows or len(df)

    def save_trades(
        self,
        trades: List[Dict],
        table_name: str = "trades",
    ) -> int:
        """Save trade records to database."""
        if not self._connected:
            raise ConnectionError("Not connected to database")

        df = pd.DataFrame(trades)
        rows = df.to_sql(table_name, self._engine, if_exists="append", index=False)
        return rows or len(df)

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
