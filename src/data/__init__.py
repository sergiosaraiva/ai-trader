"""Data layer module for fetching, processing, and storing market data."""

from .sources import (
    BaseDataSource,
    DataSourceFactory,
    CSVDataSource,
    MT5DataSource,
    AlpacaDataSource,
    YahooDataSource,
)
from .processors import (
    OHLCVProcessor,
    FeatureProcessor,
    TimeframeTransformer,
    TimeframeConfig,
    resample_ohlcv,
)
from .storage import (
    BaseStorage,
    ParquetStorage,
    DatabaseManager,
    CacheManager,
    DataStorageError,
    StorageNotFoundError,
)
from .pipeline import DataPipeline, PipelineConfig, DataQualityReport, load_data
from .loaders import (
    TrainingDataLoader,
    TradingDataset,
    DataLoaderConfig,
    LabelGenerator,
    create_dataloaders,
)

__all__ = [
    # Sources
    "BaseDataSource",
    "DataSourceFactory",
    "CSVDataSource",
    "MT5DataSource",
    "AlpacaDataSource",
    "YahooDataSource",
    # Processors
    "OHLCVProcessor",
    "FeatureProcessor",
    "TimeframeTransformer",
    "TimeframeConfig",
    "resample_ohlcv",
    # Storage
    "BaseStorage",
    "ParquetStorage",
    "DatabaseManager",
    "CacheManager",
    "DataStorageError",
    "StorageNotFoundError",
    # Pipeline
    "DataPipeline",
    "PipelineConfig",
    "DataQualityReport",
    "load_data",
    # Loaders
    "TrainingDataLoader",
    "TradingDataset",
    "DataLoaderConfig",
    "LabelGenerator",
    "create_dataloaders",
]
