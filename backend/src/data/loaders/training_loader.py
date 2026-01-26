"""Training Data Loader for PyTorch-based model training.

This module provides time-series aware data loading with:
- Proper chronological train/val/test splits (no data leakage)
- Sequence creation for LSTM/Transformer models
- Multiple labeling strategies (direction, returns, triple barrier)
- Normalization with saved scalers
- PyTorch DataLoader integration
"""

import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


class LabelMethod(Enum):
    """Methods for generating training labels."""

    DIRECTION = "direction"  # Binary: 1 if price up, 0 if down
    DIRECTION_THREE = "direction_three"  # Ternary: -1, 0, 1 (down, neutral, up)
    RETURNS = "returns"  # Continuous: percentage return
    LOG_RETURNS = "log_returns"  # Continuous: log return
    TRIPLE_BARRIER = "triple_barrier"  # -1, 0, 1 based on barriers


@dataclass
class DataLoaderConfig:
    """Configuration for training data loader.

    Attributes:
        sequence_length: Number of time steps in each input sequence.
        prediction_horizon: Steps ahead to predict.
        train_ratio: Ratio of data for training.
        val_ratio: Ratio of data for validation.
        batch_size: Batch size for DataLoader.
        shuffle_train: Whether to shuffle training data.
        label_method: Method for generating labels.
        label_column: Column to use for label generation.
        normalization: Normalization method ('zscore', 'minmax', 'robust').
        target_threshold: Threshold for direction labels (% move).
        feature_columns: Specific columns to use (None = all).
        drop_ohlcv: Whether to drop OHLCV columns from features.
        num_workers: Number of data loading workers.
        pin_memory: Pin memory for GPU transfer.
    """

    sequence_length: int = 100
    prediction_horizon: int = 1
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    batch_size: int = 64
    shuffle_train: bool = True
    label_method: LabelMethod = LabelMethod.DIRECTION
    label_column: str = "close"
    normalization: str = "zscore"
    target_threshold: float = 0.0  # For direction labels
    feature_columns: Optional[List[str]] = None
    drop_ohlcv: bool = False
    num_workers: int = 0
    pin_memory: bool = True


class LabelGenerator:
    """Generate labels for training from price data.

    Supports multiple labeling strategies including direction prediction,
    return prediction, and triple barrier method.
    """

    def __init__(self, config: DataLoaderConfig):
        """Initialize label generator.

        Args:
            config: DataLoader configuration.
        """
        self.config = config

    def generate(
        self,
        df: pd.DataFrame,
        method: Optional[LabelMethod] = None,
    ) -> pd.Series:
        """Generate labels for the DataFrame.

        Args:
            df: Feature DataFrame with price column.
            method: Label method (uses config if None).

        Returns:
            Series of labels aligned with DataFrame index.
        """
        method = method or self.config.label_method
        col = self.config.label_column
        horizon = self.config.prediction_horizon

        if col not in df.columns:
            raise ValueError(f"Label column '{col}' not in DataFrame")

        if method == LabelMethod.DIRECTION:
            return self._direction_labels(df[col], horizon)
        elif method == LabelMethod.DIRECTION_THREE:
            return self._direction_three_labels(df[col], horizon)
        elif method == LabelMethod.RETURNS:
            return self._return_labels(df[col], horizon)
        elif method == LabelMethod.LOG_RETURNS:
            return self._log_return_labels(df[col], horizon)
        elif method == LabelMethod.TRIPLE_BARRIER:
            return self._triple_barrier_labels(df, horizon)
        else:
            raise ValueError(f"Unknown label method: {method}")

    def _direction_labels(self, prices: pd.Series, horizon: int) -> pd.Series:
        """Binary direction labels (1 = up, 0 = down)."""
        future_prices = prices.shift(-horizon)
        returns = (future_prices - prices) / prices

        if self.config.target_threshold > 0:
            # Use threshold for significant moves
            labels = (returns > self.config.target_threshold).astype(float)
        else:
            labels = (returns > 0).astype(float)

        return labels

    def _direction_three_labels(self, prices: pd.Series, horizon: int) -> pd.Series:
        """Ternary direction labels (-1, 0, 1)."""
        future_prices = prices.shift(-horizon)
        returns = (future_prices - prices) / prices

        threshold = self.config.target_threshold or 0.001  # Default 0.1%

        labels = pd.Series(0, index=prices.index)
        labels[returns > threshold] = 1
        labels[returns < -threshold] = -1

        return labels

    def _return_labels(self, prices: pd.Series, horizon: int) -> pd.Series:
        """Continuous percentage return labels."""
        future_prices = prices.shift(-horizon)
        returns = (future_prices - prices) / prices
        return returns

    def _log_return_labels(self, prices: pd.Series, horizon: int) -> pd.Series:
        """Continuous log return labels."""
        future_prices = prices.shift(-horizon)
        log_returns = np.log(future_prices / prices)
        return log_returns

    def _triple_barrier_labels(
        self,
        df: pd.DataFrame,
        horizon: int,
    ) -> pd.Series:
        """Triple barrier labels based on price movement.

        Labels:
        - 1: Take-profit hit first
        - -1: Stop-loss hit first
        - 0: Time barrier hit (no significant move)

        Args:
            df: DataFrame with OHLCV data.
            horizon: Maximum time horizon.

        Returns:
            Series of labels.
        """
        # Default barriers based on ATR if available, else use fixed %
        if "atr_14" in df.columns:
            take_profit = df["atr_14"] * 2
            stop_loss = df["atr_14"] * 1.5
        else:
            # Fixed 1% take-profit, 0.75% stop-loss
            take_profit = df["close"] * 0.01
            stop_loss = df["close"] * 0.0075

        labels = pd.Series(0, index=df.index)

        for i in range(len(df) - horizon):
            entry_price = df["close"].iloc[i]
            tp = entry_price + take_profit.iloc[i]
            sl = entry_price - stop_loss.iloc[i]

            # Look forward
            for j in range(1, horizon + 1):
                if i + j >= len(df):
                    break

                high = df["high"].iloc[i + j]
                low = df["low"].iloc[i + j]

                # Check barriers
                if high >= tp:
                    labels.iloc[i] = 1  # Take-profit
                    break
                elif low <= sl:
                    labels.iloc[i] = -1  # Stop-loss
                    break

        return labels


class TradingDataset(Dataset):
    """PyTorch Dataset for trading data.

    Handles sequence creation and provides proper time-series windows
    for LSTM/Transformer models.
    """

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        sequence_length: int,
        feature_names: Optional[List[str]] = None,
    ):
        """Initialize dataset.

        Args:
            features: Feature array of shape (n_samples, n_features).
            labels: Label array of shape (n_samples,).
            sequence_length: Length of input sequences.
            feature_names: Optional list of feature names.
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
        self.sequence_length = sequence_length
        self.feature_names = feature_names or []

        # Pre-calculate valid indices
        self.n_samples = len(features) - sequence_length

    def __len__(self) -> int:
        """Return number of samples."""
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (sequence, label) tensors.
        """
        # Get sequence ending at idx + sequence_length
        seq_end = idx + self.sequence_length
        sequence = self.features[idx:seq_end]
        label = self.labels[seq_end]

        return sequence, label


class TrainingDataLoader:
    """Main class for preparing training data.

    Handles the complete pipeline from raw features to PyTorch DataLoaders:
    1. Train/val/test split (chronological)
    2. Normalization with scaler fitting on train only
    3. Label generation
    4. Sequence creation
    5. DataLoader creation

    Example:
        ```python
        config = DataLoaderConfig(
            sequence_length=100,
            prediction_horizon=1,
            batch_size=64,
            label_method=LabelMethod.DIRECTION,
        )

        loader = TrainingDataLoader(config)

        # Get DataLoaders
        train_loader, val_loader, test_loader = loader.create_dataloaders(df_features)

        # Training loop
        for batch_x, batch_y in train_loader:
            # batch_x: (batch_size, sequence_length, n_features)
            # batch_y: (batch_size,)
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
        ```
    """

    def __init__(self, config: Optional[DataLoaderConfig] = None):
        """Initialize training data loader.

        Args:
            config: Configuration for data loading.
        """
        self.config = config or DataLoaderConfig()
        self.label_generator = LabelGenerator(self.config)

        # Fitted scalers
        self.scalers: Dict[str, Dict[str, float]] = {}
        self.feature_names: List[str] = []

        # Split information
        self.train_size: int = 0
        self.val_size: int = 0
        self.test_size: int = 0
        self.split_dates: Dict[str, datetime] = {}

    def create_dataloaders(
        self,
        df: pd.DataFrame,
        *,
        return_datasets: bool = False,
    ) -> Union[
        Tuple[DataLoader, DataLoader, DataLoader],
        Tuple[DataLoader, DataLoader, DataLoader, TradingDataset, TradingDataset, TradingDataset],
    ]:
        """Create train/val/test DataLoaders from feature DataFrame.

        Args:
            df: DataFrame with features and price data.
            return_datasets: Also return Dataset objects.

        Returns:
            Tuple of (train_loader, val_loader, test_loader) or
            with datasets if return_datasets=True.
        """
        # Select features
        features_df = self._select_features(df)
        self.feature_names = features_df.columns.tolist()

        # Generate labels
        labels = self.label_generator.generate(df)

        # Align features and labels
        features_df, labels = self._align_data(features_df, labels)

        # Split chronologically
        splits = self._split_data(features_df, labels)

        # Normalize (fit on train only)
        train_features, val_features, test_features = self._normalize_splits(
            splits["train_features"],
            splits["val_features"],
            splits["test_features"],
        )

        # Create datasets
        train_dataset = TradingDataset(
            train_features,
            splits["train_labels"],
            self.config.sequence_length,
            self.feature_names,
        )
        val_dataset = TradingDataset(
            val_features,
            splits["val_labels"],
            self.config.sequence_length,
            self.feature_names,
        )
        test_dataset = TradingDataset(
            test_features,
            splits["test_labels"],
            self.config.sequence_length,
            self.feature_names,
        )

        # Store sizes
        self.train_size = len(train_dataset)
        self.val_size = len(val_dataset)
        self.test_size = len(test_dataset)

        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=self.config.shuffle_train,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
        )

        logger.info(
            f"Created DataLoaders: train={self.train_size}, "
            f"val={self.val_size}, test={self.test_size}"
        )

        if return_datasets:
            return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset

        return train_loader, val_loader, test_loader

    def _select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select feature columns."""
        if self.config.feature_columns:
            # Use specified columns
            columns = [c for c in self.config.feature_columns if c in df.columns]
        else:
            columns = df.columns.tolist()

        # Optionally drop OHLCV columns
        if self.config.drop_ohlcv:
            ohlcv = {"open", "high", "low", "close", "volume"}
            columns = [c for c in columns if c.lower() not in ohlcv]

        # Exclude label column if present
        if self.config.label_column in columns:
            columns.remove(self.config.label_column)

        return df[columns]

    def _align_data(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """Align features and labels, dropping NaN."""
        # Combine for alignment
        combined = pd.concat([features, labels.rename("_label")], axis=1)
        combined = combined.dropna()

        features_aligned = combined.drop(columns=["_label"])
        labels_aligned = combined["_label"].values

        return features_aligned, labels_aligned

    def _split_data(
        self,
        features: pd.DataFrame,
        labels: np.ndarray,
    ) -> Dict[str, Any]:
        """Split data chronologically."""
        n = len(features)
        train_end = int(n * self.config.train_ratio)
        val_end = int(n * (self.config.train_ratio + self.config.val_ratio))

        # Store split dates
        self.split_dates = {
            "train_start": features.index[0].to_pydatetime(),
            "train_end": features.index[train_end - 1].to_pydatetime(),
            "val_start": features.index[train_end].to_pydatetime(),
            "val_end": features.index[val_end - 1].to_pydatetime(),
            "test_start": features.index[val_end].to_pydatetime(),
            "test_end": features.index[-1].to_pydatetime(),
        }

        return {
            "train_features": features.iloc[:train_end].values,
            "train_labels": labels[:train_end],
            "val_features": features.iloc[train_end:val_end].values,
            "val_labels": labels[train_end:val_end],
            "test_features": features.iloc[val_end:].values,
            "test_labels": labels[val_end:],
        }

    def _normalize_splits(
        self,
        train: np.ndarray,
        val: np.ndarray,
        test: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Normalize splits, fitting only on training data."""
        method = self.config.normalization

        if method == "zscore":
            mean = train.mean(axis=0)
            std = train.std(axis=0) + 1e-8
            train_norm = (train - mean) / std
            val_norm = (val - mean) / std
            test_norm = (test - mean) / std
            self.scalers = {"mean": mean.tolist(), "std": std.tolist()}

        elif method == "minmax":
            min_val = train.min(axis=0)
            max_val = train.max(axis=0)
            range_val = max_val - min_val + 1e-8
            train_norm = (train - min_val) / range_val
            val_norm = (val - min_val) / range_val
            test_norm = (test - min_val) / range_val
            self.scalers = {"min": min_val.tolist(), "max": max_val.tolist()}

        elif method == "robust":
            median = np.median(train, axis=0)
            q75 = np.percentile(train, 75, axis=0)
            q25 = np.percentile(train, 25, axis=0)
            iqr = q75 - q25 + 1e-8
            train_norm = (train - median) / iqr
            val_norm = (val - median) / iqr
            test_norm = (test - median) / iqr
            self.scalers = {"median": median.tolist(), "iqr": iqr.tolist()}

        else:
            # No normalization
            train_norm, val_norm, test_norm = train, val, test

        return train_norm, val_norm, test_norm

    def normalize(self, features: np.ndarray) -> np.ndarray:
        """Normalize features using fitted scalers.

        Args:
            features: Feature array to normalize.

        Returns:
            Normalized features.
        """
        if not self.scalers:
            return features

        method = self.config.normalization
        features = np.array(features)

        if method == "zscore":
            mean = np.array(self.scalers["mean"])
            std = np.array(self.scalers["std"])
            return (features - mean) / std

        elif method == "minmax":
            min_val = np.array(self.scalers["min"])
            max_val = np.array(self.scalers["max"])
            return (features - min_val) / (max_val - min_val + 1e-8)

        elif method == "robust":
            median = np.array(self.scalers["median"])
            iqr = np.array(self.scalers["iqr"])
            return (features - median) / iqr

        return features

    def denormalize(self, features: np.ndarray) -> np.ndarray:
        """Reverse normalization.

        Args:
            features: Normalized features.

        Returns:
            Original scale features.
        """
        if not self.scalers:
            return features

        method = self.config.normalization
        features = np.array(features)

        if method == "zscore":
            mean = np.array(self.scalers["mean"])
            std = np.array(self.scalers["std"])
            return features * std + mean

        elif method == "minmax":
            min_val = np.array(self.scalers["min"])
            max_val = np.array(self.scalers["max"])
            return features * (max_val - min_val) + min_val

        elif method == "robust":
            median = np.array(self.scalers["median"])
            iqr = np.array(self.scalers["iqr"])
            return features * iqr + median

        return features

    def save_scalers(self, path: Union[str, Path]) -> None:
        """Save fitted scalers to file.

        Args:
            path: Path to save scalers.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "scalers": self.scalers,
            "feature_names": self.feature_names,
            "config": {
                "normalization": self.config.normalization,
                "sequence_length": self.config.sequence_length,
                "prediction_horizon": self.config.prediction_horizon,
            },
        }

        with open(path, "wb") as f:
            pickle.dump(data, f)

        logger.info(f"Saved scalers to {path}")

    def load_scalers(self, path: Union[str, Path]) -> None:
        """Load fitted scalers from file.

        Args:
            path: Path to load scalers from.
        """
        with open(path, "rb") as f:
            data = pickle.load(f)

        self.scalers = data["scalers"]
        self.feature_names = data["feature_names"]

        logger.info(f"Loaded scalers from {path}")

    def get_info(self) -> Dict[str, Any]:
        """Get information about the data loader state.

        Returns:
            Dictionary with loader information.
        """
        return {
            "sequence_length": self.config.sequence_length,
            "prediction_horizon": self.config.prediction_horizon,
            "batch_size": self.config.batch_size,
            "normalization": self.config.normalization,
            "label_method": self.config.label_method.value,
            "n_features": len(self.feature_names),
            "feature_names": self.feature_names,
            "train_size": self.train_size,
            "val_size": self.val_size,
            "test_size": self.test_size,
            "split_dates": {k: v.isoformat() for k, v in self.split_dates.items()},
        }


def create_dataloaders(
    df: pd.DataFrame,
    sequence_length: int = 100,
    prediction_horizon: int = 1,
    batch_size: int = 64,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    label_method: str = "direction",
) -> Tuple[DataLoader, DataLoader, DataLoader, TrainingDataLoader]:
    """Convenience function to create DataLoaders.

    Args:
        df: Feature DataFrame.
        sequence_length: Input sequence length.
        prediction_horizon: Prediction steps ahead.
        batch_size: Batch size.
        train_ratio: Training data ratio.
        val_ratio: Validation data ratio.
        label_method: Label generation method.

    Returns:
        Tuple of (train_loader, val_loader, test_loader, loader_instance).
    """
    config = DataLoaderConfig(
        sequence_length=sequence_length,
        prediction_horizon=prediction_horizon,
        batch_size=batch_size,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        label_method=LabelMethod(label_method),
    )

    loader = TrainingDataLoader(config)
    train, val, test = loader.create_dataloaders(df)

    return train, val, test, loader
