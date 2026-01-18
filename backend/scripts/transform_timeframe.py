#!/usr/bin/env python3
"""
Timeframe Transformation Script with Sliding Window Aggregation

This script transforms OHLCV data from a base timeframe to higher timeframes
using sliding window aggregation to preserve the total number of records.

Example:
    # Using CLI arguments
    python transform_timeframe.py --input data/forex --output data/forex/derived \
        --base-minutes 5 --target-minutes 15 60 240

    # Using config file
    python transform_timeframe.py --config configs/timeframe_transform.yaml

Author: AI Trader Development Team
"""

import argparse
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class TransformConfig:
    """Configuration for timeframe transformation."""

    # Input/Output paths
    input_path: Path
    output_path: Path

    # Timeframe settings
    base_minutes: int  # Base timeframe in minutes (e.g., 5 for 5-minute data)
    target_minutes: list[int]  # Target timeframes in minutes (e.g., [15, 60, 240])

    # Column mapping (for different CSV formats)
    timestamp_col: str = 'timestamp'
    open_col: str = 'open'
    high_col: str = 'high'
    low_col: str = 'low'
    close_col: str = 'close'
    volume_col: str = 'volume'

    # Processing options
    file_pattern: str = '*.csv'  # Pattern to match input files
    symbol_pattern: str = r'^([A-Z]{6})'  # Regex to extract symbol from filename
    combine_input_files: bool = True  # Combine all matching files before transform
    timestamp_position: str = 'start'  # 'start' or 'end' of aggregation period

    # Output options
    output_format: str = 'parquet'  # 'parquet' or 'csv'
    include_metadata: bool = True  # Include aggregation metadata columns
    compression: str = 'snappy'  # Parquet compression: 'snappy', 'gzip', 'none'

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> 'TransformConfig':
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        # Convert paths
        config_dict['input_path'] = Path(config_dict['input_path'])
        config_dict['output_path'] = Path(config_dict['output_path'])

        return cls(**config_dict)

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'TransformConfig':
        """Create configuration from CLI arguments."""
        return cls(
            input_path=Path(args.input),
            output_path=Path(args.output),
            base_minutes=args.base_minutes,
            target_minutes=args.target_minutes,
            timestamp_col=args.timestamp_col,
            open_col=args.open_col,
            high_col=args.high_col,
            low_col=args.low_col,
            close_col=args.close_col,
            volume_col=args.volume_col,
            file_pattern=args.file_pattern,
            combine_input_files=args.combine_files,
            timestamp_position=args.timestamp_position,
            output_format=args.output_format,
            include_metadata=args.include_metadata,
            compression=args.compression,
        )

    def validate(self) -> None:
        """Validate configuration."""
        if not self.input_path.exists():
            raise ValueError(f"Input path does not exist: {self.input_path}")

        if self.base_minutes <= 0:
            raise ValueError(f"Base minutes must be positive: {self.base_minutes}")

        for target in self.target_minutes:
            if target <= self.base_minutes:
                raise ValueError(
                    f"Target timeframe ({target}m) must be greater than "
                    f"base timeframe ({self.base_minutes}m)"
                )
            if target % self.base_minutes != 0:
                raise ValueError(
                    f"Target timeframe ({target}m) must be a multiple of "
                    f"base timeframe ({self.base_minutes}m)"
                )

        if self.output_format not in ('parquet', 'csv'):
            raise ValueError(f"Invalid output format: {self.output_format}")

        if self.timestamp_position not in ('start', 'end'):
            raise ValueError(f"Invalid timestamp position: {self.timestamp_position}")


class TimeframeTransformer:
    """Transforms OHLCV data between timeframes with sliding window aggregation."""

    def __init__(self, config: TransformConfig):
        self.config = config

    def load_data(self) -> dict[str, pd.DataFrame]:
        """
        Load OHLCV data from input path.

        Returns:
            Dictionary mapping symbol to DataFrame
        """
        import re

        input_path = self.config.input_path
        pattern = self.config.file_pattern

        if input_path.is_file():
            files = [input_path]
        else:
            files = sorted(input_path.glob(pattern))

        if not files:
            raise ValueError(f"No files found matching {pattern} in {input_path}")

        logger.info(f"Found {len(files)} files to process")

        # Group files by symbol
        symbol_files: dict[str, list[Path]] = {}
        symbol_regex = re.compile(self.config.symbol_pattern)

        for file in files:
            match = symbol_regex.search(file.stem)
            if match:
                symbol = match.group(1)
            else:
                symbol = file.stem.split('_')[0]

            if symbol not in symbol_files:
                symbol_files[symbol] = []
            symbol_files[symbol].append(file)

        # Load and combine data per symbol
        result: dict[str, pd.DataFrame] = {}

        for symbol, files in symbol_files.items():
            dfs = []
            for file in sorted(files):
                logger.info(f"Loading {file.name}")
                df = pd.read_csv(file, parse_dates=[self.config.timestamp_col])
                dfs.append(df)

            if self.config.combine_input_files:
                combined = pd.concat(dfs, ignore_index=True)
                combined = combined.sort_values(self.config.timestamp_col)
                combined = combined.drop_duplicates(
                    subset=[self.config.timestamp_col],
                    keep='last'
                )
                result[symbol] = combined
                logger.info(
                    f"Loaded {symbol}: {len(combined)} records "
                    f"({combined[self.config.timestamp_col].min()} to "
                    f"{combined[self.config.timestamp_col].max()})"
                )
            else:
                # Return each file separately
                for file, df in zip(files, dfs):
                    key = f"{symbol}_{file.stem}"
                    result[key] = df

        return result

    def aggregate_ohlcv(
        self,
        df: pd.DataFrame,
        target_minutes: int,
        slice_id: int = 0
    ) -> pd.DataFrame:
        """
        Aggregate OHLCV data to target timeframe with sliding window offset.

        Args:
            df: Source DataFrame with OHLCV data
            target_minutes: Target timeframe in minutes
            slice_id: Offset index for sliding window (0 to factor-1)

        Returns:
            Aggregated DataFrame
        """
        cfg = self.config
        factor = target_minutes // cfg.base_minutes

        if slice_id >= factor:
            raise ValueError(f"slice_id ({slice_id}) must be < factor ({factor})")

        # Ensure sorted by timestamp
        df = df.sort_values(cfg.timestamp_col).reset_index(drop=True)

        # Apply offset for sliding window
        df_offset = df.iloc[slice_id:].copy()

        # Create group indices (every 'factor' rows form one candle)
        df_offset['_group'] = df_offset.index // factor

        # Aggregate OHLCV
        agg_dict = {
            cfg.timestamp_col: 'first' if cfg.timestamp_position == 'start' else 'last',
            cfg.open_col: 'first',
            cfg.high_col: 'max',
            cfg.low_col: 'min',
            cfg.close_col: 'last',
            cfg.volume_col: 'sum',
        }

        # Check for spread column
        if 'spread' in df_offset.columns:
            agg_dict['spread'] = 'mean'

        aggregated = df_offset.groupby('_group').agg(agg_dict).reset_index(drop=True)

        # Remove incomplete candles (last group might have fewer than 'factor' rows)
        group_sizes = df_offset.groupby('_group').size()
        complete_groups = group_sizes[group_sizes == factor].index
        aggregated = aggregated[aggregated.index.isin(complete_groups)].reset_index(drop=True)

        # Add metadata
        if cfg.include_metadata:
            aggregated['slice_id'] = slice_id
            aggregated['base_minutes'] = cfg.base_minutes
            aggregated['target_minutes'] = target_minutes
            aggregated['candles_aggregated'] = factor

        return aggregated

    def transform_with_sliding_window(
        self,
        df: pd.DataFrame,
        target_minutes: int
    ) -> pd.DataFrame:
        """
        Transform data to target timeframe using all sliding window slices.

        Args:
            df: Source DataFrame
            target_minutes: Target timeframe in minutes

        Returns:
            DataFrame with all slices combined
        """
        factor = target_minutes // self.config.base_minutes

        logger.info(
            f"Transforming to {target_minutes}m: "
            f"factor={factor}, creating {factor} slices"
        )

        slices = []
        for slice_id in range(factor):
            slice_df = self.aggregate_ohlcv(df, target_minutes, slice_id)
            slices.append(slice_df)
            logger.debug(f"  Slice {slice_id}: {len(slice_df)} records")

        combined = pd.concat(slices, ignore_index=True)
        combined = combined.sort_values(
            [self.config.timestamp_col, 'slice_id']
            if self.config.include_metadata
            else [self.config.timestamp_col]
        ).reset_index(drop=True)

        logger.info(
            f"  Total records: {len(combined)} "
            f"(original: {len(df)}, preservation: {len(combined)/len(df)*100:.1f}%)"
        )

        return combined

    def save_output(
        self,
        df: pd.DataFrame,
        symbol: str,
        target_minutes: int
    ) -> Path:
        """Save transformed data to output file."""
        cfg = self.config

        # Create output directory
        timeframe_label = self._minutes_to_label(target_minutes)
        output_dir = cfg.output_path / timeframe_label
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename
        filename = f"{symbol}_{timeframe_label}"

        if cfg.output_format == 'parquet':
            output_file = output_dir / f"{filename}.parquet"
            compression = None if cfg.compression == 'none' else cfg.compression
            df.to_parquet(output_file, compression=compression, index=False)
        else:
            output_file = output_dir / f"{filename}.csv"
            df.to_csv(output_file, index=False)

        logger.info(f"Saved: {output_file} ({len(df)} records)")
        return output_file

    def _minutes_to_label(self, minutes: int) -> str:
        """Convert minutes to human-readable label."""
        if minutes < 60:
            return f"{minutes}m"
        elif minutes < 1440:
            hours = minutes // 60
            return f"{hours}H"
        elif minutes < 10080:
            days = minutes // 1440
            return f"{days}D"
        elif minutes < 43200:
            weeks = minutes // 10080
            return f"{weeks}W"
        else:
            months = minutes // 43200
            return f"{months}M"

    def run(self) -> dict[str, list[Path]]:
        """
        Run the full transformation pipeline.

        Returns:
            Dictionary mapping symbol to list of output files
        """
        self.config.validate()

        # Load source data
        data = self.load_data()

        # Transform each symbol to each target timeframe
        output_files: dict[str, list[Path]] = {}

        for symbol, df in data.items():
            output_files[symbol] = []

            for target_minutes in self.config.target_minutes:
                transformed = self.transform_with_sliding_window(df, target_minutes)
                output_file = self.save_output(transformed, symbol, target_minutes)
                output_files[symbol].append(output_file)

        return output_files


def create_sample_config(output_path: Path) -> None:
    """Create a sample configuration file."""
    sample_config = """# Timeframe Transformation Configuration
# =======================================

# Input/Output paths
input_path: "data/forex"
output_path: "data/forex/derived"

# Timeframe settings
base_minutes: 5  # Base timeframe of source data (5 minutes)
target_minutes:  # Target timeframes to generate
  - 15    # 15 minutes (factor: 3)
  - 30    # 30 minutes (factor: 6)
  - 60    # 1 hour (factor: 12)
  - 240   # 4 hours (factor: 48)
  - 1440  # 1 day (factor: 288)

# Column mapping (adjust if your CSV has different column names)
timestamp_col: "timestamp"
open_col: "open"
high_col: "high"
low_col: "low"
close_col: "close"
volume_col: "volume"

# Processing options
file_pattern: "*.csv"  # Pattern to match input files
symbol_pattern: "^([A-Z]{6})"  # Regex to extract symbol from filename
combine_input_files: true  # Combine all matching files before transform
timestamp_position: "start"  # 'start' or 'end' of aggregation period

# Output options
output_format: "parquet"  # 'parquet' or 'csv'
include_metadata: true  # Include slice_id and aggregation info
compression: "snappy"  # 'snappy', 'gzip', or 'none'
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(sample_config)

    logger.info(f"Created sample config: {output_path}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Transform OHLCV data between timeframes with sliding window aggregation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Transform using config file
  python transform_timeframe.py --config configs/timeframe_transform.yaml

  # Transform using CLI arguments
  python transform_timeframe.py --input data/forex --output data/forex/derived \\
      --base-minutes 5 --target-minutes 15 60 240

  # Create sample config file
  python transform_timeframe.py --create-config configs/timeframe_transform.yaml

  # Transform to CSV format instead of Parquet
  python transform_timeframe.py --config configs/timeframe_transform.yaml --output-format csv
        """
    )

    # Config file option
    parser.add_argument(
        '--config', '-c',
        type=Path,
        help='Path to YAML configuration file'
    )

    # Create sample config
    parser.add_argument(
        '--create-config',
        type=Path,
        metavar='PATH',
        help='Create a sample configuration file at the specified path'
    )

    # Input/Output options
    parser.add_argument(
        '--input', '-i',
        type=str,
        help='Input path (file or directory)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output directory'
    )

    # Timeframe options
    parser.add_argument(
        '--base-minutes', '-b',
        type=int,
        default=5,
        help='Base timeframe in minutes (default: 5)'
    )
    parser.add_argument(
        '--target-minutes', '-t',
        type=int,
        nargs='+',
        default=[15, 60, 240],
        help='Target timeframes in minutes (default: 15 60 240)'
    )

    # Column mapping
    parser.add_argument('--timestamp-col', default='timestamp')
    parser.add_argument('--open-col', default='open')
    parser.add_argument('--high-col', default='high')
    parser.add_argument('--low-col', default='low')
    parser.add_argument('--close-col', default='close')
    parser.add_argument('--volume-col', default='volume')

    # Processing options
    parser.add_argument(
        '--file-pattern',
        default='*.csv',
        help='Glob pattern for input files (default: *.csv)'
    )
    parser.add_argument(
        '--combine-files',
        action='store_true',
        default=True,
        help='Combine all input files per symbol (default: true)'
    )
    parser.add_argument(
        '--no-combine-files',
        action='store_false',
        dest='combine_files',
        help='Process each input file separately'
    )
    parser.add_argument(
        '--timestamp-position',
        choices=['start', 'end'],
        default='start',
        help='Position of timestamp in aggregated candle (default: start)'
    )

    # Output options
    parser.add_argument(
        '--output-format', '-f',
        choices=['parquet', 'csv'],
        default='parquet',
        help='Output format (default: parquet)'
    )
    parser.add_argument(
        '--include-metadata',
        action='store_true',
        default=True,
        help='Include metadata columns (default: true)'
    )
    parser.add_argument(
        '--no-metadata',
        action='store_false',
        dest='include_metadata',
        help='Exclude metadata columns'
    )
    parser.add_argument(
        '--compression',
        choices=['snappy', 'gzip', 'none'],
        default='snappy',
        help='Parquet compression (default: snappy)'
    )

    # Logging
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create sample config if requested
    if args.create_config:
        create_sample_config(args.create_config)
        return 0

    # Load configuration
    if args.config:
        if not args.config.exists():
            logger.error(f"Config file not found: {args.config}")
            return 1
        config = TransformConfig.from_yaml(args.config)

        # Allow CLI overrides
        if args.input:
            config.input_path = Path(args.input)
        if args.output:
            config.output_path = Path(args.output)
        if args.target_minutes != [15, 60, 240]:  # Non-default value
            config.target_minutes = args.target_minutes
        if args.output_format != 'parquet':
            config.output_format = args.output_format
    else:
        # Use CLI arguments
        if not args.input or not args.output:
            logger.error("Either --config or both --input and --output are required")
            return 1
        config = TransformConfig.from_args(args)

    # Run transformation
    try:
        transformer = TimeframeTransformer(config)
        output_files = transformer.run()

        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("TRANSFORMATION COMPLETE")
        logger.info("=" * 60)

        total_files = sum(len(files) for files in output_files.values())
        logger.info(f"Generated {total_files} output files:")

        for symbol, files in output_files.items():
            logger.info(f"\n{symbol}:")
            for file in files:
                logger.info(f"  - {file}")

        return 0

    except Exception as e:
        logger.error(f"Transformation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
