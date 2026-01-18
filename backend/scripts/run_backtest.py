#!/usr/bin/env python3
"""
Backtest CLI Script.

Run backtests from the command line with configurable parameters.

Usage:
    python scripts/run_backtest.py --symbol EURUSD --start 2023-01-01 --end 2024-01-01
    python scripts/run_backtest.py --config configs/backtest/example.yaml
    python scripts/run_backtest.py --symbol EURUSD --risk-profile conservative --output results/backtest_001
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
import yaml

import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.simulation import (
    EnhancedBacktester,
    BacktestConfig,
    MarketSimulator,
)
from src.data.sources.csv_source import CSVDataSource
from src.features.technical.indicators import TechnicalIndicators


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run backtests on historical data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic backtest
  python scripts/run_backtest.py --symbol EURUSD --data data/sample/EURUSD_daily.csv

  # With date range
  python scripts/run_backtest.py --symbol EURUSD --start 2023-01-01 --end 2024-01-01

  # With custom risk profile
  python scripts/run_backtest.py --symbol EURUSD --risk-profile conservative

  # Save results
  python scripts/run_backtest.py --symbol EURUSD --output results/my_backtest
        """,
    )

    # Required arguments
    parser.add_argument(
        "--symbol",
        type=str,
        default="EURUSD",
        help="Trading symbol (default: EURUSD)",
    )

    # Data source
    parser.add_argument(
        "--data",
        type=str,
        help="Path to OHLCV CSV file",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/sample",
        help="Directory containing data files (default: data/sample)",
    )

    # Date range
    parser.add_argument(
        "--start",
        type=str,
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        type=str,
        help="End date (YYYY-MM-DD)",
    )

    # Capital and risk
    parser.add_argument(
        "--capital",
        type=float,
        default=100000.0,
        help="Initial capital (default: 100000)",
    )
    parser.add_argument(
        "--risk-profile",
        type=str,
        default="moderate",
        choices=["ultra_conservative", "conservative", "moderate", "aggressive", "ultra_aggressive"],
        help="Risk profile (default: moderate)",
    )
    parser.add_argument(
        "--leverage",
        type=float,
        default=10.0,
        help="Leverage (default: 10)",
    )

    # Execution simulation
    parser.add_argument(
        "--slippage",
        type=float,
        default=0.0001,
        help="Slippage percentage (default: 0.0001 = 1 pip)",
    )
    parser.add_argument(
        "--commission",
        type=float,
        default=0.0001,
        help="Commission percentage (default: 0.0001)",
    )

    # Trading parameters
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.65,
        help="Minimum signal confidence (default: 0.65)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=100,
        help="Warmup bars before trading (default: 100)",
    )

    # Output
    parser.add_argument(
        "--output",
        type=str,
        help="Output directory for results",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="backtest",
        help="Backtest name (default: backtest)",
    )

    # Configuration file
    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML configuration file",
    )

    # Options
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Quiet output (no progress)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to files",
    )

    return parser.parse_args()


def load_config_file(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_data(
    symbol: str,
    data_path: Optional[str] = None,
    data_dir: str = "data/sample",
) -> pd.DataFrame:
    """
    Load OHLCV data from CSV.

    Args:
        symbol: Trading symbol
        data_path: Direct path to CSV file
        data_dir: Directory to search for symbol data

    Returns:
        DataFrame with OHLCV data
    """
    if data_path:
        path = Path(data_path)
    else:
        # Try to find data file
        data_dir = Path(data_dir)
        candidates = [
            data_dir / f"{symbol}_daily.csv",
            data_dir / f"{symbol}.csv",
            data_dir / f"{symbol.lower()}_daily.csv",
            data_dir / f"{symbol.lower()}.csv",
        ]

        path = None
        for candidate in candidates:
            if candidate.exists():
                path = candidate
                break

        if path is None:
            raise FileNotFoundError(
                f"Could not find data for {symbol} in {data_dir}. "
                f"Tried: {[str(c) for c in candidates]}"
            )

    logging.info(f"Loading data from {path}")

    # Load CSV
    df = pd.read_csv(path)

    # Standardize columns
    df.columns = [c.lower() for c in df.columns]

    # Parse date
    date_col = None
    for col in ["date", "timestamp", "datetime", "time"]:
        if col in df.columns:
            date_col = col
            break

    if date_col is None:
        raise ValueError("Could not find date column in data")

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)
    df = df.sort_index()

    # Ensure required columns
    required = ["open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Add volume if missing
    if "volume" not in df.columns:
        df["volume"] = 0

    logging.info(f"Loaded {len(df)} bars from {df.index[0].date()} to {df.index[-1].date()}")

    return df


def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators for features."""
    try:
        indicators = TechnicalIndicators()
        features = indicators.calculate_all(df)
        logging.info(f"Calculated {len(features.columns)} features")
        return features
    except Exception as e:
        logging.warning(f"Could not calculate features: {e}")
        return df


def create_backtest_config(args: argparse.Namespace) -> BacktestConfig:
    """Create backtest configuration from arguments."""
    # Start with defaults
    config_dict = {
        "name": args.name,
        "symbol": args.symbol,
        "initial_capital": args.capital,
        "risk_profile_name": args.risk_profile,
        "leverage": args.leverage,
        "slippage_pct": args.slippage,
        "commission_pct": args.commission,
        "min_confidence": args.min_confidence,
        "warmup_bars": args.warmup,
        "verbose": not args.quiet,
    }

    # Parse dates
    if args.start:
        config_dict["start_date"] = datetime.strptime(args.start, "%Y-%m-%d")
    if args.end:
        config_dict["end_date"] = datetime.strptime(args.end, "%Y-%m-%d")

    # Output
    if args.output and not args.no_save:
        config_dict["output_dir"] = args.output
        config_dict["save_trades"] = True
        config_dict["save_equity_curve"] = True

    # Load from config file if provided
    if args.config:
        file_config = load_config_file(args.config)
        # File config overrides defaults but not CLI args
        for key, value in file_config.items():
            if key not in config_dict or config_dict[key] is None:
                config_dict[key] = value

    return BacktestConfig(**config_dict)


def main() -> int:
    """Main entry point."""
    args = parse_args()
    setup_logging(args.verbose)

    logger = logging.getLogger(__name__)

    try:
        # Create configuration
        config = create_backtest_config(args)
        logger.info(f"Backtest configuration: {config.name}")
        logger.info(f"  Symbol: {config.symbol}")
        logger.info(f"  Capital: ${config.initial_capital:,.2f}")
        logger.info(f"  Risk Profile: {config.risk_profile_name}")

        # Load data
        data = load_data(
            symbol=args.symbol,
            data_path=args.data,
            data_dir=args.data_dir,
        )

        # Calculate features
        features = calculate_features(data)

        # Create backtester
        backtester = EnhancedBacktester(config=config)

        # Load data into backtester
        backtester.load_data(config.symbol, data, features)

        # Run backtest
        logger.info("Starting backtest...")
        result = backtester.run()

        # Print results
        if not args.quiet:
            result.print_summary()

        # Return code based on result
        if result.status.value == "completed":
            if result.total_return > 0:
                logger.info("Backtest completed successfully with positive returns")
                return 0
            else:
                logger.info("Backtest completed with negative returns")
                return 0
        else:
            logger.error(f"Backtest failed: {result.status.value}")
            return 1

    except FileNotFoundError as e:
        logger.error(f"Data file not found: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return 1
    except Exception as e:
        logger.exception(f"Backtest failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
