#!/usr/bin/env python3
"""
Trading Robot Deployment Script.

Deploy and manage the AI trading robot in various modes:
- Simulation: Local historical data simulation
- Paper: Paper trading with real market data
- Production: Live trading with real money

Usage:
    # Paper trading with Alpaca
    python scripts/deploy_robot.py --mode paper --broker alpaca --symbol EURUSD

    # Production trading (requires --confirm)
    python scripts/deploy_robot.py --mode production --broker alpaca --symbol EURUSD --confirm

    # View status
    python scripts/deploy_robot.py --status

    # Stop robot
    python scripts/deploy_robot.py --stop
"""

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.trading.brokers import (
    BrokerAdapter,
    BrokerConfig,
    BrokerType,
    AlpacaBroker,
    MT5Broker,
    create_broker,
)
from src.trading.execution import (
    ProductionExecutionEngine,
    ProductionConfig,
    OrderValidationConfig,
)
from src.trading.safety import (
    KillSwitch,
    KillSwitchConfig,
)
from src.trading.monitoring import (
    TradingMonitor,
    MonitoringConfig,
    AlertLevel,
    AlertChannel,
    AlertThreshold,
    MetricType,
)
from src.trading.risk import load_risk_profile, RiskLevel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("trading_robot.log"),
    ],
)
logger = logging.getLogger(__name__)


class RobotDeployment:
    """Trading robot deployment manager."""

    def __init__(
        self,
        mode: str,
        broker_type: str,
        symbol: str,
        risk_profile: str = "moderate",
        config_file: Optional[str] = None,
    ):
        """
        Initialize deployment.

        Args:
            mode: Deployment mode (simulation, paper, production)
            broker_type: Broker to use (alpaca, mt5)
            symbol: Trading symbol
            risk_profile: Risk profile name
            config_file: Optional config file path
        """
        self.mode = mode
        self.broker_type = broker_type
        self.symbol = symbol
        self.risk_profile_name = risk_profile

        # Components
        self.broker: Optional[BrokerAdapter] = None
        self.execution_engine: Optional[ProductionExecutionEngine] = None
        self.kill_switch: Optional[KillSwitch] = None
        self.monitor: Optional[TradingMonitor] = None

        # State
        self._running = False
        self._shutdown_event = asyncio.Event()

        # Load config
        self.config = self._load_config(config_file)

    def _load_config(self, config_file: Optional[str]) -> dict:
        """Load configuration from file or environment."""
        config = {
            "broker": {
                "api_key": os.environ.get("ALPACA_API_KEY", ""),
                "secret_key": os.environ.get("ALPACA_SECRET_KEY", ""),
                "paper": self.mode != "production",
            },
            "risk": {
                "profile": self.risk_profile_name,
            },
            "kill_switch": {
                "max_daily_loss_pct": 5.0,
                "max_daily_trades": 100,
            },
            "monitoring": {
                "enable_alerts": True,
                "webhook_url": os.environ.get("ALERT_WEBHOOK_URL"),
            },
        }

        if config_file and Path(config_file).exists():
            with open(config_file) as f:
                file_config = json.load(f)
                # Deep merge
                for key, value in file_config.items():
                    if key in config and isinstance(config[key], dict):
                        config[key].update(value)
                    else:
                        config[key] = value

        return config

    async def setup(self) -> None:
        """Set up all components."""
        logger.info(f"Setting up robot deployment: mode={self.mode}, broker={self.broker_type}")

        # Create broker
        self.broker = self._create_broker()

        # Create kill switch
        self.kill_switch = self._create_kill_switch()

        # Create monitor
        self.monitor = self._create_monitor()

        # Create execution engine
        self.execution_engine = self._create_execution_engine()

        logger.info("Robot deployment setup complete")

    def _create_broker(self) -> BrokerAdapter:
        """Create broker adapter."""
        broker_config = self.config.get("broker", {})

        if self.broker_type == "alpaca":
            config = BrokerConfig(
                broker_type=BrokerType.ALPACA,
                api_key=broker_config.get("api_key", ""),
                secret_key=broker_config.get("secret_key", ""),
                paper=broker_config.get("paper", True),
            )
            return AlpacaBroker(config)

        elif self.broker_type == "mt5":
            config = BrokerConfig(
                broker_type=BrokerType.MT5,
                login=broker_config.get("login", 0),
                password=broker_config.get("password", ""),
                server=broker_config.get("server", ""),
                path=broker_config.get("path", ""),
            )
            return MT5Broker(config)

        else:
            raise ValueError(f"Unknown broker type: {self.broker_type}")

    def _create_kill_switch(self) -> KillSwitch:
        """Create kill switch."""
        ks_config = self.config.get("kill_switch", {})

        config = KillSwitchConfig(
            max_daily_loss_pct=ks_config.get("max_daily_loss_pct", 5.0),
            max_daily_loss_amount=ks_config.get("max_daily_loss_amount", 10000.0),
            max_daily_trades=ks_config.get("max_daily_trades", 100),
            max_total_position_value=ks_config.get("max_total_position_value", 1000000.0),
            max_disconnection_seconds=ks_config.get("max_disconnection_seconds", 60.0),
            require_authorization_code=self.mode == "production",
        )

        return KillSwitch(config)

    def _create_monitor(self) -> TradingMonitor:
        """Create monitoring system."""
        mon_config = self.config.get("monitoring", {})

        # Configure alert channels
        channels = [AlertChannel.LOG]
        if mon_config.get("webhook_url"):
            channels.append(AlertChannel.WEBHOOK)

        config = MonitoringConfig(
            enable_alerts=mon_config.get("enable_alerts", True),
            alert_channels=channels,
            webhook_url=mon_config.get("webhook_url"),
            log_file_path=mon_config.get("log_file", "trading_alerts.log"),
        )

        # Add default thresholds
        config.thresholds = [
            AlertThreshold(
                metric_type=MetricType.DRAWDOWN,
                warning_threshold=5.0,
                error_threshold=10.0,
                critical_threshold=15.0,
                comparison="gt",
            ),
            AlertThreshold(
                metric_type=MetricType.MARGIN_LEVEL,
                warning_threshold=200.0,
                error_threshold=150.0,
                critical_threshold=100.0,
                comparison="lt",
            ),
        ]

        return TradingMonitor(config)

    def _create_execution_engine(self) -> ProductionExecutionEngine:
        """Create execution engine."""
        # Load risk profile
        risk_profile = load_risk_profile(self.risk_profile_name)

        # Create validation config from risk profile
        validation_config = OrderValidationConfig(
            max_order_value=1000000.0,
            max_daily_volume=10000000.0,
            max_orders_per_minute=60,
            max_orders_per_day=self.config.get("kill_switch", {}).get("max_daily_trades", 100),
            require_kill_switch_check=True,
        )

        # Create production config
        prod_config = ProductionConfig(
            validation_config=validation_config,
            reconciliation_interval_seconds=60.0,
            auto_reconnect=True,
        )

        return ProductionExecutionEngine(
            broker=self.broker,
            config=prod_config,
            kill_switch_callback=lambda: self.kill_switch.is_active,
        )

    async def connect(self) -> bool:
        """Connect to broker."""
        logger.info(f"Connecting to {self.broker_type}...")

        try:
            success = await self.broker.connect()
            if success:
                logger.info("Broker connection established")

                # Get account info
                account = await self.broker.get_account()
                logger.info(
                    f"Account: {account.account_id}, "
                    f"Balance: ${account.balance:,.2f}, "
                    f"Equity: ${account.equity:,.2f}"
                )

                return True
            else:
                logger.error("Failed to connect to broker")
                return False

        except Exception as e:
            logger.error(f"Connection error: {e}")
            return False

    async def start(self) -> None:
        """Start the trading robot."""
        self._running = True

        # Start components
        self.monitor.start()
        await self.execution_engine.start()

        logger.info(f"Trading robot started in {self.mode} mode")

        # Main loop
        try:
            while self._running and not self._shutdown_event.is_set():
                await self._trading_cycle()
                await asyncio.sleep(1.0)

        except asyncio.CancelledError:
            logger.info("Trading robot cancelled")
        finally:
            await self.stop()

    async def _trading_cycle(self) -> None:
        """Single trading cycle."""
        try:
            # Check kill switch
            if self.kill_switch.is_active:
                logger.warning("Kill switch active - skipping trading cycle")
                return

            # Check connectivity
            self.kill_switch.check_connectivity(self.broker.is_connected)

            # Update monitor metrics
            if self.broker.is_connected:
                try:
                    account = await self.broker.get_account()
                    self.monitor.record_metric(MetricType.EQUITY, account.equity)
                    self.monitor.record_metric(MetricType.BALANCE, account.balance)
                    self.monitor.record_metric(MetricType.MARGIN_LEVEL, account.margin_level)
                except Exception as e:
                    logger.error(f"Failed to update account metrics: {e}")

        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")

    async def stop(self) -> None:
        """Stop the trading robot."""
        logger.info("Stopping trading robot...")

        self._running = False
        self._shutdown_event.set()

        # Stop components
        if self.execution_engine:
            await self.execution_engine.stop()

        if self.monitor:
            self.monitor.stop()

        if self.broker:
            await self.broker.disconnect()

        logger.info("Trading robot stopped")

    def get_status(self) -> dict:
        """Get current status."""
        return {
            "mode": self.mode,
            "broker": self.broker_type,
            "symbol": self.symbol,
            "running": self._running,
            "broker_connected": self.broker.is_connected if self.broker else False,
            "kill_switch_active": self.kill_switch.is_active if self.kill_switch else False,
            "monitor_stats": self.monitor.get_stats() if self.monitor else {},
            "execution_stats": self.execution_engine.get_stats() if self.execution_engine else {},
        }

    def print_status(self) -> None:
        """Print current status to console."""
        status = self.get_status()

        print("\n" + "=" * 60)
        print("TRADING ROBOT STATUS")
        print("=" * 60)
        print(f"Mode:              {status['mode']}")
        print(f"Broker:            {status['broker']}")
        print(f"Symbol:            {status['symbol']}")
        print(f"Running:           {status['running']}")
        print(f"Broker Connected:  {status['broker_connected']}")
        print(f"Kill Switch:       {'ACTIVE' if status['kill_switch_active'] else 'inactive'}")
        print("=" * 60 + "\n")


def setup_signal_handlers(deployment: RobotDeployment) -> None:
    """Set up signal handlers for graceful shutdown."""
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, initiating shutdown...")
        deployment._shutdown_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Deploy AI Trading Robot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Paper trading with Alpaca:
    python scripts/deploy_robot.py --mode paper --broker alpaca --symbol AAPL

  Production trading (requires --confirm):
    python scripts/deploy_robot.py --mode production --broker alpaca --symbol AAPL --confirm

  View status:
    python scripts/deploy_robot.py --status
        """,
    )

    parser.add_argument(
        "--mode",
        choices=["simulation", "paper", "production"],
        default="paper",
        help="Deployment mode",
    )
    parser.add_argument(
        "--broker",
        choices=["alpaca", "mt5"],
        default="alpaca",
        help="Broker to use",
    )
    parser.add_argument(
        "--symbol",
        default="AAPL",
        help="Trading symbol",
    )
    parser.add_argument(
        "--risk-profile",
        choices=["ultra_conservative", "conservative", "moderate", "aggressive", "ultra_aggressive"],
        default="moderate",
        help="Risk profile",
    )
    parser.add_argument(
        "--config",
        help="Configuration file path",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Confirm production deployment (required for production mode)",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show status and exit",
    )
    parser.add_argument(
        "--stop",
        action="store_true",
        help="Stop running robot",
    )

    args = parser.parse_args()

    # Production mode requires confirmation
    if args.mode == "production" and not args.confirm:
        print("\n" + "!" * 60)
        print("WARNING: Production mode requires --confirm flag")
        print("This will trade with REAL MONEY!")
        print("!" * 60 + "\n")
        print("To proceed, run:")
        print(f"  python {sys.argv[0]} --mode production --broker {args.broker} --symbol {args.symbol} --confirm")
        sys.exit(1)

    # Create deployment
    deployment = RobotDeployment(
        mode=args.mode,
        broker_type=args.broker,
        symbol=args.symbol,
        risk_profile=args.risk_profile,
        config_file=args.config,
    )

    # Set up signal handlers
    setup_signal_handlers(deployment)

    # Status only
    if args.status:
        deployment.print_status()
        return

    # Initialize
    print("\n" + "=" * 60)
    print(f"AI TRADING ROBOT - {args.mode.upper()} MODE")
    print("=" * 60)
    print(f"Broker:       {args.broker}")
    print(f"Symbol:       {args.symbol}")
    print(f"Risk Profile: {args.risk_profile}")
    print("=" * 60 + "\n")

    try:
        # Setup
        await deployment.setup()

        # Connect
        if not await deployment.connect():
            logger.error("Failed to connect to broker")
            sys.exit(1)

        # Start
        await deployment.start()

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Deployment error: {e}")
        raise
    finally:
        await deployment.stop()

    print("\nRobot deployment completed.")


if __name__ == "__main__":
    asyncio.run(main())
