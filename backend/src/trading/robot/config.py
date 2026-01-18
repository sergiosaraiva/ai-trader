"""
Trading Robot Configuration.

Defines configuration dataclasses for the trading robot.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path
import yaml


@dataclass
class ModelConfig:
    """Model configuration."""
    short_term_path: str
    medium_term_path: str
    long_term_path: str
    ensemble_weights: Dict[str, float] = field(default_factory=lambda: {
        'short_term': 0.5,
        'medium_term': 0.3,
        'long_term': 0.2,
    })


@dataclass
class BrokerConfig:
    """Broker configuration."""
    name: str = "simulation"
    api_key_env: str = ""
    secret_key_env: str = ""
    paper_trading: bool = True


@dataclass
class MonitoringConfig:
    """Monitoring configuration."""
    log_level: str = "INFO"
    metrics_interval_seconds: int = 60
    alert_email: Optional[str] = None
    alert_telegram_chat_id: Optional[str] = None


@dataclass
class SimulationConfig:
    """Simulation mode configuration."""
    initial_capital: float = 100000
    base_spread_pct: float = 0.0002
    market_impact_factor: float = 0.1
    latency_base_ms: float = 50
    latency_jitter_ms: float = 20
    commission_per_trade: float = 0.0


@dataclass
class KillSwitchConfig:
    """Kill switch configuration."""
    enabled: bool = True
    max_daily_trades: int = 50
    max_position_value: float = 100000
    emergency_close_on_disconnect: bool = True


@dataclass
class RobotConfig:
    """
    Complete trading robot configuration.

    Loaded from YAML file with sensible defaults.
    """

    # Basic settings
    name: str = "AI-Trader Robot"
    version: str = "1.0.0"
    symbol: str = "EURUSD"
    timeframe_profile: str = "trader"  # scalper, trader, investor

    # Execution
    mode: str = "simulation"  # simulation, production
    cycle_interval_seconds: int = 60

    # Risk profile
    risk_profile_name: str = "moderate"

    # Components
    model_config: ModelConfig = None
    broker_config: BrokerConfig = None
    monitoring_config: MonitoringConfig = None
    simulation_config: SimulationConfig = None
    kill_switch_config: KillSwitchConfig = None

    def __post_init__(self):
        """Initialize default configs if not provided."""
        if self.model_config is None:
            self.model_config = ModelConfig(
                short_term_path="models/short_term.pt",
                medium_term_path="models/medium_term.pt",
                long_term_path="models/long_term.pt",
            )
        if self.broker_config is None:
            self.broker_config = BrokerConfig()
        if self.monitoring_config is None:
            self.monitoring_config = MonitoringConfig()
        if self.simulation_config is None:
            self.simulation_config = SimulationConfig()
        if self.kill_switch_config is None:
            self.kill_switch_config = KillSwitchConfig()

    @classmethod
    def from_yaml(cls, path: str) -> 'RobotConfig':
        """
        Load configuration from YAML file.

        Args:
            path: Path to YAML configuration file

        Returns:
            RobotConfig instance
        """
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(config_path) as f:
            data = yaml.safe_load(f)

        robot_data = data.get('robot', {})

        # Parse nested configs
        model_config = None
        if 'model_config' in robot_data:
            model_config = ModelConfig(**robot_data.pop('model_config'))
        elif 'model_paths' in robot_data:
            model_config = ModelConfig(
                short_term_path=robot_data['model_paths'].get('short_term', ''),
                medium_term_path=robot_data['model_paths'].get('medium_term', ''),
                long_term_path=robot_data['model_paths'].get('long_term', ''),
                ensemble_weights=robot_data.get('ensemble_weights', {}),
            )
            robot_data.pop('model_paths', None)
            robot_data.pop('ensemble_weights', None)

        broker_config = None
        if 'broker_config' in robot_data:
            broker_config = BrokerConfig(**robot_data.pop('broker_config'))
        elif 'broker' in data.get('production', {}):
            prod_data = data['production']
            broker_name = prod_data.get('broker', 'simulation')
            broker_data = prod_data.get(broker_name, {})
            broker_config = BrokerConfig(
                name=broker_name,
                api_key_env=broker_data.get('api_key_env', ''),
                secret_key_env=broker_data.get('secret_key_env', ''),
                paper_trading=broker_data.get('paper', True),
            )

        monitoring_config = None
        if 'monitoring_config' in robot_data:
            monitoring_config = MonitoringConfig(**robot_data.pop('monitoring_config'))
        elif 'monitoring' in robot_data:
            mon_data = robot_data.pop('monitoring')
            monitoring_config = MonitoringConfig(
                log_level=mon_data.get('log_level', 'INFO'),
                metrics_interval_seconds=mon_data.get('metrics_interval_seconds', 60),
            )

        simulation_config = None
        if 'simulation' in data:
            sim_data = data['simulation']
            simulation_config = SimulationConfig(
                initial_capital=sim_data.get('initial_capital', 100000),
                base_spread_pct=sim_data.get('slippage', {}).get('base_spread_pct', 0.0002),
                market_impact_factor=sim_data.get('slippage', {}).get('market_impact_factor', 0.1),
                latency_base_ms=sim_data.get('latency', {}).get('base_ms', 50),
                latency_jitter_ms=sim_data.get('latency', {}).get('jitter_std_ms', 20),
            )

        kill_switch_config = None
        if 'kill_switch' in robot_data:
            ks_data = robot_data.pop('kill_switch')
            kill_switch_config = KillSwitchConfig(**ks_data)

        return cls(
            name=robot_data.get('name', 'AI-Trader Robot'),
            version=robot_data.get('version', '1.0.0'),
            symbol=robot_data.get('symbol', 'EURUSD'),
            timeframe_profile=robot_data.get('timeframe_profile', 'trader'),
            mode=robot_data.get('mode', 'simulation'),
            cycle_interval_seconds=robot_data.get('cycle_interval_seconds', 60),
            risk_profile_name=robot_data.get('risk_profile', 'moderate'),
            model_config=model_config,
            broker_config=broker_config,
            monitoring_config=monitoring_config,
            simulation_config=simulation_config,
            kill_switch_config=kill_switch_config,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'version': self.version,
            'symbol': self.symbol,
            'timeframe_profile': self.timeframe_profile,
            'mode': self.mode,
            'cycle_interval_seconds': self.cycle_interval_seconds,
            'risk_profile_name': self.risk_profile_name,
        }
