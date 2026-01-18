"""
Risk Profile Definitions for Trading Robot.

Provides five predefined risk profiles from ultra-conservative to ultra-aggressive,
enabling configurable risk tolerance without code changes.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Any
import yaml
from pathlib import Path


class RiskLevel(Enum):
    """Risk level enumeration."""
    ULTRA_CONSERVATIVE = "ultra_conservative"
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    ULTRA_AGGRESSIVE = "ultra_aggressive"


@dataclass
class RiskProfile:
    """
    Comprehensive risk profile for trading robot.

    Controls all risk-related parameters including:
    - Confidence thresholds for trading
    - Position sizing limits
    - Loss limits (daily, weekly, drawdown)
    - Circuit breaker settings
    - Portfolio exposure limits
    """

    # Profile identification
    name: str
    description: str
    level: RiskLevel

    # Confidence thresholds (from Beta output)
    min_confidence_to_trade: float      # Below this = HOLD
    full_position_confidence: float     # Above this = full size

    # Position sizing
    max_position_pct: float             # Max % of equity per position
    base_position_pct: float            # Starting position size
    kelly_fraction: float               # Fraction of Kelly criterion to use

    # Loss limits
    max_daily_loss_pct: float           # Daily loss limit
    max_weekly_loss_pct: float          # Weekly loss limit
    max_drawdown_pct: float             # Maximum drawdown before halt

    # Circuit breakers
    consecutive_loss_halt: int          # N losses in a row = halt
    cooldown_hours: int                 # Hours to wait after halt

    # Portfolio limits
    max_portfolio_heat: float           # Total risk exposure
    max_correlation_exposure: float     # Max in correlated assets
    max_positions: int = 5              # Maximum concurrent positions

    # Trade limits
    max_trades_per_day: int = 20        # Maximum daily trades
    min_trade_interval_seconds: int = 300  # Minimum between trades

    # Recovery settings
    recovery_testing_trades: int = 5    # Trades needed in testing phase
    recovery_wins_to_restore: int = 3   # Consecutive wins to restore full trading

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'description': self.description,
            'level': self.level.value,
            'min_confidence_to_trade': self.min_confidence_to_trade,
            'full_position_confidence': self.full_position_confidence,
            'max_position_pct': self.max_position_pct,
            'base_position_pct': self.base_position_pct,
            'kelly_fraction': self.kelly_fraction,
            'max_daily_loss_pct': self.max_daily_loss_pct,
            'max_weekly_loss_pct': self.max_weekly_loss_pct,
            'max_drawdown_pct': self.max_drawdown_pct,
            'consecutive_loss_halt': self.consecutive_loss_halt,
            'cooldown_hours': self.cooldown_hours,
            'max_portfolio_heat': self.max_portfolio_heat,
            'max_correlation_exposure': self.max_correlation_exposure,
            'max_positions': self.max_positions,
            'max_trades_per_day': self.max_trades_per_day,
            'min_trade_interval_seconds': self.min_trade_interval_seconds,
        }


# Predefined Risk Profiles
RISK_PROFILES: Dict[RiskLevel, RiskProfile] = {
    RiskLevel.ULTRA_CONSERVATIVE: RiskProfile(
        name="Ultra Conservative",
        description="Maximum capital protection, minimal trading. Only trade on extremely clear signals.",
        level=RiskLevel.ULTRA_CONSERVATIVE,
        min_confidence_to_trade=0.85,
        full_position_confidence=0.95,
        max_position_pct=0.01,
        base_position_pct=0.005,
        kelly_fraction=0.125,
        max_daily_loss_pct=0.005,
        max_weekly_loss_pct=0.015,
        max_drawdown_pct=0.05,
        consecutive_loss_halt=2,
        cooldown_hours=48,
        max_portfolio_heat=0.03,
        max_correlation_exposure=0.02,
        max_positions=3,
        max_trades_per_day=5,
        min_trade_interval_seconds=1800,
        recovery_testing_trades=10,
        recovery_wins_to_restore=5,
    ),

    RiskLevel.CONSERVATIVE: RiskProfile(
        name="Conservative",
        description="Strong risk controls, selective trading. Good for long-term capital preservation.",
        level=RiskLevel.CONSERVATIVE,
        min_confidence_to_trade=0.75,
        full_position_confidence=0.90,
        max_position_pct=0.02,
        base_position_pct=0.01,
        kelly_fraction=0.25,
        max_daily_loss_pct=0.01,
        max_weekly_loss_pct=0.03,
        max_drawdown_pct=0.10,
        consecutive_loss_halt=3,
        cooldown_hours=24,
        max_portfolio_heat=0.06,
        max_correlation_exposure=0.04,
        max_positions=4,
        max_trades_per_day=10,
        min_trade_interval_seconds=900,
        recovery_testing_trades=7,
        recovery_wins_to_restore=4,
    ),

    RiskLevel.MODERATE: RiskProfile(
        name="Moderate",
        description="Balanced approach with industry-standard risk. Good for active trading.",
        level=RiskLevel.MODERATE,
        min_confidence_to_trade=0.65,
        full_position_confidence=0.85,
        max_position_pct=0.05,
        base_position_pct=0.02,
        kelly_fraction=0.50,
        max_daily_loss_pct=0.03,
        max_weekly_loss_pct=0.07,
        max_drawdown_pct=0.15,
        consecutive_loss_halt=5,
        cooldown_hours=12,
        max_portfolio_heat=0.10,
        max_correlation_exposure=0.06,
        max_positions=5,
        max_trades_per_day=20,
        min_trade_interval_seconds=300,
        recovery_testing_trades=5,
        recovery_wins_to_restore=3,
    ),

    RiskLevel.AGGRESSIVE: RiskProfile(
        name="Aggressive",
        description="Higher risk tolerance for higher potential returns. For experienced traders.",
        level=RiskLevel.AGGRESSIVE,
        min_confidence_to_trade=0.55,
        full_position_confidence=0.80,
        max_position_pct=0.10,
        base_position_pct=0.05,
        kelly_fraction=0.75,
        max_daily_loss_pct=0.05,
        max_weekly_loss_pct=0.10,
        max_drawdown_pct=0.25,
        consecutive_loss_halt=7,
        cooldown_hours=6,
        max_portfolio_heat=0.15,
        max_correlation_exposure=0.10,
        max_positions=7,
        max_trades_per_day=30,
        min_trade_interval_seconds=120,
        recovery_testing_trades=3,
        recovery_wins_to_restore=2,
    ),

    RiskLevel.ULTRA_AGGRESSIVE: RiskProfile(
        name="Ultra Aggressive",
        description="Maximum trading activity. USE WITH CAUTION - high risk of significant losses.",
        level=RiskLevel.ULTRA_AGGRESSIVE,
        min_confidence_to_trade=0.52,
        full_position_confidence=0.75,
        max_position_pct=0.15,
        base_position_pct=0.08,
        kelly_fraction=1.0,
        max_daily_loss_pct=0.08,
        max_weekly_loss_pct=0.15,
        max_drawdown_pct=0.35,
        consecutive_loss_halt=10,
        cooldown_hours=4,
        max_portfolio_heat=0.25,
        max_correlation_exposure=0.15,
        max_positions=10,
        max_trades_per_day=50,
        min_trade_interval_seconds=60,
        recovery_testing_trades=2,
        recovery_wins_to_restore=1,
    ),
}


def get_risk_profile(level: RiskLevel) -> RiskProfile:
    """Get predefined risk profile by level."""
    return RISK_PROFILES[level]


def load_risk_profile(
    name: str,
    config_path: Optional[str] = None,
) -> RiskProfile:
    """
    Load risk profile by name.

    Args:
        name: Profile name (e.g., 'moderate', 'conservative')
        config_path: Optional path to custom config file

    Returns:
        RiskProfile instance
    """
    # Try to match to predefined profile
    name_lower = name.lower().replace('-', '_').replace(' ', '_')

    for level in RiskLevel:
        if level.value == name_lower:
            return RISK_PROFILES[level]

    # Try to load from config file
    if config_path:
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file) as f:
                config = yaml.safe_load(f)
                if name in config.get('risk_profiles', {}):
                    profile_data = config['risk_profiles'][name]
                    return RiskProfile(
                        level=RiskLevel(profile_data.get('level', 'moderate')),
                        **profile_data
                    )

    # Default to moderate
    print(f"Warning: Unknown risk profile '{name}', defaulting to moderate")
    return RISK_PROFILES[RiskLevel.MODERATE]


def create_custom_profile(
    base_level: RiskLevel,
    **overrides,
) -> RiskProfile:
    """
    Create custom profile based on predefined level with overrides.

    Args:
        base_level: Base risk level to start from
        **overrides: Parameters to override

    Returns:
        Custom RiskProfile
    """
    base = RISK_PROFILES[base_level]
    profile_dict = base.to_dict()
    profile_dict.update(overrides)
    profile_dict['name'] = overrides.get('name', f"Custom ({base.name})")
    profile_dict['description'] = overrides.get('description', f"Custom profile based on {base.name}")
    profile_dict['level'] = base_level

    return RiskProfile(**profile_dict)


# Convenience function for getting profile comparison
def compare_profiles() -> str:
    """Generate comparison table of all profiles."""
    lines = [
        "| Parameter | Ultra-Cons | Conservative | Moderate | Aggressive | Ultra-Agg |",
        "|-----------|------------|--------------|----------|------------|-----------|",
    ]

    params = [
        ('Min Confidence', 'min_confidence_to_trade', lambda x: f"{x:.0%}"),
        ('Max Position', 'max_position_pct', lambda x: f"{x:.0%}"),
        ('Daily Loss Limit', 'max_daily_loss_pct', lambda x: f"{x:.1%}"),
        ('Max Drawdown', 'max_drawdown_pct', lambda x: f"{x:.0%}"),
        ('Loss Streak Halt', 'consecutive_loss_halt', lambda x: str(x)),
        ('Kelly Fraction', 'kelly_fraction', lambda x: f"{x:.2f}"),
    ]

    for param_name, attr, formatter in params:
        values = [formatter(getattr(RISK_PROFILES[level], attr)) for level in RiskLevel]
        lines.append(f"| {param_name} | {' | '.join(values)} |")

    return '\n'.join(lines)
