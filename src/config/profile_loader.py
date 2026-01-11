"""
Profile Loader Module
====================

Loads and merges trading profiles with inheritance support.

Profile Hierarchy:
    base.yaml (default settings)
        └── scalper.yaml (inherits from base)
        └── trader.yaml (inherits from base)
        └── investor.yaml (inherits from base)

Asset Profiles:
    assets/forex.yaml
    assets/crypto.yaml
    assets/stocks.yaml

Usage:
    from src.config.profile_loader import ProfileLoader

    # Load a trading profile
    loader = ProfileLoader()
    config = loader.load_profile("trader")

    # Load with asset-specific settings
    config = loader.load_profile("trader", asset="forex")

    # Access configuration
    timeframes = config.get("timeframes")
    indicators = config.get("indicators")
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import copy
import yaml


class ProfileLoader:
    """
    Loads and merges trading profiles with inheritance support.

    Attributes:
        profiles_dir: Path to the profiles directory
        assets_dir: Path to the asset profiles directory
    """

    def __init__(
        self,
        profiles_dir: Optional[Union[str, Path]] = None,
        assets_dir: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize the ProfileLoader.

        Args:
            profiles_dir: Path to profiles directory. Defaults to configs/profiles/
            assets_dir: Path to asset profiles directory. Defaults to configs/profiles/assets/
        """
        if profiles_dir is None:
            # Default to project root/configs/profiles
            self.profiles_dir = Path(__file__).parent.parent.parent / "configs" / "profiles"
        else:
            self.profiles_dir = Path(profiles_dir)

        if assets_dir is None:
            self.assets_dir = self.profiles_dir / "assets"
        else:
            self.assets_dir = Path(assets_dir)

        self._cache: Dict[str, Dict] = {}

    def load_profile(
        self,
        profile_name: str,
        asset: Optional[str] = None,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """
        Load a trading profile with inheritance resolution.

        Args:
            profile_name: Name of the profile (e.g., "trader", "scalper", "investor")
            asset: Optional asset class (e.g., "forex", "crypto", "stocks")
            use_cache: Whether to use cached profiles

        Returns:
            Merged configuration dictionary

        Raises:
            FileNotFoundError: If profile file doesn't exist
            ValueError: If circular inheritance is detected
        """
        cache_key = f"{profile_name}:{asset or 'none'}"

        if use_cache and cache_key in self._cache:
            return copy.deepcopy(self._cache[cache_key])

        # Load and resolve profile inheritance
        config = self._load_with_inheritance(profile_name)

        # Merge asset profile if specified
        if asset:
            asset_config = self._load_asset_profile(asset)
            config = self._merge_asset_config(config, asset_config)

        if use_cache:
            self._cache[cache_key] = config

        return copy.deepcopy(config)

    def _load_with_inheritance(
        self,
        profile_name: str,
        visited: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Load a profile and resolve its inheritance chain.

        Args:
            profile_name: Name of the profile to load
            visited: List of already visited profiles (for circular detection)

        Returns:
            Fully merged configuration

        Raises:
            ValueError: If circular inheritance is detected
        """
        if visited is None:
            visited = []

        if profile_name in visited:
            raise ValueError(
                f"Circular inheritance detected: {' -> '.join(visited)} -> {profile_name}"
            )

        visited.append(profile_name)

        # Load the profile file
        profile_path = self.profiles_dir / f"{profile_name}.yaml"
        if not profile_path.exists():
            raise FileNotFoundError(f"Profile not found: {profile_path}")

        with open(profile_path, "r") as f:
            profile_data = yaml.safe_load(f) or {}

        # Get the profile section
        profile_section = profile_data.get("profile", profile_data)

        # Check for inheritance
        parent_name = profile_section.get("inherits")

        if parent_name:
            # Load parent profile first
            parent_config = self._load_with_inheritance(parent_name, visited.copy())

            # Merge child into parent (child overrides parent)
            merged = self._deep_merge(parent_config, profile_section)
            return merged
        else:
            # No inheritance, return as is
            return profile_section

    def _load_asset_profile(self, asset_name: str) -> Dict[str, Any]:
        """
        Load an asset-specific profile.

        Args:
            asset_name: Name of the asset class (e.g., "forex", "crypto")

        Returns:
            Asset configuration dictionary

        Raises:
            FileNotFoundError: If asset profile doesn't exist
        """
        asset_path = self.assets_dir / f"{asset_name}.yaml"
        if not asset_path.exists():
            raise FileNotFoundError(f"Asset profile not found: {asset_path}")

        with open(asset_path, "r") as f:
            asset_data = yaml.safe_load(f) or {}

        return asset_data.get("asset", asset_data)

    def _merge_asset_config(
        self,
        profile_config: Dict[str, Any],
        asset_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Merge asset-specific settings into a profile.

        Asset settings are merged into specific sections:
        - asset.sessions -> profile.sessions (if not overridden)
        - asset.instruments -> profile.asset.instruments
        - asset.costs -> profile.asset.costs
        - asset.risk_adjustments -> merged with profile.risk
        - asset.indicator_adjustments -> merged with profile.indicators

        Args:
            profile_config: The loaded profile configuration
            asset_config: The asset-specific configuration

        Returns:
            Merged configuration
        """
        merged = copy.deepcopy(profile_config)

        # Merge sessions if profile doesn't override
        if "sessions" in asset_config and not merged.get("sessions", {}).get("_override"):
            merged["sessions"] = self._deep_merge(
                merged.get("sessions", {}),
                asset_config.get("sessions", {}),
            )

        # Store asset info in profile
        if "asset" not in merged:
            merged["asset"] = {}
        merged["asset"] = self._deep_merge(merged["asset"], asset_config)

        # Apply risk adjustments
        if "risk_adjustments" in asset_config:
            if "risk" not in merged:
                merged["risk"] = {}
            merged["risk"]["asset_adjustments"] = asset_config["risk_adjustments"]

        # Apply indicator adjustments
        if "indicator_adjustments" in asset_config:
            merged["indicator_adjustments"] = asset_config["indicator_adjustments"]

        # Apply news settings
        if "news" in asset_config:
            if "news" not in merged:
                merged["news"] = {}
            merged["news"] = self._deep_merge(merged["news"], asset_config["news"])

        return merged

    def _deep_merge(
        self,
        base: Dict[str, Any],
        override: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Deep merge two dictionaries. Override values take precedence.

        Args:
            base: Base dictionary
            override: Dictionary with values to override

        Returns:
            Merged dictionary
        """
        result = copy.deepcopy(base)

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                # Recursively merge dictionaries
                result[key] = self._deep_merge(result[key], value)
            else:
                # Override value
                result[key] = copy.deepcopy(value)

        return result

    def list_profiles(self) -> List[str]:
        """
        List available profile names.

        Returns:
            List of profile names (without .yaml extension)
        """
        profiles = []
        for path in self.profiles_dir.glob("*.yaml"):
            profiles.append(path.stem)
        return sorted(profiles)

    def list_assets(self) -> List[str]:
        """
        List available asset profiles.

        Returns:
            List of asset names (without .yaml extension)
        """
        assets = []
        if self.assets_dir.exists():
            for path in self.assets_dir.glob("*.yaml"):
                assets.append(path.stem)
        return sorted(assets)

    def get_profile_info(self, profile_name: str) -> Dict[str, str]:
        """
        Get basic info about a profile without full loading.

        Args:
            profile_name: Name of the profile

        Returns:
            Dictionary with name, version, description, inherits
        """
        profile_path = self.profiles_dir / f"{profile_name}.yaml"
        if not profile_path.exists():
            raise FileNotFoundError(f"Profile not found: {profile_path}")

        with open(profile_path, "r") as f:
            profile_data = yaml.safe_load(f) or {}

        profile_section = profile_data.get("profile", profile_data)

        return {
            "name": profile_section.get("name", profile_name),
            "version": profile_section.get("version", "unknown"),
            "description": profile_section.get("description", ""),
            "inherits": profile_section.get("inherits"),
        }

    def clear_cache(self):
        """Clear the profile cache."""
        self._cache.clear()

    def validate_profile(self, profile_name: str) -> Dict[str, Any]:
        """
        Validate a profile configuration.

        Args:
            profile_name: Name of the profile to validate

        Returns:
            Dictionary with validation results:
            - valid: bool
            - errors: list of error messages
            - warnings: list of warning messages
        """
        errors = []
        warnings = []

        try:
            config = self.load_profile(profile_name, use_cache=False)
        except Exception as e:
            return {
                "valid": False,
                "errors": [f"Failed to load profile: {str(e)}"],
                "warnings": [],
            }

        # Required sections
        required_sections = ["timeframes", "indicators", "signals", "risk"]
        for section in required_sections:
            if section not in config:
                errors.append(f"Missing required section: {section}")

        # Validate timeframes
        if "timeframes" in config:
            timeframes = config["timeframes"]
            required_timeframes = ["short_term", "medium_term", "long_term"]
            for tf in required_timeframes:
                if tf not in timeframes:
                    errors.append(f"Missing timeframe: {tf}")
                else:
                    tf_config = timeframes[tf]
                    if "candle_minutes" not in tf_config:
                        errors.append(f"Timeframe {tf} missing candle_minutes")
                    if "input_window" not in tf_config:
                        warnings.append(f"Timeframe {tf} missing input_window")

        # Validate ensemble weights sum to 1.0
        if "ensemble" in config and "base_weights" in config["ensemble"]:
            weights = config["ensemble"]["base_weights"]
            total = sum(weights.values())
            if abs(total - 1.0) > 0.01:
                errors.append(f"Ensemble base_weights sum to {total}, should be 1.0")

        # Validate signal thresholds
        if "signals" in config:
            signals = config["signals"]
            buy = signals.get("buy_threshold", 0)
            sell = signals.get("sell_threshold", 0)
            if buy <= 0:
                warnings.append(f"buy_threshold ({buy}) should be positive")
            if sell >= 0:
                warnings.append(f"sell_threshold ({sell}) should be negative")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
        }


class ProfileConfig:
    """
    Wrapper class for easy access to profile configuration values.

    Usage:
        loader = ProfileLoader()
        config = ProfileConfig(loader.load_profile("trader", asset="forex"))

        # Access values with dot notation
        short_term_minutes = config.timeframes.short_term.candle_minutes
        rsi_periods = config.indicators.short_term.momentum.rsi.periods
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize with a configuration dictionary.

        Args:
            config: Configuration dictionary from ProfileLoader
        """
        self._config = config

    def __getattr__(self, name: str) -> Any:
        """Get attribute with dot notation."""
        if name.startswith("_"):
            return super().__getattribute__(name)

        value = self._config.get(name)
        if isinstance(value, dict):
            return ProfileConfig(value)
        return value

    def __getitem__(self, key: str) -> Any:
        """Get item with bracket notation."""
        value = self._config.get(key)
        if isinstance(value, dict):
            return ProfileConfig(value)
        return value

    def get(self, key: str, default: Any = None) -> Any:
        """Get value with optional default."""
        value = self._config.get(key, default)
        if isinstance(value, dict):
            return ProfileConfig(value)
        return value

    def to_dict(self) -> Dict[str, Any]:
        """Return the underlying dictionary."""
        return copy.deepcopy(self._config)

    def keys(self):
        """Return configuration keys."""
        return self._config.keys()

    def __contains__(self, key: str) -> bool:
        """Check if key exists."""
        return key in self._config

    def __repr__(self) -> str:
        """String representation."""
        return f"ProfileConfig({list(self._config.keys())})"


# Convenience functions
def load_profile(
    profile_name: str,
    asset: Optional[str] = None,
) -> ProfileConfig:
    """
    Convenience function to load a profile.

    Args:
        profile_name: Name of the profile (e.g., "trader")
        asset: Optional asset class (e.g., "forex")

    Returns:
        ProfileConfig instance
    """
    loader = ProfileLoader()
    config = loader.load_profile(profile_name, asset)
    return ProfileConfig(config)


def get_timeframe_config(
    profile_name: str,
    timeframe: str = "short_term",
    asset: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Get timeframe-specific configuration.

    Args:
        profile_name: Name of the profile
        timeframe: Timeframe key (short_term, medium_term, long_term)
        asset: Optional asset class

    Returns:
        Timeframe configuration dictionary
    """
    loader = ProfileLoader()
    config = loader.load_profile(profile_name, asset)
    return config.get("timeframes", {}).get(timeframe, {})


def get_indicator_config(
    profile_name: str,
    timeframe: str = "short_term",
    asset: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Get indicator configuration for a specific timeframe.

    Args:
        profile_name: Name of the profile
        timeframe: Timeframe key (short_term, medium_term, long_term)
        asset: Optional asset class

    Returns:
        Indicator configuration dictionary
    """
    loader = ProfileLoader()
    config = loader.load_profile(profile_name, asset)
    return config.get("indicators", {}).get(timeframe, {})
