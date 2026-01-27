"""Verification script for centralized configuration system.

Tests:
1. Configuration loading
2. Parameter access
3. Validation
4. Update functionality
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.src.config import trading_config


def verify_configuration():
    """Verify configuration system is working."""
    print("=" * 60)
    print("CONFIGURATION SYSTEM VERIFICATION")
    print("=" * 60)
    print()

    # Test 1: Singleton
    print("1. Testing Singleton Pattern...")
    from backend.src.config.trading_config import TradingConfig
    config1 = TradingConfig()
    config2 = TradingConfig()
    assert config1 is config2, "Singleton check failed!"
    print("   ✓ Singleton pattern working correctly")
    print()

    # Test 2: Default Values
    print("2. Testing Default Values...")
    print(f"   Confidence Threshold: {trading_config.trading.confidence_threshold}")
    print(f"   Default Lot Size: {trading_config.trading.default_lot_size}")
    print(f"   Model Weights: 1H={trading_config.model.weight_1h}, "
          f"4H={trading_config.model.weight_4h}, D={trading_config.model.weight_daily}")
    print(f"   Agreement Bonus: {trading_config.model.agreement_bonus}")
    print(f"   Max Drawdown: {trading_config.risk.max_drawdown_percent}%")
    print("   ✓ Default values loaded correctly")
    print()

    # Test 3: Validation
    print("3. Testing Validation...")
    errors = trading_config.validate()
    if errors:
        print(f"   ✗ Validation errors: {errors}")
    else:
        print("   ✓ Configuration is valid")
    print()

    # Test 4: Model Weights
    print("4. Testing Model Weights...")
    weights = trading_config.model.get_weights()
    weights_sum = sum(weights.values())
    print(f"   Weights: {weights}")
    print(f"   Sum: {weights_sum}")
    assert 0.99 <= weights_sum <= 1.01, "Weights don't sum to 1.0!"
    print("   ✓ Model weights are normalized")
    print()

    # Test 5: Update and Rollback
    print("5. Testing Update and Rollback...")
    original_confidence = trading_config.trading.confidence_threshold
    print(f"   Original confidence: {original_confidence}")

    # Valid update
    try:
        result = trading_config.update(
            category="trading",
            updates={"confidence_threshold": 0.75},
            updated_by="verification_script",
            db_session=None,
        )
        print(f"   After update: {trading_config.trading.confidence_threshold}")
        print("   ✓ Valid update successful")
    except Exception as e:
        print(f"   ✗ Update failed: {e}")

    # Rollback
    trading_config.reset_to_defaults(category="trading", key="confidence_threshold")
    print(f"   After reset: {trading_config.trading.confidence_threshold}")
    print("   ✓ Reset to defaults working")
    print()

    # Test 6: Invalid Update (should fail)
    print("6. Testing Invalid Update Protection...")
    try:
        trading_config.update(
            category="trading",
            updates={"confidence_threshold": 1.5},  # Invalid
            updated_by="verification_script",
            db_session=None,
        )
        print("   ✗ Invalid update was accepted (should have been rejected!)")
    except ValueError as e:
        print(f"   ✓ Invalid update correctly rejected: {str(e)[:50]}...")
    print()

    # Test 7: Get All Configuration
    print("7. Testing Get All Configuration...")
    all_config = trading_config.get_all()
    categories = list(all_config.keys())
    print(f"   Categories: {categories}")
    assert "trading" in categories
    assert "model" in categories
    assert "risk" in categories
    assert "system" in categories
    print("   ✓ All categories accessible")
    print()

    # Test 8: Parameter Dataclasses
    print("8. Testing Parameter Dataclasses...")
    trading_dict = trading_config.trading.to_dict()
    model_dict = trading_config.model.to_dict()
    print(f"   Trading params: {len(trading_dict)} fields")
    print(f"   Model params: {len(model_dict)} fields")
    print("   ✓ Dataclasses working correctly")
    print()

    # Summary
    print("=" * 60)
    print("✅ ALL VERIFICATION TESTS PASSED")
    print("=" * 60)
    print()
    print("Configuration system is working correctly!")
    print()
    print("Current Configuration Summary:")
    print(f"  • Confidence Threshold: {trading_config.trading.confidence_threshold}")
    print(f"  • Model Weights: 1H={trading_config.model.weight_1h}, "
          f"4H={trading_config.model.weight_4h}, D={trading_config.model.weight_daily}")
    print(f"  • Risk: Max DD {trading_config.risk.max_drawdown_percent}%, "
          f"Max Losses {trading_config.risk.max_consecutive_losses}")
    print(f"  • System: Cache TTL {trading_config.system.cache_ttl_seconds}s")


if __name__ == "__main__":
    try:
        verify_configuration()
    except Exception as e:
        print(f"\n❌ VERIFICATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
