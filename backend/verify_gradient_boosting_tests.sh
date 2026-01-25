#!/bin/bash
# Verification script for LightGBM/CatBoost implementation tests

set -e

echo "======================================"
echo "Gradient Boosting Tests Verification"
echo "======================================"
echo ""

# Activate virtual environment
if [ -d "../.venv" ]; then
    echo "✓ Activating virtual environment..."
    source ../.venv/bin/activate
else
    echo "✗ Virtual environment not found at ../.venv"
    exit 1
fi

# Check Python version
echo ""
echo "Python version:"
python3 --version
echo ""

# Verify syntax of modified files
echo "======================================"
echo "1. Syntax Verification"
echo "======================================"
echo ""

files=(
    "src/models/multi_timeframe/improved_model.py"
    "src/models/multi_timeframe/mtf_ensemble.py"
    "scripts/train_mtf_ensemble.py"
    "scripts/compare_gradient_boosting.py"
)

for file in "${files[@]}"; do
    if python3 -m py_compile "$file" 2>/dev/null; then
        echo "✓ $file"
    else
        echo "✗ $file - SYNTAX ERROR"
        exit 1
    fi
done

# Check framework availability
echo ""
echo "======================================"
echo "2. Framework Availability"
echo "======================================"
echo ""

python3 -c "
from src.models.multi_timeframe.improved_model import HAS_LIGHTGBM, HAS_CATBOOST

print(f'LightGBM installed: {HAS_LIGHTGBM}')
print(f'CatBoost installed: {HAS_CATBOOST}')
print(f'XGBoost installed: True (required)')
"

# Verify imports work
echo ""
echo "======================================"
echo "3. Import Verification"
echo "======================================"
echo ""

python3 -c "
from src.models.multi_timeframe.improved_model import (
    ImprovedModelConfig,
    ImprovedTimeframeModel,
    HAS_LIGHTGBM,
    HAS_CATBOOST,
)
from src.models.multi_timeframe.mtf_ensemble import MTFEnsembleConfig

print('✓ All imports successful')
"

# Test configuration
echo ""
echo "======================================"
echo "4. Configuration Tests"
echo "======================================"
echo ""

python3 -c "
from src.models.multi_timeframe.improved_model import ImprovedModelConfig
from src.models.multi_timeframe.mtf_ensemble import MTFEnsembleConfig

# Test ImprovedModelConfig
config_xgb = ImprovedModelConfig(name='test', base_timeframe='1H')
assert config_xgb.model_type == 'xgboost', 'Default should be xgboost'
print('✓ Default model_type is xgboost')

config_lgb = ImprovedModelConfig(name='test', base_timeframe='1H', model_type='lightgbm')
assert config_lgb.model_type == 'lightgbm', 'Should accept lightgbm'
print('✓ Config accepts lightgbm')

config_cb = ImprovedModelConfig(name='test', base_timeframe='1H', model_type='catboost')
assert config_cb.model_type == 'catboost', 'Should accept catboost'
print('✓ Config accepts catboost')

# Test MTFEnsembleConfig
mtf_config = MTFEnsembleConfig(model_type='lightgbm')
assert mtf_config.model_type == 'lightgbm', 'MTF should accept lightgbm'
print('✓ MTFEnsembleConfig accepts model_type')
"

# Test error handling
echo ""
echo "======================================"
echo "5. Error Handling Tests"
echo "======================================"
echo ""

python3 -c "
from src.models.multi_timeframe.improved_model import (
    ImprovedModelConfig,
    ImprovedTimeframeModel,
    HAS_LIGHTGBM,
    HAS_CATBOOST,
)

# Only test error messages if frameworks are not installed
if not HAS_LIGHTGBM:
    config = ImprovedModelConfig(name='test', base_timeframe='1H', model_type='lightgbm')
    model_wrapper = ImprovedTimeframeModel(config)
    try:
        model_wrapper._create_model()
        print('✗ Should have raised ImportError for LightGBM')
    except ImportError as e:
        if 'LightGBM is not installed' in str(e):
            print('✓ LightGBM ImportError message correct')
        else:
            print(f'✗ Wrong error message: {e}')
else:
    print('⏭️  LightGBM installed - skipping error test')

if not HAS_CATBOOST:
    config = ImprovedModelConfig(name='test', base_timeframe='1H', model_type='catboost')
    model_wrapper = ImprovedTimeframeModel(config)
    try:
        model_wrapper._create_model()
        print('✗ Should have raised ImportError for CatBoost')
    except ImportError as e:
        if 'CatBoost is not installed' in str(e):
            print('✓ CatBoost ImportError message correct')
        else:
            print(f'✗ Wrong error message: {e}')
else:
    print('⏭️  CatBoost installed - skipping error test')
"

# Run pytest tests
echo ""
echo "======================================"
echo "6. Running Test Suite"
echo "======================================"
echo ""

pytest tests/unit/models/test_gradient_boosting_frameworks.py -v --tb=short

# Summary
echo ""
echo "======================================"
echo "Verification Complete!"
echo "======================================"
echo ""

python3 -c "
from src.models.multi_timeframe.improved_model import HAS_LIGHTGBM, HAS_CATBOOST

if HAS_LIGHTGBM and HAS_CATBOOST:
    print('✅ All frameworks installed - Full test suite available')
    print('   Expected: 26 tests passed, 0 skipped')
else:
    print('⚠️  Not all frameworks installed - Some tests skipped')
    print('   Expected: 18 tests passed, 8 skipped')
    print('')
    print('To enable all tests, install:')
    if not HAS_LIGHTGBM:
        print('   pip install lightgbm>=4.0.0')
    if not HAS_CATBOOST:
        print('   pip install catboost>=1.2.0')
"

echo ""
echo "Run 'pytest tests/unit/models/test_gradient_boosting_frameworks.py -v' for detailed results"
echo ""
