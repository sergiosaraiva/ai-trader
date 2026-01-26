#!/bin/bash
# Docker Volume Cleanup Script
# Reduces disk space usage by removing unnecessary data

set -e

echo "🧹 Docker Volume Cleanup Script"
echo "================================"
echo ""

# Calculate sizes before cleanup
BEFORE_MODELS=$(du -sm backend/models | cut -f1)
BEFORE_DATA=$(du -sm backend/data | cut -f1)

echo "📊 Current sizes:"
echo "  Models: ${BEFORE_MODELS}MB"
echo "  Data: ${BEFORE_DATA}MB"
echo ""

# 1. Remove tar archive parts (~550MB)
echo "🗑️  Step 1: Removing tar archive parts..."
if ls backend/models/models_trained.tar.gz.part_* 1> /dev/null 2>&1; then
    rm -f backend/models/models_trained.tar.gz.part_*
    echo "  ✅ Removed tar archive parts"
else
    echo "  ⏭️  No tar parts found"
fi

# 2. Remove crypto data (~174MB)
echo ""
echo "🗑️  Step 2: Removing crypto data..."
if [ -d "backend/data/crypto" ]; then
    rm -rf backend/data/crypto
    echo "  ✅ Removed crypto data (project uses forex only)"
else
    echo "  ⏭️  No crypto directory found"
fi

# 3. Remove cache directory (~53MB)
echo ""
echo "🗑️  Step 3: Removing cache..."
if [ -d "backend/data/cache" ]; then
    rm -rf backend/data/cache
    echo "  ✅ Removed cache (will be auto-regenerated)"
else
    echo "  ⏭️  No cache directory found"
fi

# 4. Remove experimental MTF model directories (~15MB)
echo ""
echo "🗑️  Step 4: Removing experimental model directories..."
EXPERIMENTAL_DIRS=(
    "mtf_ensemble_all_sentiment"
    "mtf_ensemble_backup_v1"
    "mtf_ensemble_baseline"
    "mtf_ensemble_baseline_backup"
    "mtf_ensemble_baseline_check"
    "mtf_ensemble_daily_sentiment"
    "mtf_ensemble_epu_daily"
    "mtf_ensemble_gdelt"
    "mtf_ensemble_pre_wavelet_backup"
    "mtf_ensemble_sentiment"
    "mtf_ensemble_sentiment_daily_only"
    "mtf_ensemble_shallow_fast"
    "mtf_ensemble_stacking"
    "mtf_ensemble_us_sentiment"
    "mtf_ensemble_wavelet"
)

REMOVED_COUNT=0
for dir in "${EXPERIMENTAL_DIRS[@]}"; do
    if [ -d "backend/models/$dir" ]; then
        rm -rf "backend/models/$dir"
        ((REMOVED_COUNT++))
    fi
done
echo "  ✅ Removed $REMOVED_COUNT experimental directories"

# 5. Remove old WFO and HPO test directories
echo ""
echo "🗑️  Step 5: Removing old test directories..."
TEST_DIRS=(
    "hpo_test_even_shallower"
    "hpo_test_higher_lr"
    "hpo_test_more_trees_shallow"
    "hpo_wfo_test"
    "wfo_baseline"
    "wfo_baseline_comparison"
    "wfo_rfecv_comparison"
    "wfo_rfecv_validation"
    "wfo_tier1_validation"
    "wfo_tier2_no_regime"
    "wfo_tier2_validation"
    "wfo_stacking"
)

REMOVED_TEST_COUNT=0
for dir in "${TEST_DIRS[@]}"; do
    if [ -d "backend/models/$dir" ]; then
        rm -rf "backend/models/$dir"
        ((REMOVED_TEST_COUNT++))
    fi
done
echo "  ✅ Removed $REMOVED_TEST_COUNT test directories"

# 6. Optional: Remove large trained models directories
echo ""
echo "⚠️  Step 6: Large trained model directories (manual review recommended):"
echo "  - backend/models/trained/ (351MB)"
echo "  - backend/models/practical_e2e/ (156MB)"
echo "  - backend/models/individual_models/ (144MB)"
echo "  - backend/models/pipeline_run/ (46MB)"
echo ""
echo "  To remove these, run:"
echo "    rm -rf backend/models/trained"
echo "    rm -rf backend/models/practical_e2e"
echo "    rm -rf backend/models/individual_models"
echo "    rm -rf backend/models/pipeline_run"

# Calculate sizes after cleanup
AFTER_MODELS=$(du -sm backend/models 2>/dev/null | cut -f1 || echo "0")
AFTER_DATA=$(du -sm backend/data 2>/dev/null | cut -f1 || echo "0")

SAVED_MODELS=$((BEFORE_MODELS - AFTER_MODELS))
SAVED_DATA=$((BEFORE_DATA - AFTER_DATA))
TOTAL_SAVED=$((SAVED_MODELS + SAVED_DATA))

echo ""
echo "✨ Cleanup Complete!"
echo "=================="
echo "  Models: ${BEFORE_MODELS}MB → ${AFTER_MODELS}MB (saved ${SAVED_MODELS}MB)"
echo "  Data: ${BEFORE_DATA}MB → ${AFTER_DATA}MB (saved ${SAVED_DATA}MB)"
echo "  Total saved: ${TOTAL_SAVED}MB"
echo ""
echo "💾 Production models preserved:"
echo "  ✅ backend/models/mtf_ensemble/ (production models)"
echo "  ✅ backend/models/wfo_validation/ (validation results)"
echo "  ✅ backend/data/forex/ (EUR/USD data)"
echo "  ✅ backend/data/sentiment/ (sentiment data)"
echo ""
echo "🐳 To clean Docker system cache, run:"
echo "  docker system prune -a --volumes"
