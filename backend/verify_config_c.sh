#!/bin/bash
# Config C Deployment Verification Script

echo "=========================================="
echo "Config C Deployment Verification"
echo "=========================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check 1: Model files
echo "1. Checking model files..."
MODEL_DIR="models/wfo_conf60_18mo/window_9"
if [ -d "$MODEL_DIR" ]; then
    echo -e "${GREEN}✓${NC} Model directory exists: $MODEL_DIR"

    # Check for required files
    REQUIRED_FILES=("1H_model.pkl" "4H_model.pkl" "D_model.pkl" "stacking_meta_learner.pkl" "ensemble_config.json")
    for file in "${REQUIRED_FILES[@]}"; do
        if [ -f "$MODEL_DIR/$file" ]; then
            echo -e "  ${GREEN}✓${NC} $file"
        else
            echo -e "  ${RED}✗${NC} $file (MISSING)"
        fi
    done
else
    echo -e "${RED}✗${NC} Model directory not found: $MODEL_DIR"
fi
echo ""

# Check 2: Production symlink
echo "2. Checking production symlink..."
if [ -L "models/production_model" ]; then
    LINK_TARGET=$(readlink models/production_model)
    echo -e "${GREEN}✓${NC} Symlink exists: models/production_model -> $LINK_TARGET"
    if [ "$LINK_TARGET" = "wfo_conf60_18mo/window_9" ]; then
        echo -e "  ${GREEN}✓${NC} Points to correct Config C model"
    else
        echo -e "  ${YELLOW}⚠${NC} Points to different model: $LINK_TARGET"
    fi
else
    echo -e "${RED}✗${NC} Production symlink not found"
fi
echo ""

# Check 3: Trading config
echo "3. Checking trading configuration..."
CONFIDENCE_COUNT=$(grep -c "confidence_threshold.*0\.60" src/config/trading_config.py)
if [ "$CONFIDENCE_COUNT" -ge 2 ]; then
    echo -e "${GREEN}✓${NC} Trading config updated (found $CONFIDENCE_COUNT occurrences of 0.60)"
else
    echo -e "${RED}✗${NC} Trading config may not be updated correctly"
fi
echo ""

# Check 4: Model service
echo "4. Checking model service..."
if grep -q "wfo_conf60_18mo" src/api/services/model_service.py; then
    echo -e "${GREEN}✓${NC} Model service points to Config C directory"
else
    echo -e "${RED}✗${NC} Model service not updated"
fi
echo ""

# Check 5: Agent config
echo "5. Checking agent configuration..."
if grep -q "0\.60" src/agent/config.py; then
    echo -e "${GREEN}✓${NC} Agent config updated to 0.60 threshold"
else
    echo -e "${YELLOW}⚠${NC} Agent config may still use old threshold"
fi
echo ""

# Check 6: WFO script
echo "6. Checking WFO script defaults..."
WFO_TRAIN=$(grep -A 1 '"--train-months"' scripts/walk_forward_optimization.py | grep "default=18")
WFO_CONF=$(grep -A 1 '"--confidence"' scripts/walk_forward_optimization.py | grep "default=0.60")

if [ -n "$WFO_TRAIN" ]; then
    echo -e "${GREEN}✓${NC} WFO train-months default: 18"
else
    echo -e "${YELLOW}⚠${NC} WFO train-months may not be updated"
fi

if [ -n "$WFO_CONF" ]; then
    echo -e "${GREEN}✓${NC} WFO confidence default: 0.60"
else
    echo -e "${YELLOW}⚠${NC} WFO confidence may not be updated"
fi
echo ""

# Check 7: Documentation
echo "7. Checking documentation..."
if grep -q "Config C" ../CLAUDE.md; then
    echo -e "${GREEN}✓${NC} CLAUDE.md updated with Config C info"
else
    echo -e "${YELLOW}⚠${NC} CLAUDE.md may not reference Config C"
fi
echo ""

# Summary
echo "=========================================="
echo "Verification Complete"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Start backend: uvicorn src.api.main:app --reload --port 8001"
echo "2. Check logs for model loading confirmation"
echo "3. Test prediction endpoint: curl http://localhost:8001/api/v1/predictions/current"
echo "4. Verify 'dynamic_threshold_used' field shows ~0.60"
echo ""
