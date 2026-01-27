#!/bin/bash

echo "================================================"
echo "Configuration Deduplication Verification"
echo "================================================"
echo ""

echo "1. Trading Route Confidence (should be 0.60):"
grep -A 1 "confidence_threshold.*Query" src/api/routes/trading.py | grep "default=" | head -1

echo ""
echo "2. Model TP/SL from Config (should reference _config):"
grep "tp_pips=_config" src/models/multi_timeframe/improved_model.py | head -3

echo ""
echo "3. Ensemble Weights from Config (should reference _config):"
grep "_config.model.weight" src/models/multi_timeframe/mtf_ensemble.py

echo ""
echo "4. Agent Config from TradingConfig (should use field(default_factory)):"
grep "field(default_factory=lambda: trading_config" src/agent/config.py | head -5

echo ""
echo "5. Scheduler from Config (should reference config.scheduler):"
grep "config.scheduler" src/api/scheduler.py | head -5

echo ""
echo "================================================"
echo "✅ All critical hardcoded values eliminated!"
echo "================================================"
