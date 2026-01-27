# Conservative Hybrid Position Sizing Implementation

## Overview

Successfully implemented the Conservative Hybrid position sizing strategy across the entire trading system. This strategy combines confidence-based scaling with comprehensive circuit breakers for risk management.

## Implementation Summary

### Order 1: Configuration System ✓
**File:** `backend/src/config/trading_config.py`

Added `ConservativeHybridParameters` dataclass with:
- **Base Risk Parameters:**
  - `base_risk_percent`: 1.5% (default)
  - `confidence_scaling_factor`: 0.5x
  - `min_risk_percent`: 0.8%
  - `max_risk_percent`: 2.5%

- **Circuit Breakers:**
  - `daily_loss_limit_percent`: -3.0%
  - `consecutive_loss_limit`: 5 trades

- **Trading Parameters:**
  - `confidence_threshold`: 0.70
  - `pip_value`: 10.0 ($ per pip for 0.1 lot EUR/USD)
  - `lot_size`: 100,000 (standard lot)

**Features:**
- Full integration with centralized config system
- Environment variable support (e.g., `CONSERVATIVE_HYBRID_BASE_RISK`)
- Hot reload capability via callbacks
- Database persistence support
- Comprehensive validation

### Order 2: Database Schema ✓
**File:** `backend/src/api/database/models.py`

Added to `Trade` model:
- `risk_percentage_used` (Float, nullable): Tracks actual risk % used for each trade

### Order 3: Position Sizer ✓
**File:** `backend/src/trading/position_sizer.py`

Created `ConservativeHybridSizer` class with:
- `calculate_position_size()` method implementing the strategy:
  1. Check confidence threshold
  2. Calculate confidence multiplier: `1.0 + (confidence - threshold) × scaling_factor`
  3. Adjust risk percentage: `base_risk × multiplier`
  4. Apply min/max caps
  5. Calculate position from risk: `(balance × risk%) / (sl_pips × pip_value)`
  6. Apply no-leverage constraint: `position ≤ balance / lot_size`

**Returns:**
- Position size in lots
- Risk percentage used
- Detailed metadata dictionary

### Order 4: Circuit Breakers ✓
**File:** `backend/src/trading/circuit_breakers.py`

Created `TradingCircuitBreaker` class with:
- `can_trade()`: Checks all circuit breaker conditions
- `get_daily_pnl()`: Calculates today's P&L
- `get_consecutive_losses()`: Counts consecutive losing trades
- `get_monthly_drawdown()`: Calculates monthly drawdown (future enhancement)

**Circuit Breaker Logic:**
1. **Daily Loss Limit:** Stops trading if daily loss exceeds configured %
2. **Consecutive Loss Limit:** Stops trading after N consecutive losses
3. Returns `(can_trade: bool, reason: Optional[str])`

### Order 5: Trading Service Integration ✓
**File:** `backend/src/api/services/trading_service.py`

**Changes:**
- Initialized `position_sizer` and `circuit_breaker` in `__init__`
- Added `_on_conservative_hybrid_change()` callback for hot reload
- Updated `execute_trade()`:
  - Check circuit breakers first
  - Calculate position size using Conservative Hybrid
  - Store `risk_percentage_used` in Trade record
  - Enhanced logging with position size and risk %
- Added helper methods:
  - `get_daily_pnl()`: Public method to get today's P&L
  - `can_trade()`: Public method to check circuit breakers

### Order 6: Backtest Script Integration ✓
**File:** `backend/scripts/backtest_dynamic_threshold.py`

**Changes:**
- Added `CONSERVATIVE_HYBRID_CONFIG` dictionary
- Imported position sizer and circuit breaker modules
- Updated `Trade` dataclass to include `risk_pct_used`
- Updated `DynamicThresholdBacktester.__init__`:
  - Initialize position sizer and circuit breaker
  - Add daily P&L tracker
- Updated trading loop:
  - Check circuit breakers before each trade
  - Calculate position size using Conservative Hybrid
  - Track daily P&L for circuit breaker logic
  - Store risk percentage in trade records
- Added helper methods:
  - `_get_consecutive_losses()`: Count recent consecutive losses
  - `_get_daily_pnl()`: Get P&L for current day
- Updated CLI arguments:
  - Replaced `--fixed-risk` with `--base-risk-percent`
  - Default: 1.5%
- Enhanced logging to show Conservative Hybrid parameters

## Key Features

### Position Sizing Formula
```python
confidence_multiplier = 1.0 + (confidence - threshold) × scaling_factor
adjusted_risk_pct = base_risk_pct × confidence_multiplier
risk_pct_used = clamp(adjusted_risk_pct, min_risk, max_risk)
position_lots = (balance × risk_pct_used / 100) / (sl_pips × pip_value)
final_position = min(position_lots, balance / lot_size)  # No leverage
```

### Example Position Sizing
- **Balance:** $100,000
- **Confidence:** 0.75 (75%)
- **Base Risk:** 1.5%
- **Threshold:** 0.70
- **Scaling Factor:** 0.5x
- **SL:** 15 pips

**Calculation:**
1. Multiplier: `1.0 + (0.75 - 0.70) × 0.5 = 1.025`
2. Adjusted Risk: `1.5% × 1.025 = 1.54%`
3. Risk Amount: `$100,000 × 0.0154 = $1,540`
4. Position Size: `$1,540 / (15 pips × $10/pip) = 10.27 lots`
5. No-Leverage Check: `max($100,000 / $100,000) = 1.0 lot`
6. **Final Position:** 1.0 lot (limited by no-leverage constraint)

### Circuit Breaker Protection
- **Daily Loss Limit:** Prevents catastrophic daily losses
- **Consecutive Loss Limit:** Stops trading during losing streaks
- **Logging:** All triggers are logged with detailed reasons

## Environment Variables

Configure via environment variables:
```bash
CONSERVATIVE_HYBRID_BASE_RISK=1.5
CONSERVATIVE_HYBRID_CONFIDENCE_SCALING=0.5
CONSERVATIVE_HYBRID_MIN_RISK=0.8
CONSERVATIVE_HYBRID_MAX_RISK=2.5
CONSERVATIVE_HYBRID_DAILY_LOSS_LIMIT=-3.0
CONSERVATIVE_HYBRID_CONSECUTIVE_LOSS_LIMIT=5
CONSERVATIVE_HYBRID_CONFIDENCE_THRESHOLD=0.70
CONSERVATIVE_HYBRID_PIP_VALUE=10.0
CONSERVATIVE_HYBRID_LOT_SIZE=100000.0
```

## Usage Examples

### Backtest with Conservative Hybrid
```bash
# Default parameters (1.5% base risk)
python scripts/backtest_dynamic_threshold.py

# Custom base risk
python scripts/backtest_dynamic_threshold.py --base-risk-percent 2.0

# Custom initial balance and risk
python scripts/backtest_dynamic_threshold.py --initial-balance 10000 --base-risk-percent 1.0
```

### API Integration
The trading service automatically uses Conservative Hybrid when executing trades:
```python
from src.api.services.trading_service import trading_service

# Execute trade (Conservative Hybrid automatically applied)
trade = trading_service.execute_trade(prediction, current_price, db)

# Check if trading is allowed
can_trade, reason = trading_service.can_trade(db)
if not can_trade:
    print(f"Trading blocked: {reason}")

# Get daily P&L
daily_pnl = trading_service.get_daily_pnl(db)
```

## Files Modified/Created

### Created Files:
1. `backend/src/trading/position_sizer.py` - Position sizing implementation
2. `backend/src/trading/circuit_breakers.py` - Circuit breaker logic
3. `backend/CONSERVATIVE_HYBRID_IMPLEMENTATION.md` - This documentation

### Modified Files:
1. `backend/src/config/trading_config.py` - Added ConservativeHybridParameters
2. `backend/src/api/database/models.py` - Added risk_percentage_used field
3. `backend/src/api/services/trading_service.py` - Integrated position sizer and circuit breakers
4. `backend/scripts/backtest_dynamic_threshold.py` - Integrated Conservative Hybrid into backtest

## Validation

All syntax checks passed:
- ✓ `trading_config.py` syntax OK
- ✓ `position_sizer.py` syntax OK
- ✓ `circuit_breakers.py` syntax OK
- ✓ `trading_service.py` syntax OK
- ✓ `backtest_dynamic_threshold.py` syntax OK

## Next Steps

1. **Database Migration:** Run migration to add `risk_percentage_used` column to trades table
2. **Testing:** Run backtest with Conservative Hybrid to validate performance
3. **Monitoring:** Track risk percentage usage and circuit breaker triggers
4. **Optimization:** Tune parameters based on backtest results:
   - Adjust base risk percentage
   - Modify confidence scaling factor
   - Fine-tune circuit breaker thresholds

## Benefits

1. **Risk Management:** Position sizes scale with confidence but remain capped
2. **Circuit Protection:** Automatic stop-loss at daily and consecutive loss limits
3. **No Leverage:** Prevents margin calls and forced liquidation
4. **Configurability:** All parameters adjustable via config system
5. **Transparency:** Full tracking of risk percentage used per trade
6. **Hot Reload:** Configuration changes without service restart

## Design Principles Followed

- **Conservative:** Base risk is moderate (1.5%), with caps at 2.5%
- **Hybrid:** Combines fixed base risk with confidence scaling
- **Defensive:** No-leverage constraint prevents overexposure
- **Transparent:** All calculations logged and stored
- **Configurable:** Easy parameter tuning via environment variables
- **Testable:** Integrated into backtest framework for validation
