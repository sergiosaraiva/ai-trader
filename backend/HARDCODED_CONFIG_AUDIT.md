# Hardcoded Configuration Audit Report

**Date:** 2026-01-27
**Status:** 🔴 **185+ hardcoded values found, 89 duplicated across files**

---

## 🚨 CRITICAL ISSUES

### 1. Trading Parameters Duplicated 40+ Times

**TP/SL Pips appear in:**
- ✅ `src/config/trading_config.py` (lines 307-322) - **CENTRALIZED**
- ❌ `src/models/multi_timeframe/improved_model.py` (lines 102-104, 116-118, 130-132) - **HARDCODED**
- ❌ `scripts/backtest_mtf_ensemble.py` (lines 55-57) - **HARDCODED**
- ❌ 40+ test files - **HARDCODED EVERYWHERE**

**Values:**
```python
# 1H: tp_pips=25.0, sl_pips=15.0, max_holding_bars=12
# 4H: tp_pips=50.0, sl_pips=25.0, max_holding_bars=18
# Daily: tp_pips=150.0, sl_pips=75.0, max_holding_bars=15
```

**Risk:** Changes require updating 40+ files manually. High chance of inconsistency.

---

### 2. Model Ensemble Weights Duplicated 8 Times

**Weights (0.6, 0.3, 0.1) appear in:**
- ✅ `src/config/trading_config.py` (line 49-51) - **CENTRALIZED**
- ❌ `src/models/multi_timeframe/mtf_ensemble.py` (line 36) - **HARDCODED**
- ❌ `src/api/schemas/prediction.py` (line 89, 196) - **HARDCODED**
- ❌ `src/api/main.py` (line 165) - **HARDCODED IN DOCS**

**Risk:** Changing ensemble weights requires updating 8 files.

---

### 3. Confidence Thresholds in 6 Different Places

| File | Line | Value | Purpose |
|------|------|-------|---------|
| `src/config/trading_config.py` | 27 | 0.60 | Global threshold ✅ |
| `src/agent/config.py` | 30 | 0.60 | Agent default ❌ |
| `src/api/routes/trading.py` | 386 | 0.70 | API param default ⚠️ **WRONG** |
| `src/models/multi_timeframe/mtf_predictor.py` | 104 | 0.65 | Model constant ❌ |
| `src/config/trading_config.py` | 217-218 | 0.55-0.75 | Dynamic bounds ✅ |

**Risk:** API route still has 0.70 default despite Config C deployment!

---

### 4. Circuit Breaker Limits Duplicated

**Risk limits appear in:**
- ✅ `src/config/trading_config.py` (lines 77-80) - **CENTRALIZED**
- ❌ `src/agent/config.py` (lines 54-55) - **DUPLICATED**

```python
# Both define:
max_consecutive_losses=5
max_drawdown_percent=15.0
max_daily_loss_percent=5.0  # or -3.0 (inconsistent!)
```

**Risk:** Agent config may override central config. Daily loss limit inconsistent (-3% vs -5%).

---

### 5. Scheduler Intervals Hardcoded

**Cron schedule hardcoded in:**
- ❌ `src/api/scheduler.py` (lines 240, 251, 262, 284) - **HARDCODED**
```python
CronTrigger(minute=55)  # Hourly at :55
IntervalTrigger(minutes=5)  # Every 5 minutes
misfire_grace_time=300  # 5 minutes
```

- ✅ `src/config/trading_config.py` (lines 164-168) - **CENTRALIZED BUT NOT USED**

**Risk:** Central config exists but scheduler.py doesn't use it.

---

### 6. API Port & Database Connection Hardcoded

**Hardcoded values:**
```python
# src/api/main.py line 183-185 - CORS origins
"http://localhost:3000"
"http://localhost:3001"
"http://localhost:5173"

# src/api/database/session.py line 21 - DB connection
"postgresql://aitrader:aitrader_dev_password@localhost:5432"

# src/api/main.py line 225 - Server params
host="0.0.0.0", port=8000
```

**Risk:** Cannot change without code modification. Credentials in code!

---

### 7. Data File Paths Not Centralized

**Hardcoded paths:**
- `src/models/multi_timeframe/enhanced_features.py` (lines 22-23)
- `src/api/services/model_service.py` (line 42)
- `src/agent/config.py` (line 186)

**Risk:** Deployment to different environments requires code changes.

---

## 📊 Summary Statistics

| Category | Instances | Duplicated | Severity | Centralized? |
|----------|-----------|------------|----------|--------------|
| **Trading Parameters** | 40+ | YES | 🔴 CRITICAL | Partial |
| **Position Sizing** | 50+ | YES | 🔴 CRITICAL | Yes (tests duplicate) |
| **Model Weights** | 8 | YES | 🔴 CRITICAL | Partial |
| **Confidence Thresholds** | 6-8 | YES | 🔴 CRITICAL | Partial |
| **Circuit Breakers** | 8 | YES | 🔴 CRITICAL | Partial |
| **Scheduler Intervals** | 6+ | YES | 🟠 HIGH | Not used |
| **API/Database** | 5+ | NO | 🟠 HIGH | No |
| **Data Paths** | 5+ | NO | 🟠 HIGH | No |
| **WFO Parameters** | 3 | NO | 🟠 HIGH | No |
| **Indicator Defaults** | 15+ | NO | 🟡 MEDIUM | No |
| **Timeouts** | 8 | YES | 🟡 MEDIUM | Partial |
| **Cache Settings** | 10+ | YES | 🟡 MEDIUM | Yes |

**Total:** 185+ hardcoded values
**Duplicated:** 89 instances in multiple files
**Properly Centralized:** ~30%

---

## 🎯 IMMEDIATE ACTIONS REQUIRED

### Priority 1: Fix Config C Deployment Issue

**FOUND:** `src/api/routes/trading.py` line 386 still has `confidence_threshold=0.70` default!

```python
# Current (WRONG):
async def get_filtered_recommendations(
    confidence_threshold: float = 0.70,  # ❌ Should be 0.60
    ...
)

# Should be:
async def get_filtered_recommendations(
    confidence_threshold: float = 0.60,  # ✅ Config C
    ...
)
```

### Priority 2: Critical Duplications

1. **Model file TP/SL values** → Import from `trading_config.py`
2. **Ensemble weights** → Import from central config
3. **Agent config duplication** → Import from `trading_config.py`
4. **Scheduler intervals** → Use centralized cron config

### Priority 3: Missing Centralization

1. **API CORS origins** → Move to environment variables
2. **Database credentials** → Use DATABASE_URL env var
3. **Data file paths** → Create path resolution helper
4. **WFO parameters** → Add to `trading_config.py`

---

## ✅ WHAT'S WORKING WELL

The `trading_config.py` file provides excellent centralization for:
- ✅ Trading parameters (TP/SL, position sizing)
- ✅ Circuit breakers (drawdown, daily loss)
- ✅ Dynamic threshold parameters
- ✅ Cache settings
- ✅ Model ensemble configuration

**The problem:** Other parts of the codebase don't always USE the centralized config.

---

## 🔧 RECOMMENDED FIXES

### Option 1: Quick Fix (Immediate)
Fix the 5 critical duplications:
1. Update `src/api/routes/trading.py` confidence default to 0.60
2. Import TP/SL from config in `improved_model.py`
3. Import weights from config in `mtf_ensemble.py`
4. Remove duplicated risk limits from `agent/config.py`
5. Use centralized scheduler config in `scheduler.py`

### Option 2: Comprehensive Refactor (Recommended)
1. Create config validation on startup
2. Add config override warnings
3. Refactor tests to use centralized config
4. Add environment variable overrides
5. Create config migration guide

---

## 📝 FILES THAT NEED UPDATES

### Critical (Fix Immediately):
1. `/home/sergio/ai-trader/backend/src/api/routes/trading.py` (line 386)
2. `/home/sergio/ai-trader/backend/src/models/multi_timeframe/improved_model.py` (lines 102-132)
3. `/home/sergio/ai-trader/backend/src/models/multi_timeframe/mtf_ensemble.py` (line 36)
4. `/home/sergio/ai-trader/backend/src/agent/config.py` (lines 30, 54-55)
5. `/home/sergio/ai-trader/backend/src/api/scheduler.py` (lines 240-284)

### High Priority:
6. `/home/sergio/ai-trader/backend/src/api/main.py` (CORS, server params)
7. `/home/sergio/ai-trader/backend/src/api/database/session.py` (DB connection)
8. `/home/sergio/ai-trader/backend/scripts/walk_forward_optimization.py` (WFO params)
9. `/home/sergio/ai-trader/backend/src/api/schemas/prediction.py` (weights)
10. `/home/sergio/ai-trader/backend/scripts/backtest_mtf_ensemble.py` (TP/SL defaults)

---

## 🚀 NEXT STEPS

1. **Immediate:** Fix the 0.70 confidence bug in trading.py
2. **This week:** Address Priority 1 critical duplications
3. **This month:** Comprehensive refactor to eliminate all hardcoded values
4. **Ongoing:** Add config validation to CI/CD pipeline

---

**Report prepared by:** Claude Code Exploration Agent
**Audit scope:** Entire backend codebase
**Date:** 2026-01-27
