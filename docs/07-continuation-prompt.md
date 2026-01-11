# Continuation Prompt for AI-Trader Implementation

## How to Use This Document

Copy the prompt below into a new Claude Code session to continue implementation of the AI-Trader system. The prompt provides full context and instructions for the implementation.

---

## Continuation Prompt

```
I need you to continue implementing the AI-Trader system - a technical analysis-focused trading system with ML-based predictions and a world-class trading robot.

## Project Context

This is the AI-Trader project located at /home/sergio/ai-trader. The project uses:
- Python 3.11+ with PyTorch for deep learning
- Beta distribution outputs for learned confidence (not sigmoid!)
- Three timeframe models (short/medium/long term) combined in an ensemble
- Configurable risk profiles and circuit breakers for protection
- Simulation mode before production trading

## Key Documentation to Read First

Before starting implementation, read these documents in order:

1. **CLAUDE.md** - Project guide with coding conventions and structure
2. **docs/01-architecture-overview.md** - System architecture
3. **docs/02-technical-analysis-model-design.md** - Model specifications
4. **docs/04-confidence-uncertainty-system.md** - Beta output layer (CRITICAL)
5. **docs/05-trading-robot-design.md** - Trading robot with risk management
6. **docs/06-implementation-plan.md** - Step-by-step implementation plan (FOLLOW THIS)

## Implementation Plan Location

The complete step-by-step implementation plan is at:
**docs/06-implementation-plan.md**

This plan has 8 phases:
1. Data Pipeline Foundation
2. Feature Engineering Pipeline
3. Model Architecture Implementation
4. Training Pipeline
5. Ensemble System
6. Trading Robot Core
7. Simulation Mode
8. Production Mode

## Agent Roles to Use

Use these specialized approaches during implementation:

### Requirements Analyst Role
When starting each phase:
- Review the phase objectives and acceptance criteria
- Clarify any ambiguous requirements
- Identify dependencies on previous phases
- Validate existing code that can be reused

### Code Engineer Role
When implementing:
- Follow the file structure in the implementation plan
- Use type hints and Google-style docstrings
- Integrate with existing code (check src/ for existing implementations)
- Follow the coding conventions in CLAUDE.md

### Quality Guardian Role
After writing code:
- Review for security vulnerabilities (especially in trading/execution)
- Check error handling and edge cases
- Ensure no data leakage in time series operations
- Validate the code follows project patterns

### Test Automator Role
For each component:
- Write unit tests in tests/ directory
- Aim for 80%+ coverage
- Test edge cases (empty data, missing values, etc.)
- Create integration tests for multi-component flows

## Current State

Check what's already implemented:
- src/models/confidence/ - Beta output layers (DONE)
- src/trading/risk/profiles.py - Risk profiles (DONE)
- src/trading/signals/ - Signal generation (DONE)
- src/trading/circuit_breakers/ - Circuit breakers (DONE)
- src/features/technical/ - Technical indicators (PARTIALLY DONE)

## Starting Point

To begin, read the implementation plan (docs/06-implementation-plan.md) and:

1. Determine which phase to start with based on current state
2. Check existing implementations in src/
3. Identify the first file to implement or complete
4. Start implementing following the acceptance criteria

## Commands to Run

```bash
# Activate virtual environment
source venv/bin/activate

# Check existing structure
ls -la src/

# Run existing tests
pytest tests/ -v

# Check what's implemented
find src -name "*.py" -type f | head -20
```

## Critical Rules

1. **ALWAYS use Beta outputs for direction prediction** - Never use raw sigmoid
2. **NEVER leak future data** - Time series splits must be chronological
3. **Test everything** - Write tests as you implement
4. **Follow the plan** - Each phase builds on the previous one
5. **Use existing code** - Don't reinvent what's already implemented

## Your Task

Start implementing the AI-Trader system following docs/06-implementation-plan.md. Begin with Phase 1 (Data Pipeline Foundation) unless you determine that phase is already complete, in which case proceed to the next incomplete phase.

For each phase:
1. Read the phase requirements in the implementation plan
2. Check what already exists in src/
3. Implement missing components
4. Write tests
5. Validate acceptance criteria
6. Move to next phase

Proceed autonomously without asking for confirmation (as specified in CLAUDE.md).
```

---

## Quick Reference

### Key Files Already Implemented

| File | Status | Description |
|------|--------|-------------|
| `src/models/confidence/learned_uncertainty.py` | DONE | Beta/Gaussian/Dirichlet outputs |
| `src/models/confidence/calibration.py` | DONE | Temperature/Platt/Isotonic |
| `src/models/confidence/uncertainty.py` | DONE | MC Dropout, Ensemble |
| `src/trading/risk/profiles.py` | DONE | 5 risk profiles |
| `src/trading/signals/generator.py` | DONE | Signal generation |
| `src/trading/signals/actions.py` | DONE | TradingSignal types |
| `src/trading/circuit_breakers/manager.py` | DONE | Circuit breaker coordinator |
| `src/trading/circuit_breakers/*.py` | DONE | Individual breakers |
| `src/features/technical/*.py` | PARTIAL | Technical indicators |

### Files to Implement (Priority Order)

**Phase 1 - Data Pipeline:**
- `src/data/sources/base.py`
- `src/data/sources/csv_source.py`
- `src/data/storage/parquet_store.py`
- `src/data/pipeline.py`

**Phase 2 - Features:**
- `src/data/processors/timeframe_transformer.py`
- `src/features/technical/calculator.py`
- `src/features/store.py`
- `src/data/loaders/training_loader.py`

**Phase 3 - Models:**
- `src/models/technical/short_term.py`
- `src/models/technical/medium_term.py`
- `src/models/technical/long_term.py`

**Phase 4 - Training:**
- `src/training/trainer.py`
- `src/training/experiment.py`
- `scripts/train_model.py`

**Phase 5 - Ensemble:**
- `src/models/ensemble/combiner.py`
- `src/models/ensemble/predictor.py`

**Phase 6-8 - Trading Robot:**
- `src/trading/robot/core.py`
- `src/simulation/backtester.py`
- `src/trading/brokers/alpaca.py`

### Key Design Decisions

1. **Beta Distribution for Direction**: Model outputs Beta(α, β) where:
   - Direction = α/(α+β) > 0.5 means UP
   - Confidence = function of concentration (α+β)

2. **Risk Profiles**: 5 levels from ultra-conservative (85% confidence required) to ultra-aggressive (52%)

3. **Circuit Breakers**:
   - Consecutive loss halt (2-10 losses depending on profile)
   - Drawdown protection (progressive reduction)
   - Model degradation detection

4. **Position Sizing**: `size = base * confidence_factor * kelly_factor * agreement_factor * breaker_multiplier`

---

## Notes for New Session

- The project is in `/home/sergio/ai-trader`
- Virtual environment is in `venv/`
- Sample data is in `data/sample/`
- Activate venv before running: `source venv/bin/activate`
- The CLAUDE.md file has all project conventions
- Always proceed autonomously (no confirmation needed)

---

*Document Version: 1.0*
*Created: 2026-01-08*
