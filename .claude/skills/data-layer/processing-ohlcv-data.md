---
name: processing-ohlcv-data
description: DEPRECATED - Merged into creating-data-processors skill
deprecated: true
redirect_to: ../backend/creating-data-processors.md
---

# Processing OHLCV Data (DEPRECATED)

**This skill has been merged into [creating-data-processors](../backend/creating-data-processors.md).**

## Redirect

All OHLCV processing patterns are now documented in:
- **Skill**: `creating-data-processors`
- **Location**: `.claude/skills/backend/creating-data-processors.md`

The merged skill includes:
- Validate/Clean/Transform pipeline pattern
- OHLCV-specific validation rules
- Timeframe resampling (OHLCV aggregation)
- Derived candlestick features
- Sequence creation for time series

## Migration

If you were using `processing-ohlcv-data`, use `creating-data-processors` instead.

The skill-router has been updated to route OHLCV tasks to the consolidated skill.

---

*Deprecated: 2026-01-07*
*Merged into: creating-data-processors*
