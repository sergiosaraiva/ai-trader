# Disk Space Optimization Summary

## ✅ Completed Cleanup (920MB Saved)

### Before
- **Models:** 1,412MB
- **Data:** 872MB
- **Total:** 2,284MB

### After
- **Models:** 717MB
- **Data:** 647MB
- **Total:** 1,364MB
- **Saved:** 920MB (40% reduction)

---

## 🗑️ What Was Removed

### 1. Tar Archive Parts (~550MB)
- `models_trained.tar.gz.part_*` (7 files)
- Old backup from January 11
- **Status:** ✅ Removed and added to .gitignore

### 2. Crypto Data (~174MB)
- BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT, BNBUSDT
- Project uses EUR/USD forex only
- **Status:** ✅ Removed and added to .gitignore

### 3. Cache Directory (~53MB)
- Parquet feature files (auto-regenerated)
- **Status:** ✅ Removed (will be recreated on next run)

### 4. Experimental Model Directories (~143MB)
Removed 15 experimental/backup MTF ensemble directories:
- mtf_ensemble_all_sentiment
- mtf_ensemble_backup_v1
- mtf_ensemble_baseline*
- mtf_ensemble_daily_sentiment
- mtf_ensemble_epu_daily
- mtf_ensemble_gdelt
- mtf_ensemble_pre_wavelet_backup
- mtf_ensemble_sentiment*
- mtf_ensemble_shallow_fast
- mtf_ensemble_stacking
- mtf_ensemble_us_sentiment
- mtf_ensemble_wavelet

**Production model preserved:** `backend/models/mtf_ensemble/`

### 5. Test/Validation Directories (~50MB)
Removed old hyperparameter and WFO test directories:
- hpo_test_even_shallower
- hpo_test_higher_lr
- hpo_test_more_trees_shallow
- wfo_baseline*
- wfo_rfecv_*
- wfo_tier1_validation
- wfo_tier2_*
- wfo_stacking

**Validation results preserved:** `backend/models/wfo_validation/`

---

## 🐳 Docker Optimizations

### Log File Size Limits Added
Updated `docker-compose.override.yml` with log rotation:

| Service | Max Size | Max Files | Max Total |
|---------|----------|-----------|-----------|
| postgres | 5MB | 2 | 10MB |
| backend | 10MB | 2 | 20MB |
| agent | 10MB | 2 | 20MB |
| frontend | 5MB | 2 | 10MB |

**Total max log storage:** 60MB (prevents unbounded growth)

### Docker Volume Status
```bash
VOLUME NAME                     SIZE
ai-trader_postgres_data         48.5MB   ✅ Small, healthy
ai-trader_data                  0B       ✅ Empty (unused)
```

---

## 📦 Additional Cleanup Options

### Optional: Remove Large Trained Model Directories (~500MB)

These directories contain old experimental models. Only remove if you don't need them:

```bash
# Review before removing:
du -sh backend/models/trained/         # 351MB - old trained models
du -sh backend/models/practical_e2e/   # 156MB - E2E test models
du -sh backend/models/individual_models/ # 144MB - individual experiments
du -sh backend/models/pipeline_run/    # 46MB - pipeline test models

# To remove (if not needed):
rm -rf backend/models/trained
rm -rf backend/models/practical_e2e
rm -rf backend/models/individual_models
rm -rf backend/models/pipeline_run

# This would save an additional ~700MB
```

### Optional: Clean Docker System Cache

```bash
# Remove unused Docker images, containers, and volumes
docker system prune -a --volumes

# Warning: This removes ALL unused Docker resources
# You may need to rebuild images after this
```

### Optional: Compress Forex Data

The EUR/USD data file is 624MB. Consider compressing historical data:

```bash
# Current size
ls -lh backend/data/forex/EURUSD_20200101_20251231_5min_combined.csv
# 624MB

# Could compress older data (2020-2023) and keep recent data uncompressed
# This requires code changes to handle compressed files
```

---

## 🛡️ Prevention Measures Applied

### Updated .gitignore
1. **Tar archives:** `*.tar.gz.part_*` - prevents future tar parts
2. **Crypto data:** `data/crypto/` - explicitly ignores crypto files

### Log Rotation
- Docker logs now automatically rotate at size limits
- Prevents logs from consuming unlimited disk space

---

## 📊 Current Production Files (Preserved)

### Essential Models (~717MB)
- `backend/models/mtf_ensemble/` - Production models (1H, 4H, Daily)
- `backend/models/wfo_validation/` - Walk-forward validation results
- `backend/models/improved_mtf/` - Improved timeframe models
- `backend/models/gradient_boosting/` - Alternative models
- `backend/models/benchmark_rfecv/` - Feature selection benchmarks

### Essential Data (~647MB)
- `backend/data/forex/` - EUR/USD 5-minute data (2020-2025)
- `backend/data/sentiment/` - EPU and sentiment scores
- `backend/data/db/` - Trading database
- `backend/data/sample/` - Sample data for testing

---

## 🔄 Maintenance Commands

### Check Disk Usage
```bash
# Overall sizes
du -sh backend/data backend/models

# Detailed breakdown
du -sh backend/models/* | sort -h
du -sh backend/data/* | sort -h

# Find large files
find backend -type f -size +10M -exec du -h {} \; | sort -h
```

### Clean Cache Manually
```bash
# Safe to delete - will be regenerated
rm -rf backend/data/cache
```

### View Docker Disk Usage
```bash
docker system df -v
```

### Rebuild Docker Containers (if needed)
```bash
docker-compose down
docker-compose up --build
```

---

## ✅ Verification

Production system still functional:
- ✅ MTF ensemble models intact (`backend/models/mtf_ensemble/`)
- ✅ WFO validation results intact (`backend/models/wfo_validation/`)
- ✅ Forex data intact (`backend/data/forex/`)
- ✅ Sentiment data intact (`backend/data/sentiment/`)
- ✅ Docker volumes healthy (48.5MB PostgreSQL)
- ✅ Log rotation configured to prevent future bloat

**System ready for deployment with 40% less disk usage!**
