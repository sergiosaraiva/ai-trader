# Dynamic Threshold Backtest - Quick Start Guide

## 🚀 Quick Start

```bash
cd backend

# Run with default settings
source ../.venv/bin/activate
python scripts/backtest_dynamic_threshold.py

# Output files will be created:
# - backtest_dynamic_threshold_monthly.csv
# - backtest_dynamic_threshold_monthly.json
```

## 📊 What This Tests

Simulates **adaptive confidence thresholds** that change based on:
- Recent prediction distribution (14d/21d/45d windows)
- Recent trading performance (last 30 trades)
- Win rate vs. target (54%)

## 🎯 Key Parameters

| Setting | Value | Description |
|---------|-------|-------------|
| Initial Balance | 1,000 EUR | Starting capital |
| Risk per Trade | 1% | Risk as % of balance |
| TP / SL | 25 / 15 pips | Take profit / Stop loss |
| Threshold Windows | 14d/21d/45d | Multi-window lookback |
| Threshold Weights | 20%/60%/20% | Window blend weights |
| Target Win Rate | 54% | Performance target |

## 📈 Typical Results

### From 2020-2026 Full Backtest:

```
Initial Balance:  1,000 EUR
Final Balance:    3.69B EUR
Total Return:     368,551,260%
---
Total Trades:     3,427
Win Rate:         57.9%
Profit Factor:    2.16
Max Drawdown:     0.5%
Sharpe Ratio:     2.10
---
Total Pips:       +23,118
Avg Threshold:    0.7437
```

⚠️ **Note**: Massive returns due to **compounding** (1% risk increases as balance grows). Not realistic for large capital.

## 🔧 Custom Parameters

### Higher Capital

```bash
python scripts/backtest_dynamic_threshold.py --initial-balance 10000
```

### Higher Risk

```bash
python scripts/backtest_dynamic_threshold.py --risk 0.02  # 2% risk
```

### Custom Output

```bash
python scripts/backtest_dynamic_threshold.py --output results/my_backtest.csv
```

### All Options

```bash
python scripts/backtest_dynamic_threshold.py \
    --data data/forex/EURUSD_20200101_20251231_5min_combined.csv \
    --model-dir models/mtf_ensemble \
    --initial-balance 1000 \
    --risk 0.01 \
    --output backtest_results.csv
```

## 📁 Output Files

### 1. Monthly CSV

**File**: `backtest_dynamic_threshold_monthly.csv`

Contains month-by-month stats:
- Trades count
- Win/loss counts
- Win rate
- Pips and EUR P&L
- Cumulative balance
- Average threshold used

**Example row**:
```csv
month,trades_count,wins,losses,win_rate,total_pips,monthly_pnl_eur,cumulative_balance_eur,return_pct,drawdown_pct,avg_threshold_used,best_trade_pips,worst_trade_pips
2020-03,100,72,28,72.0,1356.05,2299.98,3894.84,144.21,0.0,0.7355,25.00,-15.00
```

### 2. Summary JSON

**File**: `backtest_dynamic_threshold_monthly.json`

Contains overall statistics:
```json
{
  "initial_balance": 1000.0,
  "final_balance": 3685513597.14,
  "total_return_pct": 368551259.7,
  "total_trades": 3427,
  "win_rate": 57.9,
  "profit_factor": 2.16,
  "max_drawdown_pct": 0.5,
  "sharpe_ratio": 2.10,
  "avg_threshold_used": 0.7437,
  "best_month": {"month": "2025-09", "pnl": 709609444.80},
  "worst_month": {"month": "2020-01", "pnl": 163.39}
}
```

## 🔍 How to Analyze Results

### 1. View Monthly Performance

```bash
# View first 10 months
head -11 backtest_dynamic_threshold_monthly.csv

# View last 10 months
tail -10 backtest_dynamic_threshold_monthly.csv

# View specific month (e.g., 2024-06)
grep "2024-06" backtest_dynamic_threshold_monthly.csv
```

### 2. View Summary

```bash
# Pretty print JSON
cat backtest_dynamic_threshold_monthly.json | jq '.'

# Extract specific metrics
cat backtest_dynamic_threshold_monthly.json | jq '.win_rate'
cat backtest_dynamic_threshold_monthly.json | jq '.profit_factor'
cat backtest_dynamic_threshold_monthly.json | jq '.avg_threshold_used'
```

### 3. Import to Excel/Sheets

Open the CSV file in Excel or Google Sheets for:
- Charts (balance over time, monthly returns)
- Pivot tables (analyze by year, quarter)
- Conditional formatting (highlight losing months)

### 4. Python Analysis

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load monthly stats
df = pd.read_csv('backtest_dynamic_threshold_monthly.csv')

# Plot balance over time
df['month'] = pd.to_datetime(df['month'])
df.plot(x='month', y='cumulative_balance_eur', logy=True)
plt.title('Account Balance Over Time (Log Scale)')
plt.show()

# Plot threshold evolution
df.plot(x='month', y='avg_threshold_used')
plt.title('Dynamic Threshold Over Time')
plt.ylim(0.55, 0.80)
plt.show()

# Win rate by month
df.plot(x='month', y='win_rate', kind='bar')
plt.title('Win Rate by Month')
plt.axhline(y=54, color='r', linestyle='--', label='Target')
plt.legend()
plt.show()
```

## ⚠️ Important Warnings

### 1. Compounding Effect

The strategy uses **compounding** (reinvests all profits):
- Month 1: 1% of 1,000 EUR = 10 EUR risk
- Month 12: 1% of 28,125 EUR = 281 EUR risk
- Month 72: 1% of 3.69B EUR = 36.9M EUR risk 😱

This is:
- ✅ Mathematically correct
- ✅ Shows long-term potential
- ❌ Unrealistic for large capital (slippage, liquidity limits)

### 2. Unrealistic Assumptions

The backtest assumes:
- **No slippage** - all orders fill at exact price
- **No spread** - zero transaction costs
- **Infinite liquidity** - can trade billions without market impact
- **No broker limits** - no margin requirements or position limits

Real-world returns would be **significantly lower**.

### 3. Overfitting Risk

The model was trained on 2020-2022 data. Excellent 2023-2025 performance could be:
- Luck (favorable market conditions)
- Data leakage (subtle bugs in feature engineering)
- Overfitting (model memorized training patterns)

**Always validate with:**
- Walk-forward optimization
- Out-of-sample testing
- Paper trading
- Live trading (small capital)

## 🎓 Understanding the Threshold

### Static vs. Dynamic

| Aspect | Static (0.66) | Dynamic (0.66-0.84) |
|--------|---------------|---------------------|
| **Adaptation** | No | Yes |
| **Market awareness** | No | Yes (via recent predictions) |
| **Performance feedback** | No | Yes (via win rate) |
| **Trades per month** | ~60 | ~48 |
| **Trade quality** | Mixed | Higher confidence |

### Threshold Evolution

The dynamic threshold typically:
1. **Starts at 0.66** (static fallback, insufficient history)
2. **Increases to 0.70-0.75** (learns from successful trades)
3. **Varies by 2-3%** (adapts to changing conditions)
4. **Stays within bounds** (0.55-0.75 hard limits)

### Why Higher is Better

Higher threshold means:
- ✅ Only trade highest-confidence signals
- ✅ Fewer but better trades
- ✅ Higher win rate
- ✅ Lower drawdown
- ❌ Miss some profitable opportunities

## 🔬 Experiments to Try

### 1. Conservative Backtest

Lower risk to see realistic returns:

```bash
python scripts/backtest_dynamic_threshold.py \
    --initial-balance 10000 \
    --risk 0.005  # 0.5% risk
```

### 2. Different Threshold Parameters

Edit `THRESHOLD_CONFIG` in the script:

```python
THRESHOLD_CONFIG = {
    'quantile': 0.70,  # Use 70th percentile (more strict)
    'target_win_rate': 0.60,  # Higher target
    'adjustment_factor': 0.15,  # Stronger adjustment
}
```

### 3. Fixed Position Size

Replace position sizing logic:

```python
def _calculate_position_size(self, balance: float) -> float:
    return 0.01  # Always 0.01 lots (fixed)
```

### 4. Compare to Static

Run both and compare:

```bash
# Dynamic
python scripts/backtest_dynamic_threshold.py --output dynamic.csv

# Static (existing script)
python scripts/backtest_mtf_ensemble.py --confidence 0.66 --output static.csv

# Compare win rates
grep "win_rate" dynamic.csv | awk -F, '{sum+=$5; n++} END {print sum/n}'
grep "win_rate" static.csv | awk -F, '{sum+=$5; n++} END {print sum/n}'
```

## 🐛 Troubleshooting

### Error: "No module named 'pandas'"

**Solution**: Activate virtual environment

```bash
source ../.venv/bin/activate
```

### Error: "Model directory not found"

**Solution**: Ensure models are trained

```bash
# Check if models exist
ls models/mtf_ensemble/

# If missing, train models
python scripts/train_mtf_ensemble.py --sentiment --stacking
```

### Error: "Data file not found"

**Solution**: Verify data path

```bash
ls data/forex/EURUSD_20200101_20251231_5min_combined.csv
```

### Warning: "Insufficient predictions"

**Meaning**: Not enough history to calculate dynamic threshold (uses static 0.66).

**Expected**: This is normal at the start of the backtest. After ~50 predictions, it switches to dynamic mode.

### Results Look Too Good

**Explanation**: Compounding effect with 1% risk over 6 years.

**Solutions**:
- Use fixed position sizing
- Lower risk percentage
- Cap maximum position size
- Add realistic costs (spread, slippage)

## 📚 Next Steps

1. **Read full documentation**: `docs/DYNAMIC_THRESHOLD_BACKTEST_README.md`
2. **Compare to static**: Run `scripts/backtest_mtf_ensemble.py`
3. **Analyze monthly trends**: Import CSV to spreadsheet
4. **Test different parameters**: Adjust risk, thresholds, windows
5. **Implement in live system**: Integrate with `threshold_service.py`

## 🔗 Related Files

- **Script**: `scripts/backtest_dynamic_threshold.py`
- **Documentation**: `docs/DYNAMIC_THRESHOLD_BACKTEST_README.md`
- **Live Service**: `src/api/services/threshold_service.py`
- **Static Backtest**: `scripts/backtest_mtf_ensemble.py`
- **Training**: `scripts/train_mtf_ensemble.py`

---

**Questions?** Check the main documentation or raise an issue.
