# WFO Monthly Disaggregation Guide

## Overview

The `wfo_monthly_disaggregation.py` script breaks down WFO validation results into **month-by-month performance** across the entire 4-year test period (2022-2025).

## What It Does

1. Loads all 8 WFO window models
2. Re-runs backtests on each window's test period
3. Captures every trade with timestamp
4. Aggregates trades by calendar month
5. Produces CSV and JSON with monthly metrics

## Usage

```bash
cd backend

# Run with default settings (70% confidence)
python scripts/wfo_monthly_disaggregation.py

# Run with different confidence threshold
python scripts/wfo_monthly_disaggregation.py --confidence 0.75

# Custom output files
python scripts/wfo_monthly_disaggregation.py \
  --output-csv data/custom_monthly.csv \
  --output-json data/custom_monthly.json
```

## Output Files

### 1. CSV File: `data/wfo_monthly_results.csv`

Month-by-month performance in Excel-friendly format:

| year_month | trades | win_rate | total_pips | pnl_usd | balance_end | monthly_return_pct |
|------------|--------|----------|------------|---------|-------------|--------------------|
| 2022-01 | 12 | 66.7% | +78.5 | +1,245.30 | 11,245.30 | +12.45% |
| 2022-02 | 15 | 53.3% | +45.2 | +892.15 | 12,137.45 | +7.93% |
| ... | ... | ... | ... | ... | ... | ... |

**Columns:**
- `year_month` - Calendar month (YYYY-MM)
- `trades` - Number of trades executed
- `win_rate` - Percentage of winning trades
- `total_pips` - Net pips gained/lost
- `pnl_usd` - Profit/loss in USD
- `balance_end` - Account balance at end of month
- `monthly_return_pct` - Return as % of balance

### 2. JSON File: `data/wfo_monthly_results.json`

Structured data with metadata and summary:

```json
{
  "metadata": {
    "generated_at": "2026-01-27T...",
    "confidence_threshold": 0.70,
    "total_months": 48,
    "total_trades": 887
  },
  "monthly_results": [
    {
      "year_month": "2022-01",
      "trades": 12,
      "win_rate": 66.7,
      "total_pips": 78.5,
      "pnl_usd": 1245.30,
      "balance_end": 11245.30,
      "monthly_return_pct": 12.45
    },
    ...
  ],
  "summary": {
    "total_pips": 5239.1,
    "total_pnl_usd": 67954.14,
    "avg_monthly_trades": 18.5,
    "avg_monthly_return_pct": 3.5,
    "best_month": {
      "year_month": "2022-12",
      "trades": 45,
      "pnl_usd": 8234.50,
      "monthly_return_pct": 34.2
    },
    "worst_month": {
      "year_month": "2023-07",
      "trades": 8,
      "pnl_usd": -1245.30,
      "monthly_return_pct": -5.8
    }
  }
}
```

## Example Output

```
================================================================================
WFO MONTHLY DISAGGREGATION
================================================================================

Loaded WFO results: 8 windows

Processing Window 1: 2022-01 to 2022-06
  Generated 71 trades

Processing Window 2: 2022-07 to 2022-12
  Generated 163 trades

...

✅ Total trades across all windows: 887

Aggregating by month...

✅ CSV saved to: data/wfo_monthly_results.csv
✅ JSON saved to: data/wfo_monthly_results.json

================================================================================
MONTHLY SUMMARY
================================================================================

Period: 2022-01 to 2025-12
Total Months: 48
Total Trades: 887
Avg Trades/Month: 18.5
Total Pips: +5,239.1
Total PnL: $67,954.14
Avg Monthly Return: +3.5%

Best Month: 2022-12
  PnL: $8,234.50
  Return: +34.2%

Worst Month: 2023-07
  PnL: -$1,245.30
  Return: -5.8%

================================================================================
MONTH-BY-MONTH BREAKDOWN
================================================================================

Month        Trades   Win%      Pips      PnL ($)    Return%      Balance
--------------------------------------------------------------------------------
2022-01          12   66.7%    +78.5    +1,245.30    +12.45%  $11,245.30
2022-02          15   53.3%    +45.2      +892.15     +7.93%  $12,137.45
2022-03          10   70.0%   +125.8    +2,134.50    +17.58%  $14,271.95
...
2025-12          18   55.6%    +62.3    +1,456.20     +5.42%  $27,892.45
================================================================================

✅ Monthly disaggregation complete!
```

## Analysis Capabilities

### 1. Identify Strong/Weak Months

Find which months perform consistently well or poorly:

```bash
# Open CSV in Excel/Google Sheets
# Sort by "monthly_return_pct" to see best/worst months
# Look for patterns (e.g., does January always underperform?)
```

### 2. Seasonality Analysis

Check if certain months show seasonal patterns:

```python
import pandas as pd

df = pd.read_csv('data/wfo_monthly_results.csv')
df['month'] = pd.to_datetime(df['year_month']).dt.month

# Average return by calendar month
seasonal = df.groupby('month')['monthly_return_pct'].mean()
print(seasonal)
```

### 3. Drawdown Identification

Find consecutive losing months:

```bash
# Filter CSV for negative returns
# Identify longest drawdown periods
# Example: 2023-07, 2023-08, 2023-09 all negative = 3-month drawdown
```

### 4. Compounding Effect

See how balance grows month-over-month:

```python
df = pd.read_csv('data/wfo_monthly_results.csv')

# Plot balance progression
import matplotlib.pyplot as plt
plt.plot(df['year_month'], df['balance_end'])
plt.xlabel('Month')
plt.ylabel('Balance ($)')
plt.title('Account Balance Over Time')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('balance_progression.png')
```

## Key Insights from Monthly Data

### What to Look For:

1. **Consistency:**
   - How many months are profitable vs losing?
   - Are losses concentrated in specific periods?

2. **Volatility:**
   - Large swings in monthly returns indicate higher risk
   - Steady gains indicate more stable system

3. **Recovery:**
   - How quickly does system recover from losing months?
   - Do losses cluster together (regime changes)?

4. **Trade Frequency:**
   - Months with < 5 trades may have higher variance
   - Consistent trade count indicates stable signal generation

5. **Window 7 Visibility:**
   - You'll see 2025 H1 (Window 7) months with 0-1 trades
   - Confirms the regime change identified in analysis

## Integration with Frontend

To display monthly data in the dashboard:

1. **Add API endpoint:**

```python
@router.get("/backtest/monthly")
async def get_monthly_results():
    """Get month-by-month backtest results."""
    monthly_path = Path("data/wfo_monthly_results.json")
    if not monthly_path.exists():
        raise HTTPException(status_code=404)
    with open(monthly_path) as f:
        return json.load(f)
```

2. **Create Monthly Chart Component:**

```jsx
// MonthlyPerformanceChart.jsx
import { LineChart, Line, XAxis, YAxis, Tooltip, Legend } from 'recharts';

export function MonthlyPerformanceChart({ data }) {
  return (
    <LineChart data={data.monthly_results} width={800} height={400}>
      <XAxis dataKey="year_month" />
      <YAxis />
      <Tooltip />
      <Legend />
      <Line type="monotone" dataKey="balance_end" stroke="#8884d8" name="Balance" />
      <Line type="monotone" dataKey="total_pips" stroke="#82ca9d" name="Pips" />
    </LineChart>
  );
}
```

## Performance Notes

**Runtime:** ~5-10 minutes (depends on CPU)
- Loads 8 WFO models
- Runs predictions on 48 months of data
- Simulates ~900 trades

**Memory:** ~2GB RAM required
- Loads multiple models simultaneously
- Processes 5-minute data for 4 years

## Troubleshooting

### Issue: "WFO results not found"

**Solution:**
```bash
# Run WFO validation first
python scripts/walk_forward_optimization.py --sentiment --stacking
```

### Issue: "Could not load ensemble for window X"

**Solution:**
```bash
# Check window directories exist
ls -la models/wfo_validation/window_*

# Re-run WFO if missing
```

### Issue: "Insufficient test data"

**Solution:**
```bash
# Verify forex data availability
tail data/forex/EURUSD_20200101_20251231_5min_combined.csv

# Ensure data covers 2022-2025
```

## Next Steps

1. **Run the script:**
   ```bash
   python scripts/wfo_monthly_disaggregation.py
   ```

2. **Open CSV in Excel** to explore month-by-month data

3. **Identify patterns:**
   - Which months are consistently profitable?
   - Are there seasonal trends?
   - Where are the major drawdowns?

4. **Compare to market events:**
   - Correlate losing months with major forex events
   - Identify regime changes in the data

---

**Created:** 2026-01-27
**Script:** `backend/scripts/wfo_monthly_disaggregation.py`
**Outputs:** CSV + JSON with 48 months of detailed performance data
