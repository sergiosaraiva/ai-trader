#!/usr/bin/env python3
"""Analyze optimal confidence threshold using What-If API endpoint."""

import requests
import json
from typing import Dict, List

# Test different thresholds
THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
API_BASE = "http://localhost:8001/api/v1"
DAYS = 30

def get_whatif_performance(threshold: float) -> Dict:
    """Fetch What-If performance for a given threshold."""
    url = f"{API_BASE}/trading/whatif-performance"
    params = {
        "days": DAYS,
        "confidence_threshold": threshold,
        "require_agreement": True
    }

    try:
        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching threshold {threshold}: {e}")
        return None

def analyze_threshold(threshold: float, data: Dict) -> Dict:
    """Analyze performance metrics for a threshold."""
    if not data or 'summary' not in data:
        return None

    summary = data['summary']
    daily_perf = data.get('daily_performance', [])

    # Calculate drawdown from daily performance
    max_dd = 0
    peak = 0
    for day in daily_perf:
        cum_pnl = day['cumulative_pnl']
        if cum_pnl > peak:
            peak = cum_pnl
        dd = peak - cum_pnl
        if dd > max_dd:
            max_dd = dd

    # Calculate profit factor
    wins_total = sum(day['daily_pnl'] for day in daily_perf if day['daily_pnl'] > 0)
    losses_total = abs(sum(day['daily_pnl'] for day in daily_perf if day['daily_pnl'] < 0))
    profit_factor = wins_total / losses_total if losses_total > 0 else float('inf')

    # Calculate risk-adjusted return (total pips / max drawdown)
    risk_adjusted_return = summary['total_pnl'] / max_dd if max_dd > 0 else summary['total_pnl']

    return {
        'threshold': threshold,
        'total_trades': summary['total_trades'],
        'win_rate': summary['win_rate'],
        'total_pnl': summary['total_pnl'],
        'avg_daily_pnl': summary['avg_daily_pnl'],
        'profitable_days': summary['profitable_days'],
        'total_days': summary['total_days'],
        'max_drawdown': max_dd,
        'profit_factor': profit_factor,
        'risk_adjusted_return': risk_adjusted_return,
        'trades_per_day': summary['total_trades'] / DAYS if DAYS > 0 else 0
    }

def main():
    """Run threshold analysis."""
    print("=" * 80)
    print(f"CONFIDENCE THRESHOLD OPTIMIZATION ANALYSIS ({DAYS}-Day What-If)")
    print("=" * 80)
    print()

    results = []

    for threshold in THRESHOLDS:
        print(f"Testing threshold: {threshold:.2f}...", end=" ", flush=True)
        data = get_whatif_performance(threshold)

        if data:
            analysis = analyze_threshold(threshold, data)
            if analysis:
                results.append(analysis)
                print(f"✓ {analysis['total_trades']} trades, {analysis['win_rate']:.1f}% WR, {analysis['total_pnl']:.1f} pips")
            else:
                print("✗ No valid data")
        else:
            print("✗ API error")

    if not results:
        print("\n❌ No results obtained. Check if backend is running.")
        return

    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print()

    # Print table header
    header = f"{'Threshold':>10} | {'Trades':>7} | {'Win%':>6} | {'Total':>8} | {'Daily':>7} | {'MaxDD':>7} | {'PF':>5} | {'RAR':>6}"
    print(header)
    print("-" * len(header))

    # Print each result
    for r in results:
        pf_str = f"{r['profit_factor']:.2f}" if r['profit_factor'] != float('inf') else "∞"
        print(f"{r['threshold']:>10.2f} | {r['total_trades']:>7} | {r['win_rate']:>6.1f} | "
              f"{r['total_pnl']:>8.1f} | {r['avg_daily_pnl']:>7.1f} | "
              f"{r['max_drawdown']:>7.1f} | {pf_str:>5} | {r['risk_adjusted_return']:>6.2f}")

    print()
    print("Legend:")
    print("  Trades = Total trades in period")
    print("  Win%   = Win rate percentage")
    print("  Total  = Total P&L in pips")
    print("  Daily  = Average daily P&L")
    print("  MaxDD  = Maximum drawdown in pips")
    print("  PF     = Profit Factor (gross wins / gross losses)")
    print("  RAR    = Risk-Adjusted Return (Total / MaxDD)")
    print()

    # Find optimal thresholds
    print("=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print()

    # Best by total profit
    best_profit = max(results, key=lambda x: x['total_pnl'])
    print(f"✓ Highest Total Profit:  {best_profit['threshold']:.2f} ({best_profit['total_pnl']:.1f} pips)")

    # Best by win rate
    best_winrate = max(results, key=lambda x: x['win_rate'])
    print(f"✓ Highest Win Rate:      {best_winrate['threshold']:.2f} ({best_winrate['win_rate']:.1f}%)")

    # Best by risk-adjusted return
    best_rar = max(results, key=lambda x: x['risk_adjusted_return'])
    print(f"✓ Best Risk-Adjusted:    {best_rar['threshold']:.2f} (RAR: {best_rar['risk_adjusted_return']:.2f})")

    # Best by profit factor
    valid_pf = [r for r in results if r['profit_factor'] != float('inf')]
    if valid_pf:
        best_pf = max(valid_pf, key=lambda x: x['profit_factor'])
        print(f"✓ Best Profit Factor:    {best_pf['threshold']:.2f} (PF: {best_pf['profit_factor']:.2f})")

    # Filter results with minimum trade frequency (at least 0.5 trades/day = 15 trades in 30 days)
    active_results = [r for r in results if r['total_trades'] >= 15]

    if active_results:
        print()
        print("Among thresholds with sufficient activity (≥15 trades):")
        best_active_profit = max(active_results, key=lambda x: x['total_pnl'])
        best_active_rar = max(active_results, key=lambda x: x['risk_adjusted_return'])

        print(f"  → Best Total Profit:   {best_active_profit['threshold']:.2f} ({best_active_profit['total_pnl']:.1f} pips, {best_active_profit['total_trades']} trades)")
        print(f"  → Best Risk-Adjusted:  {best_active_rar['threshold']:.2f} (RAR: {best_active_rar['risk_adjusted_return']:.2f}, {best_active_rar['total_trades']} trades)")

    print()
    print("=" * 80)
    print("SWEET SPOT RECOMMENDATION")
    print("=" * 80)

    # Calculate a composite score: (Total PnL * Win Rate * Trades) / MaxDD
    for r in results:
        # Penalize very low trade counts
        trade_penalty = min(1.0, r['total_trades'] / 20)
        r['composite_score'] = (r['total_pnl'] * r['win_rate'] / 100 * trade_penalty) / (r['max_drawdown'] + 1)

    best_overall = max(results, key=lambda x: x['composite_score'])

    print()
    print(f"🎯 OPTIMAL THRESHOLD: {best_overall['threshold']:.2f}")
    print()
    print(f"   Trades:           {best_overall['total_trades']}")
    print(f"   Win Rate:         {best_overall['win_rate']:.1f}%")
    print(f"   Total P&L:        {best_overall['total_pnl']:.1f} pips")
    print(f"   Avg Daily P&L:    {best_overall['avg_daily_pnl']:.1f} pips")
    print(f"   Max Drawdown:     {best_overall['max_drawdown']:.1f} pips")
    print(f"   Profit Factor:    {best_overall['profit_factor']:.2f}")
    print(f"   Trades/Day:       {best_overall['trades_per_day']:.2f}")
    print()
    print(f"This threshold provides the best balance of profitability, risk management,")
    print(f"and trade frequency over the {DAYS}-day simulation period.")
    print()

if __name__ == "__main__":
    main()
