#!/usr/bin/env python3
"""Analyze optimal confidence threshold over extended periods using What-If API."""

import requests
import json
from typing import Dict, List

# Test different thresholds
THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70]
API_BASE = "http://localhost:8001/api/v1"
TEST_PERIODS = [30, 60, 90]  # Test over multiple periods

def get_whatif_performance(threshold: float, days: int) -> Dict:
    """Fetch What-If performance for a given threshold and period."""
    url = f"{API_BASE}/trading/whatif-performance"
    params = {
        "days": days,
        "confidence_threshold": threshold,
        "require_agreement": True
    }

    try:
        response = requests.get(url, params=params, timeout=120)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error: {e}")
        return None

def analyze_threshold(threshold: float, days: int, data: Dict) -> Dict:
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

    # Risk-adjusted return
    risk_adjusted_return = summary['total_pnl'] / max_dd if max_dd > 0 else summary['total_pnl']

    return {
        'threshold': threshold,
        'days': days,
        'total_trades': summary['total_trades'],
        'win_rate': summary['win_rate'],
        'total_pnl': summary['total_pnl'],
        'avg_daily_pnl': summary['avg_daily_pnl'],
        'max_drawdown': max_dd,
        'profit_factor': profit_factor,
        'risk_adjusted_return': risk_adjusted_return,
        'trades_per_day': summary['total_trades'] / days if days > 0 else 0
    }

def main():
    """Run extended threshold analysis."""
    print("=" * 100)
    print("EXTENDED CONFIDENCE THRESHOLD OPTIMIZATION")
    print("=" * 100)
    print()

    all_results = {}

    for days in TEST_PERIODS:
        print(f"\n{'=' * 100}")
        print(f"TESTING {days}-DAY PERIOD")
        print("=" * 100)
        print()

        results = []

        for threshold in THRESHOLDS:
            print(f"  Testing {threshold:.2f}...", end=" ", flush=True)
            data = get_whatif_performance(threshold, days)

            if data:
                analysis = analyze_threshold(threshold, days, data)
                if analysis:
                    results.append(analysis)
                    print(f"✓ {analysis['total_trades']} trades, {analysis['win_rate']:.1f}% WR, {analysis['total_pnl']:.1f} pips")
                else:
                    print("✗ No data")
            else:
                print("✗ Error")

        all_results[days] = results

        if results:
            # Print summary for this period
            print()
            header = f"{'Threshold':>10} | {'Trades':>7} | {'Win%':>6} | {'Total':>9} | {'Daily':>7} | {'MaxDD':>8} | {'PF':>6} | {'RAR':>7}"
            print(header)
            print("-" * len(header))

            for r in results:
                pf_str = f"{r['profit_factor']:.2f}" if r['profit_factor'] != float('inf') else "∞"
                print(f"{r['threshold']:>10.2f} | {r['total_trades']:>7} | {r['win_rate']:>6.1f} | "
                      f"{r['total_pnl']:>9.1f} | {r['avg_daily_pnl']:>7.1f} | "
                      f"{r['max_drawdown']:>8.1f} | {pf_str:>6} | {r['risk_adjusted_return']:>7.2f}")

    # Cross-period analysis
    print()
    print("=" * 100)
    print("CROSS-PERIOD ANALYSIS")
    print("=" * 100)
    print()

    # For each threshold, show performance across all periods
    for threshold in THRESHOLDS:
        threshold_data = []
        for days in TEST_PERIODS:
            period_results = all_results.get(days, [])
            threshold_result = next((r for r in period_results if r['threshold'] == threshold), None)
            if threshold_result:
                threshold_data.append(threshold_result)

        if threshold_data:
            print(f"\nThreshold: {threshold:.2f}")
            print("-" * 100)

            header = f"{'Period':>8} | {'Trades':>7} | {'Win%':>6} | {'Total':>9} | {'Daily':>7} | {'MaxDD':>8} | {'PF':>6} | {'T/Day':>6}"
            print(header)

            for r in threshold_data:
                pf_str = f"{r['profit_factor']:.2f}" if r['profit_factor'] != float('inf') else "∞"
                print(f"{r['days']:>6}d | {r['total_trades']:>7} | {r['win_rate']:>6.1f} | "
                      f"{r['total_pnl']:>9.1f} | {r['avg_daily_pnl']:>7.1f} | "
                      f"{r['max_drawdown']:>8.1f} | {pf_str:>6} | {r['trades_per_day']:>6.2f}")

            # Calculate averages
            avg_win_rate = sum(r['win_rate'] for r in threshold_data) / len(threshold_data)
            avg_daily_pnl = sum(r['avg_daily_pnl'] for r in threshold_data) / len(threshold_data)
            avg_trades_per_day = sum(r['trades_per_day'] for r in threshold_data) / len(threshold_data)
            total_all_trades = sum(r['total_trades'] for r in threshold_data)

            print(f"{'AVG':>6}   | {total_all_trades:>7} | {avg_win_rate:>6.1f} | {'':>9} | {avg_daily_pnl:>7.1f} | {'':>8} | {'':>6} | {avg_trades_per_day:>6.2f}")

    # Final recommendation
    print()
    print("=" * 100)
    print("FINAL RECOMMENDATION")
    print("=" * 100)
    print()

    # Calculate composite scores across all periods
    threshold_scores = {}

    for threshold in THRESHOLDS:
        scores = []
        total_trades_all = 0
        total_pnl_all = 0

        for days in TEST_PERIODS:
            period_results = all_results.get(days, [])
            result = next((r for r in period_results if r['threshold'] == threshold), None)

            if result:
                # Composite score: balance profit, win rate, and trade frequency
                trade_penalty = min(1.0, result['trades_per_day'] / 1.5)  # Aim for ~1.5 trades/day
                score = (result['total_pnl'] * result['win_rate'] / 100 * trade_penalty) / (result['max_drawdown'] + 1)
                scores.append(score)
                total_trades_all += result['total_trades']
                total_pnl_all += result['total_pnl']

        if scores:
            avg_score = sum(scores) / len(scores)
            threshold_scores[threshold] = {
                'avg_score': avg_score,
                'total_trades': total_trades_all,
                'total_pnl': total_pnl_all,
                'consistency': min(scores) / max(scores) if max(scores) > 0 else 0  # How consistent across periods
            }

    # Find best threshold
    if threshold_scores:
        best_threshold = max(threshold_scores.items(), key=lambda x: x[1]['avg_score'])
        threshold_val, metrics = best_threshold

        print(f"🎯 OPTIMAL THRESHOLD: {threshold_val:.2f}")
        print()
        print(f"   Total Trades (all periods):  {metrics['total_trades']}")
        print(f"   Total P&L (all periods):     {metrics['total_pnl']:.1f} pips")
        print(f"   Consistency Score:           {metrics['consistency']:.2f}")
        print()

        # Show detailed stats for this threshold
        print(f"Performance across test periods:")
        for days in TEST_PERIODS:
            period_results = all_results.get(days, [])
            result = next((r for r in period_results if r['threshold'] == threshold_val), None)
            if result:
                print(f"   {days:>2}d: {result['total_trades']:>3} trades, {result['win_rate']:.1f}% WR, "
                      f"{result['total_pnl']:>7.1f} pips, {result['avg_daily_pnl']:>5.1f} pips/day")

        print()
        print("This threshold offers the best balance of:")
        print("  • Profitability (total pips)")
        print("  • Win rate (accuracy)")
        print("  • Trade frequency (opportunity)")
        print("  • Risk management (drawdown)")
        print("  • Consistency across different time periods")
        print()

        # Compare to other thresholds
        print("Comparison to alternatives:")
        sorted_thresholds = sorted(threshold_scores.items(), key=lambda x: x[1]['avg_score'], reverse=True)
        for i, (t, m) in enumerate(sorted_thresholds[:3], 1):
            status = "← RECOMMENDED" if t == threshold_val else ""
            print(f"  {i}. {t:.2f}: {m['total_pnl']:>8.1f} pips, {m['total_trades']:>3} trades {status}")

    print()

if __name__ == "__main__":
    main()
