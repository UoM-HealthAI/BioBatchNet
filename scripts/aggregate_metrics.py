#!/usr/bin/env python3
"""Aggregate metrics from multiple seed runs and generate summary table."""

import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np


def aggregate_metrics(run_dir: str, output_format: str = 'table') -> pd.DataFrame:
    """Aggregate metrics across seeds from a run directory.

    Args:
        run_dir: Path to run directory containing seed_* subdirectories
        output_format: 'table' for formatted output, 'csv' for CSV

    Returns:
        DataFrame with metrics per seed and summary statistics
    """
    run_dir = Path(run_dir)
    all_metrics = []

    # Collect metrics from each seed directory
    for seed_dir in sorted(run_dir.glob('seed_*')):
        csv_path = seed_dir / 'metrics.csv'
        if not csv_path.exists():
            print(f"Warning: {csv_path} not found, skipping")
            continue

        df = pd.read_csv(csv_path)
        if df.empty:
            continue

        row = df.iloc[-1]  # last row (final evaluation)
        metrics = {'seed': seed_dir.name}

        # Parse eval metrics (JSON format)
        if 'eval' in row and pd.notna(row['eval']):
            try:
                eval_dict = json.loads(row['eval'])
                metrics.update(eval_dict)
            except json.JSONDecodeError:
                print(f"Warning: Failed to parse eval JSON in {seed_dir.name}")

        # Parse independence metrics if present
        if 'independence' in row and pd.notna(row['independence']):
            try:
                inde_dict = json.loads(row['independence'])
                metrics.update(inde_dict)
            except json.JSONDecodeError:
                pass

        all_metrics.append(metrics)

    if not all_metrics:
        print(f"No metrics found in {run_dir}")
        return pd.DataFrame()

    # Create DataFrame with all seeds
    df = pd.DataFrame(all_metrics)
    df = df.set_index('seed')

    # Compute summary statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    summary = pd.DataFrame({
        'mean': df[numeric_cols].mean(),
        'std': df[numeric_cols].std(),
    })

    return df, summary


def print_table(df: pd.DataFrame, summary: pd.DataFrame, run_name: str = None):
    """Print formatted table to console."""
    if run_name:
        print(f"\n{'='*60}")
        print(f"Run: {run_name}")
        print(f"{'='*60}")

    # Print per-seed metrics
    print("\n[Per-Seed Metrics]")
    print(df.to_string(float_format=lambda x: f'{x:.4f}'))

    # Print summary
    print(f"\n[Summary Statistics (n={len(df)})]")
    print("-" * 40)
    print(f"{'Metric':<20} {'Mean':>10} {'Std':>10}")
    print("-" * 40)
    for metric in summary.index:
        mean_val = summary.loc[metric, 'mean']
        std_val = summary.loc[metric, 'std']
        print(f"{metric:<20} {mean_val:>10.4f} {std_val:>10.4f}")
    print("-" * 40)

    # Print key metrics in compact format
    key_metrics = ['BatchScore', 'BioScore', 'TotalScore']
    available = [m for m in key_metrics if m in summary.index]
    if available:
        print("\n[Key Metrics: mean ± std]")
        for m in available:
            mean_val = summary.loc[m, 'mean']
            std_val = summary.loc[m, 'std']
            print(f"  {m}: {mean_val:.4f} ± {std_val:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Aggregate metrics from seed runs')
    parser.add_argument('run_dir', type=str, help='Path to run directory with seed_* subdirs')
    parser.add_argument('--csv', action='store_true', help='Save results as CSV')
    parser.add_argument('--output', '-o', type=str, default=None, help='Output file path')
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"Error: {run_dir} does not exist")
        return

    df, summary = aggregate_metrics(run_dir)
    if df.empty:
        return

    run_name = run_dir.name

    if args.csv:
        # Save to CSV
        output_path = args.output or (run_dir / 'summary.csv')
        summary.to_csv(output_path)
        print(f"Saved summary to {output_path}")

        # Also save per-seed metrics
        seeds_path = run_dir / 'seeds_metrics.csv'
        df.to_csv(seeds_path)
        print(f"Saved per-seed metrics to {seeds_path}")
    else:
        print_table(df, summary, run_name)


if __name__ == '__main__':
    main()
