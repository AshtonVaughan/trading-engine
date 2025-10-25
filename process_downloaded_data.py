"""
Process Downloaded M1/M5 CSV Data

Converts the downloaded CSV files to parquet format for TradingEngine.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

def process_csv_to_parquet(csv_path, output_path, symbol, timeframe):
    """Process a CSV file and convert to parquet with proper columns."""

    print(f"\nProcessing {symbol} {timeframe}...")
    print(f"  Input: {csv_path}")

    # Try to detect delimiter (tab or comma)
    with open(csv_path, 'r') as f:
        first_line = f.readline()
        delimiter = '\t' if '\t' in first_line else ','

    # Read CSV without header
    df = pd.read_csv(
        csv_path,
        names=['time', 'open', 'high', 'low', 'close', 'tick_volume'],
        parse_dates=['time'],
        sep=delimiter
    )

    print(f"  Loaded {len(df):,} rows")
    print(f"  Date range: {df['time'].min()} to {df['time'].max()}")

    # Add missing columns expected by feature engineering
    df['spread'] = 0  # Not available in this data
    df['real_volume'] = 0  # Not available in this data

    # Calculate time differences to detect gaps
    df = df.sort_values('time')
    df['time_diff'] = df['time'].diff()

    # Check for gaps
    expected_freq = pd.Timedelta('1min') if timeframe == 'M1' else pd.Timedelta('5min')
    gaps = df[df['time_diff'] > expected_freq * 2]
    if len(gaps) > 0:
        print(f"  WARNING: Found {len(gaps)} gaps >2x expected frequency")
        largest_gap = df['time_diff'].max()
        print(f"  Largest gap: {largest_gap}")

    # Set time as index
    df = df.set_index('time')

    # Remove time_diff column before saving
    df = df.drop('time_diff', axis=1)

    # Save to parquet
    df.to_parquet(output_path, compression='snappy')
    file_size_mb = Path(output_path).stat().st_size / 1024 / 1024

    print(f"  Saved {len(df):,} bars to {output_path}")
    print(f"  File size: {file_size_mb:.1f} MB")

    return df


def main():
    """Process all downloaded CSV files."""

    collectors_dir = Path("data/collectors")
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Define the CSV files to process
    files_to_process = [
        # M1 data (for scalping)
        ('EURUSD_M1.csv', 'EURUSD', 'M1'),
        ('GBPUSD_M1.csv', 'GBPUSD', 'M1'),
        ('USDCHF_M1.csv', 'USDCHF', 'M1'),
        ('USDJPY1.csv', 'USDJPY', 'M1'),  # Note: filename is USDJPY1.csv not USDJPY_M1.csv

        # M5 data (optional, for multi-timeframe analysis)
        ('EURUSD_M5.csv', 'EURUSD', 'M5'),
        ('GBPUSD_M5.csv', 'GBPUSD', 'M5'),
        ('USDCHF_M5.csv', 'USDCHF', 'M5'),
        ('USDJPY5.csv', 'USDJPY', 'M5'),
    ]

    print("="*80)
    print("Processing Downloaded FOREX Data")
    print("="*80)

    processed_files = []

    for csv_file, symbol, timeframe in files_to_process:
        csv_path = collectors_dir / csv_file

        if not csv_path.exists():
            print(f"\nWARNING: {csv_path} not found, skipping...")
            continue

        output_path = raw_dir / f"{symbol}_{timeframe}.parquet"

        try:
            df = process_csv_to_parquet(csv_path, output_path, symbol, timeframe)
            processed_files.append((symbol, timeframe, len(df)))
        except Exception as e:
            print(f"  ERROR processing {csv_file}: {e}")
            continue

    # Summary
    print("\n" + "="*80)
    print("Processing Summary")
    print("="*80)

    for symbol, timeframe, rows in processed_files:
        print(f"  {symbol} {timeframe}: {rows:,} bars")

    print(f"\nTotal files processed: {len(processed_files)}")
    print(f"\nRaw data ready in: {raw_dir.absolute()}")
    print("\nNext steps:")
    print("  1. Update config/config.yaml to use M1 timeframe")
    print("  2. Run feature engineering: python data/features/feature_engineering.py")
    print("  3. Commit to git and push to GitHub for H200 training")


if __name__ == '__main__':
    main()
