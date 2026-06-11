"""
MT5 Historical Data Collector

Collects 3+ years of 1-minute FOREX data from MetaTrader 5 for RL training.
"""

import yaml
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.logger import logger, setup_logger
from utils.mt5_bridge import MT5Bridge


class MT5HistoricalCollector:
    """
    Collect historical M1 FOREX data from MT5.
    """

    def __init__(self, config_path: str = "config/config.yaml", mt5_config_path: str = "config/mt5_config.yaml"):
        """
        Initialize collector with configuration.

        Args:
            config_path: Path to main config file
            mt5_config_path: Path to MT5 config file
        """
        # Load configs
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        with open(mt5_config_path, 'r') as f:
            self.mt5_config = yaml.safe_load(f)

        # Extract settings
        self.symbols = self.config['trading']['pairs']
        self.timeframe = self.config['trading']['timeframe']
        self.start_date = datetime.strptime(self.config['data']['start_date'], "%Y-%m-%d")
        self.end_date = datetime.strptime(self.config['data']['end_date'], "%Y-%m-%d")
        self.raw_data_dir = Path(self.config['data']['raw_data_dir'])

        # MT5 settings
        mt5_settings = self.mt5_config['mt5']
        self.mt5 = MT5Bridge(
            login=mt5_settings.get('login'),
            password=mt5_settings.get('password'),
            server=mt5_settings.get('server'),
            path=mt5_settings.get('terminal_path', {}).get('windows'),
            timeout=mt5_settings.get('timeout_ms', 60000)
        )

        # Create output directory
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"MT5HistoricalCollector initialized")
        logger.info(f"  Symbols: {self.symbols}")
        logger.info(f"  Timeframe: {self.timeframe}")
        logger.info(f"  Date range: {self.start_date} to {self.end_date}")

    def collect_symbol(self, symbol: str) -> pd.DataFrame:
        """
        Collect historical data for a single symbol.

        Args:
            symbol: Trading symbol (e.g., "EURUSD")

        Returns:
            DataFrame with OHLCV data
        """
        logger.info(f"\nCollecting data for {symbol}...")

        # Calculate expected number of bars
        days = (self.end_date - self.start_date).days
        expected_bars = days * 24 * 60  # Minutes in total period

        logger.info(f"  Expected ~{expected_bars:,} bars for {days} days")

        # Collect data in batches
        df = self.mt5.get_historical_data_batched(
            symbol=symbol,
            timeframe=self.timeframe,
            start_date=self.start_date,
            end_date=self.end_date,
            batch_size=100000
        )

        if df.empty:
            logger.error(f"  No data collected for {symbol}")
            return df

        # Data quality checks
        logger.info(f"  Collected {len(df):,} bars")
        logger.info(f"  Date range: {df['time'].min()} to {df['time'].max()}")
        logger.info(f"  Missing data: {100 * (1 - len(df) / expected_bars):.2f}%")

        # Check for gaps
        df['time_diff'] = df['time'].diff()
        large_gaps = df[df['time_diff'] > timedelta(minutes=5)]
        if len(large_gaps) > 0:
            logger.warning(f"  Found {len(large_gaps)} gaps >5 minutes")

        # Save raw data
        output_file = self.raw_data_dir / f"{symbol}_{self.timeframe}.parquet"
        df.to_parquet(output_file, index=False)
        logger.info(f"  Saved to {output_file}")

        return df

    def collect_all(self) -> Dict[str, pd.DataFrame]:
        """
        Collect data for all symbols.

        Returns:
            Dictionary mapping symbol to DataFrame
        """
        logger.info("\n" + "="*80)
        logger.info("Starting MT5 Historical Data Collection")
        logger.info("="*80)

        # Connect to MT5
        if not self.mt5.connect():
            logger.error("Failed to connect to MT5. Exiting.")
            return {}

        all_data = {}

        try:
            for symbol in self.symbols:
                df = self.collect_symbol(symbol)
                if not df.empty:
                    all_data[symbol] = df

            logger.info("\n" + "="*80)
            logger.info("Data Collection Summary")
            logger.info("="*80)

            for symbol, df in all_data.items():
                logger.info(f"{symbol}: {len(df):,} bars")

            logger.info(f"\nTotal data collected: {sum(len(df) for df in all_data.values()):,} bars")

        finally:
            self.mt5.disconnect()

        return all_data

    def validate_data(self, df: pd.DataFrame) -> Dict:
        """
        Validate collected data quality.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Dictionary with validation metrics
        """
        metrics = {
            'num_bars': len(df),
            'num_missing': df.isnull().sum().to_dict(),
            'num_duplicates': df.duplicated(subset=['time']).sum(),
            'num_zero_volume': (df['tick_volume'] == 0).sum(),
            'date_range': {
                'start': df['time'].min(),
                'end': df['time'].max()
            }
        }

        # Check for price anomalies
        df['return'] = df['close'].pct_change()
        large_moves = df[abs(df['return']) > 0.05]  # >5% moves (likely errors)
        metrics['num_large_moves'] = len(large_moves)

        return metrics


def main():
    """Main data collection script."""
    import argparse

    parser = argparse.ArgumentParser(description="Collect historical FOREX data from MT5")
    parser.add_argument('--config', default='config/config.yaml', help='Path to config file')
    parser.add_argument('--mt5-config', default='config/mt5_config.yaml', help='Path to MT5 config file')
    args = parser.parse_args()

    # Setup logger
    setup_logger(name="TradingEngine", level="INFO", log_file="logs/data_collection.log")

    # Collect data
    collector = MT5HistoricalCollector(
        config_path=args.config,
        mt5_config_path=args.mt5_config
    )

    all_data = collector.collect_all()

    if all_data:
        logger.info("\n✓ Data collection successful!")
        logger.info(f"✓ Collected data for {len(all_data)} symbols")
        logger.info(f"✓ Next step: python data/features/feature_engineer.py")
    else:
        logger.error("\n✗ Data collection failed!")


if __name__ == '__main__':
    main()
