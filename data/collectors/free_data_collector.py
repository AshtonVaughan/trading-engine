"""
Free Data Collector (No MT5 Required)

Collects FOREX data from free APIs for cloud GPU training.
Works on Windows, Linux, or cloud servers without MT5.

Data sources:
- Yahoo Finance (yfinance) - Good for hourly/daily data
- Alpha Vantage - Good for intraday (requires free API key)
"""

import yaml
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import time
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.logger import logger, setup_logger

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    logger.warning("yfinance not installed. Run: pip install yfinance")


class FreeDataCollector:
    """
    Collect FOREX data from free APIs (no MT5 required).
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize collector with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Extract settings
        self.symbols = self.config['trading']['pairs']
        self.timeframe = self.config['trading']['timeframe']
        self.start_date = self.config['data']['start_date']
        self.end_date = self.config['data']['end_date']
        self.raw_data_dir = Path(self.config['data']['raw_data_dir'])

        # Create output directory
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)

        # Map symbols to Yahoo Finance format
        self.symbol_map = {
            'EURUSD.': 'EURUSD=X',
            'EURUSD': 'EURUSD=X',
            'GBPUSD.': 'GBPUSD=X',
            'GBPUSD': 'GBPUSD=X',
            'USDJPY.': 'USDJPY=X',
            'USDJPY': 'USDJPY=X',
            'AUDUSD.': 'AUDUSD=X',
            'AUDUSD': 'AUDUSD=X',
            'USDCAD.': 'USDCAD=X',
            'USDCAD': 'USDCAD=X'
        }

        # Timeframe mapping
        self.interval_map = {
            'M1': '1m',
            'M5': '5m',
            'M15': '15m',
            'M30': '30m',
            'H1': '1h',
            'H4': '4h',
            'D1': '1d'
        }

        logger.info(f"FreeDataCollector initialized")
        logger.info(f"  Symbols: {self.symbols}")
        logger.info(f"  Timeframe: {self.timeframe}")
        logger.info(f"  Date range: {self.start_date} to {self.end_date}")

    def collect_yfinance(self, symbol: str) -> pd.DataFrame:
        """
        Collect data from Yahoo Finance.

        Note: Yahoo Finance has limitations:
        - 1m data: Only last 7 days available
        - 5m data: Only last 60 days available
        - 1h data: Up to 730 days available
        - 1d data: Years of history available

        Args:
            symbol: Trading symbol (e.g., "EURUSD" or "EURUSD.")

        Returns:
            DataFrame with OHLCV data
        """
        if not YFINANCE_AVAILABLE:
            logger.error("yfinance not installed!")
            return pd.DataFrame()

        # Map to Yahoo Finance symbol
        yf_symbol = self.symbol_map.get(symbol, symbol)
        interval = self.interval_map.get(self.timeframe, '1h')

        logger.info(f"\nCollecting data for {symbol} ({yf_symbol})...")
        logger.info(f"  Timeframe: {interval}")

        try:
            # Parse dates
            start = datetime.strptime(self.start_date, "%Y-%m-%d")
            end = datetime.strptime(self.end_date, "%Y-%m-%d")

            # Download data
            ticker = yf.Ticker(yf_symbol)
            df = ticker.history(start=start, end=end, interval=interval)

            if df.empty:
                logger.warning(f"  No data received for {symbol}")
                return df

            # Rename columns to match MT5 format
            df = df.reset_index()
            df.columns = [c.lower() for c in df.columns]

            # Rename 'datetime' to 'time' if present
            if 'datetime' in df.columns:
                df.rename(columns={'datetime': 'time'}, inplace=True)

            # Add missing columns
            if 'tick_volume' not in df.columns and 'volume' in df.columns:
                df['tick_volume'] = df['volume']
            if 'spread' not in df.columns:
                df['spread'] = 0  # Yahoo doesn't provide spread
            if 'real_volume' not in df.columns:
                df['real_volume'] = 0

            # Reorder columns
            expected_cols = ['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']
            for col in expected_cols:
                if col not in df.columns:
                    df[col] = 0

            df = df[expected_cols]

            logger.info(f"  Collected {len(df):,} bars")
            logger.info(f"  Date range: {df['time'].min()} to {df['time'].max()}")

            # Save to parquet
            output_file = self.raw_data_dir / f"{symbol.replace('.', '')}_{self.timeframe}.parquet"
            df.to_parquet(output_file, index=False)
            logger.info(f"  Saved to {output_file}")

            return df

        except Exception as e:
            logger.error(f"  Error collecting {symbol}: {e}")
            return pd.DataFrame()

    def collect_all(self) -> dict:
        """
        Collect data for all symbols.

        Returns:
            Dictionary mapping symbol to DataFrame
        """
        logger.info("\n" + "="*80)
        logger.info("Starting Free Data Collection (Yahoo Finance)")
        logger.info("="*80)

        all_data = {}

        for symbol in self.symbols:
            df = self.collect_yfinance(symbol)
            if not df.empty:
                all_data[symbol] = df
            time.sleep(1)  # Be nice to Yahoo's servers

        logger.info("\n" + "="*80)
        logger.info("Data Collection Summary")
        logger.info("="*80)

        for symbol, df in all_data.items():
            logger.info(f"{symbol}: {len(df):,} bars")

        logger.info(f"\nTotal data collected: {sum(len(df) for df in all_data.values()):,} bars")

        return all_data


def main():
    """Main data collection script."""
    import argparse

    parser = argparse.ArgumentParser(description="Collect FOREX data from free APIs")
    parser.add_argument('--config', default='config/config.yaml', help='Path to config file')
    args = parser.parse_args()

    # Setup logger
    setup_logger(name="TradingEngine", level="INFO", log_file="logs/data_collection.log")

    # Collect data
    collector = FreeDataCollector(config_path=args.config)
    all_data = collector.collect_all()

    if all_data:
        logger.info("\n✓ Data collection successful!")
        logger.info(f"✓ Collected data for {len(all_data)} symbols")
        logger.info(f"✓ Data saved to {collector.raw_data_dir}")
        logger.info(f"✓ Next step: Build feature engineering pipeline")
    else:
        logger.error("\n✗ Data collection failed!")
        logger.error("✗ Install yfinance: pip install yfinance")


if __name__ == '__main__':
    main()
