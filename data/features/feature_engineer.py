"""
Feature Engineering Pipeline for FOREX Trading

Generates 100+ features from OHLCV data:
- Price features (returns, volatility, VWAP)
- Technical indicators (RSI, MACD, Bollinger Bands, ATR, Stochastic, ADX, EMA, SMA)
- Market microstructure (bid-ask spread, volume)
- Time features (hour, day of week, session)
"""

import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.logger import logger, setup_logger


class FeatureEngineer:
    """
    Create features for RL training from raw OHLCV data.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize feature engineer with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.feature_config = self.config['features']
        self.lookback = self.feature_config['lookback_bars']

        self.raw_data_dir = Path(self.config['data']['raw_data_dir'])
        self.processed_data_dir = Path(self.config['data']['processed_data_dir'])
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)

        logger.info("FeatureEngineer initialized")
        logger.info(f"  Lookback: {self.lookback} bars")

    def calculate_returns(self, df: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
        """Calculate returns over multiple periods."""
        for period in periods:
            df[f'return_{period}'] = df['close'].pct_change(period)
        return df

    def calculate_volatility(self, df: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
        """Calculate rolling volatility (std of returns)."""
        for period in periods:
            df[f'volatility_{period}'] = df['close'].pct_change().rolling(period).std()
        return df

    def calculate_vwap(self, df: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
        """Calculate Volume-Weighted Average Price."""
        typical_price = (df['high'] + df['low'] + df['close']) / 3

        for period in periods:
            vwap = (typical_price * df['tick_volume']).rolling(period).sum() / \
                   df['tick_volume'].rolling(period).sum()
            df[f'vwap_{period}'] = vwap
            df[f'vwap_dist_{period}'] = (df['close'] - vwap) / vwap

        return df

    def calculate_rsi(self, df: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
        """Calculate Relative Strength Index."""
        for period in periods:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()

            rs = gain / loss
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))

        return df

    def calculate_macd(self, df: pd.DataFrame, fast: int, slow: int, signal: int) -> pd.DataFrame:
        """Calculate MACD (Moving Average Convergence Divergence)."""
        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False).mean()

        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        return df

    def calculate_bollinger_bands(self, df: pd.DataFrame, period: int, std: float) -> pd.DataFrame:
        """Calculate Bollinger Bands."""
        sma = df['close'].rolling(period).mean()
        rolling_std = df['close'].rolling(period).std()

        df['bb_upper'] = sma + (rolling_std * std)
        df['bb_middle'] = sma
        df['bb_lower'] = sma - (rolling_std * std)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        return df

    def calculate_atr(self, df: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
        """Calculate Average True Range."""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        for period in periods:
            df[f'atr_{period}'] = true_range.rolling(period).mean()
            df[f'atr_pct_{period}'] = df[f'atr_{period}'] / df['close']

        return df

    def calculate_stochastic(self, df: pd.DataFrame, k: int, d: int, smooth: int) -> pd.DataFrame:
        """Calculate Stochastic Oscillator."""
        low_min = df['low'].rolling(k).min()
        high_max = df['high'].rolling(k).max()

        df['stoch_k'] = 100 * (df['close'] - low_min) / (high_max - low_min)
        df['stoch_k'] = df['stoch_k'].rolling(smooth).mean()
        df['stoch_d'] = df['stoch_k'].rolling(d).mean()

        return df

    def calculate_adx(self, df: pd.DataFrame, period: int) -> pd.DataFrame:
        """Calculate Average Directional Index."""
        high_diff = df['high'].diff()
        low_diff = -df['low'].diff()

        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)

        # Calculate ATR for normalization
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = pd.concat([pd.Series(high_low), pd.Series(high_close), pd.Series(low_close)], axis=1).max(axis=1)
        atr = true_range.rolling(period).mean()

        plus_di = 100 * pd.Series(plus_dm).rolling(period).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).rolling(period).mean() / atr

        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        df['adx'] = dx.rolling(period).mean()
        df['plus_di'] = plus_di
        df['minus_di'] = minus_di

        return df

    def calculate_ema(self, df: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
        """Calculate Exponential Moving Averages."""
        for period in periods:
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
            df[f'ema_dist_{period}'] = (df['close'] - df[f'ema_{period}']) / df[f'ema_{period}']
        return df

    def calculate_sma(self, df: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
        """Calculate Simple Moving Averages."""
        for period in periods:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            df[f'sma_dist_{period}'] = (df['close'] - df[f'sma_{period}']) / df[f'sma_{period}']
        return df

    def calculate_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume-based features."""
        df['volume_sma_10'] = df['tick_volume'].rolling(10).mean()
        df['volume_sma_20'] = df['tick_volume'].rolling(20).mean()
        df['volume_ratio'] = df['tick_volume'] / df['volume_sma_20']

        # Volume-price trend
        df['vpt'] = df['tick_volume'] * ((df['close'] - df['close'].shift(1)) / df['close'].shift(1))
        df['vpt_sma'] = df['vpt'].rolling(20).mean()

        return df

    def calculate_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate time-based features (cyclical encoding)."""
        # Hour of day (0-23)
        df['hour'] = df['time'].dt.hour
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

        # Day of week (0-6)
        df['day_of_week'] = df['time'].dt.dayofweek
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        # Trading session flags
        df['asian_session'] = ((df['hour'] >= 0) & (df['hour'] < 9)).astype(int)
        df['london_session'] = ((df['hour'] >= 8) & (df['hour'] < 17)).astype(int)
        df['ny_session'] = ((df['hour'] >= 13) & (df['hour'] < 22)).astype(int)
        df['overlap_session'] = ((df['hour'] >= 13) & (df['hour'] < 17)).astype(int)

        return df

    def calculate_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate market microstructure features."""
        # Bid-ask spread (using high-low as proxy)
        df['spread_proxy'] = (df['high'] - df['low']) / df['close']
        df['spread_sma'] = df['spread_proxy'].rolling(20).mean()

        # Price impact (volume-weighted price change)
        df['price_impact'] = df['close'].pct_change() / (df['tick_volume'] + 1)

        return df

    def add_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all features to dataframe."""
        logger.info(f"  Adding features (initial: {len(df.columns)} columns)...")

        # Ensure time column is datetime
        if 'time' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['time']):
            df['time'] = pd.to_datetime(df['time'])

        # Price features
        price_config = self.feature_config['price']
        df = self.calculate_returns(df, price_config['returns'])
        df = self.calculate_volatility(df, price_config['volatility'])
        df = self.calculate_vwap(df, price_config['vwap'])

        # Technical indicators
        indicators = self.feature_config['indicators']
        df = self.calculate_rsi(df, indicators['rsi'])
        df = self.calculate_macd(df, indicators['macd']['fast'],
                                indicators['macd']['slow'],
                                indicators['macd']['signal'])
        df = self.calculate_bollinger_bands(df, indicators['bollinger']['period'],
                                           indicators['bollinger']['std'])
        df = self.calculate_atr(df, indicators['atr'])
        df = self.calculate_stochastic(df, indicators['stochastic']['k'],
                                      indicators['stochastic']['d'],
                                      indicators['stochastic']['smooth'])
        df = self.calculate_adx(df, indicators['adx'])
        df = self.calculate_ema(df, indicators['ema'])
        df = self.calculate_sma(df, indicators['sma'])

        # Volume features
        df = self.calculate_volume_features(df)

        # Time features
        df = self.calculate_time_features(df)

        # Microstructure features
        df = self.calculate_microstructure_features(df)

        logger.info(f"  Features added (final: {len(df.columns)} columns)")

        return df

    def process_symbol(self, symbol: str) -> pd.DataFrame:
        """Process a single symbol and generate features."""
        # Determine filename
        timeframe = self.config['trading']['timeframe']
        filename = f"{symbol.replace('.', '')}_{timeframe}.parquet"
        input_file = self.raw_data_dir / filename

        if not input_file.exists():
            logger.error(f"  File not found: {input_file}")
            return pd.DataFrame()

        logger.info(f"\nProcessing {symbol}...")

        # Load raw data
        df = pd.read_parquet(input_file)
        logger.info(f"  Loaded {len(df):,} bars")

        # Add features
        df = self.add_all_features(df)

        # Replace inf with NaN
        df = df.replace([np.inf, -np.inf], np.nan)

        # Drop rows with NaN (from rolling calculations)
        # Only drop rows where OHLC data has NaN (preserve data integrity)
        initial_rows = len(df)

        # Check NaN counts
        nan_counts = df.isna().sum()
        if nan_counts.max() > len(df) * 0.5:  # If more than 50% NaN in any column
            logger.warning(f"  High NaN counts detected:")
            high_nan_cols = nan_counts[nan_counts > len(df) * 0.1].sort_values(ascending=False)
            for col, count in high_nan_cols.items():
                logger.warning(f"    {col}: {count:,} ({count/len(df)*100:.1f}%)")

        # Fill inf/NaN in derived features with 0 or forward fill
        feature_cols = [c for c in df.columns if c not in ['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']]
        df[feature_cols] = df[feature_cols].fillna(method='ffill').fillna(0)

        # Drop rows where OHLC is NaN (critical data)
        df = df.dropna(subset=['open', 'high', 'low', 'close'])

        logger.info(f"  Dropped {initial_rows - len(df):,} rows with critical NaN")
        logger.info(f"  Final dataset: {len(df):,} bars with {len(df.columns)} features")

        # Save processed data
        output_file = self.processed_data_dir / filename
        df.to_parquet(output_file, index=False)
        logger.info(f"  Saved to {output_file}")

        return df

    def process_all(self) -> Dict[str, pd.DataFrame]:
        """Process all symbols."""
        logger.info("\n" + "="*80)
        logger.info("Feature Engineering Pipeline")
        logger.info("="*80)

        symbols = self.config['trading']['pairs']
        all_data = {}

        for symbol in symbols:
            df = self.process_symbol(symbol)
            if not df.empty:
                all_data[symbol] = df

        logger.info("\n" + "="*80)
        logger.info("Feature Engineering Summary")
        logger.info("="*80)

        for symbol, df in all_data.items():
            logger.info(f"{symbol}: {len(df):,} bars, {len(df.columns)} features")

        logger.info(f"\nTotal processed data: {sum(len(df) for df in all_data.values()):,} bars")

        # Create train/val/test splits
        self.create_splits(all_data)

        return all_data

    def create_splits(self, all_data: Dict[str, pd.DataFrame]):
        """Create train/val/test splits."""
        logger.info("\n" + "="*80)
        logger.info("Creating Train/Val/Test Splits")
        logger.info("="*80)

        train_pct = self.config['data']['train_pct']
        val_pct = self.config['data']['val_pct']
        test_pct = self.config['data']['test_pct']

        for symbol, df in all_data.items():
            n = len(df)
            train_end = int(n * train_pct)
            val_end = int(n * (train_pct + val_pct))

            train_df = df.iloc[:train_end]
            val_df = df.iloc[train_end:val_end]
            test_df = df.iloc[val_end:]

            # Save splits
            symbol_clean = symbol.replace('.', '')
            timeframe = self.config['trading']['timeframe']

            train_df.to_parquet(self.processed_data_dir / f"{symbol_clean}_{timeframe}_train.parquet", index=False)
            val_df.to_parquet(self.processed_data_dir / f"{symbol_clean}_{timeframe}_val.parquet", index=False)
            test_df.to_parquet(self.processed_data_dir / f"{symbol_clean}_{timeframe}_test.parquet", index=False)

            logger.info(f"{symbol}:")
            logger.info(f"  Train: {len(train_df):,} bars ({train_pct*100:.0f}%)")
            logger.info(f"  Val:   {len(val_df):,} bars ({val_pct*100:.0f}%)")
            logger.info(f"  Test:  {len(test_df):,} bars ({test_pct*100:.0f}%)")


def main():
    """Main feature engineering script."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate features from raw FOREX data")
    parser.add_argument('--config', default='config/config.yaml', help='Path to config file')
    args = parser.parse_args()

    # Setup logger
    setup_logger(name="TradingEngine", level="INFO", log_file="logs/feature_engineering.log")

    # Process features
    engineer = FeatureEngineer(config_path=args.config)
    all_data = engineer.process_all()

    if all_data:
        logger.info("\nFeature engineering complete!")
        logger.info(f"Processed data saved to {engineer.processed_data_dir}")
        logger.info("Next step: Build RL environment")
    else:
        logger.error("\nFeature engineering failed!")


if __name__ == '__main__':
    main()
