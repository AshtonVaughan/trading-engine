"""
MetaTrader 5 Python API Bridge

Provides a clean interface to MT5 for data collection and order execution.
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import time

from utils.logger import logger


class MT5Bridge:
    """
    Bridge to MetaTrader 5 terminal for data collection and trading.
    """

    def __init__(
        self,
        login: int = None,
        password: str = None,
        server: str = None,
        path: str = None,
        timeout: int = 60000
    ):
        """
        Initialize MT5 connection.

        Args:
            login: MT5 account number
            password: MT5 account password
            server: Broker server name
            path: Path to MT5 terminal executable
            timeout: Connection timeout in milliseconds
        """
        self.login = login
        self.password = password
        self.server = server
        self.path = path
        self.timeout = timeout
        self.connected = False

    def connect(self) -> bool:
        """
        Connect to MT5 terminal.

        Returns:
            True if successful, False otherwise
        """
        if self.connected:
            logger.info("Already connected to MT5")
            return True

        # Initialize MT5
        if self.path:
            if not mt5.initialize(self.path, timeout=self.timeout):
                logger.error(f"MT5 initialize failed: {mt5.last_error()}")
                return False
        else:
            if not mt5.initialize(timeout=self.timeout):
                logger.error(f"MT5 initialize failed: {mt5.last_error()}")
                return False

        logger.info("MT5 initialized successfully")

        # Login if credentials provided
        if self.login and self.password and self.server:
            if not mt5.login(self.login, password=self.password, server=self.server):
                logger.error(f"MT5 login failed: {mt5.last_error()}")
                mt5.shutdown()
                return False

            logger.info(f"Logged in to MT5 account {self.login} on {self.server}")

        self.connected = True
        return True

    def disconnect(self):
        """Disconnect from MT5."""
        if self.connected:
            mt5.shutdown()
            self.connected = False
            logger.info("Disconnected from MT5")

    def get_balance(self) -> float:
        """Get account balance."""
        if not self.connected:
            return 0.0
        account_info = mt5.account_info()
        return account_info.balance if account_info else 0.0

    def get_equity(self) -> float:
        """Get account equity."""
        if not self.connected:
            return 0.0
        account_info = mt5.account_info()
        return account_info.equity if account_info else 0.0

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current bid price for a symbol."""
        if not self.connected:
            return None
        tick = mt5.symbol_info_tick(symbol)
        return tick.bid if tick else None

    def get_recent_bars(self, symbol: str, timeframe: str, count: int) -> Optional[pd.DataFrame]:
        """Get recent OHLCV bars."""
        if not self.connected:
            return None

        # Map timeframe
        tf_map = {
            'M1': mt5.TIMEFRAME_M1,
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'M30': mt5.TIMEFRAME_M30,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1,
        }
        tf = tf_map.get(timeframe, mt5.TIMEFRAME_H1)

        # Get rates
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, count)

        if rates is None or len(rates) == 0:
            return None

        # Convert to DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')

        return df

    def execute_market_order(
        self,
        symbol: str,
        order_type: str,
        lot_size: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        magic: int = 0,
    ) -> Optional[dict]:
        """Execute a market order."""
        if not self.connected:
            logger.error("Not connected to MT5")
            return None

        # Get symbol info
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logger.error(f"Symbol {symbol} not found")
            return None

        # Prepare request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot_size,
            "type": mt5.ORDER_TYPE_BUY if order_type == 'buy' else mt5.ORDER_TYPE_SELL,
            "price": mt5.symbol_info_tick(symbol).ask if order_type == 'buy' else mt5.symbol_info_tick(symbol).bid,
            "deviation": 10,
            "magic": magic,
            "comment": "TradingEngine RL",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        # Add SL/TP if provided
        if stop_loss:
            request["sl"] = stop_loss
        if take_profit:
            request["tp"] = take_profit

        # Send order
        result = mt5.order_send(request)

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Order failed: {result.retcode} - {result.comment}")
            return None

        return {
            'ticket': result.order,
            'price': result.price,
            'volume': result.volume,
        }

    def close_position(self, position: dict) -> bool:
        """Close an open position."""
        if not self.connected:
            return False

        # Get position info
        ticket = position['ticket']
        pos = mt5.positions_get(ticket=ticket)

        if pos is None or len(pos) == 0:
            logger.error(f"Position {ticket} not found")
            return False

        pos = pos[0]

        # Prepare close request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": pos.symbol,
            "volume": pos.volume,
            "type": mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
            "position": ticket,
            "price": mt5.symbol_info_tick(pos.symbol).bid if pos.type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(pos.symbol).ask,
            "deviation": 10,
            "magic": pos.magic,
            "comment": "Close by TradingEngine",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        # Send close order
        result = mt5.order_send(request)

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Close failed: {result.retcode} - {result.comment}")
            return False

        return True

    def get_historical_data(
        self,
        symbol: str,
        timeframe: str = "M1",
        start_date: datetime = None,
        end_date: datetime = None,
        num_bars: int = None
    ) -> pd.DataFrame:
        """
        Get historical OHLCV data from MT5.

        Args:
            symbol: Trading symbol (e.g., "EURUSD")
            timeframe: Timeframe (M1, M5, M15, H1, H4, D1, etc.)
            start_date: Start date for data collection
            end_date: End date for data collection
            num_bars: Number of bars to collect (alternative to date range)

        Returns:
            DataFrame with columns: time, open, high, low, close, tick_volume, spread, real_volume
        """
        if not self.connected:
            logger.error("Not connected to MT5")
            return pd.DataFrame()

        # Convert timeframe string to MT5 constant
        timeframe_map = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
            "W1": mt5.TIMEFRAME_W1,
            "MN1": mt5.TIMEFRAME_MN1
        }

        tf = timeframe_map.get(timeframe.upper())
        if tf is None:
            logger.error(f"Invalid timeframe: {timeframe}")
            return pd.DataFrame()

        # Get data
        if num_bars:
            # Get last N bars
            rates = mt5.copy_rates_from_pos(symbol, tf, 0, num_bars)
        elif start_date and end_date:
            # Get data in date range
            rates = mt5.copy_rates_range(symbol, tf, start_date, end_date)
        else:
            logger.error("Must specify either num_bars or start_date+end_date")
            return pd.DataFrame()

        if rates is None or len(rates) == 0:
            logger.error(f"No data received for {symbol}: {mt5.last_error()}")
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')

        logger.info(f"Collected {len(df):,} {timeframe} bars for {symbol}")

        return df

    def get_historical_data_batched(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
        batch_size: int = 100000
    ) -> pd.DataFrame:
        """
        Get historical data in batches to avoid MT5 API limits.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            start_date: Start date
            end_date: End date
            batch_size: Number of bars per batch

        Returns:
            DataFrame with all data
        """
        all_data = []
        current_start = start_date

        while current_start < end_date:
            # Calculate batch end date based on timeframe
            if timeframe == "M1":
                batch_end = min(current_start + timedelta(days=70), end_date)  # ~100K minutes
            elif timeframe == "M5":
                batch_end = min(current_start + timedelta(days=347), end_date)
            elif timeframe == "H1":
                batch_end = min(current_start + timedelta(days=4167), end_date)
            else:
                batch_end = end_date

            logger.info(f"Fetching {symbol} {timeframe} from {current_start} to {batch_end}")

            df_batch = self.get_historical_data(
                symbol=symbol,
                timeframe=timeframe,
                start_date=current_start,
                end_date=batch_end
            )

            if not df_batch.empty:
                all_data.append(df_batch)
                current_start = df_batch['time'].iloc[-1] + timedelta(seconds=1)
            else:
                logger.warning(f"Empty batch for {current_start} to {batch_end}")
                break

            time.sleep(0.1)  # Rate limiting

        if not all_data:
            return pd.DataFrame()

        # Concatenate all batches
        df_all = pd.concat(all_data, ignore_index=True)
        df_all.drop_duplicates(subset=['time'], inplace=True)
        df_all.sort_values('time', inplace=True)
        df_all.reset_index(drop=True, inplace=True)

        logger.info(f"Total {len(df_all):,} bars collected for {symbol}")

        return df_all

    def get_symbol_info(self, symbol: str) -> Dict:
        """
        Get symbol information (spread, digits, contract size, etc.).

        Args:
            symbol: Trading symbol

        Returns:
            Dictionary with symbol info
        """
        if not self.connected:
            logger.error("Not connected to MT5")
            return {}

        info = mt5.symbol_info(symbol)
        if info is None:
            logger.error(f"Symbol info not found for {symbol}: {mt5.last_error()}")
            return {}

        return {
            'symbol': symbol,
            'point': info.point,
            'digits': info.digits,
            'spread': info.spread,
            'volume_min': info.volume_min,
            'volume_max': info.volume_max,
            'volume_step': info.volume_step,
            'contract_size': info.trade_contract_size,
            'currency_base': info.currency_base,
            'currency_profit': info.currency_profit,
            'currency_margin': info.currency_margin,
        }

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()


if __name__ == '__main__':
    # Test MT5 connection
    mt5_bridge = MT5Bridge()

    if mt5_bridge.connect():
        logger.info("MT5 connection successful!")

        # Get some test data
        df = mt5_bridge.get_historical_data(
            symbol="EURUSD",
            timeframe="M1",
            num_bars=1000
        )

        if not df.empty:
            logger.info(f"Sample data:\n{df.head()}")
            logger.info(f"Data shape: {df.shape}")

        # Get symbol info
        info = mt5_bridge.get_symbol_info("EURUSD")
        logger.info(f"Symbol info: {info}")

        mt5_bridge.disconnect()
    else:
        logger.error("Failed to connect to MT5")
