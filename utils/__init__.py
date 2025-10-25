"""Utility modules for TradingEngine"""

from .logger import logger, setup_logger

# MT5Bridge is Windows-only, import optionally
try:
    from .mt5_bridge import MT5Bridge
    __all__ = ['logger', 'setup_logger', 'MT5Bridge']
except ImportError:
    # MetaTrader5 not available (Linux/cloud training)
    __all__ = ['logger', 'setup_logger']
