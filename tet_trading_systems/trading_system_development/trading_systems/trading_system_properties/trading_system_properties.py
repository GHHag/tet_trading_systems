from dataclasses import dataclass
from typing import Tuple, Callable

from tet_trading_systems.trading_system_state_handler.trad_trading_system_state_handler import TradingSystemStateHandler
from tet_trading_systems.trading_system_state_handler.portfolio.portfolio import Portfolio


@dataclass(frozen=True)
class TradingSystemProperties:
    preprocess_data_function: Callable
    preprocess_data_args: Tuple
    system_state_handler: TradingSystemStateHandler
    system_state_handler_args: Tuple
    system_state_handler_call_args: Tuple
    portfolio: Portfolio
    portfolio_args: Tuple
    portfolio_call_args: Tuple
