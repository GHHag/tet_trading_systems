from dataclasses import dataclass
from typing import List, Tuple, Callable

from tet_trading_systems.trading_system_development.trading_systems.trading_system_properties.trading_system_properties import TradingSystemProperties

from tet_trading_systems.trading_system_state_handler.ml_trading_system_state_handler import MlTradingSystemStateHandler
from tet_trading_systems.trading_system_state_handler.portfolio.portfolio import Portfolio


@dataclass(frozen=True)
class MlTradingSystemProperties(TradingSystemProperties):
    system_instruments_list: List[str]

    """ def __init__(
        self,
        preprocess_data_function: Callable, preprocess_data_args: Tuple,
        system_instruments_list: List[str],
        system_state_handler: MlTradingSystemStateHandler, system_state_handler_args: Tuple,
        system_state_handler_call_args: Tuple,
        portfolio: Portfolio, portfolio_args: Tuple, portfolio_call_args: Tuple
    ):
        super().__init__(
            preprocess_data_function, preprocess_data_args,
            system_state_handler, system_state_handler_args,
            system_state_handler_call_args,
            portfolio, portfolio_args, portfolio_call_args
        )
        self.__system_instruments_list = system_instruments_list

    @property
    def system_instruments_list(self):
        return self.__system_instruments_list """
