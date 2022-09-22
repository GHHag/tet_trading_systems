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

    """ def __init__(
        self,
        preprocess_data_function: Callable, preprocess_data_args: Tuple, 
        system_state_handler: TradingSystemStateHandler, system_state_handler_args: Tuple, 
        system_state_handler_call_args: Tuple,         
        portfolio: Portfolio, portfolio_args: Tuple, portfolio_call_args: Tuple
    ):
        self.__preprocess_data_function = preprocess_data_function
        self.__preprocess_data_args = preprocess_data_args
        self.__system_state_handler = system_state_handler
        self.__system_state_handler_args = system_state_handler_args
        self.__system_state_handler_call_args = system_state_handler_call_args
        self.__portfolio = portfolio
        self.__portfolio_args = portfolio_args
        self.__portfolio_call_args = portfolio_call_args

    @property
    def preprocess_data_function(self):
        return self.__preprocess_data_function

    @property
    def preprocess_data_args(self):
        return self.__preprocess_data_args

    @property
    def system_state_handler(self):
        return self.__system_state_handler

    @property
    def system_state_handler_args(self):
        return self.__system_state_handler_args

    @property
    def system_state_handler_call_args(self):
        return self.__system_state_handler_call_args

    @property
    def portfolio(self):
        return self.__portfolio

    @property
    def portfolio_args(self):
        return self.__portfolio_args

    @property
    def portfolio_call_args(self):
        return self.__portfolio_call_args """
