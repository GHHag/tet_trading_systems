from decimal import Decimal
import json
from typing import Callable, Dict

import pandas as pd
import numpy as np

from TETrading.data.metadata.market_state_enum import MarketState
from TETrading.data.metadata.trading_system_attributes import TradingSystemAttributes
from TETrading.position.position import Position
from TETrading.position.position_manager import PositionManager
from TETrading.signal_events.signal_handler import SignalHandler

from tet_doc_db.tet_mongo_db.systems_mongo_db import TetSystemsMongoDb


class MlTradingSystemStateHandler:

    def __init__(
        self, system_name, symbol, db: TetSystemsMongoDb, df: pd.DataFrame,  
        date_format='%Y-%m-%d'
    ):
        self.__system_name = system_name
        self.__symbol = symbol
        self.__systems_db = db
        self.__df = df
        self.__date_format = date_format
        self.__model = self.__systems_db.get_ml_model(self.__system_name, self.__symbol)
        self.__position_list: list[Position] = self.__systems_db.get_single_symbol_position_list(
            self.__system_name, self.__symbol
        )

        self.__signal_handler = SignalHandler()
        if self.__model:
            self.__market_state_data = json.loads(
                self.__systems_db.get_market_state_data_for_symbol(
                    self.__system_name, self.__symbol
                )
            )
        else:
            raise Exception("Something went wrong while getting the model from database.")

        if self.__position_list[-1].exit_signal_dt:    
            mask = (self.__df['Date'] > str(self.__position_list[0].entry_dt)) & \
                (self.__df['Date'] <= str(self.__position_list[-1].exit_signal_dt))
            self.__num_testing_periods = len(self.__df.loc[mask])
        else:
            mask = (self.__df['Date'] > str(self.__position_list[0].entry_dt)) & \
                (self.__df['Date'] <= str(self.__position_list[-2].exit_signal_dt))
            periods_in_position = self.__market_state_data[TradingSystemAttributes.PERIODS_IN_POSITION] \
                if TradingSystemAttributes.PERIODS_IN_POSITION in self.__market_state_data else 1
            self.__num_testing_periods = len(self.__df.loc[mask]) + periods_in_position 

    def _generate_position_sequence(self, **kwargs):
        for pos in self.__position_list:
            yield pos

    def _handle_entry_signal(self):
        if self.__market_state_data[TradingSystemAttributes.MARKET_STATE] == MarketState.ENTRY.value and \
            self.__df['Date'].iloc[-2] == pd.Timestamp(self.__market_state_data[TradingSystemAttributes.SIGNAL_DT]) and \
                not self.__position_list[-1].active_position:
            self.__position_list[-1].enter_market(
                self.__df['Open'].iloc[-1], 'long', self.__df['Date'].iloc[-1]
            )
            print(
                f'\nEntry index {len(self.__df)}: {format(self.__df["Open"].iloc[-1], ".3f")}, ' + 
                f'{self.__df["Date"].iloc[-1]}'
            )

    def _handle_enter_market_state(
        self, entry_logic_function, entry_args,
        capital=10000, num_of_sims=2500, plot_fig=True, 
        **kwargs
    ):
        entry_signal, direction = entry_logic_function(
            self.__df.iloc[-entry_args['entry_period_lookback']:], entry_args
        )
        if entry_signal and not self.__position_list[-1].active_position and \
            not self.__position_list[-1].exit_signal_dt == self.__df['Date'].iloc[-2]:
            # create mask to filter self.__df where the 'Date' is between the first element of 
            # self.__position_lists entry_dt and the last elements exit_signal_dt
            mask = (self.__df['Date'] > str(self.__position_list[0].entry_dt)) & \
                (self.__df['Date'] <= str(self.__position_list[-1].exit_signal_dt))
            
            position_manager = PositionManager(
                self.__symbol, self.__num_testing_periods, capital, 1.0,
                asset_price_series=[float(close) for close in self.__df.loc[mask]['Close']]
            )
            position_manager.generate_positions(self._generate_position_sequence)
            position_manager.summarize_performance(plot_fig=plot_fig)

            self.__signal_handler.handle_entry_signal(
                self.__symbol, 
                {
                    TradingSystemAttributes.SIGNAL_INDEX: len(self.__df), 
                    TradingSystemAttributes.SIGNAL_DT: self.__df['Date'].iloc[-1], 
                    TradingSystemAttributes.SYMBOL: self.__symbol,
                    TradingSystemAttributes.DIRECTION: direction,
                    TradingSystemAttributes.MARKET_STATE: MarketState.ENTRY.value
                }
            )
            
            self.__position_list.pop(0)
            self.__position_list.append(Position(capital))
            print(f'\nEntry signal, buy next open\nIndex {len(self.__df)}')

    def _handle_active_pos_state(self):
        if self.__df['Date'].iloc[-1] != pd.Timestamp(self.__market_state_data[TradingSystemAttributes.SIGNAL_DT]):
            self.__position_list[-1].update(Decimal(self.__df['Close'].iloc[-1]))
        self.__position_list[-1].print_position_stats()

        if self.__market_state_data[TradingSystemAttributes.MARKET_STATE] == MarketState.EXIT.value and \
            self.__df['Date'].iloc[-2] == pd.Timestamp(self.__market_state_data[TradingSystemAttributes.SIGNAL_DT]):
            self.__position_list[-1].exit_market(
                self.__df['Open'].iloc[-1], pd.Timestamp(self.__market_state_data[TradingSystemAttributes.SIGNAL_DT])
            ) 
            print(
                f'Exit index {len(self.__df)}: {format(self.__df["Open"].iloc[-1], ".3f")}, ' + 
                f'{self.__df["Date"].iloc[-1]}\n'
                f'Realised return: {self.__position_list[-1].position_return}'
            )
            return False
        else:
            self.__signal_handler.handle_active_position(
                self.__symbol, 
                {
                    TradingSystemAttributes.SIGNAL_INDEX: len(self.__df), 
                    TradingSystemAttributes.SIGNAL_DT: self.__df['Date'].iloc[-1], 
                    TradingSystemAttributes.SYMBOL: self.__symbol, 
                    TradingSystemAttributes.PERIODS_IN_POSITION: len(self.__position_list[-1].returns_list),
                    TradingSystemAttributes.UNREALISED_RETURN: self.__position_list[-1].unrealised_return,
                    TradingSystemAttributes.MARKET_STATE: MarketState.ACTIVE.value 
                }
            )
            return True

    def _handle_exit_market_state(self, exit_logic_function, exit_args):
        exit_condition, trailing_exit, trailing_exit_price = exit_logic_function(
            self.__df.iloc[-exit_args['exit_period_lookback']:], False, None,
            self.__position_list[-1].entry_price, exit_args, len(self.__position_list[-1].returns_list)
        )
        if exit_condition:
            self.__signal_handler.handle_exit_signal(
                self.__symbol,
                {
                    TradingSystemAttributes.SIGNAL_INDEX: len(self.__df), 
                    TradingSystemAttributes.SIGNAL_DT: self.__df['Date'].iloc[-1],
                    TradingSystemAttributes.SYMBOL: self.__symbol, 
                    TradingSystemAttributes.PERIODS_IN_POSITION: len(self.__position_list[-1].returns_list),
                    TradingSystemAttributes.UNREALISED_RETURN: self.__position_list[-1].unrealised_return,
                    TradingSystemAttributes.MARKET_STATE: MarketState.EXIT.value 
                }
            )
            print(f'\nExit signal, exit next open\nIndex {len(self.__df)}')
            return True
        else:
            return False

    def __call__(
        self, entry_logic_function: Callable, exit_logic_function: Callable, 
        entry_args: Dict[str, object], exit_args: Dict[str, object], 
        pred_features: np.ndarray, 
        date_format='%Y-%m-%d', capital=10000, plot_fig=False, insert_into_db=False, 
        **kwargs
    ):
        if not 'entry_period_lookback' in entry_args.keys() or \
            not 'exit_period_lookback' in exit_args.keys():
            raise Exception("Given parameter for 'entry_args' or 'exit_args' is missing required key(s).")

        if self.__df['Date'].iloc[-1] != pd.Timestamp(self.__market_state_data[TradingSystemAttributes.SIGNAL_DT]):
            self._handle_entry_signal()
            latest_data_point = self.__df.iloc[-1].copy()
            latest_data_point['pred'] = self.__model.predict(pred_features[-1].reshape(1, -1))[0]
            self.__df = self.__df.iloc[:-1].append(latest_data_point, ignore_index=True)
           
            if isinstance(self.__position_list[-1], Position) and self.__position_list[-1].active_position:
                active_pos = self._handle_active_pos_state()
                if active_pos:
                    self._handle_exit_market_state(exit_logic_function, exit_args)
                else:
                    if insert_into_db:
                        # Position objects in json format will be inserted to database after being exited
                        self.__systems_db.insert_single_symbol_position_list(
                            self.__system_name, self.__symbol, self.__position_list, self.__num_testing_periods,
                            format='json'
                        )
            else:
                self._handle_enter_market_state(
                    entry_logic_function, entry_args,
                    capital=capital, plot_fig=plot_fig
                )

            print(self.__signal_handler)
            if insert_into_db:
                self.__signal_handler.insert_into_db(
                    {
                        MarketState.ENTRY.value: self.__systems_db.insert_market_state_data,
                        MarketState.ACTIVE.value: self.__systems_db.insert_market_state_data, 
                        MarketState.EXIT.value: self.__systems_db.insert_market_state_data
                    }, self.__system_name
                )

                result = self.__systems_db.insert_single_symbol_position_list(
                    self.__system_name, self.__symbol, self.__position_list, self.__num_testing_periods
                )

                if not result:
                    print(
                        'List of Position objects were not modified.\n' + 
                        'Insert position list result: ' + str(result)
                    )
