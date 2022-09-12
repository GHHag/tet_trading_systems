import datetime as dt
from decimal import Decimal
import json

import pandas as pd
import numpy as np

from TETrading.position.position import Position
from TETrading.position.position_manager import PositionManager
from TETrading.signal_events.signal_handler import SignalHandler
from TETrading.utils.monte_carlo_functions import calculate_safe_f, monte_carlo_simulate_returns

from tet_doc_db.doc_database_meta_classes.tet_systems_doc_db import ITetSystemsDocumentDatabase


class MlTradingSystemStateHandler:
    def __init__(
        self, system_name, system_name_suffix_bool, symbol, 
        db: ITetSystemsDocumentDatabase, df: pd.DataFrame,  
        date_format='%Y-%m-%d'
    ):
        self.__system_name = f'{system_name}_{symbol}' if system_name_suffix_bool else system_name
        self.__symbol = symbol
        self.__systems_db = db
        self.__df = df
        self.__model = self.__systems_db.get_ml_model(self.__system_name, self.__symbol)
        self.__position_list: list[Position] = self.__systems_db.get_position_list(self.__system_name)
        self.__system_metrics = json.loads(self.__systems_db.get_system_metrics(self.__system_name))
        self.__signal_handler = SignalHandler()
        if self.__model:
            self.__system_signals_list = json.loads(
               self.__systems_db.get_system_signals_for_symbol(self.__system_name, self.__symbol)
            )
            self.__entry_signal_dt, self.__active_pos_dt, self.__exit_signal_dt = tuple(
                dt.datetime.strptime(signal_data[0]['signal_dt'][:10], date_format)
                if len(signal_data) > 0 else None for signal_data in self.__system_signals_list
            )
        else:
            raise Exception(
                "Something went wrong while getting the model from database."
            )

    def _generate_position_sequence(self, **kwargs):
        for pos in self.__position_list:
            yield pos

    # gör funktionen generisk för att kunna anvanda olika typer av pos sizers och metrics
    def _insert_system_metrics(self, system_metrics, num_testing_periods):
        result = self.__systems_db.insert_system_metrics(
            self.__system_name,
            {
                'sharpe_ratio': system_metrics[-1]['Sharpe ratio'],
                'expectancy': system_metrics[-1]['Expectancy'],
                'profit_factor': float(system_metrics[-1]['Profit factor']),
                'CAR25': system_metrics[-1]['CAR25'],
                'CAR75': system_metrics[-1]['CAR75'],
                'safe_F': system_metrics[-1]['safe_F']
            },
            {
                'num_of_periods': num_testing_periods
            }
        )
        return result 

    # funktionen ska vara generisk och fungera med alla PositionSizer child-klasser? ersatter _run_monte_carlo_simulations isf?
    #def _calculate_system_metrics(self):
    #    pass

    def _run_monte_carlo_simulations( # denna function borde ligga i separat modul och endast köras för sarskilda pos_sizer objekt
        self, num_testing_periods, tolerated_pct_max_dd, dd_percentile_threshold,
        avg_yearly_positions, years_to_forecast,
        capital=10000, num_of_sims=2500, print_dataframe=False, plot_fig=False,
    ):
        for ts_run in range(2):
            if ts_run < 1:
                capital_f = round(calculate_safe_f(
                    self.__position_list, num_testing_periods, tolerated_pct_max_dd, dd_percentile_threshold,
                    forecast_positions=avg_yearly_positions * (years_to_forecast + 1),
                    forecast_data_fraction=(avg_yearly_positions / len(self.__position_list)) * years_to_forecast,
                    capital=capital, num_of_sims=num_of_sims, print_dataframe=print_dataframe
                ), 2)
            else:
                mc_data = monte_carlo_simulate_returns(
                    self.__position_list, self.__symbol, num_testing_periods,
                    start_capital=capital, capital_fraction=capital_f, num_of_sims=num_of_sims,
                    data_amount_used=(avg_yearly_positions / len(self.__position_list)) * years_to_forecast,
                    print_dataframe=print_dataframe, plot_fig=plot_fig
                )
                mc_data[-1]['safe_F'] = capital_f
                return mc_data

    def _handle_entry_signal(self):
        if self.__entry_signal_dt and \
            self.__df['Date'].iloc[-2].date() == self.__entry_signal_dt.date() and \
                not self.__position_list[-1].active_position:
            self.__position_list[-1].enter_market(
                self.__df['Open'].iloc[-1], 'long', self.__df['Date'].iloc[-1]
            )
            print(
                f'\nEntry index {len(self.__df)}: {format(self.__df["Open"].iloc[-1], ".3f")}, ' + 
                f'{self.__df["Date"].iloc[-1]}'
            )

    def _handle_enter_market_state(
        self, entry_logic_function, entry_args, position_sizer, 
        tolerated_pct_max_dd, dd_percentile_threshold,
        capital=10000, num_of_sims=2500, yearly_periods=251, years_to_forecast=2, 
        insert_into_db=False, plot_fig=True, **kwargs
    ):
        if entry_logic_function(self.__df.iloc[-entry_args['entry_period_lookback']:], entry_args) and \
            self.__position_list[-1].exit_signal_dt and \
                not self.__position_list[-1].exit_signal_dt.date() == self.__df['Date'].iloc[-2].date():
            self.__signal_handler.handle_entry_signal(
                self.__symbol, 
                {
                    'signal_index': len(self.__df), 'signal_dt': self.__df['Date'].iloc[-1], 
                    'symbol': self.__symbol
                }
            )
            mask = (self.__df['Date'] > str(self.__position_list[0].entry_dt)) & \
                (self.__df['Date'] <= str(self.__position_list[-1].exit_signal_dt))
            num_testing_periods = len(self.__df.loc[mask])
            avg_yearly_positions = int(len(self.__position_list) / (num_testing_periods / yearly_periods) + 0.5)
            position_manager = PositionManager(
                self.__symbol, num_testing_periods, capital, self.__system_metrics['metrics']['safe_F'],
                asset_price_series=[float(close) for close in self.__df.loc[mask]['Close']]
            )
            position_manager.generate_positions(self._generate_position_sequence)
            position_manager.summarize_performance(plot_fig=plot_fig)
            self.__signal_handler.add_pos_sizing_evaluation_data(
                position_sizer(
                    self.__position_list, num_testing_periods,
                    forecast_data_fraction=(avg_yearly_positions / len(self.__position_list)) * years_to_forecast,
                    capital=capital, num_of_sims=num_of_sims, symbol=self.__symbol,
                    metrics_dict=position_manager.metrics.summary_data_dict
                )
            )

            mc_data = self._run_monte_carlo_simulations(
                num_testing_periods, tolerated_pct_max_dd, dd_percentile_threshold,
                avg_yearly_positions, years_to_forecast, 
                capital=capital, num_of_sims=num_of_sims, **kwargs, plot_fig=plot_fig
            )
            if insert_into_db:
                self._insert_system_metrics(mc_data, num_testing_periods)
            
            self.__position_list.pop(0)
            self.__position_list.append(Position(capital))
            print(f'\nEntry signal, buy next open\nIndex {len(self.__df)}')

    def _handle_active_pos_state(self):
        if not self.__active_pos_dt or \
            pd.Timestamp(self.__df['Date'].iloc[-1].date()) != pd.Timestamp(self.__active_pos_dt.date()):
           self.__position_list[-1].update(Decimal(self.__df['Close'].iloc[-1]))
        self.__position_list[-1].print_position_stats()

        if self.__exit_signal_dt and self.__df['Date'].iloc[-2].date() == self.__exit_signal_dt.date():
            self.__position_list[-1].exit_market(
                self.__df['Open'].iloc[-1], self.__exit_signal_dt
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
                    'signal_index': len(self.__df), 'signal_dt': self.__df['Date'].iloc[-1], 
                    'symbol': self.__symbol, 'periods_in_position': len(self.__position_list[-1].returns_list),
                    'unrealised_return': self.__position_list[-1].unrealised_return
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
                    'signal_index': len(self.__df), 'signal_dt': self.__df['Date'].iloc[-1],
                    'symbol': self.__symbol, 'periods_in_position': len(self.__position_list[-1].returns_list),
                    'unrealised_return': self.__position_list[-1].unrealised_return
                }
            )
            print(f'\nExit signal, exit next open\nIndex {len(self.__df)}')

    def __call__(
        self, entry_logic_function, exit_logic_function, position_sizer,
        entry_args, exit_args, pred_features: np.ndarray,
        date_format='%Y-%m-%d', capital=10000,
        tolerated_pct_max_dd=10, dd_percentile_threshold=0.85, 
        years_to_forecast=2, avg_yearly_periods=251,
        plot_fig=False, insert_into_db=False, **kwargs
    ):
        if not 'entry_period_lookback' in entry_args.keys() or \
            not 'exit_period_lookback' in exit_args.keys():
            raise Exception("Given parameter for 'entry_args' or 'exit_args' is missing required key.")

        if self.__model:
            self._handle_entry_signal()
            latest_data_point = self.__df.iloc[-1].copy()
            latest_data_point['pred'] = self.__model.predict(pred_features[-1].reshape(1, -1))[0]
            self.__df = self.__df.iloc[:-1].append(latest_data_point, ignore_index=True)
        else:
            raise Exception(
                'Something went wrong while predictions with the model.'
            )

        if isinstance(self.__position_list[-1], Position) and self.__position_list[-1].active_position:
            active_pos = self._handle_active_pos_state()
            if active_pos:
                self._handle_exit_market_state(exit_logic_function, exit_args)
        else:
            self._handle_enter_market_state(
                entry_logic_function, entry_args, position_sizer,
                tolerated_pct_max_dd, dd_percentile_threshold,
                capital=capital, yearly_periods=avg_yearly_periods, years_to_forecast=years_to_forecast, 
                insert_into_db=insert_into_db, plot_fig=plot_fig
            )

        print(self.__signal_handler)            
        if insert_into_db:
            self.__signal_handler.insert_into_db(
                {
                    'entry': self.__systems_db.insert_entry_signal_data, 
                    'active': self.__systems_db.insert_active_position_data, 
                    'exit': self.__systems_db.insert_exit_signal_data
                }, self.__system_name
            )
        result = self.__systems_db.insert_position_list(self.__system_name, self.__position_list)
        if not result:
            print('List of Position objects were not modified.\n' + 
                'Insert position list result: ' + str(result))
