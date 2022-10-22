import random
from typing import List

import pandas as pd

from TETrading.position.position import Position
from TETrading.position.position_manager import PositionManager
from TETrading.utils.metric_functions import calculate_cagr
from TETrading.utils.monte_carlo_functions import monte_carlo_simulations_plot

from tet_trading_systems.trading_system_management.position_sizer.position_sizer \
    import IPositionSizer


class SafeFPositionSizer(IPositionSizer):

    __POSITION_SIZE_METRIC_STR = 'safe-f'

    def __init__(self, tolerated_pct_max_drawdown, max_drawdown_percentile_threshold):
        self.__tol_pct_max_dd = tolerated_pct_max_drawdown
        self.__max_dd_pctl_threshold = max_drawdown_percentile_threshold

    @property
    def position_size_metric_str(self):
        return self.__POSITION_SIZE_METRIC_STR

    #@property
    #def objective_function_str(self):
    #    return self.__objective_function_str

    def _monte_carlo_simulate_pos_sequence(
        self, positions, num_testing_periods, start_capital, capital_fraction=1.0,
        num_of_sims=1000, data_amount_used=0.5, symbol='', print_dataframe=False, 
        plot_monte_carlo_sims=False, **kwargs
    ):
        monte_carlo_sims_df = pd.DataFrame()

        equity_curves_list = []
        final_equity_list = []
        max_drawdowns_list = []
        sim_positions = None

        def generate_position_sequence(position_list, **kw):
            for pos in position_list[:int(len(position_list) * data_amount_used)]:
                yield pos

        for _ in range(num_of_sims):
            sim_positions = PositionManager(
                symbol, (num_testing_periods * data_amount_used), start_capital,
                capital_fraction
            )

            pos_list = random.sample(positions, len(positions))
            sim_positions.generate_positions(generate_position_sequence, pos_list)
            monte_carlo_sims_df = monte_carlo_sims_df.append(
                sim_positions.metrics.summary_data_dict, ignore_index=True
            )
            final_equity_list.append(float(sim_positions.metrics.equity_list[-1]))

            max_drawdowns_list.append(sim_positions.metrics.max_drawdown)

            equity_curves_list.append(sim_positions.metrics.equity_list)

        final_equity_list = sorted(final_equity_list)

        car25 = calculate_cagr(
            sim_positions.metrics.start_capital,
            final_equity_list[(int(len(final_equity_list) * 0.25))],
            sim_positions.metrics.num_testing_periods
        )
        car75 = calculate_cagr(
            sim_positions.metrics.start_capital,
            final_equity_list[(int(len(final_equity_list) * 0.75))],
            sim_positions.metrics.num_testing_periods
        )

        car_series = pd.Series()
        car_series['CAR25'] = car25
        car_series['CAR75'] = car75
        monte_carlo_sims_df: pd.DataFrame = monte_carlo_sims_df.append(car_series, ignore_index=True)

        if print_dataframe:
            print(monte_carlo_sims_df.to_string())
        if plot_monte_carlo_sims:
            monte_carlo_simulations_plot(
                symbol, equity_curves_list, max_drawdowns_list, final_equity_list,
                capital_fraction, car25, car75
            )

        return monte_carlo_sims_df

    def __call__(
        self, positions: List[Position], num_testing_periods, 
        #forecast_positions=100, forecast_data_fraction=0.5, persistant_safe_f=None,
        forecast_data_fraction=0.5, persistant_safe_f=None,
        capital=10000, num_of_sims=2500, symbol='', **kwargs
    ):
        #split_data_fraction = 1.0
        #if len(positions) >= forecast_positions:
        #    split_data_fraction = forecast_positions / len(positions)

        #period_len = int(period_len * split_data_fraction)
        num_testing_periods = int(num_testing_periods* forecast_data_fraction)
        # sort positions on date
        positions.sort(key=lambda tr: tr.entry_dt)

        # simulate sequences of given Position objects
        monte_carlo_sims_df: pd.DataFrame = self._monte_carlo_simulate_pos_sequence(
            #positions[-(int(len(positions) * split_data_fraction)):], 
            positions[-(int(len(positions) * forecast_data_fraction)):], 
            num_testing_periods, capital, 
            capital_fraction=persistant_safe_f if persistant_safe_f else 1.0, 
            num_of_sims=num_of_sims, data_amount_used=forecast_data_fraction, 
            symbol=symbol, **kwargs
        )

        # sort the 'Max drawdown (%)' column and convert to a list
        max_dds = sorted(monte_carlo_sims_df['Max drawdown (%)'].to_list())
        # get the drawdown value at the percentile set to be the threshold at which to limit the 
        # probability of getting a max drawdown of that magnitude at when simulating sequences 
        # of the best estimate positions
        dd_at_tolerated_threshold = max_dds[int(len(max_dds) * self.__max_dd_pctl_threshold)]
        if dd_at_tolerated_threshold < 1:
            dd_at_tolerated_threshold = 1
        # calculate the safe fraction of capital to be used to purchase assets with
        if not persistant_safe_f:
            safe_f = self.__tol_pct_max_dd / dd_at_tolerated_threshold
        else:
            safe_f = persistant_safe_f

        return {
            'symbol': symbol,
            #'sharpe_ratio': kwargs['metrics_dict']['Sharpe ratio'],
            #'profit_factor': kwargs['metrics_dict']['Profit factor'],
            #'expectancy': kwargs['metrics_dict']['Expectancy'],
            'CAR25': round(monte_carlo_sims_df.iloc[-1]['CAR25'], 3),
            'CAR75': round(monte_carlo_sims_df.iloc[-1]['CAR75'], 3),
            self.__POSITION_SIZE_METRIC_STR: round(safe_f, 3),
        }
