from typing import Dict

from tet_trading_systems.trading_system_management.position_sizer.position_sizer \
    import IPositionSizer

from TETrading.utils.monte_carlo_functions import calculate_safe_f


class ExtPositionSizer(IPositionSizer):

    __POSITION_SIZE_METRIC_STR = 'safe-f'
    __CAPITAL_FRACTION = 'capital_fraction'
    __PERSISTANT_SAFE_F = 'persistant_safe_f'

    def __init__(
        self, num_of_instruments, tolerated_pct_max_drawdown,
        max_drawdown_percentile_threshold,
        *args, **kwargs
    ):
        self.__num_of_instruments = num_of_instruments
        self.__tol_pct_max_dd = tolerated_pct_max_drawdown
        self.__max_dd_pctl_threshold = max_drawdown_percentile_threshold
        self.__position_sizer_data_dict = {}
        self.__args = args
        self.__kwargs = kwargs

    @property
    def position_size_metric_str(self):
        return self.__POSITION_SIZE_METRIC_STR

    @property
    def position_sizer_data_dict(self) -> Dict:
        return self.__position_sizer_data_dict

    def get_position_sizer_data_dict(self) -> Dict:
        return self.__position_sizer_data_dict

    def _calculate_safe_f(
        self, position_list, num_of_periods,
        avg_yearly_periods=251, years_to_forecast=2,
        capital=10000, num_of_sims=2500, print_dataframe=False
    ):
        #full_pos_list_slice_param=int(avg_yearly_positions * (years_to_forecast + years_to_forecast / 2))
        
        #sorted_position_lists = sorted(position_list, key=len, reverse=True)
        #position_list_lengths = [len(i) for i in sorted_position_lists[:int(self.__num_of_instruments / 4 + 0.5)]] \
        #position_list_lengths = [len(i) for i in position_list[:int(self.__num_of_instruments / 4 + 0.5)]] \
            #if self.__num_of_instruments > 1 \
            #else [len(sorted_position_lists[0])]
        #avg_yearly_positions = int(
        #    (sum(position_list_lengths) / len(position_list_lengths)) / years_to_forecast + 0.5
        #) 
        avg_yearly_positions = int(num_of_periods / len(position_list)) 

        #capital_f = round(calculate_safe_f(
        return round(calculate_safe_f(
            position_list, num_of_periods, self.__tol_pct_max_dd, self.__max_dd_pctl_threshold,
            forecast_positions=len(position_list), forecast_data_fraction=0.66,
            capital=capital, num_of_sims=num_of_sims, print_dataframe=print_dataframe
        ), 2)
        return round(calculate_safe_f(
            position_list, num_of_periods, self.__tol_pct_max_dd, self.__max_dd_pctl_threshold,
            forecast_positions=avg_yearly_positions * (years_to_forecast + years_to_forecast / 2), 
            forecast_data_fraction=(avg_yearly_positions * years_to_forecast) / 
                                (avg_yearly_positions * (years_to_forecast + years_to_forecast / 2)),
            capital=capital, num_of_sims=num_of_sims, print_dataframe=print_dataframe
        ), 2)

    def __call__(
        self, position_list, num_of_periods, *args, 
        persistant_safe_f=None, symbol='', **kwargs
    ):
        # få med car25, car75 med flera metrics haer
        safe_f = persistant_safe_f if persistant_safe_f else \
            self._calculate_safe_f(position_list, num_of_periods, kwargs) 
        
        self.__position_sizer_data_dict[self.__POSITION_SIZE_METRIC_STR] = safe_f
        self.__position_sizer_data_dict[self.__CAPITAL_FRACTION] = safe_f
        self.__position_sizer_data_dict[self.__PERSISTANT_SAFE_F] = safe_f
