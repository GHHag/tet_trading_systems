from tet_doc_db.doc_database_meta_classes.tet_systems_doc_db import ITetSystemsDocumentDatabase

from TETrading.utils.metadata.market_state_enum import MarketState
from TETrading.trading_system.trading_system import TradingSystem
from TETrading.utils.monte_carlo_functions import calculate_safe_f


def run_trading_system(
    data_dict, system_name, entry_func, exit_func,
    entry_args, exit_args, *args, 
    capital=10000, capital_fraction=1.0, num_of_sims=2500, 
    market_state_null_default=False,
    print_dataframe=False, plot_fig=False, plot_positions=False, write_to_file_path=None, 
    systems_db: ITetSystemsDocumentDatabase=None, 
    client_db: ITetSystemsDocumentDatabase=None, 
    insert_into_db=False,
    save_best_estimate_trades_path=None,
    **kwargs
):
    ts = TradingSystem(system_name, data_dict, entry_func, exit_func)
    ts(
        capital=capital,
        capital_fraction=capital_fraction,
        market_state_null_default=market_state_null_default,
        plot_performance_summary=plot_fig,
        save_summary_plot_to_path=None, 
        system_analysis_to_csv_path=None, 
        plot_returns_distribution=False,
        save_returns_distribution_plot_to_path=None, 
        run_monte_carlo_sims=False,
        num_of_monte_carlo_sims=num_of_sims,
        monte_carlo_data_amount=0.65,
        plot_monte_carlo=plot_fig,
        print_monte_carlo_df=False,
        monte_carlo_analysis_to_csv_path=None, 
        commission_pct_cost=0.0025,
        entry_args=entry_args,
        exit_args=exit_args,
        fixed_position_size=True,
        generate_signals=True,
        plot_positions=plot_positions,
        save_position_figs_path=None,
        write_signals_to_file_path=write_to_file_path,
        insert_data_to_db_bool=insert_into_db,
        signal_handler_db_insert_funcs=
        {
            MarketState.ENTRY.value: client_db.insert_market_state_data,
            MarketState.ACTIVE.value: client_db.insert_market_state_data,
            MarketState.EXIT.value: client_db.insert_market_state_data
        },
        single_symbol_pos_list_db_insert_func=systems_db.insert_single_symbol_position_list,
        json_format_single_symbol_pos_list_db_insert_func=client_db.insert_single_symbol_position_list,
        full_pos_list_db_insert_func=systems_db.insert_position_list,
        json_format_full_pos_list_db_insert_func=client_db.insert_position_list
    )


""" def run_multiple_run_req_pos_sizer_trading_system(
    data_dict, system_name, entry_func, exit_func, pos_sizer,
    entry_args, exit_args, *args, 
    capital=10000, capital_f=1.0,
    num_of_sims=2500, num_positions_forecasted=120, 
    market_state_null_default=False,
    print_dataframe=False, plot_fig=False, write_to_file_path=None, 
    systems_db: ITetSystemsDocumentDatabase=None, 
    client_db: ITetSystemsDocumentDatabase=None, 
    insert_into_db=False,
    save_best_estimate_trades_path=None
):
    position_sizer_dict = None
    for ts_run in range(2):
        if position_sizer_dict is not None:
            capital_f = position_sizer_dict 

        if ts_run < 1:
            db_insert_bool = False
        else:
            db_insert_bool = insert_into_db

        ts = TradingSystem(
            system_name, data_dict, entry_func, exit_func, pos_sizer
        )
        ts(
            capital=capital,
            capital_fraction=capital_f,
            market_state_null_default=market_state_null_default,
            plot_performance_summary=plot_fig,
            save_summary_plot_to_path=None, 
            system_analysis_to_csv_path=None, 
            plot_returns_distribution=False,
            save_returns_distribution_plot_to_path=None, 
            run_monte_carlo_sims=False,
            num_of_monte_carlo_sims=num_of_sims,
            monte_carlo_data_amount=0.65,
            plot_monte_carlo=plot_fig,
            print_monte_carlo_df=False,
            monte_carlo_analysis_to_csv_path=None, 
            commission_pct_cost=0.0025,
            entry_args=entry_args,
            exit_args=exit_args,
            fixed_position_size=True,
            generate_signals=True,
            plot_positions=False,
            save_position_figs_path=None,
            write_signals_to_file_path=write_to_file_path,
            insert_data_to_db_bool=db_insert_bool,
            signal_handler_db_insert_funcs=
            {
                MarketState.ENTRY.value: client_db.insert_market_state_data,
                MarketState.ACTIVE.value: client_db.insert_market_state_data,
                MarketState.EXIT.value: client_db.insert_market_state_data
            },
            single_symbol_pos_list_db_insert_func=systems_db.insert_single_symbol_position_list,
            json_format_single_symbol_pos_list_db_insert_func=client_db.insert_single_symbol_position_list,
            full_pos_list_db_insert_func=systems_db.insert_position_list,
            json_format_full_pos_list_db_insert_func=client_db.insert_position_list#,
            #full_pos_list_slice_param=num_positions_forecasted
        )
        position_sizer_dict = ts.position_sizer_dict


def run_ext_pos_sizer_trading_system(
    data_dict, system_name, entry_func, exit_func, pos_sizer, 
    entry_args, exit_args, *args, 
    capital=10000, capital_f=1.0, num_of_sims=2500, commission_pct_cost=0.0025,
    tolerated_pct_max_dd=15, dd_percentile_threshold=0.85, 
    years_to_forecast=2, avg_yearly_periods=251, 
    market_state_null_default=False,
    print_dataframe=False, plot_fig=False, write_to_file_path=None, 
    systems_db: ITetSystemsDocumentDatabase=None, 
    client_db: ITetSystemsDocumentDatabase=None, 
    insert_into_db=False,
    save_best_estimate_trades_path=None
):
    avg_yearly_positions = 0
    for ts_run in range(2):
        if ts_run < 1:
            db_insert_bool = False
        else:
            db_insert_bool = insert_into_db

        ts = TradingSystem(
            system_name, data_dict, entry_func, exit_func, pos_sizer
        )
        ts(
            capital=capital,
            capital_fraction=capital_f,
            market_state_null_default=market_state_null_default,
            plot_performance_summary=plot_fig,
            save_summary_plot_to_path=None, 
            system_analysis_to_csv_path=None, 
            plot_returns_distribution=False,
            save_returns_distribution_plot_to_path=None, 
            run_monte_carlo_sims=False,
            num_of_monte_carlo_sims=num_of_sims,
            monte_carlo_data_amount=0.65,
            plot_monte_carlo=plot_fig,
            monte_carlo_analysis_to_csv_path=None, 
            commission_pct_cost=commission_pct_cost,
            entry_args=entry_args,
            exit_args=exit_args,
            fixed_position_size=True,
            generate_signals=True,
            plot_positions=False,
            save_position_figs_path=None,
            write_signals_to_file_path=write_to_file_path,
            insert_data_to_db_bool=db_insert_bool,
            signal_handler_db_insert_funcs=
            {
                MarketState.ENTRY.value: client_db.insert_market_state_data,
                MarketState.ACTIVE.value: client_db.insert_market_state_data,
                MarketState.EXIT.value: client_db.insert_market_state_data
            },
            single_symbol_pos_list_db_insert_func=systems_db.insert_single_symbol_position_list,
            json_format_single_symbol_pos_list_db_insert_func=client_db.insert_single_symbol_position_list,
            full_pos_list_db_insert_func=systems_db.insert_position_list,
            json_format_full_pos_list_db_insert_func=client_db.insert_position_list,
            full_pos_list_slice_param=int(avg_yearly_positions * (years_to_forecast + years_to_forecast / 2))
        )

        if ts_run < 1:
            sorted_position_lists = sorted(ts.pos_lists, key=len, reverse=True)
            position_list_lengths = [len(i) for i in sorted_position_lists[:int(len(data_dict) / 4 + 0.5)]] \
                if len(data_dict) > 1 \
                else [len(sorted_position_lists[0])]
            avg_yearly_positions = int(
                (sum(position_list_lengths) / len(position_list_lengths)) / years_to_forecast + 0.5
            ) 
            capital_f = round(calculate_safe_f(
                ts.full_pos_list, ts.total_period_len, tolerated_pct_max_dd, dd_percentile_threshold,
                forecast_positions=avg_yearly_positions * (years_to_forecast + years_to_forecast / 2), 
                forecast_data_fraction=(avg_yearly_positions * years_to_forecast) / 
                                    (avg_yearly_positions * (years_to_forecast + years_to_forecast / 2)),
                capital=capital, num_of_sims=num_of_sims, print_dataframe=print_dataframe
            ), 2)
            write_to_file_path = None
        else:
            mc_data, num_of_periods = ts.run_monte_carlo_simulation(
                capital_f, forecast_positions=avg_yearly_positions * (years_to_forecast + years_to_forecast / 2), 
                forecast_data_fraction=(avg_yearly_positions * years_to_forecast) / 
                                    (avg_yearly_positions * (years_to_forecast + years_to_forecast / 2)),
                capital=capital, num_of_sims=num_of_sims, plot_fig=plot_fig, 
                save_fig_to_path=save_best_estimate_trades_path
            )
            if db_insert_bool:
                result = client_db.insert_system_metrics(
                    system_name,
                    {
                        'sharpe_ratio': mc_data[-1]['Sharpe ratio'],
                        'expectancy': mc_data[-1]['Expectancy'],
                        'profit_factor': float(mc_data[-1]['Profit factor']),
                        'CAR25': mc_data[-1]['CAR25'],
                        'CAR75': mc_data[-1]['CAR75'],
                        'safe-f': capital_f
                        #pos_sizer.position_size_metric_str: capital_f
                    },
                    {
                        'num_of_periods': num_of_periods
                    }
                ) """