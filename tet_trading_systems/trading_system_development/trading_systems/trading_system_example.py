import datetime as dt
import json

import pandas as pd

from securities_db_py_dal.dal import price_data_get_req

from tet_mongo_db.systems_mongo_db import TetSystemsMongoDb
from instruments_mongo_db.instruments_mongo_db import InstrumentsMongoDb

from TETrading.position.position_sizer.ext_position_sizer import ExtPositionSizer

from trading_system_properties.trading_system_properties import TradingSystemProperties

from trading_system_state_handler.trad_trading_system_state_handler import TradingSystemStateHandler

from system_development.live_systems.run_systems import run_ext_pos_sizer_trading_system


def entry_logic_example(df, *args, entry_args=None):
    """
    An example of an entry logic function. Returns True/False
    depending on the conditional statement.

    Parameters
    ----------
    :param df:
        'Pandas DataFrame' : Data in the form of a Pandas DataFrame
        or a slice of a Pandas DataFrame.
    :param args:
        'tuple' : A tuple with parameters used with the entry logic.
    :param entry_args:
        Keyword arg 'None/dict' : Key-value pairs with parameters used 
        with the entry logic. Default value=None
    :return:
        'bool' : True/False depending on if the entry logic
        condition is met or not.
    """

    return df['Close'].iloc[-1] >= max(df['Close'].iloc[-entry_args['entry_param_period']:]), \
        'long'


def exit_logic_example(
    df, trail, trailing_exit_price, entry_price, periods_in_pos, 
    *args, exit_args=None
):
    """
    An example of an exit logic function. Returns True/False
    depending on the conditional statement.

    Parameters
    ----------
    :param df:
        'Pandas DataFrame' : Data in the form of a Pandas DataFrame
        or a slice of a Pandas DataFrame.
    :param trail:
        'Boolean' : Additional conditions can be used to
        activate a mechanism for using trailing exit logic.
    :param trailing_exit_price:
        'float/Decimal' : Upon activating a trailing exit
        mechanism, this variable could for example be given
        a price to use as a limit for returning the exit
        condition as True if the last price would fall below it.
    :param args:
        'tuple' : A tuple with parameters used with the exit logic.
    :param exit_args:
        Keyword arg 'None/dict' : Key-value pairs with parameters used 
        with the exit logic. Default value=None
    :return:
        'bool' : True/False depending on if the exit logic
        condition is met or not.
    """

    return df['Close'].iloc[-1] <= min(df['Close'].iloc[-exit_args['exit_param_period']:]), \
        trail, trailing_exit_price


def preprocess_data(
    symbols_list, benchmark_symbol, get_data_function,
    entry_args, exit_args, start_dt, end_dt
):
    df_dict = {
        symbol: pd.json_normalize(
            get_data_function(symbol, start_dt, end_dt)['data']
        )
        for symbol in symbols_list
    }

    benchmark_col_suffix = '_benchmark'
    df_benchmark = pd.json_normalize(
        get_data_function(benchmark_symbol, start_dt, end_dt)['data']
    ).rename(
        columns={
            'Open': f'Open{benchmark_col_suffix}', 
            'High': f'High{benchmark_col_suffix}', 
            'Low': f'Low{benchmark_col_suffix}', 
            'Close': f'Close{benchmark_col_suffix}',
            'Volume': f'Volume{benchmark_col_suffix}', 
            'symbol': f'symbol{benchmark_col_suffix}'
        }
    )

    for symbol, data in dict(df_dict).items():
        if data.empty or len(data) < entry_args['req_period_iters']:
            print(symbol, 'DataFrame empty')
            del df_dict[symbol]
        else:
            df_dict[symbol] = pd.merge_ordered(data, df_benchmark, on='Date', how='inner')
            df_dict[symbol].fillna(method='ffill', inplace=True)
            df_dict[symbol]['Date'] = pd.to_datetime(df_dict[symbol]['Date'])
            df_dict[symbol].set_index('Date', inplace=True)
            #df_dict[symbol] = pd.concat([data, df_benchmark], axis=1)
            #df_dict[symbol].fillna(method='ffill', inplace=True)

            # apply indicators/features to dataframe
            df_dict[symbol]['SMA'] = df_dict[symbol]['Close'].rolling(20).mean()
            df_dict[symbol].dropna(inplace=True)

    return df_dict, None


def get_example_system_props(instruments_db: InstrumentsMongoDb):
    system_name = 'example_system'
    benchmark_symbol = '^OMX'
    entry_args = {
        'req_period_iters': 5, 'entry_param_period': 5
    }
    exit_args = {
        'exit_param_period': 5
    }
    market_list_ids = [
        instruments_db.get_market_list_id('omxs30')
    ]
    symbols_list = []
    for market_list_id in market_list_ids:
        symbols_list += json.loads(
            instruments_db.get_market_list_instrument_symbols(
                market_list_id
            )
        )

    return TradingSystemProperties( 
        preprocess_data,
        (
            symbols_list,
            benchmark_symbol, price_data_get_req,
            entry_args, exit_args
        ),
        TradingSystemStateHandler,(system_name, None),
        (
            run_ext_pos_sizer_trading_system,
            entry_logic_example, exit_logic_example,
            ExtPositionSizer('sharpe_ratio'),
            entry_args, exit_args
        ),
        None, (), ()
    )


if __name__ == '__main__':
    SYSTEMS_DB = TetSystemsMongoDb('mongodb://localhost:27017/', 'systems_db')
    INSTRUMENTS_DB = InstrumentsMongoDb('mongodb://localhost:27017/', 'instruments_db')

    start_dt = dt.datetime(1999, 1, 1)
    end_dt = dt.datetime(2011, 1, 1)

    system_props = get_example_system_props(INSTRUMENTS_DB)

    df_dict, features = system_props.preprocess_data_function(
        system_props.preprocess_data_args[0], '^OMX',
        price_data_get_req,
        system_props.preprocess_data_args[-2],
        system_props.preprocess_data_args[-1],
        start_dt, end_dt
    )

    run_ext_pos_sizer_trading_system(
        df_dict, 'example_system',
        entry_logic_example, exit_logic_example,
        ExtPositionSizer('sharpe_ratio'),
        system_props.preprocess_data_args[-2], 
        system_props.preprocess_data_args[-1], 
        plot_fig=True,
        systems_db=SYSTEMS_DB, client_db=SYSTEMS_DB, insert_into_db=False
    )