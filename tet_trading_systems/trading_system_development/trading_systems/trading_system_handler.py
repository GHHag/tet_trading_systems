import datetime as dt
from typing import List, Dict
import json

from TETrading.data.metadata.trading_system_attributes import TradingSystemAttributes
from TETrading.data.metadata.market_state_enum import MarketState

from tet_trading_systems.trading_system_development.trading_systems.trading_system_properties.trading_system_properties \
    import TradingSystemProperties
from tet_trading_systems.trading_system_development.trading_systems.trading_system_properties.ml_trading_system_properties \
    import MlTradingSystemProperties

from tet_trading_systems.trading_system_management.position_sizer.position_sizer import IPositionSizer

from tet_doc_db.doc_database_meta_classes.tet_signals_doc_db import ITetSignalsDocumentDatabase
from tet_doc_db.doc_database_meta_classes.tet_systems_doc_db import ITetSystemsDocumentDatabase
from tet_doc_db.doc_database_meta_classes.tet_portfolio_doc_db import ITetPortfolioDocumentDatabase
from tet_doc_db.doc_database_meta_classes.time_series_doc_db import ITimeSeriesDocumentDatabase
from tet_doc_db.tet_mongo_db.systems_mongo_db import TetSystemsMongoDb
from tet_doc_db.tet_mongo_db.portfolio_mongo_db import TetPortfolioMongoDb
from tet_doc_db.time_series_mongo_db.time_series_mongo_db import TimeSeriesMongoDb
from tet_doc_db.instruments_mongo_db.instruments_mongo_db import InstrumentsMongoDb


def handle_trading_system(
    system_props: TradingSystemProperties, start_dt, end_dt, 
    systems_db: ITetSystemsDocumentDatabase, 
    client_db: ITetSignalsDocumentDatabase, 
    time_series_db: ITimeSeriesDocumentDatabase=None,
    insert_into_db=False, plot_fig=False 
):
    data, pred_features_data = system_props.preprocess_data_function(
        *system_props.preprocess_data_args, start_dt, end_dt
    )
    #if time_series_db:
    #    time_series_db.insert_pandas_time_series_data(data)

    system_state_handler = system_props.system_state_handler(
        *system_props.system_state_handler_args, systems_db, data
    )
    system_position_sizer: IPositionSizer = system_props.position_sizer(
        *system_props.position_sizer_args
    )

    for _ in range(system_props.required_runs):
        system_state_handler(
            *system_props.system_state_handler_call_args, pred_features_data,
            plot_fig=plot_fig, client_db=client_db, insert_into_db=insert_into_db,
            **system_props.system_state_handler_call_kwargs,
            **system_position_sizer.position_sizer_data_dict
        )
        market_states_data: List[Dict] = json.loads(
            systems_db.get_market_state_data(
                system_props.system_name, MarketState.ENTRY.value
            )
        )

        for data_dict in market_states_data:
            position_list, num_of_periods = systems_db.get_single_symbol_position_list(
                system_props.system_name, data_dict[TradingSystemAttributes.SYMBOL],
                return_num_of_periods=True
            )
            system_position_sizer(
                position_list, num_of_periods,
                *system_props.position_sizer_call_args,
                symbol=data_dict[TradingSystemAttributes.SYMBOL], 
                **system_props.position_sizer_call_kwargs,
                **system_position_sizer.position_sizer_data_dict
            )

    pos_sizer_data_dict = system_position_sizer.get_position_sizer_data_dict()
    systems_db.insert_market_state_data(
        system_props.system_name, json.dumps(pos_sizer_data_dict)
    )


def handle_ext_pos_sizer_trading_system(
    system_props: TradingSystemProperties, start_dt, end_dt, 
    systems_db: ITetSystemsDocumentDatabase, 
    client_db: ITetSignalsDocumentDatabase, 
    time_series_db: ITimeSeriesDocumentDatabase=None,
    insert_into_db=False, plot_fig=False 
):
    data, pred_features_data = system_props.preprocess_data_function(
        *system_props.preprocess_data_args, start_dt, end_dt
    )
    #if time_series_db:
    #    time_series_db.insert_pandas_time_series_data(data)

    system_state_handler = system_props.system_state_handler(
        *system_props.system_state_handler_args, systems_db, data
    )
    system_position_sizer: IPositionSizer = system_props.position_sizer(
        *system_props.position_sizer_args
    )

    for _ in range(system_props.required_runs):
        system_state_handler(
            *system_props.system_state_handler_call_args, pred_features_data,
            plot_fig=plot_fig, client_db=client_db, insert_into_db=insert_into_db,
            **system_props.system_state_handler_call_kwargs,
            **system_position_sizer.position_sizer_data_dict
        )
        position_list, num_of_periods = systems_db.get_position_list(
            system_props.system_name, return_num_of_periods=True
        )
        system_position_sizer(
            position_list, num_of_periods,
            *system_props.position_sizer_call_args,
            **system_props.position_sizer_call_kwargs,
            **system_position_sizer.position_sizer_data_dict
        )

    pos_sizer_data_dict = system_position_sizer.get_position_sizer_data_dict()
    systems_db.insert_system_metrics(system_props.system_name, pos_sizer_data_dict)
 

def handle_ml_trading_system(
    system_props: MlTradingSystemProperties, start_dt, end_dt, 
    systems_db: ITetSystemsDocumentDatabase, 
    client_db: ITetSignalsDocumentDatabase, 
    time_series_db: ITimeSeriesDocumentDatabase=None,
    insert_into_db=False, plot_fig=False 
):
    data, pred_features_data = system_props.preprocess_data_function(
        *system_props.preprocess_data_args, start_dt, end_dt
    )
    #if time_series_db:
    #   time_series_db.insert_pandas_time_series_data(data)

    system_state_handler = system_props.system_state_handler(
        *system_props.system_state_handler_args, data, pred_features_data, systems_db,
    )
    system_position_sizer: IPositionSizer = system_props.position_sizer(
        *system_props.position_sizer_args
    )

    for _ in range(system_props.required_runs):
        system_state_handler(
            *system_props.system_state_handler_call_args,
            plot_fig=plot_fig, 
            client_db=client_db, insert_into_db=insert_into_db,
            **system_props.system_state_handler_call_kwargs,
            **system_position_sizer.position_sizer_data_dict
        )
        market_states_data: List[Dict] = json.loads(
            systems_db.get_market_state_data(
                system_props.system_name, MarketState.ENTRY.value
            )
        )

        for data_dict in market_states_data:
            position_list, num_of_periods = systems_db.get_single_symbol_position_list(
                system_props.system_name, data_dict[TradingSystemAttributes.SYMBOL],
                return_num_of_periods=True
            )
            system_position_sizer(
                position_list, num_of_periods,
                *system_props.position_sizer_call_args,
                symbol=data_dict[TradingSystemAttributes.SYMBOL], 
                **system_props.position_sizer_call_kwargs,
                **system_position_sizer.position_sizer_data_dict
            )

    pos_sizer_data_dict = system_position_sizer.get_position_sizer_data_dict()
    systems_db.insert_market_state_data(
        system_props.system_name, json.dumps(pos_sizer_data_dict)
    )


def handle_trading_system_portfolio(
    system_props: TradingSystemProperties,
    client_db: ITetSignalsDocumentDatabase,
    portfolio_db: ITetPortfolioDocumentDatabase,
    insert_into_db=False
):
    portfolio = system_props.portfolio(
        *system_props.portfolio_args, 
        client_db, portfolio_db
    )
    portfolio(
        *system_props.portfolio_call_args, 
        insert_into_db=insert_into_db
    )


if __name__ == '__main__':
    import tet_trading_systems.trading_system_development.trading_systems.env as env
    #INSTRUMENTS_DB = InstrumentsMongoDb(env.LOCALHOST_MONGO_DB_URL, 'instruments_db')
    TIME_SERIES_DB = TimeSeriesMongoDb(env.LOCALHOST_MONGO_DB_URL, 'time_series_db')
    SYSTEMS_DB = TetSystemsMongoDb(env.LOCALHOST_MONGO_DB_URL, 'systems_db')
    #CLIENT_DB = TetSystemsMongoDb(env.LOCALHOST_MONGO_DB_URL, 'client_db')
    CLIENT_DB = SYSTEMS_DB

    INSTRUMENTS_DB = InstrumentsMongoDb(env.ATLAS_MONGO_DB_URL, 'client_db')
    #TIME_SERIES_DB = TimeSeriesMongoDb(env.ATLAS_MONGO_DB_URL, 'client_db')
    #SYSTEMS_DB = TetSystemsMongoDb(env.ATLAS_MONGO_DB_URL, 'systems_db')
    #CLIENT_DB = TetSystemsMongoDb(env.ATLAS_MONGO_DB_URL, 'client_db')
    
    ML_SYSTEMS_DB = TetSystemsMongoDb(env.LOCALHOST_MONGO_DB_URL, 'systems_db')
    ML_ORDERS_DB = ML_SYSTEMS_DB 

    PORTFOLIOS_DB = TetPortfolioMongoDb(env.LOCALHOST_MONGO_DB_URL, 'client_db')

    #start_dt = dt.datetime(1999, 1, 1)
    #end_dt = dt.datetime(2011, 1, 1)
    start_dt = dt.datetime(2015, 9, 16)
    end_dt = dt.datetime.now()

    systems_props_list: List[TradingSystemProperties] = []
    ml_systems_props_list: List[MlTradingSystemProperties] = []

    #from tet_trading_systems.trading_system_development.trading_systems.trading_system_example import get_example_system_props
    #example_system_props = get_example_system_props(INSTRUMENTS_DB)
    #systems_props_list.append(example_system_props)
 
    from tet_trading_systems.trading_system_development.trading_systems.live_systems.mean_reversion_stocks import get_mean_reversion_stocks_props
    mean_reversion_stocks_props = get_mean_reversion_stocks_props(INSTRUMENTS_DB)
    systems_props_list.append(mean_reversion_stocks_props)
 
    #from system_development.systems_t1.low_vol_bo import get_low_vol_bo_props
    #low_vol_bo_props = get_low_vol_bo_props(INSTRUMENTS_DB)
    #systems_props_list.append(low_vol_bo_props)

    #from tet_trading_systems.trading_system_development.trading_systems.ml_trading_system_example import get_example_ml_system_props
    #example_ml_system_props = get_example_ml_system_props(INSTRUMENTS_DB)
    #ml_systems_props_list.append(example_ml_system_props)
    #systems_props_list.append(example_ml_system_props)

    #from system_development.systems_t1.omxs_ml_system import get_omxs_ml_system_props
    #omxs_ml_system_props = get_omxs_ml_system_props(INSTRUMENTS_DB)
    #ml_systems_props_list.append(omxs_ml_system_props)

    for system_props in systems_props_list:
        # implementera protocol för system_handler_function?
        system_props.system_handler_function(
            system_props, start_dt, end_dt, 
            SYSTEMS_DB, CLIENT_DB, 
            time_series_db=TIME_SERIES_DB, 
            insert_into_db=True, plot_fig=False
        )
        #if system_props.portfolio_args:
        #    handle_trading_system_portfolio(
        #        system_props, 
        #        CLIENT_DB, PORTFOLIOS_DB, 
        #        insert_into_db=True
        #    )
    
    for system_props in ml_systems_props_list:
        #handle_ml_trading_system(
        system_props.system_handler_function(
            system_props, start_dt, end_dt, 
            ML_SYSTEMS_DB, ML_ORDERS_DB, 
            time_series_db=TIME_SERIES_DB, 
            insert_into_db=True, plot_fig=False
        )
        #if system_props['portfolio_args']:
        #    handle_trading_system_portfolio(
        #        system_props, 
        #        SYSTEMS_DB, CLIENT_DB, PORTFOLIOS_DB, 
        #        insert_into_db=False
        #    )
