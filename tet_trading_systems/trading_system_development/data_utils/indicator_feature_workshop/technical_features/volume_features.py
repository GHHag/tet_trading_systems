import numpy as np
import pandas as pd


def apply_avg_volume(df, period_param, col_name='Volume', suffix=''):
    df[f'Avg_vol_({period_param}){suffix}'] = df[col_name].rolling(period_param).mean()


def apply_rvol(df, period_param=10, col_name='Volume', suffix=''):
    apply_avg_volume(df, period_param, col_name=col_name)

    rvol_list = []
    for i in range(len(df[col_name])):
        if i < period_param:
            rvol_list.append(np.nan)
            continue
        try:
            if df[f'Avg_vol_({period_param})'].iloc[i] <= 0:
                rvol_list.append(0)
                continue
            else:
                rvol_list.append(
                    df[col_name].iloc[i] / df[f'Avg_vol_({period_param})'].iloc[i]
                )
        except RuntimeWarning:
            print('RuntimeWarning')
        except ZeroDivisionError:
            print('ZeroDivisionError')

    df[f'RVOL_({period_param}){suffix}'] = rvol_list


def apply_volume_balance(
    df, period_param=20, col_name_price='Close', col_name_volume='Volume', 
    suffix=''
):
    volume_balance_list = []
    volume_balance_mean = []

    for index, row in enumerate(df.itertuples()):
        if index <= 1:
            volume_balance_list.append(np.nan)
            volume_balance_mean.append(np.nan)
            continue
        if df[col_name_price].iloc[index] > df[col_name_price].iloc[index-1]:
            volume_balance_list.append(float(df[col_name_volume].iloc[index]))
        elif df[col_name_price].iloc[index] < df[col_name_price].iloc[index-1]:
            volume_balance_list.append(float(df[col_name_volume].iloc[index]) * -1)
        else:
            volume_balance_list.append(0)
        if index >= period_param:
            volume_balance_mean.append(np.mean(volume_balance_list[-period_param:]))
        else:
            volume_balance_mean.append(np.nan)

    df[f'Volume_balance_{period_param}{suffix}'] = volume_balance_mean
