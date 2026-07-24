#%%
import numpy as np
import pandas as pd
from clickhouse_connect import get_client

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config
_worker_client = None
df_index_cache, df_etf_order_cache, df_etf_trade_cache = None, None, None

def get_cols(client, db, tb):
    sql = f'DESCRIBE TABLE {db}.`{tb}`'
    res = client.query(sql)
    cols = [row[0] for row in res.result_rows]
    return cols

def get_one_stk(client, db, tb, stk, cols):
    str_cols = ",".join(cols)
    sql = f"SELECT {str_cols} FROM {db}.`{tb}` WHERE SecurityID='{stk}'"

    res = client.query(sql)
    df = pd.DataFrame(res.result_rows)
    try:
        df.columns = cols
    except:
        print(db, 'Missed', stk)
        return pd.DataFrame()
    return df

def get_one_tb(client, db, tb):
    cols = get_cols(client, db, tb)
    str_cols = ",".join(cols)
    sql = f"SELECT {str_cols} FROM {db}.`{tb}`"

    res = client.query(sql)
    df = pd.DataFrame(res.result_rows)
    try:
        df.columns = cols
    except:
        print(db, 'Missed', )
        return pd.DataFrame()
    return df

def init_clickhouse_client(date:str):
    global _worker_client
    if _worker_client is None:
        _worker_client = get_client(
            host=config.default_params['clickhouse']['host'],
            port=int(config.default_params['clickhouse']['port']),
            username=config.default_params['clickhouse']['username'],
            password=config.default_params['clickhouse']['passward'],
            connect_timeout=30,
            send_receive_timeout=600,
        )
    global df_index_cache, df_etf_order_cache, df_etf_trade_cache
    df_index_cache = get_one_tb(_worker_client, 'stock_index_main_500ms_v2', date).sort_values(by=['timestamp','SecurityID']).reset_index(drop=True).rename(columns={'timestamp': 'time'})
    df_etf_order_cache = get_one_tb(_worker_client, 'ETF_Order_Test', date).sort_values(by=['timestamp','SecurityID']).reset_index(drop=True).rename(columns={'timestamp': 'time'})
    df_etf_trade_cache = get_one_tb(_worker_client, 'ETF_Trade_Test', date).sort_values(by=['timestamp','SecurityID']).reset_index(drop=True).rename(columns={'timestamp': 'time'})

def get_clickhouse_client(date:str):
    global _worker_client
    if _worker_client is None:
        init_clickhouse_client(date)
    return _worker_client

def snapshot_gap_rows() -> int:
    """Equity 500ms rows per configured snapshot_len (e.g. 1000ms -> 2)."""
    return int(int(config.default_params["snapshot_len"]) / 500)

def get_timestamp(datetime):
    return int(''.join(datetime.split(' ')[1].split('.')[0].split(':')) + datetime.split(' ')[1].split('.')[1])

def handel_df(df: pd.DataFrame) -> pd.DataFrame:
    """ due to the limit question, we have to ffill the price
        and fillna(0) of the volume and calculate the midprice carefully
    """
    price_ffill_list = ['bidprice{}'.format(i) for i in range(1, 11)] + ['askprice{}'.format(i) for i in range(1, 11)]
    volume_fillna_list = ['bidvolume{}'.format(i) for i in range(1, 11)] + ['askvolume{}'.format(i) for i in range(1, 11)]
    df[price_ffill_list] = df[price_ffill_list].ffill().fillna(0)
    df[volume_fillna_list] = df[volume_fillna_list].fillna(0)

    df['bidprice1_fill'] = df['bidprice1'].replace(0, np.nan).fillna(df['askprice1'])
    df['askprice1_fill'] = df['askprice1'].replace(0, np.nan).fillna(df['bidprice1'])
    df['midprice'] = (df['bidprice1_fill'] + df['askprice1_fill']) / 2
    df['spread'] = (df['askprice1_fill'] - df['bidprice1_fill']) / df['midprice']
    df.loc[(df['askprice1_fill'] == 0) | (df['bidprice1_fill'] == 0), 'spread'] = 0
    df['timestamp'] = df['datetime'].apply(lambda r: get_timestamp(r)).astype('int64')

    df.drop(columns=['datetime', 'bidprice1_fill', 'askprice1_fill'], inplace=True)
    df['SecurityID'] = df['SecurityID'].astype(int)
    return df

def data_loader_basic(client, date: str, securityid: str, DB: str) -> pd.DataFrame:
    if DB == '500ms':
        cols = get_cols(client, DB, date)
        df = get_one_stk(client, DB, date, securityid, cols)
        df = handel_df(df)
        df = df.sort_values(by='timestamp').reset_index(drop=True)
        snapshot_gap = int(int(config.default_params['snapshot_len']) / 500)
        df = df.iloc[::snapshot_gap]
        df = df.sort_values(by='timestamp').reset_index(drop=True)
    elif(DB in ['A_share_Trade', 'A_share_Order', 'A_share_Cancel', 'A_share_Limit']):
        cols = get_cols(client, DB, date)
        df = get_one_stk(client, DB, date, int(securityid), cols)
    elif(DB in ['stock_index_main_500ms_v2','ETF_Order_Test', 'ETF_Trade_Test']):
        df = get_one_tb(client, DB, date)
    else:
        print(f"DB {DB} not found")
        return pd.DataFrame()
    return df

def map_time2timestamp(df_data, df_snapshot):
    snap_times = df_snapshot['timestamp'].copy()
    snap_times = np.insert(snap_times, 0, 0)
    snap_times = np.append(snap_times, 160000000)

    times = df_data['time'].values
    indices = np.searchsorted(snap_times, times, side='left')

    df_data['timestamp'] = snap_times[indices]
    return df_data

def data_loader(date: str, securityid: str) -> pd.DataFrame:
    client = get_clickhouse_client(date)

    ## get snapshot data
    df_snapshot = data_loader_basic(client, date, securityid, '500ms')
    df_limit = data_loader_basic(client, date, securityid, 'A_share_Limit')
    df_limit = df_limit.rename(columns={'limitUpPrice': 'limit_up_price', 'limitDownPrice': 'limit_down_price'})
    df_snapshot = pd.merge(df_snapshot, df_limit, on='SecurityID')

    ## get trade, order, cancel data
    df_trade = data_loader_basic(client, date, securityid, 'A_share_Trade').sort_values(by=['time', 'seqno'])
    df_order = data_loader_basic(client, date, securityid, 'A_share_Order').sort_values(by=['time', 'seqno'])
    df_cancel = data_loader_basic(client, date, securityid, 'A_share_Cancel').sort_values(by=['time', 'seqno'])
    df_trade = map_time2timestamp(df_trade, df_snapshot)
    df_order = map_time2timestamp(df_order, df_snapshot)
    df_cancel = map_time2timestamp(df_cancel, df_snapshot)

    ## get ETF Info data
    global df_index_cache, df_etf_order_cache, df_etf_trade_cache
    df_index = df_index_cache.copy()
    df_etf_order = df_etf_order_cache.copy()
    df_etf_trade = df_etf_trade_cache.copy()
    df_index = map_time2timestamp(df_index, df_snapshot)
    df_etf_order = map_time2timestamp(df_etf_order, df_snapshot)
    df_etf_trade = map_time2timestamp(df_etf_trade, df_snapshot)

    return df_snapshot, df_trade, df_order, df_cancel, df_index, df_etf_order, df_etf_trade
# %%
