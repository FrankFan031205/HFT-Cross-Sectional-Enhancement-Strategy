#%%
import numpy as np
import pandas as pd
import warnings
np.seterr(divide='ignore',invalid='ignore')
warnings.filterwarnings('ignore')
precision_accuracy = 0.000001
import inspect

#%%
def _merge_feature(df_snapshot, df_feature, feature_name):
    df_snapshot = pd.merge(df_snapshot, df_feature, on='timestamp', how='left')
    df_snapshot[feature_name] = df_snapshot[feature_name].fillna(0)
    return df_snapshot


def _qty_feature(df_snapshot, df_, feature_name):
    df_ = df_[['timestamp','qty']].groupby('timestamp').sum().reset_index()
    df_.columns = ['timestamp',feature_name]
    return _merge_feature(df_snapshot, df_, feature_name)


def _amount_feature(df_snapshot, df_, feature_name):
    df_ = df_.copy()
    df_[feature_name] = df_['price'] * df_['qty']
    df_ = df_[['timestamp',feature_name]].groupby('timestamp').sum().reset_index()
    return _merge_feature(df_snapshot, df_, feature_name)


def _count_feature(df_snapshot, df_, feature_name):
    df_ = df_[['timestamp']].copy()
    df_[feature_name] = 1
    df_ = df_.groupby('timestamp')[feature_name].sum().reset_index()
    return _merge_feature(df_snapshot, df_, feature_name)


#%%
def volume(df_snapshot, df_trade, df_order, df_cancel, df_index, df_etf_order, df_etf_trade):
    """Total volume this snapshot"""

    df_trade_ = df_trade.copy()
    return _qty_feature(df_snapshot, df_trade_, 'volume')


def activate_buy_volume(df_snapshot, df_trade, df_order, df_cancel, df_index, df_etf_order, df_etf_trade):
    """Total activate buy volume this snapshot"""

    df_trade_ = df_trade[df_trade['side']==0].copy()
    return _qty_feature(df_snapshot, df_trade_, 'activate_buy_volume')


def activate_sell_volume(df_snapshot, df_trade, df_order, df_cancel, df_index, df_etf_order, df_etf_trade):
    """Total activate sell volume this snapshot"""

    df_trade_ = df_trade[df_trade['side']!=0].copy()
    return _qty_feature(df_snapshot, df_trade_, 'activate_sell_volume')


def activate_buy_amount(df_snapshot, df_trade, df_order, df_cancel, df_index, df_etf_order, df_etf_trade):
    """Total activate buy amount this snapshot"""

    df_trade_ = df_trade[df_trade['side']==0].copy()
    return _amount_feature(df_snapshot, df_trade_, 'activate_buy_amount')


def activate_sell_amount(df_snapshot, df_trade, df_order, df_cancel, df_index, df_etf_order, df_etf_trade):
    """Total activate sell amount this snapshot"""

    df_trade_ = df_trade[df_trade['side']!=0].copy()
    return _amount_feature(df_snapshot, df_trade_, 'activate_sell_amount')


def activate_buy_trade_count(df_snapshot, df_trade, df_order, df_cancel, df_index, df_etf_order, df_etf_trade):
    """Total activate buy trade count this snapshot"""

    df_trade_ = df_trade[df_trade['side']==0].copy()
    return _count_feature(df_snapshot, df_trade_, 'activate_buy_trade_count')


def activate_sell_trade_count(df_snapshot, df_trade, df_order, df_cancel, df_index, df_etf_order, df_etf_trade):
    """Total activate sell trade count this snapshot"""

    df_trade_ = df_trade[df_trade['side']!=0].copy()
    return _count_feature(df_snapshot, df_trade_, 'activate_sell_trade_count')


def trade_count(df_snapshot, df_trade, df_order, df_cancel, df_index, df_etf_order, df_etf_trade):
    """Total trade count this snapshot"""

    df_trade_ = df_trade.copy()
    return _count_feature(df_snapshot, df_trade_, 'trade_count')


def large_buy_trade_volume(df_snapshot, df_trade, df_order, df_cancel, df_index, df_etf_order, df_etf_trade):
    """Total large buy trade volume this snapshot"""

    large_threshold = df_trade['qty'].quantile(0.9) if len(df_trade) > 0 else 0
    df_trade_ = df_trade[(df_trade['side']==0) & (df_trade['qty']>=large_threshold)].copy()
    return _qty_feature(df_snapshot, df_trade_, 'large_buy_trade_volume')


def large_sell_trade_volume(df_snapshot, df_trade, df_order, df_cancel, df_index, df_etf_order, df_etf_trade):
    """Total large sell trade volume this snapshot"""

    large_threshold = df_trade['qty'].quantile(0.9) if len(df_trade) > 0 else 0
    df_trade_ = df_trade[(df_trade['side']!=0) & (df_trade['qty']>=large_threshold)].copy()
    return _qty_feature(df_snapshot, df_trade_, 'large_sell_trade_volume')


def activate_buy_order_volume(df_snapshot, df_trade, df_order, df_cancel, df_index, df_etf_order, df_etf_trade):
    """Total activate buy order volume this snapshot"""

    df_order_ = df_order[df_order['side']==0].copy()
    return _qty_feature(df_snapshot, df_order_, 'activate_buy_order_volume')


def activate_sell_order_volume(df_snapshot, df_trade, df_order, df_cancel, df_index, df_etf_order, df_etf_trade):
    """Total activate sell order volume this snapshot"""

    df_order_ = df_order[df_order['side']!=0].copy()
    return _qty_feature(df_snapshot, df_order_, 'activate_sell_order_volume')


def activate_buy_order_count(df_snapshot, df_trade, df_order, df_cancel, df_index, df_etf_order, df_etf_trade):
    """Total activate buy order count this snapshot"""

    df_order_ = df_order[df_order['side']==0].copy()
    return _count_feature(df_snapshot, df_order_, 'activate_buy_order_count')


def activate_sell_order_count(df_snapshot, df_trade, df_order, df_cancel, df_index, df_etf_order, df_etf_trade):
    """Total activate sell order count this snapshot"""

    df_order_ = df_order[df_order['side']!=0].copy()
    return _count_feature(df_snapshot, df_order_, 'activate_sell_order_count')


def aggressive_buy_order_volume(df_snapshot, df_trade, df_order, df_cancel, df_index, df_etf_order, df_etf_trade):
    """Total aggressive buy order volume this snapshot"""

    df_order_ = df_order[df_order['side']==0].copy()
    df_quote = df_snapshot[['timestamp','askprice1']].drop_duplicates('timestamp')
    df_order_ = pd.merge(df_order_, df_quote, on='timestamp', how='left')
    df_order_ = df_order_[df_order_['price']>=df_order_['askprice1']].copy()
    return _qty_feature(df_snapshot, df_order_, 'aggressive_buy_order_volume')


def aggressive_sell_order_volume(df_snapshot, df_trade, df_order, df_cancel, df_index, df_etf_order, df_etf_trade):
    """Total aggressive sell order volume this snapshot"""

    df_order_ = df_order[df_order['side']!=0].copy()
    df_quote = df_snapshot[['timestamp','bidprice1']].drop_duplicates('timestamp')
    df_order_ = pd.merge(df_order_, df_quote, on='timestamp', how='left')
    df_order_ = df_order_[df_order_['price']<=df_order_['bidprice1']].copy()
    return _qty_feature(df_snapshot, df_order_, 'aggressive_sell_order_volume')


def buy_order_distance_amount(df_snapshot, df_trade, df_order, df_cancel, df_index, df_etf_order, df_etf_trade):
    """Total buy order distance amount this snapshot"""

    df_order_ = df_order[df_order['side']==0].copy()
    df_quote = df_snapshot[['timestamp','askprice1']].drop_duplicates('timestamp')
    df_order_ = pd.merge(df_order_, df_quote, on='timestamp', how='left')
    df_order_['buy_order_distance_amount'] = (df_order_['price'] - df_order_['askprice1']).abs() * df_order_['qty']
    df_order_ = df_order_[['timestamp','buy_order_distance_amount']].groupby('timestamp').sum().reset_index()
    return _merge_feature(df_snapshot, df_order_, 'buy_order_distance_amount')


def sell_order_distance_amount(df_snapshot, df_trade, df_order, df_cancel, df_index, df_etf_order, df_etf_trade):
    """Total sell order distance amount this snapshot"""

    df_order_ = df_order[df_order['side']!=0].copy()
    df_quote = df_snapshot[['timestamp','bidprice1']].drop_duplicates('timestamp')
    df_order_ = pd.merge(df_order_, df_quote, on='timestamp', how='left')
    df_order_['sell_order_distance_amount'] = (df_order_['price'] - df_order_['bidprice1']).abs() * df_order_['qty']
    df_order_ = df_order_[['timestamp','sell_order_distance_amount']].groupby('timestamp').sum().reset_index()
    return _merge_feature(df_snapshot, df_order_, 'sell_order_distance_amount')


def cancel_buy_volume(df_snapshot, df_trade, df_order, df_cancel, df_index, df_etf_order, df_etf_trade):
    """Total buy cancel volume this snapshot"""

    df_cancel_ = df_cancel[df_cancel['side']==0].copy()
    df_cancel_ = df_cancel_[['timestamp','cancel_qty']].groupby('timestamp').sum().reset_index()
    df_cancel_.columns = ['timestamp','cancel_buy_volume']
    return _merge_feature(df_snapshot, df_cancel_, 'cancel_buy_volume')


def cancel_sell_volume(df_snapshot, df_trade, df_order, df_cancel, df_index, df_etf_order, df_etf_trade):
    """Total sell cancel volume this snapshot"""

    df_cancel_ = df_cancel[df_cancel['side']!=0].copy()
    df_cancel_ = df_cancel_[['timestamp','cancel_qty']].groupby('timestamp').sum().reset_index()
    df_cancel_.columns = ['timestamp','cancel_sell_volume']
    return _merge_feature(df_snapshot, df_cancel_, 'cancel_sell_volume')


def near_buy_cancel_volume(df_snapshot, df_trade, df_order, df_cancel, df_index, df_etf_order, df_etf_trade):
    """Total near buy cancel volume this snapshot"""

    df_cancel_ = df_cancel[df_cancel['side']==0].copy()
    df_quote = df_snapshot[['timestamp','bidprice1']].drop_duplicates('timestamp')
    df_cancel_ = pd.merge(df_cancel_, df_quote, on='timestamp', how='left')
    df_cancel_ = df_cancel_[df_cancel_['price']>=df_cancel_['bidprice1']].copy()
    df_cancel_ = df_cancel_[['timestamp','cancel_qty']].groupby('timestamp').sum().reset_index()
    df_cancel_.columns = ['timestamp','near_buy_cancel_volume']
    return _merge_feature(df_snapshot, df_cancel_, 'near_buy_cancel_volume')


def near_sell_cancel_volume(df_snapshot, df_trade, df_order, df_cancel, df_index, df_etf_order, df_etf_trade):
    """Total near sell cancel volume this snapshot"""

    df_cancel_ = df_cancel[df_cancel['side']!=0].copy()
    df_quote = df_snapshot[['timestamp','askprice1']].drop_duplicates('timestamp')
    df_cancel_ = pd.merge(df_cancel_, df_quote, on='timestamp', how='left')
    df_cancel_ = df_cancel_[df_cancel_['price']<=df_cancel_['askprice1']].copy()
    df_cancel_ = df_cancel_[['timestamp','cancel_qty']].groupby('timestamp').sum().reset_index()
    df_cancel_.columns = ['timestamp','near_sell_cancel_volume']
    return _merge_feature(df_snapshot, df_cancel_, 'near_sell_cancel_volume')


#%%
def create_function_dict():
    function_dict = {}
    current_module = inspect.currentframe().f_back.f_globals  # get a factor_func_dict

    for name, obj in current_module.items():
        if inspect.isfunction(obj) and not name.startswith('_') and name != 'create_function_dict':
            function_dict[name] = obj

    return function_dict

function_dict = create_function_dict()
