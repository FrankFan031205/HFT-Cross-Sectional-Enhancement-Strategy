#%%
import pandas as pd
import numpy as np
import random
import os
import sys
import traceback
from multiprocessing import Pool
import argparse
import polars as pl
from datetime import datetime, timedelta
import concurrent.futures

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config
from utils import get_date_security_info
from utils import data_loader
from formula_additional_feature import function_dict as AdditionalFeatureFormulaDict
from formula_factor import function_dict as FactorFormulaDict
import evaluation

# %%
def init_worker(date:str):
    data_loader.init_clickhouse_client(date)

def get_prev_month(yyyymm: str) -> str:
    dt = datetime.strptime(yyyymm, "%Y%m")
    prev_month_last_day = dt.replace(day=1) - timedelta(days=1)
    return prev_month_last_day.strftime("%Y%m")

def get_sample_securityid_list(Securityid_list: list, sample_num: int, random_seed) -> list:
    if len(Securityid_list) <= sample_num:
        return Securityid_list
    else:
        random.seed(random_seed)
        Securityid_list = random.sample(Securityid_list, sample_num)
        return Securityid_list

def calaulate_factor_date_securityid(date: str,securityid: str,) -> pd.DataFrame:
    additional_feature_list = config.additional_feature_list
    max_waiting_sequence = config.default_params['max_waiting_sequence']
    pred_horizon = config.default_params['pred_horizon']

    df_snapshot, df_trade, df_order, df_cancel, df_index, df_etf_order, df_etf_trade = data_loader.data_loader(date, securityid)
    for additional_feature in additional_feature_list:
        df_snapshot = AdditionalFeatureFormulaDict[additional_feature](df_snapshot, df_trade, df_order, df_cancel, df_index, df_etf_order, df_etf_trade)
    df_snapshot = df_snapshot.sort_values(by='timestamp').reset_index(drop=True)
    df_snapshot['f_use_check'] = [int(i % 20) for i in range(len(df_snapshot))]
    df_snapshot.loc[:max_waiting_sequence, 'f_use_check'] = -1
    df_snapshot['cost'] = df_snapshot['spread'] + 0.001

    for _pred_horizon in [int(r) for r in pred_horizon.split(',')]:
        df_snapshot['label_' + str(_pred_horizon)] = ((df_snapshot['midprice'].shift(-_pred_horizon) / df_snapshot['midprice'].shift(-1) - 1)).fillna(0)
        df_snapshot['label_cost_' + str(_pred_horizon)] = df_snapshot['label_' + str(_pred_horizon)] / df_snapshot['cost']
    df_snapshot['Date'] = int(date)
    int32_columns = [a for a in config.feature_list if ((a != 'midprice') & (a not in config.additional_feature_list) & (a != 'spread'))] + ['SecurityID', 'f_use_check', 'Date']
    float32_columns = ['midprice', 'spread'] + config.additional_feature_list + ['label_' + str(_pred_horizon) for _pred_horizon in [int(r) for r in pred_horizon.split(',')]] + ['label_cost_' + str(_pred_horizon) for _pred_horizon in [int(r) for r in pred_horizon.split(',')]]

    df_snapshot[int32_columns] = df_snapshot[int32_columns].astype(np.int32)
    df_snapshot[float32_columns] = df_snapshot[float32_columns].astype(np.float32)

    return df_snapshot

def get_factor_date_securityid(args) -> None:
    date, securityid = args

    import os
    os.environ['POLARS_MAX_THREADS'] = '1'
    import importlib
    import polars as pl
    importlib.reload(pl)
    
    try:
        df = calaulate_factor_date_securityid(date, securityid)
        df = pl.DataFrame(df)
        return df

    except Exception as e:
        print(date, securityid)
        os.makedirs(config.default_params['error_file'], exist_ok=True)
        error_message = traceback.format_exc()
        with open(os.path.join(config.default_params['error_file'], str(date) + '_' + str(securityid) + '_error_log.txt'), 'w') as file:
            file.write(error_message)

def get_feature_data(date: str):
    SecurityID_list = get_date_security_info.get_securityid_list(date)
    SecurityID_list = get_sample_securityid_list(SecurityID_list, config.default_params['sample_securityid_num'], config.default_params['random_seed'])

    task_list = list(zip(
        [date for _ in range(len(SecurityID_list))],
        SecurityID_list,
    ))

    df_list = []
    with Pool(processes=config.default_params['max_thread'],initializer=init_worker,initargs=(date,)) as pool:
        for df in pool.imap_unordered(get_factor_date_securityid, task_list):
            if df is not None:
                df_list.append(df)

    lf = pl.concat(df_list, rechunk=True)
    lf = lf.sort(['SecurityID','timestamp'])
    return lf

def handel_date_factor(date: str):
    lf = get_feature_data(date)
    for sign_layer_factor_list in config.sign_layer_factor_list:
        lf = lf.with_columns([FactorFormulaDict[name]() for name in sign_layer_factor_list])
    return lf

def save_data_parquet_v1(lf, column_list):
    lf = lf.partition_by(['Date','SecurityID'], as_dict=False)
    def writes(ll):
        ll = ll.sort('timestamp',descending=False)
        Date = ll.select(pl.col('Date').first()).item()
        SecurityID = ll.select(pl.col('SecurityID').first()).item()
        os.makedirs(os.path.join(config.default_params['factor_data_file'], 'TimeSeries', '{}'.format(Date),'{}'.format(SecurityID)),exist_ok=True)

        for factor in column_list:
            if(os.path.exists(os.path.join(config.default_params['factor_data_file'], 'TimeSeries', '{}'.format(Date),'{}'.format(SecurityID),'{}.parquet'.format(factor)))):
                continue
            lll = ll.select([factor])
            lll.write_parquet(os.path.join(config.default_params['factor_data_file'], 'TimeSeries', '{}'.format(Date),'{}'.format(SecurityID),'{}.parquet'.format(factor)))

    with concurrent.futures.ThreadPoolExecutor(max_workers=config.default_params['max_thread']) as executor:
        futures = [executor.submit(writes, ll) for ll in lf]
        for future in concurrent.futures.as_completed(futures):
            future.result()

def save_data_parquet_v2(lf):
    Date = lf.select(pl.col('Date').first()).item()
    lf = lf.partition_by('timestamp',as_dict=False)
    
    import concurrent.futures
    def writes(ll):
        timestamp = ll.select(pl.col('timestamp').first()).item()
        ll.write_parquet(os.path.join(config.default_params['factor_data_file'],'TimeSeries','{}'.format(Date),'{}_{}.parquet'.format(Date,timestamp)))
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [
        executor.submit(writes, ll)
        for ll in lf
        ]

        for future in concurrent.futures.as_completed(futures):
            future.result()

def handel_date(date: str):

    import os
    os.environ['POLARS_MAX_THREADS'] = str(config.default_params['max_thread'])
    import importlib
    import polars as pl
    importlib.reload(pl)

    lf = handel_date_factor(date)
    column_list = ['Date', 'SecurityID', 'timestamp', 'f_use_check', 'midprice', 'bidprice1', 'askprice1', 'bidvolume1', 'askvolume1'] + config.factor_list + ['label_' + str(_pred_horizon) for _pred_horizon in [int(r) for r in config.default_params['pred_horizon'].split(',')]] + ['label_cost_' + str(_pred_horizon) for _pred_horizon in [int(r) for r in config.default_params['pred_horizon'].split(',')]]
    lf = lf.select(column_list)

    if(config.default_params['dump_data'] == 1):
        if(config.default_params['save_data_parquet_version'] == 1):
            save_data_parquet_v1(lf, column_list)
        elif(config.default_params['save_data_parquet_version'] == 2):
            save_data_parquet_v2(lf)

    if(config.default_params['evaluate_data'] == 1):
        evaluation.run_all_evaluations(lf, config)


#%%
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=int)
    args = parser.parse_args()

    date = int(args.date)
    handel_date(date)


# %%
