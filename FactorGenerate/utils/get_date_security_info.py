from clickhouse_connect import get_client

import os 
import sys 
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config

def get_securityid_list(date:str) -> list:
    client = get_client(host=config.default_params['clickhouse']['host'], 
                        port=int(config.default_params['clickhouse']['port']),
                        username=config.default_params['clickhouse']['username'], 
                        password=config.default_params['clickhouse']['passward'])
    
    db = '500ms'
    sql = f"SELECT DISTINCT SecurityID FROM {db}.`{date}`"
    res = client.query(sql)
    stk_list = [row[0] for row in res.result_rows]
    return stk_list

def get_date_list(start_date:int, end_date:int):
    client = get_client(host=config.default_params['clickhouse']['host'], 
                        port=int(config.default_params['clickhouse']['port']),
                        username=config.default_params['clickhouse']['username'], 
                        password=config.default_params['clickhouse']['passward'])
    
    db = '500ms'
    sql = f"SHOW TABLES FROM {db}"
    res = client.query(sql)
    dates = [row[0] for row in res.result_rows]

    dates = [int(r) for r in dates]
    dates = sorted([r for r in dates if((r>=start_date) & (r<=end_date))])
    dates = [str(r) for r in dates]

    return dates
