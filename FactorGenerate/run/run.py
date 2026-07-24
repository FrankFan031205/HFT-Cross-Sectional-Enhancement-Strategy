#%%
import argparse 
import os 
import sys
from datetime import datetime

sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config
from utils import get_date_security_info

#%%
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate_start_date', type=int)
    parser.add_argument('--generate_end_date', type=int)
    parser.add_argument('--remove_history_data', type=int, default=1)
    args = parser.parse_args()

    start_date = int(args.generate_start_date)
    end_date = int(args.generate_end_date)
    remove_history_data = int(args.remove_history_data)

    if(remove_history_data):
        from utils import clear_cache
        clear_cache.delete_folder_by_cmd(config.default_params['factor_data_file'])
        clear_cache.delete_folder_by_cmd(config.default_params['error_file'])

  
    date_list = get_date_security_info.get_date_list(start_date, end_date)

    for date in date_list:
        if(config.default_params['dump_data'] == 1):
            os.makedirs(os.path.join(config.default_params['factor_data_file'], 'TimeSeries', '{}'.format(date)), exist_ok=True)  
        print('start handel {} at'.format(date), datetime.now())
        os.system(sys.executable + ' ' + os.path.dirname(os.path.abspath(__file__)) +'/basic.py' + ' --date {}'.format(date))

