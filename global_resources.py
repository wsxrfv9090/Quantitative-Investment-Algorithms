import os
import pandas as pd
import torch
import numpy as np
import torch
import random


# Set the working directory to a known absolute path
default_dir = r'D:\ImportanFiles\Coding Related\Repositories\Quantitative-Investment-Algorithms'
os.chdir(default_dir)

global_paths = {
    'CAPM': os.path.join(default_dir, 'CAPMlib'),
    'K means cluster': os.path.join(default_dir, 'kmc_torch'),
    'Data': os.path.join(default_dir, 'Data'),
    'SVM': os.path.join(default_dir, 'SVM'),
    'Output': os.path.join(default_dir, 'Output')
}


# has to be a interger with range: [0, 2**32)
RANDOM_SEED = random.getrandbits(32)
MAX_ITERATION = int(1e3)
TOLERANCE = 1e-6
N_RESTARTS = int(10)
DTYPE = torch.float32


def set_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def reinitiate_seed_torch() -> int:
    global RANDOM_SEED
    RANDOM_SEED = random.getrandbits(32)
    return RANDOM_SEED


# Changes the directory to specified working directory, which is the repo directory you cloned.
def ch_dir_to_repo(work_dir = default_dir):
    os.chdir(work_dir)
    global default_dir
    if work_dir != default_dir:
        default_dir = work_dir
    return work_dir


# Read files based on there sufixes.
def read_and_return_pd_df(file_path):
    print(f"Reading files from: {file_path}")

    # Check if the file is csv or excel
    if file_path.endswith('.csv'):
        input_file_pd = pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        input_file_pd = pd.read_excel(file_path)
    elif file_path.endswith('.data'):
        input_file_pd = pd.read_csv(file_path)
    else:
        print("The file format is not supported. Please provide a csv, a data or a excel file.")
        return False
    return input_file_pd

# Change the column names to English
def change_head_to_ENG(pd_df):
    # 证券代码  证券名称    交易时间	开盘价	最高价	最低价	收盘价	涨跌	涨跌幅	成交量	成交额
    pd_df.columns = ['SECU_CODE', 'SECU_NAME', 'DATE', 'OPENING', 'HIGHEST', 'LOWEST', 'CLOSING', 'CHANGE', 'PCT_CHANGE', 'VOLUME', 'AMOUNT']

def drop_and_change_head(pd_df):
    to_save = ['日期', '股票代码', '涨跌幅']
    pd_df = pd_df[to_save]
    pd_df.columns = ['Date', 'Stock ID', 'Pct_Change']
    return pd_df
    # print(pd_df.info())
    
# Change the data type of the security code to string
def change_secu_code_to_str(pd_df):
    pd_df = pd_df.copy()
    pd_df['Stock ID'] = pd_df['Stock ID'].astype('U6')
    return pd_df

# Change the data type of the numerical data to float64
def change_numerical_data_to_float64(pd_df):
    cols = ['OPENING', 'HIGHEST', 'LOWEST', 'CLOSING', 'CHANGE', 'PCT_CHANGE', 'VOLUME', 'AMOUNT']
    pd_df[cols] = pd_df[cols].replace({',': '', '--': 'NaN'}, regex = True).astype('float64')
    
def yearly_to_daily(R_f_y):
        return ((R_f_y + 1) ** (1 / 365)) - 1

def euclidean_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = x.view(-1).to(torch.float64)
    y = y.view(-1).to(torch.float64)
    
    dist = torch.norm(x - y, p=2)
    return dist

def display_nan_for(df = None):
    if df is None:
        print("None dataframe provided.")
        return None
    na_rows = df[df.isna().any(axis=1)]
    if not na_rows.empty:
        print(na_rows)
    return na_rows


# This function can only be used 

STARTDATE = '2024-01-02'
ENDDATE = '2025-01-02'
# def cut_df_by_date(start_date = None, end_date = None, pd_df = None):
#     if start_date == None or end_date == None:
#         print("You didn't pass in a date")
#         return None
#     start_ts = pd.to_datetime(start_date, format='%Y-%m-%d')
#     end_ts = pd.to_datetime(end_date, format='%Y-%m-%d')
#     pd_df = pd_df.copy()
#     pd_df['日期'] = pd.to_datetime(pd_df['日期'], format='%Y-%m-%d')
#     if (start_ts in temp_df['日期'].values) and (end_ts in temp_df['日期'].values):
#         temp_df = temp_df.loc[
#             (temp_df['日期'] >= start_ts) &
#             (temp_df['日期'] <= end_ts)
#         ]
#     else:
#         skip_count += 1
#             # skip this file if it doesn't cover the full date range
    

def get_df_dict(start_date = STARTDATE, end_date = ENDDATE, data_dir = None):
    print(start_date)
    print(end_date)
    if data_dir == None:
        print('No data_dir parameter passed, please put data directory into the parameters.')
        return None
    
    start_ts = pd.to_datetime(start_date, format='%Y-%m-%d')
    end_ts = pd.to_datetime(end_date, format='%Y-%m-%d')
    dfs = {}
    total_count = 0
    skip_count = 0
    
    for fname in os.listdir(data_dir):
        total_count += 1
        full_path = os.path.join(data_dir, fname)
        temp_df = read_and_return_pd_df(full_path)
        temp_df['日期'] = pd.to_datetime(temp_df['日期'], format='%Y-%m-%d')
        if (start_ts in temp_df['日期'].values) and (end_ts in temp_df['日期'].values):
            temp_df = temp_df.loc[
                (temp_df['日期'] >= start_ts) &
                (temp_df['日期'] <= end_ts)
            ]
        else:
            skip_count += 1
            # skip this file if it doesn't cover the full date range
            continue
        if (temp_df['收盘'] < 0).any():
            skip_count += 1
            # skip this file if it contains negative 收盘价
            continue
        # Convert to string with zero padding
        temp_df['股票代码'] = temp_df['股票代码'].astype(str).str.zfill(6)
        key = temp_df.iat[1, 1]
        dfs[key] = temp_df
    print(f"Reading complete, total skip count in {total_count}, skipped {skip_count} files. ")
    return dfs

