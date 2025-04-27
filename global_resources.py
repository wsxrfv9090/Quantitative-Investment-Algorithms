import os
import pandas as pd
import torch
import numpy as np

# Set the working directory to a known absolute path
default_dir = 'd:\Important Files\Repositories\Quantitative-Investment-Algorithms'
os.chdir(default_dir)

global_paths = {
    'CAPM': os.path.join(default_dir, 'CAPM'),
    'K means cluster': os.path.join(default_dir, 'K means cluster'),
    'Data': os.path.join(default_dir, 'Data'),
    'OVO': os.path.join(default_dir, 'OVO'),
    'SVM': os.path.join(default_dir, 'SVM'),
    'Output': os.path.join(default_dir, 'Output')
}

def set_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

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
    
# Change the data type of the security code to string
def change_secu_code_to_str(pd_df):
    pd_df['SECU_CODE'] = pd_df['SECU_CODE'].astype('str')

# Change the data type of the date to datetime
def change_date_to_datetime(pd_df):
    pd_df['DATE'] = pd.to_datetime(pd_df['DATE'])

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
    na_rows = df[df.isna().any(axis=1)]
    print(na_rows)
    return na_rows


# This function can only be used 
def get_df_dict(data_dir = None):
    if data_dir == None:
        print('No data_dir parameter passed, please put data directory into the parameters.')
        return None
    dfs = {}

    for fname in os.listdir(data_dir):
        full_path = os.path.join(data_dir, fname)
        temp_df = read_and_return_pd_df(full_path)
        key = temp_df.iat[1, 0]
        dfs[key] = temp_df
    return dfs