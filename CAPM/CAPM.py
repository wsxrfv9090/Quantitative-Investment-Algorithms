import pandas as pd
import global_resources as gr
import os
import statsmodels.api as sm
from numpy import nan as nan
from datetime import datetime

Current_Date_TAG_1 = datetime.now().strftime("%Y-%m-%d")

print(f"Current working directory: '{gr.default_dir}'.\n")
stock_data_directory = os.path.join('Data', 'Stock Data for CAPM', 'DATA_TEST')
chosen_security_file_name = 'K线导出_600873_日线数据.xlsx'
index_file_name = 'K线导出_000001_日线数据.xlsx'
R_f_file_name = 'R_f.xlsx'


def get_regression_line(stock_data_directory = stock_data_directory, security_file_name = chosen_security_file_name, index_file_name = index_file_name, Risk_free_file_name = R_f_file_name):
    
    # Concatenating paths.
    data_path = os.path.join(gr.default_dir, stock_data_directory)
    print(f"Choosing data path: '{data_path}'.\n")
    
    security_file_path = os.path.join(data_path, security_file_name)
    print(f"Choosing security file path: '{security_file_path}'.\n")
    
    index_file_path = os.path.join(data_path, index_file_name)
    print(f"Choosing market portfolio file path: '{index_file_path}'.\n")
    
    R_f_file_path = os.path.join(data_path, Risk_free_file_name)
    print(f"Choosing risk free rate file path: '{R_f_file_path}'")
    
    
    # Read Files as pd dataframe.
    i_df = gr.read_and_return_pd_df(security_file_path)
    m_df = gr.read_and_return_pd_df(index_file_path)
    R_f_df = gr.read_and_return_pd_df(R_f_file_path)
    
    
    # Preprocess useless datas and change the chinese into English, change some of the datatype into corresponding data type.
    for df in [i_df, m_df]:
        df.dropna(inplace = True)
        gr.change_head_to_ENG(df)
        gr.change_date_to_datetime(df)
        gr.change_numerical_data_to_float64(df)
        gr.change_secu_code_to_str(df)
        
    # Save important tags before Droping columns
    Stock_ID_TAG_2 = i_df.iat[0, 0]
    Stock_Name_TAG_3 = i_df.iat[0, 1]
    
    # Drop useless data for i_df and m_df
    i_df, m_df = [df[['DATE', 'CLOSING']] for df in [i_df, m_df]]
    
    ## Change the data into the same shape based on time.
    df = pd.merge(i_df, m_df, on='DATE', how='inner')
    df = df.rename(columns = {'CLOSING_x': 'Chosen Stock Closing Price', 'CLOSING_y': 'Market Portfolio Closing Price'})
    df.dropna(inplace = True)
    
    # Get percent changes
    df['i_pct_change'] = df['Chosen Stock Closing Price'].pct_change()
    df['m_pct_change'] = df['Market Portfolio Closing Price'].pct_change()
    df = df.dropna(subset=['i_pct_change', 'm_pct_change']).reset_index(drop=True)

    # Change the time range to 2022-01-01 to 2025-02-28 and Drop the useless column since now we have the percent change column.   
    Start_Date = '2022-01-01'
    End_Date = df.iat[-1, 0]
    End_Date = End_Date.strftime('%Y-%m-%d')
    Sample_Range_TAG_4 = Start_Date + " to " + End_Date
    
    # Dropping other data outside of the time range
    df = df.loc[(df['DATE'] > Start_Date) & (df['DATE'] <= End_Date)]
    
    to_drop = ['Chosen Stock Closing Price', 'Market Portfolio Closing Price']
    df.drop(to_drop, axis = 1, inplace = True)
    df = df.reset_index(drop=True)
    
    # Transform yearly R_f to daily R_f
    # def yearly_to_daily(R_f_y):
    #     return ((R_f_y + 1) ** (1 / 365)) - 1
    
    gr.change_date_to_datetime(R_f_df)
    R_f_df = R_f_df.sort_values(by='DATE')
    R_f_df['R_f'] = gr.yearly_to_daily(R_f_df['R_f'])
    
    # Merge df and R_f_df based on dates
    df = pd.merge_asof(df, R_f_df, on='DATE', direction='backward')
    
    # Get excess return for both the stock and the market
    df['excess_return_i'] = df['i_pct_change'] - df['R_f']
    df['excess_return_m'] = df['m_pct_change'] - df['R_f']
    
    # Fitting the model
    X = sm.add_constant(df['excess_return_m'])
    capm_model = sm.OLS(df['excess_return_i'], X).fit()
    alpha = capm_model.params['const']
    beta = capm_model.params['excess_return_m']
    Alpha_TAG_5 = alpha
    Beta_TAG_6 = beta
    print("Alpha:", alpha)
    print("Beta:", beta)
    
    result = {
        'Current_Date': Current_Date_TAG_1,
        'Stock_ID':     Stock_ID_TAG_2,
        'Stock_Name':   Stock_Name_TAG_3,
        'Sample_Range': Sample_Range_TAG_4,
        'Alpha':        Alpha_TAG_5,
        'Beta':         Beta_TAG_6
    }
    return pd.DataFrame([result])
    


print(get_regression_line())    