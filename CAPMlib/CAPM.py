import global_resources as gr
import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt


# Getting the dict for all files within data_dir

work_dir = gr.ch_dir_to_repo()
data_dir = os.path.join(work_dir, 'Data', 'Stock Data')
# data_dir = os.path.join(work_dir, 'Data', 'TEST Stock Data for CAPM', 'Stocks')
index_dir = os.path.join(data_dir, '000001_daily_hfq.csv')

dfs = gr.get_df_dict(data_dir = data_dir)

# Data preprocess

# Iterating
R_f_path = os.path.join(work_dir, 'Data', 'TEST Stock Data for CAPM', 'R_f.xlsx')

def CAPM(index_file_or_key = '000001', processed_dfs_dict = dfs, R_f_path = R_f_path):
    """
    Calculate CAPM alpha and beta for each stock in processed_dfs_dict against a given market index.

    Parameters:
    - index_file_or_key: key in processed_dfs_dict identifying the market index
    - processed_dfs_dict: dict of DataFrames, each with columns ['Date', 'Stock ID', 'Pct_Change']
    - rf_path: path to an Excel file containing risk-free rates with columns ['Date', 'R_f']

    Returns:
    - pandas.DataFrame with columns ['Stock ID', 'alpha', 'beta']
    """
    # Validate inputs
    if processed_dfs_dict is None or index_file_or_key not in processed_dfs_dict:
        raise ValueError("Index key not found in processed_dfs_dict")
    
    
    # Prepare index series
    index_df = processed_dfs_dict[index_file_or_key].copy()
    index_df = gr.drop_and_change_head(index_df)
    index_df = index_df.drop(columns=['Stock ID'], errors='ignore')
    index_df = index_df.sort_values('Date').reset_index(drop=True)
    
#     keys = list(dfs.keys())
#     index_key = ''
    
#     if index_file_or_key in dfs:
#         index_key = index_file_or_key
#         keys.remove(index_key)
# # elif: index_file_or_key is a directory.........
#     else:
#         return None
    
#     index_df = dfs[index_key].copy()
#     index_df = gr.drop_and_change_head(index_df)
#     index_df.drop(['Stock ID'], axis = 1, inplace = True)
#     index_df = index_df.sort_values(by='Date')
#     index_df = index_df.reset_index(drop = True)
    
    # Prepare risk-free rate
    if R_f_path is None:
        raise ValueError("Path to risk-free rate file must be provided")
    R_f_df = gr.read_and_return_pd_df(R_f_path)
    R_f_df.columns = ['Date', 'R_f']
    R_f_df['Date'] = pd.to_datetime(R_f_df['Date'], format='%Y-%m-%d')
    R_f_df = R_f_df.sort_values(by='Date')
    R_f_df['R_f'] = gr.yearly_to_daily(R_f_df['R_f'])
    
    
    
    
    results = []
    for key, temp_df in processed_dfs_dict.items():
        if key == index_file_or_key:
            continue
        temp_df = temp_df.copy()
        temp_df = gr.drop_and_change_head(temp_df)
        temp_df.dropna(inplace = True, axis = 0)
        temp_df = temp_df.sort_values(by='Date').reset_index(drop = True)
        
        # Merge with market index and risk-free rate
        df = pd.merge(temp_df, index_df, on='Date', how='inner')
        df = pd.merge_asof(df, R_f_df, on='Date', direction='backward')
        
        # Compute excess returns
        df['excess_return_i'] = df['Pct_Change_x'] - df['R_f']
        df['excess_return_m'] = df['Pct_Change_y'] - df['R_f']
        
        # Fit CAPM regression
        X = sm.add_constant(df['excess_return_m'])
        capm_model = sm.OLS(df['excess_return_i'], X).fit()
        alpha = capm_model.params['const']
        beta = capm_model.params['excess_return_m']
        
        # Collect results
        results.append({'Stock ID': key, 'alpha': alpha, 'beta': beta})
        
    return pd.DataFrame(results)



def plot_random_capm(capm_df, n_stocks = 5, x_range = None, n_points = 100):
    """
    Plot CAPM regression lines for a random subset of stocks.

    Parameters
    ----------
    capm_df : pandas.DataFrame
        Must contain columns ['Stock ID', 'alpha', 'beta'].
    n_stocks : int, default 5
        Number of random stocks to plot.
    x_range : tuple (xmin, xmax), optional
        If provided, use this as the x-axis limits. Otherwise,
        determines symmetric range based on the max |beta| sampled.
    n_points : int, default 100
        How many points to use when drawing each line.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure (also shows via plt.show()).
    """
    # Randomly sample n_stocks rows
    sample_df = capm_df.sample(n = n_stocks, replace=False)

    # Determine x-axis limits if not given
    if x_range is None:
        max_beta = sample_df['beta'].abs().max()
        # choose a symmetric range; scale a bit beyond the largest slope
        limit = max_beta * 1.5
        x_min, x_max = -limit, limit
    else:
        x_min, x_max = x_range

    # Prepare x‑values
    x = np.linspace(x_min, x_max, n_points)

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))

    for _, row in sample_df.iterrows():
        alpha = row['alpha']
        beta  = row['beta']
        sid   = row['Stock ID']

        y = alpha + beta * x
        ax.plot(x, y, label=str(sid))

    # Add zero reference lines
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)

    # Labels and legend
    ax.set_xlabel('Market Excess Return')
    ax.set_ylabel('Stock Excess Return')
    ax.set_title(f'CAPM Lines for {n_stocks} Random Stocks')
    ax.legend(title='Stock ID', loc='best')

    ax.set_xlim(x_min, x_max)
    plt.tight_layout()
    plt.show()

    return fig
