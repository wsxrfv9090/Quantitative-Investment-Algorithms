import global_resources as gr
import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt


# Getting the dict for all files within data_dir

work_dir = gr.ch_dir_to_repo()

target_dir = os.path.join(gr.global_paths['Data'], 'Stock Data')
# target_dir = os.path.join(gr.global_paths['Data'], 'TEST Stock Data for CAPM', 'Stocks')

default_market_index_path = os.path.join(gr.global_paths['Data'], 'market_index.csv') 
default_index_df = gr.read_and_return_pd_df(default_market_index_path)

R_f_name = 'R_f.xlsx'
R_f_path = os.path.join(gr.global_paths['Data'], R_f_name)

dfs = gr.get_df_dict(data_dir = target_dir)

# Preprocess default index df

def label_dfs(
    df: pd.DataFrame,
    label_column: str,
):
    label_dfs = {}
    for lbl in sorted(df[label_column].unique().tolist()):
        temp_df = df[df[label_column] == lbl].copy()
        label_dfs[lbl] = temp_df
    return label_dfs

def create_nested_dfs(label_dfs, time_series_dfs, id_column='ID'):
    """
    Create a three-level nested dictionary structure:
    - First level: label as key, dictionary as value
    - Second level: stock ID as key, time series data as value
    - Third level: the actual time series data for each stock
    
    Parameters:
    -----------
    label_dfs : dict
        Dictionary with labels as keys and DataFrames containing stock IDs as values
    time_series_dfs : dict
        Dictionary with stock IDs as keys and time series DataFrames as values
    id_column : str, default='ID'
        Column name in label_dfs that contains the stock IDs
        
    Returns:
    --------
    dict
        Three-level nested dictionary structure
    """
    nested_dfs = {}
    missing_ids = []
    
    # Iterate through each label and its corresponding DataFrame
    for label, df in label_dfs.items():
        if id_column not in df.columns:
            print(f"Warning: '{id_column}' column not found in DataFrame for label {label}. Skipping.")
            continue
            
        # Initialize the second level dictionary for this label
        nested_dfs[label] = {}
        
        # Get list of stock IDs for this label
        stock_ids = df[id_column].tolist()
        
        # Add time series data for each stock ID
        for stock_id in stock_ids:
            if stock_id in time_series_dfs:
                nested_dfs[label][stock_id] = time_series_dfs[stock_id]
            else:
                missing_ids.append((label, stock_id))
    
    # Report missing IDs
    if missing_ids:
        print(f"Warning: {len(missing_ids)} stock IDs were not found in time_series_dfs:")
        for label, stock_id in missing_ids[:10]:  # Show only first 10 to avoid clutter
            print(f"  - Label: {label}, Stock ID: {stock_id}")
        if len(missing_ids) > 10:
            print(f"  - ... and {len(missing_ids) - 10} more")
    
    # Report stock IDs in time_series_dfs but not in any label_dfs
    all_label_ids = set()
    for df in label_dfs.values():
        if id_column in df.columns:
            all_label_ids.update(df[id_column].tolist())
    
    unused_ids = set(time_series_dfs.keys()) - all_label_ids
    if unused_ids:
        print(f"Note: {len(unused_ids)} stock IDs in time_series_dfs are not associated with any label")
        if len(unused_ids) <= 10:
            print(f"Unused IDs: {list(unused_ids)}")
        else:
            print(f"First 10 unused IDs: {list(unused_ids)[:10]}")
    
    return nested_dfs



def CAPM(
    index_df: pd.DataFrame, 
    R_f: bool = True,
    processed_dfs_dict: dict = dfs, 
    ) -> pd.DataFrame:
    
    # Prepare index series
    
    cols = ['Date', 'Pct_Change']
    index_df.columns = cols
        
    # Prepare risk-free rate
    if R_f:
        R_f_df = gr.read_and_return_pd_df(R_f_path)
        R_f_df.columns = ['Date', 'R_f']
        R_f_df['Date'] = pd.to_datetime(R_f_df['Date'], format='%Y-%m-%d')
        R_f_df = R_f_df.sort_values(by='Date')
        R_f_df['R_f'] = gr.yearly_to_daily(R_f_df['R_f'])
    
    
    results = []
    for key, temp_df in processed_dfs_dict.items():
        temp_df = temp_df.copy()
        temp_df = gr.drop_and_change_head(temp_df)
        temp_df.dropna(inplace = True, axis = 0)
        temp_df['Date'] = pd.to_datetime(temp_df['Date'], format='%Y-%m-%d')
        temp_df = temp_df.sort_values(by = 'Date').reset_index(drop = True)
        
        # Merge with market index and risk-free rate
        df = pd.merge(temp_df, index_df, on='Date', how='inner')
        if R_f:
            df = pd.merge_asof(df, R_f_df, on='Date', direction='backward')
        
        # Compute excess returns
        if R_f:
            df['excess_return_i'] = df['Pct_Change_x'] - df['R_f']
            df['excess_return_m'] = df['Pct_Change_y'] - df['R_f']
        else:
            df['excess_return_i'] = df['Pct_Change_x']
            df['excess_return_m'] = df['Pct_Change_y']
        
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
        ax.plot(x, y, label = str(sid))

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
