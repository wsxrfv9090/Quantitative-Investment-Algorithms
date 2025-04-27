import global_resources as gr
import os
import pandas as pd
import numpy as np

# Getting the dict for all files within data_dir

work_dir = gr.ch_dir_to_repo()
data_dir = os.path.join(work_dir, 'Data', 'Stock Data for CAPM', 'Stocks')

print(data_dir)


dfs = gr.get_df_dict(data_dir = data_dir)

# Data preprocess
keys = list(dfs.keys())

for key in keys:
    temp_df = dfs[key]
    gr.change_head_to_ENG(temp_df)
    gr.change_secu_code_to_str(temp_df)
    pd.set_option('future.no_silent_downcasting', True)
    temp_df.replace('--', np.nan, inplace = True)
    # temp_df.infer_objects(copy = False)
    temp_df.dropna(inplace = True, axis = 0)
    
    to_drop = temp_df.columns[[1, -4, -3, -2, -1]]
    temp_df.drop(to_drop, axis = 1, inplace = True)

print(dfs.items())

