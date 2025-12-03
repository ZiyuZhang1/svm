import pandas as pd
import os
from features_reindex import get_feature, read_data, read_data_timecut
from model_diffusion import compute_kernels
import pickle
import sys
import multiprocessing as mp
from sklearn.preprocessing import MinMaxScaler
import numpy as np


root = '/itf-fi-ml/shared/users/ziyuzh/svm'

# time_spilt = True
# feature = 'ppi_'+str(time)

time_spilt = True
test_bug = True


# dga = 'opentarget'
dga = 'disgenet'

out_path = os.path.join(root,'results/temp')
out_path_pred = out_path+'_pred/pred.pkl'
time = 2019

os.makedirs(out_path, exist_ok=True)
os.makedirs(out_path_pred, exist_ok=True)

merged_df = None

if time == 2017:
    time_feature_list = ['uniport_ppi_2017','ppi_2017_dw_80','uniport_exp','uniport_seq','uniport_esm']
elif time == 2019:
    time_feature_list = ['uniport_ppi_2019','ppi_2019_dw_40','uniport_bio','uniport_seq','uniport_esm','diffusion_2019_2']

for feature in time_feature_list:
    feature_df = get_feature(root, feature)

    if 'diffusion' in feature:
        pass
    else:
        feature_cols = [col for col in feature_df.columns if col.startswith('feature')]
        if feature_cols:
            scaler = MinMaxScaler()
            feature_df[feature_cols] = scaler.fit_transform(feature_df[feature_cols])

    # Rename columns starting with 'feature'
    feature_df.rename(columns={
        col: f"{feature}_{col}" if col.startswith('feature') else col
        for col in feature_df.columns
    }, inplace=True)

    # Merge iteratively to avoid keeping all DataFrames
    if merged_df is None:
        merged_df = feature_df
    else:
        merged_df = pd.merge(merged_df, feature_df, on='string_id', how='inner')
    del feature_df  # Free memory

# feature_list = ['string_id','uniport_ppi_2019','ppi_2019_dw_40','uniport_bio','uniport_seq','uniport_esm','diffusion_2019_2']
# feature_list = ['string_id','uniport_ppi_2019','ppi_2019_dw_40','diffusion_2019_2']
feature_list = ['uniport_ppi_2019','ppi_2019_dw_40','diffusion_2019_2']

selected_merged_df = merged_df[[col for col in merged_df.columns if any(item in col for item in feature_list)]]

# selected_merged_df.columns = [
#     'string_id' if col == 'string_id' else f'feature_{i}'
#     for i, col in enumerate(selected_merged_df.columns)
# ]

# selected_merged_df.to_csv('/itf-fi-ml/shared/users/ziyuzh/svm/data/pre_processed_features/df_early_ppi.csv',index=False)
# selected_merged_df = selected_merged_df.copy()
# selected_merged_df.drop(columns=['string_id'], inplace=True)

X_concat = selected_merged_df.values


kernel_path_dict = dict()
# early_dir = '/itf-fi-ml/shared/users/ziyuzh/svm/results/df_early_kernel'

early_dir = '/itf-fi-ml/shared/users/ziyuzh/svm/results/df_early_ppi_kernel'

os.makedirs(early_dir, exist_ok=True)
feature_names, K_path = compute_kernels(X_concat, 'df_early_ppi', early_dir, True)
kernel_path_dict[feature_names] = K_path

save_path_dict = early_dir+'/kernel_dict.pkl'
with open(save_path_dict, 'wb') as f:
    pickle.dump(kernel_path_dict, f)