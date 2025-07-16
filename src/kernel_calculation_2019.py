import pandas as pd
import os
from features_reindex import get_feature, read_data, read_data_timecut
# from model_reindex_fusion_weights_uniport_cv_filter import evaluate_disease
from model_cv import compute_kernels, select_gamma_ratio, neg_bagging, eval_bagging
import sys
import multiprocessing as mp
from sklearn.preprocessing import MinMaxScaler
import pickle
from multiprocessing import Pool
import numpy as np


root = '/itf-fi-ml/shared/users/ziyuzh/svm'

# time_spilt = True
# feature = 'ppi_'+str(time)

time_spilt = True
test_bug = True
# test_bug = False

if test_bug:
    # feature_list = ['uniport_ppi_2019','uniport_bio','uniport_seq','uniport_esm']
    # feature_list = ['ppi_2019','bioconcept']
    feature_list = ['ppi_2019_dw_10','ppi_2019_dw_40','ppi_2019_dw_80','uniport_bio','uniport_seq','uniport_esm']
    out_path = os.path.join(root,'results/ppi_2019_dw_test')
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    time = 2019

merged_df = None
for feature in feature_list:
    feature_df = get_feature(root, feature)
    # Rename columns starting with 'feature'
    feature_df.rename(columns={
        col: f"{feature}_{col}" if col.startswith('feature') else col
        for col in feature_df.columns
    }, inplace=True)

    feature_cols = [col for col in feature_df.columns if col.startswith('feature')]
    if feature_cols:
        scaler = MinMaxScaler()
        feature_df[feature_cols] = scaler.fit_transform(feature_df[feature_cols])

    # Merge iteratively to avoid keeping all DataFrames
    if merged_df is None:
        merged_df = feature_df
    else:
        merged_df = pd.merge(merged_df, feature_df, on='string_id', how='inner')
    del feature_df  # Free memory

all_df = pd.read_csv('/itf-fi-ml/shared/users/ziyuzh/svm/data/disgent_2020/timecut/dga_time_uniport.csv')
all_df = all_df[all_df['string_id'].isin(merged_df['string_id'])]

methods = ['random_negative']

if time_spilt:
    selected_diseases = []
    for disease_id in all_df['disease_id'].unique():
        sub_df = all_df[all_df['disease_id']==disease_id]
        if len(sub_df) < 15:
            continue
        else:
            # print(type(time),type(sub_df['first_pub_year'].max()))
            if sub_df['first_pub_year'].max() > time and sub_df['first_pub_year'].min() <= time and len(sub_df[sub_df['first_pub_year']<time]) >=5:
                selected_diseases.append(disease_id)
else:
    selected_diseases = (
        all_df.groupby('disease_id')
        .filter(lambda x: (len(x) > 15))
        ['disease_id']
        .unique()
        .tolist())
print(feature_list, len(selected_diseases),len(merged_df))
all_results = []

def one_fold_evaluate(disease, time, feature_list, df,y,train_idx,test_idx,methods,result_df,fold):
    train_pos_df = df.loc[train_idx]
    test_pos_df = df.loc[test_idx]
    neg_num = 5*len(train_pos_df)
    neg_df = df[y == 0]

    if 'random_negative' in methods:

        kernel_dir_path = os.path.join('/itf-fi-ml/shared/users/ziyuzh/svm/results/uni_kernel_cv_scaled',str(time))
        os.makedirs(kernel_dir_path, exist_ok=True)
        kernel_pkl_path = os.path.join(kernel_dir_path,'path_save.pkl')

        if os.path.isfile(kernel_pkl_path):
            print('kernels existing')
            with open(kernel_pkl_path, 'rb') as f:
                kernels_all_dict = pickle.load(f)

        add_feature_list = set(feature_list) - set(kernels_all_dict.keys())
        if not add_feature_list:
            pass
        else:
            add_feature_list = list(add_feature_list)
        ####### calculate full kernels for each feature and their logm
            print('calculating kernels...')
            X_all = []
            
            for feature_name in add_feature_list:
                select_columns = [col for col in df.columns if col.startswith(feature_name)]
                X_all.append(df[select_columns].values)

            args_list = list(zip(X_all, add_feature_list, [kernel_dir_path] * len(X_all)))
            with Pool(min(len(add_feature_list), os.cpu_count(), 4)) as pool:
                # each tuple (X_feature, feature_id) is unpacked by starmap
                kernel_results = pool.starmap(
                    compute_kernels,
                    args_list)
            del X_all

            for fname, K_s_path in kernel_results:
                kernels_all_dict[fname] = K_s_path
                
            with open(kernel_pkl_path, 'wb') as f:
                pickle.dump(kernels_all_dict, f)
      ############################## cv get best gamma
        args_list = [(neg_df, neg_num, train_pos_df, df, kernels_all_dict[fname], fname)
            for fname in feature_list]

        with Pool(processes=len(feature_list)) as pool:
            best_ratios = pool.map(select_gamma_ratio, args_list)

        best_ratios_dict = dict()
        agg_feature = []
        for fname, best_params, best_score in best_ratios:
            print(fname, best_params, best_score)
            best_ratios_dict[fname] = best_params
            if best_score > 0.70:
                agg_feature.append(fname)
        print('collect valid feature: ', agg_feature)
      ######################### using precalculated kernels to train svm and evaluate, get weights for kernels
        print('evaluation')

        test_neg_df = neg_df
        test_df = pd.concat([test_pos_df, test_neg_df])
        test_index_loc = df.index.get_indexer(test_df.index)
        y_test = np.array([1] * len(test_pos_df) + [0] * len(test_neg_df))

        num_processes = 15
        base_seed = 42
        seed_list = [base_seed + i for i in range(num_processes)]

        for feature_name in feature_list:
            gamma = best_ratios_dict[feature_name]['gamma_ratio']
            X_path = kernels_all_dict[feature_name][gamma][0]
            C_num = best_ratios_dict[feature_name]['C_num']

            args_list = [
                (neg_df, neg_num, train_pos_df, df, X_path, C_num, test_index_loc, seed)
                for seed in seed_list]

            # Step 2: Use Pool to parallelize
            with Pool(processes=num_processes) as pool:
                bagging_y_scores = pool.map(neg_bagging, args_list)

            final_y_score = np.mean(bagging_y_scores, axis=0)

            ranked_predict_index, results = eval_bagging(final_y_score, y_test)            # Add results to the result dataframe
            result_df.loc[len(result_df.index)] = ["random_negative",fold,feature_name+'-', *results]

def evaluate_disease(disease, time, feature_list, df, y, methods,time_spilt):
    result_df = pd.DataFrame(columns=['method',"fold","para", 'top_recall_25','top_recall_300','top_recall_10%', 'top_precision_10%', 'max_precision_10%','top_recall_30%', 'top_precision_30%', 'max_precision_30%','pm_0.5%','pm_1%','pm_5%','pm_10%','pm_15%','pm_20%','pm_25%','pm_30%','auroc',"rank_ratio",'bedroc_1','bedroc_5','bedroc_10','bedroc_30'])
    
    if time_spilt:
        test_idx = df[df['test']==1].index
        train_idx = df[y==1].index.difference(test_idx)
        df.drop(columns='test', inplace=True)
        one_fold_evaluate(disease, time, feature_list, df,y,train_idx,test_idx,methods,result_df,1)
        return result_df

for disease in selected_diseases:
    # disease = 'ICD10_M34'
    print(disease,len(all_df[all_df['disease_id']==disease]))
    if time_spilt:
        df, y = read_data_timecut(disease, all_df, merged_df,time)
    else:
        df, y = read_data(disease, all_df, merged_df,time)
    result_df = evaluate_disease(disease, time, feature_list, df, y, methods,time_spilt)
    result_df.to_csv(os.path.join(out_path, f"{disease}.csv"),index = False)
    # Calculate mean metrics
    mean_df = result_df.groupby(['method'])[['top_recall_25','top_recall_300','top_recall_10%', 'top_precision_10%', 'max_precision_10%','top_recall_30%', 'top_precision_30%', 'max_precision_30%','pm_0.5%','pm_1%','pm_5%','pm_10%','pm_15%','pm_20%','pm_25%','pm_30%','auroc',"rank_ratio",'bedroc_1','bedroc_5','bedroc_10','bedroc_30']].mean().reset_index()
    # Add disease information
    mean_df['disease'] = disease
    # Append to all_results list
    all_results.append(mean_df)

# Concatenate all results into a single DataFrame
final_result = pd.concat(all_results, ignore_index=True)
final_result.to_csv(os.path.join(out_path,'all_disease.csv'),index=False)