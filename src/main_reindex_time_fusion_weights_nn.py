import pandas as pd
import os
from features_reindex import get_feature, read_data, read_data_timecut
from model_nn import enriched_set, neg_bagging, calculate_jac_sim, eval_bagging
import sys
import torch.multiprocessing as mp
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler

def one_fold_evaluate(disease, time, feature_list, df,y,train_idx,test_idx,methods,result_df,fold):
    train_pos_df = df.loc[train_idx]
    test_pos_df = df.loc[test_idx]
    neg_num = 5*len(train_pos_df)

    if 'random_negative' in methods:
        ######################### using precalculated kernels to train svm and evaluate, get weights for kernels
        print('evaluation')

        # Work with DataFrames to maintain indices
        neg_df = df[y == 0]
        test_neg_df = neg_df
        test_df = pd.concat([test_pos_df, test_neg_df])
        test_index_loc = df.index.get_indexer(test_df.index)


        test_indices = test_df.index.values
        enrich_train_genes = train_pos_df.index.values
        enrich_train_set = enriched_set(enrich_train_genes,time)

        num_processes = 2
        base_seed = 42
        seed_list = [base_seed + i for i in range(num_processes)]

        X_all = []
        for feature_name in feature_list:
            select_columns = [col for col in df.columns if col.startswith(feature_name)]
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(df[select_columns].values)
            X_all.append(X_scaled)

        args_list = [
            (neg_df, neg_num, train_pos_df, df, y, X_all, test_index_loc, seed)
            for seed in seed_list]

        # Step 2: Use Pool to parallelize
        with mp.Pool(processes=num_processes) as pool:
            bagging_y_scores = pool.map(neg_bagging, args_list)

        final_y_score = torch.stack(bagging_y_scores).mean(dim=0).cpu().numpy()

        enrich_test_genes = test_indices[np.argsort(final_y_score)[::-1]][:200]
        enrich_feature_test = enriched_set(enrich_test_genes,time)
        jac_sm = calculate_jac_sim(enrich_train_set,enrich_feature_test)

        y_test = np.array([1] * len(test_pos_df) + [0] * len(test_neg_df))
        ranked_predict_index, results = eval_bagging(final_y_score, y_test)
        # Add results to the result dataframe
        result_df.loc[len(result_df.index)] = ["random_negative",fold,'DL-'+str(round(jac_sm, 3)), *results]

def evaluate_disease(disease, time, feature_list, df, y, methods,time_spilt):
    result_df = pd.DataFrame(columns=['method',"fold","para", 'top_recall_25','top_recall_300','top_recall_10%', 'top_precision_10%', 'max_precision_10%','top_recall_30%', 'top_precision_30%', 'max_precision_30%','pm_0.5%','pm_1%','pm_5%','pm_10%','pm_15%','pm_20%','pm_25%','pm_30%','auroc',"rank_ratio",'bedroc_1','bedroc_5','bedroc_10','bedroc_30'])
    
    if time_spilt:
        test_idx = df[df['test']==1].index
        train_idx = df[y==1].index.difference(test_idx)
        df.drop(columns='test', inplace=True)
        one_fold_evaluate(disease, time, feature_list, df,y,train_idx,test_idx,methods,result_df,1)
        return result_df

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root = '/itf-fi-ml/shared/users/ziyuzh/svm'

    time_spilt = True
    # test_bug = True
    test_bug = False

    if test_bug:
        # feature_list = ['uniport_ppi_2017','uniport_exp','uniport_seq','uniport_esm']
        # feature_list = ['ppi_2019','bioconcept']
        feature_list = ['ppi_2019','bioconcept','uniport','esm2']
        out_path = os.path.join(root,'results/temp')
        time = 2019
    else:
        feature_list = sys.argv[1].split(',')
        out_path = os.path.join(root,sys.argv[2])
        time = int(sys.argv[3])

    if not os.path.exists(out_path):
        os.mkdir(out_path)

    merged_df = None
    for feature in feature_list:
        feature_df = get_feature(root, feature)
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

    all_df = pd.read_csv('/itf-fi-ml/shared/users/ziyuzh/svm/data/disgent_2020/timecut/disgent_with_time.csv')
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
    print(feature_list, len(selected_diseases),len(merged_df))
    all_results = []
    for disease in selected_diseases:
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

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()