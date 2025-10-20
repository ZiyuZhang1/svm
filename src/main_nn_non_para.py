import pandas as pd
import os
from features_reindex import get_feature, read_data, read_data_timecut
import pickle
# from model_nn_uniport import enriched_set, neg_bagging, calculate_jac_sim, eval_bagging
from model_nn_non_para import enriched_set, neg_bagging_early, neg_bagging_mid, neg_bagging_later, calculate_jac_sim, eval_bagging
from sklearn.preprocessing import MinMaxScaler
import sys
import torch
import numpy as np
from collections import defaultdict

def mask_mean(all_preds):
    arrays = np.stack([arr for arr, _ in all_preds])          # shape: (n, d)

    # Build mask matrix (True = masked / skip)
    mask = np.zeros_like(arrays, dtype=bool)
    for i, (_, m) in enumerate(all_preds):
        mask[i, m] = True

    # Invert mask: True where we keep
    keep = ~mask

    # Compute sum and count efficiently
    sum_arr = np.where(keep, arrays, 0).sum(axis=0)
    count_arr = keep.sum(axis=0)

    final_y_score = np.array(sum_arr / count_arr)
    return final_y_score

def one_fold_evaluate(disease, time, feature_list, df,y,train_idx,test_idx,methods,result_df,fold):
    train_pos_df = df.loc[train_idx]
    test_pos_df = df.loc[test_idx]
    neg_num = 5*len(train_pos_df)
    neg_df = df[y == 0]
    neg_df_add_test_pos = pd.concat([neg_df, test_pos_df])

    ######################### using precalculated kernels to train svm and evaluate, get weights for kernels
    print('train test split')

    # Work with DataFrames to maintain indices
    
    test_neg_df = neg_df
    test_df = pd.concat([test_pos_df, test_neg_df])
    test_index_loc = df.index.get_indexer(test_df.index)
    y_test = np.array([1] * len(test_pos_df) + [0] * len(test_neg_df))

    predcition_collection = dict()
    predcition_collection['true_label'] = y_test
    predcition_collection["test_genes"] = test_df.index
    predcition_collection["train_pos_genes"] = train_pos_df.index

    test_indices = test_df.index.values
    # enrich_train_genes = train_pos_df.index.values
    # enrich_train_set = enriched_set(enrich_train_genes,time)

    num_iterations = 20
    base_seed = 42
    seed_list = [base_seed + i for i in range(num_iterations)]

    args_list = [
        (neg_df_add_test_pos, neg_num, train_pos_df, df, y, feature_list, test_index_loc, seed)
        for seed in seed_list]

    if 'early_fusion' in methods:
        print('early fusion')

        bagging_y_scores = []
        for args in args_list:
            bagging_y_scores.append(neg_bagging_early(args))

        # Unzip the list of tuples into two lists
        all_preds, all_aucs = zip(*bagging_y_scores)  # all_preds is a tuple of arrays, all_aucs is a tuple of scalars

        final_y_score = mask_mean(all_preds)

        # # Compute mean predictions across all bags
        # final_y_score = np.mean(all_preds, axis=0)

        # Compute mean AUC
        mean_auc = np.mean(all_aucs)
        print('early fusion validation auc:',mean_auc)

        # enrich_test_genes = test_indices[np.argsort(final_y_score)[::-1]][:200]
        # enrich_feature_test = enriched_set(enrich_test_genes,time)
        # jac_sm = calculate_jac_sim(enrich_train_set,enrich_feature_test)

        ranked_predict_index, results = eval_bagging(final_y_score, y_test)
        # Add results to the result dataframe
        result_df.loc[len(result_df.index)] = ["random_negative",fold,'DL_early'+'-0-0-0', *results]
        predcition_collection['DL_early'] = final_y_score
    if 'mid_fusion' in methods:
        print('mid fusion')

        bagging_y_scores = []
        for args in args_list:
            bagging_y_scores.append(neg_bagging_mid(args))

        # Unzip the list of tuples into two lists
        all_preds, all_aucs = zip(*bagging_y_scores)  # all_preds is a tuple of arrays, all_aucs is a tuple of scalars

        final_y_score = mask_mean(all_preds)

        # # Compute mean predictions across all bags
        # final_y_score = np.mean(all_preds, axis=0)

        # Compute mean AUC
        mean_auc = np.mean(all_aucs)
        print('early fusion validation auc:',mean_auc)

        # enrich_test_genes = test_indices[np.argsort(final_y_score)[::-1]][:200]
        # enrich_feature_test = enriched_set(enrich_test_genes,time)
        # jac_sm = calculate_jac_sim(enrich_train_set,enrich_feature_test)

        ranked_predict_index, results = eval_bagging(final_y_score, y_test)
        # Add results to the result dataframe
        result_df.loc[len(result_df.index)] = ["random_negative",fold,'DL_mid'+'-0-0-0', *results]
        predcition_collection['DL_mid'] = final_y_score

    if 'later_fusion' in methods:
        print('later fusion')
        bagging_y_scores = []
        for args in args_list:
            bagging_y_scores.append(neg_bagging_later(args))

        feature_preds_collection = defaultdict(list)
        fused_preds_collection = []
        lf_mpl_preds_collection = []
        dict_list = []
        # mask_loc_bag = []
        # Collect predictions
        for feature_preds, fused_preds , auc_records, lf_mpl_preds , mask_loc in bagging_y_scores:
            dict_list.append(auc_records)
            for feature_name, preds in feature_preds.items():
                feature_preds_collection[feature_name].append([preds,mask_loc])
            fused_preds_collection.append([fused_preds,mask_loc])
            lf_mpl_preds_collection.append([lf_mpl_preds,mask_loc])
            # mask_loc_bag.append(mask_loc)

        aggregated = defaultdict(list)

        for d in dict_list:
            for key, value in d.items():
                aggregated[key].append(value)

        # Compute mean per key
        mean_dict = {key: np.mean(values) for key, values in aggregated.items()}

        # Average predictions per feature
        aggregated_feature_preds = {}
        for feature_name, preds_list in feature_preds_collection.items():
            final_y_score = mask_mean(preds_list)
            # final_y_score = np.mean(preds_list, axis=0)
            aggregated_feature_preds[feature_name] = final_y_score
            # enrich_test_genes = test_indices[np.argsort(final_y_score)[::-1]][:200]
            # enrich_feature_test = enriched_set(enrich_test_genes,time)
            # jac_sm = calculate_jac_sim(enrich_train_set,enrich_feature_test)

            ranked_predict_index, results = eval_bagging(final_y_score, y_test)
            # Add results to the result dataframe
            result_df.loc[len(result_df.index)] = ["random_negative",fold,'DL_'+str(feature_name)+'-0-0-0', *results]
            predcition_collection['DL_'+str(feature_name)] = final_y_score

        # Average fused predictions
        valid_preds = [p for p in fused_preds_collection if p is not None]
        if valid_preds:
            final_y_score = mask_mean(valid_preds)

            # final_y_score = np.mean(valid_preds, axis=0)

            # enrich_test_genes = test_indices[np.argsort(final_y_score)[::-1]][:200]
            # enrich_feature_test = enriched_set(enrich_test_genes,time)
            # jac_sm = calculate_jac_sim(enrich_train_set,enrich_feature_test)

            y_test = np.array([1] * len(test_pos_df) + [0] * len(test_neg_df))
            ranked_predict_index, results = eval_bagging(final_y_score, y_test)
            # Add results to the result dataframe
            result_df.loc[len(result_df.index)] = ["random_negative",fold,'DL_later_avg'+'-0-0-0', *results]
            predcition_collection['DL_later_avg'] = final_y_score

        valid_preds = [p for p in lf_mpl_preds_collection if p is not None]
        if valid_preds:
            final_y_score = mask_mean(valid_preds)

            # final_y_score = np.mean(valid_preds, axis=0)

            y_test = np.array([1] * len(test_pos_df) + [0] * len(test_neg_df))
            ranked_predict_index, results = eval_bagging(final_y_score, y_test)
            # Add results to the result dataframe
            result_df.loc[len(result_df.index)] = ["random_negative",fold,'DL_later_mlp'+'-0-0-0', *results]
            predcition_collection['DL_later_mlp'] = final_y_score
    return predcition_collection

def evaluate_disease(disease, time, feature_list, df, y, methods,time_spilt):
    result_df = pd.DataFrame(columns=['method',"fold","para", 'top_recall_25','top_recall_300','top_recall_10%', 'top_precision_10%', 'max_precision_10%','top_recall_30%', 'top_precision_30%', 'max_precision_30%','pm_0.5%','pm_1%','pm_5%','pm_10%','pm_15%','pm_20%','pm_25%','pm_30%','auroc',"rank_ratio",'bedroc_1','bedroc_5','bedroc_10','bedroc_30'])
    if time_spilt:
        test_idx = df[df['test']==1].index
        train_idx = df[y==1].index.difference(test_idx)
        df.drop(columns='test', inplace=True)
        predcition_collection = one_fold_evaluate(disease, time, feature_list, df,y,train_idx,test_idx,methods,result_df,1)
        return result_df, predcition_collection
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root = '/itf-fi-ml/shared/users/ziyuzh/svm'
    time_spilt = True
    # test_bug = True
    test_bug = False

    if test_bug:
        # feature_list = ['uniport_ppi_2017','uniport_exp','uniport_seq','uniport_esm']
        # feature_list = ['uniport_ppi_2017','ppi_2017_dw_80','uniport_exp','uniport_seq','uniport_esm']
        # feature_list = ['ppi_2019','bioconcept']
        # feature_list = ['ppi_2019_short','bioconcept_short']
        feature_list = ['uniport_ppi_2019','ppi_2019_dw_40','uniport_bio','uniport_seq','uniport_esm','diffusion_2019_2']
        out_path = os.path.join(root,'results/temp')
        out_path_pred = out_path+'_pred'
        time = 2019
    else:
        feature_list = sys.argv[1].split(',')
        out_path = os.path.join(root,sys.argv[2])
        out_path_pred = out_path+'_pred'
        time = int(sys.argv[3])

    os.makedirs(out_path, exist_ok=True)
    os.makedirs(out_path_pred, exist_ok=True)

    merged_df = None
    for feature in feature_list:
        feature_df = get_feature(root, feature)

        feature_cols = [col for col in feature_df.columns if col.startswith('feature')]
        if feature_cols:
            scaler = MinMaxScaler()
            feature_df[feature_cols] = scaler.fit_transform(feature_df[feature_cols])

        # Rename columns starting with 'feature'
        feature_df.rename(columns={
            col: f"{feature}_{col}" if col.startswith('feature') else col
            for col in feature_df.columns
        }, inplace=True)

        if merged_df is None:
            merged_df = feature_df
        else:
            merged_df = pd.merge(merged_df, feature_df, on='string_id', how='inner')
        del feature_df  # Free memory

    all_df = pd.read_csv('/itf-fi-ml/shared/users/ziyuzh/svm/data/disgent_2020/timecut/dga_time_uniport.csv')
    all_df = all_df[all_df['string_id'].isin(merged_df['string_id'])]

    methods = ['early_fusion','mid_fusion','later_fusion_avg','later_fusion']
    # methods = ['later_fusion']


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

    # feature_list = ['uniport_ppi_2019','ppi_2019_dw_40','diffusion_2019_2','string_id']
    # merged_df = merged_df[[c for c in merged_df.columns if any(c.startswith(f) for f in feature_list)]]
    print(feature_list, len(selected_diseases),len(merged_df))
    all_results = []
    for disease in selected_diseases:
        # disease = 'ICD10_N97'
        print(disease,len(all_df[all_df['disease_id']==disease]))
        if time_spilt:
            df, y = read_data_timecut(disease, all_df, merged_df,time)
        else:
            df, y = read_data(disease, all_df, merged_df,time)
        result_df, predcition_collection = evaluate_disease(disease, time, feature_list, df, y, methods,time_spilt)
        result_df.to_csv(os.path.join(out_path, f"{disease}.csv"),index = False)
        
        with open(out_path_pred+f'/{disease}_pred.pkl', 'wb') as f:
            pickle.dump(predcition_collection, f)        
        
        # Calculate mean metrics
        mean_df = result_df.groupby(['method'])[['top_recall_25','top_recall_300','top_recall_10%', 'top_precision_10%', 'max_precision_10%','top_recall_30%', 'top_precision_30%', 'max_precision_30%','pm_0.5%','pm_1%','pm_5%','pm_10%','pm_15%','pm_20%','pm_25%','pm_30%','auroc',"rank_ratio",'bedroc_1','bedroc_5','bedroc_10','bedroc_30']].mean().reset_index()
        # Add disease information
        mean_df['disease'] = disease
        # Append to all_results list
        all_results.append(mean_df)
        # break

    # Concatenate all results into a single DataFrame
    final_result = pd.concat(all_results, ignore_index=True)
    final_result.to_csv(os.path.join(out_path,'all_disease.csv'),index=False)

if __name__ == "__main__":
    main()