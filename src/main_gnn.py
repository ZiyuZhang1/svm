import os
import pickle
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.utils import add_self_loops, to_undirected, coalesce
from features_reindex import get_feature, read_data_timecut
from model_gnn import neg_bagging_gcn, neg_bagging_sage, eval_bagging
from torch_sparse import SparseTensor

METRIC_COLUMNS = [
    'top_recall_25',
    'top_recall_300',
    'top_recall_10%',
    'top_precision_10%',
    'max_precision_10%',
    'top_recall_30%',
    'top_precision_30%',
    'max_precision_30%',
    'pm_0.5%',
    'pm_1%',
    'pm_5%',
    'pm_10%',
    'pm_15%',
    'pm_20%',
    'pm_25%',
    'pm_30%',
    'auroc',
    'rank_ratio',
    'bedroc_1',
    'bedroc_5',
    'bedroc_10',
    'bedroc_30',
]
RESULT_COLUMNS = ['method', 'fold', 'para'] + METRIC_COLUMNS


def mask_mean(all_preds):
    if not all_preds:
        raise ValueError('Expected at least one set of predictions for aggregation.')

    arrays = np.stack([arr for arr, _ in all_preds])
    mask = np.zeros_like(arrays, dtype=bool)

    for i, (_, masked_positions) in enumerate(all_preds):
        if not masked_positions:
            continue
        mask[i, masked_positions] = True

    keep = ~mask
    sum_arr = np.where(keep, arrays, 0).sum(axis=0)
    count_arr = keep.sum(axis=0)
    return np.divide(sum_arr, count_arr, out=np.zeros_like(sum_arr, dtype=float), where=count_arr != 0)


def build_edge_index(node_ids, edge_path):
    edge_df = pd.read_csv(edge_path, usecols=['p1', 'p2'])
    id_to_idx = {sid: idx for idx, sid in enumerate(node_ids)}

    edge_df['src'] = edge_df['p1'].map(id_to_idx)
    edge_df['dst'] = edge_df['p2'].map(id_to_idx)
    edge_df = edge_df.dropna(subset=['src', 'dst'])

    if edge_df.empty:
        raise ValueError('No edges remain after filtering by available string_ids.')

    edge_array = edge_df[['src', 'dst']].to_numpy(dtype=np.int64).T
    print(f"Loaded {edge_df.shape[0]:,} edges after filtering.")
    return torch.tensor(edge_array, dtype=torch.long)


def evaluate_disease(disease, time, feature_list, df, y, edge_index, methods, time_split):
    df_with_test = df.copy()
    test_pos_idx = df_with_test[df_with_test['test'] == 1].index
    train_pos_idx = df_with_test[y == 1].index.difference(test_pos_idx)

    if len(train_pos_idx) == 0:
        raise ValueError(f'No training positives remain for disease {disease} at time {time}.')

    neg_idx = df_with_test.index[y == 0]
    df_clean = df_with_test.drop(columns='test')

    neg_candidates = np.concatenate([neg_idx.to_numpy(), test_pos_idx.to_numpy()])
    y_test = np.concatenate([
        np.ones(len(test_pos_idx), dtype=np.int64),
        np.zeros(len(neg_idx), dtype=np.int64),
    ])
    test_index = np.concatenate([test_pos_idx.to_numpy(), neg_idx.to_numpy()])

    predcition_collection = {
        'true_label': y_test,
        'test_genes': test_index,
        'train_pos_genes': train_pos_idx.to_numpy(),
    }

    num_iterations = 15
    base_seed = 42
    seed_list = [base_seed + i for i in range(num_iterations)]

    result_df = pd.DataFrame(columns=RESULT_COLUMNS)
    for feature_name in feature_list:
        select_columns = [col for col in df_clean.columns if col.startswith(feature_name)]
        df_features = df_clean[select_columns].copy()
        
        args_list = [
        (
            neg_candidates,
            5 * len(train_pos_idx),
            train_pos_idx.to_numpy(),
            df_features,
            y,
            edge_index,
            feature_name,
            test_index,
            seed,
        )
        for seed in seed_list]

        if 'gcn' in methods:
            print('GCN model')
            bagging_y_scores = [neg_bagging_gcn(args) for args in args_list]

            all_preds, all_aucs = zip(*bagging_y_scores)
            final_y_score = mask_mean(all_preds)
            mean_auc = float(np.mean(all_aucs))
            print(f'gcn validation auc: {mean_auc:.4f}')

            ranked_predict_index, results = eval_bagging(final_y_score, y_test)
            result_df.loc[len(result_df.index)] = ['gcn', 1, f"{feature_name}-0-0-0", *results]
            predcition_collection['gcn_'+feature_name] = final_y_score
        # if 'gat' in methods:
        #     bagging_y_scores = [neg_bagging_gat(args) for args in args_list]

        #     all_preds, all_aucs = zip(*bagging_y_scores)
        #     final_y_score = mask_mean(all_preds)
        #     mean_auc = float(np.mean(all_aucs))
        #     print(f'gat validation auc: {mean_auc:.4f}')

        #     ranked_predict_index, results = eval_bagging(final_y_score, y_test)
        #     result_df.loc[len(result_df.index)] = ['random_negative', 1, 'gat-0-0-0', *results]
        #     predcition_collection['gat'] = final_y_score
        if 'sage' in methods:
            print('graphsage model')
            bagging_y_scores = [neg_bagging_sage(args) for args in args_list]

            all_preds, all_aucs = zip(*bagging_y_scores)
            final_y_score = mask_mean(all_preds)
            mean_auc = float(np.mean(all_aucs))
            print(f'gat validation auc: {mean_auc:.4f}')

            ranked_predict_index, results = eval_bagging(final_y_score, y_test)
            result_df.loc[len(result_df.index)] = ['sage', 1, f"{feature_name}-0-0-0", *results]
            predcition_collection['sage_'+feature_name] = final_y_score

    return result_df, predcition_collection


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Running GNN pipeline on device: {device}')

    root = '/itf-fi-ml/shared/users/ziyuzh/svm'
    time_split = True
    test_bug = True
    # test_bug = True

    if test_bug:
        feature_list = ['uniport_ppi_2019',
        'ppi_2019_dw_40',
        'uniport_bio',
        'uniport_seq',
        'uniport_esm',
        'diffusion_2019_pca']
        # feature_list = ['uniport_ppi_2019']
        out_path = os.path.join(root, 'results/2019_gcn_deoversmooth_less_smooth')
        # out_path = os.path.join(root, 'results/2019_gnn_improved_sage')
        out_path_pred = out_path + '_pred'
        time = 2019
    else:
        if len(sys.argv) < 4:
            raise ValueError('Expected arguments: <comma_separated_features> <output_dir> <time_cutoff>')
        feature_list = sys.argv[1].split(',')
        out_path = os.path.join(root, sys.argv[2])
        out_path_pred = out_path + '_pred'
        time = int(sys.argv[3])

    os.makedirs(out_path, exist_ok=True)
    os.makedirs(out_path_pred, exist_ok=True)

    merged_df = None
    time_feature_list = [
        'uniport_ppi_2019',
        'ppi_2019_dw_40',
        'uniport_bio',
        'uniport_seq',
        'uniport_esm',
        'diffusion_2019_pca',
    ]

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

        if merged_df is None:
            merged_df = feature_df
        else:
            merged_df = pd.merge(merged_df, feature_df, on='string_id', how='inner')

    if merged_df is None or merged_df.empty:
        raise ValueError('Merged feature dataframe is empty. Check feature sources.')

    name_list = feature_list + ['string_id']
    merged_df = merged_df[[col for col in merged_df.columns if any(item in col for item in name_list)]]
    merged_df = merged_df.drop_duplicates(subset='string_id').reset_index(drop=True)
    merged_df = merged_df.fillna(0.0)

    all_df = pd.read_csv(os.path.join(root, 'data/disgent_2020/timecut/dga_time_uniport.csv'))
    all_df = all_df[all_df['string_id'].isin(merged_df['string_id'])]

    edge_path = os.path.join(root, 'data/stringdb/edge_2019.csv')
    edge_index = build_edge_index(merged_df['string_id'].tolist(), edge_path)

    num_nodes = len(merged_df)
    edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    edge_index = coalesce(edge_index, num_nodes=num_nodes)

    row, col = edge_index
    adj_t = SparseTensor(row=col, col=row, sparse_sizes=(num_nodes, num_nodes))
    del edge_index

    selected_diseases = []
    if time_split:
        for disease_id in all_df['disease_id'].unique():
            sub_df = all_df[all_df['disease_id'] == disease_id]
            if len(sub_df) < 15:
                continue
            if (
                sub_df['first_pub_year'].max() > time
                and sub_df['first_pub_year'].min() <= time
                and len(sub_df[sub_df['first_pub_year'] < time]) >= 5
            ):
                selected_diseases.append(disease_id)

    print(feature_list, len(selected_diseases), len(merged_df))

    all_results = []
    # methods = ['gcn']
    # methods = ['gcn','sage']
    methods = ['gcn']


    for disease in selected_diseases:
        # disease = 'ICD10_D83'
        print(disease, len(all_df[all_df['disease_id'] == disease]))
        df, y = read_data_timecut(disease, all_df, merged_df, time)
        result_df, predcition_collection = evaluate_disease(
            disease,
            time,
            feature_list,
            df,
            y,
            adj_t,
            methods,
            time_split,
        )

        result_df.to_csv(os.path.join(out_path, f'{disease}.csv'), index=False)
        with open(os.path.join(out_path_pred, f'{disease}_pred.pkl'), 'wb') as f:
            pickle.dump(predcition_collection, f)

        mean_df = result_df.groupby(['method'])[METRIC_COLUMNS].mean().reset_index()
        mean_df['disease'] = disease
        all_results.append(mean_df)
        # break
    if all_results:
        final_result = pd.concat(all_results, ignore_index=True)
        final_result.to_csv(os.path.join(out_path, 'all_disease.csv'), index=False)
    else:
        print('Warning: no diseases were evaluated; all_disease.csv was not created.')


if __name__ == '__main__':
    main()