import pandas as pd
import os
from features_reindex import get_feature
import pickle
import sys
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import coo_matrix
import numpy as np
from model_diffusion import eval_bagging
from model_mf import neg_bag
from multiprocessing import Pool

root = '/itf-fi-ml/shared/users/ziyuzh/svm'

time_spilt = True
test_bug = True
# test_bug = False

if test_bug:
    feature_list = [ 'uniport_ppi_2019',
        'ppi_2019_dw_40',
        'uniport_bio',
        'uniport_seq',
        'uniport_esm',
        'diffusion_2019_pca',
    ]
    # dga = 'opentarget'
    dga = 'disgenet'

    out_path = os.path.join(root,'results/2019_mf_add')
    out_path_pred = out_path+'_pred/'
    time = 2019

    b_para, n_para = 200, 500
else:
    feature_list = sys.argv[1].split(',')
    out_path = os.path.join(root,sys.argv[2])
    out_path_pred = out_path+'_pred'
    time = int(sys.argv[3])
    dga = sys.argv[4]
    b_para = sys.argv[5]
    n_para = sys.argv[6]

os.makedirs(out_path, exist_ok=True)
os.makedirs(out_path_pred, exist_ok=True)

def select_feature_block(merged_df: pd.DataFrame, feat: str) -> np.ndarray:
    """Return side-info matrix for ONE feature block in the same row order as merged_df's string_ids."""
    # columns belonging to this feature block + string_id
    cols = ['string_id'] + [c for c in merged_df.columns if c.startswith(f"{feat}_")]
    if len(cols) <= 1:
        raise ValueError(f"No columns found for feature block: {feat}")

    block_df = merged_df[cols].drop_duplicates("string_id").set_index("string_id")
    return block_df.to_numpy()

merged_df = None

if time == 2017:
    time_feature_list = ['uniport_ppi_2017','ppi_2017_dw_80','uniport_exp','uniport_seq','uniport_esm']
elif time == 2019:
    time_feature_list = ['uniport_ppi_2019','ppi_2019_dw_40','uniport_bio','uniport_seq','uniport_esm','diffusion_2019_pca']

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
name_list = feature_list + ['string_id']

merged_df = merged_df[[col for col in merged_df.columns if any(item in col for item in name_list)]]
# merged_df.columns = merged_df.columns.str.replace('uniport_ppi_2019_', '', regex=False)
# merged_df.to_csv('/itf-fi-ml/shared/users/ziyuzh/svm/data/input_deep_svd/node2vec_features.csv',index=False)
if dga == 'disgenet':
    all_df = pd.read_csv('/itf-fi-ml/shared/users/ziyuzh/svm/data/disgent_2020/timecut/dga_time_uniport.csv')
elif dga == 'opentarget':
    all_df = pd.read_csv('/itf-fi-ml/shared/users/ziyuzh/svm/data/opentarget/ot_dga_time_uni.csv')
    all_df = all_df[all_df['score']>=0.4]

all_df = all_df[all_df['string_id'].isin(merged_df['string_id'])]
# all_df = pd.read_csv('/itf-fi-ml/shared/users/ziyuzh/svm/data/disgent_2020/timecut/align_disgent_with_time.csv')

# methods = ['ooc','random_negative','pseudo_labeling','pseudo_labeling_mask']
# methods = ['random_negative','pseudo_labeling','pseudo_labeling_mask','pseudo_labeling_cluster_all_mask']
# methods = ['random_negative','random_negative_bagging','random_pos_negative_bagging']
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

all_df = all_df[all_df['string_id'].isin(merged_df['string_id'].unique())]
all_df = all_df[all_df['disease_id'].isin(selected_diseases)]

all_df_train = all_df[all_df['first_pub_year']<=2019][['disease_id','string_id']]
all_df_test = all_df[all_df['first_pub_year']>2019][['disease_id','string_id']]

string_ids = merged_df["string_id"].unique()
string2idx = {s: i for i, s in enumerate(string_ids)}
string_list = list(string2idx.keys())

disease_ids = all_df_train["disease_id"].unique()
disease2idx = {d: i for i, d in enumerate(disease_ids)}

df = all_df_test[all_df_test["string_id"].isin(string2idx)].copy()
rows = df["disease_id"].map(disease2idx).to_numpy()
cols = df["string_id"].map(string2idx).to_numpy()
data = np.ones(len(df), dtype=np.float32)
Y_test = coo_matrix((data, (rows, cols)), shape=(len(disease_ids), len(string_ids))).tocsr()

df = all_df_train[all_df_train["string_id"].isin(string2idx)].copy()
rows = df["disease_id"].map(disease2idx).to_numpy()
cols = df["string_id"].map(string2idx).to_numpy()
data = np.ones(len(df), dtype=np.float32)
Y_train = coo_matrix((data, (rows, cols)), shape=(len(disease_ids), len(string_ids))).tocsr()

########### fill negative values and run MF
num_processes = 20
seed_list = [42 + i for i in range(num_processes)]
all_results = []

for feat in feature_list:
    print(f"\n==== Running MF for feature: {feat} ====")

    # side info for THIS feature, aligned to merged_df string_id order
    side_info_feat = select_feature_block(merged_df, feat)

    args_list = [
        (all_df_train, string2idx, disease2idx, side_info_feat, seed, b_para, n_para)
        for seed in seed_list
    ]

    with Pool(processes=num_processes) as pool:
        bag_S = pool.starmap(neg_bag, args_list)

    S_stack = np.stack(bag_S, axis=0)
    S_mean = np.nanmean(S_stack, axis=0)

    del bag_S, S_stack, side_info_feat

    # Per disease: compute metrics and STORE scores for this feat
    for disease in selected_diseases[:1]:
        disease_idx = disease2idx[disease]

        train_genes_idx = Y_train[disease_idx, :].nonzero()[1]
        train_genes = np.array(string_list)[train_genes_idx]

        all_genes_idx = np.arange(len(string_ids))
        test_genes_idx = np.setdiff1d(all_genes_idx, train_genes_idx)
        test_genes = np.array(string_list)[test_genes_idx]

        final_y_score = S_mean[disease_idx, test_genes_idx]
        y_test = Y_test[disease_idx, test_genes_idx].toarray().ravel()

        # one disease pickle that accumulates ALL feature scores
        pred_path = os.path.join(out_path_pred, f'{disease}_pred.pkl')

        if os.path.exists(pred_path):
            with open(pred_path, 'rb') as f:
                prediction_collection = pickle.load(f)
        else:
            prediction_collection = {
                'true_label': y_test,
                'test_genes': test_genes,
                'train_pos_genes': train_genes,
            }

        # save this feature’s scores
        prediction_collection[feat] = final_y_score

        with open(pred_path, 'wb') as f:
            pickle.dump(prediction_collection, f)

        # metrics CSV: append one row per feature
        ranked_predict_index, results = eval_bagging(final_y_score, y_test)

        disease_csv = os.path.join(out_path, f"{disease}.csv")
        columns = [
            'method', 'fold', 'para',
            'top_recall_25', 'top_recall_300', 'top_recall_10%',
            'top_precision_10%', 'max_precision_10%',
            'top_recall_30%', 'top_precision_30%', 'max_precision_30%',
            'pm_0.5%', 'pm_1%', 'pm_5%', 'pm_10%', 'pm_15%', 'pm_20%', 'pm_25%', 'pm_30%',
            'auroc', 'rank_ratio', 'bedroc_1', 'bedroc_5', 'bedroc_10', 'bedroc_30'
        ]

        if os.path.exists(disease_csv):
            result_df = pd.read_csv(disease_csv)
        else:
            result_df = pd.DataFrame(columns=columns)

        # keep your method naming; para string now includes feature name
        result_df.loc[len(result_df)] = ["random_negative", '0', f"{feat}-0-0-0", *results]
        result_df.to_csv(disease_csv, index=False)

        mean_df = (
            result_df.groupby(['method'])[columns[3:]]
            .mean()
            .reset_index()
        )
        mean_df['disease'] = disease
        mean_df['feature'] = feat
        all_results.append(mean_df)

# 7) Global summary
final_result = pd.concat(all_results, ignore_index=True)
final_result.to_csv(os.path.join(out_path, 'all_disease.csv'), index=False)