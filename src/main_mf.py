import smurff
import pandas as pd
import os
from features_reindex import get_feature
import pickle
import sys
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import coo_matrix
import numpy as np
from model_diffusion import eval_bagging

root = '/itf-fi-ml/shared/users/ziyuzh/svm'

time_spilt = True
test_bug = True
# test_bug = False

if test_bug:
    feature_list = ['uniport_ppi_2019']
    # dga = 'opentarget'
    dga = 'disgenet'

    out_path = os.path.join(root,'results/2019_mf3')
    out_path_pred = out_path+'_pred/pred.pkl'
    time = 2019
else:
    feature_list = sys.argv[1].split(',')
    out_path = os.path.join(root,sys.argv[2])
    out_path_pred = out_path+'_pred'
    time = int(sys.argv[3])
    dga = sys.argv[4]

os.makedirs(out_path, exist_ok=True)
os.makedirs(out_path_pred, exist_ok=True)


merged_df = None

if time == 2017:
    time_feature_list = ['uniport_ppi_2017','ppi_2017_dw_80','uniport_exp','uniport_seq','uniport_esm']
elif time == 2019:
    time_feature_list = ['uniport_ppi_2019','ppi_2019_dw_40','uniport_bio','uniport_seq','uniport_esm','diffusion_2019']

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

# 1) Define the STRING universe from merged_df (side-info)
string_ids = merged_df["string_id"].unique()
string2idx = {s: i for i, s in enumerate(string_ids)}
string_list = list(string2idx.keys())

# 2) Define DISEASE universe from training data (you can expand this if you want)
disease_ids = all_df_train["disease_id"].unique()
disease2idx = {d: i for i, d in enumerate(disease_ids)}

# 3) Keep only training interactions whose string_id exists in merged_df
df = all_df_train[all_df_train["string_id"].isin(string2idx)].copy()

rows = df["disease_id"].map(disease2idx).to_numpy()
cols = df["string_id"].map(string2idx).to_numpy()
data = np.ones(len(df), dtype=np.float32)

Y_train = coo_matrix(
    (data, (rows, cols)),
    shape=(len(disease_ids), len(string_ids))
).tocsr()

# 4) Side info aligned to merged_df string universe order (same as Y_train columns)
side_info_string = (
    merged_df
    .drop_duplicates("string_id")
    .set_index("string_id")
    .loc[string_ids]     # exact order used in Y_train columns
    .to_numpy()
)

side_info = [None, side_info_string]



# Sanity checks
assert Y_train.shape[1] == side_info_string.shape[0]


Y = Y_train.tocsr()          # shape: (n_genes, n_diseases)
n_genes, n_dis = Y.shape

# all possible pairs (gene, disease)
rows_all = np.repeat(np.arange(n_genes, dtype=np.int32), n_dis)
cols_all = np.tile(np.arange(n_dis, dtype=np.int32), n_genes)

# mark observed (train) pairs so we can exclude them
obs = Y.tocoo()
obs_lin = (obs.row.astype(np.int64) * n_dis + obs.col.astype(np.int64))

all_lin = (rows_all.astype(np.int64) * n_dis + cols_all.astype(np.int64))

# keep only those not observed in train
mask = ~np.isin(all_lin, obs_lin, assume_unique=False)

rows_q = rows_all[mask]
cols_q = cols_all[mask]
vals_q = np.ones(rows_q.shape[0], dtype=np.float32)

Y_query = coo_matrix((vals_q, (rows_q, cols_q)), shape=Y.shape).tocsr()

predictions = smurff.MacauSession(
                       Ytrain     = Y_train,
                       Ytest      = Y_query,
                       side_info  = [None, side_info_string],
                       direct     = True,
                       num_latent = 16,
                       burnin     = 40,
                       nsamples   = 100).run()

# predictions = smurff.MacauSession(
#                        Ytrain     = Y_train,
#                        Ytest      = Y_query,
#                        side_info  = [None, side_info_string],
#                        direct     = True,
#                        num_latent = 32,
#                        burnin     = 60,
#                        nsamples   = 100).run()

# predictions = smurff.MacauSession(
#                        Ytrain     = Y_train,
#                        Ytest      = Y_query,
#                        side_info  = [None, side_info_string],
#                        direct     = True,
#                        num_latent = 16,
#                        burnin     = 40,
#                        nsamples   = 150).run()

S = np.empty((n_genes, n_dis), dtype=np.float32)

for p in predictions:
    r, c = p.coords   # <-- indices live here
    S[r, c] = p.pred_avg

# 3) Keep only training interactions whose string_id exists in merged_df
df = all_df_test[all_df_test["string_id"].isin(string2idx)].copy()

rows = df["disease_id"].map(disease2idx).to_numpy()
cols = df["string_id"].map(string2idx).to_numpy()
data = np.ones(len(df), dtype=np.float32)

Y_test = coo_matrix(
    (data, (rows, cols)),
    shape=(len(disease_ids), len(string_ids))
).tocsr()

all_results = []

for disease in selected_diseases:
    print(disease,len(all_df[all_df['disease_id']==disease]))
    disease_idx = disease2idx[disease]

    train_genes_idx = Y_train[disease_idx,:].nonzero()[1]
    train_genes  = np.array(string_list)[train_genes_idx]

    all_genes_idx = np.arange(len(string_ids))
    test_genes_idx = np.setdiff1d(all_genes_idx, train_genes_idx)
    test_genes  = np.array(string_list)[test_genes_idx]

    final_y_score = S[disease_idx, test_genes_idx]
    y_test = Y_test[disease_idx, test_genes_idx].toarray().ravel()

    result_df = pd.DataFrame(columns=['method',"fold","para", 'top_recall_25','top_recall_300','top_recall_10%', 'top_precision_10%', 'max_precision_10%','top_recall_30%', 'top_precision_30%', 'max_precision_30%','pm_0.5%','pm_1%','pm_5%','pm_10%','pm_15%','pm_20%','pm_25%','pm_30%','auroc',"rank_ratio",'bedroc_1','bedroc_5','bedroc_10','bedroc_30'])

    prediction_collection = dict()
    prediction_collection['true_label'] = y_test
    prediction_collection["test_genes"] = test_genes
    prediction_collection["train_pos_genes"] = train_genes

    feature_name = feature_list[0]

    ranked_predict_index, results = eval_bagging(final_y_score, y_test)
    # Add results to the result dataframe
    result_df.loc[len(result_df.index)] = ["random_negative",'0',feature_name+'-0-0-0', *results]
    prediction_collection[feature_name] = final_y_score
    
    with open(out_path_pred+f'/{disease}_pred.pkl', 'wb') as f:
        pickle.dump(prediction_collection, f)

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