import pandas as pd
import os
from features_reindex import get_feature, read_data, read_data_timecut
from model_lf_bag import evaluate_disease
import pickle
import sys
import multiprocessing as mp
from sklearn.preprocessing import MinMaxScaler


root = '/itf-fi-ml/shared/users/ziyuzh/svm'

# time_spilt = True
# feature = 'ppi_'+str(time)

time_spilt = True
# test_bug = True
test_bug = False

if test_bug:
    # feature_list = ['uniport_ppi_2019','uniport_bio','uniport_seq','uniport_esm']
    # feature_list = ['ppi_2019','bioconcept']
    # feature_list = ['uniport_ppi_2017','ppi_2017_dw_80','uniport_exp','uniport_seq','uniport_esm']
    # feature_list = ['uniport_ppi_2017','ppi_2017_dw_80','uniport_exp','uniport_seq']
    # feature_list = ['uniport_ppi_2019','ppi_2019_dw_40','uniport_bio','uniport_seq','uniport_esm']
    feature_list = ['uniport_ppi_2019','ppi_2019_dw_40','uniport_bio','uniport_seq','uniport_esm','diffusion_2019']
    # feature_list = ['uniport_ppi_2019','ppi_2019_dw_40','diffusion_2019']
    # feature_list = ['uniport_ppi_2019','ppi_2019_dw_40','uniport_bio','uniport_seq','uniport_esm']


    # dga = 'opentarget'
    dga = 'disgenet'

    out_path = os.path.join(root,'results/temp')
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
all_results = []

for disease in selected_diseases:
    print(disease,len(all_df[all_df['disease_id']==disease]))
    if time_spilt:
        df, y = read_data_timecut(disease, all_df, merged_df,time)
    else:
        df, y = read_data(disease, all_df, merged_df,time)
    result_df, prediction_collection = evaluate_disease(disease, time, feature_list, df, y, methods,time_spilt)
    
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