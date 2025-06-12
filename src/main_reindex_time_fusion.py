import pandas as pd
import os
from features_reindex import get_feature, read_data, read_data_timecut
from model_reindex_fusion import evaluate_disease
import sys
import multiprocessing as mp

root = '/itf-fi-ml/shared/users/ziyuzh/svm'

# time_spilt = True
# feature = 'ppi_'+str(time)

time_spilt = True
# test_bug = True
test_bug = False

if test_bug:
    # feature_list = ['ppi_2019','bioconcept','uniport','esm2']
    feature_list = ['ppi_2019','bioconcept','uniport','esm2']
    feature_list = ['ppi_2019','bioconcept']


    out_path = os.path.join(root,'results/temp')
    time = 2019
else:
    feature_list = sys.argv[1]
    feature_list = ['ppi_2019','bioconcept','uniport','esm2']
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

for disease in selected_diseases[1:]:
    print(disease,len(all_df[all_df['disease_id']==disease]))
    if time_spilt:
        df, y = read_data_timecut(disease, all_df, merged_df,time)
    else:
        df, y = read_data(disease, all_df, merged_df,time)
    result_df = evaluate_disease(disease, feature_list, df, y, methods,time_spilt)
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