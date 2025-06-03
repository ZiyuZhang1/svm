import argparse
import pandas as pd
import os
from features import get_feature, read_data
from model import evaluate_disease
import numpy as np

# # Set up argument parser
# parser = argparse.ArgumentParser(description='Process some integers.')
# parser.add_argument('--pseudo_select_function', type=int, default=1,
#                     help='an integer to select the pseudo function')
# # Parse the arguments
# args = parser.parse_args()
# # Assign the argument to the variable
# pseudo_select_function = args.pseudo_select_function
pseudo_select_function = [1,2]

root = '/itf-fi-ml/shared/users/ziyuzh/svm'
feature = 'exp'


out_path = os.path.join(root,'results',feature)
if not os.path.exists(out_path):
    os.mkdir(out_path)

feature_df = get_feature(root,feature)

all_df = pd.read_csv(os.path.join(root,'data','disgent_2020','disgenet_string.csv'))

selected_diseases = (
    all_df.groupby('disease_id')
    .filter(lambda x: len(x) > 15)
    ['disease_id']
    .unique()
    .tolist()
)

# selected_diseases = ['ICD10_G20']
methods = ['ooc','random_negative','pseudo_labeling']
all_results = []

for disease in selected_diseases:
    methods = ['ooc', 'random_negative', 'pseudo_labeling']
    df, X, y = read_data(disease, all_df, feature_df)
    if np.sum(y==1) >= 10:
        result_df = evaluate_disease(X, y, methods,functions=pseudo_select_function)
        result_df.to_csv(os.path.join(out_path, f"{disease}.csv"),index = False)
        # Calculate mean metrics
        mean_df = result_df.groupby(['method'])[["recall", "precision", 'top_recall_10', 'top_precision_10', 'max_precision_10','top_recall_30', 'top_precision_30', 'max_precision_30','auroc',"rank_ratio",'bedroc_1','bedroc_5','bedroc_10','bedroc_30']].mean().reset_index()

        # Add disease information
        mean_df['disease'] = disease

        # Append to all_results list
        all_results.append(mean_df)

# Concatenate all results into a single DataFrame
final_result = pd.concat(all_results, ignore_index=True)
final_result.to_csv(os.path.join(out_path,'all_disease.csv'),index=False)