import pandas as pd
import os
from features import get_feature, read_data
from model import evaluate_disease
# from model_assemble import evaluate_disease

# # Set up argument parser
# parser = argparse.ArgumentParser(description='Process some integers.')
# parser.add_argument('--pseudo_select_function', type=int, default=1,
#                     help='an integer to select the pseudo function')
# # Parse the arguments
# args = parser.parse_args()
# # Assign the argument to the variable
# pseudo_select_function = args.pseudo_select_function
pseudo_select_function = [1]

root = '/itf-fi-ml/shared/users/ziyuzh/svm'
feature = 'go_exp'
# feature = 'ppi'
# feature = 'txt'
# feature = 'ppi_700'
# feature = 'biograd'


out_path = os.path.join(root,'results',feature)
# out_path = os.path.join(root,'results','ppi_time_filtered')
# out_path = os.path.join(root,'ppi_10_neg')

if not os.path.exists(out_path):
    os.mkdir(out_path)

feature_df = get_feature(root,feature)

all_df = pd.read_csv(os.path.join(root,'data','disgent_2020','disgenet_string.csv'))
# all_df = pd.read_csv('/itf-fi-ml/shared/users/ziyuzh/svm/data/disgent_2020/timecut/disgent_with_time.csv')
# all_df = pd.read_csv(os.path.join(root,'data/gmb/gmb_string.csv'))


selected_diseases = (
    all_df.groupby('disease_id')
    .filter(lambda x: len(x) > 15)
    ['disease_id']
    .unique()
    .tolist()
)[36:]

# selected_diseases = ['ICD10_N80']
# methods = ['ooc','random_negative']
# methods = ['ooc','random_negative','pseudo_labeling','pseudo_on_vector']

# methods = ['pseudo_on_vector']
methods = ['random_negative']
# methods = ['random_negative','pseudo_labeling']
all_results = []




for disease in selected_diseases:
    param_fix = False
    # if disease in ['ICD10_E66','ICD10_E11','ICD10_C50','ICD10_F31','ICD10_C61','ICD10_F20']:
    #         param_fix = True
    df, X, y = read_data(disease, all_df, feature_df)
    result_df = evaluate_disease(X, y, methods,functions=pseudo_select_function, param_fix = param_fix)
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