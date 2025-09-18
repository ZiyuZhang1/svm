import os
import pickle
import numpy as np
import pandas as pd
import gseapy as gp

with open('/itf-fi-ml/shared/users/ziyuzh/svm/data/uniport_id/uni2name.pkl', 'rb') as file:
    uni2name_dict = pickle.load(file)
def enriched_set(input_ids,time):
    gene_names = set()
    for unid in input_ids:
        gene_list = uni2name_dict.get(unid, [])
        gene_names.update(gene_list)
    gene_names = list(gene_names) 
    
    if time == 2019:
        enrich_db = ['GO_Biological_Process_2021','GO_Cellular_Component_2021','GO_Molecular_Function_2021','KEGG_2019_Human','Reactome_2022']
    elif time == 2017:
        enrich_db = ['GO_Biological_Process_2021','GO_Cellular_Component_2021','GO_Molecular_Function_2021','KEGG_2016']
    try:
        enr = gp.enrichr(
            gene_list=gene_names,
            gene_sets=enrich_db,
            organism='human', 
            outdir=None
        )
        enr_df = enr.results
        if enr_df is None or enr_df.empty:
            return set()
        
        result_terms = enr_df.loc[enr_df['Adjusted P-value'] < 0.01, ['Gene_set', 'Term']]
        return set(map(tuple, result_terms.values))
    
    except Exception as e:
        # Optionally log the error: print(f"Enrichment failed: {e}")
        return set()

def calculate_jac_sim(enrich_1, enrich_2):
    intersection = enrich_1 & enrich_2
    union = enrich_1 | enrich_2
    if not union:
        return 0.0  # Define similarity as 0 if both sets are empty
    return len(intersection) / len(union)

root = '/itf-fi-ml/shared/users/ziyuzh/svm/results/2019_lf_bag_cv_all_save_pred'
for file_name in os.listdir(root):
    if not file_name.endswith('.pkl'):
        continue
    file_path = os.path.join(root,file_name)
    disease_record = dict()

    with open(file_path, "rb") as f:
        data = pickle.load(f)
    train_genes = data['train_pos_genes']
    enrich_train_set = enriched_set(train_genes,2019)
    test_indices = data['test_genes']

    for feature in data.keys():
        if feature not in ['test_genes', 'train_pos_genes', 'true_label']:
            final_y_score = data[feature]
            sim_scores = []
            for ratio in np.linspace(0, 0.5, 10):
                enrich_predict_genes = test_indices[np.argsort(final_y_score)[::-1]][:int(ratio*len(final_y_score))]
                enrich_predict_set = enriched_set(enrich_predict_genes,2019)
                jac_sm0 = calculate_jac_sim(enrich_train_set,enrich_predict_set)
                sim_scores.append(jac_sm0)
            disease_record[feature] = sim_scores    


    out_root = '/itf-fi-ml/shared/users/ziyuzh/svm/results/2019_lf_bag_cv_pred_sim'
    saved_pkl_path = os.path.join(out_root, file_name)

    # Ensure the directory exists
    os.makedirs(out_root, exist_ok=True)

    # Save pickle
    with open(saved_pkl_path, "wb") as f:
        pickle.dump(disease_record, f)

    