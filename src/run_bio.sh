#!/bin/bash

# Activate the virtual environment
source /itf-fi-ml/shared/users/ziyuzh/.venv/bin/activate

# Change to the directory containing the Python script
cd /itf-fi-ml/shared/users/ziyuzh/svm/src

# python  bio_evidence_para.py 'df_gnn_occsvm_pred' 'df_gnn_occsvm_pred_add_sim'> bio_evidence_df_gnn_occsvm_pred.log 2>&1
# python  bio_evidence_para.py '2019_mf_add_pred' '2019_mf_add_pred_sim'> bio_evidence_2019_mf_add_pred.log 2>&1
python  bio_evidence_para.py 'node_degree_check' 'node_degree_check_sim'> bio_evidence_node_degree_check.log 2>&1