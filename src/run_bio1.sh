#!/bin/bash

# Activate the virtual environment
source /itf-fi-ml/shared/users/ziyuzh/.venv/bin/activate

# Change to the directory containing the Python script
cd /itf-fi-ml/shared/users/ziyuzh/svm/src

python bio_evidence_para.py '2019_occ_deep_svd_pred/pred.pkl' '2019_occ_deep_svd_pred_add_sim' > bio_evidence_2019_occ_deep_svd_pred.log 2>&1
python bio_evidence_para.py '2019_mf_bag_2_pred' '2019_mf_bag_2_pred_add_sim' > bio_evidence_2019_mf_bag_2_pred.log 2>&1