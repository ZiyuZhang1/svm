#!/bin/bash

# Activate the virtual environment
source /itf-fi-ml/shared/users/ziyuzh/.venv/bin/activate

# Change to the directory containing the Python script
cd /itf-fi-ml/shared/users/ziyuzh/svm/src

python bio_evidence_para.py '2019_rf_renamed_pred' '2019_rf_renamed_pred_add_sim' > bio_evidence_2019_rf_renamed_pred.log 2>&1