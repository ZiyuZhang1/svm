#!/bin/bash

# Activate the virtual environment
source /itf-fi-ml/shared/users/ziyuzh/new_esm_env/bin/activate

# Change to the directory containing the Python script
cd /itf-fi-ml/shared/users/ziyuzh/svm/src
# python main_gnn.py 'uniport_ppi_2019' "results/2019_gnn" 2019 > 2019_gnn.log 2>&1
# python main_gnn.py 'uniport_ppi_2019' "results/2019_gnn_sage" 2019 > 2019_gnn_sage.log 2>&1
# python main_gnn.py 'uniport_ppi_2019' "results/2019_gnn_sage" 2019 > 2019_gnn_add_test.log 2>&1

# python main_gnn.py 'uniport_ppi_2019' "results/2019_gnn_sage" 2019 > 2019_gnn_add_improvesage.log 2>&1
# python main_gnn.py 'uniport_ppi_2019' "results/2019_gnn_sage" 2019 > 2019_gcn_deoversmooth.log 2>&1
python main_gnn.py 'uniport_ppi_2019' "results/2019_gnn_sage" 2019 > 2019_gcn_deoversmooth_less_smooth.log 2>&1


# python main_gnn_copy.py 'uniport_ppi_2019' "results/2019_gnn_sage" 2019 > 2019_gnn_copy_optimizer_01.log 2>&1
# python main_gnn_copy_copy.py 'uniport_ppi_2019' "results/2019_gnn_sage" 2019 > 2019_gnn_copy_optimizer_001.log 2>&1
# python main_gnn_less_changes.py 'uniport_ppi_2019' "results/2019_gnn_sage" 2019 > 2019_gnn_less_changes.log 2>&1

# # Activate the virtual environment
# source /itf-fi-ml/shared/users/ziyuzh/.venv/bin/activate

# # Change to the directory containing the Python script
# cd /itf-fi-ml/shared/users/ziyuzh/svm/src

# python  bio_evidence_para.py '2019_gnn_add_improvesage_pred' '2019_gnn_add_improvesage_pred_sim'> bio_evidence_2019_gnn_add_improvesage_pred.log 2>&1