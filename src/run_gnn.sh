#!/bin/bash

# Activate the virtual environment
source /itf-fi-ml/shared/users/ziyuzh/new_esm_env/bin/activate

# Change to the directory containing the Python script
cd /itf-fi-ml/shared/users/ziyuzh/svm/src
python main_gnn.py 'uniport_ppi_2019' "results/2019_gnn" 2019 > 2019_gnn.log 2>&1