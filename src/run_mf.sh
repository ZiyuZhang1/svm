#!/bin/bash

# Activate the virtual environment
source /itf-fi-ml/shared/users/ziyuzh/.venv/bin/activate

# Change to the directory containing the Python script
cd /itf-fi-ml/shared/users/ziyuzh/svm/src

# python main_mf.py 'uniport_ppi_2019' "results/2019_mf_bag_1" 2019 'disgenet' 40 100 > 2019_mf_bag_1.log 2>&1
python main_mf.py 'uniport_ppi_2019' "results/2019_mf_bag_2" 2019 'disgenet' 200 500 > 2019_mf_bag_2_add.log 2>&1
# python main_mf.py 'uniport_ppi_2019' "results/2019_mf_bag_3" 2019 'disgenet' 200 1000 > 2019_mf_bag_3.log 2>&1
# python main_mf.py 'uniport_ppi_2019' "results/2019_mf_bag_4" 2019 'disgenet' 400 500 > 2019_mf_bag_4.log 2>&1