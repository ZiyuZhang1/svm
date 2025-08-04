#!/bin/bash

# Activate the virtual environment
source /itf-fi-ml/shared/users/ziyuzh/.venv/bin/activate

# Change to the directory containing the Python script
cd /itf-fi-ml/shared/users/ziyuzh/svm/src

python main_diffusion.py 'uniport_ppi_2019,ppi_2019_dw_40,diffusion_2019' "results/2019_df_ppi" 2019 'disgenet' > 2019_df_ppi.log 2>&1
