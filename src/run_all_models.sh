#!/bin/bash

# Activate the virtual environment
source /itf-fi-ml/shared/users/ziyuzh/.venv/bin/activate

# Change to the directory containing the Python script
cd /itf-fi-ml/shared/users/ziyuzh/svm/src

python main_diffusion.py 'uniport_ppi_2019,ppi_2019_dw_40,diffusion_2019,uniport_bio,uniport_seq,uniport_esm' "results/2019_df_oob_mid" 2019 'disgenet' > 2019_df_oob_mid.log 2>&1
python main_diffusion.py 'uniport_ppi_2019,ppi_2019_dw_40,diffusion_2019' "results/2019_df_oob_ppi_mid" 2019 'disgenet' > 2019_df_oob_ppi_mid.log 2>&1

python main_mf.py 'uniport_ppi_2019' "results/2019_mf_bag_2" 2019 'disgenet' 200 500 > 2019_mf_bag_2_add.log 2>&1

python main_rf.py 'uniport_ppi_2019,ppi_2019_dw_40,diffusion_2019_2' "results/2019_rf_ppi" 2019 > 2019_rf_ppi.log 2>&1

# Activate the virtual environment
source /itf-fi-ml/shared/users/ziyuzh/new_esm_env/bin/activate

# Change to the directory containing the Python script
cd /itf-fi-ml/shared/users/ziyuzh/svm/src
python main_nn_non_para.py 'uniport_ppi_2019,ppi_2019_dw_40,uniport_bio,uniport_seq,uniport_esm,diffusion_2019_2' "results/2019_nn_oob" 2019 > 2019_nn_oob.log 2>&1
python main_nn_non_para.py 'uniport_ppi_2019,ppi_2019_dw_40,diffusion_2019_2' "results/2019_nn_oob_ppi" 2019 > 2019_nn_oob_ppi.log 2>&1
python main_gnn.py 'uniport_ppi_2019' "results/2019_gnn_sage" 2019 > 2019_gcn_deoversmooth_less_smooth.log 2>&1