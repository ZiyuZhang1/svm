#!/bin/bash

# Activate the virtual environment
source /itf-fi-ml/shared/users/ziyuzh/.venv/bin/activate

# Change to the directory containing the Python script
cd /itf-fi-ml/shared/users/ziyuzh/svm/src

# python main_diffusion.py 'uniport_ppi_2019,ppi_2019_dw_40,diffusion_2019,uniport_bio,uniport_seq,uniport_esm' "results/2019_df" 2019 'disgenet' > 2019_df.log 2>&1
# python main_late_fusion.py 'uniport_ppi_2019,ppi_2019_dw_40,diffusion_2019,uniport_bio,uniport_seq,uniport_esm' "results/2019_late" 2019 'disgenet' > 2019_late.log 2>&1
# python main_late_fusion.py 'uniport_ppi_2019,ppi_2019_dw_40,diffusion_2019,uniport_bio,uniport_seq,uniport_esm' "results/2019_late_bio" 2019 'disgenet' > 2019_late_bio.log 2>&1
# python main_lf_bag.py 'uniport_ppi_2019,ppi_2019_dw_40,diffusion_2019,uniport_bio,uniport_seq,uniport_esm' "results/2019_lf_bag" 2019 'disgenet' > 2019_lf_bag.log 2>&1
# python main_lf_bag.py 'uniport_ppi_2019,ppi_2019_dw_40,diffusion_2019,uniport_bio,uniport_seq,uniport_esm' "results/2019_lf_bag_cv" 2019 'disgenet' > 2019_lf_bag_cv.log 2>&1
# python main_lf_bag.py 'uniport_ppi_2019,ppi_2019_dw_40,diffusion_2019,uniport_bio,uniport_seq,uniport_esm' "results/2019_lf_bag_cv_all_save" 2019 'disgenet' > 2019_lf_bag_cv_all_save.log 2>&1
# python main_lf_bag_fix.py 'uniport_ppi_2019,ppi_2019_dw_40,diffusion_2019,uniport_bio,uniport_seq,uniport_esm' "results/2019_lf_bag_cv_fix_orness" 2019 'disgenet' > 2019_lf_bag_cv_fix_orness.log 2>&1
# python main_lf_bag_fix.py 'uniport_ppi_2019,ppi_2019_dw_40,diffusion_2019,uniport_bio,uniport_seq,uniport_esm' "results/2019_lf_bag_cv_fix_orness" 2019 'disgenet' > 2019_lf_bag_cv_fix_biogrid.log 2>&1
python main_biogrid.py 'uniport_ppi_2019,ppi_2019_dw_40,diffusion_2019,uniport_bio,uniport_seq,uniport_esm' "results/2019_lf_bag_cv_fix_orness" 2019 'disgenet' > 2019_biogrid.log 2>&1
