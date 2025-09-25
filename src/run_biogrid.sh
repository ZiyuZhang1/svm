#!/bin/bash

# Activate the virtual environment
source /itf-fi-ml/shared/users/ziyuzh/.venv/bin/activate

# Change to the directory containing the Python script
cd /itf-fi-ml/shared/users/ziyuzh/svm/src

python main_biogrid.py 'uniport_bio,uniport_seq,uniport_esm,biograd_2019_n2v,biograd_2019_dw_40,biogrid_diffusion_2019' "results/biogrid" 2019 'disgenet' 'biogrid'> biogrid.log 2>&1