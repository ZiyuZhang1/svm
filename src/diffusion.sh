#!/bin/bash

# Activate the virtual environment
source /itf-fi-ml/shared/users/ziyuzh/.venv/bin/activate

# Change to the script directory
cd /itf-fi-ml/shared/users/ziyuzh/svm/src

# Run the Python script and log both stdout and stderr
# python diffusion.py > diffusion.log 2>&1
python diffusion_biogrid.py > diffusion_biogrid.log 2>&1