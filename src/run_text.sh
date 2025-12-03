#!/bin/bash

# Activate the virtual environment
source /itf-fi-ml/shared/users/ziyuzh/.venv/bin/activate

# Change to the directory containing the Python script
cd /itf-fi-ml/shared/users/ziyuzh/svm/src

python precalculate_kernel_early.py
# python precalculate_kernel.py > text_kernel_calculation.log 2>&1
# python main_text.py > text_2.log 2>&1
python main_early.py > early_ppi.log 2>&1