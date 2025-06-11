#!/bin/bash

# Activate the virtual environment
source /itf-fi-ml/shared/users/ziyuzh/.venv/bin/activate

# Change to the directory containing the Python script
cd /itf-fi-ml/shared/users/ziyuzh/svm/src

# python main_reindex_time_no_para.py "ppi_2016" "results/ppi_2017_bootstrap_no_para_full" 2017
python main_reindex_time.py "bioconcept" "results/bioconcept_2019_bootstrap_full" 2019
# python main_reindex_time.py "esm2" "results/esm2_2017_posbag_full" 2017
# python main_reindex_time.py "esm2" "results/esm2_2019_posbag_full" 2019
# python main_reindex_time.py "uniport" "results/uniport_2017_full" 2017
python main_reindex_time.py "gene2vec" "results/gene2vec_2017_bootstrap_full" 2017
# python main_reindex.py "biograd" "results/biograd_full"
# python main_reindex_time.py "scgpt" "results/scgpt_full_2023" 2023