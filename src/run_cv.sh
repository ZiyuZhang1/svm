#!/bin/bash

# Activate the virtual environment
source /itf-fi-ml/shared/users/ziyuzh/.venv/bin/activate

# Change to the directory containing the Python script
cd /itf-fi-ml/shared/users/ziyuzh/svm/src

python main_reindex_time_fusion_weights_uniport_cv_filter.py 'uniport_ppi_2017,uniport_exp,uniport_seq,uniport_esm' "results/2017_cv" 2017 > output_cv_2017.log 2>&1
python main_reindex_time_fusion_weights_uniport_cv_filter.py 'uniport_ppi_2019,uniport_bio,uniport_seq,uniport_esm' "results/2019_cv" 2019 > output_cv_2019.log 2>&1

# python main_reindex_time.py "ppi_2016" "results/ppi_2017_bootstrap_po_neg_full" 2017
# python main_reindex_time.py "bioconcept" "results/bioconcept_2019_posbag_full" 2019
# python main_reindex_time_esm22019.py "esm2" "results/esm2_2019_full" 2019
# python main_reindex_time.py "uniport" "results/uniport_2019_full" 2019
# python main_reindex_time.py "gene2vec" "results/gene2vec_2017_full" 2017
# python main_reindex.py "biograd" "results/biograd_full"
# python main_reindex_time.py "scgpt" "results/scgpt_full_2023" 2023