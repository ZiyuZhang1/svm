#!/bin/bash

# Activate the virtual environment
source /itf-fi-ml/shared/users/ziyuzh/.venv/bin/activate

# Change to the directory containing the Python script
cd /itf-fi-ml/shared/users/ziyuzh/baseline/src

# python main_reindex.py "ppi" "results/ppi_full"
python main_reindex.py "bioconcept" "results/bioconcept_full"
python main_reindex.py "esm2" "results/esm2_full"
python main_reindex.py "uniport" "results/uniport_full"
python main_reindex.py "gene2vec" "results/gene2vec_full"
python main_reindex.py "biograd" "results/biograd_full"




# Run the Python script
# python main_reindex.py "ppi_align" "results/ppi_align"
# python main_reindex.py "bioconcept" "results/bioconcept_align"
# python main_reindex.py "t5_align" "results/t5_align"
# python main_reindex.py "gene2vec" "results/gene2vec_align"
# python main_reindex.py "scgpt" "results/scgpt_align"
# python main_reindex.py "esm2" "results/esm2_align"
# python main_reindex.py "MASHUP" "results/MASHUP_align"
# python main_reindex.py "GENEPT_MODEL3" "results/GENEPT_MODEL3_align"
# python main_reindex.py "GF_12L95M" "results/GF_12L95M_align"