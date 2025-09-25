#!/bin/bash

# Activate the virtual environment
source /itf-fi-ml/shared/users/ziyuzh/new_esm_env/bin/activate

# Change to the directory containing the Python script
cd /itf-fi-ml/shared/users/ziyuzh/svm/src

python main_rf.py 'uniport_ppi_2019,ppi_2019_dw_40,uniport_bio,uniport_seq,uniport_esm,diffusion_2019_2' "results/2019_rf" 2017 > 2019_rf.log 2>&1

# python main_nn_ppi_single.py 'ppi_2017_700' "results/2017_nn_700" 2017 > 2017_nn_700.log 2>&1

# python main_reindex_time_fusion_weights_uniport_nn.py 'uniport_ppi_2017,ppi_2017_dw_80,uniport_exp,uniport_seq,uniport_esm' "results/2017_nn_uni" 2017 > 2017_nn_uni.log 2>&1
# python main_reindex_time_fusion_weights_uniport_nn.py 'uniport_ppi_2019,ppi_2019_dw_40,uniport_bio,uniport_seq,uniport_esm' "results/2019_nn_uni" 2019 > 2019_nn_uni.log 2>&1

# python main_reindex_time_fusion_weights_nn.py 'ppi_2016,gene2vec,uniport,esm2' "results/2017_fused_geo_weight_bag_nn" 2017 > output_2017_nn.log 2>&1
# python main_reindex_time_fusion_weights_nn.py 'ppi_2019,bioconcept,uniport,esm2' "results/2019_fused_geo_weight_bag_nn" 2019 > output_2019_nn.log 2>&1

# python main_reindex_time.py "ppi_2016" "results/ppi_2017_bootstrap_po_neg_full" 2017
# python main_reindex_time.py "bioconcept" "results/bioconcept_2019_posbag_full" 2019
# python main_reindex_time_esm22019.py "esm2" "results/esm2_2019_full" 2019
# python main_reindex_time.py "uniport" "results/uniport_2019_full" 2019
# python main_reindex_time.py "gene2vec" "results/gene2vec_2017_full" 2017
# python main_reindex.py "biograd" "results/biograd_full"
# python main_reindex_time.py "scgpt" "results/scgpt_full_2023" 2023