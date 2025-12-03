
#!/bin/bash

# Activate the virtual environment
source /itf-fi-ml/shared/users/ziyuzh/.venv/bin/activate

# Change to the directory containing the Python script
cd /itf-fi-ml/shared/users/ziyuzh/svm/src


############################## first make sure all results are saved in that pkl file
python bio_evidence.py /itf-fi-ml/shared/users/ziyuzh/svm/results/df_gnn_occsvm_pred  > bio_df.log 2>&1
