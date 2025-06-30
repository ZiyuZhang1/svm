import numpy as np
from multiprocessing import Pool
import pickle

# Load the original dictionary
with open('/itf-fi-ml/shared/users/ziyuzh/svm/results/kerlens/2017.pkl', 'rb') as f:
    item = pickle.load(f)

# List of the first four feature names
feature_names = list(item.keys())[:4]

# Define the function to process one feature
def process_feature(feature_name):
    K = item[feature_name][0]
    K = 0.5 * (K + K.T)  # Enforce symmetry

    eigenvalues, eigenvectors = np.linalg.eigh(K)
    eigenvalues = np.clip(eigenvalues, 1e-12, None)  # Avoid log(0)
    K_log = eigenvectors @ np.diag(np.log(eigenvalues)) @ eigenvectors.T
    K_log = 0.5 * (K_log + K_log.T)  # Enforce symmetry

    return feature_name, K, K_log

# Kernel normalization function
def normalize_kernel(K):
    diag = np.sqrt(np.diag(K))
    return K / (diag[:, None] * diag[None, :])

# Use multiprocessing Pool to parallelize
with Pool(processes=4) as pool:
    results = pool.map(process_feature, feature_names)

# Assemble the updated dictionary
updated_dict = dict()
ks = []
logks = []

for feature_name, K, K_log in results:
    ks.append(K)
    logks.append(K_log)
    updated_dict[feature_name] = [K, K_log]

# Add linearly fused kernel and normalize it
K_linear_fused = np.mean(ks, axis=0)
K_linear_fused = 0.5 * (K_linear_fused + K_linear_fused.T)
K_linear_fused = normalize_kernel(K_linear_fused)
updated_dict['linear_fused'] = [K_linear_fused]

# Compute and add geometrically fused kernel
logk_avg = np.mean(logks, axis=0)
eigenvalues, eigenvectors = np.linalg.eigh(logk_avg)
eigenvalues = np.clip(eigenvalues, -50, 50)  # Prevent overflow
K_geo_mean = eigenvectors @ np.diag(np.exp(eigenvalues)) @ eigenvectors.T
K_geo_mean = 0.5 * (K_geo_mean + K_geo_mean.T)  # Enforce symmetry
K_geo_mean = normalize_kernel(K_geo_mean)
updated_dict['geo_fused'] = [K_geo_mean]

# Save updated dictionary
with open('/itf-fi-ml/shared/users/ziyuzh/svm/results/kerlens/2017_update.pkl', 'wb') as f:
    pickle.dump(updated_dict, f)
