import networkx as nx
import numpy as np
from scipy.linalg import expm
import pickle
import os
import pandas as pd
import random

# def normalize_kernel(K):
#     diag = np.sqrt(np.diag(K))
#     diag[diag == 0] = 1e-8  # Avoid division by zero
#     for i in range(K.shape[0]):
#         K[i, :] /= diag[i]
#     for j in range(K.shape[1]):
#         K[:, j] /= diag[j]
#     return K

def normalize_kernel(K):
    diag = np.sqrt(np.diag(K))
    diag[diag == 0] = 1e-8  # Avoid division by zero
    return K / (diag[:, None] * diag[None, :])


def diffusion_kernel(G, beta, normalized=True):
    nodes = list(G.nodes())
    if normalized:
        L = nx.normalized_laplacian_matrix(G, nodelist=nodes).todense()
    else:
        L = nx.laplacian_matrix(G, nodelist=nodes).todense()
    
    K = expm(-beta * L)

    return np.array(K)

def process_kernel(args):
    K= args

    eigenvalues, eigenvectors = np.linalg.eigh(K)
    eigenvalues = np.clip(eigenvalues, 1e-12, None)  # Avoid log(0)
    K_log = eigenvectors @ np.diag(np.log(eigenvalues)) @ eigenvectors.T

    K_log = 0.5 * (K_log + K_log.T)

    return K_log

def merge_similarity_matrix(sim_matrix, sample_names, merge_groups, delete_list):

    # Step 1: Create a mapping from sample to index for quick lookup
    name_to_index = {name: i for i, name in enumerate(sample_names)}

    # Step 2: Build group name and sample-to-group map
    sample_to_group = {}
    new_names = []
    for group in merge_groups:
        new_name = '_'.join(sorted(group))
        new_names.append(new_name)
        for s in group:
            sample_to_group[s] = new_name

    all_samples = set(sample_names)
    merged_samples = set(sample_to_group.keys())
    kept_samples = sorted(all_samples - merged_samples - set(delete_list))

    # Step 3: Final sample list and group mapping
    final_samples = new_names + kept_samples
    group_map = {name: [name] for name in kept_samples}
    for group in merge_groups:
        group_name = '_'.join(sorted(group))
        group_map[group_name] = group

    # Step 4: Compute average similarities between groups (on the fly)
    new_matrix = pd.DataFrame(
        np.eye(len(final_samples)), 
        index=final_samples, 
        columns=final_samples
    )

    for i, group_i in enumerate(final_samples):
        for j in range(i + 1, len(final_samples)):
            group_j = final_samples[j]
            members_i = group_map[group_i]
            members_j = group_map[group_j]

            sims = []
            for a in members_i:
                for b in members_j:
                    if a == b:
                        continue
                    idx_a = name_to_index.get(a)
                    idx_b = name_to_index.get(b)
                    if idx_a is not None and idx_b is not None:
                        sims.append(sim_matrix[idx_a, idx_b])

            sim_val = np.mean(sims) if sims else 0.0
            new_matrix.loc[group_i, group_j] = sim_val
            new_matrix.loc[group_j, group_i] = sim_val

    return new_matrix, final_samples


debug = False

# for setting in ['test']:
for setting in ['2019']:
    if setting == '2019':
        file_path = '/itf-fi-ml/shared/users/ziyuzh/svm/data/ppi_full_2019.txt'
    elif setting == '2017':
        file_path = '/itf-fi-ml/shared/users/ziyuzh/svm/data/ppi_full_2016.txt'

    if debug == True:
        G = nx.read_edgelist(file_path, nodetype=str, create_using=nx.Graph())
        center_node = random.choice(list(G.nodes()))
        radius = 1  # you can adjust this
        G = nx.ego_graph(G, center_node, radius=radius)


    else:
        G = nx.read_edgelist(file_path, nodetype=str, create_using=nx.Graph())

    nodes_order = list(G.nodes())
    save_dir = f'/itf-fi-ml/shared/users/ziyuzh/svm/results/df/{setting}'
    os.makedirs(save_dir, exist_ok=True)


    with open('/itf-fi-ml/shared/users/ziyuzh/svm/results/df/2019_map.pkl', 'rb') as f:
        map_info = pickle.load(f) #[merge_groups, delete_ensp, map_dict_aligned]

    for beta in [0.1,0.2,0.5,0.8,1,2]:
    # for beta in [0.1]:
        print('calculate kernel')
        K_full = diffusion_kernel(G, beta)
        K_full = 0.5 * (K_full + K_full.T)
        print('remap')
        new_matrix, final_samples = merge_similarity_matrix(K_full, nodes_order, map_info[0], map_info[1])
        uniport_id_order = pd.Series(final_samples).map(map_info[2]).tolist()
        print('save k and calculate logmk')
        K_full = new_matrix.to_numpy()
        K_full = normalize_kernel(K_full)
        K_full += np.eye(K_full.shape[0]) * 1e-6
        K_full = 0.5 * (K_full + K_full.T)
        
        kernel_path = os.path.join(save_dir, f'uniport_ids_difussion_K_{beta}.pkl')
        with open(kernel_path, 'wb') as f:
            pickle.dump(uniport_id_order, f)

        kernel_path = os.path.join(save_dir, f'uniport_difussion_K_{beta}.pkl')
        with open(kernel_path, 'wb') as f:
            pickle.dump(K_full, f)
            
        logm_k = process_kernel(K_full)
        kernel_path2 = os.path.join(save_dir, f'uniport_difussion_logK_{beta}.pkl')
        with open(kernel_path2, 'wb') as f:
            pickle.dump(logm_k, f)

# # for setting in ['test']:
# for setting in ['2019','2017']:
#     if setting == '2019':
#         file_path = '/itf-fi-ml/shared/users/ziyuzh/svm/data/ppi_full_2019.txt'
#         G = nx.read_edgelist(file_path, nodetype=str, create_using=nx.Graph())
#     elif setting == '2017':
#         file_path = '/itf-fi-ml/shared/users/ziyuzh/svm/data/ppi_full_2016.txt'
#         G = nx.read_edgelist(file_path, nodetype=str, create_using=nx.Graph())
#     elif setting == 'test':
#         n = 10
#         G = nx.erdos_renyi_graph(n=n, p=0.3)
#         mapping = {i: f'gene_{i+1}' for i in G.nodes()}
#         G = nx.relabel_nodes(G, mapping)
#     save_dir = f'/itf-fi-ml/shared/users/ziyuzh/svm/results/df/{setting}'
#     os.makedirs(save_dir, exist_ok=True)

#     for beta in [0.1,0.2,0.5,0.8,1,2]:
#         K_full = diffusion_kernel(G, beta)
#         K_full += np.eye(K_full.shape[0]) * 1e-6

#         K_full = 0.5 * (K_full + K_full.T)
#         kernel_path = os.path.join(save_dir, f'difussion_K_{beta}.pkl')
#         with open(kernel_path, 'wb') as f:
#             pickle.dump(K_full, f)
            
#         logm_k = process_kernel(K_full)
#         kernel_path2 = os.path.join(save_dir, f'difussion_logK_{beta}.pkl')
#         with open(kernel_path2, 'wb') as f:
#             pickle.dump(logm_k, f)