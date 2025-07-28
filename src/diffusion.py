import networkx as nx
import numpy as np
from scipy.linalg import expm
import pickle
import os
import mygene
import pandas as pd

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

def get_map_df(ensembl_ids,input_type):
    mg = mygene.MyGeneInfo()
    # Query mygene for UniProt and Entrez gene ID mappings
    results = mg.querymany(
        ensembl_ids,
        scopes=input_type,
        fields='uniprot,entrezgene',
        species='human'
    )

    results_df = pd.DataFrame(results)
    results_df = results_df[~results_df['entrezgene'].isna()]
    results_df['uniprot_ids'] = results_df['uniprot'].apply(
        lambda x: list(x.values())[0] if isinstance(x, dict) and 'Swiss-Prot' in x else None)
    results_df = results_df[~results_df['uniprot_ids'].isna()]
    results_df[results_df['uniprot_ids'].apply(lambda x: isinstance(x, list) and len(x) > 1)]
    return results_df

def diffusion_kernel(G, beta, normalized=True):

    if normalized:
        L = nx.normalized_laplacian_matrix(G).toarray()
    else:
        L = nx.laplacian_matrix(G).toarray()
    
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



# for setting in ['test']:
for setting in ['2019','2017']:
    if setting == '2019':
        file_path = '/itf-fi-ml/shared/users/ziyuzh/svm/data/ppi_full_2019.txt'
    elif setting == '2017':
        file_path = '/itf-fi-ml/shared/users/ziyuzh/svm/data/ppi_full_2016.txt'

    G = nx.read_edgelist(file_path, nodetype=str, create_using=nx.Graph())
    ensps = list(G.nodes())
    del G

    ppi_ids_map = get_map_df([s.split('.')[1] for s in ensps],'ensembl.protein')
    ppi_set = set()
    for values in ppi_ids_map['uniprot_ids']:
        if isinstance(values, list) and len(values) > 1:
            ppi_set.update(values)  # Add all elements in the list
        else:
            ppi_set.add(values)

    string_ids = []
    one2more = []
    more2one = []  # to collect subdfs with multiple or zero matches

    for uniport_ids in list(ppi_set):
        subdf = ppi_ids_map[ppi_ids_map['uniprot_ids'].str.contains(uniport_ids, na=False)]
        
        if len(subdf) == 1:
            if isinstance(subdf['uniprot_ids'], list) and len(values) > 1:
                one2more.append(subdf)
            else:
                string_ids.append(uniport_ids)
        else:
            more2one.append(subdf)

    more2one_df = pd.concat(more2one, ignore_index=True)
    more2one_df['string_id'] = '9606.'+more2one_df['query']
    merge_dict = more2one_df.groupby('uniprot_ids')['string_id'].apply(list).to_dict()
    map_dict = dict()
    merge_groups = []
    for key in merge_dict:
        merge_groups.append(merge_dict[key])
        new_key = '_'.join(sorted(merge_dict[key]))
        map_dict[new_key] = key
    flat_set = {item for sublist in merge_groups for item in sublist}
    unique_ids = ['9606.' + ensp_id for ensp_id in ppi_ids_map[ppi_ids_map['uniprot_ids'].isin(string_ids)]['query'].tolist()]
    delete_list = list(set(ensps) - flat_set - set(unique_ids))
    sample_names = ensps

    save_dir = f'/itf-fi-ml/shared/users/ziyuzh/svm/results/df/{setting}'

    for file in os.listdir(save_dir):
        if 'difussion_K' in file:
            k_path = os.path.join(save_dir,file)
            with open(k_path, 'rb') as f:
                sim_matrix = pickle.load(f)
            new_matrix, final_samples = merge_similarity_matrix(sim_matrix, sample_names, merge_groups, delete_list)
            new_matrix.rename(index=map_dict, columns=map_dict, inplace=True)
            kernel_ids = list(new_matrix.index)

            K_full = new_matrix.to_numpy()
            K_full = normalize_kernel(K_full)
            K_full += np.eye(K_full.shape[0]) * 1e-6
            K_full = 0.5 * (K_full + K_full.T)
            
            kernel_path = os.path.join(save_dir, 'uniport_ids_'+file)
            with open(kernel_path, 'wb') as f:
                pickle.dump(kernel_ids, f)

            kernel_path = os.path.join(save_dir, 'uniport_'+file)
            with open(kernel_path, 'wb') as f:
                pickle.dump(K_full, f)
                
            logm_k = process_kernel(K_full)
            kernel_path2 = os.path.join(save_dir, 'uniport_difussion_logK_'+file.split('_')[-1])
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