import numpy as np
import pandas as pd
from sklearn import svm
from rdkit.ML.Scoring.Scoring import CalcBEDROC
# from pseudo_label import select_pseudo_negatives
from sklearn.metrics import roc_auc_score
import os
import pickle
# import gseapy as gp
# from concurrent.futures import ProcessPoolExecutor
# import functools
from multiprocessing import Pool
from collections import defaultdict
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import NearestNeighbors
from scipy.linalg import eigh
from sklearn.model_selection import StratifiedKFold
from scipy.stats import rankdata
# import glob

def merge_results(results_list):
    merged = defaultdict(list)
    for res_dict in results_list:
        for key, val_list in res_dict.items():
            merged[key].extend(val_list)  # add all scores to the list
    return dict(merged)

def average_rank_ratio(y_scores, y_test):
    """
    Calculate the average predicted rank of true positives.

    Parameters:
    y_scores (array-like): Decision function scores from the classifier.
    y_test (array-like): True binary labels (0 for negative, 1 for positive).

    Returns:
    float: The average rank of true positives.
    """
    
    # Convert inputs to numpy arrays for consistency
    y_scores = np.array(y_scores)
    y_test = np.array(y_test)

    # Step 1: Sort scores in descending order and assign ranks
    sorted_indices = np.argsort(-y_scores)  # Negative for descending sort
    ranks = np.empty_like(sorted_indices)
    ranks[sorted_indices] = np.arange(1, len(y_scores) + 1)  # Rank starts from 1

    # Step 2: Identify true positives
    true_positive_indices = np.where(y_test == 1)[0]

    # Step 3: Extract ranks of true positives
    true_positive_ranks = ranks[true_positive_indices]

    # Step 4: Calculate the average rank of true positives
    average_rank = np.mean(true_positive_ranks)

    rank_ratio = average_rank/y_test.shape[0]

    return round(rank_ratio,4)

def top_recall_precision(frac,y_scores,y_test):
    if np.sum(y_test==1) == 0:
        return 0,0,0
    else:
        cut = int(len(y_scores)*frac)
        top_30_indices = np.argsort(y_scores)[-cut:][::-1]
        top_30_y_scores = y_scores[top_30_indices]
        top_30_y_test = y_test[top_30_indices]

        TP = np.sum(top_30_y_test == 1)

        recall = TP/np.sum(y_test==1)
        precision = TP/len(top_30_indices)
        max_precision = np.sum(y_test==1)/len(top_30_indices)

    return recall, precision, max_precision


def calculate_er_n(scores, y_test, n):
    """
    Calculate ER_n where the top n predictions are considered positive.
    ER_n = TPR/(TPR+FPR)
    
    Parameters:
    scores - sorted array of [label, score] pairs, highest scores first
    y_test - original labels
    n - number of top predictions to consider
    
    Returns:
    er_n - the ER_n metric value
    """
    # Ensure n doesn't exceed available data
    n = min(n, len(scores))
    
    # Count true positives in top n
    top_n_labels = scores[:n, 0]
    tp_n = np.sum(top_n_labels)
    
    # Calculate TPR and FPR for top n
    total_positives = np.sum(y_test)
    total_negatives = len(y_test) - total_positives
    
    tpr_n = tp_n / total_positives if total_positives > 0 else 0
    fpr_n = (n - tp_n) / total_negatives if total_negatives > 0 else 0
    
    # Calculate ER_n
    er_n = tpr_n / (tpr_n + fpr_n) if (tpr_n + fpr_n) > 0 else 0
    
    return er_n

def eval_bagging(y_scores, y_test):

    rank_ratio = average_rank_ratio(y_scores, y_test)
        
    ############### AUCROC
    if y_scores is not None:
        try:
            auroc = roc_auc_score(y_test, y_scores)
        except:
            auroc = "AUROC computation failed (possibly due to label issues)"
    else:
        auroc = "AUROC not available (no predict_proba or decision_function)"

    
    ############### BEDROC
    scores = np.column_stack((y_test, y_scores))  # Stack labels and scores as columns
    scores = scores[scores[:, 1].argsort()[::-1]]  # Sort by scores in descending order
    ############# top recall
    top_recall_10, top_precision_10, max_precision_10 = top_recall_precision(0.1,y_scores,y_test)
    top_recall_30, top_precision_30, max_precision_30 = top_recall_precision(0.3,y_scores,y_test)
    ############### top recall
    total_positives = np.sum(y_test)
    top_25_positives = np.sum(scores[:25, 0])
    top_300_positives = np.sum(scores[:300, 0])
    
    top_25_recall = top_25_positives / total_positives if total_positives > 0 else 0
    top_300_recall = top_300_positives / total_positives if total_positives > 0 else 0
    return (-y_scores).argsort().argsort(),(
        # recall_score(y_test, y_pred, average="binary", pos_label=1), 
        # precision_score(y_test, y_pred, average="binary", pos_label=1), 
        # f1_score(y_test, y_pred, average="binary", pos_label=1),
        top_25_recall,
        top_300_recall,
        top_recall_10, top_precision_10, max_precision_10,
        top_recall_30, top_precision_30, max_precision_30,
        calculate_er_n(scores, y_test, int(0.005*len(y_test))),
        calculate_er_n(scores, y_test, int(0.01*len(y_test))),
        calculate_er_n(scores, y_test, int(0.05*len(y_test))),
        calculate_er_n(scores, y_test, int(0.1*len(y_test))),
        calculate_er_n(scores, y_test, int(0.15*len(y_test))),
        calculate_er_n(scores, y_test, int(0.20*len(y_test))),
        calculate_er_n(scores, y_test, int(0.25*len(y_test))),
        calculate_er_n(scores, y_test, int(0.30*len(y_test))),
        auroc,
        rank_ratio,
        CalcBEDROC(scores, col=0, alpha=160.9),
        CalcBEDROC(scores, col=0, alpha=32.2),
        CalcBEDROC(scores, col=0, alpha=16.1),
        CalcBEDROC(scores, col=0, alpha=5.3)
    )

with open('/itf-fi-ml/shared/users/ziyuzh/svm/data/uniport_id/uni2name.pkl', 'rb') as file:
    uni2name_dict = pickle.load(file)

def is_spd(A, tol=1e-8):
    # Check symmetry
    if not np.allclose(A, A.T, atol=tol):
        return False
    # Check eigenvalues > 0
    eigvals = np.linalg.eigvalsh(A)
    return np.all(eigvals > tol)

def project_to_spd(A, tol=1e-8):
    # Make symmetric
    A = (A + A.T) / 2
    eigvals, eigvecs = eigh(A)
    eigvals_clipped = np.clip(eigvals, tol, None)  # set eigenvalues < tol to tol
    return eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T

def make_psd(K, min_eig=1e-6):
    K = (K + K.T) / 2
    eigvals = np.linalg.eigvalsh(K)
    if np.min(eigvals) < min_eig:
        K += np.eye(K.shape[0]) * (min_eig - np.min(eigvals))
    return K

def process_kernel(args):
    K= args

    eigenvalues, eigenvectors = np.linalg.eigh(K)
    eigenvalues = np.clip(eigenvalues, 1e-12, None)  # Avoid log(0)
    K_log = eigenvectors @ np.diag(np.log(eigenvalues)) @ eigenvectors.T
    K_log = 0.5 * (K_log + K_log.T)

    return K_log

# def enriched_set(input_ids,time):
#     gene_names = set()
#     for unid in input_ids:
#         gene_list = uni2name_dict.get(unid, [])
#         gene_names.update(gene_list)
#     gene_names = list(gene_names) 
    
#     if time == 2019:
#         enrich_db = ['GO_Biological_Process_2021','GO_Cellular_Component_2021','GO_Molecular_Function_2021','KEGG_2019_Human']
#     elif time == 2017:
#         enrich_db = ['GO_Biological_Process_2021','GO_Cellular_Component_2021','GO_Molecular_Function_2021','KEGG_2016']
#     try:
#         enr = gp.enrichr(
#             gene_list=gene_names,
#             gene_sets=enrich_db,
#             organism='human', 
#             outdir=None
#         )
#         enr_df = enr.results
#         if enr_df is None or enr_df.empty:
#             return set()
        
#         result_terms = enr_df.loc[enr_df['Adjusted P-value'] < 0.01, ['Gene_set', 'Term']]
#         return set(map(tuple, result_terms.values))
    
#     except Exception as e:
#         # Optionally log the error: print(f"Enrichment failed: {e}")
#         return set()

def calculate_jac_sim(enrich_1, enrich_2):
    intersection = enrich_1 & enrich_2
    union = enrich_1 | enrich_2
    if not union:
        return 0.0  # Define similarity as 0 if both sets are empty
    return len(intersection) / len(union)

def compute_kernels(X_feature, feature_id, save_dir,compute_log):

    ratio_list = [2,4,8]
    K_s_path = dict()
    nbrs = NearestNeighbors(n_neighbors=2).fit(X_feature)
    distances, _ = nbrs.kneighbors(X_feature)
    avg_nn_dist = np.mean(distances[:, 1])  # skip self-distance

    for ratio in ratio_list:
        gamma = 1 / (ratio * avg_nn_dist ** 2)
        K_full = rbf_kernel(X_feature, X_feature, gamma=gamma)
        K_full = 0.5 * (K_full + K_full.T)
        kernel_path = os.path.join(save_dir, f'{feature_id}_K_{ratio}_{gamma}.pkl')
        with open(kernel_path, 'wb') as f:
            pickle.dump(K_full, f)

        if compute_log:
            logm_k = process_kernel(K_full)
            kernel_path2 = os.path.join(save_dir, f'{feature_id}_logK_{ratio}_{gamma}.pkl')
            with open(kernel_path2, 'wb') as f:
                pickle.dump(logm_k, f)
            K_s_path[ratio] = [kernel_path,kernel_path2]  # Save only the path, not the matrix
        else:
            K_s_path[ratio] = [kernel_path]

    return feature_id, K_s_path

def normalize_kernel(K):
    diag = np.sqrt(np.diag(K))
    diag[diag == 0] = 1e-8  # Avoid division by zero
    return K / (diag[:, None] * diag[None, :])

def select_gamma_ratio(args):
    neg_df, neg_num, train_pos_df, df, X_dict, fname = args

    train_neg_df = neg_df.sample(n=neg_num, replace=True, random_state=42)
    train_df = pd.concat([train_pos_df, train_neg_df])
    train_index_loc = df.index.get_indexer(train_df.index)
    y_train = np.array([1] * len(train_pos_df) + [0] * len(train_neg_df))

    C_values = [1,3,9,27,81]
    # C_values = [1e-2, 1e-1, 1, 10]
    # gamma_ratios = [2,4,8]
    gamma_ratios = list(X_dict.keys())


    best_bedroc = 0
    best_auc = 0
    best_params = {'C': None, 'gamma': None}

    # Define stratified k-fold
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    for gamma_ratio in gamma_ratios:
        pre_kernel_path = X_dict[gamma_ratio][0]
        with open(pre_kernel_path, 'rb') as f:
            pre_kernel = pickle.load(f)
            pre_kernel = 0.5 * (pre_kernel + pre_kernel.T)
        for C_num in C_values:
            cv_scores = {'auc': [], 'bedroc': []}
            for fold, (train_idx, val_idx) in enumerate(skf.split(train_index_loc, y_train)):
                y_cv_train, y_cv_val = y_train[train_idx], y_train[val_idx]
                if 'diffusion' in pre_kernel_path:
                    kernel_train_idx = df.loc[df.index[train_index_loc[train_idx]], 'diffusion_2019_feature_0'].values.astype(int)
                    kernel_val_idx = df.loc[df.index[train_index_loc[val_idx]], 'diffusion_2019_feature_0'].values.astype(int)

                    X_feature_train = pre_kernel[np.ix_(kernel_train_idx, kernel_train_idx)]
                    X_feature_test = pre_kernel[np.ix_(kernel_val_idx,kernel_train_idx)]
                else:
                    X_feature_train = pre_kernel[np.ix_(train_index_loc[train_idx], train_index_loc[train_idx])]
                    X_feature_test = pre_kernel[np.ix_(train_index_loc[val_idx],train_index_loc[train_idx])]
                best_svm = svm.SVC(C=C_num, kernel='precomputed')
                best_svm.fit(X_feature_train, y_cv_train)
                y_scores = best_svm.decision_function(X_feature_test)
                auroc = roc_auc_score(y_cv_val, y_scores)
                scores = np.column_stack((y_cv_val, y_scores))  # Stack labels and scores as columns
                scores = scores[scores[:, 1].argsort()[::-1]]
                bedroc_10 = CalcBEDROC(scores, col=0, alpha=16.1)
                cv_scores['auc'].append(auroc)
                cv_scores['bedroc'].append(bedroc_10)

            avg_auc = np.mean(cv_scores['auc'])
            avg_bedroc = np.mean(cv_scores['bedroc'])

            if avg_auc > best_auc:
                best_auc = avg_auc
                best_params = {'C_num': C_num, 'gamma_ratio': gamma_ratio, 'gamma':pre_kernel_path.split('_')[-1].replace('.pkl', '')}
                best_bedroc = avg_bedroc
            # if avg_bedroc > best_bedroc:
            #     best_bedroc = avg_bedroc
            #     best_params = {'C_num': C_num, 'gamma_ratio': gamma_ratio, 'gamma':pre_kernel_path.split('_')[-1].replace('.pkl', '')}
            #     best_auc = avg_auc

    return fname, best_params, best_bedroc, best_auc

def select_C(args):
    neg_df, neg_num, train_pos_df, df, pre_kernel, fname = args

    train_neg_df = neg_df.sample(n=neg_num, replace=True, random_state=42)
    train_df = pd.concat([train_pos_df, train_neg_df])
    train_index_loc = df.index.get_indexer(train_df.index)
    y_train = np.array([1] * len(train_pos_df) + [0] * len(train_neg_df))

    C_values = [1,3,9,27,81]

    best_bedroc = 0
    best_auc = 0
    best_params = 0

    # Define stratified k-fold
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    for C_num in C_values:
        cv_scores = {'auc': [], 'bedroc': []}
        for fold, (train_idx, val_idx) in enumerate(skf.split(train_index_loc, y_train)):
            y_cv_train, y_cv_val = y_train[train_idx], y_train[val_idx]
            X_feature_train = pre_kernel[np.ix_(train_index_loc[train_idx], train_index_loc[train_idx])]
            X_feature_test = pre_kernel[np.ix_(train_index_loc[val_idx],train_index_loc[train_idx])]
            best_svm = svm.SVC(C=C_num, kernel='precomputed')
            best_svm.fit(X_feature_train, y_cv_train)
            y_scores = best_svm.decision_function(X_feature_test)
            auroc = roc_auc_score(y_cv_val, y_scores)
            scores = np.column_stack((y_cv_val, y_scores))  # Stack labels and scores as columns
            scores = scores[scores[:, 1].argsort()[::-1]]
            bedroc_10 = CalcBEDROC(scores, col=0, alpha=16.1)
            cv_scores['auc'].append(auroc)
            cv_scores['bedroc'].append(bedroc_10)

        avg_auc = np.mean(cv_scores['auc'])
        avg_bedroc = np.mean(cv_scores['bedroc'])

        if avg_auc > best_auc:
            best_auc = avg_auc
            best_bedroc = avg_bedroc
            best_params = C_num

            

    return fname, best_params, best_bedroc, best_auc

def neg_bagging(args):
    neg_df, neg_num, train_pos_df, df, X_path, C_num, test_index_loc, seed = args
    train_neg_df = neg_df.sample(n=neg_num, replace=True, random_state=seed)
    train_df = pd.concat([train_pos_df, train_neg_df])
    train_index_loc = df.index.get_indexer(train_df.index)
    y_train = np.array([1] * len(train_pos_df) + [0] * len(train_neg_df))

    if isinstance(X_path, str):
        with open(X_path, 'rb') as f:
            X_all = pickle.load(f)
            X_all = 0.5 * (X_all + X_all.T)
    else:
        X_all = X_path

    if 'diffusion' in X_path:
        kernel_train_idx = df.loc[df.index[train_index_loc], 'diffusion_2019_feature_0'].values.astype(int)
        kernel_val_idx = df.loc[df.index[test_index_loc], 'diffusion_2019_feature_0'].values.astype(int)

        X_feature_train = X_all[np.ix_(kernel_train_idx, kernel_train_idx)]
        X_feature_test = X_all[np.ix_(kernel_val_idx,kernel_train_idx)]
    else:

        X_feature_train = X_all[np.ix_(train_index_loc, train_index_loc)]
        X_feature_test = X_all[np.ix_(test_index_loc,train_index_loc)]

    best_svm = svm.SVC(C=C_num, kernel='precomputed')
    best_svm.fit(X_feature_train, y_train)
    y_scores = best_svm.decision_function(X_feature_test)
    overlap = set(train_index_loc)&set(test_index_loc)
    mask_loc = [np.where(test_index_loc == i)[0][0] for i in overlap]

    return y_scores, mask_loc

def one_fold_evaluate(disease, time, feature_list, df,y,train_idx,test_idx,methods,result_df,fold):
    train_pos_df = df.loc[train_idx]
    test_pos_df = df.loc[test_idx]
    neg_num = 5*len(train_pos_df)
    neg_df = df[y == 0]
    neg_df_add_test_pos = pd.concat([neg_df, test_pos_df])

    if 'random_negative' in methods:
        # kernel_dir_path = os.path.join('/itf-fi-ml/shared/users/ziyuzh/svm/results/dw_auc',str(time))
        # kernel_dir_path = os.path.join('/itf-fi-ml/shared/users/ziyuzh/svm/results/dw_auc_norm_test',str(time))

        kernel_dir_path = os.path.join('/itf-fi-ml/shared/users/ziyuzh/svm/results/dw_auc_norm',str(time))
        
        os.makedirs(kernel_dir_path, exist_ok=True)
        kernel_pkl_path = os.path.join(kernel_dir_path,'path_save.pkl')

        if os.path.isfile(kernel_pkl_path):
            print('kernels existing')
            with open(kernel_pkl_path, 'rb') as f:
                kernels_all_dict = pickle.load(f)
        else:
            kernels_all_dict = dict()

        add_feature_list = set(feature_list) - set(kernels_all_dict.keys())
        if not add_feature_list:
            pass
        else:
            add_feature_list = list(add_feature_list)
        ####### calculate full kernels for each feature and their logm
            print('calculating kernels...', add_feature_list)
            X_all = []
            
            for feature_name in add_feature_list:
                select_columns = [col for col in df.columns if col.startswith(feature_name)]
                X_all.append(df[select_columns].values)

            args_list = list(zip(X_all, add_feature_list, [kernel_dir_path] * len(X_all), [True] * len(X_all)))
            with Pool(min(len(add_feature_list), os.cpu_count(), 4)) as pool:
                # each tuple (X_feature, feature_id) is unpacked by starmap
                kernel_results = pool.starmap(
                    compute_kernels,
                    args_list)
            del X_all
            
            for fname, K_s_path_dict in kernel_results:
                kernels_all_dict[fname] = K_s_path_dict
                
            with open(kernel_pkl_path, 'wb') as f:
                pickle.dump(kernels_all_dict, f)
      ############################## cv get best gamma
        args_list = [(neg_df, neg_num, train_pos_df, df, kernels_all_dict[fname], fname)
            for fname in feature_list]

        with Pool(processes=len(feature_list)) as pool:
            best_ratios = pool.map(select_gamma_ratio, args_list)

        best_ratios_dict = dict()
        agg_feature = []
        for fname, best_params, best_bedroc, best_auc in best_ratios:
            print(fname, best_params, best_bedroc, best_auc)
            best_ratios_dict[fname] = best_params
            # if best_auc > 0.67 and best_bedroc > 0.5:
            agg_feature.append(fname)
        print('collect valid feature: ', agg_feature)
      ######################### using precalculated kernels to train svm and evaluate, get weights for kernels
        # print('evaluation')

        test_neg_df = neg_df
        test_df = pd.concat([test_pos_df, test_neg_df])
        test_index_loc = df.index.get_indexer(test_df.index)
        y_test = np.array([1] * len(test_pos_df) + [0] * len(test_neg_df))


        # # test_indices = test_df.index.values
        # # enrich_train_genes = train_pos_df.index.values
        # # enrich_train_set = enriched_set(enrich_train_genes,time)

        num_processes = 20
        base_seed = 42
        seed_list = [base_seed + i for i in range(num_processes)]

        # pathway_overlap_dict = dict()
        
        rank_results_per_feature = dict()
        predcition_collection = dict()
        predcition_collection['true_label'] = y_test
        predcition_collection["test_genes"] = test_df.index
        predcition_collection["train_pos_genes"] = train_pos_df.index

        for feature_name in feature_list:
            gamma = best_ratios_dict[feature_name]['gamma_ratio']
            X_path = kernels_all_dict[feature_name][gamma][0]
            C_num = best_ratios_dict[feature_name]['C_num']

            args_list = [
                (neg_df_add_test_pos, neg_num, train_pos_df, df, X_path, C_num, test_index_loc, seed)
                for seed in seed_list]

            # Step 2: Use Pool to parallelize
            with Pool(processes=num_processes) as pool:
                bagging_y_scores_with_mask = pool.map(neg_bagging, args_list)

            arrays = np.stack([arr for arr, _ in bagging_y_scores_with_mask])          # shape: (n, d)

            # Build mask matrix (True = masked / skip)
            mask = np.zeros_like(arrays, dtype=bool)
            for i, (_, m) in enumerate(bagging_y_scores_with_mask):
                mask[i, m] = True

            # Invert mask: True where we keep
            keep = ~mask

            # Compute sum and count efficiently
            sum_arr = np.where(keep, arrays, 0).sum(axis=0)
            count_arr = keep.sum(axis=0)

            final_y_score = np.array(sum_arr / count_arr)

            # final_y_score = np.mean(bagging_y_scores, axis=0)

            # pathway_overlap_dict[feature_name] = jac_sm

            ranked_predict_index, results = eval_bagging(final_y_score, y_test)
            # Add results to the result dataframe
            result_df.loc[len(result_df.index)] = ["random_negative",fold,feature_name+'-0-0-0', *results]
            rank_results_per_feature[feature_name] = rankdata(final_y_score, method='average')
            predcition_collection[feature_name] = final_y_score
        # # ################################# early fusion for now remove early fusion because not sure how to merge diffusion kernels into this stage
        # if len(agg_feature) > 0:
        #     print('early fusion')

        #     diff = any('diffusion' in f for f in agg_feature)
        #     if diff:
        #         # Load diffusion data only once if needed
        #         with open('/itf-fi-ml/shared/users/ziyuzh/svm/results/df/2019/uniport_diffusion_K_2.pkl', 'rb') as f:
        #             diffusion_data = pickle.load(f)
        #     else:
        #         diffusion_data = None

        #     # Collect columns matching any feature in agg_feature (excluding diffusion)
        #     select_columns = [
        #         col for f in agg_feature if 'diffusion' not in f
        #         for col in df.columns if col.startswith(f)
        #     ]
        #     X_concat = df[select_columns].values

        #     if diff:
        #         # Concatenate with diffusion_data column-wise
        #         X_concat = np.hstack([X_concat, diffusion_data])

        #     # train_neg_df = neg_df_add_test_pos.sample(n=neg_num, replace=True, random_state=42)
        #     # train_df = pd.concat([train_pos_df, train_neg_df])
        #     # train_index_loc = df.index.get_indexer(train_df.index)
        #     # y_train = np.array([1] * len(train_pos_df) + [0] * len(train_neg_df))
        #     # X_concat_train = X_concat[train_index_loc]

        #     feature_names = '-'.join(agg_feature)
        #     early_dir = '/itf-fi-ml/shared/users/ziyuzh/svm/results/early_concat'
        #     early_files = glob.glob(os.path.join(early_dir, f'{feature_names}*.pkl'))
            
        #     feature_names, K_path = compute_kernels(X_concat, feature_names, early_dir, False)

        #     ## read corresponding file and cv get parametrs
        #     args = neg_df_add_test_pos, neg_num, train_pos_df, df, K_path, feature_names
        #     feature_names, best_params, best_bedroc, best_auc = select_gamma_ratio(args)
        #     print(feature_names, best_params, best_bedroc, best_auc)
        #     ## evaluation
        #     gamma = best_params['gamma_ratio']
        #     X_path = K_path[gamma][0]
        #     C_num = best_params['C_num']

        #     args_list = [
        #         (neg_df_add_test_pos, neg_num, train_pos_df, df, X_path, C_num, test_index_loc, seed)
        #         for seed in seed_list]

        #     # Step 2: Use Pool to parallelize
        #     with Pool(processes=num_processes) as pool:
        #         bagging_y_scores = pool.map(neg_bagging, args_list)

        #     final_y_score = np.mean(bagging_y_scores, axis=0)
        #     predcition_collection['early_fusion'] = final_y_score
        #     enrich_test_genes = test_indices[np.argsort(final_y_score)[::-1]][:200]
        #     enrich_feature_test = enriched_set(enrich_test_genes,time)
        #     jac_sm = calculate_jac_sim(enrich_train_set,enrich_feature_test)

        #     ranked_predict_index, results = eval_bagging(final_y_score, y_test)
        #     # Add results to the result dataframe
        #     result_df.loc[len(result_df.index)] = ["random_negative",fold,'early_fusion-'+str(round(jac_sm, 3)), *results]
        # else:
        #     print('no valid features, no early fusion')
        ################################## later fusion
        # if len(agg_feature) > 0:
        #     print('later fusion')
        #     selected_ranks = np.array([rank_results_per_feature[fname] for fname in agg_feature])  # Shape: (num_features, num_elements)
        #     selected_weights = np.array([pathway_overlap_dict[fname] for fname in agg_feature])    # Shape: (num_features,)

        #     if np.sum(selected_weights) == 0:
        #         norm_weights = np.ones_like(selected_weights) / len(selected_weights)
        #         print("Warning: All selected weights are zero; using uniform weights.")
        #     else:
        #         norm_weights = selected_weights / np.sum(selected_weights)

        #     weighted_rank = np.sum(norm_weights[:, np.newaxis] * selected_ranks, axis=0)
        #     enrich_test_genes = test_indices[np.argsort(weighted_rank)[::-1]][:200]
        #     enrich_feature_test = enriched_set(enrich_test_genes,time)
        #     jac_sm = calculate_jac_sim(enrich_train_set,enrich_feature_test)
        #     ranked_predict_index, results = eval_bagging(weighted_rank, y_test)
        #     result_df.loc[len(result_df.index)] = ["random_negative",fold,'later_weighted_rank'+'-'+str(round(jac_sm, 3)), *results]

        #     avg_rank = np.mean(selected_ranks, axis=0)
        #     enrich_test_genes = test_indices[np.argsort(avg_rank)[::-1]][:200]
        #     enrich_feature_test = enriched_set(enrich_test_genes,time)
        #     jac_sm = calculate_jac_sim(enrich_train_set,enrich_feature_test)
        #     ranked_predict_index, results = eval_bagging(avg_rank, y_test)
        #     result_df.loc[len(result_df.index)] = ["random_negative",fold,'later_avg_rank'+'-'+str(round(jac_sm, 3)), *results]
        # else:
        #     print('no valid features, no later fusion')
        ############################################ kernel fusion
        if len(agg_feature) > 0:
            print('middle fusion')
            print('kernel fusion')
            ks = []
            logks = []
            for fname in agg_feature:
                gamma = best_ratios_dict[fname]['gamma_ratio']
                X_k_path = kernels_all_dict[fname][gamma][0]
                with open(X_k_path, 'rb') as f:
                    X_k = pickle.load(f)
                if 'diffusion' in fname:
                    perm = df['diffusion_2019_feature_0'].values.astype(int)
                    X_k_reorder = X_k[np.ix_(perm, perm)]
                    ks.append(X_k_reorder)
                else:
                    ks.append(X_k)

                X_logk_path = kernels_all_dict[fname][gamma][1]
                with open(X_logk_path, 'rb') as f:
                    X_logk = pickle.load(f)
                if 'diffusion' in fname:
                    perm = df['diffusion_2019_feature_0'].values.astype(int)
                    X_logk_reorder = X_logk[np.ix_(perm, perm)]
                    logks.append(X_logk_reorder)
                else:
                    logks.append(X_logk) 

                # print(X_k_path,X_logk_path)          

            K_linear_fused = np.mean(ks, axis=0)
            K_linear_fused = 0.5 * (K_linear_fused + K_linear_fused.T)
            K_linear_fused = normalize_kernel(K_linear_fused)
            kernels_all_dict['linear_fused'] = K_linear_fused
            del K_linear_fused

            logk_avg = np.mean(logks, axis=0)
            eigenvalues, eigenvectors = np.linalg.eigh(logk_avg)
            eigenvalues = np.clip(eigenvalues, -50, 50)  # Prevent overflow
            K_geo_mean = eigenvectors @ np.diag(np.exp(eigenvalues)) @ eigenvectors.T
            K_geo_mean = 0.5 * (K_geo_mean + K_geo_mean.T)  # Enforce symmetry
            K_geo_mean = normalize_kernel(K_geo_mean)
            kernels_all_dict['geo_fused'] = K_geo_mean
            del K_geo_mean

            # ######################### weighted kernels
            # pathway_overlap = []
            # for fname in agg_feature:
            #     pathway_overlap.append(pathway_overlap_dict[fname])
            # total = sum(pathway_overlap)
            # if total == 0:
            #     feature_weights = [1.0 / len(pathway_overlap)] * len(pathway_overlap)
            #     print("Warning: All pathway overlaps are zero; using uniform weights.")
            # else:
            #     feature_weights = [v / total for v in pathway_overlap]

            # first_K = ks[0]
            # K_weight_linear_all = np.zeros(first_K.shape, dtype=first_K.dtype)
            # expm_sum = np.zeros_like(logks[0])
            # # Accumulate
            # for f_index, K_i in enumerate(ks):
            #     weight = feature_weights[f_index]
            #     K_weight_linear_all += weight * K_i
            #     expm_sum     += weight * logks[f_index]

            # K_weight_linear_all = 0.5 * (K_weight_linear_all + K_weight_linear_all.T)
            # K_weight_linear_all = normalize_kernel(K_weight_linear_all)
            # kernels_all_dict['weighted_linear_fused'] = K_weight_linear_all
            # del K_weight_linear_all

            # # Geometric mean via matrix exponential
            # eigenvalues, eigenvectors = np.linalg.eigh(expm_sum)
            # eigenvalues = np.clip(eigenvalues, -50, 50)  # Prevent overflow
            # K_weight_geo_all = eigenvectors @ np.diag(np.exp(eigenvalues)) @ eigenvectors.T
            # K_weight_geo_all = 0.5 * (K_weight_geo_all + K_weight_geo_all.T)  # Enforce symmetry
            # K_weight_geo_all = normalize_kernel(K_weight_geo_all)
            # kernels_all_dict['weighted_geo_fused'] = K_weight_geo_all
            # del K_weight_geo_all
            ################################### cv choose best C for all kernels
            print('grid search C for fused kernels')
            # fusion_methods = ['linear_fused','geo_fused','weighted_linear_fused','weighted_geo_fused']
            fusion_methods = ['linear_fused','geo_fused']


            args_list = [(neg_df, neg_num, train_pos_df, df, kernels_all_dict[fname], fname)
                for fname in fusion_methods]

            with Pool(processes=len(fusion_methods)) as pool:
                best_Cs = pool.map(select_C, args_list)
            C_dict = dict()
            for fname, best_params, best_bedroc, best_auc in best_Cs:
                print(fname, best_params, best_bedroc, best_auc)
                C_dict[fname] = best_params
            ###########################################################
            print('fused kernels evaluation')
            for fusion_method in fusion_methods:
                C_num = C_dict[fusion_method]
                X_all = kernels_all_dict[fusion_method]
                args_list = [
                    (neg_df_add_test_pos, neg_num, train_pos_df, df, X_all, C_num, test_index_loc, seed)
                    for seed in seed_list]

                # Step 2: Use Pool to parallelize
                with Pool(processes=num_processes) as pool:
                    # bagging_y_scores = pool.map(neg_bagging, args_list)
                    bagging_y_scores_with_mask = pool.map(neg_bagging, args_list)

                arrays = np.stack([arr for arr, _ in bagging_y_scores_with_mask])          # shape: (n, d)

                # Build mask matrix (True = masked / skip)
                mask = np.zeros_like(arrays, dtype=bool)
                for i, (_, m) in enumerate(bagging_y_scores_with_mask):
                    mask[i, m] = True

                # Invert mask: True where we keep
                keep = ~mask

                # Compute sum and count efficiently
                sum_arr = np.where(keep, arrays, 0).sum(axis=0)
                count_arr = keep.sum(axis=0)

                final_y_score = np.array(sum_arr / count_arr)

                # final_y_score = np.mean(bagging_y_scores, axis=0)
                predcition_collection[fusion_method] = final_y_score
                ranked_predict_index, results = eval_bagging(final_y_score, y_test)


                # Add results to the result dataframe
                result_df.loc[len(result_df.index)] = ["random_negative",fold, fusion_method+'-0-0-0', *results]
        else:
            print('no valid features, no mid fusion')
        
        return predcition_collection
def evaluate_disease(disease, time, feature_list, df, y, methods,time_spilt):
    result_df = pd.DataFrame(columns=['method',"fold","para", 'top_recall_25','top_recall_300','top_recall_10%', 'top_precision_10%', 'max_precision_10%','top_recall_30%', 'top_precision_30%', 'max_precision_30%','pm_0.5%','pm_1%','pm_5%','pm_10%','pm_15%','pm_20%','pm_25%','pm_30%','auroc',"rank_ratio",'bedroc_1','bedroc_5','bedroc_10','bedroc_30'])
    
    if time_spilt:
        test_idx = df[df['test']==1].index
        train_idx = df[y==1].index.difference(test_idx)
        df.drop(columns='test', inplace=True)
        predcition_collection = one_fold_evaluate(disease, time, feature_list, df,y,train_idx,test_idx,methods,result_df,1)
        return result_df, predcition_collection
    # else:
    #     kf = KFold(n_splits=5, shuffle=True, random_state=42)
    #     for fold, (train_id, test_id) in enumerate(kf.split(df[y == 1].index)):
    #         train_idx = df[y == 1].index[train_id]
    #         test_idx = df[y == 1].index[test_id]
    #         one_fold_evaluate(disease, feature_list, df,y,train_idx,test_idx,methods,result_df,fold)                    
    #     return result_df
