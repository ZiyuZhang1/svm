import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import KFold, GridSearchCV
from rdkit.ML.Scoring.Scoring import CalcBEDROC
# from pseudo_label import select_pseudo_negatives
from pseudo_reindex import select_pseudo_negatives, select_pseudo_negatives_mask
from sklearn.metrics import roc_auc_score
from sklearn.metrics import make_scorer
import os
import pickle
import gseapy as gp
from scipy.cluster.hierarchy import fcluster
# from concurrent.futures import ProcessPoolExecutor
# import functools
from multiprocessing import Pool
from functools import partial
from collections import defaultdict
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import NearestNeighbors
from scipy.linalg import logm, expm, eigh

def merge_results(results_list):
    merged = defaultdict(list)
    for res_dict in results_list:
        for key, val_list in res_dict.items():
            merged[key].extend(val_list)  # add all scores to the list
    return dict(merged)


# Define the custom scoring function
def custom_score(y_true, y_proba):
    scores = np.column_stack((y_true, y_proba))  
    scores = scores[scores[:, 1].argsort()[::-1]] 
    return CalcBEDROC(scores, col=0, alpha=160.9)  

# Wrap the custom_score function using make_scorer
custom_scorer = make_scorer(custom_score, response_method='predict_proba')

def select_parameter(X_train, y_train, metric):
    # Define parameter grid
    param_grid = {
        'C': [0.1, 0.5, 0.8, 1, 10],
        'kernel': ['rbf'],
        'gamma': ['scale', 'auto', 0.1, 1, 10, 100]
    }
    
    # Initialize SVM with probability estimates
    clf = svm.SVC(probability=True)
    
    if metric == 'bedroc_1':
        grid_search = GridSearchCV(clf, param_grid, cv=3, scoring=custom_scorer, verbose=0)
    elif metric == 'auroc':
        grid_search = GridSearchCV(clf, param_grid, cv=3, scoring='roc_auc', verbose=0)
    grid_search.fit(X_train, y_train)
    # print(metric,grid_search.best_params_, grid_search.best_score_)
    return grid_search.best_params_

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
    return np.argsort(y_scores)[::-1],(
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

def eval(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)
    # y_pred = clf.predict(X_test)
    y_test[y_test == -1] = 0
    if hasattr(clf, "decision_function"):
        y_scores = clf.decision_function(X_test)
    elif hasattr(clf, "predict_proba"):
        y_scores = clf.predict_proba(X_test)[:, 1]
    else:
        y_scores = None  # AUROC cannot be computed without scores

    rank_ratio = average_rank_ratio(y_scores, y_test)
    
    # if -1 in y_test:
    #     y_test = np.where(y_test == -1, 0, y_test)
    # if -1 in y_pred:
    #     y_pred = np.where(y_pred == -1, 0, y_pred)
        
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
    return np.argsort(y_scores)[::-1],(
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

with open('/itf-fi-ml/shared/users/ziyuzh/svm/data/stringdb/2023/name_convert.pkl', 'rb') as file:
    loaded_data = pickle.load(file)
stringId2name,name2stringId,aliases2stringId = loaded_data
del name2stringId,aliases2stringId

def string_convert(gene):
    if gene in name2stringId.keys():
        return name2stringId[gene]
    elif gene in aliases2stringId.keys():
        return aliases2stringId[gene]
    else:
        return None

def get_top_n_predictions(ranked_predict_index,test_indices,n):
    genes = []
    sorted_indices = np.argsort(ranked_predict_index)
    top_10_indices = sorted_indices[:n]
    test_indices[top_10_indices]
    for value in test_indices[top_10_indices]:
        genes.append(stringId2name[value])
    return genes

def single_bagging_neg(neg_df,neg_num,train_pos_df,test_pos_df, _):
    result_dict_temp = dict()
    # Randomly select 'neg_num' samples from negative class
    train_neg_df = neg_df.sample(n=neg_num)
    
    # Get the remaining negative samples
    test_neg_df = neg_df
    
    # Combine positive and negative samples for training
    train_df = pd.concat([train_pos_df, train_neg_df])
    X_train = train_df.values
    y_train = np.array([1] * len(train_pos_df) + [0] * len(train_neg_df))
    
    # Store original indices for training set
    # train_indices = train_df.index.values
    
    # Combine positive and negative samples for testing
    test_df = pd.concat([test_pos_df, test_neg_df])
    X_test = test_df.values
    # y_test = np.array([1] * len(test_pos_df) + [0] * len(test_neg_df))

    # Store original indices for test set
    test_indices = test_df.index.values  
    metric = 'auroc'
    # Select parameters and train the model
    parameters = select_parameter(X_train, y_train, metric)
    best_svm = svm.SVC(**parameters)
    best_svm.fit(X_train, y_train)
    y_scores = best_svm.decision_function(X_test)

    for arrayindex, gene in enumerate(test_indices):
        if gene in result_dict_temp:
            result_dict_temp[gene].append(y_scores[arrayindex])
        else:
            result_dict_temp[gene] = [y_scores[arrayindex]]
    return result_dict_temp

def single_bagging_pos_neg(disease, neg_df,neg_num,train_pos_df,test_pos_df, _):
    result_dict_temp = dict()

    scores_df = pd.read_csv('/itf-fi-ml/shared/users/ziyuzh/svm/data/disgent_2020/timecut/disgent_with_time.csv')

    sub_df = scores_df[scores_df['disease_id'] == disease]

    # Filter only rows where string_id is in sampled_ids
    filtered = sub_df[sub_df['string_id'].isin(train_pos_df.index)]

    # Reindex to match the order of sampled_ids
    filtered = filtered.set_index('string_id').loc[train_pos_df.index]

    # Get the score column as probabilities
    probabilities = filtered['score'].values

    if probabilities.sum() == 0:
        probabilities = np.ones_like(probabilities) / len(probabilities)
    else:
        probabilities = probabilities / probabilities.sum()

    sampled_indices = np.random.choice(train_pos_df.index, size=len(train_pos_df), replace=True, p=probabilities)
    train_pos_df_ran = train_pos_df.loc[sampled_indices]

    # Randomly select 'neg_num' samples from negative class
    train_neg_df = neg_df.sample(n=neg_num)
    
    # Get the remaining negative samples
    test_neg_df = neg_df
    
    # Combine positive and negative samples for training
    train_df = pd.concat([train_pos_df_ran, train_neg_df])
    X_train = train_df.values
    y_train = np.array([1] * len(train_pos_df_ran) + [0] * len(train_neg_df))
    
    # Store original indices for training set
    # train_indices = train_df.index.values
    
    # Combine positive and negative samples for testing
    test_df = pd.concat([test_pos_df, test_neg_df])
    X_test = test_df.values
    # y_test = np.array([1] * len(test_pos_df) + [0] * len(test_neg_df))

    # Store original indices for test set
    test_indices = test_df.index.values  
    metric = 'auroc'
    # Select parameters and train the model
    parameters = select_parameter(X_train, y_train, metric)
    best_svm = svm.SVC(**parameters)
    best_svm.fit(X_train, y_train)
    y_scores = best_svm.decision_function(X_test)

    for arrayindex, gene in enumerate(test_indices):
        if gene in result_dict_temp:
            result_dict_temp[gene].append(y_scores[arrayindex])
        else:
            result_dict_temp[gene] = [y_scores[arrayindex]]
    return result_dict_temp

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

def weighted_log_euclidean_mean(kernels, weights):
    weights = np.array(weights)
    weights = weights / weights.sum()
    
    log_sum = np.zeros_like(kernels[0])
    for w, K in zip(weights, kernels):
        K = make_psd(K)
        log_K = logm(K)
        # Handle potential complex results more carefully
        if np.iscomplexobj(log_K) and np.allclose(log_K.imag, 0, atol=1e-10):
            log_K = log_K.real
        log_sum += w * log_K
    
    return expm(log_sum)

def enriched_set(input_stringids,time):
    gene_names = [stringId2name.get(sid) for sid in input_stringids if stringId2name.get(sid) is not None]
    if time == 2019:
        enrich_db = ['GO_Biological_Process_2021','GO_Cellular_Component_2021','GO_Molecular_Function_2021','KEGG_2019_Human']
    elif time == 2017:
        enrich_db = ['GO_Biological_Process_2021','GO_Cellular_Component_2021','GO_Molecular_Function_2021','KEGG_2016']
    try:
        enr = gp.enrichr(
            gene_list=gene_names,
            gene_sets=enrich_db,
            organism='human', 
            outdir=None
        )
        enr_df = enr.results
        if enr_df is None or enr_df.empty:
            return set()
        
        result_terms = enr_df.loc[enr_df['Adjusted P-value'] < 0.01, ['Gene_set', 'Term']]
        return set(map(tuple, result_terms.values))
    
    except Exception as e:
        # Optionally log the error: print(f"Enrichment failed: {e}")
        return set()

def calculate_jac_sim(enrich_1, enrich_2):
    intersection = enrich_1 & enrich_2
    union = enrich_1 | enrich_2
    if not union:
        return 0.0  # Define similarity as 0 if both sets are empty
    return len(intersection) / len(union)

def one_fold_evaluate(disease, feature_list, df,y,train_idx,test_idx,methods,result_df,fold):
    train_pos_df = df.loc[train_idx]
    test_pos_df = df.loc[test_idx]
    neg_num = 5*len(train_pos_df)

    weight_features = True

    if 'linear_fused' in feature_list:
        feature_list.remove('linear_fused')
    if 'weighted_linear_fused' in feature_list:
        feature_list.remove('weighted_linear_fused')
    if 'geo_fused' in feature_list:
        feature_list.remove('geo_fused')
    init_feature_length = len(feature_list)

    if 'random_negative' in methods:

        # Work with DataFrames to maintain indices
        neg_df = df[y == 0]

        # Randomly select 'neg_num' samples from negative class
        train_neg_df = neg_df.sample(n=neg_num, random_state=42)

        # Get the all negative samples
        test_neg_df = neg_df

        # Combine positive and negative samples for training
        train_df = pd.concat([train_pos_df, train_neg_df])
        test_df = pd.concat([test_pos_df, test_neg_df])

        X_train_mats = []
        X_test_mats = []
        for feature_name in feature_list:
            select_columns = [col for col in df.columns if col.startswith(feature_name)]
            X_train_mats.append(train_df[select_columns].values)
            X_test_mats.append(test_df[select_columns].values)

        y_train = np.array([1] * len(train_pos_df) + [0] * len(train_neg_df))
        y_test = np.array([1] * len(test_pos_df) + [0] * len(test_neg_df))

        kernels_all = []
        kernels_train = []
        kernels_test = []


        # For each feature set
        for X_tr, X_te in zip(X_train_mats, X_test_mats):
            X_all = np.concatenate([X_tr, X_te], axis=0)

            nbrs = NearestNeighbors(n_neighbors=2).fit(X_all)
            distances, _ = nbrs.kneighbors(X_all)
            avg_nn_dist = np.mean(distances[:, 1])  # skip self-distance
            gamma = 1 / (2 * avg_nn_dist ** 2)
            K_full = rbf_kernel(X_all, X_all, gamma=gamma)
            kernels_all.append(K_full)

            n_train = len(X_tr)
            kernels_train.append(K_full[:n_train, :n_train])
            kernels_test.append(K_full[n_train:, :n_train])


        time = 2019
        test_indices = test_df.index.values
        enrich_train_genes = train_pos_df.index.values
        enrich_train_set = enriched_set(enrich_train_genes,time)

        pathway_overlap = []
        for feature_index, feature_name in enumerate(feature_list):
            best_svm = svm.SVC(kernel='precomputed')
            best_svm.fit(kernels_train[feature_index], y_train)
            y_scores = best_svm.decision_function(kernels_test[feature_index])

            enrich_test_genes = test_indices[np.argsort(y_scores)[::-1]][:200]
            enrich_feature_test = enriched_set(enrich_test_genes,time)
            jac_sm = calculate_jac_sim(enrich_train_set,enrich_feature_test)
            pathway_overlap.append(jac_sm)

            ranked_predict_index, results = eval_bagging(y_scores, y_test)
            # Add results to the result dataframe
            result_df.loc[len(result_df.index)] = ["random_negative",fold,feature_name+'-'+str(round(jac_sm, 3)), *results]


        total = sum(pathway_overlap)
        feature_weights = [v / total for v in pathway_overlap]

        K_weight_linear_all = sum(w * K_train_i for w, K_train_i in zip(feature_weights, kernels_all))
        kernels_train.append(K_weight_linear_all[:n_train, :n_train])
        kernels_test.append(K_weight_linear_all[n_train:, :n_train])
        feature_list.append('weighted_linear_fused')

        # K_geo_all = weighted_log_euclidean_mean(kernels_all, feature_weights)
        # kernels_train.append(K_geo_all[:n_train, :n_train])
        # kernels_test.append(K_geo_all[n_train:, :n_train])
        # feature_list.append('geo_fused')

        feature_weights = [1 / len(feature_list)] * len(feature_list)
        K_linear_all = sum(w * K_train_i for w, K_train_i in zip(feature_weights, kernels_all))
        kernels_train.append(K_linear_all[:n_train, :n_train])
        kernels_test.append(K_linear_all[n_train:, :n_train])
        feature_list.append('linear_fused')        

        for feature_name in feature_list[init_feature_length:]:
            feature_index = feature_list.index(feature_name)
            best_svm = svm.SVC(kernel='precomputed')
            best_svm.fit(kernels_train[feature_index], y_train)
            y_scores = best_svm.decision_function(kernels_test[feature_index])
            ranked_predict_index, results = eval_bagging(y_scores, y_test)
            # Add results to the result dataframe
            result_df.loc[len(result_df.index)] = ["random_negative",fold,feature_name, *results]


    # if 'random_negative_bagging' in methods:
    #     # Work with DataFrames to maintain indices
    #     neg_df = df[y == 0]
    #     result_dict_lists = []
    #     for seed in [42,43,44]:
    #         result_dict_temp = dict()
    #         # Randomly select 'neg_num' samples from negative class
    #         train_neg_df = neg_df.sample(n=neg_num,replace=True, random_state=seed)
            
    #         # Get the all negative samples
    #         test_neg_df = neg_df
            
    #         # Combine positive and negative samples for training
    #         train_df = pd.concat([train_pos_df, train_neg_df])
    #         test_df = pd.concat([test_pos_df, test_neg_df])

    #         X_train_mats = []
    #         X_test_mats = []
    #         for feature_name in feature_list:
    #             select_columns = [col for col in df.columns if col.startswith(feature_name)]
    #             X_train_mats.append(train_df[select_columns].values)
    #             X_test_mats.append(test_df[select_columns].values)

    #         feature_weights = [1 / len(feature_list)] * len(feature_list)

    #         y_train = np.array([1] * len(train_pos_df) + [0] * len(train_neg_df))
    #         y_test = np.array([1] * len(test_pos_df) + [0] * len(test_neg_df))

    #         # Store gamma per feature set
    #         gammas = []
    #         kernels_train = []
    #         kernels_test = []

    #         # For each feature set
    #         for X_tr, X_te in zip(X_train_mats, X_test_mats):
    #             K_train, best_gamma = get_best_gamma_kernel(X_tr, y_train)
    #             K_test = rbf_kernel(X_te, X_tr, gamma=best_gamma)

    #             if not is_spd(K_train): K_train = project_to_spd(K_train)

    #             kernels_train.append(K_train)
    #             kernels_test.append(K_test)

    #         # Fuse
    #         K_fused_train = sum(w * K_train_i for w, K_train_i in zip(feature_weights, kernels_train))
    #         K_fused_test  = sum(w * K_test_i for w, K_test_i in zip(feature_weights, kernels_test))

    #         kernels_train.append(K_fused_train)
    #         kernels_test.append(K_fused_test)
    #         feature_list.append('fused')

    #         # Store original indices for training set
    #         train_indices = train_df.index.values
    #         # Store original indices for test set
    #         test_indices = test_df.index.values

    #         for feature_index, feature_name in enumerate(feature_list):

    #             best_svm = svm.SVC(kernel='precomputed')
    #             best_svm.fit(kernels_train[feature_index], y_train)
    #             y_scores = best_svm.decision_function(kernels_test[feature_index])

    #             for arrayindex, gene in enumerate(test_indices):
    #                 key_name = feature_name+'_'+gene
    #                 if key_name in result_dict_temp:
    #                     result_dict_temp[key_name].append(y_scores[arrayindex])
    #                 else:
        
    #                     result_dict_temp[key_name] = [y_scores[arrayindex]]
    #         result_dict_lists.append(result_dict_temp)

    #     result_dict = merge_results(result_dict_lists)
    #     result_averages = {key: np.mean(values) for key, values in result_dict.items()}

    #     for feature_name in feature_list:
    #         feature_result_dict = {k: v for k, v in result_averages.items() if feature_name in str(k)}
    #         feature_gene_id = [item.split('_')[1] for item in feature_result_dict if '_' in item]
    #         ranked_predict_index, results = eval_bagging(np.array(list(feature_result_dict.values())), y[df.index.get_indexer(feature_gene_id)])
    #         # Add results to the result dataframe
    #         result_df.loc[len(result_df.index)] = [
    #             "random_negative",
    #             fold,
    #             feature_list[feature_index], 
    #             *results
    #         ]

def evaluate_disease(disease, feature_list, df, y, methods,time_spilt):
    result_df = pd.DataFrame(columns=['method',"fold","para", 'top_recall_25','top_recall_300','top_recall_10%', 'top_precision_10%', 'max_precision_10%','top_recall_30%', 'top_precision_30%', 'max_precision_30%','pm_0.5%','pm_1%','pm_5%','pm_10%','pm_15%','pm_20%','pm_25%','pm_30%','auroc',"rank_ratio",'bedroc_1','bedroc_5','bedroc_10','bedroc_30'])
    
    if time_spilt:
        test_idx = df[df['test']==1].index
        train_idx = df[y==1].index.difference(test_idx)
        df.drop(columns='test', inplace=True)
        one_fold_evaluate(disease, feature_list, df,y,train_idx,test_idx,methods,result_df,1)
        return result_df
    else:
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        for fold, (train_id, test_id) in enumerate(kf.split(df[y == 1].index)):
            train_idx = df[y == 1].index[train_id]
            test_idx = df[y == 1].index[test_id]
            one_fold_evaluate(disease, feature_list, df,y,train_idx,test_idx,methods,result_df,fold)                    
        return result_df
