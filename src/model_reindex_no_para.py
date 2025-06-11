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

def eval_assemble(y_test,y_scores):
    
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
    ############# top recall percentage
    top_recall_10, top_precision_10, max_precision_10 = top_recall_precision(0.1,y_scores,y_test)
    top_recall_30, top_precision_30, max_precision_30 = top_recall_precision(0.3,y_scores,y_test)
    ############### top recall
    total_positives = np.sum(y_test)
    top_25_positives = np.sum(scores[:25, 0])
    top_300_positives = np.sum(scores[:300, 0])
    
    top_25_recall = top_25_positives / total_positives if total_positives > 0 else 0
    top_300_recall = top_300_positives / total_positives if total_positives > 0 else 0
    return (
        # recall_score(y_test, y_pred, average="binary", pos_label=1), 
        # precision_score(y_test, y_pred, average="binary", pos_label=1), 
        # f1_score(y_test, y_pred, average="binary", pos_label=1),
        top_25_recall,
        top_300_recall,
        top_recall_10, top_precision_10, max_precision_10,
        top_recall_30, top_precision_30, max_precision_30,
        auroc,
        rank_ratio,
        CalcBEDROC(scores, col=0, alpha=160.9),
        CalcBEDROC(scores, col=0, alpha=32.2),
        CalcBEDROC(scores, col=0, alpha=16.1),
        CalcBEDROC(scores, col=0, alpha=5.3)
    )
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

def string_convert(gene):
    if gene in name2stringId.keys():
        return name2stringId[gene]
    elif gene in aliases2stringId.keys():
        return aliases2stringId[gene]
    else:
        return None

def mask_enrich(df,train_pos):
    enr = gp.enrichr(gene_list=train_pos.map(stringId2name).tolist(),
                    gene_sets=['KEGG_2021_Human'],
                    organism='human', 
                    outdir=None, 
                    )
    enr_df = enr.results
    gene_lists = enr_df[enr_df['Adjusted P-value']<0.01]['Genes'].to_list()
    mask_gene_pool = set()
    for items in gene_lists:
        genes = items.split(';')
        mask_gene_pool.update(set(genes))
    mask_index = df[df.index.isin([string_convert(gene) for gene in list(mask_gene_pool)])].index
    return mask_index

def mask_ppi_loop(df,train_pos, threshold):
    ppi_connection = pd.read_csv(os.path.join('/itf-fi-ml/shared/users/ziyuzh/svm/data/stringdb/2023','9606.protein.links.v12.0.txt'), sep=' ', header=0).convert_dtypes().replace(0, float('nan'))
    ppi_connection_med = ppi_connection[ppi_connection['combined_score']>threshold]
    first_loop_genes = set(ppi_connection_med[ppi_connection_med['protein1'].isin(list(train_pos))]['protein2'].tolist())
    mask_index = df[df.index.isin(first_loop_genes)].index
    return mask_index

def get_top_n_predictions(ranked_predict_index,test_indices,n):
    genes = []
    sorted_indices = np.argsort(ranked_predict_index)
    top_10_indices = sorted_indices[:n]
    test_indices[top_10_indices]
    for value in test_indices[top_10_indices]:
        genes.append(stringId2name[value])
    return genes

def single_bagging_neg(neg_df,neg_num,train_pos_df,test_pos_df, seed, _):
    result_dict_temp = dict()
    # Randomly select 'neg_num' samples from negative class
    train_neg_df = neg_df.sample(n=neg_num,replace=True, random_state=seed)
    
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

def single_bagging_pos_neg(disease, neg_df,neg_num,train_pos_df,test_pos_df, seed, _):
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

    np.random.seed(seed)

    sampled_indices = np.random.choice(train_pos_df.index, size=len(train_pos_df), replace=True, p=probabilities)
    train_pos_df_ran = train_pos_df.loc[sampled_indices]

    # Randomly select 'neg_num' samples from negative class
    train_neg_df = neg_df.sample(n=neg_num, replace=True, random_state=seed)
    
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

# def process_neg_bagging_iteration(iter_num, neg_df, neg_num, train_pos_df, test_pos_df, select_parameter, metric='auroc'):
#     """Process a single iteration of the random negative bagging"""
#     # Randomly select 'neg_num' samples from negative class
#     train_neg_df = neg_df.sample(n=neg_num)
    
#     # Get the remaining negative samples (unchanged)
#     test_neg_df = neg_df
    
#     # Combine positive and negative samples for training
#     train_df = pd.concat([train_pos_df, train_neg_df])
#     X_train = train_df.values
#     y_train = np.array([1] * len(train_pos_df) + [0] * len(train_neg_df))
    
#     # Store original indices for training set
#     train_indices = train_df.index.values
    
#     # Combine positive and negative samples for testing
#     test_df = pd.concat([test_pos_df, test_neg_df])
#     X_test = test_df.values
#     y_test = np.array([1] * len(test_pos_df) + [0] * len(test_neg_df))

#     # Store original indices for test set
#     test_indices = test_df.index.values  
    
#     # Select parameters and train the model
#     parameters = select_parameter(X_train, y_train, metric)
#     best_svm = svm.SVC(**parameters)
#     best_svm.fit(X_train, y_train)
#     y_scores = best_svm.decision_function(X_test)
    
#     # Return results for this iteration with gene indices
#     return [(gene, y_scores[arrayindex]) for arrayindex, gene in enumerate(test_indices)]


def one_fold_evaluate(disease, df,y,train_idx,test_idx,methods,result_df,fold):
    train_pos_df = df.loc[train_idx]
    test_pos_df = df.loc[test_idx]
    neg_num = 5*len(train_pos_df)
    if 'ooc' in methods:
        # Create test set by combining positive test samples and all negative samples
        # Using DataFrames to maintain indices
        neg_df = df[y == 0]
        
        # Convert to numpy arrays for modeling but keep track of original indices
        X_test_pos_np = test_pos_df.values
        X_neg_np = neg_df.values
        X_test = np.vstack((X_test_pos_np, X_neg_np))
        
        # Create labels for evaluation
        y_test = np.array([1] * len(test_pos_df) + [-1] * len(neg_df))
        
        # Keep track of original indices for the test set
        test_indices = np.concatenate([test_pos_df.index.values, neg_df.index.values])
        
        param_grid = {
            'kernel': ['rbf'],
            'gamma': ['scale', 'auto', 0.1, 0.01, 0.001],
            'nu': [0.5, 0.8, 0.9]
        }
        
        # Use train_pos_df values for training
        X_train_pos_np = train_pos_df.values
        y_train_pos_np = np.ones(len(train_pos_df))
        
        grid_search = GridSearchCV(svm.OneClassSVM(), param_grid, cv=3, scoring='neg_mean_squared_error')
        grid_search.fit(X_train_pos_np)
        best_svm = svm.OneClassSVM(**grid_search.best_params_)
        ranked_predict_index, results = eval(best_svm, X_train_pos_np, y_train_pos_np, X_test, y_test)
        
        # For evaluation, you can use the original indices if needed
        result_df.loc[len(result_df.index)] = [
            "ooc", 
            fold, 
            str(grid_search.best_params_), 
            *results
        ]
        
        # If you need to map predictions back to the original dataframe:
        # print(get_top_n_predictions(ranked_predict_index,test_indices,10))

    # if 'random_negative' in methods:
    #     # Work with DataFrames to maintain indices
    #     neg_df = df[y == 0]
        
    #     # Randomly select 'neg_num' samples from negative class
    #     train_neg_df = neg_df.sample(n=neg_num, random_state=42)
        
    #     # Get the remaining negative samples
    #     test_neg_df = neg_df.drop(train_neg_df.index)
        
    #     # Combine positive and negative samples for training
    #     train_df = pd.concat([train_pos_df, train_neg_df])
    #     X_train = train_df.values
    #     y_train = np.array([1] * len(train_pos_df) + [0] * len(train_neg_df))
        
    #     # Store original indices for training set
    #     train_indices = train_df.index.values
        
    #     # Combine positive and negative samples for testing
    #     test_df = pd.concat([test_pos_df, test_neg_df])
    #     X_test = test_df.values
    #     y_test = np.array([1] * len(test_pos_df) + [0] * len(test_neg_df))
        
    #     # Store original indices for test set
    #     test_indices = test_df.index.values
        
    #     for metric in ['auroc']:
    #         # Select parameters and train the model
    #         parameters = select_parameter(X_train, y_train, metric)
    #         best_svm = svm.SVC(**parameters)
    #         ranked_predict_index, results = eval(best_svm, X_train, y_train, X_test, y_test)
            
    #         # Add results to the result dataframe
    #         result_df.loc[len(result_df.index)] = [
    #             "random_negative" + metric,
    #             fold,
    #             str(parameters), 
    #             *results
    #         ]

    if 'random_negative' in methods:
        # Work with DataFrames to maintain indices
        neg_df = df[y == 0]
        
        # Randomly select 'neg_num' samples from negative class
        train_neg_df = neg_df.sample(n=neg_num, random_state=42)
        
        # Get the all negative samples
        test_neg_df = neg_df
        
        # Combine positive and negative samples for training
        train_df = pd.concat([train_pos_df, train_neg_df])
        X_train = train_df.values
        y_train = np.array([1] * len(train_pos_df) + [0] * len(train_neg_df))
        
        # Store original indices for training set
        train_indices = train_df.index.values
        
        # Combine positive and negative samples for testing
        test_df = pd.concat([test_pos_df, test_neg_df])
        X_test = test_df.values
        y_test = np.array([1] * len(test_pos_df) + [0] * len(test_neg_df))
        
        # Store original indices for test set
        test_indices = test_df.index.values
        # print('neg_ran',len(test_indices))
        for metric in ['auroc']:
            # Select parameters and train the model
            parameters = select_parameter(X_train, y_train, metric)
            best_svm = svm.SVC(**parameters)
            ranked_predict_index, results = eval(best_svm, X_train, y_train, X_test, y_test)
            
            # Add results to the result dataframe
            result_df.loc[len(result_df.index)] = [
                "random_negative" + metric,
                fold,
                str(parameters), 
                *results
            ]

    if 'random_negative_bagging' in methods:
        # Work with DataFrames to maintain indices
        neg_df = df[y == 0]
        result_dict = dict()

        for iters in range(20):

            seed = iters + 42

            # Randomly select 'neg_num' samples from negative class
            train_neg_df = neg_df.sample(n=neg_num, replace=True, random_state=seed)

            # Get the remaining negative samples
            test_neg_df = neg_df

            # Combine positive and negative samples for training
            train_df = pd.concat([train_pos_df, train_neg_df])
            X_train = train_df.values
            y_train = np.array([1] * len(train_pos_df) + [0] * len(train_neg_df))

            # Store original indices for training set
            train_indices = train_df.index.values

            # Combine positive and negative samples for testing
            test_df = pd.concat([test_pos_df, test_neg_df])
            X_test = test_df.values
            y_test = np.array([1] * len(test_pos_df) + [0] * len(test_neg_df))

            # Store original indices for test set
            test_indices = test_df.index.values  
            metric = 'auroc'
            # Select parameters and train the model
            parameters = select_parameter(X_train, y_train, metric)
            best_svm = svm.SVC(**parameters)
            best_svm.fit(X_train, y_train)
            y_scores = best_svm.decision_function(X_test)

            for arrayindex, gene in enumerate(test_indices):
                if gene in result_dict:
                    result_dict[gene].append(y_scores[arrayindex])
                else:
                    result_dict[gene] = [y_scores[arrayindex]]

        result_averages = {key: np.mean(values) for key, values in result_dict.items()}
        # print('neg_ran_bagging',len(result_averages))
        ranked_predict_index, results = eval_bagging(np.array(list(result_averages.values())), y[df.index.get_indexer(list(result_averages.keys()))])

        # Add results to the result dataframe
        result_df.loc[len(result_df.index)] = [
            "random_negative_bagging" + metric,
            fold,
            str(parameters), 
            *results
        ]
        
    if 'random_pos_negative_bagging' in methods:
        # Work with DataFrames to maintain indices
        neg_df = df[y == 0]
        result_dict = dict()

        for iters in range(20):

            seed = iters + 42

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

            np.random.seed(seed)

            sampled_indices = np.random.choice(train_pos_df.index, size=len(train_pos_df), replace=True, p=probabilities)
            train_pos_df_ran = train_pos_df.loc[sampled_indices]

            # Randomly select 'neg_num' samples from negative class
            train_neg_df = neg_df.sample(n=neg_num, replace=True, random_state=seed)
            
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
                if gene in result_dict:
                    result_dict[gene].append(y_scores[arrayindex])
                else:
                    result_dict[gene] = [y_scores[arrayindex]]

        result_averages = {key: np.mean(values) for key, values in result_dict.items()}
        # print('neg_ran_bagging',len(result_averages))
        ranked_predict_index, results = eval_bagging(np.array(list(result_averages.values())), y[df.index.get_indexer(list(result_averages.keys()))])

        # Add results to the result dataframe
        result_df.loc[len(result_df.index)] = [
            "random_pos_negative_bagging" + metric,
            fold,
            str(parameters), 
            *results
        ]


    if 'pseudo_labeling' in methods:
        # Get negative samples with their indices
        neg_df = df[y == 0]
        X_neg = neg_df.values
        y_neg = np.zeros(len(neg_df))
        neg_indices = neg_df.index.values
        
        # Get positive training samples
        X_train_pos_np = train_pos_df.values
        y_train_pos_np = np.ones(len(train_pos_df))
        
        for func in [1]:
            # Use the modified function that returns indices
            all_sampled_combined_indices, neg_relative_indices, X_train_neg, y_train_neg = select_pseudo_negatives(
                neg_num, X_train_pos_np, y_train_pos_np, X_neg, y_neg, func, neg_indices
            )
            
            # Get original indices of selected negative samples
            selected_neg_original_indices = neg_indices[neg_relative_indices]
            
            # Create test set - get indices of negative samples not used in training
            test_neg_indices = np.setdiff1d(neg_indices, selected_neg_original_indices)
            X_test_neg = neg_df.loc[test_neg_indices].values
            
            # Create training set
            X_train = np.vstack((X_train_pos_np, X_train_neg))
            y_train = np.hstack((y_train_pos_np, y_train_neg))
            
            # Store original indices for training and test sets
            train_indices = np.concatenate([train_pos_df.index.values, selected_neg_original_indices])
            test_indices = np.concatenate([test_pos_df.index.values, test_neg_indices])
            
            # Create test set
            X_test_pos_np = test_pos_df.values
            X_test = np.vstack((X_test_pos_np, X_test_neg))
            y_test = np.array([1] * len(X_test_pos_np) + [0] * len(X_test_neg))
            
            for metric in ['auroc']:
                parameters = select_parameter(X_train, y_train, metric)
                best_svm = svm.SVC(**parameters)
                ranked_predict_index, results = eval(best_svm, X_train, y_train, X_test, y_test)
                
                # Add results to the result dataframe
                result_df.loc[len(result_df.index)] = [
                    "pseudo_labeling" + str(func) + metric,
                    fold,
                    str(parameters), 
                    *results
                ]
    if 'pseudo_labeling_mask' in methods:
        # Get negative samples with their indices
        neg_df = df[y == 0]
        X_neg = neg_df.values
        y_neg = np.zeros(len(neg_df))
        neg_indices = neg_df.index.values
        
        # Get positive training samples
        X_train_pos_np = train_pos_df.values
        y_train_pos_np = np.ones(len(train_pos_df))

        # mask_ids = mask_enrich(df,train_pos_df.index).tolist()
        # print(len(mask_ids))
        mask_ids = mask_ppi_loop(df,train_pos_df.index, 900).tolist()

        # mask_ids.extend(mask_ppi_loop(df,train_pos_df.index, 900).tolist())
        # print(len(mask_ids))
        
        
        for func in [1]:
            # Use the modified function that returns indices
            all_sampled_combined_indices, neg_relative_indices, X_train_neg, y_train_neg = select_pseudo_negatives_mask(
                neg_num, X_train_pos_np, y_train_pos_np, X_neg, y_neg, func, neg_indices
            )
            
            # Get original indices of selected negative samples
            selected_neg_original_indices = neg_indices[neg_relative_indices]
            
            # Create test set - get indices of negative samples not used in training
            test_neg_indices = np.setdiff1d(neg_indices, selected_neg_original_indices)
            X_test_neg = neg_df.loc[test_neg_indices].values
            
            # Create training set
            X_train = np.vstack((X_train_pos_np, X_train_neg))
            y_train = np.hstack((y_train_pos_np, y_train_neg))
            
            # Store original indices for training and test sets
            train_indices = np.concatenate([train_pos_df.index.values, selected_neg_original_indices])
            test_indices = np.concatenate([test_pos_df.index.values, test_neg_indices])
            
            # Create test set
            X_test_pos_np = test_pos_df.values
            X_test = np.vstack((X_test_pos_np, X_test_neg))
            y_test = np.array([1] * len(X_test_pos_np) + [0] * len(X_test_neg))
            
            for metric in ['auroc']:
                parameters = select_parameter(X_train, y_train, metric)
                best_svm = svm.SVC(**parameters)
                ranked_predict_index, results = eval(best_svm, X_train, y_train, X_test, y_test)
                
                # Add results to the result dataframe
                result_df.loc[len(result_df.index)] = [
                    "pseudo_labeling_mask" + str(func) + metric,
                    fold,
                    str(parameters), 
                    *results
                ]

    if 'pseudo_labeling_cluster_all_mask' in methods:
        # Get negative samples with their indices
        neg_df = df[y == 0]
        X_neg = neg_df.values
        y_neg = np.zeros(len(neg_df))
        neg_indices = neg_df.index.values

        X_train_pos_np = train_pos_df.values
        y_train_pos_np = np.ones(len(train_pos_df))
        
        with open('/itf-fi-ml/shared/users/ziyuzh/baseline/src/linkage_matrix.pkl', 'rb') as f:
            map_index , mat = pickle.load(f)
        map_index = np.array(map_index)
        n_clusters = 5
        hierarchical_labels = fcluster(mat, n_clusters, criterion='maxclust')

        mask_ids = mask_enrich(df,train_pos_df.index).tolist()
        # print(len(mask_ids))
        # mask_ids = mask_ppi_loop(df,train_pos_df.index, 900).tolist()
        mask_ids.extend(mask_ppi_loop(df,train_pos_df.index, 900).tolist())
        # print(len(mask_ids))
        mask_ids.extend(test_pos_df.index.values.tolist())
        
        
        for func in [1]:
            selected_neg_original_indices = []
            a = dict()
            for cluster_id in np.unique(hierarchical_labels):
                cluster_samples = np.where(hierarchical_labels == cluster_id)[0]
                positive_i = len(set(df[y == 1].index.values)&set(map_index[cluster_samples]))
                a[cluster_id] = len(cluster_samples)*(1-(positive_i/len(df[y == 1])))
            
            for cluster_id in np.unique(hierarchical_labels):
                neg_sample_size = int(neg_num*a[cluster_id]/sum(a.values()))
                cluster_samples = np.where(hierarchical_labels == cluster_id)[0]
                filtered = map_index[cluster_samples]
                neg_pool = filtered[~np.isin(filtered, mask_ids)]
                selected_neg_original_indices.extend(np.random.choice(neg_pool, size=neg_sample_size, replace=False).tolist())
 
            selected_neg_original_indices = np.array(selected_neg_original_indices)
            # Create test set - get indices of negative samples not used in training
            test_neg_indices = np.setdiff1d(neg_indices, selected_neg_original_indices)
            X_test_neg = neg_df.loc[test_neg_indices].values
            
            existing_indices = neg_df.index.intersection(selected_neg_original_indices)
            X_train_neg = neg_df.loc[existing_indices].values

            # Create training set
            X_train = np.vstack((X_train_pos_np, X_train_neg))
            y_train = np.array([1] * len(X_train_pos_np) + [0] * len(X_train_neg))
            
            # Store original indices for training and test sets
            train_indices = np.concatenate([train_pos_df.index.values, selected_neg_original_indices])
            test_indices = np.concatenate([test_pos_df.index.values, test_neg_indices])
            
            # Create test set
            X_test_pos_np = test_pos_df.values
            X_test = np.vstack((X_test_pos_np, X_test_neg))
            y_test = np.array([1] * len(X_test_pos_np) + [0] * len(X_test_neg))
            
            for metric in ['auroc']:
                parameters = select_parameter(X_train, y_train, metric)
                best_svm = svm.SVC(**parameters)
                ranked_predict_index, results = eval(best_svm, X_train, y_train, X_test, y_test)
                
                # Add results to the result dataframe
                result_df.loc[len(result_df.index)] = [
                    "pseudo_labeling_cluster_all_mask" + str(func) + metric,
                    fold,
                    str(parameters), 
                    *results
                ]

def evaluate_disease(disease, df, y, methods,time_spilt):
    result_df = pd.DataFrame(columns=['method',"fold","para", 'top_recall_25','top_recall_300','top_recall_10%', 'top_precision_10%', 'max_precision_10%','top_recall_30%', 'top_precision_30%', 'max_precision_30%','pm_0.5%','pm_1%','pm_5%','pm_10%','pm_15%','pm_20%','pm_25%','pm_30%','auroc',"rank_ratio",'bedroc_1','bedroc_5','bedroc_10','bedroc_30'])
    
    if time_spilt:
        test_idx = df[df['test']==1].index
        train_idx = df[y==1].index.difference(test_idx)
        df.drop(columns='test', inplace=True)
        one_fold_evaluate(disease, df,y,train_idx,test_idx,methods,result_df,1)
        return result_df
    else:
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        for fold, (train_id, test_id) in enumerate(kf.split(df[y == 1].index)):
            train_idx = df[y == 1].index[train_id]
            test_idx = df[y == 1].index[test_id]
            one_fold_evaluate(disease, df,y,train_idx,test_idx,methods,result_df,fold)                    
        return result_df
