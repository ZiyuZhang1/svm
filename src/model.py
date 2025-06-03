import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import KFold, GridSearchCV
from rdkit.ML.Scoring.Scoring import CalcBEDROC
# from pseudo_label import select_pseudo_negatives
from pseudo2 import select_pseudo_negatives
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.metrics import make_scorer
from scipy.spatial.distance import cdist


# def select_parameter(X_train,y_train):
#         # Define parameter grid
#     param_grid = {
#         'C': [0.1, 0.5, 0.8, 1, 10],
#         'kernel': ['rbf'],
#         'gamma': ['scale', 'auto',0.1,1,10,100]
#     }

#     # Initialize SVM with probability estimates
#     clf = svm.SVC(probability=True)
#     # Perform Grid Search with Cross-Validation using AUC as the scoring metric
#     grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='roc_auc', verbose=1)
#     grid_search.fit(X_train,y_train)
#     return grid_search.best_params_
#     # print("Best cross-validation AUC:", grid_search.best_score_)

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
        grid_search = GridSearchCV(clf, param_grid, cv=5, scoring=custom_scorer, verbose=1)
    elif metric == 'auroc':
        grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='roc_auc', verbose=1)
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
    return (
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

def evaluate_disease(X, y, methods,functions,param_fix):
    result_df = pd.DataFrame(columns=['method',"fold","para", 'top_recall_25','top_recall_300','top_recall_10%', 'top_precision_10%', 'max_precision_10%','top_recall_30%', 'top_precision_30%', 'max_precision_30%','pm_0.5%','pm_1%','pm_5%','pm_10%','pm_15%','pm_20%','pm_25%','pm_30%','auroc',"rank_ratio",'bedroc_1','bedroc_5','bedroc_10','bedroc_30'])
    # Initialize 5-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Get the positive class samples
    X_pos = X[y == 1]
    y_pos = y[y == 1]

    # Loop through each fold
    # X_train_pos, X_test_pos, y_train_pos, y_test_pos = train_test_split(X[y == 1], y[y == 1], test_size=0.2, random_state=42)
    for fold, (train_idx, test_idx) in enumerate(kf.split(X_pos)):
        X_train_pos, X_test_pos = X_pos[train_idx], X_pos[test_idx]
        y_train_pos, y_test_pos = y_pos[train_idx], y_pos[test_idx]

        neg_num = 5*X_train_pos.shape[0]
        if 'ooc' in methods:
            X_test = np.vstack((X_test_pos, X[y == 0]))
            y_test = np.array([1]*X_test_pos.shape[0]+[-1]*X[y == 0].shape[0])

            param_grid = {
                'kernel': ['rbf'],
                'gamma': ['scale', 'auto', 0.1, 0.01, 0.001],
                'nu': [0.5, 0.8, 0.9]
            }

            grid_search = GridSearchCV(svm.OneClassSVM(), param_grid, cv=5, scoring='neg_mean_squared_error')
            grid_search.fit(X_train_pos)
            best_svm = svm.OneClassSVM(**grid_search.best_params_)
            # best_svm = svm.OneClassSVM(nu=0.8,kernel='rbf',gamma='scale')
            result_df.loc[len(result_df.index)] = ["ooc",fold,str(grid_search.best_params_), *eval(best_svm, X_train_pos, y_train_pos, X_test, y_test)]

        if 'random_negative' in methods:
            # print(f'fold {fold}, random neg')
            # Randomly select 'neg_num' indices
            selected_neg_indices = np.random.choice(np.where(y == 0)[0], size=neg_num, replace=False)
            X_train_neg = X[selected_neg_indices]
            y_train_neg = y[selected_neg_indices]
            X_train = np.vstack((X_train_pos, X_train_neg))
            y_train = np.hstack((y_train_pos, y_train_neg))
            # Get the remaining samples by masking the selected indices
            remaining_neg_indices = np.setdiff1d(np.where(y == 0)[0], selected_neg_indices)
            X_test_neg = X[remaining_neg_indices]
            y_test_neg = y[remaining_neg_indices]
            X_test = np.vstack((X_test_pos, X_test_neg))
            y_test = np.hstack((y_test_pos, y_test_neg))
            
            for metric in ['auroc']:
            # for metric in ['bedroc_1','auroc']:
                if param_fix:
                    best_svm = svm.SVC(kernel='rbf',gamma=1,C=0.1)
                    parameters = 'kernel=rbf gamma=1 C=0.1'
                else:
                    parameters = select_parameter(X_train, y_train, metric)
                    best_svm = svm.SVC(**parameters)
                result_df.loc[len(result_df.index)] = ["random_negative"+metric,fold,str(parameters), *eval(best_svm, X_train, y_train, X_test, y_test)]
        if 'pseudo_on_vector' in methods:
            mean_vector = np.mean(X_train_pos, axis=0)
            # Calculate distances from negative samples to mean vector
            neg_indices = np.where(y == 0)[0]
            neg_samples = X[neg_indices]
            distances = cdist(neg_samples, mean_vector.reshape(1, -1)).flatten()
            
            # Select 'neg_num' negative samples farthest from the mean vector
            farthest_neg_indices = neg_indices[np.argsort(distances)[-neg_num:]]
            X_train_neg = X[farthest_neg_indices]
            y_train_neg = y[farthest_neg_indices]
            
            # Combine positive and selected negative samples
            X_train = np.vstack((X_train_pos, X_train_neg))
            y_train = np.hstack((y_train_pos, y_train_neg))
            
            # Prepare test set
            remaining_neg_indices = np.setdiff1d(np.where(y == 0)[0], farthest_neg_indices)
            X_test_neg = X[remaining_neg_indices]
            y_test_neg = y[remaining_neg_indices]
            X_test = np.vstack((X_test_pos, X_test_neg))
            y_test = np.hstack((y_test_pos, y_test_neg))
            for metric in ['auroc']:
                if param_fix:
                    best_svm = svm.SVC(kernel='rbf',gamma=1,C=0.1)
                    parameters = 'kernel=rbf gamma=1 C=0.1'
                else:
                    parameters = select_parameter(X_train, y_train, metric)
                    best_svm = svm.SVC(**parameters)
                result_df.loc[len(result_df.index)] = ["pseudo_on_vector"+metric,fold,str(parameters), *eval(best_svm, X_train, y_train, X_test, y_test)]
        if 'pseudo_labeling' in methods:
            # print(f'fold {fold}, pesudo neg')
            for func in functions:
                # X_train_neg, y_train_neg = select_pseudo_negatives(neg_num, X_train_pos, y_train_pos, X[y==0], y[y==0],func)
                # dtype = X.dtype.descr  # Get data type of elements
                # X_y0_view = X[y == 0].view(dtype=[('', X.dtype)] * X.shape[1])
                # X_train_neg_view = X_train_neg.view(dtype=[('', X.dtype)] * X.shape[1])
                # X_test_neg = np.setdiff1d(X_y0_view, X_train_neg_view).view(X.dtype).reshape(-1, X.shape[1])  

                all_sampled_negatives, X_train_neg, y_train_neg = select_pseudo_negatives(neg_num, X_train_pos, y_train_pos, X[y==0], y[y==0],func)
                select_index = all_sampled_negatives - np.sum(np.hstack((y_train_pos, y[y==0])) == 1) +1
                X_all_neg_add0 = np.vstack((np.zeros((1, X.shape[1])), X[y==0]))
                select_index_add0 = np.hstack((np.zeros((1,)), select_index))
                test_neg_index = np.setdiff1d(np.arange(X_all_neg_add0.shape[0]),select_index_add0)
                X_test_neg = X_all_neg_add0[test_neg_index]

                X_train = np.vstack((X_train_pos, X_train_neg))
                y_train = np.hstack((y_train_pos, y_train_neg))
        

                X_test = np.vstack((X_test_pos, X_test_neg))
                y_test = np.array([1]*X_test_pos.shape[0]+[0]*X_test_neg.shape[0])
                # y_test = np.concatenate((np.ones(X_test_pos.shape[0], dtype=int),np.zeros(X_test_neg.shape[0], dtype=int)))
                for metric in ['auroc']:
                # for metric in ['bedroc_1','auroc']:
                    if param_fix:
                        best_svm = svm.SVC(kernel='rbf',gamma=1,C=0.1)
                        parameters = 'kernel=rbf gamma=1 C=0.1'
                    else:
                        parameters = select_parameter(X_train, y_train, metric)
                        best_svm = svm.SVC(**parameters)
                    result_df.loc[len(result_df.index)] = ["pseudo_labeling"+str(func)+metric,fold,str(parameters), *eval(best_svm, X_train, y_train, X_test, y_test)]
    return result_df