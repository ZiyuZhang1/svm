import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from rdkit.ML.Scoring.Scoring import CalcBEDROC
import pickle

def _collect_feature_blocks(frame, feature_list):
    blocks = []
    for feature_name in feature_list:
        cols = [col for col in frame.columns if col.startswith(feature_name)]
        if cols:
            blocks.append(frame[cols].values.astype(np.float32))
    if not blocks:
        raise ValueError("No matching feature columns found for the provided feature list.")
    return np.concatenate(blocks, axis=1)


def _fit_random_forest(X, y, seed, **params):
    base_params = {
        "n_estimators": 500,
        "max_depth": None,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
    }
    model_params = {**base_params, **params}
    model = RandomForestClassifier(
        **model_params,
        class_weight={0: 1, 1: 5},
        random_state=seed,
        n_jobs=1,
    )
    model.fit(X, y)
    return model


def _train_with_validation(X, y, seed):
    y = y.astype(int)
    unique_labels = np.unique(y)
    if unique_labels.size < 2 or X.shape[0] < 5:
        model = _fit_random_forest(X, y, seed)
        return model, np.nan

    pos, neg = np.sum(y == 1), np.sum(y == 0)
    max_splits = min(3, pos, neg)
    if max_splits < 2:
        model = _fit_random_forest(X, y, seed)
        return model, np.nan

    param_grid = [
        {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 1, "max_features": "sqrt"},
        {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 2, "max_features": "sqrt"},
        {"n_estimators": 500, "max_depth": None, "min_samples_leaf": 1, "max_features": "sqrt"},
        {"n_estimators": 500, "max_depth": 20, "min_samples_leaf": 1, "max_features": "sqrt"},
        {"n_estimators": 400, "max_depth": None, "min_samples_leaf": 1, "max_features": None},
    ]

    skf = StratifiedKFold(n_splits=max_splits, shuffle=True, random_state=seed)
    best_auc = -np.inf
    best_params = None

    for params in param_grid:
        fold_aucs = []
        for fold_idx, (tr_idx, vl_idx) in enumerate(skf.split(X, y)):
            X_tr, X_vl = X[tr_idx], X[vl_idx]
            y_tr, y_vl = y[tr_idx], y[vl_idx]

            model = _fit_random_forest(X_tr, y_tr, seed + fold_idx, **params)
            try:
                val_probs = model.predict_proba(X_vl)[:, 1]
                auc = roc_auc_score(y_vl, val_probs)
                if np.isfinite(auc):
                    fold_aucs.append(auc)
            except ValueError:
                continue

        if not fold_aucs:
            continue

        mean_auc = float(np.mean(fold_aucs))
        if mean_auc > best_auc:
            best_auc = mean_auc
            best_params = params

    if best_params is None:
        model = _fit_random_forest(X, y, seed)
        return model, np.nan

    full_model = _fit_random_forest(X, y, seed, **best_params)
    return full_model, best_auc


def later_fusion_train(train_matrix, labels, seed):
    """Train a lightweight ensemble to fuse per-feature predictions."""
    fusion_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_leaf=1,
        class_weight={0: 1, 1: 5},
        random_state=seed,
        n_jobs=1,
    )
    fusion_model.fit(train_matrix, labels)
    return fusion_model


def neg_bagging_early(args):
    neg_df, neg_num, train_pos_df, df, y, feature_list, test_index_loc, seed = args
    train_neg_df = neg_df.sample(n=neg_num, replace=True, random_state=seed)
    train_df = pd.concat([train_pos_df, train_neg_df])

    y_train = np.concatenate([
        np.ones(len(train_pos_df), dtype=int),
        np.zeros(len(train_neg_df), dtype=int),
    ])

    X_train = _collect_feature_blocks(train_df, feature_list)
    model, val_auc = _train_with_validation(X_train, y_train, seed)

    test_df = df.iloc[test_index_loc]
    X_test = _collect_feature_blocks(test_df, feature_list)
    preds = model.predict_proba(X_test)[:, 1]
    return preds, val_auc


def neg_bagging_mid(args):
    return neg_bagging_early(args)


def neg_bagging_later(args):
    neg_df, neg_num, train_pos_df, df, y, feature_list, test_index_loc, seed = args
    train_neg_df = neg_df.sample(n=neg_num, replace=True, random_state=seed)
    train_df = pd.concat([train_pos_df, train_neg_df])

    y_train = np.concatenate([
        np.ones(len(train_pos_df), dtype=int),
        np.zeros(len(train_neg_df), dtype=int),
    ])

    test_df = df.iloc[test_index_loc]

    feature_preds = {}
    feature_train_preds = {}
    auc_records = {}
    fusion_candidates = {}
    train_preds = []
    fusion_feature_names = []

    for feature_name in feature_list:
        cols = [col for col in train_df.columns if col.startswith(feature_name)]
        if not cols:
            continue

        X_train = train_df[cols].values.astype(np.float32)
        model, val_auc = _train_with_validation(X_train, y_train, seed)
        auc_records[feature_name] = val_auc

        X_test = test_df[cols].values.astype(np.float32)
        preds = model.predict_proba(X_test)[:, 1]
        preds_path = f'/itf-fi-ml/shared/users/ziyuzh/svm/results/temp_pred/{seed}{feature_name}.pkl'
        with open(preds_path, "wb") as f:
            pickle.dump(preds, f)
        print('save file:', preds_path)
        feature_preds[feature_name] = preds_path
        feature_train_preds[feature_name] = model.predict_proba(X_train)[:, 1]

        if not np.isnan(val_auc):
            fusion_candidates[feature_name] = preds
            train_preds.append(feature_train_preds[feature_name])
            fusion_feature_names.append(feature_name)

    fused_preds = None
    if fusion_candidates:
        fused_matrix = np.vstack(list(fusion_candidates.values()))
        fused_preds = fused_matrix.mean(axis=0)

    lf_fusion_preds = None
    if fusion_feature_names:
        train_matrix = np.column_stack(train_preds).astype(np.float32)
        fusion_model = later_fusion_train(train_matrix, y_train, seed)
        test_matrix = np.column_stack([fusion_candidates[name] for name in fusion_feature_names]).astype(np.float32)
        lf_fusion_preds = fusion_model.predict_proba(test_matrix)[:, 1]

    return feature_preds, fused_preds, auc_records, lf_fusion_preds


def average_rank_ratio(y_scores, y_test):
    y_scores = np.asarray(y_scores)
    y_test = np.asarray(y_test)

    sorted_indices = np.argsort(-y_scores)
    ranks = np.empty_like(sorted_indices)
    ranks[sorted_indices] = np.arange(1, len(y_scores) + 1)

    true_positive_indices = np.where(y_test == 1)[0]
    if true_positive_indices.size == 0:
        return np.nan

    true_positive_ranks = ranks[true_positive_indices]
    average_rank = np.mean(true_positive_ranks)
    rank_ratio = average_rank / y_test.shape[0]
    return round(rank_ratio, 4)


def top_recall_precision(frac, y_scores, y_test):
    y_scores = np.asarray(y_scores)
    y_test = np.asarray(y_test)
    positives = (y_test == 1).sum()
    if positives == 0:
        return 0.0, 0.0, 0.0

    cut = max(int(len(y_scores) * frac), 1)
    top_indices = np.argsort(y_scores)[-cut:][::-1]
    top_labels = y_test[top_indices]

    tp = (top_labels == 1).sum()
    recall = tp / positives
    precision = tp / len(top_indices)
    max_precision = positives / len(top_indices)
    return recall, precision, max_precision


def calculate_er_n(scores, y_test, n):
    if n <= 0:
        return 0.0
    n = min(n, len(scores))
    top_n_labels = scores[:n, 0]
    tp_n = np.sum(top_n_labels)

    total_positives = np.sum(y_test)
    total_negatives = len(y_test) - total_positives

    if total_positives == 0 or total_negatives == 0:
        return 0.0

    tpr_n = tp_n / total_positives
    fpr_n = (n - tp_n) / total_negatives

    denom = tpr_n + fpr_n
    return tpr_n / denom if denom > 0 else 0.0


def eval_bagging(y_scores, y_test):
    y_scores = np.asarray(y_scores)
    y_test = np.asarray(y_test)

    rank_ratio = average_rank_ratio(y_scores, y_test)

    try:
        auroc = roc_auc_score(y_test, y_scores)
    except ValueError:
        auroc = np.nan

    scores = np.column_stack((y_test, y_scores))
    order = np.argsort(scores[:, 1])[::-1]
    scores = scores[order]

    top_recall_10, top_precision_10, max_precision_10 = top_recall_precision(0.1, y_scores, y_test)
    top_recall_30, top_precision_30, max_precision_30 = top_recall_precision(0.3, y_scores, y_test)

    total_positives = np.sum(y_test)
    top_25_recall = (np.sum(scores[:25, 0]) / total_positives) if total_positives > 0 else 0.0
    top_300_recall = (np.sum(scores[:300, 0]) / total_positives) if total_positives > 0 else 0.0

    return np.argsort(y_scores)[::-1], (
        top_25_recall,
        top_300_recall,
        top_recall_10,
        top_precision_10,
        max_precision_10,
        top_recall_30,
        top_precision_30,
        max_precision_30,
        calculate_er_n(scores, y_test, int(0.005 * len(y_test))),
        calculate_er_n(scores, y_test, int(0.01 * len(y_test))),
        calculate_er_n(scores, y_test, int(0.05 * len(y_test))),
        calculate_er_n(scores, y_test, int(0.1 * len(y_test))),
        calculate_er_n(scores, y_test, int(0.15 * len(y_test))),
        calculate_er_n(scores, y_test, int(0.20 * len(y_test))),
        calculate_er_n(scores, y_test, int(0.25 * len(y_test))),
        calculate_er_n(scores, y_test, int(0.30 * len(y_test))),
        auroc,
        rank_ratio,
        CalcBEDROC(scores, col=0, alpha=160.9),
        CalcBEDROC(scores, col=0, alpha=32.2),
        CalcBEDROC(scores, col=0, alpha=16.1),
        CalcBEDROC(scores, col=0, alpha=5.3),
    )
