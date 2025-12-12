import os
import pickle
import sys

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import MinMaxScaler

import macau
from features_reindex import get_feature, read_data, read_data_timecut
from model_nn_non_para import eval_bagging


def extract_macau_predictions(preds, test_mask, num_rows):
    """Normalize different Macau prediction shapes to a full matrix."""
    if hasattr(preds, "predictions"):
        preds = preds.predictions

    if isinstance(preds, dict):
        if "test" in preds:
            preds = preds["test"]
        elif "pred" in preds:
            preds = preds["pred"]

    try:
        preds_arr = np.asarray(preds)
    except Exception:
        print("Macau predictions could not be converted to an array; returning zeros.")
        return np.zeros_like(test_mask, dtype=float)

    if preds_arr.shape == test_mask.shape:
        return preds_arr

    flat_mask = test_mask.reshape(-1)
    flat_preds = preds_arr.reshape(-1)
    if flat_preds.shape[0] == flat_mask.sum():
        full = np.zeros_like(test_mask, dtype=float)
        full[test_mask] = flat_preds
        return full

    if preds_arr.ndim == 2 and preds_arr.shape[0] == num_rows and preds_arr.shape[1] == 1:
        full = np.zeros_like(test_mask, dtype=float)
        full[:, 0] = preds_arr[:, 0]
        return full

    print("Unexpected Macau prediction shape; returning zeros.")
    return np.zeros_like(test_mask, dtype=float)


def run_macau_iteration(
    num_rows,
    num_diseases,
    side_info,
    disease_training,
    disease_tests,
    num_latent=32,
    precision=5.0,
    burnin=400,
    nsamples=1600,
    seed=None,
):
    """Train a single Macau model and return predictions for all disease test sets."""
    Y = np.full((num_rows, num_diseases), np.nan)
    Ytest_mask = np.zeros_like(Y, dtype=bool)

    for col, info in enumerate(disease_training):
        Y[info["train_rows"], col] = info["train_labels"]
    for col, info in enumerate(disease_tests):
        Ytest_mask[info["test_rows"], col] = True

    if seed is not None:
        np.random.seed(seed)

    result = macau.macau(
        Y=Y,
        Ytest=Ytest_mask,
        side=[side_info, None],
        num_latent=num_latent,
        precision=precision,
        burnin=burnin,
        nsamples=nsamples,
    )
    return extract_macau_predictions(result, Ytest_mask, num_rows)


def prepare_disease_metadata(disease_id, all_df, merged_df, base_index, time):
    """Build consistent train/test splits and labels for a single disease."""
    df, y = read_data_timecut(disease_id, all_df, merged_df, time)
    df = df.loc[base_index]
    y_series = pd.Series(y, index=df.index).loc[base_index]

    test_idx = df[df["test"] == 1].index
    train_pos_idx = y_series.index[y_series == 1].difference(test_idx)
    test_pos_idx = test_idx
    all_neg_idx = y_series.index[y_series == 0]

    test_rows = np.concatenate([test_pos_idx, all_neg_idx])
    y_test = np.concatenate(
        [np.ones(len(test_pos_idx), dtype=float), np.zeros(len(all_neg_idx), dtype=float)]
    )

    return {
        "train_pos_idx": train_pos_idx,
        "test_pos_idx": test_pos_idx,
        "all_neg_idx": all_neg_idx,
        "test_rows": test_rows,
        "y_test": y_test,
    }


def bagged_macau_for_all(disease_meta, features, seed_list, neg_multiplier=5):
    """Run Macau jointly over all diseases for each bag and aggregate predictions."""
    index_to_pos = {idx: pos for pos, idx in enumerate(features.index)}
    num_rows = len(features)
    num_diseases = len(disease_meta)
    disease_ids = list(disease_meta.keys())
    side_info = sparse.csr_matrix(features.values)

    # Prepare constant test row positions per disease
    disease_tests = []
    for disease_id in disease_ids:
        meta = disease_meta[disease_id]
        test_rows = np.array([index_to_pos[idx] for idx in meta["test_rows"]])
        disease_tests.append({"test_rows": test_rows})

    bagged_scores = {disease_id: np.zeros(len(disease_meta[disease_id]["test_rows"])) for disease_id in disease_ids}

    for seed in seed_list:
        disease_training = []
        for disease_id in disease_ids:
            meta = disease_meta[disease_id]
            train_pos = np.array([index_to_pos[idx] for idx in meta["train_pos_idx"]])
            neg_num = neg_multiplier * len(train_pos)
            all_neg_idx = meta["all_neg_idx"]
            sampled_neg_ids = all_neg_idx.to_series().sample(
                n=neg_num,
                replace=neg_num > len(all_neg_idx),
                random_state=seed,
            ).index
            train_neg = np.array([index_to_pos[idx] for idx in sampled_neg_ids])
            train_rows = np.concatenate([train_pos, train_neg])
            train_labels = np.concatenate(
                [np.ones(len(train_pos), dtype=float), np.zeros(len(train_neg), dtype=float)]
            )
            disease_training.append({"train_rows": train_rows, "train_labels": train_labels})

        preds_matrix = run_macau_iteration(
            num_rows=num_rows,
            num_diseases=num_diseases,
            side_info=side_info,
            disease_training=disease_training,
            disease_tests=disease_tests,
            seed=seed,
        )

        for col, disease_id in enumerate(disease_ids):
            test_rows = disease_tests[col]["test_rows"]
            bagged_scores[disease_id] += preds_matrix[test_rows, col]

    final_scores = {
        disease_id: bagged_scores[disease_id] / len(seed_list) for disease_id in disease_ids
    }
    return disease_ids, final_scores


def main():
    root = "/itf-fi-ml/shared/users/ziyuzh/svm"
    time_spilt = True
    # test_bug = True
    test_bug = False

    if test_bug:
        feature_list = ["uniport_ppi_2019"]
        out_path = os.path.join(root, "results/temp")
        out_path_pred = out_path + "_pred"
        time = 2019
    else:
        if len(sys.argv) < 4:
            raise ValueError("Usage: python main_mf.py <features> <out_dir> <time>")
        _requested_features = sys.argv[1].split(",")
        feature_list = ["uniport_ppi_2019"]
        if set(_requested_features) != set(feature_list):
            print(
                "Using uniport_ppi_2019 only for Macau MF; overriding provided feature list."
            )
        out_path = os.path.join(root, sys.argv[2])
        out_path_pred = out_path + "_pred"
        time = int(sys.argv[3])

    os.makedirs(out_path, exist_ok=True)
    os.makedirs(out_path_pred, exist_ok=True)

    merged_df = None
    for feature in feature_list:
        feature_df = get_feature(root, feature)

        feature_cols = [col for col in feature_df.columns if col.startswith("feature")]
        if feature_cols:
            scaler = MinMaxScaler()
            feature_df[feature_cols] = scaler.fit_transform(feature_df[feature_cols])

        feature_df.rename(
            columns={
                col: f"{feature}_{col}" if col.startswith("feature") else col
                for col in feature_df.columns
            },
            inplace=True,
        )

        if merged_df is None:
            merged_df = feature_df
        else:
            merged_df = pd.merge(merged_df, feature_df, on="string_id", how="inner")
        del feature_df

    all_df = pd.read_csv(
        "/itf-fi-ml/shared/users/ziyuzh/svm/data/disgent_2020/timecut/dga_time_uniport.csv"
    )
    all_df = all_df[all_df["string_id"].isin(merged_df["string_id"])]

    selected_diseases = []
    if time_spilt:
        for disease_id in all_df["disease_id"].unique():
            sub_df = all_df[all_df["disease_id"] == disease_id]
            if len(sub_df) < 15:
                continue
            if (
                sub_df["first_pub_year"].max() > time
                and sub_df["first_pub_year"].min() <= time
                and len(sub_df[sub_df["first_pub_year"] < time]) >= 5
            ):
                selected_diseases.append(disease_id)

    print(feature_list, len(selected_diseases), len(merged_df))
    base_features = merged_df.set_index("string_id")
    base_index = base_features.index

    disease_meta = {}
    for disease in selected_diseases:
        print(disease, len(all_df[all_df["disease_id"] == disease]))
        disease_meta[disease] = prepare_disease_metadata(
            disease_id=disease,
            all_df=all_df,
            merged_df=merged_df,
            base_index=base_index,
            time=time,
        )

    num_iterations = 20
    base_seed = 42
    seed_list = [base_seed + i for i in range(num_iterations)]

    disease_ids, final_scores = bagged_macau_for_all(
        disease_meta=disease_meta, features=base_features, seed_list=seed_list
    )

    columns = [
        "method",
        "fold",
        "para",
        "top_recall_25",
        "top_recall_300",
        "top_recall_10%",
        "top_precision_10%",
        "max_precision_10%",
        "top_recall_30%",
        "top_precision_30%",
        "max_precision_30%",
        "pm_0.5%",
        "pm_1%",
        "pm_5%",
        "pm_10%",
        "pm_15%",
        "pm_20%",
        "pm_25%",
        "pm_30%",
        "auroc",
        "rank_ratio",
        "bedroc_1",
        "bedroc_5",
        "bedroc_10",
        "bedroc_30",
    ]
    all_results = []
    prediction_collection = {
        "true_label": {},
        "test_genes": {},
        "train_pos_genes": {},
        "macau_mf": {},
    }

    for disease in disease_ids:
        meta = disease_meta[disease]
        y_test = meta["y_test"]
        scores = final_scores[disease]
        ranked_predict_index, results = eval_bagging(scores, y_test)

        df_row = ["random_negative", 1, "macau_mf", *results]
        result_df = pd.DataFrame([df_row], columns=columns)
        result_df.to_csv(os.path.join(out_path, f"{disease}.csv"), index=False)

        prediction_collection["true_label"][disease] = y_test
        prediction_collection["test_genes"][disease] = meta["test_rows"]
        prediction_collection["train_pos_genes"][disease] = meta["train_pos_idx"]
        prediction_collection["macau_mf"][disease] = scores

        mean_df = result_df.copy()
        mean_df["disease"] = disease
        all_results.append(mean_df)

    final_result = pd.concat(all_results, ignore_index=True)
    final_result.to_csv(os.path.join(out_path, "all_disease.csv"), index=False)

    with open(os.path.join(out_path_pred, "all_disease_pred.pkl"), "wb") as f:
        pickle.dump(prediction_collection, f)


if __name__ == "__main__":
    main()
