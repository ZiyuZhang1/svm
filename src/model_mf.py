import random
import pandas as pd
import numpy as np
import smurff
from scipy.sparse import coo_matrix



def neg_bag(all_df_train,string2idx,disease2idx,side_info_string, seed, b_para, n_para):

    rng = random.Random(seed)

    # container for negative samples
    all_df_train = all_df_train[all_df_train["string_id"].isin(string2idx)].copy()

    train_neg_records = []
    disease_ids = all_df_train['disease_id'].unique()
    string_ids = list(string2idx.keys())

    pos_rows = all_df_train['disease_id'].map(disease2idx).values
    pos_cols = all_df_train['string_id'].map(string2idx).values
    pos_data = np.ones(len(all_df_train), dtype=np.int8)

    for disease in disease_ids:
        # positive genes for this disease
        trainset = set(all_df_train.loc[all_df_train['disease_id'] == disease, 'string_id'])

        # sample negatives (5x positives)
        train_neg = rng.sample(list(set(string_ids) - trainset),k=5 * len(trainset))

        # store disease–gene pairs
        for gene in train_neg:
            train_neg_records.append({'disease_id': disease,'string_id': gene})

    # build DataFrame
    train_neg_df = pd.DataFrame(train_neg_records)

    neg_rows = train_neg_df['disease_id'].map(disease2idx).values
    neg_cols = train_neg_df['string_id'].map(string2idx).values
    neg_data = np.zeros(len(train_neg_df), dtype=np.int8)

    rows = np.concatenate([pos_rows, neg_rows])
    cols = np.concatenate([pos_cols, neg_cols])
    data = np.concatenate([pos_data, neg_data])

    Y_train = coo_matrix(
        (data, (rows, cols)),
        shape=(len(disease_ids), len(string_ids))
    ).tocsr()

    Y = Y_train.tocsr()          # shape: (n_genes, n_diseases)
    n_genes, n_dis = Y.shape

    # all possible pairs (gene, disease)
    rows_all = np.repeat(np.arange(n_genes, dtype=np.int32), n_dis)
    cols_all = np.tile(np.arange(n_dis, dtype=np.int32), n_genes)

    # mark observed (train) pairs so we can exclude them
    obs = Y.tocoo()
    obs_lin = (obs.row.astype(np.int64) * n_dis + obs.col.astype(np.int64))

    all_lin = (rows_all.astype(np.int64) * n_dis + cols_all.astype(np.int64))

    # keep only those not observed in train
    mask = ~np.isin(all_lin, obs_lin, assume_unique=False)

    rows_q = rows_all[mask]
    cols_q = cols_all[mask]
    vals_q = np.ones(rows_q.shape[0], dtype=np.float32)

    Y_query = coo_matrix((vals_q, (rows_q, cols_q)), shape=Y.shape).tocsr()

    predictions = smurff.MacauSession(
                        Ytrain     = Y_train,
                        Ytest      = Y_query,
                        side_info  = [None, side_info_string],
                        direct     = True,
                        num_latent = 16,
                        burnin     = int(b_para),
                        nsamples   = int(n_para)).run()

    # S = np.empty((n_genes, n_dis), dtype=np.float32)
    S = np.full((n_genes, n_dis), np.nan, dtype=np.float32)

    for p in predictions:
        r, c = p.coords   # <-- indices live here
        S[r, c] = p.pred_avg

    return S