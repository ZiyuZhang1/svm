import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import numpy as np
import pandas as pd
from rdkit.ML.Scoring.Scoring import CalcBEDROC
# from pseudo_label import select_pseudo_negatives
from sklearn.metrics import roc_auc_score
import gseapy as gp
import pickle
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.model_selection import train_test_split

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
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
with open('/itf-fi-ml/shared/users/ziyuzh/svm/data/uniport_id/uni2name.pkl', 'rb') as file:
    uni2name_dict = pickle.load(file)

def enriched_set(input_ids,time):
    gene_names = set()
    for unid in input_ids:
        gene_list = uni2name_dict.get(unid, [])
        gene_names.update(gene_list)
    gene_names = list(gene_names)

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


def stratified_tensor_split(features, labels, val_ratio=0.2, random_state=42, max_tries=50):
    indices = np.arange(len(labels))
    labels_np = labels.cpu().numpy() if torch.is_tensor(labels) else np.array(labels)

    # Try stratified split until val has both classes
    for attempt in range(max_tries):
        stratify = labels_np if len(np.unique(labels_np)) > 1 else None
        train_idx, val_idx = train_test_split(
            indices,
            test_size=val_ratio,
            stratify=stratify,
            random_state=random_state + attempt
        )
        if len(np.unique(labels_np[val_idx])) > 1:
            break
    else:
        # Fallback: if we cannot get both classes, just do a random split
        print("⚠️ Only one class available for validation — falling back to random split.")
        train_idx, val_idx = train_test_split(
            indices, test_size=val_ratio, random_state=random_state
        )

    # Build datasets
    if isinstance(features, list):
        train_features = [f[train_idx] for f in features]
        val_features = [f[val_idx] for f in features]
        train_dataset = TensorDataset(*train_features, labels[train_idx])
        val_dataset = TensorDataset(*val_features, labels[val_idx])
    else:
        train_dataset = TensorDataset(features[train_idx], labels[train_idx])
        val_dataset = TensorDataset(features[val_idx], labels[val_idx])

    return train_dataset, val_dataset

class IntegratedMLP(nn.Module):
    def __init__(self, input_dims, hidden_dim=64, n_hidden_layers=1, output_dim=1, task='classification', dropout_rate=0.3):
        super(IntegratedMLP, self).__init__()
        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, 32),
                nn.ReLU(),
                nn.LayerNorm(32)
            )
            for dim in input_dims
        ])

        total_encoded_dim = 32 * len(input_dims)

        layers = []
        layers.append(nn.Linear(total_encoded_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))

        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))

        layers.append(nn.Linear(hidden_dim, output_dim))
        self.mlp = nn.Sequential(*layers)
        self.task = task

    def forward(self, inputs):
        encoded = [encoder(x) for encoder, x in zip(self.encoders, inputs)]
        x_cat = torch.cat(encoded, dim=1)
        out = self.mlp(x_cat)

        if self.task == 'classification':
            return out
        return out


# Assume the model is predefined
class SimpleModel(nn.Module):
    def __init__(self, input_size):
        super(SimpleModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.5),  # Stronger dropout
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)



def neg_bagging_early(args):
    neg_df, neg_num, train_pos_df, df, y, feature_list, test_index_loc, seed = args
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Prepare training data
    train_neg_df = neg_df.sample(n=neg_num, replace=True, random_state=seed)
    train_df = pd.concat([train_pos_df, train_neg_df])

    y_train = np.array([1] * len(train_pos_df) + [0] * len(train_neg_df))
    train_labels = torch.from_numpy(y_train).to(device).float()

    X_all = []
    for feature_name in feature_list:
        select_columns = [col for col in train_df.columns if col.startswith(feature_name)]
        X_all.append(train_df[select_columns].values)

    # Concatenate all features
    X_train = np.concatenate(X_all, axis=1)
    train_features = torch.from_numpy(X_train).to(device).float()

    train_dataset, val_dataset = stratified_tensor_split(train_features, train_labels, val_ratio=0.2)

    num_epochs = 100
    patience = 5
    best_val_loss = float('inf')
    patience_counter = 0
    batch_size = 32
    lr = 0.001

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = SimpleModel(input_size=X_train.shape[1]).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_auc = 0
    best_train_auc = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_probs = []
        train_targets = []

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch).squeeze(dim=1)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)

            train_probs.extend(outputs.detach().cpu().numpy())
            train_targets.extend(y_batch.detach().cpu().numpy())

        train_loss /= len(train_loader.dataset)
        train_auc = roc_auc_score(train_targets, train_probs)

        # Validation
        model.eval()
        val_loss = 0
        val_probs = []
        val_targets = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch).squeeze(dim=1)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)

                val_probs.extend(outputs.cpu().numpy())
                val_targets.extend(y_batch.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        val_auc = roc_auc_score(val_targets, val_probs)

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}")

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()  # Save best model
            best_val_auc = val_auc
            best_train_auc = train_auc
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    print(f"Best Train AUC: {best_train_auc:.4f}, Best Val AUC: {best_val_auc:.4f}")

    # Load the best model before testing
    model.load_state_dict(best_model_state)

    # Prepare test data
    test_df = df.iloc[test_index_loc]
    X_test = []
    for feature_name in feature_list:
        select_columns = [col for col in test_df.columns if col.startswith(feature_name)]
        X_test.append(test_df[select_columns].values)

    X_test = np.concatenate(X_test, axis=1)
    test_features = torch.from_numpy(X_test).to(device).float()

    model.eval()
    with torch.no_grad():
        preds = model(test_features).squeeze(dim=1)

    train_index_loc = df.index.get_indexer(train_df.index)
    overlap = set(train_index_loc)&set(test_index_loc)
    mask_loc = [np.where(test_index_loc == i)[0][0] for i in overlap]

    return (preds.cpu().numpy(), mask_loc), best_val_auc



class FeatureEncoder(nn.Module):
    """A simple feedforward encoder for each feature source."""
    def __init__(self, input_size, hidden_size=64):
        super(FeatureEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size)
        )

    def forward(self, x):
        return self.encoder(x)


class MidFusionModel(nn.Module):
    """Mid-fusion model with separate encoders for each feature source."""
    def __init__(self, input_sizes, hidden_size=32):
        super(MidFusionModel, self).__init__()
        self.encoders = nn.ModuleList([FeatureEncoder(size, hidden_size) for size in input_sizes])
        fusion_input_size = hidden_size * len(input_sizes)

        self.classifier = nn.Sequential(
            nn.Linear(fusion_input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x_list):
        encoded_features = [encoder(x) for encoder, x in zip(self.encoders, x_list)]
        fused = torch.cat(encoded_features, dim=1)
        return self.classifier(fused)



def neg_bagging_mid(args):
    neg_df, neg_num, train_pos_df, df, y, feature_list, test_index_loc, seed = args
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_neg_df = neg_df.sample(n=neg_num, replace=True, random_state=seed)
    train_df = pd.concat([train_pos_df, train_neg_df])

    y_train = np.array([1] * len(train_pos_df) + [0] * len(train_neg_df))
    train_labels = torch.from_numpy(y_train).to(device).float()

    feature_data = []
    input_sizes = []
    for feature_name in feature_list:
        select_columns = [col for col in train_df.columns if col.startswith(feature_name)]
        feature_values = train_df[select_columns].values
        feature_data.append(torch.from_numpy(feature_values).to(device).float())
        input_sizes.append(feature_values.shape[1])

    train_dataset, val_dataset = stratified_tensor_split(feature_data, train_labels, val_ratio=0.2)

    num_epochs = 100
    patience = 8
    best_val_loss = float('inf')
    best_val_auc = 0
    best_train_auc = 0
    patience_counter = 0
    batch_size = 32
    lr = 0.0005

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = MidFusionModel(input_sizes).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_preds = []
        train_targets = []

        for batch in train_loader:
            *X_batches, y_batch = batch
            optimizer.zero_grad()
            outputs = model(X_batches).squeeze(dim=1)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * y_batch.size(0)

            train_preds.extend(outputs.detach().cpu().numpy())
            train_targets.extend(y_batch.cpu().numpy())

        train_loss /= len(train_loader.dataset)
        train_auc = roc_auc_score(train_targets, train_preds)

        model.eval()
        val_loss = 0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch in val_loader:
                *X_batches, y_batch = batch
                outputs = model(X_batches).squeeze(dim=1)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * y_batch.size(0)

                val_preds.extend(outputs.cpu().numpy())
                val_targets.extend(y_batch.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        val_auc = roc_auc_score(val_targets, val_preds)

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_auc = val_auc
            best_train_auc = train_auc
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    model.load_state_dict(best_model_state)

    print(f"Best Train AUC: {best_train_auc:.4f}, Best Val AUC: {best_val_auc:.4f}")

    # Prepare test data
    test_df = df.iloc[test_index_loc]
    test_features = []

    for feature_name in feature_list:
        select_columns = [col for col in test_df.columns if col.startswith(feature_name)]
        feature_values = torch.from_numpy(test_df[select_columns].values).to(device).float()
        test_features.append(feature_values)

    model.eval()
    with torch.no_grad():
        preds = model(test_features).squeeze(dim=1)

    train_index_loc = df.index.get_indexer(train_df.index)
    overlap = set(train_index_loc)&set(test_index_loc)
    mask_loc = [np.where(test_index_loc == i)[0][0] for i in overlap]

    return (preds.cpu().numpy(), mask_loc), best_val_auc

class FusionHead(nn.Module):
    """Trainable late-fusion head that ingests per-feature probabilities."""
    def __init__(self, num_sources):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_sources, 3),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(3, 1),
            nn.Sigmoid()
        )
    def forward(self, x):  # x: [B, num_sources] probabilities
        return self.net(x).squeeze(1)

def safe_train_val_split(X, Y, test_size=0.2, max_tries=100, random_state=42):
    rng = np.random.RandomState(random_state)
    for attempt in range(max_tries):
        X_train, X_val, y_train, y_val = train_test_split(
            X, Y, test_size=test_size, random_state=rng.randint(0, 1e6), stratify=Y
        )
        if len(np.unique(y_val)) > 1:  # at least one pos & one neg
            return X_train, X_val, y_train, y_val
    raise ValueError("Could not create a validation split with both classes after many tries")


# ---- training function ----
def later_fusion_train(X, Y, num_epochs=50, lr=1e-3, batch_size=32):
    """
    Train a FusionHead on (X, Y).
    X: numpy array of shape [N, num_sources]
    Y: numpy array of shape [N] (binary labels: 0/1)
    """
    print('train later fusion mlp')
    # 1. Train/val split
    X_train, X_val, y_train, y_val = safe_train_val_split(X, Y)

    # Convert to torch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_val   = torch.tensor(X_val, dtype=torch.float32)
    y_val   = torch.tensor(y_val, dtype=torch.float32)

    # 2. Model, loss, optimizer
    num_sources = X.shape[1]
    model = FusionHead(num_sources)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # 3. Training loop
    best_val_auc = float('-inf')
    best_model_state = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        model.train()
        # mini-batch training
        permutation = torch.randperm(X_train.size(0))
        for i in range(0, X_train.size(0), batch_size):
            idx = permutation[i:i+batch_size]
            xb, yb = X_train[idx], y_train[idx]

            # forward
            preds = model(xb)
            loss = criterion(preds, yb)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # validation
        model.eval()
        with torch.no_grad():
            val_probs = model(X_val).detach().numpy()   # probabilities
            val_auc = roc_auc_score(y_val.numpy(), val_probs)

        if np.isnan(val_auc):
            print('Validation AUC is NaN; retaining previous best model state.')
            continue

        # track best
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = copy.deepcopy(model.state_dict())

        print(f"Epoch {epoch+1}/{num_epochs} - Val AUC: {val_auc:.4f}")

    # 4. Return best model
    if best_model_state is None:
        best_model_state = copy.deepcopy(model.state_dict())
    best_model = FusionHead(num_sources)
    best_model.load_state_dict(best_model_state)
    print(f"Best Val AUC: {best_val_auc:.4f}")
    return best_model


def neg_bagging_later(args):
    neg_df, neg_num, train_pos_df, df, y, feature_list, test_index_loc, seed = args
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Prepare training data
    train_neg_df = neg_df.sample(n=neg_num, replace=True, random_state=seed)
    train_df = pd.concat([train_pos_df, train_neg_df])

    y_train = np.array([1] * len(train_pos_df) + [0] * len(train_neg_df))
    train_labels = torch.from_numpy(y_train).to(device).float()

    feature_preds = {}
    train_preds = []
    fusion_candidates = {}
    auc_records = {}
    # Loop through each feature source
    for feature_name in feature_list:
        select_columns = [col for col in train_df.columns if col.startswith(feature_name)]
        X_train = train_df[select_columns].values
        train_features = torch.from_numpy(X_train).to(device).float()

        train_dataset, val_dataset = stratified_tensor_split(train_features, train_labels, val_ratio=0.2)

        num_epochs = 100
        patience = 5
        best_val_loss = float('inf')
        patience_counter = 0
        batch_size = 32
        lr = 0.001

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        model = SimpleModel(input_size=X_train.shape[1]).to(device)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

        best_val_auc = 0
        best_train_auc = 0

        for epoch in range(num_epochs):
            model.train()
            train_loss = 0
            train_probs = []
            train_targets = []

            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch).squeeze(dim=1)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * X_batch.size(0)

                train_probs.extend(outputs.detach().cpu().numpy())
                train_targets.extend(y_batch.detach().cpu().numpy())

            train_loss /= len(train_loader.dataset)
            train_auc = roc_auc_score(train_targets, train_probs)

            # Validation
            model.eval()
            val_loss = 0
            val_probs = []
            val_targets = []

            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    outputs = model(X_batch).squeeze(dim=1)
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item() * X_batch.size(0)

                    val_probs.extend(outputs.cpu().numpy())
                    val_targets.extend(y_batch.cpu().numpy())

            val_loss /= len(val_loader.dataset)
            val_auc = roc_auc_score(val_targets, val_probs)

            print(f"Feature: {feature_name}, Epoch {epoch + 1}/{num_epochs}, "
                  f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                  f"Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_auc = val_auc
                best_train_auc = train_auc
                patience_counter = 0
                best_model_state = model.state_dict()
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping for feature {feature_name} at epoch {epoch + 1}")
                break

        print(f"Best Train AUC for feature {feature_name}: {best_train_auc:.4f}")
        print(f"Best Val AUC for feature {feature_name}: {best_val_auc:.4f}")
        auc_records[feature_name] = best_val_auc

        # Load the best model
        model.load_state_dict(best_model_state)

        # Prepare test data for this feature
        test_df = df.iloc[test_index_loc]
        select_columns = [col for col in test_df.columns if col.startswith(feature_name)]
        X_test = test_df[select_columns].values
        test_features = torch.from_numpy(X_test).to(device).float()

        model.eval()
        with torch.no_grad():
            preds = model(test_features).squeeze(dim=1).cpu().numpy()
            train_preds.append(model(train_features).squeeze(dim=1).cpu().numpy())

        # Always save individual feature predictions
        feature_preds[feature_name] = preds

        # Only include features with val AUC > 0.7 in fusion
        if best_val_auc >= 0:
            fusion_candidates[feature_name] = preds
        else:
            print(f"Feature {feature_name} excluded from fusion due to low Val AUC: {best_val_auc:.4f}")

    # Late fusion: average predictions from all eligible feature sources
    if fusion_candidates:
        all_preds = np.array(list(fusion_candidates.values()))
        fused_preds = np.mean(all_preds, axis=0)
    else:
        fused_preds = None
        print("No features passed the AUC threshold. Fusion result is None.")

    lf_mlp = later_fusion_train(np.array(train_preds).T,y_train)

    lf_mlp.eval()  # set to evaluation mode
    with torch.no_grad():
        X_new = torch.tensor(all_preds.T, dtype=torch.float32)
        probs = lf_mlp(X_new).numpy() 

    train_index_loc = df.index.get_indexer(train_df.index)
    overlap = set(train_index_loc)&set(test_index_loc)
    mask_loc = [np.where(test_index_loc == i)[0][0] for i in overlap]

    return feature_preds, fused_preds, auc_records, probs, mask_loc


# def neg_bagging_later(args):
#     """
#     Train per-feature models; then train a small classifier to fuse their DECISIONS (probabilities).
#     Model selection uses validation AUC of the fused outputs (not train).
#     """
#     (neg_df, neg_num, train_pos_df, df, y, feature_list, test_index_loc, seed) = args

#     np.random.seed(seed); torch.manual_seed(seed)
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     # 1) Prepare training frame and labels
#     train_neg_df = neg_df.sample(n=neg_num, replace=True, random_state=seed)
#     train_df = pd.concat([train_pos_df, train_neg_df])
#     y_train_np = np.array([1]*len(train_pos_df) + [0]*len(train_neg_df), dtype=np.int64)
#     y_train = torch.from_numpy(y_train_np).float().to(device)

#     # 2) Train each per-feature base model and collect VAL predictions
#     feature_models = {}
#     feature_val_preds = {}   # dict[name] -> np.array shape [N_val]
#     feature_test_preds = {}  # dict[name] -> np.array shape [N_test]
#     feature_val_auc = {}     # dict[name] -> float

#     # We’ll build a single stratified split once, and reuse indices across features
#     from sklearn.model_selection import StratifiedShuffleSplit
#     sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
#     idx = np.arange(len(train_df))
#     (train_idx, val_idx), = sss.split(idx, y_train_np)

#     # For convenience, keep validation labels on CPU for roc_auc_score
#     y_val_np = y_train_np[val_idx]
#     y_val_t = torch.from_numpy(y_val_np).float().to(device)

#     # Define your per-feature model (same as your SimpleModel)
#     class SimpleModel(nn.Module):
#         def __init__(self, input_size):
#             super().__init__()
#             self.net = nn.Sequential(
#                 nn.Linear(input_size, 128),
#                 nn.ReLU(),
#                 nn.Linear(128, 1),
#                 nn.Sigmoid()
#             )
#         def forward(self, x):
#             return self.net(x).squeeze(1)

#     auc_threshold = 0
#     kept_feature_names = []

#     for feature_name in feature_list:
#         cols = [c for c in train_df.columns if c.startswith(feature_name)]
#         if not cols:
#             continue

#         X = train_df[cols].values.astype(np.float32)
#         X_train = torch.from_numpy(X[train_idx]).to(device)
#         X_val   = torch.from_numpy(X[val_idx]).to(device)

#         # Dataloaders
#         batch_size = 32
#         train_dataset = torch.utils.data.TensorDataset(X_train, y_train[train_idx])
#         val_dataset   = torch.utils.data.TensorDataset(X_val,   y_val_t)
#         train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#         val_loader    = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

#         # Train base model with early stopping on val loss
#         model = SimpleModel(input_size=X.shape[1]).to(device)
#         crit = nn.BCELoss()
#         opt  = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
#         best_state = None
#         best_val_loss = float('inf')
#         best_val_auc  = 0.0
#         patience = 5
#         wait = 0
#         max_epochs = 100

#         for epoch in range(max_epochs):
#             model.train()
#             for xb, yb in train_loader:
#                 opt.zero_grad()
#                 p = model(xb)
#                 loss = crit(p, yb)
#                 loss.backward()
#                 opt.step()

#             # validate
#             model.eval()
#             val_losses, val_probs = [], []
#             with torch.no_grad():
#                 for xb, yb in val_loader:
#                     p = model(xb)
#                     val_losses.append(crit(p, yb).item() * xb.size(0))
#                     val_probs.append(p.detach().cpu().numpy())
#             val_loss = np.sum(val_losses) / len(val_dataset)
#             val_probs = np.concatenate(val_probs, axis=0)
#             val_auc = roc_auc_score(y_val_np, val_probs)

#             if val_loss < best_val_loss:
#                 best_val_loss = val_loss
#                 best_val_auc  = val_auc
#                 best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
#                 wait = 0
#             else:
#                 wait += 1
#                 if wait >= patience:
#                     break

#         # restore best
#         model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
#         feature_models[feature_name] = model
#         feature_val_auc[feature_name] = best_val_auc

#         # Store validation predictions (will be used to train fusion head)
#         model.eval()
#         with torch.no_grad():
#             val_probs = model(X_val).detach().cpu().numpy()
#         feature_val_preds[feature_name] = val_probs

#         # Also pre-compute TEST predictions per feature
#         test_df_local = df.iloc[test_index_loc]
#         test_cols = [c for c in test_df_local.columns if c.startswith(feature_name)]
#         if not test_cols:
#             continue
#         X_test = torch.from_numpy(test_df_local[test_cols].values.astype(np.float32)).to(device)
#         with torch.no_grad():
#             test_probs = model(X_test).detach().cpu().numpy()
#         feature_test_preds[feature_name] = test_probs

#     # 3) Choose which feature streams to include in fusion (based on their *validation* AUC)
#     kept_feature_names = [n for n in feature_val_preds.keys() if feature_val_auc.get(n, 0) >= auc_threshold]

#     # Edge case: if none pass, fall back to best single feature
#     if not kept_feature_names:
#         if not feature_val_auc:
#             # no features available at all
#             return feature_test_preds, None, feature_val_auc, None
#         best_name = max(feature_val_auc, key=feature_val_auc.get)
#         kept_feature_names = [best_name]

#     # 4) Build the matrix of validation decisions for the fusion head
#     Z_val = np.column_stack([feature_val_preds[n] for n in kept_feature_names]).astype(np.float32)  # [N_val, K]
#     y_val = torch.from_numpy(y_val_np).float().to(device)
#     Z_val_t = torch.from_numpy(Z_val).to(device)

#     # Optional: split Z_val again to early-stop the fusion head (simple 80/20 split here)
#     N = Z_val.shape[0]
#     perm = np.random.RandomState(seed).permutation(N)
#     cut = int(0.8 * N)
#     tr_idx, fu_val_idx = perm[:cut], perm[cut:]

#     Ztr = Z_val_t[tr_idx]; ytr = y_val[tr_idx]
#     Zvl = Z_val_t[fu_val_idx]; yvl = y_val[fu_val_idx]

#     # 5) Train the trainable late-fusion head on decision-level inputs
#     fusion = FusionHead(num_sources=Z_val.shape[1]).to(device)
#     crit = nn.BCELoss()
#     opt  = torch.optim.Adam(fusion.parameters(), lr=5e-3, weight_decay=1e-4)
#     patience, wait = 10, 0
#     best_state = None
#     best_auc  = 0.0
#     max_epochs = 200
#     batch_size = 64

#     def batches(X, y, bs):
#         for i in range(0, X.size(0), bs):
#             yield X[i:i+bs], y[i:i+bs]

#     for epoch in range(max_epochs):
#         fusion.train()
#         for xb, yb in batches(Ztr, ytr, batch_size):
#             opt.zero_grad()
#             p = fusion(xb)
#             loss = crit(p, yb)
#             loss.backward()
#             opt.step()

#         # meta-validation for the fusion head
#         fusion.eval()
#         with torch.no_grad():
#             pv = fusion(Zvl).detach().cpu().numpy()
#         val_auc = roc_auc_score(yvl.detach().cpu().numpy(), pv)

#         if val_auc > best_auc:
#             best_auc = val_auc
#             best_state = {k: v.cpu().clone() for k, v in fusion.state_dict().items()}
#             wait = 0
#         else:
#             wait += 1
#             if wait >= patience:
#                 break

#     fusion.load_state_dict({k: v.to(device) for k, v in best_state.items()})

#     # 6) Produce fused TEST predictions (decision-level inputs from base models → fusion head)
#     Z_test = np.column_stack([feature_test_preds[n] for n in kept_feature_names]).astype(np.float32)
#     Z_test_t = torch.from_numpy(Z_test).to(device)
#     fusion.eval()
#     with torch.no_grad():
#         fused_test_probs = fusion(Z_test_t).detach().cpu().numpy()

#     # # Also return the raw per-feature test preds for analysis
#     # return feature_test_preds, fused_test_probs, feature_val_auc, {
#     #     "kept_features": kept_feature_names,
#     #     "fusion_val_auc": float(best_auc)
#     # }
#     return fused_test_probs, best_auc
