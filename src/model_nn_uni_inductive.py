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


def stratified_tensor_split(features, labels, val_ratio=0.2, random_state=42):
    indices = np.arange(len(labels))

    # Move labels to CPU and numpy for stratification
    labels_np = labels.cpu().numpy()

    # Stratified split
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_ratio,
        stratify=labels_np,
        random_state=random_state
    )

    if isinstance(features, list):  # Case 1: list of feature arrays
        train_features = [f[train_idx] for f in features]
        val_features = [f[val_idx] for f in features]
        train_dataset = TensorDataset(*train_features, labels[train_idx])
        val_dataset = TensorDataset(*val_features, labels[val_idx])
    else:  # Case 2: single feature array
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

    return preds.cpu().numpy(), best_val_auc



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

    return preds.cpu().numpy(), best_val_auc


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

        # Always save individual feature predictions
        feature_preds[feature_name] = preds

        # Only include features with val AUC > 0.7 in fusion
        if best_val_auc >= 0.7:
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

    return feature_preds, fused_preds, auc_records
