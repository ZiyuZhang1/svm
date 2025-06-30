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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# ----------------------------
# Model Definition
# ----------------------------
class IntegratedMLP(nn.Module):
    def __init__(self, input_dims, hidden_dim=128, output_dim=1, task='classification'):
        super(IntegratedMLP, self).__init__()
        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, 64),
                nn.ReLU(),
                nn.BatchNorm1d(64)
            )
            for dim in input_dims
        ])

        total_encoded_dim = 64 * len(input_dims)
        self.mlp = nn.Sequential(
            nn.Linear(total_encoded_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.task = task

    def forward(self, inputs):
        encoded = [encoder(x) for encoder, x in zip(self.encoders, inputs)]
        x_cat = torch.cat(encoded, dim=1)
        out = self.mlp(x_cat)

        if self.task == 'classification':
            return out
        return out
    
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
with open('/itf-fi-ml/shared/users/ziyuzh/svm/data/stringdb/2023/name_convert.pkl', 'rb') as file:
    loaded_data = pickle.load(file)
stringId2name,name2stringId,aliases2stringId = loaded_data
del name2stringId,aliases2stringId
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


def neg_bagging(args):
    neg_df, neg_num, train_pos_df, df, y, X_all, test_index_loc, seed = args
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Sample negatives and build training set
    train_neg_df = neg_df.sample(n=neg_num, replace=True, random_state=seed)
    train_df = pd.concat([train_pos_df, train_neg_df])
    train_index_loc = df.index.get_indexer(train_df.index)

    train_mask = torch.zeros(len(df), dtype=torch.bool, device=device)
    train_mask[train_index_loc] = True

    test_mask = torch.zeros(len(df), dtype=torch.bool, device=device)
    test_mask[test_index_loc] = True   

    labels = torch.from_numpy(y).to(device).float()
    torch_tensors = [torch.from_numpy(arr).to(device).float() for arr in X_all]
    input_dims = [arr.shape[1] for arr in X_all]

    # Prepare training data
    train_inputs = [tensor[train_mask] for tensor in torch_tensors]
    train_labels = labels[train_mask]

    # Build TensorDataset for multi-input
    dataset = TensorDataset(*train_inputs, train_labels)

    # Split into training and validation (80% train, 20% val)
    val_ratio = 0.2
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed))

    # Create DataLoaders
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model setup
    model = IntegratedMLP(input_dims=input_dims, hidden_dim=128, output_dim=1, task='classification')
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    # optimizer = optim.Adam(model.parameters(), lr=1e-3)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)


    num_epochs = 100
    patience = 10  # Number of epochs to wait for improvement
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()

            batch_inputs = [b.to(device) for b in batch[:-1]]
            batch_labels = batch[-1].to(device)

            preds = model(batch_inputs).squeeze()
            loss = criterion(preds, batch_labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Validation step
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch_inputs = [b.to(device) for b in batch[:-1]]
                batch_labels = batch[-1].to(device)

                preds = model(batch_inputs).squeeze()
                loss = criterion(preds, batch_labels)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {avg_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

    # Load the best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Final test predictions
    model.eval()
    with torch.no_grad():
        all_preds = model(torch_tensors).squeeze()

    test_preds = all_preds[test_mask]

    return test_preds

# def neg_bagging(args):
#     neg_df, neg_num, train_pos_df, df, y, X_all, test_index_loc, seed = args
#     train_neg_df = neg_df.sample(n=neg_num, replace=True, random_state=seed)
#     train_df = pd.concat([train_pos_df, train_neg_df])
#     train_index_loc = df.index.get_indexer(train_df.index)

#     train_mask = torch.zeros(len(df), dtype=torch.bool, device=device)
#     train_mask[train_index_loc] = True

#     test_mask = torch.zeros(len(df), dtype=torch.bool, device=device)
#     test_mask[test_index_loc] = True   

#     labels = torch.from_numpy(y).to(device).float()

#     torch_tensors = [torch.from_numpy(arr).to(device).float() for arr in X_all]
#     input_dims = [arr.shape[1] for arr in X_all]

#     num_epochs = 50
#     all_test_predictions = []

#     # Model setup
#     model = IntegratedMLP(input_dims=input_dims, hidden_dim=128, output_dim=1, task='classification')
#     model = model.to(device)

#     criterion = nn.BCEWithLogitsLoss()

#     optimizer = optim.Adam(model.parameters(), lr=1e-3)

#     # Training loop
#     for epoch in range(num_epochs):
#         model.train()
#         optimizer.zero_grad()

#         preds = model(torch_tensors).squeeze()
#         loss = criterion(preds[train_mask], labels[train_mask])
#         loss.backward()
#         optimizer.step()

#     # After training, get predictions for all test (unlabeled) samples
#     model.eval()
#     with torch.no_grad():
#         all_preds = model(torch_tensors).squeeze()

#     test_preds = all_preds[test_mask]

#     return test_preds
