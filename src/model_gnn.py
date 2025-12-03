import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from rdkit.ML.Scoring.Scoring import CalcBEDROC
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch_geometric.nn import GCNConv, SAGEConv, BatchNorm, global_mean_pool
from torch_geometric.utils import add_self_loops, to_undirected

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def average_rank_ratio(y_scores, y_test):
    """Average rank of the true positives normalised by the list length."""
    y_scores = np.array(y_scores)
    y_test = np.array(y_test)

    sorted_indices = np.argsort(-y_scores)
    ranks = np.empty_like(sorted_indices)
    ranks[sorted_indices] = np.arange(1, len(y_scores) + 1)

    true_positive_indices = np.where(y_test == 1)[0]
    if true_positive_indices.size == 0:
        return 0.0

    average_rank = np.mean(ranks[true_positive_indices])
    rank_ratio = average_rank / y_test.shape[0]
    return round(rank_ratio, 4)


def top_recall_precision(frac, y_scores, y_test):
    positives = np.sum(y_test == 1)
    if positives == 0:
        return 0.0, 0.0, 0.0

    cut = max(1, int(len(y_scores) * frac))
    top_indices = np.argsort(y_scores)[-cut:][::-1]
    top_labels = y_test[top_indices]

    tp = np.sum(top_labels == 1)
    recall = tp / positives
    precision = tp / len(top_indices)
    max_precision = positives / len(top_indices)
    return recall, precision, max_precision


def calculate_er_n(scores, y_test, n):
    n = min(n, len(scores))
    if n == 0:
        return 0.0

    top_n_labels = scores[:n, 0]
    tp_n = np.sum(top_n_labels)

    total_positives = np.sum(y_test)
    total_negatives = len(y_test) - total_positives

    tpr_n = tp_n / total_positives if total_positives > 0 else 0.0
    fpr_n = (n - tp_n) / total_negatives if total_negatives > 0 else 0.0
    return tpr_n / (tpr_n + fpr_n) if (tpr_n + fpr_n) > 0 else 0.0


def eval_bagging(y_scores, y_test):
    rank_ratio = average_rank_ratio(y_scores, y_test)

    if y_scores is not None:
        try:
            auroc = roc_auc_score(y_test, y_scores)
        except ValueError:
            auroc = "AUROC computation failed (possibly due to label issues)"
    else:
        auroc = "AUROC not available (no predict_proba or decision_function)"

    scores = np.column_stack((y_test, y_scores))
    scores = scores[scores[:, 1].argsort()[::-1]]

    top_recall_10, top_precision_10, max_precision_10 = top_recall_precision(0.1, y_scores, y_test)
    top_recall_30, top_precision_30, max_precision_30 = top_recall_precision(0.3, y_scores, y_test)

    total_positives = np.sum(y_test)
    top_25_recall = np.sum(scores[:25, 0]) / total_positives if total_positives > 0 else 0.0
    top_300_recall = np.sum(scores[:300, 0]) / total_positives if total_positives > 0 else 0.0

    return np.argsort(y_scores)[::-1], (
        top_25_recall,
        top_300_recall,
        top_recall_10, top_precision_10, max_precision_10,
        top_recall_30, top_precision_30, max_precision_30,
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


class SimpleGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels=128, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.output = nn.Linear(hidden_channels, 2)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.output(x)


def _safe_roc_auc(labels, scores):
    try:
        return roc_auc_score(labels, scores)
    except ValueError:
        return float('nan')

class ImprovedGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels=128, num_layers=2, dropout=0.5, num_classes=2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # First layer
        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.bns.append(BatchNorm(hidden_channels))

        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(BatchNorm(hidden_channels))

        # MLP output head for better feature interaction
        self.output = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, num_classes)
        )

        self.dropout = dropout

    def forward(self, x, edge_index, batch=None):
        for conv, bn in zip(self.convs, self.bns):
            residual = x  # Residual connection
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + residual if x.shape == residual.shape else x

        # For graph-level tasks: pool node features
        if batch is not None:
            x = global_mean_pool(x, batch)

        return self.output(x)

def neg_bagging_gcn(args):
    (
        neg_candidates,
        neg_num,
        train_pos_idx,
        df,
        y,
        edge_index,
        feature_list,
        test_index,
        seed,
    ) = args

    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    feature_matrix = df.to_numpy(dtype=np.float32, copy=True)
    feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    x = torch.from_numpy(feature_matrix).to(device)

    labels = torch.from_numpy(np.asarray(y, dtype=np.int64)).to(device)

    if edge_index.size(0) != 2:
        raise ValueError('edge_index must have shape [2, num_edges].')

    num_nodes = x.size(0)
    edge_index = edge_index.to(device)
    edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

    node_to_idx = {sid: idx for idx, sid in enumerate(df.index)}
    if len(node_to_idx) != num_nodes:
        raise ValueError('Found duplicated string_id entries while building node index.')

    neg_candidates = np.asarray(neg_candidates, dtype=object)
    train_pos_idx = np.asarray(train_pos_idx, dtype=object)
    test_index = np.asarray(test_index, dtype=object)

    valid_train_pos = [gene for gene in train_pos_idx if gene in node_to_idx]
    if not valid_train_pos:
        raise ValueError('No valid training positives found for GNN training.')

    valid_neg_candidates = [gene for gene in neg_candidates if gene in node_to_idx]
    if not valid_neg_candidates:
        raise ValueError('No valid negative candidates available for sampling.')

    sampled_neg = rng.choice(valid_neg_candidates, size=neg_num, replace=True) if neg_num > 0 else np.array([], dtype=object)
    train_neg_idx = np.asarray(sampled_neg, dtype=object)

    train_pos_nodes = np.array([node_to_idx[g] for g in valid_train_pos], dtype=np.int64)
    train_neg_nodes = np.array([node_to_idx[g] for g in train_neg_idx], dtype=np.int64)
    train_nodes = np.unique(np.concatenate([train_pos_nodes, train_neg_nodes]))

    train_targets = labels.clone()
    if train_neg_nodes.size > 0:
        train_targets[torch.from_numpy(train_neg_nodes).to(device)] = 0
    train_targets[torch.from_numpy(train_pos_nodes).to(device)] = 1

    test_missing = [gene for gene in test_index if gene not in node_to_idx]
    if test_missing:
        raise KeyError(f'Test genes missing from feature set: {len(test_missing)} items.')
    test_nodes = np.array([node_to_idx[g] for g in test_index], dtype=np.int64)

    train_labels_np = train_targets.cpu().numpy()[train_nodes]
    unique_labels = np.unique(train_labels_np)

    if train_nodes.size == 0 or unique_labels.size == 0:
        raise ValueError('Training set for GNN is empty.')

    restarts = 3 if train_nodes.size >= 10 else 1
    splits = []

    if train_nodes.size < 2 or unique_labels.size < 2:
        splits.append((train_nodes, np.array([], dtype=np.int64), train_labels_np, np.array([], dtype=np.int64)))
    else:
        test_size = max(1, int(round(train_nodes.size * 0.2)))
        test_size = min(test_size, train_nodes.size - 1)
        for split_idx in range(restarts):
            for attempt in range(10):
                result = train_test_split(
                    train_nodes,
                    train_labels_np,
                    test_size=test_size,
                    stratify=train_labels_np,
                    random_state=seed + split_idx * 31 + attempt,
                )
                train_split, val_split, train_split_labels, val_split_labels = result
                if val_split.size == 0 or np.unique(val_split_labels).size < 2:
                    continue
                splits.append((train_split, val_split, train_split_labels, val_split_labels))
                break
            else:
                splits.append((train_nodes, np.array([], dtype=np.int64), train_labels_np, np.array([], dtype=np.int64)))

    best_overall_metric = float('-inf')
    best_val_auc = float('nan')
    best_preds = None

    criterion = nn.CrossEntropyLoss()
    fold_num = 0
    for train_split, val_split, train_split_labels, val_split_labels in splits:
        fold_num += 1
        # model = SimpleGCN(x.size(1)).to(device)
        model = ImprovedGCN(x.size(1)).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
        # optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

        train_split_tensor = torch.from_numpy(train_split).to(device)
        val_split_tensor = torch.from_numpy(val_split).to(device) if val_split.size > 0 else None

        best_split_metric = float('-inf')
        best_split_auc = float('nan')
        best_state = None
        patience = 30
        patience_counter = 0
        max_epochs = 300

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',          # we monitor val AUC (higher is better)
            factor=0.7,          # reduce LR by 30% each plateau
            patience=10,         # wait 10 epochs of no improvement before reducing
            min_lr=1e-5
        )

        for epoch in range(max_epochs):
            model.train()
            optimizer.zero_grad()
            logits = model(x, edge_index)
            loss = criterion(logits[train_split_tensor], train_targets[train_split_tensor])
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                logits = model(x, edge_index)
                probs = torch.softmax(logits, dim=1)[:, 1]

            # --- Validation ---
            if val_split.size > 0:
                val_scores = probs[val_split_tensor].detach().cpu().numpy()
                val_auc = _safe_roc_auc(val_split_labels, val_scores)
                compare_metric = val_auc if np.isfinite(val_auc) else float('-inf')
            else:
                val_auc = float('nan')
                compare_metric = -loss.item()

            # --- Scheduler step ---
            scheduler.step(compare_metric if np.isfinite(compare_metric) else 0.0)

            # --- Early stopping check ---
            if compare_metric > best_split_metric + 1e-6:
                best_split_metric = compare_metric
                best_split_auc = val_auc
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            current_lr = scheduler.optimizer.param_groups[0]["lr"]
            print(f"fold: {fold_num:2d}  epoch: {epoch:3d}  loss: {loss.item():.4f}  val auc: {val_auc:.4f}  lr: {current_lr:.6f}")

            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch} (best val AUC = {best_split_auc:.4f})")
                break

        if best_state is None:
            best_state = copy.deepcopy(model.state_dict())

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            final_logits = model(x, edge_index)
            final_probs = torch.softmax(final_logits, dim=1)[:, 1].detach().cpu().numpy()

        test_preds = final_probs[test_nodes]

        if best_split_metric > best_overall_metric:
            best_overall_metric = best_split_metric
            best_val_auc = best_split_auc
            best_preds = test_preds

    test_index_list = test_index.tolist()
    test_lookup = {gene: idx for idx, gene in enumerate(test_index_list)}
    overlap = set(train_neg_idx.tolist()) & set(test_index_list)
    mask_loc = sorted(test_lookup[gene] for gene in overlap)

    if best_preds is None:
        best_preds = np.zeros(len(test_index_list), dtype=float)

    return (best_preds, mask_loc), float(best_val_auc) if np.isfinite(best_val_auc) else float('nan')

class ImprovedGraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels=128, num_layers=2, dropout=0.5, num_classes=2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.num_layers = num_layers
        self.dropout = dropout

        # Input layer
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns.append(BatchNorm(hidden_channels))

        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(BatchNorm(hidden_channels))

        # MLP output head (two-layer improves expressiveness)
        self.output = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, num_classes)
        )

    def forward(self, x, edge_index):
        for conv, bn in zip(self.convs, self.bns):
            residual = x
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            # Residual connection (if same shape)
            x = x + residual if x.shape == residual.shape else x
        return self.output(x)


class SimpleGraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels=128, dropout=0.5):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.output = nn.Linear(hidden_channels, 2)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.output(x)

def neg_bagging_sage(args):
    (
        neg_candidates,
        neg_num,
        train_pos_idx,
        df,
        y,
        edge_index,
        feature_list,
        test_index,
        seed,
    ) = args

    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    feature_matrix = df.to_numpy(dtype=np.float32, copy=True)
    feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    x = torch.from_numpy(feature_matrix).to(device)

    labels = torch.from_numpy(np.asarray(y, dtype=np.int64)).to(device)

    if edge_index.size(0) != 2:
        raise ValueError('edge_index must have shape [2, num_edges].')

    num_nodes = x.size(0)
    edge_index = edge_index.to(device)
    edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

    node_to_idx = {sid: idx for idx, sid in enumerate(df.index)}
    if len(node_to_idx) != num_nodes:
        raise ValueError('Found duplicated string_id entries while building node index.')

    neg_candidates = np.asarray(neg_candidates, dtype=object)
    train_pos_idx = np.asarray(train_pos_idx, dtype=object)
    test_index = np.asarray(test_index, dtype=object)

    valid_train_pos = [gene for gene in train_pos_idx if gene in node_to_idx]
    if not valid_train_pos:
        raise ValueError('No valid training positives found for GNN training.')

    valid_neg_candidates = [gene for gene in neg_candidates if gene in node_to_idx]
    if not valid_neg_candidates:
        raise ValueError('No valid negative candidates available for sampling.')

    sampled_neg = rng.choice(valid_neg_candidates, size=neg_num, replace=True) if neg_num > 0 else np.array([], dtype=object)
    train_neg_idx = np.asarray(sampled_neg, dtype=object)

    train_pos_nodes = np.array([node_to_idx[g] for g in valid_train_pos], dtype=np.int64)
    train_neg_nodes = np.array([node_to_idx[g] for g in train_neg_idx], dtype=np.int64)
    train_nodes = np.unique(np.concatenate([train_pos_nodes, train_neg_nodes]))

    train_targets = labels.clone()
    if train_neg_nodes.size > 0:
        train_targets[torch.from_numpy(train_neg_nodes).to(device)] = 0
    train_targets[torch.from_numpy(train_pos_nodes).to(device)] = 1

    test_missing = [gene for gene in test_index if gene not in node_to_idx]
    if test_missing:
        raise KeyError(f'Test genes missing from feature set: {len(test_missing)} items.')
    test_nodes = np.array([node_to_idx[g] for g in test_index], dtype=np.int64)

    train_labels_np = train_targets.cpu().numpy()[train_nodes]
    unique_labels = np.unique(train_labels_np)

    if train_nodes.size == 0 or unique_labels.size == 0:
        raise ValueError('Training set for GNN is empty.')

    restarts = 3 if train_nodes.size >= 10 else 1
    splits = []

    if train_nodes.size < 2 or unique_labels.size < 2:
        splits.append((train_nodes, np.array([], dtype=np.int64), train_labels_np, np.array([], dtype=np.int64)))
    else:
        test_size = max(1, int(round(train_nodes.size * 0.2)))
        test_size = min(test_size, train_nodes.size - 1)
        for split_idx in range(restarts):
            for attempt in range(10):
                result = train_test_split(
                    train_nodes,
                    train_labels_np,
                    test_size=test_size,
                    stratify=train_labels_np,
                    random_state=seed + split_idx * 31 + attempt,
                )
                train_split, val_split, train_split_labels, val_split_labels = result
                if val_split.size == 0 or np.unique(val_split_labels).size < 2:
                    continue
                splits.append((train_split, val_split, train_split_labels, val_split_labels))
                break
            else:
                splits.append((train_nodes, np.array([], dtype=np.int64), train_labels_np, np.array([], dtype=np.int64)))

    best_overall_metric = float('-inf')
    best_val_auc = float('nan')
    best_preds = None

    criterion = nn.CrossEntropyLoss()
    fold_num = 0
    for train_split, val_split, train_split_labels, val_split_labels in splits:
        fold_num += 1
        
        model = ImprovedGraphSAGE(x.size(1)).to(device)
        # model = SimpleGraphSAGE(x.size(1)).to(device)
        # optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

        train_split_tensor = torch.from_numpy(train_split).to(device)
        val_split_tensor = torch.from_numpy(val_split).to(device) if val_split.size > 0 else None

        best_split_metric = float('-inf')
        best_split_auc = float('nan')
        best_state = None
        patience = 30
        patience_counter = 0
        max_epochs = 300

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',          # we monitor val AUC (higher is better)
            factor=0.7,          # reduce LR by 30% each plateau
            patience=10,         # wait 10 epochs of no improvement before reducing
            min_lr=1e-5
        )

        for epoch in range(max_epochs):
            model.train()
            optimizer.zero_grad()
            logits = model(x, edge_index)
            loss = criterion(logits[train_split_tensor], train_targets[train_split_tensor])
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                logits = model(x, edge_index)
                probs = torch.softmax(logits, dim=1)[:, 1]

            # --- Validation ---
            if val_split.size > 0:
                val_scores = probs[val_split_tensor].detach().cpu().numpy()
                val_auc = _safe_roc_auc(val_split_labels, val_scores)
                compare_metric = val_auc if np.isfinite(val_auc) else float('-inf')
            else:
                val_auc = float('nan')
                compare_metric = -loss.item()

            # --- Scheduler step ---
            scheduler.step(compare_metric if np.isfinite(compare_metric) else 0.0)

            # --- Early stopping check ---
            if compare_metric > best_split_metric + 1e-6:
                best_split_metric = compare_metric
                best_split_auc = val_auc
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
            current_lr = scheduler.optimizer.param_groups[0]["lr"]
            print(f"fold: {fold_num:2d}  epoch: {epoch:3d}  loss: {loss.item():.4f}  val auc: {val_auc:.4f}  lr: {current_lr:.6f}")

            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch} (best val AUC = {best_split_auc:.4f})")
                break

        if best_state is None:
            best_state = copy.deepcopy(model.state_dict())

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            final_logits = model(x, edge_index)
            final_probs = torch.softmax(final_logits, dim=1)[:, 1].detach().cpu().numpy()

        test_preds = final_probs[test_nodes]

        if best_split_metric > best_overall_metric:
            best_overall_metric = best_split_metric
            best_val_auc = best_split_auc
            best_preds = test_preds

    test_index_list = test_index.tolist()
    test_lookup = {gene: idx for idx, gene in enumerate(test_index_list)}
    overlap = set(train_neg_idx.tolist()) & set(test_index_list)
    mask_loc = sorted(test_lookup[gene] for gene in overlap)

    if best_preds is None:
        best_preds = np.zeros(len(test_index_list), dtype=float)

    return (best_preds, mask_loc), float(best_val_auc) if np.isfinite(best_val_auc) else float('nan')

# class SimpleGAT(nn.Module):
#     def __init__(self, in_channels, hidden_channels=128, heads=4, dropout=0.5):
#         super().__init__()
#         self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, concat=False, dropout=dropout)
#         self.conv2 = GATConv(hidden_channels, hidden_channels, heads=heads, concat=False, dropout=dropout)
#         self.output = nn.Linear(hidden_channels, 2)
#         self.dropout = dropout

#     def forward(self, x, edge_index):
#         x = self.conv1(x, edge_index)
#         x = F.elu(x)
#         x = F.dropout(x, p=self.dropout, training=self.training)
#         x = self.conv2(x, edge_index)
#         x = F.elu(x)
#         x = F.dropout(x, p=self.dropout, training=self.training)
#         return self.output(x)

# def neg_bagging_gat(args):
#     (
#         neg_candidates,
#         neg_num,
#         train_pos_idx,
#         df,
#         y,
#         edge_index,
#         feature_list,
#         test_index,
#         seed,
#     ) = args

#     np.random.seed(seed)
#     rng = np.random.default_rng(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)

#     feature_matrix = df.to_numpy(dtype=np.float32, copy=True)
#     feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)
#     x = torch.from_numpy(feature_matrix).to(device)

#     labels = torch.from_numpy(np.asarray(y, dtype=np.int64)).to(device)

#     if edge_index.size(0) != 2:
#         raise ValueError('edge_index must have shape [2, num_edges].')

#     num_nodes = x.size(0)
#     edge_index = edge_index.to(device)
#     edge_index = to_undirected(edge_index, num_nodes=num_nodes)
#     edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

#     node_to_idx = {sid: idx for idx, sid in enumerate(df.index)}
#     if len(node_to_idx) != num_nodes:
#         raise ValueError('Found duplicated string_id entries while building node index.')

#     neg_candidates = np.asarray(neg_candidates, dtype=object)
#     train_pos_idx = np.asarray(train_pos_idx, dtype=object)
#     test_index = np.asarray(test_index, dtype=object)

#     valid_train_pos = [gene for gene in train_pos_idx if gene in node_to_idx]
#     if not valid_train_pos:
#         raise ValueError('No valid training positives found for GNN training.')

#     valid_neg_candidates = [gene for gene in neg_candidates if gene in node_to_idx]
#     if not valid_neg_candidates:
#         raise ValueError('No valid negative candidates available for sampling.')

#     sampled_neg = rng.choice(valid_neg_candidates, size=neg_num, replace=True) if neg_num > 0 else np.array([], dtype=object)
#     train_neg_idx = np.asarray(sampled_neg, dtype=object)

#     train_pos_nodes = np.array([node_to_idx[g] for g in valid_train_pos], dtype=np.int64)
#     train_neg_nodes = np.array([node_to_idx[g] for g in train_neg_idx], dtype=np.int64)
#     train_nodes = np.unique(np.concatenate([train_pos_nodes, train_neg_nodes]))

#     train_targets = labels.clone()
#     if train_neg_nodes.size > 0:
#         train_targets[torch.from_numpy(train_neg_nodes).to(device)] = 0
#     train_targets[torch.from_numpy(train_pos_nodes).to(device)] = 1

#     test_missing = [gene for gene in test_index if gene not in node_to_idx]
#     if test_missing:
#         raise KeyError(f'Test genes missing from feature set: {len(test_missing)} items.')
#     test_nodes = np.array([node_to_idx[g] for g in test_index], dtype=np.int64)

#     train_labels_np = train_targets.cpu().numpy()[train_nodes]
#     unique_labels = np.unique(train_labels_np)

#     if train_nodes.size == 0 or unique_labels.size == 0:
#         raise ValueError('Training set for GNN is empty.')

#     restarts = 3 if train_nodes.size >= 10 else 1
#     splits = []

#     if train_nodes.size < 2 or unique_labels.size < 2:
#         splits.append((train_nodes, np.array([], dtype=np.int64), train_labels_np, np.array([], dtype=np.int64)))
#     else:
#         test_size = max(1, int(round(train_nodes.size * 0.2)))
#         test_size = min(test_size, train_nodes.size - 1)
#         for split_idx in range(restarts):
#             for attempt in range(10):
#                 result = train_test_split(
#                     train_nodes,
#                     train_labels_np,
#                     test_size=test_size,
#                     stratify=train_labels_np,
#                     random_state=seed + split_idx * 31 + attempt,
#                 )
#                 train_split, val_split, train_split_labels, val_split_labels = result
#                 if val_split.size == 0 or np.unique(val_split_labels).size < 2:
#                     continue
#                 splits.append((train_split, val_split, train_split_labels, val_split_labels))
#                 break
#             else:
#                 splits.append((train_nodes, np.array([], dtype=np.int64), train_labels_np, np.array([], dtype=np.int64)))

#     best_overall_metric = float('-inf')
#     best_val_auc = float('nan')
#     best_preds = None

#     criterion = nn.CrossEntropyLoss()
#     fold_num = 0
#     for train_split, val_split, train_split_labels, val_split_labels in splits:
#         fold_num += 1
#         model = SimpleGAT(x.size(1)).to(device)
#         optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

#         train_split_tensor = torch.from_numpy(train_split).to(device)
#         val_split_tensor = torch.from_numpy(val_split).to(device) if val_split.size > 0 else None

#         best_split_metric = float('-inf')
#         best_split_auc = float('nan')
#         best_state = None
#         patience = 30
#         patience_counter = 0
#         max_epochs = 300

#         for epoch in range(max_epochs):
#             model.train()
#             optimizer.zero_grad()
#             logits = model(x, edge_index)
#             loss = criterion(logits[train_split_tensor], train_targets[train_split_tensor])
#             loss.backward()
#             optimizer.step()

#             model.eval()
#             with torch.no_grad():
#                 logits = model(x, edge_index)
#                 probs = torch.softmax(logits, dim=1)[:, 1]

#             if val_split.size > 0:
#                 val_scores = probs[val_split_tensor].detach().cpu().numpy()
#                 val_auc = _safe_roc_auc(val_split_labels, val_scores)
#                 compare_metric = val_auc if np.isfinite(val_auc) else float('-inf')
#             else:
#                 val_auc = float('nan')
#                 compare_metric = -loss.item()

#             if compare_metric > best_split_metric + 1e-6:
#                 best_split_metric = compare_metric
#                 best_split_auc = val_auc
#                 best_state = copy.deepcopy(model.state_dict())
#                 patience_counter = 0
#             else:
#                 patience_counter += 1

#             if patience_counter >= patience:
#                 break
#             print('fold: ',fold_num, 'epoch: ',epoch,' loss: ',loss, ' val auc: ', val_auc)

#         if best_state is None:
#             best_state = copy.deepcopy(model.state_dict())

#         model.load_state_dict(best_state)
#         model.eval()
#         with torch.no_grad():
#             final_logits = model(x, edge_index)
#             final_probs = torch.softmax(final_logits, dim=1)[:, 1].detach().cpu().numpy()

#         test_preds = final_probs[test_nodes]

#         if best_split_metric > best_overall_metric:
#             best_overall_metric = best_split_metric
#             best_val_auc = best_split_auc
#             best_preds = test_preds

#     test_index_list = test_index.tolist()
#     test_lookup = {gene: idx for idx, gene in enumerate(test_index_list)}
#     overlap = set(train_neg_idx.tolist()) & set(test_index_list)
#     mask_loc = sorted(test_lookup[gene] for gene in overlap)

#     if best_preds is None:
#         best_preds = np.zeros(len(test_index_list), dtype=float)

#     return (best_preds, mask_loc), float(best_val_auc) if np.isfinite(best_val_auc) else float('nan')

# class SimpleGraphTransformer(nn.Module):
#     def __init__(self, in_channels, hidden_channels=128, heads=4, dropout=0.5):
#         super().__init__()
#         self.conv1 = TransformerConv(in_channels, hidden_channels, heads=heads, concat=False, dropout=dropout)
#         self.conv2 = TransformerConv(hidden_channels, hidden_channels, heads=heads, concat=False, dropout=dropout)
#         self.output = nn.Linear(hidden_channels, 2)
#         self.dropout = dropout

#     def forward(self, x, edge_index):
#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         x = F.dropout(x, p=self.dropout, training=self.training)
#         x = self.conv2(x, edge_index)
#         x = F.relu(x)
#         x = F.dropout(x, p=self.dropout, training=self.training)
#         return self.output(x)