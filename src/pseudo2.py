import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import fcluster, linkage

def calculate_proportion(hierarchical_labels, y_train, func):
    proportion = []
    unique_clusters = np.unique(hierarchical_labels)
    
    # Total number of samples with label 1
    total_y1 = np.sum(y_train == 1)
    if total_y1 == 0:
        raise ValueError("No positive samples in training data")

    for c in unique_clusters:
        cluster_indices = np.where(hierarchical_labels == c)[0]
        cluster_y1 = np.sum(y_train[cluster_indices] == 1)
        
        if func == 1:
            A_i = len(cluster_indices) * (1 - (cluster_y1 / total_y1))
        elif func == 2:
            A_i = max(0, 2 * len(cluster_indices) / len(y_train) - (cluster_y1 / total_y1))
        elif func == 3:
            A_i = max(0, 2 * len(cluster_indices) / len(y_train) - (cluster_y1 / total_y1))
            if cluster_y1 == 0:
                # A_i = np.exp(0.05 * A_i)
                A_i = A_i**0.5
        
        proportion.append(A_i)
    
    return proportion

def cluster_negative_sampling(hierarchical_labels, y_train, proportion, neg_size):
    neg_cluster = []
    all_sampled_negatives = []
    unique_clusters = np.unique(hierarchical_labels)
    total_proportion = sum(proportion)

    if total_proportion == 0:
        raise ValueError("Sum of proportions is zero. Check the 'proportion' calculation.")

    for index, c in enumerate(unique_clusters):
        A_i = proportion[index]
        neg_i = int((A_i / total_proportion) * neg_size)

        cluster_indices = np.where(hierarchical_labels == c)[0]
        negative_indices = cluster_indices[y_train[cluster_indices] == 0]

        if len(negative_indices) == 0:
            continue  
        neg_i = min(neg_i, len(negative_indices))  # Adjust to max available

        sampled_indices = np.random.choice(negative_indices, size=neg_i, replace=False)
        all_sampled_negatives.extend(sampled_indices)
        neg_cluster.append(neg_i)

    return all_sampled_negatives

def select_pseudo_negatives(neg_num, X_train_pos, y_train_pos, X_all_neg, y_all_neg, func):
    # Combine positive and negative samples
    X_train = np.vstack((X_train_pos, X_all_neg))
    y_train = np.hstack((y_train_pos, y_all_neg))
    # print(f'start clustering')

    # Hierarchical clustering
    model = AgglomerativeClustering(distance_threshold=0, n_clusters=None, compute_distances=True)
    model.fit(X_train)

    # Get linkage matrix
    mat = linkage(X_train, method='ward')

    # Get first-level labels
    n_clusters = 5
    hierarchical_labels = fcluster(mat, n_clusters, criterion='maxclust')

    # Sample negatives
    proportion = calculate_proportion(hierarchical_labels, y_train, func)
    all_sampled_negatives = cluster_negative_sampling(hierarchical_labels, y_train, proportion, neg_num)
    # print(f'get neg samples')

    return all_sampled_negatives, X_train[all_sampled_negatives], y_train[all_sampled_negatives]