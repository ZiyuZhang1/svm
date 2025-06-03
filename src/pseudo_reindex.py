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

def cluster_negative_sampling(hierarchical_labels, y_train, proportion, neg_size, original_indices):
    neg_cluster = []
    all_sampled_indices = []  # Store the indices within the combined dataset
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
        all_sampled_indices.extend(sampled_indices)
        neg_cluster.append(neg_i)

    return all_sampled_indices

def select_pseudo_negatives(neg_num, X_train_pos, y_train_pos, X_all_neg, y_all_neg, func, neg_indices=None):
    # If no indices provided, create dummy ones
    if neg_indices is None:
        neg_indices = np.arange(len(X_all_neg))
    
    # Track original positions
    pos_count = len(X_train_pos)
    
    # Combine positive and negative samples
    X_train = np.vstack((X_train_pos, X_all_neg))
    y_train = np.hstack((y_train_pos, y_all_neg))
    
    # Keep track of original indices
    combined_indices = np.arange(len(X_train))  # Indices in the combined array
    
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
    sampled_combined_indices = cluster_negative_sampling(hierarchical_labels, y_train, proportion, neg_num, combined_indices)
    
    # Convert to indices within the original negative set
    # Adjust indices to be relative to the negative set
    neg_relative_indices = [idx - pos_count for idx in sampled_combined_indices]
    
    # Get the original indices from the provided neg_indices
    selected_original_indices = [neg_indices[i] for i in neg_relative_indices]
    
    # Get the selected samples
    X_selected = X_all_neg[neg_relative_indices]
    y_selected = y_all_neg[neg_relative_indices]
    
    return sampled_combined_indices, neg_relative_indices, X_selected, y_selected

def calculate_proportion_mask(hierarchical_labels, y_train, func, original_indices=None, mask_ids=None):
    proportion = []
    unique_clusters = np.unique(hierarchical_labels)
    
    # Create a modified y_train array that treats mask_ids as positives
    y_train_modified = y_train.copy()
    
    # If mask_ids are provided, mark them as positive samples in y_train_modified
    if mask_ids is not None and original_indices is not None:
        mask_ids_set = set(mask_ids)
        # Starting from the first negative sample in the combined dataset
        offset = len(y_train) - len(original_indices)
        for i in range(len(original_indices)):
            if original_indices[i] in mask_ids_set:
                y_train_modified[i + offset] = 1  # Mark as positive
    
    # Total number of samples with label 1 (including those in mask_ids)
    total_y1 = np.sum(y_train_modified == 1)
    if total_y1 == 0:
        raise ValueError("No positive samples in training data")

    for c in unique_clusters:
        cluster_indices = np.where(hierarchical_labels == c)[0]
        cluster_y1 = np.sum(y_train_modified[cluster_indices] == 1)
        
        if func == 1:
            A_i = len(cluster_indices) * (1 - (cluster_y1 / total_y1))
        elif func == 2:
            A_i = max(0, 2 * len(cluster_indices) / len(y_train_modified) - (cluster_y1 / total_y1))
        elif func == 3:
            A_i = max(0, 2 * len(cluster_indices) / len(y_train_modified) - (cluster_y1 / total_y1))
            if cluster_y1 == 0:
                # A_i = np.exp(0.05 * A_i)
                A_i = A_i**0.5
        
        proportion.append(A_i)
    
    return proportion

def cluster_negative_sampling_mask(hierarchical_labels, y_train, proportion, neg_size, original_indices, mask_ids=None):
    neg_cluster = []
    all_sampled_indices = []  # Store the indices within the combined dataset
    unique_clusters = np.unique(hierarchical_labels)
    total_proportion = sum(proportion)

    if total_proportion == 0:
        raise ValueError("Sum of proportions is zero. Check the 'proportion' calculation.")
    
    # Convert mask_ids to a set for faster lookup if provided
    mask_ids_set = set(mask_ids) if mask_ids is not None else set()

    for index, c in enumerate(unique_clusters):
        A_i = proportion[index]
        neg_i = int((A_i / total_proportion) * neg_size)

        cluster_indices = np.where(hierarchical_labels == c)[0]
        negative_indices = cluster_indices[y_train[cluster_indices] == 0]

        if len(negative_indices) == 0:
            continue  
            
        # Filter out mask_ids
        valid_indices = []
        for idx in negative_indices:
            # Check if this is a positive sample (will be in first positions) or if it's a negative sample not in mask_ids
            if idx < len(y_train) - len(original_indices) or original_indices[idx - (len(y_train) - len(original_indices))] not in mask_ids_set:
                valid_indices.append(idx)
                
        if not valid_indices:
            continue  # Skip if no valid indices for this cluster
            
        neg_i = min(neg_i, len(valid_indices))  # Adjust to max available

        sampled_indices = np.random.choice(valid_indices, size=neg_i, replace=False)
        all_sampled_indices.extend(sampled_indices)
        neg_cluster.append(neg_i)

    return all_sampled_indices

def select_pseudo_negatives_mask(neg_num, X_train_pos, y_train_pos, X_all_neg, y_all_neg, func, neg_indices=None, mask_ids=None):
    # If no indices provided, create dummy ones
    if neg_indices is None:
        neg_indices = np.arange(len(X_all_neg))
    
    # Track original positions
    pos_count = len(X_train_pos)
    
    # Combine positive and negative samples
    X_train = np.vstack((X_train_pos, X_all_neg))
    y_train = np.hstack((y_train_pos, y_all_neg))
    
    # Keep track of original indices
    combined_indices = np.arange(len(X_train))  # Indices in the combined array
    
    # Hierarchical clustering
    model = AgglomerativeClustering(distance_threshold=0, n_clusters=None, compute_distances=True)
    model.fit(X_train)

    # Get linkage matrix
    mat = linkage(X_train, method='ward')

    # Get first-level labels
    n_clusters = 5
    hierarchical_labels = fcluster(mat, n_clusters, criterion='maxclust')

    # Sample negatives - pass original indices and mask_ids to consider mask_ids as positives
    proportion = calculate_proportion_mask(hierarchical_labels, y_train, func, neg_indices, mask_ids)
    sampled_combined_indices = cluster_negative_sampling_mask(hierarchical_labels, y_train, proportion, neg_num, neg_indices, mask_ids)
    
    # Convert to indices within the original negative set
    # Adjust indices to be relative to the negative set
    neg_relative_indices = [idx - pos_count for idx in sampled_combined_indices]
    
    # Get the selected samples
    X_selected = X_all_neg[neg_relative_indices]
    y_selected = y_all_neg[neg_relative_indices]
    
    return sampled_combined_indices, neg_relative_indices, X_selected, y_selected