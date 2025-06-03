import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import fcluster

def getLinkageMat(model):
    children = model.children_
    cs = np.zeros(len(children))
    N = len(model.labels_)
    for i,child in enumerate(children):
        count = 0
        for idx in child:
            count += 1 if idx < N else cs[idx - N]
        cs[i] = count
    return np.column_stack([children, model.distances_, cs])

def calculate_proportion(hierarchical_labels,y_train,func):
    proportion = []
    unique_clusters = np.unique(hierarchical_labels)
    # Total number of samples with label 1
    total_y1 = np.sum(y_train == 1)
    # print('total positive',total_y1)
    for c in unique_clusters:
        # Indices of samples in the current cluster
        cluster_indices = np.where(hierarchical_labels == c)[0]
        
        # Number of samples with label 1 in this cluster
        cluster_y1 = np.sum(y_train[cluster_indices] == 1)
        if func == 1:
            # Calculate A_i using the corrected formula
            A_i = len(cluster_indices) * (1 - (cluster_y1 / total_y1))
        elif func == 2:
            A_i = 2*len(cluster_indices)/len(y_train) - (cluster_y1 / total_y1)
            if A_i < 0:
                A_i = 0
        
        proportion.append(A_i)
    return proportion

def cluster_negative_sampling(hierarchical_labels,y_train,proportion,neg_size):
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
        negative_indices = np.array(cluster_indices)[y_train[cluster_indices] == 0]
        # Check if there are enough negative samples to draw from
        if len(negative_indices) == 0:
            continue  
        elif neg_i > len(negative_indices):
            neg_i = len(negative_indices)  # Adjust to the maximum available
        # Sample negative indices without replacement
        sampled_indices = np.random.choice(negative_indices, size=neg_i, replace=False)
        # Store the sampled indices and the count
        all_sampled_negatives.extend(sampled_indices)
        neg_cluster.append(neg_i)

    return all_sampled_negatives

def select_pseudo_negatives(neg_num,X_train_pos,y_train_pos,X_all_neg,y_all_neg,func):
    # clustering
    X_train = np.vstack((X_train_pos,X_all_neg))
    y_train = np.hstack((y_train_pos,y_all_neg))
    model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
    model = model.fit(X_train)
    mat = getLinkageMat(model)
    # get first level labels
    n_clusters = 5
    hierarchical_labels = fcluster(mat, n_clusters, criterion='maxclust')
    # sample negatives
    proportion = calculate_proportion(hierarchical_labels,y_train,func)
    all_sampled_negatives = cluster_negative_sampling(hierarchical_labels,y_train,proportion,neg_num)

    return X_train[all_sampled_negatives], y_train[all_sampled_negatives]