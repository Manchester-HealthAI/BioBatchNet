import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import seaborn as sns
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import random
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from collections import defaultdict

def visualize(X, y, centers=None):
    if centers is not None:
        combined = np.vstack([X, centers])
    else:
        combined = X

    tsne = TSNE(n_components=2, random_state=42)
    combined_tsne = tsne.fit_transform(combined)

    X_tsne = combined_tsne[:len(X)]
    centers_tsne = combined_tsne[len(X):] if centers is not None else None

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', alpha=0.5, s=10, label='Data Points')

    if centers_tsne is not None:
        plt.scatter(
            centers_tsne[:, 0], 
            centers_tsne[:, 1], 
            c='red', 
            marker='X', 
            s=200, 
            label='Cluster Centers'
        )
    plt.title('K-Means Clustering Results (t-SNE Reduced to 2D)')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend()
    plt.colorbar(scatter, label='Cluster Labels')
    plt.show()

def get_cluster_labels(model, data_loader, device):
    model.eval()
    y_pred = []
    embeddings = []
    labels = []

    with torch.no_grad():
        for batch_data in tqdm(data_loader, desc="get labels"):
            x, y = batch_data
            x = x.to(device)
            z, _ = model.autoencoder(x)
            q = model(x)
            _, pred = torch.max(q, dim=1)
    
            embeddings.append(z.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())   
            labels.append(y.detach().cpu().numpy()) 
            
        embeddings = np.vstack(embeddings)
        labels = np.concatenate(labels, axis=0)
    return embeddings, np.array(y_pred), labels

def target_distribution(q, order=2):
    weight = q**order / (q.sum(0) + 1) 
    return (weight.t() / weight.sum(1)).t()

def soft_assign(z, center, alpha=1):
    q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - center, 2), 2) / alpha)
    q = q.pow((alpha + 1.0) / 2.0)
    q = (q.t() / (torch.sum(q, 1) + 1e-8) ).t()
    return q

def pairwise_loss(p1, p2, cons_type):
    if cons_type == "ML":
        ml_loss = torch.mean(-torch.log(torch.sum(p1 * p2, dim=1)))
        return ml_loss
    else:
        cl_loss = torch.mean(-torch.log(1.0 - torch.sum(p1 * p2, dim=1)))
        return cl_loss

def generate_random_pair(y, num):
    """
    Generate random pairwise constraints.
    """
    ml_ind1, ml_ind2 = [], []
    cl_ind1, cl_ind2 = [], []
    while num > 0:
        tmp1 = random.randint(0, y.shape[0] - 1)
        tmp2 = random.randint(0, y.shape[0] - 1)
        if tmp1 == tmp2:
            continue
        if y[tmp1] == y[tmp2]:
            ml_ind1.append(tmp1)
            ml_ind2.append(tmp2)
        else:
            cl_ind1.append(tmp1)
            cl_ind2.append(tmp2)
        num -= 1
    return np.array(ml_ind1), np.array(ml_ind2), np.array(cl_ind1), np.array(cl_ind2)


def transitive_closure(ml_ind1, ml_ind2, cl_ind1, cl_ind2, n):
    """
    This function calculate the total transtive closure for must-links and the full entailment
    for cannot-links. 
    
    # Arguments
        ml_ind1, ml_ind2 = instances within a pair of must-link constraints
        cl_ind1, cl_ind2 = instances within a pair of cannot-link constraints
        n = total training instance number

    # Return
        transtive closure (must-links)
        entailment of cannot-links
    """
    ml_graph = dict()
    cl_graph = dict()
    for i in range(n):
        ml_graph[i] = set()
        cl_graph[i] = set()

    def add_both(d, i, j):
        d[i].add(j)
        d[j].add(i)

    for (i, j) in zip(ml_ind1, ml_ind2):
        add_both(ml_graph, i, j)

    def dfs(i, graph, visited, component):
        visited[i] = True
        for j in graph[i]:
            if not visited[j]:
                dfs(j, graph, visited, component)
        component.append(i)

    visited = [False] * n
    for i in range(n):
        if not visited[i]:
            component = []
            dfs(i, ml_graph, visited, component)
            for x1 in component:
                for x2 in component:
                    if x1 != x2:
                        ml_graph[x1].add(x2)
    for (i, j) in zip(cl_ind1, cl_ind2):
        add_both(cl_graph, i, j)
        for y in ml_graph[j]:
            add_both(cl_graph, i, y)
        for x in ml_graph[i]:
            add_both(cl_graph, x, j)
            for y in ml_graph[j]:
                add_both(cl_graph, x, y)
    ml_res_set = set()
    cl_res_set = set()
    for i in ml_graph:
        for j in ml_graph[i]:
            if j != i and j in cl_graph[i]:
                raise Exception('inconsistent constraints between %d and %d' % (i, j))
            if i <= j:
                ml_res_set.add((i, j))
            else:
                ml_res_set.add((j, i))
    for i in cl_graph:
        for j in cl_graph[i]:
            if i <= j:
                cl_res_set.add((i, j))
            else:
                cl_res_set.add((j, i))
    ml_res1, ml_res2 = [], []
    cl_res1, cl_res2 = [], []
    for (x, y) in ml_res_set:
        ml_res1.append(x)
        ml_res2.append(y)
    for (x, y) in cl_res_set:
        cl_res1.append(x)
        cl_res2.append(y)
    return np.array(ml_res1), np.array(ml_res2), np.array(cl_res1), np.array(cl_res2)

def align_labels(reference_labels, labels, n_clusters):
    cost_matrix = np.zeros((n_clusters, n_clusters))
    for c1 in range(n_clusters):
        for c2 in range(n_clusters):
            cost_matrix[c1, c2] = np.sum((reference_labels == c1) & (labels == c2))

    row_ind, col_ind = linear_sum_assignment(-cost_matrix)  
    label_mapping = {col_ind[i]: row_ind[i] for i in range(n_clusters)}
    aligned_labels = np.vectorize(label_mapping.get)(labels)  
    return aligned_labels

def compute_coassociation_matrix(X, n_clusters=7, n_runs=100, feature_ratio=1):
    if isinstance(X, torch.Tensor):
        tem_X = X.detach().cpu().numpy()
    else:
        tem_X = X

    n_samples, n_features = tem_X.shape
    cluster_labels = np.zeros((n_samples, n_runs), dtype=int)

    for i in range(n_runs):
        feature_indices = np.random.choice(n_features, size=int(n_features * feature_ratio), replace=False)
        X_features = tem_X[:, feature_indices]  
        
        # using the whole data
        kmeans = KMeans(n_clusters=n_clusters, init='random', random_state=i, n_init=20)
        labels = kmeans.fit_predict(X_features)
        
        if i == 0:
            reference_labels = labels  
        else:
            # Align the labels using reference labels
            labels = align_labels(reference_labels, labels, n_clusters)  
        
        cluster_labels[:, i] = labels

    co_matrix = np.zeros((n_samples, n_samples))

    for i in range(n_runs):
        labels = cluster_labels[:, i]
        same_cluster = labels[:, np.newaxis] == labels[np.newaxis, :]
        co_matrix += same_cluster

    # Normalize co-association matrix
    co_matrix /= n_runs

    return co_matrix, cluster_labels

def hierarchy_cluster_with_centroids(X: np.ndarray, eac_matrix: np.ndarray, n_clusters: int = 7) -> tuple:

    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()
    else:
        X = X
    distance_matrix = 1 - eac_matrix
    np.fill_diagonal(distance_matrix, 0)
    condensed_distance = squareform(distance_matrix, checks=False)
    linkage_matrix = linkage(condensed_distance, method='average')
    clusters = fcluster(linkage_matrix, t=n_clusters, criterion='maxclust')
    
    centroids = np.zeros((n_clusters, X.shape[1]))
    for cluster_id in range(1, n_clusters + 1):
        cluster_points = X[clusters == cluster_id]
        centroids[cluster_id - 1] = cluster_points.mean(axis=0)
    
    return clusters, centroids

def plot_with_links(x, y, ml_ind1, ml_ind2, cl_ind1, cl_ind2, sample_frac=0.1, tsne_perplexity=40, tsne_iter=350):
    data = pd.DataFrame(x)
    data['label'] = y

    # Sample data
    sample = data.sample(frac=sample_frac, random_state=42)
    sampled_x = sample.iloc[:, :-1]
    sampled_y = sample['label']

    # Perform t-SNE
    tsne = TSNE(n_components=2, perplexity=tsne_perplexity, n_iter=tsne_iter, verbose=1)
    tsne_results = tsne.fit_transform(sampled_x)

    # Add t-SNE results back to the sampled DataFrame
    sample = sample.copy()
    sample['tsne-1'] = tsne_results[:, 0]
    sample['tsne-2'] = tsne_results[:, 1]

    plt.figure(figsize=(16, 10))

    # Plot scatter
    sns.scatterplot(
        x=sample['tsne-1'], y=sample['tsne-2'],
        hue=sample['label'],
        palette=sns.color_palette("hls", len(sample['label'].unique())),
        legend="auto",
        alpha=0.8,
        s=20
    )

    # Plot must-links
    valid_ml_pairs = [(idx1, idx2) for idx1, idx2 in zip(ml_ind1, ml_ind2) if idx1 in sample.index and idx2 in sample.index]
    print(f"number of must-link: {len(valid_ml_pairs)}")
    for i, (idx1, idx2) in enumerate(valid_ml_pairs):
        plt.plot(
            [sample.loc[idx1, 'tsne-1'], sample.loc[idx2, 'tsne-1']],
            [sample.loc[idx1, 'tsne-2'], sample.loc[idx2, 'tsne-2']],
            'b-', alpha=0.3, label="must link" if i == 0 else None
        )

    # 绘制 cannot-links
    valid_cl_pairs = [(idx1, idx2) for idx1, idx2 in zip(cl_ind1, cl_ind2) if idx1 in sample.index and idx2 in sample.index]
    print(f"number of cannot-link: {len(valid_cl_pairs)}")
    for i, (idx1, idx2) in enumerate(valid_cl_pairs):
        plt.plot(
            [sample.loc[idx1, 'tsne-1'], sample.loc[idx2, 'tsne-1']],
            [sample.loc[idx1, 'tsne-2'], sample.loc[idx2, 'tsne-2']],
            'r--', alpha=0.3, label="cannot link" if i == 0 else None
        )


    # Remove duplicate labels
    handles, labels = plt.gca().get_legend_handles_labels()
    newLabels, newHandles = [], []
    for handle, label in zip(handles, labels):
        if label not in newLabels:
            newLabels.append(label)
            newHandles.append(handle)

    # Save plot
    plt.show()

