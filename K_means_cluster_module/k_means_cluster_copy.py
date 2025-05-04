import torch
import os
import numpy as np
import global_resources as gr
import random

# Example Data process
# Read data
data_path = os.path.join(gr.default_dir, r'Data\breast-cancer-wisconsin.data')
df = gr.read_and_return_pd_df(data_path)

df.replace('?', np.nan, inplace=True)
df.dropna(inplace=True)
df.drop(['id'], axis=1, inplace=True)
df['bare_nuclei'] = df['bare_nuclei'].astype(np.int64)
df.dropna(inplace=True)

# Device & default dtype
device = gr.set_device()
print(f"Current device: {device.capitalize()}.")
DEFAULT_DTYPE = torch.float64

# Prepare data
X_cpu = np.array(df.drop(['class'], axis=1)).astype('float32')
X_gpu = torch.tensor(X_cpu, device=device)

# Core functions with flexible dtype

def initiate_centroids(X, k=3, random_seed=None, dtype=None):
    """
    Initialize k centroids sampled from X.
    Optionally specify dtype to cast centroids.
    """
    if random_seed is not None:
        random.seed(random_seed)
    N = X.size(0)
    idx_cpu = random.sample(range(N), k)
    idx = torch.tensor(idx_cpu, device=X.device, dtype=torch.long)
    centroids = X[idx]
    if dtype is not None:
        centroids = centroids.to(dtype)
    return centroids


def optimize_centroids(
    X,
    centroids=None,
    k=3,
    max_iters=1000,
    tol=1e-6,
    random_seed=None,
    dtype=None
):
    """
    Perform k-means until convergence or max_iters.
    If centroids is None, they will be initialized.
    dtype: if provided, casts intermediate tensors.
    """
    # ensure X in correct dtype
    if dtype is not None and X.dtype != dtype:
        X = X.to(dtype)

    if centroids is None:
        centroids = initiate_centroids(X, k=k, random_seed=random_seed, dtype=dtype)
    
    for it in range(max_iters):
        labels = torch.cdist(X, centroids).argmin(dim=1)
        new_centroids = torch.zeros(
            centroids.size(), device=X.device,
            dtype=dtype or centroids.dtype
        )
        for j in range(centroids.size(0)):
            members = X[labels == j]
            if members.numel() == 0:
                # reinitialize empty cluster
                centroids = initiate_centroids(X, k=centroids.size(0), random_seed=random_seed, dtype=dtype)
                break
            new_centroids[j] = members.mean(dim=0)
        if torch.allclose(new_centroids, centroids, atol=tol):
            labels = torch.cdist(X, new_centroids).argmin(dim=1)
            centroids = new_centroids
            break
        centroids = new_centroids
    return centroids, labels


def calculate_variation(X, centroids, labels, dtype=None):
    """
    Compute within-cluster sum of squares.
    X, centroids: tensors. labels: assignment.
    dtype: optional cast.
    """
    if dtype is not None and X.dtype != dtype:
        X = X.to(dtype)
    variation = 0.0
    for j in range(centroids.size(0)):
        members = X[labels == j]
        diffs = members - centroids[j].unsqueeze(0)
        variation += torch.sum(diffs * diffs).item()
    return variation


def WCSS_for_single_k(
    X,
    k,
    n_restarts=10,
    tol=1e-6,
    max_iters=1000,
    dtype=None
):
    """
    Compute best WCSS for given k over multiple restarts.
    dtype: if provided, used for all computations; else uses X.dtype.
    """
    best = {'variation': float('inf'), 'centroids': None, 'labels': None}
    for r in range(n_restarts):
        seed = None if r is None else r
        centroids_init = initiate_centroids(X, k=k, random_seed=seed, dtype=dtype)
        centroids, labels = optimize_centroids(
            X, centroids_init, k=k,
            max_iters=max_iters, tol=tol,
            random_seed=seed, dtype=dtype
        )
        var = calculate_variation(X, centroids, labels, dtype=dtype)
        if var < best['variation']:
            best.update({'variation': var, 'centroids': centroids.clone(), 'labels': labels.clone()})
    return X, best['labels'], best['centroids'], best['variation']
