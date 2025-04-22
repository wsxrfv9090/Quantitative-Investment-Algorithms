import torch
import os
import numpy as np
import global_resources as gr
import random
import matplotlib.pyplot as plt

# Example Data process
# Read data
data_path = os.path.join(gr.default_dir, r'Data\breast-cancer-wisconsin.data')
df = gr.read_and_return_pd_df(data_path)


df.replace('?', np.nan, inplace = True)
df.dropna(inplace = True)
# df.replace('?', -99999, inplace = True)
df.drop(['id'], axis = 1, inplace = True)
df["bare_nuclei"] = df["bare_nuclei"].astype(np.int64)
df.dropna(inplace = True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Current device: {device.capitalize()}.")

X = np.array(df.drop(['class'], axis = 1)).astype('float64')
X_gpu = torch.tensor(X, device = device)

# Important Parameters
RANDOM_SEED = random.randint(0, 2**32 - 1)
MAX_ITERATION = int(1e3)
TOLERANCE = 1e-6
N_RESTARTS = int(10)

# Initiates k centroids for X tensor
def initiate_centroids(X = X_gpu, k = 3, random_seed = RANDOM_SEED):
    N = X.size(0)
    # GPU ways of doing it, but it randomly permute all data before sampling, and isn't the best practice:
    # generator = torch.Generator(device=X_gpu.device).manual_seed(RANDOM_SEED)
    # idx = torch.randperm(X_gpu.size(0), generator = generator, device = X_gpu.device)[:k]
    random.seed(random_seed)
    # draw k unique ints in [0..N-1]
    idx_cpu = random.sample(range(N), k)    
    # convert & move to GPU
    idx = torch.tensor(idx_cpu, device=X_gpu.device)
    centroids = X[idx]
    return centroids

# Usage for init_centroids
# Centroids = init_centroids(X_gpu)

# This function uses max_iters to optimize centroids base on the difference between new and old centroids, with tol being the tolerance, returns the optimized centroids and the corresponding labels
def optimize_centroids(X = X_gpu, centroids = None, max_iters = MAX_ITERATION, tol = TOLERANCE):
    labels = 0
    global RANDOM_SEED
    if centroids == None:
        centroids = initiate_centroids(X)
    for it in range(max_iters):
        labels = torch.cdist(X_gpu, centroids).argmin(dim = 1)
        new_centroids = torch.zeros_like(centroids)
        
        for j in range(centroids.size(0)):
            members = X_gpu[labels == j]
            if members.numel() == 0:
                RANDOM_SEED += 1
                centroids = initiate_centroids(X, k=centroids.size(0), random_seed = RANDOM_SEED)
            else:
                new_centroids[j] = members.mean(dim = 0)

        # If the new_centroids and old centroids are not more different than tol, it will break the loop
        if torch.allclose(new_centroids, centroids, atol = tol):
            # print(f"Converged at iter {it}")
            labels = torch.cdist(X_gpu, new_centroids).argmin(dim = 1)
            break
        if it == max_iters:
            print(f"Hit max_iters in optimize_centroids function!!!!")
        centroids = new_centroids
        
    return new_centroids, labels



# Usage for optimize_centroids
# Centroids, Labels = optimize_centroids(Centroids)

# Calculates the variation for a given tensor X, it's centroids and it's corresponding labels
def calculate_variation(X = X_gpu, centroids = None, labels = None):
    variation = 0.0
    if centroids == None or labels == None:
        centroids, labels = optimize_centroids(X)
    for j in range(centroids.size(0)):
        members = X_gpu[labels == j]
        diffs = members - centroids[j].unsqueeze(0)
        variation += torch.sum(diffs * diffs).item()
    return variation

# variation = calculate_variation() 



# Within cluster sum of squares
# This function calculates the sum of squares within clusters for each restarts, and choose the best one, and it does that by accessing the variation for each restarts, note that this n_restarts is the way you adjust your parameters, when N_RESTARTS is too low, it probably won't get the "best" variation, but when N_RESTARTS is too high, it'll take forever but increase the reliability.
def WCSS_for_single_k(X = X_gpu, k = 3, n_restarts = N_RESTARTS, tol = TOLERANCE, max_iters = MAX_ITERATION):
    global RANDOM_SEED
    best_variation = float('inf')
    best_centroids = None
    best_labels = None

    for _ in range(n_restarts):
        RANDOM_SEED += 1
        Centroids = initiate_centroids(X, k, random_seed = RANDOM_SEED)
        
        Centroids, Labels = optimize_centroids(
            X = X,
            centroids = Centroids,
            max_iters = max_iters,
            tol = tol
        )

        var = calculate_variation(X, Centroids, Labels)

        # keep best
        if var < best_variation:
            best_variation = var
            best_centroids = Centroids.clone()
            best_labels = Labels.clone()

    return X, best_labels, best_centroids, best_variation



# This visualizes a plot and return the variations for different k, by difault the k is in range of 1 to 10
# Note that this function calls all of the function above, and it's time complexity is highly depandent on MAX_ITERATION & N_RESTARTS & k
def k_means_assessment(X = X_gpu, k = 10):
    ks = list(range(1, k + 1))
    variations = []

    for k in ks:
        _, _, _, var = WCSS_for_single_k(X, k)
        variations.append(var)
    
    if not variations:
        print("No valid results to plot.")
        return
    
    plt.figure(figsize=(8, 5))
    plt.scatter(ks[:len(variations)], variations, marker='o')
    plt.plot(ks[:len(variations)], variations, linestyle='-', alpha=0.7)
    plt.xticks(ks)
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Within-cluster variation")
    plt.title("Elbow Method for Optimal k")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()
    return variations

k_means_assessment()