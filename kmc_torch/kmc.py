import torch
import os
import numpy as np
import global_resources as gr
import random
import matplotlib.pyplot as plt

# Important Parameters
# The seed have to be a 32 bit interger value, because manual_seed only takes a 32 bit interger
RANDOM_SEED = random.getrandbits(32)
MAX_ITERATION = int(1e3)
TOLERANCE = 1e-6
N_RESTARTS = int(10)
DTYPE = torch.float32


# Initiates k centroids for X tensor
def initiate_centroids(
    X: torch.Tensor, 
    k: int = 3, 
    random_seed: int = RANDOM_SEED
    ) -> torch.Tensor:
    # size(0) meaning the total row number of X
    print(f"Initiating centroids with k being {k}...")
    N = X.size(0)
    
    # Use a generator to get reproductivity
    generator = torch.Generator(device = X.device).manual_seed(random_seed)
    
    # Generate all 1s to get a evenly distributed probabilities for multinomial to drow, because torch.multinomial draws non zero values based on their total probability, e.g. if [0, 10, 3, 0] is weights, the indices will be drawed as only 1 or 2 because the probability of draw first and last element is: 0/13 which is zero and will never be drawed as indeces. 
    # Hence the ons initiating, with size of X's row number as it's shape.
    weights = torch.ones(N, device = X.device)
    idx = torch.multinomial(weights, k, replacement = False, generator = generator)
    
    # convert & move to GPU
    centroids = X[idx]
    return centroids


# This function uses max_iters to optimize centroids base on the difference between new and old centroids, with tol being the tolerance, returns the optimized centroids and the corresponding labels
def optimize_centroids(
    X: torch.Tensor, 
    k: int = 3, 
    centroids: torch.Tensor = None, 
    max_iters: int = MAX_ITERATION, 
    tol: float = TOLERANCE
    ) -> tuple[torch.Tensor, torch.Tensor]:
    labels = 0
    global RANDOM_SEED
    
    # If centroids is none, initiate them
    if centroids == None:
        centroids = initiate_centroids(X, k)
        
    
    # If the rows number of centroids doesn't match with k, raise error
    # This will only happen if you pass in centroids yourself as a parameter which doesn't match the k
    if centroids.size(0) != k:
        print(f"Total row number of centroids is not equal to k passed in.")
        return None
    
    for it in range(max_iters):
        labels = torch.cdist(X, centroids).argmin(dim = 1)
        new_centroids = torch.zeros_like(centroids, device = X.device, dtype = centroids.dtype)
        
        for j in range(centroids.size(0)):
            members = X[labels == j]
            if members.numel() == 0:
                RANDOM_SEED = gr.reinitiate_seed_torch()
                centroids = initiate_centroids(X, k = k, random_seed = RANDOM_SEED)
            else:
                new_centroids[j] = members.mean(dim = 0)

        # If the new_centroids and old centroids are not more different than tol, it will break the loop
        if torch.allclose(new_centroids, centroids, atol = tol):
            labels = torch.cdist(X, new_centroids).argmin(dim = 1)
            break
        if it == max_iters:
            print(f"Hit max_iters in optimize_centroids function!!!! Breaking it without cenrtoids being successfully optimized.")
        centroids = new_centroids
        
    return new_centroids, labels


# Calculates the variation for a given tensor X, it's centroids and it's corresponding labels
def calculate_variation(
    X: torch.Tensor, 
    centroids: torch.Tensor = None, 
    labels: torch.Tensor = None
    ) -> float:
    variation = 0.0
    if centroids == None or labels == None:
        print(f"Centroids or labels was not passed in to function: calculate_variation.")
        return None
    for j in range(centroids.size(0)):
        members = X[labels == j]
        diffs = members - centroids[j].unsqueeze(0)
        variation += torch.sum(diffs * diffs).item()
    return variation


# Within cluster sum of squares
# This function calculates the sum of squares within clusters for each restarts, and choose the best one, and it does that by accessing the variation for each restarts, note that this n_restarts is the way you adjust your parameters, when N_RESTARTS is too low, it probably won't get the "best" variation, but when N_RESTARTS is too high, it'll take forever but increase the reliability.
def WCSS_for_single_k(
    X: torch.Tensor, 
    k: int = 3, 
    n_restarts: int = N_RESTARTS, 
    tol: float = TOLERANCE, 
    max_iters: int = MAX_ITERATION,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    global RANDOM_SEED
    best_variation = float('inf')
    best_centroids = None
    best_labels = None

    print(f"Clustering with: k = {k}.")
    old_centroids = torch.zeros(k, X.shape[1], device = X.device, dtype = X.dtype)
    for _ in range(n_restarts):
        RANDOM_SEED += 1
        centroids = initiate_centroids(X, k, random_seed = RANDOM_SEED)
        centroids, Labels = optimize_centroids(
            X = X,
            k = k,
            centroids = centroids,
            max_iters = max_iters,
            tol = tol
        )

        var = calculate_variation(X, centroids, Labels)

        # keep best
        if var < best_variation:
            best_variation = var
            best_centroids = centroids.clone()
            best_labels = Labels.clone()
        if torch.allclose(old_centroids, centroids, atol = tol):
            best_variation = var
            best_centroids = centroids.clone()
            best_labels = Labels.clone()
            break
            
        old_centroids = centroids.clone()
        zeros = torch.zeros_like(centroids, device=X.device, dtype=X.dtype)
        if torch.equal(old_centroids, zeros):
            print("Error: old_centroids are zeros like without being modified by actual old centroids.")

    return X, best_labels, best_centroids, best_variation



# This visualizes a plot and return the variations for different k, by difault the k is in range of 1 to 10
# Note that this function calls all of the function above, and it's time complexity is highly depandent on MAX_ITERATION & N_RESTARTS & k
def k_means_assessment(
    X: torch.Tensor, 
    k: int = 10
    ) -> list:
    ks = list(range(1, k + 1))
    variations = []

    for k in ks:
        _, _, _, var = WCSS_for_single_k(X, k)
        variations.append(var)
    
    if not variations:
        print("No valid results to plot in k_means_accessment.")
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

# k_means_assessment()