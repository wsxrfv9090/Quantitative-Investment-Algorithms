import global_resources as gr
import K_means_cluster_module.k_means_cluster as kmc
import SVM.SVC as svc
import CAPM_module.get_regression_line as capm
import torch

main_dir = 'D:\Important Files\Repositories\Quantitative-Investment-Algorithms'
gr.ch_dir_to_repo(main_dir)


# Import data

# K means cluster

    # Elbow accessing:
# kmc.k_means_assessment()

# Choose K according to the plot
k = 2
X, y, centroids, var = kmc.WCSS_for_single_k(k = k)
print(f"K clustering completed, with variance of {var}, k of {k}.")
print(f"Type of X: {X.type}")
print(f"Shpe of X: {X.shape}")
print(f"Type of y: {y.type}")
print(f"Shape of y: {y.shape}")

# SVM/OVO
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Running on {device}")
X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
y_tensor = torch.tensor(y, dtype=torch.float32, device=device)
svc.test(X=X_tensor, y=y_tensor, num_epochs=400, learning_rate=0.0001)

# weights, bias = svc.train_svm(X_tensor, y_tensor,
#                               epochs=400,
#                               learning_rate=1e-3,
#                               C=1.0,
#                               batch_size=64,
#                               device=device)
# weights, bias = svc.test(y = y, num_epochs = 400, learning_rate = 0.01)
# CAPM
