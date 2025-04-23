import global_resources as gr
import K_means_cluster_module.k_means_cluster as kmc
import SVM.SVC as svc
import CAPM_module.get_regression_line as capm
import torch

main_dir = 'D:\ImportanFiles\Coding Related\Repositories\Quantitative-Investment-Algorithms'
gr.ch_dir_to_repo(main_dir)


# Import data

# K means cluster

    # Elbow accessing:
# kmc.k_means_assessment()

# Choose K according to the plot
k = 2
X, y, centroids, var = kmc.WCSS_for_single_k(k = k)
print(f"K clustering completed, with variance of {var}, k of {k}.")


# SVM/OVO
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Running on {device}")
X_tensor = torch.tensor(X, dtype = torch.float32, device = device)
y_tensor = torch.tensor(y, dtype = torch.float32, device = device)
svc.test(X=X_tensor, y=y_tensor, num_epochs=10000, learning_rate=0.0001)


# CAPM
