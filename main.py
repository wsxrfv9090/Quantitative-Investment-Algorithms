import global_resources as gr
import K_means_cluster_module.k_means_cluster as kmc
import SVM.SVC as svc
import torch
import CAPMlib.CAPM as capm

main_dir = 'D:\ImportanFiles\Coding Related\Repositories\Quantitative-Investment-Algorithms'
gr.ch_dir_to_repo(main_dir)

def nl():
    print("\n")

# Import data

# K means cluster

    # Elbow accessing:
# kmc.k_means_assessment()


# Choose K according to the plot
k = 2
X, y, centroids, var = kmc.WCSS_for_single_k(k = k)
print(f"K clustering completed, with variance of {var}, k of {k}.")
y = torch.where(y == 0, torch.tensor(-1, dtype = y.dtype, device = y.device, requires_grad = y.requires_grad), y)

# SVM/OVO
device = gr.set_device()
print(f"Running on {device}.")


weights_bias = svc.create_random_weights_bias(X.shape[-1] + 1, dtype = X.dtype)
svc_trained_weights, svc_trained_bias = svc.train(X, y, weights = weights_bias[:-1], bias = weights_bias[-1], l2_penalty = False, num_epochs = 10000, start_learning_rate = 0.01)

accuracy = svc.score(svc_trained_weights, svc_trained_bias, X, y)
print(f"Testing Completed with the accuracy of {accuracy}. ")

test = capm.CAPM()