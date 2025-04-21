import torch
import pandas as pd
import numpy as np
import global_resources as gr
import os

# GLOBAL VARIABLES
LEARNING_RATE = 0.01


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Current training device: {device.capitalize()}.")

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

X = np.array(df.drop(['class'], axis = 1)).astype('float64')
y = np.array(df['class']).astype('float64')

y = np.where(y == 4, 1, np.where(y == 2, -1, y))

X_example = torch.tensor(X, dtype = torch.float32, device = device, requires_grad = True)
y_example = torch.tensor(y, dtype = torch.float32, device = device, requires_grad = True)


n, d = X_example.shape

def init_params(n_features, device):
    w = torch.zeros(n_features, dtype=torch.float32, device=device, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float32, device=device, requires_grad=True)
    return w, b

w, b = init_params(d, device)


optimizer = torch.optim.SGD([w, b], lr=LEARNING_RATE)









# Shuffle row wise
def shuffle_tensor_row_wise(ts = X_gpu):
    indices = torch.randperm(ts.shape[0])
    ts = ts[indices]

shuffle_tensor_row_wise(X_gpu)

def create_random_weights_bias(shape = X_gpu.shape[-1] + 1):
    weights_bias = torch.rand(1, shape, dtype = torch.float64, device = device)
    weights_bias = torch.squeeze(weights_bias)
    weights_bias[-1] = 0
    return weights_bias

weights_bias = create_random_weights_bias()


def cal_signed_distance(n_point, hyperplain_weights = weights_bias[:-1], hyperplain_bias = weights_bias[-1].item()):
    raw_scores = torch.matmul(n_point, hyperplain_weights) + hyperplain_bias
    weight_norm = torch.norm(hyperplain_weights)
    distances = raw_scores / weight_norm
    return distances

distances = cal_signed_distance(X_gpu, hyperplain_weights = weights_bias[:-1] + 1, hyperplain_bias = weights_bias[-1])

def hinge_loss(distances = distances, labels = y_gpu, margin=1.0):
    distances = distances.squeeze()
    
    losses = torch.clamp(margin - labels * distances, min = 0)
    
    return losses.mean()

loss = hinge_loss(distances, y_gpu)

def update_model(X = X_gpu, y = y_gpu, weights = weights_bias[:-1], bias = weights_bias[-1], learning_rate = 0.01):
    
    weights.requires_grad_(True)
    bias.requires_grad_(True)
    
    distances = cal_signed_distance(X, weights, bias)
    
    loss = hinge_loss(distances, y)
    
    loss.backward()
    
    with torch.no_grad():
        weights -= learning_rate * weights.grad
        bias -= learning_rate * bias.grad
    
    weights.grad.zero_()
    bias.grad.zero_()
    
    return loss.item(), weights, bias

num_epochs = 100
learning_rate = 0.01

for epoch in range(num_epochs):
    loss_value, weights, bias = update_model(X_gpu, y_gpu)
    print(f"Epoch {epoch + 1:03d} | Loss: {loss_value:.4f}")