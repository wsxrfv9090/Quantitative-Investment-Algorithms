import torch
import pandas as pd
import numpy as np
import global_resources as gr
import os

# GLOBAL VARIABLES
DTYPE = torch.float32
NUM_EPOCHS = 3000
LEARNING_RATE = 0.01
DELTA_LOSS_BREAKER = 1e-12

device = gr.set_device()
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

X = np.array(df.drop(['class'], axis = 1)).astype('float32')
y = np.array(df['class']).astype('float32')

y = np.where(y == 4, 1, np.where(y == 2, -1, y))

X_gpu = torch.tensor(X, dtype = DTYPE, device = device, requires_grad = True)
y_gpu = torch.tensor(y, dtype = DTYPE, device = device, requires_grad = True)


# n, d = X_example.shape

# def init_params(n_features, device):
#     w = torch.zeros(n_features, dtype=torch.float32, device=device, requires_grad=True)
#     b = torch.zeros(1, dtype=torch.float32, device=device, requires_grad=True)
#     return w, b

# w, b = init_params(d, device)


# optimizer = torch.optim.SGD([w, b], lr=LEARNING_RATE)



# Shuffle row wise
def shuffle_tensor_row_wise(ts = X_gpu):
    indices = torch.randperm(ts.shape[0])
    ts = ts[indices]

# shuffle_tensor_row_wise(X_gpu)

def create_random_weights_bias(shape = X_gpu.shape[-1] + 1, dtype = DTYPE):
    print(f"Creating random weights and bias with dtype: {dtype}")
    weights_bias = torch.rand(1, shape, dtype = dtype, device = device)
    weights_bias = torch.squeeze(weights_bias)
    weights_bias[-1] = 0
    return weights_bias

weights_bias = create_random_weights_bias()

def cal_distances(n_point = X_gpu, hyperplain_weights = weights_bias[:-1], hyperplain_bias = weights_bias[-1].unsqueeze(0), dtype = DTYPE):
    if n_point.dtype != dtype:
        print(f"Dtype mismatch in cal_distances function, n_point has {n_point.dtype}, while hyperplain_weights has {hyperplain_weights.dtype}, and hyperplain_bias has {hyperplain_bias.dtype}.")
    raw_scores = torch.matmul(n_point, hyperplain_weights) + hyperplain_bias
    # weight_norm = torch.norm(hyperplain_weights)
    # distances = raw_scores / weight_norm
    return raw_scores

distances = cal_distances(X_gpu, hyperplain_weights = weights_bias[:-1] + 1, hyperplain_bias = weights_bias[-1])

def hinge_loss(distances = distances, labels = y_gpu, margin=1.0):
    distances = distances.squeeze()
    losses = torch.clamp(margin - labels * distances, min = 0)
    return losses.mean()

def hinge_loss_l2_panalty(distances = distances, weights = weights_bias[:-1], labels = y_gpu, margin=1.0, l2_reg = 1e-4):
    distances = distances.squeeze()
    losses = torch.clamp(margin - labels * distances, min = 0)
    l2_penalty = 0.5 * l2_reg * torch.dot(weights, weights)
    return losses.mean() + l2_penalty

loss = hinge_loss(distances, y_gpu)

def update_model(X = X_gpu, y = y_gpu, weights = weights_bias[:-1], bias = weights_bias[-1], learning_rate = 0.01, dtype = DTYPE, l2_penalty = False):
    weights.requires_grad_(True)
    bias.requires_grad_(True)
    
    distances = cal_distances(X, weights, bias, dtype = dtype)
    if l2_penalty:
        loss = hinge_loss(distances, y)
    else:
        loss = hinge_loss_l2_panalty(distances, weights, y)
    loss.backward()
    
    with torch.no_grad():
        weights -= learning_rate * weights.grad
        bias -= learning_rate * bias.grad
    
    weights.grad.zero_()
    bias.grad.zero_()
    
    return loss.item(), weights, bias

    
def train(X = X_gpu, 
          y = y_gpu, 
          weights = weights_bias[:-1], 
          bias = weights_bias[-1], 
          num_epochs = NUM_EPOCHS, 
          start_learning_rate = LEARNING_RATE, 
          l2_penalty = False, 
          print_every = int(100), 
          delta_loss_breaker = DELTA_LOSS_BREAKER, 
          patience = 10):
    print(f"Training with loss function: {'hinge loss.' if not l2_penalty else 'hinge loss with l2 penalty on weights.'}")
    prev_loss = None
    lr = start_learning_rate
    streak = 0
    for epoch in range(num_epochs):
        loss_value, weights, bias = update_model(X, y, weights = weights, bias = bias, learning_rate = lr, dtype = DTYPE, l2_penalty = l2_penalty)
        delta_loss = prev_loss - loss_value if prev_loss is not None else None
        lr = adjust_lr(delta_loss, lr)
        prev_loss = loss_value
        
        small = None
        if delta_loss:
            small = abs(delta_loss) <= delta_loss_breaker and delta_loss != 0 and delta_loss != None
        if small:
            streak += 1
        else:
            streak = 0
        if epoch == 1 or epoch % print_every == 0 or epoch == num_epochs or streak >= patience or epoch == num_epochs - 1:
            print(f"Epoch {epoch} | Loss: {loss_value:} | Delta loss: {delta_loss}")  
        if streak >= patience:
            print(f"Exited with delta_Loss squared consecutively being smaller than {delta_loss_breaker} from epoch {epoch - 10} to epoch {epoch}.")
            break

    return weights, bias
    

def adjust_lr(delta_loss,
              lr,
              decay_factor: float = 0.5,
              growth_factor: float = 1.05,
              min_lr: float = 1e-6,
              max_lr: float = 1.0,
              thresh: float = 1e-4):
    if delta_loss == None:
        return lr
    
    if delta_loss >= thresh:
        # good improvement → bump lr
        lr = min(lr * growth_factor, max_lr)
    elif delta_loss <= -thresh:
        # loss got worse → cut lr
        lr = max(lr * decay_factor, min_lr)
    # else: tiny change → keep lr
    return lr


def score(weights: torch.Tensor, 
          bias: torch.Tensor,
          X_test: torch.Tensor,
          y_test: torch.Tensor,
          normed = False):
    if normed:
        with torch.no_grad():
            weights_normed = torch.norm(weights)
            weights = weights / weights_normed
            bias = bias / weights_normed
    raw_scores = torch.matmul(X_test, weights) + bias

    preds = torch.sign(raw_scores).to(dtype = y_test.dtype)       # +1 or -1

    correct = (preds == y_test).sum().item()
    total   = y_test.shape[0]
    
    accuracy = 100.0 * correct / total
    return accuracy


# train()