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

# Shuffle row wise
def shuffle_tensor_row_wise(
    X: torch.Tensor
    ) -> torch.Tensor:
    idx = torch.randperm(X.shape[0], device=X.device)
    return X[idx]


# Creates weights and bias based on the shape of row vectors within 
def create_random_weights_bias(
    X: torch.Tensor, 
    ) -> torch.Tensor:
    print(f"Creating random weights and bias with dtype: {X.dtype}")
    # 0.0 <= weights_bias[i] < 1.0
    weights_bias = torch.rand(1, X.shape[-1] + 1, dtype = X.dtype, device = device)
    weights_bias = torch.squeeze(weights_bias)
    weights_bias[-1] = 0
    return weights_bias

def cal_distances(
    X: torch.Tensor, 
    weights_and_bias: torch.Tensor,
    ) -> torch.Tensor:
    weights = weights_and_bias[:-1]
    bias = weights_and_bias[-1].unsqueeze(0)
    raw_scores = torch.matmul(X, weights) + bias
    
    # Without normalization
    return raw_scores

def hinge_loss(
    distances: torch.Tensor, 
    labels: torch.Tensor, 
    margin: float = 1.0
    ) -> torch.Tensor:
    if distances.dtype != labels.dtype:
        raise ValueError("Mismatching dtype in hinge_loss function! Maybe you didn't initiate y and X as the same data type???")
    distances = distances.squeeze()
    losses = torch.clamp(margin - labels * distances, min = 0)
    return losses.mean().unsqueeze(0)

def hinge_loss_l2_panalty(
    distances: torch.Tensor, 
    weights_and_bias: torch.Tensor,
    labels: torch.Tensor, 
    margin: float = 1.0, 
    l2_reg: float = 1e-4
    ) -> torch.Tensor:
    weights = weights_and_bias.squeeze(0)[:-1]
    distances = distances.squeeze()
    losses = torch.clamp(margin - labels * distances, min = 0)
    l2_penalty = 0.5 * l2_reg * torch.dot(weights, weights)
    return (losses.mean() + l2_penalty).unsqueeze(0)

def update_model(
    X: torch.Tensor, 
    y: torch.Tensor, 
    weights_and_bias: torch.Tensor,
    learning_rate: torch.Tensor, 
    l2_penalty: bool = False,
    ) -> tuple[float, torch.Tensor, torch.Tensor]:
    
    # Making sure weights and bias is gradient tracked
    if not weights_and_bias.requires_grad:
        raise AttributeError("Weights and bias is not gradient-tracked when updating model!!!!")
    
    # Making sure all parameters is on the same device:
    if X.device != y.device or X.device != weights_and_bias.device or X.device != learning_rate.device:
        raise MemoryError("Four parameters passed into update_model function is not on same device!!! Check if learning rate is set correctly using Torch and on X.device.")
    
    # Makeing sure it's one dimensional
    weights_and_bias = weights_and_bias.squeeze(0)
    # Retain original gradient before squeeze operation
    weights_and_bias.retain_grad()
    
    weights = weights_and_bias[:-1]   # this is a view
    bias = weights_and_bias[-1]
    
    # Forward pass
    distances = cal_distances(X, weights_and_bias)
    if l2_penalty:
        loss = hinge_loss_l2_panalty(distances, weights_and_bias, y)
    else:
        loss = hinge_loss(distances, y)
        
    
    # Backword    
    loss.backward()
    
    # Step
    with torch.no_grad():
        weights_and_bias -= learning_rate * weights_and_bias.grad
        weights_and_bias.grad.zero_()
    
    return loss.item(), weights, bias

    
def train(
    X: torch.Tensor, 
    y: torch.Tensor, 
    weights_and_bias: torch.Tensor, 
    num_epochs = NUM_EPOCHS, 
    start_learning_rate = LEARNING_RATE, 
    l2_penalty = False, 
    print_every = int(100), 
    delta_loss_breaker = DELTA_LOSS_BREAKER, 
    patience = 10,
    ) -> tuple[torch.Tensor, torch.Tensor]:
    print(f"Training with loss function: {'hinge loss.' if not l2_penalty else 'hinge loss with l2 penalty on weights.'}")
    
    prev_loss = None
    lr = start_learning_rate
    streak = 0
    weights_and_bias.requires_grad_(True)
    
    for epoch in range(num_epochs):
        loss_value, weights, bias = update_model(X, y, weights_and_bias, learning_rate = lr, l2_penalty = l2_penalty)
        delta_loss = prev_loss - loss_value if prev_loss is not None else None
        lr = adjust_lr(delta_loss, lr)
        
        small = None
        if delta_loss:
            small = abs(delta_loss) <= delta_loss_breaker and delta_loss != 0 and delta_loss != None
        if small:
            print(f"Delta Loss smaller than threshold: Epoch {epoch} | Loss: {loss_value:} | Delta loss: {delta_loss}")  
            streak += 1
        else:
            streak = 0
            
        if epoch == 1 or epoch % print_every == 0 or streak >= patience or epoch == num_epochs - 1:
            print(f"Epoch {epoch} | Loss: {loss_value:} | Delta loss: {delta_loss}")  
        if streak >= patience:
            print(f"Exited with delta_Loss squared consecutively being smaller than {delta_loss_breaker} from epoch {epoch - patience} to epoch {epoch}.")
            break
        if epoch == NUM_EPOCHS - 1:
            print("Max epoch reached. ")
        
        prev_loss = loss_value
        
    return weights, bias.unsqueeze(0)
    
def get_norm_weights_bias(
    weights: torch.Tensor,
    bias: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
    return weights / weights.norm(), bias / weights.norm()





def adjust_lr(
    delta_loss,
    lr,
    decay_factor: float = 0.5,
    growth_factor: float = 1.05,
    min_lr: float = 1e-6,
    max_lr: float = 1.0,
    thresh: float = 1e-4
    ) -> float:
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


def score(
    weights: torch.Tensor, 
    bias: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    ) -> float:
    
    raw_scores = torch.matmul(X_test, weights) + bias

    preds = torch.sign(raw_scores).to(dtype = y_test.dtype)       # +1 or -1

    correct = (preds == y_test).sum().item()
    total   = y_test.shape[0]
    
    accuracy = 100.0 * correct / total
    return accuracy


def extract_svc(
    weights: torch.Tensor,
    bias: torch.Tensor,
    norm: bool = True,
) -> torch.Tensor:
    if norm:
        weights_bias = torch.cat((weights.squeeze(), bias.squeeze().unsqueeze(0)), dim = 0)
        svc = weights_bias / weights.norm()
        return svc
    else:
        unnormed = torch.cat((weights.squeeze(), bias.squeeze().unsqueeze(0)), dim = 0)
        return unnormed
    
    
def ovo():
    for i in range(10):
        for j in range(10):
            print("do something.")