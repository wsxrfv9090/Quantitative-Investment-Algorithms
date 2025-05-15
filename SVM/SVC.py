import torch
import pandas as pd
import numpy as np
import global_resources as gr
import os


device = gr.set_device()
print(f"Current training device: {device.capitalize()}.")

# GLOBAL VARIABLES
DTYPE = torch.float64
NUM_EPOCHS = 3000
LEARNING_RATE = torch.tensor(0.01, device = device, dtype = DTYPE)
DELTA_LOSS_BREAKER = 1e-12
RELATIVE_BREAKER = 1e-7


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
    num_epochs: int = NUM_EPOCHS, 
    start_learning_rate: torch.Tensor = LEARNING_RATE, 
    l2_penalty: bool = False, 
    print_every: int = int(100), 
    delta_loss_breaker: float = DELTA_LOSS_BREAKER, 
    patience: int = 10,
    relative: bool = True,
    relative_breaker: float = RELATIVE_BREAKER
    ) -> tuple[torch.Tensor, torch.Tensor]:
    print(f"Training with loss function: {'hinge loss.' if not l2_penalty else 'hinge loss with l2 penalty on weights.'}")
    
    prev_loss = None
    lr = start_learning_rate
    streak = 0
    weights_and_bias.requires_grad_(True)
    
    if not relative:
        print("Training with delta loss breaker.")
    else:
        print("Training with relative breaker.")
    
    for epoch in range(num_epochs):
        loss_value, weights, bias = update_model(X, y, weights_and_bias, learning_rate = lr, l2_penalty = l2_penalty)
        if not relative:
            delta_loss = prev_loss - loss_value if prev_loss is not None else None
        else:
            relative_ratio = abs(prev_loss / loss_value) if prev_loss is not None else None
        
        
        # lr = adjust_lr(delta_loss, lr)
        if not relative:
            delta = prev_loss - loss_value if prev_loss is not None else None
            lr = adjust_lr(
                lr=lr,
                relative=False,
                delta_loss=delta,
                abs_thresh=delta_loss_breaker,
            )
        else:
            ratio = abs(prev_loss / loss_value) if prev_loss is not None else None
            # print(f"Prev learning rate: {lr.item()}")
            lr = adjust_lr(
                lr=lr,
                relative=True,
                relative_ratio=ratio,
                rel_thresh=relative_breaker,
            )
            # print(f"After adjust learning rate: {lr.item()}")
        
        
        small = None
        if not relative:
            if delta_loss:
                small = abs(delta_loss) <= delta_loss_breaker and delta_loss != 0
        else:
            if relative_ratio != 0 and relative_ratio:
                small = (relative_ratio - 1) <= relative_breaker
            elif relative_ratio == 0:
                raise ValueError("Relative ratio is zero somehow.")
            
        if small:
            streak += 1
        else:
            streak = 0
        
        if not relative:
            if epoch == 1 or epoch % print_every == 0 or streak >= patience or epoch == num_epochs - 1:
                print(f"Epoch {epoch} | Loss: {loss_value:} | Delta loss: {delta_loss}")  
            if streak >= patience:
                print(f"Exited with delta_Loss squared consecutively being smaller than {delta_loss_breaker} from epoch {epoch - patience} to epoch {epoch}.")
                break
            if epoch == num_epochs - 1:
                print("Max epoch reached. ")
        else:
            if epoch == 1 or epoch % print_every == 0 or streak >= patience or epoch == num_epochs - 1:
                print(f"Epoch {epoch} | Loss: {loss_value:} | Relative Ratio: {relative_ratio}")
            if streak >= patience:
                print(f"Exited with relative_ratio consecutively being smaller than {relative_breaker} from epoch {epoch - patience} to epoch {epoch}.")
                break
            if epoch == num_epochs - 1:
                print("Max epoch reached. ")
        
        prev_loss = loss_value
        
    return weights, bias.unsqueeze(0)
    
def get_norm_weights_bias(
    weights: torch.Tensor,
    bias: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
    return weights / weights.norm(), bias / weights.norm()


# def adjust_lr(
#     delta_loss,
#     lr: torch.Tensor,
#     decay_factor: float = 0.5,
#     growth_factor: float = 1.05,
#     min_lr: float = 1e-6,
#     max_lr: float = 1.0,
#     thresh: float = 1e-4,
#     ) -> float:
#     if delta_loss == None:
#         # print(f"Start learning rate: {lr}")
#         return lr
    
#     if delta_loss >= thresh:
#         # good improvement → bump lr
#         lr = lr * growth_factor
#         lr = torch.clamp(lr, max = max_lr)
        
#     elif delta_loss <= -thresh:
#         # loss got worse → cut lr
#         lr = lr * decay_factor
#         lr = torch.clamp(lr, min = min_lr)
#     # else: tiny change → keep lr
#     return lr


def adjust_lr(
    *,
    lr: torch.Tensor,
    relative: bool = False,
    delta_loss: float | None = None,
    relative_ratio: float | None = None,
    decay_factor: float = 0.5,
    growth_factor: float = 1.05,
    min_lr: float = 1e-6,
    max_lr: float = 5.0,
    abs_thresh: float = 1e-7,
    rel_thresh: float = 0.01,
) -> torch.Tensor:
    """
    Adjust learning rate based on either absolute change in loss (delta_loss)
    or relative change (relative_ratio).

    Parameters
    ----------
    lr : torch.Tensor
        Current learning rate.
    relative : bool
        If True, use relative_ratio mode; otherwise use delta_loss mode.
    delta_loss : float or None
        prev_loss - curr_loss (only if relative=False).
    relative_ratio : float or None
        abs(prev_loss / curr_loss) (only if relative=True).
    decay_factor : float
        Factor to multiply lr when performance worsens.
    growth_factor : float
        Factor to multiply lr when performance improves.
    min_lr : float
        Floor for lr after decay.
    max_lr : float
        Ceiling for lr after growth.
    abs_thresh : float
        Minimum |delta_loss| to count as “significant” improvement or decline.
    rel_thresh : float
        Minimum deviation from 1.0 to count as “significant” in relative mode.

    Returns
    -------
    torch.Tensor
        The updated (and clamped) learning rate.
    """
    # nothing to compare yet
    if relative:
        if relative_ratio is None:
            return lr
        # relative_ratio > 1+rel_thresh → improvement
        if relative_ratio >= 1.0 + rel_thresh:
            lr = lr * growth_factor
        # relative_ratio < 1-rel_thresh → got worse
        elif relative_ratio <= 1.0 - rel_thresh:
            lr = lr * decay_factor
        # Else the relative ratio is not "significant" so it doesn't change
    else:
        if delta_loss is None:
            return lr
        # delta_loss = prev_loss - curr_loss
        if delta_loss >= abs_thresh:
            lr = lr * growth_factor
        elif delta_loss <= -abs_thresh:
            lr = lr * decay_factor
        # Else the relative ratio is not "significant" so it doesn't change

    # clamp to [min_lr, max_lr]
    return torch.clamp(lr, min = min_lr, max = max_lr)



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
    

def ovo_train(
    X: torch.Tensor, 
    y: torch.Tensor, 
    num_epochs: int = NUM_EPOCHS, 
    start_learning_rate: torch.Tensor = LEARNING_RATE, 
    l2_penalty: bool = False, 
    print_every: int = int(100), 
    delta_loss_breaker: float = DELTA_LOSS_BREAKER, 
    patience: int = 10,
    relative: bool = False,
    relative_breaker: float = RELATIVE_BREAKER
    ) -> dict:
    uniq_labels = y.unique().tolist()
    if len(uniq_labels) == 2:
        raise BufferError(f"Unique labels in y: {y} has only two value, istead of using ovo_train, please use train instead.")
    D = X.size(1)  # Change this to the actual size of your row vector.
    # weights_tensor = torch.empty((0, D), dtype = X.dtype, device = X.device)
    # bias_tensor = torch.empty((0), dtype = X.dtype, device = X.device)
    dic = {}
    for idx, label_a in enumerate(uniq_labels):
        for label_b in uniq_labels[idx + 1:]:
            print("-------------------------------------------------------------------------------------------------------")
            print(f"Training on label a: {label_a} and label b: {label_b}")
            first_members = X[y == label_a]
            second_members = X[y == label_b]
            X_temp = torch.cat([first_members, second_members], dim = 0)
            y_temp = torch.cat([torch.ones(first_members.size(0), device = X.device, dtype = X.dtype), -torch.ones(second_members.size(0), device = X.device, dtype = X.dtype)], dim = 0)
            weights, bias = train(
                X_temp, 
                y_temp, 
                weights_and_bias = create_random_weights_bias(first_members),
                num_epochs = num_epochs,
                start_learning_rate = start_learning_rate,
                l2_penalty = l2_penalty,
                print_every = print_every,
                delta_loss_breaker = delta_loss_breaker, 
                patience = patience,
                relative = relative,
                relative_breaker = relative_breaker
                )
            key = (label_a, label_b)
            weights_bias = torch.cat([weights, bias], dim = 0)
            # weights_tensor = torch.cat([weights_tensor, weights.squeeze().unsqueeze(0)], dim = 0)
            # bias_tensor = torch.cat([bias_tensor, bias.unsqueeze(0)], dim = 0)
            # dic[key] = (weights_tensor, bias_tensor)
            dic[key] = weights_bias
    return dic

def ovo_predict(
    dic: dict,
    X: torch.Tensor,
    dtype: torch.dtype = None
) -> torch.Tensor:
    if dtype == None:
        dtype = X.dtype
    # X.size(0)
    votes = torch.empty((X.size(0), 0), dtype = dtype, device = X.device)
    for (label_i, label_j), weights_bias in dic.items():
        if weights_bias.device != X.device:
            raise MemoryError('Weights and bias tensor and X tensor not on same device!!!!')
        distances = cal_distances(X = X, weights_and_bias = weights_bias)
        preds = torch.sign(distances).to(dtype = dtype)
        votes_temp = torch.where(preds == 1, label_i, label_j).unsqueeze(0)
        votes = torch.cat([votes, votes_temp.T], dim = 1)
    final_predictions, _ = torch.mode(votes, dim = 1)
    return final_predictions


def ovo_score(
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    dic: dict
) -> float:
    preds = ovo_predict(dic, X_test, y_test.dtype)
    correct = (preds == y_test).sum().item()
    total   = y_test.numel()
    acc     = correct / total
    print(f"Accuracy: {correct}/{total} = {acc*100:.2f}%")
    return acc
    


# def ovo_test(dic: dict,
#              X_test: torch.Tensor,
#              y_test: torch.Tensor
#             ) -> float:
#     """
#     Runs OvO prediction on X_test, compares to y_test, and prints accuracy.
#     Returns:
#       accuracy (float between 0 and 1)
#     """
#     preds = ovo_predict(dic, X_test)
#     correct = (preds == y_test).sum().item()
#     total   = y_test.numel()
#     acc     = correct / total
#     print(f"Accuracy: {correct}/{total} = {acc*100:.2f}%")
#     return acc
