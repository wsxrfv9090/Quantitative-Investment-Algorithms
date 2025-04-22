# svc_module.py
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

def load_data(path):
    df = pd.read_csv(path, header=None, na_values='?')
    df = df.apply(pd.to_numeric, errors='coerce').dropna()
    df.drop(columns = 0, inplace=True)   # drop ID col
    X = df.iloc[:, :-1].values.astype(np.float32)
    y_raw = df.iloc[:, -1].values
    y = np.where(y_raw == 4, 1.0, np.where(y_raw == 2, -1.0, np.nan))
    if np.isnan(y).any():
        raise ValueError("labels must be 2 or 4")
    return X, y.astype(np.float32)

def svm_objective(w, b, Xb, yb, C):
    # primal objective: ½||w||² + C * mean(hinge)
    raw = Xb @ w + b
    hinge = torch.clamp(1 - yb * raw, min=0).mean()
    return 0.5 * (w @ w) + C * hinge

def train_svm(X, y,
              epochs=100,
              learning_rate=1e-3,
              C=1.0,
              batch_size=64,
              device=None,
              print_every=10):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    X = torch.tensor(X, device=device)
    y = torch.tensor(y, device=device)

    # init
    N, D = X.shape
    w = torch.zeros(D, requires_grad=True, device=device)
    b = torch.zeros(1, requires_grad=True, device=device)
    optimizer = torch.optim.SGD([w, b], lr=learning_rate)

    # DataLoader gives you *all* batches every epoch
    ds     = TensorDataset(X, y)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for Xb, yb in loader:
            optimizer.zero_grad()
            loss = svm_objective(w, b, Xb, yb, C)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * Xb.size(0)

        epoch_loss /= N
        if epoch == 1 or epoch % print_every == 0:
            print(f"[Epoch {epoch:03d}/{epochs:03d}]  Loss: {epoch_loss:.4f}  ||w||={w.norm().item():.4f}")

    return w.detach(), b.detach()


if __name__ == "__main__":
    data_path = os.path.join("Data", "breast-cancer-wisconsin.data")
    X, y = load_data(data_path)

    # feature‐scale!
    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)

    w, b = train_svm(
        X, y,
        epochs=10000,
        learning_rate=1e-3,
        C=1.0,
        batch_size=64,    # now you’ll see ALL 699 samples per epoch
        print_every=20
    )
