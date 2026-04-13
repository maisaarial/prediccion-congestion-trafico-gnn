from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import TensorDataset



def crear_ventanas(data: np.ndarray, window: int = 12, horizon: int = 1) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    T = data.shape[0]

    for t in range(T - window - horizon + 1):
        x_t = data[t : t + window]
        y_t = data[t + window + horizon - 1]
        X.append(x_t[..., np.newaxis])
        y.append(y_t)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)



def split_temporal(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return X[:train_end], y[:train_end], X[train_end:val_end], y[train_end:val_end], X[val_end:], y[val_end:]



def to_torch_tensors(*arrays: np.ndarray) -> list[torch.Tensor]:
    return [torch.tensor(arr, dtype=torch.float32) for arr in arrays]



def build_tensor_datasets(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[TensorDataset, TensorDataset, TensorDataset]:
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
    return train_dataset, val_dataset, test_dataset
