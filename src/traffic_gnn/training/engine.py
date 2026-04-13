from __future__ import annotations

import numpy as np
import torch



def train_one_epoch(model, loader, optimizer, criterion, edge_index, edge_weight, device):
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        y_pred = model(X_batch, edge_index, edge_weight)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)

    return total_loss / len(loader.dataset)



def evaluate(model, loader, criterion, edge_index, edge_weight, device):
    model.eval()
    total_loss = 0.0
    preds = []
    trues = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            y_pred = model(X_batch, edge_index, edge_weight)
            loss = criterion(y_pred, y_batch)
            total_loss += loss.item() * X_batch.size(0)
            preds.append(y_pred.cpu())
            trues.append(y_batch.cpu())

    preds = torch.cat(preds, dim=0).numpy()
    trues = torch.cat(trues, dim=0).numpy()
    mae = float(np.mean(np.abs(preds - trues)))
    rmse = float(np.sqrt(np.mean((preds - trues) ** 2)))
    return total_loss / len(loader.dataset), mae, rmse, preds, trues
