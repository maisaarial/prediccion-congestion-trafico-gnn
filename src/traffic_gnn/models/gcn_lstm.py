from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class GCN_LSTM(nn.Module):
    def __init__(self, num_nodes: int, in_channels: int = 1, gcn_hidden: int = 32, lstm_hidden: int = 64):
        super().__init__()
        self.num_nodes = num_nodes
        self.gcn1 = GCNConv(in_channels, gcn_hidden)
        self.gcn2 = GCNConv(gcn_hidden, gcn_hidden)
        self.lstm = nn.LSTM(input_size=gcn_hidden, hidden_size=lstm_hidden, batch_first=True)
        self.fc = nn.Linear(lstm_hidden, 1)

    def forward(self, x_seq: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, window, N, _ = x_seq.shape
        gcn_outputs = []

        for t in range(window):
            x_t = x_seq[:, t, :, :]
            batch_embeddings = []
            for b in range(batch_size):
                x_b = x_t[b]
                h = torch.relu(self.gcn1(x_b, edge_index, edge_weight))
                h = torch.relu(self.gcn2(h, edge_index, edge_weight))
                batch_embeddings.append(h)
            batch_embeddings = torch.stack(batch_embeddings, dim=0)
            gcn_outputs.append(batch_embeddings)

        h_seq = torch.stack(gcn_outputs, dim=1)
        h_seq = h_seq.permute(0, 2, 1, 3)
        h_seq = h_seq.reshape(batch_size * N, window, -1)
        lstm_out, _ = self.lstm(h_seq)
        last_out = lstm_out[:, -1, :]
        out = self.fc(last_out).reshape(batch_size, N)
        return out
