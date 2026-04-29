from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class GCN_GRU(nn.Module):
    def __init__(self, num_nodes: int, in_channels: int = 1, gcn_hidden: int = 32, lstm_hidden: int = 64, **kwargs):
        super().__init__()
        if "hidden_channels" in kwargs:
            gcn_hidden = kwargs["hidden_channels"]
        if "gru_hidden" in kwargs:
            lstm_hidden = kwargs["gru_hidden"]
        self.num_nodes = num_nodes
        self.gcn1 = GCNConv(in_channels, gcn_hidden)
        self.gcn2 = GCNConv(gcn_hidden, gcn_hidden)
        self.gru = nn.GRU(input_size=gcn_hidden, hidden_size=lstm_hidden, batch_first=True)
        self.fc = nn.Linear(lstm_hidden, 1)

    def forward(self, x_seq: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, window, N, _ = x_seq.shape
        outputs = []
        for t in range(window):
            x_t = x_seq[:, t, :, :]
            batch_embeddings = []
            for b in range(batch_size):
                h = torch.relu(self.gcn1(x_t[b], edge_index, edge_weight))
                h = torch.relu(self.gcn2(h, edge_index, edge_weight))
                batch_embeddings.append(h)
            outputs.append(torch.stack(batch_embeddings, dim=0))
        h_seq = torch.stack(outputs, dim=1).permute(0, 2, 1, 3).reshape(batch_size * N, window, -1)
        gru_out, _ = self.gru(h_seq)
        return self.fc(gru_out[:, -1, :]).reshape(batch_size, N)
