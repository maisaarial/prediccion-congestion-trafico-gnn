from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv


class GAT_LSTM(nn.Module):
    def __init__(self, num_nodes: int, in_channels: int = 1, gat_hidden: int = 32, lstm_hidden: int = 64, heads: int = 1, **kwargs):
        super().__init__()
        if "hidden_channels" in kwargs:
            gat_hidden = kwargs["hidden_channels"]
        self.num_nodes = num_nodes
        self.gat1 = GATConv(in_channels=in_channels, out_channels=gat_hidden, heads=heads, concat=False, dropout=0.0)
        self.gat2 = GATConv(in_channels=gat_hidden, out_channels=gat_hidden, heads=heads, concat=False, dropout=0.0)
        self.lstm = nn.LSTM(input_size=gat_hidden, hidden_size=lstm_hidden, batch_first=True)
        self.fc = nn.Linear(lstm_hidden, 1)

    def forward(self, x_seq: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, window, N, _ = x_seq.shape
        outputs = []
        for t in range(window):
            x_t = x_seq[:, t, :, :]
            batch_embeddings = []
            for b in range(batch_size):
                # GATConv usa la topología; en esta versión básica no usamos edge_weight.
                h = torch.relu(self.gat1(x_t[b], edge_index))
                h = torch.relu(self.gat2(h, edge_index))
                batch_embeddings.append(h)
            outputs.append(torch.stack(batch_embeddings, dim=0))
        h_seq = torch.stack(outputs, dim=1).permute(0, 2, 1, 3).reshape(batch_size * N, window, -1)
        lstm_out, _ = self.lstm(h_seq)
        return self.fc(lstm_out[:, -1, :]).reshape(batch_size, N)
