import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNBaseline(nn.Module):
    def __init__(self, n_nodes, window_size, n_classes, n_hidden=64, n_layers=2, kernel_size=3, dropout=0.2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.pools = nn.ModuleList()

        self.convs.append(nn.Conv1d(n_nodes, n_hidden, kernel_size, padding=kernel_size//2))
        self.bns.append(nn.BatchNorm1d(n_hidden))
        self.pools.append(nn.MaxPool1d(2, 2))

        curr = n_hidden
        for _ in range(1, n_layers):
            nxt = min(curr * 2, 512)
            self.convs.append(nn.Conv1d(curr, nxt, kernel_size, padding=kernel_size//2))
            self.bns.append(nn.BatchNorm1d(nxt))
            self.pools.append(nn.MaxPool1d(2, 2))
            curr = nxt

        self._to_linear = self._compute_linear_size(n_nodes, window_size)

        self.fc1 = nn.Linear(self._to_linear, n_hidden * 2)
        self.fc2 = nn.Linear(n_hidden * 2, n_classes)
        self.dropout = nn.Dropout(dropout)

    def _compute_linear_size(self, n_nodes, window_size):
        with torch.no_grad():
            x = torch.zeros(1, n_nodes, window_size)
            for conv, bn, pool in zip(self.convs, self.bns, self.pools):
                x = F.relu(bn(conv(x)))
                x = pool(x)
            x = x.mean(dim=2)
            return x.size(1)

    def forward(self, x):
        for conv, bn, pool in zip(self.convs, self.bns, self.pools):
            x = F.relu(bn(conv(x)))
            x = pool(x)
        x = x.mean(dim=2)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)