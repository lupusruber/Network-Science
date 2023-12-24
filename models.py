import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric_temporal import A3TGCN2, DCRNN, TGCN


class TemporalGNN(torch.nn.Module):
    def __init__(self, node_features, hidden_units, out_periods, batch_size) -> None:
        super(TemporalGNN, self).__init__()
        self.tgnn = A3TGCN2(
            in_channels=node_features,
            out_channels=hidden_units,
            periods=out_periods,
            batch_size=batch_size,
        )
        self.linear = torch.nn.Linear(hidden_units, out_periods)

    def forward(self, x, edge_index, edge_weight=None, h=None) -> Tensor:
        h = self.tgnn(x, edge_index, edge_weight, h)
        h: Tensor = F.relu(h)
        h = self.linear(h)
        return h


class TemporalGraphCN(torch.nn.Module):
    def __init__(self, node_features, hidden_units, out_periods) -> None:
        super(TemporalGraphCN, self).__init__()
        self.linear1 = torch.nn.Linear(node_features, hidden_units)
        self.tgnn = TGCN(
            in_channels=hidden_units,
            out_channels=hidden_units,
        )
        self.linear2 = torch.nn.Linear(hidden_units, out_periods)

    def forward(
        self,
        x,
        edge_index,
    ) -> Tensor:
        h: Tensor = self.linear1(x)
        h = self.tgnn(h, edge_index)
        h = F.relu(h)
        h = self.linear2(h)
        return h


class RecurrentGNN(torch.nn.Module):
    def __init__(self, node_features, out_periods, hidden_units, K) -> None:
        super(RecurrentGNN, self).__init__()

        # (B, C, W, H) = (B, 325, 2, 12)
        B = 1
        C = 325
        W = 2
        H = 12
        D = C * W * H
        self.flat = torch.nn.Flatten()
        self.linear_2 = torch.nn.Linear(W * H, hidden_units)
        self.tgnn = DCRNN(in_channels=hidden_units, out_channels=hidden_units, K=K)
        self.linear = torch.nn.Linear(hidden_units, out_periods)

    def forward(self, x, edge_index, edge_weight=None) -> Tensor:
        h = self.flat(x)
        h = self.linear_2(h)
        h = F.sigmoid(h)
        h = self.tgnn(h, edge_index, edge_weight)
        h: Tensor = F.sigmoid(h)
        h = self.linear(h)
        return h
