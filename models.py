import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric_temporal import A3TGCN2, DCRNN, TGCN
from typing import Any, Dict
from copy import deepcopy

from data import (
    create_test_data_loader,
    create_train_data_loader,
    test_data_set,
    train_data_set,
    static_edge_index,
)
from pathlib import Path

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"


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


def train_and_eval_DCRNN(number_of_epochs: int, BATCH_SIZE: int) -> Dict[str, Any]:
    """_summary_
    :param number_of_epochs: The number of epoch for the model training
    :type number_of_epochs: int
    :param BATCH_SIZE: How big is the batch size
    :type BATCH_SIZE: int
    """
    train_loader = create_train_data_loader(train_data_set, BATCH_SIZE)
    test_loader = create_test_data_loader(test_data_set, BATCH_SIZE)

    model = RecurrentGNN(
        node_features=12,
        out_periods=12,
        hidden_units=32,
        K=3,
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()

    best_error = float("inf")
    model_dict: Dict[str, Any] = deepcopy(model.state_dict())

    for epoch in range(number_of_epochs):
        print(f"Epoch number: {epoch+1}")

        model.train()
        time = 0
        train_loss_list = []
        for X, y in train_loader:
            X = X.squeeze()
            y = y.squeeze()[:, 0]  # da se popravi

            y_hat_train = model(X, static_edge_index)
            train_cost = loss_fn(target=y, input=y_hat_train)

            optimizer.zero_grad()
            train_cost.backward()
            optimizer.step()
            train_loss_list.append(train_cost.item())

            train_mse: float = sum(train_loss_list) / len(train_loss_list)

            if time % 1000 == 0:
                print(f"Train MSE in step {time}: {train_mse}")

            time += 1

        model.eval()
        time = 0
        with torch.inference_mode():
            test_loss_list = []
            for X, y in test_loader:
                X = X.squeeze()
                y = y.squeeze()[:, 0]
                y_hat_test = model(X, static_edge_index)
                test_cost = loss_fn(target=y, input=y_hat_test)
                test_loss_list.append(test_cost.item())

                if time % 1000 == 0:
                    print(
                        f"Test MSE in step {time}: {sum(test_loss_list) / len(test_loss_list)}"
                    )

                time += 1

            test_mse: float = sum(test_loss_list) / len(test_loss_list)

            if test_mse < best_error:
                best_error = test_mse
                model_dict = deepcopy(model.state_dict())

    return model_dict


def train_and_eval_TGCN(number_of_epochs: int, BATCH_SIZE: int) -> Dict[str, Any]:
    train_loader = create_train_data_loader(train_data_set, BATCH_SIZE)
    test_loader = create_test_data_loader(test_data_set, BATCH_SIZE)

    model = TemporalGraphCN(
        node_features=12,
        hidden_units=128,
        out_periods=12,
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()

    best_error = float("inf")
    model_dict: Dict[str, Any] = deepcopy(model.state_dict())

    for epoch in range(number_of_epochs):
        print(f"Epoch number: {epoch+1}")

        model.train()
        time = 0
        train_loss_list = []
        for X, y in train_loader:
            y = y[:, :, 0].squeeze()
            X = X[:, :, 0].squeeze()

            train_y_hat = model(X, static_edge_index)
            train_loss = loss_fn(target=y, input=train_y_hat)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            train_loss_list.append(train_loss.item())

            if time % 100 == 0:
                print(
                    f"MSE Train in step {time}: {sum(train_loss_list) / len(train_loss_list)}"
                )

            time += 1

        model.eval()
        with torch.inference_mode():
            time = 0
            test_total_loss = []

            for X, y in test_loader:
                y = y[:, :, 0].squeeze()
                X = X[:, :, 0].squeeze()

                test_y_hat = model(X, static_edge_index)
                test_loss = loss_fn(target=y, input=test_y_hat)
                test_total_loss.append(test_loss.item())

                if time % 100 == 0:
                    print(
                        f"MSE Test in step {time}: {sum(test_total_loss) / len(test_total_loss)}"
                    )

                time += 1

            test_mse: float = sum(test_total_loss) / len(test_total_loss)
            if test_mse < best_error:
                best_error = test_mse
                model_dict = deepcopy(model.state_dict())

    return model_dict


def train_and_eval_TGNN(number_of_epochs: int, BATCH_SIZE: int) -> Dict[str, Any]:
    train_loader = create_train_data_loader(train_data_set, BATCH_SIZE)
    test_loader = create_test_data_loader(test_data_set, BATCH_SIZE)

    model = TemporalGNN(
        node_features=2,
        out_periods=12,
        hidden_units=32,
        batch_size=BATCH_SIZE,
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()

    best_error = float("inf")
    model_dict: Dict[str, Any] = deepcopy(model.state_dict())

    for epoch in range(number_of_epochs):
        print(f"Epoch number: {epoch+1}")

        model.train()
        time = 0
        train_loss_list = []
        for X, y in train_loader:
            y_speed = y[:, :, 0]
            # y_con = y[:, :, 1]

            train_y_hat = model(X, static_edge_index)
            train_loss = loss_fn(target=y_speed, input=train_y_hat)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            train_loss_list.append(train_loss.item())

            if time % 100 == 0:
                print(
                    f"MSE Train in step {time}: {sum(train_loss_list) / len(train_loss_list)}"
                )
            time += 1

        model.eval()
        with torch.inference_mode():
            time = 0
            test_total_loss = []

            for X, y in test_loader:
                y_speed = y[:, :, 0]

                test_y_hat = model(X, static_edge_index)
                test_loss = loss_fn(target=y_speed, input=test_y_hat)
                test_total_loss.append(test_loss.item())

                if time % 100 == 0:
                    print(
                        f"MSE Test in step {time}: {sum(test_total_loss) / len(test_total_loss)}"
                    )

                time += 1

            test_mse: float = sum(test_total_loss) / len(test_total_loss)

            if test_mse < best_error:
                best_error = test_mse
                model_dict = deepcopy(model.state_dict())

    return model_dict


def get_models() -> tuple[TemporalGNN, RecurrentGNN, TemporalGraphCN]:
    model_TGNN = TemporalGNN(
        node_features=2,
        out_periods=12,
        hidden_units=32,
        batch_size=32,
    ).to(DEVICE)

    model_DCRNN = RecurrentGNN(
        node_features=12,
        out_periods=12,
        hidden_units=32,
        K=3,
    ).to(DEVICE)

    model_TGCN = TemporalGraphCN(
        node_features=12,
        hidden_units=128,
        out_periods=12,
    ).to(DEVICE)

    return model_TGNN, model_DCRNN, model_TGCN


def get_paths():
    path_TGNN = Path(r"saved_models/model_TGNN_state_dict.pth")
    path_DCRNN = Path(r"saved_models/model_DCRNN_state_dict.pth")
    path_TGCN = Path(r"saved_models/model_TGCN_state_dict.pth")
    return path_TGNN, path_DCRNN, path_TGCN


def get_trained_models():
    path_TGNN, path_DCRNN, path_TGCN = get_paths()

    model_TGNN, model_DCRNN, model_TGCN = get_models()

    if path_TGNN.is_file():
        loaded_state_dict = torch.load(path_TGNN, map_location=DEVICE)
        model_TGNN.load_state_dict(loaded_state_dict)
        print("TGNN dict found !")
    else:
        state_dict = train_and_eval_TGNN(number_of_epochs=2, BATCH_SIZE=32)
        torch.save(state_dict, path_TGNN)

    if path_DCRNN.is_file():
        loaded_state_dict = torch.load(path_DCRNN, map_location=DEVICE)
        model_DCRNN.load_state_dict(loaded_state_dict)
        print("DCRNN dict found !")
    else:
        state_dict = train_and_eval_DCRNN(number_of_epochs=2, BATCH_SIZE=1)
        torch.save(state_dict, path_DCRNN)

    if path_TGCN.is_file():
        loaded_state_dict = torch.load(path_TGCN, map_location=DEVICE)
        model_TGCN.load_state_dict(loaded_state_dict)
        print("TGCN dict found !")
    else:
        state_dict = train_and_eval_TGCN(number_of_epochs=2, BATCH_SIZE=1)
        torch.save(state_dict, path_TGCN)

    return model_TGNN, model_DCRNN, model_TGCN


model_TGNN, model_DCRNN, model_TGCN = get_trained_models()
