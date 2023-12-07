from typing import Any, Dict
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import Tensor
from torch_geometric_temporal import PemsBayDatasetLoader, temporal_signal_split
from torch_geometric_temporal.signal import StaticGraphTemporalSignal
from models import RecurrentGNN, TemporalGNN, TemporalGraphCN

from copy import deepcopy

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"


data_loader = PemsBayDatasetLoader()
dataset: StaticGraphTemporalSignal = data_loader.get_dataset()
train_data_set, test_data_set = temporal_signal_split(
    data_iterator=dataset, train_ratio=0.8
)

static_edge_index = None
for snapshot in train_data_set:
    static_edge_index = snapshot.edge_index.to(DEVICE)
    break


def get_sample_data_for_viz():
    test_input = np.array(test_data_set.features)
    test_target = np.array(test_data_set.targets)

    test_x_tensor = torch.from_numpy(test_input).to(DEVICE, dtype=torch.float32)
    test_target_tensor = torch.from_numpy(test_target).to(DEVICE, dtype=torch.float32)

    return test_x_tensor, test_target_tensor


def create_train_data_loader(
    train_data_set: StaticGraphTemporalSignal, BATCH_SIZE: int
) -> DataLoader:
    """_summary_

    :param train_data_set: _description_
    :type train_data_set: _type_
    :param BATCH_SIZE: _description_
    :type BATCH_SIZE: _type_
    :return: _description_
    :rtype: DataLoader
    """
    train_input = np.array(train_data_set.features)
    train_target = np.array(train_data_set.targets)

    train_x_tensor: Tensor = torch.from_numpy(train_input).to(
        DEVICE, dtype=torch.float32
    )
    train_target_tensor: Tensor = torch.from_numpy(train_target).to(
        DEVICE, dtype=torch.float32
    )

    train_data_set_new = torch.utils.data.TensorDataset(
        train_x_tensor, train_target_tensor
    )
    train_loader: DataLoader = torch.utils.data.DataLoader(
        train_data_set_new, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )
    return train_loader


def create_test_data_loader(
    test_data_set: StaticGraphTemporalSignal, BATCH_SIZE: int
) -> DataLoader:
    test_input = np.array(test_data_set.features)
    test_target = np.array(test_data_set.targets)

    test_x_tensor = torch.from_numpy(test_input).to(DEVICE, dtype=torch.float32)
    test_target_tensor = torch.from_numpy(test_target).to(DEVICE, dtype=torch.float32)

    test_data_set_new = torch.utils.data.TensorDataset(
        test_x_tensor, test_target_tensor
    )
    test_loader = torch.utils.data.DataLoader(
        test_data_set_new, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )
    return test_loader


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
            y_con = y[:, :, 1]

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
