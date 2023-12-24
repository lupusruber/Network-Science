import torch
from matplotlib import pyplot as plt
from torch import Tensor
import seaborn as sns
import numpy as np
from torch.utils.data import DataLoader

from data import (
    create_test_data_loader,
    dataset,
    test_data_set,
    static_edge_index,
    get_sample_data_for_viz,
)
from models import model_DCRNN, model_TGCN, model_TGNN


DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"


def get_all_y_hats(test_loader: DataLoader, model_name: str) -> tuple[Tensor, Tensor]:
    pred_val = list()
    true_val = list()
    if "DCRNN" == model_name:
        for X, y in test_loader:
            X = X.squeeze()
            X = X.reshape(325, 24)
            y = y.squeeze()
            with torch.inference_mode():
                model_TGCN.eval()
                Y_hat = model_DCRNN(X, static_edge_index).to(DEVICE)
            pred_val.append(Y_hat.cpu())
            true_val.append(y[:, 0].cpu())
    if "TGCN" == model_name:
        pass
    if "TGNNN" == model_name:
        pass

    return torch.tensor(np.array(pred_val)), torch.tensor(np.array(true_val))


def visualize_data(sensor_number=1, time_steps=24):
    sensor_labels = [
        bucket.y[sensor_number][0].item() for bucket in list(dataset)[:time_steps]
    ]
    plt.plot(sensor_labels)
    plt.show()


def visualize_sensors_for_every_time_stamp(
    n: int, predicted: Tensor, true: Tensor, title: str
):
    true_output = true.cpu()
    predicted_output = predicted.cpu()

    plt.figure(figsize=(30, 4), dpi=80)
    plt.title(title)

    averages_pred = [
        torch.mean(time_stamp[:, 0]).item() for time_stamp in predicted_output[:n]
    ]
    averages_true = [
        torch.mean(time_stamp[:, 0]).item() for time_stamp in true_output[:n]
    ]

    sns.lineplot(
        x=list(range(n)), y=averages_pred, label="Predicted for sensors by time stamp"
    )
    sns.lineplot(
        x=list(range(n)), y=averages_true, label="True for sensors by time stamp"
    )

    plt.xlabel("Predicted Values")
    plt.ylabel("True Values")
    plt.title("True vs. Predicted Values")

    plt.legend()
    plt.show()
    plt.savefig("predicted_vs_true.png")
    plt.close()


def prediction_of_first_n_detectors(
    n: int, predicted: Tensor, true: Tensor, next: int, title: str
):
    true_output = true.cpu()
    predicted_output = predicted.cpu()

    plt.figure(figsize=(30, 4), dpi=80)
    plt.title(title)

    count = 0
    flag = True
    for i in range(next * n, (next + 1) * n, 1):
        next_i = 12 * count
        if flag:
            plt.plot(
                range(0 + next_i, 12 + next_i),
                true_output[i].numpy(),
                color="blue",
                label="true",
            )
            plt.plot(
                range(0 + next_i, 12 + next_i),
                predicted_output[i].detach().numpy(),
                color="red",
                label="predicted",
            )
            flag = False
        else:
            plt.plot(
                range(0 + next_i, 12 + next_i), true_output[i].numpy(), color="blue"
            )
            plt.plot(
                range(0 + next_i, 12 + next_i),
                predicted_output[i].detach().numpy(),
                color="red",
            )
        count += 1
    plt.legend()
    plt.show()


def visualise_sensors(*visualisations):
    if "TGCN" in visualisations:
        pass

    if "DCRNN" in visualisations:
        pred, true = get_all_y_hats(
            test_loader=create_test_data_loader(
                test_data_set=test_data_set, BATCH_SIZE=1
            ),
            model_name="DCRNN",
        )
        visualize_sensors_for_every_time_stamp(
            n=1000, predicted=pred, true=true, title="DCRNN"
        )

    if "TGNN" in visualisations:
        pass


def visualise(*visualisations):
    X_test, y_test = get_sample_data_for_viz()
    timestep = 5
    X_test_tgcn = X_test[timestep, :, 0]
    y_test = y_test[timestep, :, 0]

    if "TGCN" in visualisations:
        y_hat_tgcn = model_TGCN(X_test_tgcn, static_edge_index).to(DEVICE)
        prediction_of_first_n_detectors(
            n=20, next=0, predicted=y_hat_tgcn, true=y_test, title="TGCN"
        )

    if "DCRNN" in visualisations:
        X_test_dcrn = X_test[timestep, :].reshape(325, 24)
        y_hat_dcrn = model_DCRNN(X_test_dcrn, static_edge_index).to(DEVICE)
        prediction_of_first_n_detectors(
            n=20, next=0, predicted=y_hat_dcrn, true=y_test, title="DCRNN"
        )

    if "TGNN" in visualisations:
        X_data = create_test_data_loader(test_data_set=test_data_set, BATCH_SIZE=32)
        X_from_loader, y_from_loader = None, None
        for index, (X, y) in enumerate(X_data):
            if index == timestep // 32:
                X_from_loader, y_from_loader = X, y[timestep % 32]
                break

        y_hat_tgnn = model_TGNN(X_from_loader, static_edge_index).to(DEVICE)

        prediction_of_first_n_detectors(
            n=20,
            next=0,
            predicted=y_hat_tgnn[timestep % 32],
            true=y_from_loader[:, 0],
            title="TGNN",
        )
