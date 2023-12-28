from pathlib import Path

import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from torch import Tensor

from data import (
    dataset,
    static_edge_index,
    test_loader_batched,
    test_loader_no_batch,
)
from models import model_DCRNN, model_TGCN, model_TGNN

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"


def get_all_y_hats(model_name: str) -> tuple[Tensor, Tensor]:
    """_summary_
    :param model_name: Name of the model
    :type model_name: str
    :return: predicted and true tensor values for each time step
    :rtype: tuple[Tensor, Tensor]
    """
    pred_val = list()
    true_val = list()
    if "DCRNN" == model_name:
        for X, y in test_loader_no_batch:
            X = X.squeeze()
            X = X.reshape(325, 24)
            y = y.squeeze()
            with torch.inference_mode():
                model_DCRNN.eval()
                y_hat = model_DCRNN(X, static_edge_index).to(DEVICE)
            pred_val.append(y_hat.cpu())
            true_val.append(y[:, 0].cpu())
    if "TGCN" == model_name:
        for X, y in test_loader_no_batch:
            X = X[:, :, 0].squeeze()
            y = y[:, :, 0].squeeze()
            with torch.inference_mode():
                model_TGCN.eval()
                y_hat = model_TGCN(X, static_edge_index).to(DEVICE)
            pred_val.append(y_hat.cpu())
            true_val.append(y.cpu())
    if "TGNN" == model_name:
        for X, y in test_loader_batched:
            with torch.inference_mode():
                model_TGNN.eval()
                y_hat = model_TGNN(X, static_edge_index).to(DEVICE)
            for timestamp in y:
                true_val.append(timestamp[:, 0].cpu())
            for timestamp in y_hat:
                pred_val.append(timestamp.cpu())

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

    plt.xlabel("Time stamps")
    plt.ylabel("Velocities")
    plt.title(f"True vs. Predicted Values {title}")

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
        y_hat, true = save_and_load_y("TGCN")
        visualize_sensors_for_every_time_stamp(
            n=1000, predicted=y_hat, true=true, title="TGCN"
        )

    if "DCRNN" in visualisations:
        y_hat, true = save_and_load_y('DCRNN')
        visualize_sensors_for_every_time_stamp(
            n=1000, predicted=y_hat, true=true, title="DCRNN"
        )

    if "TGNN" in visualisations:
        y_hat, true = save_and_load_y('TGNN')
        visualize_sensors_for_every_time_stamp(
            n=1000, predicted=y_hat, true=true, title="TGNN"
        )


def get_all_y_for_DCRNN() -> tuple[Tensor, Tensor]:
    """_summary_
    :return: tuple of predicted and true values
    :rtype: tuple[Tensor, Tensor]
    """
    return save_and_load_y(
        model_name="DCRNN",
    )


def save_and_load_y(model_name: str) -> tuple[Tensor, Tensor]:
    """_summary_

    :return: tuple of predicted and true values
    :rtype: tuple[Tensor, Tensor]
    """

    true = Path(f"saved_data/true_y_from_{model_name}.pth")
    predicted = Path(f"saved_data/predicted_y_from_{model_name}.pth")

    if true.is_file() and predicted.is_file():
        true_values = torch.load(true, map_location=DEVICE)
        predicted_values = torch.load(predicted, map_location=DEVICE)

        return predicted_values.cpu(), true_values.cpu()

    predicted_values, true_values = get_all_y_hats(model_name=model_name)

    torch.save(true_values, true)
    torch.save(predicted_values, predicted)

    return predicted_values, true_values


def visualise(*visualisations, timestep=0, number_of_sensors=20):
    if "TGCN" in visualisations:
        y_hat, true = save_and_load_y("TGCN")
        prediction_of_first_n_detectors(
            n=number_of_sensors,
            next=0,
            predicted=y_hat[timestep],
            true=true[timestep],
            title=f"TGCN true vs predicted in timestep {timestep}",
        )

    if "DCRNN" in visualisations:
        y_hat, true = save_and_load_y("DCRNN")
        prediction_of_first_n_detectors(
            n=number_of_sensors,
            next=0,
            predicted=y_hat[timestep],
            true=true[timestep],
            title=f"DCRNN true vs predicted in timestep {timestep}",
        )
    if "TGNN" in visualisations:
        y_hat, true = save_and_load_y("TGNN")
        prediction_of_first_n_detectors(
            n=number_of_sensors,
            next=0,
            predicted=y_hat[timestep],
            true=true[timestep],
            title=f"TGNN true vs predicted in timestep {timestep}",
        )


if __name__ == "__main__":
    ...
