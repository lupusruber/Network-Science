import torch
from matplotlib import pyplot as plt
from torch import Tensor
import seaborn as sns

from train_and_test import dataset


def visualize_data(sensor_number=1, time_steps=24):
    sensor_labels = [
        bucket.y[sensor_number][0].item() for bucket in list(dataset)[:time_steps]
    ]
    plt.plot(sensor_labels)
    plt.show()


def visualize_sensors_for_every_time_stamp(n: int, predicted: Tensor, true: Tensor, title: str):
    true_output = true.cpu()
    predicted_output = predicted.cpu()

    plt.figure(figsize=(30, 4), dpi=80)
    plt.title(title)

    averages_pred = [torch.mean(time_stamp[:, 0]).item() for time_stamp in predicted_output[:n]]
    averages_true = [torch.mean(time_stamp[:, 0]).item() for time_stamp in true_output[:n]]

    sns.lineplot(x=list(range(n)), y=averages_pred,  label="Predicted for sensors by time stamp")
    sns.lineplot(x=list(range(n)), y=averages_true,  label="True for sensors by time stamp")

    plt.xlabel('Predicted Values')
    plt.ylabel('True Values')
    plt.title('True vs. Predicted Values')

    plt.legend()
    plt.show()


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
