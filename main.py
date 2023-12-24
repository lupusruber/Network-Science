from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

import train_and_test
from train_and_test import static_edge_index, create_test_data_loader
import torch
import torch.utils.data

import viz
from models import TemporalGNN, RecurrentGNN, TemporalGraphCN
from train_and_test import (
    train_and_eval_DCRNN,
    train_and_eval_TGCN,
    train_and_eval_TGNN,
    get_sample_data_for_viz,
)

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"


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


def get_all_y_hats(test_loader: DataLoader, model_name: str):
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


def visualise_sensors(*visualisations):
    if "TGCN" in visualisations: pass

    if "DCRNN" in visualisations:
        pred, true = get_all_y_hats(create_test_data_loader(test_data_set=train_and_test.test_data_set, BATCH_SIZE=1),
                                    "DCRNN")
        viz.visualize_sensors_for_every_time_stamp(n=1000, predicted=pred, true=true, title="DCRNN")

    if "TGNN" in visualisations: pass


def visualise(*visualisations):
    X_test, y_test = get_sample_data_for_viz()
    timestep = 5
    X_test_tgcn = X_test[timestep, :, 0]
    y_test = y_test[timestep, :, 0]

    if "TGCN" in visualisations:
        y_hat_tgcn = model_TGCN(X_test_tgcn, static_edge_index).to(DEVICE)
        viz.prediction_of_first_n_detectors(
            n=20, next=0, predicted=y_hat_tgcn, true=y_test, title="TGCN"
        )

    if "DCRNN" in visualisations:
        X_test_dcrn = X_test[timestep, :].reshape(325, 24)
        y_hat_dcrn = model_DCRNN(X_test_dcrn, static_edge_index).to(DEVICE)
        viz.prediction_of_first_n_detectors(
            n=20, next=0, predicted=y_hat_dcrn, true=y_test, title="DCRNN"
        )

    if "TGNN" in visualisations:
        X_data = create_test_data_loader(
            test_data_set=train_and_test.test_data_set, BATCH_SIZE=32
        )
        X_from_loader, y_from_loader = None, None
        for index, (X, y) in enumerate(X_data):
            if index == timestep // 32:
                X_from_loader, y_from_loader = X, y[timestep % 32]
                break

        y_hat_tgnn = model_TGNN(X_from_loader, static_edge_index).to(DEVICE)

        viz.prediction_of_first_n_detectors(
            n=20,
            next=0,
            predicted=y_hat_tgnn[timestep % 32],
            true=y_from_loader[:, 0],
            title="TGNN",
        )


if __name__ == "__main__":
    model_TGNN, model_DCRNN, model_TGCN = get_trained_models()
    # visualise("DCRNN", "TGNN", "TGCN")
    visualise_sensors("DCRNN")
