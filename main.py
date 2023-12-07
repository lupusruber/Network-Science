from pathlib import Path
from train_and_test import static_edge_index
import torch
import torch.utils.data

import viz
from models import TemporalGNN, RecurrentGNN, TemporalGraphCN
from train_and_test import train_and_eval_DCRNN, train_and_eval_TGCN, train_and_eval_TGNN, get_sample_data_for_viz

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"


def get_models():
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
    path_TGNN = Path(r'saved_models/model_TGNN_state_dict.pth')
    path_DCRNN = Path(r'saved_models/model_DCRNN_state_dict.pth')
    path_TGCN = Path(r'saved_models/model_TGCN_state_dict.pth')
    return path_TGNN, path_DCRNN, path_TGCN


if __name__ == "__main__":

    path_TGNN, path_DCRNN, path_TGCN = get_paths()

    model_TGNN, model_DCRNN, model_TGCN = get_models()

    if path_TGNN.is_file():
        loaded_state_dict = torch.load(path_TGNN, map_location=DEVICE)
        model_TGNN.load_state_dict(loaded_state_dict)
        print("TGNN device found !")
    else:
        state_dict = train_and_eval_TGNN(number_of_epochs=2, BATCH_SIZE=32)
        torch.save(state_dict, path_TGNN)

    if path_DCRNN.is_file():
        loaded_state_dict = torch.load(path_DCRNN, map_location=DEVICE)
        model_DCRNN.load_state_dict(loaded_state_dict)
        print("DCRNN device found !")
    else:
        state_dict = train_and_eval_DCRNN(number_of_epochs=2, BATCH_SIZE=1)
        torch.save(state_dict, path_DCRNN)

    if path_TGCN.is_file():
        loaded_state_dict = torch.load(path_TGCN, map_location=DEVICE)
        model_TGCN.load_state_dict(loaded_state_dict)
        print("TGCN device found !")
    else:
        state_dict = train_and_eval_TGCN(number_of_epochs=2, BATCH_SIZE=1)
        torch.save(state_dict, path_TGCN)

    X_test, y_test = get_sample_data_for_viz()
    timestep = 0
    print(X_test.shape)
    X_test = X_test[timestep, :, 0]
    y_test = y_test[timestep, :, 0]
    print(X_test.shape)
    y_hat = model_TGCN(X_test, static_edge_index).to(DEVICE)
    viz.prediction_of_first_n_detectors(n=20, next=0, predicted=y_hat, true=y_test)
