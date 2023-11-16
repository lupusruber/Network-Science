from pathlib import Path

import torch
import torch.utils.data

from models import TemporalGNN, RecurrentGNN, TemporalGraphCN
from train_and_test import train_and_eval_DCRNN, train_and_eval_TGCN, train_and_eval_TGNN

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
        loaded_state_dict = torch.load(path_TGNN)
        model_TGNN.load_state_dict(loaded_state_dict)
        print("TGNN device found !")
    else:
        state_dict = train_and_eval_TGNN(number_of_epochs=2, BATCH_SIZE=32)
        torch.save(state_dict, path_TGNN)

    if path_DCRNN.is_file():
        loaded_state_dict = torch.load(path_DCRNN)
        model_DCRNN.load_state_dict(loaded_state_dict)
        print("TGNN device found !")
    else:
        state_dict = train_and_eval_DCRNN(number_of_epochs=2, BATCH_SIZE=1)
        torch.save(state_dict, path_DCRNN)

    if path_TGCN.is_file():
        loaded_state_dict = torch.load(path_TGCN)
        model_TGCN.load_state_dict(loaded_state_dict)
        print("TGNN device found !")
    else:
        state_dict = train_and_eval_TGCN(number_of_epochs=2, BATCH_SIZE=1)
        torch.save(state_dict, path_TGCN)