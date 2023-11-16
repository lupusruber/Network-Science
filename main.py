from pathlib import Path

import torch
import torch.utils.data

from models import TemporalGNN
from train_and_test import train_and_eval_DCRNN, train_and_eval_TGCN, train_and_eval_TGNN

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":

    path_TGNN = Path(r'saved_models/model_TGNN_state_dict.pth')

    model_TGNN = TemporalGNN(
        node_features=2,
        out_periods=12,
        hidden_units=32,
        batch_size=32,
    ).to(DEVICE)

    if path_TGNN.is_file():
        loaded_state_dict = torch.load(path_TGNN)
        model_TGNN.load_state_dict(loaded_state_dict)
        print("TGNN device found !")
    else:
        state_dict = train_and_eval_TGNN(number_of_epochs=2, BATCH_SIZE=32)
        torch.save(state_dict, path_TGNN)