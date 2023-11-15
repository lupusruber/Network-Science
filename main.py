import torch
import torch.utils.data

from train_and_test import train_and_eval_DCRNN, train_and_eval_TGNN

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    ...
