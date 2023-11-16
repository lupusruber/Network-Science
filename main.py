import torch
import torch.utils.data

from train_and_test import train_and_eval_DCRNN, train_and_eval_TGCN, train_and_eval_TGNN

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    print(DEVICE)
    train_and_eval_TGCN(number_of_epochs=5, BATCH_SIZE=1)
    
    print("bla bla")
