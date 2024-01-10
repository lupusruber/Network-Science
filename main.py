import torch.utils.data
from viz import save_and_load_y

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    models = ("DCRNN", "TGCN", "TGNN")
    for model in models:
        save_and_load_y(model)

