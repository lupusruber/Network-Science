import torch.utils.data

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    models = ("DCRNN", "TGCN", "TGNN")

