import torch
import torch.utils.data
from viz import visualise_sensors


DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    visualise_sensors("DCRNN")
