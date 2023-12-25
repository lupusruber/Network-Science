import torch.utils.data
from viz import visualise_sensors
from models import train_and_eval_TGNN

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    visualise_sensors("TGNN")
