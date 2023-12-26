import torch.utils.data
from viz import get_all_y_for_DCRNN, visualise_sensors

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    # visualise_sensors("DCRNN", "TGCN", "TGNN")
    get_all_y_for_DCRNN()