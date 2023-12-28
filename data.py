import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch_geometric_temporal import PemsBayDatasetLoader, temporal_signal_split
from torch_geometric_temporal.signal import StaticGraphTemporalSignal

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"


data_loader = PemsBayDatasetLoader()
dataset: StaticGraphTemporalSignal = data_loader.get_dataset()
train_data_set, test_data_set = temporal_signal_split(
    data_iterator=dataset, train_ratio=0.8
)

static_edge_index = None
for snapshot in train_data_set:
    static_edge_index = snapshot.edge_index.to(DEVICE)
    break


# def get_sample_data_for_viz() -> tuple[Tensor, Tensor]:
#     test_input = np.array(test_data_set.features)
#     test_target = np.array(test_data_set.targets)

#     test_x_tensor = torch.from_numpy(test_input).to(DEVICE, dtype=torch.float32)
#     test_target_tensor = torch.from_numpy(test_target).to(DEVICE, dtype=torch.float32)

#     return test_x_tensor, test_target_tensor


def create_train_data_loader(
    train_data_set: StaticGraphTemporalSignal, BATCH_SIZE: int
) -> DataLoader:
    train_input = np.array(train_data_set.features)
    train_target = np.array(train_data_set.targets)

    train_x_tensor: Tensor = torch.from_numpy(train_input).to(
        DEVICE, dtype=torch.float32
    )
    train_target_tensor: Tensor = torch.from_numpy(train_target).to(
        DEVICE, dtype=torch.float32
    )

    train_data_set_new = torch.utils.data.TensorDataset(
        train_x_tensor, train_target_tensor
    )
    train_loader: DataLoader = torch.utils.data.DataLoader(
        train_data_set_new, batch_size=BATCH_SIZE, shuffle=False, drop_last=True
    )
    return train_loader


def create_test_data_loader(
    test_data_set: StaticGraphTemporalSignal, BATCH_SIZE: int
) -> DataLoader:
    test_input = np.array(test_data_set.features)
    test_target = np.array(test_data_set.targets)

    test_x_tensor = torch.from_numpy(test_input).to(DEVICE, dtype=torch.float32)
    test_target_tensor = torch.from_numpy(test_target).to(DEVICE, dtype=torch.float32)

    test_data_set_new = torch.utils.data.TensorDataset(
        test_x_tensor, test_target_tensor
    )
    test_loader = torch.utils.data.DataLoader(
        test_data_set_new, batch_size=BATCH_SIZE, shuffle=False, drop_last=True
    )
    return test_loader


test_loader_no_batch = create_test_data_loader(
    test_data_set=test_data_set, BATCH_SIZE=1
)
test_loader_batched = create_test_data_loader(
    test_data_set=test_data_set, BATCH_SIZE=32
)
