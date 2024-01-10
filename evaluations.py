import torch
from viz import *


def evaluate(type: str, y_hat, y_test) -> None:
    if type == "MAE":
        loss_fn = torch.nn.L1Loss()
        train_loss_list = []
        for time, (hat, test) in enumerate(zip(y_hat, y_test)):
            hat = hat[:, 0]
            test = test[:, 0]
            train_cost = loss_fn(target=hat, input=test)
            train_loss_list.append(train_cost.item())
            train_mae: float = sum(train_loss_list) / len(train_loss_list)
            print(f"Test MAE in step {time}: {train_mae}")

    elif type == "RMSE":
        loss_fn = torch.nn.MSELoss()
        train_loss_list = []
        for time, (hat, test) in enumerate(zip(y_hat, y_test)):
            train_cost = torch.sqrt(loss_fn(target=hat, input=test))
            train_loss_list.append(train_cost.item())
            train_mae: float = sum(train_loss_list) / len(train_loss_list)
            print(f"Test MSE in step {time}: {train_mae}")

    elif type == "MSE":
        loss_fn = torch.nn.MSELoss()
        train_loss_list = []
        for time, (hat, test) in enumerate(zip(y_hat, y_test)):
            hat = hat[:, 0]
            test = test[:, 0]
            train_cost = loss_fn(target=hat, input=test)
            train_loss_list.append(train_cost.item())
            train_mae: float = sum(train_loss_list) / len(train_loss_list)
            print(f"Test MSE in step {time}: {train_mae}")


if __name__ == '__main__':
    pred_DCRNN, true_DCRNN = get_all_y_for_DCRNN()
    evaluate(type="RMSE", y_hat=true_DCRNN, y_test=pred_DCRNN)