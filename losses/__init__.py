import torch
import torch.nn.functional as F

from .multitask_loss import MultitaskLoss, get_multitask_loss
from .time_aware_loss import TimeAwareLoss, get_time_aware_loss


def get_loss(y_pred, y_true, task, time_aware=False, criterion=None, los_info=None):
    if task == "outcome":
        loss = F.binary_cross_entropy(y_pred, y_true[:, 0])
    elif task == "los":
        loss = F.mse_loss(y_pred, y_true[:, 1])
    elif task == "multitask":
        if criterion is None:
            loss = get_multitask_loss(y_pred[:,0], y_pred[:,1], y_true[:,0], y_true[:,1])
        else:
            loss = criterion(y_pred[:,0], y_pred[:,1], y_true[:,0], y_true[:,1])
    else:
        raise ValueError("Task not supported")

    # If use time aware loss:
    if task == "outcome" and time_aware:
        if criterion is None:
            loss = get_time_aware_loss(y_pred, y_true[:, 0], y_true[:, 1], los_info)
        else:
            loss = criterion(y_pred, y_true[:, 0], y_true[:, 1])

    return loss
