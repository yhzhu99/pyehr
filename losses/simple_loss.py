import torch
import torch.nn.functional as F


def get_simple_loss(y_pred, y_true, task):
    if task == "outcome":
        y_true = y_true[:, 0]
        loss = F.binary_cross_entropy_with_logits(y_pred, y_true)
    elif task == "los":
        y_true = y_true[:, 1]
        loss = F.mse_loss(y_pred, y_true)
    return loss
