import torch
import torch.nn.functional as F

from .multitask_loss import get_multitask_loss


def get_simple_loss(y_pred, y_true, task):
    if task == "outcome":
        loss = F.binary_cross_entropy_with_logits(y_pred, y_true[:, 0])
    elif task == "los":
        loss = F.mse_loss(y_pred, y_true[:, 1])
    elif task == "multitask":
        loss = get_multitask_loss(y_pred[:,0], y_pred[:,1], y_true[:,0], y_true[:,1])
    return loss
