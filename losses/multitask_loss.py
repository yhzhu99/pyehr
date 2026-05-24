import torch
import torch.nn.functional as F
from torch import nn


class MultitaskLoss(nn.Module):
    def __init__(self, task_num=2):
        super(MultitaskLoss, self).__init__()
        self.task_num = task_num
        self.log_vars = nn.Parameter(torch.zeros(task_num))

    def forward(self, outcome_pred, los_pred, outcome, los):
        loss0 = F.binary_cross_entropy(outcome_pred, outcome)
        loss1 = F.mse_loss(los_pred, los)
        task_losses = torch.stack([loss0, loss1])
        precision = torch.exp(-self.log_vars)
        return torch.sum(precision * task_losses + self.log_vars)

    @property
    def alpha(self):
        return torch.exp(-self.log_vars).detach()

def get_multitask_loss(outcome_pred, los_pred, outcome, los):
    mtl = MultitaskLoss(task_num=2)
    return mtl(outcome_pred, los_pred, outcome, los)
