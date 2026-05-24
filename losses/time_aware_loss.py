import torch
from torch import nn


class TimeAwareLoss(nn.Module):
    def __init__(self, decay_rate=0.1, reward_factor=1.0, los_mean=None, los_std=None):
        super(TimeAwareLoss, self).__init__()
        self.bce = nn.BCELoss(reduction='none')
        self.decay_rate = decay_rate
        self.reward_factor = reward_factor
        self.los_mean = los_mean
        self.los_std = los_std

    def forward(self, outcome_pred, outcome_true, los_true):
        raw_los = los_true
        if self.los_mean is not None and self.los_std is not None:
            raw_los = los_true * self.los_std + self.los_mean
        raw_los = raw_los.clamp_min(0)
        los_weights = 1 + self.reward_factor * (1 - torch.exp(-self.decay_rate * raw_los))
        los_weights = los_weights / los_weights.mean().detach().clamp_min(1e-12)
        loss_unreduced = self.bce(outcome_pred, outcome_true)
        return (loss_unreduced * los_weights).mean()

def get_time_aware_loss(outcome_pred, outcome_true, los_true, los_info=None):
    los_info = los_info or {}
    time_aware_loss = TimeAwareLoss(
        los_mean=los_info.get("los_mean"),
        los_std=los_info.get("los_std"),
    )
    return time_aware_loss(outcome_pred, outcome_true, los_true)

if __name__ == "__main__":
    outcome_pred = torch.tensor([0.1])
    outcome_true = torch.tensor([1.])
    los_true = torch.tensor([-4.0])
    print(get_time_aware_loss(outcome_pred, outcome_true, los_true))
