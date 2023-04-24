import torch
from torch.nn.utils.rnn import unpad_sequence


def unpad_y(y_pred, y_true, lens):
    y_pred_unpad = unpad_sequence(y_pred, batch_first=True, lengths=lens)
    y_pred_stack = torch.vstack(y_pred_unpad).squeeze(dim=-1)
    y_true_unpad = unpad_sequence(y_true, batch_first=True, lengths=lens)
    y_true_stack = torch.vstack(y_true_unpad).squeeze(dim=-1)
    return y_pred_stack, y_true_stack