import torch
from torchmetrics import AUROC, Accuracy, AveragePrecision

threshold = 0.5
accuracy = Accuracy(task="binary", threshold=threshold)
auroc = AUROC(task="binary", threshold=threshold)
auprc = AveragePrecision(task="binary", threshold=threshold)

def get_binary_metrics(preds, labels):
    labels = labels[:, 0]
    # convert labels type to int
    labels = labels.type(torch.int)
    accuracy(preds, labels)
    auroc(preds, labels)
    auprc(preds, labels)
    # return a dictionary
    return {
        "accuracy": accuracy.compute(),
        "auroc": auroc.compute(),
        "auprc": auprc.compute(),
    }