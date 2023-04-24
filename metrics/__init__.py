from .binary_classification_metrics import get_binary_metrics
from .regression_metrics import get_regression_metrics


def get_all_metrics(preds, labels, task):
    if task == "outcome":
        return get_binary_metrics(preds, labels[:,0])
    elif task == "los":
        return get_regression_metrics(preds, labels[:,1])
    elif task == "multitask":
        return get_binary_metrics(preds[:,0], labels[:,0]) | get_regression_metrics(preds[:,1], labels[:,1])
    else:
        raise ValueError("Task not supported")