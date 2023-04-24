from .binary_classification_metrics import get_binary_metrics
from .regression_metrics import get_regression_metrics


def get_all_metrics(preds, labels, task):
    if task == "outcome":
        return get_binary_metrics(preds, labels)
    elif task == "los":
        return get_regression_metrics(preds, labels)
    else:
        raise ValueError("Task not supported")