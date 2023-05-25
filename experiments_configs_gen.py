import copy
import json

from configs.dl import dl_best_hparams

models = ["MCGRU"]
tasks = ["outcome", "los", "multitask"]
datasets = ["tjh", "cdsl"]

def dataset_related_config_map(dataset):
    if dataset=="tjh":
        return {"demo_dim":2, "lab_dim":73, "batch_size": 64}
    elif dataset=="cdsl":
        return {"demo_dim":2, "lab_dim":97, "batch_size": 128}

def main_metric_map(task):
    if task in ["outcome", "multitask"]:
        return {"main_metric": "auprc"}
    elif task=="los":
        return {"main_metric": "mae"}


base_config = {
    "epochs": 100,
    "patience": 10,
    "learning_rate": 0.001,
    "output_dim": 1,
    "hidden_dim": 64,
}


"""
Part 01
----------
Use time-aware loss for deep learning models
(two datasets, top-5 performance model)
"""

part01_configs = [
 {'model': 'RNN',
  'dataset': 'tjh',
  'task': 'outcome',
  'epochs': 100,
  'patience': 10,
  'batch_size': 64,
  'learning_rate': 0.0001,
  'main_metric': 'auprc',
  'demo_dim': 2,
  'lab_dim': 73,
  'hidden_dim': 64,
  'output_dim': 1,
  'time_aware': True,
 },
 {'model': 'GRU',
  'dataset': 'tjh',
  'task': 'outcome',
  'epochs': 100,
  'patience': 10,
  'batch_size': 64,
  'learning_rate': 0.0001,
  'main_metric': 'auprc',
  'demo_dim': 2,
  'lab_dim': 73,
  'hidden_dim': 32,
  'output_dim': 1,
  'time_aware': True,
 },
 {'model': 'TCN',
  'dataset': 'tjh',
  'task': 'outcome',
  'epochs': 100,
  'patience': 10,
  'batch_size': 64,
  'learning_rate': 0.001,
  'main_metric': 'auprc',
  'demo_dim': 2,
  'lab_dim': 73,
  'hidden_dim': 64,
  'output_dim': 1,
  'time_aware': True,
 },
 {'model': 'StageNet',
  'dataset': 'tjh',
  'task': 'outcome',
  'epochs': 100,
  'patience': 10,
  'batch_size': 64,
  'learning_rate': 0.001,
  'main_metric': 'auprc',
  'demo_dim': 2,
  'lab_dim': 73,
  'hidden_dim': 32,
  'output_dim': 1,
  'time_aware': True,
 },
 {'model': 'RNN',
  'dataset': 'cdsl',
  'task': 'outcome',
  'epochs': 100,
  'patience': 10,
  'batch_size': 128,
  'learning_rate': 0.01,
  'main_metric': 'auprc',
  'demo_dim': 2,
  'lab_dim': 97,
  'hidden_dim': 64,
  'output_dim': 1,
  'time_aware': True,
 },
 {'model': 'GRU',
  'dataset': 'cdsl',
  'task': 'outcome',
  'epochs': 100,
  'patience': 10,
  'batch_size': 128,
  'learning_rate': 0.01,
  'main_metric': 'auprc',
  'demo_dim': 2,
  'lab_dim': 97,
  'hidden_dim': 32,
  'output_dim': 1,
  'time_aware': True,
 },
 {'model': 'TCN',
  'dataset': 'cdsl',
  'task': 'outcome',
  'epochs': 100,
  'patience': 10,
  'batch_size': 128,
  'learning_rate': 0.001,
  'main_metric': 'auprc',
  'demo_dim': 2,
  'lab_dim': 97,
  'hidden_dim': 64,
  'output_dim': 1,
  'time_aware': True,
 },
 {'model': 'StageNet',
  'dataset': 'cdsl',
  'task': 'outcome',
  'epochs': 100,
  'patience': 10,
  'batch_size': 128,
  'learning_rate': 0.001,
  'main_metric': 'auprc',
  'demo_dim': 2,
  'lab_dim': 97,
  'hidden_dim': 32,
  'output_dim': 1,
  'time_aware': True,
 },
]


"""
Part 02
----------
MC-GRU Series, w/ or w/o time-aware loss
Tasks: outcome/los/multitask
& outcome+time_aware_loss (add two configs for both datasets)
"""

part02_configs = [
 {'model': 'MCGRU',
  'dataset': 'cdsl',
  'task': 'outcome',
  'epochs': 100,
  'patience': 10,
  'batch_size': 128,
  'learning_rate': 0.001,
  'main_metric': 'auprc',
  'demo_dim': 2,
  'lab_dim': 97,
  'hidden_dim': 64,
  'output_dim': 1,
  'time_aware': True,
 },
 {'model': 'MCGRU',
  'dataset': 'tjh',
  'task': 'outcome',
  'epochs': 100,
  'patience': 10,
  'batch_size': 64,
  'learning_rate': 0.001,
  'main_metric': 'auprc',
  'demo_dim': 2,
  'lab_dim': 73,
  'hidden_dim': 64,
  'output_dim': 1,
  'time_aware': True,
 },
]
models = ["MCGRU"]
for task in tasks:
    for dataset in datasets:
        for model in models:
            config = copy.copy(base_config)
            config["task"]=task
            config["dataset"]=dataset
            config.update(dataset_related_config_map(dataset))
            config.update(main_metric_map(task))
            config["model"]=model
            part02_configs.append(config)


to_add_configs = []
to_add_configs.extend(part01_configs)
to_add_configs.extend(part02_configs)
print(len(to_add_configs))
def unique(list1):
    # intilize a null list
    unique_list = []
    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    return unique_list
unique_list = unique(to_add_configs)
print("unique configs", len(unique_list))

with open('configs/experiments.py', 'w') as fout:
    json.dump(unique_list, fout, indent=4)