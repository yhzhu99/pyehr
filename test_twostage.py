"""
step 1: find the best config combination (outcome/los) of the same model
step 2: get the prediction results
step 3: calculate the metric
"""

import pandas as pd
import numpy as np
import lightning as L
import torch

from datasets.loader.datamodule import EhrDataModule
from datasets.loader.load_los_info import get_los_info
from metrics import get_all_metrics, check_metric_is_better
from pipelines import DlPipeline, MlPipeline

from configs.dl import dl_best_hparams
from configs.ml import ml_best_hparams
from configs.experiments import experiments_configs

models = ["RF", "DT", "GBDT", "XGBoost", "CatBoost", 'MLP', 'GRU', 'RNN', 'LSTM', 'TCN', 'Transformer', 'AdaCare', 'Agent', 'GRASP', 'RETAIN', 'StageNet', 'MCGRU']

all_configs = []
all_configs.extend(dl_best_hparams)
all_configs.extend(ml_best_hparams)
all_configs.extend(experiments_configs)

filterrd_configs = []
for c in all_configs:
    if "time_aware" not in c and c["task"] in ["outcome", "los"]:
        filterrd_configs.append(c)

print(len(filterrd_configs))

# same dataset, same model, pair outcome and los task configs

pair_configs = []
def find_pair(dataset, model):
    for c in filterrd_configs:
        if c["dataset"]==dataset and c["model"]==model and c["task"]=="outcome":
            outcome = c
    for c in filterrd_configs:
        if c["dataset"]==dataset and c["model"]==model and c["task"]=="los":
            los = c
    return (outcome, los)

for dataset in ["tjh", "cdsl"]:
    for model in models:
        pair_configs.append(find_pair(dataset, model))

def run_ml_experiment(config):
    los_config = get_los_info(f'datasets/{config["dataset"]}/processed/fold_{config["fold"]}')
    config.update({"los_info": los_config})
    # data
    dm = EhrDataModule(f'datasets/{config["dataset"]}/processed/fold_{config["fold"]}', batch_size=config["batch_size"])
    # train/val/test
    pipeline = MlPipeline(config)
    trainer = L.Trainer(accelerator="cpu", max_epochs=1, logger=False, num_sanity_val_steps=0)
    trainer.test(pipeline, dm)
    return pipeline.test_outputs

def run_dl_experiment(config):
    los_config = get_los_info(f'datasets/{config["dataset"]}/processed/fold_{config["fold"]}')
    config.update({"los_info": los_config})

    # data
    dm = EhrDataModule(f'datasets/{config["dataset"]}/processed/fold_{config["fold"]}', batch_size=config["batch_size"])
    # checkpoint
    checkpoint_path = f'logs/train/{config["dataset"]}/{config["task"]}/{config["model"]}-fold{config["fold"]}-seed{config["seed"]}/checkpoints/best.ckpt'
    if "time_aware" in config and config["time_aware"] == True:
        checkpoint_path = f'logs/train/{config["dataset"]}/{config["task"]}/{config["model"]}-fold{config["fold"]}-seed{config["seed"]}-ta/checkpoints/best.ckpt' # time-aware loss applied
    # train/val/test
    pipeline = DlPipeline(config)
    trainer = L.Trainer(accelerator="cpu", max_epochs=1, logger=False, num_sanity_val_steps=0)
    trainer.test(pipeline, dm, ckpt_path=checkpoint_path)
    return pipeline.test_outputs


if __name__ == "__main__":
    best_hparams = experiments_configs # [TO-SPECIFY]
    performance_table = {'dataset':[], 'task': [], 'model': [], 'fold': [], 'seed': [], 'accuracy': [], 'auroc': [], 'auprc': [], 'es': [], 'mae': [], 'mse': [], 'rmse': [], 'r2': [], 'osmae': []}
    for pair_config in pair_configs:
    # for i in range(0, 1):
        config, config_los = pair_config
        run_func = run_ml_experiment if config["model"] in ["RF", "DT", "GBDT", "XGBoost", "CatBoost"] else run_dl_experiment
        if config["dataset"]=="cdsl":
            seeds = [0]
            folds = [0,1,2,3,4,5,6,7,8,9]
        else: # tjh dataset
            seeds = [0]
            folds = [0,1,2,3,4,5,6,7,8,9]
        for fold in folds:
            config["fold"] = fold
            config_los["fold"] = fold
            for seed in seeds:
                config["seed"] = seed
                config_los["seed"] = seed
                outcome_model_outputs = run_func(config)
                los_model_outputs = run_func(config_los)
                pred_outcome = outcome_model_outputs["preds"]
                pred_los = los_model_outputs["preds"]
                if run_func==run_ml_experiment: y_pred = np.stack([pred_outcome, pred_los], axis=1)
                else: 
                    y_pred = torch.cat([pred_outcome.unsqueeze(1), pred_los.unsqueeze(1)], dim=1)
                y_true = outcome_model_outputs["labels"]
                los_config = get_los_info(f'datasets/{config["dataset"]}/processed/fold_{fold}')
                perf = get_all_metrics(y_pred, y_true, "multitask", los_config)

                if "time_aware" in config and config["time_aware"] == True:
                    model_name = config['model']+"-ta"
                else:
                    model_name = config['model']

                performance_table['dataset'].append(config['dataset'])
                performance_table['task'].append("twostage")
                performance_table['model'].append(model_name)
                performance_table['fold'].append(config['fold'])
                performance_table['seed'].append(config['seed'])

                performance_table['accuracy'].append(perf['accuracy'])
                performance_table['auroc'].append(perf['auroc'])
                performance_table['auprc'].append(perf['auprc'])
                performance_table['es'].append(perf['es'])
                performance_table['mae'].append(perf['mae'])
                performance_table['mse'].append(perf['mse'])
                performance_table['rmse'].append(perf['rmse'])
                performance_table['r2'].append(perf['r2'])
                performance_table['osmae'].append(perf['osmae'])

                print(model_name, config["fold"], config["dataset"], perf)
    pd.DataFrame(performance_table).to_csv('perf_twostage_0528.csv', index=False) # [TO-SPECIFY]