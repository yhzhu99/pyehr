import random
from pathlib import Path

import pandas as pd
import os
import hydra
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from omegaconf import DictConfig, OmegaConf
import numpy as np
from tqdm import tqdm

from datasets.loader.datamodule import EhrDataModule
from datasets.loader.load_los_info import get_los_info
from pipelines import DlPipeline, MlPipeline
from configs.dl import dl_best_hparams
from configs.ml import ml_best_hparams

project_name = "pyehr"

def run_ml_experiment(config):
    los_config = get_los_info(f'datasets/{config["dataset"]}/processed/fold_{config["fold"]}')
    config.update({"los_info": los_config})
    # data
    dm = EhrDataModule(f'datasets/{config["dataset"]}/processed/fold_{config["fold"]}', batch_size=config["batch_size"])
    # train/val/test
    pipeline = MlPipeline(config)
    trainer = L.Trainer(accelerator="cpu", max_epochs=1, logger=False, num_sanity_val_steps=0)
    trainer.test(pipeline, dm)
    perf = pipeline.test_performance
    return perf

def run_dl_experiment(config):
    los_config = get_los_info(f'datasets/{config["dataset"]}/processed/fold_{config["fold"]}')
    config.update({"los_info": los_config})

    # data
    dm = EhrDataModule(f'datasets/{config["dataset"]}/processed/fold_{config["fold"]}', batch_size=config["batch_size"])
    # checkpoint
    checkpoint_path = f'logs/train/{config["dataset"]}/{config["task"]}/{config["model"]}-fold{config["fold"]}-seed{config["seed"]}/checkpoints/test.ckpt'
    # train/val/test
    pipeline = DlPipeline(config)
    trainer = L.Trainer(accelerator="cpu", max_epochs=1, logger=False, num_sanity_val_steps=0)
    trainer.test(pipeline, dm, ckpt_path=checkpoint_path)
    perf = pipeline.test_performance
    return perf

if __name__ == "__main__":
    performance_table = {'dataset':[], 'task': [], 'model': [], 'fold': [], 'seed': [], 'accuracy': [], 'auroc': [], 'auprc': [], 'mae': [], 'mse': [], 'rmse': [], 'r2': []}
    config = {'model': 'MCGRU',
                'dataset': 'cdsl',
                'task': 'outcome',
                'epochs': 100,
                'patience': 10,
                'batch_size': 64,
                'learning_rate': 0.001,
                'main_metric': 'auprc',
                'demo_dim': 2,
                'lab_dim': 97,
                'hidden_dim': 64,
                'output_dim': 1}
    run_func = run_ml_experiment if config["model"] in ["RF", "DT", "GBDT", "XGBoost", "CatBoost"] else run_dl_experiment
    if config["dataset"]=="cdsl":
        seeds = [0]
        folds = [0,1,2,3,4,5,6,7,8,9]
    else: # tjh dataset
        seeds = [0]
        folds = [0,1,2,3,4,5,6,7,8,9]
    for fold in folds:
        config["fold"] = fold
        for seed in seeds:
            config["seed"] = seed
            perf = run_func(config)
            print(f"{config}, Test Performance: {perf}")
            performance_table['dataset'].append(config['dataset'])
            performance_table['task'].append(config['task'])
            performance_table['model'].append(config['model'])
            performance_table['fold'].append(config['fold'])
            performance_table['seed'].append(config['seed'])
            if config['task'] == 'outcome':
                performance_table['accuracy'].append(perf['accuracy'])
                performance_table['auroc'].append(perf['auroc'])
                performance_table['auprc'].append(perf['auprc'])
                performance_table['mae'].append(None)
                performance_table['mse'].append(None)
                performance_table['rmse'].append(None)
                performance_table['r2'].append(None)
            elif config['task'] == 'los':
                performance_table['accuracy'].append(None)
                performance_table['auroc'].append(None)
                performance_table['auprc'].append(None)
                performance_table['mae'].append(perf['mae'])
                performance_table['mse'].append(perf['mse'])
                performance_table['rmse'].append(perf['rmse'])
                performance_table['r2'].append(perf['r2'])
            else:
                performance_table['accuracy'].append(perf['accuracy'])
                performance_table['auroc'].append(perf['auroc'])
                performance_table['auprc'].append(perf['auprc'])
                performance_table['mae'].append(perf['mae'])
                performance_table['mse'].append(perf['mse'])
                performance_table['rmse'].append(perf['rmse'])
                performance_table['r2'].append(perf['r2'])
    pd.DataFrame(performance_table).to_csv('performance_mcgru_outcome_base.csv', index=False) # [TO-SPECIFY]