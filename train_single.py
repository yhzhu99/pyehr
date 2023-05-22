import random
from pathlib import Path

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
    # logger
    checkpoint_filename = f'{config["model"]}-fold{config["fold"]}-seed{config["seed"]}'
    logger = CSVLogger(save_dir="logs", name=f'train/{config["dataset"]}/{config["task"]}', version=checkpoint_filename)
    L.seed_everything(config["seed"]) # seed for reproducibility

    # train/val/test
    pipeline = MlPipeline(config)
    trainer = L.Trainer(accelerator="cpu", max_epochs=1, logger=logger, num_sanity_val_steps=0)
    trainer.fit(pipeline, dm)
    perf = pipeline.cur_best_performance
    return perf

def run_dl_experiment(config):
    los_config = get_los_info(f'datasets/{config["dataset"]}/processed/fold_{config["fold"]}')
    config.update({"los_info": los_config})

    # data
    dm = EhrDataModule(f'datasets/{config["dataset"]}/processed/fold_{config["fold"]}', batch_size=config["batch_size"])
    # logger
    checkpoint_filename = f'{config["model"]}-fold{config["fold"]}-seed{config["seed"]}'
    logger = CSVLogger(save_dir="logs", name=f'train/{config["dataset"]}/{config["task"]}', version=checkpoint_filename)

    # EarlyStop and checkpoint callback
    if config["task"] in ["outcome", "multitask"]:
        early_stopping_callback = EarlyStopping(monitor="auprc", patience=config["patience"], mode="max",)
        checkpoint_callback = ModelCheckpoint(filename="test", monitor="auprc", mode="max")
    elif config["task"] == "los":
        early_stopping_callback = EarlyStopping(monitor="mae", patience=config["patience"], mode="min",)
        checkpoint_callback = ModelCheckpoint(filename="test", monitor="mae", mode="min")

    L.seed_everything(config["seed"]) # seed for reproducibility

    # train/val/test
    pipeline = DlPipeline(config)
    trainer = L.Trainer(accelerator="gpu", devices=[1], max_epochs=config["epochs"], logger=logger, callbacks=[early_stopping_callback, checkpoint_callback])
    trainer.fit(pipeline, dm)
    perf = pipeline.cur_best_performance
    return perf

if __name__ == "__main__":

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
        folds = [0]
    else: # tjh dataset
        seeds = [0]
        # folds = [0,1,2,3,4,5,6,7,8,9]
        folds = [0]
    for fold in folds:
        config["fold"] = fold
        for seed in seeds:
            config["seed"] = seed
            perf = run_func(config)
            print(f"{config}, Val Performance: {perf}")
