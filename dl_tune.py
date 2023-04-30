import random

import hydra
import lightning as L
import wandb
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from omegaconf import DictConfig, OmegaConf

from datasets.loader.datamodule import EhrDataModule
from datasets.loader.load_los_info import get_los_info
from pipelines import DlPipeline

# import os
# os.environ['WANDB_MODE'] = 'offline'
# os.environ['WANDB_LOG_LEVEL'] = 'debug'

project_name = "pyehr"

hydra.initialize(config_path="configs", version_base=None)
cfg = OmegaConf.to_container(hydra.compose(config_name="config"))

dataset_config = {
    'tjh': {'demo_dim': 2, 'lab_dim': 73},
    'cdsl': {'demo_dim': 2, 'lab_dim': 97},
}

sweep_configuration = {
    'method': 'grid',
    'name': 'sweep',
    'metric': {'goal': 'minimize', 'name': 'val_loss'},
    'parameters': 
    {
        'task': {'values': ['outcome', 'los', 'multitask']},
        'dataset': {'values': ['tjh']},
        'model': {'values': ['MLP', 'GRU', 'RNN', 'LSTM', 'TCN', 'Transformer', 'AdaCare', 'Agent', 'GRASP', 'RETAIN', 'StageNet', 'ConCare']},
        'batch_size': {'values': [64]},
        'hidden_dim': {'values': [32, 64, 128]},
        'learning_rate': {'values': [1e-2, 1e-3, 1e-4]},
        'fold': {'values': [0]},
    }
}

sweep_id = wandb.sweep(sweep_configuration, project=project_name)

def run_experiment():
    run = wandb.init(project=project_name, config=cfg)
    wandb_logger = WandbLogger(project=project_name, log_model=True) # log only the last (best) checkpoint
    wandb.config.update(dataset_config[wandb.config['dataset']])
    los_config = get_los_info(f'datasets/{wandb.config["dataset"]}/processed/fold_{wandb.config["fold"]}')
    main_metric = "mae" if wandb.config["task"] == "los" else "auprc"
    wandb.config.update({"los_info": los_config, "main_metric": main_metric})
    
    # data
    dm = EhrDataModule(f'datasets/{wandb.config["dataset"]}/processed/fold_{wandb.config["fold"]}', batch_size=wandb.config["batch_size"])

    # EarlyStop and checkpoint callback
    if wandb.config["task"] in ["outcome", "multitask"]:
        early_stopping_callback = EarlyStopping(monitor="auprc", patience=wandb.config["patience"], mode="max",)
        checkpoint_callback = ModelCheckpoint(monitor="auprc", mode="max")
    elif wandb.config["task"] == "los":
        early_stopping_callback = EarlyStopping(monitor="mae", patience=wandb.config["patience"], mode="min",)
        checkpoint_callback = ModelCheckpoint(monitor="mae", mode="min")

    # train/val/test
    pipeline = DlPipeline(wandb.config.as_dict())
    trainer = L.Trainer(accelerator="gpu", devices=[1], max_epochs=wandb.config["epochs"], logger=wandb_logger, callbacks=[early_stopping_callback, checkpoint_callback])
    trainer.fit(pipeline, dm)
    print("Best Score", checkpoint_callback.best_model_score)

if __name__ == "__main__":
   wandb.agent(sweep_id, function=run_experiment)
