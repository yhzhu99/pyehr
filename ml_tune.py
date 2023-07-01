import hydra
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from omegaconf import DictConfig, OmegaConf

import wandb
from datasets.loader.datamodule import EhrDataModule
from datasets.loader.load_los_info import get_los_info
from pipelines import DlPipeline, MlPipeline

# import os
# os.environ['WANDB_MODE'] = 'offline'
# os.environ['WANDB_LOG_LEVEL'] = 'debug'

project_name = "pyehr"

hydra.initialize(config_path="configs", version_base=None)
cfg = OmegaConf.to_container(hydra.compose(config_name="config"))

sweep_configuration = {
    'method': 'grid',
    'name': 'sweep_ml',
    'parameters': 
    {
        'task': {'values': ['outcome', 'los']},
        'dataset': {'values': ['tjh', 'cdsl']},
        'model': {'values': ['CatBoost', 'RF', 'XGBoost', 'GBDT', 'DT']},
        'learning_rate': {'values': [0.01, 0.1, 1.0]},
        'n_estimators': {'values': [10, 50, 100]},
        'max_depth': {'values': [5, 10, 20]},
        'fold': {'values': [0]},
        'seed': {'values': [42]},
    }
}

sweep_id = wandb.sweep(sweep_configuration, project=project_name)

def run_experiment():
    run = wandb.init(project=project_name, config=cfg)
    wandb_logger = WandbLogger(project=project_name, log_model=True) # log only the last (best) checkpoint
    config = wandb.config
    los_config = get_los_info(f'datasets/{config["dataset"]}/processed/fold_{config["fold"]}')
    main_metric = "mae" if config["task"] == "los" else "auprc"
    config.update({"los_info": los_config, "main_metric": main_metric})
    
    # data
    dm = EhrDataModule(f'datasets/{config["dataset"]}/processed/fold_{config["fold"]}', batch_size=config["batch_size"])

    # checkpoint callback
    if config["task"] in ["outcome"]:
        checkpoint_callback = ModelCheckpoint(monitor="best_auprc", mode="max")
    elif config["task"] == "los":
        checkpoint_callback = ModelCheckpoint(monitor="best_mae", mode="min")

    L.seed_everything(config["seed"]) # seed for reproducibility
    
    # train/val/test
    pipeline = MlPipeline(config.as_dict())
    trainer = L.Trainer(accelerator="cpu", max_epochs=1, logger=wandb_logger, callbacks=[checkpoint_callback], num_sanity_val_steps=0)
    trainer.fit(pipeline, dm)
    print("Best Score", checkpoint_callback.best_model_score)

if __name__ == "__main__":
    wandb.agent(sweep_id, function=run_experiment, project=project_name)
