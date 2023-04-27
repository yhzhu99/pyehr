import lightning as L
import optuna
import toml
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from datasets.loader.datamodule import EhrDataModule
from datasets.loader.load_los_info import get_los_info
from pipelines import DlPipeline

model_name = "StageNet"
stage = "tune"
dataset = "tjh"
task = "outcome" # ["outcome", "los", "multitask"]
fold = 0
tjh_config = {"demo_dim": 2, "lab_dim": 73, "input_dim": 75,}
cdsl_config = {"demo_dim": 2, "lab_dim": 97, "input_dim": 99,}
dataset_config = {}
if dataset == "tjh": dataset_config = tjh_config
elif dataset == "cdsl": dataset_config = cdsl_config
output_dim = 1
main_metric = "mae" if task == "los" else "auprc"
epochs = 100
patience = 10
learning_rate = 1e-3

config = {"stage": stage, "task": task, "dataset": dataset, "output_dim": output_dim, "fold": fold, "epochs": epochs, "patience": patience, "model_name": model_name, "main_metric": main_metric, "learning_rate": learning_rate, "chunk_size": 64}
config = config | dataset_config


def objective(trial: optuna.trial.Trial):
    global config
    # config
    trial_config = {
        "hidden_dim": trial.suggest_int("hidden_dim", 1, 8192),
        "batch_size": trial.suggest_int("batch_size", 1, 8192),
    }
    config = config | trial_config
    los_config = get_los_info(f'datasets/{config["dataset"]}/processed/fold_{config["fold"]}')
    config["los_info"] = los_config

    # data
    dm = EhrDataModule(f'datasets/{config["dataset"]}/processed/fold_{config["fold"]}', batch_size=config["batch_size"])
    
    # callbacks
    checkpoint_filename = f'{config["model_name"]}-fold{config["fold"]}'
    if config["task"] in ["outcome", "multitask"]:
        early_stopping_callback = EarlyStopping(monitor="auprc", patience=config["patience"], mode="max",)
        checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="auprc", mode="max", dirpath=f'./checkpoints/{config["stage"]}/{config["dataset"]}/{config["task"]}', filename=checkpoint_filename,)
    elif config["task"] == "los":
        early_stopping_callback = EarlyStopping(monitor="mae", patience=config["patience"], mode="min",)
        checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="mae", mode="min", dirpath=f'./checkpoints/{config["stage"]}/{config["dataset"]}/{config["task"]}', filename=checkpoint_filename,)
    
    # logger
    logger = CSVLogger(save_dir="logs", name=f'{config["stage"]}/{config["dataset"]}/{config["task"]}', version=checkpoint_filename)
    
    # train/val/test
    pipeline = DlPipeline(config)
    trainer = L.Trainer(max_epochs=config["epochs"], logger=logger, callbacks=[early_stopping_callback, checkpoint_callback])
    trainer.fit(pipeline, dm)

    # return best metric score
    best_metric_score = checkpoint_callback.best_model_score
    return best_metric_score

direction = "minimize" if config["task"] == "los" else "maximize"
search_space = {"hidden_dim": [64], "batch_size": [64]}
study = optuna.create_study(direction=direction, sampler=optuna.samplers.GridSampler(search_space))
study.optimize(objective, n_trials=100)

trial = study.best_trial
config = config | trial.params

# save the config dict to toml file, with name of {model}-{task}-{score}.toml
with open(f'./checkpoints/{config["stage"]}/{config["dataset"]}/{config["task"]}/{config["model_name"]}_best.toml', 'w') as f:
    toml.dump(config, f)

print(config)
