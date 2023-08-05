import lightning as L
import pandas as pd

from configs.dl import dl_best_hparams
from configs.experiments import experiments_configs
from configs.ml import ml_best_hparams
from datasets.loader.datamodule import EhrDataModule
from datasets.loader.load_los_info import get_los_info
from pipelines import DlPipeline, MlPipeline


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
    checkpoint_path = f'logs/train/{config["dataset"]}/{config["task"]}/{config["model"]}-fold{config["fold"]}-seed{config["seed"]}/checkpoints/best.ckpt'
    if "time_aware" in config and config["time_aware"] == True:
        checkpoint_path = f'logs/train/{config["dataset"]}/{config["task"]}/{config["model"]}-fold{config["fold"]}-seed{config["seed"]}-ta/checkpoints/best.ckpt' # time-aware loss applied
    # train/val/test
    pipeline = DlPipeline(config)
    trainer = L.Trainer(accelerator="cpu", max_epochs=1, logger=False, num_sanity_val_steps=0)
    trainer.test(pipeline, dm, ckpt_path=checkpoint_path)
    # perf = pipeline.test_performance
    outs = pipeline.test_outputs
    return outs

if __name__ == "__main__":
    best_hparams = dl_best_hparams # [TO-SPECIFY]
    performance_table = {'dataset':[], 'task': [], 'model': [], 'fold': [], 'seed': [], 'accuracy': [], 'auroc': [], 'auprc': [], 'es': [], 'mae': [], 'mse': [], 'rmse': [], 'r2': [], 'osmae': []}
    for i in range(0, len(best_hparams)):
    # for i in range(0, 1):
        config = best_hparams[i]
        if config["task"] not in ["outcome", "multitask"]:
            print("Skipping...", config['task'])
            continue
        print(f"Testing... {i}/{len(best_hparams)}")
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
                outs = run_func(config)
                print(f"{config}", "Done")

                if "time_aware" in config and config["time_aware"] == True:
                    model_name = config['model']+"_ta"
                else:
                    model_name = config['model']

                save_name = f"{config['dataset']}-{config['task']}-{model_name}-fold{fold}-seed{seed}.pkl"
                pd.to_pickle(outs, f"logs/analysis/{save_name}")

