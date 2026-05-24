import os
import shutil
import tempfile
import unittest

import lightning as L
import pandas as pd
import torch

from datasets.loader.datamodule import EhrDataModule
from losses import TimeAwareLoss
from pipelines import DlPipeline, MlPipeline


def dl_config(model="MLP", task="outcome"):
    return {
        "demo_dim": 2,
        "lab_dim": 5,
        "hidden_dim": 8,
        "output_dim": 1,
        "learning_rate": 1e-2,
        "task": task,
        "los_info": {
            "los_mean": 10.0,
            "los_std": 2.0,
            "threshold": 5.0,
            "large_los": 20.0,
        },
        "model": model,
        "main_metric": "auprc",
    }


class LossRegressionTests(unittest.TestCase):
    def test_multitask_loss_is_registered_and_updates(self):
        pipeline = DlPipeline(dl_config(task="multitask"))
        self.assertIn("criterion.log_vars", dict(pipeline.named_parameters()))

        x = torch.randn(3, 4, 7)
        y = torch.zeros(3, 4, 2)
        y[:, :, 0] = torch.randint(0, 2, (3, 4)).float()
        y[:, :, 1] = torch.randn(3, 4)
        lens = torch.tensor([4, 2, 3])

        optimizer = pipeline.configure_optimizers()
        before = pipeline.criterion.log_vars.detach().clone()
        loss, *_ = pipeline._get_loss(x, y, lens)
        loss.backward()
        optimizer.step()

        self.assertFalse(torch.equal(before, pipeline.criterion.log_vars.detach()))

    def test_time_aware_loss_keeps_gradient_for_large_los(self):
        loss_fn = TimeAwareLoss(los_mean=0.0, los_std=1.0)
        pred = torch.tensor([0.1], requires_grad=True)
        loss = loss_fn(pred, torch.tensor([1.0]), torch.tensor([100.0]))
        loss.backward()

        self.assertGreater(loss.item(), 0)
        self.assertIsNotNone(pred.grad)
        self.assertNotEqual(pred.grad.abs().item(), 0)


class ModelRegressionTests(unittest.TestCase):
    def test_grasp_and_concare_support_batch_size_one(self):
        x = torch.randn(1, 1, 7)
        y = torch.tensor([[[1.0, 0.2]]])
        lens = torch.tensor([1])

        for model in ["GRASP", "ConCare"]:
            cfg = dl_config(model=model)
            cfg["cluster_num"] = 12
            pipeline = DlPipeline(cfg)
            loss, y_true, y_hat, embedding = pipeline._get_loss(x, y, lens)
            self.assertTrue(torch.isfinite(loss))
            self.assertEqual(tuple(y_hat.shape), (1,))

    def test_agent_action_layers_receive_gradients(self):
        pipeline = DlPipeline(dl_config(model="Agent"))
        x = torch.randn(3, 4, 7)
        y = torch.zeros(3, 4, 2)
        y[:, :, 0] = torch.randint(0, 2, (3, 4)).float()
        lens = torch.tensor([4, 3, 2])

        loss, *_ = pipeline._get_loss(x, y, lens)
        loss.backward()

        grads = {
            name: param.grad
            for name, param in pipeline.named_parameters()
            if "agent_encoder.agent" in name
        }
        self.assertGreater(len(grads), 0)
        self.assertTrue(all(grad is not None for grad in grads.values()))
        self.assertTrue(any(grad.abs().sum().item() > 0 for grad in grads.values()))


class PipelineRegressionTests(unittest.TestCase):
    def test_dl_test_outputs_flatten_patient_ids(self):
        pipeline = DlPipeline(dl_config())
        batch1 = (
            torch.randn(2, 3, 7),
            torch.zeros(2, 3, 2),
            torch.tensor([3, 2]),
            ("p1", "p2"),
        )
        batch2 = (
            torch.randn(1, 1, 7),
            torch.ones(1, 1, 2),
            torch.tensor([1]),
            ("p3",),
        )
        batch1[1][:, :, 0] = torch.randint(0, 2, (2, 3)).float()

        pipeline.test_step(batch1, 0)
        pipeline.test_step(batch2, 1)
        pipeline.on_test_epoch_end()

        self.assertEqual(pipeline.test_outputs["pids"], ["p1", "p2", "p3"])
        self.assertEqual(tuple(pipeline.test_outputs["embeddings"].shape), (3, 3, 8))

    def test_ml_pipeline_fits_and_evaluates_multiple_batches(self):
        dataset_name = "unit_test_tmp"
        try:
            with tempfile.TemporaryDirectory() as data_dir:
                x = [
                    [[0.1, 0.2], [0.2, 0.1]],
                    [[1.0, 1.1]],
                    [[0.9, 1.2]],
                    [[0.0, 0.3]],
                ]
                y = [
                    [[0.0, 0.1], [1.0, 0.2]],
                    [[1.0, 0.3]],
                    [[1.0, 0.4]],
                    [[0.0, 0.5]],
                ]
                pids = ["a", "b", "c", "d"]
                for split in ["train", "val", "test"]:
                    pd.to_pickle(x, os.path.join(data_dir, f"{split}_x.pkl"))
                    pd.to_pickle(y, os.path.join(data_dir, f"{split}_y.pkl"))
                    pd.to_pickle(pids, os.path.join(data_dir, f"{split}_pid.pkl"))

                config = {
                    "task": "outcome",
                    "los_info": {
                        "los_mean": 0.0,
                        "los_std": 1.0,
                        "threshold": 1.0,
                        "large_los": 2.0,
                    },
                    "model": "DT",
                    "main_metric": "auprc",
                    "dataset": dataset_name,
                    "fold": 0,
                    "seed": 0,
                    "max_depth": 2,
                    "n_estimators": 1,
                    "learning_rate": 0.1,
                    "batch_size": 2,
                }
                data_module = EhrDataModule(data_dir, batch_size=2)
                pipeline = MlPipeline(config)
                trainer = L.Trainer(
                    accelerator="cpu",
                    max_epochs=1,
                    logger=False,
                    num_sanity_val_steps=0,
                    enable_checkpointing=False,
                    enable_model_summary=False,
                    enable_progress_bar=False,
                )
                trainer.fit(pipeline, data_module)
                self.assertIn("auprc", pipeline.cur_best_performance)

                trainer.test(pipeline, data_module)
                self.assertEqual(pipeline.test_outputs["labels"].shape, (5, 2))
        finally:
            shutil.rmtree(os.path.join("logs", "train", dataset_name), ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
