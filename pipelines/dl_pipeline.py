import os

import lightning as L
import torch
import torch.nn as nn

import models
from datasets.loader.unpad import unpad_y
from losses import get_simple_loss
from metrics import get_all_metrics
from models.utils import generate_mask, get_last_visit


class DlPipeline(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        self.demo_dim = config["demo_dim"]
        self.lab_dim = config["lab_dim"]
        self.input_dim = config["input_dim"]
        assert self.input_dim == self.demo_dim + self.lab_dim
        self.hidden_dim = config["hidden_dim"]
        self.output_dim = config["output_dim"]
        self.learning_rate = config["learning_rate"]
        self.task = config["task"]
        self.los_info = config["los_info"]
        self.model_name = config["model_name"]
        self.main_metric = config["main_metric"]

        model_class = getattr(models, self.model_name)
        self.ehr_encoder = model_class(**config)
        if self.task == "outcome":
            self.head = nn.Sequential(nn.Linear(self.hidden_dim, self.output_dim), nn.Dropout(0.0), nn.Sigmoid())
        elif self.task == "los":
            self.head = nn.Sequential(nn.Linear(self.hidden_dim, self.output_dim), nn.Dropout(0.0))
        elif self.task == "multitask":
            self.head = models.heads.MultitaskHead(self.hidden_dim, self.output_dim, drop=0.0)

        self.validation_step_outputs = []

    def forward(self, x, lens):
        if self.model_name == "ConCare":
            x_demo, x_lab, mask = x[:, 0, :self.demo_dim], x[:, :, self.demo_dim:], generate_mask(lens)
            embedding, decov_loss = self.ehr_encoder(x_lab, x_demo, mask)
            y_hat = self.head(embedding)
            return y_hat, embedding, decov_loss
        elif self.model_name in ["GRASP", "Agent"]:
            x_demo, x_lab, mask = x[:, 0, :self.demo_dim], x[:, :, self.demo_dim:], generate_mask(lens)
            embedding = self.ehr_encoder(x_lab, x_demo, mask)
            y_hat = self.head(embedding)
            return y_hat, embedding
        elif self.model_name in ["AdaCare", "RETAIN", "TCN"]:
            mask = generate_mask(lens)
            embedding = self.ehr_encoder(x, mask)
            y_hat = self.head(embedding)
            return y_hat, embedding
        elif self.model_name in ["GRU", "LSTM", "RNN", "MLP"]:
            embedding = self.ehr_encoder(x)
            y_hat = self.head(embedding)
            return y_hat, embedding

    def _get_loss(self, x, y, lens):
        if self.model_name == "ConCare":
            y_hat, embedding, decov_loss = self(x, lens)
            y_hat, y = unpad_y(y_hat, y, lens)
            loss = get_simple_loss(y_hat, y, self.task)
            loss += decov_loss
        else:
            y_hat, embedding = self(x, lens)
            y_hat, y = unpad_y(y_hat, y, lens)
            loss = get_simple_loss(y_hat, y, self.task)
        return loss, y, y_hat
    def training_step(self, batch, batch_idx):
        x, y, lens, pid = batch
        loss, y, y_hat = self._get_loss(x, y, lens)
        self.log("train_loss", loss)
        return loss
    def validation_step(self, batch, batch_idx):
        x, y, lens, pid = batch
        loss, y, y_hat = self._get_loss(x, y, lens)
        self.log("val_loss", loss)
        outs = {'y_pred': y_hat, 'y_true': y, 'val_loss': loss}
        self.validation_step_outputs.append(outs)
        return loss
    def on_validation_epoch_end(self):
        y_pred = torch.cat([x['y_pred'] for x in self.validation_step_outputs])
        y_true = torch.cat([x['y_true'] for x in self.validation_step_outputs])
        loss = torch.stack([x['val_loss'] for x in self.validation_step_outputs]).mean()
        self.log("val_loss_epoch", loss)
        metrics = get_all_metrics(y_pred, y_true, self.task, self.los_info)
        for k, v in metrics.items(): self.log(k, v)
        main_metric = metrics[self.main_metric]
        self.validation_step_outputs.clear()
        return main_metric
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer