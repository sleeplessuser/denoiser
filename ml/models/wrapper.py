import torch
import lightning.pytorch as pl
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility
from torchmetrics.audio.snr import SignalNoiseRatio

from .utils import get_model

class LitWrapper(pl.LightningModule):
    def __init__(self, config, loss_fn=None) -> None:
        super().__init__()
        self.save_hyperparameters("config")

        self.config = config
        self.loss_fn = loss_fn
        self.model = get_model(config['model']['name'])

        self.metrics = {
            "pesq": PerceptualEvaluationSpeechQuality(self.config["model"]["sr"], "wb"),
            "stoi": ShortTimeObjectiveIntelligibility(self.config["model"]["sr"]),
            "snr": SignalNoiseRatio(),
        }

    def on_test_start(self) -> None:
        for metric in self.metrics:
            self.metrics[metric].to(self.device)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        self.log("val/loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        for metric_name, metric_fn in self.metrics.items():
            self.log(metric_name, metric_fn(y_hat, y).mean())

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self.model(data)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.config["optim"]["lr"],
            weight_decay=self.config["optim"]["wd"],
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            "min",
            patience=self.config["train"]["patience"],
            threshold=self.config["train"]["delta"],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val/loss",
        }
