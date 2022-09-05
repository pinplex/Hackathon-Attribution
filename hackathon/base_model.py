import torch
import pytorch_lightning as pl

from typing import Any, Union, Optional
from torch import Tensor


class BaseModel(pl.LightningModule):
    def __init__(
            self,
            custom_model: torch.nn.Module,
            learning_rate: float,
            weight_decay: float):
        super().__init__()

        self.model = custom_model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.loss_fn = torch.nn.MSELoss()

        self.save_hyperparameters(ignore=['custom_model'])

    def common_step(self, batch: dict[str, Union[Tensor, dict[str, Any]]],
                    step_name: Optional[str] = None) -> tuple[Tensor, Tensor]:
        x = batch['x']
        y = batch['y']
        data_sel = batch['data_sel']

        y_hat = self.model(x)

        pred_len = data_sel['pred_len'].min()
        loss = self.loss_fn(y_hat[:, -pred_len:, :], y[:, -pred_len:, :])

        if step_name:
            self.log(f'{step_name}_loss', loss, on_step=step_name == 'train', on_epoch=True, batch_size=x.shape[0])
        return y_hat, loss

    def training_step(
            self,
            batch: dict[str, Union[Tensor, dict[str, Any]]],
            batch_idx: int) -> Tensor:
        _, loss = self.common_step(batch, 'train')

        return loss

    def validation_step(
            self,
            batch: dict[str, Union[Tensor, dict[str, Any]]],
            batch_idx: int) -> None:
        self.common_step(batch, 'val')

    def test_step(
            self,
            batch: dict[str, Union[Tensor, dict[str, Any]]],
            batch_idx: int) -> None:
        _, loss = self.common_step(batch, 'test')

    def predict_step(
            self,
            batch: dict[str, Union[Tensor, dict[str, Any]]],
            batch_idx: int,
            dataloader_idx: int = 0) -> None:
        y_hat, _ = self.common_step(batch)

        dataset = self.trainer.predict_dataloaders[dataloader_idx].dataset

        dataset.assign_predictions(pred=y_hat, data_sel=batch['data_sel'])

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        return optimizer
