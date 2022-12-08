import torch
import pytorch_lightning as pl
import xarray as xr

from typing import Any, Union, Optional, Tuple
from torch import Tensor


class BaseModel(pl.LightningModule):
    """Implements a base model.
    
    Meant to be subclassed.

    Note that the prediction step (i.e., `my_trainer.predict(...)`) automatically
    assigns the predictions to the datamodule's dataset with name '<target>_hat'.
    """

    def __init__(
            self,
            learning_rate: float,
            weight_decay: float,
            norm_stats: dict[str, Tensor]):
        """Initialize a BeseModel.

        Parameters
        ----------
        learning_rate: The learning rate.
        weight_decat: The weight decay.
        norm_stats: Feature normalization stats of format {'mean': Tensor, 'std': Tensor}, with
            both tensors of shape (num_features,). The order of the features must be the same
            as in `batch['x']` (see `.common_step(...)2`).
        """
        super().__init__()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.norm_mean: Tensor
        self.register_buffer(name='norm_mean', tensor=norm_stats['mean'])
        self.norm_std: Tensor
        self.register_buffer(name='norm_std', tensor=norm_stats['std'])

        self.loss_fn = torch.nn.MSELoss()

        self.save_hyperparameters()

    def normalize(self, x: Tensor) -> Tensor:
        """Normalize a numpy array.

        Parameters
        ----------
        x: A tensor of features to be normalized.

        Returns
        -------
        The normalized Tensor.
        """

        return (x - self.norm_mean) / self.norm_std

    def common_step(self, batch: dict[str, Union[Tensor, dict[str, Any]]],
                    step_name: Optional[str] = None,
                    return_x_norm: bool = False) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]:
        # [batch, sequence, features]
        x = self.normalize(batch['x'])
        # [batch, sequence, targets]
        y = batch['y']

        data_sel = batch['data_sel']

        y_hat = self(x)

        pred_len = data_sel['pred_len'].min()
        loss = self.loss_fn(y_hat[:, -pred_len:, :], y[:, -pred_len:, :])

        if step_name:
            self.log(f'{step_name}_loss', loss, on_step=step_name == 'train', on_epoch=True, batch_size=x.shape[0])

        if return_x_norm:
            return y_hat, loss, x

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
