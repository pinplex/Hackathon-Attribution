
import os
from abc import abstractmethod

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
import torch
from torch import Tensor
import xarray as xr

from typing import Optional

from hackathon.data_pipeline import DataModule
from hackathon.base_model import BaseModel


class Ensemble(pl.LightningModule):
    """Create model ensemble from multiple pytorch models."""
    def __init__(self, model_list: list[pl.LightningModule], aggregate_fun: str = 'mean') -> None:
        """Initializes Ensemble.

        Parameters
        ----------
        model_list: A list of models, each a pl.LightningModule.
        aggregate_fun: the aggregation function to reduces ensemble runs, one of
            \'mean\' (default) or \'median\'.
        """
        super().__init__()
        self.models = torch.nn.ModuleList(model_list)
        if aggregate_fun == 'mean':
            self.aggr_fn = lambda x: torch.mean(x, dim=0)
        elif aggregate_fun == 'median':
            self.aggr_fn = lambda x: torch.quantile(x, q=0.5, dim=0)
        else:
            raise ValueError(
                f'`aggregate_fun` must be \'mean\' or \'median\', is \'{aggregate_fun}\'.'
            )

    def forward(self, x: Tensor) -> Tensor:
        """Forward call, all ensemble members and reduces the output."""
        y_hats = []
        for module in self.models:
            y_hats.append(module(x))

        y_hats = torch.stack(y_hats, dim=0)
        y_hat = self.aggr_fn(y_hats)

        return y_hat


class BaseRunner(object):
    """BaseRunner implements the training scheme.
    
    * Meant to be subclassed.
    * In subclass, override
        * `MySubclass.data_setup` to define the dataloader,
        * `MySubclass.model_setup` to define the model, and
        * `MySubclass.train` to define the training routine.

    Example:
        >>> class MyRunner(BaseRunner):
                def data_setup(self) -> DataModule:
                    train_selection, valid_selection = f(fold)
                    datamodule = DataModule(...)
                    return datamodule, model
        >>> runner = MyRunner()
        >>> runner.train()

    """
    def __init__(
        self,
        log_dir: str,
        seed: Optional[int] = None) -> None:
        """Initialize BaseRunner.

        Parameters
        ----------
        log_dir: The root experiment directory to save logs and checkpoints to.
        seed: The random seed.        
        """

        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        pl.seed_everything(seed)

    @abstractmethod
    def data_setup(self, fold: int, **kwargs) -> DataModule:
        """Setup datamodule of class DataModule with a given cross validation fold.
        
        Must be overridden in subclass.

        Parameters
        ----------
        fold (int): the fold id, a dummy parameter that must be 0.
            Override and implement your own data splitting routine.

        Returns
        -------
        A datamodule of type DataModule.
        """

        pass

    @abstractmethod
    def model_setup(self) -> pl.LightningModule:
        """Create a model whihc must be a subclass of hackathon.base_model.BaseModel.
        
        Must be overridden in subclass.

        Returns
        -------
        A model.
        """
        pass

    def trainer_setup(
            self,
            version: str,
            patience: int = 20,
            **kwargs) -> pl.Trainer:
        """Trainer setup.

        Subclass only to change default behavior.

        Parameters
        ----------
        version: The run version (e.g., 'fold_00' if you run a cross-validation, or 'final').
            Logs will be saved to `log_dir/version`.
        patience: The patiance, training will be stopped after n iterations
            with no improvement.

        Returns
        -------
        A pl.Trainer.
        """
        logger = TensorBoardLogger(save_dir=self.log_dir, name='', version=version)
        early_stopper = EarlyStopping(patience=patience, monitor='val_loss', mode='min', verbose=True)
        checkpointer = ModelCheckpoint(save_top_k=1, monitor='val_loss')

        trainer = pl.Trainer(
            logger=logger,
            default_root_dir=self.log_dir,
            callbacks=[
                early_stopper,
                checkpointer
            ],
            log_every_n_steps=1,
            max_epochs=-1,
            **kwargs
        )

        return trainer

    @abstractmethod
    def train(self) -> tuple[pl.Trainer, DataModule, pl.LightningModule]:
        """Runs training.

        Must be overridden in subclass.

        Note
        ----

        If multiple models are trained (e.g., cross validation) within this method:
        * use `trainer_setup(version=...)` to log multiple models. Each one will have its own subdirectory (`log_dir/version`).
        * you can pass a list of trained models to the `hackathon.base_runner.Ensemble` class and return it as a single model.

        Returns
        -------
        - the trainer: pl.Trainer
        - the data module: DataModule
        - the trained model: pl.LightningModule
        """

        pass

    def predict(
            self,
            trainer: pl.Trainer,
            model: BaseModel,
            datamodule: DataModule,
            version: str) -> xr.Dataset:
        """Make predictions with the `datamodule.predict_dataloader`.
        
        Parameters
        ----------
        trainer: a trainer on which '.fit(...)' has been run before.
        model: a trained model.
        datamodule: the datamodule.
        version: The run version (e.g., 'fold_00' if you run a cross-validation, or 'final').
            Predictions will be saved to `log_dir/version/predictions.nc`. Make sure to match with
            `MyRunner.trainer_setup(version='same_as_here')` to have predictions along with the logs.
        """

        dataloader = datamodule.predict_dataloader()
        trainer.predict(model=model, ckpt_path='best', dataloaders=dataloader)
        save_path = os.path.join(self.log_dir, version, 'predictions.nc')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        dataloader.dataset.ds.to_netcdf(save_path)

    def save_model(self, model: BaseModel, version: str) -> None:
        """Save a pytorch lightning model.
        
        Parameters
        ----------
        model: the model.
        version: The run version (e.g., 'fold_00' if you run a cross-validation, or 'final').
            The model will be saved to `log_dir/version/final.chkpt`. Make sure to match with
            `MyRunner.trainer_setup(version='same_as_here')` to have the model along with the logs.
        """

        save_path = os.path.join(self.log_dir, version, 'final.ckpt')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model, save_path)
