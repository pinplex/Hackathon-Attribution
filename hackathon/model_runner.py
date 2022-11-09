
import os
from abc import abstractmethod

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
import torch
from torch import Tensor
import xarray as xr

from typing import Optional, Callable, Union, Any

from hackathon.data_pipeline import DataModule
from hackathon.base_model import BaseModel


class Ensemble(pl.LightningModule):
    """Create model ensemble from multiple pytorch models."""
    def __init__(self, model_type_list: list[type], checkpoint_path_list: list[str], aggregate_fun: str = 'mean') -> None:
        """Initializes Ensemble.

        Parameters
        ----------
        model_list: A list of models, each a BaseModel.
        aggregate_fun: the aggregation function to reduces ensemble runs, one of
            \'mean\' (default) or \'median\'.
        """
        super().__init__()

        # Directly passing the trained models causes an error (`ReferenceError: weakly-referenced
        # object no longer exists`), that's why we reload the models here.
        if len(model_type_list) != len(checkpoint_path_list):
            raise ValueError(
                'length of `model_type_list` must be equal to length of `checkpoint_paths`.'
            )

        models = []
        for model_type, checkpoint_path in zip(model_type_list, checkpoint_path_list):
            models.append(model_type.load_from_checkpoint(checkpoint_path))

        self.models = torch.nn.ModuleList(models)

        if aggregate_fun == 'mean':
            self.aggr_fn = self.mean_agg
        elif aggregate_fun == 'median':
            self.aggr_fn = self.median_agg
        else:
            raise ValueError(
                f'`aggregate_fun` must be \'mean\' or \'median\', is \'{aggregate_fun}\'.'
            )

    def forward(self, x: Tensor) -> Tensor:
        """Forward call all ensemble members and reduces the output."""
        y_hats = []
        for module in self.models:
            y_hat, _ = module.common_step(x)
            y_hats.append(y_hat)

        y_hats = torch.stack(y_hats, dim=0)

        y_hat = self.aggr_fn(y_hats)

        return y_hat

    def predict_step(
            self,
            batch: dict[str, Union[Tensor, dict[str, Any]]],
            batch_idx: int,
            dataloader_idx: int = 0) -> None:
        y_hat = self(batch)

        dataset = self.trainer.predict_dataloaders[dataloader_idx].dataset

        dataset.assign_predictions(pred=y_hat, data_sel=batch['data_sel'])

    @staticmethod
    def mean_agg(x: Tensor) -> Tensor:
        return torch.mean(x, dim=0)

    @staticmethod
    def median_agg(x: Tensor) -> Tensor:
        return torch.median(x, dim=0)

class ModelRunner(object):
    """ModelRunner implements the training scheme.
    """
    def __init__(
        self,
        log_dir: str,
        quickrun: bool = False,
        seed: Optional[int] = None) -> None:
        """Initialize BaseRunner.

        Parameters
        ----------
        log_dir: The root experiment directory to save logs and checkpoints to.
        quickrun: If set to true, less data is used for training and only two CV folds are run.
        seed: The random seed.        
        """

        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        self.quickrun = quickrun

        pl.seed_everything(seed)

    def data_setup(self, fold: int, **kwargs) -> pl.LightningDataModule:
        """Setup datamodule of class pl.LightningDataModule with a given cross validation fold.

        Parameters
        ----------
        fold: the fold id, a dummy parameter that must be in the range 0 to `num_folds`-1.
            Override and implement your own data splitting routine. If fold = -1, the test
            data is returned.
        kwargs: Are passed to the DataModule.

        Returns
        -------
        A datamodule of type pl.LightningDataModule.
        """

        if fold == -1:
            train_sel = {
                'location': [1],
                'time': slice('1850', '1855')
            }
            valid_sel = {
                'location': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                'time': slice('2005', '2014')
            }
        else:
            train_locs, valid_locs = self.get_cv_loc_split(fold)
            train_sel = {
                'location': [fold + 1] if self.quickrun else train_locs,
                'time': slice('1850', '1855') if self.quickrun else slice('1850', '2004')
            }
            valid_sel = {
                'location': [2 - fold] if self.quickrun else valid_locs,
                'time': slice('2005', '2010') if self.quickrun else slice('2005', '2014')
            }

        datamodule = DataModule(
            # You may keep these:
            data_path='./simple_gpp_model/data/CMIP6/predictor-variables_historical+GPP.nc',
            features=[f'var{i}' for i in range(1, 8)] + ['co2'],
            targets=['GPP'],
            # You may change these:
            train_subset=train_sel,
            valid_subset=valid_sel,
            test_subset=valid_sel,
            window_size=3,
            context_size=1,
            **kwargs)

        return datamodule

    def trainer_setup(
            self,
            version: str,
            patience: int = 15,
            max_epochs: int = -1,
            **kwargs) -> pl.Trainer:
        """Trainer setup.

        Subclass only to change default behavior.

        Parameters
        ----------
        version: The run version (e.g., 'fold_00' if you run a cross-validation, or 'final').
            Logs will be saved to `log_dir/version`.
        patience: The patiance, training will be stopped after n iterations
            with no improvement.
        max_epochs: Maximum number of epochs to run, default is -1 (infinite).

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
            max_epochs=max_epochs,
            **kwargs
        )

        return trainer

    def train(
            self,
            model_fn: Callable[[dict[str, Tensor]], BaseModel],
            fold: Optional[int] = None,
            **kwargs) -> tuple[pl.Trainer, BaseModel]:
        """Runs training.

        Note:
        This is just a blueprint, you may implement your own training routine here, e.g.,
        cross validation. At the end, one single model must be returned.

        Parameters
        ----------
        model_fn: a function that returns an initialized model and takes norm_stats
            as argument.
        fold: optional fold ID, an integer in the range [0, 9]. If passed, only the given fold
            will be used for training. Else, the 10 folds are iterated (2 if `quickrun=True`)
        kwargs: passed to pl.Trainer.

        Returns
        -------
        - The pl.trainer.
        - The trained model.
        """

        model_types = []
        checkpoint_paths = []

        if fold is None:
            iter_folds = range(2) if self.quickrun else range(10)
        else:
            iter_folds = [fold]
        for fold in iter_folds:
            version = f'fold_{fold:02d}'

            datamodule = self.data_setup(
                fold=fold,
                batch_size=10,
                num_workers=0
            )

            model = model_fn(norm_stats=datamodule.norm_stats)

            trainer = self.trainer_setup(
                version=version,
                **kwargs
            )

            # Fit model with training data (and valid data for early stopping.)
            trainer.fit(model, datamodule=datamodule)

            # Load best model.
            checkpoint_path = self.load_best_model(trainer=trainer, model=model)

            # Final predictions on the test set.
            self.predict(model=model, trainer=trainer, datamodule=datamodule, version=version)

            model_types.append(type(model))
            checkpoint_paths.append(checkpoint_path)

        ensemble = Ensemble(model_type_list=model_types, checkpoint_path_list=checkpoint_paths)
        self.save_model(model=ensemble, version='final')

        return trainer, ensemble

    def predict(
            self,
            trainer: pl.Trainer,
            model: BaseModel,
            datamodule: DataModule,
            version: str) -> xr.Dataset:
        """Make predictions with the `datamodule.predict_dataloader`.

        !!!DOES NOT LOAD BEST MODEL!!!
        ------------------------------
        Use `.load_best_model(...)` to restore best parameters and pass the model to `.predict(...)`.

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
        trainer.predict(model=model, dataloaders=dataloader)
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

    @staticmethod
    def load_best_model(trainer: pl.Trainer, model: BaseModel) -> str:
        """Load best model (inplace).
        
        Parameters
        ----------
        trainer: a trainer on which '.fit(...)' has been run before.
        model: a model.

        Returns
        -------
        The best model checkpoint path.
        """

        best_model = trainer.checkpoint_callback.best_model_path
        model.load_from_checkpoint(best_model)

        return best_model

    @staticmethod
    def get_cv_loc_split(fold: int) -> tuple[list[int], list[int]]:
        """Split clusters of sites into training and validation set.

        Parameters
        ----------
        fold: the fold ID, a value between (including) 0 and 9.

        Returns
        -------
        A tuple, the training and the validation locations.
        """

        locs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        if (fold < 10) and (fold >= 0):
            valid_loc = [locs.pop(fold)]
            train_loc = locs
        else:
            raise ValueError(
                f'argument `fold` must be a value between (including) 0 and 9, is {fold}.'
            )

        return train_loc, valid_loc
