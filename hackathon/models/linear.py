
import pytorch_lightning as pl
import torch
from torch import Tensor

from hackathon import BaseModel, BaseRunner, DataModule, Ensemble


class Linear(BaseModel):
    def __init__(self, num_features: int, num_targets: int, **kwargs) -> None:
        super(Linear, self).__init__(**kwargs)

        self.linear = torch.nn.Linear(num_features, num_targets)
        self.relu = torch.nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        out = self.relu(self.linear(x))
        return out


class LinearRunner(BaseRunner):
    """Implements a linear model with training routine."""
    def data_setup(self, fold: int, **kwargs) -> pl.LightningDataModule:
        """Setup datamodule of class pl.LightningDataModule with a given cross validation fold.

        Parameters
        ----------
        fold: the fold id, a dummy parameter that must be 0.
            Override and implement your own data splitting routine.
        kwargs: Are passed to the DataModule.

        Returns
        -------
        A datamodule of type pl.LightningDataModule.
        """

        train_locs, valid_locs = self.get_cv_loc_split(fold)
        train_sel = {
            'location': train_locs,
            'time': slice('1850', '2004')
        }
        valid_sel = {
            'location': valid_locs,
            'time': slice('2005', '2014')
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
            window_size=1,
            context_size=1,
            **kwargs)

        return datamodule

    def model_setup(self, num_features: int, num_targets: int):
        """Create a model as subclass of hackathon.base_model.BaseModel.
        
        Parameters
        ----------
        num_features: The number of features.
        num_targets: The number of targets.

        Returns
        -------
        A model.
        """
        model = Linear(
            num_features=num_features,
            num_targets=num_targets,
            learning_rate=0.01,
            weight_decay=0.0,
        )

        return model

    def train(self) -> tuple[pl.Trainer, pl.LightningDataModule, pl.LightningModule]:
        """Runs training.

        Note:
        This is just a blueprint, you may implement your own training routine here, e.g.,
        cross validation. At the end, one single model must be returned.

        Returns
        -------
        A trained model.
        """

        models = []

        fold = 0
        for fold in range(10):
            version = f'fold_{fold:02d}'

            datamodule = self.data_setup(
                fold=fold,
                batch_size=4,
                num_workers=0
            )

            model = self.model_setup(
                num_features=datamodule.num_features,
                num_targets=datamodule.num_targets
            )

            trainer = self.trainer_setup(version=version)

            # Fit model with training data (and valid data for early stopping.)
            trainer.fit(model, datamodule=datamodule)

            # Load best model.
            self.load_best_model(trainer=trainer, model=model)

            # Final predictions on the prediction set.
            self.predict(trainer=trainer, model=model, datamodule=datamodule, version=version)

        ensemble = Ensemble(models)

        return trainer, datamodule, ensemble
