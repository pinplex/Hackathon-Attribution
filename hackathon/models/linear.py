
import pytorch_lightning as pl
import torch
from torch import Tensor

from hackathon import BaseModel, BaseRunner, DataModule


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

        # `fold` only selects locations, you may also create splits in time.
        if fold == 0:
            train_subset_locations = [1, 2]
            valid_subset_locations = [3]
            test_subset_locations = [4]
        else:
            raise ValueError(
                f'`fold` must be 0 but is {fold}.'
            )

        datamodule = DataModule(
            # You may keep these:
            data_path='./simple_gpp_model/data/CMIP6/predictor-variables_historical+GPP.nc',
            features=[f'var{i}' for i in range(1, 8)] + ['co2'],
            targets=['GPP'],
            # You may change these:
            train_subset={
                'location': train_subset_locations,
                'time': slice('1850', '1855')
            },
            valid_subset={
                'location': valid_subset_locations,
                'time': slice('1855', '1860')
            },
            test_subset={
                'location': test_subset_locations,
                'time': slice('1855', '1860')
            },
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
        self.predict(trainer=trainer, datamodule=datamodule, version=version)

        return trainer, datamodule, model
