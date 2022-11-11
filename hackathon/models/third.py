import pytorch_lightning as pl
import torch
from torch import Tensor

from hackathon import BaseModel, BaseRunner, DataModule

class VeryRandomBaseline(BaseModel):
    """
    A very simple (one hidden layer) MLP to predict ahead

    :param num_features: The number of input features.
    :param num_targets: The number of target features.
    :param receptive_field: The number of time steps used in each prediction.
    :param hidden_size: The hidden size of the MLP.

    :type num_features: Integer
    :type num_targets: Integer
    :type receptive_field: Integer
    :type hidden_size: Integer
    """

    def __init__(
        self,
        num_features: int,
        num_features_1: int,
        num_features_2: int,
        num_features_3: int,
        num_features_4: int,
        num_targets: int,
        **kwargs,
    ):
        super(VeryRandomBaseline, self).__init__(**kwargs)

        self.num_targets = num_targets

        self.layer1 = torch.nn.Conv1d(
            in_channels = num_features,
            out_channels = num_features_1,
            kernel_size = 14,
            stride = 13,
            padding = 'valid'
        )
        
        self.layer2 = torch.nn.Conv1d(
            in_channels = num_features_1,
            out_channels = num_features_2,
            kernel_size = 7,
            stride = 7,
            padding = 'valid'
        )
        
        self.layer3 = torch.nn.Conv1d(
            in_channels = num_features_2,
            out_channels = num_features_3,
            kernel_size = 2,
            stride = 2,
            padding = 'valid'
        )
        
        self.layer4 = torch.nn.Conv1d(
            in_channels = num_features_3,
            out_channels = num_features_4,
            kernel_size = 2,
            stride = 2,
            padding = 'valid'
        )
        
        self.layer5 = torch.nn.Conv1d(
            in_channels = num_features_4,
            out_channels = num_features,
            kernel_size = 2,
            stride = 2,
            padding = 'valid'
        )

        self.relu = torch.nn.ReLU()

    def forward(self, input_tensor: Tensor) -> Tensor:
        """
        The forward function of the simple MLP

        :param input_tensor: the input tensor
        :type input_tensor: Tensor
        """

        x = torch.transpose(input_tensor, -1, -2)
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.layer3(x)
        x = self.relu(x)
        x = self.layer4(x)
        x = self.relu(x)
        x = self.layer5(x)
        input_tensor = input_tensor.unsqueeze(-2)

        z = torch.matmul(input_tensor, x) 
        z = z.squeeze(-1)
        return z


class VeryRandomRunner(BaseRunner):
    """Implements a linear model with training routine."""

    def __init__(self, **kwargs):
        super(SimpleMLPRunner, self).__init__(**kwargs)

    def data_setup(self, fold: int, **kwargs) -> pl.LightningDataModule:
        """
        Setup datamodule of class pl.LightningDataModule with a given cross 
        validation fold.

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
            raise ValueError(f"`fold` must be 0 but is {fold}.")

        datamodule = DataModule(
            # You may keep these:
            data_path="./simple_gpp_model/data/CMIP6/predictor-variables_historical+GPP.nc",
            features=[f"var{i}" for i in range(1, 8)] + ["co2"],
            targets=["GPP"],
            # You may change these:
            train_subset={
                "location": train_subset_locations,
                "time": slice("1850", "2004"),
            },
            valid_subset={
                "location": valid_subset_locations,
                "time": slice("2005", "2014"),
            },
            test_subset={
                "location": test_subset_locations,
                "time": slice("2005", "2014"),
            },
            window_size=1,
            context_size=1,
            **kwargs,
        )

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
        model = VeryRandomBaseline(
            num_features=num_features,
            num_targets=num_targets,
            num_features_1 = 16,
            num_features_2 = 32,
            num_features_3 = 64,
            num_features_4 = 128, 
            learning_rate=3e-4,
            weight_decay=1e-5,
        )

        return model

    def train(
        self,
    ) -> tuple[pl.Trainer, pl.LightningDataModule, pl.LightningModule]:
        """Runs training.

        Note:
        This is just a blueprint, you may implement your own training routine here, e.g.,
        cross validation. At the end, one single model must be returned.

        Returns
        -------
        A trained model.
        """

        fold = 0
        version = f"fold_{fold:02d}"

        datamodule = self.data_setup(fold=fold, batch_size=4, num_workers=0)

        model = self.model_setup(
            num_features=datamodule.num_features,
            num_targets=datamodule.num_targets,
        )

        trainer = self.trainer_setup(version=version, patience = 400)

        # Fit model with training data (and valid data for early stopping.)
        trainer.fit(model, datamodule=datamodule)

        trainer.test(model, datamodule = datamodule)
        # Final predictions on the test set.
        self.predict(trainer=trainer, datamodule=datamodule, version=version)

        return trainer, datamodule, model


if __name__ == '__main__':
    model = VeryRandomBaseline(8, 16, 32, 64, 128, 1)
    input_tensor = torch.randn(1, 730, 8)
    print(model(input_tensor).shape)
