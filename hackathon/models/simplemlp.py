import torch
from torch import Tensor

from hackathon import BaseModel


class SimpleMLP(BaseModel):
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
        num_targets: int,
        receptive_field: int,
        **kwargs,
    ):
        super(SimpleMLP, self).__init__(**kwargs)

        self.context_pad = torch.nn.ConstantPad1d((receptive_field - 1, 0), 0.0)

        self.mod1layer1 = torch.nn.Conv1d(
            in_channels=num_features,
            out_channels=32,
            kernel_size=receptive_field
        )
        self.relu = torch.nn.ReLU()

        self.mod1layer2 = torch.nn.Conv1d(
            in_channels=32,
            out_channels=16,
            kernel_size=1
        )

        self.mod1layer3 = torch.nn.Conv1d(
            in_channels=16,
            out_channels=8,
            kernel_size=1
        )

        self.mod1layer4 = torch.nn.Conv1d(
            in_channels=8,
            out_channels=4,
            kernel_size=1
        )

        self.mod1layer5 = torch.nn.Conv1d(
            in_channels=4,
            out_channels=num_targets,
            kernel_size=1
        )

    def forward(self, in_features: Tensor) -> Tensor:
        """
        The forward function of the simple MLP

        :param x: the input tensor
        :type x: Tensor
        """

        in_features = torch.transpose(in_features, -1, -2)
        in_features_pad = self.context_pad(in_features)
        x = self.mod1layer1(in_features_pad)
        x = self.relu(x)
        x = self.mod1layer2(x)
        x = self.relu(x)
        x = self.mod1layer3(x)
        x = self.relu(x)
        x = self.mod1layer4(x)
        x = self.relu(x)
        x = self.mod1layer5(x)

        x = torch.transpose(x, -1, -2)
        return x


def model_setup(norm_stats: dict[str, Tensor]) -> BaseModel:
    """Create a model as subclass of hackathon.base_model.BaseModel.

    Parameters
    ----------
    norm_stats: Feature normalization stats with signature {'mean': Tensor, 'std': Tensor},
        both tensors with shape (num_features,).

    Returns
    -------
    A model.
    """
    model = SimpleMLP(
        num_features=8,
        num_targets=1,
        receptive_field=30,
        learning_rate=3e-4,
        weight_decay=1e-5,
        norm_stats=norm_stats
    )

    return model
