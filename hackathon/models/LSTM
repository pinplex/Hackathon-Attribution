import torch
from torch import Tensor

from base_model import BaseModel


class LSTM(BaseModel):
    """
    A relatively deep LSTM

    :param num_features: The number of input features.
    :param num_targets: The number of target features.
    :param num_layers: The number of hidden layers.
    :param num_hidden: The hidden node size.

    :type num_features: Integer
    :type num_targets: Integer
    :type num_layers: Integer
    :type num_hidden: Integer
    """


    def __init__(self,num_features: int, num_targets: int,num_hidden: int, num_layers:int, **kwargs) -> None:
        super(LSTM, self).__init__(**kwargs)

        self.lstm = torch.nn.LSTM(input_size=num_features, hidden_size=num_hidden, num_layers=num_layers, batch_first=True)

        self.linear =torch.nn.Linear(in_features=num_hidden, out_features=num_targets)


    def forward(self, x: Tensor) -> Tensor:
        out, _ = self.lstm(x)

        out = self.linear(out)

        return out


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
    model = LSTM(
            num_features=8,
            num_targets=1,
            num_hidden= 128,
            num_layers=2,
            learning_rate=0.01,
            weight_decay=0,
            norm_stats=norm_stats
    )

    return model

