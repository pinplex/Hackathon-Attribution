
import torch
from torch import Tensor

from hackathon import BaseModel


class Linear(BaseModel):
    def __init__(self, num_features: int, num_targets: int, **kwargs) -> None:
        super(Linear, self).__init__(**kwargs)

        self.linear = torch.nn.Linear(num_features, num_targets)
        self.relu = torch.nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        out = self.relu(self.linear(x))
        return out


def model_setup():
    """Create a model as subclass of hackathon.base_model.BaseModel.

    Returns
    -------
    A model.
    """
    model = Linear(
        num_features=8,
        num_targets=1,
        learning_rate=0.01,
        weight_decay=0.0,
    )

    return model
