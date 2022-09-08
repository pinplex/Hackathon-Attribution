
import torch
from torch import Tensor

from hackathon import BaseModel


class Linear(BaseModel):
    def __init__(self, num_features: int, num_targets: int, **kwargs) -> None:
        super(Linear, self).__init__(**kwargs)

        self.linear = torch.nn.Linear(num_features, num_targets)
        self.softplus = torch.nn.Softplus()

    def forward(self, x: Tensor) -> Tensor:
        out = self.softplus(self.linear(x))
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
    model = Linear(
        num_features=8,
        num_targets=1,
        learning_rate=0.01,
        weight_decay=0.0,
        norm_stats=norm_stats
    )

    return model
