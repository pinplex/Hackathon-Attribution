#%%
import pytorch_lightning as pl
from torch import nn
from torch import Tensor

from hackathon import BaseModel

class Conv1D(BaseModel):
    def __init__(self, num_features: int, num_targets: int, kernel_size: int, **kwargs) -> None:
        super(Conv1D, self).__init__(**kwargs)

        # padding='same' makes in sequence = out_sequence. Cutting the temporal 'context' is then
        # done by the framework, should work out of the box.
        self.conv1d = nn.Conv1d(
            in_channels=num_features,
            out_channels=num_targets,
            kernel_size=kernel_size,
            padding=0)

        self.softplus = nn.Softplus()

    def forward(self, x: Tensor) -> Tensor:
        # Convolution expects features (aka channels) at second position
        # [batch, sequence, features] -> [batch, features, sequence]
        out = x.permute(0, 2, 1)
        #print(out.shape)

        # Apply convolution over sequence.
        # [batch, features, sequence] -> [batch, outputs, sequence]
        out = self.conv1d(out)
        #print(out.shape)

        # Move features back to last position and apply softplus (to force x>0)
        # [batch, outputs, sequence] -> [batch, sequence, outputs]
        out = self.softplus(out).permute(0, 2, 1)
        #print(out.shape)

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
    model = Conv1D(
        num_features=8,
        num_targets=1,
        kernel_size=30,
        learning_rate=0.01,
        weight_decay=0.0,
        norm_stats=norm_stats
    )

    return model
