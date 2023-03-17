"""
Function to reproduce the ground truth model as an nn.Module

author = Christian Reimers (creimers@bgc-jena.mpg.de)
credit = Max-Planck-Institute for Biogeochemistry
"""

import torch
from torch import Tensor

from hackathon import BaseModel


class GTmodel(BaseModel):
    """
    The ground truth model as a nn.Module
    """

    def __init__(self, **kwargs):
        super(GTmodel, self).__init__(**kwargs)

        self.useless = torch.nn.Parameter(torch.zeros(1))
        self.relu = torch.nn.ReLU()

    def normalize(self, x: Tensor) -> Tensor:
        """
        replacing the inherited method by the identity
        """

        return x

    def forward(self, x: Tensor) -> Tensor:
        """
        The forward function of the simple MLP

        :param x: the input tensor
        :type x: Tensor
        """

        vpd, tmin, tp, ssrd, sfcwind, e, fpar, co2 = torch.tensor_split(x, 8, dim=-1)

        # Calculate the bucket soil water content
        bswc = 200 + torch.cumsum(tp, dim=1) + torch.cumsum(e, dim=1)
        for i in range(1, bswc.shape[1]):
            if bswc[0, i, 0] > 200:
                bswc[0, i - 1:, 0] = bswc[0, i - 1:, 0] - bswc[0, i, 0] + 200

        bswc = torch.clip(bswc, 25, 200)

        # set epsilon max
        epsilon_max = 0.001051

        # calculate apar
        apar = ssrd * 0.45 * fpar

        # calculate the effect of co2 fertilization
        co2_0 = co2[:, 0:1, :]
        f_co2 = (co2 - co2_0) / co2 + 1

        # calculate the effect of temperature
        m = 1 / (9.5 + 7)
        t = 1 - m * 9.5
        tmin_scalar = m * tmin + t
        tmin_scalar = torch.clip(tmin_scalar, 0, 1)

        # calculate the effect of vpd
        m_vpd = -1 / (2400 - 650)
        t_vpd = 1 - m_vpd * 650
        vpd_scalar = m_vpd * vpd + t_vpd
        vpd_scalar = torch.clip(vpd_scalar, 0, 1)

        # calculate the effect of soil-water effect
        rew = (bswc - 25) / (100 - 25)
        swc_scalar = torch.pow(rew, 0.383)
        swc_scalar = torch.clip(swc_scalar, 0, 1)

        # Combine everything into GPP
        gpp = epsilon_max * apar * f_co2 * tmin_scalar * vpd_scalar * swc_scalar * 1000 + self.useless - self.useless

        # Add the expected values of the noise
        pi = torch.acos(torch.zeros(1)) * 2
        noise_mean = 1 / torch.sqrt(2 * pi)
        gpp = gpp + noise_mean

        return gpp


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
    model = GTmodel(
        learning_rate=3e-4,
        weight_decay=1e-5,
        norm_stats=norm_stats
    )

    return model
