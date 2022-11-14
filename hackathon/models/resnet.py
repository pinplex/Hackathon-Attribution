from dataclasses import dataclass
from typing import Optional, Callable, Tuple

import numpy as np
import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F

from hackathon import BaseModel

ACTIVATIONS = {'swish': nn.SiLU,
               'relu': nn.ReLU}


def activation(name: str) -> nn.Module:
    return ACTIVATIONS[name](True)


class ZeroPad(nn.ConstantPad2d):
    def __init__(self, padding):
        super(ZeroPad, self).__init__(padding=(0, 0, padding, padding), value=0)


class ConvXx1(nn.Module):

    def __init__(self, in_feat: int, out_feat: int, stride: int = 1, padding: int = 1, kernel_size: int = 3):
        super(ConvXx1, self).__init__()
        self.padding = padding

        self.pad = nn.Identity() if padding == 0 else ZeroPad(padding)

        self.conv = nn.Conv2d(in_feat, out_feat,
                              kernel_size=(kernel_size, 1),
                              stride=(stride, 1),
                              bias=False)

    def forward(self, x):
        x = self.pad(x)
        return self.conv(x)


class SpecialBN2d(nn.BatchNorm2d):
    def __init__(self, batch_dim: int, *args, **kwargs):
        super(SpecialBN2d, self).__init__(*args, **kwargs)
        self.batch_dim = batch_dim

    def forward(self, x: Tensor) -> Tensor:
        x = self._flip_batch(x)
        x = super(SpecialBN2d, self).forward(x)
        return self._flip_batch(x)

    def _flip_batch(self, x) -> Tensor:
        return torch.transpose(x, 0, self.batch_dim)


@dataclass(eq=False)
class BasicBlock1d(nn.Module):
    in_features: int
    out_features: int
    stride: int = 1
    kernel_size: int = 3

    sampling_dir: Optional[str] = None
    do_sampling: bool = False
    normlayer: Optional[Callable[..., nn.Module]] = None

    activation_type: str = 'relu'

    def __post_init__(self):
        super(BasicBlock1d, self).__init__()

        if self.normlayer is None:
            self.normlayer = nn.BatchNorm2d

        self.down_layer = self._init_down_layer()

        self.activation = activation(self.activation_type)

        self.conv1 = ConvXx1(self.in_features, self.out_features, self.stride)
        self.norm1 = self.normlayer(num_features=self.out_features)

        self.conv2 = ConvXx1(self.out_features, self.out_features)
        self.norm2 = self.normlayer(num_features=self.out_features)

    def _init_down_layer(self):
        if not self.do_sampling:
            return nn.Identity()

        conv1x1 = ConvXx1(self.in_features, self.out_features,
                          kernel_size=1,
                          stride=self.stride,
                          padding=0, )

        return nn.Sequential(conv1x1, self.normlayer(num_features=self.out_features))

    def forward(self, x):

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.norm2(out)

        identity = self.down_layer(x)

        out += identity

        out = self.activation(out)
        return out


@dataclass(eq=False)
class ResNet1d(nn.Module):
    context_size: int
    layers: Tuple[int, ...]

    target_feats: int = 1
    data_feats: int = 8
    apply_maxpool: bool = True

    normlayer: Optional[Callable[..., nn.Module]] = None
    in_feats: int = 64

    activation_type: str = 'relu'

    def __post_init__(self):
        super(ResNet1d, self).__init__()

        if len(self.layers) < 1:
            raise ValueError('you need to specify min on layer.')

        if self.normlayer is None:
            self.normlayer = nn.BatchNorm2d

        self.conv1 = nn.Sequential(ZeroPad(3),
                                   nn.Conv2d(self.data_feats, self.in_feats, kernel_size=(7, 1), padding=0, bias=False))

        self.norm1 = self.normlayer(num_features=self.in_feats)

        if self.apply_maxpool:
            self.maxpool = nn.Sequential(ZeroPad(1),
                                         nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)))

        self.activation = activation(self.activation_type)

        features = self.in_feats * 2
        self.l1 = self._make_layer(features, self.layers[0], stride=1)

        self.layers_sub1 = nn.ModuleList()
        for blocks in self.layers[1:]:
            features *= 2
            self.layers_sub1.append(self._make_layer(features, blocks, stride=2))

        self.layer_out = nn.Conv2d(features, self.target_feats, kernel_size=1, bias=False)
        in_features = np.int32(np.ceil(self._fold_size() / 2 ** len(self.layers)))
        self.fc_out = nn.Linear(in_features, self.target_feats)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity=self.activation_type)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, feats: int, blocks: int, stride: int = 1, sampling_dir='down') -> nn.Sequential:
        do_do_sampling = stride != 1 or self.in_feats != feats

        params = dict(out_features=feats,
                      normlayer=self.normlayer, do_sampling=do_do_sampling,
                      sampling_dir=sampling_dir,
                      activation_type=self.activation_type)

        block_1 = BasicBlock1d(in_features=self.in_feats, stride=stride, **params)

        layers = [block_1]

        self.in_feats = feats
        for _ in range(1, blocks):
            layers.append(BasicBlock1d(in_features=feats, stride=1, **params))

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        in_shape = x.shape  # shape (batch x time x channels)

        fold_size = self._fold_size()
        x = x.unfold(1, fold_size, 1)  # shape (batch x n_folds x channels x fold_size)
        x = torch.permute(x, (0, 2, 3, 1))  # shape (batch x channels x fold_size x n_folds)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)

        if self.apply_maxpool:
            x = self.maxpool(x)

        x = self.l1(x)

        for layer in self.layers_sub1:
            x = layer(x)

        x = self.layer_out(x)

        x = torch.transpose(x, -1, -2)

        x = self.fc_out(x)

        x = torch.squeeze(x, 1)

        diff = in_shape[1] - x.shape[1]
        x = F.pad(x, (0, 0, diff, 0), value=0)
        return x

    def _fold_size(self):
        # plus 1 since we are doing nowcasting,. i.e. ctx + now
        return self.context_size + 1


class ResNetModule(BaseModel):
    def __init__(self, context_size: int, layers: Tuple[int, ...], data_feats=12, in_feats=64,
                 activation_type: str = 'relu',
                 target_feats: int = 1, normlayer: Optional[Callable[..., nn.Module]] = None, **kwargs) -> None:
        super(ResNetModule, self).__init__(**kwargs)

        self.model = ResNet1d(context_size=context_size, target_feats=target_feats, layers=layers, apply_maxpool=True,
                              normlayer=normlayer, in_feats=in_feats, activation_type=activation_type,
                              data_feats=data_feats)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


def model_setup(norm_stats: dict[str, Tensor]) -> BaseModel:
    """Create a model as subclass of hackathon.base_model.BaseModel.


    Parameters
    ----------
    norm_stats: Feature normalization stats with signature {'mean': Tensor, 'std': Tensor}, both tensors with shape (num_features,).

    Returns
    -------
    A model.
    """
    base_model_kwargs = dict(norm_stats=norm_stats,
                             learning_rate=0.013590,
                             weight_decay=0.000007)

    return ResNetModule(
        context_size=5,
        data_feats=8,
        in_feats=8,
        activation_type='relu',
        layers=(2, 1, 1),
        target_feats=1,
        # normlayer=partial(SpecialBN2d, batch_dim=-1),
        **base_model_kwargs
    )
