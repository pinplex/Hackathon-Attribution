import torch
from torch import Tensor

from hackathon import BaseModel


class EfficiencyModel(BaseModel):
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
        super(EfficiencyModel, self).__init__(**kwargs)
        

        self.layer11 = torch.nn.Conv1d(
            in_channels = num_features + 2,
            out_channels = 128,
            kernel_size = 30
        )
        
        self.layer12 = torch.nn.Conv1d(
            in_channels = 128,
            out_channels = 1,
            kernel_size = 1
        )

        self.layer2 = torch.nn.Conv1d(
            in_channels = num_features,
            out_channels = 1,
            kernel_size = 30
        )

        #self.layer21 = torch.nn.Conv1d(
        #    in_channels = num_features,
        #    out_channels = 32,
        #    kernel_size = 30
        #)
        #self.layer22 = torch.nn.Conv1d(
        #    in_channels = 32,
        #    out_channels = 1,
        #    kernel_size = 1
        #)
        
        self.layer3 = torch.nn.Conv1d(
            in_channels = num_features,
            out_channels = 1,
            kernel_size = 30
        )
        
        #self.layer31 = torch.nn.Conv1d(
        #    in_channels = num_features,
        #    out_channels = 32,
        #    kernel_size = 30
        #)
        #self.layer32 = torch.nn.Conv1d(
        #    in_channels = 32,
        #    out_channels = 1,
        #    kernel_size = 1
        #)
        
        self.layer4 = torch.nn.Conv1d(
            in_channels = num_features,
            out_channels = 1,
            kernel_size = 30
        )
        
        #self.layer41 = torch.nn.Conv1d(
        #    in_channels = num_features,
        #    out_channels = 32,
        #    kernel_size = 30
        #)
        #self.layer42 = torch.nn.Conv1d(
        #    in_channels = 32,
        #    out_channels = 1,
        #    kernel_size = 1
        #)
        
        self.layer5 = torch.nn.Conv1d(
            in_channels = num_features,
            out_channels = 1,
            kernel_size = 30
        )
        
        #self.layer51 = torch.nn.Conv1d(
        #    in_channels = num_features,
        #    out_channels = 32,
        #    kernel_size = 30
        #)
        #self.layer52 = torch.nn.Conv1d(
        #    in_channels = 32,
        #    out_channels = 1,
        #    kernel_size = 1
        #)
        
        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()


    def forward(self, x: Tensor) -> Tensor:
        """
        The forward function of the simple MLP

        :param x: the input tensor
        :type x: Tensor
        """
        
        n = x.shape[1]
        bs = x.shape[0]
        time = torch.linspace(0, n / 365.25 * 6.282, n)
        sin = torch.sin(time).unsqueeze(0).unsqueeze(-1).to('cuda')
        cos = torch.cos(time).unsqueeze(0).unsqueeze(-1).to('cuda')

        sin = sin.tile((bs,1,1))
        cos = cos.tile((bs,1,1))

        x_time = torch.cat([x,sin,cos], dim = -1)

        x = torch.transpose(x, -1, -2)
        x_time = torch.transpose(x_time, -1, -2)

        max_val = self.layer11(x_time)
        max_val = self.relu(max_val)
        max_val = self.layer12(max_val)

        eff1 = self.layer2(x)
        #eff1 = self.layer21(x)
        #eff1 = self.relu(eff1)
        #eff1 = self.layer22(eff1)
        eff1 = self.tanh(eff1) + 1

        eff2 = self.layer3(x)
        #eff2 = self.layer31(x)
        #eff2 = self.relu(eff2)
        #eff2 = self.layer32(eff2)
        eff2 = self.tanh(eff2) + 1
        
        eff3 = self.layer4(x)
        #eff3 = self.layer41(x)
        #eff3 = self.relu(eff3)
        #eff3 = self.layer42(eff3)
        eff3 = self.tanh(eff3) + 1
        
        eff4 = self.layer5(x)
        #eff4 = self.layer51(x)
        #eff4 = self.relu(eff4)
        #eff4 = self.layer52(eff4)
        eff4 = self.tanh(eff4) + 1
    
        x = max_val * eff1 * eff2 * eff3 * eff4

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
    model = EfficiencyModel(
        num_features=8,
        num_targets=1,
        receptive_field = 30,
        learning_rate=3e-5,
        weight_decay=1e-6,
        norm_stats = norm_stats
    )

    return model
