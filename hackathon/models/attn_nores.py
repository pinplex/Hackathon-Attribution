import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math

from typing import Iterable, Any, Optional

from hackathon import BaseModel


class PermuteBatchSeq(nn.Module):
    def __init__(self) -> None:
        """Switch first two dimensions ('batch_first' <-> 'sequence first')
        """
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x.permute(1, 0, 2)


class UnfoldToContextLength(nn.Module):
    def __init__(self, context_len: int) -> None:
        """Unfold and reshape input to chunks of `context_len` and reshape attention layout.

        Shapes:
            x: batch, sequence, input
            output: context_len, batch x sequence, innput
 
        Args:
            context_len: the context length, the input sequence will be cut into windows of this length.

        """

        super().__init__()
        self.context_len = context_len
        self.context_pad = torch.nn.ConstantPad2d((0, 0, self.context_len - 1, 0), 0.0)

    def forward(self, x: Tensor) -> Tensor:

        # [B, S, I] -> [B, S + C, I]
        x_t = self.context_pad(x)

        # [B, S + C, I] -> [B, S, I, C]
        x_t = x_t.unfold(dimension=1, size=self.context_len, step=1)

        # [B, S, I, C] -> [B * S, I, C]
        x_t = x_t.flatten(0, 1)

        # [B * S, I, C] -> [C, B * S, I]
        x_t = x_t.permute(2, 0, 1)

        return x_t


class PositionalEncoding(nn.Module):
    def __init__(
            self,
            d_model: int,
            dropout: float = 0.1,
            max_len: int = 100000,
            n: int = 365 * 50):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(n) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.permute(1, 0, 2))
        self.pe: Tensor


    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_input].

        Returns:
            Tensor, shape [batch_dim, seq_len, embedding_dim].
        """
        
        out = x + self.pe[:, :x.size(1), :]
        return self.dropout(out)


class MultiheadAttnNoRes(BaseModel):
    def __init__(
            self,
            num_inputs: int,
            num_outputs: int,
            d_model: int,
            num_head: int,
            num_hidden: int,
            num_layers: int,
            dropout: float = 0.1,
            context_len: int = 30,
            **kwargs):
        """Implements a multihead self-attention model.

        Shapes:
            src: [batch_size, seq_len, num_inputs]
            src_mask: [seq_len, seq_len]

        Returns:
            output Tensor of shape [batch_size, seq_len, d_model]

        Args:
            num_inputs (int):
                The number of inputs.
            num_outputs (int):
                The output dimensionality.
            d_model (int):
                The number of expected features in the input. Bust be an even number.
            num_head (int):
                the number of attention heads per layer.
            num_hidden (int):
                The number of hidden units.
            num_layers (int):
                The number of hidden fully-connected layers.
            dropout (float):
                The dropout applied after each layer, in range [0, 1).
            context_len (int):
                The maximum length of attention, default is 30.
            **kwargs:
                Keyword arguments passed to BaseModel (parent class).
        """

        super().__init__(**kwargs)

        self.context_len = context_len

        self.model_type = 'MultiheadAttention'

        self.input_encoder = nn.Linear(in_features=num_inputs, out_features=d_model)

        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len = 100000, dropout=dropout)

        self.unfold_and_reshape = UnfoldToContextLength(context_len=context_len)

        encoder_layers = TransformerEncoderLayer(
            d_model=d_model, nhead=num_head, dim_feedforward=num_hidden, dropout=dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)

        self.to_batch_first = PermuteBatchSeq()

        self.linear_y = nn.Linear(
            in_features=d_model,
            out_features=num_outputs
        )

        self.activation_out = nn.Softplus()

        # self.attn_scores = {}
        # for i, layer in enumerate(self.transformer_encoder.layers):
        #     layer.self_attn.register_forward_hook(self.get_activation(f'attn_layer_{i:02d}'))

        self.save_hyperparameters()

    def get_activation(self, name: str):
        # The hook signature.
        def hook(model, input, output):
            self.attn_scores[name] = output[1].detach()

        return hook

    # def forward(self, src: Tensor) -> Tensor:
    #     # Quick fix for forward run using too much memory:
    #     # Cut sequence in three overlapping pieces, predict, combine.
    #     S = src.shape[1]
    #     overlap = self.max_temp_context
    #     res = []
    #     if S > (50 * 365.25):
    #         s = S // 3
    #         starts = torch.clamp(s * torch.arange(3) - overlap, min=0)
    #         ends = torch.clamp(s * torch.arange(1, 4), max=S)
    #         for i, (start, end) in enumerate(zip(starts, ends)):
    #             out = self.forward_(src[:, start:end, :])
    #             if i == 0:
    #                 res.append(out)
    #             else:
    #                 res.append(out[:, overlap:, :])
    #         return torch.concat(res, dim=1)

    #     else:
    #         return self.forward_(src)

    def forward(self, src: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size, num_features]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, num_targets]
        """

        b, s, _ = src.shape

        # [B, S, I] -> [B, S, D]
        src = self.input_encoder(src)

        # [B, S, D] -> [B, S, D]
        src_enc = self.pos_encoder(src)

        # [B, S, D] -> [C, B * S, D]
        src_enc = self.unfold_and_reshape(src_enc)

        # [S, S]
        src_mask = self.generate_square_subsequent_mask(sz=self.context_len, max_len=self.context_len)

        # It is not most efficient to compute full self attention here.
        # [C, B * S, D] -> [C, B * S, D]
        out = self.transformer_encoder(src_enc, src_mask)

        # [C, B * S, D] -> [B, S, D]
        out = out[-1].view(b, s, -1)

        # [B, S, D] -> [B, S, O]
        out = self.linear_y(out)
        out = self.activation_out(out)

        return out

    def generate_square_subsequent_mask(self, sz: int, max_len: Optional[int] = None) -> Tensor:
        """Generates an upper-triangular matrix of -inf, with zeros on diag."""
        mask = torch.triu(torch.ones(sz, sz, device=self.device) * float('-inf'), diagonal=1)

        if max_len:
            mask += torch.tril(torch.ones(sz, sz, device=self.device) * float('-inf'), diagonal=-max_len)

        return mask


def model_setup(norm_stats: dict[str, Tensor], **kwargs) -> BaseModel:
    """Create a model as subclass of hackathon.base_model.BaseModel.

    Returns
    -------
    A model.
    """

    default_params = dict(
        d_model=8,
        num_head=2,
        num_hidden=32,
        num_layers=3,
        dropout=0.15,
        learning_rate=0.01,
        weight_decay=0.0001,
    )
    default_params.update(kwargs)

    model = MultiheadAttnNoRes(
        num_inputs=8,
        num_outputs=1,
        norm_stats=norm_stats,
        **default_params
    )


    return model


def get_search_space() -> dict[str, Iterable[Any]]:
    search_space = {
        'd_model': [8, 16, 32],
        'num_head': [1, 2, 4],
        'num_hidden': [32, 64],
        'num_layers': [1, 2, 3],
        'dropout': [0.0, 0.15, 0.3],
        'learning_rate': [1e-4, 1e-3, 1e-2],
        'weight_decay': [1e-4, 1e-3, 1e-2],
    }

    return search_space
