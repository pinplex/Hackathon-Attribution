import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math

from hackathon import BaseModel


class PermuteBatchSeq(nn.Module):
    def __init__(self) -> None:
        """Switch first two dimensions ('batch_first' <-> 'sequence first')
        """
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x.permute(1, 0, 2)


class PositionalEncoding(nn.Module):

    def __init__(
            self,
            d_model: int,
            dropout: float = 0.1,
            max_len: int = 15000,
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


class MultiheadAttn(BaseModel):
    def __init__(
            self,
            num_inputs: int,
            num_outputs: int,
            d_model: int,
            num_head: int,
            num_hidden: int,
            num_layers: int,
            dropout: float = 0.1,
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
            **kwargs:
                Keyword arguments passed to BaseModel (parent class).
        """

        super().__init__(**kwargs)

        self.model_type = 'Transformer'

        self.input_encoder = nn.Linear(in_features=num_inputs, out_features=d_model)

        self.pos_encoder = PositionalEncoding(d_model=d_model, dropout=dropout)

        self.to_sequence_first = PermuteBatchSeq()

        encoder_layers = TransformerEncoderLayer(
            d_model=d_model, nhead=num_head, dim_feedforward=num_hidden, dropout=dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)

        self.to_batch_first = PermuteBatchSeq()

        self.linear_y = nn.Linear(
            in_features=d_model,
            out_features=num_outputs
        )

        # self.attn_scores = {}
        # for i, layer in enumerate(self.transformer_encoder.layers):
        #     layer.self_attn.register_forward_hook(self.get_activation(f'attn_layer_{i:02d}'))

        self.save_hyperparameters()

    def get_activation(self, name: str):
        # The hook signature.
        def hook(model, input, output):
            self.attn_scores[name] = output[1].detach()

        return hook

    def forward(self, src: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len / 24, seq_len / 24]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """

        # [B, S, I] -> [B, S, D]
        src = self.input_encoder(src)

        # [B, S, D] -> [B, S, D]
        src_enc = self.pos_encoder(src)

        # [B, S, D] -> [S, B, D]
        src_enc = self.to_sequence_first(src_enc)

        # [S, S]
        src_mask = self.generate_square_subsequent_mask(sz=src_enc.shape[0])

        # [S, B, D] -> [S, B, D]
        out = self.transformer_encoder(src_enc, src_mask)

        # [S, B, D] -> [B, S, D]
        out = self.to_batch_first(out)

        # [B, S, D] -> [B, S, D]
        out = out + src

        # [B, S, D] -> [B, S, O]
        out = self.linear_y(out)        

        return out

    def generate_square_subsequent_mask(self, sz: int) -> Tensor:
        """Generates an upper-triangular matrix of -inf, with zeros on diag."""
        return torch.triu(torch.ones(sz, sz, device=self.device) * float('-inf'), diagonal=1)


def model_setup(norm_stats: dict[str, Tensor], **kwargs) -> BaseModel:
    """Create a model as subclass of hackathon.base_model.BaseModel.

    Returns
    -------
    A model.
    """

    default_params = dict(
        d_model=4,
        num_head=4,
        num_hidden=8,
        num_layers=2,
        dropout=0.1,
        learning_rate=0.001,
        weight_decay=0.001,
    )
    default_params.update(kwargs)

    model = MultiheadAttn(
        num_inputs=8,
        num_outputs=1,
        norm_stats=norm_stats,
        **default_params
    )

    return model
