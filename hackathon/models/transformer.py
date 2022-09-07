import pytorch_lightning as pl
import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math

from hackathon import BaseModel, BaseRunner, DataModule, Ensemble


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


class AttnRunner(BaseRunner):
    """Implements a linear model with training routine."""
    def data_setup(self, fold: int, **kwargs) -> pl.LightningDataModule:
        """Setup datamodule of class pl.LightningDataModule with a given cross validation fold.

        Parameters
        ----------
        fold: the fold id, a dummy parameter that must be in the range 0 to `num_folds`-1.
            Override and implement your own data splitting routine.
        kwargs: Are passed to the DataModule.

        Returns
        -------
        A datamodule of type pl.LightningDataModule.
        """

        train_locs, valid_locs = self.get_loc_split(fold)
        train_sel = {
            'location': train_locs,
            'time': slice('1850', '2004')
        }
        valid_sel = {
            'location': valid_locs,
            'time': slice('2005', '2014')
        }

        datamodule = DataModule(
            # You may keep these:
            data_path='./simple_gpp_model/data/CMIP6/predictor-variables_historical+GPP.nc',
            features=[f'var{i}' for i in range(1, 8)] + ['co2'],
            targets=['GPP'],
            # You may change these:
            train_subset=train_sel,
            valid_subset=valid_sel,
            test_subset=valid_sel,
            window_size=3,
            context_size=1,
            **kwargs)

        return datamodule

    def model_setup(self, num_features: int, num_targets: int):
        """Create a model as subclass of hackathon.base_model.BaseModel.

        Parameters
        ----------
        num_features: The number of features.
        num_targets: The number of targets.

        Returns
        -------
        A model.
        """
        model = MultiheadAttn(
            num_inputs=num_features,
            num_outputs=num_targets,
            d_model=4,
            num_head=4,
            num_hidden=8,
            num_layers=2,
            dropout=0.1,
            learning_rate=0.001,
            weight_decay=0.001,
        )

        return model

    def train(self) -> tuple[pl.Trainer, pl.LightningDataModule, pl.LightningModule]:
        """Runs training.

        Note:
        This is just a blueprint, you may implement your own training routine here, e.g.,
        cross validation. At the end, one single model must be returned.

        Returns
        -------
        A trained model.
        """

        models = []
        for fold in range(10):
            version = f'fold_{fold:02d}'

            datamodule = self.data_setup(
                fold=fold,
                batch_size=10,
                num_workers=10
            )

            model = self.model_setup(
                num_features=datamodule.num_features,
                num_targets=datamodule.num_targets
            )

            trainer = self.trainer_setup(version=version, max_epochs=-1)

            # Fit model with training data (and valid data for early stopping.)
            trainer.fit(model, datamodule=datamodule)

            # Load best model.
            self.load_best_model(trainer=trainer, model=model)

            # Final predictions on the test set.
            self.predict(model=model, trainer=trainer, datamodule=datamodule, version=version)

            models.append(model)

        ensemble = Ensemble(models)

        return trainer, datamodule, ensemble

    @staticmethod
    def get_cv_loc_split(fold: int) -> tuple[list[int], list[int]]:
        """Split clusters of sites into training and validation set.

        Parameters
        ----------
        fold: the fold ID, a value between (including) 0 and 9.

        Returns
        -------
        A tuple, the training and the validation locations.
        """

        locs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        if (fold < 10) and (fold >= 0):
            valid_loc = locs.pop(fold)
            train_loc = locs
        else:
            raise ValueError(
                f'argument `fold` must be a value between (including) 0 and 9, is {fold}.'
            )

        return train_loc, valid_loc
