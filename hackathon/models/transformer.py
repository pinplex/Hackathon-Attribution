import os
import shutil
from glob import glob

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from torch import Tensor

from hackathon import BaseModel
from hackathon import DataModule

ROOT_DIR = f'./logs/{os.path.basename(__file__).split(".py")[0]}'


class Linear(torch.nn.Module):
    def __init__(self, num_features: int, num_targets: int):
        super(Linear, self).__init__()

        self.linear = torch.nn.Linear(num_features, num_targets)

    def forward(self, x: Tensor) -> Tensor:
        out = self.linear(x)
        return out


def run(
        rerun=True,
        root_dir: str = ROOT_DIR,
        version=None,
        seed=None) -> tuple[pl.LightningModule, torch.utils.data.DataLoader]:
    pl.seed_everything(seed)

    dataloader_kwargs = dict(batch_size=4, num_workers=4)

    datamodule = DataModule(data_path='./simple_gpp_model/data/CMIP6/predictor-variables_historical+GPP.nc',
                            training_subset={'location': [1, 2], 'time': slice('1850', '1855')},
                            validation_subset={'location': [3, 4], 'time': slice('1855', '1860')},
                            features=[f'var{i}' for i in range(1, 8)] + ['co2'],
                            targets=['GPP'],
                            window_size=2,
                            context_size=1,
                            **dataloader_kwargs)

    custom_model = Linear(
        num_features=datamodule.num_features,
        num_targets=datamodule.num_targets
    )

    model = BaseModel(
        custom_model=custom_model,
        learning_rate=0.001,
        weight_decay=0.0
    )

    # DON'T CHANGE ------
    logger = TensorBoardLogger(save_dir=root_dir, name='', version=version)
    early_stopper = EarlyStopping(patience=10, monitor='val_loss', mode='min')
    checkpointer = ModelCheckpoint(save_top_k=1, monitor='val_loss')
    if os.path.isdir(logger.root_dir) and rerun:
        shutil.rmtree(logger.root_dir)
    # -----------------

    trainer = pl.Trainer(
        logger=logger,
        default_root_dir=root_dir,
        callbacks=[
            early_stopper,
            checkpointer
        ],
        max_epochs=-1
    )

    if rerun:
        trainer.fit(model, datamodule=datamodule)

        # eval_loader = datamodule.test_dataloader()
        eval_loader = datamodule.val_dataloader()  # Use validation dataloader for testing.
        trainer.predict(ckpt_path='best', dataloaders=eval_loader)
    else:
        checkpoint = glob(os.path.join(logger.root_dir, 'checkpoints/*'))
        if len(checkpoint) != 1:
            raise AssertionError(
                f'the number of checkpoints is {len(checkpoint)}, must be 1.'
            )
        checkpoint = checkpoint[0]
        model.load_from_checkpoint(checkpoint)

        datamodule.setup()
        eval_loader = datamodule.val_dataloader()

    return model, eval_loader
