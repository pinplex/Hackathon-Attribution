import os
from typing import List

import optuna
import pytorch_lightning as pl
from optuna.exceptions import StorageInternalError
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning import Callback
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping

import hackathon
from hackathon.models import resnet

MAX_EPOCHS = 500  # does not matter since we have early stopping! :)


def objective(trial: optuna.trial.Trial) -> float:
    monitor = 'val_loss'

    hyperparams = {}

    context_size = trial.suggest_int('context_size', 1, 200, log=True)
    hyperparams['context_size'] = context_size

    max_n_layers = 4
    n_layers = trial.suggest_int('n_layers', 1, max_n_layers)
    hyperparams['n_layers'] = n_layers

    layers = []
    for i in range(max_n_layers):
        layer_name = f'layer_{i}'
        if int(i < n_layers):
            layer_i = trial.suggest_int(layer_name, 1, 4)
        else:
            layer_i = trial.suggest_int(layer_name, 0, 0)

        layers.append(layer_i)
        hyperparams[layer_name] = layer_i

    in_feats = trial.suggest_int('in_feats', 8, 128)
    hyperparams['in_feats'] = in_feats

    learning_rate = trial.suggest_float('learning_rate', 5e-6, 1e-1, log=True)
    hyperparams['learning_rate'] = learning_rate

    weight_decay = trial.suggest_float('weight_decay', 1e-10, 1e-3, log=True)
    hyperparams['weight_decay'] = weight_decay

    datamodule = setup_data()
    model = resnet.ResNetModule(context_size=context_size,
                                layers=tuple(layers),
                                data_feats=8,
                                in_feats=in_feats,
                                activation_type='relu',
                                target_feats=1,
                                learning_rate=learning_rate,
                                weight_decay=weight_decay,
                                norm_stats=datamodule.norm_stats)

    jobid = os.getenv('SLURM_ARRAY_JOB_ID')
    array_task_id = int(os.getenv('SLURM_ARRAY_TASK_ID'))
    tb_logger = pl_loggers.TensorBoardLogger(save_dir="slurm_logs/",
                                             name=f'{jobid}[{array_task_id:02d}]',
                                             version=f"v{trial.number:03d}",
                                             default_hp_metric=False)

    pruner = PyTorchLightningPruningCallback(trial, monitor=monitor)
    early_stopper = EarlyStopping(monitor=monitor, mode="min", patience=10, verbose=True)

    callbacks: List[Callback] = [pruner, early_stopper]
    trainer = pl.Trainer(logger=tb_logger,
                         enable_checkpointing=False,
                         max_epochs=MAX_EPOCHS,
                         accelerator="gpu",
                         devices=-1,
                         callbacks=callbacks
                         )

    tb_logger.log_hyperparams(hyperparams)
    trainer.fit(model=model, datamodule=datamodule)

    return trainer.callback_metrics[monitor].item()


def setup_data() -> hackathon.DataModule:
    locs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    valid_locs = locs
    train_locs = locs
    train_sel = {
        'location': train_locs,
        'time': slice('1850', '2004')
    }
    valid_sel = {
        'location': valid_locs,
        'time': slice('2005', '2014')
    }
    return hackathon.DataModule(  # You may keep these:
        data_path='./simple_gpp_model/data/CMIP6/predictor-variables_historical+GPP.nc',
        features=[f'var{i}' for i in range(1, 8)] + ['co2'],
        targets=['GPP'],
        # You may change these:
        train_subset=train_sel,
        valid_subset=valid_sel,
        test_subset=valid_sel,
        window_size=3,
        context_size=1,
        # dataloader kwargs
        num_workers=3)


def main():
    pruner: optuna.pruners.BasePruner = optuna.pruners.MedianPruner()

    jobid = os.getenv('SLURM_ARRAY_JOB_ID')

    study = optuna.create_study(direction="minimize",
                                pruner=pruner,
                                sampler=optuna.samplers.TPESampler(),
                                load_if_exists=True,
                                study_name='resnet',
                                storage=f'sqlite:////Net/Groups/BGI/scratch/aschall/hackathon-attribution/resnet-{jobid}.db')

    # storage='postgresql://aschall:3dfbadae-10e3-4aab-bf81-af98e2583a7f@node-r5-he21.bgc-jena.mpg.de:61842/optuna')

    study.optimize(objective, n_trials=100,
                   timeout=60 * 60 * 8 - 60 * 3,  # run 8 h - 3 min :)
                   catch=(RuntimeError, RuntimeWarning, StorageInternalError)
                   )
    print("Number of finished trials: {}".format(len(study.trials)))
    print("Best trial:")
    trial = study.best_trial
    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


if __name__ == "__main__":
    main()
