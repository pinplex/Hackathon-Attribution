
from argparse import ArgumentParser, Namespace
import torch
import os
import shutil
from tqdm import tqdm

from hackathon.model_runner import ModelRunner
from hackathon.base_model import BaseModel
from hackathon.models.attn_nores import MultiheadAttnNoRes as attn_model
from hackathon.models.Conv1D import Conv1D as conv1d_model
from hackathon.models.linear import Linear as linear_model
from hackathon.models.LSTM import LSTM as lstm_model
from hackathon.models.simplemlp import SimpleMLP as simplemlp_model

model_funs = [
    linear_model,
    attn_model,
    conv1d_model,
    lstm_model,
    simplemlp_model
]


def main(args: Namespace):
    model: BaseModel

    print('\n+++ Predictions are saved to `./hackathon/logs/<model_name>/extrap/<space|time>/explanations.nc>`+++\n')

    for model in (pbar0 := tqdm(model_funs)):
        model_name = model.__module__.split('.')[-1].lower()
        log_dir = f'./hackathon/logs/{model_name}/extrap'
        if os.path.isdir(log_dir):
            shutil.rmtree(log_dir)
        checkpoint_path = f'./hackathon/logs/{model_name}/xval/final/final.ckpt'

        runner_space = ModelRunner(
            log_dir=log_dir,
            quickrun=args.quickrun,
            seed=910,
            data_dir='./simple_gpp_model/data/CMIP6/predictor-variables+GPP_space_test.nc')
        runner_time = ModelRunner(
            log_dir=log_dir,
            quickrun=args.quickrun,
            seed=910,
            data_dir='./simple_gpp_model/data/CMIP6/predictor-variables+GPP_time+space_test.nc')

        model = torch.load(checkpoint_path)

        pbar0.set_description(f'Model     {"<"+model_name+">":>30}')

        for runner, runner_name in (pbar1 := tqdm(zip(
                [runner_space, runner_time],
                ['space', 'time']), leave=False)):

            pbar1.set_description(f'Predicting {"<"+runner_name+">":>30}')

            if runner_name == 'space':
                data_module = runner.data_setup(
                    fold=-1,
                    custom_test_sel={
                        'time': slice('1850', '2014'),
                        'location': range(11, 16)
                    },
                    batch_size=1
                )
            else:
                data_module = runner.data_setup(
                    fold=-1,
                    custom_test_sel={
                        'time': slice('2015', '2100'),
                        'location': range(1, 16)
                    },
                    batch_size=1
                )

            trainer = runner.trainer_setup(
                version=runner_name,
                accelerator=None if args.gpu == -1 else 'gpu',
                devices=None if args.gpu == -1 else f'{args.gpu},'
            )

            runner.predict(
                trainer=trainer,
                model=model,
                datamodule=data_module,
                version=runner_name
            )


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--quickrun',
        action='store_true',
        help='Quick development run, explanations are computed for only one location .')
    parser.add_argument(
        '--gpu',
        type=int,
        default=-1,
        help='GPU ID to use, -1 (default) deactivates GPU.')

    args = parser.parse_args()

    main(args)
