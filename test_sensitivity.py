from argparse import ArgumentParser, Namespace
import torch
import os
import shutil
from tqdm import tqdm

from hackathon.model_runner import ModelRunner
from hackathon.base_model import BaseModel
from hackathon.models.attn import MultiheadAttn as attn_model
from hackathon.models.Conv1D import Conv1D as conv1d_model
from hackathon.models.linear import Linear as linear_model
from hackathon.models.LSTM import LSTM as lstm_model
from hackathon.models.multimodel import EfficiencyModel as efficiency_model
from hackathon.models.resnet import ResNetModule as resnet_model
from hackathon.models.simplemlp import SimpleMLP as simplemlp_model
from hackathon.models.attn_nores import MultiheadAttnNoRes as attn_nores_model

model_funs = [ # uncomment models you like to process
    #attn_model,
    #conv1d_model,
    #linear_model,
    #lstm_model,
    #simplemlp_model,
    attn_nores_model
]


def main(args: Namespace):
    model: BaseModel

    print('\n+++ Predictions are saved to `./hackathon/logs/<model_name>/sensitivity/<CO2|causal>/predictions.nc>`+++\n')

    for model in (pbar0 := tqdm(model_funs)):
        model_name = model.__module__.split('.')[-1].lower()
        log_dir = f'./hackathon/logs/{model_name}/sensitivity'
        if os.path.isdir(log_dir):
            shutil.rmtree(log_dir)
        checkpoint_path = f'./hackathon/logs/{model_name}/xval/final/final.ckpt'

        runner_CO2 = ModelRunner(
            log_dir=log_dir,
            quickrun=args.quickrun,
            seed=910,
            data_dir='./simple_gpp_model/data/CMIP6/predictor-variables_historical+ssp585+GPP_no-CO2-change.nc')
        runner_causal = ModelRunner(
            log_dir=log_dir,
            quickrun=args.quickrun,
            seed=910,
            data_dir='./simple_gpp_model/data/CMIP6/predictor-variables_historical+ssp585+GPP_non-causal-constant.nc')

        model = torch.load(checkpoint_path)

        pbar0.set_description(f'Model     {"<"+model_name+">":>30}')

        for runner, runner_name in (pbar1 := tqdm(zip(
                [runner_CO2, runner_causal],
                ['CO2', 'causal']), leave=False)):

            pbar1.set_description(f'Predicting {"<"+runner_name+">":>30}')

            if runner_name == 'CO2':
                data_module = runner.data_setup(
                    fold=-1,
                    custom_test_sel={
                        'time': slice('1850', '2100'),
                        'location': range(1, 2)
                    },
                    batch_size=1
                )
            else:
                data_module = runner.data_setup(
                    fold=-1,
                    custom_test_sel={
                        'time': slice('1850', '2100'),
                        'location': range(1, 2)
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
