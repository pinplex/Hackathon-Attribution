
from argparse import ArgumentParser, Namespace
import torch
import os
import shutil
from tqdm import tqdm

from hackathon.model_runner import ModelRunner
from hackathon.explainers.test_explainer import TestExplainer
from hackathon.base_model import BaseModel
# from hackathon.models.attn import MultiheadAttn as attn_model
# from hackathon.models.Conv1D import Conv1D as conv1d_model
from hackathon.models.linear import Linear as linear_model
# from hackathon.models.LSTM import LSTM as lstm_model
# from hackathon.models.multimodel import EfficiencyModel as efficiency_model
# from hackathon.models.resnet import ResNetModule as resnet_model
# from hackathon.models.simplemlp import SimpleMLP as simplemlp_model

model_funs = [
    # attn_model,
    # conv1d_model,
    linear_model,
    # lstm_model,
    # efficiency_model,
    # resnet_model,
    # simplemlp_model
]

explainers = [
    TestExplainer()
]


def main(args: Namespace):
    model: BaseModel

    print('\n+++ Explanations are saved to `./hackathon/logs/<model_name>/expl/<explainer_name>/explanations.nc>`+++\n')

    for model in (pbar0 := tqdm(model_funs)):
        model_name = model.__module__.split('.')[-1].lower()
        log_dir = f'./hackathon/logs/{model_name}/expl'
        if os.path.isdir(log_dir):
            shutil.rmtree(log_dir)
        checkpoint_path = f'./hackathon/logs/{model_name}/xval/final/final.ckpt'

        runner = ModelRunner(log_dir=log_dir, quickrun=args.quickrun, seed=910)

        model = torch.load(checkpoint_path)
        val_dataloader = runner.data_setup(fold=-1).val_dataloader()

        pbar0.set_description(f'Model     {"<"+model_name+">":>30}')

        for explainer in (pbar1 := tqdm(explainers, leave=False)):
            explainer_name = type(explainer).__name__.lower()
            pbar1.set_description(f'Explainer {"<"+explainer_name+">":>30}')

            log_dir_explainer = os.path.join(log_dir, explainer_name)
            if os.path.isdir(log_dir_explainer):
                shutil.rmtree(log_dir_explainer)
            else:
                os.makedirs(log_dir_explainer)

            explanations = explainer.get_explanations(
                model=model,
                val_dataloader=val_dataloader,
                gpu=None if args.gpu == -1 else args.gpu
            )

            explanations.to_netcdf(os.path.join(log_dir_explainer, 'explanations.nc'))

            val_dataloader.dataset.reset_sensitivity_ds()


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
