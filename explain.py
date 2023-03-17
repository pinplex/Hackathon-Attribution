
from argparse import ArgumentParser, Namespace
import torch
import os
import shutil
from tqdm import tqdm

from hackathon.model_runner import ModelRunner, Ensemble
# from hackathon.explainers.test_explainer import TestExplainer
from hackathon.explainers.gradients_based_explanation import \
    GradExplainer, InputXGradExplainer, IntegratedGradientsExplainer

from hackathon.base_model import BaseModel
from hackathon.models.gt_model import GTmodel as gt_model
from hackathon.models.gt_model import model_setup as gt_model_setup
from hackathon.models.attn import MultiheadAttn as attn_model
from hackathon.models.linear import Linear as linear_model
from hackathon.models.LSTM import LSTM as lstm_model
from hackathon.models.simplemlp import SimpleMLP as simplemlp_model

# Needed to run LSTM backpropagation in eval mode.
torch.backends.cudnn.enabled = False


model_funs = [
    gt_model,
    attn_model,
    linear_model,
    lstm_model,
    simplemlp_model
]

explainers = [
    GradExplainer(pbar_loops=True),
    InputXGradExplainer(pbar_loops=True),
    IntegratedGradientsExplainer(pbar_loops=True, n_step=20),
]


def main(args: Namespace):
    model: BaseModel

    print('\n+++ Explanations are saved to `./hackathon/logs/<model_name>/expl/<explainer_name>/explanations.nc`+++\n')

    for model in (pbar0 := tqdm(model_funs)):
        model_name = model.__module__.split('.')[-1].lower()
        log_dir = f'./hackathon/logs/{model_name}/expl'
        if os.path.isdir(log_dir):
            shutil.rmtree(log_dir)
        checkpoint_path = f'./hackathon/logs/{model_name}/xval/final/final.ckpt'

        runner = ModelRunner(log_dir=log_dir, quickrun=args.quickrun, seed=910)

        if model_name == 'gt_model':
            model = gt_model_setup({'mean': torch.zeros(8), 'std': torch.zeros(8)})
            model = Ensemble(model_type_list=[model], is_gt_model=True)
        else:
            model = torch.load(checkpoint_path)
        val_dataloader = runner.data_setup(fold=-1, batch_size=1).xai_dataloader()

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
